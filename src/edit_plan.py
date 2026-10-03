"""
Canonical edit decision list builder.
Schedules integer project audio samples, rational video frame indices,
overlap crossfades, pauses, and loudness normalization into an immutable EditPlan.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional

import numpy as np
import soundfile as sf

from contracts import (
    EDIT_PLAN_SCHEMA_VERSION,
    EditPlan,
    InventoryIndex,
    PlanSegment,
    PlanTransition,
    RationalTime,
    SpanCandidate,
    TargetGraph,
)
from media_time import cumulative_frame_ceiling, samples_to_seconds


def build_edit_plan(
    selected_spans: List[SpanCandidate],
    graph: TargetGraph,
    inventory: InventoryIndex,
    project_sample_rate: int = 48000,
    video_fps: Optional[RationalTime] = None,
    rendering_mode: Literal["source-derived", "cuts-only"] = "source-derived",
    overlap_duration_ms: float = 14.0,
    lead_in_sec: float = 0.0,
) -> EditPlan:
    """
    Constructs an immutable, sample-accurate EditPlan from selected spans.
    """
    if video_fps is None:
        video_fps = RationalTime(30000, 1001)

    if not selected_spans:
        raise ValueError("Cannot build EditPlan with empty spans")

    plan_id = f"plan_{uuid.uuid4().hex[:12]}"
    segments: List[PlanSegment] = []
    transitions: List[PlanTransition] = []

    current_output_sample = 0
    overlap_samples_const = int(round(overlap_duration_ms * project_sample_rate / 1000.0))

    seg_idx = 0

    # Check for leading gap (boundary index -1)
    leading_gap = getattr(graph, "explicit_gaps", {}).get(-1, lead_in_sec)
    if leading_gap > 0.0 and selected_spans:
        first_span = selected_spans[0]
        first_src_rec = inventory.sources[first_span.source_content_id]
        pause_samples = int(round(leading_gap * project_sample_rate))
        p_start = current_output_sample
        p_end = p_start + pause_samples

        p_start_frame = cumulative_frame_ceiling(p_start, project_sample_rate, video_fps)
        p_end_frame = cumulative_frame_ceiling(p_end, project_sample_rate, video_fps)
        p_source_pts = samples_to_seconds(first_span.start_sample, first_src_rec.native_sample_rate)

        pause_seg = PlanSegment(
            segment_index=seg_idx,
            source_content_id="silence",
            speaker_id=first_span.speaker_id,
            source_start_sample=0,
            source_end_sample=pause_samples,
            output_start_sample=p_start,
            output_end_sample=p_end,
            target_text="[pause]",
            target_phones=["sil"],
            span_type="pause",
            source_video_path=first_src_rec.source_path,
            is_silence=True,
            freeze_video=True,
            video_start_frame=p_start_frame,
            video_frame_count=p_end_frame - p_start_frame,
            video_source_pts_sec=p_source_pts,
        )
        segments.append(pause_seg)
        seg_idx += 1
        current_output_sample = p_end

        transitions.append(PlanTransition(
            transition_index=len(transitions),
            from_segment_index=pause_seg.segment_index,
            to_segment_index=seg_idx,
            transition_type="hard_splice",
            overlap_samples=0,
        ))

    for i, span in enumerate(selected_spans):
        src_rec = inventory.sources[span.source_content_id]
        native_sr = src_rec.native_sample_rate

        # Duration in native samples
        native_duration = span.end_sample - span.start_sample
        # Resampled duration at project sample rate
        resampled_duration = int(round(float(native_duration) * project_sample_rate / float(native_sr)))

        seg_start_sample = current_output_sample
        seg_end_sample = seg_start_sample + resampled_duration

        # Compute video frame bounds using cumulative ceiling quantizer (drift-free)
        start_frame = cumulative_frame_ceiling(seg_start_sample, project_sample_rate, video_fps)
        end_frame = cumulative_frame_ceiling(seg_end_sample, project_sample_rate, video_fps)
        frame_count = end_frame - start_frame  # may be 0 for very short clips
        source_pts = samples_to_seconds(span.start_sample, native_sr)

        target_phones = [graph.nodes[nid].phone for nid in span.covered_nodes if nid < len(graph.nodes)]

        realized = getattr(span, "realized_phones", None)
        if not realized and inventory is not None:
            realized = [inventory.events[eid].label for eid in span.event_ids if eid in inventory.events and inventory.events[eid].kind == "phone"]
        if not realized:
            realized = list(target_phones)

        target_words_covered: List[str] = []
        if graph and hasattr(graph, "nodes"):
            for nid in span.covered_nodes:
                if nid < len(graph.nodes):
                    w_txt = getattr(graph.nodes[nid], "word_text", "")
                    if w_txt and (not target_words_covered or target_words_covered[-1] != w_txt):
                        target_words_covered.append(w_txt)
        target_label = " ".join(target_words_covered) if target_words_covered else span.matched_word

        seg = PlanSegment(
            segment_index=seg_idx,
            source_content_id=span.source_content_id,
            speaker_id=span.speaker_id,
            source_start_sample=span.start_sample,
            source_end_sample=span.end_sample,
            output_start_sample=seg_start_sample,
            output_end_sample=seg_end_sample,
            target_text=target_label,
            target_phones=target_phones,
            span_type=span.span_type,
            source_video_path=src_rec.source_path,
            video_start_frame=start_frame,
            video_frame_count=frame_count,
            video_source_pts_sec=source_pts,
            realized_phones=realized,
        )
        segments.append(seg)
        seg_idx += 1

        # Advance output timeline
        current_output_sample = seg_end_sample

        # Check transition to next segment
        if i < len(selected_spans) - 1:
            next_span = selected_spans[i + 1]

            # Determine if this is a natural continuation
            is_cont = False
            if span.source_content_id == next_span.source_content_id:
                if span.end_sample == next_span.start_sample:
                    is_cont = True
                elif span.event_ids and next_span.event_ids:
                    last_ev = inventory.events.get(span.event_ids[-1])
                    if last_ev and last_ev.successor_id == next_span.event_ids[0]:
                        is_cont = True

            # Check if there is an explicit pause between these words
            last_node = graph.nodes[span.covered_nodes[-1]]
            next_first_node = graph.nodes[next_span.covered_nodes[0]]

            pause_sec = 0.0
            if last_node.token_index != next_first_node.token_index and last_node.is_word_final:
                explicit_gaps = getattr(graph, "explicit_gaps", {})
                if last_node.token_index in explicit_gaps:
                    pause_sec = explicit_gaps[last_node.token_index]
                elif rendering_mode != "cuts-only":
                    pause_sec = graph.punctuation_pauses.get(last_node.token_index, 0.0)

            if pause_sec > 0.0:
                # Insert explicit pause segment
                pause_samples = int(round(pause_sec * project_sample_rate))
                p_start = current_output_sample
                p_end = p_start + pause_samples

                p_start_frame = cumulative_frame_ceiling(p_start, project_sample_rate, video_fps)
                p_end_frame = cumulative_frame_ceiling(p_end, project_sample_rate, video_fps)
                p_source_pts = source_pts + (float(resampled_duration) / float(project_sample_rate))

                pause_seg = PlanSegment(
                    segment_index=seg_idx,
                    source_content_id="silence",
                    speaker_id=span.speaker_id,
                    source_start_sample=0,
                    source_end_sample=pause_samples,
                    output_start_sample=p_start,
                    output_end_sample=p_end,
                    target_text="[pause]",
                    target_phones=["sil"],
                    span_type="pause",
                    source_video_path=src_rec.source_path,
                    is_silence=True,
                    freeze_video=True,
                    video_start_frame=p_start_frame,
                    video_frame_count=p_end_frame - p_start_frame,  # may be 0 for <gap:0ms>
                    video_source_pts_sec=p_source_pts,
                )
                segments.append(pause_seg)
                seg_idx += 1

                transitions.append(PlanTransition(
                    transition_index=len(transitions),
                    from_segment_index=seg.segment_index,
                    to_segment_index=pause_seg.segment_index,
                    transition_type="pause",
                    overlap_samples=0,
                ))

                current_output_sample = p_end

                transitions.append(PlanTransition(
                    transition_index=len(transitions),
                    from_segment_index=pause_seg.segment_index,
                    to_segment_index=seg_idx,  # points to next segment
                    transition_type="hard_splice",
                    overlap_samples=0,
                ))
            else:
                if is_cont:
                    transitions.append(PlanTransition(
                        transition_index=len(transitions),
                        from_segment_index=seg.segment_index,
                        to_segment_index=seg_idx,
                        transition_type="continuation",
                        overlap_samples=0,
                    ))
                else:
                    if rendering_mode == "source-derived":
                        next_src_rec = inventory.sources.get(next_span.source_content_id)
                        next_sr = next_src_rec.native_sample_rate if next_src_rec and next_src_rec.native_sample_rate > 0 else project_sample_rate
                        next_resampled_dur = int(round(float(next_span.end_sample - next_span.start_sample) * project_sample_rate / float(next_sr)))
                        max_overlap = min(resampled_duration // 5, next_resampled_dur // 5)
                        overlap = min(overlap_samples_const, max_overlap)
                        current_output_sample -= overlap
                        transitions.append(PlanTransition(
                            transition_index=len(transitions),
                            from_segment_index=seg.segment_index,
                            to_segment_index=seg_idx,
                            transition_type="overlap",
                            overlap_samples=overlap,
                            fade_samples=overlap,
                        ))
                    else:  # cuts-only mode
                        transitions.append(PlanTransition(
                            transition_index=len(transitions),
                            from_segment_index=seg.segment_index,
                            to_segment_index=seg_idx,
                            transition_type="hard_splice",
                            overlap_samples=0,
                        ))

    # Check for trailing gap (boundary after last token)
    if selected_spans and segments:
        last_node = graph.nodes[selected_spans[-1].covered_nodes[-1]]
        explicit_gaps = getattr(graph, "explicit_gaps", {})
        if last_node.token_index in explicit_gaps:
            trailing_gap = explicit_gaps[last_node.token_index]
        elif rendering_mode != "cuts-only":
            trailing_gap = graph.punctuation_pauses.get(last_node.token_index, 0.0)
        else:
            trailing_gap = 0.0

        if trailing_gap > 0.0:
            last_speech_seg = segments[-1]
            pause_samples = int(round(trailing_gap * project_sample_rate))
            p_start = current_output_sample
            p_end = p_start + pause_samples

            p_start_frame = cumulative_frame_ceiling(p_start, project_sample_rate, video_fps)
            p_end_frame = cumulative_frame_ceiling(p_end, project_sample_rate, video_fps)

            pause_seg = PlanSegment(
                segment_index=seg_idx,
                source_content_id="silence",
                speaker_id=last_speech_seg.speaker_id,
                source_start_sample=0,
                source_end_sample=pause_samples,
                output_start_sample=p_start,
                output_end_sample=p_end,
                target_text="[pause]",
                target_phones=["sil"],
                span_type="pause",
                source_video_path=last_speech_seg.source_video_path,
                is_silence=True,
                freeze_video=True,
                video_start_frame=p_start_frame,
                video_frame_count=p_end_frame - p_start_frame,
                video_source_pts_sec=last_speech_seg.video_source_pts_sec,
            )
            segments.append(pause_seg)
            transitions.append(PlanTransition(
                transition_index=len(transitions),
                from_segment_index=last_speech_seg.segment_index,
                to_segment_index=pause_seg.segment_index,
                transition_type="pause",
                overlap_samples=0,
            ))
            seg_idx += 1
            current_output_sample = p_end

    total_output_samples = current_output_sample
    total_video_frames = cumulative_frame_ceiling(total_output_samples, project_sample_rate, video_fps)

    # Deterministic picture-switch policy during audio overlap:
    # Partition cumulative timeline at crossfade midpoint so sum(video_frame_count) == total_video_frames
    if segments:
        overlap_by_from_seg = {}
        for t in transitions:
            if t.transition_type == "overlap":
                overlap_by_from_seg[t.from_segment_index] = t.overlap_samples

        v_display_starts = [0] * len(segments)
        v_display_ends = [0] * len(segments)

        v_display_starts[0] = 0
        for k in range(len(segments) - 1):
            curr_seg = segments[k]
            next_seg = segments[k + 1]
            ov = overlap_by_from_seg.get(curr_seg.segment_index, 0)
            if ov > 0:
                midpoint = next_seg.output_start_sample + (ov // 2)
                v_display_ends[k] = midpoint
                v_display_starts[k + 1] = midpoint
            else:
                v_display_ends[k] = curr_seg.output_end_sample
                v_display_starts[k + 1] = next_seg.output_start_sample

        v_display_ends[-1] = total_output_samples

        for k, seg in enumerate(segments):
            v_start = v_display_starts[k]
            v_end = v_display_ends[k]
            start_f = cumulative_frame_ceiling(v_start, project_sample_rate, video_fps)
            end_f = cumulative_frame_ceiling(v_end, project_sample_rate, video_fps)
            seg.video_start_frame = start_f
            seg.video_frame_count = max(0, end_f - start_f)

            # Adjust video_source_pts_sec for audio overlap offset if not silence
            if not seg.is_silence:
                offset_samples = v_start - seg.output_start_sample
                if offset_samples > 0:
                    seg.video_source_pts_sec += (offset_samples / float(project_sample_rate))

    plan = EditPlan(
        schema_version=EDIT_PLAN_SCHEMA_VERSION,
        plan_id=plan_id,
        target_text=graph.raw_text,
        speaker_id=selected_spans[0].speaker_id,
        selector_mode="span",
        rendering_mode="source-derived" if rendering_mode == "source-derived" else "cuts-only",
        project_sample_rate=project_sample_rate,
        video_fps=video_fps,
        total_output_samples=total_output_samples,
        total_video_frames=total_video_frames,
        segments=segments,
        transitions=transitions,
        sources=inventory.sources,
        metadata={"num_cuts": len(transitions)},
    )
    plan.validate()
    finalize_edit_plan(plan, inventory, normalize_loudness=(rendering_mode != "cuts-only"))
    return plan


def finalize_edit_plan(
    plan: EditPlan,
    inventory: Optional[InventoryIndex] = None,
    normalize_loudness: bool = True,
    target_i: float = -16.0,
    target_tp: float = -1.0,
    target_lra: float = 11.0,
) -> EditPlan:
    """
    Freezes all DSP decisions into the EditPlan before rendering:
    1. Computes float gain_db for all non-silence segments over consolidated natural runs.
       In cuts-only mode or if normalize_loudness is False, gain_db is 0.0.
    2. Measures EBU R128 loudness on an unpublished analysis master and records the
       analysis stats and target parameters in plan metadata.
    3. Guarantees that rendering will not need to make DSP decisions or mutate the plan.
    """
    is_cuts_only = (plan.rendering_mode == "cuts-only")

    # Step 1: Consolidate natural runs and compute gain_db for every segment
    runs: List[List[PlanSegment]] = []
    curr_run: List[PlanSegment] = []

    for seg in plan.segments:
        if seg.is_silence:
            seg.gain_db = 0.0
            if curr_run:
                runs.append(curr_run)
                curr_run = []
            continue

        if not curr_run:
            curr_run.append(seg)
        else:
            prev_seg = curr_run[-1]
            if (
                prev_seg.source_content_id == seg.source_content_id
                and prev_seg.source_end_sample == seg.source_start_sample
            ):
                curr_run.append(seg)
            else:
                runs.append(curr_run)
                curr_run = [seg]

    if curr_run:
        runs.append(curr_run)

    segment_rms_db: Dict[int, float] = {}

    for run in runs:
        if is_cuts_only or not normalize_loudness:
            for s in run:
                s.gain_db = 0.0
            continue

        first_seg = run[0]
        src_rec = plan.sources.get(first_seg.source_content_id) if plan.sources else None
        src_path = None
        if src_rec:
            if src_rec.cached_pcm_path and Path(src_rec.cached_pcm_path).exists():
                src_path = src_rec.cached_pcm_path
            elif src_rec.source_path and Path(src_rec.source_path).exists():
                src_path = src_rec.source_path
        elif first_seg.source_video_path and Path(first_seg.source_video_path).exists():
            src_path = first_seg.source_video_path

        run_audio_chunks = []
        if src_path:
            try:
                arr, native_rate = sf.read(str(src_path))
                rec_sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else native_rate
                for s in run:
                    if native_rate != rec_sr:
                        st = int(round(s.source_start_sample * native_rate / rec_sr))
                        en = int(round(s.source_end_sample * native_rate / rec_sr))
                    else:
                        st = s.source_start_sample
                        en = s.source_end_sample
                    st = max(0, min(len(arr), st))
                    en = max(0, min(len(arr), en))
                    chunk = arr[st:en]
                    if native_rate != plan.project_sample_rate and len(chunk) > 0:
                        import librosa
                        if chunk.ndim > 1:
                            chunk = librosa.resample(chunk, orig_sr=native_rate, target_sr=plan.project_sample_rate, axis=0)
                        else:
                            chunk = librosa.resample(chunk, orig_sr=native_rate, target_sr=plan.project_sample_rate)
                    run_audio_chunks.append(chunk)
            except Exception:
                pass

        if run_audio_chunks:
            run_audio = np.concatenate(run_audio_chunks, axis=0)
            if run_audio.ndim > 1:
                run_audio = np.mean(run_audio, axis=1)
            active_mask = np.abs(run_audio) > 1e-4
            if np.sum(active_mask) > 50:
                rms = float(np.sqrt(np.mean(run_audio[active_mask] ** 2)))
            else:
                rms = float(np.sqrt(np.mean(run_audio ** 2))) if len(run_audio) > 0 else 0.0

            if rms > 1e-4:
                rms_db = 20.0 * np.log10(rms)
                auto_gain_db = -20.0 - rms_db
                is_subword_or_phone = any(s.span_type in ("subword", "phone") for s in run)
                limit_db = 12.0 if is_subword_or_phone else 9.0
                gain_val = float(max(-limit_db, min(limit_db, auto_gain_db)))
            else:
                rms_db = -20.0
                gain_val = 0.0
        else:
            rms_db = -20.0
            gain_val = 0.0

        for s in run:
            s.gain_db = gain_val
            segment_rms_db[s.segment_index] = rms_db

    # Ensure all segments have a valid float gain_db
    for s in plan.segments:
        if s.gain_db is None:
            s.gain_db = 0.0

    # Step 1b: Dialogue acoustic loudness smoothing and intra-word consistency
    # Limit rapid loudness transitions between consecutive spliced phonetic fragments (subwords/phones)
    # to <= 3.0 dB to prevent jarring volume jumps between consonants and vowels, while allowing
    # full multi-clip leveling range between long independent dialogue phrases.
    if not is_cuts_only and normalize_loudness:
        speech_indices = [idx for idx, s in enumerate(plan.segments) if not s.is_silence]
        threshold_samples = int(round(0.250 * plan.project_sample_rate))
        if len(speech_indices) > 1:
            for i in range(len(speech_indices) - 1):
                idx_curr = speech_indices[i]
                idx_next = speech_indices[i + 1]
                s_c = plan.segments[idx_curr]
                s_n = plan.segments[idx_next]
                is_fragment = (
                    s_c.target_text == s_n.target_text
                    or s_c.span_type in ("subword", "phone")
                    or s_n.span_type in ("subword", "phone")
                    or s_c.duration_samples < threshold_samples
                    or s_n.duration_samples < threshold_samples
                )
                max_jump = 3.0 if (is_fragment and idx_next == idx_curr + 1) else 18.0
                level_curr = segment_rms_db.get(s_c.segment_index, -20.0) + (s_c.gain_db or 0.0)
                level_next = segment_rms_db.get(s_n.segment_index, -20.0) + (s_n.gain_db or 0.0)
                if level_next > level_curr + max_jump:
                    excess = level_next - (level_curr + max_jump)
                    s_n.gain_db = float(max(-12.0, (s_n.gain_db or 0.0) - excess))
                elif level_next < level_curr - max_jump:
                    deficit = (level_curr - max_jump) - level_next
                    s_n.gain_db = float(min(12.0, (s_n.gain_db or 0.0) + deficit))

            for i in range(len(speech_indices) - 1, 0, -1):
                idx_curr = speech_indices[i]
                idx_prev = speech_indices[i - 1]
                s_c = plan.segments[idx_curr]
                s_p = plan.segments[idx_prev]
                is_fragment = (
                    s_c.target_text == s_p.target_text
                    or s_c.span_type in ("subword", "phone")
                    or s_p.span_type in ("subword", "phone")
                    or s_c.duration_samples < threshold_samples
                    or s_p.duration_samples < threshold_samples
                )
                max_jump = 3.0 if (is_fragment and idx_curr == idx_prev + 1) else 18.0
                level_curr = segment_rms_db.get(s_c.segment_index, -20.0) + (s_c.gain_db or 0.0)
                level_prev = segment_rms_db.get(s_p.segment_index, -20.0) + (s_p.gain_db or 0.0)
                if level_prev > level_curr + max_jump:
                    excess = level_prev - (level_curr + max_jump)
                    s_p.gain_db = float(max(-12.0, (s_p.gain_db or 0.0) - excess))
                elif level_prev < level_curr - max_jump:
                    deficit = (level_curr - max_jump) - level_prev
                    s_p.gain_db = float(min(12.0, (s_p.gain_db or 0.0) + deficit))

        # Step 1c: Sentence-level speech leveling anchor
        # Prevent any speech segment inside a sentence from dropping > 3.5 dB below the median
        # sentence speech level to prevent quiet words or suffixes (e.g. 'pudding') from being swallowed.
        speech_segs = [s for s in plan.segments if not s.is_silence]
        if speech_segs:
            speech_levels = [segment_rms_db.get(s.segment_index, -20.0) + (s.gain_db or 0.0) for s in speech_segs]
            median_level = float(np.median(speech_levels))
            min_allowed_level = median_level - 3.5
            for s in speech_segs:
                current_level = segment_rms_db.get(s.segment_index, -20.0) + (s.gain_db or 0.0)
                if current_level < min_allowed_level:
                    deficit = min_allowed_level - current_level
                    is_subword_or_phone = s.span_type in ("subword", "phone")
                    limit_db = 12.0 if is_subword_or_phone else 9.0
                    s.gain_db = float(min(limit_db, (s.gain_db or 0.0) + deficit))

    # Step 2: Measure loudness on an unpublished analysis master and record metadata
    if not is_cuts_only and normalize_loudness:
        try:
            from video_assembler import render_master_pcm
            with tempfile.TemporaryDirectory(prefix="sm_loudness_analysis_") as tmp_d:
                analysis_wav = Path(tmp_d) / "analysis_master.wav"
                render_master_pcm(plan, str(analysis_wav), normalize_loudness=False)
                if analysis_wav.exists() and analysis_wav.stat().st_size > 0:
                    cmd1 = [
                        "ffmpeg", "-nostats",
                        "-i", str(analysis_wav),
                        "-af", f"loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA={target_lra:.1f}:print_format=json",
                        "-f", "null", "-"
                    ]
                    proc1 = subprocess.run(cmd1, capture_output=True, text=True)
                    if proc1.returncode == 0:
                        match = re.search(r"\{\s*\"input_i\"\s*:.*?\}", proc1.stderr, re.DOTALL)
                        if match:
                            stats = json.loads(match.group(0))
                            plan.metadata["ebur128_analysis"] = stats
                            plan.metadata["ebur128_target"] = {
                                "target_i": target_i,
                                "target_tp": target_tp,
                                "target_lra": target_lra,
                            }
        except Exception:
            pass

    plan.metadata["normalize_loudness"] = normalize_loudness
    return plan
