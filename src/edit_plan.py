"""
Canonical edit decision list builder.
Schedules integer project audio samples, rational video frame indices,
overlap crossfades, pauses, and loudness normalization into an immutable EditPlan.
"""

from __future__ import annotations

import uuid
from typing import List, Literal, Optional

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
from media_time import samples_to_seconds


def samples_to_frame_index_ceil(samples: int, sample_rate: int, fps: RationalTime) -> int:
    num = samples * fps.numerator
    den = sample_rate * fps.denominator
    return (num + den - 1) // den if num > 0 else 0


def build_edit_plan(
    selected_spans: List[SpanCandidate],
    graph: TargetGraph,
    inventory: InventoryIndex,
    project_sample_rate: int = 48000,
    video_fps: Optional[RationalTime] = None,
    rendering_mode: Literal["source-derived", "cuts-only"] = "source-derived",
    overlap_duration_ms: float = 14.0,
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
    for i, span in enumerate(selected_spans):
        src_rec = inventory.sources[span.source_content_id]
        native_sr = src_rec.native_sample_rate

        # Duration in native samples
        native_duration = span.end_sample - span.start_sample
        # Resampled duration at project sample rate
        resampled_duration = int(round(float(native_duration) * project_sample_rate / float(native_sr)))

        seg_start_sample = current_output_sample
        seg_end_sample = seg_start_sample + resampled_duration

        # Compute video frame bounds using end-covering CFR ceil policy bounded by source duration
        start_frame = samples_to_frame_index_ceil(seg_start_sample, project_sample_rate, video_fps)
        end_frame = samples_to_frame_index_ceil(seg_end_sample, project_sample_rate, video_fps)
        frame_count = max(1, end_frame - start_frame)
        source_pts = samples_to_seconds(span.start_sample, native_sr)

        # Eliminate video frame overrun past source word duration
        source_dur_sec = float(span.end_sample - span.start_sample) / float(native_sr)
        max_source_frames = max(1, int(source_dur_sec * video_fps.numerator / video_fps.denominator + 1e-4))
        frame_count = max(1, min(frame_count, max_source_frames))

        target_phones = [graph.nodes[nid].phone for nid in span.covered_nodes if nid < len(graph.nodes)]

        seg = PlanSegment(
            segment_index=seg_idx,
            source_content_id=span.source_content_id,
            speaker_id=span.speaker_id,
            source_start_sample=span.start_sample,
            source_end_sample=span.end_sample,
            output_start_sample=seg_start_sample,
            output_end_sample=seg_end_sample,
            target_text=span.matched_word,
            target_phones=target_phones,
            span_type=span.span_type,
            source_video_path=src_rec.source_path,
            video_start_frame=start_frame,
            video_frame_count=frame_count,
            video_source_pts_sec=source_pts,
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

            # Check if there is an explicit punctuation pause between these words
            # Only consider a pause if:
            # 1. We are not in 'cuts-only' mode
            # 2. Transitioning between DIFFERENT words (token indices differ)
            # 3. The current span ends the preceding word
            last_node = graph.nodes[span.covered_nodes[-1]]
            next_first_node = graph.nodes[next_span.covered_nodes[0]]

            pause_sec = 0.0
            if rendering_mode != "cuts-only" and last_node.token_index != next_first_node.token_index and last_node.is_word_final:
                pause_sec = graph.punctuation_pauses.get(last_node.token_index, 0.0)

            if pause_sec > 0.0:
                # Insert explicit pause segment
                pause_samples = int(round(pause_sec * project_sample_rate))
                p_start = current_output_sample
                p_end = p_start + pause_samples

                p_start_frame = samples_to_frame_index_ceil(p_start, project_sample_rate, video_fps)
                p_end_frame = samples_to_frame_index_ceil(p_end, project_sample_rate, video_fps)
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
                    video_frame_count=max(1, p_end_frame - p_start_frame),
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
                        overlap = min(overlap_samples_const, resampled_duration // 4)
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

    total_output_samples = current_output_sample
    total_video_frames = samples_to_frame_index_ceil(total_output_samples, project_sample_rate, video_fps)

    # Reconcile video slice frame counts so concatenated video matches timeline bounds without overrunning source duration
    if segments:
        for k in range(len(segments) - 1):
            s_curr = segments[k]
            s_next = segments[k + 1]
            raw_frames = max(1, s_next.video_start_frame - s_curr.video_start_frame)
            if not s_curr.is_silence and s_curr.source_content_id in inventory.sources:
                src_rec = inventory.sources[s_curr.source_content_id]
                source_dur = float(s_curr.source_end_sample - s_curr.source_start_sample) / float(src_rec.native_sample_rate)
                max_f = max(1, int(source_dur * video_fps.numerator / video_fps.denominator + 1e-4))
                s_curr.video_frame_count = max(1, min(raw_frames, max_f))
            else:
                s_curr.video_frame_count = raw_frames

        # Handle last segment
        s_last = segments[-1]
        raw_last_frames = max(1, total_video_frames - s_last.video_start_frame)
        if not s_last.is_silence and s_last.source_content_id in inventory.sources:
            src_rec = inventory.sources[s_last.source_content_id]
            source_dur = float(s_last.source_end_sample - s_last.source_start_sample) / float(src_rec.native_sample_rate)
            max_f = max(1, int(source_dur * video_fps.numerator / video_fps.denominator + 1e-4))
            s_last.video_frame_count = max(1, min(raw_last_frames, max_f))
        else:
            s_last.video_frame_count = raw_last_frames

        total_video_frames = sum(s.video_frame_count for s in segments)

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
    return plan
