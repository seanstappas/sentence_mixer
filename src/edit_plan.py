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
from media_time import cumulative_frame_ceiling, samples_to_seconds


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

    # Check for leading gap (boundary index -1)
    leading_gap = getattr(graph, "explicit_gaps", {}).get(-1, 0.0)
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

    # Check for trailing gap (boundary after last token)
    if selected_spans and segments:
        last_node = graph.nodes[selected_spans[-1].covered_nodes[-1]]
        trailing_gap = getattr(graph, "explicit_gaps", {}).get(last_node.token_index, 0.0)
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
