"""Unit tests for dialogue scene synthesizer and word-highlighted subtitle generator."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dialogue_mixer import (
    format_ass_time,
    parse_script_text,
    concat_edit_plans,
    color_to_ass,
    resolve_speaker_color,
    synthesize_and_render_dialogue,
)
from contracts import (
    EditPlan,
    InventoryIndex,
    PlanSegment,
    PlanTransition,
    RationalTime,
    SourceRecord,
    SpanCandidate,
    WordOccurrence,
)
from features import ContinuousFeatureExtractor
from span_selector import SpanSelector


def test_format_ass_time():
    assert format_ass_time(0.0) == "0:00:00.00"
    assert format_ass_time(1.234) == "0:00:01.23"
    assert format_ass_time(65.5) == "0:01:05.50"
    assert format_ass_time(3661.05) == "1:01:01.05"


def test_parse_script_text():
    script = """
    # This is a comment
    anakin: It's not the Jedi way - he must live!
    mace-windu: That's a suspiciously specific request, but okay.
    """
    lines = parse_script_text(script)
    assert len(lines) == 2
    assert lines[0][0] == "anakin"
    assert "Jedi way, he must live!" in lines[0][1]
    assert lines[1][0] == "mace-windu"
    assert lines[1][1] == "That's a suspiciously specific request, but okay."


def test_concat_edit_plans():
    fps = RationalTime(24000, 1001)
    seg1 = PlanSegment(
        segment_index=0,
        source_content_id="src1",
        speaker_id="anakin",
        source_start_sample=0,
        source_end_sample=48000,
        output_start_sample=0,
        output_end_sample=48000,
        target_text="test one",
        target_phones=["T", "EH1", "S", "T", "W", "AH1", "N"],
        span_type="phrase",
        source_video_path="fake.mkv",
        gain_db=0.0,
        video_start_frame=0,
        video_frame_count=24,
    )
    plan1 = EditPlan(
        plan_id="p1",
        target_text="test one",
        speaker_id="anakin",
        project_sample_rate=48000,
        video_fps=fps,
        total_output_samples=48000,
        total_video_frames=24,
        segments=[seg1],
        sources={"src1": None},
    )

    seg2 = PlanSegment(
        segment_index=0,
        source_content_id="src2",
        speaker_id="mace-windu",
        source_start_sample=0,
        source_end_sample=24000,
        output_start_sample=0,
        output_end_sample=24000,
        target_text="test two",
        target_phones=["T", "EH1", "S", "T", "T", "UW1"],
        span_type="phrase",
        source_video_path="fake2.mkv",
        gain_db=0.0,
        video_start_frame=0,
        video_frame_count=12,
    )
    plan2 = EditPlan(
        plan_id="p2",
        target_text="test two",
        speaker_id="mace-windu",
        project_sample_rate=48000,
        video_fps=fps,
        total_output_samples=24000,
        total_video_frames=12,
        segments=[seg2],
        sources={"src2": None},
    )

    combined = concat_edit_plans([plan1, plan2], pause_between_sec=0.5)
    combined.validate()

    assert len(combined.segments) == 3  # seg1, pause, seg2
    assert combined.segments[1].is_silence
    assert combined.segments[1].span_type == "pause"
    assert combined.segments[2].speaker_id == "mace-windu"
    assert combined.total_output_samples == 48000 + 24000 + 24000  # 48k + 0.5s*48k + 24k = 96k


def test_color_to_ass():
    assert color_to_ass("yellow") == "&H0000FFFF&"
    assert color_to_ass("purple") == "&H00FF40D0&"
    assert color_to_ass("blue") == "&H00FFB432&"
    assert color_to_ass("green") == "&H0050F032&"
    assert color_to_ass("red") == "&H003C3CFF&"
    # Hex input
    assert color_to_ass("#FF0000") == "&H000000FF&"  # red in BGR
    # ASS tag input
    assert color_to_ass("&H00AABBCC&") == "&H00AABBCC&"


def test_resolve_speaker_color():
    colors = {
        "mace": "purple",
        "anakin": "blue",
        "luke": "green",
        "sidious": "red",
        "vader": "blue",
    }
    assert resolve_speaker_color("anakin", colors) == "&H00FFB432&"
    assert resolve_speaker_color("mace-windu", colors) == "&H00FF40D0&"
    assert resolve_speaker_color("luke", colors) == "&H0050F032&"
    assert resolve_speaker_color("palpatine", colors) == "&H003C3CFF&"  # alias sidious
    assert resolve_speaker_color("vader", colors) == "&H00FFB432&"
    # Unmatched fallback
    assert resolve_speaker_color("yoda", colors) == "&H0000FFFF&"


def test_dialogue_mixer_feature_extractor_wired_and_scene_loudnorm(monkeypatch, tmp_path):
    """
    Validates Fix P1.1 and Fix P1.3:
    1. ContinuousFeatureExtractor(sr=16000) is instantiated and passed to selector.select.
    2. Per-turn build_edit_plan is invoked with finalize=False to bypass redundant loudnorm runs.
    3. finalize_edit_plan is invoked once on combined_plan for single-pass scene loudness finalization.
    """
    import dialogue_mixer

    src = SourceRecord(
        content_id="src1",
        source_path="fake.mp4",
        native_sample_rate=48000,
        native_channels=2,
    )
    wo1 = WordOccurrence(
        word_id="w1",
        source_content_id="src1",
        speaker_id="anakin",
        word_normalized="hello",
        word_original="hello",
        start_sample=0,
        end_sample=24000,
        phone_event_ids=["ev1"],
        realized_phones=["HH"],
    )
    wo2 = WordOccurrence(
        word_id="w2",
        source_content_id="src1",
        speaker_id="mace-windu",
        word_normalized="there",
        word_original="there",
        start_sample=24000,
        end_sample=48000,
        phone_event_ids=["ev2"],
        realized_phones=["DH"],
    )
    inv = InventoryIndex(
        sources={"src1": src},
        words={"w1": wo1, "w2": wo2},
    )

    passed_extractors = []

    def mock_select(self, graph, speaker_id=None, feature_extractor=None, **kwargs):
        passed_extractors.append(feature_extractor)
        return [
            SpanCandidate(
                candidate_id=f"c_{speaker_id}",
                target_node_start=0,
                target_node_end=0,
                covered_nodes=[0],
                source_content_id="src1",
                speaker_id=speaker_id or "anakin",
                start_sample=0,
                end_sample=24000,
                event_ids=[],
                span_type="word",
                matched_word="hello",
            )
        ]

    build_plan_finalize_flags = []
    real_build_edit_plan = dialogue_mixer.build_edit_plan

    def mock_build_edit_plan(*args, **kwargs):
        build_plan_finalize_flags.append(kwargs.get("finalize"))
        return real_build_edit_plan(*args, **kwargs)

    finalized_plans = []

    def mock_finalize_edit_plan(plan, *args, **kwargs):
        finalized_plans.append((plan, kwargs.get("normalize_loudness")))
        return plan

    monkeypatch.setattr(SpanSelector, "select", mock_select)
    monkeypatch.setattr(dialogue_mixer, "build_edit_plan", mock_build_edit_plan)
    monkeypatch.setattr(dialogue_mixer, "finalize_edit_plan", mock_finalize_edit_plan)
    def mock_render(plan, out_path, **kwargs):
        Path(out_path).write_bytes(b"dummy_mp4_bytes")

    monkeypatch.setattr(dialogue_mixer, "render_sentence_mix", mock_render)

    out_video = tmp_path / "scene.mp4"
    synthesize_and_render_dialogue(
        lines=[("anakin", "hello"), ("mace-windu", "there")],
        inventory_index=inv,
        output_video_path=out_video,
        burn_subtitles=False,
    )

    # 1. Feature extractor wired
    assert len(passed_extractors) == 2
    for fe in passed_extractors:
        assert isinstance(fe, ContinuousFeatureExtractor)
        assert fe.sr == 16000

    # 2. Per-turn finalize=False
    assert build_plan_finalize_flags == [False, False]

    # 3. Scene loudnorm on combined_plan exactly once
    assert len(finalized_plans) == 1
    comb_plan, norm_flag = finalized_plans[0]
    assert comb_plan.speaker_id == "multi"
    assert norm_flag is True


def test_concat_edit_plans_video_midpoint_partitioning_no_overcounting():
    """
    Validates Fix P1.3 video midpoint partitioning:
    When plans containing internal 14ms audio crossfades are concatenated,
    video display bounds partition at the crossfade midpoints, ensuring:
      sum(seg.video_frame_count) == total_video_frames
    and adjacent video frame intervals abut without overlapping or gaps.
    """
    fps = RationalTime(24, 1)
    sr = 48000
    ov_samples = 672  # 14ms at 48kHz

    # Turn 1: 2 segments overlapping by 14ms
    seg1 = PlanSegment(
        segment_index=0,
        source_content_id="src1",
        speaker_id="anakin",
        source_start_sample=0,
        source_end_sample=48000,
        output_start_sample=0,
        output_end_sample=48000,
        target_text="seg one",
        target_phones=["S", "EH1", "G"],
        span_type="word",
        source_video_path="video1.mkv",
        gain_db=0.0,
        video_start_frame=0,
        video_frame_count=24,
    )
    seg2 = PlanSegment(
        segment_index=1,
        source_content_id="src1",
        speaker_id="anakin",
        source_start_sample=48000,
        source_end_sample=96000,
        output_start_sample=48000 - ov_samples,  # 47328
        output_end_sample=96000 - ov_samples,  # 95328
        target_text="seg two",
        target_phones=["T", "UW1"],
        span_type="word",
        source_video_path="video1.mkv",
        gain_db=0.0,
        video_start_frame=24,
        video_frame_count=24,
    )
    t1 = PlanTransition(
        transition_index=0,
        from_segment_index=0,
        to_segment_index=1,
        transition_type="overlap",
        overlap_samples=ov_samples,
        fade_samples=ov_samples,
    )
    p1_total_samples = 96000 - ov_samples
    plan1 = EditPlan(
        plan_id="p1",
        target_text="seg one seg two",
        speaker_id="anakin",
        project_sample_rate=sr,
        video_fps=fps,
        total_output_samples=p1_total_samples,
        total_video_frames=48,
        segments=[seg1, seg2],
        transitions=[t1],
        sources={"src1": None},
    )

    # Turn 2: 2 segments overlapping by 14ms
    seg3 = PlanSegment(
        segment_index=0,
        source_content_id="src2",
        speaker_id="mace-windu",
        source_start_sample=0,
        source_end_sample=48000,
        output_start_sample=0,
        output_end_sample=48000,
        target_text="seg three",
        target_phones=["TH", "R", "IY1"],
        span_type="word",
        source_video_path="video2.mkv",
        gain_db=0.0,
        video_start_frame=0,
        video_frame_count=24,
    )
    seg4 = PlanSegment(
        segment_index=1,
        source_content_id="src2",
        speaker_id="mace-windu",
        source_start_sample=48000,
        source_end_sample=96000,
        output_start_sample=48000 - ov_samples,
        output_end_sample=96000 - ov_samples,
        target_text="seg four",
        target_phones=["F", "AO1", "R"],
        span_type="word",
        source_video_path="video2.mkv",
        gain_db=0.0,
        video_start_frame=24,
        video_frame_count=24,
    )
    t2 = PlanTransition(
        transition_index=0,
        from_segment_index=0,
        to_segment_index=1,
        transition_type="overlap",
        overlap_samples=ov_samples,
        fade_samples=ov_samples,
    )
    plan2 = EditPlan(
        plan_id="p2",
        target_text="seg three seg four",
        speaker_id="mace-windu",
        project_sample_rate=sr,
        video_fps=fps,
        total_output_samples=p1_total_samples,
        total_video_frames=48,
        segments=[seg3, seg4],
        transitions=[t2],
        sources={"src2": None},
    )

    combined = concat_edit_plans([plan1, plan2], pause_between_sec=0.5)
    combined.validate()

    # Exact video frame partitioning invariant
    total_frames_sum = sum(s.video_frame_count for s in combined.segments)
    assert total_frames_sum == combined.total_video_frames, (
        f"Sum of video_frame_count ({total_frames_sum}) must strictly equal total_video_frames ({combined.total_video_frames})"
    )

    # Frame continuity: every segment's start_frame must equal the preceding segment's end_frame
    for k in range(len(combined.segments) - 1):
        curr_s = combined.segments[k]
        next_s = combined.segments[k + 1]
        assert curr_s.video_start_frame + curr_s.video_frame_count == next_s.video_start_frame, (
            f"Segment {k} ends at frame {curr_s.video_start_frame + curr_s.video_frame_count} "
            f"but segment {k+1} starts at frame {next_s.video_start_frame}"
        )


