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
)
from contracts import EditPlan, PlanSegment, RationalTime


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

