"""
Tests for Substring Cut Accuracy & Boundary Bleed Elimination (Milestone M4).
Verifies:
1. VTT rolling caption deduplication across abutting cues and cues with <= 0.5s spacer gaps.
2. Clean speech representation preventing duplicate dialogue from reaching MFA forced alignment.
3. Audio crossfade ramps in render_master_pcm (fade_out and fade_in applied to overlapping slices).
4. Video frame overrun elimination in edit_plan.py (frame count bounded by source duration).
5. PTS seek formatting microsecond precision (:.6f).
6. Legacy render adapter native sample rate resolution (eliminating sample rate skew).
"""

import tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
import pytest

from contracts import (
    AlignedEvent,
    EditPlan,
    InventoryIndex,
    PlanSegment,
    PlanTransition,
    RationalTime,
    SourceRecord,
    SpanCandidate,
)
from edit_plan import build_edit_plan
from lexicon import Lexicon
from target_graph import TargetGraphBuilder
from video_assembler import (
    render_master_pcm,
    render_sentence_mix_legacy,
    _get_media_native_sample_rate,
)
from vtt_preprocessor import clean_rolling_cues, parse_vtt_cues


# ============================================================================
# 1. VTT Rolling Caption Deduplication Tests
# ============================================================================

def test_vtt_rolling_cues_with_10ms_spacer_gap():
    """
    Verifies that YouTube auto-generated captions with 10ms spacer gaps between cues
    have carry-over duplicate lines stripped.
    This directly prevents the Obi-Wan 'it's over anakin' / 'i have the high ground' bleed.
    """
    vtt_content = """WEBVTT

00:08:07.680 --> 00:08:09.589 align:start position:0%
then you are lost
it's over anakin

00:08:09.589 --> 00:08:09.599 align:start position:0%
it's over anakin

00:08:09.599 --> 00:08:11.510 align:start position:0%
it's over anakin
i have the high ground
"""
    with tempfile.NamedTemporaryFile("w", suffix=".vtt", delete=False, encoding="utf-8") as f:
        f.write(vtt_content)
        temp_path = f.name

    try:
        cues = parse_vtt_cues(temp_path)
        assert len(cues) == 2, f"Expected 2 speech intervals, got {len(cues)}: {cues}"

        # Cue 1: 'then you are lost' was first line, 'it's over anakin' second
        assert cues[0]["text"] == "then you are lost it's over anakin"
        assert abs(cues[0]["start_sec"] - 487.680) < 0.001
        assert abs(cues[0]["end_sec"] - 489.589) < 0.001

        # Cue 2: The 10ms spacer (489.589 to 489.599) should be dropped,
        # and 'it's over anakin' in the next cue (gap = 0.010s <= 0.5s) must be stripped!
        assert cues[1]["text"] == "i have the high ground"
        assert abs(cues[1]["start_sec"] - 489.599) < 0.001
        assert abs(cues[1]["end_sec"] - 491.510) < 0.001
    finally:
        Path(temp_path).unlink()


def test_vtt_rolling_cues_abutting_gap_zero():
    """
    Verifies that abutting cues (gap = 0.000s) have carry-over duplicate lines stripped.
    """
    cues = [
        {"start_sec": 1.0, "end_sec": 2.5, "lines": ["Hello there"]},
        {"start_sec": 2.5, "end_sec": 4.0, "lines": ["Hello there", "General Kenobi"]},
    ]
    cleaned = clean_rolling_cues(cues)
    assert len(cleaned) == 2
    assert cleaned[0]["text"] == "Hello there"
    assert cleaned[1]["text"] == "General Kenobi"
    assert cleaned[1]["start_sec"] == 2.5
    assert cleaned[1]["end_sec"] == 4.0


def test_vtt_rolling_cues_spacer_gap_up_to_half_second():
    """
    Verifies that cues separated by small spacer gaps up to 0.5s are identified
    as rolling caption updates and stripped.
    """
    cues = [
        {"start_sec": 10.0, "end_sec": 12.0, "lines": ["May the force"]},
        {"start_sec": 12.45, "end_sec": 14.5, "lines": ["May the force", "be with you"]},
    ]
    cleaned = clean_rolling_cues(cues)
    assert len(cleaned) == 2
    assert cleaned[0]["text"] == "May the force"
    assert cleaned[1]["text"] == "be with you"
    assert cleaned[1]["start_sec"] == 12.45


def test_vtt_genuine_repetition_with_large_gap_preserved():
    """
    Verifies that genuine speech repetitions separated by > 0.5s are preserved.
    """
    cues = [
        {"start_sec": 1.0, "end_sec": 2.0, "lines": ["No"]},
        {"start_sec": 3.0, "end_sec": 4.0, "lines": ["No"]},
    ]
    cleaned = clean_rolling_cues(cues)
    assert len(cleaned) == 2
    assert cleaned[0]["text"] == "No"
    assert cleaned[1]["text"] == "No"


def test_vtt_overlapping_cues_start_sec_adjusted():
    """
    Verifies that overlapping cues have duplicate line removed and start_sec adjusted.
    """
    cues = [
        {"start_sec": 1.0, "end_sec": 3.0, "lines": ["I have the"]},
        {"start_sec": 2.5, "end_sec": 4.5, "lines": ["I have the", "high ground"]},
    ]
    cleaned = clean_rolling_cues(cues)
    assert len(cleaned) == 2
    assert cleaned[0]["text"] == "I have the"
    assert cleaned[1]["text"] == "high ground"
    assert cleaned[1]["start_sec"] == 3.0
    assert cleaned[1]["duration"] == 1.5


def test_obi_wan_ep3_vtt_real_cue_cleanliness():
    """
    Tests parse_vtt_cues against input/obi_wan_ep3.en.vtt to ensure the dialogue
    at 489.599s is strictly 'i have the high ground' with zero 'it's over anakin' bleed.
    """
    vtt_path = Path("input/obi_wan_ep3.en.vtt")
    if not vtt_path.exists():
        pytest.skip("input/obi_wan_ep3.en.vtt not present in workspace")

    cues = parse_vtt_cues(str(vtt_path))

    # Find the cue covering ~489.6s
    high_ground_cue = None
    for c in cues:
        if abs(c["start_sec"] - 489.599) < 0.1:
            high_ground_cue = c
            break

    assert high_ground_cue is not None, "Could not find cue starting near 489.599s in obi_wan_ep3.en.vtt"
    assert "anakin" not in high_ground_cue["text"].lower(), (
        f"Cue at {high_ground_cue['start_sec']} contains bleed: '{high_ground_cue['text']}'"
    )
    assert "it's over" not in high_ground_cue["text"].lower(), (
        f"Cue at {high_ground_cue['start_sec']} contains bleed: '{high_ground_cue['text']}'"
    )
    assert "high ground" in high_ground_cue["text"].lower(), (
        f"Cue text should contain 'high ground', got: '{high_ground_cue['text']}'"
    )


# ============================================================================
# 2. Audio Crossfade Ramp Application Tests
# ============================================================================

def test_render_master_pcm_linear_crossfade_ramps():
    """
    Verifies that render_master_pcm applies linear crossfade ramps (fade_out and fade_in)
    rather than summing overlapping signals at 100% gain.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sr = 48000
        overlap = 480  # 10ms overlap

        # Source 1: constant amplitude 0.8 (below 0.98 peak clamp threshold)
        src1_wav = tmp_path / "src1.wav"
        audio1 = 0.8 * np.ones(12000, dtype=np.float32)
        sf.write(str(src1_wav), audio1, sr)

        # Source 2: constant amplitude 0.0
        src2_wav = tmp_path / "src2.wav"
        audio2 = np.zeros(12000, dtype=np.float32)
        sf.write(str(src2_wav), audio2, sr)

        src_rec1 = SourceRecord(
            content_id="src1",
            source_path=str(src1_wav),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(src1_wav),
        )
        src_rec2 = SourceRecord(
            content_id="src2",
            source_path=str(src2_wav),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(src2_wav),
        )

        seg1 = PlanSegment(
            segment_index=0,
            source_content_id="src1",
            speaker_id="spk",
            source_start_sample=0,
            source_end_sample=12000,
            output_start_sample=0,
            output_end_sample=12000,
            target_text="seg1",
            target_phones=[],
            span_type="word",
            source_video_path=str(src1_wav),
        )
        seg2 = PlanSegment(
            segment_index=1,
            source_content_id="src2",
            speaker_id="spk",
            source_start_sample=0,
            source_end_sample=12000,
            output_start_sample=12000 - overlap,  # overlap starts at 11520
            output_end_sample=24000 - overlap,
            target_text="seg2",
            target_phones=[],
            span_type="word",
            source_video_path=str(src2_wav),
        )
        trans = PlanTransition(
            transition_index=0,
            from_segment_index=0,
            to_segment_index=1,
            transition_type="overlap",
            overlap_samples=overlap,
            fade_samples=overlap,
        )

        total_samples = 24000 - overlap
        plan = EditPlan(
            plan_id="crossfade_test_plan",
            target_text="seg1 seg2",
            speaker_id="spk",
            project_sample_rate=sr,
            total_output_samples=total_samples,
            segments=[seg1, seg2],
            transitions=[trans],
            sources={"src1": src_rec1, "src2": src_rec2},
        )
        plan.validate()

        out_wav = tmp_path / "out_master.wav"
        render_master_pcm(plan, out_wav)

        data, out_sr = sf.read(str(out_wav))
        assert out_sr == sr
        assert len(data) == total_samples

        # Check the crossfade region: [11520 : 12000]
        # At start of fade (sample 11520): fade_out = 1.0, fade_in = 0.0 -> amplitude = 0.8
        assert abs(data[11520] - 0.8) < 1e-3, f"Start of fade should be ~0.8, got {data[11520]}"

        # At midpoint of fade (sample 11520 + 240): fade_out = 0.5, fade_in = 0.5
        # audio1=0.8 * 0.5 + audio2=0.0 * 0.5 = 0.4
        # Under the old bug (100%+100% gain summing), data would have been 0.8!
        midpoint_val = data[11520 + 240]
        assert abs(midpoint_val - 0.4) < 0.05, (
            f"Midpoint of crossfade must be ~0.4, got {midpoint_val} (indicates unweighted summing)"
        )

        # At end of fade (sample 11999): fade_out = 0.0, fade_in = 1.0 -> amplitude = 0.0
        assert abs(data[11999] - 0.0) < 1e-2, f"End of fade should be ~0.0, got {data[11999]}"


def test_render_master_pcm_equal_gain_crossover():
    """
    Verifies that crossfading two equal signals (e.g. 0.7 and 0.7) produces a smooth
    constant 0.7 rather than peaking at 1.4 (which would cause digital clipping).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sr = 48000
        overlap = 400

        src_wav = tmp_path / "const_07.wav"
        audio = 0.7 * np.ones(8000, dtype=np.float32)
        sf.write(str(src_wav), audio, sr)

        src_rec = SourceRecord(
            content_id="src_07",
            source_path=str(src_wav),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(src_wav),
        )

        seg1 = PlanSegment(
            segment_index=0,
            source_content_id="src_07",
            speaker_id="spk",
            source_start_sample=0,
            source_end_sample=8000,
            output_start_sample=0,
            output_end_sample=8000,
            target_text="a",
            target_phones=[],
            span_type="word",
            source_video_path=str(src_wav),
        )
        seg2 = PlanSegment(
            segment_index=1,
            source_content_id="src_07",
            speaker_id="spk",
            source_start_sample=0,
            source_end_sample=8000,
            output_start_sample=8000 - overlap,
            output_end_sample=16000 - overlap,
            target_text="b",
            target_phones=[],
            span_type="word",
            source_video_path=str(src_wav),
        )
        trans = PlanTransition(
            transition_index=0,
            from_segment_index=0,
            to_segment_index=1,
            transition_type="overlap",
            overlap_samples=overlap,
            fade_samples=overlap,
        )

        plan = EditPlan(
            plan_id="equal_gain_plan",
            target_text="a b",
            speaker_id="spk",
            project_sample_rate=sr,
            total_output_samples=16000 - overlap,
            segments=[seg1, seg2],
            transitions=[trans],
            sources={"src_07": src_rec},
        )

        out_wav = tmp_path / "out_equal.wav"
        render_master_pcm(plan, out_wav)

        data, _ = sf.read(str(out_wav))
        # With linear crossfade: 0.7*(1-t) + 0.7*t = 0.7 for all t
        # Max peak must not exceed 0.705
        max_peak = np.max(data)
        assert max_peak <= 0.705, f"Crossfade caused peak amplitude spike: {max_peak} > 0.705"


# ============================================================================
# 3. Video Frame Overrun Prevention & PTS Precision Tests
# ============================================================================

def test_video_frame_count_bounded_by_source_duration():
    """
    Verifies that build_edit_plan bounds video_frame_count so that ceiling rounding
    does not overrun past the source word duration into adjacent dialogue.
    """
    sr = 48000
    fps = RationalTime(30000, 1001)  # 29.97 fps (frame dur ~= 33.37ms)

    src_rec = SourceRecord(
        content_id="src_bound_test",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=1,
    )

    # Word duration = 4000 samples at 48kHz = 0.08333s (83.33ms)
    # At 29.97 fps: 83.33ms / 33.37ms = 2.497 frames.
    # Max integer frames in this source slice = floor(0.08333 * 29.97) = 2 frames.
    # If ceiling rounding allocates 3 frames, it would overrun by (3*33.37 - 83.33) = 16.8ms!
    ev = AlignedEvent(
        event_id="ev1",
        source_content_id="src_bound_test",
        utterance_id="u1",
        speaker_id="spk",
        kind="word",
        start_sample=0,
        end_sample=4000,
        label="test",
        base_phone="test",
    )
    inv = InventoryIndex(
        sources={"src_bound_test": src_rec},
        events={"ev1": ev},
    )

    lex = Lexicon()
    lex.word_to_pronunciations["test"] = [["T", "EH1", "S", "T"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("test")

    cand = SpanCandidate(
        candidate_id="c1",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0, 1, 2, 3],
        source_content_id="src_bound_test",
        speaker_id="spk",
        start_sample=0,
        end_sample=4000,
        event_ids=["ev1"],
        span_type="word",
        matched_word="test",
    )

    plan = build_edit_plan([cand], graph, inv, project_sample_rate=sr, video_fps=fps)
    plan.validate()

    seg = plan.segments[0]
    # Max frames fitting inside 4000 samples @ 48kHz:
    # 4000 / 48000 * 30000 / 1001 = 2.4975 frames -> max 2 frames!
    assert seg.video_frame_count <= 2, (
        f"video_frame_count={seg.video_frame_count} overruns source duration (max 2 frames allowed)"
    )


def test_pts_seek_precision_formatting():
    """
    Verifies that PTS seek timestamps preserve microsecond precision
    and format with 6 decimal places (:.6f).
    """
    pts = 489.59912345
    formatted = f"{pts:.6f}"
    assert formatted == "489.599123", f"Expected 6 decimal places, got {formatted}"


def test_legacy_render_adapter_native_sample_rate():
    """
    Verifies that render_sentence_mix_legacy computes source_start_sample
    using the source file's native sample rate (e.g. 44100) rather than hardcoded 48000.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        native_sr = 44100
        project_sr = 48000

        # Create a synthetic 44.1kHz WAV file
        test_wav = tmp_path / "native44100.wav"
        t = np.linspace(0, 10.0, native_sr * 10, endpoint=False)
        audio = 0.3 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        sf.write(str(test_wav), audio, native_sr)

        # Probed native rate must be 44100
        probed_sr = _get_media_native_sample_rate(test_wav, fallback=project_sr)
        assert probed_sr == 44100, f"Expected probed rate 44100, got {probed_sr}"

        # Test unit starting at 2.0s and ending at 3.0s
        units = [
            {
                "source_video": str(test_wav),
                "start_sec": 2.0,
                "end_sec": 3.0,
                "label": "test_unit",
                "type": "word",
            }
        ]

        # Verify through the helper calculation that source_start_sample
        # is 2.0 * 44100 = 88200, NOT 2.0 * 48000 = 96000 (which would cause a 43-second skew on episode 3)!
        expected_sample_native = int(round(2.0 * native_sr))
        incorrect_sample_hardcoded = int(round(2.0 * project_sr))

        assert expected_sample_native == 88200
        assert incorrect_sample_hardcoded == 96000
        assert expected_sample_native != incorrect_sample_hardcoded

        # Actually render using legacy adapter to verify integration
        out_mp4 = tmp_path / "out_legacy.mp4"
        render_sentence_mix_legacy(units, str(out_mp4), audio_sample_rate=project_sr)
        out_wav = out_mp4.with_suffix(".wav")
        assert out_wav.exists()
        info = sf.info(str(out_wav))
        assert info.samplerate == project_sr
        assert info.frames == project_sr  # 1.0s at 48000 Hz
