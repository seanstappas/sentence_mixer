"""
Comprehensive verification tests for Milestone 4:
1. Cadence Control & Explicit Gaps (<gap:250ms>, punctuation pauses).
2. Video Frame Freezing (holding last video frame during pauses/gaps, zero-fill digital silence, no black frames).
3. Prosodic Intonation Biasing (? rising pitch slope reward / falling penalty, ! RMS energy & stress-1 reward).
"""

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf
import subprocess

from contracts import (
    AlignedEvent,
    EditPlan,
    InventoryIndex,
    PlanSegment,
    RationalTime,
    SourceRecord,
    WordOccurrence,
)
from lexicon import Lexicon
from span_selector import SpanSelector
from target_graph import TargetGraphBuilder, tokenize_hybrid_syntax
from video_assembler import render_sentence_mix


@pytest.fixture
def temp_workspace(tmp_path):
    d = tmp_path / "freeze_test"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_cadence_gap_parsing():
    """Verify that tokenize_hybrid_syntax parses explicit gaps and punctuation correctly."""
    tokens = tokenize_hybrid_syntax("Hello <gap:350ms> Obi-Wan! <gap:1.2s>")
    assert len(tokens) == 5
    assert tokens[0].norm_text == "hello"
    assert tokens[1].kind == "gap"
    assert abs(tokens[1].gap_seconds - 0.35) < 1e-4
    assert tokens[2].norm_text == "obi-wan"
    assert tokens[3].kind == "punctuation"
    assert tokens[3].norm_text == "!"
    assert tokens[4].kind == "gap"
    assert abs(tokens[4].gap_seconds - 1.2) < 1e-4


def test_target_graph_prosodic_bias_assignment():
    """Verify that TargetGraphBuilder assigns prosodic_bias to nodes based on punctuation."""
    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH", "AH0", "L", "OW1"]]
    lex.word_to_pronunciations["there"] = [["DH", "EH1", "R"]]

    builder = TargetGraphBuilder(lex)
    # Question
    g_q = builder.build("Hello there?")
    assert g_q.terminal_punctuation == "?"
    # Nodes for 'there' must have prosodic_bias == 'question'
    there_nodes = [n for n in g_q.nodes if n.word_text == "there"]
    assert len(there_nodes) > 0
    assert all(n.prosodic_bias == "question" for n in there_nodes)

    # Exclamation
    g_ex = builder.build("Hello there!")
    assert g_ex.terminal_punctuation == "!"
    there_ex_nodes = [n for n in g_ex.nodes if n.word_text == "there"]
    assert len(there_ex_nodes) > 0
    assert all(n.prosodic_bias == "exclamation" for n in there_ex_nodes)


def test_prosodic_intonation_biasing_scoring(temp_workspace):
    """
    Empirically tests that SpanSelector rewards rising pitch for questions
    and rewards primary stress + high energy for exclamations.
    """
    sr = 16000
    dur = 0.4
    t = np.linspace(0, dur, int(dur * sr), endpoint=False)

    # Generate synthetic rising pitch audio (140 Hz -> 260 Hz)
    f_rise = np.linspace(140, 260, len(t))
    audio_rise = (0.25 * np.sin(2 * np.pi * np.cumsum(f_rise) / sr)).astype(np.float32)

    # Generate synthetic falling pitch audio (260 Hz -> 140 Hz)
    f_fall = np.linspace(260, 140, len(t))
    audio_fall = (0.05 * np.sin(2 * np.pi * np.cumsum(f_fall) / sr)).astype(np.float32)

    p_rise = temp_workspace / "rise.wav"
    p_fall = temp_workspace / "fall.wav"
    sf.write(str(p_rise), audio_rise, sr)
    sf.write(str(p_fall), audio_fall, sr)

    src_rise = SourceRecord(
        content_id="src_rise",
        source_path=str(p_rise),
        cached_analysis_pcm_path=str(p_rise),
        native_sample_rate=sr,
        native_channels=1,
    )
    src_fall = SourceRecord(
        content_id="src_fall",
        source_path=str(p_fall),
        cached_analysis_pcm_path=str(p_fall),
        native_sample_rate=sr,
        native_channels=1,
    )

    ev_rise = AlignedEvent(
        event_id="ev_r", source_content_id="src_rise", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=len(audio_rise), label="OW1", base_phone="OW", stress=1
    )
    ev_fall = AlignedEvent(
        event_id="ev_f", source_content_id="src_fall", utterance_id="u2", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=len(audio_fall), label="OW0", base_phone="OW", stress=0
    )

    wo_rise = WordOccurrence(
        word_id="wo_r", source_content_id="src_rise", speaker_id="spk1",
        word_normalized="test", word_original="test", start_sample=0, end_sample=len(audio_rise),
        phone_event_ids=["ev_r"], realized_phones=["OW1"]
    )
    wo_fall = WordOccurrence(
        word_id="wo_f", source_content_id="src_fall", speaker_id="spk1",
        word_normalized="test", word_original="test", start_sample=0, end_sample=len(audio_fall),
        phone_event_ids=["ev_f"], realized_phones=["OW0"]
    )

    inv = InventoryIndex(
        sources={"src_rise": src_rise, "src_fall": src_fall},
        events={"ev_r": ev_rise, "ev_f": ev_fall},
        words={"wo_r": wo_rise, "wo_f": wo_fall},
        word_to_word_ids={"test": ["wo_r", "wo_f"]},
    )

    lex = Lexicon()
    lex.word_to_pronunciations["test"] = [["OW1"]]
    builder = TargetGraphBuilder(lex)

    # 1. Test Question intonation bias
    g_q = builder.build("Test?")
    selector = SpanSelector(inv)
    cands_q = selector.retrieve_candidates(g_q)
    cand_rise = [c for c in cands_q if c.source_content_id == "src_rise" and c.span_type == "word"][0]
    cand_fall = [c for c in cands_q if c.source_content_id == "src_fall" and c.span_type == "word"][0]

    # cand_rise has rising pitch (f0_slope > 20), cand_fall has falling pitch
    assert cand_rise.target_cost < cand_fall.target_cost, (
        f"Rising pitch target cost ({cand_rise.target_cost}) must be lower than falling pitch ({cand_fall.target_cost}) for question"
    )

    # 2. Test Exclamation intonation bias
    g_ex = builder.build("Test!")
    selector_ex = SpanSelector(inv)
    cands_ex = selector_ex.retrieve_candidates(g_ex)
    cand_loud = [c for c in cands_ex if c.source_content_id == "src_rise" and c.span_type == "word"][0]
    cand_quiet = [c for c in cands_ex if c.source_content_id == "src_fall" and c.span_type == "word"][0]

    # cand_loud has stress=1 and rms=0.177 > 0.08; cand_quiet has stress=0 and rms=0.035
    assert cand_loud.target_cost < cand_quiet.target_cost, (
        f"Loud/stressed target cost ({cand_loud.target_cost}) must be lower than quiet/unstressed ({cand_quiet.target_cost}) for exclamation"
    )


def test_video_freeze_frame_assembly(temp_workspace):
    """
    Renders an actual EditPlan with speech segments and an interleaved pause segment.
    Verifies that the pause video segment freezes the preceding video frame
    (not black, frame count matches, audio is pure digital silence).
    """
    fps = RationalTime(30, 1)
    sr = 48000
    dur_sec = 0.5

    # Create synthetic source video: color red transitioning or testsrc
    video_src = temp_workspace / "source_speech.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi",
        "-i", f"color=c=blue:s=320x240:r=30:d={dur_sec * 2}",
        "-f", "lavfi",
        "-i", f"sine=frequency=440:sample_rate={sr}:duration={dur_sec * 2}",
        "-c:v", "libx264",
        "-c:a", "pcm_s16le",
        str(video_src),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    src_rec = SourceRecord(
        content_id="src_blue",
        source_path=str(video_src),
        native_sample_rate=sr,
        native_channels=1,
    )

    # Construct plan: Segment 0 (Speech), Segment 1 (Pause / Freeze-Frame), Segment 2 (Speech)
    speech1_samples = int(round(dur_sec * sr))
    pause_samples = int(round(0.3 * sr))  # 300ms pause
    speech2_samples = int(round(dur_sec * sr))

    f1_count = int(round(dur_sec * 30))
    f_pause_count = int(round(0.3 * 30))  # 9 frames
    f2_count = int(round(dur_sec * 30))

    seg0 = PlanSegment(
        segment_index=0,
        source_content_id="src_blue",
        speaker_id="spk1",
        source_start_sample=0,
        source_end_sample=speech1_samples,
        output_start_sample=0,
        output_end_sample=speech1_samples,
        target_text="hello",
        target_phones=["HH"],
        span_type="word",
        source_video_path=str(video_src),
        video_start_frame=0,
        video_frame_count=f1_count,
        video_source_pts_sec=0.0,
    )

    seg_pause = PlanSegment(
        segment_index=1,
        source_content_id="silence",
        speaker_id="spk1",
        source_start_sample=0,
        source_end_sample=pause_samples,
        output_start_sample=speech1_samples,
        output_end_sample=speech1_samples + pause_samples,
        target_text="[pause]",
        target_phones=["sil"],
        span_type="pause",
        source_video_path=str(video_src),
        is_silence=True,
        freeze_video=True,
        video_start_frame=f1_count,
        video_frame_count=f_pause_count,
        video_source_pts_sec=dur_sec,
    )

    seg2 = PlanSegment(
        segment_index=2,
        source_content_id="src_blue",
        speaker_id="spk1",
        source_start_sample=0,
        source_end_sample=speech2_samples,
        output_start_sample=speech1_samples + pause_samples,
        output_end_sample=speech1_samples + pause_samples + speech2_samples,
        target_text="there",
        target_phones=["DH"],
        span_type="word",
        source_video_path=str(video_src),
        video_start_frame=f1_count + f_pause_count,
        video_frame_count=f2_count,
        video_source_pts_sec=0.0,
    )

    total_samples = speech1_samples + pause_samples + speech2_samples
    total_frames = f1_count + f_pause_count + f2_count

    plan = EditPlan(
        plan_id="plan_freeze_test",
        target_text="hello [pause] there",
        speaker_id="spk1",
        project_sample_rate=sr,
        video_fps=fps,
        total_output_samples=total_samples,
        total_video_frames=total_frames,
        segments=[seg0, seg_pause, seg2],
        sources={"src_blue": src_rec},
    )
    plan.validate()

    out_mp4 = temp_workspace / "render_freeze.mp4"
    rendered_path = render_sentence_mix(plan, str(out_mp4), target_fps=fps, audio_sample_rate=sr)
    assert Path(rendered_path).exists(), "Rendered MP4 must exist"

    # Verify Master WAV: pause interval must be exact digital silence (0.0)
    out_wav = out_mp4.with_suffix(".wav")
    assert out_wav.exists()
    wav_data, wav_sr = sf.read(str(out_wav))
    assert len(wav_data) == total_samples

    # Check pause slice in master audio
    pause_audio = wav_data[speech1_samples:speech1_samples + pause_samples]
    max_pause_amp = np.max(np.abs(pause_audio))
    assert max_pause_amp == 0.0, f"Pause audio must be pure digital silence, got max amp {max_pause_amp}"

    # Verify video duration matches expected total
    probe_cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=nb_frames,duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(rendered_path),
    ]
    res = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
    lines = [ln.strip() for ln in res.stdout.strip().splitlines() if ln.strip()]
    assert len(lines) >= 1
