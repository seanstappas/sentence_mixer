"""
Regression test suite for Milestone 5 (B10, U23-U24).
Verifies:
- U23: Rising- and falling-pitch candidates for one word:
  Rising guide chooses the rising candidate; falling guide chooses the falling one;
  costs differ measurably; works without WavLM.
- U24: --guide without transcript; nonexistent transcript path; mismatched transcript:
  Each exits 2.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np
import soundfile as sf

from contracts import (
    AlignedEvent,
    InventoryIndex,
    SourceRecord,
    WordOccurrence,
)
from lexicon import Lexicon
from prosody_guide import ProsodyGuide
from span_selector import SpanSelector
from target_graph import TargetGraphBuilder


def test_u23_rising_and_falling_pitch_candidates_for_one_word(tmp_path: Path):
    """
    U23: Rising- and falling-pitch candidates for one word:
    Rising guide chooses the rising candidate; falling guide chooses the falling one;
    costs differ; works without WavLM.
    """
    sr = 16000
    dur = 0.5
    t = np.linspace(0, dur, int(sr * dur), endpoint=False)

    # 1. Candidate audio
    # Rising candidate: 150 Hz -> 300 Hz
    f_rise = np.linspace(150, 300, len(t))
    sig_rise = (0.3 * np.sin(2 * np.pi * f_rise * t)).astype(np.float32)
    rise_pcm = tmp_path / "cand_rise.wav"
    sf.write(str(rise_pcm), sig_rise, sr)

    # Falling candidate: 300 Hz -> 150 Hz
    f_fall = np.linspace(300, 150, len(t))
    sig_fall = (0.3 * np.sin(2 * np.pi * f_fall * t)).astype(np.float32)
    fall_pcm = tmp_path / "cand_fall.wav"
    sf.write(str(fall_pcm), sig_fall, sr)

    # 2. Guide audio recordings
    # Rising guide: 150 -> 350 Hz
    f_guide_rise = np.linspace(150, 350, len(t))
    sig_guide_rise = (0.4 * np.sin(2 * np.pi * f_guide_rise * t)).astype(np.float32)
    guide_rise_path = tmp_path / "guide_rise.wav"
    sf.write(str(guide_rise_path), sig_guide_rise, sr)

    # Falling guide: 350 -> 150 Hz
    f_guide_fall = np.linspace(350, 150, len(t))
    sig_guide_fall = (0.4 * np.sin(2 * np.pi * f_guide_fall * t)).astype(np.float32)
    guide_fall_path = tmp_path / "guide_fall.wav"
    sf.write(str(guide_fall_path), sig_guide_fall, sr)

    # Build inventory containing both candidates for 'hello'
    inv = InventoryIndex()
    src_rise = SourceRecord(
        content_id="src_rise",
        source_path=str(rise_pcm),
        cached_analysis_pcm_path=str(rise_pcm),
        native_sample_rate=sr,
        native_channels=1,
    )
    src_fall = SourceRecord(
        content_id="src_fall",
        source_path=str(fall_pcm),
        cached_analysis_pcm_path=str(fall_pcm),
        native_sample_rate=sr,
        native_channels=1,
    )
    inv.sources["src_rise"] = src_rise
    inv.sources["src_fall"] = src_fall

    # Word occurrence 1: rising candidate
    wo_rise = WordOccurrence(
        word_id="wo_rise",
        source_content_id="src_rise",
        speaker_id="speaker_a",
        word_normalized="hello",
        word_original="hello",
        start_sample=0,
        end_sample=len(sig_rise),
        phone_event_ids=["ev_r1"],
        realized_phones=["HH", "AH0", "L", "OW1"],
        is_eligible=True,
    )
    inv.words["wo_rise"] = wo_rise
    inv.word_to_word_ids.setdefault("hello", []).append("wo_rise")

    ev_rise = AlignedEvent(
        event_id="ev_r1",
        source_content_id="src_rise",
        utterance_id="utt_1",
        speaker_id="speaker_a",
        kind="phone",
        start_sample=0,
        end_sample=len(sig_rise),
        label="OW1",
        base_phone="OW",
        is_eligible=True,
    )
    inv.events["ev_r1"] = ev_rise
    inv.phone_to_event_ids.setdefault("OW1", []).append("ev_r1")
    inv.base_phone_to_event_ids.setdefault("OW", []).append("ev_r1")

    # Word occurrence 2: falling candidate
    wo_fall = WordOccurrence(
        word_id="wo_fall",
        source_content_id="src_fall",
        speaker_id="speaker_a",
        word_normalized="hello",
        word_original="hello",
        start_sample=0,
        end_sample=len(sig_fall),
        phone_event_ids=["ev_f1"],
        realized_phones=["HH", "AH0", "L", "OW1"],
        is_eligible=True,
    )
    inv.words["wo_fall"] = wo_fall
    inv.word_to_word_ids["hello"].append("wo_fall")

    ev_fall = AlignedEvent(
        event_id="ev_f1",
        source_content_id="src_fall",
        utterance_id="utt_2",
        speaker_id="speaker_a",
        kind="phone",
        start_sample=0,
        end_sample=len(sig_fall),
        label="OW1",
        base_phone="OW",
        is_eligible=True,
    )
    inv.events["ev_f1"] = ev_fall
    inv.phone_to_event_ids.setdefault("OW1", []).append("ev_f1")
    inv.base_phone_to_event_ids.setdefault("OW", []).append("ev_f1")

    lex = Lexicon([])
    lex.word_to_pronunciations["hello"] = [["HH", "AH0", "L", "OW1"]]

    graph_builder = TargetGraphBuilder(lex)
    graph = graph_builder.build("hello")

    selector = SpanSelector(inv)
    candidates = selector.retrieve_candidates(graph)
    word_cands = [c for c in candidates if c.span_type == "word"]
    assert len(word_cands) == 2

    # A. Test with Rising Guide
    guide_rising = ProsodyGuide(guide_rise_path, guide_transcript="hello")
    cost_rise_on_rise = guide_rising.get_candidate_cost(word_cands[0], graph, selector=selector)
    cost_fall_on_rise = guide_rising.get_candidate_cost(word_cands[1], graph, selector=selector)

    # Costs must differ measurably!
    assert cost_rise_on_rise != cost_fall_on_rise
    # Rising candidate must have lower cost under rising guide
    assert cost_rise_on_rise < cost_fall_on_rise

    selected_rise = selector.select(graph, prosody_guide=guide_rising)
    assert len(selected_rise) == 1
    assert selected_rise[0].source_content_id == "src_rise"

    # B. Test with Falling Guide
    guide_falling = ProsodyGuide(guide_fall_path, guide_transcript="hello")
    cost_rise_on_fall = guide_falling.get_candidate_cost(word_cands[0], graph, selector=selector)
    cost_fall_on_fall = guide_falling.get_candidate_cost(word_cands[1], graph, selector=selector)

    assert cost_rise_on_fall != cost_fall_on_fall
    # Falling candidate must have lower cost under falling guide
    assert cost_fall_on_fall < cost_rise_on_fall

    selected_fall = selector.select(graph, prosody_guide=guide_falling)
    assert len(selected_fall) == 1
    assert selected_fall[0].source_content_id == "src_fall"


def test_u24_guide_cli_transcript_validation(tmp_path: Path):
    """
    U24: --guide without transcript; nonexistent transcript path; mismatched transcript:
    Each exits 2.
    """
    repo_root = Path(__file__).resolve().parent.parent
    run_mixer_py = repo_root / "run_mixer.py"
    py_exe = sys.executable

    # Create dummy guide audio
    guide_audio = tmp_path / "guide.wav"
    sf.write(str(guide_audio), np.zeros(16000, dtype=np.float32), 16000)

    # 1. --guide without transcript
    cmd1 = [
        py_exe, str(run_mixer_py), "mix",
        "--text", "hello",
        "--selector", "span",
        "--guide", str(guide_audio),
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert res1.returncode == 2, f"Expected exit code 2 without --guide-transcript, got {res1.returncode}"
    assert "guide-transcript" in res1.stderr

    # 2. Nonexistent transcript path
    nonexistent = tmp_path / "does_not_exist.txt"
    cmd2 = [
        py_exe, str(run_mixer_py), "mix",
        "--text", "hello",
        "--selector", "span",
        "--guide", str(guide_audio),
        "--guide-transcript", str(nonexistent),
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode == 2, f"Expected exit code 2 for nonexistent transcript, got {res2.returncode}"
    assert "not found" in res2.stderr.lower()

    # 3. Mismatched transcript
    mismatch_file = tmp_path / "mismatch.txt"
    mismatch_file.write_text("goodbye world", encoding="utf-8")
    cmd3 = [
        py_exe, str(run_mixer_py), "mix",
        "--text", "hello there",
        "--selector", "span",
        "--guide", str(guide_audio),
        "--guide-transcript", str(mismatch_file),
    ]
    res3 = subprocess.run(cmd3, capture_output=True, text=True)
    assert res3.returncode == 2, f"Expected exit code 2 for mismatched transcript, got {res3.returncode}"
    assert "not match" in res3.stderr.lower()
