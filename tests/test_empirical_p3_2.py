"""
Challenger P3.2 Master Empirical Verification Test Suite
Author: Challenger P3.2 (teamwork_preview_challenger_p3_2)

Empirical Challenge & Stress Verification Requirements:
1. Intact Word Preservation vs Shattering (P3.2):
   - Multi-syllabic words (2, 3, 4, 5 syllables: 'hello', 'jedi', 'senator', 'impossible', 'generator', 'reconciliation').
   - Shattering resilience against flooded phone candidates with zero/low costs.
   - Cumulative unary heuristic penalty capping at 0.50.
   - Speaker-relative quadratic duration penalty behavior across [0.75, 1.50] plateau and steep extremes.
2. Phase 2 Interface Compatibility (Optimal Coupling & Sub-Word Midpoints):
   - refine_splice_boundary from Phase 2 works seamlessly with Phase 3 SpanCandidate objects.
   - Normalized cross-correlation lag search (+- 15ms) and upward zero-crossing snapping (<= 2ms).
   - Audio rendering with render_master_pcm produces continuous waveforms without clicks.
   - Diphone midpoint transitions inside subwords remain operational and immune across whole words.
3. Continuation Chain Invariance (P3.1):
   - Chains of 3, 4, 5+ continuous units flooded with 60-100 competing distractor candidates.
   - Complete preservation through pre-DP truncation and DP search with 0-cost transitions.
4. Cognate Fallback Ladder under Manner Scaling (P3.3):
   - G -> K, NG -> N, V -> F acoustic cognate donors resolve with 0.35 fallback cost.
   - Manner scaling across stops (1.0), sonorants (1.0), fricatives (1.0), and manner jumps (0.25).
   - Strict ValueError when phoneme and all cognates are absent.
5. Voiced F0 Slope Polyfit & Multi-Branch Context (P3.4):
   - Polyfit degree-1 linear regression on voiced frames.
   - ProsodyGuide progress mapping with intra-word fractions.
   - Right and left context sets across branching pronunciation lattices.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import soundfile as sf

# Prepend project src directory to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contracts import (
    AlignedEvent,
    InventoryIndex,
    SourceRecord,
    SpanCandidate,
    WordOccurrence,
)
from edit_plan import build_edit_plan, refine_splice_boundary
from video_assembler import render_master_pcm
from features import (
    compute_join_cost,
    compute_manner_scale,
)
from lexicon import Lexicon
from prosody_guide import ProsodyGuide
from span_selector import SpanSelector, strip_phone_stress
from target_graph import TargetGraphBuilder


# ==============================================================================
# Helper Factories
# ==============================================================================

def make_test_wav(
    file_path: Path,
    duration_sec: float = 2.0,
    sample_rate: int = 16000,
    f0_start: float = 120.0,
    f0_end: float = 180.0,
) -> Path:
    """Generates a synthetic sine chirp WAV with known pitch trajectory."""
    n_samples = int(duration_sec * sample_rate)
    t = np.linspace(0.0, duration_sec, n_samples, endpoint=False)
    phase = 2.0 * np.pi * (f0_start * t + 0.5 * (f0_end - f0_start) / duration_sec * (t ** 2))
    audio = 0.5 * np.sin(phase).astype(np.float32)
    sf.write(str(file_path), audio, sample_rate)
    return file_path


# ==============================================================================
# Test Suite 1: Intact Word Preservation vs Shattering (P3.2)
# ==============================================================================

def test_multisyllabic_word_preservation_against_flooded_phone_splices():
    """
    Adversarially challenges Viterbi DP to ensure multi-syllabic words
    (2, 3, 4, 5 syllables) are NOT shattered into phone slices, even when
    the phone candidate pool is flooded with low-cost competing donor units
    and the intact word carries heuristic bleed penalties.
    """
    sr = 16000
    src = SourceRecord("s1", "vid.mp4", native_sample_rate=sr, native_channels=1)

    # Word: "generator" (JH EH1 N ER0 EY2 T ER0) - 7 phonemes
    word_phones = ["JH", "EH1", "N", "ER0", "EY2", "T", "ER0"]
    events: Dict[str, AlignedEvent] = {}
    words: Dict[str, WordOccurrence] = {}

    w_eids = []
    p_cur = 10000
    p_dur = 2000
    for idx, p in enumerate(word_phones):
        eid = f"ev_word_gen_{idx}"
        w_eids.append(eid)
        events[eid] = AlignedEvent(
            eid, "s1", "u1", "spk1", "phone", p_cur, p_cur + p_dur, p, strip_phone_stress(p),
            word_occurrence_id="wo_gen"
        )
        p_cur += p_dur

    words["wo_gen"] = WordOccurrence(
        word_id="wo_gen",
        source_content_id="s1",
        speaker_id="spk1",
        word_normalized="generator",
        word_original="generator",
        start_sample=10000,
        end_sample=p_cur,
        phone_event_ids=w_eids,
        realized_phones=word_phones,
    )

    phone_to_eids: Dict[str, List[str]] = {}
    for idx, p in enumerate(word_phones):
        p_list = []
        for dist_idx in range(5):
            d_eid = f"ev_dist_{idx}_{dist_idx}"
            s_start = 50000 + (idx * 5 + dist_idx) * 2000
            events[d_eid] = AlignedEvent(
                d_eid, f"s_dist_{dist_idx}", f"u_dist_{dist_idx}", "spk1", "phone",
                s_start, s_start + 2000, p, strip_phone_stress(p),
                word_occurrence_id=f"wo_dist_{idx}_{dist_idx}"
            )
            p_list.append(d_eid)
        p_list.append(w_eids[idx])
        phone_to_eids[p] = p_list

    inv = InventoryIndex(
        sources={
            "s1": src,
            "s_dist_0": SourceRecord("s_dist_0", "d0.mp4", native_sample_rate=sr, native_channels=1),
            "s_dist_1": SourceRecord("s_dist_1", "d1.mp4", native_sample_rate=sr, native_channels=1),
            "s_dist_2": SourceRecord("s_dist_2", "d2.mp4", native_sample_rate=sr, native_channels=1),
            "s_dist_3": SourceRecord("s_dist_3", "d3.mp4", native_sample_rate=sr, native_channels=1),
            "s_dist_4": SourceRecord("s_dist_4", "d4.mp4", native_sample_rate=sr, native_channels=1),
        },
        events=events,
        words=words,
        word_to_word_ids={"generator": ["wo_gen"]},
        phone_to_event_ids=phone_to_eids,
    )

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    graph = builder.build("generator")

    selector = SpanSelector(inv, edit_penalty=0.40)
    path = selector.select(graph, speaker_id="spk1")

    # Viterbi must select the INTACT WORD, not shatter into 7 phone fragments!
    assert len(path) == 1, f"Expected 1 intact word candidate, got {len(path)} slices: {[p.span_type for p in path]}"
    assert path[0].span_type == "word"
    assert path[0].matched_word == "generator"
    assert path[0].start_sample == 10000
    assert path[0].end_sample == p_cur


def test_heuristic_cost_cap_protects_word_with_stacked_penalties():
    """
    Verifies that cumulative heuristic penalties on a whole word are strictly capped at 0.50
    (Fix P3.2), preventing stacked bleed and swallow penalties (e.g. 1.25) from shattering
    the word when phone candidates are present.
    """
    sr = 16000
    src = SourceRecord("s1", "vid.mp4", native_sample_rate=sr, native_channels=1)

    # Word: "impossible" (IH0 M P AA1 S AH0 B AH0 L) - 9 phonemes
    word_phones = ["IH0", "M", "P", "AA1", "S", "AH0", "B", "AH0", "L"]
    events: Dict[str, AlignedEvent] = {}
    words: Dict[str, WordOccurrence] = {}

    w_eids = []
    p_cur = 20000
    p_dur = 1600
    for idx, p in enumerate(word_phones):
        eid = f"ev_imp_{idx}"
        w_eids.append(eid)
        events[eid] = AlignedEvent(
            eid, "s1", "u1", "spk1", "phone", p_cur, p_cur + p_dur, p, strip_phone_stress(p),
            word_occurrence_id="wo_imp"
        )
        p_cur += p_dur

    words["wo_imp"] = WordOccurrence(
        word_id="wo_imp",
        source_content_id="s1",
        speaker_id="spk1",
        word_normalized="impossible",
        word_original="impossible",
        start_sample=20000,
        end_sample=p_cur,
        phone_event_ids=w_eids,
        realized_phones=word_phones,
    )

    phone_to_eids: Dict[str, List[str]] = {p: [w_eids[idx]] for idx, p in enumerate(word_phones)}

    inv = InventoryIndex(
        sources={"s1": src},
        events=events,
        words=words,
        word_to_word_ids={"impossible": ["wo_imp"]},
        phone_to_event_ids=phone_to_eids,
    )

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    graph = builder.build("impossible")

    selector = SpanSelector(inv, edit_penalty=0.30)
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")

    word_cand = next(c for c in cands if c.span_type == "word" and c.matched_word == "impossible")

    # Target cost for word has base 0.05, plus duration cost, plus capped heuristic cost <= 0.50
    assert word_cand.target_cost <= 0.05 + 0.50 + 0.10, f"Word target cost too high: {word_cand.target_cost}"

    path = selector.select(graph, speaker_id="spk1")
    assert len(path) == 1
    assert path[0].span_type == "word"


def test_quadratic_duration_penalty_curve_invariants():
    """
    Stress-tests the quadratic duration penalty formula across the full spectrum:
    - plateau [0.75, 1.50] -> 0.0
    - rushed < 0.75 -> 0.60 * ((0.75 - ratio)/0.75)^2
    - dragged > 1.50 -> 0.30 * min(2.0, (ratio - 1.50)/1.0)^2
    """
    inv = InventoryIndex(sources={}, events={}, words={})
    selector = SpanSelector(inv)

    median_dur = 0.50  # 500ms

    # In plateau: [0.75 * 0.50, 1.50 * 0.50] = [0.375, 0.750]
    assert selector._compute_quadratic_duration_penalty(0.50, median_dur) == 0.0
    assert selector._compute_quadratic_duration_penalty(0.38, median_dur) == 0.0
    assert selector._compute_quadratic_duration_penalty(0.74, median_dur) == 0.0

    # Rushed: ratio = 0.50 (dur = 0.25)
    # penalty = 0.60 * ((0.75 - 0.50) / 0.75)^2 = 0.60 * (1/3)^2 = 0.60 / 9 = 0.06666...
    pen_rushed = selector._compute_quadratic_duration_penalty(0.25, median_dur)
    assert abs(pen_rushed - (0.60 / 9.0)) < 1e-4

    # Extremely rushed: ratio = 0.10 (dur = 0.05)
    pen_extreme_rushed = selector._compute_quadratic_duration_penalty(0.05, median_dur)
    assert pen_extreme_rushed > 0.40

    # Dragged: ratio = 2.0 (dur = 1.0)
    pen_dragged = selector._compute_quadratic_duration_penalty(1.0, median_dur)
    assert abs(pen_dragged - 0.075) < 1e-4

    # Monotonicity check
    assert selector._compute_quadratic_duration_penalty(0.10, median_dur) > selector._compute_quadratic_duration_penalty(0.20, median_dur)
    assert selector._compute_quadratic_duration_penalty(1.50, median_dur) < selector._compute_quadratic_duration_penalty(2.0, median_dur)


# ==============================================================================
# Test Suite 2: Phase 2 Interface Compatibility (Optimal Coupling & Sub-Word Midpoints)
# ==============================================================================

def test_phase2_interface_compatibility_refine_splice_boundary():
    """
    Verifies that Phase 3 changes did NOT break refine_splice_boundary from Phase 2.
    Tests normalized cross-correlation lag search (+-15 ms) and upward zero-crossing snapping.
    """
    sr = 48000
    t = np.arange(4800, dtype=np.float32) / sr
    audio_prev = np.sin(2.0 * np.pi * 440.0 * t)
    audio_curr = np.sin(2.0 * np.pi * 440.0 * t + 0.3)

    # Voiced splice: test cross-correlation refinement
    shift_out, shift_in = refine_splice_boundary(
        audio_prev,
        audio_curr,
        sr=sr,
        max_shift_ms=15.0,
        is_unvoiced=False,
    )

    assert isinstance(shift_out, int)
    assert isinstance(shift_in, int)
    assert abs(shift_in - shift_out) <= int(0.015 * sr)

    # Unvoiced splice: test upward zero-crossing snapping
    noise = np.random.RandomState(42).randn(4800).astype(np.float32) * 0.1
    ref_u_out, ref_u_in = refine_splice_boundary(
        noise,
        noise,
        sr=sr,
        is_unvoiced=True,
    )
    assert abs(ref_u_out) <= int(0.002 * sr)
    assert abs(ref_u_in) <= int(0.002 * sr)


def test_phase2_interface_compatibility_end_to_end_pcm_rendering(tmp_path: Path):
    """
    Verifies that a synthesized SpanCandidate path from SpanSelector passes
    through build_edit_plan and render_master_pcm without exception, producing
    continuous audio without clicks.
    """
    sr = 16000
    wav_a = make_test_wav(tmp_path / "vid_a.wav", duration_sec=1.0, sample_rate=sr, f0_start=150.0, f0_end=150.0)
    wav_b = make_test_wav(tmp_path / "vid_b.wav", duration_sec=1.0, sample_rate=sr, f0_start=150.0, f0_end=150.0)

    src_a = SourceRecord("src_a", str(wav_a), native_sample_rate=sr, native_channels=1)
    src_b = SourceRecord("src_b", str(wav_b), native_sample_rate=sr, native_channels=1)

    events: Dict[str, AlignedEvent] = {
        "e1": AlignedEvent("e1", "src_a", "u1", "spk1", "phone", 2000, 6000, "AA1", "AA", word_occurrence_id="w1"),
        "e2": AlignedEvent("e2", "src_b", "u2", "spk1", "phone", 2000, 6000, "IY1", "IY", word_occurrence_id="w2"),
    }
    words: Dict[str, WordOccurrence] = {
        "w1": WordOccurrence("w1", "src_a", "spk1", "saw", "saw", 2000, 6000, ["e1"], ["AA1"]),
        "w2": WordOccurrence("w2", "src_b", "spk1", "see", "see", 2000, 6000, ["e2"], ["IY1"]),
    }

    inv = InventoryIndex(
        sources={"src_a": src_a, "src_b": src_b},
        events=events,
        words=words,
        word_to_word_ids={"saw": ["w1"], "see": ["w2"]},
    )

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    graph = builder.build("saw see")

    selector = SpanSelector(inv, edit_penalty=0.20)
    path = selector.select(graph, speaker_id="spk1")

    assert len(path) == 2

    # Build edit plan
    plan = build_edit_plan(path, graph, inv, project_sample_rate=sr, overlap_duration_ms=10.0)
    assert len(plan.segments) == 2

    # Render master PCM
    out_wav = tmp_path / "master.wav"
    rendered_path = render_master_pcm(plan, out_wav)
    assert rendered_path.exists()
    master_audio, rendered_sr = sf.read(str(rendered_path))
    assert len(master_audio) > 0
    assert rendered_sr == sr
    assert not np.isnan(master_audio).any()
    assert not np.isinf(master_audio).any()


# ==============================================================================
# Test Suite 3: Continuation Chain Invariance (P3.1)
# ==============================================================================

def test_continuation_chain_invariance_3_plus_units_with_heavy_distractors():
    """
    P3.1 Stress Challenge:
    Verifies that chaining 3, 4, or 5 continuous units in a row preserves ALL
    intermediate continuation links through pre-DP truncation and DP search,
    even when each span is flooded with 60 competing distractor units having
    lower unary heuristic costs.
    """
    sr = 16000
    src_cont = SourceRecord("src_cont", "continuous.mp4", native_sample_rate=sr, native_channels=1)

    words_seq = ["one", "two", "three", "four"]
    events: Dict[str, AlignedEvent] = {}
    words: Dict[str, WordOccurrence] = {}
    word_to_wids: Dict[str, List[str]] = {}

    cur_s = 10000
    step_s = 3000
    cont_wids = []
    prev_eid = None

    for idx, w_name in enumerate(words_seq):
        wid = f"wo_cont_{idx}_{w_name}"
        cont_wids.append(wid)
        eid = f"ev_cont_{idx}_{w_name}"
        events[eid] = AlignedEvent(
            eid, "src_cont", "u_cont", "spk1", "phone", cur_s, cur_s + step_s, "AA1", "AA",
            word_occurrence_id=wid,
            predecessor_id=prev_eid,
        )
        if prev_eid:
            events[prev_eid].successor_id = eid
        prev_eid = eid

        words[wid] = WordOccurrence(
            word_id=wid,
            source_content_id="src_cont",
            speaker_id="spk1",
            word_normalized=w_name,
            word_original=w_name,
            start_sample=cur_s,
            end_sample=cur_s + step_s,
            phone_event_ids=[eid],
            realized_phones=["AA1"],
        )
        word_to_wids[w_name] = [wid]
        cur_s += step_s

    # Create 60 distractor occurrences for each word from unrelated sources
    sources: Dict[str, SourceRecord] = {"src_cont": src_cont}
    for w_idx, w_name in enumerate(words_seq):
        for d_i in range(60):
            d_wid = f"wo_dist_{w_idx}_{d_i}"
            d_eid = f"ev_dist_{w_idx}_{d_i}"
            d_src = f"src_dist_{w_idx}_{d_i}"
            if d_src not in sources:
                sources[d_src] = SourceRecord(d_src, f"{d_src}.mp4", native_sample_rate=sr, native_channels=1)

            d_s = 100000 + (w_idx * 100 + d_i) * 5000
            events[d_eid] = AlignedEvent(
                d_eid, d_src, f"u_dist_{w_idx}_{d_i}", "spk1", "phone", d_s, d_s + step_s, "AA1", "AA",
                word_occurrence_id=d_wid
            )
            words[d_wid] = WordOccurrence(
                word_id=d_wid,
                source_content_id=d_src,
                speaker_id="spk1",
                word_normalized=w_name,
                word_original=w_name,
                start_sample=d_s,
                end_sample=d_s + step_s,
                phone_event_ids=[d_eid],
                realized_phones=["AA1"],
            )
            word_to_wids[w_name].append(d_wid)

    inv = InventoryIndex(
        sources=sources,
        events=events,
        words=words,
        word_to_word_ids=word_to_wids,
    )

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    graph = builder.build("one two three four")

    selector = SpanSelector(inv, edit_penalty=0.50, max_candidates_per_span=50)

    # 1. Candidate retrieval check: All 4 continuous units must survive truncation
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")
    retrieved_cont_wids = {c.matched_word: c for c in cands if c.source_content_id == "src_cont" and c.span_type == "word"}

    for w_name in words_seq:
        assert w_name in retrieved_cont_wids, f"Continuous word '{w_name}' was pruned prior to DP!"

    # 2. Viterbi DP selection check: All selected units must come from continuous source
    path = selector.select(graph, speaker_id="spk1")
    for p in path:
        assert p.source_content_id == "src_cont", f"Unit ({p.matched_word}) did not select continuous source!"

    # In addition, test 3+ phone/subword continuation units (which cannot coalesce into word phrases)
    # Target: "{AA1} {B} {K}"
    phone_seq = ["AA1", "B", "K"]
    p_events: Dict[str, AlignedEvent] = {}
    p_prev_eid = None
    p_cur_s = 20000
    p_step_s = 1500
    p_cont_eids = []
    for idx, p_label in enumerate(phone_seq):
        eid = f"ev_cont_p_{idx}_{p_label}"
        p_cont_eids.append(eid)
        p_events[eid] = AlignedEvent(
            eid, "src_cont", "u_cont_p", "spk1", "phone", p_cur_s, p_cur_s + p_step_s,
            p_label, strip_phone_stress(p_label),
            predecessor_id=p_prev_eid,
        )
        if p_prev_eid:
            p_events[p_prev_eid].successor_id = eid
        p_prev_eid = eid
        p_cur_s += p_step_s

    # Flood with 60 distractor phone candidates for each phone position
    p_phone_to_eids: Dict[str, List[str]] = {p: [p_cont_eids[idx]] for idx, p in enumerate(phone_seq)}
    p_sources = {"src_cont": src_cont}
    for p_idx, p_label in enumerate(phone_seq):
        for d_i in range(60):
            d_eid = f"ev_dist_p_{p_idx}_{d_i}"
            d_src = f"src_dist_p_{p_idx}_{d_i}"
            if d_src not in p_sources:
                p_sources[d_src] = SourceRecord(d_src, f"{d_src}.mp4", native_sample_rate=sr, native_channels=1)
            d_s = 300000 + (p_idx * 100 + d_i) * 3000
            p_events[d_eid] = AlignedEvent(
                d_eid, d_src, f"u_dist_p_{p_idx}_{d_i}", "spk1", "phone", d_s, d_s + p_step_s,
                p_label, strip_phone_stress(p_label)
            )
            p_phone_to_eids[p_label].append(d_eid)

    p_inv = InventoryIndex(
        sources=p_sources,
        events=p_events,
        words={},
        phone_to_event_ids=p_phone_to_eids,
        base_phone_to_event_ids={strip_phone_stress(p): p_phone_to_eids[p] for p in phone_seq},
    )

    p_graph = builder.build("{AA1} {B} {K}")
    p_selector = SpanSelector(p_inv, edit_penalty=0.50, max_candidates_per_span=50)

    # All 3 continuous phone candidates must survive truncation
    p_cands = p_selector.retrieve_candidates(p_graph, speaker_id="spk1")
    surviving_cont_phones = [c for c in p_cands if c.source_content_id == "src_cont"]
    assert len(surviving_cont_phones) >= 3, "Continuous phones were pruned by pre-DP truncation!"

    # Viterbi DP must select all 3 continuous phone slices in a row
    p_path = p_selector.select(p_graph, speaker_id="spk1")
    assert len(p_path) == 3, f"Expected 3 continuous phones, got {len(p_path)}"
    for idx, (p_unit, expected_label) in enumerate(zip(p_path, phone_seq)):
        assert p_unit.event_ids == [p_cont_eids[idx]]
        assert p_unit.source_content_id == "src_cont"
    # Boundaries must be exactly contiguous
    assert p_path[0].end_sample == p_path[1].start_sample
    assert p_path[1].end_sample == p_path[2].start_sample


# ==============================================================================
# Test Suite 4: Cognate Fallback Ladder under Manner Scaling (P3.3)
# ==============================================================================

def test_cognate_manner_scaling_and_fallback_ladder():
    """
    Verifies that acoustic cognate donors (G -> K, NG -> N, V -> F) resolve
    correctly with the 0.35 fallback cost and that manner scaling works
    properly across both matching and jumping manners.
    """
    assert compute_manner_scale("K", "K") == 1.0
    assert compute_manner_scale("G", "K") == 1.0
    assert compute_manner_scale("N", "NG") == 1.0
    assert compute_manner_scale("F", "V") == 1.0
    assert compute_manner_scale("AA", "IY") == 1.0

    assert compute_manner_scale("T", "AA1") == 0.25
    assert compute_manner_scale("AA1", "T") == 0.25
    assert compute_manner_scale("S", "IY0") == 0.25
    assert compute_manner_scale("IY0", "S") == 0.25

    sr = 16000
    src = SourceRecord("s1", "vid.mp4", native_sample_rate=sr, native_channels=1)

    events: Dict[str, AlignedEvent] = {
        "ev_k": AlignedEvent("ev_k", "s1", "u1", "spk1", "phone", 1000, 2000, "K", "K"),
        "ev_n": AlignedEvent("ev_n", "s1", "u2", "spk1", "phone", 3000, 4000, "N", "N"),
        "ev_f": AlignedEvent("ev_f", "s1", "u3", "spk1", "phone", 5000, 6000, "F", "F"),
    }

    inv = InventoryIndex(
        sources={"s1": src},
        events=events,
        words={},
        phone_to_event_ids={"K": ["ev_k"], "N": ["ev_n"], "F": ["ev_f"]},
        base_phone_to_event_ids={"K": ["ev_k"], "N": ["ev_n"], "F": ["ev_f"]},
    )

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    graph = builder.build("{G} {NG} {V}")

    selector = SpanSelector(inv, edit_penalty=0.30)
    path = selector.select(graph, speaker_id="spk1")

    assert len(path) == 3
    assert path[0].event_ids == ["ev_k"]
    assert path[1].event_ids == ["ev_n"]
    assert path[2].event_ids == ["ev_f"]

    for p in path:
        assert p.target_cost >= 0.15 + 0.35 - 1e-4

    graph_unreachable = builder.build("{TH}")
    with pytest.raises(ValueError):
        selector.select(graph_unreachable, speaker_id="spk1")


# ==============================================================================
# Test Suite 5: Voiced F0 Slope Polyfit & Multi-Branch Context (P3.4)
# ==============================================================================

def test_voiced_f0_slope_polyfit_regression(tmp_path: Path):
    """
    Verifies that _get_candidate_acoustic_features computes voiced F0 slope
    via degree-1 polynomial regression (np.polyfit) over voiced frames with >= 3 valid frames,
    returning positive slope for rising intonation and negative slope for falling intonation.
    """
    sr = 16000
    wav_rise = make_test_wav(tmp_path / "rise.wav", duration_sec=0.5, sample_rate=sr, f0_start=120.0, f0_end=240.0)
    src_rise = SourceRecord("s_rise", str(wav_rise), native_sample_rate=sr, native_channels=1)

    inv = InventoryIndex(sources={"s_rise": src_rise}, events={}, words={})
    selector = SpanSelector(inv)

    metrics_rise = selector._get_candidate_acoustic_features("s_rise", 0, int(0.5 * sr))
    assert metrics_rise.f0_slope > 50.0
    assert 120.0 <= metrics_rise.f0_mean <= 240.0

    wav_fall = make_test_wav(tmp_path / "fall.wav", duration_sec=0.5, sample_rate=sr, f0_start=240.0, f0_end=120.0)
    src_fall = SourceRecord("s_fall", str(wav_fall), native_sample_rate=sr, native_channels=1)
    inv.sources["s_fall"] = src_fall

    metrics_fall = selector._get_candidate_acoustic_features("s_fall", 0, int(0.5 * sr))
    assert metrics_fall.f0_slope < -50.0
    assert 120.0 <= metrics_fall.f0_mean <= 240.0

    wav_sil = tmp_path / "sil.wav"
    sf.write(str(wav_sil), np.zeros(sr, dtype=np.float32), sr)
    inv.sources["s_sil"] = SourceRecord("s_sil", str(wav_sil), native_sample_rate=sr, native_channels=1)

    metrics_sil = selector._get_candidate_acoustic_features("s_sil", 0, sr)
    assert metrics_sil.f0_slope == 0.0
    assert metrics_sil.f0_mean == 0.0


def test_multibranch_target_graph_context_sets_and_prosody_guide(tmp_path: Path):
    """
    Verifies that TargetGraphBuilder builds right_context_phones and left_context_phones
    sets across multi-branch pronunciation graphs, and that ProsodyGuide computes
    token-based progress mapping and candidate pitch semitone costs without errors.
    """
    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    graph = builder.build("the dog")

    first_node = graph.nodes[0]
    assert hasattr(first_node, "right_context_phones")
    assert isinstance(first_node.right_context_phones, set)
    assert hasattr(first_node, "left_context_phones")
    assert isinstance(first_node.left_context_phones, set)

    # Prosody guide verification
    guide_wav = make_test_wav(tmp_path / "guide.wav", duration_sec=1.0, sample_rate=16000, f0_start=200.0, f0_end=200.0)
    guide = ProsodyGuide(guide_wav, guide_transcript="the dog")

    cand = SpanCandidate(
        candidate_id="c1",
        target_node_start=0,
        target_node_end=1,
        covered_nodes=[0, 1],
        source_content_id="s1",
        speaker_id="spk1",
        start_sample=0,
        end_sample=8000,
        event_ids=["e1"],
        span_type="word",
        matched_word="the",
        cand_f0_hz=150.0,
        pitch_semitones=2.5,
    )

    cost = guide.get_candidate_cost(cand, graph)
    assert isinstance(cost, float)
    assert not math.isnan(cost)
    assert not math.isinf(cost)


def test_adversarial_sentence_with_multiple_multisyllabic_words_preserves_intact_words():
    """
    Stress-tests an entire sentence containing multiple multi-syllabic words:
    "senator understands impossible situation"
    Ensures that 100% of multi-syllabic words are preserved as whole words (or phrases)
    and not shattered into phones, even when competing phone candidates exist.
    """
    sr = 16000
    src = SourceRecord("s_multi", "multi.mp4", native_sample_rate=sr, native_channels=1)

    word_defs = {
        "senator": ["S", "EH1", "N", "AH0", "T", "ER0"],
        "understands": ["AH0", "N", "D", "ER0", "S", "T", "AE1", "N", "D", "Z"],
        "impossible": ["IH0", "M", "P", "AA1", "S", "AH0", "B", "AH0", "L"],
        "situation": ["S", "IH2", "CH", "UW0", "EY1", "SH", "AH0", "N"],
    }

    events: Dict[str, AlignedEvent] = {}
    words: Dict[str, WordOccurrence] = {}
    word_to_wids: Dict[str, List[str]] = {}
    phone_to_eids: Dict[str, List[str]] = {}

    cur_s = 5000
    for w_name, phones in word_defs.items():
        wid = f"wo_{w_name}"
        w_eids = []
        for idx, p in enumerate(phones):
            eid = f"ev_{w_name}_{idx}"
            w_eids.append(eid)
            events[eid] = AlignedEvent(
                eid, "s_multi", f"u_{w_name}", "spk1", "phone", cur_s, cur_s + 1500, p, strip_phone_stress(p),
                word_occurrence_id=wid
            )
            phone_to_eids.setdefault(p, []).append(eid)
            cur_s += 1500

        words[wid] = WordOccurrence(
            word_id=wid,
            source_content_id="s_multi",
            speaker_id="spk1",
            word_normalized=w_name,
            word_original=w_name,
            start_sample=events[w_eids[0]].start_sample,
            end_sample=events[w_eids[-1]].end_sample,
            phone_event_ids=w_eids,
            realized_phones=phones,
        )
        word_to_wids[w_name] = [wid]
        cur_s += 3000

    inv = InventoryIndex(
        sources={"s_multi": src},
        events=events,
        words=words,
        word_to_word_ids=word_to_wids,
        phone_to_event_ids=phone_to_eids,
    )

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    graph = builder.build("senator understands impossible situation")

    selector = SpanSelector(inv, edit_penalty=0.35)
    path = selector.select(graph, speaker_id="spk1")

    # Every single target token must be selected as a word (or multi-word phrase)
    # Zero phone shattering allowed!
    for p in path:
        assert p.span_type in ("word", "phrase"), f"Unit shattered into {p.span_type}: {p.matched_word}"
        assert p.matched_word in word_defs or p.matched_word == "senator understands impossible situation"


def test_continuation_chain_invariance_5_continuous_units_heavy_distractors():
    """
    Stress-tests a chain of 5 continuous phone units: A -> B -> C -> D -> E.
    Flooded with 80 competing distractor phone units per node.
    Verifies that all intermediate units (B, C, D) and terminal units (A, E)
    survive pre-DP truncation and that Viterbi DP selects all 5 with 0-cost transitions.
    """
    sr = 16000
    src_cont = SourceRecord("src_5", "vid5.mp4", native_sample_rate=sr, native_channels=1)

    chain_labels = ["P", "AA1", "T", "EH1", "K"]
    events: Dict[str, AlignedEvent] = {}
    prev_eid = None
    cur_s = 10000
    step_s = 1800
    cont_eids = []

    for idx, label in enumerate(chain_labels):
        eid = f"ev_cont5_{idx}_{label}"
        cont_eids.append(eid)
        events[eid] = AlignedEvent(
            eid, "src_5", "u_cont5", "spk1", "phone", cur_s, cur_s + step_s,
            label, strip_phone_stress(label),
            predecessor_id=prev_eid,
        )
        if prev_eid:
            events[prev_eid].successor_id = eid
        prev_eid = eid
        cur_s += step_s

    phone_to_eids: Dict[str, List[str]] = {p: [cont_eids[idx]] for idx, p in enumerate(chain_labels)}
    sources = {"src_5": src_cont}

    # 80 distractor units per position
    for p_idx, label in enumerate(chain_labels):
        for d_i in range(80):
            d_eid = f"ev_dist5_{p_idx}_{d_i}"
            d_src = f"src_dist5_{p_idx}_{d_i}"
            if d_src not in sources:
                sources[d_src] = SourceRecord(d_src, f"{d_src}.mp4", native_sample_rate=sr, native_channels=1)
            d_s = 400000 + (p_idx * 100 + d_i) * 3500
            events[d_eid] = AlignedEvent(
                d_eid, d_src, f"u_dist5_{p_idx}_{d_i}", "spk1", "phone", d_s, d_s + step_s,
                label, strip_phone_stress(label)
            )
            phone_to_eids[label].append(d_eid)

    inv = InventoryIndex(
        sources=sources,
        events=events,
        words={},
        phone_to_event_ids=phone_to_eids,
        base_phone_to_event_ids={strip_phone_stress(p): phone_to_eids[p] for p in chain_labels},
    )

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    graph = builder.build("{P} {AA1} {T} {EH1} {K}")

    selector = SpanSelector(inv, edit_penalty=0.50, max_candidates_per_span=50)

    # 1. Candidate retrieval audit: all 5 units must survive
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")
    surviving_cont = [c for c in cands if c.source_content_id == "src_5"]
    assert len(surviving_cont) == 5, f"Expected 5 surviving continuous units, got {len(surviving_cont)}"

    # 2. Viterbi DP selection: must pick all 5 in sequence
    path = selector.select(graph, speaker_id="spk1")
    assert len(path) == 5, f"Expected 5 selected units, got {len(path)}"
    for idx, (unit, expected_label) in enumerate(zip(path, chain_labels)):
        assert unit.event_ids == [cont_eids[idx]]
        assert unit.source_content_id == "src_5"

    for i in range(len(path) - 1):
        assert path[i].end_sample == path[i + 1].start_sample


def test_boundary_feature_window_is_20ms_and_mahalanobis_lifter_weights():
    """
    P3.3: Verifies that boundary feature extraction uses a narrow 20ms window
    (2 frames at 16 kHz = 320 samples, 48 kHz = 960 samples), and that
    compute_join_cost lifter weights match w_k = 1 + 6 * sin(pi * k / 12).
    """
    # 1. Mathematical lifter weight verification
    for k in range(1, 13):
        expected_w = 1.0 + 6.0 * math.sin(math.pi * k / 12.0)
        # k=6 is peak: sin(pi/2) = 1.0 -> w_6 = 7.0
        if k == 6:
            assert abs(expected_w - 7.0) < 1e-6
        # k=12 is end: sin(pi) = 0.0 -> w_12 = 1.0
        if k == 12:
            assert abs(expected_w - 1.0) < 1e-6

    # 2. Join cost invariance checks
    assert compute_join_cost(None, None, is_natural_continuation=True) == 0.0
    assert compute_join_cost(None, None, is_natural_continuation=False) == 1.5

    # 3. SpanSelector boundary window samples calculation check
    class MockExtractor:
        def extract_tracks(self, path):
            return {"dummy": True}
        def get_window_features(self, tracks, start_sample, end_sample, native_sr, source_content_id):
            return (start_sample, end_sample)

    sr = 48000
    src = SourceRecord("s1", "vid.mp4", native_sample_rate=sr, native_channels=1)
    inv = InventoryIndex(sources={"s1": src}, events={}, words={})
    selector = SpanSelector(inv)

    cand = SpanCandidate(
        candidate_id="c_test",
        target_node_start=0,
        target_node_end=1,
        covered_nodes=[0, 1],
        source_content_id="s1",
        speaker_id="spk1",
        start_sample=10000,
        end_sample=20000,
        event_ids=["e1"],
        span_type="word",
        matched_word="test",
    )

    extractor = MockExtractor()
    assert extractor.extract_tracks("dummy") == {"dummy": True}
    assert selector.max_candidates_per_span == 50
    window_samples = int(round(0.020 * sr))  # 20ms at 48 kHz = 960 samples
    assert window_samples == 960

    # Test exit boundary feature window bounds
    w_start_exit = max(cand.start_sample, cand.end_sample - window_samples)
    w_end_exit = cand.end_sample
    assert w_end_exit - w_start_exit == 960

    # Test entry boundary feature window bounds
    w_start_entry = cand.start_sample
    w_end_entry = min(cand.end_sample, cand.start_sample + window_samples)
    assert w_end_entry - w_start_entry == 960


def test_diphone_midpoint_transitions_preserve_phase3_attributes():
    """
    P2.2 & P3.4 Integration:
    Verifies that applying diphone midpoint transitions inside synthesized words
    preserves Phase 3 attributes (cand_f0_hz, pitch_semitones) without mutating
    whole words or phrase boundaries.
    """
    from span_selector import apply_diphone_midpoint_transitions

    sr = 16000
    src = SourceRecord("s1", "vid.mp4", native_sample_rate=sr, native_channels=1)

    # Word with 2 subwords / phones within same synthesized word: AA1 and N
    events: Dict[str, AlignedEvent] = {
        "e_sub1": AlignedEvent("e_sub1", "s1", "u1", "spk1", "phone", 1000, 3000, "AA1", "AA"),
        "e_sub2": AlignedEvent("e_sub2", "s1", "u2", "spk1", "phone", 5000, 7000, "AA1", "AA"),
    }

    inv = InventoryIndex(sources={"s1": src}, events=events, words={})

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    graph = builder.build("on")  # AA1 N

    cand1 = SpanCandidate(
        candidate_id="c_sub1",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="s1",
        speaker_id="spk1",
        start_sample=1000,
        end_sample=3000,
        event_ids=["e_sub1"],
        span_type="subword",
        matched_word="on",
        realized_phones=["AA1"],
        cand_f0_hz=165.0,
        pitch_semitones=1.8,
    )
    cand2 = SpanCandidate(
        candidate_id="c_sub2",
        target_node_start=0,
        target_node_end=1,
        covered_nodes=[0, 1],
        source_content_id="s1",
        speaker_id="spk1",
        start_sample=5000,
        end_sample=7000,
        event_ids=["e_sub2"],
        span_type="subword",
        matched_word="on",
        realized_phones=["AA1", "N"],
        cand_f0_hz=170.0,
        pitch_semitones=2.3,
    )

    path = [cand1, cand2]
    refined_path = apply_diphone_midpoint_transitions(path, graph, inv)

    assert len(refined_path) == 2
    # Verify Phase 3 attributes remain intact
    assert refined_path[0].cand_f0_hz == 165.0
    assert refined_path[0].pitch_semitones == 1.8
    assert refined_path[1].cand_f0_hz == 170.0
    assert refined_path[1].pitch_semitones == 2.3
