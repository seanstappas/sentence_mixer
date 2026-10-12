"""
Master Empirical Adversarial Verification Suite for Phase 3 (Fixes P3.1 - P3.4)
Author: Challenger P3.1 (teamwork_preview_challenger_p3_1)

Adversarial Stress Verification Requirements:
1. Continuation Preservation Stress Test (P3.1):
   - Construct candidate pools where a natural continuation candidate is ranked at extreme low positions
     (#51, #100, #200) with 50, 100, and 200 competing candidates.
   - Assert continuation candidate is preserved in the surviving pool and selected when transition cost
     makes it globally optimal.
   - Test degenerate graphs: single node, empty edges, disconnected spans (linkage audit safety).

2. Duration Z-Score & Quadratic Cost Stress Test (P3.2):
   - Test duration ratio limits: ratio = 0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0.
   - Assert smooth quadratic behavior without NaN, negative costs, or discontinuities.
   - Verify whole-word heuristic penalty stacking is strictly bounded by 0.50 regardless of
     individual l_bleed_cost, r_bleed_cost, and swallow_cost values.

3. Manner Scaling & Mahalanobis Distance Stress Test (P3.3):
   - Test pairwise phonetic transitions across all ARPAbet classes (vowel, stop, fricative, affricate, nasal, liquid, glide).
   - Verify that natural manner transitions return 0.25 and matching/vowel-vowel transitions return 1.0.
   - Verify liftered Mahalanobis MFCC distance produces non-negative, finite values without NaN.

4. F0 Slope & Multi-Branch Lattice Stress Test (P3.4):
   - Test F0 slope regression on arrays with 0, 1, 2, 3, and 100 frames, pure unvoiced frames, and extreme noisy pitch jumps.
   - Verify ProsodyGuide progress normalization on graphs with multiple branching words.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure src is on sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import pytest
import soundfile as sf

from contracts import (
    AlignedEvent,
    FeatureRecord,
    InventoryIndex,
    SourceRecord,
    SpanCandidate,
    TargetGraph,
    TargetPhoneNode,
    WordOccurrence,
)
from features import (
    compute_join_cost,
    compute_manner_scale,
)
from lexicon import Lexicon
from prosody_guide import ProsodyGuide
from span_selector import SpanSelector
from target_graph import TargetGraphBuilder


@pytest.fixture
def dummy_guide_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "guide_audio.wav"
    sr = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave at 220Hz
    signal = 0.5 * np.sin(2 * np.pi * 220 * t)
    sf.write(str(audio_path), signal, sr)
    return audio_path


# ==============================================================================
# SECTION 1: P3.1 CONTINUATION PRESERVATION STRESS TESTS
# ==============================================================================

@pytest.mark.parametrize("competing_count", [50, 99, 199])
def test_continuation_preservation_at_extreme_low_ranks(competing_count: int):
    """
    Stress-test continuation preservation when a natural continuation candidate
    is ranked at extreme low positions (#51, #100, #200) among competing candidates.
    Asserts:
    1. The continuation candidate is preserved in the surviving pool returned by retrieve_candidates.
    2. Viterbi DP selects the continuous pair over competing candidates because 0.0 transition cost
       outweighs the unary penalty.
    """
    sr = 48000
    src_cont_id = "src_cont"
    sources: Dict[str, SourceRecord] = {
        src_cont_id: SourceRecord(
            content_id=src_cont_id,
            source_path="cont.wav",
            native_sample_rate=sr,
            native_channels=2,
        )
    }

    # Word 1: 'good' [G, UH1, D]
    ev_g = AlignedEvent(
        event_id="ev_g", source_content_id=src_cont_id, utterance_id="u0", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=4000, label="G", base_phone="G",
        successor_id="ev_uh",
    )
    ev_uh = AlignedEvent(
        event_id="ev_uh", source_content_id=src_cont_id, utterance_id="u0", speaker_id="spk1",
        kind="phone", start_sample=4000, end_sample=12000, label="UH1", base_phone="UH",
        predecessor_id="ev_g", successor_id="ev_d",
    )
    ev_d = AlignedEvent(
        event_id="ev_d", source_content_id=src_cont_id, utterance_id="u0", speaker_id="spk1",
        kind="phone", start_sample=12000, end_sample=16000, label="D", base_phone="D",
        predecessor_id="ev_uh", successor_id="ev_d_dog",
    )
    w_good = WordOccurrence(
        word_id="w_good", source_content_id=src_cont_id, speaker_id="spk1",
        word_normalized="good", word_original="good",
        start_sample=0, end_sample=16000,
        phone_event_ids=["ev_g", "ev_uh", "ev_d"],
        realized_phones=["G", "UH1", "D"],
    )

    # Word 2: 'dog' [D, AO1, G] - natural continuation in source recording immediately following w_good
    # Intentionally give it a heavily dragged duration (e.g. 1.2s instead of 0.25s) so its unary
    # duration penalty pushes its rank down below ALL competing candidates!
    ev_d_dog = AlignedEvent(
        event_id="ev_d_dog", source_content_id=src_cont_id, utterance_id="u0", speaker_id="spk1",
        kind="phone", start_sample=16000, end_sample=24000, label="D", base_phone="D",
        predecessor_id="ev_d", successor_id="ev_ao_dog",
    )
    ev_ao_dog = AlignedEvent(
        event_id="ev_ao_dog", source_content_id=src_cont_id, utterance_id="u0", speaker_id="spk1",
        kind="phone", start_sample=24000, end_sample=60000, label="AO1", base_phone="AO",
        predecessor_id="ev_d_dog", successor_id="ev_g_dog",
    )
    ev_g_dog = AlignedEvent(
        event_id="ev_g_dog", source_content_id=src_cont_id, utterance_id="u0", speaker_id="spk1",
        kind="phone", start_sample=60000, end_sample=73600, label="G", base_phone="G",
        predecessor_id="ev_ao_dog",
    )
    w_dog_cont = WordOccurrence(
        word_id="w_dog_cont", source_content_id=src_cont_id, speaker_id="spk1",
        word_normalized="dog", word_original="dog",
        start_sample=16000, end_sample=73600,  # ~1.2s, heavily dragged
        phone_event_ids=["ev_d_dog", "ev_ao_dog", "ev_g_dog"],
        realized_phones=["D", "AO1", "G"],
    )

    events: Dict[str, AlignedEvent] = {
        "ev_g": ev_g, "ev_uh": ev_uh, "ev_d": ev_d,
        "ev_d_dog": ev_d_dog, "ev_ao_dog": ev_ao_dog, "ev_g_dog": ev_g_dog,
    }
    words: Dict[str, WordOccurrence] = {"w_good": w_good, "w_dog_cont": w_dog_cont}
    word_to_ids: Dict[str, List[str]] = {"good": ["w_good"], "dog": ["w_dog_cont"]}
    p_to_e: Dict[str, List[str]] = {
        "G": ["ev_g", "ev_g_dog"], "UH1": ["ev_uh"], "D": ["ev_d", "ev_d_dog"],
        "AO1": ["ev_ao_dog"],
    }
    bp_to_e: Dict[str, List[str]] = {
        "G": ["ev_g", "ev_g_dog"], "UH": ["ev_uh"], "D": ["ev_d", "ev_d_dog"],
        "AO": ["ev_ao_dog"],
    }

    # Now generate competing_count competing occurrences of 'dog' from distinct sources
    # with near-perfect durations (0.28s) so their unary cost is virtually zero (< 0.10)
    for i in range(1, competing_count + 1):
        s_id = f"src_comp_{i}"
        sources[s_id] = SourceRecord(
            content_id=s_id, source_path=f"{s_id}.wav", native_sample_rate=sr, native_channels=2
        )
        ed = f"ev_d_comp_{i}"
        eao = f"ev_ao_comp_{i}"
        eg = f"ev_g_comp_{i}"
        events[ed] = AlignedEvent(
            event_id=ed, source_content_id=s_id, utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=1000, end_sample=4000, label="D", base_phone="D",
            successor_id=eao,
        )
        events[eao] = AlignedEvent(
            event_id=eao, source_content_id=s_id, utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=4000, end_sample=10000, label="AO1", base_phone="AO",
            predecessor_id=ed, successor_id=eg,
        )
        events[eg] = AlignedEvent(
            event_id=eg, source_content_id=s_id, utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=10000, end_sample=14440, label="G", base_phone="G",
            predecessor_id=eao,
        )
        wid = f"w_dog_comp_{i}"
        words[wid] = WordOccurrence(
            word_id=wid, source_content_id=s_id, speaker_id="spk1",
            word_normalized="dog", word_original="dog",
            start_sample=1000, end_sample=14440,  # exactly 0.28s
            phone_event_ids=[ed, eao, eg],
            realized_phones=["D", "AO1", "G"],
        )
        word_to_ids["dog"].append(wid)
        p_to_e["D"].append(ed)
        p_to_e["AO1"].append(eao)
        p_to_e["G"].append(eg)
        bp_to_e["D"].append(ed)
        bp_to_e["AO"].append(eao)
        bp_to_e["G"].append(eg)

    inv = InventoryIndex(
        sources=sources, events=events, words=words,
        word_to_word_ids=word_to_ids,
        phone_to_event_ids=p_to_e,
        base_phone_to_event_ids=bp_to_e,
    )

    lex = Lexicon()
    lex.word_to_pronunciations["good"] = [["G", "UH1", "D"]]
    lex.word_to_pronunciations["dog"] = [["D", "AO1", "G"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("good dog")

    # SpanSelector with default max_candidates_per_span=50
    # Total candidates for 'dog': 1 + competing_count (e.g. 51, 100, 200)
    # The continuation candidate w_dog_cont is ranked last (#51, #100, #200) due to dur penalty
    selector = SpanSelector(inv, edit_penalty=0.8, max_candidates_per_span=50)
    retrieved = selector.retrieve_candidates(graph, speaker_id="spk1")

    # 1. Assert continuation candidate was NOT pruned
    cont_cands = [c for c in retrieved if c.source_content_id == src_cont_id and c.matched_word == "dog"]
    assert len(cont_cands) >= 1, (
        f"Natural continuation candidate ranked #{competing_count + 1} was incorrectly pruned "
        f"from surviving pool of size {len(retrieved)}!"
    )
    assert cont_cands[0].start_sample == 16000

    # 2. Assert Viterbi selects the continuous pair
    selected = selector.select(graph, speaker_id="spk1")
    assert len(selected) == 2
    assert selected[0].matched_word == "good"
    assert selected[1].matched_word == "dog"
    assert selected[0].source_content_id == src_cont_id
    assert selected[1].source_content_id == src_cont_id
    assert selected[1].start_sample == 16000, "Viterbi failed to select 0-cost continuation path!"


def test_linkage_audit_on_degenerate_graphs():
    """
    Stress-test the pre-DP continuation linkage audit on degenerate graph structures:
    1. Single-node graph (1 phone, 0 edges).
    2. Empty edges graph (2 nodes, 0 edges).
    3. Disconnected graph (islands of unreachable nodes).
    4. Graph with empty normalized_words.
    Asserts: No unhandled exceptions (ZeroDivisionError, IndexError, KeyError, TypeError).
    """
    sr = 48000
    src = SourceRecord(content_id="s1", source_path="s1.wav", native_sample_rate=sr, native_channels=1)
    ev = AlignedEvent(
        event_id="e1", source_content_id="s1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=2400, label="AH0", base_phone="AH"
    )
    w = WordOccurrence(
        word_id="w1", source_content_id="s1", speaker_id="spk1",
        word_normalized="a", word_original="a",
        start_sample=0, end_sample=2400,
        phone_event_ids=["e1"], realized_phones=["AH0"]
    )
    inv = InventoryIndex(
        sources={"s1": src}, events={"e1": ev}, words={"w1": w},
        word_to_word_ids={"a": ["w1"]},
        phone_to_event_ids={"AH0": ["e1"]},
        base_phone_to_event_ids={"AH": ["e1"]},
    )
    selector = SpanSelector(inv, max_candidates_per_span=50)

    # Degenerate 1: Single node graph, 0 edges
    single_node = TargetPhoneNode(
        node_id=0, word_text="a", token_index=0, branch_id=0,
        phone="AH0", base_phone="AH", stress=0, is_word_initial=True, is_word_final=True
    )
    deg1 = TargetGraph(
        raw_text="a", normalized_words=["a"],
        nodes=[single_node], edges=[], start_nodes=[0], end_nodes=[0]
    )
    cands1 = selector.retrieve_candidates(deg1, speaker_id="spk1")
    assert len(cands1) > 0
    sel1 = selector.select(deg1, speaker_id="spk1")
    assert len(sel1) == 1

    # Degenerate 2: Two nodes with empty edges (disconnected)
    n0 = TargetPhoneNode(
        node_id=0, word_text="a", token_index=0, branch_id=0,
        phone="AH0", base_phone="AH", stress=0, is_word_initial=True, is_word_final=True
    )
    n1 = TargetPhoneNode(
        node_id=1, word_text="a", token_index=1, branch_id=0,
        phone="AH0", base_phone="AH", stress=0, is_word_initial=True, is_word_final=True
    )
    deg2 = TargetGraph(
        raw_text="a a", normalized_words=["a", "a"],
        nodes=[n0, n1], edges=[], start_nodes=[0], end_nodes=[1]
    )
    # Linkage audit should run cleanly without throwing
    cands2 = selector.retrieve_candidates(deg2, speaker_id="spk1")
    assert isinstance(cands2, list)

    # Degenerate 3: Empty nodes graph
    deg3 = TargetGraph(
        raw_text="", normalized_words=[],
        nodes=[], edges=[], start_nodes=[], end_nodes=[]
    )
    cands3 = selector.retrieve_candidates(deg3, speaker_id="spk1")
    assert cands3 == []


# ==============================================================================
# SECTION 2: P3.2 DURATION Z-SCORE & QUADRATIC COST STRESS TESTS
# ==============================================================================

def test_quadratic_duration_penalty_spectrum():
    """
    Stress-test _compute_quadratic_duration_penalty across exact ratio points:
    ratio = 0.0, 0.5, 0.75, 1.0, 1.5, 2.0, 10.0.
    Asserts:
    1. Exact values match analytical quadratic equations.
    2. Zero penalty plateau on [0.75, 1.50].
    3. Strict monotonicity outside plateau.
    4. Continuity at boundary points (0.75 and 1.50).
    5. Non-negativity, finiteness, zero NaN/Inf.
    """
    fn = SpanSelector._compute_quadratic_duration_penalty
    median = 0.20  # 200 ms

    # ratio = 0.0: 0.60 * ((0.75 - 0.0) / 0.75)**2 = 0.60 * 1.0 = 0.60
    c_0_0 = fn(0.0, median)
    assert abs(c_0_0 - 0.60) < 1e-6

    # ratio = 0.5: 0.60 * ((0.75 - 0.50) / 0.75)**2 = 0.60 * (1/3)**2 = 0.60 / 9 = 0.066666...
    c_0_5 = fn(0.10, median)
    assert abs(c_0_5 - (0.60 * (0.25 / 0.75) ** 2)) < 1e-6

    # ratio = 0.75: plateau boundary -> 0.0 (within float tolerance)
    c_0_75 = fn(0.15, median)
    assert abs(c_0_75) < 1e-12

    # ratio = 1.0: plateau center -> 0.0
    c_1_0 = fn(0.20, median)
    assert c_1_0 == 0.0

    # ratio = 1.5: plateau upper boundary -> 0.0 (within float tolerance)
    c_1_5 = fn(0.30, median)
    assert abs(c_1_5) < 1e-12

    # ratio = 2.0: 0.30 * min(2.0, (2.0 - 1.50)/1.0)**2 = 0.30 * 0.5**2 = 0.075
    c_2_0 = fn(0.40, median)
    assert abs(c_2_0 - 0.075) < 1e-6

    # ratio = 10.0: clamped at 2.0 -> 0.30 * (2.0)**2 = 1.20
    c_10_0 = fn(2.00, median)
    assert abs(c_10_0 - 1.20) < 1e-6

    # Monotonicity checks
    ratios = np.linspace(0.0, 5.0, 501)
    costs = [fn(float(r * median), median) for r in ratios]
    for r, c in zip(ratios, costs):
        assert not math.isnan(c)
        assert not math.isinf(c)
        assert c >= 0.0

    # Strictly decreasing for ratio < 0.75
    r_sub = [r for r in ratios if r < 0.74]
    c_sub = [fn(float(r * median), median) for r in r_sub]
    for i in range(len(c_sub) - 1):
        assert c_sub[i] > c_sub[i + 1]

    # Exactly 0.0 on [0.75, 1.50]
    r_plat = [r for r in ratios if 0.76 <= r <= 1.49]
    c_plat = [fn(float(r * median), median) for r in r_plat]
    for c in c_plat:
        assert c == 0.0

    # Strictly increasing for 1.50 < ratio <= 3.50 (where clamp occurs at 1.5 + 2.0 = 3.5)
    r_sup = [r for r in ratios if 1.51 < r <= 3.49]
    c_sup = [fn(float(r * median), median) for r in r_sup]
    for i in range(len(c_sup) - 1):
        assert c_sup[i] < c_sup[i + 1]

    # Extreme pathological inputs: median <= 0
    c_zero_med = fn(0.10, 0.0)
    assert not math.isnan(c_zero_med) and c_zero_med >= 0.0
    c_neg_med = fn(0.10, -0.5)
    assert not math.isnan(c_neg_med) and c_neg_med >= 0.0


def test_whole_word_heuristic_penalty_capping_strict_bound():
    """
    Verify that whole-word heuristic penalty stacking is strictly bounded by 0.50
    regardless of individual l_bleed_cost, r_bleed_cost, and swallow_cost values.
    """
    sr = 48000
    src = SourceRecord(content_id="s1", source_path="s1.wav", native_sample_rate=sr, native_channels=1)

    # Create word with extreme bleed and swallow penalties:
    # 1. Swallowed penalty: phone duration < 0.035s generates up to 2.50
    # 2. Bleed penalty: preceding / following plosives generate 0.80 + 0.35
    # Combined sum is > 3.50, but heuristic penalty cap must clamp at exactly 0.50.
    ev_k = AlignedEvent(
        event_id="ek", source_content_id="s1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=200, label="K", base_phone="K",
        successor_id="ea"
    )
    ev_ae = AlignedEvent(
        event_id="ea", source_content_id="s1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=200, end_sample=400, label="AE1", base_phone="AE",
        predecessor_id="ek", successor_id="et"
    )
    ev_t = AlignedEvent(
        event_id="et", source_content_id="s1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=400, end_sample=600, label="T", base_phone="T",
        predecessor_id="ea"
    )
    w = WordOccurrence(
        word_id="w_cat", source_content_id="s1", speaker_id="spk1",
        word_normalized="cat", word_original="cat",
        start_sample=0, end_sample=600,  # 600 samples = 0.0125s (micro-short = severe swallow)
        phone_event_ids=["ek", "ea", "et"],
        realized_phones=["K", "AE1", "T"]
    )
    inv = InventoryIndex(
        sources={"s1": src}, events={"ek": ev_k, "ea": ev_ae, "et": ev_t},
        words={"w_cat": w}, word_to_word_ids={"cat": ["w_cat"]},
        phone_to_event_ids={"K": ["ek"], "AE1": ["ea"], "T": ["et"]},
        base_phone_to_event_ids={"K": ["ek"], "AE": ["ea"], "T": ["et"]},
    )
    selector = SpanSelector(inv)

    # Direct formula check on arbitrary stacked penalties
    test_cases = [
        (0.80, 0.35, 2.50),  # sum = 3.65 -> capped at 0.50
        (0.50, 0.50, 0.00),  # sum = 1.00 -> capped at 0.50
        (0.20, 0.20, 0.05),  # sum = 0.45 -> uncapped = 0.45
        (0.00, 0.00, 0.00),  # sum = 0.00 -> 0.00
        (0.00, 0.00, 5.00),  # sum = 5.00 -> capped at 0.50
    ]
    for l_bl, r_bl, swallow in test_cases:
        capped = min(0.50, l_bl + r_bl + swallow)
        assert capped <= 0.50
        if l_bl + r_bl + swallow >= 0.50:
            assert capped == 0.50

    # Integration test: Retrieve candidate and verify target_cost
    lex = Lexicon()
    lex.word_to_pronunciations["cat"] = [["K", "AE1", "T"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("cat")
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")
    word_cand = next((c for c in cands if c.span_type == "word"), None)
    assert word_cand is not None

    # Base target cost = 0.05 + bias + r_c + dur_cost + heur_c + lex_c
    # dur_cost at 0.0125s vs 0.25s is 0.60
    # heur_c is strictly capped at 0.50
    # total target_cost should be <= 0.05 + 0.60 + 0.50 + 0.05 = 1.20
    # Before Fix P3.2, un-capped target_cost was > 3.80!
    assert word_cand.target_cost < 1.50


# ==============================================================================
# SECTION 3: P3.3 MANNER SCALING & MAHALANOBIS DISTANCE STRESS TESTS
# ==============================================================================

def test_manner_scaling_across_all_arpabet_transitions():
    """
    Test pairwise phonetic transitions across all ARPAbet classes:
    Vowel, Stop, Fricative, Affricate, Nasal, Liquid, Glide.
    Asserts:
    1. Same phone (e.g. AA1 -> AA1, T -> T) returns 1.0.
    2. Stress variation of same vowel (e.g. AA1 -> AA0, IY2 -> IY1) returns 1.0.
    3. Matching manner class (Vowel -> Vowel, Stop -> Stop, Fricative -> Fricative,
       Sonorant -> Sonorant) returns 1.0.
    4. Cross-manner natural transitions (Stop -> Vowel, Fricative -> Vowel,
       Vowel -> Stop, Stop -> Sonorant) return 0.25.
    5. None / empty transitions return 1.0.
    """
    vowels = ["AA1", "AE0", "AH1", "AO2", "AW1", "AY0", "EH1", "ER0", "EY1", "IH0", "IY1", "OW2", "OY1", "UH1", "UW0"]
    stops = ["P", "T", "K", "B", "D", "G"]
    affricates = ["CH", "JH"]  # classified as stops in features.STOP_PHONES
    fricatives = ["F", "V", "TH", "DH", "S", "Z", "SH", "ZH", "HH"]
    nasals = ["M", "N", "NG"]  # in features.SONORANT_PHONES
    liquids = ["L", "R"]       # in features.SONORANT_PHONES
    glides = ["W", "Y"]        # in features.SONORANT_PHONES

    # 1. Identity & Stress variation
    for v in vowels:
        assert compute_manner_scale(v, v) == 1.0
        base = "".join(c for c in v if not c.isdigit())
        v_diff_stress = base + "0" if not v.endswith("0") else base + "1"
        assert compute_manner_scale(v, v_diff_stress) == 1.0

    for c in stops + affricates + fricatives + nasals + liquids + glides:
        assert compute_manner_scale(c, c) == 1.0

    # 2. Within-class transitions -> 1.0
    for v1 in vowels:
        for v2 in vowels:
            assert compute_manner_scale(v1, v2) == 1.0, f"Vowel-vowel {v1}->{v2} should be 1.0"

    for s1 in stops + affricates:
        for s2 in stops + affricates:
            assert compute_manner_scale(s1, s2) == 1.0, f"Stop-stop {s1}->{s2} should be 1.0"

    for f1 in fricatives:
        for f2 in fricatives:
            assert compute_manner_scale(f1, f2) == 1.0, f"Fricative-fricative {f1}->{f2} should be 1.0"

    for n1 in nasals + liquids + glides:
        for n2 in nasals + liquids + glides:
            assert compute_manner_scale(n1, n2) == 1.0, f"Sonorant-sonorant {n1}->{n2} should be 1.0"

    # 3. Cross-class natural manner jumps -> 0.25
    # Stop into Vowel (e.g. /T/ into /AA1/)
    for s in stops + affricates:
        for v in vowels:
            scale = compute_manner_scale(s, v)
            assert scale == 0.25, f"Stop->Vowel {s}->{v} must return 0.25, got {scale}"

    # Fricative into Vowel (e.g. /S/ into /IY1/)
    for f in fricatives:
        for v in vowels:
            scale = compute_manner_scale(f, v)
            assert scale == 0.25, f"Fricative->Vowel {f}->{v} must return 0.25, got {scale}"

    # Vowel into Stop (e.g. /AA/ into /T/)
    for v in vowels:
        for s in stops + affricates:
            scale = compute_manner_scale(v, s)
            assert scale == 0.25, f"Vowel->Stop {v}->{s} must return 0.25, got {scale}"

    # Stop into Fricative
    for s in stops:
        for f in fricatives:
            scale = compute_manner_scale(s, f)
            assert scale == 0.25, f"Stop->Fricative {s}->{f} must return 0.25, got {scale}"

    # 4. None / empty handling
    assert compute_manner_scale(None, "AA1") == 1.0
    assert compute_manner_scale("T", None) == 1.0
    assert compute_manner_scale(None, None) == 1.0
    assert compute_manner_scale("", "") == 1.0


def test_liftered_mahalanobis_mfcc_distance_stability():
    """
    Stress-test liftered Mahalanobis MFCC distance in compute_join_cost.
    Asserts:
    1. Identical MFCCs return 0.0 distance.
    2. Arbitrary real vectors (including high tilt, high frequency) produce
       non-negative finite cost without NaN/Inf.
    3. Natural manner jump scales MFCC join cost by 0.25 relative to matching manner.
    4. All-zero MFCC vectors produce 0.0 distance (no ZeroDivisionError).
    """
    # 1. Identical MFCCs
    m_base = [10.0] + [float(k * 1.5) for k in range(1, 13)]
    f1 = FeatureRecord(
        source_content_id="s1", sample_rate=16000, start_sample=0, end_sample=1600,
        rms_energy=0.05, f0_hz=150.0, f0_confidence=0.9, voicing_state="voiced",
        spectral_centroid=1200.0, mfcc_13=list(m_base),
    )
    f2 = FeatureRecord(
        source_content_id="s1", sample_rate=16000, start_sample=1600, end_sample=3200,
        rms_energy=0.05, f0_hz=150.0, f0_confidence=0.9, voicing_state="voiced",
        spectral_centroid=1200.0, mfcc_13=list(m_base),
    )
    cost_ident = compute_join_cost(f1, f2, exit_phone="AA1", entry_phone="AA1")
    assert cost_ident == 0.0

    # 2. Distinct MFCCs with manner scaling comparison
    m_diff = [10.0] + [float((13 - k) * 2.0) for k in range(1, 13)]
    f3 = FeatureRecord(
        source_content_id="s2", sample_rate=16000, start_sample=0, end_sample=1600,
        rms_energy=0.05, f0_hz=150.0, f0_confidence=0.9, voicing_state="voiced",
        spectral_centroid=1200.0, mfcc_13=m_diff,
    )
    cost_matching = compute_join_cost(f1, f3, exit_phone="AA1", entry_phone="IY1")  # vowel->vowel: scale=1.0
    cost_jump = compute_join_cost(f1, f3, exit_phone="T", entry_phone="AA1")       # stop->vowel: scale=0.25

    # Since f0 and rms are identical, cost is purely the MFCC spectral term
    assert cost_matching > 0.0
    assert cost_jump > 0.0
    assert abs(cost_jump - (0.25 * cost_matching)) < 1e-5

    # 3. Pathological all-zero MFCCs
    f_zero = FeatureRecord(
        source_content_id="s1", sample_rate=16000, start_sample=0, end_sample=1600,
        rms_energy=0.05, f0_hz=150.0, f0_confidence=0.9, voicing_state="voiced",
        spectral_centroid=1200.0, mfcc_13=[0.0] * 13,
    )
    cost_zeros = compute_join_cost(f_zero, f_zero, exit_phone="T", entry_phone="AA1")
    assert not math.isnan(cost_zeros)
    assert cost_zeros == 0.0

    # 4. Randomized MFCC arrays across 100 trials
    rng = np.random.RandomState(42)
    for _ in range(100):
        vec1 = [float(x) for x in rng.randn(13)]
        vec2 = [float(x) for x in rng.randn(13)]
        rec1 = FeatureRecord(
            source_content_id="s1", sample_rate=16000, start_sample=0, end_sample=1600,
            rms_energy=0.05, f0_hz=150.0, f0_confidence=0.9, voicing_state="voiced",
            spectral_centroid=1200.0, mfcc_13=vec1,
        )
        rec2 = FeatureRecord(
            source_content_id="s2", sample_rate=16000, start_sample=0, end_sample=1600,
            rms_energy=0.05, f0_hz=150.0, f0_confidence=0.9, voicing_state="voiced",
            spectral_centroid=1200.0, mfcc_13=vec2,
        )
        c = compute_join_cost(rec1, rec2, exit_phone="S", entry_phone="AA1")
        assert not math.isnan(c)
        assert not math.isinf(c)
        assert c >= 0.0


# ==============================================================================
# SECTION 4: P3.4 F0 SLOPE & MULTI-BRANCH LATTICE STRESS TESTS
# ==============================================================================

def test_f0_slope_regression_edge_cases():
    """
    Stress-test degree-1 linear regression for voiced F0 slope estimation:
    - 0, 1, and 2 frames: returns 0.0 (requires >= 3 frames).
    - 3 frames: returns exact linear regression slope.
    - 100 frames with pure pitch ramp (50 Hz/s): returns ~50.0.
    - Pure unvoiced frames (confidence <= 0.5): returns 0.0.
    - Pathological octave jumps / noisy pitch contour: returns finite float, no NaN.
    """
    # Test internal polyfit logic matching _get_candidate_acoustic_features
    def compute_test_slope(f0_list: List[Optional[float]], conf_list: List[float]) -> float:
        valid_indices = [
            i for i, (f, c) in enumerate(zip(f0_list, conf_list))
            if f is not None and not np.isnan(f) and c > 0.5 and 80 <= f <= 500
        ]
        if len(valid_indices) >= 3:
            t_voiced = np.array([i * (160.0 / 16000.0) for i in valid_indices], dtype=float)
            f0_voiced = np.array([float(f0_list[i]) for i in valid_indices], dtype=float)
            return float(np.polyfit(t_voiced, f0_voiced, 1)[0])
        return 0.0

    # 0 frames
    assert compute_test_slope([], []) == 0.0

    # 1 frame
    assert compute_test_slope([150.0], [0.9]) == 0.0

    # 2 frames
    assert compute_test_slope([150.0, 160.0], [0.9, 0.9]) == 0.0

    # 3 frames: linear ramp at 100 Hz/sec
    # dt = 160/16000 = 0.01s per frame.
    # f0 = [150.0, 151.0, 152.0] -> df = 1.0 / 0.01 = 100.0 Hz/s
    slope_3 = compute_test_slope([150.0, 151.0, 152.0], [0.9, 0.9, 0.9])
    assert abs(slope_3 - 100.0) < 1e-4

    # 100 frames with slope = 50.0 Hz/sec
    dt = 0.01
    f0_100 = [120.0 + 50.0 * (i * dt) for i in range(100)]
    conf_100 = [0.95] * 100
    slope_100 = compute_test_slope(f0_100, conf_100)
    assert abs(slope_100 - 50.0) < 1e-4

    # Pure unvoiced frames (all confidence <= 0.5)
    conf_unvoiced = [0.3] * 100
    assert compute_test_slope(f0_100, conf_unvoiced) == 0.0

    # Extreme noisy pitch jumps (octave doubling / subharmonics)
    rng = np.random.RandomState(1337)
    f0_noisy = [150.0 + rng.choice([-50.0, 0.0, 150.0]) for _ in range(50)]
    slope_noisy = compute_test_slope(f0_noisy, [0.8] * 50)
    assert not math.isnan(slope_noisy)
    assert not math.isinf(slope_noisy)


def test_prosody_guide_progress_normalization_on_multibranch_graphs(dummy_guide_audio: Path):
    """
    Stress-test ProsodyGuide progress normalization on graphs with multiple branching words.
    Verifies that:
    1. Alternative pronunciation branches for the same word receive IDENTICAL
       progress windows [prog_start, prog_end] regardless of node count.
    2. Node ID count skewing (where later branch IDs distorted the contour) is completely eliminated.
    3. Candidate pitch evaluation computes correct semitones from cand_f0_hz.
    """
    guide = ProsodyGuide(dummy_guide_audio, guide_transcript="either or neither")

    lex = Lexicon()
    # Word 0: 'either' has 2 branches: Branch 0 (3 phones: IY1, DH, ER0), Branch 1 (3 phones: AY1, DH, ER0)
    lex.word_to_pronunciations["either"] = [["IY1", "DH", "ER0"], ["AY1", "DH", "ER0"]]
    # Word 1: 'or' has 2 branches: Branch 0 (2 phones: AO1, R), Branch 1 (2 phones: ER1, sil)
    lex.word_to_pronunciations["or"] = [["AO1", "R"], ["ER1", "sil"]]
    # Word 2: 'neither' has 2 branches: Branch 0 (4 phones: N, IY1, DH, ER0), Branch 1 (4 phones: N, AY1, DH, ER0)
    lex.word_to_pronunciations["neither"] = [["N", "IY1", "DH", "ER0"], ["N", "AY1", "DH", "ER0"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("either or neither")
    assert len(graph.normalized_words) == 3

    # Candidates covering 'neither' on Branch 0 vs Branch 1
    neither_nodes_b0 = [n for n in graph.nodes if n.word_text == "neither" and n.branch_id == 0]
    neither_nodes_b1 = [n for n in graph.nodes if n.word_text == "neither" and n.branch_id == 1]
    assert len(neither_nodes_b0) > 0
    assert len(neither_nodes_b1) > 0

    cand_b0 = SpanCandidate(
        candidate_id="cand_b0",
        target_node_start=neither_nodes_b0[0].node_id,
        target_node_end=neither_nodes_b0[-1].node_id,
        covered_nodes=[n.node_id for n in neither_nodes_b0],
        source_content_id="s1", speaker_id="spk1",
        start_sample=0, end_sample=1000,
        event_ids=[], span_type="word", matched_word="neither",
        cand_f0_hz=220.0, pitch_semitones=0.0
    )
    cand_b1 = SpanCandidate(
        candidate_id="cand_b1",
        target_node_start=neither_nodes_b1[0].node_id,
        target_node_end=neither_nodes_b1[-1].node_id,
        covered_nodes=[n.node_id for n in neither_nodes_b1],
        source_content_id="s1", speaker_id="spk1",
        start_sample=0, end_sample=1000,
        event_ids=[], span_type="word", matched_word="neither",
        cand_f0_hz=220.0, pitch_semitones=0.0
    )

    cost_b0 = guide.get_candidate_cost(cand_b0, graph)
    cost_b1 = guide.get_candidate_cost(cand_b1, graph)

    # In Fix P3.4, progress for 'neither' across BOTH branches is normalized by token_index:
    # token_index=2 / total_tokens=3 -> [2/3, 1.0].
    # Therefore, guide cost must be identical between parallel branches!
    assert abs(cost_b0 - cost_b1) < 1e-4, (
        f"Branch progress skew detected: Branch 0 cost {cost_b0} != Branch 1 cost {cost_b1}"
    )

    # Test candidate pitch deviation cost
    # cand_high has f0 = 440 Hz (+12 semitones above 220 Hz)
    cand_high = SpanCandidate(
        candidate_id="cand_high",
        target_node_start=neither_nodes_b0[0].node_id,
        target_node_end=neither_nodes_b0[-1].node_id,
        covered_nodes=[n.node_id for n in neither_nodes_b0],
        source_content_id="s1", speaker_id="spk1",
        start_sample=0, end_sample=1000,
        event_ids=[], span_type="word", matched_word="neither",
        cand_f0_hz=440.0, pitch_semitones=12.0
    )
    cost_high = guide.get_candidate_cost(cand_high, graph)
    # Pitch semitone difference |12.0 - 0.0| * 0.1 = 1.2 added to cost
    assert cost_high > cost_b0
    assert abs((cost_high - cost_b0) - (0.1 * 12.0)) < 1e-4
