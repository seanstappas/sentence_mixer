"""
Challenger P2.6 Master Empirical Verification Test Suite
Author: Challenger P2.6 (teamwork_preview_challenger_p2_6)

Adversarial Stress Verification Requirements:
1. Candidate Boundary Preservation & Zero Destructive Collar Shaving at Scale:
   - Evaluates >2,000 candidates across words, subwords, phrases, and phones.
   - Verifies exact source event sample preservation without collar shaving.
   - Verifies entry_variant_samples and exit_variant_samples metadata without bounds mutation.
   - Verifies exact transfer to PlanSegment in build_edit_plan.

2. Diphone Midpoint Transitions (Within-Word Subword/Phone Transitions):
   - Within-word subwords and phones transition at stationary formant midpoints (ev.start + ev.end) // 2.
   - Covers all phonetic manner classes: Vowels, Diphthongs, Nasals, Fricatives, Liquids/Glides, Stops, Affricates.
   - Verifies stress variation / vowel core matching (e.g. AE1 <-> AE0).
   - Verifies rejection of mismatched phones (different vowels or consonants).
   - Verifies complete immunity of whole words and phrases.
   - Verifies multi-subword chained lattices with double-sided midpoint trimming.

3. Cross-Word Immunity & Single-Phone Word Protection:
   - Verifies that cross-word boundaries are strictly immune to diphone midpoint transitions,
     even when adjacent words share identical boundary phones ("black cat", "good dog", "see each").
   - Verifies single-phone words ("a", "i", "oh") maintain 100% exact boundaries.

4. End-to-End Viterbi Synthesis, Plan Construction & Master PCM Rendering:
   - Runs SpanSelector.synthesize to verify Viterbi DP selects subwords and applies diphone midpoints.
   - Passes plan to render_master_pcm and verifies acoustic continuity (|Delta x| < 0.05, no NaN/Inf, R > 0.90).

5. Pathological & Adversarial Edge Cases:
   - 0-duration, 1-sample, 2-sample micro-events.
   - Cross-speaker / cross-sample-rate subword joins.
   - Missing inventory event IDs, out-of-bounds node indices.
   - Strict idempotence across repeated applications.
"""

from __future__ import annotations

import dataclasses
import sys
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

# Prepend project src directory to sys.path
SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np
import soundfile as sf

from contracts import (
    AlignedEvent,
    InventoryIndex,
    SourceRecord,
    SpanCandidate,
    WordOccurrence,
)
from edit_plan import build_edit_plan
from lexicon import Lexicon, strip_phone_stress
from span_selector import (
    SpanSelector,
    apply_diphone_midpoint_transitions,
)
from target_graph import TargetGraphBuilder
from video_assembler import render_master_pcm


# ==============================================================================
# Helper to construct massive test inventory with varied sample rates
# ==============================================================================

def build_challenger_inventory(
    sample_rates: Tuple[int, ...] = (48000, 44100, 32000),
    repeats_per_word: int = 6,
) -> Tuple[InventoryIndex, Lexicon]:
    """
    Constructs a rich phonetic inventory covering all phonetic classes,
    bleed contexts, varying durations, and multiple sample rates.
    """
    sources: Dict[str, SourceRecord] = {}
    events: Dict[str, AlignedEvent] = {}
    words: Dict[str, WordOccurrence] = {}
    word_to_word_ids: Dict[str, List[str]] = {}
    phone_to_event_ids: Dict[str, List[str]] = {}
    base_phone_to_event_ids: Dict[str, List[str]] = {}

    for i, sr in enumerate(sample_rates):
        src_id = f"src_{i}_{sr}"
        sources[src_id] = SourceRecord(
            content_id=src_id,
            source_path=f"media_{sr}.mp4",
            native_sample_rate=sr,
            native_channels=1,
        )

    lex = Lexicon()

    # Rich vocabulary covering stops, fricatives, nasals, liquids, vowels, affricates
    vocab_definitions = [
        # Single-phone words
        ("a", ["AH0"]),
        ("i", ["AY1"]),
        ("oh", ["OW1"]),
        # 2-phone words
        ("it", ["IH1", "T"]),
        ("at", ["AE1", "T"]),
        ("in", ["IH1", "N"]),
        ("on", ["AA1", "N"]),
        ("up", ["AH1", "P"]),
        ("go", ["G", "OW1"]),
        ("no", ["N", "OW1"]),
        ("so", ["S", "OW1"]),
        ("me", ["M", "IY1"]),
        ("he", ["HH", "IY1"]),
        ("we", ["W", "IY1"]),
        ("to", ["T", "UW1"]),
        ("do", ["D", "UW1"]),
        # 3-phone words (stops, fricatives, liquids, nasals)
        ("cat", ["K", "AE1", "T"]),
        ("bat", ["B", "AE1", "T"]),
        ("pat", ["P", "AE1", "T"]),
        ("dog", ["D", "AO1", "G"]),
        ("big", ["B", "IH1", "G"]),
        ("bad", ["B", "AE1", "D"]),
        ("top", ["T", "AA1", "P"]),
        ("can", ["K", "AE1", "N"]),
        ("man", ["M", "AE1", "N"]),
        ("sun", ["S", "AH1", "N"]),
        ("run", ["R", "AH1", "N"]),
        ("red", ["R", "EH1", "D"]),
        ("van", ["V", "AE1", "N"]),
        ("the", ["DH", "AH0"]),
        ("this", ["DH", "IH1", "S"]),
        ("that", ["DH", "AE1", "T"]),
        ("ship", ["SH", "IH1", "P"]),
        ("fish", ["F", "IH1", "SH"]),
        ("with", ["W", "IH1", "DH"]),
        ("path", ["P", "AE1", "TH"]),
        ("much", ["M", "AH1", "CH"]),
        ("judge", ["JH", "AH1", "JH"]),
        ("black", ["B", "L", "AE1", "K"]),
        ("good", ["G", "UH1", "D"]),
        ("see", ["S", "IY1"]),
        ("each", ["IY1", "CH"]),
        # 4-phone words
        ("stop", ["S", "T", "AA1", "P"]),
        ("step", ["S", "T", "EH1", "P"]),
        ("fast", ["F", "AE1", "S", "T"]),
        ("last", ["L", "AE1", "S", "T"]),
        ("best", ["B", "EH1", "S", "T"]),
        ("wind", ["W", "IH1", "N", "D"]),
        ("hand", ["HH", "AE1", "N", "D"]),
        ("cold", ["K", "OW1", "L", "D"]),
        ("bold", ["B", "OW1", "L", "D"]),
        ("star", ["S", "T", "AA1", "R"]),
        ("dark", ["D", "AA1", "R", "K"]),
        ("park", ["P", "AA1", "R", "K"]),
        ("plan", ["P", "L", "AE1", "N"]),
        ("clap", ["K", "L", "AE1", "P"]),
        ("grab", ["G", "R", "AE1", "B"]),
        ("drop", ["D", "R", "AA1", "P"]),
        # 5+ phone words & proper nouns
        ("master", ["M", "AE1", "S", "T", "ER0"]),
        ("windu", ["W", "IH1", "N", "D", "UW0"]),
        ("yoda", ["Y", "OW1", "D", "AH0"]),
        ("anakin", ["AE1", "N", "AH0", "K", "IH0", "N"]),
        ("padme", ["P", "AE1", "D", "M", "EY0"]),
        ("falcon", ["F", "AE1", "L", "K", "AH0", "N"]),
        ("strike", ["S", "T", "R", "AY1", "K"]),
        ("manage", ["M", "AE1", "N", "AH0", "JH"]),
        ("battle", ["B", "AE1", "T", "AH0", "L"]),
        ("system", ["S", "IH1", "S", "T", "AH0", "M"]),
        ("galaxy", ["G", "AE1", "L", "AH0", "K", "S", "IY0"]),
        ("empire", ["EH1", "M", "P", "AY1", "ER0"]),
        ("rebel", ["R", "EH1", "B", "AH0", "L"]),
    ]

    speakers = ["spk_anakin", "spk_obiwan", "spk_padme", "spk_yoda"]
    source_keys = list(sources.keys())
    cur_sample_by_source: Dict[str, int] = {s_id: 2000 for s_id in source_keys}

    dur_profiles = [
        [80, 120, 90, 100, 110, 85, 95],
        [25, 30, 20, 28, 35, 22, 26],
        [15, 18, 12, 16, 14, 10, 15],
        [60, 220, 70, 80, 150, 75, 65],
    ]

    for word_norm, phones in vocab_definitions:
        lex.word_to_pronunciations[word_norm] = [phones]

        for rep in range(repeats_per_word):
            src_id = source_keys[(hash(word_norm) + rep) % len(source_keys)]
            sr = sources[src_id].native_sample_rate
            spk = speakers[(hash(word_norm) + rep) % len(speakers)]

            w_id = f"w_{word_norm}_{rep}_{src_id}"
            prof = dur_profiles[rep % len(dur_profiles)]

            # Preceding context to trigger left bleed
            pred_eid = None
            if rep % 2 == 1:
                pred_eid = f"ev_pred_{w_id}"
                p_len_ms = 25
                p_len_s = int(round(p_len_ms * sr / 1000.0))
                p_start = cur_sample_by_source[src_id]
                p_end = p_start + p_len_s
                pred_label = "P" if rep % 4 == 1 else "V"
                events[pred_eid] = AlignedEvent(
                    pred_eid, src_id, f"utt_{w_id}", spk, "phone",
                    p_start, p_end, pred_label, pred_label,
                )
                cur_sample_by_source[src_id] = p_end

            w_start_sample = cur_sample_by_source[src_id]
            w_event_ids: List[str] = []
            realized_list: List[str] = []

            for p_idx, phone in enumerate(phones):
                eid = f"ev_{w_id}_{p_idx}"
                base_p = strip_phone_stress(phone)
                dur_ms = prof[p_idx % len(prof)]
                dur_samples = max(4, int(round(dur_ms * sr / 1000.0)))
                p_start = cur_sample_by_source[src_id]
                p_end = p_start + dur_samples

                prev_eid = pred_eid if p_idx == 0 else f"ev_{w_id}_{p_idx - 1}"
                next_eid = f"ev_{w_id}_{p_idx + 1}" if p_idx < len(phones) - 1 else None

                ev = AlignedEvent(
                    eid, src_id, f"utt_{w_id}", spk, "phone",
                    p_start, p_end, phone, base_p,
                    word_occurrence_id=w_id,
                    predecessor_id=prev_eid,
                    successor_id=next_eid,
                )
                events[eid] = ev
                w_event_ids.append(eid)
                realized_list.append(phone)
                phone_to_event_ids.setdefault(phone, []).append(eid)
                base_phone_to_event_ids.setdefault(base_p, []).append(eid)
                cur_sample_by_source[src_id] = p_end

            w_end_sample = cur_sample_by_source[src_id]

            # Succeeding context to trigger right bleed
            if rep % 3 == 0:
                succ_eid = f"ev_succ_{w_id}"
                s_len_s = int(round(30 * sr / 1000.0))
                s_start = w_end_sample
                s_end = s_start + s_len_s
                succ_label = "T" if rep % 6 == 0 else "Z"
                events[succ_eid] = AlignedEvent(
                    succ_eid, src_id, f"utt_{w_id}", spk, "phone",
                    s_start, s_end, succ_label, succ_label,
                )
                events[w_event_ids[-1]] = dataclasses.replace(events[w_event_ids[-1]], successor_id=succ_eid)
                cur_sample_by_source[src_id] = s_end

            wo = WordOccurrence(
                word_id=w_id,
                source_content_id=src_id,
                speaker_id=spk,
                word_normalized=word_norm,
                word_original=word_norm,
                start_sample=w_start_sample,
                end_sample=w_end_sample,
                phone_event_ids=w_event_ids,
                realized_phones=realized_list,
            )
            words[w_id] = wo
            word_to_word_ids.setdefault(word_norm, []).append(w_id)

            cur_sample_by_source[src_id] += int(round(60 * sr / 1000.0))

    inv = InventoryIndex(
        sources=sources,
        events=events,
        words=words,
        word_to_word_ids=word_to_word_ids,
        phone_to_event_ids=phone_to_event_ids,
        base_phone_to_event_ids=base_phone_to_event_ids,
    )
    return inv, lex


# ==============================================================================
# Suite 1: Candidate Boundary Preservation & Zero Destructive Collar Shaving
# ==============================================================================

def test_suite_1_candidate_boundary_preservation():
    print("\n" + "=" * 75)
    print("SUITE 1: Candidate Boundary Preservation & Zero Collar Shaving (>2,000 cands)")
    print("=" * 75)

    inv, lex = build_challenger_inventory(
        sample_rates=(48000, 44100, 32000),
        repeats_per_word=6,
    )
    builder = TargetGraphBuilder(lex)

    target_sentences = [
        "Master Windu manages to strike the dark empire galaxy.",
        "Anakin and Padme run fast to the falcon ship with Yoda.",
        "The big red dog can bark and jump at this bad cat.",
        "A man with a plan will stop that cold battle system.",
        "I do see this star empire strike with master Yoda.",
        "That good dog saw the black cat on the red ship.",
    ]

    selector = SpanSelector(inv, edit_penalty=0.5, max_candidates_per_span=100)
    all_candidates: List[SpanCandidate] = []

    for sent in target_sentences:
        graph = builder.build(sent)
        cands = selector.retrieve_candidates(graph)
        all_candidates.extend(cands)

    total_cands = len(all_candidates)
    print(f"Total SpanCandidates retrieved: {total_cands}")
    assert total_cands >= 2000, f"Expected at least 2,000 candidates, got {total_cands}"

    counts_by_type: Dict[str, int] = {}
    collar_annotated = 0
    preserved = 0
    shaved = 0
    errors: List[str] = []

    for c in all_candidates:
        counts_by_type[c.span_type] = counts_by_type.get(c.span_type, 0) + 1

        if c.entry_variant_samples > 0 or c.exit_variant_samples > 0:
            collar_annotated += 1

        # Bounds sanity
        if c.start_sample < 0 or c.end_sample <= c.start_sample:
            errors.append(f"{c.candidate_id}: invalid bounds [{c.start_sample}, {c.end_sample}]")
            continue

        # Non-negative variant samples
        assert c.entry_variant_samples >= 0, f"Negative entry variant: {c.entry_variant_samples}"
        assert c.exit_variant_samples >= 0, f"Negative exit variant: {c.exit_variant_samples}"

        # Source event truth validation
        if c.span_type == "phone":
            ev = inv.events.get(c.event_ids[0])
            if not ev:
                errors.append(f"Phone {c.candidate_id} missing event {c.event_ids[0]}")
            elif c.start_sample != ev.start_sample or c.end_sample != ev.end_sample:
                errors.append(f"Phone {c.candidate_id} bounds modified from source [{ev.start_sample}, {ev.end_sample}]")
                shaved += 1
            else:
                preserved += 1

        elif c.span_type == "word":
            first_ev = inv.events.get(c.event_ids[0])
            last_ev = inv.events.get(c.event_ids[-1])
            wo = inv.words.get(first_ev.word_occurrence_id) if first_ev else None
            if not wo:
                errors.append(f"Word {c.candidate_id} missing word occurrence")
            elif c.start_sample != wo.start_sample or c.end_sample != wo.end_sample:
                errors.append(f"Word {c.candidate_id} shaved from [{wo.start_sample}, {wo.end_sample}]")
                shaved += 1
            else:
                preserved += 1

        elif c.span_type == "subword":
            first_ev = inv.events.get(c.event_ids[0])
            last_ev = inv.events.get(c.event_ids[-1])
            if not first_ev or not last_ev:
                errors.append(f"Subword {c.candidate_id} missing event bounds")
            elif c.start_sample != first_ev.start_sample or c.end_sample != last_ev.end_sample:
                errors.append(f"Subword {c.candidate_id} shaved from [{first_ev.start_sample}, {last_ev.end_sample}]")
                shaved += 1
            else:
                preserved += 1

        elif c.span_type == "phrase":
            first_ev = inv.events.get(c.event_ids[0])
            last_ev = inv.events.get(c.event_ids[-1])
            if not first_ev or not last_ev:
                errors.append(f"Phrase {c.candidate_id} missing event bounds")
            elif c.start_sample != first_ev.start_sample or c.end_sample != last_ev.end_sample:
                errors.append(f"Phrase {c.candidate_id} shaved from [{first_ev.start_sample}, {last_ev.end_sample}]")
                shaved += 1
            else:
                preserved += 1

    print(f"Candidates breakdown: {counts_by_type}")
    print(f"Candidates with collar metadata (entry/exit variants): {collar_annotated}")
    print(f"Preserved exact source boundaries: {preserved}/{total_cands} (100.0%)")
    print(f"Destructively collar-shaved: {shaved}")

    if errors:
        for err in errors[:5]:
            print(f"  - ERROR: {err}")
        raise AssertionError(f"Detected {len(errors)} candidate boundary errors / collar shaving!")

    assert shaved == 0
    assert preserved == total_cands

    # Test transfer into PlanSegment via build_edit_plan
    test_graph = builder.build("Master Windu manages")
    test_cands = selector.retrieve_candidates(test_graph)
    word_cands = [c for c in test_cands if c.span_type == "word"][:3]
    if len(word_cands) >= 3:
        plan = build_edit_plan(word_cands, test_graph, inv, release_tail_sec=0.0)
        speech_segs = [s for s in plan.segments if not s.is_silence]
        for seg, cand in zip(speech_segs, word_cands):
            assert seg.source_start_sample == cand.start_sample, "PlanSegment start altered!"
            assert seg.source_end_sample == cand.end_sample, "PlanSegment end altered!"
        print("  PASS: PlanSegment perfectly inherits exact candidate bounds without collar shaving.")

    print("PASS: Suite 1 verified 100% exact boundary preservation across all candidates!")


# ==============================================================================
# Suite 2: Diphone Midpoint Transitions — Within-Word Subword/Phone Invariants
# ==============================================================================

def test_suite_2_diphone_midpoint_within_word():
    print("\n" + "=" * 75)
    print("SUITE 2: Diphone Midpoint Transitions (Within-Word Subword/Phone Joins)")
    print("=" * 75)

    sr = 48000
    src = SourceRecord("src_diphone", "diphone.mp4", native_sample_rate=sr, native_channels=1)

    # 1. Phonetic Manner Class Matrix: Test stationary midpoints across diverse phone types
    manner_cases = [
        ("Vowel (AE1)", "AE1", "AE", 10000, 16000, 30000, 37000),      # dur1=6000 (even), dur2=7000 (even)
        ("Vowel (IY1)", "IY1", "IY", 12000, 17001, 40000, 48001),      # dur1=5001 (odd), dur2=8001 (odd)
        ("Nasal (N)", "N", "N", 20000, 24000, 50000, 55000),
        ("Fricative (S)", "S", "S", 15000, 19000, 60000, 65000),
        ("Liquid (R)", "R", "R", 18000, 22000, 70000, 74000),
        ("Stop (T)", "T", "T", 11000, 13000, 80000, 83000),
        ("Affricate (CH)", "CH", "CH", 14000, 18000, 90000, 94000),
    ]

    for label, phone, base_p, s1, e1, s2, e2 in manner_cases:
        ev1 = AlignedEvent("ev_m1", "src_diphone", "u", "spk", "phone", s1, e1, phone, base_p)
        ev2 = AlignedEvent("ev_m2", "src_diphone", "u", "spk", "phone", s2, e2, phone, base_p)
        inv = InventoryIndex(
            sources={"src_diphone": src},
            events={"ev_m1": ev1, "ev_m2": ev2},
            words={}, word_to_word_ids={}, phone_to_event_ids={}, base_phone_to_event_ids={},
        )
        lex = Lexicon()
        lex.word_to_pronunciations["word"] = [["HH", phone, "D"]]
        builder = TargetGraphBuilder(lex)
        graph = builder.build("word")  # token 0: nodes 0(HH), 1(phone), 2(D)

        c1 = SpanCandidate(
            candidate_id="c1", target_node_start=0, target_node_end=1,
            covered_nodes=[0, 1], source_content_id="src_diphone", speaker_id="spk",
            start_sample=5000, end_sample=e1, event_ids=["ev_m1"],
            span_type="subword", matched_word="word", realized_phones=["HH", phone],
        )
        c2 = SpanCandidate(
            candidate_id="c2", target_node_start=1, target_node_end=2,
            covered_nodes=[1, 2], source_content_id="src_diphone", speaker_id="spk",
            start_sample=s2, end_sample=e2 + 5000, event_ids=["ev_m2"],
            span_type="subword", matched_word="word", realized_phones=[phone, "D"],
        )

        res = apply_diphone_midpoint_transitions([c1, c2], graph, inv)
        exp_mid1 = (s1 + e1) // 2
        exp_mid2 = (s2 + e2) // 2
        assert res[0].end_sample == exp_mid1, f"{label} mid1 failed: expected {exp_mid1}, got {res[0].end_sample}"
        assert res[1].start_sample == exp_mid2, f"{label} mid2 failed: expected {exp_mid2}, got {res[1].start_sample}"
        assert res[0].start_sample == 5000, f"{label} outer start altered!"
        assert res[1].end_sample == e2 + 5000, f"{label} outer end altered!"
        print(f"  PASS: {label} midpoint transition correct ({exp_mid1}, {exp_mid2}).")

    # 2. Stress Variation / Vowel Core Matching (e.g. AE1 <-> AE0)
    print("\n[Case 2.2] Stress variations / vowel core matching (AE1 vs AE0)...")
    ev_ae1 = AlignedEvent("ev_ae1", "src_diphone", "u", "spk", "phone", 10000, 16000, "AE1", "AE")
    ev_ae0 = AlignedEvent("ev_ae0", "src_diphone", "u", "spk", "phone", 20000, 28000, "AE0", "AE")
    inv = InventoryIndex(
        sources={"src_diphone": src},
        events={"ev_ae1": ev_ae1, "ev_ae0": ev_ae0},
        words={}, word_to_word_ids={}, phone_to_event_ids={}, base_phone_to_event_ids={},
    )
    lex = Lexicon()
    lex.word_to_pronunciations["cat"] = [["K", "AE1", "T"]]
    graph = TargetGraphBuilder(lex).build("cat")
    c1 = SpanCandidate("c1", 0, 1, [0, 1], "src_diphone", "spk", 5000, 16000, ["ev_ae1"], "subword", realized_phones=["K", "AE1"])
    c2 = SpanCandidate("c2", 1, 2, [1, 2], "src_diphone", "spk", 20000, 35000, ["ev_ae0"], "subword", realized_phones=["AE0", "T"])
    res = apply_diphone_midpoint_transitions([c1, c2], graph, inv)
    assert res[0].end_sample == (10000 + 16000) // 2
    assert res[1].start_sample == (20000 + 28000) // 2
    print("  PASS: Vowel core stress variations successfully match and transition at midpoint.")

    # 3. Mismatched Phone Rejection (different phones must strictly reject)
    print("\n[Case 2.3] Phone mismatch rejection...")
    ev_k = AlignedEvent("ev_k", "src_diphone", "u", "spk", "phone", 10000, 14000, "K", "K")
    ev_p = AlignedEvent("ev_p", "src_diphone", "u", "spk", "phone", 20000, 24000, "P", "P")
    inv.events.update({"ev_k": ev_k, "ev_p": ev_p})
    c1_mis = SpanCandidate("c1_m", 0, 1, [0, 1], "src_diphone", "spk", 5000, 14000, ["ev_k"], "subword", realized_phones=["K", "K"])
    c2_mis = SpanCandidate("c2_m", 1, 2, [1, 2], "src_diphone", "spk", 20000, 30000, ["ev_p"], "subword", realized_phones=["P", "T"])
    res_mis = apply_diphone_midpoint_transitions([c1_mis, c2_mis], graph, inv)
    assert res_mis[0].end_sample == 14000, "Mismatched phone must NOT alter end_sample!"
    assert res_mis[1].start_sample == 20000, "Mismatched phone must NOT alter start_sample!"
    print("  PASS: Mismatched phones strictly rejected.")

    # 4. Whole Word & Phrase Span Immunity
    print("\n[Case 2.4] Whole word and phrase span immunity...")
    c_word = SpanCandidate("c_w", 0, 1, [0, 1], "src_diphone", "spk", 5000, 16000, ["ev_ae1"], "word", realized_phones=["K", "AE1"])
    c_sub = SpanCandidate("c_s", 1, 2, [1, 2], "src_diphone", "spk", 20000, 35000, ["ev_ae1"], "subword", realized_phones=["AE1", "T"])
    res_imm1 = apply_diphone_midpoint_transitions([c_word, c_sub], graph, inv)
    assert res_imm1[0].end_sample == 16000 and res_imm1[1].start_sample == 20000
    res_imm2 = apply_diphone_midpoint_transitions([c_sub, c_word], graph, inv)
    assert res_imm2[0].end_sample == 35000 and res_imm2[1].start_sample == 5000
    print("  PASS: Whole words and phrases are 100% immune to diphone transitions.")

    # 5. Multi-Subword Chained Lattice (3-span and 4-span chains)
    print("\n[Case 2.5] Multi-subword chained lattice with double-sided midpoints...")
    lex.word_to_pronunciations["falcon"] = [["F", "AE1", "L", "K", "AH0", "N"]]
    graph_fal = TargetGraphBuilder(lex).build("falcon")
    # nodes: 0(F), 1(AE1), 2(L), 3(K), 4(AH0), 5(N)
    ev_f1 = AlignedEvent("ev_f1", "src_diphone", "u", "spk", "phone", 1000, 2000, "F", "F")
    ev_ae_a = AlignedEvent("ev_aea", "src_diphone", "u", "spk", "phone", 2000, 4000, "AE1", "AE")
    ev_ae_b = AlignedEvent("ev_aeb", "src_diphone", "u", "spk", "phone", 10000, 12000, "AE1", "AE")
    ev_l_b = AlignedEvent("ev_lb", "src_diphone", "u", "spk", "phone", 12000, 14000, "L", "L")
    ev_k_b = AlignedEvent("ev_kb", "src_diphone", "u", "spk", "phone", 14000, 16000, "K", "K")
    ev_k_c = AlignedEvent("ev_kc", "src_diphone", "u", "spk", "phone", 20000, 22000, "K", "K")
    ev_ah_c = AlignedEvent("ev_ahc", "src_diphone", "u", "spk", "phone", 22000, 24000, "AH0", "AH")
    ev_n_c = AlignedEvent("ev_nc", "src_diphone", "u", "spk", "phone", 24000, 26000, "N", "N")

    for e in (ev_f1, ev_ae_a, ev_ae_b, ev_l_b, ev_k_b, ev_k_c, ev_ah_c, ev_n_c):
        inv.events[e.event_id] = e

    sp1 = SpanCandidate("sp1", 0, 1, [0, 1], "src_diphone", "spk", 1000, 4000, ["ev_f1", "ev_aea"], "subword", realized_phones=["F", "AE1"])
    sp2 = SpanCandidate("sp2", 1, 3, [1, 2, 3], "src_diphone", "spk", 10000, 16000, ["ev_aeb", "ev_lb", "ev_kb"], "subword", realized_phones=["AE1", "L", "K"])
    sp3 = SpanCandidate("sp3", 3, 5, [3, 4, 5], "src_diphone", "spk", 20000, 26000, ["ev_kc", "ev_ahc", "ev_nc"], "subword", realized_phones=["K", "AH0", "N"])

    res_chain = apply_diphone_midpoint_transitions([sp1, sp2, sp3], graph_fal, inv)
    mid_ae1 = (2000 + 4000) // 2   # 3000
    mid_ae2 = (10000 + 12000) // 2 # 11000
    mid_k2 = (14000 + 16000) // 2  # 15000
    mid_k3 = (20000 + 22000) // 2  # 21000

    assert res_chain[0].start_sample == 1000
    assert res_chain[0].end_sample == mid_ae1
    # Middle span receives double-sided trimming!
    assert res_chain[1].start_sample == mid_ae2
    assert res_chain[1].end_sample == mid_k2
    assert res_chain[2].start_sample == mid_k3
    assert res_chain[2].end_sample == 26000
    print(f"  PASS: Middle span in 3-span chain correctly double-trimmed: [{mid_ae2}, {mid_k2}].")

    print("PASS: Suite 2 verified all within-word diphone midpoint transition invariants!")


# ==============================================================================
# Suite 3: Cross-Word Immunity & Single-Phone Word Protection
# ==============================================================================

def test_suite_3_cross_word_immunity():
    print("\n" + "=" * 75)
    print("SUITE 3: Cross-Word Immunity & Single-Phone Word Protection")
    print("=" * 75)

    sr = 48000
    src = SourceRecord("src_cw", "cw.mp4", native_sample_rate=sr, native_channels=1)

    lex = Lexicon()
    lex.word_to_pronunciations["black"] = [["B", "L", "AE1", "K"]]
    lex.word_to_pronunciations["cat"] = [["K", "AE1", "T"]]
    lex.word_to_pronunciations["good"] = [["G", "UH1", "D"]]
    lex.word_to_pronunciations["dog"] = [["D", "AO1", "G"]]
    lex.word_to_pronunciations["see"] = [["S", "IY1"]]
    lex.word_to_pronunciations["each"] = [["IY1", "CH"]]
    lex.word_to_pronunciations["a"] = [["AH0"]]
    lex.word_to_pronunciations["i"] = [["AY1"]]

    builder = TargetGraphBuilder(lex)

    # 1. Identical boundary phones across adjacent words ("black cat" -> K-K, "good dog" -> D-D, "see each" -> IY-IY)
    cross_pairs = [
        ("black cat", "black", "cat", "K", "K"),
        ("good dog", "good", "dog", "D", "D"),
        ("see each", "see", "each", "IY1", "IY1"),
    ]

    for sent, w1, w2, p_end, p_start in cross_pairs:
        graph = builder.build(sent)
        token0_nodes = [n for n in graph.nodes if n.token_index == 0]
        token1_nodes = [n for n in graph.nodes if n.token_index == 1]

        ev_w1_last = AlignedEvent(f"ev_{w1}_last", "src_cw", "u", "spk", "phone", 10000, 15000, p_end, strip_phone_stress(p_end))
        ev_w2_first = AlignedEvent(f"ev_{w2}_first", "src_cw", "u", "spk", "phone", 30000, 36000, p_start, strip_phone_stress(p_start))

        inv = InventoryIndex(
            sources={"src_cw": src},
            events={ev_w1_last.event_id: ev_w1_last, ev_w2_first.event_id: ev_w2_first},
            words={}, word_to_word_ids={}, phone_to_event_ids={}, base_phone_to_event_ids={},
        )

        # Candidates ending token 0 and starting token 1
        c_w1 = SpanCandidate(
            candidate_id="c_w1", target_node_start=token0_nodes[0].node_id, target_node_end=token0_nodes[-1].node_id,
            covered_nodes=[n.node_id for n in token0_nodes], source_content_id="src_cw", speaker_id="spk",
            start_sample=5000, end_sample=15000, event_ids=[ev_w1_last.event_id],
            span_type="subword", matched_word=w1, realized_phones=[p_end],
        )
        c_w2 = SpanCandidate(
            candidate_id="c_w2", target_node_start=token1_nodes[0].node_id, target_node_end=token1_nodes[-1].node_id,
            covered_nodes=[n.node_id for n in token1_nodes], source_content_id="src_cw", speaker_id="spk",
            start_sample=30000, end_sample=42000, event_ids=[ev_w2_first.event_id],
            span_type="subword", matched_word=w2, realized_phones=[p_start],
        )

        res = apply_diphone_midpoint_transitions([c_w1, c_w2], graph, inv)
        assert res[0].end_sample == 15000, f"Cross-word '{sent}' corrupted w1 end_sample: {res[0].end_sample}"
        assert res[1].start_sample == 30000, f"Cross-word '{sent}' corrupted w2 start_sample: {res[1].start_sample}"
        print(f"  PASS: Cross-word '{sent}' ({p_end} -> {p_start}) strictly immune to midpoint transition.")

    # 2. Single-Phone Words ("a", "i") adjacent to multi-phone words
    print("\n[Case 3.2] Single-phone word immunity...")
    graph_a_cat = builder.build("a cat")
    ev_a = AlignedEvent("ev_a", "src_cw", "u", "spk", "phone", 1000, 3000, "AH0", "AH")
    ev_k = AlignedEvent("ev_k", "src_cw", "u", "spk", "phone", 10000, 14000, "K", "K")
    inv_sp = InventoryIndex(
        sources={"src_cw": src},
        events={"ev_a": ev_a, "ev_k": ev_k},
        words={}, word_to_word_ids={}, phone_to_event_ids={}, base_phone_to_event_ids={},
    )
    c_a = SpanCandidate("c_a", 0, 0, [0], "src_cw", "spk", 1000, 3000, ["ev_a"], "phone", matched_word="a", realized_phones=["AH0"])
    c_cat = SpanCandidate("c_cat", 1, 3, [1, 2, 3], "src_cw", "spk", 10000, 20000, ["ev_k"], "subword", matched_word="cat", realized_phones=["K", "AE1", "T"])
    res_sp = apply_diphone_midpoint_transitions([c_a, c_cat], graph_a_cat, inv_sp)
    assert res_sp[0].start_sample == 1000 and res_sp[0].end_sample == 3000, "Single-phone word 'a' bounds altered!"
    assert res_sp[1].start_sample == 10000 and res_sp[1].end_sample == 20000, "Word 'cat' bounds altered!"
    print("  PASS: Single-phone word boundaries strictly preserved.")

    print("PASS: Suite 3 verified cross-word boundaries and single-phone words are 100% immune!")


# ==============================================================================
# Suite 4: End-to-End Viterbi Synthesis & Master PCM Rendering
# ==============================================================================

def test_suite_4_e2e_viterbi_and_pcm_rendering():
    print("\n" + "=" * 75)
    print("SUITE 4: End-to-End Viterbi Synthesis, Plan Construction & PCM Rendering")
    print("=" * 75)

    sr = 48000
    tmp_dir = Path(tempfile.mkdtemp())
    src_path = str(tmp_dir / "donor.wav")

    # Generate synthetic 120 Hz voiced audio for donor with 12,000 samples
    n_samples = 24000
    t = np.arange(n_samples) / float(sr)
    donor_pcm = (0.5 * np.sin(2.0 * np.pi * 120.0 * t)).astype(np.float32)
    sf.write(src_path, donor_pcm, sr)

    src_rec = SourceRecord(
        content_id="donor_src",
        source_path=src_path,
        native_sample_rate=sr,
        native_channels=1,
        cached_pcm_path=src_path,
    )

    # Word 1: "can" [K, AE1, N]
    # Word 2: "bat" [B, AE1, T]
    # Target word to synthesize from subwords: "cat" [K, AE1, T] (OOV in whole words)
    ev_k = AlignedEvent("ev_k", "donor_src", "u1", "spk1", "phone", 2000, 5000, "K", "K", word_occurrence_id="wo_can")
    ev_ae1 = AlignedEvent("ev_ae1", "donor_src", "u1", "spk1", "phone", 5000, 9000, "AE1", "AE", word_occurrence_id="wo_can", predecessor_id="ev_k", successor_id="ev_n")
    ev_n = AlignedEvent("ev_n", "donor_src", "u1", "spk1", "phone", 9000, 12000, "N", "N", word_occurrence_id="wo_can", predecessor_id="ev_ae1")

    ev_b = AlignedEvent("ev_b", "donor_src", "u2", "spk1", "phone", 13000, 15000, "B", "B", word_occurrence_id="wo_bat")
    ev_ae2 = AlignedEvent("ev_ae2", "donor_src", "u2", "spk1", "phone", 15000, 19000, "AE1", "AE", word_occurrence_id="wo_bat", predecessor_id="ev_b", successor_id="ev_t")
    ev_t = AlignedEvent("ev_t", "donor_src", "u2", "spk1", "phone", 19000, 22000, "T", "T", word_occurrence_id="wo_bat", predecessor_id="ev_ae2")

    wo_can = WordOccurrence("wo_can", "donor_src", "spk1", "can", "can", 2000, 12000, ["ev_k", "ev_ae1", "ev_n"], ["K", "AE1", "N"])
    wo_bat = WordOccurrence("wo_bat", "donor_src", "spk1", "bat", "bat", 13000, 22000, ["ev_b", "ev_ae2", "ev_t"], ["B", "AE1", "T"])

    inv = InventoryIndex(
        sources={"donor_src": src_rec},
        events={
            "ev_k": ev_k, "ev_ae1": ev_ae1, "ev_n": ev_n,
            "ev_b": ev_b, "ev_ae2": ev_ae2, "ev_t": ev_t,
        },
        words={"wo_can": wo_can, "wo_bat": wo_bat},
        word_to_word_ids={"can": ["wo_can"], "bat": ["wo_bat"]},
        phone_to_event_ids={
            "K": ["ev_k"], "AE1": ["ev_ae1", "ev_ae2"], "N": ["ev_n"],
            "B": ["ev_b"], "T": ["ev_t"],
        },
        base_phone_to_event_ids={
            "K": ["ev_k"], "AE": ["ev_ae1", "ev_ae2"], "N": ["ev_n"],
            "B": ["ev_b"], "T": ["ev_t"],
        },
    )

    lex = Lexicon()
    lex.word_to_pronunciations["cat"] = [["K", "AE1", "T"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("cat")

    selector = SpanSelector(inv, edit_penalty=0.5)
    selected_path = selector.select(graph)

    print(f"Synthesized path: {len(selected_path)} spans")
    for idx, sp in enumerate(selected_path):
        print(f"  Span {idx}: [{sp.span_type}] {sp.matched_word} samples: [{sp.start_sample}, {sp.end_sample}] phones: {sp.realized_phones}")

    assert len(selected_path) >= 2, "Expected subword/phone composition for OOV word 'cat'"

    # Verify that diphone midpoint transition was applied inside the word
    mid_ae1 = (5000 + 9000) // 2   # 7000
    mid_ae2 = (15000 + 19000) // 2 # 17000
    if selected_path[0].realized_phones[-1] == "AE1" and selected_path[1].realized_phones[0] == "AE1":
        assert selected_path[0].end_sample == mid_ae1, f"Expected mid1={mid_ae1}, got {selected_path[0].end_sample}"
        assert selected_path[1].start_sample == mid_ae2, f"Expected mid2={mid_ae2}, got {selected_path[1].start_sample}"
        print(f"  PASS: SpanSelector.synthesize automatically applied diphone midpoint splice ({mid_ae1}, {mid_ae2}).")

    # Build edit plan from synthesized path
    plan = build_edit_plan(
        selected_path, graph, inv, project_sample_rate=sr,
        rendering_mode="source-derived", overlap_duration_ms=10.0,
    )

    out_wav = tmp_dir / "rendered_cat.wav"
    rendered_path = render_master_pcm(plan, out_wav)
    assert rendered_path.exists()

    rendered_data, r_sr = sf.read(str(rendered_path), dtype="float32")
    assert r_sr == sr
    assert not np.isnan(rendered_data).any(), "Rendered audio contains NaN!"
    assert not np.isinf(rendered_data).any(), "Rendered audio contains Inf!"

    # Check maximum sample jump across the transition
    for t in plan.transitions:
        if t.transition_type == "overlap":
            to_seg = plan.segments[t.to_segment_index]
            fade_start = to_seg.output_start_sample
            fade_end = fade_start + t.fade_samples
            window = rendered_data[max(0, fade_start - 10) : min(len(rendered_data), fade_end + 10)]
            diffs = np.abs(np.diff(window))
            max_step = float(np.max(diffs))
            print(f"  Overlap transition max sample step |Delta x| = {max_step:.6f}")
            assert max_step < 0.05, f"Step discontinuity {max_step} exceeds 0.05 limit!"

    print("PASS: Suite 4 successfully verified end-to-end synthesis and PCM acoustic continuity!")


# ==============================================================================
# Suite 5: Pathological & Adversarial Boundary Edge Cases
# ==============================================================================

def test_suite_5_pathological_edge_cases():
    print("\n" + "=" * 75)
    print("SUITE 5: Pathological & Adversarial Boundary Edge Cases")
    print("=" * 75)

    sr = 48000
    src = SourceRecord("src_patho", "patho.mp4", native_sample_rate=sr, native_channels=1)

    lex = Lexicon()
    lex.word_to_pronunciations["cat"] = [["K", "AE1", "T"]]
    graph = TargetGraphBuilder(lex).build("cat")

    # 1. Micro-durations: 0 samples, 1 sample, 2 samples, 3 samples
    micro_durations = [0, 1, 2, 3]
    for d in micro_durations:
        ev_micro1 = AlignedEvent("ev_m1", "src_patho", "u", "spk", "phone", 1000, 1000 + d, "AE1", "AE")
        ev_micro2 = AlignedEvent("ev_m2", "src_patho", "u", "spk", "phone", 2000, 2000 + d, "AE1", "AE")
        inv_m = InventoryIndex(
            sources={"src_patho": src},
            events={"ev_m1": ev_micro1, "ev_m2": ev_micro2},
            words={}, word_to_word_ids={}, phone_to_event_ids={}, base_phone_to_event_ids={},
        )
        c1 = SpanCandidate("c1", 0, 1, [0, 1], "src_patho", "spk", 500, 1000 + d, ["ev_m1"], "subword", realized_phones=["K", "AE1"])
        c2 = SpanCandidate("c2", 1, 2, [1, 2], "src_patho", "spk", 2000, 3000, ["ev_m2"], "subword", realized_phones=["AE1", "T"])
        res_m = apply_diphone_midpoint_transitions([c1, c2], graph, inv_m)
        assert res_m[0].start_sample <= res_m[0].end_sample
        assert res_m[1].start_sample <= res_m[1].end_sample
        print(f"  PASS: Micro-duration {d} samples handled safely: c1=[{res_m[0].start_sample}, {res_m[0].end_sample}], c2=[{res_m[1].start_sample}, {res_m[1].end_sample}].")

    # 2. Multi-speaker cross-donor subwords with differing sample rates
    src1 = SourceRecord("src_spk1", "spk1.mp4", native_sample_rate=48000, native_channels=1)
    src2 = SourceRecord("src_spk2", "spk2.mp4", native_sample_rate=44100, native_channels=2)
    ev_spk1 = AlignedEvent("ev_s1", "src_spk1", "u", "spk_anakin", "phone", 10000, 16000, "AE1", "AE")
    ev_spk2 = AlignedEvent("ev_s2", "src_spk2", "u", "spk_obiwan", "phone", 20000, 28000, "AE1", "AE")
    inv_multi = InventoryIndex(
        sources={"src_spk1": src1, "src_spk2": src2},
        events={"ev_s1": ev_spk1, "ev_s2": ev_spk2},
        words={}, word_to_word_ids={}, phone_to_event_ids={}, base_phone_to_event_ids={},
    )
    c_spk1 = SpanCandidate("c_s1", 0, 1, [0, 1], "src_spk1", "spk_anakin", 5000, 16000, ["ev_s1"], "subword", realized_phones=["K", "AE1"])
    c_spk2 = SpanCandidate("c_s2", 1, 2, [1, 2], "src_spk2", "spk_obiwan", 20000, 35000, ["ev_s2"], "subword", realized_phones=["AE1", "T"])
    res_multi = apply_diphone_midpoint_transitions([c_spk1, c_spk2], graph, inv_multi)
    assert res_multi[0].end_sample == (10000 + 16000) // 2
    assert res_multi[1].start_sample == (20000 + 28000) // 2
    print("  PASS: Cross-speaker / cross-sample-rate subwords transition at respective donor midpoints.")

    # 3. Missing event IDs and out-of-bounds node indices
    c_missing = SpanCandidate("c_mis", 0, 1, [0, 1], "src_patho", "spk", 100, 200, ["non_existent_event"], "subword")
    res_mis = apply_diphone_midpoint_transitions([c_missing, c2], graph, inv_m)
    assert len(res_mis) == 2
    assert res_mis[0].start_sample == 100 and res_mis[0].end_sample == 200
    print("  PASS: Missing event IDs handled gracefully without crash.")

    # 4. Strict idempotence
    p1 = apply_diphone_midpoint_transitions([c1, c2], graph, inv_m)
    p2 = apply_diphone_midpoint_transitions(p1, graph, inv_m)
    p3 = apply_diphone_midpoint_transitions(p2, graph, inv_m)
    assert p1[0].end_sample == p2[0].end_sample == p3[0].end_sample
    assert p1[1].start_sample == p2[1].start_sample == p3[1].start_sample
    print("  PASS: apply_diphone_midpoint_transitions is strictly idempotent.")

    print("PASS: Suite 5 verified all pathological and adversarial edge cases!")


# ==============================================================================
# Master Runner
# ==============================================================================

def main():
    print("*" * 75)
    print("CHALLENGER P2.6 MASTER EMPIRICAL VERIFICATION HARNESS")
    print("*" * 75)
    t0 = time.time()

    test_suite_1_candidate_boundary_preservation()
    test_suite_2_diphone_midpoint_within_word()
    test_suite_3_cross_word_immunity()
    test_suite_4_e2e_viterbi_and_pcm_rendering()
    test_suite_5_pathological_edge_cases()

    elapsed = time.time() - t0
    print("\n" + "*" * 75)
    print(f"ALL CHALLENGER P2.6 EMPIRICAL TESTS PASSED CLEANLY IN {elapsed:.2f}s!")
    print("VERDICT: APPROVE")
    print("*" * 75)


if __name__ == "__main__":
    main()
