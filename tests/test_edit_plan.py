"""
Tests for canonical EditPlan construction and timing invariants.
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from contracts import (
    AlignedEvent,
    EditPlan,
    InventoryIndex,
    PlanSegment,
    PlanTransition,
    SourceRecord,
    SpanCandidate,
)
from edit_plan import build_edit_plan, refine_splice_boundary
from lexicon import Lexicon
from target_graph import TargetGraphBuilder
from video_assembler import render_master_pcm


def test_build_edit_plan_timing_and_pauses():
    sr = 48000
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )

    ev1 = AlignedEvent(
        event_id="ev1", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=24000, label="HH", base_phone="HH"
    )
    ev2 = AlignedEvent(
        event_id="ev2", source_content_id="src_1", utterance_id="u2", speaker_id="spk1",
        kind="phone", start_sample=48000, end_sample=72000, label="DH", base_phone="DH"
    )

    inv = InventoryIndex(
        sources={"src_1": src},
        events={"ev1": ev1, "ev2": ev2},
    )

    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH"]]
    lex.word_to_pronunciations["there"] = [["DH"]]

    builder = TargetGraphBuilder(lex)
    # Target text with comma pause: 'Hello, there'
    graph = builder.build("Hello, there")

    cand1 = SpanCandidate(
        candidate_id="c1",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="src_1",
        speaker_id="spk1",
        start_sample=0,
        end_sample=24000,
        event_ids=["ev1"],
        span_type="word",
        matched_word="hello",
    )

    cand2 = SpanCandidate(
        candidate_id="c2",
        target_node_start=1,
        target_node_end=1,
        covered_nodes=[1],
        source_content_id="src_1",
        speaker_id="spk1",
        start_sample=48000,
        end_sample=72000,
        event_ids=["ev2"],
        span_type="word",
        matched_word="there",
    )

    plan = build_edit_plan([cand1, cand2], graph, inv, project_sample_rate=sr)
    plan.validate()

    # Verify pause insertion: comma creates pause segment between word 0 and word 1
    pause_segs = [s for s in plan.segments if s.is_silence]
    assert len(pause_segs) == 1, "Punctuation comma must schedule an explicit silence pause segment"
    assert pause_segs[0].target_text == "[pause]"

    # Invariant: sample count equation
    seg_samples = sum(s.duration_samples for s in plan.segments)
    overlap_samples = sum(t.overlap_samples for t in plan.transitions if t.transition_type == "overlap")
    assert plan.total_output_samples == seg_samples - overlap_samples


def test_t01_six_50ms_cuts_cumulative_frames():
    """T01: 6 x 50-ms cuts, 48 kHz, 30 fps, cuts-only -> 14,400 master samples; 9 actual video frames."""
    from contracts import RationalTime, TargetGraph, TargetPhoneNode

    sr = 48000
    fps = RationalTime(30, 1)
    cut_samples = int(round(0.050 * sr))  # 2400
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )
    cands = []
    events = {}
    nodes = []
    for i in range(6):
        ev_id = f"ev_{i}"
        events[ev_id] = AlignedEvent(
            event_id=ev_id, source_content_id="src_1", utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=i * cut_samples, end_sample=(i + 1) * cut_samples,
            label=f"P_{i}", base_phone="AA"
        )
        nodes.append(TargetPhoneNode(
            node_id=i, token_index=i, word_text=f"w{i}", phone=f"P_{i}", base_phone="AA",
            stress=1, is_word_initial=True, is_word_final=True, is_phrase_final=(i == 5)
        ))
        cands.append(SpanCandidate(
            candidate_id=f"c_{i}",
            target_node_start=i,
            target_node_end=i,
            covered_nodes=[i],
            source_content_id="src_1",
            speaker_id="spk1",
            start_sample=i * cut_samples,
            end_sample=(i + 1) * cut_samples,
            event_ids=[ev_id],
            span_type="word",
            matched_word=f"w{i}",
        ))

    inv = InventoryIndex(sources={"src_1": src}, events=events)
    graph = TargetGraph(
        raw_text="w0 w1 w2 w3 w4 w5",
        normalized_words=[f"w{i}" for i in range(6)],
        nodes=nodes,
        edges=[(i, i + 1) for i in range(5)],
        start_nodes=[0],
        end_nodes=[5],
    )
    plan = build_edit_plan(cands, graph, inv, project_sample_rate=sr, video_fps=fps, rendering_mode="cuts-only")
    plan.validate()
    assert plan.total_output_samples == 14400
    assert plan.total_video_frames == 9
    assert sum(s.video_frame_count for s in plan.segments) == 9


def test_t02_1000_cuts_cumulative_frames():
    """T02: 1,000 x 50-ms cuts -> 2,400,000 samples; 1,500 actual frames; bounded switch error."""
    from contracts import RationalTime, TargetGraph, TargetPhoneNode

    sr = 48000
    fps = RationalTime(30, 1)
    cut_samples = int(round(0.050 * sr))  # 2400
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )
    N = 1000
    cands = []
    events = {}
    nodes = []
    for i in range(N):
        ev_id = f"ev_{i}"
        events[ev_id] = AlignedEvent(
            event_id=ev_id, source_content_id="src_1", utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=i * cut_samples, end_sample=(i + 1) * cut_samples,
            label="AA", base_phone="AA"
        )
        nodes.append(TargetPhoneNode(
            node_id=i, token_index=i, word_text=f"w{i}", phone="AA", base_phone="AA",
            stress=1, is_word_initial=True, is_word_final=True, is_phrase_final=(i == N - 1)
        ))
        cands.append(SpanCandidate(
            candidate_id=f"c_{i}",
            target_node_start=i,
            target_node_end=i,
            covered_nodes=[i],
            source_content_id="src_1",
            speaker_id="spk1",
            start_sample=i * cut_samples,
            end_sample=(i + 1) * cut_samples,
            event_ids=[ev_id],
            span_type="word",
            matched_word=f"w{i}",
        ))

    inv = InventoryIndex(sources={"src_1": src}, events=events)
    graph = TargetGraph(
        raw_text=" ".join(f"w{i}" for i in range(N)),
        normalized_words=[f"w{i}" for i in range(N)],
        nodes=nodes,
        edges=[(i, i + 1) for i in range(N - 1)],
        start_nodes=[0],
        end_nodes=[N - 1],
    )
    plan = build_edit_plan(cands, graph, inv, project_sample_rate=sr, video_fps=fps, rendering_mode="cuts-only")
    plan.validate()
    assert plan.total_output_samples == 2400000
    assert plan.total_video_frames == 1500
    assert sum(s.video_frame_count for s in plan.segments) == 1500


def test_t03_subframe_cuts_and_fractional_fps():
    """T03: Contributions shorter than a frame; fractional FPS. Audio preserved; zero-frame picture spans allowed."""
    from contracts import RationalTime, TargetGraph, TargetPhoneNode
    from media_time import cumulative_frame_ceiling

    sr = 48000
    fps = RationalTime(30000, 1001)  # 29.97 fps NTSC
    # 5ms cuts = 240 samples each (each frame is ~1601.6 samples)
    cut_samples = 240
    N = 20
    src = SourceRecord(
        content_id="src_sub",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=1,
    )
    cands = []
    events = {}
    nodes = []
    for i in range(N):
        ev_id = f"ev_sub_{i}"
        events[ev_id] = AlignedEvent(
            event_id=ev_id, source_content_id="src_sub", utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=i * cut_samples, end_sample=(i + 1) * cut_samples,
            label="AA", base_phone="AA"
        )
        nodes.append(TargetPhoneNode(
            node_id=i, token_index=i, word_text=f"w{i}", phone="AA", base_phone="AA",
            stress=1, is_word_initial=True, is_word_final=True, is_phrase_final=(i == N - 1)
        ))
        cands.append(SpanCandidate(
            candidate_id=f"c_{i}",
            target_node_start=i,
            target_node_end=i,
            covered_nodes=[i],
            source_content_id="src_sub",
            speaker_id="spk1",
            start_sample=i * cut_samples,
            end_sample=(i + 1) * cut_samples,
            event_ids=[ev_id],
            span_type="word",
            matched_word=f"w{i}",
        ))

    inv = InventoryIndex(sources={"src_sub": src}, events=events)
    graph = TargetGraph(
        raw_text=" ".join(f"w{i}" for i in range(N)),
        normalized_words=[f"w{i}" for i in range(N)],
        nodes=nodes,
        edges=[(i, i + 1) for i in range(N - 1)],
        start_nodes=[0],
        end_nodes=[N - 1],
    )
    plan = build_edit_plan(cands, graph, inv, project_sample_rate=sr, video_fps=fps, rendering_mode="cuts-only")
    plan.validate()

    assert plan.total_output_samples == N * cut_samples  # 4800
    expected_frames = cumulative_frame_ceiling(plan.total_output_samples, sr, fps)
    assert plan.total_video_frames == expected_frames
    assert sum(s.video_frame_count for s in plan.segments) == expected_frames
    # Subframe segments must contain zero-frame spans without error
    zero_frame_segs = [s for s in plan.segments if s.video_frame_count == 0]
    assert len(zero_frame_segs) > 0, "Subframe cuts must produce valid zero-frame picture spans"


def test_t04_exact_gap_250ms_samples_and_phrase_blocking():
    """T04: 'hello <gap:250ms> there' with contiguous source phrase available -> exactly 12,000 gap samples; no swallowed gap."""
    from contracts import WordOccurrence
    from span_selector import SpanSelector

    sr = 48000
    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH", "EH1", "L", "OW0"]]
    lex.word_to_pronunciations["there"] = [["DH", "EH1", "R"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("hello <gap:250ms> there")

    # Verify explicit_gaps is populated on graph
    assert graph.explicit_gaps.get(0) == 0.250

    src = SourceRecord(
        content_id="src_phrase",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=1,
    )
    # Source has consecutive words "hello" and "there"
    ev1 = AlignedEvent(
        event_id="ev_h", source_content_id="src_phrase", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=24000, label="HH", base_phone="HH",
        word_occurrence_id="wo_h", successor_id="ev_t"
    )
    ev2 = AlignedEvent(
        event_id="ev_t", source_content_id="src_phrase", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=24000, end_sample=48000, label="DH", base_phone="DH",
        word_occurrence_id="wo_t"
    )
    wo1 = WordOccurrence(
        word_id="wo_h", source_content_id="src_phrase", speaker_id="spk1",
        word_normalized="hello", word_original="hello", start_sample=0, end_sample=24000,
        phone_event_ids=["ev_h"], realized_phones=["HH"]
    )
    wo2 = WordOccurrence(
        word_id="wo_t", source_content_id="src_phrase", speaker_id="spk1",
        word_normalized="there", word_original="there", start_sample=24000, end_sample=48000,
        phone_event_ids=["ev_t"], realized_phones=["DH"]
    )


    inv = InventoryIndex(
        sources={"src_phrase": src},
        events={"ev_h": ev1, "ev_t": ev2},
        words={"wo_h": wo1, "wo_t": wo2},
        word_to_word_ids={"hello": ["wo_h"], "there": ["wo_t"]},
    )

    selector = SpanSelector(inv)
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")


    # SpanSelector MUST NOT create a single phrase candidate covering both words across <gap:250ms>
    phrase_cands = [c for c in cands if c.span_type == "phrase"]
    assert len(phrase_cands) == 0, "Phrase candidates must not span across explicit gaps"

    # Select word candidates
    c1 = next(c for c in cands if c.matched_word == "hello")
    c2 = next(c for c in cands if c.matched_word == "there")

    plan = build_edit_plan([c1, c2], graph, inv, project_sample_rate=sr, rendering_mode="cuts-only")
    plan.validate()

    pause_segs = [s for s in plan.segments if s.is_silence]
    assert len(pause_segs) == 1
    # Exactly 250ms at 48000 Hz = 12,000 samples
    assert pause_segs[0].duration_samples == 12000


def test_t05_zero_leading_trailing_consecutive_gaps():
    """T05: Zero, leading, trailing, consecutive gaps; cuts-only."""
    sr = 48000
    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH"]]
    lex.word_to_pronunciations["there"] = [["DH"]]

    builder = TargetGraphBuilder(lex)

    # 1. Zero gap: <gap:0ms> should have 0 samples
    g_zero = builder.build("hello <gap:0ms> there")
    assert g_zero.explicit_gaps.get(0) == 0.0

    # 2. Consecutive gaps: <gap:100ms> <gap:150ms> add together to 250ms = 12000 samples
    g_consec = builder.build("hello <gap:100ms> <gap:150ms> there")
    assert abs(g_consec.explicit_gaps.get(0) - 0.250) < 1e-5

    # 3. Leading and trailing gaps
    g_flanked = builder.build("<gap:100ms> hello <gap:200ms> there <gap:300ms>")
    assert abs(g_flanked.explicit_gaps.get(-1) - 0.100) < 1e-5
    assert abs(g_flanked.explicit_gaps.get(0) - 0.200) < 1e-5
    assert abs(g_flanked.explicit_gaps.get(1) - 0.300) < 1e-5

    # Build plan for flanked target
    src = SourceRecord(
        content_id="src_1", source_path="video.mp4", native_sample_rate=sr, native_channels=1
    )
    ev1 = AlignedEvent(
        event_id="ev1", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=24000, label="HH", base_phone="HH"
    )
    ev2 = AlignedEvent(
        event_id="ev2", source_content_id="src_1", utterance_id="u2", speaker_id="spk1",
        kind="phone", start_sample=48000, end_sample=72000, label="DH", base_phone="DH"
    )
    inv = InventoryIndex(sources={"src_1": src}, events={"ev1": ev1, "ev2": ev2})
    cand1 = SpanCandidate(
        candidate_id="c1", target_node_start=0, target_node_end=0, covered_nodes=[0],
        source_content_id="src_1", speaker_id="spk1", start_sample=0, end_sample=24000,
        event_ids=["ev1"], span_type="word", matched_word="hello",
    )
    cand2 = SpanCandidate(
        candidate_id="c2", target_node_start=1, target_node_end=1, covered_nodes=[1],
        source_content_id="src_1", speaker_id="spk1", start_sample=48000, end_sample=72000,
        event_ids=["ev2"], span_type="word", matched_word="there",
    )

    plan = build_edit_plan([cand1, cand2], g_flanked, inv, project_sample_rate=sr, rendering_mode="cuts-only")
    plan.validate()

    pauses = [s for s in plan.segments if s.is_silence]
    assert len(pauses) == 3, "Leading, internal, and trailing gaps must produce 3 pause segments"
    assert pauses[0].duration_samples == int(round(0.100 * sr))  # 4800
    assert pauses[1].duration_samples == int(round(0.200 * sr))  # 9600
    assert pauses[2].duration_samples == int(round(0.300 * sr))  # 14400


def test_continuation_splice_requires_exact_sample_abutment():
    """
    Validates Fix P1.2:
    Continuation splice requires BOTH exact sample abutment (span.end_sample == next_span.start_sample)
    AND successor event linkage.
    A collar-trimmed or excised pair (end_sample != start_sample) must NOT be marked continuation;
    it must receive a standard crossfaded transition (overlap).
    """
    sr = 48000
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )
    ev1 = AlignedEvent(
        event_id="ev1",
        source_content_id="src_1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        start_sample=0,
        end_sample=24000,
        label="HH",
        base_phone="HH",
        successor_id="ev2",
    )
    ev2 = AlignedEvent(
        event_id="ev2",
        source_content_id="src_1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        start_sample=24000,
        end_sample=48000,
        label="DH",
        base_phone="DH",
        predecessor_id="ev1",
    )
    inv = InventoryIndex(sources={"src_1": src}, events={"ev1": ev1, "ev2": ev2})

    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH"]]
    lex.word_to_pronunciations["there"] = [["DH"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("hello there")

    # Case A: True abutting continuation (end_sample == 24000, start_sample == 24000)
    cand1_abut = SpanCandidate(
        candidate_id="c1",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="src_1",
        speaker_id="spk1",
        start_sample=0,
        end_sample=24000,
        event_ids=["ev1"],
        span_type="word",
        matched_word="hello",
    )
    cand2_abut = SpanCandidate(
        candidate_id="c2",
        target_node_start=1,
        target_node_end=1,
        covered_nodes=[1],
        source_content_id="src_1",
        speaker_id="spk1",
        start_sample=24000,
        end_sample=48000,
        event_ids=["ev2"],
        span_type="word",
        matched_word="there",
    )
    plan_abut = build_edit_plan([cand1_abut, cand2_abut], graph, inv, project_sample_rate=sr, finalize=False)
    assert len(plan_abut.transitions) == 1
    assert plan_abut.transitions[0].transition_type == "continuation"
    assert plan_abut.transitions[0].overlap_samples == 0

    # Case B: Non-abutting pair (collar-trimmed: end_sample=23000, next start_sample=24000)
    # Despite ev1.successor_id == ev2.event_id, the 1000-sample excised gap must NOT be continuation.
    cand1_gap = SpanCandidate(
        candidate_id="c1_gap",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="src_1",
        speaker_id="spk1",
        start_sample=0,
        end_sample=23000,
        event_ids=["ev1"],
        span_type="word",
        matched_word="hello",
    )
    plan_gap = build_edit_plan(
        [cand1_gap, cand2_abut],
        graph,
        inv,
        project_sample_rate=sr,
        rendering_mode="source-derived",
        finalize=False,
    )
    assert len(plan_gap.transitions) == 1
    assert plan_gap.transitions[0].transition_type == "overlap"
    assert plan_gap.transitions[0].overlap_samples > 0
    assert plan_gap.transitions[0].fade_samples > 0


def test_build_edit_plan_finalize_flag():
    """
    Validates Fix P1.3 finalize parameter on build_edit_plan:
    When finalize=False, loudnorm Pass-1 and finalize_edit_plan are bypassed.
    When finalize=True, finalize_edit_plan is invoked.
    """
    sr = 48000
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )
    ev1 = AlignedEvent(
        event_id="ev1",
        source_content_id="src_1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        start_sample=0,
        end_sample=24000,
        label="HH",
        base_phone="HH",
    )
    inv = InventoryIndex(sources={"src_1": src}, events={"ev1": ev1})

    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("hello")

    cand1 = SpanCandidate(
        candidate_id="c1",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="src_1",
        speaker_id="spk1",
        start_sample=0,
        end_sample=24000,
        event_ids=["ev1"],
        span_type="word",
        matched_word="hello",
    )

    # finalize=False: finalize_edit_plan is bypassed
    plan_unfinalized = build_edit_plan([cand1], graph, inv, project_sample_rate=sr, finalize=False)
    assert "normalize_loudness" not in plan_unfinalized.metadata
    assert "ebur128_analysis" not in plan_unfinalized.metadata

    # finalize=True: runs finalize_edit_plan and populates metadata
    plan_finalized = build_edit_plan([cand1], graph, inv, project_sample_rate=sr, finalize=True)
    assert "normalize_loudness" in plan_finalized.metadata
    assert plan_finalized.metadata["normalize_loudness"] is True


def test_span_candidate_word_bleed_variant_samples():
    """
    Validates Fix P1.4:
    Whole-word candidate retrieval populates entry_variant_samples and exit_variant_samples
    on SpanCandidate non-destructively without mutating start_sample or end_sample.
    """
    from contracts import WordOccurrence
    from span_selector import SpanSelector

    sr = 48000
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )

    # Predecessor phone 'V', word 'all' [AA1, L], successor phone 'K'
    # Start node 'AA' is a vowel; predecessor 'V' triggers l_bleed_cost = 0.80 -> entry_variant = 45ms.
    # Target utterance 'all' is phrase-final; successor phone 'K' triggers r_bleed_cost = 0.35 -> exit_variant = 15ms.
    ev_pred = AlignedEvent(
        event_id="ev_pred", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=5000, label="V", base_phone="V",
        successor_id="ev_aa"
    )
    ev_aa = AlignedEvent(
        event_id="ev_aa", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=5000, end_sample=15000, label="AA1", base_phone="AA",
        predecessor_id="ev_pred", successor_id="ev_l"
    )
    ev_l = AlignedEvent(
        event_id="ev_l", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=15000, end_sample=25000, label="L", base_phone="L",
        predecessor_id="ev_aa", successor_id="ev_succ"
    )
    ev_succ = AlignedEvent(
        event_id="ev_succ", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=25000, end_sample=30000, label="K", base_phone="K",
        predecessor_id="ev_l"
    )

    wo_bleed = WordOccurrence(
        word_id="w_all_1", source_content_id="src_1", speaker_id="spk1",
        word_normalized="all", word_original="all",
        start_sample=5000, end_sample=25000,
        phone_event_ids=["ev_aa", "ev_l"],
        realized_phones=["AA1", "L"],
    )

    inv = InventoryIndex(
        sources={"src_1": src},
        events={"ev_pred": ev_pred, "ev_aa": ev_aa, "ev_l": ev_l, "ev_succ": ev_succ},
        words={"w_all_1": wo_bleed},
        word_to_word_ids={"all": ["w_all_1"]},
    )

    lex = Lexicon()
    lex.word_to_pronunciations["all"] = [["AA1", "L"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("all")  # Single-word phrase-final utterance

    selector = SpanSelector(inv)
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")

    word_cands = [c for c in cands if c.matched_word == "all" and c.span_type == "word"]
    assert len(word_cands) >= 1
    cand = word_cands[0]

    # Non-destructive sample boundaries
    assert cand.start_sample == 5000
    assert cand.end_sample == 25000

    # Populated variant samples
    expected_exit = int(round(0.015 * sr))   # 15 ms right bleed
    expected_entry = int(round(0.045 * sr))  # 45 ms left bleed (l_bleed_cost >= 0.80)
    assert cand.exit_variant_samples == expected_exit
    assert cand.entry_variant_samples == expected_entry


def test_refine_splice_boundary_cross_correlation():
    """
    P2.1: Verifies that refine_splice_boundary finds the optimal lag offset
    on synthetic out-of-phase periodic signals, phase-aligning pitch periods
    and maximizing normalized cross-correlation without pitch or time modification.
    """
    sr = 48000
    dur_sec = 0.050
    t = np.linspace(0.0, dur_sec, int(round(dur_sec * sr)), endpoint=False, dtype=np.float32)

    # Fundamental frequency: 200 Hz (period = 5 ms = 240 samples at 48 kHz)
    f0 = 200.0
    out_pcm = np.sin(2.0 * np.pi * f0 * t).astype(np.float32)

    # in_pcm is 180 degrees out of phase (exact comb-filtering cancellation)
    in_pcm_inverted = (-out_pcm).copy()

    # Baseline unaligned normalized cross-correlation
    denom = np.sqrt(np.sum(out_pcm ** 2) * np.sum(in_pcm_inverted ** 2) + 1e-12)
    r_unaligned = np.sum(out_pcm * in_pcm_inverted) / denom
    assert r_unaligned < -0.95, "Uninverted wave must have negative correlation"

    shift_out, shift_in = refine_splice_boundary(out_pcm, in_pcm_inverted, sr=sr, max_shift_ms=15.0)

    # After optimal lag refinement:
    # Cross-correlation between aligned slices must be maximized (> +0.95)
    l_eval = min(len(out_pcm), len(in_pcm_inverted)) - max(shift_out, shift_in)
    aligned_out = out_pcm[shift_out : shift_out + l_eval]
    aligned_in = in_pcm_inverted[shift_in : shift_in + l_eval]
    r_aligned = np.sum(aligned_out * aligned_in) / np.sqrt(np.sum(aligned_out ** 2) * np.sum(aligned_in ** 2) + 1e-12)
    assert r_aligned > 0.95, f"Optimal coupling must phase-align out-of-phase waves, got {r_aligned}"

    # Also test positive lag delay: wave delayed by 48 samples (1 ms)
    delay_samples = 48
    in_pcm_delayed = np.roll(out_pcm, delay_samples)
    shift_out_del, shift_in_del = refine_splice_boundary(out_pcm, in_pcm_delayed, sr=sr, max_shift_ms=15.0)
    rel_shift = (shift_in_del - shift_out_del) % 240
    assert rel_shift == delay_samples % 240, f"Expected relative lag {delay_samples}, got {rel_shift}"


def test_refine_splice_boundary_zero_crossing():
    """
    P2.1: Verifies that refine_splice_boundary with is_unvoiced=True snaps cut points
    to upward-going zero-crossings within <= 2 ms inside the lowest-energy valley.
    """
    sr = 48000
    max_snap_samples = int(round(0.002 * sr))  # 96 samples at 48k

    # Create unvoiced-like signal with DC step and zero crossings
    rng = np.random.RandomState(42)
    n = 2000
    noise = (rng.randn(n) * 0.05).astype(np.float32)

    # Insert clear upward zero crossings
    # For outgoing: ensure an upward zero crossing near the end (index 1950, which is 50 samples from end)
    out_sig = noise.copy()
    out_sig[1949] = -0.05
    out_sig[1950] = +0.05

    # For incoming: ensure an upward zero crossing near the start (index 40)
    in_sig = noise.copy()
    in_sig[39] = -0.05
    in_sig[40] = +0.05

    shift_out, shift_in = refine_splice_boundary(out_sig, in_sig, sr=sr, is_unvoiced=True)

    # Shift must be <= 2 ms
    assert abs(shift_out) <= max_snap_samples, f"shift_out {shift_out} exceeds 2 ms ({max_snap_samples})"
    assert abs(shift_in) <= max_snap_samples, f"shift_in {shift_in} exceeds 2 ms ({max_snap_samples})"

    # Outgoing cut point: at n + shift_out
    cut_out = len(out_sig) + shift_out
    assert out_sig[cut_out - 1] <= 0.0, f"Expected non-positive sample before cut, got {out_sig[cut_out - 1]}"
    assert out_sig[cut_out] >= 0.0, f"Expected non-negative sample at cut, got {out_sig[cut_out]}"

    # Incoming cut point: at shift_in
    assert in_sig[shift_in - 1] <= 0.0, f"Expected non-positive sample before in-cut, got {in_sig[shift_in - 1]}"
    assert in_sig[shift_in] >= 0.0, f"Expected non-negative sample at in-cut, got {in_sig[shift_in]}"


def test_refine_splice_boundary_pathological_inputs():
    """
    Validates that refine_splice_boundary handles pathological inputs safely:
    - 1-element 1D arrays (np.squeeze collapses to 0-D scalar without fix)
    - 0-element empty arrays
    - 2D single-column arrays (N, 1) and (1, N)
    - All-zero arrays (silent energy)
    - Extremely short arrays (< 16 samples)
    All must return (0, 0) without exceptions.
    """
    sr = 48000

    # 1-element arrays (formerly TypeError: len() of unsized object)
    assert refine_splice_boundary(np.array([0.5], dtype=np.float32), np.zeros(100, dtype=np.float32), sr=sr) == (0, 0)
    assert refine_splice_boundary(np.zeros(100, dtype=np.float32), np.array([0.5], dtype=np.float32), sr=sr) == (0, 0)
    assert refine_splice_boundary(np.array([0.5], dtype=np.float32), np.array([0.5], dtype=np.float32), sr=sr) == (0, 0)

    # Empty arrays
    assert refine_splice_boundary(np.array([], dtype=np.float32), np.ones(100, dtype=np.float32), sr=sr) == (0, 0)
    assert refine_splice_boundary(np.ones(100, dtype=np.float32), np.array([], dtype=np.float32), sr=sr) == (0, 0)

    # 2D shapes
    assert refine_splice_boundary(np.ones((1, 1), dtype=np.float32), np.ones((100, 1), dtype=np.float32), sr=sr) == (0, 0)
    assert refine_splice_boundary(np.ones((100, 1), dtype=np.float32), np.ones((1, 1), dtype=np.float32), sr=sr) == (0, 0)
    s1 = np.ones((200, 1), dtype=np.float32)
    s2 = np.ones((200, 1), dtype=np.float32)
    out_shift, in_shift = refine_splice_boundary(s1, s2, sr=sr)
    assert isinstance(out_shift, int) and isinstance(in_shift, int)

    # All-zero arrays (unvoiced and voiced)
    assert refine_splice_boundary(np.zeros(200, dtype=np.float32), np.zeros(200, dtype=np.float32), sr=sr, is_unvoiced=False) == (0, 0)
    assert refine_splice_boundary(np.zeros(200, dtype=np.float32), np.zeros(200, dtype=np.float32), sr=sr, is_unvoiced=True) == (0, 0)

    # Very short arrays (len < 16)
    assert refine_splice_boundary(np.ones(5, dtype=np.float32), np.ones(5, dtype=np.float32), sr=sr) == (0, 0)


@pytest.mark.parametrize("phase_offset", [np.pi / 2, np.pi, -np.pi / 3, 2.5])
def test_render_master_pcm_crossfade_continuity(phase_offset: float):
    """
    P2.1: Verifies that render_master_pcm phase-aligns out-of-phase voiced audio
    and maintains sample-accurate waveform continuity (|delta_x| < 0.05) across
    BOTH fade_start and fade_end for arbitrary non-zero phase offsets.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sr = 48000
        fade_samples = int(round(0.010 * sr))  # 480 samples
        f0 = 100.0  # Period = 480 samples
        n_seg = 12000

        t1 = np.arange(n_seg, dtype=np.float32) / sr
        audio1 = (0.5 * np.sin(2.0 * np.pi * f0 * t1)).astype(np.float32)

        t2 = np.arange(n_seg, dtype=np.float32) / sr
        audio2 = (0.5 * np.sin(2.0 * np.pi * f0 * t2 + phase_offset)).astype(np.float32)

        src1_wav = tmp_path / "src1.wav"
        src2_wav = tmp_path / "src2.wav"
        sf.write(str(src1_wav), audio1, sr)
        sf.write(str(src2_wav), audio2, sr)

        src_rec1 = SourceRecord("src1", str(src1_wav), native_sample_rate=sr, native_channels=1, cached_pcm_path=str(src1_wav))
        src_rec2 = SourceRecord("src2", str(src2_wav), native_sample_rate=sr, native_channels=1, cached_pcm_path=str(src2_wav))

        seg1 = PlanSegment(
            segment_index=0, source_content_id="src1", speaker_id="spk",
            source_start_sample=0, source_end_sample=n_seg,
            output_start_sample=0, output_end_sample=n_seg,
            target_text="seg1", target_phones=["AA"], realized_phones=["AA"], span_type="word",
            source_video_path=str(src1_wav), gain_db=0.0,
        )
        seg2 = PlanSegment(
            segment_index=1, source_content_id="src2", speaker_id="spk",
            source_start_sample=0, source_end_sample=n_seg,
            output_start_sample=n_seg - fade_samples, output_end_sample=2 * n_seg - fade_samples,
            target_text="seg2", target_phones=["AA"], realized_phones=["AA"], span_type="word",
            source_video_path=str(src2_wav), gain_db=0.0,
        )
        trans = PlanTransition(
            transition_index=0, from_segment_index=0, to_segment_index=1,
            transition_type="overlap", overlap_samples=fade_samples, fade_samples=fade_samples,
        )

        plan = EditPlan(
            plan_id=f"continuity_test_{phase_offset}", target_text="seg1 seg2", speaker_id="spk",
            project_sample_rate=sr, total_output_samples=2 * n_seg - fade_samples,
            segments=[seg1, seg2], transitions=[trans],
            sources={"src1": src_rec1, "src2": src_rec2},
            rendering_mode="cuts-only",
        )
        plan.validate()

        out_wav = tmp_path / "out_master.wav"
        rendered_path = render_master_pcm(plan, out_wav)
        rendered, _ = sf.read(str(rendered_path), dtype="float32")

        fade_start = seg2.output_start_sample
        fade_end = seg1.output_end_sample

        # Check correlation R > 0.90 between p_tail and n_head
        margin = min(int(round(0.015 * sr)), max(0, n_seg - fade_samples))
        _s_out, shift_in = refine_splice_boundary(
            audio1[-fade_samples:], audio2[:fade_samples + margin], sr=sr, max_shift_ms=15.0, is_unvoiced=False
        )
        p_tail = audio1[-fade_samples:]
        n_head = audio2[shift_in : shift_in + fade_samples]
        denom = float(np.sqrt(np.sum(p_tail ** 2) * np.sum(n_head ** 2)))
        r = float(np.sum(p_tail * n_head) / denom) if denom > 0 else -1.0
        assert r > 0.90, f"Correlation between p_tail and n_head is {r:.4f} <= 0.90"

        # Check boundary step discontinuity at fade_start
        step_start = abs(float(rendered[fade_start] - rendered[fade_start - 1]))
        assert step_start < 0.05, f"Step discontinuity at fade_start is {step_start:.6f}, exceeding 0.05"

        # Check boundary step discontinuity at fade_end
        step_end = abs(float(rendered[fade_end] - rendered[fade_end - 1]))
        assert step_end < 0.05, f"Step discontinuity at fade_end is {step_end:.6f}, exceeding 0.05"

        # Check all consecutive sample differences in the transition region
        diffs = np.abs(np.diff(rendered[fade_start - 10 : fade_end + 10]))
        max_diff = float(np.max(diffs))
        assert max_diff < 0.05, f"Max consecutive sample jump in crossfade region is {max_diff:.6f}"


def test_render_master_pcm_unvoiced_snapping_active():
    """
    Validates that unvoiced transitions (e.g. stop consonants 'T', 'K')
    actively apply zero-crossing boundary snapping during crossfade rendering
    and maintain sample-accurate waveform continuity (|delta_x| < 0.05).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sr = 48000
        fade_samples = int(round(0.010 * sr))  # 480 samples
        n_seg = 12000

        # Create two slices with clear unvoiced upward zero crossings
        audio1 = np.full(n_seg, 0.2, dtype=np.float32)
        audio2 = np.full(n_seg, 0.2, dtype=np.float32)

        # In outgoing slice: place an upward zero crossing inside the search margin
        audio1[n_seg - 51] = -0.2
        audio1[n_seg - 50] = +0.2

        # In incoming slice: place an upward zero crossing inside head search margin at index 40
        audio2[39] = -0.2
        audio2[40] = +0.2

        src1_wav = tmp_path / "src1_unv.wav"
        src2_wav = tmp_path / "src2_unv.wav"
        sf.write(str(src1_wav), audio1, sr)
        sf.write(str(src2_wav), audio2, sr)

        src_rec1 = SourceRecord("src1", str(src1_wav), native_sample_rate=sr, native_channels=1, cached_pcm_path=str(src1_wav))
        src_rec2 = SourceRecord("src2", str(src2_wav), native_sample_rate=sr, native_channels=1, cached_pcm_path=str(src2_wav))

        # Realized stop consonants 'T' and 'K' trigger is_unv = True
        seg1 = PlanSegment(
            segment_index=0, source_content_id="src1", speaker_id="spk",
            source_start_sample=0, source_end_sample=n_seg,
            output_start_sample=0, output_end_sample=n_seg,
            target_text="seg1", target_phones=["T"], realized_phones=["T"], span_type="word",
            source_video_path=str(src1_wav), gain_db=0.0,
        )
        seg2 = PlanSegment(
            segment_index=1, source_content_id="src2", speaker_id="spk",
            source_start_sample=0, source_end_sample=n_seg,
            output_start_sample=n_seg - fade_samples, output_end_sample=2 * n_seg - fade_samples,
            target_text="seg2", target_phones=["K"], realized_phones=["K"], span_type="word",
            source_video_path=str(src2_wav), gain_db=0.0,
        )
        trans = PlanTransition(
            transition_index=0, from_segment_index=0, to_segment_index=1,
            transition_type="overlap", overlap_samples=fade_samples, fade_samples=fade_samples,
        )

        plan = EditPlan(
            plan_id="unv_test_plan", target_text="seg1 seg2", speaker_id="spk",
            project_sample_rate=sr, total_output_samples=2 * n_seg - fade_samples,
            segments=[seg1, seg2], transitions=[trans],
            sources={"src1": src_rec1, "src2": src_rec2},
            rendering_mode="cuts-only",
        )
        plan.validate()

        out_wav = tmp_path / "out_unv_master.wav"
        rendered_path = render_master_pcm(plan, out_wav)
        rendered, _ = sf.read(str(rendered_path), dtype="float32")

        fade_start = seg2.output_start_sample
        fade_end = seg1.output_end_sample

        # Step jump at fade_start must be < 0.05
        step_start = abs(float(rendered[fade_start] - rendered[fade_start - 1]))
        assert step_start < 0.05, f"Step discontinuity at fade_start is {step_start:.6f}"

        # Step jump at fade_end must be < 0.05
        step_end = abs(float(rendered[fade_end] - rendered[fade_end - 1]))
        assert step_end < 0.05, f"Step discontinuity at fade_end is {step_end:.6f}"

        # Consecutive sample diffs across transition region must be < 0.05
        diffs = np.abs(np.diff(rendered[fade_start - 10 : fade_end + 10]))
        max_diff = float(np.max(diffs))
        assert max_diff < 0.05, f"Max consecutive sample jump in unvoiced crossfade is {max_diff:.6f}"
        assert len(rendered) == 2 * n_seg - fade_samples


