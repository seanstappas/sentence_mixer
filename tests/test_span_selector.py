"""
Tests for variable-span unit selection (Milestone M3).
Verifies the Phrase -> Word -> Whole Phone hierarchy, intact word preservation (rank-31 regression),
repeated 14ms unit edit handling, and exact DP vs brute-force agreement.
"""

from typing import List

from contracts import AlignedEvent, InventoryIndex, SourceRecord, WordOccurrence
from lexicon import Lexicon
from span_selector import SpanSelector
from target_graph import TargetGraphBuilder


def build_synthetic_inventory() -> InventoryIndex:
    sr = 48000
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )

    # Word 'cat' realized as [K, AE1, T]
    # Sample range: 10000 to 24400 (0.3s)
    # K: 10000..13000, AE1: 13000..20000, T: 20000..24400
    ev_k = AlignedEvent(
        event_id="ev_k", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=10000, end_sample=13000, label="K", base_phone="K",
        successor_id="ev_ae"
    )
    ev_ae = AlignedEvent(
        event_id="ev_ae", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=13000, end_sample=20000, label="AE1", base_phone="AE",
        predecessor_id="ev_k", successor_id="ev_t"
    )
    ev_t = AlignedEvent(
        event_id="ev_t", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=20000, end_sample=24400, label="T", base_phone="T",
        predecessor_id="ev_ae"
    )

    w_cat = WordOccurrence(
        word_id="w_cat_1", source_content_id="src_1", speaker_id="spk1",
        word_normalized="cat", word_original="cat",
        start_sample=10000, end_sample=24400,
        phone_event_ids=["ev_k", "ev_ae", "ev_t"],
        realized_phones=["K", "AE1", "T"]
    )

    # Isolated phone units for K, AE1, T located elsewhere in recording
    # (simulating phones that might be near 90ms duration under the old heuristic)
    ev_k_iso = AlignedEvent(
        event_id="ev_k_iso", source_content_id="src_1", utterance_id="u2", speaker_id="spk1",
        kind="phone", start_sample=40000, end_sample=44320, label="K", base_phone="K"
    )
    ev_ae_iso = AlignedEvent(
        event_id="ev_ae_iso", source_content_id="src_1", utterance_id="u3", speaker_id="spk1",
        kind="phone", start_sample=50000, end_sample=54320, label="AE1", base_phone="AE"
    )
    ev_t_iso = AlignedEvent(
        event_id="ev_t_iso", source_content_id="src_1", utterance_id="u4", speaker_id="spk1",
        kind="phone", start_sample=60000, end_sample=64320, label="T", base_phone="T"
    )

    # A short 14ms unit (672 samples at 48kHz)
    ev_short_14ms = AlignedEvent(
        event_id="ev_short", source_content_id="src_1", utterance_id="u5", speaker_id="spk1",
        kind="phone", start_sample=70000, end_sample=70672, label="B", base_phone="B"
    )

    inv = InventoryIndex(
        sources={"src_1": src},
        events={
            "ev_k": ev_k, "ev_ae": ev_ae, "ev_t": ev_t,
            "ev_k_iso": ev_k_iso, "ev_ae_iso": ev_ae_iso, "ev_t_iso": ev_t_iso,
            "ev_short": ev_short_14ms,
        },
        words={"w_cat_1": w_cat},
        word_to_word_ids={"cat": ["w_cat_1"]},
        phone_to_event_ids={
            "K": ["ev_k", "ev_k_iso"],
            "AE1": ["ev_ae", "ev_ae_iso"],
            "T": ["ev_t", "ev_t_iso"],
            "B": ["ev_short"],
        },
        base_phone_to_event_ids={
            "K": ["ev_k", "ev_k_iso"],
            "AE": ["ev_ae", "ev_ae_iso"],
            "T": ["ev_t", "ev_t_iso"],
            "B": ["ev_short"],
        },
    )
    return inv


def test_protected_rank_31_intact_word():
    """
    Mandatory regression: an intact word whose individual phones might not have the
    lowest phone-level duration cost must still win as a single intact candidate
    rather than 3 separate cuts.
    """
    inv = build_synthetic_inventory()
    lex = Lexicon()
    lex.word_to_pronunciations["cat"] = [["K", "AE1", "T"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("cat")

    selector = SpanSelector(inv, edit_penalty=0.5)
    selected_path = selector.select(graph)

    # Must select 1 single intact word candidate, not 3 separate phone slices
    assert len(selected_path) == 1
    assert selected_path[0].span_type == "word"
    assert selected_path[0].matched_word == "cat"
    assert selected_path[0].start_sample == 10000
    assert selected_path[0].end_sample == 24400


def test_repeated_14ms_unit_not_merged():
    """
    Mandatory regression: A 14 ms source unit reused for two target occurrences
    must be treated as two contributions with an edit, not a self-continuation.
    """
    inv = build_synthetic_inventory()
    lex = Lexicon()
    lex.word_to_pronunciations["b"] = [["B"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("b b")

    selector = SpanSelector(inv, edit_penalty=0.5)
    cands = selector.retrieve_candidates(graph)

    # Check candidates for node 0 and node 1
    cand_0 = next(c for c in cands if c.target_node_start == 0 and c.event_ids == ["ev_short"])
    cand_1 = next(c for c in cands if c.target_node_start == 1 and c.event_ids == ["ev_short"])

    # Invariant: reusing the identical source range cannot be claimed as a zero-cost continuation
    is_cont = selector.is_source_continuation(cand_0, cand_1)
    assert not is_cont, "Reusing the same source slice must not be recognized as a natural continuation"


def test_exact_dp_versus_brute_force():
    """
    Validates dynamic programming against exhaustive brute-force search on a small graph.
    """
    inv = build_synthetic_inventory()
    lex = Lexicon()
    lex.word_to_pronunciations["cat"] = [["K", "AE1", "T"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("cat")

    selector = SpanSelector(inv, edit_penalty=0.5)
    dp_path = selector.select(graph)

    # Compute DP total cost
    dp_cost = dp_path[0].target_cost + dp_path[0].quality_cost
    for i in range(len(dp_path) - 1):
        is_cont = selector.is_source_continuation(dp_path[i], dp_path[i+1])
        trans = 0.0 if is_cont else selector.edit_penalty
        dp_cost += trans + dp_path[i+1].target_cost + dp_path[i+1].quality_cost

    # Brute-force enumerate all complete paths covering nodes 0, 1, 2
    candidates = selector.retrieve_candidates(graph)

    def enumerate_paths(curr_node: int, current_path: List) -> List[List]:
        if curr_node > 2:
            return [current_path]
        res = []
        for c in candidates:
            if c.target_node_start == curr_node:
                res.extend(enumerate_paths(c.target_node_end + 1, current_path + [c]))
        return res

    all_paths = enumerate_paths(0, [])
    assert len(all_paths) > 0

    best_bf_cost = float("inf")
    for path in all_paths:
        cost = path[0].target_cost + path[0].quality_cost
        for i in range(len(path) - 1):
            is_cont = selector.is_source_continuation(path[i], path[i+1])
            trans = 0.0 if is_cont else selector.edit_penalty
            cost += trans + path[i+1].target_cost + path[i+1].quality_cost
        if cost < best_bf_cost:
            best_bf_cost = cost

    assert abs(dp_cost - best_bf_cost) < 1e-6, f"DP cost {dp_cost} does not match brute force {best_bf_cost}"
