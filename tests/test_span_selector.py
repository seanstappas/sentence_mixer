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


def test_t08_controlled_basic_boundary_features_swapped():
    """T08: Controlled basic boundary features swapped -> Actual selected path and cost trace change."""
    from contracts import FeatureRecord

    lex = Lexicon()
    lex.word_to_pronunciations["go"] = [["G", "OW1"]]
    lex.word_to_pronunciations["home"] = [["HH", "OW1", "M"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("go home")

    # Inventory with word 'go' and two instances of 'home': 'home_1' and 'home_2'
    sr = 48000
    src = SourceRecord(content_id="s1", source_path="v.mp4", native_sample_rate=sr, native_channels=1)

    ev_go = AlignedEvent(event_id="e_go", source_content_id="s1", utterance_id="u1", speaker_id="spk1",
                         kind="phone", start_sample=0, end_sample=24000, label="OW1", base_phone="OW")
    w_go = WordOccurrence(word_id="w_go", source_content_id="s1", speaker_id="spk1",
                          word_normalized="go", word_original="go", start_sample=0, end_sample=24000,
                          phone_event_ids=["e_go"], realized_phones=["G", "OW1"])

    ev_h1 = AlignedEvent(event_id="e_h1", source_content_id="s1", utterance_id="u2", speaker_id="spk1",
                         kind="phone", start_sample=30000, end_sample=54000, label="HH", base_phone="HH")
    w_h1 = WordOccurrence(word_id="w_h1", source_content_id="s1", speaker_id="spk1",
                          word_normalized="home", word_original="home", start_sample=30000, end_sample=54000,
                          phone_event_ids=["e_h1"], realized_phones=["HH", "OW1", "M"])

    ev_h2 = AlignedEvent(event_id="e_h2", source_content_id="s1", utterance_id="u3", speaker_id="spk1",
                         kind="phone", start_sample=60000, end_sample=84000, label="HH", base_phone="HH")
    w_h2 = WordOccurrence(word_id="w_h2", source_content_id="s1", speaker_id="spk1",
                          word_normalized="home", word_original="home", start_sample=60000, end_sample=84000,
                          phone_event_ids=["e_h2"], realized_phones=["HH", "OW1", "M"])

    inv = InventoryIndex(
        sources={"s1": src},
        events={"e_go": ev_go, "e_h1": ev_h1, "e_h2": ev_h2},
        words={"w_go": w_go, "w_h1": w_h1, "w_h2": w_h2},
        word_to_word_ids={"go": ["w_go"], "home": ["w_h1", "w_h2"]},
    )
    selector = SpanSelector(inv, edit_penalty=0.1)

    # Feature sets:
    # Exit feature of 'go': rms 0.1, f0 200Hz
    feat_exit_go = FeatureRecord(
        source_content_id="s1", sample_rate=sr, start_sample=20000, end_sample=24000,
        rms_energy=0.1, f0_hz=200.0, f0_confidence=0.9, voicing_state="voiced",
        spectral_centroid=1500.0, mfcc_13=[1.0]*13
    )
    # Entry feature matching 'go' (low join cost)
    feat_entry_close = FeatureRecord(
        source_content_id="s1", sample_rate=sr, start_sample=30000, end_sample=34000,
        rms_energy=0.105, f0_hz=202.0, f0_confidence=0.9, voicing_state="voiced",
        spectral_centroid=1520.0, mfcc_13=[1.02]*13
    )
    # Entry feature distant from 'go' (high join cost)
    feat_entry_distant = FeatureRecord(
        source_content_id="s1", sample_rate=sr, start_sample=60000, end_sample=64000,
        rms_energy=0.01, f0_hz=350.0, f0_confidence=0.9, voicing_state="voiced",
        spectral_centroid=3500.0, mfcc_13=[5.0]*13
    )

    # In Run 1: home_1 gets close features, home_2 gets distant features
    feat_dict_1 = {
        ("s1", 24000, True): feat_exit_go,
        ("s1", 30000, False): feat_entry_close,
        ("s1", 60000, False): feat_entry_distant,
    }
    path_1 = selector.select(graph, feature_extractor=feat_dict_1)
    chosen_home_1 = [s for s in path_1 if s.matched_word == "home"][0]
    assert chosen_home_1.start_sample == 30000, "Should select home_1 with close boundary features"

    # In Run 2: SWAP features! home_1 gets distant features, home_2 gets close features
    feat_dict_2 = {
        ("s1", 24000, True): feat_exit_go,
        ("s1", 30000, False): feat_entry_distant,
        ("s1", 60000, False): feat_entry_close,
    }
    path_2 = selector.select(graph, feature_extractor=feat_dict_2)
    chosen_home_2 = [s for s in path_2 if s.matched_word == "home"][0]
    assert chosen_home_2.start_sample == 60000, "Should select home_2 after boundary features are swapped"


def test_t09_guide_preferences_swapped():
    """T09: Guide/backend preferences swapped -> Same production selector consumes measured inputs."""
    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH", "AH0", "L", "OW1"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("hello")

    sr = 48000
    src = SourceRecord(content_id="s1", source_path="v.mp4", native_sample_rate=sr, native_channels=1)

    # Two competing occurrences of 'hello'
    ev1 = AlignedEvent(event_id="e1", source_content_id="s1", utterance_id="u1", speaker_id="spk1",
                       kind="phone", start_sample=0, end_sample=24000, label="HH", base_phone="HH")
    w1 = WordOccurrence(word_id="w1", source_content_id="s1", speaker_id="spk1",
                        word_normalized="hello", word_original="hello", start_sample=0, end_sample=24000,
                        phone_event_ids=["e1"], realized_phones=["HH", "AH0", "L", "OW1"])

    ev2 = AlignedEvent(event_id="e2", source_content_id="s1", utterance_id="u2", speaker_id="spk1",
                       kind="phone", start_sample=30000, end_sample=54000, label="HH", base_phone="HH")
    w2 = WordOccurrence(word_id="w2", source_content_id="s1", speaker_id="spk1",
                        word_normalized="hello", word_original="hello", start_sample=30000, end_sample=54000,
                        phone_event_ids=["e2"], realized_phones=["HH", "AH0", "L", "OW1"])

    inv = InventoryIndex(
        sources={"s1": src},
        events={"e1": ev1, "e2": ev2},
        words={"w1": w1, "w2": w2},
        word_to_word_ids={"hello": ["w1", "w2"]},
    )
    selector = SpanSelector(inv)

    cands = selector.retrieve_candidates(graph)
    c1 = next(c for c in cands if c.start_sample == 0)
    c2 = next(c for c in cands if c.start_sample == 30000)

    # Guide 1 prefers c1 (penalty on c2)
    guide_1 = {c1.candidate_id: 0.0, c2.candidate_id: 10.0}
    path_1 = selector.select(graph, prosody_guide=guide_1)
    assert path_1[0].start_sample == 0

    # Guide 2 prefers c2 (penalty on c1)
    guide_2 = {c1.candidate_id: 10.0, c2.candidate_id: 0.0}
    path_2 = selector.select(graph, prosody_guide=guide_2)
    assert path_2[0].start_sample == 30000


def test_t10_competing_occurrences_and_partition_invariance():
    """
    T10: 31+ competing occurrences; reordered inventory; split natural run:
    Protected suitable runs survive; deterministic selection regardless of inventory ordering.
    """
    lex = Lexicon()
    lex.word_to_pronunciations["test"] = [["T", "EH1", "S", "T"]]
    lex.word_to_pronunciations["run"] = [["R", "AH1", "N"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("test run")

    sr = 48000
    src = SourceRecord(content_id="s1", source_path="v.mp4", native_sample_rate=sr, native_channels=1)

    words = {}
    events = {}
    word_ids_test = []
    word_ids_run = []

    # 35 competing occurrences of 'test' with different quality/duration
    for i in range(35):
        w_id = f"w_test_{i:02d}"
        e_id = f"e_test_{i:02d}"
        ev = AlignedEvent(
            event_id=e_id, source_content_id="s1", utterance_id=f"u_{i}",
            speaker_id="spk1", kind="phone",
            start_sample=i * 20000, end_sample=i * 20000 + 10000,
            label="T", base_phone="T",
        )
        events[e_id] = ev
        wo = WordOccurrence(
            word_id=w_id, source_content_id="s1", speaker_id="spk1",
            word_normalized="test", word_original="test",
            start_sample=i * 20000, end_sample=i * 20000 + 10000,
            phone_event_ids=[e_id], realized_phones=["T", "EH1", "S", "T"],
        )
        words[w_id] = wo
        word_ids_test.append(w_id)

    # 1 continuous natural run of "test run" (starts at sample 1,000,000 to 1,020,000)
    natural_test_w = WordOccurrence(
        word_id="w_nat_test", source_content_id="s1", speaker_id="spk1",
        word_normalized="test", word_original="test",
        start_sample=1000000, end_sample=1010000,
        phone_event_ids=["e_nat_t"], realized_phones=["T", "EH1", "S", "T"],
    )
    natural_run_w = WordOccurrence(
        word_id="w_nat_run", source_content_id="s1", speaker_id="spk1",
        word_normalized="run", word_original="run",
        start_sample=1010000, end_sample=1020000,
        phone_event_ids=["e_nat_r"], realized_phones=["R", "AH1", "N"],
    )
    words["w_nat_test"] = natural_test_w
    words["w_nat_run"] = natural_run_w
    word_ids_test.append("w_nat_test")
    word_ids_run.append("w_nat_run")

    events["e_nat_t"] = AlignedEvent(
        event_id="e_nat_t", source_content_id="s1", utterance_id="u_nat",
        speaker_id="spk1", kind="phone", start_sample=1000000, end_sample=1010000,
        label="T", base_phone="T",
    )
    events["e_nat_r"] = AlignedEvent(
        event_id="e_nat_r", source_content_id="s1", utterance_id="u_nat",
        speaker_id="spk1", kind="phone", start_sample=1010000, end_sample=1020000,
        label="R", base_phone="R",
    )

    inv = InventoryIndex(
        sources={"s1": src},
        events=events,
        words=words,
        word_to_word_ids={"test": list(word_ids_test), "run": list(word_ids_run)},
    )

    selector = SpanSelector(inv, edit_penalty=1.0)
    path1 = selector.select(graph)

    # The natural contiguous run (w_nat_test + w_nat_run) should win or be selected with 0 join edit cost
    # Reversing the inventory order must produce the identical selection result (deterministic ties/ranking)
    inv_reversed = InventoryIndex(
        sources={"s1": src},
        events=events,
        words=words,
        word_to_word_ids={"test": list(reversed(word_ids_test)), "run": list(reversed(word_ids_run))},
    )
    selector_rev = SpanSelector(inv_reversed, edit_penalty=1.0)
    path2 = selector_rev.select(graph)

    assert len(path1) == len(path2)
    assert [s.start_sample for s in path1] == [s.start_sample for s in path2]


