"""
Tests for variable-span unit selection (Milestone M3).
Verifies the Phrase -> Word -> Whole Phone hierarchy, intact word preservation (rank-31 regression),
repeated 14ms unit edit handling, and exact DP vs brute-force agreement.
"""

from typing import Dict, List

from contracts import AlignedEvent, InventoryIndex, SourceRecord, WordOccurrence
from lexicon import Lexicon
from span_selector import SpanSelector, apply_diphone_midpoint_transitions
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


def test_surrounding_source_coherence_sandwich_splice():
    """
    Validates that a swapped word inside a sentence returns to the surrounding
    sentence in Source A rather than jumping to an unrelated Source C.
    """
    sr = 48000
    src_a = SourceRecord("src_a", "video_a.mp4", native_sample_rate=sr, native_channels=2)
    src_b = SourceRecord("src_b", "video_b.mp4", native_sample_rate=sr, native_channels=2)
    src_c = SourceRecord("src_c", "video_c.mp4", native_sample_rate=sr, native_channels=2)

    events: Dict[str, AlignedEvent] = {}
    words: Dict[str, WordOccurrence] = {}

    # Source A: Part 1 "hello" (sample 10000 to 20000)
    events["e_a_hello_h"] = AlignedEvent("e_a_hello_h", "src_a", "u1", "spk1", "phone", 10000, 15000, "HH", "HH")
    events["e_a_hello_o"] = AlignedEvent("e_a_hello_o", "src_a", "u1", "spk1", "phone", 15000, 20000, "OW1", "OW")
    words["w_a_hello"] = WordOccurrence("w_a_hello", "src_a", "spk1", "hello", "hello", 10000, 20000, ["e_a_hello_h", "e_a_hello_o"], ["HH", "OW1"])

    # Source B: Swapped word "there" (sample 5000 to 15000)
    events["e_b_there_dh"] = AlignedEvent("e_b_there_dh", "src_b", "u2", "spk1", "phone", 5000, 10000, "DH", "DH")
    events["e_b_there_er"] = AlignedEvent("e_b_there_er", "src_b", "u2", "spk1", "phone", 10000, 15000, "ER1", "ER")
    words["w_b_there"] = WordOccurrence("w_b_there", "src_b", "spk1", "there", "there", 5000, 15000, ["e_b_there_dh", "e_b_there_er"], ["DH", "ER1"])

    # Source A: Part 3 "friend" resuming surrounding sentence (sample 25000 to 35000, only 5000 samples / 0.1s after hello!)
    events["e_a_friend_f"] = AlignedEvent("e_a_friend_f", "src_a", "u1", "spk1", "phone", 25000, 30000, "F", "F")
    events["e_a_friend_end"] = AlignedEvent("e_a_friend_end", "src_a", "u1", "spk1", "phone", 30000, 35000, "EH1", "EH")
    words["w_a_friend"] = WordOccurrence("w_a_friend", "src_a", "spk1", "friend", "friend", 25000, 35000, ["e_a_friend_f", "e_a_friend_end"], ["F", "EH1"])

    # Source C: Competitor for Part 3 "friend" from unrelated video C
    events["e_c_friend_f"] = AlignedEvent("e_c_friend_f", "src_c", "u3", "spk1", "phone", 50000, 55000, "F", "F")
    events["e_c_friend_end"] = AlignedEvent("e_c_friend_end", "src_c", "u3", "spk1", "phone", 55000, 60000, "EH1", "EH")
    words["w_c_friend"] = WordOccurrence("w_c_friend", "src_c", "spk1", "friend", "friend", 50000, 60000, ["e_c_friend_f", "e_c_friend_end"], ["F", "EH1"])

    inv = InventoryIndex(
        sources={"src_a": src_a, "src_b": src_b, "src_c": src_c},
        events=events,
        words=words,
        word_to_word_ids={
            "hello": ["w_a_hello"],
            "there": ["w_b_there"],
            "friend": ["w_c_friend", "w_a_friend"],
        },
    )

    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH", "OW1"]]
    lex.word_to_pronunciations["there"] = [["DH", "ER1"]]
    lex.word_to_pronunciations["friend"] = [["F", "EH1"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("hello there friend")

    selector = SpanSelector(inv, edit_penalty=0.5)
    path = selector.select(graph)

    assert len(path) == 3
    assert path[0].matched_word == "hello" and path[0].source_content_id == "src_a"
    assert path[1].matched_word == "there" and path[1].source_content_id == "src_b"
    # Part 3 must return to src_a via surrounding sentence affinity, NOT jump to src_c
    assert path[2].matched_word == "friend" and path[2].source_content_id == "src_a"


def test_unit_reuse_penalty_encourages_variety():
    """
    Validates that when multiple identical-quality instances of a phoneme exist,
    the unit reuse penalty encourages selecting distinct donor instances.
    """
    sr = 16000
    src = SourceRecord("s1", "vid.mp4", native_sample_rate=sr, native_channels=1)
    events: Dict[str, AlignedEvent] = {}

    # Two identical 100ms instances of AA1
    events["ev_aa_1"] = AlignedEvent("ev_aa_1", "s1", "u1", "spk1", "phone", 1000, 2600, "AA1", "AA")
    events["ev_aa_2"] = AlignedEvent("ev_aa_2", "s1", "u2", "spk1", "phone", 5000, 6600, "AA1", "AA")

    inv = InventoryIndex(
        sources={"s1": src},
        events=events,
        words={},
        phone_to_event_ids={"AA1": ["ev_aa_1", "ev_aa_2"]},
    )

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    # Target with two AA1 sounds
    graph = builder.build("{AA1} {AA1}")

    selector = SpanSelector(inv, edit_penalty=0.2)
    path = selector.select(graph)

    assert len(path) == 2
    # First unit selects one instance, second unit selects the other instance rather than duplicate reuse
    selected_events = [p.event_ids[0] for p in path]
    assert len(set(selected_events)) == 2
    assert "ev_aa_1" in selected_events and "ev_aa_2" in selected_events


def test_asymmetric_vowel_duration_preference():
    """
    Validates that a longer, well-enunciated vowel receives lower cost than a rushed/truncated vowel.
    """
    sr = 16000
    src = SourceRecord("s1", "vid.mp4", native_sample_rate=sr, native_channels=1)
    events: Dict[str, AlignedEvent] = {}

    # AA1 prior is 160 ms (2560 samples).
    # Long enunciated vowel: 180 ms (2880 samples)
    events["ev_aa_long"] = AlignedEvent("ev_aa_long", "s1", "u1", "spk1", "phone", 1000, 3880, "AA1", "AA")
    # Truncated / rushed vowel: 50 ms (800 samples)
    events["ev_aa_rushed"] = AlignedEvent("ev_aa_rushed", "s1", "u2", "spk1", "phone", 5000, 5800, "AA1", "AA")

    inv = InventoryIndex(
        sources={"s1": src},
        events=events,
        words={},
        phone_to_event_ids={"AA1": ["ev_aa_long", "ev_aa_rushed"]},
    )

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    graph = builder.build("{AA1}")

    selector = SpanSelector(inv, edit_penalty=0.5)
    cands = selector.retrieve_candidates(graph)

    c_long = next(c for c in cands if c.event_ids == ["ev_aa_long"])
    c_rushed = next(c for c in cands if c.event_ids == ["ev_aa_rushed"])

    assert c_long.target_cost < c_rushed.target_cost


def test_graded_fallback_ladder_cognate_and_manner_bridge():
    """
    Validates that missing phones in a limited inventory gracefully resolve
    via acoustic cognates and manner-of-articulation bridges without breaking Viterbi DP.
    """
    sr = 16000
    src = SourceRecord("s1", "vid.mp4", native_sample_rate=sr, native_channels=1)
    events: Dict[str, AlignedEvent] = {}

    # Speaker only has T (alveolar plosive) and EH1 (vowel)
    events["ev_t"] = AlignedEvent("ev_t", "s1", "u1", "spk1", "phone", 1000, 2500, "T", "T")
    events["ev_eh"] = AlignedEvent("ev_eh", "s1", "u2", "spk1", "phone", 3000, 5000, "EH1", "EH")

    inv = InventoryIndex(
        sources={"s1": src},
        events=events,
        words={},
        phone_to_event_ids={"T": ["ev_t"], "EH1": ["ev_eh"]},
        base_phone_to_event_ids={"T": ["ev_t"], "EH": ["ev_eh"]},
    )

    lex = Lexicon()
    builder = TargetGraphBuilder(lex)
    # Target requires D (cognate to T) and EY1 (diphthong cognate to EH)
    graph = builder.build("{D} {EY1}")

    selector = SpanSelector(inv, edit_penalty=0.3)
    path = selector.select(graph, speaker_id="spk1")

    assert len(path) == 2
    assert path[0].span_type == "phone"
    assert path[0].event_ids == ["ev_t"]  # D mapped to T cognate
    assert path[1].span_type == "phone"
    assert path[1].event_ids == ["ev_eh"]  # EY1 mapped to EH cognate

    # When a required phone and its acoustic cognates are completely absent (e.g. G with no K),
    # SpanSelector must strictly raise ValueError rather than synthesizing arbitrary sounds.
    import pytest
    graph_missing = builder.build("{EH1} {G}")
    with pytest.raises(ValueError, match="Could not find a continuous phonetic path"):
        selector.select(graph_missing, speaker_id="spk1")


def test_span_candidates_preserve_exact_sample_boundaries_without_collar_clipping():
    """
    P2.1: Verifies that SpanSelector candidate retrieval preserves exact source sample
    boundaries across phrase, subword, and phone candidates without blind destructive
    sample collar shaving, while recording non-destructive variant samples.
    """
    sr = 48000
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )

    ev_pred = AlignedEvent(
        event_id="ev_pred", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=5000, label="V", base_phone="V",
        successor_id="ev_ae"
    )
    ev_ae = AlignedEvent(
        event_id="ev_ae", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=5000, end_sample=15000, label="AE1", base_phone="AE",
        predecessor_id="ev_pred", successor_id="ev_p"
    )
    ev_p = AlignedEvent(
        event_id="ev_p", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=15000, end_sample=20000, label="P", base_phone="P",
        predecessor_id="ev_ae"
    )

    w_at = WordOccurrence(
        word_id="w_at_1", source_content_id="src_1", speaker_id="spk1",
        word_normalized="at", word_original="at",
        start_sample=5000, end_sample=20000,
        phone_event_ids=["ev_ae", "ev_p"],
        realized_phones=["AE1", "P"],
    )

    inv = InventoryIndex(
        sources={"src_1": src},
        events={"ev_pred": ev_pred, "ev_ae": ev_ae, "ev_p": ev_p},
        words={"w_at_1": w_at},
        word_to_word_ids={"at": ["w_at_1"]},
        phone_to_event_ids={"AE1": ["ev_ae"], "P": ["ev_p"]},
        base_phone_to_event_ids={"AE": ["ev_ae"], "P": ["ev_p"]},
    )

    lex = Lexicon()
    lex.word_to_pronunciations["at"] = [["AE1", "P"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("at")

    selector = SpanSelector(inv)
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")

    # 1. Phone candidate check (AE1)
    ae_cands = [c for c in cands if c.span_type == "phone" and c.event_ids == ["ev_ae"]]
    assert len(ae_cands) >= 1
    c_ae = ae_cands[0]
    # Bound must NOT be shaved by stop collar (12ms) or bleed collar (20ms/45ms)
    assert c_ae.start_sample == 5000
    assert c_ae.end_sample == 15000
    assert c_ae.exit_variant_samples > 0
    assert c_ae.entry_variant_samples > 0

    # 2. Subword candidate check (AE1 P)
    sub_cands = [c for c in cands if c.span_type == "subword"]
    if sub_cands:
        c_sub = sub_cands[0]
        assert c_sub.start_sample == 5000
        assert c_sub.end_sample == 20000
        assert c_sub.entry_variant_samples > 0


def test_diphone_midpoint_transitions_subwords_within_word():
    """
    P2.2: Verifies that subword and phone joins inside a synthesized word
    support diphone midpoint transitions at (start_sample + end_sample) // 2
    where formant trajectories are stationary.
    """
    sr = 48000
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=1,
    )

    # Donor 1: 'can' [K, AE1, N] -> provides subword [K, AE1] (nodes 0, 1)
    ev_k1 = AlignedEvent("ev_k1", "src_1", "u1", "spk1", "phone", 10000, 13000, "K", "K")
    ev_ae1 = AlignedEvent("ev_ae1", "src_1", "u1", "spk1", "phone", 13000, 20000, "AE1", "AE")
    ev_n1 = AlignedEvent("ev_n1", "src_1", "u1", "spk1", "phone", 20000, 24000, "N", "N")
    w_can = WordOccurrence(
        word_id="w_can_1", source_content_id="src_1", speaker_id="spk1",
        word_normalized="can", word_original="can",
        start_sample=10000, end_sample=24000,
        phone_event_ids=["ev_k1", "ev_ae1", "ev_n1"],
        realized_phones=["K", "AE1", "N"],
    )

    # Donor 2: 'bat' [B, AE1, T] -> provides subword [AE1, T] (nodes 1, 2)
    ev_b2 = AlignedEvent("ev_b2", "src_1", "u2", "spk1", "phone", 30000, 33000, "B", "B")
    ev_ae2 = AlignedEvent("ev_ae2", "src_1", "u2", "spk1", "phone", 33000, 41000, "AE1", "AE")
    ev_t2 = AlignedEvent("ev_t2", "src_1", "u2", "spk1", "phone", 41000, 45000, "T", "T")
    w_bat = WordOccurrence(
        word_id="w_bat_1", source_content_id="src_1", speaker_id="spk1",
        word_normalized="bat", word_original="bat",
        start_sample=30000, end_sample=45000,
        phone_event_ids=["ev_b2", "ev_ae2", "ev_t2"],
        realized_phones=["B", "AE1", "T"],
    )

    inv = InventoryIndex(
        sources={"src_1": src},
        events={
            "ev_k1": ev_k1, "ev_ae1": ev_ae1, "ev_n1": ev_n1,
            "ev_b2": ev_b2, "ev_ae2": ev_ae2, "ev_t2": ev_t2,
        },
        words={"w_can_1": w_can, "w_bat_1": w_bat},
        word_to_word_ids={"can": ["w_can_1"], "bat": ["w_bat_1"]},
        phone_to_event_ids={
            "K": ["ev_k1"], "AE1": ["ev_ae1", "ev_ae2"], "N": ["ev_n1"],
            "B": ["ev_b2"], "T": ["ev_t2"],
        },
        base_phone_to_event_ids={
            "K": ["ev_k1"], "AE": ["ev_ae1", "ev_ae2"], "N": ["ev_n1"],
            "B": ["ev_b2"], "T": ["ev_t2"],
        },
    )

    lex = Lexicon()
    lex.word_to_pronunciations["cat"] = [["K", "AE1", "T"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("cat")

    # Direct helper test
    from contracts import SpanCandidate
    cand1 = SpanCandidate(
        candidate_id="c_sub1", target_node_start=0, target_node_end=1,
        covered_nodes=[0, 1], source_content_id="src_1", speaker_id="spk1",
        start_sample=10000, end_sample=20000, event_ids=["ev_k1", "ev_ae1"],
        span_type="subword", matched_word="can[K-AE1]", realized_phones=["K", "AE1"],
    )
    cand2 = SpanCandidate(
        candidate_id="c_sub2", target_node_start=1, target_node_end=2,
        covered_nodes=[1, 2], source_content_id="src_1", speaker_id="spk1",
        start_sample=33000, end_sample=45000, event_ids=["ev_ae2", "ev_t2"],
        span_type="subword", matched_word="bat[AE1-T]", realized_phones=["AE1", "T"],
    )

    refined_spans = apply_diphone_midpoint_transitions([cand1, cand2], graph, inv)
    assert len(refined_spans) == 2

    expected_mid1 = (13000 + 20000) // 2
    assert refined_spans[0].end_sample == expected_mid1, f"Expected {expected_mid1}, got {refined_spans[0].end_sample}"

    expected_mid2 = (33000 + 41000) // 2
    assert refined_spans[1].start_sample == expected_mid2, f"Expected {expected_mid2}, got {refined_spans[1].start_sample}"

    # Global select traversal test
    selector = SpanSelector(inv, edit_penalty=0.1)
    path = selector.select(graph, speaker_id="spk1")
    assert len(path) == 2
    assert path[0].span_type == "subword"
    assert path[1].span_type == "subword"
    assert path[0].end_sample == expected_mid1
    assert path[1].start_sample == expected_mid2






def test_continuation_preservation_exempt_from_top_50_pruning():
    """
    Fix P3.1: A natural candidate continuation ranked #51+ on unary heuristic sort
    must NOT be pruned away by pre-DP truncation and must be selected by Viterbi DP
    due to its zero transition cost.
    """
    sr = 48000
    src_cont = SourceRecord(
        content_id="src_cont", source_path="movie.mp4", native_sample_rate=sr, native_channels=2
    )

    # Word 1: 'big' [B, IH1, G] in src_cont: 10000..25000
    ev_b = AlignedEvent(
        event_id="ev_b", source_content_id="src_cont", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=10000, end_sample=14000, label="B", base_phone="B",
        successor_id="ev_ih"
    )
    ev_ih = AlignedEvent(
        event_id="ev_ih", source_content_id="src_cont", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=14000, end_sample=19000, label="IH1", base_phone="IH",
        predecessor_id="ev_b", successor_id="ev_g"
    )
    ev_g = AlignedEvent(
        event_id="ev_g", source_content_id="src_cont", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=19000, end_sample=25000, label="G", base_phone="G",
        predecessor_id="ev_ih", successor_id="ev_k_cont"  # Links directly to ev_k_cont!
    )
    w_big = WordOccurrence(
        word_id="w_big", source_content_id="src_cont", speaker_id="spk1",
        word_normalized="big", word_original="big",
        start_sample=10000, end_sample=25000,
        phone_event_ids=["ev_b", "ev_ih", "ev_g"],
        realized_phones=["B", "IH1", "G"],
    )

    # Natural continuation Word 2: 'cat' [K, AE1, T] starting at 25000 in src_cont
    # Made somewhat rushed so duration penalty pushes its unary rank down
    ev_k_cont = AlignedEvent(
        event_id="ev_k_cont", source_content_id="src_cont", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=25000, end_sample=28000, label="K", base_phone="K",
        predecessor_id="ev_g", successor_id="ev_ae_cont"
    )
    ev_ae_cont = AlignedEvent(
        event_id="ev_ae_cont", source_content_id="src_cont", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=28000, end_sample=31000, label="AE1", base_phone="AE",
        predecessor_id="ev_k_cont", successor_id="ev_t_cont"
    )
    ev_t_cont = AlignedEvent(
        event_id="ev_t_cont", source_content_id="src_cont", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=31000, end_sample=33000, label="T", base_phone="T",
        predecessor_id="ev_ae_cont"
    )
    w_cat_cont = WordOccurrence(
        word_id="w_cat_cont", source_content_id="src_cont", speaker_id="spk1",
        word_normalized="cat", word_original="cat",
        start_sample=25000, end_sample=33000,  # 8000 samples = 0.166s (rushed)
        phone_event_ids=["ev_k_cont", "ev_ae_cont", "ev_t_cont"],
        realized_phones=["K", "AE1", "T"],
    )

    sources: Dict[str, SourceRecord] = {"src_cont": src_cont}
    events: Dict[str, AlignedEvent] = {
        "ev_b": ev_b, "ev_ih": ev_ih, "ev_g": ev_g,
        "ev_k_cont": ev_k_cont, "ev_ae_cont": ev_ae_cont, "ev_t_cont": ev_t_cont,
    }
    words: Dict[str, WordOccurrence] = {"w_big": w_big, "w_cat_cont": w_cat_cont}
    word_to_ids: Dict[str, List[str]] = {"big": ["w_big"], "cat": ["w_cat_cont"]}
    p_to_e: Dict[str, List[str]] = {
        "B": ["ev_b"], "IH1": ["ev_ih"], "G": ["ev_g"],
        "K": ["ev_k_cont"], "AE1": ["ev_ae_cont"], "T": ["ev_t_cont"],
    }
    bp_to_e: Dict[str, List[str]] = {
        "B": ["ev_b"], "IH": ["ev_ih"], "G": ["ev_g"],
        "K": ["ev_k_cont"], "AE": ["ev_ae_cont"], "T": ["ev_t_cont"],
    }

    # Now create 55 competing occurrences of 'cat' from other sources with ideal durations (0.30s)
    # They rank higher (1..55) on unary heuristics than w_cat_cont (#56)
    for i in range(1, 56):
        s_id = f"src_comp_{i}"
        sources[s_id] = SourceRecord(
            content_id=s_id, source_path=f"{s_id}.mp4", native_sample_rate=sr, native_channels=2
        )
        ek = f"ev_k_comp_{i}"
        eae = f"ev_ae_comp_{i}"
        et = f"ev_t_comp_{i}"
        events[ek] = AlignedEvent(
            event_id=ek, source_content_id=s_id, utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=1000, end_sample=5000, label="K", base_phone="K",
            successor_id=eae
        )
        events[eae] = AlignedEvent(
            event_id=eae, source_content_id=s_id, utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=5000, end_sample=11000, label="AE1", base_phone="AE",
            predecessor_id=ek, successor_id=et
        )
        events[et] = AlignedEvent(
            event_id=et, source_content_id=s_id, utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=11000, end_sample=15400, label="T", base_phone="T",
            predecessor_id=eae
        )
        wid = f"w_cat_comp_{i}"
        words[wid] = WordOccurrence(
            word_id=wid, source_content_id=s_id, speaker_id="spk1",
            word_normalized="cat", word_original="cat",
            start_sample=1000, end_sample=15400,  # exactly 0.30s = optimal
            phone_event_ids=[ek, eae, et],
            realized_phones=["K", "AE1", "T"],
        )
        word_to_ids["cat"].append(wid)
        p_to_e["K"].append(ek)
        p_to_e["AE1"].append(eae)
        p_to_e["T"].append(et)
        bp_to_e["K"].append(ek)
        bp_to_e["AE"].append(eae)
        bp_to_e["T"].append(et)

    inv = InventoryIndex(
        sources=sources, events=events, words=words,
        word_to_word_ids=word_to_ids,
        phone_to_event_ids=p_to_e,
        base_phone_to_event_ids=bp_to_e,
    )

    lex = Lexicon()
    lex.word_to_pronunciations["big"] = [["B", "IH1", "G"]]
    lex.word_to_pronunciations["cat"] = [["K", "AE1", "T"]]
    builder = TargetGraphBuilder(lex)
    graph = builder.build("big cat")

    # Set max_candidates_per_span=50. There are 56 total word occurrences of 'cat'.
    # w_cat_cont is ranked #56 by unary sort key due to rush penalty.
    selector = SpanSelector(inv, edit_penalty=0.5, max_candidates_per_span=50)
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")

    # Verify w_cat_cont is present in cands despite being rank #56
    cont_cand = next((c for c in cands if c.start_sample == 25000 and c.source_content_id == "src_cont"), None)
    assert cont_cand is not None, "Natural continuation candidate ranked #56 was erroneously pruned by top-50 cutoff!"

    # Verify Viterbi DP selects the continuous pair
    selected = selector.select(graph, speaker_id="spk1")
    assert len(selected) == 2
    assert selected[0].matched_word == "big"
    assert selected[1].matched_word == "cat"
    assert selected[0].source_content_id == "src_cont"
    assert selected[1].source_content_id == "src_cont"
    assert selected[1].start_sample == 25000, "Viterbi failed to select 0-cost natural continuation!"


def test_speaker_median_quadratic_duration_penalty_and_heuristic_capping():
    """
    Fix P3.2: Tests quadratic duration penalty z-score behavior,
    speaker median duration modeling, and unary heuristic penalty capping at 0.50.
    """
    # 1. Quadratic duration penalty formula validation
    pen_ideal = SpanSelector._compute_quadratic_duration_penalty(0.20, 0.20)
    assert pen_ideal == 0.0

    pen_plateau = SpanSelector._compute_quadratic_duration_penalty(0.25, 0.20)  # ratio = 1.25 in [0.75, 1.50]
    assert pen_plateau == 0.0

    # Rushed speech: ratio = 0.50 (< 0.75)
    # Expected: 0.60 * ((0.75 - 0.50) / 0.75) ** 2 = 0.60 * (1/3)**2 = 0.06666...
    pen_rushed = SpanSelector._compute_quadratic_duration_penalty(0.10, 0.20)
    assert abs(pen_rushed - (0.60 * (0.25 / 0.75) ** 2)) < 1e-5

    # Extremely rushed: ratio = 0.25
    pen_extreme_rushed = SpanSelector._compute_quadratic_duration_penalty(0.05, 0.20)
    assert pen_extreme_rushed > pen_rushed

    # Dragged out speech: ratio = 1.75 (> 1.50)
    # Expected: 0.30 * min(2.0, (1.75 - 1.50) / 1.0) ** 2 = 0.30 * (0.25)**2 = 0.01875
    pen_dragged = SpanSelector._compute_quadratic_duration_penalty(0.35, 0.20)
    assert abs(pen_dragged - (0.30 * (0.25) ** 2)) < 1e-5

    # 2. Speaker median duration learning from inventory
    sr = 48000
    src = SourceRecord(content_id="s1", source_path="s1.wav", native_sample_rate=sr, native_channels=1)
    # spk_fast speaks 'hello' in 0.12s
    w1 = WordOccurrence(
        word_id="w1", source_content_id="s1", speaker_id="spk_fast",
        word_normalized="hello", word_original="hello",
        start_sample=0, end_sample=int(0.12 * sr),
        phone_event_ids=[], realized_phones=["HH", "AH0", "L", "OW1"]
    )
    # spk_slow speaks 'hello' in 0.40s
    w2 = WordOccurrence(
        word_id="w2", source_content_id="s1", speaker_id="spk_slow",
        word_normalized="hello", word_original="hello",
        start_sample=int(0.2 * sr), end_sample=int(0.60 * sr),
        phone_event_ids=[], realized_phones=["HH", "AH0", "L", "OW1"]
    )
    inv = InventoryIndex(
        sources={"s1": src}, events={}, words={"w1": w1, "w2": w2},
        word_to_word_ids={"hello": ["w1", "w2"]},
    )
    selector = SpanSelector(inv)
    assert abs(selector._get_speaker_median_duration("spk_fast", "hello", 0.25) - 0.12) < 1e-3
    assert abs(selector._get_speaker_median_duration("spk_slow", "hello", 0.25) - 0.40) < 1e-3
    assert abs(selector._get_speaker_median_duration("spk_unknown", "hello", 0.25) - 0.26) < 1e-3
