"""
Tests for target pronunciation graph construction and context preservation.
Verifies cross-word context (e.g. T followed by K), pronunciation branching, and manner-based duration priors.
"""

from lexicon import Lexicon
from target_graph import TargetGraphBuilder


def test_cross_word_context_preservation():
    # Mandatory regression: cross-word boundary T followed by K
    # E.g. 'hot key' -> 'hot' ends in T, 'key' starts with K
    lex = Lexicon()
    # Ensure known pronunciations
    lex.word_to_pronunciations["hot"] = [["HH", "AA1", "T"]]
    lex.word_to_pronunciations["key"] = [["K", "IY1"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("hot key")

    # Find the T node in 'hot' and the K node in 'key'
    t_nodes = [n for n in graph.nodes if n.word_text == "hot" and n.base_phone == "T"]
    k_nodes = [n for n in graph.nodes if n.word_text == "key" and n.base_phone == "K"]

    assert len(t_nodes) == 1
    assert len(k_nodes) == 1
    t_node = t_nodes[0]
    k_node = k_nodes[0]

    # Invariant: cross-word context must be preserved, not replaced by SIL
    assert t_node.right_context_phone == "K"
    assert k_node.left_context_phone == "T"


def test_pronunciation_branching():
    lex = Lexicon()
    # 'the' has two pronunciations: DH AH0 and DH IY0
    lex.word_to_pronunciations["the"] = [["DH", "AH0"], ["DH", "IY0"]]
    lex.word_to_pronunciations["force"] = [["F", "AO1", "R", "S"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("the force")

    # There should be two start nodes for 'the' (both starting with DH)
    assert len(graph.start_nodes) == 2

    # Graph should contain both AH0 and IY0 nodes
    phones = [n.phone for n in graph.nodes if n.word_text == "the"]
    assert "AH0" in phones
    assert "IY0" in phones


def test_duration_priors_manner_categories():
    lex = Lexicon()
    lex.word_to_pronunciations["kenobi"] = [["K", "AH0", "N", "OW1", "B", "IY0"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("Kenobi")

    k_node = next(n for n in graph.nodes if n.base_phone == "K")
    ow_node = next(n for n in graph.nodes if n.base_phone == "OW")

    # Invariant: Stop consonant K prior is shorter than diphthong OW prior; no universal 90ms
    assert k_node.duration_prior_sec != ow_node.duration_prior_sec
    assert k_node.duration_prior_sec < ow_node.duration_prior_sec


def test_punctuation_pause():
    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH", "AH0", "L", "OW1"]]
    lex.word_to_pronunciations["there"] = [["DH", "EH1", "R"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("Hello, there!")

    # Comma after word 0 gives 0.25s pause, exclamation after word 1 gives 0.45s pause
    assert 0 in graph.punctuation_pauses
    assert graph.punctuation_pauses[0] == 0.25
    assert 1 in graph.punctuation_pauses
    assert graph.punctuation_pauses[1] == 0.45


def test_multibranch_context_set_preservation():
    """
    Fix P3.4: TargetPhoneNode preserves right_context_phones: Set[str]
    and left_context_phones: Set[str] across multiple pronunciation branches.
    """
    lex = Lexicon()
    # Word 1: 'either' with two branches: [IY1, DH, ER0] and [AY1, DH, ER0]
    lex.word_to_pronunciations["either"] = [["IY1", "DH", "ER0"], ["AY1", "DH", "ER0"]]
    # Word 2: 'way' [W, EY1]
    lex.word_to_pronunciations["way"] = [["W", "EY1"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("either way")

    # Final nodes of 'either' across both branches should point to 'W' in both single and set context
    er_nodes = [n for n in graph.nodes if n.word_text == "either" and n.base_phone == "ER"]
    assert len(er_nodes) == 2
    for er_node in er_nodes:
        assert er_node.right_context_phone == "W"
        assert "W" in er_node.right_context_phones
        assert isinstance(er_node.right_context_phones, set)

    # Initial node of 'way' should have left_context_phones matching both branches
    w_nodes = [n for n in graph.nodes if n.word_text == "way" and n.base_phone == "W"]
    assert len(w_nodes) == 1
    w_node = w_nodes[0]
    assert "ER0" in w_node.left_context_phones
    assert isinstance(w_node.left_context_phones, set)
