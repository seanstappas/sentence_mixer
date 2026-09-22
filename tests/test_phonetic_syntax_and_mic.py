"""
Unit and integration tests for Milestone 3 (R2):
- Hybrid Phonetic Syntax Parsing ({B AE1 K AH0 N})
- Reverse Pronunciation Index in SpanSelector for Intact Word Retrieval
- Strict Syntax & Phone Alphabet Validation
- Microphone / Audio-to-Phoneme Recognition & Analysis-Only Invariants
"""

from __future__ import annotations

import numpy as np
import pytest

from contracts import EditPlan, InventoryIndex, PlanSegment, SourceRecord, WordOccurrence, AlignedEvent
from lexicon import Lexicon
from phoneme_recognizer import PhonemeRecognizer, assert_no_mic_audio_in_plan
from span_selector import SpanSelector
from target_graph import TargetGraphBuilder, tokenize_hybrid_syntax


def test_tokenize_hybrid_syntax_valid():
    text = "Hello {B AE1 K AH0 N}, <gap:250ms> Obi-Wan!"
    tokens = tokenize_hybrid_syntax(text)

    # Expected: 'hello', '{B AE1 K AH0 N}', ',', '<gap:250ms>', 'obi-wan', '!'
    kinds = [t.kind for t in tokens]
    assert kinds == ["word", "phonetic", "punctuation", "gap", "word", "punctuation"]

    phone_tok = tokens[1]
    assert phone_tok.phones == ["B", "AE1", "K", "AH0", "N"]
    assert phone_tok.norm_text == "{B AE1 K AH0 N}"

    gap_tok = tokens[3]
    assert np.isclose(gap_tok.gap_seconds, 0.250)


def test_tokenize_hybrid_syntax_validation_errors():
    # 1. Unclosed brace
    with pytest.raises(ValueError, match="Unmatched braces"):
        tokenize_hybrid_syntax("Hello {B AE1 K AH0 N")

    # 2. Unexpected closing brace
    with pytest.raises(ValueError, match="Unmatched braces"):
        tokenize_hybrid_syntax("Hello B AE1} there")

    # 3. Invalid ARPAbet phone
    with pytest.raises(ValueError, match="Invalid ARPAbet phone 'NOT_A_PHONE'"):
        tokenize_hybrid_syntax("Hello {NOT_A_PHONE} there")

    # 4. Unclosed gap tag
    with pytest.raises(ValueError, match="Unmatched angle brackets"):
        tokenize_hybrid_syntax("Hello <gap:250ms there")

    # 5. Invalid gap format
    with pytest.raises(ValueError, match="Invalid gap format"):
        tokenize_hybrid_syntax("Hello <gap:invalid_duration> there")

    # 6. Empty phonetic group
    with pytest.raises(ValueError, match="Empty phonetic group"):
        tokenize_hybrid_syntax("Hello {} there")


def test_target_graph_builder_with_hybrid_tokens():
    lex = Lexicon([])
    lex.word_to_pronunciations["hello"] = [["HH", "AH0", "L", "OW1"]]
    lex.word_to_pronunciations["there"] = [["DH", "EH1", "R"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("Hello {B AE1 K AH0 N} there")

    assert graph.normalized_words == ["hello", "{B AE1 K AH0 N}", "there"]
    assert len(graph.nodes) == 4 + 5 + 3  # 4 for hello, 5 for bacon, 3 for there

    # Verify nodes for {B AE1 K AH0 N}
    bacon_nodes = [n for n in graph.nodes if n.token_index == 1]
    assert len(bacon_nodes) == 5
    assert [n.phone for n in bacon_nodes] == ["B", "AE1", "K", "AH0", "N"]
    assert bacon_nodes[0].is_word_initial is True
    assert bacon_nodes[-1].is_word_final is True


def test_reverse_pronunciation_index_retrieves_intact_word():
    """
    Verifies that when an explicit phonetic group {B AE1 K AH0 N} is requested,
    SpanSelector uses its reverse pronunciation index to retrieve the recorded
    word 'bacon' as an intact whole-word candidate instead of breaking into phones.
    """
    inv = InventoryIndex()
    inv.sources["src1"] = SourceRecord(
        content_id="src1",
        source_path="video1.mp4",
        native_sample_rate=16000,
        native_channels=1,
    )

    # Word 'bacon' in inventory
    bacon_events = []
    phones = ["B", "AE1", "K", "AH0", "N"]
    for i, p in enumerate(phones):
        eid = f"ev_bacon_{i}"
        ev = AlignedEvent(
            event_id=eid,
            source_content_id="src1",
            utterance_id="utt_bacon",
            speaker_id="speaker_0",
            kind="phone",
            start_sample=i * 1000,
            end_sample=(i + 1) * 1000,
            label=p,
            word_occurrence_id="w_bacon_01",
        )
        inv.events[eid] = ev
        inv.phone_to_event_ids.setdefault(p, []).append(eid)
        bacon_events.append(eid)

    # Link events
    for i in range(len(bacon_events) - 1):
        inv.events[bacon_events[i]].successor_id = bacon_events[i + 1]
        inv.events[bacon_events[i + 1]].predecessor_id = bacon_events[i]

    wo_bacon = WordOccurrence(
        word_id="w_bacon_01",
        source_content_id="src1",
        speaker_id="speaker_0",
        word_normalized="bacon",
        word_original="Bacon",
        start_sample=0,
        end_sample=5000,
        phone_event_ids=bacon_events,
        realized_phones=phones,
    )
    inv.words["w_bacon_01"] = wo_bacon
    inv.word_to_word_ids["bacon"] = ["w_bacon_01"]

    # Target sentence requesting explicit phonetic group
    lex = Lexicon([])
    builder = TargetGraphBuilder(lex)
    graph = builder.build("{B AE1 K AH0 N}")

    selector = SpanSelector(inv)
    candidates = selector.retrieve_candidates(graph)

    # Check that a whole word candidate was formed from the reverse pronunciation index
    word_cands = [c for c in candidates if c.span_type == "word"]
    assert len(word_cands) >= 1
    assert word_cands[0].matched_word == "bacon"
    assert word_cands[0].start_sample == 0
    assert word_cands[0].end_sample == 5000

    # Ensure search selects the intact whole word
    selected = selector.select(graph)
    assert len(selected) == 1
    assert selected[0].span_type == "word"
    assert selected[0].matched_word == "bacon"


def test_phoneme_recognizer_and_mic_safety():
    rec = PhonemeRecognizer()

    # Test text to ARPAbet conversion
    phones = rec.text_to_arpabet("bacon")
    assert phones in (["B", "EY1", "K", "AH0", "N"], ["B", "EY1", "K", "AH1", "N"])

    # Test safety invariant: assert_no_mic_audio_in_plan
    clean_plan = EditPlan(
        target_text="test text",
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="obi_wan_ep2",
                speaker_id="obi-wan",
                source_start_sample=0,
                source_end_sample=16000,
                output_start_sample=0,
                output_end_sample=16000,
                target_text="hello",
                target_phones=["HH", "AH0", "L", "OW1"],
                span_type="word",
                source_video_path="input/obi_wan_ep2.mp4",
            )
        ],
    )
    assert_no_mic_audio_in_plan(clean_plan)

    # Contaminated plan with mic audio must raise AssertionError
    contaminated_plan = EditPlan(
        target_text="test text",
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="mic_audio_01",
                speaker_id="user",
                source_start_sample=0,
                source_end_sample=16000,
                output_start_sample=0,
                output_end_sample=16000,
                target_text="hello",
                target_phones=["HH", "AH0", "L", "OW1"],
                span_type="word",
                source_video_path="temp/mic_recorded.wav",
            )
        ],
    )
    with pytest.raises(AssertionError, match="Safety violation: microphone analysis audio"):
        assert_no_mic_audio_in_plan(contaminated_plan)
