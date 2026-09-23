"""
Unit and integration tests for Milestone 3 (R2):
- Hybrid Phonetic Syntax Parsing ({B AE1 K AH0 N})
- Reverse Pronunciation Index in SpanSelector for Intact Word Retrieval
- Strict Syntax & Phone Alphabet Validation
- Microphone / Audio-to-Phoneme Recognition & Analysis-Only Invariants
"""

from __future__ import annotations

from pathlib import Path
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


def test_phoneme_recognizer_interface_and_audio_to_arpabet(monkeypatch):
    """Verify PhonemeRecognizer exposes real interface and connects audio to arpabet."""
    rec = PhonemeRecognizer()
    assert hasattr(rec, "record_audio_snippet")
    assert hasattr(rec, "audio_to_arpabet")
    assert hasattr(rec, "transcribe_audio_to_text")
    assert hasattr(rec, "text_to_arpabet")
    assert hasattr(rec, "recognize_phonetic_string")

    # audio_to_arpabet with mocked transcription
    monkeypatch.setattr(rec, "transcribe_audio_to_text", lambda audio, sample_rate=16000: "bacon")
    phones = rec.audio_to_arpabet(np.zeros(16000, dtype=np.float32))
    assert phones in (["B", "EY1", "K", "AH0", "N"], ["B", "EY1", "K", "AH1", "N"])

    # Empty audio transcription returns empty list
    monkeypatch.setattr(rec, "transcribe_audio_to_text", lambda audio, sample_rate=16000: "")
    assert rec.audio_to_arpabet(np.zeros(16000, dtype=np.float32)) == []

    # Calling record_audio_snippet without sounddevice raises RuntimeError
    with pytest.raises(RuntimeError, match="Microphone recording unavailable"):
        rec.record_audio_snippet(duration=0.1)


def test_t18_empty_recognition_raises_runtime_error(monkeypatch):
    """T18: Empty recognition must raise RuntimeError, never fabricating {AH0}."""
    rec = PhonemeRecognizer()
    monkeypatch.setattr(rec, "transcribe_audio_to_text", lambda audio, sample_rate=16000: "")
    with pytest.raises(RuntimeError, match="Empty audio recognition result"):
        rec.recognize_phonetic_string(np.zeros(16000, dtype=np.float32), interactive=False)


def test_t06_pronunciation_branch_isolation_and_compounds():
    """T06: Two-pronunciation 'read', compounds, repeated ambiguous words. One connected pronunciation per occurrence/component."""
    lex = Lexicon([])
    # 'read' has two pronunciations: present (R IY1 D) and past (R EH1 D)
    lex.word_to_pronunciations["read"] = [["R", "IY1", "D"], ["R", "EH1", "D"]]
    lex.word_to_pronunciations["off"] = [["AO1", "F"]]

    # 1. Compound 'read-off' must produce bounded Cartesian product (2 distinct pronunciations)
    prons = lex.get_pronunciations("read-off")
    assert len(prons) == 2, f"Expected 2 compound pronunciations, got {len(prons)}: {prons}"
    assert ["R", "IY1", "D", "AO1", "F"] in prons
    assert ["R", "EH1", "D", "AO1", "F"] in prons

    # 2. TargetGraphBuilder assigns distinct branch_ids
    builder = TargetGraphBuilder(lex)
    graph = builder.build("read")
    nodes_b0 = [n for n in graph.nodes if n.branch_id == 0]
    nodes_b1 = [n for n in graph.nodes if n.branch_id == 1]
    assert [n.phone for n in nodes_b0] == ["R", "IY1", "D"]
    assert [n.phone for n in nodes_b1] == ["R", "EH1", "D"]

    # 3. Span candidate branches do not cross: candidate covering branch 0 does not contain nodes from branch 1
    inv = InventoryIndex()
    inv.sources["src1"] = SourceRecord(
        content_id="src1", source_path="v.mp4", native_sample_rate=16000, native_channels=1
    )
    # Source occurrence of "read" pronounced R IY1 D
    evs = [
        AlignedEvent(event_id=f"e{i}", source_content_id="src1", utterance_id="u1", speaker_id="spk1",
                     kind="phone", start_sample=i*1600, end_sample=(i+1)*1600, label=p, base_phone=p[:2])
        for i, p in enumerate(["R", "IY1", "D"])
    ]
    for i in range(len(evs)-1):
        evs[i].successor_id = evs[i+1].event_id
    wo = WordOccurrence(
        word_id="wo_read", source_content_id="src1", speaker_id="spk1",
        word_normalized="read", word_original="read", start_sample=0, end_sample=4800,
        phone_event_ids=[e.event_id for e in evs], realized_phones=["R", "IY1", "D"],
    )
    inv.events = {e.event_id: e for e in evs}
    inv.words = {"wo_read": wo}
    inv.word_to_word_ids = {"read": ["wo_read"]}

    selector = SpanSelector(inv)
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")
    word_cands = [c for c in cands if c.span_type == "word"]
    for c in word_cands:
        # Branch IDs of all covered nodes must be identical
        b_ids = {graph.nodes[nid].branch_id for nid in c.covered_nodes}
        assert len(b_ids) == 1, f"Candidate {c.candidate_id} spans mixed branch IDs: {b_ids}"


def test_t07_strict_phone_syntax_validation():
    """T07: {AE9}, {AH10}, {T1}, malformed groups/gaps. Location-aware rejection; no silent repair/persistence."""
    # 1. Invalid stress digit > 2
    with pytest.raises(ValueError, match="Invalid stress '9'"):
        tokenize_hybrid_syntax("Hello {AE9} world")

    # 2. Extra stress digits
    with pytest.raises(ValueError, match="Invalid stress '10'"):
        tokenize_hybrid_syntax("Hello {AH10} world")

    # 3. Consonant stress
    with pytest.raises(ValueError, match="consonants cannot take stress"):
        tokenize_hybrid_syntax("Hello {T1} world")

    # 4. Negative gap duration
    with pytest.raises(ValueError, match="must be finite and non-negative"):
        tokenize_hybrid_syntax("Hello <gap:-50ms> world")

    # 5. Empty phonetic group
    with pytest.raises(ValueError, match="Empty phonetic group"):
        tokenize_hybrid_syntax("Hello {} world")


def test_t17_microphone_and_wav_adapter_rates_and_safety(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    T17: Microphone action through actual adapter; WAV at 16/44.1/48 kHz:
    Real API is connected; correct rate/duration; editable result; no mic audio in plan.
    """
    import soundfile as sf
    from phoneme_recognizer import PhonemeRecognizer, assert_no_mic_audio_in_plan
    from contracts import EditPlan, PlanSegment, RationalTime

    rec = PhonemeRecognizer()
    # Mock transcribe_audio_to_text to avoid downloading/running ASR model in unit test
    monkeypatch.setattr(rec, "transcribe_audio_to_text", lambda audio, sample_rate=16000: "hello")

    # Test WAV files at 16 kHz, 44.1 kHz, and 48 kHz
    for target_sr in [16000, 44100, 48000]:
        dur = 0.5
        sine_audio = (0.2 * np.sin(2 * np.pi * 440.0 * np.linspace(0, dur, int(dur * target_sr)))).astype(np.float32)
        wav_path = tmp_path / f"test_{target_sr}.wav"
        sf.write(str(wav_path), sine_audio, target_sr, subtype="PCM_16")

        result = rec.recognize_phonetic_string(wav_path, interactive=False)
        assert result.startswith("{") and result.endswith("}")
        assert "HH" in result

    # Test interactive edit capability
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("builtins.input", lambda _: "{HH AH0 L OW0}")
    edited_res = rec.recognize_phonetic_string(wav_path, interactive=True)
    assert edited_res == "{HH AH0 L OW0}"

    # Verify assert_no_mic_audio_in_plan catches leaked microphone audio
    leaked_plan = EditPlan(
        target_text="test",
        speaker_id="speaker_01",
        project_sample_rate=48000,
        video_fps=RationalTime(30, 1),
        total_output_samples=4800,
        total_video_frames=3,
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="mic_audio_clip",
                speaker_id="speaker_01",
                source_start_sample=0,
                source_end_sample=4800,
                output_start_sample=0,
                output_end_sample=4800,
                target_text="test",
                target_phones=["T"],
                span_type="word",
                source_video_path="recorded_mic_temp.wav",
            )
        ],
    )
    with pytest.raises(AssertionError, match="Safety violation: microphone analysis audio detected"):
        assert_no_mic_audio_in_plan(leaked_plan)




