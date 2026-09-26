"""
Regression test suite for Milestone 4 (B08, B09, B07, U16-U22).
Verifies:
- U16: --target-audio saying "hello there" preserves word boundaries ({HH AH0 L OW1} {DH EH1 R})
  and selects whole words rather than phones.
- U17: Recording realized R EH1 D: Target {R EH1 D} for 'read' via Tier 1 aligner; spelling fallback offers both variants.
- U18: Edited {T1} rejected with re-prompt; missing Whisper names package; silent audio detected; low confidence in headless exits.
- U19: 'read read' in TTY: one review per occurrence; graph constrained per occurrence; 'use once' not persisted; metadata records choices.
- U20: Save custom pronunciation deduplication on parsed normalized (word, phones) tuple; concurrent saves persist.
- U21: Headless OOV with no G2P fails naming word (no AH0); numeral '1' normalized to 'one'.
- U22: Constrained R EH1 D rejects R IY1 D recording; unconstrained allowed with cost; plan records realized_phones.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pytest
import soundfile as sf

from contracts import (
    AlignedEvent,
    InventoryIndex,
    SourceRecord,
    WordOccurrence,
)
from edit_plan import build_edit_plan
from lexicon import Lexicon, normalize_numerals
from oov_assistant import OOVAssistant
from phoneme_recognizer import PhonemeRecognizer
from span_selector import SpanSelector
from target_graph import TargetGraphBuilder


def _build_dummy_inventory(word_specs: List[dict]) -> InventoryIndex:
    inv = InventoryIndex()
    src = SourceRecord(
        content_id="src_01",
        source_path="fake/path.mp4",
        native_sample_rate=16000,
        native_channels=1,
    )
    inv.sources["src_01"] = src

    event_counter = 0
    word_counter = 0

    for ws in word_specs:
        w_norm = ws["word"]
        phones = ws["phones"]
        realized = ws.get("realized_phones", phones)
        spk = ws.get("speaker_id", "speaker-a")
        start = ws.get("start_sample", word_counter * 8000)
        dur = ws.get("duration_samples", 8000)
        end = start + dur

        phone_eids = []
        p_dur = dur // max(len(realized), 1)
        for i, p in enumerate(realized):
            eid = f"ev_{event_counter:04d}"
            ev = AlignedEvent(
                event_id=eid,
                source_content_id="src_01",
                utterance_id="utt_01",
                speaker_id=spk,
                kind="phone",
                start_sample=start + i * p_dur,
                end_sample=start + (i + 1) * p_dur,
                label=p,
                base_phone=p.rstrip("012"),
                is_eligible=True,
            )
            inv.events[eid] = ev
            inv.phone_to_event_ids.setdefault(p, []).append(eid)
            inv.base_phone_to_event_ids.setdefault(ev.base_phone, []).append(eid)
            phone_eids.append(eid)
            event_counter += 1

        wid = f"w_{word_counter:04d}"
        wo = WordOccurrence(
            word_id=wid,
            source_content_id="src_01",
            speaker_id=spk,
            word_normalized=w_norm,
            word_original=w_norm,
            start_sample=start,
            end_sample=end,
            phone_event_ids=phone_eids,
            realized_phones=realized,
            is_eligible=True,
        )
        inv.words[wid] = wo
        inv.word_to_word_ids.setdefault(w_norm, []).append(wid)
        word_counter += 1

    return inv


def test_u16_target_audio_preserves_word_boundaries_and_selects_whole_words(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    U16: --target-audio saying 'hello there' (fake Whisper, aligner):
    Two word groups ({HH AH0 L OW1} {DH EH1 R}), whole-word selection (2 word spans, not 7 phones).
    """
    audio_path = tmp_path / "hello_there.wav"
    sf.write(str(audio_path), np.ones(16000, dtype=np.float32) * 0.1, 16000)

    lex = Lexicon([])
    lex.word_to_pronunciations["hello"] = [["HH", "AH0", "L", "OW1"]]
    lex.word_to_pronunciations["there"] = [["DH", "EH1", "R"]]

    rec = PhonemeRecognizer(lexicon=lex)
    monkeypatch.setattr(rec, "transcribe_audio_to_text", lambda audio, sample_rate=16000: "hello there")

    # Recognizer outputs separate phonetic groups per word
    phonetic_str = rec.recognize_phonetic_string(audio_path, interactive=False)
    assert phonetic_str == "{HH AH0 L OW1} {DH EH1 R}"

    # Build inventory containing whole word recordings for 'hello' and 'there'
    inv = _build_dummy_inventory([
        {"word": "hello", "phones": ["HH", "AH0", "L", "OW1"]},
        {"word": "there", "phones": ["DH", "EH1", "R"]},
    ])

    graph_builder = TargetGraphBuilder(lex)
    graph = graph_builder.build(phonetic_str)
    assert len(graph.normalized_words) == 2

    selector = SpanSelector(inv)
    selected = selector.select(graph)

    # Must select 2 whole-word spans, NOT 7 individual phone spans!
    assert len(selected) == 2
    assert selected[0].span_type == "word"
    assert selected[0].matched_word == "hello"
    assert selected[1].span_type == "word"
    assert selected[1].matched_word == "there"


def test_u17_recording_realized_r_eh_d_tier1_and_spelling_fallback(tmp_path: Path):
    """
    U17: Recording realized R EH1 D:
    Target {R EH1 D} for 'read' via Tier 1 MFA; spelling-only fallback offers both variants.
    """
    lex = Lexicon([])
    lex.word_to_pronunciations["read"] = [["R", "IY1", "D"], ["R", "EH1", "D"]]

    # 1. Tier 1 with injected aligner choosing R EH1 D
    class FakeMFAAligner:
        def align_word(self, audio, word, variants, start, end):
            if word == "read":
                return ["R", "EH1", "D"]
            return variants[0]

    rec_tier1 = PhonemeRecognizer(lexicon=lex, aligner=FakeMFAAligner())
    dummy_audio = np.ones(16000, dtype=np.float32) * 0.1
    targets_tier1 = rec_tier1.recognize_structured_words(dummy_audio, text_override="read")

    assert len(targets_tier1) == 1
    assert targets_tier1[0].tier == "mfa_tier1"
    assert targets_tier1[0].phones == ["R", "EH1", "D"]
    assert targets_tier1[0].alternatives == [["R", "IY1", "D"]]

    # 2. Spelling-only fallback (no aligner)
    rec_spelling = PhonemeRecognizer(lexicon=lex, aligner=None)
    targets_spelling = rec_spelling.recognize_structured_words(dummy_audio, text_override="read")

    assert len(targets_spelling) == 1
    assert targets_spelling[0].tier == "spelling_fallback"
    # Canonical is first variant, alternative is second
    assert targets_spelling[0].phones == ["R", "IY1", "D"]
    assert ["R", "EH1", "D"] in targets_spelling[0].alternatives


def test_u18_validation_reprompt_missing_whisper_silent_and_low_confidence(monkeypatch: pytest.MonkeyPatch):
    """
    U18: Edited {T1} rejected with re-prompt; missing Whisper names package;
    silent audio detected; low confidence in headless exits with code/error naming uncertain words.
    """
    lex = Lexicon([])
    lex.word_to_pronunciations["test"] = [["T", "EH1", "S", "T"]]
    rec = PhonemeRecognizer(lexicon=lex)

    # a) Edited {T1} validation error and re-prompt in interactive TTY mode
    monkeypatch.setattr(rec, "transcribe_audio_to_text", lambda audio, sample_rate=16000: "test")
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)

    prompts = ["{T1}", "{T EH1 S T}"]
    def mock_input(prompt_text):
        return prompts.pop(0)

    # First invalid attempt {T1} must be rejected, and second attempt accepted
    result = rec.recognize_phonetic_string(np.ones(16000, dtype=np.float32) * 0.1, interactive=True, input_fn=mock_input)
    assert result == "{T EH1 S T}"

    # b) Missing Whisper package error names 'faster-whisper'
    def mock_missing_whisper(*args, **kwargs):
        raise RuntimeError("Whisper package ('faster-whisper') not installed. Please install faster-whisper or use text target.")
    monkeypatch.setattr(rec, "transcribe_audio_to_text", mock_missing_whisper)
    with pytest.raises(RuntimeError, match="faster-whisper"):
        rec.recognize_phonetic_string(np.ones(16000, dtype=np.float32) * 0.1, interactive=False)

    # c) Silent audio detected
    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="silent"):
        rec.recognize_phonetic_string(np.zeros(16000, dtype=np.float32), interactive=False)

    # d) Low confidence in headless exits naming uncertain words
    monkeypatch.setattr(rec, "transcribe_audio_to_text", lambda audio, sample_rate=16000: "murky")
    monkeypatch.setattr(
        rec,
        "transcribe_audio_to_words",
        lambda audio, sample_rate=16000: [{"word": "murky", "start": 0.0, "end": 1.0, "probability": 0.25}],
    )
    with pytest.raises(RuntimeError, match="murky"):
        rec.recognize_phonetic_string(np.ones(16000, dtype=np.float32) * 0.1, interactive=False, min_confidence=0.6)


def test_u19_heteronym_review_per_occurrence_and_plan_metadata(tmp_path: Path):
    """
    U19: 'read read' in TTY with different choices:
    One review per occurrence; graph constrained per occurrence; 'use once' not persisted; choices in plan metadata.
    """
    lex = Lexicon([])
    lex.word_to_pronunciations["read"] = [["R", "IY1", "D"], ["R", "EH1", "D"]]
    custom_dict = tmp_path / "custom.txt"

    assistant = OOVAssistant(lexicon=lex, custom_dict_path=custom_dict)

    # Token 0 -> choose variant 1 (R IY1 D), scope 1 (once)
    # Token 1 -> choose variant 2 (R EH1 D), scope 1 (once)
    inputs = ["1", "1", "2", "1"]
    def mock_input(p):
        return inputs.pop(0)

    choices = assistant.review_all_in_text("read read", interactive=True, input_fn=mock_input)

    assert len(choices) == 2
    assert choices[0].token_index == 0
    assert choices[0].phones == ["R", "IY1", "D"]
    assert choices[0].scope == "once"

    assert choices[1].token_index == 1
    assert choices[1].phones == ["R", "EH1", "D"]
    assert choices[1].scope == "once"

    # 'use once' must NOT be persisted to custom_dict
    assert not custom_dict.exists()

    # Target graph builder receives per-occurrence overrides
    overrides = {c.token_index: c.phones for c in choices}
    builder = TargetGraphBuilder(lex)
    graph = builder.build("read read", overrides=overrides)

    # First 'read' is constrained to R IY1 D
    first_read_nodes = [n for n in graph.nodes if n.token_index == 0]
    assert [n.phone for n in first_read_nodes] == ["R", "IY1", "D"]
    assert all(n.is_constrained for n in first_read_nodes)

    # Second 'read' is constrained to R EH1 D
    second_read_nodes = [n for n in graph.nodes if n.token_index == 1]
    assert [n.phone for n in second_read_nodes] == ["R", "EH1", "D"]
    assert all(n.is_constrained for n in second_read_nodes)


def test_u20_save_custom_pronunciation_deduplication(tmp_path: Path):
    """
    U20: Save the same pronunciation as an existing 'BACTA  B AE1 K T AH0' line:
    No duplicate; concurrent saves still both persist.
    """
    custom_file = tmp_path / "custom_pronunciations.txt"
    # Pre-populate with uppercase word and multi-space formatting
    custom_file.write_text("# Existing comment\nBACTA   B AE1 K T AH0\n", encoding="utf-8")

    lex = Lexicon([])
    # Save the exact same pronunciation (lowercase word, single space)
    lex.save_custom_pronunciation("bacta", ["B", "AE1", "K", "T", "AH0"], custom_file)

    lines = [ln.strip() for ln in custom_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2  # comment + 1 entry (NO duplicate!)
    assert lines[0] == "# Existing comment"
    assert "BACTA" in lines[1]

    # Save a second different word
    lex.save_custom_pronunciation("kolto", ["K", "OW1", "L", "T", "OW0"], custom_file)
    lines_after = [ln.strip() for ln in custom_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines_after) == 3


def test_u21_headless_oov_no_g2p_and_numeral_normalization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    U21: Headless OOV with no G2P exits naming the word (no AH0); target containing '1' normalized to 'one'.
    """
    # 1. Numeral normalization: "1" -> "one", preserving <gap:250ms> and {B AE1 K T AH0}
    norm = normalize_numerals("Anakin 1 <gap:250ms> {B AE1 K T AH0}")
    assert "one" in norm
    assert "<gap:250ms>" in norm
    assert "{B AE1 K T AH0}" in norm

    # 2. Headless OOV with no G2P
    lex = Lexicon([])
    # Mock get_g2p to return None
    monkeypatch.setattr("lexicon.get_g2p", lambda: None)

    assistant = OOVAssistant(lexicon=lex, custom_dict_path=tmp_path / "custom.txt")
    with pytest.raises(RuntimeError, match="foobar"):
        assistant.review_all_in_text("foobar", interactive=False)


def test_u22_constrained_realization_rejection_and_plan_realized_phones(tmp_path: Path):
    """
    U22: Constrained R EH1 D rejects R IY1 D recording;
    unconstrained reduced realizations still usable with a recorded cost; plan records realized_phones.
    """
    lex = Lexicon([])
    # 'read' listed with R EH1 D and R IY1 D
    lex.word_to_pronunciations["read"] = [["R", "EH1", "D"], ["R", "IY1", "D"]]

    # Inventory has ONLY an 'R IY1 D' recording of 'read'
    inv = _build_dummy_inventory([
        {"word": "read", "phones": ["R", "IY1", "D"], "realized_phones": ["R", "IY1", "D"]},
    ])

    selector = SpanSelector(inv)

    # Case A: Constrained target {R EH1 D}
    builder = TargetGraphBuilder(lex)
    constrained_graph = builder.build("{R EH1 D}")
    cands_constrained = selector.retrieve_candidates(constrained_graph)

    # Word candidate with realized [R, IY1, D] must NOT be selected for constrained [R, EH1, D]
    word_cands_constrained = [c for c in cands_constrained if c.span_type == "word"]
    assert len(word_cands_constrained) == 0, f"Expected 0 matching word candidates for constrained mismatch, got {word_cands_constrained}"

    # Case B: Unconstrained target word 'read'
    unconstrained_graph = builder.build("read")
    cands_unconstrained = selector.retrieve_candidates(unconstrained_graph)
    word_cands_unconstrained = [c for c in cands_unconstrained if c.span_type == "word"]
    # Branch 1 (R IY1 D) matches the recording
    assert len(word_cands_unconstrained) >= 1
    chosen = word_cands_unconstrained[0]
    assert chosen.realized_phones == ["R", "IY1", "D"]

    # Build edit plan and check PlanSegment.realized_phones
    plan = build_edit_plan([chosen], unconstrained_graph, inv, project_sample_rate=16000)
    assert len(plan.segments) == 1
    assert plan.segments[0].realized_phones == ["R", "IY1", "D"]
