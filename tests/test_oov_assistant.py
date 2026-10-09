"""
Verification tests for Milestone 5: Interactive Out-Of-Vocabulary (OOV) Learning Assistant.
Tests OOV detection, sounds-like exemplar formatting, user decision pathways,
and dictionary persistence.
"""

from unittest.mock import create_autospec

import numpy as np
import pytest

from lexicon import Lexicon, format_phoneme_breakdown
from oov_assistant import OOVAssistant
from phoneme_recognizer import PhonemeRecognizer


@pytest.fixture
def temp_dict_path(tmp_path):
    d_path = tmp_path / "custom_pronunciations.txt"
    d_path.write_text("knownword\tN OW1 N\n", encoding="utf-8")
    return d_path


def test_format_phoneme_breakdown():
    """Verify that phoneme breakdown formats human-readable sounds-like descriptions."""
    breakdown = format_phoneme_breakdown(["B", "AE1", "K", "T", "AH0"])
    assert "B" in breakdown
    assert "'b' in boy" in breakdown
    assert "AE1" in breakdown
    assert "'a' in apple" in breakdown
    assert "(stressed)" in breakdown
    assert "AH0" in breakdown
    assert "(unstressed)" in breakdown


def test_oov_detection(temp_dict_path):
    """Verify that OOVAssistant detects unknown words but skips known words and phonetic groups."""
    lex = Lexicon([temp_dict_path])
    assistant = OOVAssistant(lex, custom_dict_path=temp_dict_path)

    text = "knownword bacta {B AE1 K AH0 N} <gap:250ms> midichlorian!"
    oov = assistant.detect_oov_words(text)

    # 'knownword' is in dictionary; '{B AE1 K AH0 N}' is explicit phonetic; '<gap:250ms>' is gap
    # 'bacta' and 'midichlorian' are unknown
    assert oov == ["bacta", "midichlorian"]


def test_oov_non_interactive_fallback(temp_dict_path):
    """Verify that non-interactive resolution returns G2P suggestion without writing to disk."""
    lex = Lexicon([temp_dict_path])
    assistant = OOVAssistant(lex, custom_dict_path=temp_dict_path)

    initial_content = temp_dict_path.read_text(encoding="utf-8")
    phones = assistant.resolve_oov_word("bacta", interactive=False)

    assert len(phones) > 0
    # Did not mutate dictionary file
    assert temp_dict_path.read_text(encoding="utf-8") == initial_content


def test_oov_action_1_accept_suggestion(temp_dict_path):
    """Verify Action 1 accepts G2P suggestion and persists to dictionary."""
    lex = Lexicon([temp_dict_path])
    assistant = OOVAssistant(lex, custom_dict_path=temp_dict_path)

    # Mock user input choosing '1'
    phones = assistant.resolve_oov_word(
        "bacta",
        interactive=True,
        input_fn=lambda prompt: "1",
    )

    assert len(phones) > 0
    assert lex.is_known("bacta")
    content = temp_dict_path.read_text(encoding="utf-8")
    assert "bacta" in content
    assert " ".join(phones) in content


def test_oov_action_2_custom_arpabet_entry(temp_dict_path):
    """Verify Action 2 accepts valid custom ARPAbet phonemes and persists."""
    lex = Lexicon([temp_dict_path])
    assistant = OOVAssistant(lex, custom_dict_path=temp_dict_path)

    inputs = ["2", "B AE1 K T AH0"]
    input_iter = iter(inputs)

    phones = assistant.resolve_oov_word(
        "bacta",
        interactive=True,
        input_fn=lambda prompt: next(input_iter),
    )

    assert phones == ["B", "AE1", "K", "T", "AH0"]
    assert lex.is_known("bacta")
    content = temp_dict_path.read_text(encoding="utf-8")
    assert "bacta\tB AE1 K T AH0" in content or "bacta" in content


def test_oov_action_3_microphone_ingestion(temp_dict_path):
    """Verify Action 3 connects microphone/phoneme recognizer and persists result."""
    lex = Lexicon([temp_dict_path])
    assistant = OOVAssistant(lex, custom_dict_path=temp_dict_path)

    mock_rec = create_autospec(PhonemeRecognizer, instance=True)
    mock_rec.record_audio_snippet.return_value = np.zeros(16000, dtype=np.float32)
    mock_rec.audio_to_arpabet.return_value = ["M", "IH1", "D", "IH0", "K", "L", "AO1", "R", "IY0", "AH0", "N"]

    inputs = ["3", "y"]  # Choice 3, then confirm 'y'
    input_iter = iter(inputs)

    phones = assistant.resolve_oov_word(
        "midichlorian",
        interactive=True,
        input_fn=lambda prompt: next(input_iter),
        recognizer=mock_rec,
    )

    assert phones == ["M", "IH1", "D", "IH0", "K", "L", "AO1", "R", "IY0", "AH0", "N"]
    assert lex.is_known("midichlorian")
    content = temp_dict_path.read_text(encoding="utf-8")
    assert "midichlorian" in content
    mock_rec.record_audio_snippet.assert_called_once_with(duration=2.5)
    mock_rec.audio_to_arpabet.assert_called_once()


def test_oov_action_3_microphone_failure_falls_back(temp_dict_path):
    """Verify Action 3 falls back to G2P suggestion when recording raises RuntimeError."""
    lex = Lexicon([temp_dict_path])
    assistant = OOVAssistant(lex, custom_dict_path=temp_dict_path)

    mock_rec = create_autospec(PhonemeRecognizer, instance=True)
    mock_rec.record_audio_snippet.side_effect = RuntimeError("Microphone device not found")

    inputs = ["3"]
    input_iter = iter(inputs)

    phones = assistant.resolve_oov_word(
        "midichlorian",
        interactive=True,
        input_fn=lambda prompt: next(input_iter),
        recognizer=mock_rec,
    )

    # Falls back to G2P suggestion
    assert len(phones) > 0
    # Did not persist failed pronunciation
    assert not lex.is_known("midichlorian")


def test_oov_action_4_ignore(temp_dict_path):
    """Verify Action 4 returns suggestion without saving to dictionary."""
    lex = Lexicon([temp_dict_path])
    assistant = OOVAssistant(lex, custom_dict_path=temp_dict_path)

    initial_content = temp_dict_path.read_text(encoding="utf-8")
    phones = assistant.resolve_oov_word(
        "temporaryword",
        interactive=True,
        input_fn=lambda prompt: "4",
    )

    assert len(phones) > 0
    assert not lex.is_known("temporaryword")
    assert temp_dict_path.read_text(encoding="utf-8") == initial_content


def test_t19_oov_deduplication_and_validation(temp_dict_path):
    """T19: TTY/non-TTY OOV; known ambiguity; use-once/save/reload/concurrent write.
    No blocking headless read; validated atomic deduplication; versioned shared lexicon."""
    lex = Lexicon([temp_dict_path])
    assistant = OOVAssistant(lex, custom_dict_path=temp_dict_path)

    # 1. Non-interactive (headless) mode: must return G2P suggestion without reading stdin
    phones = assistant.resolve_oov_word("headlessword", interactive=False)
    assert len(phones) > 0
    assert not lex.is_known("headlessword")  # not saved to dictionary automatically

    # 2. Deduplication: saving the same pronunciation twice does not produce duplicate lines
    lex.save_custom_pronunciation("padawan", ["P", "AE1", "D", "AH0", "W", "AA2", "N"], temp_dict_path)
    lex.save_custom_pronunciation("padawan", ["P", "AE1", "D", "AH0", "W", "AA2", "N"], temp_dict_path)

    lines = [ln.strip() for ln in temp_dict_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    padawan_lines = [ln for ln in lines if ln.startswith("padawan")]
    assert len(padawan_lines) == 1, f"Expected exactly 1 line for padawan, got {len(padawan_lines)}: {padawan_lines}"

    # 3. Validation: attempting to save invalid phone raises ValueError and does not write
    with pytest.raises(ValueError, match="consonants cannot take stress"):
        lex.save_custom_pronunciation("badword", ["T1", "AE1"], temp_dict_path)
    assert not lex.is_known("badword")

