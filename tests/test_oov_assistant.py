"""
Verification tests for Milestone 5: Interactive Out-Of-Vocabulary (OOV) Learning Assistant.
Tests OOV detection, sounds-like exemplar formatting, user decision pathways,
and dictionary persistence.
"""

import pytest
import numpy as np

from lexicon import Lexicon, format_phoneme_breakdown
from oov_assistant import OOVAssistant


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

    class MockRecognizer:
        def record_audio_snippet(self, duration=2.5):
            return np.zeros(16000, dtype=np.float32)

        def audio_to_arpabet(self, audio):
            return ["M", "IH1", "D", "IH0", "K", "L", "AO1", "R", "IY0", "AH0", "N"]

    inputs = ["3", "y"]  # Choice 3, then confirm 'y'
    input_iter = iter(inputs)

    phones = assistant.resolve_oov_word(
        "midichlorian",
        interactive=True,
        input_fn=lambda prompt: next(input_iter),
        recognizer=MockRecognizer(),
    )

    assert phones == ["M", "IH1", "D", "IH0", "K", "L", "AO1", "R", "IY0", "AH0", "N"]
    assert lex.is_known("midichlorian")
    content = temp_dict_path.read_text(encoding="utf-8")
    assert "midichlorian" in content


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
