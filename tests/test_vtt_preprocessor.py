"""
Tests for WebVTT subtitle preprocessing.
Verifies multiline retention, rolling caption deduplication, and genuine repetition preservation.
"""

import tempfile
from pathlib import Path
from vtt_preprocessor import parse_vtt_cues, clean_vtt_text


def test_ordinary_multiline_cue():
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:04.000
Hello there
General Kenobi
"""
    with tempfile.NamedTemporaryFile("w", suffix=".vtt", delete=False, encoding="utf-8") as f:
        f.write(vtt_content)
        temp_path = f.name

    try:
        cues = parse_vtt_cues(temp_path)
        assert len(cues) == 1
        assert cues[0]["text"] == "Hello there General Kenobi"
        assert cues[0]["start_sec"] == 1.0
        assert cues[0]["end_sec"] == 4.0
    finally:
        Path(temp_path).unlink()


def test_rolling_captions_display_dedup():
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:03.000
Hello there

00:00:02.500 --> 00:00:05.000
Hello there
General Kenobi
"""
    with tempfile.NamedTemporaryFile("w", suffix=".vtt", delete=False, encoding="utf-8") as f:
        f.write(vtt_content)
        temp_path = f.name

    try:
        cues = parse_vtt_cues(temp_path)
        assert len(cues) == 2
        assert cues[0]["text"] == "Hello there"
        # The rolling display duplicate 'Hello there' should be omitted from cue 2
        assert cues[1]["text"] == "General Kenobi"
        assert cues[1]["start_sec"] >= 2.5
    finally:
        Path(temp_path).unlink()


def test_genuine_repetition_preserved():
    # Genuinely repeated word without temporal overlap
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:02.000
Kenobi

00:00:03.000 --> 00:00:04.000
Kenobi
"""
    with tempfile.NamedTemporaryFile("w", suffix=".vtt", delete=False, encoding="utf-8") as f:
        f.write(vtt_content)
        temp_path = f.name

    try:
        cues = parse_vtt_cues(temp_path)
        assert len(cues) == 2
        assert cues[0]["text"] == "Kenobi"
        assert cues[1]["text"] == "Kenobi"
    finally:
        Path(temp_path).unlink()


def test_clean_vtt_text():
    assert clean_vtt_text("<00:00:01.000><c> Obi-Wan's </c> [Applause]") == "Obi-Wan's"
    assert clean_vtt_text(">> Chancellor Palpatine!") == "Chancellor Palpatine!"
