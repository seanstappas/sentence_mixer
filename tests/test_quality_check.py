"""
Tests for tools/quality_check.py.
Verifies acoustic and ASR enunciation checks, duration floor verification,
suite verification, JSON reporting, and CLI exit codes.
"""

from __future__ import annotations

import json
from pathlib import Path
import pytest
import soundfile as sf
import numpy as np

from tools.quality_check import (
    check_render_quality,
    check_suite_quality,
    main as quality_check_main,
)


@pytest.fixture
def dummy_media_and_plan(tmp_path: Path):
    sr = 16000
    duration_sec = 1.0
    samples = int(sr * duration_sec)
    audio = (0.2 * np.sin(2 * np.pi * 440 * np.linspace(0, duration_sec, samples))).astype(np.float32)

    wav_path = tmp_path / "test_render.wav"
    sf.write(str(wav_path), audio, sr)

    plan_path = tmp_path / "test_render.plan.json"
    plan_data = {
        "project_sample_rate": sr,
        "segments": [
            {
                "segment_index": 0,
                "target_text": "hello",
                "output_start_sample": 0,
                "output_end_sample": int(sr * 0.4),  # 400ms
                "gain_db": 0.0,
                "is_silence": False,
            },
            {
                "segment_index": 1,
                "target_text": "there",
                "output_start_sample": int(sr * 0.4),
                "output_end_sample": int(sr * 0.8),  # 400ms
                "gain_db": 0.0,
                "is_silence": False,
            },
        ],
    }
    plan_path.write_text(json.dumps(plan_data), encoding="utf-8")
    return wav_path, plan_path


def test_quality_check_pass(monkeypatch: pytest.MonkeyPatch, dummy_media_and_plan):
    wav_path, plan_path = dummy_media_and_plan

    # Mock verify_listening to return matching transcript
    mock_asr = {
        "transcribed_text": "hello there",
        "assisted_metrics": {"transcription": "hello there", "wer": 0.0},
        "assisted_wer": 0.0,
        "wer": 0.0,
    }
    monkeypatch.setattr("tools.quality_check.verify_listening", lambda **kwargs: mock_asr)

    passed, report = check_render_quality(
        input_media=wav_path,
        expected_text="Hello there",
        plan_path=plan_path,
        critical_words=["hello", "there"],
    )
    assert passed is True
    assert report["overall_passed"] is True
    assert report["checks"]["critical_words_detected"]["passed"] is True
    assert len(report["issues"]) == 0


def test_quality_check_missing_critical_word(monkeypatch: pytest.MonkeyPatch, dummy_media_and_plan):
    wav_path, plan_path = dummy_media_and_plan

    # Mock ASR missing critical word
    mock_asr = {
        "transcribed_text": "hello",
        "assisted_metrics": {"transcription": "hello", "wer": 0.5},
        "assisted_wer": 0.5,
        "wer": 0.5,
    }
    monkeypatch.setattr("tools.quality_check.verify_listening", lambda **kwargs: mock_asr)

    passed, report = check_render_quality(
        input_media=wav_path,
        expected_text="Hello there",
        plan_path=plan_path,
        critical_words=["hello", "there"],
    )
    assert passed is False
    assert report["overall_passed"] is False
    assert report["checks"]["critical_words_detected"]["passed"] is False
    assert "there" in report["checks"]["critical_words_detected"]["missing"]


def test_quality_check_duration_floor(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    sr = 16000
    wav_path = tmp_path / "short_render.wav"
    sf.write(str(wav_path), np.zeros(16000, dtype=np.float32), sr)

    plan_path = tmp_path / "short_render.plan.json"
    plan_data = {
        "project_sample_rate": sr,
        "segments": [
            {
                "segment_index": 0,
                "target_text": "quick",
                "output_start_sample": 0,
                "output_end_sample": int(sr * 0.03),  # 30ms (below 65ms floor)
                "gain_db": 0.0,
                "is_silence": False,
            },
        ],
    }
    plan_path.write_text(json.dumps(plan_data), encoding="utf-8")

    # ASR fails to hear 'quick'
    mock_asr = {
        "transcribed_text": "unintelligible",
        "assisted_metrics": {"transcription": "", "wer": 1.0},
        "assisted_wer": 1.0,
        "wer": 1.0,
    }
    monkeypatch.setattr("tools.quality_check.verify_listening", lambda **kwargs: mock_asr)

    passed, report = check_render_quality(
        input_media=wav_path,
        expected_text="quick",
        plan_path=plan_path,
        critical_words=["quick"],
        min_word_duration_ms=65.0,
    )
    assert passed is False
    assert report["checks"]["duration_check"]["passed"] is False


def test_quality_check_suite_mode(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, dummy_media_and_plan):
    wav_path, plan_path = dummy_media_and_plan

    suite_path = tmp_path / "test_suite.json"
    suite_data = {
        "suite_name": "Test Suite",
        "cases": [
            {
                "case_id": "test_render",
                "target_text": "Hello there",
                "critical_words": ["hello"],
            }
        ],
    }
    suite_path.write_text(json.dumps(suite_data), encoding="utf-8")

    mock_asr = {
        "transcribed_text": "hello there",
        "assisted_metrics": {"transcription": "hello there", "wer": 0.0},
        "assisted_wer": 0.0,
        "wer": 0.0,
    }
    monkeypatch.setattr("tools.quality_check.verify_listening", lambda **kwargs: mock_asr)

    passed, reports = check_suite_quality(
        suite_path=suite_path,
        render_dir=tmp_path,
    )
    assert passed is True
    assert len(reports) == 1
    assert reports[0]["case_id"] == "test_render"


def test_quality_check_cli_exit_codes(monkeypatch: pytest.MonkeyPatch, dummy_media_and_plan):
    wav_path, _ = dummy_media_and_plan

    # Pass case
    monkeypatch.setattr(
        "tools.quality_check.verify_listening",
        lambda **kwargs: {
            "transcribed_text": "hello there",
            "assisted_metrics": {"transcription": "hello there", "wer": 0.0},
            "assisted_wer": 0.0,
            "wer": 0.0,
        },
    )
    code = quality_check_main(["--input", str(wav_path), "--expected", "Hello there", "--json"])
    assert code == 0

    # Nonexistent file -> exit code 2
    code_missing = quality_check_main(["--input", "nonexistent.mp4", "--expected", "foo"])
    assert code_missing == 2
