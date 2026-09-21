"""
Unit tests for ProsodyGuide and safety invariant assertions (M6).
"""

import numpy as np
import pytest
import soundfile as sf
from pathlib import Path

from src.contracts import EditPlan, PlanSegment, RationalTime
from src.prosody_guide import ProsodyGuide, assert_no_guide_audio_in_plan


@pytest.fixture
def dummy_guide_audio(tmp_path: Path) -> Path:
    audio_path = tmp_path / "guide.wav"
    sr = 16000
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Sine wave at 220Hz
    signal = 0.5 * np.sin(2 * np.pi * 220 * t)
    sf.write(str(audio_path), signal, sr)
    return audio_path


def test_prosody_guide_transcript_validation(dummy_guide_audio: Path):
    guide = ProsodyGuide(dummy_guide_audio, guide_transcript="Hello there")
    assert guide.validate_transcript("hello there") is True
    assert guide.validate_transcript("Hello there!") is True
    assert guide.validate_transcript("General Kenobi") is False


def test_prosody_guide_relative_pitch_contour(dummy_guide_audio: Path):
    guide = ProsodyGuide(dummy_guide_audio, guide_transcript="Hello there")
    contour = guide.get_relative_pitch_contour()
    assert len(contour) > 0
    # Sine wave at 220Hz should produce steady pitch near 0 relative semitones
    voiced_semitones = [c for c in contour if c is not None]
    assert len(voiced_semitones) > 0
    assert abs(np.mean(voiced_semitones)) < 1.0


def test_assert_no_guide_audio_in_plan(dummy_guide_audio: Path, tmp_path: Path):
    safe_video = tmp_path / "source.mp4"
    safe_video.touch()

    # Valid plan with safe source
    safe_plan = EditPlan(
        target_text="Hello there",
        video_fps=RationalTime(30, 1),
        project_sample_rate=48000,
        total_output_samples=24000,
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="source_a",
                speaker_id="speaker_a",
                source_start_sample=0,
                source_end_sample=24000,
                output_start_sample=0,
                output_end_sample=24000,
                target_text="hello",
                target_phones=["HH", "AH0", "L", "OW1"],
                span_type="word",
                source_video_path=str(safe_video),
                is_silence=False,
            )
        ]
    )

    # Should not raise
    assert_no_guide_audio_in_plan(safe_plan, dummy_guide_audio)

    # Leaked plan pointing to guide audio path
    leaked_plan = EditPlan(
        target_text="Hello there",
        video_fps=RationalTime(30, 1),
        project_sample_rate=48000,
        total_output_samples=24000,
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="source_a",
                speaker_id="speaker_a",
                source_start_sample=0,
                source_end_sample=24000,
                output_start_sample=0,
                output_end_sample=24000,
                target_text="hello",
                target_phones=["HH", "AH0", "L", "OW1"],
                span_type="word",
                source_video_path=str(dummy_guide_audio),
                is_silence=False,
            )
        ]
    )

    with pytest.raises(AssertionError, match="Safety Violation"):
        assert_no_guide_audio_in_plan(leaked_plan, dummy_guide_audio)

    # Leaked plan with source_content_id='guide'
    leaked_id_plan = EditPlan(
        target_text="Hello there",
        video_fps=RationalTime(30, 1),
        project_sample_rate=48000,
        total_output_samples=24000,
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="guide",
                speaker_id="speaker_a",
                source_start_sample=0,
                source_end_sample=24000,
                output_start_sample=0,
                output_end_sample=24000,
                target_text="hello",
                target_phones=["HH", "AH0", "L", "OW1"],
                span_type="word",
                source_video_path=str(safe_video),
                is_silence=False,
            )
        ]
    )

    with pytest.raises(AssertionError, match="Safety Violation"):
        assert_no_guide_audio_in_plan(leaked_id_plan, dummy_guide_audio)
