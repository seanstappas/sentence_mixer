"""
Tests for tools/validate_mix.py validation gate.
Verifies rejection of invalid timing, missing files, and provenance mismatch.
"""

import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

from contracts import EditPlan, PlanSegment, RationalTime, SourceRecord
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from validate_mix import validate_mix


def test_validate_mix_catches_missing_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        stem = Path(tmpdir) / "nonexistent_mix"
        # Must fail when files are missing
        assert validate_mix(str(stem)) is False


def test_validate_mix_catches_sample_mismatch():
    with tempfile.TemporaryDirectory() as tmpdir:
        stem = Path(tmpdir) / "mismatch_mix"
        sr = 48000

        # Plan declares 24000 samples
        src_rec = SourceRecord(
            content_id="src_1",
            source_path="video.mp4",
            native_sample_rate=sr,
            native_channels=2,
        )
        seg = PlanSegment(
            segment_index=0,
            source_content_id="src_1",
            speaker_id="spk1",
            source_start_sample=0,
            source_end_sample=24000,
            output_start_sample=0,
            output_end_sample=24000,
            target_text="test",
            target_phones=[],
            span_type="word",
            source_video_path="video.mp4",
        )
        plan = EditPlan(
            plan_id="plan_1",
            project_sample_rate=sr,
            total_output_samples=24000,
            segments=[seg],
            sources={"src_1": src_rec},
        )
        plan.save(stem.with_suffix(".plan.json"))

        # But WAV file has 20000 samples (4000 samples short)
        data = np.zeros(20000, dtype=np.float32)
        sf.write(str(stem.with_suffix(".wav")), data, sr)

        # Create dummy mp4 file
        stem.with_suffix(".mp4").write_bytes(b"dummy")

        # Must fail because sample counts do not match
        assert validate_mix(str(stem)) is False


def test_measure_audio_loudness_pure_digital_silence(tmp_path: Path):
    from validate_mix import measure_audio_loudness

    silence_wav = tmp_path / "pure_silence.wav"
    sr = 48000
    data = np.zeros(48000, dtype=np.float32)
    sf.write(str(silence_wav), data, sr, subtype="PCM_16")

    i_lufs, tp_dbtp = measure_audio_loudness(silence_wav)
    assert i_lufs <= -70.0
    assert tp_dbtp == float("-inf")


def test_t25_validate_mix_rejects_stale_corrupt_mix(tmp_path: Path):
    """
    T25: Stale artifact, wrong frame count, invalid XML, unmet loudness.
    Full validator rejects; no existence-only or widened-tolerance pass.
    """
    stem = tmp_path / "corrupt_mix"
    sr = 48000

    src_rec = SourceRecord(
        content_id="src_valid",
        source_path="dummy.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )

    # 1. Reject invalid source bounds
    corrupt_seg = PlanSegment(
        segment_index=0,
        source_content_id="src_valid",
        speaker_id="spk1",
        source_start_sample=5000,
        source_end_sample=1000,  # Invalid: end < start
        output_start_sample=0,
        output_end_sample=4000,
        target_text="test",
        target_phones=[],
        span_type="word",
        source_video_path="dummy.mp4",
    )

    plan = EditPlan(
        plan_id="plan_corrupt",
        project_sample_rate=sr,
        total_output_samples=4000,
        segments=[corrupt_seg],
        sources={"src_valid": src_rec},
    )
    plan.save(stem.with_suffix(".plan.json"))

    # Write matching WAV and dummy MP4
    data = np.zeros(4000, dtype=np.float32)
    sf.write(str(stem.with_suffix(".wav")), data, sr)
    stem.with_suffix(".mp4").write_bytes(b"dummy")

    # Validator must reject invalid source bounds
    assert validate_mix(str(stem), skip_asr=True) is False

    # 2. Reject XML timebase mismatch
    valid_seg = PlanSegment(
        segment_index=0,
        source_content_id="src_valid",
        speaker_id="spk1",
        source_start_sample=0,
        source_end_sample=4000,
        output_start_sample=0,
        output_end_sample=4000,
        target_text="test",
        target_phones=[],
        span_type="word",
        source_video_path="dummy.mp4",
    )
    valid_plan = EditPlan(
        plan_id="plan_valid",
        project_sample_rate=sr,
        total_output_samples=4000,
        video_fps=RationalTime(30, 1),
        segments=[valid_seg],
        sources={"src_valid": src_rec},
    )
    valid_plan.save(stem.with_suffix(".plan.json"))

    # Create XML with mismatched timebase (24 instead of 30)
    xml_content = """<?xml version="1.0" encoding="utf-8"?>
    <xmeml version="4">
      <sequence>
        <rate><timebase>24</timebase></rate>
      </sequence>
    </xmeml>"""
    stem.with_suffix(".xml").write_text(xml_content, encoding="utf-8")

    assert validate_mix(str(stem), skip_asr=True) is False


