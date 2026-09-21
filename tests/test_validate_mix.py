"""
Tests for tools/validate_mix.py validation gate.
Verifies rejection of invalid timing, missing files, and provenance mismatch.
"""

import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

from contracts import EditPlan, PlanSegment, SourceRecord
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

