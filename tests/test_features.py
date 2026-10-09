"""
Tests for continuous acoustic feature extraction and join cost computation.
Verifies voicing detection, pitch tracking, missing-data handling, and continuation zero-cost.
"""

import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

from contracts import FeatureRecord
from features import ContinuousFeatureExtractor, compute_join_cost


def create_synthetic_audio(path: Path, sr: int = 16000):
    """Creates a 1-second audio file: 0.5s 220Hz sine wave (voiced) + 0.5s silence (unvoiced)."""
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    sine = 0.5 * np.sin(2 * np.pi * 220.0 * t)  # 220 Hz A3 tone
    silence = np.zeros(int(sr * 0.5))
    signal = np.concatenate([sine, silence])
    sf.write(str(path), signal, sr)


def test_continuous_features_voicing_and_pitch():
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = Path(tmpdir) / "test_synth.wav"
        create_synthetic_audio(wav_path, sr=16000)

        extractor = ContinuousFeatureExtractor(sr=16000)
        tracks = extractor.extract_tracks(wav_path)

        assert tracks["n_frames"] > 0
        assert len(tracks["rms"]) == tracks["n_frames"]
        assert len(tracks["f0"]) == tracks["n_frames"]

        # Interval 1: 0.0s to 0.4s (voiced tone)
        feat_voiced = extractor.get_window_features(
            tracks, start_sample=0, end_sample=6400, native_sr=16000
        )
        assert feat_voiced.voicing_state == "voiced"
        assert feat_voiced.f0_hz is not None
        assert abs(feat_voiced.f0_hz - 220.0) < 15.0  # close to 220 Hz

        # Interval 2: 0.6s to 0.9s (silence/unvoiced)
        feat_unvoiced = extractor.get_window_features(
            tracks, start_sample=9600, end_sample=14400, native_sr=16000
        )
        assert feat_unvoiced.voicing_state == "unvoiced"
        # Invariant: unvoiced frame must NOT fabricate 120 Hz or arbitrary pitch
        assert feat_unvoiced.f0_hz is None
        assert feat_unvoiced.rms_energy < 1e-3


def test_join_cost_invariants():
    # Invariant: natural original continuations have exactly 0.0 join cost
    assert compute_join_cost(None, None, is_natural_continuation=True) == 0.0

    feat_a = FeatureRecord(
        source_content_id="s1",
        sample_rate=16000,
        start_sample=0,
        end_sample=1600,
        rms_energy=0.1,
        f0_hz=200.0,
        f0_confidence=0.9,
        voicing_state="voiced",
        spectral_centroid=1500.0,
        mfcc_13=[1.0] * 13,
    )

    feat_b_close = FeatureRecord(
        source_content_id="s2",
        sample_rate=16000,
        start_sample=1600,
        end_sample=3200,
        rms_energy=0.105,
        f0_hz=202.0,
        f0_confidence=0.9,
        voicing_state="voiced",
        spectral_centroid=1520.0,
        mfcc_13=[1.02] * 13,
    )

    feat_c_distant = FeatureRecord(
        source_content_id="s3",
        sample_rate=16000,
        start_sample=3200,
        end_sample=4800,
        rms_energy=0.01,
        f0_hz=350.0,
        f0_confidence=0.9,
        voicing_state="voiced",
        spectral_centroid=3500.0,
        mfcc_13=[5.0] * 13,
    )

    cost_close = compute_join_cost(feat_a, feat_b_close, is_natural_continuation=False)
    cost_distant = compute_join_cost(feat_a, feat_c_distant, is_natural_continuation=False)

    assert cost_close < cost_distant, "Acoustically distant features must have higher join cost"

    # Invariant: missing features must NOT earn zero-cost joins
    cost_missing = compute_join_cost(feat_a, None, is_natural_continuation=False)
    assert cost_missing > cost_close
