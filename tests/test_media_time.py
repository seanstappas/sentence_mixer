"""
Tests for media time mapping, sample/frame conversions, and rational timing invariants.
"""

from fractions import Fraction
from contracts import RationalTime
from media_time import (
    samples_to_seconds,
    seconds_to_samples,
    samples_to_frame_index,
    frame_index_to_samples,
)


def test_time_and_samples_conversion():
    sr = 48000
    sec = 2.5
    s = seconds_to_samples(sec, sr)
    assert s == 120000
    assert samples_to_seconds(s, sr) == 2.5


def test_rational_frame_mapping_no_drift():
    # Test at 29.97 fps (30000/1001) over 10,000 frames
    fps = RationalTime(30000, 1001)
    sr = 48000

    # Test round-trip conversion for various frame indices
    for f in [0, 1, 30, 300, 1000, 30000]:
        sample = frame_index_to_samples(f, fps, sr)
        recalculated_frame = samples_to_frame_index(sample, sr, fps)
        assert recalculated_frame == f, f"Frame mismatch at frame {f}: got {recalculated_frame}"


def test_rational_time_fraction():
    r = RationalTime(30000, 1001)
    frac = r.as_fraction()
    assert frac == Fraction(30000, 1001)
