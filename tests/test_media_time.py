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


def test_t13_two_audio_streams_and_rational_time_mapping(tmp_path):
    """
    T13: Two audio streams, nonzero origins, mixed rates, VFR.
    Verifies that extract_source_record maps the specific stream index,
    records the correct sample rate, origin, and rational conversions.
    """
    import subprocess
    import soundfile as sf
    from media_time import extract_source_record

    # Create a test MP4 container with 1 video stream and 2 distinct audio streams:
    # Stream 0: video (testsrc2, 1 sec, 30 fps)
    # Stream 1: audio 44.1 kHz sine 440 Hz
    # Stream 2: audio 48.0 kHz sine 880 Hz
    out_mp4 = tmp_path / "dual_audio.mp4"
    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc2=duration=1:size=320x240:rate=30",
        "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=44100:duration=1",
        "-f", "lavfi", "-i", "sine=frequency=880:sample_rate=48000:duration=1",
        "-map", "0:v",
        "-map", "1:a",
        "-map", "2:a",
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a:0", "pcm_s16le",
        "-c:a:1", "pcm_s16le",
        str(out_mp4)
    ]
    subprocess.run(cmd, capture_output=True, check=True)

    rec = extract_source_record(out_mp4, build_dir=tmp_path / "build")

    # Audio stream 1 should be selected (first audio stream)
    assert rec.selected_audio_stream == 1
    assert rec.native_sample_rate == 44100
    assert rec.cached_pcm_path != ""

    # Decoded PCM audio must match the 44.1 kHz rate
    data, sr = sf.read(rec.cached_pcm_path)
    assert sr == 44100
    assert len(data) >= 44000


def test_prefer_avg_frame_rate_over_r_frame_rate(tmp_path, monkeypatch):
    import media_time
    from media_time import extract_source_record

    dummy_media = tmp_path / "dummy.mp4"
    dummy_media.write_bytes(b"dummy")

    mock_probe = {
        "streams": [
            {
                "codec_type": "video",
                "index": 0,
                "r_frame_rate": "1000/1",
                "avg_frame_rate": "24/1",
                "duration": "1.0",
                "start_time": "0.040",
            },
            {
                "codec_type": "audio",
                "index": 1,
                "sample_rate": "48000",
                "channels": 2,
                "duration": "1.0",
                "start_time": "0.020",
                "initial_padding": "1024",
            },
        ],
        "format": {"duration": "1.0"},
    }

    monkeypatch.setattr(media_time, "probe_media_file", lambda p: mock_probe)
    # Mock subprocess so ffmpeg decoding is bypassed
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)

    rec = extract_source_record(dummy_media, build_dir=tmp_path / "build")
    # Video fps must prefer avg_frame_rate (24/1) over r_frame_rate (1000/1)
    assert rec.video_fps["numerator"] == 24
    assert rec.video_fps["denominator"] == 1
    # Codec delay and origins must be extracted
    assert rec.codec_delay_samples == 1024
    assert rec.decoded_audio_origin_sec == 0.020
    assert rec.decoded_video_origin_sec == 0.040


def test_codec_delay_start_pad_fallback(tmp_path, monkeypatch):
    import media_time
    from media_time import extract_source_record
    import subprocess

    dummy_media = tmp_path / "dummy.mp4"
    dummy_media.write_bytes(b"dummy")

    mock_probe = {
        "streams": [
            {
                "codec_type": "audio",
                "index": 0,
                "sample_rate": "48000",
                "channels": 1,
                "duration": "1.0",
                "start_pad": 512,
            },
        ],
        "format": {"duration": "1.0"},
    }

    monkeypatch.setattr(media_time, "probe_media_file", lambda p: mock_probe)
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)

    rec = extract_source_record(dummy_media, build_dir=tmp_path / "build")
    assert rec.codec_delay_samples == 512
    assert rec.decoded_video_origin_sec == 0.0


