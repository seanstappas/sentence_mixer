"""
Adversarial stress-testing suite for Milestone 4 (Phase 4: Fixes P4.1–P4.3).
Focuses on edge cases, mathematical limits, corruption handling, and algorithmic regressions.
"""

import json
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Dict, List

import numpy as np
import pytest
import soundfile as sf

from contracts import EDIT_PLAN_SCHEMA_VERSION, EditPlan, PlanSegment, RationalTime
from media_time import extract_source_record
from movie_ingest import SegmentRecord, run_voice_checks
from diarization import SpeakerDiarizer
from speaker_registry import SpeakerRegistry


class _MockEmbedder:
    def __init__(self, vectors: Dict[float, np.ndarray]):
        self._vectors = vectors

    def embed(self, start: float, end: float) -> np.ndarray:
        return self._vectors[round(float(start), 3)]

    def save(self) -> None:
        pass


# ==============================================================================
# FIX P4.1 TESTS: RationalTime & SourceRecord Edge Cases
# ==============================================================================

def test_rational_time_zero_edge_cases():
    """Verify RationalTime(0, D) arithmetic, boundary behavior, and validation."""
    # Zero numerator with various positive denominators
    for denom in (1, 1000, 1001, 48000, 1000000):
        rt = RationalTime(0, denom)
        rt.validate()
        assert rt.as_float() == 0.0
        assert rt.as_fraction() == Fraction(0, 1)
        d = rt.to_dict()
        assert d == {"numerator": 0, "denominator": denom}
        # Roundtrip via dict unpacking
        rt_rt = RationalTime(**d)
        assert rt_rt.numerator == 0 and rt_rt.denominator == denom

    # Invalid zero denominator
    with pytest.raises(ValueError, match="denominator must be positive"):
        RationalTime(0, 0).validate()

    # Invalid negative denominator
    with pytest.raises(ValueError, match="denominator must be positive"):
        RationalTime(0, -1000).validate()

    # Invalid negative numerator
    with pytest.raises(ValueError, match="numerator cannot be negative"):
        RationalTime(-1, 1000).validate()
    with pytest.raises(ValueError, match="numerator cannot be negative"):
        RationalTime(-100, 1).validate()

    # Parsing 0.0 from string and number
    rt_str_frac = RationalTime.from_str("0/1000")
    assert rt_str_frac.numerator == 0 and rt_str_frac.denominator == 1000

    rt_str_zero = RationalTime.from_str("0")
    assert rt_str_zero.as_float() == 0.0

    rt_str_float = RationalTime.from_str("0.0")
    assert rt_str_float.as_float() == 0.0

    rt_num_int = RationalTime.from_number(0)
    assert rt_num_int.as_float() == 0.0

    rt_num_float = RationalTime.from_number(0.0)
    assert rt_num_float.as_float() == 0.0

    rt_num_frac = RationalTime.from_number(Fraction(0, 1))
    assert rt_num_frac.as_float() == 0.0


def test_edit_plan_frame_rate_must_be_strictly_positive():
    """EditPlan.video_fps must reject 0 FPS even though timestamps allow 0."""
    fps_zero = RationalTime(0, 1)
    seg = PlanSegment(
        segment_index=0,
        source_content_id="src_1",
        speaker_id="spk_1",
        source_start_sample=0,
        source_end_sample=24000,
        output_start_sample=0,
        output_end_sample=24000,
        target_text="test",
        target_phones=["T", "EH1", "S", "T"],
        span_type="word",
        source_video_path="input/test.mp4",
    )
    plan = EditPlan(
        schema_version=EDIT_PLAN_SCHEMA_VERSION,
        plan_id="plan_zero_fps",
        target_text="test",
        speaker_id="spk_1",
        project_sample_rate=48000,
        video_fps=fps_zero,
        total_output_samples=24000,
        segments=[seg],
    )
    with pytest.raises(ValueError, match="video_fps numerator must be positive"):
        plan.validate()


def test_container_probe_missing_and_malformed_fields(tmp_path, monkeypatch):
    """Stress test media_time.extract_source_record under various probe edge cases."""
    import media_time
    import subprocess

    dummy_media = tmp_path / "dummy.mp4"
    dummy_media.write_bytes(b"dummy")

    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)

    # Case A: avg_frame_rate is "0/0", r_frame_rate is valid "25/1"
    probe_a = {
        "streams": [
            {
                "codec_type": "video",
                "index": 0,
                "r_frame_rate": "25/1",
                "avg_frame_rate": "0/0",
            },
            {
                "codec_type": "audio",
                "index": 1,
                "sample_rate": "48000",
                "channels": 2,
            },
        ],
        "format": {"duration": "10.0"},
    }
    monkeypatch.setattr(media_time, "probe_media_file", lambda p: probe_a)
    rec_a = extract_source_record(dummy_media, build_dir=tmp_path / "build_a")
    assert rec_a.video_fps["numerator"] == 25
    assert rec_a.video_fps["denominator"] == 1
    assert rec_a.codec_delay_samples == 0
    assert rec_a.decoded_video_origin_sec == 0.0

    # Case B: avg_frame_rate missing completely, r_frame_rate missing, fallback to 30000/1001
    probe_b = {
        "streams": [
            {
                "codec_type": "video",
                "index": 0,
            },
            {
                "codec_type": "audio",
                "index": 1,
                "sample_rate": "44100",
                "channels": 1,
                "initial_padding": "invalid_string",
            },
        ],
        "format": {},
    }
    monkeypatch.setattr(media_time, "probe_media_file", lambda p: probe_b)
    rec_b = extract_source_record(dummy_media, build_dir=tmp_path / "build_b")
    assert rec_b.video_fps["numerator"] == 30000
    assert rec_b.video_fps["denominator"] == 1001
    assert rec_b.codec_delay_samples == 0

    # Case C: audio-only file (no video stream at all)
    probe_c = {
        "streams": [
            {
                "codec_type": "audio",
                "index": 0,
                "sample_rate": "48000",
                "channels": 2,
                "start_time": "0.150",
                "start_pad": "256",
            },
        ],
        "format": {"duration": "5.0"},
    }
    monkeypatch.setattr(media_time, "probe_media_file", lambda p: probe_c)
    rec_c = extract_source_record(dummy_media, build_dir=tmp_path / "build_c")
    assert rec_c.video_fps["numerator"] == 30000
    assert rec_c.video_fps["denominator"] == 1001
    assert rec_c.codec_delay_samples == 256
    assert rec_c.decoded_audio_origin_sec == 0.150
    assert rec_c.decoded_video_origin_sec == 0.0


# ==============================================================================
# FIX P4.2 TESTS: Movie Voice Checks for Minor Characters
# ==============================================================================

def test_movie_voice_checks_minor_character_variations():
    """Test minor characters with 1, 2, 3, 4 lines against major centroids."""
    rng = np.random.default_rng(123)
    # Define major characters with orthogonal unit vectors
    centers = {
        "emperor": np.array([1.0, 0.0, 0.0, 0.0] + [0.0]*12),
        "vader":   np.array([0.0, 1.0, 0.0, 0.0] + [0.0]*12),
        "luke":    np.array([0.0, 0.0, 1.0, 0.0] + [0.0]*12),
    }

    records: List[SegmentRecord] = []
    vectors: Dict[float, np.ndarray] = {}
    t = 0.0

    def add_line(label: str, vec: np.ndarray) -> SegmentRecord:
        nonlocal t
        v = vec / np.linalg.norm(vec)
        r = SegmentRecord(
            key=f"k_{t:.3f}",
            chunk_id="c",
            seg_index=0,
            cue_start=t,
            start=round(t, 3),
            end=round(t + 1.0, 3),
            label=label,
            label_source="subtitle_tag",
            text_hash="thash",
        )
        r.speaker, r.status, r.eligible = label, "label", True
        vectors[round(t, 3)] = v
        records.append(r)
        t += 2.0
        return r

    # Create major centroids with 15 lines each
    for spk in ("emperor", "vader", "luke"):
        for _ in range(15):
            noise = 0.03 * rng.normal(size=16)
            add_line(spk, centers[spk] + noise)

    # Sub-case 1: Minor character 'officer' with 1 line, strongly matching 'vader'
    # High similarity (>0.90), high margin (>0.80) -> should be voice_mismatch!
    off_line = add_line("officer", centers["vader"] + 0.02 * rng.normal(size=16))

    # Sub-case 2: Minor character 'trooper' with 3 lines, ALL strongly matching 'vader'
    tr_lines = [add_line("trooper", centers["vader"] + 0.02 * rng.normal(size=16)) for _ in range(3)]

    # Sub-case 3: Minor character 'pilot' with 3 lines: 2 matching 'luke', but 1 is distinct/noisy
    # Thin evidence safeguard: not all lines match the same centroid with high confidence
    pi_lines = [
        add_line("pilot", centers["luke"] + 0.02 * rng.normal(size=16)),
        add_line("pilot", centers["luke"] + 0.02 * rng.normal(size=16)),
        add_line("pilot", np.array([0.0, 0.0, 0.0, 1.0] + [0.0]*12)),  # Distinct 4th dimension
    ]

    # Sub-case 4: Minor character 'droid' with 1 line having a distinct acoustic voice (orthogonal)
    # Cosine similarity to all major centroids is ~0.0 (< 0.50)
    dr_line = add_line("droid", np.array([0.0, 0.0, 0.0, 1.0] + [0.0]*12))

    embedder = _MockEmbedder(vectors)
    _stats = run_voice_checks(records, embedder, log=lambda m: None)

    # 1. Single line officer matching vader is flagged as voice_mismatch
    assert off_line.voice_best == "vader"
    assert off_line.status == "voice_mismatch"
    assert not off_line.eligible

    # 2. Trooper (3 lines all matching vader) are flagged as voice_mismatch
    for tr in tr_lines:
        assert tr.voice_best == "vader"
        assert tr.status == "voice_mismatch"
        assert not tr.eligible

    # 3. Pilot (mixed lines) must NOT be marked voice_mismatch (protected from overrule)
    for pi in pi_lines:
        assert pi.status != "voice_mismatch"
        assert pi.eligible

    # 4. Droid (distinct voice, sim < 0.50) must NOT be marked voice_mismatch
    assert dr_line.status != "voice_mismatch"
    assert dr_line.eligible


def test_movie_voice_checks_empty_or_single_centroid():
    """Verify movie voice check resilience when 0 or 1 major centroids exist."""
    rng = np.random.default_rng(42)
    records: List[SegmentRecord] = []
    vectors: Dict[float, np.ndarray] = {}

    # Case A: Zero major centroids (all speakers have < 5 lines)
    r1 = SegmentRecord(
        key="k1", chunk_id="c", seg_index=0, cue_start=0.0, start=0.0, end=1.0,
        label="spk1", label_source="sub", text_hash="h"
    )
    r1.speaker, r1.status, r1.eligible = "spk1", "label", True
    records.append(r1)
    vectors[0.0] = rng.normal(size=16)

    embedder = _MockEmbedder(vectors)
    stats_a = run_voice_checks(records, embedder, log=lambda m: None)
    assert stats_a["propagation_margin_threshold"] is None
    assert r1.status == "label"
    assert r1.eligible

    # Case B: Exactly 1 major centroid (only 1 character has >= 5 lines)
    records.clear()
    vectors.clear()
    c_solo = np.array([1.0] + [0.0]*15)
    for i in range(10):
        t = float(i)
        r = SegmentRecord(
            key=f"solo_{t}", chunk_id="c", seg_index=0, cue_start=t, start=t, end=t+1.5,
            label="solo", label_source="sub", text_hash="h"
        )
        r.speaker, r.status, r.eligible = "solo", "label", True
        records.append(r)
        vectors[t] = c_solo + 0.01 * rng.normal(size=16)

    # Minor character with 1 line matching solo
    r_minor = SegmentRecord(
        key="minor", chunk_id="c", seg_index=0, cue_start=20.0, start=20.0, end=21.5,
        label="minor", label_source="sub", text_hash="h"
    )
    r_minor.speaker, r_minor.status, r_minor.eligible = "minor", "label", True
    records.append(r_minor)
    vectors[20.0] = c_solo + 0.01 * rng.normal(size=16)

    _stats_b = run_voice_checks(records, _MockEmbedder(vectors), log=lambda m: None)
    # With only 1 centroid, ranked has len == 1, second == -1.0, margin = best_sim - (-1.0)
    assert r_minor.voice_best == "solo"
    assert r_minor.voice_margin > 1.0


# ==============================================================================
# FIX P4.3 TESTS: Multi-Channel Audio Slicing, Resampling & Corrupted Inputs
# ==============================================================================

def test_soundfile_slicing_multi_channel_and_resampling(tmp_path, monkeypatch):
    """Stress test vtt_preprocessor soundfile slicing across various channel counts and sample rates."""
    from vtt_preprocessor import preprocess_source

    test_rates = [48000, 44100, 22050, 96000]
    channel_counts = [1, 2, 6]  # Mono, stereo, 5.1 surround

    for sr in test_rates:
        for ch in channel_counts:
            dur_sec = 3.0
            num_samples = int(dur_sec * sr)
            # Generate distinguishable multichannel tone/ramp
            if ch == 1:
                sig = np.linspace(-0.5, 0.5, num_samples, dtype=np.float32)
            else:
                sig = np.zeros((num_samples, ch), dtype=np.float32)
                for c in range(ch):
                    sig[:, c] = np.linspace(-0.5 * (c + 1), 0.5 * (c + 1), num_samples, dtype=np.float32)

            audio_file = tmp_path / f"test_{sr}_{ch}.wav"
            sf.write(str(audio_file), sig, sr, subtype="PCM_16")

            vtt_content = """WEBVTT

00:00:00.500 --> 00:00:02.000
Multichannel cue
"""
            vtt_file = tmp_path / f"sub_{sr}_{ch}.vtt"
            vtt_file.write_text(vtt_content, encoding="utf-8")

            # Monkeypatch subprocess to verify FFmpeg was NOT called
            def fail_ffmpeg(*args, **kwargs):
                raise AssertionError(f"FFmpeg called unexpectedly for sr={sr}, ch={ch}")
            monkeypatch.setattr(subprocess, "run", fail_ffmpeg)

            out_corpus = tmp_path / f"corpus_{sr}_{ch}"
            manifest_str = preprocess_source(
                video_path=str(audio_file),
                vtt_path=str(vtt_file),
                output_corpus_dir=str(out_corpus),
            )
            manifest = json.loads(Path(manifest_str).read_text(encoding="utf-8"))
            assert manifest["source_video"] == str(audio_file)
            chunk_wav = out_corpus / "chunk_0000.wav"
            assert chunk_wav.exists()

            data, read_sr = sf.read(str(chunk_wav))
            assert read_sr == 16000
            # Sliced duration is 1.5 seconds -> 24000 samples (+/- 1 sample due to polyphase resampling)
            assert abs(len(data) - 24000) <= 2
            assert data.ndim == 1  # Successfully downmixed to mono


def test_soundfile_slicing_corrupted_audio_fallback(tmp_path, monkeypatch):
    """Verify that preprocess_source falls back to FFmpeg if audio file is corrupted or unreadable by SoundFile."""
    from vtt_preprocessor import preprocess_source

    corrupt_audio = tmp_path / "corrupted.wav"
    corrupt_audio.write_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00garbage_data")

    vtt_content = """WEBVTT

00:00:00.000 --> 00:00:01.000
Fallback cue
"""
    vtt_file = tmp_path / "sub.vtt"
    vtt_file.write_text(vtt_content, encoding="utf-8")

    ffmpeg_called = False

    def mock_ffmpeg(cmd, *args, **kwargs):
        nonlocal ffmpeg_called
        ffmpeg_called = True
        # Create a dummy output wav
        out_wav = Path(cmd[-1])
        sf.write(str(out_wav), np.zeros(16000, dtype=np.int16), 16000, subtype="PCM_16")
        class MockRes:
            returncode = 0
        return MockRes()

    monkeypatch.setattr(subprocess, "run", mock_ffmpeg)

    out_corpus = tmp_path / "corpus_corrupt"
    preprocess_source(
        video_path=str(corrupt_audio),
        vtt_path=str(vtt_file),
        output_corpus_dir=str(out_corpus),
    )
    assert ffmpeg_called, "FFmpeg fallback was not invoked on corrupted audio file!"


def test_diarization_padding_edge_limits(tmp_path):
    """Stress test SpeakerDiarizer hangover padding right edge limits."""
    sr = 16000
    # 10s audio
    audio = np.zeros(10 * sr, dtype=np.float32)
    wav_path = tmp_path / "padding_test.wav"
    sf.write(str(wav_path), audio, sr)

    reg = SpeakerRegistry(config_path=tmp_path / "reg.json")
    diarizer = SpeakerDiarizer(registry=reg)

    # Candidate turns with tiny gap (0.05s) and large gap (4.0s)
    # Gap 1: end=1.0, next_start=1.05 -> gap is 0.05s - 0.04s = 0.01s padding
    # Gap 2: end=2.0, next_start=6.0 -> gap is 4.0s - 0.04s = 3.96s -> capped at 0.15s
    intervals = diarizer.diarize(
        audio_path=wav_path,
        candidate_turns=[(0.5, 1.0), (1.05, 2.0), (6.0, 7.0)],
        interactive=False,
    )
    assert len(intervals) == 3

    # Turn 0: padding should be min(0.15, 0.5 * max(0.0, 1.05 - 1.0 - 0.04)) = 0.005s -> end = 1.005
    assert abs(intervals[0].end_sec - 1.005) < 1e-3

    # Turn 1: padding should be min(0.15, max(0.0, 6.0 - 2.0 - 0.04)) = 0.15s -> end = 2.15
    assert abs(intervals[1].end_sec - 2.15) < 1e-3

    # Turn 2: gap to end of file (10.0s) is 3.0s - 0.04s -> capped at 0.15s -> end = 7.15
    assert abs(intervals[2].end_sec - 7.15) < 1e-3
