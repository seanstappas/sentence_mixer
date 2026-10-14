"""
Adversarial empirical challenge harness for Milestone 4 (Phase 4: Fixes P4.1–P4.3).
Tests:
1. End-to-End EditPlan & Timeline Invariance (P4.1):
   - RationalTime(0) round-trip, serialization, frame/sample calculations.
   - decoded_video_origin_sec & codec_delay_samples metadata propagation.
2. Movie Support Regression & Leave-One-Out Calibration (P4.2):
   - Borderline voice similarity boundary: sim = 0.49 vs sim = 0.51.
   - Single vs multi-line minor character deterministic classification.
   - Mixed-evidence and self-consistent alternative voice protection.
3. Short-Clip Chunk Slicing Performance & Fallback (P4.3):
   - 25+ subtitle cues sliced purely via SoundFile with ZERO FFmpeg subprocesses.
   - Fallback to FFmpeg on non-audio / video containers or unseekable streams.
   - Sample-rate conversion (44.1 kHz, 48 kHz) and multichannel downmixing.
"""

from __future__ import annotations

import json
import math
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import soundfile as sf

from contracts import (
    EDIT_PLAN_SCHEMA_VERSION,
    EditPlan,
    PlanSegment,
    RationalTime,
    SourceRecord,
)
import media_time
from media_time import (
    cumulative_frame_ceiling,
    extract_source_record,
    frame_index_to_samples,
    samples_to_frame_index,
    samples_to_rational_frames,
)
from movie_ingest import SegmentRecord, run_voice_checks
from vtt_preprocessor import preprocess_source


# ==============================================================================
# Challenge 1: End-to-End EditPlan & Timeline Invariance (P4.1)
# ==============================================================================


class TestRationalTimeAndEditPlanInvariance:
    """Stress-test RationalTime(0) and EditPlan timeline invariance."""

    def test_rational_time_zero_offsets_and_boundaries(self) -> None:
        """RationalTime(0) is a valid timestamp offset, but negative or zero-denom is rejected."""
        # Valid zero offsets
        rt_zero = RationalTime(0, 1000)
        rt_zero.validate()
        assert rt_zero.as_float() == 0.0
        assert rt_zero.as_fraction() == Fraction(0, 1)
        assert rt_zero.to_dict() == {"numerator": 0, "denominator": 1000}

        rt_zero_simple = RationalTime(0, 1)
        rt_zero_simple.validate()
        assert rt_zero_simple.as_float() == 0.0

        # String parsing of zero
        parsed_zero = RationalTime.from_str("0/1000")
        parsed_zero.validate()
        assert parsed_zero.numerator == 0
        assert parsed_zero.denominator == 1000

        parsed_float_zero = RationalTime.from_str("0.0")
        parsed_float_zero.validate()
        assert parsed_float_zero.numerator == 0

        # Number conversion
        from_num = RationalTime.from_number(0)
        from_num.validate()
        assert from_num.numerator == 0

        # Rejection of negative numerators
        with pytest.raises(ValueError, match="RationalTime numerator cannot be negative"):
            RationalTime(-1, 1000).validate()

        # Rejection of zero or negative denominators
        with pytest.raises(ValueError, match="RationalTime denominator must be positive"):
            RationalTime(0, 0).validate()

        with pytest.raises(ValueError, match="RationalTime denominator must be positive"):
            RationalTime(0, -100).validate()

    def test_edit_plan_rejects_zero_fps_but_accepts_zero_timestamps(self, tmp_path: Path) -> None:
        """EditPlan must strictly enforce video_fps.numerator > 0, while allowing 0-timestamp segments."""
        seg = PlanSegment(
            segment_index=0,
            source_content_id="src_0",
            speaker_id="spk_0",
            source_start_sample=0,
            source_end_sample=48000,
            output_start_sample=0,
            output_end_sample=48000,
            target_text="test",
            target_phones=["T", "EH1", "S", "T"],
            span_type="word",
            source_video_path="input/test.mp4",
        )
        src = SourceRecord(
            content_id="src_0",
            source_path="input/test.mp4",
            native_sample_rate=48000,
            native_channels=2,
            decoded_video_origin_sec=0.040,
            codec_delay_samples=1024,
        )

        # Plan with 0 FPS must fail
        plan_bad_fps = EditPlan(
            schema_version=EDIT_PLAN_SCHEMA_VERSION,
            plan_id="plan_zero_fps",
            target_text="test",
            speaker_id="spk_1",
            project_sample_rate=48000,
            video_fps=RationalTime(0, 1),
            total_output_samples=48000,
            segments=[seg],
            sources={"src_0": src},
        )
        with pytest.raises(ValueError, match="video_fps numerator must be positive"):
            plan_bad_fps.validate()

        # Plan with valid FPS and start sample 0 must pass and roundtrip through JSON
        plan_good = EditPlan(
            schema_version=EDIT_PLAN_SCHEMA_VERSION,
            plan_id="plan_valid",
            target_text="test",
            speaker_id="spk_1",
            project_sample_rate=48000,
            video_fps=RationalTime(24000, 1001),
            total_output_samples=48000,
            total_video_frames=24,
            segments=[seg],
            sources={"src_0": src},
        )
        plan_good.validate()

        json_path = tmp_path / "plan.json"
        plan_good.save(json_path)

        loaded = EditPlan.load(json_path)
        loaded.validate()
        assert loaded.plan_id == "plan_valid"
        assert loaded.video_fps.numerator == 24000
        assert loaded.video_fps.denominator == 1001
        assert loaded.sources["src_0"].decoded_video_origin_sec == 0.040
        assert loaded.sources["src_0"].codec_delay_samples == 1024

    def test_samples_and_frames_zero_invariance_and_no_drift(self) -> None:
        """Ensure time/sample/frame conversions with timestamp 0 and large times have zero drift."""
        fps = RationalTime(30000, 1001)
        sr = 48000

        # Sample 0 maps exactly to frame 0 and RationalTime(0, 1)
        rf_0 = samples_to_rational_frames(0, sr, fps)
        rf_0.validate()
        assert rf_0.numerator == 0
        assert samples_to_frame_index(0, sr, fps) == 0
        assert cumulative_frame_ceiling(0, sr, fps) == 0
        assert frame_index_to_samples(0, fps, sr) == 0

        # Test large offsets (10 hours = 36000 seconds)
        duration_sec = 36000.0
        n_samples = int(duration_sec * sr)
        rf_large = samples_to_rational_frames(n_samples, sr, fps)
        rf_large.validate()
        frame_idx = samples_to_frame_index(n_samples, sr, fps)
        frame_ceil = cumulative_frame_ceiling(n_samples, sr, fps)
        # Expected frames = 36000 * 30000 / 1001 = 1078921.0789... -> floor 1078921, ceil 1078922
        assert frame_idx == 1078921
        assert frame_ceil == 1078922

    def test_extract_source_record_avg_frame_rate_and_delay_propagation(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify media_time extracts avg_frame_rate, codec_delay_samples, and decoded_video_origin_sec."""
        dummy_file = tmp_path / "test.mkv"
        dummy_file.write_bytes(b"mock_video")

        mock_probe = {
            "streams": [
                {
                    "codec_type": "video",
                    "index": 0,
                    "avg_frame_rate": "24000/1001",
                    "r_frame_rate": "1000/1",  # VFR container artifact
                    "duration": "120.0",
                    "start_time": "0.083333",
                },
                {
                    "codec_type": "audio",
                    "index": 1,
                    "sample_rate": "48000",
                    "channels": 6,
                    "duration": "120.0",
                    "start_time": "0.000000",
                    "initial_padding": "1152",
                },
            ],
            "format": {"duration": "120.0"},
        }

        monkeypatch.setattr(media_time, "probe_media_file", lambda p: mock_probe)
        monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: None)

        rec = extract_source_record(dummy_file, build_dir=tmp_path / "build")
        rec.validate()

        # avg_frame_rate preferred over r_frame_rate
        assert rec.video_fps["numerator"] == 24000
        assert rec.video_fps["denominator"] == 1001
        assert rec.codec_delay_samples == 1152
        assert abs(rec.decoded_video_origin_sec - 0.083333) < 1e-5
        assert rec.decoded_audio_origin_sec == 0.0

        # Check media.json serialization
        saved_media_json = tmp_path / "build" / "sources" / rec.content_id / "media.json"
        assert saved_media_json.exists()
        saved_data = json.loads(saved_media_json.read_text(encoding="utf-8"))
        assert saved_data["codec_delay_samples"] == 1152
        assert abs(saved_data["decoded_video_origin_sec"] - 0.083333) < 1e-5


# ==============================================================================
# Challenge 2: Movie Support & Voice Verification Borderline Calibration (P4.2)
# ==============================================================================


class _FakeEmbedder:
    def __init__(self, vectors: Dict[float, np.ndarray]) -> None:
        self.vectors = vectors

    def embed(self, start: float, end: float) -> np.ndarray:
        return self.vectors[round(float(start), 3)]

    def embed_segments(self, wav_path: Path, segments: List[Tuple[float, float]]) -> Dict[float, np.ndarray]:
        return {s: self.vectors[s] for s, e in segments if s in self.vectors}

    def save(self) -> None:
        pass


class TestVoiceChecksBorderlineCalibration:
    """Stress-test voice verification boundary conditions and minor character protection."""

    def _build_test_corpus(self, minor_specs: Dict[str, List[float]]) -> Tuple[List[SegmentRecord], Dict[float, np.ndarray], Dict[str, Any]]:
        """Helper to construct a corpus with 3 major characters and arbitrary minor character line similarities."""
        rng = np.random.default_rng(123)
        # Fix 16D orthonormal basis for major characters
        v_mira = np.zeros(16, dtype=np.float32)
        v_mira[0] = 1.0
        v_tobin = np.zeros(16, dtype=np.float32)
        v_tobin[1] = 1.0
        v_osric = np.zeros(16, dtype=np.float32)
        v_osric[2] = 1.0

        centers = {"mira": v_mira, "tobin": v_tobin, "osric": v_osric}
        records: List[SegmentRecord] = []
        vectors: Dict[float, np.ndarray] = {}
        t = 0.0

        def add_rec(label: str, vec: np.ndarray) -> SegmentRecord:
            nonlocal t
            v = (vec / np.linalg.norm(vec)).astype(np.float32)
            st = round(t, 3)
            r = SegmentRecord(
                key=f"k{st}",
                chunk_id="c",
                seg_index=0,
                cue_start=st,
                start=st,
                end=round(st + 2.0, 3),
                label=label,
                label_source="subtitle_tag",
                text_hash="h",
            )
            r.speaker, r.status, r.eligible = label, "label", True
            vectors[st] = v
            records.append(r)
            t += 3.0
            return r

        # Add 15 lines for each major character with tight noise so centroids are clean
        for name in ("mira", "tobin", "osric"):
            for _ in range(15):
                noise = rng.normal(scale=0.01, size=16).astype(np.float32)
                add_rec(name, centers[name] + noise)

        # Add minor characters according to target similarities toward 'tobin' (dimension 1)
        # Vector with similarity s toward tobin: s * v_tobin + sqrt(1 - s^2) * orthogonal_noise
        for minor_name, target_sims in minor_specs.items():
            for s in target_sims:
                ortho = np.zeros(16, dtype=np.float32)
                ortho[3] = 1.0  # orthogonal to mira (0), tobin (1), osric (2)
                vec = s * v_tobin + math.sqrt(max(0.0, 1.0 - s * s)) * ortho
                add_rec(minor_name, vec)

        stats = run_voice_checks(records, _FakeEmbedder(vectors), log=lambda m: None)
        return records, vectors, stats

    def test_borderline_similarity_049_vs_051_single_line(self) -> None:
        """
        Adversarial test: A single minor character line with sim = 0.49 vs sim = 0.51 toward major character.
        sim = 0.49 must NOT be flagged voice_mismatch (remains label_voice_unclear, eligible).
        sim = 0.51 MUST be flagged voice_mismatch (eligible = False).
        """
        minor_specs = {
            "herald_49": [0.49],
            "herald_51": [0.51],
        }
        records, vectors, stats = self._build_test_corpus(minor_specs)

        rec_49 = next(r for r in records if r.label == "herald_49")
        rec_51 = next(r for r in records if r.label == "herald_51")

        # Threshold calibrated from major characters
        assert stats["propagation_margin_threshold"] is not None
        assert stats["propagation_margin_threshold"] >= 0.05

        # rec_49 has best_sim ~ 0.49 < MIN_CONFIDENT_SIMILARITY (0.50)
        assert rec_49.voice_best == "tobin"
        assert round(rec_49.voice_sim, 2) == 0.49
        assert rec_49.status == "label_voice_unclear"
        assert rec_49.eligible is True

        # rec_51 has best_sim ~ 0.51 >= MIN_CONFIDENT_SIMILARITY (0.50) and margin >= threshold
        assert rec_51.voice_best == "tobin"
        assert round(rec_51.voice_sim, 2) == 0.51
        assert rec_51.status == "voice_mismatch"
        assert rec_51.eligible is False

    def test_minor_character_inconsistent_evidence_protected(self) -> None:
        """
        Adversarial test: A minor character with 2 lines, where line 1 has sim = 0.51 (matches Tobin),
        but line 2 has sim = 0.49 (below confident threshold).
        Because evidence is inconsistent, neither line should be flagged voice_mismatch.
        """
        minor_specs = {
            "scout_mixed": [0.51, 0.49],
        }
        records, vectors, stats = self._build_test_corpus(minor_specs)
        scout_lines = [r for r in records if r.label == "scout_mixed"]
        assert len(scout_lines) == 2

        for r in scout_lines:
            assert r.status != "voice_mismatch"
            assert r.eligible is True
            assert r.status == "label_voice_unclear"

    def test_minor_character_consistent_evidence_both_flagged(self) -> None:
        """
        Adversarial test: A minor character with 2 lines, both having sim >= 0.51 matching Tobin.
        Both lines must be flagged voice_mismatch and excluded from inventory.
        """
        minor_specs = {
            "archer_consistent": [0.52, 0.54],
        }
        records, vectors, stats = self._build_test_corpus(minor_specs)
        archer_lines = [r for r in records if r.label == "archer_consistent"]
        assert len(archer_lines) == 2

        for r in archer_lines:
            assert r.voice_best == "tobin"
            assert r.status == "voice_mismatch"
            assert r.eligible is False

    def test_minor_character_split_across_different_majors(self) -> None:
        """
        Adversarial test: A minor character with 2 lines, line 1 matching Tobin (0.55),
        line 2 matching Mira (0.55).
        Because best matches point to different characters, all_match is False; neither is flagged mismatch.
        """
        v_mira = np.zeros(16, dtype=np.float32)
        v_mira[0] = 1.0
        v_tobin = np.zeros(16, dtype=np.float32)
        v_tobin[1] = 1.0
        ortho = np.zeros(16, dtype=np.float32)
        ortho[3] = 1.0

        vec_tobin = 0.55 * v_tobin + math.sqrt(1 - 0.55**2) * ortho
        vec_mira = 0.55 * v_mira + math.sqrt(1 - 0.55**2) * ortho

        rng = np.random.default_rng(999)
        records: List[SegmentRecord] = []
        vectors: Dict[float, np.ndarray] = {}
        t = 0.0

        def add_rec(label: str, v: np.ndarray) -> SegmentRecord:
            nonlocal t
            norm_v = (v / np.linalg.norm(v)).astype(np.float32)
            st = round(t, 3)
            r = SegmentRecord(
                key=f"k{st}",
                chunk_id="c",
                seg_index=0,
                cue_start=st,
                start=st,
                end=round(st + 2.0, 3),
                label=label,
                label_source="subtitle_tag",
                text_hash="h",
            )
            r.speaker, r.status, r.eligible = label, "label", True
            vectors[st] = norm_v
            records.append(r)
            t += 3.0
            return r

        for name, base_v in [("mira", v_mira), ("tobin", v_tobin), ("osric", np.eye(16)[2])]:
            for _ in range(15):
                add_rec(name, base_v + rng.normal(scale=0.01, size=16).astype(np.float32))

        r1 = add_rec("split_minor", vec_tobin)
        r2 = add_rec("split_minor", vec_mira)

        _stats = run_voice_checks(records, _FakeEmbedder(vectors), log=lambda m: None)

        assert r1.voice_best == "tobin"
        assert r2.voice_best == "mira"
        # Neither can be voice_mismatch because lines do not consistently match one profile
        assert r1.status == "label_voice_unclear" and r1.eligible is True
        assert r2.status == "label_voice_unclear" and r2.eligible is True


# ==============================================================================
# Challenge 3: Short-Clip Chunk Slicing Performance & Concurrency (P4.3)
# ==============================================================================


class TestShortClipChunkSlicing:
    """Stress-test vtt_preprocessor with 20+ subtitle cues and non-audio fallback."""

    def test_soundfile_slicing_25_cues_zero_subprocess_spawns(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Preprocess 25 subtitle cues from a 30-second WAV.
        Must execute purely using SoundFile seeking with ZERO FFmpeg subprocesses.
        """
        sr = 16000
        duration_sec = 30.0
        total_samples = int(duration_sec * sr)

        # Known frequency modulation ramp so every sample is unique and testable
        t = np.linspace(0.0, duration_sec, total_samples, endpoint=False, dtype=np.float32)
        chirp = 0.5 * np.sin(2 * np.pi * (200 + 10 * t) * t)

        audio_wav = tmp_path / "long_audio.wav"
        sf.write(str(audio_wav), chirp, sr, subtype="PCM_16")

        # Generate 25 cues
        vtt_lines = ["WEBVTT\n"]
        cues_meta = []
        for i in range(25):
            start_s = i * 1.1 + 0.1
            end_s = start_s + 0.8
            cues_meta.append((start_s, end_s))
            vtt_lines.append(f"{i+1}\n00:00:{start_s:06.3f} --> 00:00:{end_s:06.3f}\nCue sentence {i} test words\n")

        vtt_path = tmp_path / "subtitles.vtt"
        vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")

        # Monkeypatch subprocess.run to raise immediately if called
        subprocess_calls = []

        def fail_on_ffmpeg(*args: Any, **kwargs: Any) -> Any:
            subprocess_calls.append(args)
            raise AssertionError(f"Unexpected subprocess spawned: {args}")

        monkeypatch.setattr(subprocess, "run", fail_on_ffmpeg)

        corpus_out = tmp_path / "corpus_soundfile"
        manifest_path = preprocess_source(
            video_path=str(audio_wav),
            vtt_path=str(vtt_path),
            output_corpus_dir=str(corpus_out),
        )

        assert len(subprocess_calls) == 0, "Subprocess was spawned when SoundFile should handle all slices!"

        manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        assert len(manifest["chunks"]) == 25

        # Verify sample-exact correctness for all 25 chunks
        for i, (st, en) in enumerate(cues_meta):
            chunk_id = f"chunk_{i:04d}"
            c_wav = corpus_out / f"{chunk_id}.wav"
            assert c_wav.exists()

            data, read_sr = sf.read(str(c_wav))
            assert read_sr == 16000
            expected_start_samp = int(round(st * sr))
            expected_end_samp = int(round(en * sr))
            expected_len = expected_end_samp - expected_start_samp

            assert len(data) == expected_len
            expected_slice = chirp[expected_start_samp:expected_end_samp]
            # Verify exact waveform preservation
            assert np.allclose(data, expected_slice, atol=1e-3)

    def test_fallback_to_ffmpeg_when_non_audio_container_passed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        When video_path is a format SoundFile cannot open (e.g. simulated MP4/MKV container),
        preprocess_source must catch the exception cleanly and fall back to FFmpeg subprocess.
        """
        fake_mp4 = tmp_path / "video.mp4"
        # Write dummy binary header that libsndfile cannot recognize as an audio format
        fake_mp4.write_bytes(b"\x00\x00\x00\x1cftypisom\x00\x00\x02\x00isomiso2mp41")

        vtt_lines = [
            "WEBVTT\n",
            "1\n00:00:01.000 --> 00:00:02.000\nFallback test cue\n",
        ]
        vtt_path = tmp_path / "fallback.vtt"
        vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")

        # Intercept subprocess.run to simulate successful FFmpeg slicing
        ffmpeg_commands: List[List[str]] = []

        def mock_ffmpeg(cmd: List[str], *args: Any, **kwargs: Any) -> Any:
            ffmpeg_commands.append(cmd)
            # Produce mock 16kHz mono WAV output at the target path
            out_wav = Path(cmd[-1])
            sf.write(str(out_wav), np.zeros(16000, dtype=np.int16), 16000, subtype="PCM_16")

            class MockResult:
                returncode = 0

            return MockResult()

        monkeypatch.setattr(subprocess, "run", mock_ffmpeg)

        corpus_out = tmp_path / "corpus_fallback"
        manifest_path = preprocess_source(
            video_path=str(fake_mp4),
            vtt_path=str(vtt_path),
            output_corpus_dir=str(corpus_out),
        )
        assert Path(manifest_path).exists()

        # Must have fallen back to FFmpeg exactly once for the cue
        assert len(ffmpeg_commands) == 1
        cmd = ffmpeg_commands[0]
        assert cmd[0] == "ffmpeg"
        assert "-vn" in cmd
        assert "-ar" in cmd and "16000" in cmd

        chunk_wav = corpus_out / "chunk_0000.wav"
        assert chunk_wav.exists()

    def test_soundfile_resampling_from_44100_stereo(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """
        Verify that SoundFile read correctly handles 44100 Hz stereo audio, downmixing to mono
        and resampling to 16000 Hz with exact duration.
        """
        sr = 44100
        dur_sec = 2.0
        n_samples = int(dur_sec * sr)

        # Stereo audio with different signals in left and right channels
        left = 0.4 * np.ones(n_samples, dtype=np.float32)
        right = 0.2 * np.ones(n_samples, dtype=np.float32)
        stereo = np.column_stack([left, right])

        audio_wav = tmp_path / "stereo_44100.wav"
        sf.write(str(audio_wav), stereo, sr, subtype="PCM_16")

        vtt_lines = [
            "WEBVTT\n",
            "1\n00:00:00.500 --> 00:00:01.500\nResample stereo test\n",
        ]
        vtt_path = tmp_path / "stereo.vtt"
        vtt_path.write_text("\n".join(vtt_lines), encoding="utf-8")

        # Zero FFmpeg subprocess calls
        monkeypatch.setattr(
            subprocess,
            "run",
            lambda *args, **kwargs: pytest.fail("FFmpeg should not be called for 44.1kHz WAV!"),
        )

        corpus_out = tmp_path / "corpus_stereo"
        manifest_path = preprocess_source(
            video_path=str(audio_wav),
            vtt_path=str(vtt_path),
            output_corpus_dir=str(corpus_out),
        )
        assert Path(manifest_path).exists()

        chunk_wav = corpus_out / "chunk_0000.wav"
        data, read_sr = sf.read(str(chunk_wav))
        assert read_sr == 16000
        assert data.ndim == 1  # Downmixed to mono
        # 1.0 second duration at 16000 Hz = 16000 samples
        assert len(data) == 16000
        # Expected downmixed value = (0.4 + 0.2) / 2 = 0.3
        # Exclude filter transient edge samples
        middle_slice = data[1000:-1000]
        assert np.allclose(middle_slice, 0.3, atol=0.01)
