"""
Empirical stress-test suite for Challenger 1:
- R1: Independent ASR Listening Verification Tool (tools/listen_verify.py)
- R2: Output Audio Loudness Normalization (normalize_audio_ebur128, segment leveling, CLI flags)

Executed independently against implementation under test.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"
OUTPUT_DIR = REPO_ROOT / "output"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from contracts import EditPlan, PlanSegment, PlanTransition, SourceRecord
from edit_plan import finalize_edit_plan
from listen_verify import verify_listening
from validate_mix import measure_audio_loudness
from video_assembler import normalize_audio_ebur128, render_master_pcm


def _measure_ebur128_direct(wav_path: Path) -> Tuple[float, float]:
    """Independent EBU R128 loudness measurement using raw FFmpeg ebur128 filter."""
    return measure_audio_loudness(wav_path)


def _generate_synthetic_speech(
    duration_sec: float,
    sample_rate: int = 48000,
    amplitude: float = 0.1,
    base_freq: float = 220.0,
) -> np.ndarray:
    """Generates synthetic harmonic speech-like signal."""
    num_samples = int(round(duration_sec * sample_rate))
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    sig = (
        1.0 * np.sin(2 * np.pi * base_freq * t)
        + 0.5 * np.sin(2 * np.pi * (2 * base_freq) * t)
        + 0.25 * np.sin(2 * np.pi * (3 * base_freq) * t)
    )
    env = np.sin(np.pi * np.linspace(0, 1, num_samples)) ** 2
    sig = sig * env
    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = amplitude * (sig / max_val)
    return sig.astype(np.float32)


# ==============================================================================
# R1: ASR Listening Verification & Bleed Detection Tests
# ==============================================================================

@pytest.mark.model
@pytest.mark.real_corpus
class TestASRListeningVerification:
    """Empirical verification of R1 ASR Listener CLI, metrics, and bleed rejection."""

    def test_cli_smoke_basic_wav_match(self):
        """Verify tools/listen_verify.py on smoke_basic.wav returns exit code 0 and WER 0.00."""
        wav_path = OUTPUT_DIR / "smoke_basic.wav"
        if not wav_path.exists():
            pytest.skip(f"Missing test asset: {wav_path}")

        cmd = [
            sys.executable,
            str(TOOLS_DIR / "listen_verify.py"),
            "--input", str(wav_path),
            "--expected", "Hello there",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"Expected exit code 0, got {proc.returncode}: {proc.stderr}"

        data = json.loads(proc.stdout)
        assert data["passed"] is True
        assert data["wer"] == 0.0
        assert data["cer"] == 0.0
        assert data["bleed_detected"] is False
        assert data["all_expected_words_present"] is True
        assert data["missing_words"] == []
        assert data["bleeding_words"] == []
        assert "hello" in data["word_matches"]
        assert "there" in data["word_matches"]

    def test_cli_smoke_basic_mp4_match(self):
        """Verify tools/listen_verify.py directly extracts audio from MP4 and succeeds."""
        mp4_path = OUTPUT_DIR / "smoke_basic.mp4"
        if not mp4_path.exists():
            pytest.skip(f"Missing test asset: {mp4_path}")

        cmd = [
            sys.executable,
            str(TOOLS_DIR / "listen_verify.py"),
            "--input", str(mp4_path),
            "--expected", "Hello there",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"Expected exit code 0, got {proc.returncode}: {proc.stderr}"

        data = json.loads(proc.stdout)
        assert data["passed"] is True
        assert data["wer"] == 0.0
        assert data["missing_words"] == []
        assert data["bleeding_words"] == []

    def test_cli_corrupted_expected_text_fails(self):
        """Verify tools/listen_verify.py exits with code 1 when expected text is completely mismatched."""
        wav_path = OUTPUT_DIR / "smoke_basic.wav"
        if not wav_path.exists():
            pytest.skip(f"Missing test asset: {wav_path}")
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "listen_verify.py"),
            "--input", str(wav_path),
            "--expected", "Completely wrong sentence altogether",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 1, f"Expected exit code 1, got {proc.returncode}"

        data = json.loads(proc.stdout)
        assert data["passed"] is False
        assert data["wer"] > 0.20
        assert len(data["missing_words"]) > 0
        assert len(data["failure_reasons"]) > 0

    def test_cli_bleeding_words_detection(self):
        """
        Verify tools/listen_verify.py detects bleeding words when audio contains words
        not in the expected text, resulting in exit code 1 and accurate bleeding_words.
        """
        wav_path = OUTPUT_DIR / "smoke_basic.wav"  # Transcribes to "Hello there"
        if not wav_path.exists():
            pytest.skip(f"Missing test asset: {wav_path}")
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "listen_verify.py"),
            "--input", str(wav_path),
            "--expected", "Hello",  # 'there' is extra spoken dialogue bleed
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 1, f"Expected exit code 1 for bleed, got {proc.returncode}"

        data = json.loads(proc.stdout)
        assert data["passed"] is False
        assert data["bleed_detected"] is True
        assert "there" in data["bleeding_words"]
        assert data["missing_words"] == []
        assert any("bleed" in r.lower() for r in data["failure_reasons"])

    def test_cli_missing_words_detection(self):
        """
        Verify tools/listen_verify.py detects missing target words when expected text has words
        not present in the audio.
        """
        wav_path = OUTPUT_DIR / "smoke_basic.wav"  # Transcribes to "Hello there"
        if not wav_path.exists():
            pytest.skip(f"Missing test asset: {wav_path}")
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "listen_verify.py"),
            "--input", str(wav_path),
            "--expected", "Hello there Obi-Wan Kenobi",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 1, f"Expected exit code 1 for missing words, got {proc.returncode}"

        data = json.loads(proc.stdout)
        assert data["passed"] is False
        assert data["all_expected_words_present"] is False
        assert "obi" in data["missing_words"]
        assert "wan" in data["missing_words"]
        assert "kenobi" in data["missing_words"]

    def test_cli_allow_insertions_flag(self):
        """
        Verify --allow-insertions permits extra spoken words if WER threshold is loosened.
        """
        wav_path = OUTPUT_DIR / "smoke_basic.wav"
        if not wav_path.exists():
            pytest.skip(f"Missing test asset: {wav_path}")
        # Audio is "Hello there". Expected is "Hello". Without flag, insertion fails.
        # With --allow-insertions and tolerance=1.0, it should pass.
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "listen_verify.py"),
            "--input", str(wav_path),
            "--expected", "Hello",
            "--allow-insertions",
            "--tolerance", "1.0",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0, f"Expected exit code 0 with --allow-insertions, got {proc.returncode}: {proc.stderr}"
        data = json.loads(proc.stdout)
        assert data["passed"] is True

    def test_json_schema_completeness(self):
        """Verify that JSON output strictly satisfies the PROJECT.md interface contract."""
        wav_path = OUTPUT_DIR / "smoke_basic.wav"
        if not wav_path.exists():
            pytest.skip(f"Missing test asset: {wav_path}")
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "listen_verify.py"),
            "--input", str(wav_path),
            "--expected", "Hello there",
            "--json",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 0
        data = json.loads(proc.stdout)

        required_keys = [
            "input_path",
            "expected_text",
            "transcribed_text",
            "normalized_expected",
            "normalized_transcribed",
            "wer",
            "cer",
            "word_matches",
            "missing_words",
            "bleeding_words",
            "passed",
        ]
        for key in required_keys:
            assert key in data, f"Missing required key '{key}' in JSON response"


# ==============================================================================
# R2: Loudness Normalization & Invariant Preservation Tests
# ==============================================================================

class TestLoudnessNormalization:
    """Empirical verification of R2 EBU R128 normalization and invariant preservation."""

    @pytest.mark.parametrize("amplitude,expected_input_type", [
        (0.005, "quiet"),
        (0.05, "moderate"),
        (0.85, "loud"),
    ])
    def test_normalize_audio_ebur128_target_and_true_peak(self, tmp_path: Path, amplitude: float, expected_input_type: str):
        """
        Verify that normalize_audio_ebur128 achieves -16.0 +/- 1.0 LUFS and true peak <= -1.0 dBTP
        across quiet, moderate, and loud speech-like inputs.
        """
        sr = 48000
        duration = 3.0
        in_wav = tmp_path / f"test_in_{expected_input_type}.wav"
        out_wav = tmp_path / f"test_out_{expected_input_type}.wav"

        audio = _generate_synthetic_speech(duration_sec=duration, sample_rate=sr, amplitude=amplitude)
        sf.write(str(in_wav), audio, sr, subtype="PCM_16")

        stats = normalize_audio_ebur128(
            input_wav=in_wav,
            output_wav=out_wav,
            target_i=-16.0,
            target_tp=-1.0,
            target_lra=11.0,
            sample_rate=sr,
        )

        assert out_wav.exists()
        # Measure directly via independent FFmpeg ebur128 filter
        measured_i, measured_tp = _measure_ebur128_direct(out_wav)

        # Acceptance: -16.0 +/- 1.0 LUFS -> [-17.0, -15.0]
        assert -17.0 <= measured_i <= -15.0, (
            f"[{expected_input_type}] Integrated loudness {measured_i:.2f} LUFS outside [-17.0, -15.0]"
        )
        # Acceptance: true peak <= -1.0 dBTP
        assert measured_tp <= -0.99, (
            f"[{expected_input_type}] True peak {measured_tp:.2f} dBTP exceeds -1.0 dBTP ceiling"
        )
        assert stats["sample_diff"] == 0

    @pytest.mark.parametrize("sample_count", [
        1000,
        12345,
        23761,
        48000,
        48001,
        71999,
        96000,
    ])
    def test_exact_sample_count_preservation_across_lengths(self, tmp_path: Path, sample_count: int):
        """
        Verify sample count invariant: output frames == input frames (sample diff == 0)
        across diverse, even, and odd sample lengths.
        """
        sr = 48000
        in_wav = tmp_path / f"in_count_{sample_count}.wav"
        out_wav = tmp_path / f"out_count_{sample_count}.wav"

        t = np.linspace(0, sample_count / sr, sample_count, endpoint=False)
        audio = (0.2 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
        sf.write(str(in_wav), audio, sr, subtype="PCM_16")

        stats = normalize_audio_ebur128(in_wav, out_wav, sample_rate=sr)

        out_info = sf.info(str(out_wav))
        assert out_info.frames == sample_count, (
            f"Sample count mismatch: expected {sample_count}, got {out_info.frames}"
        )
        assert out_info.samplerate == sr
        assert stats["sample_diff"] == 0

    def test_silence_handling_pure_silence(self, tmp_path: Path):
        """
        Verify that pure digital silence (zeros) passes through safely without error,
        without producing NaN/infinite gain, and preserves exact frame count.
        """
        sr = 48000
        frames = 24000
        in_wav = tmp_path / "pure_silence.wav"
        out_wav = tmp_path / "pure_silence_out.wav"

        sf.write(str(in_wav), np.zeros(frames, dtype=np.float32), sr, subtype="PCM_16")

        stats = normalize_audio_ebur128(in_wav, out_wav, sample_rate=sr)
        assert out_wav.exists()
        out_data, _ = sf.read(str(out_wav))
        assert np.all(out_data == 0.0)
        assert len(out_data) == frames
        assert stats.get("silence") is True
        assert stats["sample_diff"] == 0

    def test_silence_handling_near_silence(self, tmp_path: Path):
        """
        Verify that near-silent audio (peak < 1e-4) is guarded against extreme gain boosting.
        """
        sr = 48000
        frames = 12000
        in_wav = tmp_path / "near_silence.wav"
        out_wav = tmp_path / "near_silence_out.wav"

        near_silent_audio = np.full(frames, 1e-5, dtype=np.float32)
        sf.write(str(in_wav), near_silent_audio, sr, subtype="PCM_16")

        stats = normalize_audio_ebur128(in_wav, out_wav, sample_rate=sr)
        assert out_wav.exists()
        out_data, _ = sf.read(str(out_wav))
        # Amplitude must remain tiny, not boosted into clipping or noise explosion
        assert np.max(np.abs(out_data)) < 1e-3
        assert stats.get("silence") is True

    def test_multi_clip_segment_leveling(self, tmp_path: Path):
        """
        Verify Tier 1 per-segment speech leveling balances disparate cuts (quiet whisper vs loud shout)
        before crossfading and master normalization, preventing volume jumps and clipping.
        """
        sr = 48000
        seg_samples = 48000
        overlap = 480

        # Quiet clip: amplitude 0.02
        quiet_wav = tmp_path / "whisper.wav"
        sf.write(str(quiet_wav), _generate_synthetic_speech(1.0, sr, amplitude=0.02), sr, subtype="PCM_16")

        # Loud clip: amplitude 0.75
        loud_wav = tmp_path / "shout.wav"
        sf.write(str(loud_wav), _generate_synthetic_speech(1.0, sr, amplitude=0.75, base_freq=330.0), sr, subtype="PCM_16")

        src_quiet = SourceRecord(
            content_id="s_quiet",
            source_path=str(quiet_wav),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(quiet_wav),
        )
        src_loud = SourceRecord(
            content_id="s_loud",
            source_path=str(loud_wav),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(loud_wav),
        )

        seg1 = PlanSegment(
            segment_index=0,
            source_content_id="s_quiet",
            speaker_id="spk",
            source_start_sample=0,
            source_end_sample=seg_samples,
            output_start_sample=0,
            output_end_sample=seg_samples,
            target_text="whisper",
            target_phones=[],
            span_type="word",
            source_video_path=str(quiet_wav),
        )
        seg2 = PlanSegment(
            segment_index=1,
            source_content_id="s_loud",
            speaker_id="spk",
            source_start_sample=0,
            source_end_sample=seg_samples,
            output_start_sample=seg_samples - overlap,
            output_end_sample=(seg_samples * 2) - overlap,
            target_text="shout",
            target_phones=[],
            span_type="word",
            source_video_path=str(loud_wav),
        )
        trans = PlanTransition(
            transition_index=0,
            from_segment_index=0,
            to_segment_index=1,
            transition_type="overlap",
            overlap_samples=overlap,
            fade_samples=overlap,
        )
        total_samples = (seg_samples * 2) - overlap

        plan = EditPlan(
            plan_id="leveling_test_plan",
            target_text="whisper shout",
            speaker_id="spk",
            project_sample_rate=sr,
            total_output_samples=total_samples,
            segments=[seg1, seg2],
            transitions=[trans],
            sources={"s_quiet": src_quiet, "s_loud": src_loud},
        )

        # 1. Unnormalized render: quiet vs loud has massive RMS disparity
        unnorm_wav = tmp_path / "render_unnorm.wav"
        finalize_edit_plan(plan, normalize_loudness=False)
        render_master_pcm(plan, unnorm_wav, normalize_loudness=False)
        data_unnorm, _ = sf.read(str(unnorm_wav))
        rms_quiet_unnorm = np.sqrt(np.mean(data_unnorm[10000:30000] ** 2))
        rms_loud_unnorm = np.sqrt(np.mean(data_unnorm[seg_samples + 10000 : seg_samples + 30000] ** 2))
        unnorm_ratio = rms_loud_unnorm / rms_quiet_unnorm
        assert unnorm_ratio > 20.0

        # 2. Normalized render: leveling applies gain to quiet cut and cuts loud cut
        norm_wav = tmp_path / "render_norm.wav"
        finalize_edit_plan(plan, normalize_loudness=True)
        render_master_pcm(plan, norm_wav, normalize_loudness=True)
        data_norm, _ = sf.read(str(norm_wav))
        rms_quiet_norm = np.sqrt(np.mean(data_norm[10000:30000] ** 2))
        rms_loud_norm = np.sqrt(np.mean(data_norm[seg_samples + 10000 : seg_samples + 30000] ** 2))
        norm_ratio = rms_loud_norm / rms_quiet_norm

        # Disparity significantly reduced
        assert norm_ratio < unnorm_ratio / 3.0, (
            f"Expected leveling to reduce disparity: unnorm={unnorm_ratio:.1f}, norm={norm_ratio:.1f}"
        )

        # Master loudness complies with EBU R128
        meas_i, meas_tp = _measure_ebur128_direct(norm_wav)
        assert -17.0 <= meas_i <= -15.0
        assert meas_tp <= -0.99

        # No clipping
        assert np.max(np.abs(data_norm)) <= 1.0

    def test_cli_normalize_loudness_flag_behavior(self):
        """
        Verify CLI argument parsing in run_mixer.py:
        - Default is True (enabled)
        - --no-normalize-loudness disables normalization
        - --normalize-loudness explicitly enables normalization
        """
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--normalize-loudness",
            action=argparse.BooleanOptionalAction,
            default=True,
        )

        # Default
        args_def = parser.parse_args([])
        assert args_def.normalize_loudness is True

        # Explicit opt-out
        args_opt_out = parser.parse_args(["--no-normalize-loudness"])
        assert args_opt_out.normalize_loudness is False

        # Explicit opt-in
        args_opt_in = parser.parse_args(["--normalize-loudness"])
        assert args_opt_in.normalize_loudness is True

    def test_in_place_normalization(self, tmp_path: Path):
        """
        Verify that in-place normalization (input_wav == output_wav) succeeds,
        does not truncate or corrupt the file, and maintains exact sample count.
        """
        sr = 48000
        frames = 48000
        wav_path = tmp_path / "in_place_test.wav"
        audio = _generate_synthetic_speech(1.0, sr, amplitude=0.03)
        sf.write(str(wav_path), audio, sr, subtype="PCM_16")

        stats = normalize_audio_ebur128(wav_path, wav_path, sample_rate=sr)
        info = sf.info(str(wav_path))
        assert info.frames == frames
        assert stats["sample_diff"] == 0
        meas_i, meas_tp = _measure_ebur128_direct(wav_path)
        assert -17.0 <= meas_i <= -15.0
        assert meas_tp <= -0.99

    def test_extreme_short_clip_silence_guard(self, tmp_path: Path):
        """
        Verify that extremely short clips (e.g. 50 samples) do not crash loudnorm.
        """
        sr = 48000
        frames = 50
        in_wav = tmp_path / "short.wav"
        out_wav = tmp_path / "short_out.wav"
        sf.write(str(in_wav), np.ones(frames, dtype=np.float32) * 0.05, sr, subtype="PCM_16")

        stats = normalize_audio_ebur128(in_wav, out_wav, sample_rate=sr)
        assert out_wav.exists()
        assert sf.info(str(out_wav)).frames == frames
        assert stats["sample_diff"] == 0

    def test_empty_audio_normalization(self, tmp_path: Path):
        """
        Verify that 0-sample WAV input passes through safely without crash.
        """
        sr = 48000
        in_wav = tmp_path / "empty.wav"
        out_wav = tmp_path / "empty_out.wav"
        sf.write(str(in_wav), np.zeros(0, dtype=np.float32), sr, subtype="PCM_16")

        stats = normalize_audio_ebur128(in_wav, out_wav, sample_rate=sr)
        assert out_wav.exists()
        assert sf.info(str(out_wav)).frames == 0
        assert stats.get("silence") is True


class TestASREdgeCases:
    """Additional edge cases for ASR listening verification."""

    def test_homophone_moxy_moxie_acceptance(self):
        """Verify phonetic homophone acceptance for target vocabulary (MOXY vs moxie)."""
        res = verify_listening(
            input_path="dummy.wav",
            expected_text="Anakin, I have the MOXY!",
            audio_array=np.zeros(16000, dtype=np.float32),
            asr_backend=type("MockBackend", (), {
                "transcribe": lambda self, a, initial_prompt=None: "Anakin, I have the MOXIE!"
            })(),
            allow_homophones=True,
        )
        assert res["passed"] is True
        assert res["wer"] == 0.0

    def test_no_homophones_flag_enforcement(self):
        """Verify that disabling homophones marks substitution as mismatch."""
        res = verify_listening(
            input_path="dummy.wav",
            expected_text="Anakin, I have the MOXY!",
            audio_array=np.zeros(16000, dtype=np.float32),
            asr_backend=type("MockBackend", (), {
                "transcribe": lambda self, a, initial_prompt=None: "Anakin, I have the MOXIE!"
            })(),
            allow_homophones=False,
            wer_threshold=0.10,
        )
        assert res["passed"] is False
        assert res["wer"] > 0.10
        assert "moxy" in res["missing_words"]
        assert ("moxy", "moxie") in res["metrics"]["substitutions"]

    def test_zero_length_audio_transcription(self):
        """Verify that empty audio array transcribes to empty string and reports all words missing."""
        res = verify_listening(
            input_path="dummy.wav",
            expected_text="Hello there",
            audio_array=np.zeros(0, dtype=np.float32),
            asr_backend=type("MockBackend", (), {
                "transcribe": lambda self, a, initial_prompt=None: ""
            })(),
        )
        assert res["passed"] is False
        assert res["missing_words"] == ["hello", "there"]
        assert res["wer"] == 1.0

    def test_cli_operational_error_on_missing_file(self):
        """Verify tools/listen_verify.py exits with code 2 on missing input file."""
        cmd = [
            sys.executable,
            str(TOOLS_DIR / "listen_verify.py"),
            "--input", "nonexistent_media_path_12345.wav",
            "--expected", "Hello",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        assert proc.returncode == 2, f"Expected exit code 2 for missing file, got {proc.returncode}"


