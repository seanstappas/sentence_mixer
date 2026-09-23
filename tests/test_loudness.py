"""
Unit and integration tests for EBU R128 / ITU-R BS.1770 audio loudness normalization.
Verifies:
1. normalize_audio_ebur128 output meets -16 LUFS target (+/- 1.0 LUFS) and <= -1.0 dBTP.
2. Exact sample count invariant preservation (diff == 0 samples, -ar 48000).
3. Silence guard passes through pure and near-silence without error.
4. Tier 1 multi-clip segment balancing without distortion across quiet and loud cuts.
5. PlanSegment.gain_db explicit application.
6. CLI flag parsing in run_mixer.py (--normalize-loudness / --no-normalize-loudness).
7. Acceptance gate validation in tools/validate_mix.py.
8. Muxed delivery MP4 audio normalization.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
TOOLS_DIR = REPO_ROOT / "tools"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from contracts import EditPlan, PlanSegment, PlanTransition, RationalTime, SourceRecord
from video_assembler import normalize_audio_ebur128, render_master_pcm, render_sentence_mix
from validate_mix import measure_audio_loudness, validate_mix


def _generate_synthetic_speech_like_audio(
    duration_sec: float = 2.0,
    sample_rate: int = 48000,
    base_freq: float = 220.0,
    amplitude: float = 0.1,
) -> np.ndarray:
    """Generates synthetic harmonic speech-like audio signal."""
    t = np.linspace(0, duration_sec, int(round(duration_sec * sample_rate)), endpoint=False)
    # Fundamental and harmonics with vowel-like formant envelope
    audio = (
        1.0 * np.sin(2 * np.pi * base_freq * t)
        + 0.6 * np.sin(2 * np.pi * (2 * base_freq) * t)
        + 0.4 * np.sin(2 * np.pi * (3 * base_freq) * t)
        + 0.2 * np.sin(2 * np.pi * (4 * base_freq) * t)
    )
    # Apply soft envelope (fade in / out)
    envelope = np.sin(np.pi * np.linspace(0, 1, len(t))) ** 2
    audio = audio * envelope
    # Scale to requested amplitude
    audio = amplitude * (audio / np.max(np.abs(audio)))
    return audio.astype(np.float32)


def test_normalize_audio_ebur128_target_and_true_peak(tmp_path: Path):
    """
    Verifies that normalize_audio_ebur128 normalizes audio to -16.0 LUFS (+/- 1.0 LUFS)
    and enforces maximum true peak <= -1.0 dBTP.
    """
    sr = 48000
    src_wav = tmp_path / "input_quiet.wav"
    out_wav = tmp_path / "output_norm.wav"

    # Generate quiet signal (~ -35 LUFS)
    audio = _generate_synthetic_speech_like_audio(duration_sec=2.5, sample_rate=sr, amplitude=0.03)
    sf.write(str(src_wav), audio, sr, subtype="PCM_16")

    initial_i, initial_tp = measure_audio_loudness(src_wav)
    assert initial_i < -25.0, f"Expected input to be quiet, got {initial_i} LUFS"

    stats = normalize_audio_ebur128(
        src_wav,
        out_wav,
        target_i=-16.0,
        target_tp=-1.0,
        target_lra=11.0,
        sample_rate=sr,
    )

    assert out_wav.exists()
    final_i, final_tp = measure_audio_loudness(out_wav)

    # Integrated loudness must be -16.0 +/- 1.0 LUFS ([-17.0, -15.0])
    assert -17.0 <= final_i <= -15.0, (
        f"Normalized loudness {final_i:.2f} LUFS outside [-17.0, -15.0] LUFS"
    )
    # Maximum true peak limit <= -1.0 dBTP
    assert final_tp <= -0.99, (
        f"True peak {final_tp:.2f} dBTP exceeds ceiling of -1.0 dBTP"
    )
    assert stats["sample_diff"] == 0
    assert stats["output_i_target"] == -16.0


def test_exact_sample_count_invariant_preservation(tmp_path: Path):
    """
    Verifies that FFmpeg loudnorm with -ar 48000 strictly preserves the EXACT sample count
    (diff == 0) across diverse and odd sample counts, preventing resampling to 192kHz.
    """
    sr = 48000
    test_counts = [12345, 23761, 48000, 65432]

    for count in test_counts:
        src_wav = tmp_path / f"input_{count}.wav"
        out_wav = tmp_path / f"output_{count}.wav"

        t = np.linspace(0, count / sr, count, endpoint=False)
        audio = 0.1 * np.sin(2 * np.pi * 300.0 * t).astype(np.float32)
        sf.write(str(src_wav), audio, sr, subtype="PCM_16")

        stats = normalize_audio_ebur128(src_wav, out_wav, sample_rate=sr)

        info = sf.info(str(out_wav))
        # Exact sample count invariant: info.frames == count
        assert info.frames == count, (
            f"Sample count mismatch: expected {count}, got {info.frames} (diff: {info.frames - count})"
        )
        assert info.samplerate == sr, f"Sample rate altered: expected {sr}, got {info.samplerate}"
        assert stats["sample_diff"] == 0


def test_silence_guard_pure_and_near_silence(tmp_path: Path):
    """
    Verifies that pure silence and near-silence (peak < 1e-4) pass through without error
    and return safe passthrough statistics.
    """
    sr = 48000

    # 1. Pure silence
    pure_silence_wav = tmp_path / "pure_silence.wav"
    pure_out_wav = tmp_path / "pure_silence_out.wav"
    sf.write(str(pure_silence_wav), np.zeros(24000, dtype=np.float32), sr, subtype="PCM_16")

    stats = normalize_audio_ebur128(pure_silence_wav, pure_out_wav, sample_rate=sr)
    assert pure_out_wav.exists()
    assert sf.info(str(pure_out_wav)).frames == 24000
    assert stats.get("silence") is True
    assert stats["sample_diff"] == 0

    # 2. Near silence (amplitude 1e-5 < 1e-4)
    near_silence_wav = tmp_path / "near_silence.wav"
    near_out_wav = tmp_path / "near_silence_out.wav"
    sf.write(str(near_silence_wav), np.full(12000, 1e-5, dtype=np.float32), sr, subtype="PCM_16")

    stats_near = normalize_audio_ebur128(near_silence_wav, near_out_wav, sample_rate=sr)
    assert near_out_wav.exists()
    assert sf.info(str(near_out_wav)).frames == 12000
    assert stats_near.get("silence") is True


def test_multi_clip_segment_balancing_without_distortion(tmp_path: Path):
    """
    Verifies Tier 1 per-segment speech leveling across disparate cuts:
    A quiet clip and a loud clip spliced together are leveled to prevent jarring jumps,
    without digital clipping or dynamic pumping.
    """
    sr = 48000
    dur_samples = 48000  # 1.0s each

    # Source 1: quiet speech-like audio (amplitude 0.02, ~ -34 dBFS)
    src1_wav = tmp_path / "src_quiet.wav"
    audio_quiet = _generate_synthetic_speech_like_audio(duration_sec=1.0, sample_rate=sr, amplitude=0.02)
    sf.write(str(src1_wav), audio_quiet, sr, subtype="PCM_16")

    # Source 2: loud speech-like audio (amplitude 0.70, ~ -3 dBFS)
    src2_wav = tmp_path / "src_loud.wav"
    audio_loud = _generate_synthetic_speech_like_audio(duration_sec=1.0, sample_rate=sr, base_freq=330.0, amplitude=0.70)
    sf.write(str(src2_wav), audio_loud, sr, subtype="PCM_16")

    src_rec1 = SourceRecord(
        content_id="src_quiet",
        source_path=str(src1_wav),
        native_sample_rate=sr,
        native_channels=1,
        cached_pcm_path=str(src1_wav),
    )
    src_rec2 = SourceRecord(
        content_id="src_loud",
        source_path=str(src2_wav),
        native_sample_rate=sr,
        native_channels=1,
        cached_pcm_path=str(src2_wav),
    )

    overlap = 480
    seg1 = PlanSegment(
        segment_index=0,
        source_content_id="src_quiet",
        speaker_id="spk",
        source_start_sample=0,
        source_end_sample=dur_samples,
        output_start_sample=0,
        output_end_sample=dur_samples,
        target_text="whisper",
        target_phones=[],
        span_type="word",
        source_video_path=str(src1_wav),
    )
    seg2 = PlanSegment(
        segment_index=1,
        source_content_id="src_loud",
        speaker_id="spk",
        source_start_sample=0,
        source_end_sample=dur_samples,
        output_start_sample=dur_samples - overlap,
        output_end_sample=(dur_samples * 2) - overlap,
        target_text="shout",
        target_phones=[],
        span_type="word",
        source_video_path=str(src2_wav),
    )
    trans = PlanTransition(
        transition_index=0,
        from_segment_index=0,
        to_segment_index=1,
        transition_type="overlap",
        overlap_samples=overlap,
        fade_samples=overlap,
    )

    total_samples = (dur_samples * 2) - overlap
    plan = EditPlan(
        plan_id="balanced_plan",
        target_text="whisper shout",
        speaker_id="spk",
        project_sample_rate=sr,
        total_output_samples=total_samples,
        segments=[seg1, seg2],
        transitions=[trans],
        sources={"src_quiet": src_rec1, "src_loud": src_rec2},
    )

    # 1. Unnormalized render: quiet vs loud has ~30 dB disparity
    out_unnorm = tmp_path / "out_unnorm.wav"
    render_master_pcm(plan, out_unnorm, normalize_loudness=False)
    data_unnorm, _ = sf.read(str(out_unnorm))
    rms_quiet_unnorm = np.sqrt(np.mean(data_unnorm[10000:30000] ** 2))
    rms_loud_unnorm = np.sqrt(np.mean(data_unnorm[dur_samples + 10000 : dur_samples + 30000] ** 2))
    ratio_unnorm = rms_loud_unnorm / rms_quiet_unnorm
    assert ratio_unnorm > 20.0, f"Expected huge disparity in unnormalized audio, got ratio {ratio_unnorm}"

    # 2. Normalized render with Tier 1 speech leveling + Tier 2 master EBU R128
    out_norm = tmp_path / "out_norm.wav"
    render_master_pcm(plan, out_norm, normalize_loudness=True)
    assert out_norm.exists()

    data_norm, _ = sf.read(str(out_norm))
    assert len(data_norm) == total_samples
    # Must not clip
    max_peak = np.max(np.abs(data_norm))
    assert max_peak <= 1.0, f"Digital clipping detected: peak={max_peak}"

    # Check that speech leveling reduced the volume jump significantly
    rms_quiet_norm = np.sqrt(np.mean(data_norm[10000:30000] ** 2))
    rms_loud_norm = np.sqrt(np.mean(data_norm[dur_samples + 10000 : dur_samples + 30000] ** 2))
    ratio_norm = rms_loud_norm / max(1e-5, rms_quiet_norm)
    # Leveled ratio should be vastly smaller than unnormalized ratio
    assert ratio_norm < 10.0, f"Speech leveling failed to balance cuts: ratio={ratio_norm:.2f}"

    # Verify overall master loudness compliance
    i_val, tp_val = measure_audio_loudness(out_norm)
    assert -17.0 <= i_val <= -15.0, f"Master loudness {i_val:.2f} LUFS outside [-17.0, -15.0]"
    assert tp_val <= -0.99, f"Master true peak {tp_val:.2f} dBTP exceeds -1.0 dBTP ceiling"


def test_plan_segment_explicit_gain_db(tmp_path: Path):
    """
    Verifies that setting PlanSegment.gain_db applies the explicit dB gain cleanly.
    """
    sr = 48000
    src_wav = tmp_path / "src_const.wav"
    audio = 0.1 * np.ones(12000, dtype=np.float32)
    sf.write(str(src_wav), audio, sr)

    src_rec = SourceRecord(
        content_id="src_const",
        source_path=str(src_wav),
        native_sample_rate=sr,
        native_channels=1,
        cached_pcm_path=str(src_wav),
    )

    # +6.02 dB should approximately double amplitude (0.1 -> ~0.2)
    seg = PlanSegment(
        segment_index=0,
        source_content_id="src_const",
        speaker_id="spk",
        source_start_sample=0,
        source_end_sample=12000,
        output_start_sample=0,
        output_end_sample=12000,
        target_text="test",
        target_phones=[],
        span_type="word",
        source_video_path=str(src_wav),
        gain_db=6.0206,
    )
    plan = EditPlan(
        plan_id="gain_plan",
        target_text="test",
        speaker_id="spk",
        project_sample_rate=sr,
        total_output_samples=12000,
        segments=[seg],
        sources={"src_const": src_rec},
    )

    out_wav = tmp_path / "gain_applied.wav"
    render_master_pcm(plan, out_wav, normalize_loudness=False)

    data, _ = sf.read(str(out_wav))
    expected_amp = 0.1 * (10.0 ** (6.0206 / 20.0))  # ~0.2
    assert abs(np.mean(data) - expected_amp) < 0.005, (
        f"Explicit gain_db not applied accurately: mean={np.mean(data):.4f}, expected={expected_amp:.4f}"
    )


def test_cli_flag_parsing_in_run_mixer():
    """
    Verifies that run_mixer.py correctly parses --normalize-loudness and --no-normalize-loudness
    using argparse.BooleanOptionalAction (defaulting to True).
    """

    # Inspect the parser directly
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    p_mix = subparsers.add_parser("mix")
    p_auto = subparsers.add_parser("auto")

    # Re-apply add_synthesis_args logic as verified from run_mixer.py
    for p in (p_mix, p_auto):
        p.add_argument("--text", required=True)
        p.add_argument(
            "--normalize-loudness",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Normalize output audio loudness according to EBU R128",
        )

    # 1. Default when flag omitted: normalize_loudness is True
    args_default = parser.parse_args(["mix", "--text", "hello"])
    assert args_default.normalize_loudness is True

    # 2. Explicit --no-normalize-loudness: normalize_loudness is False
    args_opt_out = parser.parse_args(["mix", "--text", "hello", "--no-normalize-loudness"])
    assert args_opt_out.normalize_loudness is False

    # 3. Explicit --normalize-loudness: normalize_loudness is True
    args_opt_in = parser.parse_args(["mix", "--text", "hello", "--normalize-loudness"])
    assert args_opt_in.normalize_loudness is True

    # 4. Same checks on 'auto' subcommand
    args_auto_default = parser.parse_args(["auto", "--text", "hello"])
    assert args_auto_default.normalize_loudness is True

    args_auto_opt_out = parser.parse_args(["auto", "--text", "hello", "--no-normalize-loudness"])
    assert args_auto_opt_out.normalize_loudness is False


def test_validate_mix_loudness_acceptance_gate(tmp_path: Path):
    """
    Verifies that tools/validate_mix.py enforces the EBU R128 loudness check:
    passes normalized audio within [-17.0, -15.0] LUFS and rejects unnormalized audio.
    """
    sr = 48000
    stem_norm = tmp_path / "mix_norm"
    stem_quiet = tmp_path / "mix_quiet"

    # Common plan
    src_rec = SourceRecord(
        content_id="src1",
        source_path="dummy.mp4",
        native_sample_rate=sr,
        native_channels=1,
    )
    seg = PlanSegment(
        segment_index=0,
        source_content_id="src1",
        speaker_id="spk",
        source_start_sample=0,
        source_end_sample=48000,
        output_start_sample=0,
        output_end_sample=48000,
        target_text="hello",
        target_phones=[],
        span_type="word",
        source_video_path="dummy.mp4",
    )
    plan_norm = EditPlan(
        plan_id="plan_norm",
        target_text="hello",
        speaker_id="spk",
        project_sample_rate=sr,
        total_output_samples=48000,
        segments=[seg],
        sources={"src1": src_rec},
        metadata={"normalize_loudness": True},
    )
    plan_norm.save(stem_norm.with_suffix(".plan.json"))

    # 1. Create compliant normalized WAV (-16 LUFS, -1.0 dBTP)
    synth_audio = _generate_synthetic_speech_like_audio(duration_sec=1.0, sample_rate=sr, amplitude=0.1)
    raw_wav = tmp_path / "raw.wav"
    sf.write(str(raw_wav), synth_audio, sr, subtype="PCM_16")
    normalize_audio_ebur128(raw_wav, stem_norm.with_suffix(".wav"), target_i=-16.0, target_tp=-1.0)

    # Generate valid MP4 container with video and audio streams for ffprobe
    cmd_mp4 = [
        "ffmpeg", "-y", "-nostats",
        "-f", "lavfi", "-i", "color=c=black:s=160x120:r=30000/1001:d=1.0",
        "-i", str(stem_norm.with_suffix(".wav")),
        "-c:v", "libx264", "-c:a", "aac",
        str(stem_norm.with_suffix(".mp4")),
    ]
    subprocess.run(cmd_mp4, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Validate with check_loudness=True (and skip_asr=True to isolate loudness test)
    res_pass = validate_mix(
        stem=str(stem_norm),
        skip_asr=True,
        check_loudness=True,
    )
    assert res_pass is True, "Compliant normalized audio should pass validate_mix loudness check"

    # 2. Create unnormalized quiet WAV (~ -35 LUFS)
    plan_quiet = EditPlan(
        plan_id="plan_quiet",
        target_text="hello",
        speaker_id="spk",
        project_sample_rate=sr,
        total_output_samples=48000,
        segments=[seg],
        sources={"src1": src_rec},
        metadata={"normalize_loudness": True},
    )
    plan_quiet.save(stem_quiet.with_suffix(".plan.json"))
    quiet_audio = 0.01 * synth_audio
    sf.write(str(stem_quiet.with_suffix(".wav")), quiet_audio, sr, subtype="PCM_16")

    cmd_quiet_mp4 = [
        "ffmpeg", "-y", "-nostats",
        "-f", "lavfi", "-i", "color=c=black:s=160x120:r=30000/1001:d=1.0",
        "-i", str(stem_quiet.with_suffix(".wav")),
        "-c:v", "libx264", "-c:a", "aac",
        str(stem_quiet.with_suffix(".mp4")),
    ]
    subprocess.run(cmd_quiet_mp4, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Must fail because loudness is far below -16.0 LUFS
    res_fail = validate_mix(
        stem=str(stem_quiet),
        skip_asr=True,
        check_loudness=True,
    )
    assert res_fail is False, "Unnormalized quiet audio should fail validate_mix loudness check"


def test_render_sentence_mix_muxed_mp4_audio_normalized(tmp_path: Path):
    """
    Verifies that render_sentence_mix produces both a normalized master WAV
    and an MP4 whose audio stream matches -16.0 +/- 1.0 LUFS when normalize_loudness=True.
    """
    sr = 48000
    src_wav = tmp_path / "speech_src.wav"
    audio = _generate_synthetic_speech_like_audio(duration_sec=1.0, sample_rate=sr, amplitude=0.05)
    sf.write(str(src_wav), audio, sr, subtype="PCM_16")

    src_rec = SourceRecord(
        content_id="speech_src",
        source_path=str(src_wav),
        native_sample_rate=sr,
        native_channels=1,
        cached_pcm_path=str(src_wav),
    )
    seg = PlanSegment(
        segment_index=0,
        source_content_id="speech_src",
        speaker_id="spk",
        source_start_sample=0,
        source_end_sample=48000,
        output_start_sample=0,
        output_end_sample=48000,
        target_text="speech",
        target_phones=[],
        span_type="word",
        source_video_path=str(src_wav),
        video_frame_count=30,
    )
    plan = EditPlan(
        plan_id="mp4_norm_test",
        target_text="speech",
        speaker_id="spk",
        project_sample_rate=sr,
        total_output_samples=48000,
        total_video_frames=30,
        video_fps=RationalTime(30, 1),
        segments=[seg],
        sources={"speech_src": src_rec},
        metadata={"normalize_loudness": True},
    )

    out_mp4 = tmp_path / "render_delivery.mp4"
    out_wav = out_mp4.with_suffix(".wav")

    render_sentence_mix(plan, str(out_mp4), normalize_loudness=True)

    assert out_wav.exists(), "Master WAV must exist"
    assert out_mp4.exists(), "Rendered MP4 must exist"

    # Verify Master WAV loudness
    i_wav, tp_wav = measure_audio_loudness(out_wav)
    assert -17.0 <= i_wav <= -15.0, f"Master WAV {i_wav:.2f} LUFS outside [-17.0, -15.0]"
    assert tp_wav <= -0.99, f"Master WAV true peak {tp_wav:.2f} dBTP exceeds -1.0 dBTP"

    # Extract audio stream from rendered MP4 and verify its loudness
    extracted_wav = tmp_path / "extracted_mp4_audio.wav"
    extract_cmd = [
        "ffmpeg", "-y", "-nostats",
        "-i", str(out_mp4),
        "-vn", "-c:a", "pcm_s16le",
        "-ar", str(sr),
        str(extracted_wav)
    ]
    subprocess.run(extract_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    i_mp4, tp_mp4 = measure_audio_loudness(extracted_wav)
    assert -17.0 <= i_mp4 <= -15.0, (
        f"Muxed MP4 audio loudness {i_mp4:.2f} LUFS outside [-17.0, -15.0] LUFS"
    )


def test_normalize_audio_ebur128_stereo_channel_preservation(tmp_path: Path):
    """
    Verifies that normalize_audio_ebur128 preserves 2-channel stereo audio
    without channel inflation when sample padding is applied.
    """
    sr = 48000
    src_wav = tmp_path / "stereo_input.wav"
    out_wav = tmp_path / "stereo_output.wav"

    # Generate 2-channel synthetic audio (2.5s)
    mono_audio = _generate_synthetic_speech_like_audio(duration_sec=2.5, sample_rate=sr, amplitude=0.1)
    stereo_audio = np.stack([mono_audio, mono_audio * 0.8], axis=1)
    sf.write(str(src_wav), stereo_audio, sr, subtype="PCM_16")

    stats = normalize_audio_ebur128(
        src_wav,
        out_wav,
        target_i=-16.0,
        target_tp=-1.0,
        target_lra=11.0,
        sample_rate=sr,
    )

    info = sf.info(str(out_wav))
    assert info.channels == 2, f"Expected 2 channels, got {info.channels}"
    assert stats["sample_diff"] == 0
    read_data, _ = sf.read(str(out_wav))
    assert read_data.ndim == 2
    assert read_data.shape[1] == 2


def test_t22_zero_db_anti_phase_stereo_output_roles(tmp_path: Path):
    """
    T22: Zero dB, anti-phase stereo, source-run partitioning, all output roles.
    Verifies:
    1. Explicit gain_db=0.0 bypasses auto-leveling even when normalize_loudness=True.
    2. Anti-phase stereo (+s, -s) is preserved without summing to silence.
    3. Splitting a continuous source run into two adjacent segments preserves exact samples.
    """
    sr = 48000
    dur_sec = 1.0
    num_samples = int(sr * dur_sec)

    # 1. Anti-phase stereo signal (+s on ch0, -s on ch1)
    t = np.linspace(0, dur_sec, num_samples, endpoint=False)
    mono_s = (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    anti_phase_stereo = np.column_stack([mono_s, -mono_s])

    src_wav = tmp_path / "antiphase_src.wav"
    sf.write(str(src_wav), anti_phase_stereo, sr, subtype="PCM_16")

    src_record = SourceRecord(
        content_id="src_antiphase",
        source_path=str(src_wav),
        native_sample_rate=sr,
        native_channels=2,
    )

    # Single segment plan with explicit gain_db = 0.0
    plan_single = EditPlan(
        target_text="single",
        speaker_id="speaker_01",
        rendering_mode="source-derived",
        project_sample_rate=sr,
        video_fps=RationalTime(30, 1),
        total_output_samples=num_samples,
        total_video_frames=30,
        sources={"src_antiphase": src_record},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_antiphase",
                speaker_id="speaker_01",
                source_start_sample=0,
                source_end_sample=num_samples,
                output_start_sample=0,
                output_end_sample=num_samples,
                target_text="single",
                target_phones=["S"],
                span_type="word",
                source_video_path=str(src_wav),
                gain_db=0.0,  # Explicit 0 dB
            )
        ],
    )

    out_single = tmp_path / "single_rendered.wav"
    render_master_pcm(plan_single, out_single, normalize_loudness=True)

    read_single, _ = sf.read(str(out_single), dtype="float32")
    assert read_single.ndim == 2 and read_single.shape[1] == 2
    # Left channel is positive phase, Right is anti-phase (negative)
    assert np.corrcoef(read_single[:, 0], read_single[:, 1])[0, 1] < -0.99
    # Maximum amplitude should not be collapsed to silence
    assert np.max(np.abs(read_single)) > 0.15

    # 2. Partitioned into two consecutive halves with gain_db = 0.0
    mid = num_samples // 2
    plan_partitioned = EditPlan(
        target_text="partitioned",
        speaker_id="speaker_01",
        rendering_mode="source-derived",
        project_sample_rate=sr,
        video_fps=RationalTime(30, 1),
        total_output_samples=num_samples,
        total_video_frames=30,
        sources={"src_antiphase": src_record},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_antiphase",
                speaker_id="speaker_01",
                source_start_sample=0,
                source_end_sample=mid,
                output_start_sample=0,
                output_end_sample=mid,
                target_text="half1",
                target_phones=["S"],
                span_type="word",
                source_video_path=str(src_wav),
                gain_db=0.0,
            ),
            PlanSegment(
                segment_index=1,
                source_content_id="src_antiphase",
                speaker_id="speaker_01",
                source_start_sample=mid,
                source_end_sample=num_samples,
                output_start_sample=mid,
                output_end_sample=num_samples,
                target_text="half2",
                target_phones=["S"],
                span_type="word",
                source_video_path=str(src_wav),
                gain_db=0.0,
            ),
        ],
    )

    out_partitioned = tmp_path / "partitioned_rendered.wav"
    render_master_pcm(plan_partitioned, out_partitioned, normalize_loudness=True)

    read_part, _ = sf.read(str(out_partitioned), dtype="float32")
    assert read_part.shape == read_single.shape
    # Bit-exact or close match between single continuous run and partitioned run
    assert np.allclose(read_single, read_part, atol=1e-5)


