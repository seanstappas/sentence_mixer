"""
Adversarial Stress Test Suite — Milestone 1 Invariants Verification
Author: teamwork_preview_challenger_m1_it2_2

Rigorous empirical challenge for:
1. Extreme/negative sample range bounds checking in `src/video_assembler.py` (must raise ValueError).
2. Stereo channel preservation throughout assembly in `src/video_assembler.py` (must maintain 2 distinct channels).
3. Vocal stem discrimination and hash-based invalidation in `src/vocal_isolation.py`
   (reject `no_vocals.wav`, never invert stems, invalidate on content hash mismatch).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from contracts import EditPlan, PlanSegment, RationalTime, SourceRecord
from video_assembler import normalize_audio_ebur128, render_master_pcm, render_sentence_mix
from vocal_isolation import (
    _find_instrumental_candidates,
    _find_vocal_candidates,
    compute_file_sha256,
    get_source_id,
    isolate_vocals,
)


# ============================================================================
# Section 1: Extreme & Negative Sample Range Bounds Checking (video_assembler.py)
# ============================================================================

@pytest.mark.parametrize(
    "start_sample,end_sample,expected_error_regex",
    [
        (-1, 1000, "Invalid source sample range"),
        (-500, 1000, "Invalid source sample range"),
        (-999999, 1000, "Invalid source sample range"),
        (500, 200, "Invalid source sample range"),       # Inverted: end < start
        (1000, 0, "Invalid source sample range"),         # Inverted: end < start
        (0, 48001, "Invalid source sample range"),        # 1 sample past buffer end (buffer len = 48000)
        (0, 999999, "Invalid source sample range"),       # Far past buffer end
        (48000, 48001, "Invalid source sample range"),    # Starting at end, asking for 1 more
        (-100, -50, "Invalid source sample range"),       # Both negative
    ],
)
def test_adversarial_bounds_checking_combined_track(
    tmp_path: Path,
    start_sample: int,
    end_sample: int,
    expected_error_regex: str,
):
    """
    Adversarial Challenge 1.1:
    Verify that render_master_pcm strictly raises ValueError across a battery
    of negative, inverted, and overflowing sample boundaries in default 'combined' mode.
    Under NO circumstances may it silently pad (np.pad) or return zeros.
    """
    sr = 48000
    source_len = sr  # Exactly 48,000 samples (1.0s)
    dummy_wav = tmp_path / "source_combined.wav"
    audio_data = np.sin(np.linspace(0, 100, source_len, dtype=np.float32))
    sf.write(str(dummy_wav), audio_data, sr)

    out_wav = tmp_path / "out_combined.wav"

    source_rec = SourceRecord(
        content_id="src_comb",
        source_path=str(dummy_wav),
        cached_pcm_path=str(dummy_wav),
        native_sample_rate=sr,
        native_channels=1,
    )

    plan = EditPlan(
        target_text="bounds test",
        project_sample_rate=sr,
        total_output_samples=max(1000, end_sample - start_sample if end_sample > start_sample else 1000),
        sources={"src_comb": source_rec},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_comb",
                speaker_id="spk_test",
                source_start_sample=start_sample,
                source_end_sample=end_sample,
                output_start_sample=0,
                output_end_sample=max(1000, end_sample - start_sample if end_sample > start_sample else 1000),
                target_text="bounds",
                target_phones=["B", "AW1", "N", "D", "Z"],
                span_type="word",
                source_video_path=str(dummy_wav),
            )
        ],
    )

    with pytest.raises(ValueError, match=expected_error_regex):
        render_master_pcm(plan, out_wav)


@pytest.mark.parametrize(
    "start_sample,end_sample",
    [
        (-1, 500),
        (-10000, 500),
        (500, 100),       # Inverted
        (0, 48005),       # Overflowing
        (50000, 60000),   # Far past length
    ],
)
def test_adversarial_bounds_checking_isolated_track(
    tmp_path: Path,
    start_sample: int,
    end_sample: int,
):
    """
    Adversarial Challenge 1.2:
    Verify that render_master_pcm with audio_track='isolated' strictly raises ValueError
    when requesting out-of-bounds or negative sample ranges against the vocal stem.
    """
    sr = 48000
    source_len = sr  # 48,000 samples
    vocal_wav = tmp_path / "vocals.wav"
    sf.write(str(vocal_wav), np.ones(source_len, dtype=np.float32) * 0.5, sr)

    out_wav = tmp_path / "out_isolated.wav"

    source_rec = SourceRecord(
        content_id="src_iso",
        source_path=str(vocal_wav),
        cached_pcm_path=str(vocal_wav),
        cached_vocal_pcm_path=str(vocal_wav),
        native_sample_rate=sr,
        native_channels=1,
    )

    plan = EditPlan(
        target_text="isolated bounds test",
        project_sample_rate=sr,
        total_output_samples=max(500, end_sample - start_sample if end_sample > start_sample else 500),
        sources={"src_iso": source_rec},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_iso",
                speaker_id="spk_iso",
                source_start_sample=start_sample,
                source_end_sample=end_sample,
                output_start_sample=0,
                output_end_sample=max(500, end_sample - start_sample if end_sample > start_sample else 500),
                target_text="isolated",
                target_phones=["AY1", "S", "AH0", "L", "EY1", "T", "IH0", "D"],
                span_type="word",
                source_video_path=str(vocal_wav),
            )
        ],
    )

    with pytest.raises(ValueError, match="Invalid source sample range"):
        render_master_pcm(plan, out_wav, audio_track="isolated")


@pytest.mark.parametrize(
    "start_sample,end_sample",
    [
        (-10, 1000),
        (1000, 500),       # Inverted
        (0, 50000),        # Past stem length (48000)
    ],
)
def test_adversarial_bounds_checking_recombined_track(
    tmp_path: Path,
    start_sample: int,
    end_sample: int,
):
    """
    Adversarial Challenge 1.3:
    Verify that render_master_pcm with audio_track='recombined' checks bounds on both
    the vocal stem and the instrumental stem, strictly raising ValueError.
    """
    sr = 48000
    source_len = sr
    vocal_wav = tmp_path / "vocals.wav"
    inst_wav = tmp_path / "instrumental.wav"
    sf.write(str(vocal_wav), np.ones(source_len, dtype=np.float32) * 0.3, sr)
    sf.write(str(inst_wav), np.ones(source_len, dtype=np.float32) * 0.2, sr)

    out_wav = tmp_path / "out_recombined.wav"

    source_rec = SourceRecord(
        content_id="src_recomb",
        source_path=str(vocal_wav),
        cached_pcm_path=str(vocal_wav),
        cached_vocal_pcm_path=str(vocal_wav),
        cached_instrumental_pcm_path=str(inst_wav),
        native_sample_rate=sr,
        native_channels=1,
    )

    plan = EditPlan(
        target_text="recombined bounds test",
        project_sample_rate=sr,
        total_output_samples=max(1000, end_sample - start_sample if end_sample > start_sample else 1000),
        sources={"src_recomb": source_rec},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_recomb",
                speaker_id="spk_recomb",
                source_start_sample=start_sample,
                source_end_sample=end_sample,
                output_start_sample=0,
                output_end_sample=max(1000, end_sample - start_sample if end_sample > start_sample else 1000),
                target_text="recomb",
                target_phones=["R", "IY0", "K", "AA1", "M"],
                span_type="word",
                source_video_path=str(vocal_wav),
            )
        ],
    )

    with pytest.raises(ValueError, match="Invalid source sample range"):
        render_master_pcm(plan, out_wav, audio_track="recombined")


def test_adversarial_bounds_checking_slice_shorter_than_duration_raises_value_error(tmp_path: Path):
    """
    Adversarial Challenge 1.4:
    If source slice length is shorter than duration_samples (and native_rate == sr),
    render_master_pcm MUST raise ValueError and refuse to silently pad with silence.
    """
    sr = 48000
    source_len = 10000
    dummy_wav = tmp_path / "short_source.wav"
    sf.write(str(dummy_wav), np.zeros(source_len, dtype=np.float32), sr)

    out_wav = tmp_path / "out_short.wav"

    source_rec = SourceRecord(
        content_id="src_short",
        source_path=str(dummy_wav),
        cached_pcm_path=str(dummy_wav),
        native_sample_rate=sr,
        native_channels=1,
    )

    # Segment claims duration_samples = 2000, but source range is only 1000 samples
    seg = PlanSegment(
        segment_index=0,
        source_content_id="src_short",
        speaker_id="spk_test",
        source_start_sample=0,
        source_end_sample=1000,   # length 1000
        output_start_sample=0,
        output_end_sample=2000,   # duration_samples = 2000
        target_text="short",
        target_phones=["SH", "AO1", "R", "T"],
        span_type="word",
        source_video_path=str(dummy_wav),
    )

    plan = EditPlan(
        target_text="short test",
        project_sample_rate=sr,
        total_output_samples=2000,
        sources={"src_short": source_rec},
        segments=[seg],
    )

    with pytest.raises(ValueError, match="is less than expected duration"):
        render_master_pcm(plan, out_wav)


# ============================================================================
# Section 2: Stereo Channel Preservation Throughout Assembly (video_assembler.py)
# ============================================================================

def test_adversarial_stereo_preservation_uncorrelated_channels(tmp_path: Path):
    """
    Adversarial Challenge 2.1:
    Test stereo integrity with completely uncorrelated signals:
    - Left channel: 440 Hz tone + distinct DC bias
    - Right channel: 880 Hz tone + distinct DC bias
    Verify:
    1. Rendered master WAV is 2-channel.
    2. Left channel perfectly preserves 440 Hz signal.
    3. Right channel perfectly preserves 880 Hz signal.
    4. Cross-channel difference has high RMS energy (no mono downmix).
    """
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    ch_left = (0.5 * np.sin(2 * np.pi * 440 * t) + 0.1).astype(np.float32)
    ch_right = (0.3 * np.sin(2 * np.pi * 880 * t) - 0.1).astype(np.float32)
    stereo_matrix = np.column_stack([ch_left, ch_right])

    stereo_wav = tmp_path / "stereo_uncorrelated.wav"
    sf.write(str(stereo_wav), stereo_matrix, sr)

    out_wav = tmp_path / "stereo_master_uncorrelated.wav"

    source_rec = SourceRecord(
        content_id="src_stereo_test",
        source_path=str(stereo_wav),
        cached_pcm_path=str(stereo_wav),
        native_sample_rate=sr,
        native_channels=2,
    )

    plan = EditPlan(
        target_text="stereo integrity",
        rendering_mode="cuts-only",  # Purity: bit-exact
        project_sample_rate=sr,
        total_output_samples=int(sr * duration),
        sources={"src_stereo_test": source_rec},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_stereo_test",
                speaker_id="spk_stereo",
                source_start_sample=0,
                source_end_sample=int(sr * duration),
                output_start_sample=0,
                output_end_sample=int(sr * duration),
                target_text="stereo",
                target_phones=["S", "T", "EH1", "R", "IY0", "OW0"],
                span_type="phrase",
                source_video_path=str(stereo_wav),
            )
        ],
    )

    render_master_pcm(plan, out_wav, normalize_loudness=False)

    data, read_sr = sf.read(str(out_wav))
    assert read_sr == sr
    assert data.ndim == 2
    assert data.shape[1] == 2
    assert data.shape[0] == int(sr * duration)

    # Verbatim channel match within 16-bit PCM precision (1/32768 ~= 3e-5)
    np.testing.assert_allclose(data[:, 0], ch_left, atol=1e-4)
    np.testing.assert_allclose(data[:, 1], ch_right, atol=1e-4)

    # Ensure channels are distinctly different
    diff = data[:, 0] - data[:, 1]
    rms_diff = np.sqrt(np.mean(diff ** 2))
    assert rms_diff > 0.2, f"Channels are too similar (rms_diff={rms_diff}), possible mono downmixing!"


def test_adversarial_stereo_preservation_with_loudnorm_normalization(tmp_path: Path):
    """
    Adversarial Challenge 2.2:
    EBU R128 loudness normalization (`normalize_audio_ebur128`) must NOT collapse stereo
    audio into mono. The output must maintain 2 channels with distinct L/R energy.
    """
    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Different frequency tones on Left and Right
    ch_l = (0.4 * np.sin(2 * np.pi * 330 * t)).astype(np.float32)
    ch_r = (0.4 * np.sin(2 * np.pi * 660 * t)).astype(np.float32)
    stereo_audio = np.column_stack([ch_l, ch_r])

    in_wav = tmp_path / "pre_norm_stereo.wav"
    out_wav = tmp_path / "post_norm_stereo.wav"
    sf.write(str(in_wav), stereo_audio, sr)

    stats = normalize_audio_ebur128(
        input_wav=in_wav,
        output_wav=out_wav,
        target_i=-16.0,
        target_tp=-1.0,
        sample_rate=sr,
    )
    assert stats["output_i_target"] == -16.0

    assert out_wav.exists()
    norm_data, norm_sr = sf.read(str(out_wav))
    assert norm_sr == sr
    assert norm_data.ndim == 2
    assert norm_data.shape[1] == 2
    assert norm_data.shape[0] == int(sr * duration)

    # Channels must remain distinct after loudnorm filter
    assert not np.allclose(norm_data[:, 0], norm_data[:, 1], atol=1e-2)


def test_adversarial_stereo_crossfade_transitions(tmp_path: Path):
    """
    Adversarial Challenge 2.3:
    Transitions with crossfade overlaps between two stereo segments must preserve
    2-channel layout and apply independent crossfades on both Left and Right channels.
    """
    from contracts import PlanTransition

    sr = 48000
    seg_len = 24000  # 0.5s each
    overlap = 4800   # 0.1s overlap

    # Segment 1 stereo
    t1 = np.linspace(0, 0.5, seg_len, endpoint=False)
    s1 = np.column_stack([np.sin(2 * np.pi * 200 * t1), np.sin(2 * np.pi * 400 * t1)]).astype(np.float32)
    wav1 = tmp_path / "s1.wav"
    sf.write(str(wav1), s1, sr)

    # Segment 2 stereo
    t2 = np.linspace(0, 0.5, seg_len, endpoint=False)
    s2 = np.column_stack([np.sin(2 * np.pi * 600 * t2), np.sin(2 * np.pi * 800 * t2)]).astype(np.float32)
    wav2 = tmp_path / "s2.wav"
    sf.write(str(wav2), s2, sr)

    total_samples = (seg_len * 2) - overlap
    out_wav = tmp_path / "crossfade_stereo.wav"

    source_recs = {
        "src_1": SourceRecord("src_1", str(wav1), native_sample_rate=sr, native_channels=2, cached_pcm_path=str(wav1)),
        "src_2": SourceRecord("src_2", str(wav2), native_sample_rate=sr, native_channels=2, cached_pcm_path=str(wav2)),
    }

    plan = EditPlan(
        target_text="crossfade test",
        rendering_mode="cuts-only",
        project_sample_rate=sr,
        total_output_samples=total_samples,
        sources=source_recs,
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_1",
                speaker_id="spk_1",
                source_start_sample=0,
                source_end_sample=seg_len,
                output_start_sample=0,
                output_end_sample=seg_len,
                target_text="one",
                target_phones=["W", "AH1", "N"],
                span_type="word",
                source_video_path=str(wav1),
            ),
            PlanSegment(
                segment_index=1,
                source_content_id="src_2",
                speaker_id="spk_2",
                source_start_sample=0,
                source_end_sample=seg_len,
                output_start_sample=seg_len - overlap,
                output_end_sample=total_samples,
                target_text="two",
                target_phones=["T", "UW1"],
                span_type="word",
                source_video_path=str(wav2),
            ),
        ],
        transitions=[
            PlanTransition(
                transition_index=0,
                from_segment_index=0,
                to_segment_index=1,
                transition_type="overlap",
                overlap_samples=overlap,
            )
        ],
    )

    render_master_pcm(plan, out_wav, normalize_loudness=False)

    data, _ = sf.read(str(out_wav))
    assert data.ndim == 2
    assert data.shape[1] == 2
    assert data.shape[0] == total_samples
    assert not np.allclose(data[:, 0], data[:, 1], atol=1e-3)


def test_adversarial_stereo_channel_preservation_in_full_mp4_render(tmp_path: Path):
    """
    Adversarial Challenge 2.4:
    Test render_sentence_mix with stereo input through full video multiplexing.
    Confirm the output MP4 contains an audio stream with channels == 2.
    """
    from media_time import probe_media_file

    sr = 48000
    dur_samples = 48000  # 1 second
    t = np.linspace(0, 1.0, dur_samples, endpoint=False)
    stereo_data = np.column_stack([0.3 * np.sin(2 * np.pi * 400 * t), 0.3 * np.sin(2 * np.pi * 800 * t)]).astype(np.float32)

    src_wav = tmp_path / "stereo_source_mp4.wav"
    sf.write(str(src_wav), stereo_data, sr)

    out_mp4 = tmp_path / "stereo_render.mp4"

    source_rec = SourceRecord(
        content_id="src_mp4_stereo",
        source_path=str(src_wav),
        cached_pcm_path=str(src_wav),
        native_sample_rate=sr,
        native_channels=2,
    )

    plan = EditPlan(
        target_text="stereo mp4 test",
        project_sample_rate=sr,
        video_fps=RationalTime(24, 1),
        total_output_samples=dur_samples,
        total_video_frames=24,
        sources={"src_mp4_stereo": source_rec},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_mp4_stereo",
                speaker_id="spk_test",
                source_start_sample=0,
                source_end_sample=dur_samples,
                output_start_sample=0,
                output_end_sample=dur_samples,
                target_text="stereo",
                target_phones=["S", "T", "EH1", "R", "IY0", "OW0"],
                span_type="phrase",
                source_video_path=str(src_wav),
                video_start_frame=0,
                video_frame_count=24,
                video_source_pts_sec=0.0,
            )
        ],
    )

    rendered_mp4 = render_sentence_mix(
        plan_or_units=plan,
        output_mp4_path=str(out_mp4),
        normalize_loudness=False,
    )

    assert Path(rendered_mp4).exists()

    # Probe MP4 streams using media_time
    probe_info = probe_media_file(rendered_mp4)
    audio_stream = next((s for s in probe_info.get("streams", []) if s.get("codec_type") == "audio"), None)
    assert audio_stream is not None, "Output MP4 does not contain an audio stream"
    assert int(audio_stream.get("channels", 0)) == 2, (
        f"Expected 2 audio channels in rendered MP4, got {audio_stream.get('channels')}"
    )


# ============================================================================
# Section 3: Stem Discrimination & Hash Invalidation (vocal_isolation.py)
# ============================================================================

def test_adversarial_find_vocal_candidates_rejects_sole_no_vocals_file(tmp_path: Path):
    """
    Adversarial Challenge 3.1:
    If a directory contains ONLY 'no_vocals.wav' and NO genuine vocal file,
    _find_vocal_candidates MUST return an empty list [], NEVER falling back to no_vocals.wav.
    """
    test_dir = tmp_path / "solo_no_vocals"
    test_dir.mkdir(parents=True)

    no_vocals = test_dir / "no_vocals.wav"
    no_vocals.write_bytes(b"RIFF_INSTRUMENTAL_ONLY")

    vocal_cands = _find_vocal_candidates(test_dir)
    assert vocal_cands == [], f"Expected empty candidates list, got {vocal_cands}"

    # Conversely, _find_instrumental_candidates MUST recognize it
    inst_cands = _find_instrumental_candidates(test_dir)
    assert len(inst_cands) == 1
    assert inst_cands[0] == no_vocals


def test_adversarial_isolate_vocals_raises_runtime_error_if_only_no_vocals_produced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Adversarial Challenge 3.2 (Stem Inversion Protection):
    If audio separation produces ONLY an instrumental track ('no_vocals.wav') and fails
    to produce a vocal stem, isolate_vocals MUST raise RuntimeError rather than
    inverting stems and serving instrumental music as speech.
    """
    src_file = tmp_path / "speech.wav"
    src_file.write_bytes(b"INPUT_AUDIO")

    vocals_out = tmp_path / "vocals_out"

    def mock_separator_broken_stems(cmd, *args, **kwargs):
        out_d = Path(cmd[cmd.index("--output_dir") + 1])
        out_d.mkdir(parents=True, exist_ok=True)
        # Only write no_vocals.wav!
        (out_d / "no_vocals.wav").write_bytes(b"RIFF_ONLY_INSTRUMENTAL")
        return MagicMock(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: "mock_sep")
    monkeypatch.setattr("subprocess.run", mock_separator_broken_stems)

    with pytest.raises(RuntimeError, match="Vocal separation failed"):
        isolate_vocals(src_file, output_dir=vocals_out, fallback_to_demucs=False)


def test_adversarial_hash_invalidation_byte_modification_same_size_and_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Adversarial Challenge 3.3 (Stealth Modification Invalidation):
    Verify that modifying 1 byte in the source audio while keeping file size identical
    and manually restoring st_mtime triggers SHA-256 cache invalidation.
    Stale stems MUST be cleaned up and separation re-executed.
    """
    src_file = tmp_path / "voice.wav"
    content_v1 = b"SPEECH_DATA_V1_" + (b"X" * 1000)
    src_file.write_bytes(content_v1)
    fixed_time = time.time() - 3600
    os.utime(src_file, (fixed_time, fixed_time))

    v1_hash = compute_file_sha256(src_file)

    vocals_base = tmp_path / "vocals_cache"
    sid = get_source_id(src_file)
    stem_dir = vocals_base / sid
    stem_dir.mkdir(parents=True)

    vocal_stem = stem_dir / "vocals.wav"
    vocal_stem.write_bytes(b"OLD_STALE_VOCAL_STEM")

    inst_stem = stem_dir / "instrumental.wav"
    inst_stem.write_bytes(b"OLD_STALE_INST_STEM")

    meta_file = stem_dir / "metadata.json"
    meta_file.write_text(
        json.dumps({
            "source_id": sid,
            "source_sha256": v1_hash,
            "model_name": "test_model",
        }),
        encoding="utf-8",
    )

    rerun_executed = 0

    def mock_separator(cmd, *args, **kwargs):
        nonlocal rerun_executed
        rerun_executed += 1
        out_d = Path(cmd[cmd.index("--output_dir") + 1])
        (out_d / "vocals.wav").write_bytes(b"FRESH_RESEPARATED_VOCAL_STEM")
        return MagicMock(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: "mock_sep")
    monkeypatch.setattr("subprocess.run", mock_separator)

    # Step 1: Baseline call -> Cache hit
    hit = isolate_vocals(src_file, output_dir=vocals_base, force=False)
    assert rerun_executed == 0
    assert hit.read_bytes() == b"OLD_STALE_VOCAL_STEM"

    # Step 2: Stealth tamper (same size, same timestamp, different 1 byte)
    content_v2 = b"SPEECH_DATA_V2_" + (b"X" * 1000)
    assert len(content_v2) == len(content_v1)
    src_file.write_bytes(content_v2)
    os.utime(src_file, (fixed_time, fixed_time))

    assert src_file.stat().st_size == len(content_v1)
    assert src_file.stat().st_mtime == fixed_time
    assert compute_file_sha256(src_file) != v1_hash

    # Step 3: Call isolate_vocals -> Must detect hash divergence, unlink old stems, rerun
    fresh_stem = isolate_vocals(src_file, output_dir=vocals_base, force=False)
    assert rerun_executed == 1
    assert fresh_stem.read_bytes() == b"FRESH_RESEPARATED_VOCAL_STEM"

    # Metadata must be updated with v2 hash
    new_meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert new_meta["source_sha256"] == compute_file_sha256(src_file)


def test_adversarial_force_flag_bypasses_and_purges_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Adversarial Challenge 3.4:
    Calling isolate_vocals with force=True MUST unlink existing stems and metadata,
    even if the source content hash perfectly matches.
    """
    src_file = tmp_path / "voice_force.wav"
    src_file.write_bytes(b"SPEECH_FORCE_TEST")

    vocals_base = tmp_path / "vocals_cache"
    sid = get_source_id(src_file)
    stem_dir = vocals_base / sid
    stem_dir.mkdir(parents=True)

    vocal_stem = stem_dir / "vocals.wav"
    vocal_stem.write_bytes(b"ORIGINAL_VOCALS")
    inst_stem = stem_dir / "instrumental.wav"
    inst_stem.write_bytes(b"ORIGINAL_INST")

    meta_file = stem_dir / "metadata.json"
    meta_file.write_text(
        json.dumps({
            "source_id": sid,
            "source_sha256": compute_file_sha256(src_file),
            "model_name": "original_model",
        }),
        encoding="utf-8",
    )

    reseparated = False

    def mock_separator(cmd, *args, **kwargs):
        nonlocal reseparated
        reseparated = True
        out_d = Path(cmd[cmd.index("--output_dir") + 1])
        (out_d / "vocals.wav").write_bytes(b"FORCED_REBUILD_VOCALS")
        return MagicMock(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: "mock_sep")
    monkeypatch.setattr("subprocess.run", mock_separator)

    res = isolate_vocals(src_file, output_dir=vocals_base, force=True)
    assert reseparated is True
    assert res.read_bytes() == b"FORCED_REBUILD_VOCALS"


def test_adversarial_stereo_mixed_mono_and_stereo_inputs(tmp_path: Path):
    """
    Adversarial Challenge 2.5:
    Splicing a mono segment with a stereo segment must promote the entire master
    buffer to stereo (2 channels). Mono audio is duplicated to L+R, while stereo
    audio retains separate L and R channels.
    """
    sr = 48000
    seg_len = 24000  # 0.5s

    # Segment 0: Mono 440 Hz
    t0 = np.linspace(0, 0.5, seg_len, endpoint=False)
    mono_data = (0.5 * np.sin(2 * np.pi * 440 * t0)).astype(np.float32)
    mono_wav = tmp_path / "mono.wav"
    sf.write(str(mono_wav), mono_data, sr)

    # Segment 1: Stereo (Left 600 Hz, Right 800 Hz)
    t1 = np.linspace(0, 0.5, seg_len, endpoint=False)
    stereo_data = np.column_stack([0.4 * np.sin(2 * np.pi * 600 * t1), 0.4 * np.sin(2 * np.pi * 800 * t1)]).astype(np.float32)
    stereo_wav = tmp_path / "stereo.wav"
    sf.write(str(stereo_wav), stereo_data, sr)

    out_wav = tmp_path / "mixed_master.wav"

    source_recs = {
        "src_mono": SourceRecord("src_mono", str(mono_wav), native_sample_rate=sr, native_channels=1, cached_pcm_path=str(mono_wav)),
        "src_stereo": SourceRecord("src_stereo", str(stereo_wav), native_sample_rate=sr, native_channels=2, cached_pcm_path=str(stereo_wav)),
    }

    plan = EditPlan(
        target_text="mixed mono stereo",
        rendering_mode="cuts-only",
        project_sample_rate=sr,
        total_output_samples=seg_len * 2,
        sources=source_recs,
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_mono",
                speaker_id="spk_mono",
                source_start_sample=0,
                source_end_sample=seg_len,
                output_start_sample=0,
                output_end_sample=seg_len,
                target_text="mono",
                target_phones=["M", "AA1", "N", "OW0"],
                span_type="word",
                source_video_path=str(mono_wav),
            ),
            PlanSegment(
                segment_index=1,
                source_content_id="src_stereo",
                speaker_id="spk_stereo",
                source_start_sample=0,
                source_end_sample=seg_len,
                output_start_sample=seg_len,
                output_end_sample=seg_len * 2,
                target_text="stereo",
                target_phones=["S", "T", "EH1", "R", "IY0", "OW0"],
                span_type="word",
                source_video_path=str(stereo_wav),
            ),
        ],
    )

    render_master_pcm(plan, out_wav, normalize_loudness=False)

    data, read_sr = sf.read(str(out_wav))
    assert read_sr == sr
    assert data.ndim == 2
    assert data.shape == (seg_len * 2, 2)

    # Segment 0 slice (samples 0 to 24000) should have L == R (dual mono)
    np.testing.assert_allclose(data[:seg_len, 0], data[:seg_len, 1], atol=1e-4)

    # Segment 1 slice (samples 24000 to 48000) must have L != R (stereo preserved)
    assert not np.allclose(data[seg_len:, 0], data[seg_len:, 1], atol=1e-3)
    np.testing.assert_allclose(data[seg_len:, 0], stereo_data[:, 0], atol=1e-4)
    np.testing.assert_allclose(data[seg_len:, 1], stereo_data[:, 1], atol=1e-4)


def test_adversarial_stereo_dynamic_runtime_expansion(tmp_path: Path):
    """
    Adversarial Challenge 2.6:
    When SourceRecord.native_channels is incorrectly flagged as 1, but the underlying
    source audio is discovered to be stereo during decoding, render_master_pcm must
    dynamically promote master_audio to (total_samples, 2) and preserve both channels.
    """
    sr = 48000
    seg_len = 24000
    t = np.linspace(0, 0.5, seg_len, endpoint=False)
    stereo_data = np.column_stack([0.5 * np.sin(2 * np.pi * 300 * t), 0.5 * np.sin(2 * np.pi * 500 * t)]).astype(np.float32)

    # Use a dummy extension so upfront probe in plan doesn't catch it
    raw_pcm = tmp_path / "audio_source.dat"
    sf.write(str(raw_pcm), stereo_data, sr, format="WAV")

    out_wav = tmp_path / "dynamic_expanded.wav"

    source_rec = SourceRecord(
        content_id="src_dyn",
        source_path=str(raw_pcm),
        cached_pcm_path=str(raw_pcm),
        native_sample_rate=sr,
        native_channels=1,  # Claim 1 channel!
    )

    plan = EditPlan(
        target_text="dynamic expansion test",
        rendering_mode="cuts-only",
        project_sample_rate=sr,
        total_output_samples=seg_len,
        sources={"src_dyn": source_rec},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_dyn",
                speaker_id="spk_dyn",
                source_start_sample=0,
                source_end_sample=seg_len,
                output_start_sample=0,
                output_end_sample=seg_len,
                target_text="dynamic",
                target_phones=["D", "AY0", "N", "AE1", "M", "IH0", "K"],
                span_type="word",
                source_video_path=str(raw_pcm),
            )
        ],
    )

    render_master_pcm(plan, out_wav, normalize_loudness=False)

    data, _ = sf.read(str(out_wav))
    assert data.ndim == 2
    assert data.shape == (seg_len, 2)
    assert not np.allclose(data[:, 0], data[:, 1], atol=1e-3)


def test_adversarial_empty_legacy_units_raises_value_error():
    """
    Adversarial Challenge 1.5:
    render_sentence_mix with empty legacy units list must raise ValueError.
    """
    with pytest.raises(ValueError, match="Cannot render empty units list"):
        render_sentence_mix([], "dummy_output.mp4")


def test_adversarial_demucs_four_stem_candidate_discrimination(tmp_path: Path):
    """
    Adversarial Challenge 3.5:
    In a standard Demucs 4-stem directory structure containing:
    - vocals.wav
    - drums.wav
    - bass.wav
    - other.wav
    _find_vocal_candidates MUST return ONLY vocals.wav.
    _find_instrumental_candidates MUST return drums, bass, and other (not vocals).
    """
    demucs_dir = tmp_path / "htdemucs" / "track1"
    demucs_dir.mkdir(parents=True)

    v_file = demucs_dir / "vocals.wav"
    v_file.write_bytes(b"VOCALS")
    d_file = demucs_dir / "drums.wav"
    d_file.write_bytes(b"DRUMS")
    b_file = demucs_dir / "bass.wav"
    b_file.write_bytes(b"BASS")
    o_file = demucs_dir / "other.wav"
    o_file.write_bytes(b"OTHER")

    vocal_cands = _find_vocal_candidates(demucs_dir)
    assert len(vocal_cands) == 1
    assert vocal_cands[0] == v_file

    inst_cands = _find_instrumental_candidates(demucs_dir)
    assert v_file not in inst_cands
    assert d_file in inst_cands or b_file in inst_cands or o_file in inst_cands

