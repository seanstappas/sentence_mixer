"""
Tests for video and master PCM assembly.
Verifies exact master WAV sample count assertion and CFR slice rendering.
"""

import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

from contracts import EditPlan, PlanSegment, PlanTransition, SourceRecord
from video_assembler import render_master_pcm


def test_render_master_pcm_exact_sample_count():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sr = 48000

        # Create synthetic source audio
        src_wav = tmp_path / "source1.wav"
        t = np.linspace(0, 1.0, sr, endpoint=False)
        audio = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        sf.write(str(src_wav), audio, sr)

        src_rec = SourceRecord(
            content_id="src_1",
            source_path=str(src_wav),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(src_wav),
        )

        seg1 = PlanSegment(
            segment_index=0,
            source_content_id="src_1",
            speaker_id="spk1",
            source_start_sample=0,
            source_end_sample=12000,  # 0.25s
            output_start_sample=0,
            output_end_sample=12000,
            target_text="test1",
            target_phones=[],
            span_type="word",
            source_video_path=str(src_wav),
            gain_db=0.0,
        )

        seg2 = PlanSegment(
            segment_index=1,
            source_content_id="src_1",
            speaker_id="spk1",
            source_start_sample=12000,
            source_end_sample=24000,  # 0.25s
            output_start_sample=11760,  # 240 sample overlap
            output_end_sample=23760,
            target_text="test2",
            target_phones=[],
            span_type="word",
            source_video_path=str(src_wav),
            gain_db=0.0,
        )

        trans = PlanTransition(
            transition_index=0,
            from_segment_index=0,
            to_segment_index=1,
            transition_type="overlap",
            overlap_samples=240,
        )

        total_samples = 23760

        plan = EditPlan(
            plan_id="plan_pcm_test",
            target_text="test1 test2",
            speaker_id="spk1",
            project_sample_rate=sr,
            total_output_samples=total_samples,
            segments=[seg1, seg2],
            transitions=[trans],
            sources={"src_1": src_rec},
        )
        plan.validate()

        out_wav = tmp_path / "master.wav"
        rendered_wav = render_master_pcm(plan, out_wav)

        assert rendered_wav.exists()
        info = sf.info(str(rendered_wav))
        # Invariant: exact master samples
        assert info.frames == total_samples
        assert info.samplerate == sr


def test_render_master_pcm_audio_track_selection():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sr = 48000
        dur_samples = 24000

        # Create synthetic combined (440 Hz) and vocal (880 Hz) audio
        comb_wav = tmp_path / "combined.wav"
        vocal_wav = tmp_path / "vocals.wav"
        t = np.linspace(0, float(dur_samples) / sr, dur_samples, endpoint=False)
        comb_audio = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)
        vocal_audio = 0.3 * np.sin(2 * np.pi * 880.0 * t).astype(np.float32)
        sf.write(str(comb_wav), comb_audio, sr)
        sf.write(str(vocal_wav), vocal_audio, sr)

        src_rec = SourceRecord(
            content_id="src_dual",
            source_path=str(comb_wav),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(comb_wav),
            cached_vocal_pcm_path=str(vocal_wav),
        )

        seg = PlanSegment(
            segment_index=0,
            source_content_id="src_dual",
            speaker_id="spk1",
            source_start_sample=0,
            source_end_sample=dur_samples,
            output_start_sample=0,
            output_end_sample=dur_samples,
            target_text="dual_test",
            target_phones=[],
            span_type="word",
            source_video_path=str(comb_wav),
            gain_db=0.0,
        )

        plan = EditPlan(
            plan_id="plan_dual_test",
            target_text="dual_test",
            speaker_id="spk1",
            project_sample_rate=sr,
            total_output_samples=dur_samples,
            segments=[seg],
            sources={"src_dual": src_rec},
        )

        out_comb = tmp_path / "out_comb.wav"
        render_master_pcm(plan, out_comb, audio_track="combined")
        comb_read, _ = sf.read(str(out_comb))

        out_vocal = tmp_path / "out_vocal.wav"
        render_master_pcm(plan, out_vocal, audio_track="isolated")
        vocal_read, _ = sf.read(str(out_vocal))

        # Audio tracks must differ since combined is 440Hz and vocal is 880Hz
        assert np.max(np.abs(comb_read - vocal_read)) > 0.05
        assert len(comb_read) == dur_samples
        assert len(vocal_read) == dur_samples


def test_render_sentence_mix_audio_track_both():
    from video_assembler import render_sentence_mix
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sr = 48000
        dur_samples = 48000  # 1 second

        comb_wav = tmp_path / "comb.wav"
        vocal_wav = tmp_path / "vocals.wav"
        t = np.linspace(0, 1.0, sr, endpoint=False)
        sf.write(str(comb_wav), 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32), sr)
        sf.write(str(vocal_wav), 0.5 * np.sin(2 * np.pi * 880.0 * t).astype(np.float32), sr)

        src_rec = SourceRecord(
            content_id="src_mix",
            source_path=str(comb_wav),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(comb_wav),
            cached_vocal_pcm_path=str(vocal_wav),
        )

        seg = PlanSegment(
            segment_index=0,
            source_content_id="src_mix",
            speaker_id="spk1",
            source_start_sample=0,
            source_end_sample=dur_samples,
            output_start_sample=0,
            output_end_sample=dur_samples,
            target_text="both_test",
            target_phones=[],
            span_type="word",
            source_video_path="",  # triggers fallback color frame
            video_start_frame=0,
            video_frame_count=30,
            gain_db=0.0,
        )

        plan = EditPlan(
            plan_id="plan_both_test",
            target_text="both_test",
            speaker_id="spk1",
            project_sample_rate=sr,
            total_output_samples=dur_samples,
            total_video_frames=30,
            segments=[seg],
            sources={"src_mix": src_rec},
        )

        out_mp4 = tmp_path / "result.mp4"
        res = render_sentence_mix(plan, str(out_mp4), audio_track="both", normalize_loudness=False)

        assert res == str(out_mp4)
        assert out_mp4.exists()
        assert out_mp4.stat().st_size > 0

        vocals_mp4 = tmp_path / "result_vocals.mp4"
        assert vocals_mp4.exists()
        assert vocals_mp4.stat().st_size > 0

        # Both WAV masters should also exist
        assert (tmp_path / "result.wav").exists()
        assert (tmp_path / "result_vocals.wav").exists()


def test_render_master_pcm_audio_track_recombined():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sr = 48000
        dur_samples = 48000

        comb_wav = tmp_path / "comb.wav"
        vocal_wav = tmp_path / "vocals.wav"
        inst_wav = tmp_path / "instrumental.wav"

        t = np.linspace(0, 1.0, sr, endpoint=False)
        vocal_audio = (0.3 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        inst_audio = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)
        comb_audio = vocal_audio + inst_audio

        sf.write(str(comb_wav), comb_audio, sr)
        sf.write(str(vocal_wav), vocal_audio, sr)
        sf.write(str(inst_wav), inst_audio, sr)

        src_rec = SourceRecord(
            content_id="src_recomb",
            source_path=str(comb_wav),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(comb_wav),
            cached_vocal_pcm_path=str(vocal_wav),
            cached_instrumental_pcm_path=str(inst_wav),
        )

        seg = PlanSegment(
            segment_index=0,
            source_content_id="src_recomb",
            speaker_id="spk1",
            source_start_sample=0,
            source_end_sample=dur_samples,
            output_start_sample=0,
            output_end_sample=dur_samples,
            target_text="recomb_test",
            target_phones=[],
            span_type="word",
            source_video_path="",
            gain_db=0.0,
        )

        plan = EditPlan(
            plan_id="plan_recomb_test",
            target_text="recomb_test",
            speaker_id="spk1",
            project_sample_rate=sr,
            total_output_samples=dur_samples,
            total_video_frames=30,
            segments=[seg],
            sources={"src_recomb": src_rec},
        )

        out_recomb = tmp_path / "out_recomb.wav"
        render_master_pcm(plan, out_recomb, audio_track="recombined", normalize_loudness=False)
        recomb_read, _ = sf.read(str(out_recomb))

        # Output should be the sum of vocal and instrumental
        expected_sum = vocal_audio + inst_audio
        assert np.max(np.abs(recomb_read - expected_sum)) < 1e-4
        assert len(recomb_read) == dur_samples


def test_render_sentence_mix_audio_track_all():
    from video_assembler import render_sentence_mix
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        sr = 48000
        dur_samples = 48000

        comb_wav = tmp_path / "comb.wav"
        vocal_wav = tmp_path / "vocals.wav"
        inst_wav = tmp_path / "instrumental.wav"

        t = np.linspace(0, 1.0, sr, endpoint=False)
        vocal_audio = (0.3 * np.sin(2 * np.pi * 880.0 * t)).astype(np.float32)
        inst_audio = (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

        sf.write(str(comb_wav), (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32), sr)
        sf.write(str(vocal_wav), vocal_audio, sr)
        sf.write(str(inst_wav), inst_audio, sr)

        src_rec = SourceRecord(
            content_id="src_all",
            source_path=str(comb_wav),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(comb_wav),
            cached_vocal_pcm_path=str(vocal_wav),
            cached_instrumental_pcm_path=str(inst_wav),
        )

        seg = PlanSegment(
            segment_index=0,
            source_content_id="src_all",
            speaker_id="spk1",
            source_start_sample=0,
            source_end_sample=dur_samples,
            output_start_sample=0,
            output_end_sample=dur_samples,
            target_text="all_test",
            target_phones=[],
            span_type="word",
            source_video_path="",
            video_start_frame=0,
            video_frame_count=30,
            gain_db=0.0,
        )

        plan = EditPlan(
            plan_id="plan_all_test",
            target_text="all_test",
            speaker_id="spk1",
            project_sample_rate=sr,
            total_output_samples=dur_samples,
            total_video_frames=30,
            segments=[seg],
            sources={"src_all": src_rec},
        )

        out_mp4 = tmp_path / "result.mp4"
        res = render_sentence_mix(plan, str(out_mp4), audio_track="all", normalize_loudness=False)

        assert res == str(out_mp4)
        assert out_mp4.exists()
        assert (tmp_path / "result_vocals.mp4").exists()
        assert (tmp_path / "result_recombined.mp4").exists()

        # All 3 WAV masters should exist
        assert (tmp_path / "result.wav").exists()
        assert (tmp_path / "result_vocals.wav").exists()
        assert (tmp_path / "result_recombined.wav").exists()


