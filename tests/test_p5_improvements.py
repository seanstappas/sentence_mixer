"""
Unit and integration tests for Phase 5 improvements (Fixes P5.1 – P5.4).
"""

import numpy as np
import soundfile as sf
from pathlib import Path

from contracts import EditPlan, PlanSegment, SourceRecord
from video_assembler import render_master_pcm
from tools.quality_check import detect_splice_clicks


def test_p5_2_silence_cosine_ramp(tmp_path: Path):
    """Verify that post-loudnorm silence zeroing applies smooth cosine tapers at s_start and s_end."""
    sr = 48000
    total_samples = 48000

    # 1 second of continuous 440Hz tone
    t = np.linspace(0, 1.0, total_samples, endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440.0 * t).astype(np.float32)

    src_wav = tmp_path / "source.wav"
    sf.write(str(src_wav), audio, sr)

    src_rec = SourceRecord(
        content_id="src_1",
        source_path=str(src_wav),
        native_sample_rate=sr,
        native_channels=1,
        cached_pcm_path=str(src_wav),
    )

    # Segment 1: speech [0 to 12000]
    seg1 = PlanSegment(
        segment_index=0,
        source_content_id="src_1",
        speaker_id="spk1",
        source_start_sample=0,
        source_end_sample=12000,
        output_start_sample=0,
        output_end_sample=12000,
        target_text="speech1",
        target_phones=[],
        span_type="word",
        source_video_path="",
        gain_db=0.0,
    )

    # Segment 2: silence [12000 to 24000]
    seg2 = PlanSegment(
        segment_index=1,
        source_content_id="",
        speaker_id="",
        source_start_sample=0,
        source_end_sample=12000,
        output_start_sample=12000,
        output_end_sample=24000,
        target_text="",
        target_phones=[],
        span_type="pause",
        source_video_path="",
        is_silence=True,
    )

    # Segment 3: speech [24000 to 36000]
    seg3 = PlanSegment(
        segment_index=2,
        source_content_id="src_1",
        speaker_id="spk1",
        source_start_sample=0,
        source_end_sample=12000,
        output_start_sample=24000,
        output_end_sample=36000,
        target_text="speech2",
        target_phones=[],
        span_type="word",
        source_video_path="",
        gain_db=0.0,
    )

    plan = EditPlan(
        plan_id="plan_silence_ramp",
        target_text="speech1 speech2",
        speaker_id="spk1",
        project_sample_rate=sr,
        total_output_samples=36000,
        segments=[seg1, seg2, seg3],
        sources={"src_1": src_rec},
    )

    out_wav = tmp_path / "master_silence.wav"
    render_master_pcm(plan, out_wav, normalize_loudness=True)

    read_data, _ = sf.read(str(out_wav), dtype="float32")
    assert len(read_data) == 36000

    # Interior of silence interval [12000 to 24000] must be exactly 0.0
    assert np.all(read_data[12500:23500] == 0.0)

    # Boundary right at 12000 must be zeroed out smoothly
    assert read_data[12000] == 0.0
    assert read_data[24000] == 0.0

    # 3ms ramp length is 144 samples
    # Check that in [12000 - 144, 12000], signal smoothly decays to 0.0 without step jump
    ramp_in = read_data[12000 - 144:12000]
    diffs = np.abs(np.diff(ramp_in))
    assert np.max(diffs) < 0.05  # Smooth transition, no jump


def test_p5_3_artifact_keys_and_alias_resolution(tmp_path: Path):
    # Verify subtitle key distinction
    srt_file = tmp_path / "mix.srt"
    vtt_file = tmp_path / "mix.vtt"

    def artifact_key(f: Path) -> str:
        suf = f.suffix.lower()
        if suf == ".srt":
            return "subtitles_srt"
        if suf == ".vtt":
            return "subtitles_vtt"
        return f.stem

    assert artifact_key(srt_file) == "subtitles_srt"
    assert artifact_key(vtt_file) == "subtitles_vtt"
    assert artifact_key(srt_file) != artifact_key(vtt_file)

    # Verify CharacterResolver alias resolution
    from subtitle_parsing import CharacterResolver
    char_config = {
        "speakers": {
            "palpatine": {
                "display": "Palpatine",
                "aliases": ["Chancellor Palpatine", "The Emperor", "Darth Sidious"],
            }
        }
    }
    resolver = CharacterResolver(char_config)
    resolved_id, kind = resolver.resolve("Chancellor Palpatine")
    assert resolved_id == "palpatine"
    assert kind == "speaker"


def test_p5_4_click_detection_at_boundaries(tmp_path: Path):
    """Verify click detector flags transient spikes at transitions and passes smooth audio."""
    sr = 48000
    total_samples = 48000

    # Baseline smooth sine wave
    t = np.linspace(0, 1.0, total_samples, endpoint=False)
    smooth_audio = 0.3 * np.sin(2 * np.pi * 300.0 * t).astype(np.float32)

    smooth_wav = tmp_path / "smooth.wav"
    sf.write(str(smooth_wav), smooth_audio, sr)

    plan_data = {
        "project_sample_rate": sr,
        "segments": [
            {"output_start_sample": 0, "output_end_sample": 24000},
            {"output_start_sample": 24000, "output_end_sample": 48000},
        ],
        "transitions": [
            {"transition_index": 0, "to_segment_index": 1},
        ],
    }

    # Clean audio should pass click detection with 0 clicks
    passed, clicks = detect_splice_clicks(smooth_wav, plan_data)
    assert passed is True
    assert len(clicks) == 0

    # Inject an audible click spike at sample 24000
    clicked_audio = smooth_audio.copy()
    clicked_audio[24000] = 0.95  # Massive step jump spike
    clicked_audio[24001] = -0.90

    clicked_wav = tmp_path / "clicked.wav"
    sf.write(str(clicked_wav), clicked_audio, sr)

    passed_bad, clicks_bad = detect_splice_clicks(clicked_wav, plan_data)
    assert passed_bad is False
    assert len(clicks_bad) >= 1
    assert clicks_bad[0]["transition_index"] == 0
