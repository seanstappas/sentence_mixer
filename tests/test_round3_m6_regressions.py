"""
Regression tests for Milestone 6 (B12, B13, U28-U29).
Verifies:
- U28: Pause segment; 24 fps source in a 30 fps sequence; mono master:
  Hold still in XML matching the rendered frame; native in/out; one audio track.
- U29: Finalized plan rendered twice:
  Every non-silence segment has a float gain_db; identical WAVs; plan hash unchanged;
  renderer rejects undecided gains (None).
"""

from __future__ import annotations

import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import soundfile as sf

from contracts import (
    EditPlan,
    PlanSegment,
    RationalTime,
    SourceRecord,
)
from edit_plan import finalize_edit_plan
from fcp_xml_generator import generate_fcp_xml
from video_assembler import render_master_pcm


def test_u28_pause_segment_24fps_source_30fps_seq_mono_master(tmp_path: Path):
    """
    U28: Pause segment; 24 fps source in a 30 fps sequence; mono master:
    Hold still in XML matching the rendered frame; native in/out; one audio track.
    """
    # 1. Create a mono master WAV (1 channel)
    sr = 48000
    dur_samples = 48000
    t = np.linspace(0, 1.0, dur_samples, endpoint=False)
    mono_audio = (0.3 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    master_wav = tmp_path / "master_mono.wav"
    sf.write(str(master_wav), mono_audio, sr)

    # 2. Source video: 24 fps source
    src_video_path = tmp_path / "source_24fps.mp4"
    # Create synthetic 24 fps video with test tone
    import subprocess
    cmd_gen = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", "testsrc=duration=2:size=1920x1080:rate=24",
        "-f", "lavfi", "-i", "sine=frequency=440:duration=2:sample_rate=48000",
        "-c:v", "libx264", "-c:a", "pcm_s16le",
        "-r", "24",
        str(src_video_path),
    ]
    subprocess.run(cmd_gen, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    src_rec = SourceRecord(
        content_id="src_24",
        source_path=str(src_video_path),
        native_sample_rate=sr,
        native_channels=1,
        video_fps={"numerator": 24, "denominator": 1},
    )

    # Segment 0: speech cut (15 sequence frames at 30 fps = 0.5s -> in source at 24 fps: 12 frames)
    seg0 = PlanSegment(
        segment_index=0,
        source_content_id="src_24",
        speaker_id="speaker_a",
        source_start_sample=0,
        source_end_sample=24000,
        output_start_sample=0,
        output_end_sample=24000,
        target_text="speech",
        target_phones=["S", "P", "IY1", "CH"],
        span_type="word",
        source_video_path=str(src_video_path),
        video_start_frame=0,
        video_frame_count=15,  # 15 frames at 30 fps
        video_source_pts_sec=0.25,  # 0.25s start in source
        gain_db=0.0,
    )

    # Segment 1: pause segment / hold still (15 sequence frames at 30 fps = 0.5s)
    seg1 = PlanSegment(
        segment_index=1,
        source_content_id="src_24",
        speaker_id="speaker_a",
        source_start_sample=24000,
        source_end_sample=48000,
        output_start_sample=24000,
        output_end_sample=48000,
        target_text="[pause]",
        target_phones=["sil"],
        span_type="pause",
        source_video_path=str(src_video_path),
        is_silence=True,
        freeze_video=True,
        video_start_frame=15,
        video_frame_count=15,
        video_source_pts_sec=0.75,
        gain_db=0.0,
    )

    plan = EditPlan(
        plan_id="plan_u28",
        target_text="speech [pause]",
        speaker_id="speaker_a",
        video_fps=RationalTime(30, 1),  # Sequence FPS is 30
        project_sample_rate=sr,
        total_output_samples=48000,
        total_video_frames=30,
        segments=[seg0, seg1],
        sources={"src_24": src_rec},
    )
    plan.validate()

    out_xml = tmp_path / "timeline_u28.xml"
    generate_fcp_xml(
        plan,
        str(out_xml),
        sequence_name="U28_Test",
        master_wav_path=str(master_wav),
    )
    assert out_xml.exists()

    tree = ET.parse(str(out_xml))
    root = tree.getroot()

    # 1. Check sequence FPS (30 fps)
    seq_timebase = root.find(".//sequence/rate/timebase")
    assert seq_timebase is not None and seq_timebase.text == "30"

    # 2. Check audio tracks: mono master must produce exactly ONE track
    audio_tracks = root.findall(".//media/audio/track")
    assert len(audio_tracks) == 1, f"Expected 1 audio track for mono master, got {len(audio_tracks)}"
    aclip = audio_tracks[0].find("clipitem")
    assert aclip is not None
    assert "Master_Audio" in aclip.find("name").text

    # 3. Check video clipitems: 2 items (one speech, one hold)
    vclips = root.findall(".//media/video/track/clipitem")
    assert len(vclips) == 2

    # Speech clip: native source rate 24 fps
    vclip_speech = vclips[0]
    speech_file = vclip_speech.find("file")
    assert speech_file is not None
    speech_file_rate = speech_file.find("rate/timebase")
    assert speech_file_rate is not None and speech_file_rate.text == "24", f"Expected file timebase 24, got {speech_file_rate.text}"

    # Native in/out: start_pts=0.25s * 24fps = 6 in_frame; 0.5s duration * 24fps = 12 -> out_frame = 18
    in_val = int(vclip_speech.find("in").text)
    out_val = int(vclip_speech.find("out").text)
    assert in_val == 6, f"Expected native in_frame=6 for 0.25s @ 24fps, got {in_val}"
    assert out_val == 18, f"Expected native out_frame=18 for 0.75s @ 24fps, got {out_val}"

    # Hold clip: pause represented as still image
    vclip_hold = vclips[1]
    assert "hold" in vclip_hold.find("name").text
    hold_file = vclip_hold.find("file")
    assert hold_file is not None
    assert "hold_0001.png" in hold_file.find("name").text
    # Still image exists beside XML
    holds_dir = out_xml.parent / f"{out_xml.stem}_holds"
    hold_png = holds_dir / "hold_0001.png"
    assert hold_png.exists() and hold_png.stat().st_size > 0


def test_u29_freeze_dsp_decisions_in_plan_and_renderer_checks(tmp_path: Path):
    """
    U29: Finalized plan rendered twice:
    - Every non-silence segment has a float gain_db
    - Renderer rejects undecided gains (None) with ValueError
    - Rendering does NOT mutate plan or plan hash
    - Identical WAVs produced across multiple renders
    """
    sr = 48000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    # Source A: quiet tone (-26 dBFS RMS approx)
    sig_a = (0.05 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_a = tmp_path / "src_a.wav"
    sf.write(str(wav_a), sig_a, sr)

    src_rec = SourceRecord(
        content_id="src_a",
        source_path=str(wav_a),
        cached_pcm_path=str(wav_a),
        native_sample_rate=sr,
        native_channels=1,
    )

    # 1. Test renderer rejects undecided gain (seg.gain_db is None)
    seg_undecided = PlanSegment(
        segment_index=0,
        source_content_id="src_a",
        speaker_id="speaker_a",
        source_start_sample=0,
        source_end_sample=len(sig_a),
        output_start_sample=0,
        output_end_sample=len(sig_a),
        target_text="undecided",
        target_phones=["AH0"],
        span_type="word",
        source_video_path=str(wav_a),
        gain_db=None,  # UNDECIDED!
    )
    plan_undecided = EditPlan(
        plan_id="plan_undecided",
        target_text="undecided",
        speaker_id="speaker_a",
        project_sample_rate=sr,
        total_output_samples=len(sig_a),
        segments=[seg_undecided],
        sources={"src_a": src_rec},
    )

    out_wav_test = tmp_path / "out_should_fail.wav"
    with pytest.raises(ValueError, match="(?i)undecided gain_db"):
        render_master_pcm(plan_undecided, str(out_wav_test))

    # 2. Finalize edit plan: freezes DSP decisions into plan
    finalize_edit_plan(plan_undecided, normalize_loudness=True)
    assert plan_undecided.segments[0].gain_db is not None
    assert isinstance(plan_undecided.segments[0].gain_db, float)
    # Quiet signal should receive positive gain
    assert plan_undecided.segments[0].gain_db > 0.0

    # Save plan and record hash
    plan_json_1 = json.dumps(plan_undecided.to_dict(), sort_keys=True)

    # 3. Render 1
    render1_wav = tmp_path / "render1.wav"
    render_master_pcm(plan_undecided, str(render1_wav), normalize_loudness=False)
    assert render1_wav.exists()

    plan_json_after_render1 = json.dumps(plan_undecided.to_dict(), sort_keys=True)
    assert plan_json_1 == plan_json_after_render1, "Rendering mutated the EditPlan!"

    # 4. Render 2
    render2_wav = tmp_path / "render2.wav"
    render_master_pcm(plan_undecided, str(render2_wav), normalize_loudness=False)
    assert render2_wav.exists()

    plan_json_after_render2 = json.dumps(plan_undecided.to_dict(), sort_keys=True)
    assert plan_json_1 == plan_json_after_render2, "Rendering mutated the EditPlan!"

    # Check identical audio bytes produced
    bytes1 = render1_wav.read_bytes()
    bytes2 = render2_wav.read_bytes()
    assert bytes1 == bytes2, "Multiple renders of finalized plan produced non-identical WAV files!"
