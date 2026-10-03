import json
from pathlib import Path
from src.contracts import EditPlan, PlanSegment, RationalTime
from src.subtitles import (
    generate_subtitles,
    export_subtitles,
    format_timestamp_srt,
    format_timestamp_vtt,
)


def _create_sample_plan() -> EditPlan:
    plan = EditPlan(
        target_text="I am going to invest in bitcoin",
        project_sample_rate=48000,
        video_fps=RationalTime(30, 1),
        total_output_samples=96000,
        total_video_frames=60,
    )
    # Segment 0: Pause (0 to 0.1s: 0 to 4800 samples)
    s0 = PlanSegment(
        segment_index=0,
        source_content_id="silence",
        speaker_id="anakin",
        source_start_sample=0,
        source_end_sample=4800,
        output_start_sample=0,
        output_end_sample=4800,
        target_text="[pause]",
        target_phones=["sil"],
        span_type="pause",
        source_video_path="",
        is_silence=True,
    )
    # Segment 1: Phrase "I am" (0.1s to 0.6s: 4800 to 28800 samples)
    s1 = PlanSegment(
        segment_index=1,
        source_content_id="src1",
        speaker_id="anakin",
        source_start_sample=1000,
        source_end_sample=25000,
        output_start_sample=4800,
        output_end_sample=28800,
        target_text="i am",
        target_phones=["AY1", "AE1", "M"],
        span_type="phrase",
        source_video_path="v1.mp4",
        is_silence=False,
    )
    # Segment 2: Word "going" (0.6s to 0.9s: 28800 to 43200 samples)
    s2 = PlanSegment(
        segment_index=2,
        source_content_id="src1",
        speaker_id="anakin",
        source_start_sample=25000,
        source_end_sample=39400,
        output_start_sample=28800,
        output_end_sample=43200,
        target_text="going",
        target_phones=["G", "OW1", "IH0", "NG"],
        span_type="word",
        source_video_path="v1.mp4",
        is_silence=False,
    )
    # Segment 3: Pause gap (0.9s to 1.1s: 43200 to 52800 samples)
    s3 = PlanSegment(
        segment_index=3,
        source_content_id="silence",
        speaker_id="anakin",
        source_start_sample=0,
        source_end_sample=9600,
        output_start_sample=43200,
        output_end_sample=52800,
        target_text="<gap:200ms>",
        target_phones=["sil"],
        span_type="pause",
        source_video_path="",
        is_silence=True,
    )
    # Segment 4: Word "to" (1.1s to 1.3s: 52800 to 62400 samples)
    s4 = PlanSegment(
        segment_index=4,
        source_content_id="src1",
        speaker_id="anakin",
        source_start_sample=40000,
        source_end_sample=49600,
        output_start_sample=52800,
        output_end_sample=62400,
        target_text="to",
        target_phones=["T", "UW1"],
        span_type="word",
        source_video_path="v1.mp4",
        is_silence=False,
    )
    # Segment 5: Subwords forming "invest" (1.3s to 1.7s: 62400 to 81600 samples)
    s5 = PlanSegment(
        segment_index=5,
        source_content_id="src1",
        speaker_id="anakin",
        source_start_sample=50000,
        source_end_sample=69200,
        output_start_sample=62400,
        output_end_sample=81600,
        target_text="investigation[IH0-N-V-EH1-S-T]",
        target_phones=["IH0", "N", "V", "EH1", "S", "T"],
        span_type="subword",
        source_video_path="v1.mp4",
        is_silence=False,
    )
    plan.segments = [s0, s1, s2, s3, s4, s5]
    return plan


def test_format_timestamp_srt():
    assert format_timestamp_srt(0.0) == "00:00:00,000"
    assert format_timestamp_srt(1.234) == "00:00:01,234"
    assert format_timestamp_srt(65.5) == "00:01:05,500"
    assert format_timestamp_srt(3661.009) == "01:01:01,009"


def test_format_timestamp_vtt():
    assert format_timestamp_vtt(0.0) == "00:00:00.000"
    assert format_timestamp_vtt(1.234) == "00:00:01.234"
    assert format_timestamp_vtt(65.5) == "00:01:05.500"


def test_generate_subtitles_srt_phrase_level():
    plan = _create_sample_plan()
    srt_content = generate_subtitles(plan, format="srt", cue_mode="phrase")

    assert "1\n00:00:00,100 --> 00:00:00,900" in srt_content
    assert "i am going" in srt_content.lower()
    assert "2\n00:00:01,100 --> 00:00:01,700" in srt_content
    assert "to invest" in srt_content.lower()


def test_generate_subtitles_vtt_format():
    plan = _create_sample_plan()
    vtt_content = generate_subtitles(plan, format="vtt", cue_mode="phrase")

    assert vtt_content.startswith("WEBVTT")
    assert "00:00:00.100 --> 00:00:00.900" in vtt_content
    assert "00:00:01.100 --> 00:00:01.700" in vtt_content


def test_generate_subtitles_word_level():
    plan = _create_sample_plan()
    srt_content = generate_subtitles(plan, format="srt", cue_mode="word")

    # In word mode, each spoken unit gets its own cue
    assert "00:00:00,100 --> 00:00:00,600" in srt_content
    assert "00:00:00,600 --> 00:00:00,900" in srt_content
    assert "going" in srt_content.lower()
    assert "00:00:01,100 --> 00:00:01,300" in srt_content
    assert "to" in srt_content.lower()


def test_export_subtitles_file(tmp_path: Path):
    plan = _create_sample_plan()
    out_file = tmp_path / "test_mix.srt"
    res_path = export_subtitles(plan, out_file)

    assert res_path.exists()
    content = res_path.read_text(encoding="utf-8")
    assert "00:00:00,100 --> 00:00:00,900" in content


def test_subtitles_from_legacy_units():
    units = [
        {"label": "hello", "start_sec": 0.0, "end_sec": 0.5},
        {"label": "there", "start_sec": 1.0, "end_sec": 1.5},
    ]
    srt = generate_subtitles(units, format="srt", cue_mode="phrase")
    assert "hello there" in srt
    assert "00:00:00,000 --> 00:00:01,000" in srt


def test_burn_subtitles_to_video_smoke(tmp_path: Path):
    import subprocess
    from src.subtitles import burn_subtitles_to_video

    vid_path = tmp_path / "dummy.mp4"
    sub_path = tmp_path / "dummy.srt"
    out_path = tmp_path / "burned.mp4"

    # Generate small 0.5s dummy black video with silent audio
    subprocess.run([
        "ffmpeg", "-y", "-f", "lavfi", "-i", "color=c=black:s=160x120:d=0.5",
        "-f", "lavfi", "-i", "anullsrc=r=48000:cl=mono", "-t", "0.5",
        "-c:v", "libx264", "-c:a", "aac", str(vid_path)
    ], check=True, capture_output=True)

    sub_path.write_text("1\n00:00:00,000 --> 00:00:00,500\nHello Test\n", encoding="utf-8")
    res = burn_subtitles_to_video(vid_path, sub_path, out_path, style="classic")
    assert res.exists()
    assert res.stat().st_size > 0


def test_run_mixer_subtitles_cli_integration(tmp_path: Path):
    import subprocess
    import sys
    from tests.test_round3_m1_regressions import create_synthetic_media, create_synthetic_inventory_file

    media = create_synthetic_media(tmp_path, "test.mp4", duration_sec=1.0)
    inv_file = tmp_path / "units_index.json"
    create_synthetic_inventory_file(
        inv_file, media, ["obi-wan"], {"obi-wan": ["hello", "there"]}
    )

    out_mp4 = tmp_path / "mix.mp4"
    out_srt = tmp_path / "mix.srt"

    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "run_mixer.py"),
        "mix",
        "--inventory", str(inv_file),
        "--text", "hello there",
        "--selector", "span",
        "--render", str(out_mp4),
        "--subtitles",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Mix failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert out_mp4.exists()
    assert out_srt.exists()
    content = out_srt.read_text(encoding="utf-8")
    assert "hello there" in content

    # Check completion manifest
    manifest_p = tmp_path / "mix.manifest.json"
    assert manifest_p.exists()
    data = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert "subtitles" in data["artifacts"]


def test_run_mixer_burn_subtitles_cli_integration(tmp_path: Path):
    import subprocess
    import sys
    from tests.test_round3_m1_regressions import create_synthetic_media, create_synthetic_inventory_file

    media = create_synthetic_media(tmp_path, "test.mp4", duration_sec=1.0)
    inv_file = tmp_path / "units_index.json"
    create_synthetic_inventory_file(
        inv_file, media, ["obi-wan"], {"obi-wan": ["hello", "there"]}
    )

    out_mp4 = tmp_path / "mix_burned.mp4"

    repo_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        str(repo_root / "run_mixer.py"),
        "mix",
        "--inventory", str(inv_file),
        "--text", "hello there",
        "--selector", "span",
        "--render", str(out_mp4),
        "--burn-subtitles",
        "--subtitle-style", "classic",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"Mix failed:\nSTDOUT: {res.stdout}\nSTDERR: {res.stderr}"
    assert out_mp4.exists()
    assert out_mp4.stat().st_size > 0
    # Standalone .srt was not requested, so it shouldn't linger
    assert not (tmp_path / "mix_burned.srt").exists()
