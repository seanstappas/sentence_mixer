"""
Tests for source discovery, subtitle stem pairing, and SRT subtitle support.
Validates Milestone 2 (Features F5, F6, F7):
- Video file discovery (.mp4, .mkv, .mov, .avi, .webm)
- Ignored directories (.build, .mamba, .venv, .git, __pycache__)
- Subtitle stem pairing priority (.en.vtt > .vtt > .en.srt > .srt > .en-US.vtt > .en-US.srt > wildcard)
- SRT comma-timestamp parsing in timestamp_to_seconds and parse_vtt_cues
- parse_manifest_items directory and folder entry expansion with attribute inheritance
- CLI --folder argument parsing
- Incremental delta alignment cache hit speed (< 5 ms per clip)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from run_mixer import parse_manifest_items
from source_discovery import (
    EXCLUDED_DIR_NAMES,
    SUPPORTED_VIDEO_EXTENSIONS,
    discover_folder_sources,
    find_associated_subtitles,
    should_skip_dir,
)
from vtt_preprocessor import parse_vtt_cues, timestamp_to_seconds


def test_supported_video_extensions():
    """All required video formats must be supported."""
    expected = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
    assert expected.issubset(SUPPORTED_VIDEO_EXTENSIONS)


def test_excluded_dir_names():
    """Build and environment directories must be explicitly excluded."""
    expected = {".build", ".mamba", ".venv", ".git", "__pycache__"}
    assert expected.issubset(EXCLUDED_DIR_NAMES)
    for name in expected:
        assert should_skip_dir(name) is True
    assert should_skip_dir(".custom_hidden") is True
    assert should_skip_dir("normal_folder") is False


def test_discover_folder_sources_all_extensions(tmp_path: Path):
    """Recursively finds all supported video extensions case-insensitively."""
    # Create subdirectories
    sub1 = tmp_path / "season_1"
    sub1.mkdir()
    sub2 = tmp_path / "season_2" / "nested"
    sub2.mkdir(parents=True)

    # Create various video files with uppercase and lowercase suffixes
    v_mp4 = sub1 / "clip1.mp4"
    v_mkv = sub1 / "clip2.MKV"
    v_mov = sub2 / "clip3.mov"
    v_avi = sub2 / "clip4.AVI"
    v_webm = tmp_path / "clip5.webm"

    # Create non-video files that must NOT be discovered
    non_video1 = sub1 / "notes.txt"
    non_video2 = sub2 / "image.png"

    for f in [v_mp4, v_mkv, v_mov, v_avi, v_webm, non_video1, non_video2]:
        f.write_text("dummy", encoding="utf-8")

    discovered = discover_folder_sources(tmp_path)
    discovered_videos = [Path(d["video"]).name.lower() for d in discovered]

    assert len(discovered) == 5
    assert set(discovered_videos) == {"clip1.mp4", "clip2.mkv", "clip3.mov", "clip4.avi", "clip5.webm"}

    for item in discovered:
        assert "video" in item
        assert "vtt" in item
        assert item["vtt"] is None  # No subtitles created yet
        assert item["speaker_id"] is None
        assert item["single_speaker_verified"] is False


def test_discover_folder_sources_ignores_build_and_hidden(tmp_path: Path):
    """Explicitly ignores files located inside build or hidden directories."""
    valid_dir = tmp_path / "content"
    valid_dir.mkdir()
    valid_vid = valid_dir / "real.mp4"
    valid_vid.write_text("data", encoding="utf-8")

    # Create ignored directories with video files
    for ign in [".build", ".mamba", ".venv", ".git", "__pycache__", ".hidden_folder"]:
        ign_dir = tmp_path / ign / "sub"
        ign_dir.mkdir(parents=True)
        ign_vid = ign_dir / "should_ignore.mp4"
        ign_vid.write_text("data", encoding="utf-8")

    discovered = discover_folder_sources(tmp_path)
    assert len(discovered) == 1
    assert Path(discovered[0]["video"]).name == "real.mp4"


def test_discover_folder_sources_errors_on_invalid_path(tmp_path: Path):
    """Raises FileNotFoundError for nonexistent path and NotADirectoryError for file."""
    nonexistent = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError):
        discover_folder_sources(nonexistent)

    regular_file = tmp_path / "file.txt"
    regular_file.write_text("hello", encoding="utf-8")
    with pytest.raises(NotADirectoryError):
        discover_folder_sources(regular_file)


def test_subtitle_priority_en_vtt_over_all(tmp_path: Path):
    """Priority 1 (.en.vtt) beats all other candidates."""
    vid = tmp_path / "scene.mp4"
    vid.write_text("video", encoding="utf-8")

    sub_en_vtt = tmp_path / "scene.en.vtt"
    sub_vtt = tmp_path / "scene.vtt"
    sub_en_srt = tmp_path / "scene.en.srt"
    sub_srt = tmp_path / "scene.srt"

    for s in [sub_en_vtt, sub_vtt, sub_en_srt, sub_srt]:
        s.write_text("sub", encoding="utf-8")

    match = find_associated_subtitles(vid)
    assert match is not None
    assert Path(match).name == "scene.en.vtt"


def test_subtitle_priority_vtt_over_srt(tmp_path: Path):
    """Priority 2 (.vtt) beats .en.srt and .srt."""
    vid = tmp_path / "scene.mp4"
    vid.write_text("video", encoding="utf-8")

    sub_vtt = tmp_path / "scene.vtt"
    sub_en_srt = tmp_path / "scene.en.srt"
    sub_srt = tmp_path / "scene.srt"

    for s in [sub_vtt, sub_en_srt, sub_srt]:
        s.write_text("sub", encoding="utf-8")

    match = find_associated_subtitles(vid)
    assert match is not None
    assert Path(match).name == "scene.vtt"


def test_subtitle_priority_en_srt_over_srt(tmp_path: Path):
    """Priority 3 (.en.srt) beats .srt and .en-US variants."""
    vid = tmp_path / "scene.mp4"
    vid.write_text("video", encoding="utf-8")

    sub_en_srt = tmp_path / "scene.en.srt"
    sub_srt = tmp_path / "scene.srt"
    sub_en_us_vtt = tmp_path / "scene.en-US.vtt"

    for s in [sub_en_srt, sub_srt, sub_en_us_vtt]:
        s.write_text("sub", encoding="utf-8")

    match = find_associated_subtitles(vid)
    assert match is not None
    assert Path(match).name == "scene.en.srt"


def test_subtitle_priority_srt_over_en_us(tmp_path: Path):
    """Priority 4 (.srt) beats .en-US.vtt and .en-US.srt."""
    vid = tmp_path / "scene.mp4"
    vid.write_text("video", encoding="utf-8")

    sub_srt = tmp_path / "scene.srt"
    sub_en_us_vtt = tmp_path / "scene.en-US.vtt"
    sub_en_us_srt = tmp_path / "scene.en-US.srt"

    for s in [sub_srt, sub_en_us_vtt, sub_en_us_srt]:
        s.write_text("sub", encoding="utf-8")

    match = find_associated_subtitles(vid)
    assert match is not None
    assert Path(match).name == "scene.srt"


def test_subtitle_priority_en_us_vtt_over_en_us_srt(tmp_path: Path):
    """Priority 5 (.en-US.vtt) beats .en-US.srt."""
    vid = tmp_path / "scene.mp4"
    vid.write_text("video", encoding="utf-8")

    sub_en_us_vtt = tmp_path / "scene.en-US.vtt"
    sub_en_us_srt = tmp_path / "scene.en-US.srt"

    for s in [sub_en_us_vtt, sub_en_us_srt]:
        s.write_text("sub", encoding="utf-8")

    match = find_associated_subtitles(vid)
    assert match is not None
    assert Path(match).name == "scene.en-US.vtt"


def test_subtitle_priority_fallback_wildcard(tmp_path: Path):
    """Priority 7 matches any {stem}*.vtt, then {stem}*.srt alphabetically."""
    vid = tmp_path / "episode_01.mkv"
    vid.write_text("video", encoding="utf-8")

    sub_custom_vtt = tmp_path / "episode_01_cleaned.vtt"
    sub_custom_srt = tmp_path / "episode_01_cleaned.srt"
    sub_custom_vtt.write_text("vtt", encoding="utf-8")
    sub_custom_srt.write_text("srt", encoding="utf-8")

    # .vtt fallback beats .srt fallback
    match = find_associated_subtitles(vid)
    assert match is not None
    assert Path(match).name == "episode_01_cleaned.vtt"


def test_srt_comma_timestamps_in_timestamp_to_seconds():
    """timestamp_to_seconds correctly converts SRT comma-separated timestamps."""
    assert timestamp_to_seconds("00:00:01,000") == pytest.approx(1.0)
    assert timestamp_to_seconds("00:01:23,456") == pytest.approx(83.456)
    assert timestamp_to_seconds("01:02:03,040") == pytest.approx(3723.040)
    assert timestamp_to_seconds("04:05,678") == pytest.approx(245.678)
    assert timestamp_to_seconds("  12,345  ") == pytest.approx(12.345)

    # Standard period-separated timestamps continue to work identically
    assert timestamp_to_seconds("00:00:01.000") == pytest.approx(1.0)
    assert timestamp_to_seconds("00:01:23.456") == pytest.approx(83.456)


def test_parse_srt_cues_with_commas_and_multiline(tmp_path: Path):
    """parse_vtt_cues parses SRT files with numeric index headers, comma timestamps, and multiline cues."""
    srt_content = """1
00:00:01,000 --> 00:00:03,500
Hello there!
General Kenobi.

2
00:00:04,200 --> 00:00:07,800
<i>You are a bold one.</i>
Kill him!
"""
    srt_file = tmp_path / "test.srt"
    srt_file.write_text(srt_content, encoding="utf-8")

    cues = parse_vtt_cues(str(srt_file))
    assert len(cues) == 2

    assert cues[0]["start_sec"] == pytest.approx(1.0)
    assert cues[0]["end_sec"] == pytest.approx(3.5)
    assert cues[0]["duration"] == pytest.approx(2.5)
    assert cues[0]["text"] == "Hello there! General Kenobi."

    assert cues[1]["start_sec"] == pytest.approx(4.2)
    assert cues[1]["end_sec"] == pytest.approx(7.8)
    assert cues[1]["duration"] == pytest.approx(3.6)
    assert cues[1]["text"] == "You are a bold one. Kill him!"


def test_parse_manifest_items_direct_folder(tmp_path: Path):
    """parse_manifest_items directly accepts a folder path and discovers its videos and subtitles."""
    v1 = tmp_path / "clip_a.mp4"
    s1 = tmp_path / "clip_a.en.vtt"
    v2 = tmp_path / "clip_b.webm"
    s2 = tmp_path / "clip_b.srt"
    v1.write_text("v1", encoding="utf-8")
    s1.write_text("s1", encoding="utf-8")
    v2.write_text("v2", encoding="utf-8")
    s2.write_text("s2", encoding="utf-8")

    items = parse_manifest_items(str(tmp_path))
    assert len(items) == 2
    videos = [Path(it["video"]).name for it in items]
    assert videos == ["clip_a.mp4", "clip_b.webm"]

    assert Path(items[0]["vtt"]).name == "clip_a.en.vtt"
    assert Path(items[1]["vtt"]).name == "clip_b.srt"


def test_parse_manifest_items_json_folder_expansion(tmp_path: Path):
    """JSON manifest containing a 'folder' entry expands its sources with inherited attributes."""
    media_dir = tmp_path / "media_sources"
    media_dir.mkdir()

    v1 = media_dir / "vid1.mp4"
    s1 = media_dir / "vid1.vtt"
    v2 = media_dir / "vid2.mkv"
    s2 = media_dir / "vid2.en.srt"
    for f in [v1, s1, v2, s2]:
        f.write_text("data", encoding="utf-8")

    # Outside standalone video
    ext_v = tmp_path / "extra.mov"
    ext_v.write_text("data", encoding="utf-8")

    manifest_data = [
        {
            "folder": "media_sources",
            "speaker_id": "anakin",
            "single_speaker_verified": True,
            "recording_id": "rec_01",
        },
        {
            "video": "extra.mov",
            "speaker_id": "obi-wan",
            "single_speaker_verified": False,
        },
    ]

    manifest_file = tmp_path / "sources_manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    parsed = parse_manifest_items(str(manifest_file))
    assert len(parsed) == 3

    # Expanded folder items inherit speaker_id and single_speaker_verified
    vid1_item = next(it for it in parsed if Path(it["video"]).name == "vid1.mp4")
    assert vid1_item["speaker_id"] == "anakin"
    assert vid1_item["single_speaker_verified"] is True
    assert vid1_item["recording_id"] == "rec_01"
    assert Path(vid1_item["vtt"]).name == "vid1.vtt"

    vid2_item = next(it for it in parsed if Path(it["video"]).name == "vid2.mkv")
    assert vid2_item["speaker_id"] == "anakin"
    assert vid2_item["single_speaker_verified"] is True
    assert Path(vid2_item["vtt"]).name == "vid2.en.srt"

    # Standalone video item
    extra_item = next(it for it in parsed if Path(it["video"]).name == "extra.mov")
    assert extra_item["speaker_id"] == "obi-wan"
    assert extra_item["single_speaker_verified"] is False


def test_parse_manifest_items_string_directory(tmp_path: Path):
    """JSON manifest containing a bare string path that points to a directory expands it."""
    sub_dir = tmp_path / "clips"
    sub_dir.mkdir()
    (sub_dir / "video1.avi").write_text("avi", encoding="utf-8")
    (sub_dir / "video1.srt").write_text("srt", encoding="utf-8")

    manifest_data = ["clips"]
    manifest_file = tmp_path / "manifest.json"
    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f)

    items = parse_manifest_items(str(manifest_file))
    assert len(items) == 1
    assert Path(items[0]["video"]).name == "video1.avi"
    assert Path(items[0]["vtt"]).name == "video1.srt"


def test_cli_p_mix_folder_argument():
    """p_mix parser accepts --folder and populates args.folder."""
    import run_mixer

    parser = run_mixer.build_cli_parser()
    args = parser.parse_args(["mix", "--folder", "input/test_dir", "--text", "hello there"])
    assert args.folder == ["input/test_dir"]
    assert args.text == "hello there"


def test_incremental_delta_alignment_cache_hit_performance(tmp_path: Path, monkeypatch):
    """
    Verifies incremental delta alignment hits Layer 1 cache in milliseconds (< 5 ms per clip).
    Simulates repeated folder ingestion where inventory and metadata already exist.
    """
    from cache import compute_alignment_cache_key, compute_labeled_inventory_cache_key, hash_file_content
    from run_mixer import ensure_source_aligned
    import lexicon

    cache_dir = tmp_path / ".build"
    cache_dir.mkdir()
    monkeypatch.setattr("run_mixer.CACHE_BASE_DIR", cache_dir)

    # Create 3 clips in a folder
    source_folder = tmp_path / "watched_folder"
    source_folder.mkdir()

    items = []
    for i in range(3):
        v_file = source_folder / f"clip_{i}.mp4"
        s_file = source_folder / f"clip_{i}.vtt"
        v_file.write_bytes(f"dummy_video_stream_content_{i}".encode("utf-8"))
        s_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n", encoding="utf-8")
        items.append({"video": str(v_file), "vtt": str(s_file)})

    # Mock stage dictionary
    dict_hash = "mock_dict_hash_12345"
    monkeypatch.setattr(
        lexicon,
        "stage_mfa_dictionary",
        lambda *args, **kwargs: (kwargs.get("output_path", Path("staged.dict")), dict_hash),
    )

    # Pre-populate valid Layer 1 cached inventory for all 3 clips
    for item in items:
        v_path = Path(item["video"]).resolve()
        vtt_path = Path(item["vtt"]).resolve()
        from cache import compute_media_hash
        from run_mixer import get_source_id

        src_id = get_source_id(v_path)
        media_hash = compute_media_hash(v_path, cache_dir=cache_dir)
        vtt_hash = hash_file_content(str(vtt_path))

        align_key = compute_alignment_cache_key(
            media_hash=media_hash,
            selected_audio_stream=0,
            selected_video_stream=0,
            transcript_hash=vtt_hash,
            dict_hash=dict_hash,
            acoustic_model="english_us_arpa",
            mfa_options={},
            preprocessor_version="2.0",
        )
        inv_key = compute_labeled_inventory_cache_key(
            alignment_key=align_key,
            speaker_id=None,
            single_speaker_verified=False,
            assume_single_speaker=False,
            speaker_intervals_hash="",
        )

        inv_dir = cache_dir / "inventory" / src_id
        inv_dir.mkdir(parents=True, exist_ok=True)
        (inv_dir / "units_index.json").write_text(json.dumps({"words": {}, "events": {}}), encoding="utf-8")
        (inv_dir / "inventory_metadata.json").write_text(
            json.dumps({"inventory_key": inv_key, "alignment_key": align_key}), encoding="utf-8"
        )

    # First pass: initial folder ingestion
    for item in items:
        ensure_source_aligned(
            item["video"],
            item["vtt"],
            force_realign=False,
            assume_single_speaker=False,
        )

    # Second pass: repeated folder ingestion measurement
    t_start = time.perf_counter()
    for item in items:
        inv_res = ensure_source_aligned(
            item["video"],
            item["vtt"],
            force_realign=False,
            assume_single_speaker=False,
        )
        assert Path(inv_res).exists()
    t_total = time.perf_counter() - t_start

    ms_per_clip = (t_total / len(items)) * 1000.0
    print(f"\n[Performance] Cache hit latency: {ms_per_clip:.3f} ms per clip (threshold: < 5.0 ms)")

    # Assert cache hit latency is well below 5.0 ms per clip
    assert ms_per_clip < 5.0, f"Cache hit latency too slow: {ms_per_clip:.3f} ms per clip (expected < 5 ms)"
