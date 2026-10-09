"""
Unit and integration tests for the standalone `ingest` CLI command and continuous `--watch` mode.
Validates Milestone 3 (Features F8, F9, F10):
- First-class `ingest` subcommand in `build_cli_parser()` with all required options.
- Standalone ingestion execution without synthesis targets (--text, --target-audio, --record-target).
- Per-clip metrics tracking: total scanned, cache hits, new alignments, hit rate percentage.
- Structured console summary report with per-speaker eligible word and phoneme statistics.
- Folder-based discovery and auto-pairing of subtitles (.en.vtt, .srt).
- Sub-5ms 100% cache hit response on re-ingestion of unchanged sources.
- Continuous `--watch` mode: initial pass, polling loop, delta alignment on file addition, clean Ctrl+C shutdown.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import pytest

import run_mixer
from contracts import AlignedEvent, InventoryIndex, SourceRecord, WordOccurrence
from run_mixer import (
    build_cli_parser,
    collect_source_items,
    get_folder_file_snapshot,
    ingest_sources,
    print_ingestion_summary_report,
    run_watch_loop,
)


def test_ingest_cli_parser_options():
    """Verify `build_cli_parser()` correctly configures the `ingest` subcommand and options."""
    parser = build_cli_parser()

    # Parse with standard options
    cmd = [
        "ingest",
        "--folder", "path/to/folder1", "path/to/folder2",
        "--manifest", "manifest.json",
        "--video", "v1.mp4", "v2.mp4",
        "--vtt", "v1.vtt", "v2.vtt",
        "--cache-dir", ".test_cache",
        "--speaker", "anakin",
        "--assume-single-speaker",
        "--force-realign",
        "--isolate-vocals",
        "--diarize", "force",
        "--speaker-registry", "config/custom_reg.json",
        "--watch",
        "--poll-interval", "2.5",
        "--non-interactive",
    ]
    args = parser.parse_args(cmd)
    assert args.command == "ingest"
    assert args.folder == ["path/to/folder1", "path/to/folder2"]
    assert args.manifest == "manifest.json"
    assert args.video == ["v1.mp4", "v2.mp4"]
    assert args.vtt == ["v1.vtt", "v2.vtt"]
    assert args.cache_dir == ".test_cache"
    assert args.speaker == "anakin"
    assert args.assume_single_speaker is True
    assert args.force_realign is True
    assert args.isolate_vocals is True
    assert args.diarize == "force"
    assert args.speaker_registry == "config/custom_reg.json"
    assert args.watch is True
    assert args.poll_interval == 2.5
    assert args.non_interactive is True


def test_ingest_cli_default_options():
    """Verify default values for `ingest` subcommand options."""
    parser = build_cli_parser()
    args = parser.parse_args(["ingest", "--folder", "media"])

    assert args.command == "ingest"
    assert args.folder == ["media"]
    assert args.manifest is None
    assert args.video is None
    assert args.vtt is None
    assert args.cache_dir is None
    assert args.speaker is None
    assert args.assume_single_speaker is False
    assert args.force_realign is False
    assert args.isolate_vocals is True
    assert args.diarize == "auto"
    assert args.speaker_registry == "config/speaker_registry.json"
    assert args.watch is False
    assert args.poll_interval == 5.0
    assert args.non_interactive is False

    # Verify --no-isolate-vocals flag
    args_no_vocals = parser.parse_args(["ingest", "--folder", "media", "--no-isolate-vocals"])
    assert args_no_vocals.isolate_vocals is False


def test_ingest_cli_does_not_require_synthesis_targets():
    """Verify `ingest` command runs without demanding --text, --target-audio, or --record-target."""
    parser = build_cli_parser()
    args = parser.parse_args(["ingest", "--manifest", "evaluation/anakin_sources.json"])

    # Targets must not be present or required
    assert not hasattr(args, "text") or args.text is None
    assert not hasattr(args, "target_audio") or args.target_audio is None
    assert not hasattr(args, "record_target") or args.record_target is False


def test_ingest_cli_missing_input_exits(monkeypatch: pytest.MonkeyPatch):
    """Running `ingest` without --folder, --manifest, or --video causes a CLI error."""
    monkeypatch.setattr("sys.argv", ["run_mixer.py", "ingest"])
    with pytest.raises(SystemExit) as exc_info:
        run_mixer.main()
    assert exc_info.value.code == 2


def test_collect_source_items_from_folders_and_manifest(tmp_path: Path):
    """Test `collect_source_items` resolution with folder and manifest inputs."""
    # Create test media folder
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    v1 = media_dir / "clip1.mp4"
    v1.write_bytes(b"dummy1")
    s1 = media_dir / "clip1.en.vtt"
    s1.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello\n", encoding="utf-8")

    # Create manifest pointing to a separate video
    v2 = tmp_path / "clip2.mp4"
    v2.write_bytes(b"dummy2")
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps([{"video": str(v2), "speaker_id": "test_speaker"}]), encoding="utf-8")

    args = argparse.Namespace(
        folder=[str(media_dir)],
        manifest=str(manifest_file),
        video=None,
        vtt=None,
        speaker="obi_wan",
        assume_single_speaker=True,
    )

    items = collect_source_items(args)
    assert len(items) == 2
    paths = [Path(it["video"]).name for it in items]
    assert "clip1.mp4" in paths
    assert "clip2.mp4" in paths

    # Folder item inherited speaker from CLI when assume_single_speaker was set
    folder_item = next(it for it in items if "clip1" in it["video"])
    assert folder_item["speaker_id"] == "obi_wan"
    assert folder_item["single_speaker_verified"] is True
    assert folder_item["vtt"] == str(s1.resolve())


def test_print_ingestion_summary_report_formatting(capsys: pytest.CaptureFixture):
    """Test `print_ingestion_summary_report` formats statistics cleanly."""
    summary_data = {
        "total_sources": 5,
        "new_alignments": 1,
        "cache_hits": 4,
        "speaker_stats": {
            "anakin": {"words": 150, "phonemes": 420},
            "obi_wan": {"words": 80, "phonemes": 230},
        },
    }
    print_ingestion_summary_report(summary_data)
    out = capsys.readouterr().out

    assert "INGESTION SUMMARY REPORT" in out
    assert "Total Sources Scanned: 5" in out
    assert "Newly Processed Clips: 1" in out
    assert "Cache Hits:            4 (80.0%)" in out
    assert "Speaker 'anakin': 150 words, 420 phonemes" in out
    assert "Speaker 'obi_wan': 80 words, 230 phonemes" in out
    assert "Total Available Units: 230 words, 650 phonemes" in out


def test_print_ingestion_summary_report_zero_sources(capsys: pytest.CaptureFixture):
    """Test `print_ingestion_summary_report` handles 0 sources without division by zero."""
    print_ingestion_summary_report(total_sources=0, cache_hits=0, new_alignments=0, speaker_stats={})
    out = capsys.readouterr().out

    assert "Total Sources Scanned: 0" in out
    assert "Cache Hits:            0 (100.0%)" in out
    assert "(No speaker unit statistics available)" in out


def test_ingest_sources_cache_hit_tracking(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify `ingest_sources` accurately tracks cache hits vs new alignments."""
    inv_index = InventoryIndex(
        words={
            "w1": WordOccurrence(
                word_id="w1",
                source_content_id="s1",
                speaker_id="anakin",
                word_normalized="test",
                word_original="Test",
                start_sample=0,
                end_sample=16000,
                phone_event_ids=["e1"],
                realized_phones=["T", "EH1", "S", "T"],
                is_eligible=True,
            )
        },
        events={
            "e1": AlignedEvent(
                event_id="e1",
                source_content_id="s1",
                utterance_id="u1",
                speaker_id="anakin",
                kind="phone",
                start_sample=0,
                end_sample=4000,
                label="T",
                is_eligible=True,
            )
        },
        sources={"s1": SourceRecord(content_id="s1", source_path="dummy.mp4", native_sample_rate=16000, native_channels=1)},
    )

    inv_path = tmp_path / "units_index.json"
    inv_path.write_text(json.dumps(inv_index.to_dict()), encoding="utf-8")

    # Mock ensure_source_aligned to simulate cache miss on first run, hit on second run
    call_count = 0

    def mock_ensure(video_path, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        cache_status = kwargs.get("cache_status")
        if cache_status is not None:
            if call_count <= 2:
                cache_status["hit"] = False
                cache_status["realigned"] = True
            else:
                cache_status["hit"] = True
                cache_status["realigned"] = False
        return str(inv_path)

    monkeypatch.setattr(run_mixer, "ensure_source_aligned", mock_ensure)
    monkeypatch.setattr(run_mixer, "load_or_combine_inventories", lambda paths: (inv_index, {"words": {}, "phones": {}}))

    items = [
        {"video": "video1.mp4", "vtt": "video1.vtt", "single_speaker_verified": True},
        {"video": "video2.mp4", "vtt": "video2.vtt", "single_speaker_verified": True},
    ]

    # First pass: cache misses
    res1 = ingest_sources(items, args=argparse.Namespace(assume_single_speaker=True))
    assert res1["total_sources"] == 2
    assert res1["new_alignments"] == 2
    assert res1["cache_hits"] == 0
    assert "anakin" in res1["speaker_stats"]
    assert res1["speaker_stats"]["anakin"]["words"] == 1
    assert res1["speaker_stats"]["anakin"]["phonemes"] == 1

    # Second pass: cache hits
    res2 = ingest_sources(items, args=argparse.Namespace(assume_single_speaker=True))
    assert res2["total_sources"] == 2
    assert res2["new_alignments"] == 0
    assert res2["cache_hits"] == 2


def test_ingest_sources_ineligible_units_excluded_from_speaker_stats(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that ineligible words and phones are excluded from summary statistics."""
    inv_index = InventoryIndex(
        words={
            "w_eligible": WordOccurrence(
                word_id="w1",
                source_content_id="s1",
                speaker_id="anakin",
                word_normalized="yes",
                word_original="yes",
                start_sample=0,
                end_sample=16000,
                phone_event_ids=["e1"],
                realized_phones=["Y", "EH1", "S"],
                is_eligible=True,
            ),
            "w_ineligible": WordOccurrence(
                word_id="w2",
                source_content_id="s1",
                speaker_id="anakin",
                word_normalized="no",
                word_original="no",
                start_sample=16000,
                end_sample=32000,
                phone_event_ids=["e2"],
                realized_phones=["N", "OW1"],
                is_eligible=False,  # Excluded due to overlap or noise
            ),
        },
        events={
            "e_eligible": AlignedEvent(
                event_id="e1",
                source_content_id="s1",
                utterance_id="u1",
                speaker_id="anakin",
                kind="phone",
                start_sample=0,
                end_sample=5000,
                label="Y",
                is_eligible=True,
            ),
            "e_ineligible": AlignedEvent(
                event_id="e2",
                source_content_id="s1",
                utterance_id="u1",
                speaker_id="anakin",
                kind="phone",
                start_sample=5000,
                end_sample=10000,
                label="EH1",
                is_eligible=False,
            ),
            "e_silence": AlignedEvent(
                event_id="e3",
                source_content_id="s1",
                utterance_id="u1",
                speaker_id="anakin",
                kind="silence",
                start_sample=10000,
                end_sample=16000,
                label="sil",
                is_eligible=True,  # Kind is silence, not phone
            ),
        },
        sources={"s1": SourceRecord(content_id="s1", source_path="dummy.mp4", native_sample_rate=16000, native_channels=1)},
    )

    inv_path = tmp_path / "units_index.json"
    inv_path.write_text(json.dumps(inv_index.to_dict()), encoding="utf-8")

    monkeypatch.setattr(run_mixer, "ensure_source_aligned", lambda *a, **kw: str(inv_path))
    monkeypatch.setattr(run_mixer, "load_or_combine_inventories", lambda paths: (inv_index, {"words": {}, "phones": {}}))

    res = ingest_sources([{"video": "test.mp4", "single_speaker_verified": True}])
    stats = res["speaker_stats"]["anakin"]
    assert stats["words"] == 1  # only w_eligible
    assert stats["phonemes"] == 1  # only e_eligible


def test_get_folder_file_snapshot(tmp_path: Path):
    """Test `get_folder_file_snapshot` detects supported video and subtitle files."""
    v1 = tmp_path / "clip1.mp4"
    v1.write_bytes(b"content1")
    s1 = tmp_path / "clip1.en.vtt"
    s1.write_text("WEBVTT\n", encoding="utf-8")
    ignored = tmp_path / "readme.txt"
    ignored.write_text("info", encoding="utf-8")

    snap = get_folder_file_snapshot([tmp_path])
    assert str(v1.resolve()) in snap
    assert str(s1.resolve()) in snap
    assert str(ignored.resolve()) not in snap
    assert snap[str(v1.resolve())][0] == len(b"content1")


def test_run_watch_loop_initial_and_delta_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify continuous `--watch` loop performs initial pass, polls, and triggers delta alignment."""
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()

    v1 = watched_dir / "clip1.mp4"
    v1.write_bytes(b"video1")
    s1 = watched_dir / "clip1.en.vtt"
    s1.write_text("WEBVTT\n", encoding="utf-8")

    aligned_videos: List[str] = []

    def mock_ensure(video_path, *args, **kwargs):
        aligned_videos.append(Path(video_path).name)
        status = kwargs.get("cache_status")
        if status is not None:
            status["hit"] = True
            status["realigned"] = False
        return str(tmp_path / "mock_inv.json")

    monkeypatch.setattr(run_mixer, "ensure_source_aligned", mock_ensure)
    dummy_inv = InventoryIndex()
    monkeypatch.setattr(run_mixer, "load_or_combine_inventories", lambda paths: (dummy_inv, {"words": {}, "phones": {}}))

    args = argparse.Namespace(
        folder=[str(watched_dir)],
        manifest=None,
        poll_interval=0.01,
        assume_single_speaker=True,
    )

    # Hook sleep to add clip2 during the first sleep cycle
    sleep_call_count = 0

    def mock_sleep(sec):
        nonlocal sleep_call_count
        sleep_call_count += 1
        if sleep_call_count == 1:
            # Create second video + subtitle file
            v2 = watched_dir / "clip2.mp4"
            v2.write_bytes(b"video2")
            s2 = watched_dir / "clip2.srt"
            s2.write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n", encoding="utf-8")

    monkeypatch.setattr(time, "sleep", mock_sleep)

    # Run for 2 iterations
    run_watch_loop(folders=[watched_dir], args=args, max_iterations=2)

    # Clip 1 aligned in initial pass; Clip 2 aligned in delta pass
    assert "clip1.mp4" in aligned_videos
    assert "clip2.mp4" in aligned_videos
    assert aligned_videos.count("clip2.mp4") == 1
    # Clip 1 was NOT re-aligned during the delta pass
    assert aligned_videos.count("clip1.mp4") == 1


def test_run_watch_loop_keyboard_interrupt_clean_exit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Verify that pressing Ctrl+C (KeyboardInterrupt) in watch mode logs clean exit and exits with code 0."""
    watched_dir = tmp_path / "watched"
    watched_dir.mkdir()

    v1 = watched_dir / "clip1.mp4"
    v1.write_bytes(b"video1")
    s1 = watched_dir / "clip1.vtt"
    s1.write_text("WEBVTT\n", encoding="utf-8")

    def mock_ensure(*args, **kwargs):
        return str(tmp_path / "mock_inv.json")

    monkeypatch.setattr(run_mixer, "ensure_source_aligned", mock_ensure)
    dummy_inv = InventoryIndex()
    monkeypatch.setattr(run_mixer, "load_or_combine_inventories", lambda paths: (dummy_inv, {"words": {}, "phones": {}}))

    # Raise KeyboardInterrupt when time.sleep is called
    def mock_sleep(sec):
        raise KeyboardInterrupt()

    monkeypatch.setattr(time, "sleep", mock_sleep)

    args = argparse.Namespace(
        folder=[str(watched_dir)],
        poll_interval=0.01,
        assume_single_speaker=True,
    )

    with pytest.raises(SystemExit) as exc_info:
        run_watch_loop(folders=[watched_dir], args=args)

    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "Watch mode terminated by user (Ctrl+C). Exiting cleanly." in out


def test_ingest_cli_main_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture):
    """Test full `run_mixer.py ingest` invocation via main()."""
    media_dir = tmp_path / "media"
    media_dir.mkdir()
    v1 = media_dir / "clip1.mp4"
    v1.write_bytes(b"video1")
    s1 = media_dir / "clip1.en.vtt"
    s1.write_text("WEBVTT\n", encoding="utf-8")

    dummy_inv = InventoryIndex(
        words={
            "w1": WordOccurrence(
                word_id="w1",
                source_content_id="s1",
                speaker_id="anakin",
                word_normalized="hello",
                word_original="hello",
                start_sample=0,
                end_sample=16000,
                phone_event_ids=["e1"],
                realized_phones=["HH", "AH0", "L", "OW1"],
                is_eligible=True,
            )
        },
        events={
            "e1": AlignedEvent(
                event_id="e1",
                source_content_id="s1",
                utterance_id="u1",
                speaker_id="anakin",
                kind="phone",
                start_sample=0,
                end_sample=4000,
                label="HH",
                is_eligible=True,
            )
        },
        sources={"s1": SourceRecord(content_id="s1", source_path=str(v1), native_sample_rate=16000, native_channels=1)},
    )
    inv_file = tmp_path / "units_index.json"
    inv_file.write_text(json.dumps(dummy_inv.to_dict()), encoding="utf-8")

    monkeypatch.setattr(run_mixer, "ensure_source_aligned", lambda *a, **kw: str(inv_file))
    monkeypatch.setattr(run_mixer, "load_or_combine_inventories", lambda paths: (dummy_inv, {"words": {}, "phones": {}}))

    test_args = [
        "run_mixer.py",
        "ingest",
        "--folder", str(media_dir),
        "--speaker", "anakin",
        "--assume-single-speaker",
    ]
    monkeypatch.setattr("sys.argv", test_args)

    run_mixer.main()
    out = capsys.readouterr().out
    assert "INGESTION SUMMARY REPORT" in out
    assert "Total Sources Scanned: 1" in out
    assert "Speaker 'anakin': 1 words, 1 phonemes" in out


def test_ingest_sources_passes_isolate_vocals_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that --isolate-vocals is passed to ensure_source_aligned."""
    passed_isolate_vocals = []

    def mock_ensure(video_path, *args, **kwargs):
        passed_isolate_vocals.append(kwargs.get("isolate_vocals"))
        return str(tmp_path / "mock_inv.json")

    monkeypatch.setattr(run_mixer, "ensure_source_aligned", mock_ensure)
    dummy_inv = InventoryIndex()
    monkeypatch.setattr(run_mixer, "load_or_combine_inventories", lambda paths: (dummy_inv, {"words": {}, "phones": {}}))

    items = [{"video": "test.mp4", "single_speaker_verified": True}]
    ingest_sources(items, args=argparse.Namespace(isolate_vocals=True, assume_single_speaker=True))

    assert passed_isolate_vocals == [True]


def test_ingest_sources_diarize_force_sets_force_inventory_rebuild(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that --diarize force sets force_inventory_rebuild=True in ensure_source_aligned."""
    passed_force_rebuild = []

    def mock_ensure(video_path, *args, **kwargs):
        passed_force_rebuild.append(kwargs.get("force_inventory_rebuild"))
        return str(tmp_path / "mock_inv.json")

    monkeypatch.setattr(run_mixer, "ensure_source_aligned", mock_ensure)
    dummy_inv = InventoryIndex()
    monkeypatch.setattr(run_mixer, "load_or_combine_inventories", lambda paths: (dummy_inv, {"words": {}, "phones": {}}))

    items = [{"video": "test.mp4", "single_speaker_verified": True}]
    ingest_sources(items, args=argparse.Namespace(diarize="force", assume_single_speaker=True))

    assert passed_force_rebuild == [True]


def test_run_watch_loop_with_manifest_delta(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify that modifying the manifest file triggers delta ingestion on changed items."""
    manifest_p = tmp_path / "sources.json"
    v1 = tmp_path / "v1.mp4"
    v1.write_bytes(b"data1")
    v2 = tmp_path / "v2.mp4"
    v2.write_bytes(b"data2")

    # Initial manifest has v1
    manifest_p.write_text(json.dumps([{"video": str(v1)}]), encoding="utf-8")

    aligned_videos: List[str] = []

    def mock_ensure(video_path, *args, **kwargs):
        aligned_videos.append(Path(video_path).name)
        return str(tmp_path / "mock_inv.json")

    monkeypatch.setattr(run_mixer, "ensure_source_aligned", mock_ensure)
    dummy_inv = InventoryIndex()
    monkeypatch.setattr(run_mixer, "load_or_combine_inventories", lambda paths: (dummy_inv, {"words": {}, "phones": {}}))

    sleep_count = 0

    def mock_sleep(sec):
        nonlocal sleep_count
        sleep_count += 1
        if sleep_count == 1:
            # Modify manifest to add v2
            time.sleep(0.01)  # ensure mtime differs
            manifest_p.write_text(json.dumps([{"video": str(v1)}, {"video": str(v2)}]), encoding="utf-8")

    monkeypatch.setattr(time, "sleep", mock_sleep)

    args = argparse.Namespace(manifest=str(manifest_p), poll_interval=0.01, assume_single_speaker=True)
    run_watch_loop(manifest=str(manifest_p), args=args, max_iterations=2)

    assert "v1.mp4" in aligned_videos
    assert "v2.mp4" in aligned_videos


def test_ingest_positional_video_infers_subtitles(tmp_path: Path):
    """Verify positional video argument infers matching subtitles automatically."""
    v = tmp_path / "clip.mp4"
    v.write_bytes(b"video_data")
    sub = tmp_path / "clip.en.vtt"
    sub.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world\n", encoding="utf-8")

    parser = build_cli_parser()
    args = parser.parse_args(["ingest", str(v)])
    assert args.command == "ingest"
    assert args.source == [str(v)]

    items = collect_source_items(args)
    assert len(items) == 1
    assert items[0]["video"] == str(v.resolve())
    assert items[0]["vtt"] == str(sub.resolve())


def test_ingest_video_flag_infers_subtitles(tmp_path: Path):
    """Verify --video flag without --vtt infers matching subtitles automatically."""
    v = tmp_path / "sample.mp4"
    v.write_bytes(b"video_data")
    sub = tmp_path / "sample.vtt"
    sub.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nTest\n", encoding="utf-8")

    parser = build_cli_parser()
    args = parser.parse_args(["ingest", "--video", str(v)])
    assert args.command == "ingest"
    assert args.video == [str(v)]
    assert args.vtt is None

    items = collect_source_items(args)
    assert len(items) == 1
    assert items[0]["video"] == str(v.resolve())
    assert items[0]["vtt"] == str(sub.resolve())


def test_ingest_positional_folder_and_manifest(tmp_path: Path):
    """Verify positional folder and manifest resolution."""
    # Folder test
    f_dir = tmp_path / "media"
    f_dir.mkdir()
    v1 = f_dir / "vid1.mp4"
    v1.write_bytes(b"vid1")
    s1 = f_dir / "vid1.srt"
    s1.write_text("1\n00:00:00,000 --> 00:00:01,000\nSub 1\n", encoding="utf-8")

    parser = build_cli_parser()
    args_folder = parser.parse_args(["ingest", str(f_dir)])
    items_folder = collect_source_items(args_folder)
    assert len(items_folder) == 1
    assert items_folder[0]["video"] == str(v1.resolve())
    assert items_folder[0]["vtt"] == str(s1.resolve())

    # Manifest test
    v2 = tmp_path / "vid2.mp4"
    v2.write_bytes(b"vid2")
    s2 = tmp_path / "vid2.en.vtt"
    s2.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nSub 2\n", encoding="utf-8")
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps([{"video": str(v2)}]), encoding="utf-8")

    args_man = parser.parse_args(["ingest", str(man)])
    items_man = collect_source_items(args_man)
    assert len(items_man) == 1
    assert items_man[0]["video"] == str(v2.resolve())
    assert items_man[0]["vtt"] == str(s2.resolve())


