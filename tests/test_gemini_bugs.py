"""
Unit tests for Gemini Bugs 1–5 fixes.
Milestone 1 / Engine Foundation & Bug Fixes:
- Bug 1: Serialize cached_vocal_pcm_path into inventory metadata (units_index.json["sources"]).
- Bug 2: Replace flat globbing with recursive globbing (rglob) in run_mixer and inventory_builder.
- Bug 3: Synchronize effective isolate_vocals state into plan.metadata["isolate_vocals"].
- Bug 4: Preserve user-specified --speaker flag (do not overwrite in single-speaker inventories).
- Bug 5: Remove dead Path expression in run_mixer.py (line 236) and ensure directory exists.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import run_mixer
from contracts import AlignedEvent, InventoryIndex, SourceRecord, WordOccurrence
from inventory_builder import build_units_inventory
from media_time import SourceRecord as MediaSourceRecord
from run_mixer import get_source_id


def create_synthetic_textgrid(tg_path: Path, word: str, phone: str) -> None:
    """Creates a sample Praat Long Format TextGrid with phone and word tiers."""
    content = f"""File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "{word}"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0
            xmax = 1.0
            text = "{phone}"
"""
    tg_path.write_text(content, encoding="utf-8")


def test_bug1_cached_vocal_pcm_path_serialized_in_inventory(tmp_path: Path) -> None:
    """
    Bug 1 Fix Verification:
    build_units_inventory must serialize cached_vocal_pcm_path into units_index.json["sources"]
    when provided via SourceRecord, manifest audio_source_path, or --vocal-stem argument.
    This prevents ensure_source_aligned from entering infinite alignment invalidation loops.
    """
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)
    align_dir = tmp_path / "aligned"
    align_dir.mkdir(parents=True, exist_ok=True)
    out_inventory = tmp_path / "inventory" / "units_index.json"

    # Create dummy chunk wav and textgrid
    chunk_wav = corpus_dir / "chunk_000.wav"
    chunk_wav.write_bytes(b"RIFFdummy")
    chunk_tg = align_dir / "chunk_000.TextGrid"
    chunk_tg.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
        'xmin = 0\nxmax = 1.0\ntiers? <exists>\nsize = 1\n'
        'item []:\n    item [1]:\n        class = "IntervalTier"\n'
        '        name = "words"\n        xmin = 0\n        xmax = 1.0\n'
        '        intervals: size = 1\n        intervals [1]:\n'
        '            xmin = 0\n            xmax = 1.0\n            text = "hello"\n',
        encoding="utf-8",
    )

    vocal_stem = tmp_path / "vocals.wav"
    vocal_stem.write_bytes(b"RIFFvocalstem")

    # 1. Test via source_record parameter
    source_rec = MediaSourceRecord(
        source_path=str(tmp_path / "video.mp4"),
        content_id="source_test_01",
        cached_pcm_path=str(tmp_path / "original.wav"),
        cached_vocal_pcm_path=str(vocal_stem.resolve()),
        native_sample_rate=48000,
        native_channels=2,
    )

    manifest_file = corpus_dir / "corpus_manifest.json"
    manifest_file.write_text(
        json.dumps({
            "source_video": str(tmp_path / "video.mp4"),
            "chunks": {"chunk_000": {"start_sec": 0.0, "end_sec": 1.0, "text": "hello"}},
        }),
        encoding="utf-8",
    )

    build_units_inventory(
        aligned_dir=str(align_dir),
        manifest_path=str(manifest_file),
        output_index_path=str(out_inventory),
        speaker_id="speaker_0",
        source_record=source_rec,
    )

    assert out_inventory.exists()
    data = json.loads(out_inventory.read_text(encoding="utf-8"))
    assert "sources" in data
    assert "source_test_01" in data["sources"]
    assert data["sources"]["source_test_01"]["cached_vocal_pcm_path"] == str(vocal_stem.resolve())

    # 2. Test via manifest audio_source_path
    manifest_file2 = corpus_dir / "corpus_manifest_vocal.json"
    manifest_file2.write_text(
        json.dumps({
            "source_video": str(tmp_path / "video2.mp4"),
            "audio_source_path": str(vocal_stem.resolve()),
            "chunks": {"chunk_000": {"start_sec": 0.0, "end_sec": 1.0, "text": "hello"}},
        }),
        encoding="utf-8",
    )

    out_inv2 = tmp_path / "inventory2" / "units_index.json"
    build_units_inventory(
        aligned_dir=str(align_dir),
        manifest_path=str(manifest_file2),
        output_index_path=str(out_inv2),
    )

    data2 = json.loads(out_inv2.read_text(encoding="utf-8"))
    first_source_id = next(iter(data2["sources"]))
    assert data2["sources"][first_source_id]["cached_vocal_pcm_path"] == str(vocal_stem.resolve())


def test_bug2_build_units_inventory_rglob_discovers_nested_speakers(tmp_path: Path) -> None:
    """
    Bug 2 Fix Verification:
    build_units_inventory must use recursive globbing (rglob) to discover
    TextGrid files located in nested speaker subdirectories (e.g., aligned/spk_1/chunk_001.TextGrid).
    If flat globbing is used, nested TextGrids are missed and 0 units are indexed.
    """
    corpus_dir = tmp_path / "corpus"
    aligned_dir = tmp_path / "aligned"
    out_inventory = tmp_path / "inventory" / "units_index.json"

    spk1_align = aligned_dir / "spk_1"
    spk2_align = aligned_dir / "spk_2"
    spk1_align.mkdir(parents=True)
    spk2_align.mkdir(parents=True)

    create_synthetic_textgrid(spk1_align / "chunk_001.TextGrid", "hello", "HH")
    create_synthetic_textgrid(spk2_align / "chunk_002.TextGrid", "world", "W")

    manifest_file = corpus_dir / "corpus_manifest.json"
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(
        json.dumps({
            "source_video": str(tmp_path / "video.mp4"),
            "chunks": {
                "chunk_001": {"start_sec": 0.0, "end_sec": 1.0, "text": "hello"},
                "chunk_002": {"start_sec": 1.0, "end_sec": 2.0, "text": "world"},
            },
        }),
        encoding="utf-8",
    )

    build_units_inventory(
        aligned_dir=str(aligned_dir),
        manifest_path=str(manifest_file),
        output_index_path=str(out_inventory),
    )

    assert out_inventory.exists()
    data = json.loads(out_inventory.read_text(encoding="utf-8"))
    assert len(data.get("words", {})) == 2, "build_units_inventory failed to discover nested TextGrids"
    words_indexed = sorted(w["word_normalized"] for w in data["words"].values())
    assert words_indexed == ["hello", "world"]


def test_bug2_run_mfa_align_rglob_discovers_nested_chunks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Bug 2 Fix Verification (run_mixer.py):
    run_mfa_align must use recursive globbing (rglob) to discover chunk_*.wav files
    located in nested speaker subdirectories (e.g., corpus/spk_1/chunk_001.wav).
    """
    corpus_dir = tmp_path / "corpus"
    aligned_dir = tmp_path / "aligned"
    corpus_dir.mkdir(parents=True)
    aligned_dir.mkdir(parents=True)

    spk1 = corpus_dir / "spk_1"
    spk2 = corpus_dir / "spk_2"
    spk1.mkdir(parents=True)
    spk2.mkdir(parents=True)

    (spk1 / "chunk_001.wav").write_bytes(b"RIFFdummy1")
    (spk1 / "chunk_001.txt").write_text("hello", encoding="utf-8")
    (spk2 / "chunk_002.wav").write_bytes(b"RIFFdummy2")
    (spk2 / "chunk_002.txt").write_text("world", encoding="utf-8")

    def mock_subp_run(cmd: list[str], *args: object, **kwargs: object) -> MagicMock:
        # In _execute_mfa, cmd is ["mfa", "align", src_dir, dict, acoustic, dst_dir, ...]
        # cmd[2] is src_dir, cmd[5] is dst_dir
        src_dir = Path(cmd[2])
        dst_dir = Path(cmd[5])
        for w in src_dir.rglob("chunk_*.wav"):
            (dst_dir / f"{w.stem}.TextGrid").write_text("dummy_tg", encoding="utf-8")
        return MagicMock(returncode=0)

    monkeypatch.setattr(run_mixer, "get_mamba_config", lambda: (None, None))
    monkeypatch.setattr("subprocess.run", mock_subp_run)

    run_mixer.run_mfa_align(
        corpus_dir=str(corpus_dir),
        output_aligned_dir=str(aligned_dir),
    )

    aligned_tgs = sorted(p.name for p in aligned_dir.glob("*.TextGrid"))
    assert aligned_tgs == ["chunk_001.TextGrid", "chunk_002.TextGrid"]


def test_bug3_isolate_vocals_synchronized_in_plan_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Bug 3 Fix Verification:
    Effective isolate_vocals state derived in run_mixer (from --isolate-vocals or
    --audio-track in isolated/recombined/both/all) must be synchronized into plan.metadata["isolate_vocals"].
    If reverted to getattr(args, "isolate_vocals", False), --audio-track isolated incorrectly leaves
    plan.metadata["isolate_vocals"] as False.
    """
    # Create a minimal synthetic inventory index
    src = SourceRecord(
        content_id="src_1",
        source_path=str(tmp_path / "dummy.mp4"),
        native_sample_rate=48000,
        native_channels=2,
    )
    ev = AlignedEvent(
        event_id="ev1",
        source_content_id="src_1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        start_sample=0,
        end_sample=24000,
        label="HH",
        base_phone="HH",
    )
    w = WordOccurrence(
        word_id="w1",
        source_content_id="src_1",
        speaker_id="spk1",
        word_normalized="hello",
        word_original="hello",
        start_sample=0,
        end_sample=24000,
        phone_event_ids=["ev1"],
        realized_phones=["HH"],
    )
    inv = InventoryIndex(
        sources={"src_1": src},
        events={"ev1": ev},
        words={"w1": w},
        word_to_word_ids={"hello": ["w1"]},
        phone_to_event_ids={"HH": ["ev1"]},
    )

    inv_file = tmp_path / "units_index.json"
    with open(inv_file, "w", encoding="utf-8") as f:
        json.dump(inv.to_dict(), f)

    # Mock render_sentence_mix so no FFmpeg subprocess is executed
    def mock_render(plan_or_units, out_mp4, *args, **kwargs):
        p = Path(out_mp4)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"dummy_mp4")
        p.with_suffix(".wav").write_bytes(b"dummy_wav")

    monkeypatch.setattr(run_mixer, "render_sentence_mix", mock_render)

    # Case A: User specifies --audio-track isolated without --isolate-vocals
    plan_isolated = tmp_path / "plan_isolated.json"
    monkeypatch.setattr(sys, "argv", [
        "run_mixer.py", "mix",
        "--text", "hello",
        "--inventory", str(inv_file),
        "--audio-track", "isolated",
        "--selector", "span",
        "--plan", str(plan_isolated),
        "--render", str(tmp_path / "render_isolated.mp4"),
    ])
    run_mixer.main()

    assert plan_isolated.exists()
    with open(plan_isolated, "r", encoding="utf-8") as f:
        pdata_isolated = json.load(f)
    assert pdata_isolated["metadata"].get("isolate_vocals") is True
    assert pdata_isolated["metadata"].get("audio_track") == "isolated"

    # Case B: User specifies --audio-track combined with --no-isolate-vocals
    plan_combined = tmp_path / "plan_combined.json"
    monkeypatch.setattr(sys, "argv", [
        "run_mixer.py", "mix",
        "--text", "hello",
        "--inventory", str(inv_file),
        "--audio-track", "combined",
        "--no-isolate-vocals",
        "--selector", "span",
        "--plan", str(plan_combined),
        "--render", str(tmp_path / "render_combined.mp4"),
    ])
    run_mixer.main()

    assert plan_combined.exists()
    with open(plan_combined, "r", encoding="utf-8") as f:
        pdata_combined = json.load(f)
    assert pdata_combined["metadata"].get("isolate_vocals") is False
    assert pdata_combined["metadata"].get("audio_track") == "combined"


def test_bug4_speaker_flag_preserved_in_single_speaker_inventory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Bug 4 Fix Verification:
    Preserve user-specified --speaker flag in run_mixer instead of unconditionally overwriting
    it with the inventory's single speaker ID.
    If reverted to buggy behavior (if len(unique_speakers) == 1: effective_speaker = next(iter(unique_speakers))),
    the user's explicit --speaker flag is silently ignored.
    """
    src = SourceRecord(
        content_id="src_1",
        source_path=str(tmp_path / "dummy.mp4"),
        native_sample_rate=48000,
        native_channels=2,
    )
    ev = AlignedEvent(
        event_id="ev1",
        source_content_id="src_1",
        utterance_id="u1",
        speaker_id="single_inventory_spk",
        kind="phone",
        start_sample=0,
        end_sample=24000,
        label="HH",
        base_phone="HH",
    )
    w = WordOccurrence(
        word_id="w1",
        source_content_id="src_1",
        speaker_id="single_inventory_spk",
        word_normalized="hello",
        word_original="hello",
        start_sample=0,
        end_sample=24000,
        phone_event_ids=["ev1"],
        realized_phones=["HH"],
    )
    inv = InventoryIndex(
        sources={"src_1": src},
        events={"ev1": ev},
        words={"w1": w},
        word_to_word_ids={"hello": ["w1"]},
        phone_to_event_ids={"HH": ["ev1"]},
    )

    inv_file = tmp_path / "units_index.json"
    with open(inv_file, "w", encoding="utf-8") as f:
        json.dump(inv.to_dict(), f)

    monkeypatch.setattr(run_mixer, "render_sentence_mix", MagicMock())
    mock_select = MagicMock(return_value=[])
    monkeypatch.setattr(run_mixer.SpanSelector, "select", mock_select)

    # Case A: User explicitly provides --speaker="custom_speaker" absent from single-speaker inventory
    # Under Round 3 B02 / U03 identity governance, requesting an absent speaker exits with code 2
    monkeypatch.setattr(sys, "argv", [
        "run_mixer.py", "mix",
        "--text", "hello",
        "--inventory", str(inv_file),
        "--speaker", "custom_speaker",
        "--selector", "span",
        "--plan", str(tmp_path / "plan1.json"),
        "--render", str(tmp_path / "render1.mp4"),
    ])
    with pytest.raises(SystemExit) as exc_info:
        run_mixer.main()
    assert exc_info.value.code == 2

    # Case A2: User explicitly provides valid --speaker="single_inventory_spk"
    mock_select.reset_mock()
    monkeypatch.setattr(sys, "argv", [
        "run_mixer.py", "mix",
        "--text", "hello",
        "--inventory", str(inv_file),
        "--speaker", "single_inventory_spk",
        "--selector", "span",
        "--plan", str(tmp_path / "plan1_valid.json"),
        "--render", str(tmp_path / "render1_valid.mp4"),
    ])
    try:
        run_mixer.main()
    except Exception:
        pass  # Empty span exception is expected since mock returns []

    assert mock_select.call_count >= 1
    passed_speaker = mock_select.call_args[1].get("speaker_id")
    assert passed_speaker == "single_inventory_spk", f"Expected single_inventory_spk to be preserved, got {passed_speaker}"

    # Case B: User does NOT provide --speaker (should fallback to single_inventory_spk)
    mock_select.reset_mock()
    monkeypatch.setattr(sys, "argv", [
        "run_mixer.py", "mix",
        "--text", "hello",
        "--inventory", str(inv_file),
        "--selector", "span",
        "--plan", str(tmp_path / "plan2.json"),
        "--render", str(tmp_path / "render2.mp4"),
    ])
    try:
        run_mixer.main()
    except Exception:
        pass

    assert mock_select.call_count >= 1
    passed_speaker = mock_select.call_args[1].get("speaker_id")
    assert passed_speaker == "single_inventory_spk", f"Expected fallback single_inventory_spk, got {passed_speaker}"


def test_bug5_source_cache_dir_created_without_dead_expression(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Bug 5 Fix Verification:
    Ensure line 236 dead Path expression CACHE_BASE_DIR / "sources" / source_id is replaced
    with active directory creation so the sources directory exists on disk.
    """
    from run_mixer import ensure_source_aligned

    build_dir = tmp_path / ".build"
    monkeypatch.setattr("run_mixer.CACHE_BASE_DIR", build_dir)

    dummy_media = tmp_path / "sample.mp4"
    dummy_media.write_bytes(b"dummy_sample_video")
    dummy_vtt = tmp_path / "sample.vtt"
    dummy_vtt.write_text("WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nhello\n", encoding="utf-8")

    expected_source_id = get_source_id(str(dummy_media.resolve()))
    target_dir = build_dir / "sources" / expected_source_id
    assert not target_dir.exists()

    # Mock downstream processing to isolate and verify line 236 directory creation in ensure_source_aligned
    monkeypatch.setattr("run_mixer.preprocess_source", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_mixer.run_mfa_align", lambda *args, **kwargs: None)
    monkeypatch.setattr("run_mixer.build_units_inventory", lambda *args, **kwargs: None)

    ensure_source_aligned(video_path=str(dummy_media), vtt_path=str(dummy_vtt))

    assert target_dir.exists()
    assert target_dir.is_dir()
