import json
import os
from pathlib import Path

import pytest

import run_mixer
from cache import (
    compute_media_hash,
)


def create_dummy_vtt(path: Path) -> Path:
    content = """WEBVTT

00:00:00.500 --> 00:00:02.500
Hello world test.
"""
    path.write_text(content, encoding="utf-8")
    return path


def test_u10_middle_byte_changes_media_hash_and_source_id(tmp_path: Path):
    """
    U10: 300,000-byte media, middle byte changed.
    Produces a new media hash and new source ID.
    Fails loudly if file does not exist (no path hash fallback).
    """
    media_file = tmp_path / "sample_video.mp4"
    data = bytearray(b"A" * 300000)
    media_file.write_bytes(bytes(data))

    hash1 = compute_media_hash(media_file, cache_dir=tmp_path)
    id1 = run_mixer.get_source_id(str(media_file))

    # Flip middle byte (index 150000)
    data[150000] = ord(b"Z")
    media_file.write_bytes(bytes(data))

    # Touch mtime to ensure cache index notices modification
    new_mtime = media_file.stat().st_mtime + 5.0
    os.utime(media_file, (new_mtime, new_mtime))

    hash2 = compute_media_hash(media_file, cache_dir=tmp_path)
    id2 = run_mixer.get_source_id(str(media_file))

    assert hash1 != hash2, "Middle-byte change in 300KB media must produce different media hash"
    assert id1 != id2, "Middle-byte change must produce different source_id"


def test_u10_compute_media_hash_fails_loudly_on_missing_file(tmp_path: Path):
    missing_media = tmp_path / "does_not_exist.mp4"
    with pytest.raises(FileNotFoundError):
        compute_media_hash(missing_media, cache_dir=tmp_path)

    with pytest.raises(FileNotFoundError):
        run_mixer.get_source_id(str(missing_media))


def test_u10_middle_byte_changed_rebuilds_alignment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    U10: When middle byte of 300,000-byte file changes, ensure_source_aligned
    must rebuild alignment (MFA must be called again).
    """
    cache_dir = tmp_path / ".build"
    monkeypatch.setattr(run_mixer, "CACHE_BASE_DIR", cache_dir)

    media_file = tmp_path / "video.mp4"
    data = bytearray(b"X" * 300000)
    media_file.write_bytes(bytes(data))
    vtt_file = create_dummy_vtt(tmp_path / "video.vtt")

    mfa_call_count = 0

    def fake_mfa(corpus_dir, output_aligned_dir, dict_path, acoustic_model="english_us_arpa"):
        nonlocal mfa_call_count
        mfa_call_count += 1
        out = Path(output_aligned_dir)
        out.mkdir(parents=True, exist_ok=True)
        # Create a dummy TextGrid
        (out / "chunk_0000.TextGrid").write_text("dummy textgrid", encoding="utf-8")

    def fake_preprocess(v_abs, vtt, out_dir, **kwargs):
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "corpus_manifest.json").write_text(json.dumps({"chunks": {"chunk_0000": {}}}), encoding="utf-8")

    def fake_inventory_builder(aligned_dir, manifest_path, output_index_path, **kwargs):
        p = Path(output_index_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"schema_version": "2.0.0", "words": {}, "events": {}, "sources": {}}), encoding="utf-8")

    monkeypatch.setattr(run_mixer, "run_mfa_align", fake_mfa)
    monkeypatch.setattr(run_mixer, "preprocess_source", fake_preprocess)
    monkeypatch.setattr(run_mixer, "build_units_inventory", fake_inventory_builder)

    # 1st run: fresh alignment
    inv1 = run_mixer.ensure_source_aligned(str(media_file), str(vtt_file))
    assert Path(inv1).exists()
    assert mfa_call_count == 1

    # Invalidate cache by flipping middle byte
    data[150000] = ord(b"Y")
    media_file.write_bytes(bytes(data))
    new_mtime = media_file.stat().st_mtime + 5.0
    os.utime(media_file, (new_mtime, new_mtime))

    # 2nd run: should trigger alignment again
    inv2 = run_mixer.ensure_source_aligned(str(media_file), str(vtt_file))
    assert Path(inv2).exists()
    assert mfa_call_count == 2, "Middle byte change must trigger MFA re-alignment"


def test_u11_identity_assertion_change_skips_mfa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    U11: Changing identity assertion (single_speaker_verified) rebuilds only the labeled inventory,
    making NO MFA call.
    """
    cache_dir = tmp_path / ".build"
    monkeypatch.setattr(run_mixer, "CACHE_BASE_DIR", cache_dir)

    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"A" * 1000)
    vtt_file = create_dummy_vtt(tmp_path / "video.vtt")

    mfa_call_count = 0
    inv_build_count = 0

    def fake_mfa(corpus_dir, output_aligned_dir, dict_path, acoustic_model="english_us_arpa"):
        nonlocal mfa_call_count
        mfa_call_count += 1
        out = Path(output_aligned_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "chunk_0000.TextGrid").write_text("dummy textgrid", encoding="utf-8")

    def fake_preprocess(v_abs, vtt, out_dir, **kwargs):
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "corpus_manifest.json").write_text(json.dumps({"chunks": {"chunk_0000": {}}}), encoding="utf-8")

    def fake_inventory_builder(aligned_dir, manifest_path, output_index_path, **kwargs):
        nonlocal inv_build_count
        inv_build_count += 1
        p = Path(output_index_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"schema_version": "2.0.0", "words": {}, "events": {}, "sources": {}}), encoding="utf-8")

    monkeypatch.setattr(run_mixer, "run_mfa_align", fake_mfa)
    monkeypatch.setattr(run_mixer, "preprocess_source", fake_preprocess)
    monkeypatch.setattr(run_mixer, "build_units_inventory", fake_inventory_builder)

    # Run 1: unverified
    inv1 = run_mixer.ensure_source_aligned(str(media_file), str(vtt_file), single_speaker_verified=False)
    assert Path(inv1).exists()
    assert mfa_call_count == 1
    assert inv_build_count == 1

    # Run 2: identity change only (single_speaker_verified=True, speaker_id="anakin")
    inv2 = run_mixer.ensure_source_aligned(
        str(media_file), str(vtt_file), speaker_id="anakin", single_speaker_verified=True
    )
    assert Path(inv2).exists()
    # MFA call count MUST STILL BE 1 (NO MFA CALL)!
    assert mfa_call_count == 1, "Identity change must NOT trigger MFA"
    # Inventory builder was called again to update labeled layer
    assert inv_build_count == 2, "Identity change must rebuild labeled inventory"


def test_u11_speaker_intervals_change_skips_mfa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    U11: Changing speaker intervals rebuilds only the labeled inventory,
    making NO MFA call.
    """
    cache_dir = tmp_path / ".build"
    monkeypatch.setattr(run_mixer, "CACHE_BASE_DIR", cache_dir)

    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"A" * 1000)
    vtt_file = create_dummy_vtt(tmp_path / "video.vtt")

    mfa_call_count = 0
    inv_build_count = 0

    def fake_mfa(corpus_dir, output_aligned_dir, dict_path, acoustic_model="english_us_arpa"):
        nonlocal mfa_call_count
        mfa_call_count += 1
        out = Path(output_aligned_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "chunk_0000.TextGrid").write_text("dummy textgrid", encoding="utf-8")

    def fake_preprocess(v_abs, vtt, out_dir, **kwargs):
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "corpus_manifest.json").write_text(json.dumps({"chunks": {"chunk_0000": {}}}), encoding="utf-8")

    def fake_inventory_builder(aligned_dir, manifest_path, output_index_path, **kwargs):
        nonlocal inv_build_count
        inv_build_count += 1
        p = Path(output_index_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"schema_version": "2.0.0", "words": {}, "events": {}, "sources": {}}), encoding="utf-8")

    monkeypatch.setattr(run_mixer, "run_mfa_align", fake_mfa)
    monkeypatch.setattr(run_mixer, "preprocess_source", fake_preprocess)
    monkeypatch.setattr(run_mixer, "build_units_inventory", fake_inventory_builder)

    # Run 1: interval 0 to 1
    run_mixer.ensure_source_aligned(
        str(media_file), str(vtt_file), speaker_intervals=[{"start_sec": 0.0, "end_sec": 1.0}]
    )
    assert mfa_call_count == 1
    assert inv_build_count == 1

    # Run 2: interval 1 to 2
    run_mixer.ensure_source_aligned(
        str(media_file), str(vtt_file), speaker_intervals=[{"start_sec": 1.0, "end_sec": 2.0}]
    )
    assert mfa_call_count == 1, "Interval change must NOT trigger MFA"
    assert inv_build_count == 2, "Interval change must rebuild labeled inventory"
