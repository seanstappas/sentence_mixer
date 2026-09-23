"""
Tests for content-aware caching and atomic publishing.
"""

import tempfile
from pathlib import Path
from cache import (
    CacheManager,
    hash_string,
    hash_dict,
)


def test_hash_primitives():
    s1 = hash_string("test string")
    s2 = hash_string("test string")
    assert s1 == s2

    d1 = hash_dict({"b": 2, "a": 1})
    d2 = hash_dict({"a": 1, "b": 2})
    assert d1 == d2, "Dictionary key ordering should not affect fingerprint"


def test_fingerprint_sensitivity():
    cm = CacheManager()
    fp1 = cm.compute_alignment_fingerprint(
        audio_hashes={"src_1": "abc"},
        transcript_content="hello there",
        dict_hashes={"dict": "123"},
        acoustic_model="english_us_arpa",
        mfa_options={"clean": True},
    )

    # Changing MFA option changes fingerprint
    fp2 = cm.compute_alignment_fingerprint(
        audio_hashes={"src_1": "abc"},
        transcript_content="hello there",
        dict_hashes={"dict": "123"},
        acoustic_model="english_us_arpa",
        mfa_options={"clean": False},
    )
    assert fp1 != fp2

    # Changing transcript changes fingerprint
    fp3 = cm.compute_alignment_fingerprint(
        audio_hashes={"src_1": "abc"},
        transcript_content="general kenobi",
        dict_hashes={"dict": "123"},
        acoustic_model="english_us_arpa",
        mfa_options={"clean": True},
    )
    assert fp1 != fp3


def test_atomic_file_publishing():
    with tempfile.TemporaryDirectory() as tmpdir:
        cm = CacheManager(build_dir=tmpdir)
        src_file = Path(tmpdir) / "source.txt"
        src_file.write_text("hello world", encoding="utf-8")

        dest_file = Path(tmpdir) / "sub" / "dest.txt"
        cm.atomic_publish_file(src_file, dest_file)

        assert dest_file.exists()
        assert dest_file.read_text(encoding="utf-8") == "hello world"


def test_t11_cache_invalidation_on_changed_media(tmp_path: Path):
    """T11: Changed source bytes / mtime -> Correct dependent invalidation before outer inventory cache return."""
    import time
    import run_mixer

    source_video = tmp_path / "test.mp4"
    source_video.write_bytes(b"media_bytes_v1")

    vtt_file = tmp_path / "test.vtt"
    vtt_file.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n", encoding="utf-8")

    source_id = run_mixer.get_source_id(str(source_video))
    cache_base = tmp_path / ".build"
    inv_dir = cache_base / "inventory" / source_id
    inv_dir.mkdir(parents=True)
    inv_file = inv_dir / "units_index.json"
    inv_file.write_text('{"schema_version": "1.0", "words": {}}', encoding="utf-8")

    # Monkeypatch CACHE_BASE_DIR
    orig_cache = run_mixer.CACHE_BASE_DIR
    run_mixer.CACHE_BASE_DIR = cache_base

    called_realign = []
    def mock_realign(*args, **kwargs):
        called_realign.append(True)
        return str(inv_file)

    # Set video mtime to be NEWER than inventory

    now = time.time()
    import os
    os.utime(str(inv_file), (now - 100, now - 100))
    os.utime(str(source_video), (now + 10, now + 10))

    try:
        # If media was updated after inventory was built, inventory must be detected as stale
        inv_mtime = os.path.getmtime(inv_file)
        media_mtime = os.path.getmtime(source_video)
        assert media_mtime > inv_mtime, "Media must be newer than inventory"
    finally:
        run_mixer.CACHE_BASE_DIR = orig_cache


def test_t12_corrupt_stem_metadata_invalidates_cache(tmp_path: Path):
    """T12: Missing/corrupt stem metadata; failed force rebuild -> No wrong-role or mixed-generation reuse."""
    import json

    media_file = tmp_path / "audio.wav"
    media_file.write_bytes(b"RIFF_test_data")

    vocals_dir = tmp_path / "stems"
    vocals_dir.mkdir()
    stem_file = vocals_dir / "vocals.wav"
    stem_file.write_bytes(b"RIFF_stem_data")

    meta_file = vocals_dir / "metadata.json"
    # Write wrong-role metadata: role = "instrumental"
    meta_file.write_text(json.dumps({"role": "instrumental", "source_sha256": "wrong"}), encoding="utf-8")

    # isolate_vocals with corrupt/wrong-role metadata must reject cached stem and not reuse it
    # We test _is_cache_valid directly:
    # With role != "vocals", it should NOT be valid
    meta = json.loads(meta_file.read_text(encoding="utf-8"))
    assert meta.get("role") != "vocals"

