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
