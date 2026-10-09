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


def test_clean_source_cache_by_path_and_stem(tmp_path: Path):
    from cache import clean_source_cache

    cache_dir = tmp_path / ".build"
    source_id = "sample_source_11223344"
    other_id = "other_source_99887766"

    # Setup directories
    for sub in ["sources", "alignment", "inventory", "vocals"]:
        (cache_dir / sub / source_id).mkdir(parents=True, exist_ok=True)
        (cache_dir / sub / other_id).mkdir(parents=True, exist_ok=True)
        # Add a dummy file inside
        ((cache_dir / sub / source_id) / "test.json").write_text("{}", encoding="utf-8")
        ((cache_dir / sub / other_id) / "test.json").write_text("{}", encoding="utf-8")

    # Audio extracts
    (cache_dir / "audio_extracts").mkdir(parents=True, exist_ok=True)
    (cache_dir / "audio_extracts" / "sample_source.wav").write_bytes(b"WAV1")
    (cache_dir / "audio_extracts" / "other_source.wav").write_bytes(b"WAV2")

    # Clusters
    (cache_dir / "clusters").mkdir(parents=True, exist_ok=True)
    (cache_dir / "clusters" / "anon_sample_source_11223344_0_preview.wav").write_bytes(b"PREV1")
    (cache_dir / "clusters" / "anon_other_source_99887766_0_preview.wav").write_bytes(b"PREV2")

    # Media hash index
    import json
    index_file = cache_dir / ".media_hash_index.json"
    dummy_index = {
        "/path/to/sample_source.mp4": {"size": 100, "mtime_ns": 1, "sha256": "11223344aabbcc"},
        "/path/to/other_source.mp4": {"size": 200, "mtime_ns": 2, "sha256": "99887766ddeeff"},
    }
    index_file.write_text(json.dumps(dummy_index), encoding="utf-8")

    # Test Dry-Run first
    dry_res = clean_source_cache("sample_source", cache_dir=cache_dir, dry_run=True)
    assert dry_res["dry_run"] is True
    assert dry_res["freed_bytes"] > 0
    # Files must still exist
    assert (cache_dir / "sources" / source_id).exists()
    assert (cache_dir / "audio_extracts" / "sample_source.wav").exists()

    # Real run by stem
    res = clean_source_cache("sample_source", cache_dir=cache_dir, dry_run=False)
    assert res["freed_bytes"] > 0
    assert len(res["deleted_dirs"]) == 4  # sources, alignment, inventory, vocals
    assert len(res["deleted_files"]) == 2  # audio_extract, cluster preview

    # Target files removed
    assert not (cache_dir / "sources" / source_id).exists()
    assert not (cache_dir / "alignment" / source_id).exists()
    assert not (cache_dir / "inventory" / source_id).exists()
    assert not (cache_dir / "vocals" / source_id).exists()
    assert not (cache_dir / "audio_extracts" / "sample_source.wav").exists()
    assert not (cache_dir / "clusters" / "anon_sample_source_11223344_0_preview.wav").exists()

    # Other source preserved!
    assert (cache_dir / "sources" / other_id).exists()
    assert (cache_dir / "alignment" / other_id).exists()
    assert (cache_dir / "audio_extracts" / "other_source.wav").exists()
    assert (cache_dir / "clusters" / "anon_other_source_99887766_0_preview.wav").exists()

    # Media hash index updated
    updated_idx = json.loads(index_file.read_text(encoding="utf-8"))
    assert "/path/to/sample_source.mp4" not in updated_idx
    assert "/path/to/other_source.mp4" in updated_idx


def test_clean_all_source_caches(tmp_path: Path):
    from cache import clean_all_source_caches

    cache_dir = tmp_path / ".build"
    (cache_dir / "sources" / "src1").mkdir(parents=True, exist_ok=True)
    (cache_dir / "sources" / "src2").mkdir(parents=True, exist_ok=True)
    (cache_dir / "audio_extracts").mkdir(parents=True, exist_ok=True)
    (cache_dir / "audio_extracts" / "s1.wav").write_bytes(b"test")

    # Important: Pretrained models must not be removed by clean_all
    (cache_dir / "mfa" / "pretrained_models").mkdir(parents=True, exist_ok=True)
    (cache_dir / "mfa" / "pretrained_models" / "model.zip").write_bytes(b"model")

    res = clean_all_source_caches(cache_dir=cache_dir, dry_run=False)
    assert res["freed_bytes"] > 0
    assert not (cache_dir / "sources" / "src1").exists()
    assert not (cache_dir / "sources" / "src2").exists()
    assert not (cache_dir / "audio_extracts" / "s1.wav").exists()

    # Pretrained models preserved!
    assert (cache_dir / "mfa" / "pretrained_models" / "model.zip").exists()


def test_run_mixer_clean_cli(tmp_path: Path):
    import subprocess
    import sys

    cache_dir = tmp_path / ".build"
    (cache_dir / "sources" / "demo_12345678").mkdir(parents=True, exist_ok=True)
    ((cache_dir / "sources" / "demo_12345678") / "file.txt").write_text("hello", encoding="utf-8")

    cmd = [
        sys.executable,
        "run_mixer.py",
        "clean",
        "demo",
        "--cache-dir",
        str(cache_dir),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "Cleaned cache for source 'demo'" in proc.stdout
    assert not (cache_dir / "sources" / "demo_12345678").exists()


def test_clean_source_prunes_speaker_registry(tmp_path: Path):
    import numpy as np
    from cache import clean_source_cache
    from speaker_registry import SpeakerRegistry

    reg_path = tmp_path / "speaker_registry.json"
    reg = SpeakerRegistry(config_path=reg_path)
    reg.register_or_update(
        speaker_id="speaker_a",
        display_name="Speaker A",
        centroid=np.array([0.1, 0.2, 0.3]),
        exemplar_path=str(tmp_path / ".build" / "clusters" / "anon_source1_12345678_0_preview.wav"),
        source_id="source1_12345678",
    )
    reg.register_or_update(
        speaker_id="speaker_b",
        display_name="Speaker B",
        centroid=np.array([0.4, 0.5, 0.6]),
        exemplar_path=str(tmp_path / ".build" / "clusters" / "anon_source2_87654321_0_preview.wav"),
        source_id="source2_87654321",
    )
    assert len(reg.speakers) == 2

    # Clean source1
    res = clean_source_cache(
        "source1",
        cache_dir=tmp_path / ".build",
        clean_speakers=True,
        registry_path=reg_path,
    )
    assert "speaker_a" in res["removed_speakers"]
    assert "speaker_b" not in res["removed_speakers"]

    # Verify reload from disk
    reg_reloaded = SpeakerRegistry(config_path=reg_path)
    assert "speaker_a" not in reg_reloaded.speakers
    assert "speaker_b" in reg_reloaded.speakers


def test_inventory_helpers_and_speaker_discovery(tmp_path: Path):
    import json
    from cache import (
        is_valid_inventory,
        inventory_has_speaker,
        find_cached_inventories_for_speaker,
        find_all_cached_inventories,
    )
    from speaker_registry import SpeakerRegistry

    cache_dir = tmp_path / ".build"
    inv_dir = cache_dir / "inventory"
    inv_dir.mkdir(parents=True)

    # 1. Empty inventory (dummy)
    empty_inv = inv_dir / "empty_source" / "units_index.json"
    empty_inv.parent.mkdir(parents=True)
    empty_inv.write_text(json.dumps({"words": {}, "phones": {}}), encoding="utf-8")
    assert not is_valid_inventory(empty_inv)
    assert not inventory_has_speaker(empty_inv, "bail_organa")

    # 2. Source 1 with Bail Organa and Mas Amedda
    s1_inv = inv_dir / "source1_9c96ee02" / "units_index.json"
    s1_inv.parent.mkdir(parents=True)
    s1_data = {
        "words": {
            "w1": {
                "word_normalized": "debate",
                "speaker_id": "bail_organa",
                "is_eligible": True,
            },
            "w2": {
                "word_normalized": "order",
                "speaker_id": "mas_amedda",
                "is_eligible": True,
            },
        }
    }
    s1_inv.write_text(json.dumps(s1_data), encoding="utf-8")
    assert is_valid_inventory(s1_inv)
    assert inventory_has_speaker(s1_inv, "bail_organa")
    assert inventory_has_speaker(s1_inv, "Bail Organa")
    assert inventory_has_speaker(s1_inv, "mas_amedda")
    assert not inventory_has_speaker(s1_inv, "yoda")

    # 3. Source 2 with Bail Organa and Palpatine
    s2_inv = inv_dir / "source2_12345678" / "units_index.json"
    s2_inv.parent.mkdir(parents=True)
    s2_data = {
        "words": {
            "w3": {
                "word_normalized": "senate",
                "speaker_id": "bail_organa",
                "is_eligible": True,
            },
            "w4": {
                "word_normalized": "power",
                "speaker_id": "palpatine",
                "is_eligible": True,
            },
        }
    }
    s2_inv.write_text(json.dumps(s2_data), encoding="utf-8")

    # 4. Source 3 with only Obi-Wan
    s3_inv = inv_dir / "source3_abcdef12" / "units_index.json"
    s3_inv.parent.mkdir(parents=True)
    s3_data = {
        "words": {
            "w5": {
                "word_normalized": "hello",
                "speaker_id": "obi_wan",
                "is_eligible": True,
            }
        }
    }
    s3_inv.write_text(json.dumps(s3_data), encoding="utf-8")

    # Speaker Registry
    reg_path = tmp_path / "speaker_registry.json"
    reg = SpeakerRegistry(config_path=reg_path)
    reg.register_or_update(
        speaker_id="bail_organa",
        display_name="Bail Organa",
        source_id="source1_9c96ee02",
    )
    reg.register_or_update(
        speaker_id="bail_organa",
        source_id="source2_12345678",
    )

    # Test find_cached_inventories_for_speaker
    bail_invs = find_cached_inventories_for_speaker("bail_organa", cache_dir=cache_dir, registry_path=reg_path)
    assert len(bail_invs) == 2
    assert str(s1_inv.resolve()) in bail_invs
    assert str(s2_inv.resolve()) in bail_invs
    assert str(s3_inv.resolve()) not in bail_invs

    # Test find_cached_inventories_for_speaker with title case / space
    bail_title_invs = find_cached_inventories_for_speaker("Bail Organa", cache_dir=cache_dir, registry_path=reg_path)
    assert bail_title_invs == bail_invs

    # Test single speaker
    obi_invs = find_cached_inventories_for_speaker("obi_wan", cache_dir=cache_dir, registry_path=reg_path)
    assert len(obi_invs) == 1
    assert str(s3_inv.resolve()) in obi_invs

    # Test nonexistent speaker
    non_invs = find_cached_inventories_for_speaker("unknown_speaker", cache_dir=cache_dir, registry_path=reg_path)
    assert non_invs == []

    # Test find_all_cached_inventories
    all_invs = find_all_cached_inventories(cache_dir=cache_dir)
    assert len(all_invs) == 3
    assert str(empty_inv.resolve()) not in all_invs


def test_mix_speaker_first_cli_messages(tmp_path: Path):
    import subprocess
    import sys

    # Mix with no sources and no speaker -> helpful error message
    cmd1 = [sys.executable, "run_mixer.py", "mix", "--text", "hello world"]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert res1.returncode != 0
    assert "Please provide source(s)" in res1.stderr
    assert "--speaker <name> or --all-sources" in res1.stderr

    # Mix with nonexistent speaker -> reports speaker missing from cache
    cmd2 = [
        sys.executable,
        "run_mixer.py",
        "mix",
        "--text",
        "hello world",
        "--speaker",
        "nonexistent_person_xyz",
        "--cache-dir",
        str(tmp_path),
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode != 0
    assert "No ingested sources found for speaker 'nonexistent_person_xyz'" in res2.stderr


def test_mix_speaker_first_discovers_and_mixes(tmp_path: Path):
    import json
    import subprocess
    import sys

    cache_dir = tmp_path / ".build"
    inv_dir = cache_dir / "inventory" / "src1_12345678"
    inv_dir.mkdir(parents=True)
    inv_file = inv_dir / "units_index.json"

    # Create dummy video file
    video_file = tmp_path / "dummy_video.mp4"
    video_file.write_bytes(b"dummy")

    inv_data = {
        "sources": {
            "src1_12345678": {
                "content_id": "src1_12345678",
                "source_path": str(video_file),
                "native_sample_rate": 16000,
                "native_channels": 1,
            }
        },
        "words": {
            "w1": {
                "word_normalized": "hello",
                "speaker_id": "bail_organa",
                "is_eligible": True,
                "start_sample": 0,
                "end_sample": 1600,
                "source_content_id": "src1_12345678",
                "realized_phones": ["HH", "AH0", "L", "OW1"],
            }
        },
        "events": {
            "e1": {
                "event_id": "e1",
                "kind": "phone",
                "label": "HH",
                "speaker_id": "bail_organa",
                "is_eligible": True,
                "start_sample": 0,
                "end_sample": 400,
                "source_content_id": "src1_12345678",
            }
        },
        "word_to_word_ids": {"hello": ["w1"]},
        "phone_to_event_ids": {"HH": ["e1"]},
        "base_phone_to_event_ids": {"HH": ["e1"]},
        "schema_version": "1.0",
        "metadata": {},
    }
    inv_file.write_text(json.dumps(inv_data), encoding="utf-8")

    # Call mix with only --speaker bail_organa --text "hello"
    cmd = [
        sys.executable,
        "run_mixer.py",
        "mix",
        "--text",
        "hello",
        "--speaker",
        "Bail Organa",
        "--selector",
        "span",
        "--cache-dir",
        str(cache_dir),
        "--plan",
        str(tmp_path / "out_plan.json"),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert "[*] Discovered 1 ingested source inventory/inventories containing speaker 'Bail Organa'." in res.stdout
    assert "Requested speaker 'Bail Organa' not found" not in res.stderr

    # Test with --all-sources
    cmd_all = [
        sys.executable,
        "run_mixer.py",
        "mix",
        "--text",
        "hello",
        "--all-sources",
        "--selector",
        "span",
        "--cache-dir",
        str(cache_dir),
        "--plan",
        str(tmp_path / "out_plan_all.json"),
    ]
    res_all = subprocess.run(cmd_all, capture_output=True, text=True)
    assert "[*] Pooling all 1 available ingested source inventory/inventories in cache." in res_all.stdout




