"""
Adversarial test suite by teamwork_preview_challenger_m1_2:
Milestone 1 Verification:
1. Stem discrimination: strictly reject 'no_vocals.wav' and negative terms even when mtime is newer.
2. Stem cache invalidation: modifying source audio invalidates stem cache via SHA-256 hash checking.
3. Recursive globbing: rglob correctly discovers nested chunk files and TextGrids in multi-speaker directories.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from inventory_builder import build_units_inventory
from vocal_isolation import (
    _find_instrumental_candidates,
    _find_vocal_candidates,
    compute_file_sha256,
    get_source_id,
    isolate_vocals,
)


# ============================================================================
# 1. Stem Discrimination: Strict Rejection of no_vocals.wav Even With Newer mtime
# ============================================================================

def test_find_vocal_candidates_rejects_no_vocals_with_newer_mtime(tmp_path: Path):
    """
    Empirical Challenge 1:
    Even when no_vocals.wav is created/modified later than vocals.wav (newer mtime),
    _find_vocal_candidates MUST completely exclude no_vocals.wav and return vocals.wav.
    """
    test_dir = tmp_path / "demucs_run"
    test_dir.mkdir(parents=True)

    vocal_stem = test_dir / "vocals.wav"
    vocal_stem.write_bytes(b"RIFF_vocal_audio_data")

    # Set older mtime on vocals.wav
    t_old = time.time() - 3600  # 1 hour ago
    os.utime(vocal_stem, (t_old, t_old))

    # Create no_vocals.wav with newer mtime
    no_vocal_stem = test_dir / "no_vocals.wav"
    no_vocal_stem.write_bytes(b"RIFF_instrumental_audio_data")
    t_new = time.time()  # now
    os.utime(no_vocal_stem, (t_new, t_new))

    assert no_vocal_stem.stat().st_mtime > vocal_stem.stat().st_mtime

    candidates = _find_vocal_candidates(test_dir)
    assert vocal_stem in candidates
    assert no_vocal_stem not in candidates
    assert len(candidates) == 1
    assert candidates[0] == vocal_stem

    # Conversely, verify _find_instrumental_candidates finds no_vocals.wav and excludes vocals.wav
    inst_candidates = _find_instrumental_candidates(test_dir)
    assert no_vocal_stem in inst_candidates
    assert vocal_stem not in inst_candidates


@pytest.mark.parametrize(
    "negative_filename",
    [
        "no_vocals.wav",
        "no-vocals.wav",
        "novocals.wav",
        "NO_VOCALS.WAV",
        "track_no_vocals.flac",
        "non_vocal.wav",
        "non-vocal.mp3",
        "instrumental.wav",
        "accompaniment.wav",
        "drums.wav",
        "bass.wav",
        "other.wav",
        "subfolder/no_vocals.wav",
    ],
)
def test_find_vocal_candidates_rejects_all_negative_variations(tmp_path: Path, negative_filename: str):
    """
    Empirical Challenge 1 (Parametric):
    Verify every negative term variation is strictly rejected from vocal candidates.
    """
    test_dir = tmp_path / "test_neg"
    neg_path = test_dir / negative_filename
    neg_path.parent.mkdir(parents=True, exist_ok=True)
    neg_path.write_bytes(b"RIFF_negative_content")

    # Ensure timestamp is fresh
    now = time.time()
    os.utime(neg_path, (now, now))

    candidates = _find_vocal_candidates(test_dir)
    assert neg_path not in candidates
    assert len(candidates) == 0


def test_isolate_vocals_selects_vocals_over_newer_no_vocals(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Empirical Challenge 1 (End-to-End isolate_vocals):
    When separator creates both vocals.wav and no_vocals.wav (where no_vocals.wav is written last),
    isolate_vocals must copy vocals.wav, never no_vocals.wav.
    """
    src_file = tmp_path / "song.mp4"
    src_file.write_bytes(b"dummy_media_source_payload")

    vocals_cache_dir = tmp_path / "cache_vocals"

    def mock_run(cmd, *args, **kwargs):
        out_dir = Path(cmd[cmd.index("-o") + 1 if "-o" in cmd else cmd[cmd.index("--output_dir") + 1]])
        nested = out_dir / "htdemucs" / "song"
        nested.mkdir(parents=True, exist_ok=True)

        v_file = nested / "vocals.wav"
        v_file.write_bytes(b"RIFF_GENUINE_VOCALS")
        t_v = time.time() - 500
        os.utime(v_file, (t_v, t_v))

        # no_vocals is created after vocals with newer timestamp
        nv_file = nested / "no_vocals.wav"
        nv_file.write_bytes(b"RIFF_BACKGROUND_MUSIC")
        t_nv = time.time()
        os.utime(nv_file, (t_nv, t_nv))

        return MagicMock(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: f"/mock/{name}")
    monkeypatch.setattr("subprocess.run", mock_run)

    result_path = isolate_vocals(
        source_path=src_file,
        output_dir=vocals_cache_dir,
        fallback_to_demucs=True,
    )

    assert result_path.exists()
    assert result_path.name == "vocals.wav"
    assert result_path.read_bytes() == b"RIFF_GENUINE_VOCALS"

    # Instrumental stem should capture no_vocals
    inst_path = result_path.parent / "instrumental.wav"
    assert inst_path.exists()
    assert inst_path.read_bytes() == b"RIFF_BACKGROUND_MUSIC"


# ============================================================================
# 2. Cache Invalidation: Source Modification Invalidates Stem Cache via SHA-256
# ============================================================================

def test_source_audio_modification_invalidates_stem_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Empirical Challenge 2:
    1. Populate valid cache with matching SHA-256 hash in metadata.json.
    2. Calling isolate_vocals returns cached stem without running separator.
    3. Modify source file audio content (altering SHA-256).
    4. Next call to isolate_vocals MUST detect the hash mismatch, invalidate the cache,
       unlink old stems, and re-run separation.
    """
    src_file = tmp_path / "source_speech.wav"
    src_file.write_bytes(b"ORIGINAL_AUDIO_PAYLOAD_V1")
    initial_hash = compute_file_sha256(src_file)

    vocals_base = tmp_path / "vocals_cache"
    sid = get_source_id(src_file)
    target_dir = vocals_base / sid
    target_dir.mkdir(parents=True)

    vocal_wav = target_dir / "vocals.wav"
    vocal_wav.write_bytes(b"RIFF_VOCALS_V1")
    inst_wav = target_dir / "instrumental.wav"
    inst_wav.write_bytes(b"RIFF_INST_V1")

    meta_file = target_dir / "metadata.json"
    meta_file.write_text(
        json.dumps({
            "source_id": sid,
            "source_path": str(src_file),
            "source_sha256": initial_hash,
            "model_name": "test_model",
        }),
        encoding="utf-8",
    )

    separation_count = 0

    def mock_separator_run(cmd, *args, **kwargs):
        nonlocal separation_count
        separation_count += 1
        out_d = Path(cmd[cmd.index("--output_dir") + 1])
        (out_d / "vocals.wav").write_bytes(b"RIFF_VOCALS_V2_FRESH")
        return MagicMock(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: "mock_audio_separator")
    monkeypatch.setattr("subprocess.run", mock_separator_run)

    # Step 1: Initial call with untouched source -> CACHE HIT
    stem_1 = isolate_vocals(src_file, output_dir=vocals_base, force=False)
    assert stem_1 == vocal_wav
    assert stem_1.read_bytes() == b"RIFF_VOCALS_V1"
    assert separation_count == 0  # No separation executed

    # Step 2: Modify source audio file
    src_file.write_bytes(b"MODIFIED_AUDIO_PAYLOAD_V2_DIFFERENT_HASH")
    new_hash = compute_file_sha256(src_file)
    assert new_hash != initial_hash

    # Step 3: Call isolate_vocals with modified source -> CACHE INVALIDATED & REBUILD
    stem_2 = isolate_vocals(src_file, output_dir=vocals_base, force=False)
    assert separation_count == 1  # Separation was invoked due to hash mismatch
    assert stem_2.read_bytes() == b"RIFF_VOCALS_V2_FRESH"

    # Verify metadata.json was updated with the new hash
    new_meta_file = vocals_base / get_source_id(src_file) / "metadata.json"
    updated_meta = json.loads(new_meta_file.read_text(encoding="utf-8"))
    assert updated_meta["source_sha256"] == new_hash


def test_empty_or_zero_byte_cached_stem_triggers_reseparation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Empirical Challenge 2 (Edge case):
    If vocal stem exists but is 0 bytes (e.g. aborted previous write),
    cache is considered invalid and re-separation is executed.
    """
    src_file = tmp_path / "source.mp4"
    src_file.write_bytes(b"MEDIA_DATA")

    vocals_base = tmp_path / "vocals_cache"
    sid = get_source_id(src_file)
    target_dir = vocals_base / sid
    target_dir.mkdir(parents=True)

    zero_vocal = target_dir / "vocals.wav"
    zero_vocal.write_bytes(b"")  # 0 bytes

    meta_file = target_dir / "metadata.json"
    meta_file.write_text(
        json.dumps({"source_sha256": compute_file_sha256(src_file)}),
        encoding="utf-8",
    )

    rerun = False

    def mock_run(cmd, *args, **kwargs):
        nonlocal rerun
        rerun = True
        out_d = Path(cmd[cmd.index("--output_dir") + 1])
        (out_d / "vocals.wav").write_bytes(b"RIFF_VALID_STEM")
        return MagicMock(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: "mock_sep")
    monkeypatch.setattr("subprocess.run", mock_run)

    result = isolate_vocals(src_file, output_dir=vocals_base, force=False)
    assert rerun is True
    assert result.stat().st_size > 0


# ============================================================================
# 3. Recursive Globbing: rglob Discovers Nested Chunks and TextGrids
# ============================================================================

def test_rglob_deeply_nested_chunks_and_textgrids(tmp_path: Path):
    """
    Empirical Challenge 3:
    Verify that recursive globbing (rglob) correctly discovers chunks and TextGrids
    in arbitrary multi-speaker nested folder hierarchies (speaker/scene/take/...).
    """
    corpus_dir = tmp_path / "corpus"
    aligned_dir = tmp_path / "aligned"

    # Create deeply nested multi-speaker paths
    paths_wav = [
        corpus_dir / "speaker_alpha" / "chunk_001.wav",
        corpus_dir / "speaker_alpha" / "takes" / "chunk_002.wav",
        corpus_dir / "speaker_beta" / "scene_1" / "sub" / "chunk_003.wav",
        corpus_dir / "speaker_gamma" / "chunk_004.wav",
    ]
    paths_tg = [
        aligned_dir / "speaker_alpha" / "chunk_001.TextGrid",
        aligned_dir / "speaker_alpha" / "takes" / "chunk_002.TextGrid",
        aligned_dir / "speaker_beta" / "scene_1" / "sub" / "chunk_003.TextGrid",
        aligned_dir / "speaker_gamma" / "chunk_004.TextGrid",
    ]

    for p in paths_wav:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"RIFF_chunk")

    for p in paths_tg:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("dummy_textgrid", encoding="utf-8")

    # Flat glob must fail to find them (proving why rglob was required)
    assert len(list(corpus_dir.glob("chunk_*.wav"))) == 0
    assert len(list(aligned_dir.glob("*.TextGrid"))) == 0

    # rglob finds all 4 nested files
    found_wavs = sorted(corpus_dir.rglob("chunk_*.wav"))
    found_tgs = sorted(aligned_dir.rglob("*.TextGrid"))
    assert len(found_wavs) == 4
    assert len(found_tgs) == 4
    assert [p.name for p in found_wavs] == ["chunk_001.wav", "chunk_002.wav", "chunk_003.wav", "chunk_004.wav"]
    assert [p.name for p in found_tgs] == ["chunk_001.TextGrid", "chunk_002.TextGrid", "chunk_003.TextGrid", "chunk_004.TextGrid"]


def test_build_units_inventory_processes_nested_speaker_textgrids(tmp_path: Path):
    """
    Empirical Challenge 3 (End-to-End Inventory Builder):
    build_units_inventory must discover and index TextGrids situated inside
    nested speaker subdirectories.
    """
    aligned_dir = tmp_path / "aligned"
    spk1_dir = aligned_dir / "spk_anakin"
    spk2_dir = aligned_dir / "spk_obiwan" / "scene1"
    spk1_dir.mkdir(parents=True)
    spk2_dir.mkdir(parents=True)

    tg_anakin = spk1_dir / "chunk_001.TextGrid"
    tg_anakin.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
        'xmin = 0\nxmax = 1.0\ntiers? <exists>\nsize = 2\n'
        'item []:\n    item [1]:\n        class = "IntervalTier"\n'
        '        name = "words"\n        xmin = 0\n        xmax = 1.0\n'
        '        intervals: size = 1\n        intervals [1]:\n'
        '            xmin = 0\n            xmax = 1.0\n            text = "power"\n'
        '    item [2]:\n        class = "IntervalTier"\n'
        '        name = "phones"\n        xmin = 0\n        xmax = 1.0\n'
        '        intervals: size = 1\n        intervals [1]:\n'
        '            xmin = 0\n            xmax = 1.0\n            text = "P"\n',
        encoding="utf-8",
    )

    tg_obiwan = spk2_dir / "chunk_002.TextGrid"
    tg_obiwan.write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
        'xmin = 0\nxmax = 1.0\ntiers? <exists>\nsize = 2\n'
        'item []:\n    item [1]:\n        class = "IntervalTier"\n'
        '        name = "words"\n        xmin = 0\n        xmax = 1.0\n'
        '        intervals: size = 1\n        intervals [1]:\n'
        '            xmin = 0\n            xmax = 1.0\n            text = "ground"\n'
        '    item [2]:\n        class = "IntervalTier"\n'
        '        name = "phones"\n        xmin = 0\n        xmax = 1.0\n'
        '        intervals: size = 1\n        intervals [1]:\n'
        '            xmin = 0\n            xmax = 1.0\n            text = "G"\n',
        encoding="utf-8",
    )

    manifest_file = tmp_path / "corpus_manifest.json"
    manifest_file.write_text(
        json.dumps({
            "source_video": str(tmp_path / "source.mp4"),
            "chunks": {
                "chunk_001": {"start_sec": 0.0, "end_sec": 1.0, "text": "power"},
                "chunk_002": {"start_sec": 5.0, "end_sec": 6.0, "text": "ground"},
            },
        }),
        encoding="utf-8",
    )

    out_inventory = tmp_path / "output_inventory" / "units_index.json"

    build_units_inventory(
        aligned_dir=str(aligned_dir),
        manifest_path=str(manifest_file),
        output_index_path=str(out_inventory),
    )

    assert out_inventory.exists()
    data = json.loads(out_inventory.read_text(encoding="utf-8"))

    # Both words from nested TextGrids must be successfully indexed
    assert "words" in data
    indexed_words = {w["word_normalized"] for w in data["words"].values()}
    assert "power" in indexed_words
    assert "ground" in indexed_words
    assert "power" in data.get("word_to_word_ids", {})
    assert "ground" in data.get("word_to_word_ids", {})


def test_single_byte_flip_with_preserved_mtime_invalidates_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Empirical Challenge 2 (Adversarial):
    Flip 1 byte in source file while preserving file size and restoring exact st_mtime.
    Naive timestamp/size caches fail this test, but SHA-256 hash checking MUST detect it
    and invalidate the cache.
    """
    src_file = tmp_path / "audio_source.wav"
    original_data = bytearray(b"A" * 1024)
    src_file.write_bytes(original_data)
    fixed_time = time.time() - 7200
    os.utime(src_file, (fixed_time, fixed_time))

    vocals_base = tmp_path / "vocals_cache"
    sid = get_source_id(src_file)
    target_dir = vocals_base / sid
    target_dir.mkdir(parents=True)

    vocal_wav = target_dir / "vocals.wav"
    vocal_wav.write_bytes(b"RIFF_OLD_STEM")

    meta_file = target_dir / "metadata.json"
    meta_file.write_text(
        json.dumps({
            "source_id": sid,
            "source_sha256": compute_file_sha256(src_file),
        }),
        encoding="utf-8",
    )

    reseparation_called = False

    def mock_run(cmd, *args, **kwargs):
        nonlocal reseparation_called
        reseparation_called = True
        out_d = Path(cmd[cmd.index("--output_dir") + 1])
        (out_d / "vocals.wav").write_bytes(b"RIFF_FRESH_STEM")
        return MagicMock(returncode=0)

    monkeypatch.setattr("shutil.which", lambda name: "mock_sep")
    monkeypatch.setattr("subprocess.run", mock_run)

    # Flip exactly 1 byte in the middle of the file
    tampered_data = bytearray(original_data)
    tampered_data[512] = ord("B")
    assert len(tampered_data) == len(original_data)
    src_file.write_bytes(tampered_data)
    # Artificially restore exact original mtime
    os.utime(src_file, (fixed_time, fixed_time))

    assert src_file.stat().st_size == 1024
    assert src_file.stat().st_mtime == fixed_time

    # isolate_vocals must detect SHA-256 hash mismatch despite identical size and mtime
    res = isolate_vocals(src_file, output_dir=vocals_base, force=False)
    assert reseparation_called is True
    assert res.read_bytes() == b"RIFF_FRESH_STEM"


def test_mfa_batching_rglob_moves_nested_textgrids(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Empirical Challenge 3 (Adversarial MFA batching):
    When MFA batching runs (batches > 45 chunks) and MFA outputs TextGrids
    in nested speaker directories within b_aligned, b_aligned.rglob('*.TextGrid')
    must locate and move every TextGrid to out_aligned.
    """
    import run_mixer

    corpus_dir = tmp_path / "corpus"
    aligned_dir = tmp_path / "aligned"
    corpus_dir.mkdir(parents=True)
    aligned_dir.mkdir(parents=True)

    # Create 50 chunks (exceeding batch size of 45) across multiple speaker subdirectories
    for i in range(50):
        spk = f"spk_{i % 3}"
        spk_dir = corpus_dir / spk
        spk_dir.mkdir(parents=True, exist_ok=True)
        chunk_w = spk_dir / f"chunk_{i:03d}.wav"
        chunk_w.write_bytes(b"RIFF_data")
        chunk_t = spk_dir / f"chunk_{i:03d}.txt"
        chunk_t.write_text(f"text {i}", encoding="utf-8")

    # Mock _execute_mfa to simulate MFA producing nested TextGrids inside dst_dir
    def mock_execute_mfa(src_dir: Path, dst_dir: Path):
        for wav in src_dir.rglob("chunk_*.wav"):
            # Simulate MFA putting TextGrid in nested speaker folder
            speaker_sub = dst_dir / "speaker_subfolder"
            speaker_sub.mkdir(parents=True, exist_ok=True)
            tg = speaker_sub / f"{wav.stem}.TextGrid"
            tg.write_text("TextGrid content", encoding="utf-8")

    monkeypatch.setattr(run_mixer, "get_mamba_config", lambda: (None, None))
    # Intercept run_mfa_alignment's internal _execute_mfa by monkeypatching subprocess.run
    def mock_subp_run(cmd, *args, **kwargs):
        src_dir = Path(cmd[2])
        dst_dir = Path(cmd[5])
        mock_execute_mfa(src_dir, dst_dir)
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", mock_subp_run)

    run_mixer.run_mfa_align(
        corpus_dir=str(corpus_dir),
        output_aligned_dir=str(aligned_dir),
    )

    # Verify that all 50 TextGrids were moved to output_aligned_dir
    final_tgs = list(aligned_dir.rglob("*.TextGrid"))
    assert len(final_tgs) == 50

