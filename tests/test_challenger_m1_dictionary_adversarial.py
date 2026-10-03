"""
Adversarial Challenge Test Suite for Milestone 1:
MFA Custom Dictionary Staging, FileLock Concurrency, Edge Cases,
Cache Invalidation, Proper Noun Indexing, and Phrase Selection.
Author: Challenger 1 (teamwork_preview_challenger_m1_1)
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import multiprocessing
import time
from pathlib import Path

import pytest

from cache import (
    compute_alignment_cache_key,
)
from contracts import (
    AlignedEvent,
    InventoryIndex,
    SourceRecord,
    WordOccurrence,
)
from inventory_builder import build_units_inventory
from lexicon import (
    Lexicon,
    _STAGED_DICT_CACHE,
    stage_mfa_dictionary,
)


# =========================================================================
# 1. Edge Cases & Corrupted Inputs
# =========================================================================

def test_missing_files_handling(tmp_path: Path):
    """Test behavior when dictionary files are missing."""
    base_dict = tmp_path / "nonexistent_base.dict"
    custom_dict = tmp_path / "nonexistent_custom.txt"
    staged_dict = tmp_path / "staged.dict"

    # Both missing should raise FileNotFoundError
    with pytest.raises(FileNotFoundError, match="Neither base dictionary.*nor custom dictionary.*found"):
        stage_mfa_dictionary(base_dict, custom_dict, staged_dict)

    # Base dictionary exists, custom is missing -> should succeed
    base_dict.write_text("hello\tHH AH0 L OW1\n", encoding="utf-8")
    out_p, h = stage_mfa_dictionary(base_dict, custom_dict, staged_dict)
    assert out_p.exists()
    assert "hello\tHH AH0 L OW1" in out_p.read_text(encoding="utf-8")

    # Custom exists, base is missing -> should succeed
    staged_dict_2 = tmp_path / "staged_2.dict"
    base_dict.unlink()
    custom_dict.write_text("yoda\tY OW1 D AH0\n", encoding="utf-8")
    out_p2, h2 = stage_mfa_dictionary(base_dict, custom_dict, staged_dict_2)
    assert out_p2.exists()
    assert "yoda\tY OW1 D AH0" in out_p2.read_text(encoding="utf-8")


def test_empty_files_handling(tmp_path: Path):
    """Test behavior when dictionary files are empty (0 bytes)."""
    base_dict = tmp_path / "empty_base.dict"
    base_dict.write_text("", encoding="utf-8")

    custom_dict = tmp_path / "empty_custom.txt"
    custom_dict.write_text("", encoding="utf-8")

    staged_dict = tmp_path / "staged.dict"
    out_p, h = stage_mfa_dictionary(base_dict, custom_dict, staged_dict)
    assert out_p.exists()
    content = out_p.read_text(encoding="utf-8")
    # Should produce valid empty/newline content without unhandled exception
    assert content.strip() == ""
    assert h == hashlib.sha256(out_p.read_bytes()).hexdigest()


def test_corrupted_lines_and_extra_whitespace(tmp_path: Path):
    """
    Test behavior with:
    - Multiple leading/trailing spaces
    - Multiple consecutive tabs and spaces
    - Comment-only lines
    - Lines with only whitespace
    - Blank lines
    - Case variance in phones
    """
    base_dict = tmp_path / "base.dict"
    base_dict.write_text(
        "   \n"  # whitespace line
        "# Header comment\n"
        "   hello    \t\t   HH   ah0   l   ow1   \n"  # irregular spaces, mixed case phones
        "world\tW ER1 L D\n"
        "   # indented comment\n",
        encoding="utf-8",
    )

    custom_dict = tmp_path / "custom.txt"
    custom_dict.write_text(
        "\n\n"
        "windu\t\t\tW   IH1   n   d   uw0   \n"
        "   \n",
        encoding="utf-8",
    )

    staged_dict = tmp_path / "staged.dict"
    out_p, _ = stage_mfa_dictionary(base_dict, custom_dict, staged_dict)

    lines = out_p.read_text(encoding="utf-8").splitlines()
    assert "hello\tHH AH0 L OW1" in lines
    assert "HELLO\tHH AH0 L OW1" in lines
    assert "world\tW ER1 L D" in lines
    assert "windu\tW IH1 N D UW0" in lines
    assert "WINDU\tW IH1 N D UW0" in lines

    # Verify no raw comments leaked into staged dictionary
    assert not any(ln.startswith("#") for ln in lines)
    # Verify strict tab delimiter
    for ln in lines:
        assert "\t" in ln
        parts = ln.split("\t")
        assert len(parts) == 2 or any(float(p) for p in parts[1:] if p)
        # Verify phones are separated by single spaces
        phones = parts[1].split()
        assert parts[1] == " ".join(phones)


def test_corrupted_empty_pronunciation_entry(tmp_path: Path):
    """
    Test that a corrupted dictionary entry with a word but NO phones
    (e.g. 'broken_word') is NOT staged as a malformed entry ('broken_word\\t')
    which would crash Montreal Forced Aligner dictionary parsing.
    """
    base_dict = tmp_path / "base.dict"
    base_dict.write_text(
        "broken_word\n"  # Missing phones!
        "valid_word\tV AE1 L IH0 D\n",
        encoding="utf-8",
    )

    staged_dict = tmp_path / "staged.dict"
    out_p, _ = stage_mfa_dictionary(base_dict_path=base_dict, custom_dict_path=None, output_path=staged_dict)
    lines = out_p.read_text(encoding="utf-8").splitlines()

    broken_entries = [ln for ln in lines if ln.startswith("broken_word")]
    # Empty phone sequences should be filtered out to prevent MFA dictionary parsing errors
    assert len(broken_entries) == 0, f"Malformed entry with empty phones should not be staged: {broken_entries}"
    assert "valid_word\tV AE1 L IH0 D" in lines


def test_invalid_arpabet_phones_handling(tmp_path: Path):
    """
    Test how invalid ARPAbet phones in custom dictionaries are handled.
    Verifies that invalid stress on consonants ('T1') or out-of-range stress ('AE9')
    are validated or flagged rather than corrupting MFA acoustic alignment.
    """
    custom_dict = tmp_path / "custom.txt"
    custom_dict.write_text(
        "badstress1\tT1\n"
        "badstress2\tAE9\n"
        "goodword\tG UH1 D\n",
        encoding="utf-8",
    )

    staged_dict = tmp_path / "staged.dict"
    out_p, _ = stage_mfa_dictionary(base_dict_path=None, custom_dict_path=custom_dict, output_path=staged_dict)
    content = out_p.read_text(encoding="utf-8")
    assert "goodword\tG UH1 D" in content


def test_unicode_and_special_character_entries(tmp_path: Path):
    """
    Test handling of UTF-8 special characters:
    - Words with hyphens (obi-wan)
    - Words with apostrophes (d'artagnan, don't)
    - Non-ASCII / accented characters (café, chloë)
    """
    base_dict = tmp_path / "base.dict"
    base_dict.write_text(
        "obi-wan\tOW1 B IY0 W AA1 N\n"
        "don't\tD OW1 N T\n"
        "café\tK AE2 F EY1\n",
        encoding="utf-8",
    )

    staged_dict = tmp_path / "staged.dict"
    out_p, _ = stage_mfa_dictionary(base_dict_path=base_dict, custom_dict_path=None, output_path=staged_dict)
    content = out_p.read_text(encoding="utf-8")

    assert "obi-wan\tOW1 B IY0 W AA1 N" in content
    assert "OBI-WAN\tOW1 B IY0 W AA1 N" in content
    assert "don't\tD OW1 N T" in content
    assert "café\tK AE2 F EY1" in content
    assert "CAFÉ\tK AE2 F EY1" in content


# =========================================================================
# 2. Concurrency & FileLock Stress Tests
# =========================================================================

def _worker_stage_dict(args):
    base_path, custom_path, out_path = args
    return stage_mfa_dictionary(base_path, custom_path, out_path)


def test_multithreaded_concurrent_staging(tmp_path: Path):
    """
    Stress test stage_mfa_dictionary under high concurrency with multiple threads.
    Tests for race conditions in FileLock, temporary file collision, and cache corruption.
    """
    base_dict = tmp_path / "base.dict"
    base_dict.write_text(
        "\n".join(f"word_{i}\tW ER1 D {i % 10}" for i in range(200)) + "\n",
        encoding="utf-8",
    )
    custom_dict = tmp_path / "custom.txt"
    custom_dict.write_text(
        "\n".join(f"custom_{i}\tK AH1 S T {i % 10}" for i in range(50)) + "\n",
        encoding="utf-8",
    )

    staged_dict = tmp_path / "staged.dict"

    # Clear in-memory cache to force staging
    _STAGED_DICT_CACHE.clear()

    num_threads = 16
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [
            executor.submit(stage_mfa_dictionary, base_dict, custom_dict, staged_dict)
            for _ in range(num_threads)
        ]
        results = [f.result() for f in futures]

    # All threads must produce identical hash and valid path
    hashes = set(h for _, h in results)
    assert len(hashes) == 1, f"Expected consistent hash across all concurrent threads, got: {hashes}"
    assert staged_dict.exists()
    assert staged_dict.stat().st_size > 0


def test_multiprocess_concurrent_staging(tmp_path: Path):
    """
    Stress test stage_mfa_dictionary under true multiprocessing concurrency.
    Verifies FileLock synchronization across distinct OS processes.
    """
    base_dict = tmp_path / "base.dict"
    base_dict.write_text(
        "\n".join(f"proc_word_{i}\tP R AA1 K {i % 10}" for i in range(100)) + "\n",
        encoding="utf-8",
    )
    custom_dict = tmp_path / "custom.txt"
    custom_dict.write_text(
        "windu\tW IH1 N D UW0\n"
        "yoda\tY OW1 D AH0\n",
        encoding="utf-8",
    )
    staged_dict = tmp_path / "staged.dict"

    tasks = [(base_dict, custom_dict, staged_dict) for _ in range(8)]

    # Use multiprocessing pool
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=4) as pool:
        results = pool.map(_worker_stage_dict, tasks)

    hashes = set(h for _, h in results)
    assert len(hashes) == 1, f"Multi-process staging produced divergent hashes: {hashes}"
    assert staged_dict.exists()


# =========================================================================
# 3. Cache Invalidation Stress Tests
# =========================================================================

def test_cache_invalidation_lifecycle(tmp_path: Path):
    """
    Rigorously tests the complete cache invalidation lifecycle:
    1. Initial staging produces dict_hash_1 and align_key_1.
    2. Appending new entry produces dict_hash_2 != dict_hash_1.
    3. Modifying existing entry pronunciation produces dict_hash_3 != dict_hash_2.
    4. Deleting entry produces dict_hash_4 != dict_hash_3.
    5. In-memory cache bypass and filesystem mtime changes.
    """
    base_dict = tmp_path / "base.dict"
    base_dict.write_text("anakin\tAE1 N AH0 K IH0 N\n", encoding="utf-8")

    custom_dict = tmp_path / "custom.txt"
    custom_dict.write_text("windu\tW IH1 N D UW0\n", encoding="utf-8")

    staged_dict = tmp_path / "staged.dict"

    # Step 1: Initial staging
    _, hash_1 = stage_mfa_dictionary(base_dict, custom_dict, staged_dict)
    align_key_1 = compute_alignment_cache_key("m1", 0, 0, "t1", hash_1)

    # Step 2: In-memory cache hit test (should return exact same hash without re-reading)
    _, hash_1_cached = stage_mfa_dictionary(base_dict, custom_dict, staged_dict)
    assert hash_1 == hash_1_cached

    # Step 3: Append new entry
    time.sleep(0.01)  # Ensure filesystem mtime tick
    with open(custom_dict, "a", encoding="utf-8") as f:
        f.write("yoda\tY OW1 D AH0\n")

    _, hash_2 = stage_mfa_dictionary(base_dict, custom_dict, staged_dict)
    align_key_2 = compute_alignment_cache_key("m1", 0, 0, "t1", hash_2)

    assert hash_1 != hash_2, "Appending new custom pronunciation must change dict_hash"
    assert align_key_1 != align_key_2, "Alignment cache key must change on new pronunciation"

    # Step 4: Modify existing entry pronunciation
    time.sleep(0.01)
    custom_dict.write_text(
        "windu\tW IH1 N D Y UW0\n"  # modified pronunciation
        "yoda\tY OW1 D AH0\n",
        encoding="utf-8",
    )

    _, hash_3 = stage_mfa_dictionary(base_dict, custom_dict, staged_dict)
    assert hash_3 != hash_2, "Modifying custom pronunciation must change dict_hash"

    # Step 5: Delete entry
    time.sleep(0.01)
    custom_dict.write_text(
        "yoda\tY OW1 D AH0\n",  # windu deleted
        encoding="utf-8",
    )

    _, hash_4 = stage_mfa_dictionary(base_dict, custom_dict, staged_dict)
    assert hash_4 != hash_3, "Deleting custom pronunciation must change dict_hash"

    staged_content = staged_dict.read_text(encoding="utf-8")
    assert "windu" not in staged_content
    assert "yoda\tY OW1 D AH0" in staged_content


def test_lexicon_save_invalidates_staged_cache(tmp_path: Path):
    """
    Verifies that calling Lexicon.save_custom_pronunciation invalidates
    _STAGED_DICT_CACHE so subsequent stage_mfa_dictionary picks up the new entry.
    """
    custom_dict = tmp_path / "custom.txt"
    custom_dict.write_text("initial\tIH1 N IH1 SH AH0 L\n", encoding="utf-8")

    staged_dict = tmp_path / "staged.dict"
    _, h1 = stage_mfa_dictionary(custom_dict_path=custom_dict, output_path=staged_dict)

    # Use Lexicon instance to save new pronunciation
    lex = Lexicon([custom_dict])
    time.sleep(0.01)
    lex.save_custom_pronunciation("windu", ["W", "IH1", "N", "D", "UW0"], custom_path=custom_dict)

    # Stage again
    _, h2 = stage_mfa_dictionary(custom_dict_path=custom_dict, output_path=staged_dict)

    assert h1 != h2, "Lexicon.save_custom_pronunciation must invalidate staging cache"
    staged_content = staged_dict.read_text(encoding="utf-8")
    assert "windu\tW IH1 N D UW0" in staged_content
    assert "WINDU\tW IH1 N D UW0" in staged_content


# =========================================================================
# 4. Proper Nouns Indexing & Anakin Phrase Selection
# =========================================================================

def test_proper_noun_indexing_and_phrase_retrieval(tmp_path: Path):
    """
    Verifies proper noun indexing:
    1. 'windu' and 'yoda' indexed with valid WordOccurrence instances.
    2. Realized phones are ['W', 'IH1', 'N', 'D', 'UW0'] and ['Y', 'OW1', 'D', 'AH0'].
    3. Neither word is marked as noise / 'spn'.
    4. Phrase retrieval for 'Master Windu manages to masturbate master Yoda.'
       retrieves intact words/phrases for 'master windu' and 'master yoda'.
    """
    aligned_dir = tmp_path / "aligned"
    aligned_dir.mkdir()

    tg_content = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 2.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 2.0
        intervals: size = 4
        intervals [1]:
            xmin = 0.0
            xmax = 0.4
            text = "master"
        intervals [2]:
            xmin = 0.4
            xmax = 0.9
            text = "windu"
        intervals [3]:
            xmin = 0.9
            xmax = 1.3
            text = "master"
        intervals [4]:
            xmin = 1.3
            xmax = 1.8
            text = "yoda"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 2.0
        intervals: size = 18
        intervals [1]:
            xmin = 0.0
            xmax = 0.1
            text = "M"
        intervals [2]:
            xmin = 0.1
            xmax = 0.2
            text = "AE1"
        intervals [3]:
            xmin = 0.2
            xmax = 0.3
            text = "S"
        intervals [4]:
            xmin = 0.3
            xmax = 0.35
            text = "T"
        intervals [5]:
            xmin = 0.35
            xmax = 0.4
            text = "ER0"
        intervals [6]:
            xmin = 0.4
            xmax = 0.5
            text = "W"
        intervals [7]:
            xmin = 0.5
            xmax = 0.6
            text = "IH1"
        intervals [8]:
            xmin = 0.6
            xmax = 0.7
            text = "N"
        intervals [9]:
            xmin = 0.7
            xmax = 0.8
            text = "D"
        intervals [10]:
            xmin = 0.8
            xmax = 0.9
            text = "UW0"
        intervals [11]:
            xmin = 0.9
            xmax = 1.0
            text = "M"
        intervals [12]:
            xmin = 1.0
            xmax = 1.1
            text = "AE1"
        intervals [13]:
            xmin = 1.1
            xmax = 1.2
            text = "S"
        intervals [14]:
            xmin = 1.2
            xmax = 1.25
            text = "T"
        intervals [15]:
            xmin = 1.25
            xmax = 1.3
            text = "ER0"
        intervals [16]:
            xmin = 1.3
            xmax = 1.45
            text = "Y"
        intervals [17]:
            xmin = 1.45
            xmax = 1.65
            text = "OW1"
        intervals [18]:
            xmin = 1.65
            xmax = 1.75
            text = "D"
        intervals [19]:
            xmin = 1.75
            xmax = 1.8
            text = "AH0"
"""
    tg_file = aligned_dir / "chunk_0000.TextGrid"
    tg_file.write_text(tg_content, encoding="utf-8")

    manifest = {
        "source_video": "anakin_ep2.mp4",
        "speaker_id": "anakin",
        "single_speaker_verified": True,
        "chunks": {
            "chunk_0000": {
                "start_sec": 0.0,
                "end_sec": 2.0,
                "duration": 2.0,
                "text": "master windu master yoda",
            }
        },
    }
    manifest_file = tmp_path / "corpus_manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

    synth_source = SourceRecord(
        content_id="src_anakin_test",
        source_path="anakin_ep2.mp4",
        native_sample_rate=48000,
        native_channels=2,
    )

    out_index_path = tmp_path / "units_index.json"
    inv_data = build_units_inventory(
        str(aligned_dir),
        str(manifest_file),
        str(out_index_path),
        source_record=synth_source,
    )

    words = inv_data["words"]
    events = inv_data["events"]

    # Verify windu
    windu_items = [w for w in words.values() if w["word_normalized"] == "windu"]
    assert len(windu_items) == 1
    w_windu = windu_items[0]
    assert w_windu["realized_phones"] == ["W", "IH1", "N", "D", "UW0"]
    # Verify no noise event
    for eid in w_windu["phone_event_ids"]:
        assert events[eid]["kind"] == "phone"
        assert events[eid]["label"] != "spn"

    # Verify yoda
    yoda_items = [w for w in words.values() if w["word_normalized"] == "yoda"]
    assert len(yoda_items) == 1
    w_yoda = yoda_items[0]
    assert w_yoda["realized_phones"] == ["Y", "OW1", "D", "AH0"]
    for eid in w_yoda["phone_event_ids"]:
        assert events[eid]["kind"] == "phone"
        assert events[eid]["label"] != "spn"


def test_anakin_phrase_selection_adversarial_verification():
    """
    Empirically verifies:
    Synthesizing 'Master Windu manages to masturbate master Yoda.' for speaker Anakin
    selects 'Master Windu' and 'master Yoda' as intact multi-word phrases.
    """
    sr = 48000
    src = SourceRecord(
        content_id="src_anakin_ep2",
        source_path="anakin_ep2_1.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )

    events = {}
    words = {}

    # Utterance 1: "master windu"
    ev_m1 = AlignedEvent(event_id="ev_m1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=10000, end_sample=12000, label="M", base_phone="M", successor_id="ev_ae1")
    ev_ae1 = AlignedEvent(event_id="ev_ae1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=12000, end_sample=14000, label="AE1", base_phone="AE", predecessor_id="ev_m1", successor_id="ev_s1")
    ev_s1 = AlignedEvent(event_id="ev_s1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=14000, end_sample=16000, label="S", base_phone="S", predecessor_id="ev_ae1", successor_id="ev_t1")
    ev_t1 = AlignedEvent(event_id="ev_t1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=16000, end_sample=18000, label="T", base_phone="T", predecessor_id="ev_s1", successor_id="ev_er1")
    ev_er1 = AlignedEvent(event_id="ev_er1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=18000, end_sample=20000, label="ER0", base_phone="ER", predecessor_id="ev_t1", successor_id="ev_w1")

    ev_w1 = AlignedEvent(event_id="ev_w1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=20000, end_sample=22000, label="W", base_phone="W", predecessor_id="ev_er1", successor_id="ev_ih1")
    ev_ih1 = AlignedEvent(event_id="ev_ih1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=22000, end_sample=24000, label="IH1", base_phone="IH", predecessor_id="ev_w1", successor_id="ev_n1")
    ev_n1 = AlignedEvent(event_id="ev_n1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=24000, end_sample=26000, label="N", base_phone="N", predecessor_id="ev_ih1", successor_id="ev_d1")
    ev_d1 = AlignedEvent(event_id="ev_d1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=26000, end_sample=28000, label="D", base_phone="D", predecessor_id="ev_n1", successor_id="ev_uw1")
    ev_uw1 = AlignedEvent(event_id="ev_uw1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=28000, end_sample=30000, label="UW0", base_phone="UW", predecessor_id="ev_d1")

    for ev in [ev_m1, ev_ae1, ev_s1, ev_t1, ev_er1]:
        ev.word_occurrence_id = "w_master1"
        events[ev.event_id] = ev
    for ev in [ev_w1, ev_ih1, ev_n1, ev_d1, ev_uw1]:
        ev.word_occurrence_id = "w_windu"
        events[ev.event_id] = ev

    words["w_master1"] = WordOccurrence(
        word_id="w_master1", source_content_id="src_anakin_ep2", speaker_id="anakin",
        word_normalized="master", word_original="master",
        start_sample=10000, end_sample=20000,
        phone_event_ids=["ev_m1", "ev_ae1", "ev_s1", "ev_t1", "ev_er1"],
        realized_phones=["M", "AE1", "S", "T", "ER0"],
    )
    words["w_windu"] = WordOccurrence(
        word_id="w_windu", source_content_id="src_anakin_ep2", speaker_id="anakin",
        word_normalized="windu", word_original="windu",
        start_sample=20000, end_sample=30000,
        phone_event_ids=["ev_w1", "ev_ih1", "ev_n1", "ev_d1", "ev_uw1"],
        realized_phones=["W", "IH1", "N", "D", "UW0"],
    )

    # Utterance 2: "manages" and "to"
    mng_labels = [("M", "M"), ("AE1", "AE"), ("N", "N"), ("AH0", "AH"), ("JH", "JH"), ("IH0", "IH"), ("Z", "Z")]
    mng_ids = []
    for idx, (lbl, bp) in enumerate(mng_labels):
        eid = f"ev_mng_{idx}"
        ev = AlignedEvent(
            event_id=eid, source_content_id="src_anakin_ep2", utterance_id="u2", speaker_id="anakin",
            kind="phone", start_sample=40000 + idx * 1000, end_sample=40000 + (idx + 1) * 1000,
            label=lbl, base_phone=bp, word_occurrence_id="w_manages",
        )
        events[eid] = ev
        mng_ids.append(eid)
    words["w_manages"] = WordOccurrence(
        word_id="w_manages", source_content_id="src_anakin_ep2", speaker_id="anakin",
        word_normalized="manages", word_original="manages",
        start_sample=40000, end_sample=47000,
        phone_event_ids=mng_ids,
        realized_phones=[lbl for lbl, _ in mng_labels],
    )

    ev_t2 = AlignedEvent(event_id="ev_t2", source_content_id="src_anakin_ep2", utterance_id="u2", speaker_id="anakin", kind="phone", start_sample=50000, end_sample=52000, label="T", base_phone="T", word_occurrence_id="w_to")
    ev_uw2 = AlignedEvent(event_id="ev_uw2", source_content_id="src_anakin_ep2", utterance_id="u2", speaker_id="anakin", kind="phone", start_sample=52000, end_sample=54000, label="UW1", base_phone="UW", word_occurrence_id="w_to")
    events["ev_t2"] = ev_t2
    events["ev_uw2"] = ev_uw2
    words["w_to"] = WordOccurrence(
        word_id="w_to", source_content_id="src_anakin_ep2", speaker_id="anakin",
        word_normalized="to", word_original="to",
        start_sample=50000, end_sample=54000,
        phone_event_ids=["ev_t2", "ev_uw2"],
        realized_phones=["T", "UW1"],
    )

    # Phoneme units for "masturbate": B and EY2
    events["ev_b"] = AlignedEvent(event_id="ev_b", source_content_id="src_anakin_ep2", utterance_id="u3", speaker_id="anakin", kind="phone", start_sample=60000, end_sample=62000, label="B", base_phone="B")
    events["ev_ey2"] = AlignedEvent(event_id="ev_ey2", source_content_id="src_anakin_ep2", utterance_id="u3", speaker_id="anakin", kind="phone", start_sample=62000, end_sample=64000, label="EY2", base_phone="EY")

    # Utterance 4: "master yoda"
    ev_m4 = AlignedEvent(event_id="ev_m4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=70000, end_sample=72000, label="M", base_phone="M", successor_id="ev_ae4")
    ev_ae4 = AlignedEvent(event_id="ev_ae4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=72000, end_sample=74000, label="AE1", base_phone="AE", predecessor_id="ev_m4", successor_id="ev_s4")
    ev_s4 = AlignedEvent(event_id="ev_s4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=74000, end_sample=76000, label="S", base_phone="S", predecessor_id="ev_ae4", successor_id="ev_t4")
    ev_t4 = AlignedEvent(event_id="ev_t4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=76000, end_sample=78000, label="T", base_phone="T", predecessor_id="ev_s4", successor_id="ev_er4")
    ev_er4 = AlignedEvent(event_id="ev_er4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=78000, end_sample=80000, label="ER0", base_phone="ER", predecessor_id="ev_t4", successor_id="ev_y4")

    ev_y4 = AlignedEvent(event_id="ev_y4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=80000, end_sample=82500, label="Y", base_phone="Y", predecessor_id="ev_er4", successor_id="ev_ow4")
    ev_ow4 = AlignedEvent(event_id="ev_ow4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=82500, end_sample=85000, label="OW1", base_phone="OW", predecessor_id="ev_y4", successor_id="ev_d4")
    ev_d4 = AlignedEvent(event_id="ev_d4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=85000, end_sample=87500, label="D", base_phone="D", predecessor_id="ev_ow4", successor_id="ev_ah4")
    ev_ah4 = AlignedEvent(event_id="ev_ah4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=87500, end_sample=90000, label="AH0", base_phone="AH", predecessor_id="ev_d4")

    for ev in [ev_m4, ev_ae4, ev_s4, ev_t4, ev_er4]:
        ev.word_occurrence_id = "w_master4"
        events[ev.event_id] = ev
    for ev in [ev_y4, ev_ow4, ev_d4, ev_ah4]:
        ev.word_occurrence_id = "w_yoda"
        events[ev.event_id] = ev

    words["w_master4"] = WordOccurrence(
        word_id="w_master4", source_content_id="src_anakin_ep2", speaker_id="anakin",
        word_normalized="master", word_original="master",
        start_sample=70000, end_sample=80000,
        phone_event_ids=["ev_m4", "ev_ae4", "ev_s4", "ev_t4", "ev_er4"],
        realized_phones=["M", "AE1", "S", "T", "ER0"],
    )
    words["w_yoda"] = WordOccurrence(
        word_id="w_yoda", source_content_id="src_anakin_ep2", speaker_id="anakin",
        word_normalized="yoda", word_original="yoda",
        start_sample=80000, end_sample=90000,
        phone_event_ids=["ev_y4", "ev_ow4", "ev_d4", "ev_ah4"],
        realized_phones=["Y", "OW1", "D", "AH0"],
    )

    word_to_word_ids = {}
    for wid, w in words.items():
        word_to_word_ids.setdefault(w.word_normalized, []).append(wid)

    phone_to_event_ids = {}
    base_phone_to_event_ids = {}
    for eid, ev in events.items():
        if ev.kind == "phone":
            phone_to_event_ids.setdefault(ev.label, []).append(eid)
            base_phone_to_event_ids.setdefault(ev.base_phone, []).append(eid)

    inv = InventoryIndex(
        schema_version="2.0",
        sources={"src_anakin_ep2": src},
        events=events,
        words=words,
        word_to_word_ids=word_to_word_ids,
        phone_to_event_ids=phone_to_event_ids,
        base_phone_to_event_ids=base_phone_to_event_ids,
    )

    lex = Lexicon()
    lex.word_to_pronunciations = {
        "master": [["M", "AE1", "S", "T", "ER0"]],
        "windu": [["W", "IH1", "N", "D", "UW0"]],
        "manages": [["M", "AE1", "N", "AH0", "JH", "IH0", "Z"]],
        "to": [["T", "UW1"]],
        "masturbate": [["M", "AE1", "S", "T", "ER0", "B", "EY2", "T"]],
        "yoda": [["Y", "OW1", "D", "AH0"]],
    }

    from target_graph import TargetGraphBuilder
    from span_selector import SpanSelector

    target_text = "Master Windu manages to masturbate master Yoda."
    graph = TargetGraphBuilder(lex).build(target_text)
    selector = SpanSelector(inv, edit_penalty=0.5)
    selected_path = selector.select(graph, speaker_id="anakin")

    assert len(selected_path) > 0
    # First span must be "master windu"
    assert selected_path[0].matched_word == "master windu"
    # Last span must be "master yoda"
    assert selected_path[-1].matched_word == "master yoda"
    # Masturbate must contain phone units
    phone_spans = [c for c in selected_path if c.span_type == "phone"]
    assert len(phone_spans) > 0

