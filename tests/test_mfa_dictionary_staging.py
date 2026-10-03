"""
Tests for Milestone 1: Dynamic MFA Custom Dictionary Staging, Content Hashing,
Outer Alignment Cache Invalidation, Proper Noun Indexing, and Anakin Phrase Selection.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Dict, List

import pytest

from cache import (
    compute_alignment_cache_key,
    compute_labeled_inventory_cache_key,
    hash_file_content,
)
from contracts import (
    AlignedEvent,
    InventoryIndex,
    SourceRecord,
    WordOccurrence,
)
from inventory_builder import build_units_inventory
from lexicon import Lexicon, stage_mfa_dictionary
from span_selector import SpanSelector
from target_graph import TargetGraphBuilder


def test_stage_mfa_dictionary_combines_base_and_custom(tmp_path: Path):
    """
    Verifies that stage_mfa_dictionary correctly merges base and custom dictionaries,
    formats as tab-separated ARPAbet (WORD\\tP H O N E S), generates case variants,
    and returns a valid SHA-256 hash.
    """
    base_dict = tmp_path / "base.dict"
    base_dict.write_text(
        "master\t0.99\t0.37\t0.6\t1.05\tM AE1 S T ER0\n"
        "manages\tM AE1 N AH0 JH IH0 Z\n"
        "to\tT UW1\n",
        encoding="utf-8",
    )

    custom_dict = tmp_path / "custom.txt"
    custom_dict.write_text(
        "# Custom pronunciations\n"
        "windu\tW IH1 N D UW0\n"
        "yoda\tY OW1 D AH0\n"
        "masturbate   M AE1 S T ER0 B EY2 T\n",  # multiple spaces to test normalization
        encoding="utf-8",
    )

    staged_dict = tmp_path / "staged.dict"
    out_path, dict_hash = stage_mfa_dictionary(
        base_dict_path=base_dict,
        custom_dict_path=custom_dict,
        output_path=staged_dict,
    )

    assert out_path.exists()
    assert out_path == staged_dict

    # Check content hash
    expected_hash = hashlib.sha256(staged_dict.read_bytes()).hexdigest()
    assert dict_hash == expected_hash

    content = staged_dict.read_text(encoding="utf-8").splitlines()
    entries_map: Dict[str, List[str]] = {}
    for line in content:
        assert "\t" in line, f"Line missing tab delimiter: {line}"
        word, rest = line.split("\t", 1)
        entries_map.setdefault(word, []).append(rest)

    # Check that custom words have both lowercase and uppercase variants
    assert "windu" in entries_map
    assert "WINDU" in entries_map
    assert entries_map["windu"] == ["W IH1 N D UW0"]
    assert entries_map["WINDU"] == ["W IH1 N D UW0"]

    assert "yoda" in entries_map
    assert "YODA" in entries_map
    assert entries_map["yoda"] == ["Y OW1 D AH0"]
    assert entries_map["YODA"] == ["Y OW1 D AH0"]

    assert "masturbate" in entries_map
    assert "MASTURBATE" in entries_map
    assert entries_map["masturbate"] == ["M AE1 S T ER0 B EY2 T"]

    # Check probability preservation for base Kaldi entry
    assert "master" in entries_map
    assert any("0.99" in r and "M AE1 S T ER0" in r for r in entries_map["master"])


def test_stage_mfa_dictionary_deduplication(tmp_path: Path):
    """
    Verifies that identical entries within or across files are deduplicated,
    and variant markers like 'word(2)' are normalized.
    """
    base_dict = tmp_path / "base.dict"
    base_dict.write_text(
        "anakin\tAE1 N AH0 K IH0 N\n"
        "anakin\tAE1 N AH0 K IH0 N\n",  # duplicate in base
        encoding="utf-8",
    )

    custom_dict = tmp_path / "custom.txt"
    custom_dict.write_text(
        "anakin\tAE1 N AH0 K IH0 N\n"   # duplicate across files
        "anakin(2)\tAE1 N AH0 K IH0 N\n"  # variant marker
        "kenobi\tK AH0 N OW1 B IY0\n",
        encoding="utf-8",
    )

    staged_dict = tmp_path / "staged.dict"
    out_path, _ = stage_mfa_dictionary(
        base_dict_path=base_dict,
        custom_dict_path=custom_dict,
        output_path=staged_dict,
    )

    lines = out_path.read_text(encoding="utf-8").splitlines()
    anakin_lower_lines = [ln for ln in lines if ln == "anakin\tAE1 N AH0 K IH0 N"]
    anakin_upper_lines = [ln for ln in lines if ln == "ANAKIN\tAE1 N AH0 K IH0 N"]

    assert len(anakin_lower_lines) == 1, f"Expected 1 deduplicated entry for 'anakin', found {len(anakin_lower_lines)}"
    assert len(anakin_upper_lines) == 1, f"Expected 1 deduplicated entry for 'ANAKIN', found {len(anakin_upper_lines)}"
    assert "kenobi\tK AH0 N OW1 B IY0" in lines
    assert "KENOBI\tK AH0 N OW1 B IY0" in lines


def test_custom_pronunciation_invalidates_alignment_cache(tmp_path: Path):
    """
    Verifies that modifying custom_pronunciations.txt changes the staged dictionary hash,
    which automatically changes compute_alignment_cache_key and compute_labeled_inventory_cache_key.
    """
    base_dict = tmp_path / "base.dict"
    base_dict.write_text("hello\tHH AH0 L OW1\n", encoding="utf-8")

    custom_dict = tmp_path / "custom.txt"
    custom_dict.write_text("windu\tW IH1 N D UW0\n", encoding="utf-8")

    staged_dict = tmp_path / "staged.dict"

    # Stage 1: Initial custom pronunciations
    _, hash_1 = stage_mfa_dictionary(base_dict, custom_dict, staged_dict)

    align_key_1 = compute_alignment_cache_key(
        media_hash="media_abc_123",
        selected_audio_stream=0,
        selected_video_stream=0,
        transcript_hash="transcript_xyz",
        dict_hash=hash_1,
    )
    inv_key_1 = compute_labeled_inventory_cache_key(
        alignment_key=align_key_1,
        speaker_id="anakin",
    )

    # Stage 2: Add yoda to custom dictionary
    with open(custom_dict, "a", encoding="utf-8") as f:
        f.write("yoda\tY OW1 D AH0\n")

    # Clear staging cache to test invalidation
    _, hash_2 = stage_mfa_dictionary(base_dict, custom_dict, staged_dict)

    align_key_2 = compute_alignment_cache_key(
        media_hash="media_abc_123",
        selected_audio_stream=0,
        selected_video_stream=0,
        transcript_hash="transcript_xyz",
        dict_hash=hash_2,
    )
    inv_key_2 = compute_labeled_inventory_cache_key(
        alignment_key=align_key_2,
        speaker_id="anakin",
    )

    assert hash_1 != hash_2, "Dictionary hash must change when custom pronunciations are added"
    assert align_key_1 != align_key_2, "Alignment cache key must invalidate when custom pronunciations change"
    assert inv_key_1 != inv_key_2, "Inventory cache key must invalidate when alignment key changes"


def test_production_repo_dictionaries_contain_windu_and_yoda():
    """
    Verifies that staging the real production repo dictionaries combines
    merged_dictionary.dict and custom_pronunciations.txt and contains windu and yoda.
    """
    out_path, dict_hash = stage_mfa_dictionary()
    assert out_path.exists()
    assert len(dict_hash) == 64

    content = out_path.read_text(encoding="utf-8")
    assert "windu\tW IH1 N D UW0" in content
    assert "WINDU\tW IH1 N D UW0" in content
    assert "yoda\tY OW1 D AH0" in content
    assert "YODA\tY OW1 D AH0" in content
    assert "masturbate\tM AE1 S T ER0 B EY2 T" in content
    assert "MASTURBATE\tM AE1 S T ER0 B EY2 T" in content


def test_proper_nouns_indexing_in_units_inventory(tmp_path: Path):
    """
    Verifies that when TextGrids contain windu and yoda (recognized via staged dictionary),
    inventory_builder indexes them as valid WordOccurrence instances with realized_phones
    rather than discarding them as 'spn' noise.
    """
    aligned_dir = tmp_path / "aligned"
    aligned_dir.mkdir()

    # Synthetic TextGrid with recognized phones for "master windu" and "master yoda"
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
        "source_video": "anakin_source.mp4",
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
        source_path="anakin_source.mp4",
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

    assert "words" in inv_data
    words = inv_data["words"]

    # Verify windu was indexed as a whole word occurrence
    windu_wos = [w for w in words.values() if w["word_normalized"] == "windu"]
    assert len(windu_wos) == 1, "Expected 'windu' to be indexed as a whole word occurrence"
    assert windu_wos[0]["realized_phones"] == ["W", "IH1", "N", "D", "UW0"]
    assert windu_wos[0]["speaker_id"] == "anakin"

    # Verify yoda was indexed as a whole word occurrence
    yoda_wos = [w for w in words.values() if w["word_normalized"] == "yoda"]
    assert len(yoda_wos) == 1, "Expected 'yoda' to be indexed as a whole word occurrence"
    assert yoda_wos[0]["realized_phones"] == ["Y", "OW1", "D", "AH0"]
    assert yoda_wos[0]["speaker_id"] == "anakin"


def test_unrecognized_spn_words_are_dropped():
    """
    Contrasts with test_proper_nouns_indexing_in_units_inventory:
    verifies that when an OOV word has only 'spn' phone intervals,
    inventory_builder discards it (confirming the problem that dictionary staging solves).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        aligned_dir = tmp_path / "aligned"
        aligned_dir.mkdir()

        # TextGrid where "windu" has only an 'spn' phone interval
        tg_content = """File type = "ooTextFile"
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
            xmin = 0.1
            xmax = 0.9
            text = "windu"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.0
        intervals: size = 1
        intervals [1]:
            xmin = 0.1
            xmax = 0.9
            text = "spn"
"""
        tg_file = aligned_dir / "chunk_0000.TextGrid"
        tg_file.write_text(tg_content, encoding="utf-8")

        manifest = {
            "source_video": "dummy.mp4",
            "speaker_id": "anakin",
            "single_speaker_verified": True,
            "chunks": {
                "chunk_0000": {
                    "start_sec": 0.0,
                    "end_sec": 1.0,
                    "duration": 1.0,
                    "text": "windu",
                }
            },
        }
        manifest_file = tmp_path / "corpus_manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        synth_source = SourceRecord(
            content_id="src_dummy",
            source_path="dummy.mp4",
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

        # windu with spn constituent should be discarded
        windu_wos = [w for w in inv_data["words"].values() if w["word_normalized"] == "windu"]
        assert len(windu_wos) == 0, "OOV word with 'spn' phone should be discarded"


def test_anakin_phrase_selection_master_windu_and_master_yoda():
    """
    Verifies Requirement 4:
    Synthesizing 'Master Windu manages to masturbate master Yoda.' for speaker Anakin
    selects 'Master Windu' and 'master Yoda' as intact words or phrases.
    """
    sr = 48000
    src = SourceRecord(
        content_id="src_anakin_ep2",
        source_path="anakin_ep2_1.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )

    events: Dict[str, AlignedEvent] = {}
    words: Dict[str, WordOccurrence] = {}

    # Utterance 1: "master windu" (consecutive words spoken by anakin)
    # master: 10000..20000 (M, AE1, S, T, ER0)
    ev_m1 = AlignedEvent(event_id="ev_m1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=10000, end_sample=12000, label="M", base_phone="M", successor_id="ev_ae1")
    ev_ae1 = AlignedEvent(event_id="ev_ae1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=12000, end_sample=14000, label="AE1", base_phone="AE", predecessor_id="ev_m1", successor_id="ev_s1")
    ev_s1 = AlignedEvent(event_id="ev_s1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=14000, end_sample=16000, label="S", base_phone="S", predecessor_id="ev_ae1", successor_id="ev_t1")
    ev_t1 = AlignedEvent(event_id="ev_t1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=16000, end_sample=18000, label="T", base_phone="T", predecessor_id="ev_s1", successor_id="ev_er1")
    ev_er1 = AlignedEvent(event_id="ev_er1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=18000, end_sample=20000, label="ER0", base_phone="ER", predecessor_id="ev_t1", successor_id="ev_w1")

    # windu: 20000..30000 (W, IH1, N, D, UW0)
    ev_w1 = AlignedEvent(event_id="ev_w1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=20000, end_sample=22000, label="W", base_phone="W", predecessor_id="ev_er1", successor_id="ev_ih1")
    ev_ih1 = AlignedEvent(event_id="ev_ih1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=22000, end_sample=24000, label="IH1", base_phone="IH", predecessor_id="ev_w1", successor_id="ev_n1")
    ev_n1 = AlignedEvent(event_id="ev_n1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=24000, end_sample=26000, label="N", base_phone="N", predecessor_id="ev_ih1", successor_id="ev_d1")
    ev_d1 = AlignedEvent(event_id="ev_d1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=26000, end_sample=28000, label="D", base_phone="D", predecessor_id="ev_n1", successor_id="ev_uw1")
    ev_uw1 = AlignedEvent(event_id="ev_uw1", source_content_id="src_anakin_ep2", utterance_id="u1", speaker_id="anakin", kind="phone", start_sample=28000, end_sample=30000, label="UW0", base_phone="UW", predecessor_id="ev_d1")

    w_master1 = WordOccurrence(
        word_id="w_master1", source_content_id="src_anakin_ep2", speaker_id="anakin",
        word_normalized="master", word_original="master",
        start_sample=10000, end_sample=20000,
        phone_event_ids=["ev_m1", "ev_ae1", "ev_s1", "ev_t1", "ev_er1"],
        realized_phones=["M", "AE1", "S", "T", "ER0"],
    )
    w_windu = WordOccurrence(
        word_id="w_windu", source_content_id="src_anakin_ep2", speaker_id="anakin",
        word_normalized="windu", word_original="windu",
        start_sample=20000, end_sample=30000,
        phone_event_ids=["ev_w1", "ev_ih1", "ev_n1", "ev_d1", "ev_uw1"],
        realized_phones=["W", "IH1", "N", "D", "UW0"],
    )

    for ev in [ev_m1, ev_ae1, ev_s1, ev_t1, ev_er1]:
        ev.word_occurrence_id = "w_master1"
        events[ev.event_id] = ev
    for ev in [ev_w1, ev_ih1, ev_n1, ev_d1, ev_uw1]:
        ev.word_occurrence_id = "w_windu"
        events[ev.event_id] = ev
    words["w_master1"] = w_master1
    words["w_windu"] = w_windu

    # Utterance 2: "manages" and "to"
    # manages: 40000..48000 (M, AE1, N, AH0, JH, IH0, Z)
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

    # to: 50000..54000 (T, UW1)
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

    # Isolated phone units for "masturbate": M, AE1, S, T, ER0, B, EY2, T
    # Note: B and EY2 provided here as donor units
    ev_b = AlignedEvent(event_id="ev_b", source_content_id="src_anakin_ep2", utterance_id="u3", speaker_id="anakin", kind="phone", start_sample=60000, end_sample=62000, label="B", base_phone="B")
    ev_ey2 = AlignedEvent(event_id="ev_ey2", source_content_id="src_anakin_ep2", utterance_id="u3", speaker_id="anakin", kind="phone", start_sample=62000, end_sample=64000, label="EY2", base_phone="EY")
    events["ev_b"] = ev_b
    events["ev_ey2"] = ev_ey2

    # Utterance 4: "master yoda" (consecutive words spoken by anakin)
    # master: 70000..80000
    ev_m4 = AlignedEvent(event_id="ev_m4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=70000, end_sample=72000, label="M", base_phone="M", successor_id="ev_ae4")
    ev_ae4 = AlignedEvent(event_id="ev_ae4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=72000, end_sample=74000, label="AE1", base_phone="AE", predecessor_id="ev_m4", successor_id="ev_s4")
    ev_s4 = AlignedEvent(event_id="ev_s4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=74000, end_sample=76000, label="S", base_phone="S", predecessor_id="ev_ae4", successor_id="ev_t4")
    ev_t4 = AlignedEvent(event_id="ev_t4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=76000, end_sample=78000, label="T", base_phone="T", predecessor_id="ev_s4", successor_id="ev_er4")
    ev_er4 = AlignedEvent(event_id="ev_er4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=78000, end_sample=80000, label="ER0", base_phone="ER", predecessor_id="ev_t4", successor_id="ev_y4")

    # yoda: 80000..90000 (Y, OW1, D, AH0)
    ev_y4 = AlignedEvent(event_id="ev_y4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=80000, end_sample=82500, label="Y", base_phone="Y", predecessor_id="ev_er4", successor_id="ev_ow4")
    ev_ow4 = AlignedEvent(event_id="ev_ow4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=82500, end_sample=85000, label="OW1", base_phone="OW", predecessor_id="ev_y4", successor_id="ev_d4")
    ev_d4 = AlignedEvent(event_id="ev_d4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=85000, end_sample=87500, label="D", base_phone="D", predecessor_id="ev_ow4", successor_id="ev_ah4")
    ev_ah4 = AlignedEvent(event_id="ev_ah4", source_content_id="src_anakin_ep2", utterance_id="u4", speaker_id="anakin", kind="phone", start_sample=87500, end_sample=90000, label="AH0", base_phone="AH", predecessor_id="ev_d4")

    w_master4 = WordOccurrence(
        word_id="w_master4", source_content_id="src_anakin_ep2", speaker_id="anakin",
        word_normalized="master", word_original="master",
        start_sample=70000, end_sample=80000,
        phone_event_ids=["ev_m4", "ev_ae4", "ev_s4", "ev_t4", "ev_er4"],
        realized_phones=["M", "AE1", "S", "T", "ER0"],
    )
    w_yoda = WordOccurrence(
        word_id="w_yoda", source_content_id="src_anakin_ep2", speaker_id="anakin",
        word_normalized="yoda", word_original="yoda",
        start_sample=80000, end_sample=90000,
        phone_event_ids=["ev_y4", "ev_ow4", "ev_d4", "ev_ah4"],
        realized_phones=["Y", "OW1", "D", "AH0"],
    )

    for ev in [ev_m4, ev_ae4, ev_s4, ev_t4, ev_er4]:
        ev.word_occurrence_id = "w_master4"
        events[ev.event_id] = ev
    for ev in [ev_y4, ev_ow4, ev_d4, ev_ah4]:
        ev.word_occurrence_id = "w_yoda"
        events[ev.event_id] = ev
    words["w_master4"] = w_master4
    words["w_yoda"] = w_yoda

    word_to_word_ids: Dict[str, List[str]] = {}
    for wid, w in words.items():
        word_to_word_ids.setdefault(w.word_normalized, []).append(wid)

    phone_to_event_ids: Dict[str, List[str]] = {}
    base_phone_to_event_ids: Dict[str, List[str]] = {}
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

    target_text = "Master Windu manages to masturbate master Yoda."
    builder = TargetGraphBuilder(lex)
    graph = builder.build(target_text)

    selector = SpanSelector(inv, edit_penalty=0.5)
    selected_path = selector.select(graph, speaker_id="anakin")

    assert len(selected_path) > 0

    # Verify "master windu" is selected as an intact multi-word phrase
    first_span = selected_path[0]
    assert first_span.span_type in ("phrase", "word")
    if first_span.span_type == "phrase":
        assert first_span.matched_word == "master windu"
    else:
        assert first_span.matched_word == "master"
        assert selected_path[1].matched_word == "windu"

    # Verify "master yoda" is selected as an intact multi-word phrase
    last_span = selected_path[-1]
    assert last_span.span_type in ("phrase", "word")
    if last_span.span_type == "phrase":
        assert last_span.matched_word == "master yoda"
    else:
        assert last_span.matched_word == "yoda"
        assert selected_path[-2].matched_word == "master"

    # Verify "masturbate" was synthesized from phones
    phone_spans = [c for c in selected_path if c.span_type == "phone"]
    assert len(phone_spans) > 0, "Expected 'masturbate' to be synthesized from phone splices"
    phone_labels = [events[eid].label for c in phone_spans for eid in c.event_ids]
    assert "B" in phone_labels
    assert "EY2" in phone_labels


def test_ensure_source_aligned_stages_mfa_dictionary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verifies that ensure_source_aligned calls stage_mfa_dictionary,
    passes the staged dictionary path to run_mfa_align, and incorporates
    its hash into alignment_metadata.json.
    """
    import run_mixer

    cache_dir = tmp_path / ".build"
    monkeypatch.setattr(run_mixer, "CACHE_BASE_DIR", cache_dir)

    media_file = tmp_path / "video.mp4"
    media_file.write_bytes(b"VIDEO_TEST_DATA" * 100)

    vtt_file = tmp_path / "video.vtt"
    vtt_file.write_text(
        "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nmaster windu\n",
        encoding="utf-8",
    )

    passed_dict_path = []

    def mock_mfa_align(corpus_dir, aligned_dir, dict_path, acoustic_model="english_us_arpa"):
        passed_dict_path.append(dict_path)
        out = Path(aligned_dir)
        out.mkdir(parents=True, exist_ok=True)
        (out / "chunk_0000.TextGrid").write_text("fake textgrid", encoding="utf-8")

    def mock_preprocess(v_abs, vtt, out_dir, **kwargs):
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "corpus_manifest.json").write_text(json.dumps({"chunks": {"chunk_0000": {}}}), encoding="utf-8")

    def mock_inventory_builder(aligned_dir, manifest_path, output_index_path, **kwargs):
        p = Path(output_index_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({"schema_version": "2.0.0", "words": {}, "events": {}, "sources": {}}), encoding="utf-8")

    monkeypatch.setattr(run_mixer, "run_mfa_align", mock_mfa_align)
    monkeypatch.setattr(run_mixer, "preprocess_source", mock_preprocess)
    monkeypatch.setattr(run_mixer, "build_units_inventory", mock_inventory_builder)

    inv_path = run_mixer.ensure_source_aligned(str(media_file), str(vtt_file))
    assert Path(inv_path).exists()

    # Verify run_mfa_align received the staged dictionary path
    assert len(passed_dict_path) == 1
    staged_p = Path(passed_dict_path[0])
    assert staged_p.name == "staged_dictionary.dict"
    assert staged_p.exists()

    # Verify alignment metadata contains dict_hash
    source_id = run_mixer.get_source_id(str(media_file.resolve()))
    align_meta_p = cache_dir / "alignment" / source_id / "alignment_metadata.json"
    assert align_meta_p.exists()
    meta = json.loads(align_meta_p.read_text(encoding="utf-8"))
    assert "dict_hash" in meta
    assert len(meta["dict_hash"]) == 64
    assert meta["dict_hash"] == hash_file_content(staged_p)

