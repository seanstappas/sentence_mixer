#!/usr/bin/env python3
"""
Rebuilds unit inventories for ingested movie corpora using Subtitle-Anchored OOV Recovery.
Parses existing aligned TextGrids without re-running audio separation or MFA alignment,
restoring proper nouns and out-of-vocabulary words that MFA labeled as 'spn'.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from contracts import SourceRecord
from inventory_builder import build_units_inventory
from lexicon import Lexicon
from cache import load_inventory_json, resolve_inventory_path

DEFAULT_MOVIES = [
    "Star_Wars_Episode_I_-_The_Phantom_Menace_aa16ff9a",
    "Star_Wars_Episode_II_-_Attack_Of_The_Clo_56214896",
    "Star_Wars_Episode_III_-_Revenge_Of_The_S_a39114e3",
    "Star_Wars_Episode_IV_-_A_New_Hope_1977_2_04865440",
    "Star_Wars_Episode_VI_-_Return_Of_The_Jed_3d083b52",
]

CLIP_SOURCES = [
    "anakin_ep2_1_bce006b9",
    "anakin_ep2_2_5623dd18",
    "anakin_ep3_2_0ed12f5b",
    "crisis_9c96ee02",
    "obi_wan_ep2_a3766933",
    "obi_wan_ep3_a0bcd862",
    "test_source_0ac68f87",
]

ALL_SOURCES = DEFAULT_MOVIES + CLIP_SOURCES

TRACKED_PROPER_NOUNS = [
    "alderaan", "naboo", "tatooine", "dantooine", "mustafar", "utapau",
    "skywalker", "amidala", "padme", "leia", "organa", "chewie", "chewbacca",
    "jabba", "calrissian", "valorum", "sebulba", "plagueis", "sidious",
    "sifo", "padawan", "cloners", "droid", "droids", "lightsaber", "tydirium",
    "r2", "3po",
]


def rebuild_inventory_for_movie(
    source_id: str,
    alignment_base: Path,
    inventory_base: Path,
    lexicon: Lexicon,
    dry_run: bool = False,
) -> Dict[str, Any]:
    align_dir = alignment_base / source_id
    inv_dir = inventory_base / source_id
    inv_path = inv_dir / "units_index.json.gz"

    # Detect whether it is a movie (blocks/ + movie_manifest.json) or clip (aligned/ + corpus/corpus_manifest.json)
    if (align_dir / "blocks").exists() and (align_dir / "movie_manifest.json").exists():
        aligned_dir = align_dir / "blocks"
        manifest_p = align_dir / "movie_manifest.json"
    elif (align_dir / "aligned").exists() and (align_dir / "corpus" / "corpus_manifest.json").exists():
        aligned_dir = align_dir / "aligned"
        manifest_p = align_dir / "corpus" / "corpus_manifest.json"
    else:
        return {"source_id": source_id, "status": "skipped_missing_alignment_or_manifest"}

    # Extract existing SourceRecord and speaker metadata if available
    src_rec: Optional[SourceRecord] = None
    old_word_count = 0
    old_words_set = set()
    raw_inv_path = resolve_inventory_path(inv_dir)
    if raw_inv_path.exists():
        try:
            old_data = load_inventory_json(raw_inv_path)
            old_words = old_data.get("words", {})
            if isinstance(old_words, dict) and old_words and isinstance(next(iter(old_words.values())), dict):
                old_word_count = len(old_words)
                old_words_set = {v.get("word_normalized") for v in old_words.values() if isinstance(v, dict)}
            else:
                old_w_inv = old_data.get("words_inventory", {})
                old_word_count = sum(len(v) for v in old_w_inv.values())
                old_words_set = set(old_w_inv.keys())

            sources = old_data.get("sources", {})
            if source_id in sources:
                allowed_fields = {f.name for f in dataclasses.fields(SourceRecord)}
                raw_src = {k: v for k, v in sources[source_id].items() if k in allowed_fields}
                src_rec = SourceRecord(**raw_src)
        except Exception as e:
            print(f"  [!] Warning: could not parse existing inventory for {source_id}: {e}")

    # Read speaker metadata from existing inventory_metadata.json if present
    spk_id = None
    single_spk = False
    assume_single = False
    meta_p = inv_dir / "inventory_metadata.json"
    if meta_p.exists():
        try:
            with open(meta_p, "r", encoding="utf-8") as f:
                meta = json.load(f)
            spk_id = meta.get("speaker_id")
            single_spk = bool(meta.get("single_speaker_verified", False))
            assume_single = bool(meta.get("assume_single_speaker", False))
        except Exception:
            pass

    if dry_run:
        return {
            "source_id": source_id,
            "status": "dry_run",
            "old_words": old_word_count,
        }

    t0 = time.time()
    tmp_out = inv_dir / f"units_index.tmp_{sys.platform}_{time.time_ns()}.json.gz"
    inv_dir.mkdir(parents=True, exist_ok=True)

    try:
        new_inv = build_units_inventory(
            aligned_dir=str(aligned_dir),
            manifest_path=str(manifest_p),
            output_index_path=str(tmp_out),
            source_record=src_rec,
            lexicon=lexicon,
            recover_oov_words=True,
            speaker_id=spk_id,
            single_speaker_verified=single_spk,
            assume_single_speaker=assume_single,
        )
        # Atomic replace
        if inv_path.exists():
            inv_path.unlink()
        tmp_out.replace(inv_path)

        # Remove uncompressed units_index.json if present to keep clean
        raw_json = inv_dir / "units_index.json"
        if raw_json.exists():
            try:
                raw_json.unlink()
            except OSError:
                pass
    finally:
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except OSError:
                pass

    dt = time.time() - t0
    new_words = new_inv.get("words", {})
    if isinstance(new_words, dict) and new_words and isinstance(next(iter(new_words.values())), dict):
        new_word_count = len(new_words)
        new_words_set = {v.get("word_normalized") for v in new_words.values() if isinstance(v, dict)}
    else:
        new_w_inv = new_inv.get("words_inventory", {})
        new_word_count = sum(len(v) for v in new_w_inv.values())
        new_words_set = set(new_w_inv.keys())

    recovered_words = new_words_set - old_words_set
    tracked_found = {w: sum(1 for v in new_words.values() if isinstance(v, dict) and v.get("word_normalized") == w) for w in TRACKED_PROPER_NOUNS if w in new_words_set}

    return {
        "source_id": source_id,
        "status": "success",
        "duration_sec": round(dt, 2),
        "old_words": old_word_count,
        "new_words": new_word_count,
        "words_added": new_word_count - old_word_count,
        "unique_added": len(recovered_words),
        "tracked_proper_nouns": tracked_found,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild unit inventories with OOV recovery and storage optimization.")
    parser.add_argument(
        "--movie",
        default="all",
        help="Source ID to rebuild or 'all' for all 12 ingested sources",
    )
    parser.add_argument(
        "--movies-only",
        action="store_true",
        help="Rebuild only the 5 Star Wars movies, skipping the 7 clip sources",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Check status without writing new inventory files",
    )
    args = parser.parse_args()

    alignment_base = REPO_ROOT / ".build" / "alignment"
    inventory_base = REPO_ROOT / ".build" / "inventory"

    dict_files = [
        REPO_ROOT / "dictionaries" / "custom_pronunciations.txt",
        REPO_ROOT / "dictionaries" / "merged_dictionary.dict",
    ]
    lexicon = Lexicon([p for p in dict_files if p.exists()])

    target_sources: List[str] = []
    if args.movies_only:
        target_sources = list(DEFAULT_MOVIES)
    elif args.movie == "all":
        target_sources = list(ALL_SOURCES)
    else:
        target_sources = [args.movie]

    print("\n" + "=" * 65)
    print("REBUILDING INVENTORIES (SUBTITLE-ANCHORED OOV RECOVERY & COMPACT STORAGE)")
    print("=" * 65)

    results = []
    total_added = 0
    t_start = time.time()

    for idx, mov in enumerate(target_sources, 1):
        print(f"\n[{idx}/{len(target_sources)}] Processing: {mov}")
        res = rebuild_inventory_for_movie(
            source_id=mov,
            alignment_base=alignment_base,
            inventory_base=inventory_base,
            lexicon=lexicon,
            dry_run=args.dry_run,
        )
        results.append(res)
        if res.get("status") == "success":
            added = res["words_added"]
            total_added += added
            print(f"  [+] Completed in {res['duration_sec']}s")
            print(f"  [+] Words: {res['old_words']} -> {res['new_words']} (+{added} instances, +{res['unique_added']} unique)")
            if res['tracked_proper_nouns']:
                print(f"  [+] Tracked proper nouns recovered: {res['tracked_proper_nouns']}")
        else:
            print(f"  [-] Result: {res}")

    print("\n" + "=" * 65)
    print(f"REBUILD COMPLETE in {time.time() - t_start:.2f}s! Total words processed across sources: {sum(r.get('new_words', 0) for r in results)}")
    print("=" * 65)


if __name__ == "__main__":
    main()
