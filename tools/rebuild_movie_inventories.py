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

DEFAULT_MOVIES = [
    "Star_Wars_Episode_I_-_The_Phantom_Menace_aa16ff9a",
    "Star_Wars_Episode_II_-_Attack_Of_The_Clo_56214896",
    "Star_Wars_Episode_III_-_Revenge_Of_The_S_a39114e3",
    "Star_Wars_Episode_IV_-_A_New_Hope_1977_2_04865440",
    "Star_Wars_Episode_VI_-_Return_Of_The_Jed_3d083b52",
]

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
    blocks_dir = align_dir / "blocks"
    manifest_p = align_dir / "movie_manifest.json"
    inv_dir = inventory_base / source_id
    inv_path = inv_dir / "units_index.json"

    if not blocks_dir.exists() or not manifest_p.exists():
        return {"source_id": source_id, "status": "skipped_missing_blocks"}

    # Extract existing SourceRecord if available
    src_rec: Optional[SourceRecord] = None
    old_word_count = 0
    old_words_set = set()
    if inv_path.exists():
        try:
            with open(inv_path, "r", encoding="utf-8") as f:
                old_data = json.load(f)
            old_words = old_data.get("words_inventory", {})
            old_word_count = sum(len(v) for v in old_words.values())
            old_words_set = set(old_words.keys())
            sources = old_data.get("sources", {})
            if source_id in sources:
                allowed_fields = {f.name for f in dataclasses.fields(SourceRecord)}
                raw_src = {k: v for k, v in sources[source_id].items() if k in allowed_fields}
                src_rec = SourceRecord(**raw_src)
        except Exception as e:
            print(f"  [!] Warning: could not parse existing inventory for {source_id}: {e}")

    if dry_run:
        return {
            "source_id": source_id,
            "status": "dry_run",
            "old_words": old_word_count,
        }

    t0 = time.time()
    tmp_out = inv_dir / f"units_index.tmp_{sys.platform}_{time.time_ns()}.json"
    inv_dir.mkdir(parents=True, exist_ok=True)

    try:
        new_inv = build_units_inventory(
            aligned_dir=str(blocks_dir),
            manifest_path=str(manifest_p),
            output_index_path=str(tmp_out),
            source_record=src_rec,
            lexicon=lexicon,
        )
        # Atomic replace
        tmp_out.replace(inv_path)
    finally:
        if tmp_out.exists():
            try:
                tmp_out.unlink()
            except OSError:
                pass

    dt = time.time() - t0
    new_words = new_inv.get("words_inventory", {})
    new_word_count = sum(len(v) for v in new_words.values())
    new_words_set = set(new_words.keys())

    recovered_words = new_words_set - old_words_set
    tracked_found = {w: len(new_words[w]) for w in TRACKED_PROPER_NOUNS if w in new_words}

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
    parser = argparse.ArgumentParser(description="Rebuild movie unit inventories with OOV recovery.")
    parser.add_argument(
        "--movie",
        default="all",
        help="Source ID to rebuild or 'all' for all ingested movies",
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

    target_movies: List[str] = []
    if args.movie == "all":
        target_movies = list(DEFAULT_MOVIES)
    else:
        target_movies = [args.movie]

    print("\n" + "=" * 65)
    print("REBUILDING MOVIE INVENTORIES (SUBTITLE-ANCHORED OOV RECOVERY)")
    print("=" * 65)

    results = []
    total_added = 0
    t_start = time.time()

    for idx, mov in enumerate(target_movies, 1):
        print(f"\n[{idx}/{len(target_movies)}] Processing: {mov}")
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
            print(f"  [+] Tracked proper nouns recovered: {res['tracked_proper_nouns']}")
        else:
            print(f"  [-] Result: {res}")

    print("\n" + "=" * 65)
    print(f"REBUILD COMPLETE in {time.time() - t_start:.2f}s! Total words recovered across movies: +{total_added}")
    print("=" * 65)


if __name__ == "__main__":
    main()
