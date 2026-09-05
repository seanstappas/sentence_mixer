#!/usr/bin/env python3
"""
Inventory Builder for Sentence Mixing
Parses Praat .TextGrid alignment files and source video manifests into a searchable
unit database (words and phonemes) with global video timestamps.
"""

import argparse
import glob
import json
import os
import re
import sys
from typing import Dict, List, Any, Optional

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def parse_textgrid_pure_python(tg_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pure Python parser for Praat .TextGrid files (both short and long text formats).
    Returns dict mapping tier name -> list of intervals [{'xmin': float, 'xmax': float, 'text': str}].
    """
    tiers = {}
    with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]

    current_tier = None
    i = 0
    while i < len(lines):
        line = lines[i]

        # Match Long Format tier header: name = "words" or class = "IntervalTier"
        name_match = re.match(r'name\s*=\s*"([^"]+)"', line)
        if name_match:
            current_tier = name_match.group(1).lower()
            if current_tier not in tiers:
                tiers[current_tier] = []
            i += 1
            continue

        # Match Long Format intervals: xmin = ..., xmax = ..., text = "..."
        if current_tier:
            xmin_m = re.match(r'xmin\s*=\s*([0-9.]+)', line)
            if xmin_m:
                xmin = float(xmin_m.group(1))
                xmax = None
                text = ""
                # Scan next lines for xmax and text
                j = i + 1
                while j < min(i + 5, len(lines)):
                    xmax_m = re.match(r'xmax\s*=\s*([0-9.]+)', lines[j])
                    if xmax_m:
                        xmax = float(xmax_m.group(1))
                    text_m = re.match(r'text\s*=\s*"([^"]*)"', lines[j])
                    if text_m:
                        text = text_m.group(1)
                        i = j
                        break
                    j += 1
                if xmax is not None:
                    tiers[current_tier].append({"xmin": xmin, "xmax": xmax, "text": text})

        i += 1

    return tiers


def build_units_inventory(
    aligned_dir: str,
    manifest_path: str,
    output_index_path: str
) -> Dict[str, Any]:
    """
    Builds a JSON unit inventory of all words and phonemes from aligned TextGrids.
    """
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    source_video = manifest["source_video"]
    chunks_meta = manifest["chunks"]

    words_inventory: Dict[str, List[Dict[str, Any]]] = {}
    phones_inventory: Dict[str, List[Dict[str, Any]]] = {}

    tg_files = glob.glob(os.path.join(aligned_dir, "*.TextGrid"))
    print(f"[*] Found {len(tg_files)} .TextGrid files in {aligned_dir}")

    total_words_indexed = 0
    total_phones_indexed = 0

    for tg_file in tg_files:
        chunk_id = os.path.splitext(os.path.basename(tg_file))[0]
        if chunk_id not in chunks_meta:
            continue

        chunk_start_sec = float(chunks_meta[chunk_id]["start_sec"])

        # Attempt to parse TextGrid using praatio if installed, else fallback to built-in parser
        parsed_tiers = None
        try:
            from praatio import textgrid
            tg = textgrid.openTextgrid(tg_file, False)
            parsed_tiers = {}
            for tname in tg.tierNames:
                entries = []
                for entry in tg.getTier(tname).entries:
                    entries.append({"xmin": entry.start, "xmax": entry.end, "text": entry.label})
                parsed_tiers[tname.lower()] = entries
        except Exception:
            parsed_tiers = parse_textgrid_pure_python(tg_file)

        # Process words tier
        words_tier = parsed_tiers.get("words", [])
        for entry in words_tier:
            raw_text = entry["text"].strip()
            clean_word = re.sub(r"[^\w\']", "", raw_text).lower()
            # Skip empty, silence, or noise tokens
            if not clean_word or clean_word in {"sil", "sp", "spn", "eps", "<eps>"}:
                continue

            local_start = float(entry["xmin"])
            local_end = float(entry["xmax"])
            duration = local_end - local_start
            if duration <= 0.01:
                continue

            global_start = chunk_start_sec + local_start
            global_end = chunk_start_sec + local_end

            item = {
                "source_video": source_video,
                "chunk_id": chunk_id,
                "label": clean_word,
                "start_sec": round(global_start, 4),
                "end_sec": round(global_end, 4),
                "duration": round(duration, 4),
                "type": "word"
            }

            if clean_word not in words_inventory:
                words_inventory[clean_word] = []
            words_inventory[clean_word].append(item)
            total_words_indexed += 1

        # Process phones tier
        phones_tier = parsed_tiers.get("phones", [])
        for entry in phones_tier:
            raw_phone = entry["text"].strip().upper()
            if not raw_phone or raw_phone in {"SIL", "SP", "SPN", "EPS", "<EPS>", ""}:
                continue

            # Strip stress numbers for base phone matching (e.g., IY1 -> IY)
            base_phone = re.sub(r"\d", "", raw_phone)
            if not base_phone:
                continue

            local_start = float(entry["xmin"])
            local_end = float(entry["xmax"])
            duration = local_end - local_start
            if duration <= 0.01:
                continue

            global_start = chunk_start_sec + local_start
            global_end = chunk_start_sec + local_end

            item = {
                "source_video": source_video,
                "chunk_id": chunk_id,
                "label": raw_phone,
                "base_phone": base_phone,
                "start_sec": round(global_start, 4),
                "end_sec": round(global_end, 4),
                "duration": round(duration, 4),
                "type": "phone"
            }

            # Index by both exact phone (with stress) and base phone
            for key in {raw_phone, base_phone}:
                if key not in phones_inventory:
                    phones_inventory[key] = []
                phones_inventory[key].append(item)

            total_phones_indexed += 1

    inventory_data = {
        "source_video": source_video,
        "total_unique_words": len(words_inventory),
        "total_words_instances": total_words_indexed,
        "total_unique_phones": len(phones_inventory),
        "total_phones_instances": total_phones_indexed,
        "words": words_inventory,
        "phones": phones_inventory
    }

    os.makedirs(os.path.dirname(os.path.abspath(output_index_path)), exist_ok=True)
    with open(output_index_path, "w", encoding="utf-8") as f:
        json.dump(inventory_data, f, indent=2)

    print(f"[+] Inventory build complete:")
    print(f"    - Unique words: {len(words_inventory)} (Total instances: {total_words_indexed})")
    print(f"    - Unique phonemes: {len(phones_inventory)} (Total instances: {total_phones_indexed})")
    print(f"    - Index saved to: {output_index_path}")
    return inventory_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build unit inventory from aligned TextGrids.")
    parser.add_argument("--aligned", required=True, help="Directory containing .TextGrid files")
    parser.add_argument("--manifest", required=True, help="Path to corpus_manifest.json")
    parser.add_argument("--output", default="./inventory/units_index.json", help="Output JSON index path")
    args = parser.parse_args()

    build_units_inventory(args.aligned, args.manifest, args.output)
