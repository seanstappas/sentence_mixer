#!/usr/bin/env python3
"""
YTP Sentence Mixing Pipeline CLI
Unified CLI tool to align video sources via MFA and synthesize target sentences.
Supports multiple source videos, in-memory corpus merging, and per-video alignment caching.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR / "src"
BIN_DIR = SCRIPT_DIR / "bin"
MICROMAMBA_EXE = BIN_DIR / "micromamba.exe"
MFA_ENV_PATH = Path("C:/Users/Sean/micromamba/envs/mfa_env")
CACHE_BASE_DIR = SCRIPT_DIR / "cache"

# Add src/ to sys.path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from vtt_preprocessor import preprocess_source
from inventory_builder import build_units_inventory
from unit_selector import synthesize_sentence
from fcp_xml_generator import generate_fcp_xml
from video_assembler import render_sentence_mix


def get_source_id(video_path: str) -> str:
    """Generate a clean filesystem-safe source ID with a hash for unique caching."""
    clean_stem = re.sub(r"[^\w\-_]", "_", Path(video_path).stem)[:40]
    path_hash = hashlib.sha256(Path(video_path).resolve().as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{clean_stem}_{path_hash}"


def find_associated_vtt(video_path: str) -> str:
    """Locate the corresponding .vtt file alongside the video if not explicitly provided."""
    v_path = Path(video_path)
    parent = v_path.parent
    stem = v_path.stem

    # Common subtitle naming schemes
    candidates = [
        parent / f"{stem}.en.vtt",
        parent / f"{stem}.vtt",
        parent / f"{stem}.en-US.vtt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    # Glob search for any matching .vtt
    matches = list(parent.glob(f"{glob_escape(stem)}*.vtt"))
    if matches:
        return str(matches[0])

    return ""


def glob_escape(s: str) -> str:
    """Escape special characters for globbing."""
    return re.sub(r"([*?\[\]])", r"[\1]", s)


def run_mfa_align(
    corpus_dir: str,
    output_aligned_dir: str,
    dictionary_path: str = None,
    acoustic_model: str = "english_us_arpa"
):
    """Executes Montreal Forced Aligner inside the mfa_env micromamba environment."""
    os.makedirs(output_aligned_dir, exist_ok=True)
    dict_arg = dictionary_path if dictionary_path and os.path.exists(dictionary_path) else "english_us_arpa"

    cmd = [
        str(MICROMAMBA_EXE),
        "run",
        "-r", "C:/Users/Sean/micromamba",
        "-n", "mfa_env",
        "mfa", "align",
        corpus_dir,
        dict_arg,
        acoustic_model,
        output_aligned_dir,
        "--clean",
        "--use_mp"
    ]

    print(f"[*] Running MFA Alignment command:")
    print("    " + " ".join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print(f"[!] MFA alignment with multiprocessing returned code {res.returncode}. Retrying single-process...")
        cmd_fallback = [
            str(MICROMAMBA_EXE),
            "run",
            "-r", "C:/Users/Sean/micromamba",
            "-n", "mfa_env",
            "mfa", "align",
            corpus_dir,
            dict_arg,
            acoustic_model,
            output_aligned_dir,
            "--clean",
            "-j", "1"
        ]
        res = subprocess.run(cmd_fallback)
        if res.returncode != 0:
            raise RuntimeError("MFA alignment failed. Check MFA log files.")

    print(f"[+] MFA Alignment complete! TextGrids saved to: {output_aligned_dir}")


def ensure_source_aligned(video_path: str, vtt_path: str = None, force_realign: bool = False) -> str:
    """
    Checks if a video has already been aligned in the cache.
    If cached, returns the path to its units_index.json.
    If not cached (or force_realign is True), executes preprocessing, MFA alignment, and inventory indexing.
    """
    v_abs = os.path.abspath(video_path)
    if not os.path.exists(v_abs):
        raise FileNotFoundError(f"Source video not found: {v_abs}")

    source_id = get_source_id(v_abs)
    source_cache_dir = CACHE_BASE_DIR / source_id
    corpus_dir = source_cache_dir / "corpus"
    aligned_dir = source_cache_dir / "aligned"
    manifest_path = corpus_dir / "corpus_manifest.json"
    inventory_path = source_cache_dir / "units_index.json"
    dict_path = SCRIPT_DIR / "dictionaries" / "merged_dictionary.dict"
    if not dict_path.exists():
        dict_path = SCRIPT_DIR / "dictionaries" / "merged_beato_dict.dict"
    if not dict_path.exists():
        dict_path = SCRIPT_DIR / "dictionaries" / "custom_pronunciations.txt"
    if not dict_path.exists():
        dict_path = SCRIPT_DIR / "dictionaries" / "custom_beato_dict.txt"

    # Check for legacy alignment for the default reference video
    legacy_inventory = SCRIPT_DIR / "inventory" / "units_index.json"
    if not inventory_path.exists() and legacy_inventory.exists() and not force_realign:
        try:
            with open(legacy_inventory, "r", encoding="utf-8") as lif:
                data = json.load(lif)
            if os.path.abspath(data.get("source_video", "")) == v_abs:
                print(f"[+] Found existing cached alignment for: {Path(v_abs).name}")
                return str(legacy_inventory)
        except Exception:
            pass

    # Check if cached inventory already exists
    if inventory_path.exists() and not force_realign:
        print(f"[+] Using cached alignment index: {inventory_path}")
        return str(inventory_path)

    # Need to run alignment
    if not vtt_path:
        vtt_path = find_associated_vtt(v_abs)
        if not vtt_path or not os.path.exists(vtt_path):
            raise FileNotFoundError(
                f"No matching .vtt file found for {v_abs}. Please specify --vtt explicitly."
            )

    print(f"\n[+] Aligning source: {Path(v_abs).name}")
    print(f"    Subtitles: {Path(vtt_path).name}")
    print(f"    Cache Dir: {source_cache_dir}")

    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)

    # 1. Preprocess
    print("\n--- [1/3] Preprocessing Subtitles & Audio ---")
    preprocess_source(v_abs, vtt_path, str(corpus_dir))

    # 2. MFA Alignment
    print("\n--- [2/3] Running Forced Alignment ---")
    run_mfa_align(str(corpus_dir), str(aligned_dir), str(dict_path))

    # 3. Build Inventory
    print("\n--- [3/3] Building Unit Inventory Index ---")
    build_units_inventory(str(aligned_dir), str(manifest_path), str(inventory_path))

    return str(inventory_path)


def merge_inventories(inventories: list) -> dict:
    """
    Combines multiple unit inventories in memory into a single unified corpus index.
    Preserves source_video reference on each unit for accurate multi-source rendering.
    """
    combined = {
        "source_videos": [],
        "total_unique_words": 0,
        "total_words_instances": 0,
        "total_unique_phones": 0,
        "total_phones_instances": 0,
        "words": {},
        "phones": {}
    }

    words_db = combined["words"]
    phones_db = combined["phones"]
    seen_videos = set()

    for inv in inventories:
        src = inv.get("source_video")
        if src and src not in seen_videos:
            seen_videos.add(src)
            combined["source_videos"].append(src)
        elif "source_videos" in inv:
            for s in inv["source_videos"]:
                if s not in seen_videos:
                    seen_videos.add(s)
                    combined["source_videos"].append(s)

        # Merge words
        for word, instances in inv.get("words", {}).items():
            if word not in words_db:
                words_db[word] = []
            words_db[word].extend(instances)
            combined["total_words_instances"] += len(instances)

        # Merge phones
        for phone, instances in inv.get("phones", {}).items():
            if phone not in phones_db:
                phones_db[phone] = []
            phones_db[phone].extend(instances)
            combined["total_phones_instances"] += len(instances)

    combined["total_unique_words"] = len(words_db)
    combined["total_unique_phones"] = len(phones_db)
    return combined


def main():
    parser = argparse.ArgumentParser(description="YTP Sentence Mixing Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Mix (Primary command: accepts multiple --video or --inventory)
    p_mix = subparsers.add_parser("mix", help="Synthesize target sentence using one or more source videos (cached or newly aligned)")
    p_mix.add_argument("--text", required=True, help="Target text / sentence to synthesize")
    p_mix.add_argument("--video", nargs="+", help="Path(s) to source video(s) (.mp4). Accepts multiple paths or comma-separated lists. Checks cache or aligns.")
    p_mix.add_argument("--vtt", nargs="+", help="Optional path(s) to WebVTT subtitles (.vtt). Auto-detected if alongside videos.")
    p_mix.add_argument("--manifest", help="Path to JSON manifest containing a list of objects with 'video' and optional 'vtt' paths.")
    p_mix.add_argument("--inventory", nargs="+", help="Direct path(s) to units_index.json (bypasses video cache lookup)")
    p_mix.add_argument("--xml", help="Output Premiere Pro XML path (e.g. output/my_mix.xml)")
    p_mix.add_argument("--render", help="Output rendered MP4 path (e.g. output/my_mix.mp4)")
    p_mix.add_argument("--fps", type=float, default=29.97, help="Timeline framerate (default: 29.97)")
    p_mix.add_argument("--force-realign", action="store_true", help="Force re-align even if source is cached")

    # 2. Preprocess standalone
    p_prep = subparsers.add_parser("preprocess", help="Preprocess video and VTT into corpus chunks")
    p_prep.add_argument("--video", required=True, help="Path to input video (.mp4)")
    p_prep.add_argument("--vtt", required=True, help="Path to WebVTT subtitles (.vtt)")
    p_prep.add_argument("--output", default=str(SCRIPT_DIR / "corpus"), help="Output corpus directory")

    # 3. Align standalone
    p_align = subparsers.add_parser("align", help="Run MFA alignment on corpus")
    p_align.add_argument("--corpus", default=str(SCRIPT_DIR / "corpus"), help="Corpus directory with WAV and TXT")
    p_align.add_argument("--output", default=str(SCRIPT_DIR / "aligned"), help="Output aligned TextGrid directory")
    p_align.add_argument("--dict", default=str(SCRIPT_DIR / "dictionaries" / "merged_beato_dict.dict"), help="Dictionary path")

    # 4. Build Inventory standalone
    p_inv = subparsers.add_parser("build-inventory", help="Build units index JSON from aligned TextGrids")
    p_inv.add_argument("--aligned", default=str(SCRIPT_DIR / "aligned"), help="Aligned TextGrids directory")
    p_inv.add_argument("--manifest", default=str(SCRIPT_DIR / "corpus" / "corpus_manifest.json"), help="Manifest JSON")
    p_inv.add_argument("--output", default=str(SCRIPT_DIR / "inventory" / "units_index.json"), help="Output units index")

    # 5. End-to-End Pipeline
    p_full = subparsers.add_parser("auto", help="Run full pipeline: Preprocess -> Align -> Inventory -> Mix")
    p_full.add_argument("--video", required=True, help="Input video path")
    p_full.add_argument("--vtt", help="Input VTT path (optional, auto-detected if alongside video)")
    p_full.add_argument("--text", required=True, help="Target text to synthesize")
    p_full.add_argument("--output-name", default="ytp_mix", help="Base name for output mix files")
    p_full.add_argument("--fps", type=float, default=29.97, help="Timeline framerate (default: 29.97)")

    args = parser.parse_args()

    if args.command == "mix":
        # Parse video paths (supports space-separated nargs, comma-separated strings, or --manifest)
        raw_videos = []
        raw_vtts = []

        if args.manifest:
            if not os.path.exists(args.manifest):
                sys.exit(f"Error: Manifest file not found: {args.manifest}")
            with open(args.manifest, "r", encoding="utf-8") as mf:
                manifest_data = json.load(mf)
            for item in manifest_data:
                if isinstance(item, dict) and "video" in item:
                    raw_videos.append(item["video"])
                    raw_vtts.append(item.get("vtt"))
                elif isinstance(item, str):
                    raw_videos.append(item)
                    raw_vtts.append(None)

        if args.video:
            for item in args.video:
                for part in item.split(","):
                    clean = part.strip()
                    if clean:
                        raw_videos.append(clean)

        if args.vtt:
            for item in args.vtt:
                for part in item.split(","):
                    clean = part.strip()
                    if clean:
                        raw_vtts.append(clean)

        raw_inventories = []
        if args.inventory:
            for item in args.inventory:
                for part in item.split(","):
                    clean = part.strip()
                    if clean:
                        raw_inventories.append(clean)

        loaded_inventories = []

        # 1. Load any directly specified inventory JSONs
        for inv_p in raw_inventories:
            if not os.path.exists(inv_p):
                sys.exit(f"Error: Inventory index not found at {inv_p}")
            with open(inv_p, "r", encoding="utf-8") as f:
                loaded_inventories.append(json.load(f))

        # 2. Load / align video sources
        for idx, vid_p in enumerate(raw_videos):
            custom_vtt = raw_vtts[idx] if idx < len(raw_vtts) else None
            inv_p = ensure_source_aligned(vid_p, custom_vtt, force_realign=args.force_realign)
            with open(inv_p, "r", encoding="utf-8") as f:
                loaded_inventories.append(json.load(f))

        # 3. Fallback to default reference inventory if nothing provided
        if not loaded_inventories:
            default_inv = SCRIPT_DIR / "inventory" / "units_index.json"
            if default_inv.exists():
                print(f"[*] No sources specified. Falling back to reference inventory: {default_inv}")
                with open(default_inv, "r", encoding="utf-8") as f:
                    loaded_inventories.append(json.load(f))
            else:
                sys.exit("Error: Please provide --manifest <file.json>, --video <path_to_video...>, or --inventory <path_to_index.json...>.")

        # Combine inventories in memory
        if len(loaded_inventories) == 1:
            inv = loaded_inventories[0]
            print(f"[+] Loaded single corpus: {inv.get('total_unique_words', len(inv.get('words', {})))} unique words, "
                  f"{inv.get('total_unique_phones', len(inv.get('phones', {})))} phonemes")
        else:
            inv = merge_inventories(loaded_inventories)
            print(f"[+] Successfully merged {len(loaded_inventories)} source inventories in memory:")
            print(f"    - Unique words across corpus: {inv['total_unique_words']} (total instances: {inv['total_words_instances']})")
            print(f"    - Unique phonemes across corpus: {inv['total_unique_phones']} (total instances: {inv['total_phones_instances']})")

        units, plan = synthesize_sentence(args.text, inv)
        print("\n--- Synthesized Cut Plan ---")
        for p in plan:
            print(f"  {p}")

        # Video render (defaults to output/mix_<name>.mp4 if not specified)
        clean_text_name = re.sub(r"[^\w\-_]", "_", args.text[:20])
        out_render = args.render or str(SCRIPT_DIR / "output" / f"mix_{clean_text_name}.mp4")

        # Only generate XML if explicitly requested via --xml
        if args.xml:
            generate_fcp_xml(units, args.xml, sequence_name=f"YTP_{clean_text_name}", fps=args.fps)

        render_sentence_mix(units, out_render, target_fps=args.fps)

    elif args.command == "auto":
        inventory_path = ensure_source_aligned(args.video, args.vtt)
        with open(inventory_path, "r", encoding="utf-8") as f:
            inv = json.load(f)

        units, plan = synthesize_sentence(args.text, inv)
        print("\n--- Synthesized Cut Plan ---")
        for p in plan:
            print(f"  {p}")

        out_mp4 = str(SCRIPT_DIR / "output" / f"{args.output_name}.mp4")
        render_sentence_mix(units, out_mp4, target_fps=args.fps)

        if getattr(args, "xml", None):
            generate_fcp_xml(units, args.xml, sequence_name=f"YTP_{args.output_name}", fps=args.fps)

    elif args.command == "preprocess":
        preprocess_source(args.video, args.vtt, args.output)

    elif args.command == "align":
        run_mfa_align(args.corpus, args.output, args.dict)

    elif args.command == "build-inventory":
        build_units_inventory(args.aligned, args.manifest, args.output)


if __name__ == "__main__":
    main()
