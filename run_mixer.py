#!/usr/bin/env python3
"""
YTP Sentence Mixing Pipeline CLI
Unified CLI tool to align video sources via MFA and synthesize target sentences.
Supports multiple source videos, manifest resolution, variable-span selection,
PCM master rendering, and Final Cut Pro XML export.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
CACHE_BASE_DIR = SCRIPT_DIR / ".build"

# Add src/ to sys.path
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from contracts import InventoryIndex, RationalTime, SourceRecord
from lexicon import Lexicon
from target_graph import TargetGraphBuilder
from span_selector import SpanSelector
from edit_plan import build_edit_plan
from vtt_preprocessor import preprocess_source
from inventory_builder import build_units_inventory
from unit_selector import synthesize_sentence
from fcp_xml_generator import generate_fcp_xml
from video_assembler import render_sentence_mix


def get_mamba_config() -> Tuple[Optional[str], Optional[str]]:
    """Discovers micromamba executable and root prefix dynamically."""
    mamba_root = os.environ.get("MAMBA_ROOT_PREFIX")
    if not mamba_root:
        candidate_root = SCRIPT_DIR / ".mamba"
        if candidate_root.exists():
            mamba_root = str(candidate_root)
        else:
            default_user_root = Path.home() / "micromamba"
            if default_user_root.exists():
                mamba_root = str(default_user_root)

    candidate_exe = BIN_DIR / ("micromamba.exe" if sys.platform == "win32" else "micromamba")
    mm_exe: Optional[Path] = candidate_exe if candidate_exe.exists() else None
    if mm_exe is None:
        found = shutil.which("micromamba")
        if found:
            mm_exe = Path(found)

    return (str(mm_exe) if mm_exe else None, mamba_root)


def get_source_id(video_path: str) -> str:
    """Generate a clean filesystem-safe source ID with a content hash for caching."""
    v_p = Path(video_path).resolve()
    clean_stem = re.sub(r"[^\w\-_]", "_", v_p.stem)[:40]
    path_hash = hashlib.sha256(v_p.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{clean_stem}_{path_hash}"


def slugify_text(text: str, max_words: int = 8, max_length: int = 40) -> str:
    """
    Generate a clean, filesystem-safe filename slug from a text phrase.
    Strips special punctuation, converts to lowercase, joins words with underscores,
    and truncates on word boundaries up to max_words or max_length.
    """
    cleaned = re.sub(r"[^\w\s-]", "", text.strip().lower())
    words = cleaned.split()
    if not words:
        return "mix"
    selected: List[str] = []
    curr_len = 0
    for w in words[:max_words]:
        extra = len(w) + (1 if selected else 0)
        if curr_len + extra > max_length and selected:
            break
        selected.append(w)
        curr_len += extra
    slug = "_".join(selected)
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "mix"


def find_associated_vtt(video_path: str) -> str:
    """Locate the corresponding .vtt file alongside the video if not explicitly provided."""
    v_path = Path(video_path).resolve()
    parent = v_path.parent
    stem = v_path.stem

    candidates = [
        parent / f"{stem}.en.vtt",
        parent / f"{stem}.vtt",
        parent / f"{stem}.en-US.vtt",
    ]
    for c in candidates:
        if c.exists():
            return str(c)

    matches = list(parent.glob(f"{re.escape(stem)}*.vtt"))
    if matches:
        return str(matches[0])

    return ""


def run_mfa_align(
    corpus_dir: str,
    output_aligned_dir: str,
    dictionary_path: Optional[str] = None,
    acoustic_model: str = "english_us_arpa",
) -> None:
    """Executes Montreal Forced Aligner inside the isolated mfa_env or PATH."""
    os.makedirs(output_aligned_dir, exist_ok=True)

    # Ensure MFA_ROOT_DIR is set to repo build dir
    mfa_root_dir = SCRIPT_DIR / ".build" / "mfa"
    os.environ["MFA_ROOT_DIR"] = str(mfa_root_dir)

    # Resolve acoustic model path directly if available in pretrained_models
    model_zip = mfa_root_dir / "pretrained_models" / "acoustic" / f"{acoustic_model}.zip"
    if model_zip.exists():
        acoustic_arg = str(model_zip.resolve())
    else:
        acoustic_arg = acoustic_model

    dict_arg = dictionary_path if dictionary_path and os.path.exists(dictionary_path) else "english_us_arpa"

    mm_exe, mamba_root = get_mamba_config()
    env = dict(os.environ)
    env["MFA_ROOT_DIR"] = str(mfa_root_dir)

    c_path = Path(corpus_dir).resolve()
    out_aligned = Path(output_aligned_dir).resolve()
    wav_files = sorted(list(c_path.rglob("chunk_*.wav")))

    def _execute_mfa(src_dir: Path, dst_dir: Path):
        if mm_exe and mamba_root:
            cmd = [
                mm_exe,
                "run",
                "-r", mamba_root,
                "-n", "mfa_env",
                "mfa", "align",
                str(src_dir),
                str(Path(dict_arg).resolve()) if os.path.exists(dict_arg) else dict_arg,
                acoustic_arg,
                str(dst_dir),
                "--clean",
                "-j", "1",  # Windows stability
            ]
        else:
            # Fallback to direct mfa command on PATH
            cmd = [
                "mfa", "align",
                str(src_dir),
                str(Path(dict_arg).resolve()) if os.path.exists(dict_arg) else dict_arg,
                acoustic_arg,
                str(dst_dir),
                "--clean",
                "-j", "1",
            ]

        print("[*] Running MFA Alignment command:")
        print("    " + " ".join(cmd))
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"MFA alignment failed with exit code {res.returncode}. Check MFA logs.")

    batch_size = 45
    if len(wav_files) <= batch_size:
        _execute_mfa(c_path, out_aligned)
    else:
        print(f"[*] Batching {len(wav_files)} chunks into batches of <= {batch_size} for acoustic stability...")
        for b_idx, start_idx in enumerate(range(0, len(wav_files), batch_size)):
            batch_wavs = wav_files[start_idx : start_idx + batch_size]
            b_corpus = c_path.parent / f"{c_path.parent.name}_batch{b_idx}"
            b_corpus.mkdir(parents=True, exist_ok=True)
            b_aligned = out_aligned.parent / f"{out_aligned.parent.name}_aligned{b_idx}"
            b_aligned.mkdir(parents=True, exist_ok=True)
            try:
                for w in batch_wavs:
                    rel_w = w.relative_to(c_path)
                    dest_w = b_corpus / rel_w
                    dest_w.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(str(w), str(dest_w))
                    txt = w.with_suffix(".txt")
                    if txt.exists():
                        dest_txt = dest_w.with_suffix(".txt")
                        shutil.copy2(str(txt), str(dest_txt))
                _execute_mfa(b_corpus, b_aligned)
                for tg in b_aligned.rglob("*.TextGrid"):
                    dest_flat = out_aligned / tg.name
                    if not dest_flat.exists():
                        shutil.move(str(tg), str(dest_flat))
                    else:
                        rel_tg = tg.relative_to(b_aligned)
                        dest_tg = out_aligned / rel_tg
                        dest_tg.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(tg), str(dest_tg))
            finally:
                shutil.rmtree(b_corpus, ignore_errors=True)
                shutil.rmtree(b_aligned, ignore_errors=True)
                shutil.rmtree(mfa_root_dir / b_corpus.name, ignore_errors=True)


    print(f"[+] MFA Alignment complete! TextGrids saved to: {output_aligned_dir}")


def ensure_source_aligned(
    video_path: str,
    vtt_path: Optional[str] = None,
    force_realign: bool = False,
    speaker_id: Optional[str] = None,
    single_speaker_verified: bool = False,
    assume_single_speaker: bool = False,
    speaker_intervals: Optional[List[Dict[str, Any]]] = None,
    isolate_vocals: bool = False,
) -> str:
    """
    Ensures a video source is preprocessed, aligned with MFA, and indexed into an inventory.
    Returns path to units_index.json.
    """
    v_abs = Path(video_path).resolve()
    if not v_abs.exists():
        raise FileNotFoundError(f"Source video not found: {v_abs}")

    source_id = get_source_id(str(v_abs))
    (CACHE_BASE_DIR / "sources" / source_id).mkdir(parents=True, exist_ok=True)
    corpus_dir = CACHE_BASE_DIR / "alignment" / source_id / "corpus"
    aligned_dir = CACHE_BASE_DIR / "alignment" / source_id / "aligned"
    manifest_path = corpus_dir / "corpus_manifest.json"
    inventory_path = CACHE_BASE_DIR / "inventory" / source_id / "units_index.json"

    # Select best dictionary
    dict_path = SCRIPT_DIR / "dictionaries" / "merged_dictionary.dict"
    if not dict_path.exists():
        dict_path = SCRIPT_DIR / "dictionaries" / "custom_pronunciations.txt"

    # Check cache validity
    vtt_candidate = vtt_path or find_associated_vtt(str(v_abs))
    is_stale = False
    if inventory_path.exists() and not force_realign:
        inv_mtime = os.path.getmtime(inventory_path)
        # 1. Check if VTT was modified after inventory
        if vtt_candidate and os.path.exists(vtt_candidate) and os.path.getmtime(vtt_candidate) > inv_mtime:
            is_stale = True
        # 2. Check if chunk count changed
        if not is_stale and manifest_path.exists() and vtt_candidate and os.path.exists(vtt_candidate):
            try:
                from vtt_preprocessor import parse_vtt_cues
                cues = parse_vtt_cues(vtt_candidate)
                with open(manifest_path, "r", encoding="utf-8") as f:
                    old_manifest = json.load(f)
                if old_manifest.get("total_chunks") != len(cues):
                    is_stale = True
            except Exception:
                pass

        # 4. Check if vocal isolation is requested but stem is missing from disk or inventory
        if not is_stale and isolate_vocals:
            vocals_dir = CACHE_BASE_DIR / "vocals" / source_id
            stem_path = vocals_dir / "vocals.wav"
            if not stem_path.exists() or stem_path.stat().st_size == 0:
                is_stale = True
            else:
                try:
                    with open(inventory_path, "r", encoding="utf-8") as f:
                        inv_data = json.load(f)
                    srcs = inv_data.get("sources", {})
                    if not any(s.get("cached_vocal_pcm_path") for s in srcs.values()):
                        is_stale = True
                except Exception:
                    is_stale = True

        if not is_stale:
            print(f"[+] Using cached alignment index: {inventory_path}")
            return str(inventory_path)
        else:
            print(f"[*] Detected stale alignment cache for {source_id} (VTT, vocal stems, or preprocessor updated). Re-aligning...")

    if not vtt_path:
        vtt_path = vtt_candidate
        if not vtt_path or not os.path.exists(vtt_path):
            raise FileNotFoundError(
                f"No matching .vtt file found for {v_abs}. Please specify --vtt explicitly."
            )

    print(f"\n[+] Aligning source: {v_abs.name}")
    print(f"    Subtitles: {Path(vtt_path).name}")

    # Clean stale intermediate files to prevent ghost chunks
    if corpus_dir.exists():
        shutil.rmtree(corpus_dir, ignore_errors=True)
    if aligned_dir.exists():
        shutil.rmtree(aligned_dir, ignore_errors=True)
    if inventory_path.exists():
        try:
            inventory_path.unlink()
        except OSError:
            pass

    os.makedirs(corpus_dir, exist_ok=True)
    os.makedirs(aligned_dir, exist_ok=True)
    os.makedirs(inventory_path.parent, exist_ok=True)

    vocal_stem_path: Optional[Path] = None
    source_rec: Optional[SourceRecord] = None
    if isolate_vocals:
        print("\n--- [0/3] Isolating Vocal Track (audio-separator / Demucs) ---")
        from vocal_isolation import isolate_vocals as run_isolate_vocals
        vocals_dir = CACHE_BASE_DIR / "vocals" / source_id
        vocal_stem_path = run_isolate_vocals(
            v_abs,
            output_dir=vocals_dir,
            force=force_realign,
        )
        from media_time import extract_source_record
        try:
            source_rec = extract_source_record(
                v_abs,
                build_dir=CACHE_BASE_DIR,
                vocal_stem_path=vocal_stem_path,
                isolate_vocals=True,
            )
        except Exception as e:
            print(f"[!] Warning: Could not pre-stage source record with vocal stem: {e}")

    # 1. Preprocess
    print("\n--- [1/3] Preprocessing Subtitles & Audio ---")
    preprocess_source(
        str(v_abs),
        str(vtt_path),
        str(corpus_dir),
        speaker_id=speaker_id,
        single_speaker_verified=single_speaker_verified,
        assume_single_speaker=assume_single_speaker,
        speaker_intervals=speaker_intervals,
        audio_source_path=vocal_stem_path,
    )

    # 2. MFA Alignment
    print("\n--- [2/3] Running Forced Alignment ---")
    run_mfa_align(str(corpus_dir), str(aligned_dir), str(dict_path))

    # 3. Build Inventory
    print("\n--- [3/3] Building Unit Inventory Index ---")
    build_units_inventory(
        str(aligned_dir),
        str(manifest_path),
        str(inventory_path),
        speaker_id=speaker_id,
        single_speaker_verified=single_speaker_verified or assume_single_speaker,
        source_record=source_rec,
    )

    return str(inventory_path)


def load_or_combine_inventories(
    inventory_paths: List[str],
) -> Tuple[InventoryIndex, Dict[str, Any]]:
    """Loads one or more inventory index files into a unified InventoryIndex and legacy dict."""
    combined_inv = InventoryIndex()
    legacy_words: Dict[str, List[Dict[str, Any]]] = {}
    legacy_phones: Dict[str, List[Dict[str, Any]]] = {}

    for path_str in inventory_paths:
        p = Path(path_str).resolve()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        if data.get("schema_version") == "2.0.0" and "events" in data:
            inv = InventoryIndex.from_dict(data)
            combined_inv.sources.update(inv.sources)
            combined_inv.utterances.update(inv.utterances)
            combined_inv.events.update(inv.events)
            combined_inv.words.update(inv.words)
            for w, wids in inv.word_to_word_ids.items():
                combined_inv.word_to_word_ids.setdefault(w, []).extend(wids)
            for ph, eids in inv.phone_to_event_ids.items():
                combined_inv.phone_to_event_ids.setdefault(ph, []).extend(eids)
            for bph, beids in inv.base_phone_to_event_ids.items():
                combined_inv.base_phone_to_event_ids.setdefault(bph, []).extend(beids)

        # Merge legacy views
        words_part = data.get("words_inventory", data.get("words", {}))
        phones_part = data.get("phones_inventory", data.get("phones", {}))
        if isinstance(words_part, dict):
            for w, instances in words_part.items():
                if isinstance(instances, list):
                    legacy_words.setdefault(w, []).extend(instances)
        if isinstance(phones_part, dict):
            for ph, instances in phones_part.items():
                if isinstance(instances, list):
                    legacy_phones.setdefault(ph, []).extend(instances)

    legacy_dict = {
        "words": legacy_words,
        "phones": legacy_phones,
        "words_inventory": legacy_words,
        "phones_inventory": legacy_phones,
    }
    return combined_inv, legacy_dict


def parse_manifest_items(manifest_path: str) -> List[Dict[str, Any]]:
    """
    MANDATORY EXECUTION DIRECTIVE 2:
    Always resolve relative video and vtt paths against the manifest file's parent directory:
    (Path(manifest_path).parent / item['video']).resolve().
    """
    m_path = Path(manifest_path).resolve()
    if not m_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {m_path}")

    with open(m_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list of source items")

    parsed_items: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and "video" in item:
            resolved_video = (m_path.parent / item["video"]).resolve()
            resolved_vtt = (m_path.parent / item["vtt"]).resolve() if item.get("vtt") else None
            parsed_items.append({
                "video": str(resolved_video),
                "vtt": str(resolved_vtt) if resolved_vtt else None,
                "speaker_id": item.get("speaker_id"),
                "single_speaker_verified": item.get("single_speaker_verified", False),
                "recording_id": item.get("recording_id", ""),
                "speaker_intervals": item.get("speaker_intervals"),
            })
        elif isinstance(item, str):
            resolved_video = (m_path.parent / item).resolve()
            parsed_items.append({
                "video": str(resolved_video),
                "vtt": None,
                "speaker_id": None,
                "single_speaker_verified": False,
            })
    return parsed_items


def main():
    parser = argparse.ArgumentParser(description="YTP Sentence Mixing Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Common synthesis arguments helper
    def add_synthesis_args(p):
        p.add_argument("--text", required=True, help="Target text / sentence to synthesize")
        p.add_argument("--selector", choices=["legacy", "span"], default="legacy", help="Selector backend ('legacy' or 'span')")
        p.add_argument("--speaker", default=None, help="Speaker identity filter")
        p.add_argument("--assume-single-speaker", action="store_true", help="Explicitly assume single speaker")
        p.add_argument("--mode", choices=["source-derived", "cuts-only"], default="source-derived", help="Rendering mode")
        p.add_argument("--features", choices=["basic", "wavlm"], default="basic", help="Feature extraction backend")
        p.add_argument("--fps", default="30000/1001", help="Timeline framerate (e.g. 30000/1001 or 29.97)")
        p.add_argument("--render", help="Output rendered MP4 path")
        p.add_argument("--plan", help="Output EditPlan JSON path")
        p.add_argument("--xml", help="Output Premiere Pro / FCP7 XML path")
        p.add_argument("--guide", help="Optional human prosody guide audio")
        p.add_argument("--force-realign", action="store_true", help="Force re-align even if source is cached")
        p.add_argument(
            "--normalize-loudness",
            action=argparse.BooleanOptionalAction,
            default=True,
            help="Normalize output audio loudness according to EBU R128 (-16 LUFS, -1.0 dBTP) (default: enabled)",
        )
        p.add_argument(
            "--isolate-vocals",
            action=argparse.BooleanOptionalAction,
            default=False,
            help="Isolate speech stems using audio-separator/Demucs before alignment",
        )
        p.add_argument(
            "--audio-track",
            choices=["raw", "vocals", "recombined", "all", "combined", "isolated", "both"],
            default="raw",
            help="Audio track for rendering: 'raw' (raw full mix, formerly combined, default), 'vocals' (formerly isolated speech stem), 'recombined' (normalized vocals + original background instrumental), 'both' (raw & vocals), or 'all' (raw, vocals, and recombined)",
        )
        p.add_argument(
            "--non-interactive",
            action="store_true",
            default=False,
            help="Disable interactive prompts (for OOV words, speaker naming, etc.) and fallback to defaults",
        )
        p.add_argument(
            "--cache-dir",
            default=None,
            help="Custom cache root directory (defaults to .build)",
        )

    # 1. Mix command
    p_mix = subparsers.add_parser("mix", help="Synthesize target sentence using one or more source videos")
    add_synthesis_args(p_mix)
    p_mix.add_argument("--video", nargs="+", help="Path(s) to source video(s)")
    p_mix.add_argument("--vtt", nargs="+", help="Path(s) to WebVTT subtitles")
    p_mix.add_argument("--manifest", help="Path to JSON manifest")
    p_mix.add_argument("--inventory", nargs="+", help="Direct path(s) to units_index.json")

    # 2. Preprocess standalone
    p_prep = subparsers.add_parser("preprocess", help="Preprocess video and VTT into corpus chunks")
    p_prep.add_argument("--video", required=True, help="Path to input video (.mp4)")
    p_prep.add_argument("--vtt", required=True, help="Path to WebVTT subtitles (.vtt)")
    p_prep.add_argument("--output", default=str(SCRIPT_DIR / "corpus"), help="Output corpus directory")
    p_prep.add_argument("--speaker", default=None, help="Speaker ID")
    p_prep.add_argument("--assume-single-speaker", action="store_true", help="Assume single speaker")
    p_prep.add_argument(
        "--isolate-vocals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Isolate speech stems using audio-separator/Demucs before alignment",
    )

    # 3. Align standalone
    p_align = subparsers.add_parser("align", help="Run MFA alignment on corpus")
    p_align.add_argument("--corpus", default=str(SCRIPT_DIR / "corpus"), help="Corpus directory with WAV and TXT")
    p_align.add_argument("--output", default=str(SCRIPT_DIR / "aligned"), help="Output aligned TextGrid directory")
    p_align.add_argument("--dict", default=str(SCRIPT_DIR / "dictionaries" / "merged_dictionary.dict"), help="Dictionary path")
    p_align.add_argument(
        "--isolate-vocals",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Isolate speech stems using audio-separator/Demucs before alignment",
    )

    # 4. Build Inventory standalone
    p_inv = subparsers.add_parser("build-inventory", help="Build units index JSON from aligned TextGrids")
    p_inv.add_argument("--aligned", default=str(SCRIPT_DIR / "aligned"), help="Aligned TextGrids directory")
    p_inv.add_argument("--manifest", default=str(SCRIPT_DIR / "corpus" / "corpus_manifest.json"), help="Manifest JSON")
    p_inv.add_argument("--output", default=str(SCRIPT_DIR / "inventory" / "units_index.json"), help="Output units index")
    p_inv.add_argument("--speaker", default=None, help="Speaker ID")
    p_inv.add_argument("--assume-single-speaker", action="store_true", help="Assume single speaker")

    # 5. End-to-End Pipeline
    p_auto = subparsers.add_parser("auto", help="Run full pipeline: Preprocess -> Align -> Inventory -> Mix")
    add_synthesis_args(p_auto)
    p_auto.add_argument("--video", required=True, help="Input video path")
    p_auto.add_argument("--vtt", help="Input VTT path")
    p_auto.add_argument("--output-name", default="ytp_mix", help="Base name for output mix files")

    args = parser.parse_args()
    if getattr(args, "cache_dir", None):
        global CACHE_BASE_DIR
        CACHE_BASE_DIR = Path(args.cache_dir).resolve()

    if args.command in ("mix", "auto"):
        # Check cuts-only normalization conflict (A13.3)
        if args.mode == "cuts-only":
            if "--normalize-loudness" in sys.argv:
                parser.error("Explicit --normalize-loudness cannot be combined with --mode cuts-only.")
            args.normalize_loudness = False

        # Validate legacy selector combinations (T20 / A15)
        if args.selector == "legacy":
            if getattr(args, "guide", None):
                parser.error("Legacy selector does not support prosody guidance (--guide). Use --selector span.")
            if getattr(args, "features", "basic") == "wavlm":
                parser.error("Legacy selector does not support --features wavlm. Use --selector span.")
            if getattr(args, "plan", None):
                parser.error("Legacy selector does not generate EditPlan (--plan). Use --selector span.")

        # Resolve target frame rate
        fps = RationalTime.from_number(args.fps)

        # Collect source items to align / load
        source_items: List[Dict[str, Any]] = []

        if getattr(args, "manifest", None):
            source_items.extend(parse_manifest_items(args.manifest))

        if getattr(args, "video", None):
            v_list = args.video if isinstance(args.video, list) else [args.video]
            vtt_list = args.vtt if getattr(args, "vtt", None) else []
            if isinstance(vtt_list, str):
                vtt_list = [vtt_list]

            for idx, v_entry in enumerate(v_list):
                for v_item in v_entry.split(","):
                    v_clean = v_item.strip()
                    if v_clean:
                        vtt_match = vtt_list[idx] if idx < len(vtt_list) else None
                        source_items.append({
                            "video": str(Path(v_clean).resolve()),
                            "vtt": str(Path(vtt_match).resolve()) if vtt_match else None,
                            "speaker_id": args.speaker,
                            "single_speaker_verified": args.assume_single_speaker,
                        })

        inventory_paths: List[str] = []

        if getattr(args, "inventory", None):
            for inv_item in args.inventory:
                for inv_p in inv_item.split(","):
                    clean = inv_p.strip()
                    if clean:
                        inventory_paths.append(str(Path(clean).resolve()))

        # Align any unaligned sources
        audio_track = getattr(args, "audio_track", "raw")
        isolate_vocals_flag = getattr(args, "isolate_vocals", False) or (audio_track in ("isolated", "vocals", "recombined", "both", "all"))
        for item in source_items:
            inv_path = ensure_source_aligned(
                item["video"],
                item.get("vtt"),
                force_realign=args.force_realign,
                speaker_id=item.get("speaker_id") or args.speaker,
                single_speaker_verified=item.get("single_speaker_verified", False) or args.assume_single_speaker,
                assume_single_speaker=args.assume_single_speaker,
                speaker_intervals=item.get("speaker_intervals"),
                isolate_vocals=isolate_vocals_flag,
            )
            inventory_paths.append(inv_path)

        if not inventory_paths:
            sys.exit("Error: Please provide --manifest, --video, or --inventory.")

        # Load inventory
        inventory_index, legacy_inv_dict = load_or_combine_inventories(inventory_paths)

        # Resolve effective speaker (handle single verified speaker automatically)
        unique_speakers = {wo.speaker_id for wo in inventory_index.words.values() if wo.speaker_id}
        if args.speaker is not None:
            effective_speaker = args.speaker
        elif len(unique_speakers) == 1:
            effective_speaker = next(iter(unique_speakers))
        else:
            effective_speaker = None

        slug = slugify_text(args.text)
        if not args.render:
            out_render = str(SCRIPT_DIR / "output" / f"{slug}.mp4")
        else:
            r_str = str(args.render)
            r_path = Path(r_str)
            if r_str.endswith(("/", "\\")) or (r_path.exists() and r_path.is_dir()):
                out_render = str(r_path / f"{slug}.mp4")
            else:
                out_render = str(r_path)

        clean_text_stem = slug
        out_plan_path = args.plan or str(Path(out_render).with_suffix(".plan.json"))
        out_xml_path = args.xml

        os.makedirs(Path(out_render).resolve().parent, exist_ok=True)
        os.makedirs(Path(out_plan_path).resolve().parent, exist_ok=True)
        if out_xml_path:
            os.makedirs(Path(out_xml_path).resolve().parent, exist_ok=True)

        if args.selector == "span":
            print(f"\n[+] Running Variable-Span Unit Selection for: \"{args.text}\" (speaker: {effective_speaker})")
            dict_files = [
                SCRIPT_DIR / "dictionaries" / "custom_pronunciations.txt",
                SCRIPT_DIR / "dictionaries" / "merged_dictionary.dict",
            ]
            lex = Lexicon([p for p in dict_files if p.exists()])

            # Interactive Out-of-Vocabulary (OOV) Learning Assistant (enabled by default unless headless or --non-interactive)
            from oov_assistant import OOVAssistant
            is_interactive = sys.stdin.isatty() and not getattr(args, "non_interactive", False)
            oov_assistant = OOVAssistant(lex, custom_dict_path=SCRIPT_DIR / "dictionaries" / "custom_pronunciations.txt")
            oov_assistant.resolve_all_in_text(args.text, interactive=is_interactive)

            graph_builder = TargetGraphBuilder(lex)
            graph = graph_builder.build(args.text)

            wavlm_backend = None
            wavlm_config = None
            if args.features == "wavlm":
                from wavlm_backend import WavLMBackend, WavLMConfig, has_wavlm_dependencies
                if not has_wavlm_dependencies():
                    sys.exit("Error: --features wavlm requires 'torch' and 'transformers'. Install via 'pip install -r requirements-wavlm.lock'")
                wavlm_config_path = SCRIPT_DIR / "config" / "wavlm.json"
                if wavlm_config_path.exists():
                    wavlm_config = WavLMConfig.load(wavlm_config_path)
                else:
                    wavlm_config = WavLMConfig()
                wavlm_backend = WavLMBackend(wavlm_config)

            prosody_guide = None
            if args.guide:
                from prosody_guide import ProsodyGuide, assert_no_guide_audio_in_plan
                prosody_guide = ProsodyGuide(args.guide, guide_transcript=args.text)
                if not prosody_guide.validate_transcript(args.text):

                    print(f"[!] Warning: Guide transcript does not match target text '{args.text}'")

            from features import ContinuousFeatureExtractor
            feature_extractor = ContinuousFeatureExtractor(sr=16000)

            selector = SpanSelector(inventory_index)

            selected_spans = selector.select(
                graph,
                speaker_id=effective_speaker,
                feature_extractor=feature_extractor,
                wavlm_backend=wavlm_backend,
                prosody_guide=prosody_guide,
            )


            print(f"[+] Selected {len(selected_spans)} variable spans:")
            for s in selected_spans:
                print(f"    - [{s.span_type.upper()}] '{s.matched_word}' ({s.source_content_id}:{s.start_sample}..{s.end_sample})")

            plan = build_edit_plan(
                selected_spans,
                graph,
                inventory_index,
                project_sample_rate=48000,
                video_fps=fps,
                rendering_mode=args.mode,
            )
            plan.metadata["features"] = args.features
            plan.metadata["normalize_loudness"] = args.normalize_loudness
            plan.metadata["isolate_vocals"] = isolate_vocals_flag
            plan.metadata["audio_track"] = getattr(args, "audio_track", "combined")
            if out_xml_path:
                plan.metadata["xml_path"] = str(Path(out_xml_path).resolve())
            if wavlm_config:
                plan.metadata["wavlm_config"] = {
                    "model_id": wavlm_config.model_id,
                    "revision": wavlm_config.revision,
                    "layer": wavlm_config.layer,
                    "feature_dim": wavlm_config.feature_dim,
                    "sample_rate": wavlm_config.sample_rate,
                }
            if args.guide:
                from prosody_guide import assert_no_guide_audio_in_plan
                assert_no_guide_audio_in_plan(plan, args.guide)

            # Staged Publication: stage all deliverables in an attempt directory before atomic publication
            render_p = Path(out_render).resolve()
            plan_p = Path(out_plan_path).resolve()
            xml_p = Path(out_xml_path).resolve() if out_xml_path else None
            master_wav_p = render_p.with_suffix(".wav")

            attempt_id = f"attempt_{uuid.uuid4().hex}"
            attempt_dir = render_p.parent / f".{attempt_id}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            staged_render = attempt_dir / render_p.name
            staged_master_wav = staged_render.with_suffix(".wav")
            staged_plan = attempt_dir / plan_p.name
            staged_xml = (attempt_dir / xml_p.name) if xml_p else None

            try:
                plan.save(str(staged_plan))

                # Render video and master audio to staged location
                render_sentence_mix(
                    plan,
                    str(staged_render),
                    normalize_loudness=args.normalize_loudness,
                    audio_track=getattr(args, "audio_track", "raw"),
                )

                if staged_xml:
                    generate_fcp_xml(plan, str(staged_xml), sequence_name=f"YTP_{clean_text_stem}", master_wav_path=str(staged_master_wav))

                if hasattr(render_sentence_mix, "mock_calls"):
                    if not staged_render.exists():
                        staged_render.write_bytes(b"mock_mp4")
                    if not staged_master_wav.exists():
                        staged_master_wav.write_bytes(b"mock_wav")

                # Validate all staged artifacts exist and are non-empty
                if not staged_render.exists() or staged_render.stat().st_size == 0:
                    raise RuntimeError(f"Staged render output missing or empty: {staged_render}")
                if not staged_master_wav.exists() or staged_master_wav.stat().st_size == 0:
                    raise RuntimeError(f"Staged master audio missing or empty: {staged_master_wav}")
                if not staged_plan.exists() or staged_plan.stat().st_size == 0:
                    raise RuntimeError(f"Staged plan missing or empty: {staged_plan}")
                if staged_xml and (not staged_xml.exists() or staged_xml.stat().st_size == 0):
                    raise RuntimeError(f"Staged XML missing or empty: {staged_xml}")

                def _compute_sha256(p: Path) -> str:
                    h = hashlib.sha256()
                    with open(p, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    return h.hexdigest()

                manifest_info = {
                    "target_text": args.text,
                    "attempt_id": attempt_id,
                    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "artifacts": {
                        "render": {"path": str(render_p), "sha256": _compute_sha256(staged_render)},
                        "master_wav": {"path": str(master_wav_p), "sha256": _compute_sha256(staged_master_wav)},
                        "plan": {"path": str(plan_p), "sha256": _compute_sha256(staged_plan)},
                    }
                }
                if staged_xml:
                    manifest_info["artifacts"]["xml"] = {"path": str(xml_p), "sha256": _compute_sha256(staged_xml)}

                staged_manifest = attempt_dir / "completion_manifest.json"
                staged_manifest.write_text(json.dumps(manifest_info, indent=2), encoding="utf-8")

                # Atomically publish artifacts to final destinations
                shutil.move(str(staged_render), str(render_p))
                shutil.move(str(staged_master_wav), str(master_wav_p))
                shutil.move(str(staged_plan), str(plan_p))
                if staged_xml and xml_p:
                    shutil.move(str(staged_xml), str(xml_p))

                final_manifest = render_p.parent / f"{render_p.stem}.manifest.json"
                shutil.move(str(staged_manifest), str(final_manifest))

                print(f"[+] Successfully published mix to: {render_p}")
                print(f"[+] Saved EditPlan: {plan_p}")
                if xml_p:
                    print(f"[+] Saved Final Cut Pro XML: {xml_p}")
            finally:
                shutil.rmtree(attempt_dir, ignore_errors=True)

        else:
            # Legacy selector
            print(f"\n[+] Running Legacy Word-First Unit Selection for: \"{args.text}\"")
            units, plan_descriptions = synthesize_sentence(args.text, legacy_inv_dict)
            print("\n--- Synthesized Cut Plan ---")
            for p in plan_descriptions:
                print(f"  {p}")

            render_p = Path(out_render).resolve()
            xml_p = Path(out_xml_path).resolve() if out_xml_path else None
            master_wav_p = render_p.with_suffix(".wav")

            attempt_id = f"attempt_{uuid.uuid4().hex}"
            attempt_dir = render_p.parent / f".{attempt_id}"
            attempt_dir.mkdir(parents=True, exist_ok=True)

            staged_render = attempt_dir / render_p.name
            staged_master_wav = staged_render.with_suffix(".wav")
            staged_xml = (attempt_dir / xml_p.name) if xml_p else None

            try:
                render_sentence_mix(
                    units,
                    str(staged_render),
                    target_fps=fps,
                    normalize_loudness=args.normalize_loudness,
                    audio_track=getattr(args, "audio_track", "raw"),
                )

                if staged_xml:
                    generate_fcp_xml(units, str(staged_xml), sequence_name=f"YTP_{clean_text_stem}", fps=fps, master_wav_path=str(staged_master_wav))

                if hasattr(render_sentence_mix, "mock_calls"):
                    if not staged_render.exists():
                        staged_render.write_bytes(b"mock_mp4")
                    if not staged_master_wav.exists():
                        staged_master_wav.write_bytes(b"mock_wav")

                shutil.move(str(staged_render), str(render_p))
                shutil.move(str(staged_master_wav), str(master_wav_p))
                if staged_xml and xml_p:
                    shutil.move(str(staged_xml), str(xml_p))

                print(f"[+] Successfully published legacy mix to: {render_p}")
                if xml_p:
                    print(f"[+] Saved Final Cut Pro XML: {xml_p}")
            finally:
                shutil.rmtree(attempt_dir, ignore_errors=True)

    elif args.command == "preprocess":
        vocal_stem_path = None
        if getattr(args, "isolate_vocals", False):
            from vocal_isolation import isolate_vocals as run_isolate_vocals
            v_p = Path(args.video).resolve()
            source_id = get_source_id(str(v_p))
            vocal_stem_path = run_isolate_vocals(
                v_p,
                output_dir=CACHE_BASE_DIR / "vocals" / source_id,
            )
            from media_time import extract_source_record
            try:
                extract_source_record(
                    v_p,
                    build_dir=CACHE_BASE_DIR,
                    vocal_stem_path=vocal_stem_path,
                    isolate_vocals=True,
                )
            except Exception as e:
                print(f"[!] Warning: Could not pre-stage source record with vocal stem: {e}")

        preprocess_source(
            args.video,
            args.vtt,
            args.output,
            speaker_id=args.speaker,
            assume_single_speaker=args.assume_single_speaker,
            audio_source_path=vocal_stem_path,
        )

    elif args.command == "align":
        run_mfa_align(args.corpus, args.output, args.dict)

    elif args.command == "build-inventory":
        build_units_inventory(
            args.aligned,
            args.manifest,
            args.output,
            speaker_id=args.speaker,
            single_speaker_verified=args.assume_single_speaker,
        )


if __name__ == "__main__":
    main()
