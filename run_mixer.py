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
from cache import (
    compute_media_hash,
    compute_alignment_cache_key,
    compute_labeled_inventory_cache_key,
    hash_file_content,
    hash_dict,
)


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
    """Generate a clean filesystem-safe source ID with a full content hash for caching."""
    v_p = Path(video_path).resolve()
    clean_stem = re.sub(r"[^\w\-_]", "_", v_p.stem)[:40]
    full_sha = compute_media_hash(v_p, cache_dir=CACHE_BASE_DIR)
    content_hash = full_sha[:8]
    return f"{clean_stem}_{content_hash}"


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
    force_inventory_rebuild: bool = False,
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
    alignment_base = CACHE_BASE_DIR / "alignment" / source_id
    corpus_dir = alignment_base / "corpus"
    aligned_dir = alignment_base / "aligned"
    manifest_path = corpus_dir / "corpus_manifest.json"
    alignment_meta_path = alignment_base / "alignment_metadata.json"

    inventory_base = CACHE_BASE_DIR / "inventory" / source_id
    inventory_path = inventory_base / "units_index.json"
    inventory_meta_path = inventory_base / "inventory_metadata.json"

    # Select best dictionary
    dict_path = SCRIPT_DIR / "dictionaries" / "merged_dictionary.dict"
    if not dict_path.exists():
        dict_path = SCRIPT_DIR / "dictionaries" / "custom_pronunciations.txt"

    # Resolve VTT
    vtt_candidate = vtt_path or find_associated_vtt(str(v_abs))
    if not vtt_candidate or not os.path.exists(vtt_candidate):
        raise FileNotFoundError(
            f"No matching .vtt file found for {v_abs}. Please specify --vtt explicitly."
        )

    # Compute cache fingerprints
    media_hash = compute_media_hash(v_abs, cache_dir=CACHE_BASE_DIR)
    vtt_hash = hash_file_content(vtt_candidate)
    dict_hash = hash_file_content(dict_path) if dict_path.exists() else ""
    acoustic_model = "english_us_arpa"
    mfa_options: Dict[str, Any] = {}
    preprocessor_version = "2.0"

    align_key = compute_alignment_cache_key(
        media_hash=media_hash,
        selected_audio_stream=0,
        selected_video_stream=0,
        transcript_hash=vtt_hash,
        dict_hash=dict_hash,
        acoustic_model=acoustic_model,
        mfa_options=mfa_options,
        preprocessor_version=preprocessor_version,
    )

    intervals_hash = hash_dict({"intervals": speaker_intervals}) if speaker_intervals else ""
    inv_key = compute_labeled_inventory_cache_key(
        alignment_key=align_key,
        speaker_id=speaker_id,
        single_speaker_verified=single_speaker_verified,
        assume_single_speaker=assume_single_speaker,
        speaker_intervals_hash=intervals_hash,
    )

    # Check Layer 1: Labeled Inventory Cache
    inv_hit = False
    if inventory_path.exists() and not force_realign and not force_inventory_rebuild:
        if inventory_meta_path.exists():
            try:
                with open(inventory_meta_path, "r", encoding="utf-8") as f:
                    inv_meta = json.load(f)
                if inv_meta.get("inventory_key") == inv_key:
                    inv_hit = True
            except Exception:
                inv_hit = False
        else:
            # Legacy cache: verify VTT mtime is older than inventory
            inv_mtime = os.path.getmtime(inventory_path)
            vtt_mtime = os.path.getmtime(vtt_candidate)
            if vtt_mtime <= inv_mtime:
                inv_hit = True

        if inv_hit and isolate_vocals:
            vocals_dir = CACHE_BASE_DIR / "vocals" / source_id
            stem_path = vocals_dir / "vocals.wav"
            if not stem_path.exists() or stem_path.stat().st_size == 0:
                inv_hit = False
            else:
                try:
                    with open(inventory_path, "r", encoding="utf-8") as f:
                        inv_data = json.load(f)
                    srcs = inv_data.get("sources", {})
                    if not any(s.get("cached_vocal_pcm_path") for s in srcs.values()):
                        inv_hit = False
                except Exception:
                    inv_hit = False

    if inv_hit:
        print(f"[+] Using cached labeled inventory: {inventory_path}")
        return str(inventory_path)

    # Check Layer 2: Alignment Cache
    align_hit = False
    if manifest_path.exists() and aligned_dir.exists() and alignment_meta_path.exists() and not force_realign:
        tg_files = list(aligned_dir.glob("*.TextGrid"))
        if tg_files:
            try:
                with open(alignment_meta_path, "r", encoding="utf-8") as f:
                    align_meta = json.load(f)
                if align_meta.get("alignment_key") == align_key:
                    align_hit = True
            except Exception:
                align_hit = False

        if align_hit and isolate_vocals:
            vocals_dir = CACHE_BASE_DIR / "vocals" / source_id
            stem_path = vocals_dir / "vocals.wav"
            if not stem_path.exists() or stem_path.stat().st_size == 0:
                align_hit = False

    vocal_stem_path: Optional[Path] = None
    source_rec: Optional[SourceRecord] = None

    if align_hit:
        print(f"[+] Alignment cache hit for {source_id}; reusing alignment and rebuilding labeled inventory...")
        if isolate_vocals:
            vocals_dir = CACHE_BASE_DIR / "vocals" / source_id
            vocal_stem_path = vocals_dir / "vocals.wav"
            from media_time import extract_source_record
            try:
                source_rec = extract_source_record(
                    v_abs,
                    build_dir=CACHE_BASE_DIR,
                    vocal_stem_path=vocal_stem_path,
                    isolate_vocals=True,
                )
            except Exception as e:
                print(f"[!] Warning: Could not pre-stage source record: {e}")

        inventory_base.mkdir(parents=True, exist_ok=True)
        build_units_inventory(
            str(aligned_dir),
            str(manifest_path),
            str(inventory_path),
            speaker_id=speaker_id,
            single_speaker_verified=single_speaker_verified or assume_single_speaker,
            source_record=source_rec,
            speaker_intervals=speaker_intervals,
        )

        with open(inventory_meta_path, "w", encoding="utf-8") as f:
            json.dump({
                "inventory_key": inv_key,
                "alignment_key": align_key,
                "speaker_id": speaker_id,
                "single_speaker_verified": single_speaker_verified,
                "assume_single_speaker": assume_single_speaker,
                "speaker_intervals_hash": intervals_hash,
            }, f, indent=2)

        return str(inventory_path)

    # Alignment miss or force_realign: rebuild both alignment and labeled inventory
    print(f"\n[+] Aligning source: {v_abs.name}")
    print(f"    Subtitles: {Path(vtt_candidate).name}")

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
    os.makedirs(inventory_base, exist_ok=True)

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
        str(vtt_candidate),
        str(corpus_dir),
        speaker_id=speaker_id,
        single_speaker_verified=single_speaker_verified,
        assume_single_speaker=assume_single_speaker,
        audio_source_path=vocal_stem_path,
    )

    # 2. MFA Alignment
    print("\n--- [2/3] Running Forced Alignment ---")
    run_mfa_align(str(corpus_dir), str(aligned_dir), str(dict_path), acoustic_model=acoustic_model)

    alignment_base.mkdir(parents=True, exist_ok=True)
    with open(alignment_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "alignment_key": align_key,
            "media_hash": media_hash,
            "transcript_hash": vtt_hash,
            "dict_hash": dict_hash,
            "acoustic_model": acoustic_model,
            "mfa_options": mfa_options,
            "preprocessor_version": preprocessor_version,
        }, f, indent=2)

    # 3. Build Inventory
    print("\n--- [3/3] Building Unit Inventory Index ---")
    build_units_inventory(
        str(aligned_dir),
        str(manifest_path),
        str(inventory_path),
        speaker_id=speaker_id,
        single_speaker_verified=single_speaker_verified or assume_single_speaker,
        source_record=source_rec,
        speaker_intervals=speaker_intervals,
    )

    with open(inventory_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "inventory_key": inv_key,
            "alignment_key": align_key,
            "speaker_id": speaker_id,
            "single_speaker_verified": single_speaker_verified,
            "assume_single_speaker": assume_single_speaker,
            "speaker_intervals_hash": intervals_hash,
        }, f, indent=2)

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

        if ("events" in data and ("schema_version" in data or "metadata" in data)):
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

        # Merge legacy views (support legacy_words/legacy_phones, words_inventory/phones_inventory, or words/phones)
        words_part = data.get("legacy_words") or data.get("words_inventory")
        if not words_part and isinstance(data.get("words"), dict):
            # check if it's a legacy dictionary where values are lists of instances
            first_val = next(iter(data["words"].values()), None)
            if isinstance(first_val, list):
                words_part = data["words"]
        phones_part = data.get("legacy_phones") or data.get("phones_inventory")
        if not phones_part and isinstance(data.get("phones"), dict):
            first_val = next(iter(data["phones"].values()), None)
            if isinstance(first_val, list):
                phones_part = data["phones"]

        if isinstance(words_part, dict):
            for w, instances in words_part.items():
                if isinstance(instances, list):
                    legacy_words.setdefault(w, []).extend(instances)
        if isinstance(phones_part, dict):
            for ph, instances in phones_part.items():
                if isinstance(instances, list):
                    legacy_phones.setdefault(ph, []).extend(instances)

    # Derive legacy views from InventoryIndex if legacy entries were not explicitly in file
    if not legacy_words and combined_inv.words:
        for wo in combined_inv.words.values():
            w_norm = wo.word_normalized
            src = combined_inv.sources.get(wo.source_content_id)
            src_path = src.source_path if src else ""
            sr = src.native_sample_rate if src and src.native_sample_rate else 16000
            legacy_words.setdefault(w_norm, []).append({
                "source_video": src_path,
                "chunk_id": wo.source_content_id,
                "label": w_norm,
                "start_sec": wo.start_sample / sr,
                "end_sec": wo.end_sample / sr,
                "duration": (wo.end_sample - wo.start_sample) / sr,
                "sample_start": wo.start_sample,
                "sample_end": wo.end_sample,
                "phones": wo.realized_phones,
                "speaker_id": wo.speaker_id,
                "is_eligible": getattr(wo, "is_eligible", True),
            })

    if not legacy_phones and combined_inv.events:
        for ev in combined_inv.events.values():
            if ev.kind == "phone":
                src = combined_inv.sources.get(ev.source_content_id)
                src_path = src.source_path if src else ""
                sr = src.native_sample_rate if src and src.native_sample_rate else 16000
                legacy_phones.setdefault(ev.label, []).append({
                    "source_video": src_path,
                    "chunk_id": ev.source_content_id,
                    "label": ev.label,
                    "start_sec": ev.start_sample / sr,
                    "end_sec": ev.end_sample / sr,
                    "duration": (ev.end_sample - ev.start_sample) / sr,
                    "sample_start": ev.start_sample,
                    "sample_end": ev.end_sample,
                    "speaker_id": ev.speaker_id,
                    "is_eligible": getattr(ev, "is_eligible", True),
                })

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
        p.add_argument("--text", default=None, help="Target text / sentence to synthesize")
        p.add_argument("--target-audio", default=None, help="Path to reference audio file for target pronunciation recognition")
        p.add_argument("--record-target", action="store_true", default=False, help="Record target pronunciation from microphone")
        p.add_argument("--diarize", choices=["auto", "off", "force"], default="auto", help="Diarization mode ('auto', 'off', 'force')")
        p.add_argument("--speaker-registry", default="config/speaker_registry.json", help="Path to speaker registry JSON")
        p.add_argument("--guide-transcript", default=None, help="Path to transcript file for prosody guide audio")
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
        # Check mutual exclusivity of target representations
        targets = [t for t in (getattr(args, "text", None), getattr(args, "target_audio", None), getattr(args, "record_target", False)) if t]
        if not targets:
            parser.error("One of --text, --target-audio, or --record-target is required.")
        if len(targets) > 1:
            parser.error("Options --text, --target-audio, and --record-target are mutually exclusive.")

        # Check cuts-only normalization conflict (A13.3)
        if args.mode == "cuts-only":
            if "--normalize-loudness" in sys.argv:
                parser.error("Explicit --normalize-loudness cannot be combined with --mode cuts-only.")
            args.normalize_loudness = False

        # Validate legacy selector combinations (T20 / A15 / B03)
        if args.selector == "legacy":
            if getattr(args, "guide", None):
                parser.error("Legacy selector does not support prosody guidance (--guide). Use --selector span.")
            if getattr(args, "features", "basic") == "wavlm":
                parser.error("Legacy selector does not support --features wavlm. Use --selector span.")
            if getattr(args, "plan", None):
                parser.error("Legacy selector does not generate EditPlan (--plan). Use --selector span.")
            if getattr(args, "target_audio", None) or getattr(args, "record_target", False):
                parser.error("Legacy selector does not support spoken audio targets (--target-audio, --record-target). Use --selector span.")
            if getattr(args, "text", None):
                if "<gap:" in args.text or "{" in args.text:
                    parser.error("Legacy selector does not support explicit gaps (<gap:...>) or phonetic syntax ({...}). Use --selector span.")

        # Validate guide arguments early (B10, U24)
        if getattr(args, "guide", None):
            guide_txt_path = getattr(args, "guide_transcript", None)
            if not guide_txt_path:
                parser.error("--guide requires --guide-transcript pointing to a transcript file matching target text.")
            gt_p = Path(guide_txt_path)
            if not gt_p.exists() or not gt_p.is_file():
                parser.error(f"--guide-transcript file not found: {guide_txt_path}")
            if getattr(args, "text", None):
                try:
                    guide_txt_content = gt_p.read_text(encoding="utf-8").strip()
                except Exception as e:
                    parser.error(f"Cannot read --guide-transcript file: {e}")
                from prosody_guide import ProsodyGuide
                try:
                    guide_obj = ProsodyGuide(args.guide, guide_transcript=guide_txt_content)
                    if not guide_obj.validate_transcript(args.text):
                        parser.error(f"--guide-transcript words do not match target text '{args.text}'.")
                except FileNotFoundError:
                    parser.error(f"Guide audio file not found: {args.guide}")

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
                            "speaker_id": args.speaker if args.assume_single_speaker else None,
                            "single_speaker_verified": args.assume_single_speaker,
                        })

        inventory_paths: List[str] = []

        if getattr(args, "inventory", None):
            for inv_item in args.inventory:
                for inv_p in inv_item.split(","):
                    clean = inv_p.strip()
                    if clean:
                        inventory_paths.append(str(Path(clean).resolve()))

        # Resolve speaker registry path against repository root
        registry_path = Path(args.speaker_registry)
        if not registry_path.is_absolute():
            registry_path = (SCRIPT_DIR / registry_path).resolve()
        else:
            registry_path = registry_path.resolve()

        # Align any unaligned sources
        audio_track = getattr(args, "audio_track", "raw")
        isolate_vocals_flag = getattr(args, "isolate_vocals", False) or (audio_track in ("isolated", "vocals", "recombined", "both", "all"))
        for item in source_items:
            is_asserted = bool(
                item.get("single_speaker_verified")
                or args.assume_single_speaker
                or item.get("speaker_intervals")
            )
            if not is_asserted:
                if args.diarize in ("auto", "force"):
                    from diarization import SpeakerDiarizer
                    from speaker_registry import SpeakerRegistry
                    backend_available = False
                    try:
                        from wavlm_backend import has_wavlm_dependencies
                        backend_available = has_wavlm_dependencies()
                    except Exception:
                        backend_available = False

                    if not backend_available:
                        if args.speaker is not None:
                            parser.error(
                                "Source is not asserted as single-speaker and diarization backend is unavailable.\n"
                                "To proceed with this speaker, either:\n"
                                "  1. Specify --assume-single-speaker if the video contains only this speaker.\n"
                                "  2. Provide verified speaker_intervals in a manifest.\n"
                                "  3. Install diarization backend dependencies."
                            )
                        else:
                            print("[!] Warning: Diarization backend unavailable and no speaker asserted. Proceeding with unverified identity; different voices may be mixed.")
                    else:
                        reg = SpeakerRegistry(config_path=registry_path)
                        diarizer = SpeakerDiarizer(registry=reg)
                        intervals = diarizer.diarize(
                            item["video"],
                            interactive=not getattr(args, "non_interactive", False),
                            content_id=get_source_id(item["video"]),
                        )
                        item["speaker_intervals"] = intervals

            inv_path = ensure_source_aligned(
                item["video"],
                item.get("vtt"),
                force_realign=args.force_realign,
                force_inventory_rebuild=(args.diarize == "force"),
                speaker_id=item.get("speaker_id") if item.get("single_speaker_verified") else None,
                single_speaker_verified=item.get("single_speaker_verified", False) or args.assume_single_speaker,
                assume_single_speaker=args.assume_single_speaker,
                speaker_intervals=item.get("speaker_intervals"),
                isolate_vocals=isolate_vocals_flag,
            )
            inventory_paths.append(inv_path)

        if not inventory_paths:
            parser.error("Please provide --manifest, --video, or --inventory.")

        # Load inventory
        inventory_index, legacy_inv_dict = load_or_combine_inventories(inventory_paths)

        # Collect all available speaker identities across words, events, and legacy views
        unique_speakers = sorted(list({
            wo.speaker_id for wo in inventory_index.words.values() if wo.speaker_id and getattr(wo, "is_eligible", True)
        } | {
            ev.speaker_id for ev in inventory_index.events.values() if ev.speaker_id and getattr(ev, "is_eligible", True)
        } | {
            inst.get("speaker_id")
            for insts in legacy_inv_dict.get("words", {}).values()
            for inst in insts
            if inst.get("speaker_id") and inst.get("is_eligible", True)
        } | {
            inst.get("speaker_id")
            for insts in legacy_inv_dict.get("phones", {}).values()
            for inst in insts
            if inst.get("speaker_id") and inst.get("is_eligible", True)
        }))

        if args.speaker is not None:
            if args.speaker not in unique_speakers:
                avail_str = ", ".join(sorted(unique_speakers)) if unique_speakers else "none"
                parser.error(
                    f"Requested speaker '{args.speaker}' not found in inventory. "
                    f"Available speakers: {avail_str}"
                )
            effective_speaker = args.speaker
        elif len(unique_speakers) > 1:
            parser.error(
                f"Multiple speakers found in inventory ({', '.join(unique_speakers)}). "
                "Please specify --speaker to avoid mixing different voices."
            )
        elif len(unique_speakers) == 1:
            effective_speaker = unique_speakers[0]
        else:
            effective_speaker = None

        pron_overrides: Dict[int, Any] = {}
        acoustic_target_meta: Optional[Dict[str, Any]] = None

        if getattr(args, "record_target", False):
            from phoneme_recognizer import PhonemeRecognizer
            is_interactive = sys.stdin.isatty() and not getattr(args, "non_interactive", False)
            rec = PhonemeRecognizer()
            print("[*] Recording target speech from microphone (2.5 seconds)...")
            audio = rec.record_audio_snippet(duration=2.5)
            structured_words = rec.recognize_structured_words(audio)
            if not structured_words:
                parser.error("Empty audio recognition result from microphone.")
            args.text = " ".join(w.word for w in structured_words)
            pron_overrides = {i: w.phones for i, w in enumerate(structured_words)}
            acoustic_target_meta = {
                "source": "microphone",
                "words": [{"word": w.word, "phones": w.phones, "tier": w.tier, "confidence": w.confidence} for w in structured_words],
            }
            print(f"[+] Recognized target text: {args.text}")
        elif getattr(args, "target_audio", None):
            from phoneme_recognizer import PhonemeRecognizer
            is_interactive = sys.stdin.isatty() and not getattr(args, "non_interactive", False)
            rec = PhonemeRecognizer()
            print(f"[*] Recognizing target pronunciation from: {args.target_audio}...")
            target_p = Path(args.target_audio).resolve()
            if not target_p.exists():
                parser.error(f"Target audio file not found: {args.target_audio}")
            import soundfile as sf
            data, sr = sf.read(str(target_p))
            structured_words = rec.recognize_structured_words(data, sample_rate=sr)
            if not structured_words:
                parser.error("Empty audio recognition result from target audio.")
            args.text = " ".join(w.word for w in structured_words)
            pron_overrides = {i: w.phones for i, w in enumerate(structured_words)}
            acoustic_target_meta = {
                "source": str(target_p),
                "words": [{"word": w.word, "phones": w.phones, "tier": w.tier, "confidence": w.confidence} for w in structured_words],
            }
            print(f"[+] Recognized target text: {args.text}")

        # Shared target preparation and numeral normalization
        from lexicon import normalize_numerals
        args.text = normalize_numerals(args.text)

        dict_files = [
            SCRIPT_DIR / "dictionaries" / "custom_pronunciations.txt",
            SCRIPT_DIR / "dictionaries" / "merged_dictionary.dict",
        ]
        lex = Lexicon([p for p in dict_files if p.exists()])

        from oov_assistant import OOVAssistant
        is_interactive = sys.stdin.isatty() and not getattr(args, "non_interactive", False)
        oov_assistant = OOVAssistant(lex, custom_dict_path=SCRIPT_DIR / "dictionaries" / "custom_pronunciations.txt")
        review_choices = []
        try:
            review_choices = oov_assistant.review_all_in_text(args.text, interactive=is_interactive)
            for ch in review_choices:
                if is_interactive or ch.source != "dictionary":
                    pron_overrides[ch.token_index] = ch.phones
        except RuntimeError as e:
            parser.error(str(e))

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

            graph_builder = TargetGraphBuilder(lex)
            graph = graph_builder.build(args.text, overrides=pron_overrides)

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
                guide_txt_path = getattr(args, "guide_transcript", None)
                if not guide_txt_path:
                    parser.error("--guide requires --guide-transcript pointing to a transcript file matching target text.")
                gt_p = Path(guide_txt_path)
                if not gt_p.exists() or not gt_p.is_file():
                    parser.error(f"--guide-transcript file not found: {guide_txt_path}")
                try:
                    guide_txt_content = gt_p.read_text(encoding="utf-8").strip()
                except Exception as e:
                    parser.error(f"Cannot read --guide-transcript file: {e}")
                prosody_guide = ProsodyGuide(args.guide, guide_transcript=guide_txt_content)
                if not prosody_guide.validate_transcript(args.text):
                    parser.error(f"--guide-transcript words do not match target text '{args.text}'.")

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
            if acoustic_target_meta:
                plan.metadata["acoustic_target"] = acoustic_target_meta
            if review_choices:
                plan.metadata["pronunciation_review"] = [
                    {"token_index": c.token_index, "text": c.text, "phones": c.phones, "source": c.source, "scope": c.scope}
                    for c in review_choices
                ]

            from edit_plan import finalize_edit_plan
            finalize_edit_plan(plan, inventory_index, normalize_loudness=args.normalize_loudness)

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
                    if not staged_master_wav.exists() or staged_master_wav.stat().st_size == 0:
                        raise RuntimeError(f"Staged master audio missing or empty: {staged_master_wav}")
                    # Point XML to the finalized published master WAV path, inspecting the staged master WAV
                    generate_fcp_xml(
                        plan,
                        str(staged_xml),
                        sequence_name=f"YTP_{clean_text_stem}",
                        master_wav_path=str(master_wav_p),
                        inspect_wav_path=str(staged_master_wav),
                    )

                # Validate required primary staged artifacts exist and are non-empty
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

                def _artifact_key(staged_file: Path) -> str:
                    stem = staged_file.stem
                    suffix = staged_file.suffix.lower()
                    if "vocals" in stem:
                        return "vocals_render" if suffix == ".mp4" else "vocals_wav"
                    if "recombined" in stem:
                        return "recombined_render" if suffix == ".mp4" else "recombined_wav"
                    clean_ext = suffix.lstrip(".")
                    return f"{stem}_{clean_ext}" if clean_ext else stem

                # Collect all staged delivery files (including multi-variant tracks)
                file_mappings: Dict[Path, Path] = {
                    staged_render: render_p,
                    staged_master_wav: master_wav_p,
                    staged_plan: plan_p,
                }
                if staged_xml and xml_p:
                    file_mappings[staged_xml] = xml_p

                # Add all variant deliverables (_vocals.mp4, _vocals.wav, _recombined.mp4, etc.)
                for staged_file in attempt_dir.iterdir():
                    if staged_file.is_file() and staged_file not in file_mappings:
                        if not staged_file.name.endswith(".tmp") and staged_file.name != "completion_manifest.json":
                            dest_file = render_p.parent / staged_file.name
                            file_mappings[staged_file] = dest_file

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

                for sf_item, df_item in file_mappings.items():
                    if sf_item not in (staged_render, staged_master_wav, staged_plan, staged_xml):
                        key = _artifact_key(sf_item)
                        if key in manifest_info["artifacts"]:
                            raise ValueError(
                                f"Duplicate manifest artifact key '{key}' for file '{sf_item}' and '{manifest_info['artifacts'][key]['path']}'"
                            )
                        manifest_info["artifacts"][key] = {"path": str(df_item), "sha256": _compute_sha256(sf_item)}

                staged_manifest = attempt_dir / "completion_manifest.json"
                staged_manifest.write_text(json.dumps(manifest_info, indent=2), encoding="utf-8")
                final_manifest = render_p.parent / f"{render_p.stem}.manifest.json"

                # Transactional Publication with backup and rollback
                backup_dir = render_p.parent / f".backup_{attempt_id}"
                backed_up: List[Tuple[Path, Path]] = []
                published: List[Path] = []
                try:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    for sf_item, df_item in file_mappings.items():
                        if df_item.exists():
                            b_copy = backup_dir / df_item.name
                            shutil.copy2(str(df_item), str(b_copy))
                            backed_up.append((df_item, b_copy))
                    if final_manifest.exists():
                        b_man = backup_dir / final_manifest.name
                        shutil.copy2(str(final_manifest), str(b_man))
                        backed_up.append((final_manifest, b_man))

                    for sf_item, df_item in file_mappings.items():
                        df_item.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(sf_item), str(df_item))
                        published.append(df_item)

                    shutil.move(str(staged_manifest), str(final_manifest))
                    published.append(final_manifest)

                    # Success: clean up backup
                    shutil.rmtree(backup_dir, ignore_errors=True)
                except Exception as pub_err:
                    print(f"[!] Publication error ({pub_err}), rolling back to previous generation...", file=sys.stderr)
                    for p_item in published:
                        if p_item.exists():
                            try:
                                p_item.unlink()
                            except OSError:
                                pass
                    for d_orig, b_saved in backed_up:
                        if b_saved.exists():
                            shutil.move(str(b_saved), str(d_orig))
                    shutil.rmtree(backup_dir, ignore_errors=True)
                    raise

                print(f"[+] Successfully published mix to: {render_p}")
                print(f"[+] Saved EditPlan: {plan_p}")
                if xml_p:
                    print(f"[+] Saved Final Cut Pro XML: {xml_p}")
            finally:
                shutil.rmtree(attempt_dir, ignore_errors=True)

        else:
            # Legacy selector
            print(f"\n[+] Running Legacy Word-First Unit Selection for: \"{args.text}\"")
            legacy_units_dict = legacy_inv_dict
            if effective_speaker is not None:
                filtered_w = {
                    w: [inst for inst in insts if inst.get("speaker_id") == effective_speaker and inst.get("is_eligible", True)]
                    for w, insts in legacy_inv_dict.get("words", {}).items()
                }
                filtered_w = {w: insts for w, insts in filtered_w.items() if insts}
                filtered_p = {
                    ph: [inst for inst in insts if inst.get("speaker_id") == effective_speaker and inst.get("is_eligible", True)]
                    for ph, insts in legacy_inv_dict.get("phones", {}).items()
                }
                filtered_p = {ph: insts for ph, insts in filtered_p.items() if insts}
                legacy_units_dict = {
                    "words": filtered_w,
                    "words_inventory": filtered_w,
                    "phones": filtered_p,
                    "phones_inventory": filtered_p,
                }

            units, plan_descriptions = synthesize_sentence(args.text, legacy_units_dict, lexicon=lex, overrides=pron_overrides)
            if not units:
                raise RuntimeError(f"Legacy selector could not find any unit candidates for '{args.text}'")
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
                    if not staged_master_wav.exists() or staged_master_wav.stat().st_size == 0:
                        raise RuntimeError(f"Staged master audio missing or empty: {staged_master_wav}")
                    generate_fcp_xml(
                        units,
                        str(staged_xml),
                        sequence_name=f"YTP_{clean_text_stem}",
                        fps=fps,
                        master_wav_path=str(master_wav_p),
                        inspect_wav_path=str(staged_master_wav),
                    )

                def _compute_sha256(p: Path) -> str:
                    h = hashlib.sha256()
                    with open(p, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    return h.hexdigest()

                def _artifact_key(staged_file: Path) -> str:
                    stem = staged_file.stem
                    suffix = staged_file.suffix.lower()
                    if "vocals" in stem:
                        return "vocals_render" if suffix == ".mp4" else "vocals_wav"
                    if "recombined" in stem:
                        return "recombined_render" if suffix == ".mp4" else "recombined_wav"
                    clean_ext = suffix.lstrip(".")
                    return f"{stem}_{clean_ext}" if clean_ext else stem

                file_mappings = {
                    staged_render: render_p,
                    staged_master_wav: master_wav_p,
                }
                if staged_xml and xml_p:
                    file_mappings[staged_xml] = xml_p

                for staged_file in attempt_dir.iterdir():
                    if staged_file.is_file() and staged_file not in file_mappings:
                        if not staged_file.name.endswith(".tmp") and staged_file.name != "completion_manifest.json":
                            dest_file = render_p.parent / staged_file.name
                            file_mappings[staged_file] = dest_file

                manifest_info = {
                    "target_text": args.text,
                    "attempt_id": attempt_id,
                    "completed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                    "artifacts": {
                        "render": {"path": str(render_p), "sha256": _compute_sha256(staged_render)},
                        "master_wav": {"path": str(master_wav_p), "sha256": _compute_sha256(staged_master_wav)},
                    }
                }
                if staged_xml:
                    manifest_info["artifacts"]["xml"] = {"path": str(xml_p), "sha256": _compute_sha256(staged_xml)}

                for sf_item, df_item in file_mappings.items():
                    if sf_item not in (staged_render, staged_master_wav, staged_xml):
                        key = _artifact_key(sf_item)
                        if key in manifest_info["artifacts"]:
                            raise ValueError(
                                f"Duplicate manifest artifact key '{key}' for file '{sf_item}' and '{manifest_info['artifacts'][key]['path']}'"
                            )
                        manifest_info["artifacts"][key] = {"path": str(df_item), "sha256": _compute_sha256(sf_item)}

                staged_manifest = attempt_dir / "completion_manifest.json"
                staged_manifest.write_text(json.dumps(manifest_info, indent=2), encoding="utf-8")
                final_manifest = render_p.parent / f"{render_p.stem}.manifest.json"

                backup_dir = render_p.parent / f".backup_{attempt_id}"
                backed_up = []
                published = []
                try:
                    backup_dir.mkdir(parents=True, exist_ok=True)
                    for sf_item, df_item in file_mappings.items():
                        if df_item.exists():
                            b_copy = backup_dir / df_item.name
                            shutil.copy2(str(df_item), str(b_copy))
                            backed_up.append((df_item, b_copy))
                    if final_manifest.exists():
                        b_man = backup_dir / final_manifest.name
                        shutil.copy2(str(final_manifest), str(b_man))
                        backed_up.append((final_manifest, b_man))

                    for sf_item, df_item in file_mappings.items():
                        df_item.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(sf_item), str(df_item))
                        published.append(df_item)

                    shutil.move(str(staged_manifest), str(final_manifest))
                    published.append(final_manifest)

                    shutil.rmtree(backup_dir, ignore_errors=True)
                except Exception as pub_err:
                    print(f"[!] Legacy publication error ({pub_err}), rolling back...", file=sys.stderr)
                    for p_item in published:
                        if p_item.exists():
                            try:
                                p_item.unlink()
                            except OSError:
                                pass
                    for d_orig, b_saved in backed_up:
                        if b_saved.exists():
                            shutil.move(str(b_saved), str(d_orig))
                    shutil.rmtree(backup_dir, ignore_errors=True)
                    raise

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
