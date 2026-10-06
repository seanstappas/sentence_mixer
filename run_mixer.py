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
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

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
from subtitles import burn_subtitles_to_video, export_subtitles
from cache import (
    compute_media_hash,
    compute_alignment_cache_key,
    compute_labeled_inventory_cache_key,
    hash_file_content,
    hash_dict,
)


def format_file_url(p: Union[str, Path, None]) -> str:
    """Format path as clickable file:/// URL."""
    if not p:
        return ""
    path = Path(p).resolve()
    try:
        return path.as_uri()
    except Exception:
        as_posix = path.as_posix()
        if not as_posix.startswith("/"):
            as_posix = "/" + as_posix
        return f"file://{as_posix}"



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


MOVIE_MFA_BATCH_SIZE = 400
_MOVIE_SOURCE_CACHE: Dict[str, bool] = {}


def is_movie_source(video_path: Union[str, Path]) -> bool:
    """True for long or surround sources that use the block-cached movie pipeline."""
    p = Path(video_path).resolve()
    key = str(p)
    if key not in _MOVIE_SOURCE_CACHE:
        from movie_media import is_long_source
        _MOVIE_SOURCE_CACHE[key] = p.exists() and is_long_source(p)
    return _MOVIE_SOURCE_CACHE[key]


def print_movie_ingest_summary(result: Dict[str, Any]) -> None:
    voice = result.get("voice") or {}
    counts = result.get("status_counts") or {}
    print(
        f"[+] Movie ready: {result['blocks_aligned'].__len__()}/{result['blocks_total']} blocks aligned "
        f"({result['blocks_aligned_this_run']} this run), stems {'complete' if result['stems_complete'] else 'not available'}, "
        f"{result['seconds']:.0f}s"
    )
    if counts:
        print("    line labels: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    if voice and "skipped" not in voice:
        print(
            f"    voice checks: leave-one-out accuracy {voice.get('loo_accuracy')}, "
            f"{voice.get('voice_mismatch')} labeled lines flagged, {voice.get('propagated')} unlabeled lines labeled by voice"
        )
    top = list((result.get("speaker_seconds") or {}).items())[:8]
    if top:
        print("    usable speech: " + ", ".join(f"{k} {v:.0f}s" for k, v in top))


def load_movie_scenes(video_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Scenes exported by tools/label_subtitles.py into the labeled subtitle report."""
    stem = Path(video_path).stem
    for name in (f"{stem}.en.vtt.report.json", f"{stem}.vtt.report.json"):
        rp = CACHE_BASE_DIR / "subtitles" / "labeled" / name
        if rp.exists():
            try:
                return json.loads(rp.read_text(encoding="utf-8")).get("scenes", []) or []
            except Exception:
                return []
    return []


def resolve_time_ranges(
    range_args: Optional[List[str]],
    scene_args: Optional[List[str]],
    video_paths: List[str],
) -> List[Tuple[float, float]]:
    """Combines --range values and --scene selections (number or heading keyword) into ranges."""
    from movie_ingest import parse_ranges
    ranges = parse_ranges(range_args)
    for sel in scene_args or []:
        found = False
        key = str(sel).strip()
        for vp in video_paths:
            scenes = load_movie_scenes(vp)
            for sc in scenes:
                # Numbers select by scene index only (screenplay headings often contain numbers)
                if key.isdigit():
                    hit = str(sc.get("index")) == key
                else:
                    hit = key.lower() in str(sc.get("heading", "")).lower()
                if hit:
                    ranges.append((float(sc["start"]), float(sc["end"])))
                    found = True
            if found:
                break
        if not found:
            raise ValueError(
                f"Scene '{sel}' not found. List scenes with: run_mixer.py scenes <movie>"
            )
    return sorted(ranges)


def pick_timeline_fps(inventory: InventoryIndex, default: RationalTime) -> RationalTime:
    """Most common video frame rate among the inventory's sources (so movies keep 23.976 fps)."""
    counts: Dict[Tuple[int, int], int] = {}
    for src in inventory.sources.values():
        vf = getattr(src, "video_fps", None) or {}
        num, den = int(vf.get("numerator", 0) or 0), int(vf.get("denominator", 0) or 0)
        if num > 0 and den > 0:
            counts[(num, den)] = counts.get((num, den), 0) + 1
    if not counts:
        return default
    num, den = max(counts.items(), key=lambda kv: kv[1])[0]
    return RationalTime(num, den)


def restrict_inventory_to_ranges(
    inventory: InventoryIndex,
    ranges: List[Tuple[float, float]],
) -> int:
    """Marks words/events outside the time ranges ineligible. Returns eligible word count."""
    kept = 0
    for wo in inventory.words.values():
        src = inventory.sources.get(wo.source_content_id)
        sr = src.native_sample_rate if src and src.native_sample_rate else 48000
        s, e = wo.start_sample / sr, wo.end_sample / sr
        inside = any(rs <= s and e <= re_ for rs, re_ in ranges)
        if not inside and wo.is_eligible:
            wo.is_eligible = False
            wo.eligibility_reason = "outside_range"
            for ev_id in wo.phone_event_ids:
                ev = inventory.events.get(ev_id)
                if ev is not None:
                    ev.is_eligible = False
                    ev.eligibility_reason = "outside_range"
        elif inside and wo.is_eligible:
            kept += 1
    for ev in inventory.events.values():
        if ev.kind != "phone" or not ev.is_eligible:
            continue
        src = inventory.sources.get(ev.source_content_id)
        sr = src.native_sample_rate if src and src.native_sample_rate else 48000
        if not any(rs <= ev.start_sample / sr and ev.end_sample / sr <= re_ for rs, re_ in ranges):
            ev.is_eligible = False
            ev.eligibility_reason = "outside_range"
    return kept


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
    """Locate the corresponding subtitle (.vtt or .srt) file alongside the video if not explicitly provided."""
    from source_discovery import find_associated_subtitles
    sub = find_associated_subtitles(video_path)
    return sub or ""


def run_mfa_align(
    corpus_dir: str,
    output_aligned_dir: str,
    dictionary_path: Optional[str] = None,
    acoustic_model: str = "english_us_arpa",
    batch_size: int = 45,
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
    # Prevent OpenBLAS/OpenMP thread exhaustion/access violations (error 1073741819 / 0xC0000005)
    # on high-core-count Windows CPUs (e.g. 24/32 threads)
    env["OMP_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["VECLIB_MAXIMUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"

    # Prefer direct invocation of mfa executable inside mfa_env with required DLL PATHs,
    # avoiding micromamba.exe launcher crash bugs (0xC0000005 / error 1073741819) on Windows.
    direct_mfa_exe: Optional[Path] = None
    if mamba_root:
        env_dir = Path(mamba_root) / "envs" / "mfa_env"
        if env_dir.exists():
            candidate_mfa = env_dir / ("Scripts/mfa.exe" if sys.platform == "win32" else "bin/mfa")
            if candidate_mfa.exists():
                direct_mfa_exe = candidate_mfa
                extra_paths = [
                    str(env_dir),
                    str(env_dir / "Library" / "mingw-w64" / "bin"),
                    str(env_dir / "Library" / "usr" / "bin"),
                    str(env_dir / "Library" / "bin"),
                    str(env_dir / "Scripts"),
                    str(env_dir / "bin"),
                ]
                env["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + env.get("PATH", "")

    c_path = Path(corpus_dir).resolve()
    out_aligned = Path(output_aligned_dir).resolve()
    wav_files = sorted(list(c_path.rglob("chunk_*.wav")))

    def _execute_mfa(src_dir: Path, dst_dir: Path):
        mfa_temp_dir = mfa_root_dir / "temp"
        mfa_temp_dir.mkdir(parents=True, exist_ok=True)
        target_temp = mfa_temp_dir / src_dir.name

        # Pre-clean stale DB / WAL / SHM files left behind from any interrupted run
        # to prevent SQLAlchemy 'identity_token' NoneType errors on Windows.
        for stale_path in [
            target_temp,
            mfa_root_dir / src_dir.name,
            Path.home() / "Documents" / "MFA" / src_dir.name,
        ]:
            if stale_path.exists():
                shutil.rmtree(stale_path, ignore_errors=True)

        base_cmd = [
            "mfa", "align",
            str(src_dir),
            str(Path(dict_arg).resolve()) if os.path.exists(dict_arg) else dict_arg,
            acoustic_arg,
            str(dst_dir),
            "-t", str(mfa_temp_dir),
            "--clean",
            "-j", "1",  # Windows stability
        ]

        if direct_mfa_exe:
            cmd = [str(direct_mfa_exe)] + base_cmd[1:]
        elif mm_exe and mamba_root:
            cmd = [mm_exe, "run", "-r", mamba_root, "-n", "mfa_env"] + base_cmd
        else:
            cmd = base_cmd

        print("[*] Running MFA Alignment command:")
        print("    " + " ".join(cmd))
        res = subprocess.run(cmd, env=env)
        if res.returncode != 0:
            print("[!] MFA alignment returned non-zero code. Purging temp cache and retrying with clean state...")
            time.sleep(1.0)
            for stale_path in [
                target_temp,
                mfa_root_dir / src_dir.name,
                Path.home() / "Documents" / "MFA" / src_dir.name,
            ]:
                if stale_path.exists():
                    for _ in range(5):
                        try:
                            shutil.rmtree(stale_path)
                            break
                        except OSError:
                            time.sleep(0.5)

            # Use a completely fresh isolated temp directory on retry to ensure no corrupted SQLite state is reused
            retry_temp_dir = mfa_temp_dir / f"retry_{int(time.time())}"
            retry_temp_dir.mkdir(parents=True, exist_ok=True)
            retry_base_cmd = list(base_cmd)
            if "-t" in retry_base_cmd:
                t_idx = retry_base_cmd.index("-t")
                retry_base_cmd[t_idx + 1] = str(retry_temp_dir)
            if direct_mfa_exe:
                retry_cmd = [str(direct_mfa_exe)] + retry_base_cmd[1:]
            elif mm_exe and mamba_root:
                retry_cmd = (
                    [mm_exe, "run", "-r", mamba_root, "-n", "mfa_env"] + retry_base_cmd
                )
            else:
                retry_cmd = retry_base_cmd
            res = subprocess.run(retry_cmd, env=env)
            if retry_temp_dir.exists():
                shutil.rmtree(retry_temp_dir, ignore_errors=True)
            if res.returncode != 0:
                raise RuntimeError(f"MFA alignment failed with exit code {res.returncode}. Check MFA logs.")

        if target_temp.exists():
            shutil.rmtree(target_temp, ignore_errors=True)

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
    cache_status: Optional[Dict[str, Any]] = None,
    asr_fill_tails: bool = False,
    ranges: Optional[List[Tuple[float, float]]] = None,
    subtitle_stream: Optional[str] = None,
    characters_path: Optional[str] = None,
    voice_checks: bool = True,
) -> str:
    """
    Ensures a video source is preprocessed, aligned with MFA, and indexed into an inventory.
    Returns path to units_index.json.
    asr_fill_tails enables opt-in Whisper tail enrichment of subtitle text before alignment.
    Long sources (whole movies) use the block-cached movie pipeline (src/movie_ingest.py);
    ranges limits which blocks are processed, and labels come from subtitles, scripts and
    voice checks instead of diarization.
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

    # Stage MFA dictionary combining merged_dictionary.dict and custom_pronunciations.txt
    from lexicon import stage_mfa_dictionary
    staged_dict_dir = CACHE_BASE_DIR / "mfa"
    dict_path, dict_hash = stage_mfa_dictionary(
        base_dict_path=SCRIPT_DIR / "dictionaries" / "merged_dictionary.dict",
        custom_dict_path=SCRIPT_DIR / "dictionaries" / "custom_pronunciations.txt",
        output_path=staged_dict_dir / "staged_dictionary.dict",
    )

    if is_movie_source(v_abs):
        from movie_ingest import ensure_movie_aligned
        result = ensure_movie_aligned(
            v_abs,
            source_id=source_id,
            build_dir=CACHE_BASE_DIR,
            run_mfa=lambda c, a: run_mfa_align(c, a, str(dict_path), acoustic_model="english_us_arpa", batch_size=MOVIE_MFA_BATCH_SIZE),
            dict_hash=dict_hash,
            subtitle_path=vtt_path,
            subtitle_stream=subtitle_stream,
            characters_path=characters_path,
            ranges=ranges,
            separate_stems=isolate_vocals,
            voice_checks=voice_checks,
            force_realign=force_realign,
        )
        print_movie_ingest_summary(result)
        if cache_status is not None:
            cache_status["hit"] = bool(result["cache_hit"])
            cache_status["realigned"] = result["blocks_aligned_this_run"] > 0
            cache_status["layer"] = 1 if result["cache_hit"] else 0
            cache_status["inventory_path"] = result["inventory_path"]
            cache_status["movie"] = result
        return str(result["inventory_path"])

    # Resolve VTT
    vtt_candidate = vtt_path or find_associated_vtt(str(v_abs))
    if not vtt_candidate or not os.path.exists(vtt_candidate):
        raise FileNotFoundError(
            f"No matching .vtt file found for {v_abs}. Please specify --vtt explicitly."
        )

    # Compute cache fingerprints
    media_hash = compute_media_hash(v_abs, cache_dir=CACHE_BASE_DIR)
    vtt_hash = hash_file_content(vtt_candidate)
    acoustic_model = "english_us_arpa"
    # The enrichment setting changes the MFA transcript, so it is part of the alignment key.
    mfa_options: Dict[str, Any] = {"asr_tail_enrichment": bool(asr_fill_tails)}
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

    intervals_serializable = (
        [iv.to_dict() if hasattr(iv, "to_dict") else iv for iv in speaker_intervals]
        if speaker_intervals
        else None
    )
    intervals_hash = hash_dict({"intervals": intervals_serializable}) if intervals_serializable else ""
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
        print(f"[+] Using cached labeled inventory: {format_file_url(inventory_path)}")
        if cache_status is not None:
            cache_status["hit"] = True
            cache_status["realigned"] = False
            cache_status["layer"] = 1
            cache_status["inventory_path"] = str(inventory_path)
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

        if cache_status is not None:
            cache_status["hit"] = True
            cache_status["realigned"] = False
            cache_status["layer"] = 2
            cache_status["inventory_path"] = str(inventory_path)
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
            pre_extract_audio=True,
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
        asr_tail_enrichment=asr_fill_tails,
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

    if cache_status is not None:
        cache_status["hit"] = False
        cache_status["realigned"] = True
        cache_status["layer"] = 0
        cache_status["inventory_path"] = str(inventory_path)
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

    Supports:
      - Direct directory paths (invoking discover_folder_sources).
      - Manifest entries with {"folder": "relative/path", ...} expanding folder sources
        and inheriting explicit speaker attributes.
      - Standard {"video": "...", "vtt": "..."} entries and string video paths.
    """
    m_path = Path(manifest_path).resolve()
    if not m_path.exists():
        raise FileNotFoundError(f"Manifest file not found: {m_path}")

    if m_path.is_dir():
        from source_discovery import discover_folder_sources
        return discover_folder_sources(m_path)

    with open(m_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Manifest must be a JSON list of source items")

    parsed_items: List[Dict[str, Any]] = []
    for item in data:
        if isinstance(item, dict) and "folder" in item:
            from source_discovery import discover_folder_sources
            folder_p = (m_path.parent / item["folder"]).resolve()
            folder_items = discover_folder_sources(folder_p)
            for f_item in folder_items:
                parsed_items.append({
                    "video": f_item["video"],
                    "vtt": f_item.get("vtt"),
                    "speaker_id": item.get("speaker_id", f_item.get("speaker_id")),
                    "single_speaker_verified": item.get(
                        "single_speaker_verified", f_item.get("single_speaker_verified", False)
                    ),
                    "recording_id": item.get("recording_id", f_item.get("recording_id", "")),
                    "speaker_intervals": item.get("speaker_intervals", f_item.get("speaker_intervals")),
                })
        elif isinstance(item, dict) and "video" in item:
            resolved_video = (m_path.parent / item["video"]).resolve()
            if item.get("vtt"):
                resolved_vtt = (m_path.parent / item["vtt"]).resolve()
            else:
                from source_discovery import find_associated_subtitles
                sub_str = find_associated_subtitles(resolved_video)
                resolved_vtt = Path(sub_str).resolve() if sub_str else None
            parsed_items.append({
                "video": str(resolved_video),
                "vtt": str(resolved_vtt) if resolved_vtt else None,
                "speaker_id": item.get("speaker_id"),
                "single_speaker_verified": item.get("single_speaker_verified", False),
                "recording_id": item.get("recording_id", ""),
                "speaker_intervals": item.get("speaker_intervals"),
            })
        elif isinstance(item, str):
            resolved_target = (m_path.parent / item).resolve()
            if resolved_target.is_dir():
                from source_discovery import discover_folder_sources
                folder_items = discover_folder_sources(resolved_target)
                parsed_items.extend(folder_items)
            else:
                from source_discovery import find_associated_subtitles
                sub_match = find_associated_subtitles(resolved_target)
                parsed_items.append({
                    "video": str(resolved_target),
                    "vtt": sub_match,
                    "speaker_id": None,
                    "single_speaker_verified": False,
                    "recording_id": "",
                    "speaker_intervals": None,
                })
    return parsed_items


def collect_source_items(args: argparse.Namespace) -> List[Dict[str, Any]]:
    """Collect and resolve source items from positional source arguments, --folder, --manifest, and --video CLI arguments."""
    source_items: List[Dict[str, Any]] = []

    if getattr(args, "source", None):
        s_list = args.source if isinstance(args.source, list) else [args.source]
        for s_entry in s_list:
            for s_item in s_entry.split(","):
                s_clean = s_item.strip()
                if not s_clean:
                    continue
                s_path = Path(s_clean).resolve()
                if not s_path.exists():
                    raise FileNotFoundError(f"Source path not found: {s_path}")
                if s_path.is_dir():
                    from source_discovery import discover_folder_sources
                    discovered = discover_folder_sources(s_path)
                    for d_item in discovered:
                        if getattr(args, "speaker", None) and getattr(args, "assume_single_speaker", False):
                            d_item["speaker_id"] = args.speaker
                            d_item["single_speaker_verified"] = True
                        elif getattr(args, "assume_single_speaker", False):
                            d_item["single_speaker_verified"] = True
                        source_items.append(d_item)
                elif s_path.suffix.lower() == ".json":
                    source_items.extend(parse_manifest_items(str(s_path)))
                else:
                    from source_discovery import find_associated_subtitles
                    vtt_match = find_associated_subtitles(s_path)
                    source_items.append({
                        "video": str(s_path),
                        "vtt": vtt_match,
                        "speaker_id": getattr(args, "speaker", None) if getattr(args, "assume_single_speaker", False) else None,
                        "single_speaker_verified": getattr(args, "assume_single_speaker", False),
                    })

    if getattr(args, "folder", None):
        f_list = args.folder if isinstance(args.folder, list) else [args.folder]
        for f_entry in f_list:
            for f_item in f_entry.split(","):
                f_clean = f_item.strip()
                if f_clean:
                    from source_discovery import discover_folder_sources
                    discovered = discover_folder_sources(Path(f_clean).resolve())
                    for d_item in discovered:
                        if getattr(args, "speaker", None) and getattr(args, "assume_single_speaker", False):
                            d_item["speaker_id"] = args.speaker
                            d_item["single_speaker_verified"] = True
                        elif getattr(args, "assume_single_speaker", False):
                            d_item["single_speaker_verified"] = True
                        source_items.append(d_item)

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
                    if not vtt_match:
                        from source_discovery import find_associated_subtitles
                        vtt_match = find_associated_subtitles(Path(v_clean).resolve())
                    source_items.append({
                        "video": str(Path(v_clean).resolve()),
                        "vtt": str(Path(vtt_match).resolve()) if vtt_match else None,
                        "speaker_id": getattr(args, "speaker", None) if getattr(args, "assume_single_speaker", False) else None,
                        "single_speaker_verified": getattr(args, "assume_single_speaker", False),
                    })

    # Deduplicate items by resolved video path preserving order
    seen_videos: Set[str] = set()
    deduped_items: List[Dict[str, Any]] = []
    for item in source_items:
        v_res = str(Path(item["video"]).resolve())
        if v_res not in seen_videos:
            seen_videos.add(v_res)
            deduped_items.append(item)

    return deduped_items


def ingest_sources(
    items: List[Dict[str, Any]],
    args: Optional[argparse.Namespace] = None,
    cache_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Ingest a list of source items: runs vocal isolation (if requested),
    diarization / clustering (if enabled and needed), MFA alignment, and inventory indexing.
    Tracks per-clip metrics (cache hits vs new alignments) and returns an aggregation dict.
    """
    if cache_dir is not None:
        global CACHE_BASE_DIR
        CACHE_BASE_DIR = Path(cache_dir).resolve()

    if not items:
        return {
            "total_sources": 0,
            "processed_clips": 0,
            "cache_hits": 0,
            "new_alignments": 0,
            "inventory_paths": [],
            "combined_inventory": InventoryIndex(),
            "legacy_dict": {"words": {}, "phones": {}},
            "speaker_stats": {},
        }

    if args is None:
        args = argparse.Namespace()

    registry_path_str = getattr(args, "speaker_registry", "config/speaker_registry.json")
    registry_path = Path(registry_path_str)
    if not registry_path.is_absolute():
        registry_path = (SCRIPT_DIR / registry_path).resolve()
    else:
        registry_path = registry_path.resolve()

    audio_track = getattr(args, "audio_track", "raw")
    isolate_vocals_flag = bool(
        getattr(args, "isolate_vocals", False)
        or (audio_track in ("isolated", "vocals", "recombined", "both", "all"))
    )

    force_realign = bool(getattr(args, "force_realign", False))
    diarize_mode = getattr(args, "diarize", "auto")
    assume_single = bool(getattr(args, "assume_single_speaker", False))
    cli_speaker = getattr(args, "speaker", None)
    non_interactive = bool(getattr(args, "non_interactive", False))

    cache_hits = 0
    new_alignments = 0
    inventory_paths: List[str] = []

    for item in items:
        v_abs = Path(item["video"]).resolve()
        source_id = get_source_id(str(v_abs)) if v_abs.exists() else Path(item["video"]).stem

        is_asserted = bool(
            item.get("single_speaker_verified")
            or assume_single
            or item.get("speaker_intervals")
            or (v_abs.exists() and is_movie_source(v_abs))
        )
        if not is_asserted:
            if diarize_mode in ("auto", "force"):
                backend_available = False
                try:
                    from wavlm_backend import has_wavlm_dependencies
                    backend_available = has_wavlm_dependencies()
                except Exception:
                    backend_available = False

                if not backend_available:
                    if cli_speaker is not None:
                        raise ValueError(
                            "Source is not asserted as single-speaker and diarization backend is unavailable.\n"
                            "To proceed with this speaker, either:\n"
                            "  1. Specify --assume-single-speaker if the video contains only this speaker.\n"
                            "  2. Provide verified speaker_intervals in a manifest.\n"
                            "  3. Install diarization backend dependencies."
                        )
                    else:
                        print("[!] Warning: Diarization backend unavailable and no speaker asserted. Proceeding with unverified identity; different voices may be mixed.")
                else:
                    diarize_media_path: Union[str, Path] = v_abs
                    if isolate_vocals_flag and v_abs.exists():
                        print(f"\n--- [0/3] Isolating Vocal Track (audio-separator / Demucs) for {v_abs.name} ---")
                        from vocal_isolation import isolate_vocals as run_isolate_vocals
                        vocals_dir = CACHE_BASE_DIR / "vocals" / source_id
                        try:
                            vocal_stem_path = run_isolate_vocals(
                                v_abs,
                                output_dir=vocals_dir,
                                force=force_realign,
                                pre_extract_audio=True,
                            )
                            if vocal_stem_path and vocal_stem_path.exists():
                                diarize_media_path = vocal_stem_path
                        except Exception as v_err:
                            print(f"[!] Warning: Vocal isolation before diarization failed: {v_err}")

                    from diarization import SpeakerDiarizer
                    from speaker_registry import SpeakerRegistry
                    reg = SpeakerRegistry(config_path=registry_path)
                    diarizer = SpeakerDiarizer(
                        registry=reg,
                        distance_threshold=getattr(args, "diarize_threshold", None),
                    )

                    candidate_turns = None
                    try:
                        # Extract clean acoustic speech activity turns using VAD on the audio/vocal track
                        import soundfile as _sf
                        _audio_data, _audio_sr = _sf.read(str(diarize_media_path))
                        vad_turns = diarizer.detect_speech_turns(_audio_data, _audio_sr)
                        if vad_turns:
                            candidate_turns = vad_turns
                    except Exception as vad_err:
                        print(f"[!] Warning: Speech turn detection failed ({vad_err}); falling back to default segmentation.")
                        candidate_turns = None

                    intervals = diarizer.diarize(
                        diarize_media_path,
                        candidate_turns=candidate_turns,
                        interactive=not non_interactive,
                        content_id=source_id,
                        subtitle_path=item.get("vtt"),
                        cluster_preview_dir=Path(cache_dir) / "clusters" if cache_dir else None,
                    )
                    item["speaker_intervals"] = intervals

        item_status: Dict[str, Any] = {}
        inv_path = ensure_source_aligned(
            item["video"],
            item.get("vtt"),
            force_realign=force_realign,
            force_inventory_rebuild=(diarize_mode == "force"),
            speaker_id=item.get("speaker_id") if item.get("single_speaker_verified") else (cli_speaker if assume_single else None),
            single_speaker_verified=item.get("single_speaker_verified", False) or assume_single,
            assume_single_speaker=assume_single,
            speaker_intervals=item.get("speaker_intervals"),
            isolate_vocals=isolate_vocals_flag,
            cache_status=item_status,
            asr_fill_tails=bool(getattr(args, "asr_fill_tails", False)),
            ranges=getattr(args, "resolved_ranges", None) or None,
            subtitle_stream=getattr(args, "subtitle_stream", None),
            characters_path=getattr(args, "characters", None),
            voice_checks=bool(getattr(args, "voice_checks", True)),
        )
        inventory_paths.append(inv_path)

        if "hit" in item_status:
            if item_status["hit"]:
                cache_hits += 1
            else:
                new_alignments += 1
        else:
            if Path(inv_path).exists():
                cache_hits += 1
            else:
                new_alignments += 1

    combined_inv, legacy_inv_dict = load_or_combine_inventories(inventory_paths)

    speaker_stats: Dict[str, Dict[str, int]] = {}
    if combined_inv.words or combined_inv.events:
        for wo in combined_inv.words.values():
            if getattr(wo, "is_eligible", True):
                spk = wo.speaker_id or "unassigned"
                stats = speaker_stats.setdefault(spk, {"words": 0, "phonemes": 0})
                stats["words"] += 1

        for ev in combined_inv.events.values():
            if ev.kind == "phone" and getattr(ev, "is_eligible", True):
                spk = ev.speaker_id or "unassigned"
                stats = speaker_stats.setdefault(spk, {"words": 0, "phonemes": 0})
                stats["phonemes"] += 1
    else:
        for w_norm, instances in legacy_inv_dict.get("words", {}).items():
            for inst in instances:
                if inst.get("is_eligible", True):
                    spk = inst.get("speaker_id") or "unassigned"
                    stats = speaker_stats.setdefault(spk, {"words": 0, "phonemes": 0})
                    stats["words"] += 1

        for ph_label, instances in legacy_inv_dict.get("phones", {}).items():
            for inst in instances:
                if inst.get("is_eligible", True):
                    spk = inst.get("speaker_id") or "unassigned"
                    stats = speaker_stats.setdefault(spk, {"words": 0, "phonemes": 0})
                    stats["phonemes"] += 1

    return {
        "total_sources": len(items),
        "processed_clips": len(items),
        "cache_hits": cache_hits,
        "new_alignments": new_alignments,
        "inventory_paths": inventory_paths,
        "combined_inventory": combined_inv,
        "legacy_dict": legacy_inv_dict,
        "speaker_stats": speaker_stats,
    }


def print_ingestion_summary_report(
    summary_data: Optional[Dict[str, Any]] = None,
    *,
    total_sources: Optional[int] = None,
    cache_hits: Optional[int] = None,
    new_alignments: Optional[int] = None,
    speaker_stats: Optional[Dict[str, Dict[str, int]]] = None,
) -> None:
    """Print a clean, structured console report of ingestion results and inventory statistics."""
    if summary_data is not None:
        total = summary_data.get("total_sources", 0) if total_sources is None else total_sources
        hits = summary_data.get("cache_hits", 0) if cache_hits is None else cache_hits
        new_proc = summary_data.get("new_alignments", 0) if new_alignments is None else new_alignments
        stats = summary_data.get("speaker_stats", {}) if speaker_stats is None else speaker_stats
    else:
        total = total_sources or 0
        hits = cache_hits or 0
        new_proc = new_alignments or 0
        stats = speaker_stats or {}

    hit_rate = 100.0 if total == 0 else (hits / total) * 100.0

    print("\n" + "=" * 60)
    print("                INGESTION SUMMARY REPORT")
    print("=" * 60)
    print(f"Total Sources Scanned: {total}")
    print(f"Newly Processed Clips: {new_proc}")
    print(f"Cache Hits:            {hits} ({hit_rate:.1f}%)")
    print("-" * 60)
    print("Speaker Breakdown:")
    if stats:
        total_words = 0
        total_phones = 0
        for spk in sorted(stats.keys()):
            w_cnt = stats[spk].get("words", 0)
            p_cnt = stats[spk].get("phonemes", 0)
            total_words += w_cnt
            total_phones += p_cnt
            print(f"  - Speaker '{spk}': {w_cnt} words, {p_cnt} phonemes")
        print("-" * 60)
        print(f"Total Available Units: {total_words} words, {total_phones} phonemes")
    else:
        print("  (No speaker unit statistics available)")
    print("=" * 60 + "\n")


def get_folder_file_snapshot(folders: List[Path]) -> Dict[str, Tuple[int, int]]:
    """Scan watched folders and return map of filepath -> (size, mtime_ns)."""
    from source_discovery import SUPPORTED_VIDEO_EXTENSIONS, should_skip_dir
    snapshot: Dict[str, Tuple[int, int]] = {}
    sub_exts = {".vtt", ".srt"}

    for folder in folders:
        f_path = Path(folder).resolve()
        if not f_path.exists() or not f_path.is_dir():
            continue

        for root, dirnames, filenames in os.walk(f_path):
            dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]
            for fname in filenames:
                p = Path(root) / fname
                suf = p.suffix.lower()
                if suf in SUPPORTED_VIDEO_EXTENSIONS or suf in sub_exts or fname.lower().endswith((".vtt", ".srt")):
                    try:
                        st = p.stat()
                        snapshot[str(p.resolve())] = (st.st_size, st.st_mtime_ns)
                    except OSError:
                        pass
    return snapshot


def run_watch_loop(
    folders: Optional[List[Union[str, Path]]] = None,
    manifest: Optional[str] = None,
    args: Optional[argparse.Namespace] = None,
    cache_dir: Optional[Path] = None,
    max_iterations: Optional[int] = None,
) -> None:
    """
    Run continuous monitoring loop over watched folders and manifest sources.
    Performs an initial pass, then polls every poll_interval seconds for additions/modifications.
    Gracefully handles KeyboardInterrupt (Ctrl+C) and exits with code 0.
    """
    if args is None:
        args = argparse.Namespace()

    poll_interval = float(getattr(args, "poll_interval", 5.0))

    resolved_folders: List[Path] = []
    if folders:
        for f in folders:
            resolved_folders.append(Path(f).resolve())

    if manifest:
        m_p = Path(manifest).resolve()
        if m_p.is_dir():
            resolved_folders.append(m_p)
        elif m_p.is_file():
            try:
                with open(m_p, "r", encoding="utf-8") as mf:
                    m_data = json.load(mf)
                if isinstance(m_data, list):
                    for m_item in m_data:
                        if isinstance(m_item, dict) and "folder" in m_item:
                            resolved_folders.append((m_p.parent / m_item["folder"]).resolve())
            except Exception:
                pass

    watched_folders = list(dict.fromkeys(resolved_folders))

    if watched_folders and not getattr(args, "folder", None):
        setattr(args, "folder", [str(f) for f in watched_folders])
    if manifest and not getattr(args, "manifest", None):
        setattr(args, "manifest", manifest)

    initial_items = collect_source_items(args)
    print(f"\n[WATCH] Performing initial ingestion pass on {len(initial_items)} source(s)...")
    initial_summary = ingest_sources(initial_items, args, cache_dir=cache_dir)
    print_ingestion_summary_report(initial_summary)

    current_snapshot = get_folder_file_snapshot(watched_folders)
    manifest_sig: Optional[Tuple[int, int]] = None
    if manifest and Path(manifest).is_file():
        try:
            m_stat = Path(manifest).stat()
            manifest_sig = (m_stat.st_size, m_stat.st_mtime_ns)
        except OSError:
            manifest_sig = None

    print(f"[WATCH] Watching {len(watched_folders)} folder(s) for changes (poll interval: {poll_interval:.1f}s). Press Ctrl+C to stop.\n")

    iteration = 0
    try:
        while True:
            if max_iterations is not None and iteration >= max_iterations:
                break
            iteration += 1

            time.sleep(poll_interval)

            new_snapshot = get_folder_file_snapshot(watched_folders)

            changed_files: Set[str] = set()
            for f_path, f_sig in new_snapshot.items():
                if f_path not in current_snapshot or current_snapshot[f_path] != f_sig:
                    changed_files.add(f_path)

            manifest_changed = False
            if manifest and Path(manifest).is_file():
                try:
                    m_stat = Path(manifest).stat()
                    new_m_sig = (m_stat.st_size, m_stat.st_mtime_ns)
                    if new_m_sig != manifest_sig:
                        manifest_changed = True
                        manifest_sig = new_m_sig
                except OSError:
                    pass

            if changed_files or manifest_changed:
                delta_items: List[Dict[str, Any]] = []

                if changed_files and watched_folders:
                    from source_discovery import discover_folder_sources
                    for wf in watched_folders:
                        discovered = discover_folder_sources(wf)
                        for d_item in discovered:
                            v_abs = str(Path(d_item["video"]).resolve())
                            vtt_abs = str(Path(d_item["vtt"]).resolve()) if d_item.get("vtt") else None
                            if v_abs in changed_files or (vtt_abs and vtt_abs in changed_files):
                                if getattr(args, "speaker", None) and getattr(args, "assume_single_speaker", False):
                                    d_item["speaker_id"] = args.speaker
                                    d_item["single_speaker_verified"] = True
                                elif getattr(args, "assume_single_speaker", False):
                                    d_item["single_speaker_verified"] = True
                                delta_items.append(d_item)

                if manifest_changed and manifest:
                    parsed_man = parse_manifest_items(manifest)
                    for m_item in parsed_man:
                        v_abs = str(Path(m_item["video"]).resolve())
                        vtt_abs = str(Path(m_item["vtt"]).resolve()) if m_item.get("vtt") else None
                        if v_abs in changed_files or (vtt_abs and vtt_abs in changed_files) or manifest_changed:
                            delta_items.append(m_item)

                seen_delta: Set[str] = set()
                unique_delta: List[Dict[str, Any]] = []
                for d_it in delta_items:
                    v_key = str(Path(d_it["video"]).resolve())
                    if v_key not in seen_delta:
                        seen_delta.add(v_key)
                        unique_delta.append(d_it)

                if unique_delta:
                    print(f"\n[WATCH] Detected changes in {len(changed_files)} file(s); running delta ingestion on {len(unique_delta)} source(s)...")
                    delta_summary = ingest_sources(unique_delta, args, cache_dir=cache_dir)
                    print_ingestion_summary_report(delta_summary)

                current_snapshot = new_snapshot
            else:
                if set(current_snapshot.keys()) != set(new_snapshot.keys()):
                    current_snapshot = new_snapshot

    except KeyboardInterrupt:
        print("\n[WATCH] Watch mode terminated by user (Ctrl+C). Exiting cleanly.")
        sys.exit(0)


ASR_FILL_TAILS_HELP = (
    "Opt-in: append words that Whisper hears after a subtitle cue ends to the alignment transcript "
    "(off by default; Whisper output is unverified and can add hallucinated or other speakers' words)"
)


def add_synthesis_args(p: argparse.ArgumentParser) -> None:
    """Add common synthesis arguments to a parser."""
    p.add_argument("--text", default=None, help="Target text / sentence to synthesize")
    p.add_argument("--target-audio", default=None, help="Path to reference audio file for target pronunciation recognition")
    p.add_argument("--record-target", action="store_true", default=False, help="Record target pronunciation from microphone")
    p.add_argument("--diarize", choices=["auto", "off", "force"], default="auto", help="Diarization mode ('auto', 'off', 'force')")
    p.add_argument(
        "--diarize-threshold",
        type=float,
        default=None,
        help="Cosine distance threshold for speaker clustering (default: auto: 0.18 for WavLM-SV, 0.45 for acoustic fallback)",
    )
    p.add_argument("--speaker-registry", default="config/speaker_registry.json", help="Path to speaker registry JSON")
    p.add_argument("--guide-transcript", default=None, help="Path to transcript file for prosody guide audio")
    p.add_argument("--selector", choices=["legacy", "span"], default="span", help="Selector backend ('span' or 'legacy', default: span)")
    p.add_argument("--speaker", default=None, help="Speaker identity filter")
    p.add_argument("--assume-single-speaker", action="store_true", help="Explicitly assume single speaker")
    p.add_argument("--mode", choices=["source-derived", "cuts-only"], default="source-derived", help="Rendering mode")
    p.add_argument("--features", choices=["basic", "wavlm"], default="basic", help="Feature extraction backend")
    p.add_argument(
        "--fps",
        default="auto",
        help="Timeline framerate: auto (the sources' own rate, e.g. 24000/1001 for movies; default) or a value such as 30000/1001",
    )
    p.add_argument("--render", help="Output rendered MP4 path")
    p.add_argument("--plan", help="Output EditPlan JSON path")
    p.add_argument("--xml", help="Output Premiere Pro / FCP7 XML path")
    p.add_argument("--guide", help="Optional human prosody guide audio")
    p.add_argument("--force-realign", action="store_true", help="Force re-align even if source is cached")
    p.add_argument(
        "--asr-fill-tails",
        action="store_true",
        default=False,
        help=ASR_FILL_TAILS_HELP,
    )
    p.add_argument(
        "--normalize-loudness",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Normalize output audio loudness according to EBU R128 (-16 LUFS, -1.0 dBTP) (default: enabled)",
    )
    p.add_argument(
        "--isolate-vocals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Isolate speech stems using audio-separator/Demucs before alignment (default: True)",
    )
    p.add_argument(
        "--audio-track",
        choices=["raw", "vocals", "recombined", "all", "combined", "isolated", "both"],
        default="recombined",
        help="Audio track for rendering: 'recombined' (normalized vocals + original background instrumental, default), 'raw' (raw full mix, formerly combined), 'vocals' (formerly isolated speech stem), 'both' (raw & vocals), or 'all' (raw, vocals, and recombined)",
    )
    p.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Disable interactive prompts (for OOV words, speaker naming, etc.) and fallback to defaults",
    )
    p.add_argument(
        "--review-pronunciations",
        action="store_true",
        default=False,
        help="Interactively review and select pronunciations for OOV words and variants (default: auto-select recommended)",
    )
    p.add_argument(
        "--interactive-clusters",
        action="store_true",
        default=False,
        help="Interactively review unassigned speaker clusters during diarization (default: false unless running ingest or explicitly set)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="Custom cache root directory (defaults to .build)",
    )
    p.add_argument(
        "--subtitles",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export matching .srt subtitle file alongside output video",
    )
    p.add_argument(
        "--subtitles-path",
        default=None,
        help="Custom output path for exported subtitle file",
    )
    p.add_argument(
        "--burn-subtitles",
        action="store_true",
        default=False,
        help="Hardcode/burn subtitles directly into rendered output video using FFmpeg",
    )
    p.add_argument(
        "--subtitle-style",
        choices=["classic", "modern"],
        default="classic",
        help="Subtitle visual style for burned subtitles ('classic' YTP yellow or 'modern' clean white)",
    )
    p.add_argument(
        "--subtitle-mode",
        choices=["phrase", "word"],
        default="phrase",
        help="Subtitle timing mode: 'phrase' (default natural clauses) or 'word' (per-word highlights)",
    )


def build_cli_parser() -> argparse.ArgumentParser:
    """Build and return top-level CLI argument parser."""
    parser = argparse.ArgumentParser(description="YTP Sentence Mixing Engine")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # 1. Mix command
    p_mix = subparsers.add_parser("mix", help="Synthesize target sentence using one or more source videos")
    add_synthesis_args(p_mix)
    p_mix.add_argument("source", nargs="*", default=None, help="Optional source path(s) to video(s), folder(s), or manifest(s)")
    p_mix.add_argument("--video", nargs="+", help="Path(s) to source video(s)")
    p_mix.add_argument("--vtt", nargs="+", help="Path(s) to WebVTT subtitles")
    p_mix.add_argument("--manifest", help="Path to JSON manifest")
    p_mix.add_argument("--folder", nargs="+", help="Path(s) to folder(s) containing source videos and subtitles")
    p_mix.add_argument("--inventory", nargs="+", help="Direct path(s) to units_index.json")
    p_mix.add_argument(
        "--all-sources",
        "--all",
        dest="all_sources",
        action="store_true",
        default=False,
        help="Mix across all available ingested sources in cache (optionally filtered by --speaker)",
    )

    # Ingest command
    p_ingest = subparsers.add_parser(
        "ingest",
        help="Ingest source video(s) and subtitle(s) into alignment cache and inventory without synthesizing",
    )
    p_ingest.add_argument("source", nargs="*", default=None, help="Source path(s) to video(s), folder(s), or manifest(s)")
    p_ingest.add_argument("--manifest", default=None, help="Path to JSON manifest or directory")
    p_ingest.add_argument("--folder", nargs="+", default=None, help="Path(s) to folder(s) containing source videos and subtitles")
    p_ingest.add_argument("--video", nargs="+", default=None, help="Path(s) to source video(s)")
    p_ingest.add_argument("--vtt", nargs="+", default=None, help="Path(s) to WebVTT or SRT subtitles")
    p_ingest.add_argument("--cache-dir", default=None, help="Custom cache root directory (defaults to .build)")
    p_ingest.add_argument("--speaker", default=None, help="Speaker identity filter or assignment")
    p_ingest.add_argument("--assume-single-speaker", action="store_true", default=False, help="Explicitly assume single speaker")
    p_ingest.add_argument("--force-realign", action="store_true", default=False, help="Force re-align even if source is cached")
    p_ingest.add_argument(
        "--asr-fill-tails",
        action="store_true",
        default=False,
        help=ASR_FILL_TAILS_HELP,
    )
    p_ingest.add_argument(
        "--isolate-vocals",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Isolate speech stems using audio-separator/Demucs before alignment (default: True)",
    )
    p_ingest.add_argument(
        "--diarize",
        choices=["auto", "off", "force"],
        default="auto",
        help="Diarization mode ('auto', 'off', 'force')",
    )
    p_ingest.add_argument(
        "--diarize-threshold",
        type=float,
        default=None,
        help="Cosine distance threshold for speaker clustering (default: auto: 0.18 for WavLM-SV, 0.45 for acoustic fallback)",
    )
    p_ingest.add_argument(
        "--speaker-registry",
        default="config/speaker_registry.json",
        help="Path to speaker registry JSON",
    )
    p_ingest.add_argument(
        "--watch",
        action="store_true",
        default=False,
        help="Continuously monitor input folders for new or modified media and ingest incrementally",
    )
    p_ingest.add_argument(
        "--poll-interval",
        type=float,
        default=5.0,
        help="Polling interval in seconds for continuous --watch mode (default: 5.0)",
    )
    p_ingest.add_argument(
        "--non-interactive",
        action="store_true",
        default=False,
        help="Disable interactive prompts (for speaker naming, OOV words, etc.) and fallback to defaults",
    )

    # 2. Preprocess standalone
    p_prep = subparsers.add_parser("preprocess", help="Preprocess video and VTT into corpus chunks")
    p_prep.add_argument("--video", required=True, help="Path to input video (.mp4)")
    p_prep.add_argument("--vtt", required=True, help="Path to WebVTT subtitles (.vtt)")
    p_prep.add_argument("--output", default=str(SCRIPT_DIR / "corpus"), help="Output corpus directory")
    p_prep.add_argument("--speaker", default=None, help="Speaker ID")
    p_prep.add_argument("--assume-single-speaker", action="store_true", help="Assume single speaker")
    p_prep.add_argument(
        "--asr-fill-tails",
        action="store_true",
        default=False,
        help=ASR_FILL_TAILS_HELP,
    )
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

    # 6. Clean command
    add_movie_args(p_mix)
    add_movie_args(p_ingest)
    for p_render in (p_mix, p_auto):
        p_render.add_argument(
            "--resolution",
            default="1080p",
            help="Output video size: 1080p (default), 1440p, 4k (3840x2160), source (native size), or WxH; letterboxed to fit",
        )

    p_scenes = subparsers.add_parser("scenes", help="List named scenes (with times) for a movie")
    p_scenes.add_argument("source", help="Movie file")
    p_scenes.add_argument("--cache-dir", default=None, help="Custom cache root directory (defaults to .build)")

    p_speakers = subparsers.add_parser("speakers", help="List speakers and usable speech per ingested movie")
    p_speakers.add_argument("source", nargs="*", default=None, help="Movie file(s); default: all ingested movies")
    p_speakers.add_argument("--cache-dir", default=None, help="Custom cache root directory (defaults to .build)")

    p_review = subparsers.add_parser(
        "review-speakers",
        help="Interactively label unlabeled or voice-flagged movie lines (saved to config/speaker_labels/)",
    )
    p_review.add_argument("source", help="Movie file (must be ingested)")
    p_review.add_argument(
        "--only",
        choices=["unlabeled", "flagged", "voice", "all"],
        default="flagged",
        help="Which lines to review: flagged (voice mismatch + unlabeled, default), unlabeled, voice (voice-propagated), all",
    )
    p_review.add_argument("--range", action="append", default=None, metavar="START-END", help="Only lines inside this range")
    p_review.add_argument("--limit", type=int, default=50, help="Maximum lines per session (default 50)")
    p_review.add_argument("--no-play", action="store_true", default=False, help="Do not play audio previews")
    p_review.add_argument(
        "--audio",
        choices=["mix", "center", "vocals"],
        default="mix",
        help="What to play: mix (the film's stereo mix, default), center (raw 5.1 center channel), or vocals "
        "(separated dialogue with music and effects removed; what alignment and voice checks analyze)",
    )
    p_review.add_argument("--cache-dir", default=None, help="Custom cache root directory (defaults to .build)")

    p_clean = subparsers.add_parser(
        "clean",
        help="Clean up cache and build artifacts for specific source(s) or all sources",
    )
    p_clean.add_argument(
        "source",
        nargs="*",
        default=None,
        help="Source path(s) (e.g. input/crisis.mp4), source stem (e.g. crisis), or source ID (e.g. crisis_9c96ee02)",
    )
    p_clean.add_argument("--video", nargs="+", default=None, help="Path(s) to source video(s)")
    p_clean.add_argument("--cache-dir", default=None, help="Custom cache root directory (defaults to .build)")
    p_clean.add_argument(
        "--all",
        action="store_true",
        default=False,
        help="Clean cache/build files for all sources",
    )
    p_clean.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Show what cache files and directories would be removed without deleting them",
    )
    p_clean.add_argument(
        "--keep-speakers",
        action="store_true",
        default=False,
        help="Retain speaker profiles registered from the cleaned source in the speaker registry",
    )
    p_clean.add_argument(
        "--speaker-registry",
        default="config/speaker_registry.json",
        help="Path to speaker registry JSON (defaults to config/speaker_registry.json)",
    )

    return parser


def add_movie_args(p: argparse.ArgumentParser) -> None:
    """Options for long sources (whole movies)."""
    p.add_argument(
        "--range",
        action="append",
        default=None,
        metavar="START-END",
        help="Limit a movie to a time range, e.g. 01:30:00-01:35:00 (repeatable). Ingest aligns only the "
        "overlapping 10-minute blocks; mix only uses words inside the range(s).",
    )
    p.add_argument(
        "--scene",
        action="append",
        default=None,
        help="Limit a movie to a scene by number or heading keyword (see 'run_mixer.py scenes <movie>'); repeatable",
    )
    p.add_argument(
        "--subtitle-stream",
        default=None,
        help="Embedded subtitle track to use for a movie: auto, sdh, plain, or a stream index "
        "(default: labeled VTT from tools/label_subtitles.py if present, else sidecar file, else auto)",
    )
    p.add_argument(
        "--characters",
        default=None,
        help="Character alias list for resolving subtitle speaker labels (default: config/characters/star_wars.json)",
    )
    p.add_argument(
        "--voice-checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="For movies: verify subtitle speaker labels by voice and label unlabeled lines by voice when confident (default: on)",
    )


def main():
    parser = build_cli_parser()
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
        fps_auto = str(args.fps).strip().lower() == "auto"
        fps = RationalTime(30000, 1001) if fps_auto else RationalTime.from_number(args.fps)

        # Collect source items to align / load
        try:
            source_items = collect_source_items(args)
            args.resolved_ranges = resolve_time_ranges(
                getattr(args, "range", None), getattr(args, "scene", None), [i["video"] for i in source_items]
            )
        except (ValueError, FileNotFoundError) as err:
            parser.error(str(err))

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
            v_abs = Path(item["video"]).resolve()
            source_id = get_source_id(str(v_abs)) if v_abs.exists() else Path(item["video"]).stem

            is_asserted = bool(
                item.get("single_speaker_verified")
                or args.assume_single_speaker
                or item.get("speaker_intervals")
                or (v_abs.exists() and is_movie_source(v_abs))
            )
            if not is_asserted:
                if args.diarize in ("auto", "force"):
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
                        diarize_media_path: Union[str, Path] = v_abs
                        if isolate_vocals_flag and v_abs.exists():
                            print(f"\n--- [0/3] Isolating Vocal Track (audio-separator / Demucs) for {v_abs.name} ---")
                            from vocal_isolation import isolate_vocals as run_isolate_vocals
                            vocals_dir = CACHE_BASE_DIR / "vocals" / source_id
                            try:
                                vocal_stem_path = run_isolate_vocals(
                                    v_abs,
                                    output_dir=vocals_dir,
                                    force=args.force_realign,
                                    pre_extract_audio=True,
                                )
                                if vocal_stem_path and vocal_stem_path.exists():
                                    diarize_media_path = vocal_stem_path
                            except Exception as v_err:
                                print(f"[!] Warning: Vocal isolation before diarization failed: {v_err}")

                        from diarization import SpeakerDiarizer
                        from speaker_registry import SpeakerRegistry

                        reg = SpeakerRegistry(config_path=registry_path)
                        diarizer = SpeakerDiarizer(
                            registry=reg,
                            distance_threshold=getattr(args, "diarize_threshold", None),
                        )

                        candidate_turns = None
                        try:
                            # Extract clean acoustic speech activity turns using VAD on the audio/vocal track
                            import soundfile as _sf
                            _audio_data, _audio_sr = _sf.read(str(diarize_media_path))
                            vad_turns = diarizer.detect_speech_turns(_audio_data, _audio_sr)
                            if vad_turns:
                                candidate_turns = vad_turns
                        except Exception as vad_err:
                            print(f"[!] Warning: Speech turn detection failed ({vad_err}); falling back to default segmentation.")
                            candidate_turns = None

                        is_cluster_interactive = (
                            bool(getattr(args, "interactive_clusters", False))
                            and sys.stdin.isatty()
                            and not getattr(args, "non_interactive", False)
                        )
                        intervals = diarizer.diarize(
                            diarize_media_path,
                            candidate_turns=candidate_turns,
                            interactive=is_cluster_interactive,
                            content_id=source_id,
                            subtitle_path=item.get("vtt"),
                            cluster_preview_dir=CACHE_BASE_DIR / "clusters",
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
                asr_fill_tails=bool(getattr(args, "asr_fill_tails", False)),
                ranges=getattr(args, "resolved_ranges", None) or None,
                subtitle_stream=getattr(args, "subtitle_stream", None),
                characters_path=getattr(args, "characters", None),
                voice_checks=bool(getattr(args, "voice_checks", True)),
            )
            inventory_paths.append(inv_path)

        if not inventory_paths or getattr(args, "all_sources", False):
            from cache import find_all_cached_inventories, find_cached_inventories_for_speaker
            if getattr(args, "all_sources", False) and not getattr(args, "speaker", None):
                discovered_invs = find_all_cached_inventories(cache_dir=CACHE_BASE_DIR)
                if discovered_invs:
                    print(f"[*] Pooling all {len(discovered_invs)} available ingested source inventory/inventories in cache.")
                    for inv_p in discovered_invs:
                        if inv_p not in inventory_paths:
                            inventory_paths.append(inv_p)
                elif not inventory_paths:
                    parser.error(
                        "No valid ingested sources found in cache. "
                        "Please ingest source media first with 'run_mixer.py ingest <source>' or specify sources via --video / --folder / --manifest."
                    )
            elif not inventory_paths:
                if getattr(args, "speaker", None):
                    discovered_invs = find_cached_inventories_for_speaker(
                        args.speaker,
                        cache_dir=CACHE_BASE_DIR,
                        registry_path=registry_path,
                    )
                    if discovered_invs:
                        print(f"[*] Discovered {len(discovered_invs)} ingested source inventory/inventories containing speaker '{args.speaker}'.")
                        inventory_paths.extend(discovered_invs)
                    else:
                        from speaker_registry import SpeakerRegistry
                        reg = SpeakerRegistry(config_path=registry_path)
                        known_speakers = [s.speaker_id for s in reg.list_speakers()]
                        avail_spk_str = f" Available registered speakers: {', '.join(sorted(known_speakers))}." if known_speakers else ""
                        parser.error(
                            f"No ingested sources found for speaker '{args.speaker}'.{avail_spk_str} "
                            "Please ingest source media first with 'run_mixer.py ingest <source>' or specify source(s) via --video / --folder / --manifest."
                        )
                else:
                    parser.error(
                        "Please provide source(s) (--video, --folder, --manifest, or positional), "
                        "or specify --speaker <name> or --all-sources to mix across ingested sources in cache."
                    )

        # Load inventory
        inventory_index, legacy_inv_dict = load_or_combine_inventories(inventory_paths)
        if fps_auto:
            fps = pick_timeline_fps(inventory_index, default=fps)
        if getattr(args, "resolved_ranges", None):
            if args.selector == "legacy":
                parser.error("--range/--scene require --selector span.")
            kept_words = restrict_inventory_to_ranges(inventory_index, args.resolved_ranges)
            legacy_inv_dict = {"words": {}, "phones": {}, "words_inventory": {}, "phones_inventory": {}}
            span_desc = ", ".join(f"{s / 60:.1f}-{e / 60:.1f} min" for s, e in args.resolved_ranges)
            print(f"[*] Restricted to time range(s) {span_desc}: {kept_words} eligible words")

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
            norm_req = args.speaker.strip().lower().replace(" ", "_").replace("-", "_")
            matched_spk = next(
                (s for s in unique_speakers if s.lower().replace("-", "_") == norm_req),
                None,
            )
            if matched_spk is None:
                avail_str = ", ".join(sorted(unique_speakers)) if unique_speakers else "none"
                parser.error(
                    f"Requested speaker '{args.speaker}' not found in inventory. "
                    f"Available speakers: {avail_str}"
                )
            effective_speaker = matched_spk
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
        is_interactive = (
            bool(getattr(args, "review_pronunciations", False))
            and sys.stdin.isatty()
            and not getattr(args, "non_interactive", False)
        )
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

            lead_in = 0.080 if (args.mode != "cuts-only" and -1 not in getattr(graph, "explicit_gaps", {})) else 0.0
            plan = build_edit_plan(
                selected_spans,
                graph,
                inventory_index,
                project_sample_rate=48000,
                video_fps=fps,
                rendering_mode=args.mode,
                lead_in_sec=lead_in,
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
                    output_resolution=getattr(args, "resolution", None),
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

                # Subtitle generation & optional burning
                staged_subtitles: Optional[Path] = None
                subtitles_p: Optional[Path] = None
                if getattr(args, "subtitles", False) or getattr(args, "subtitles_path", None) or getattr(args, "burn_subtitles", False):
                    sub_ext = ".vtt" if (args.subtitles_path and Path(args.subtitles_path).suffix.lower() == ".vtt") else ".srt"
                    staged_subtitles = attempt_dir / f"{render_p.stem}{sub_ext}"
                    export_subtitles(
                        plan,
                        staged_subtitles,
                        cue_mode=getattr(args, "subtitle_mode", "phrase"),
                    )
                    if getattr(args, "subtitles_path", None):
                        subtitles_p = Path(args.subtitles_path).resolve()
                    elif getattr(args, "subtitles", False):
                        subtitles_p = render_p.with_suffix(sub_ext)

                if getattr(args, "burn_subtitles", False):
                    if not staged_subtitles or not staged_subtitles.exists():
                        raise RuntimeError("Cannot burn subtitles: subtitle file was not generated.")
                    for mp4_f in list(attempt_dir.glob("*.mp4")):
                        if mp4_f.name.endswith("_burned.mp4"):
                            continue
                        burned_temp = attempt_dir / f"{mp4_f.stem}_burned.mp4"
                        burn_subtitles_to_video(
                            mp4_f,
                            staged_subtitles,
                            burned_temp,
                            style=getattr(args, "subtitle_style", "classic"),
                        )
                        shutil.move(str(burned_temp), str(mp4_f))
                    # Clean up temporary subtitle file if standalone export was not requested
                    if not getattr(args, "subtitles", False) and not getattr(args, "subtitles_path", None):
                        staged_subtitles.unlink(missing_ok=True)
                        staged_subtitles = None

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
                    if suffix in (".srt", ".vtt"):
                        return "subtitles"
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
                if staged_subtitles and subtitles_p:
                    file_mappings[staged_subtitles] = subtitles_p

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

                print(f"[+] Successfully published mix to: {format_file_url(render_p)}")
                print(f"[+] Saved EditPlan: {format_file_url(plan_p)}")
                if xml_p:
                    print(f"[+] Saved Final Cut Pro XML: {format_file_url(xml_p)}")
                if subtitles_p:
                    print(f"[+] Saved Subtitles: {format_file_url(subtitles_p)}")
            finally:
                shutil.rmtree(attempt_dir, ignore_errors=True)

        else:
            # Legacy selector
            print(f"\n[+] Running Legacy Word-First Unit Selection for: \"{args.text}\"")
            legacy_units_dict = legacy_inv_dict
            words_source = legacy_inv_dict.get("words_inventory") or legacy_inv_dict.get("words", {})
            phones_source = legacy_inv_dict.get("phones_inventory") or legacy_inv_dict.get("phones", {})

            if effective_speaker is not None:
                filtered_w = {
                    w: [inst for inst in insts if inst.get("speaker_id") == effective_speaker and inst.get("is_eligible", True)]
                    for w, insts in words_source.items()
                    if isinstance(insts, list)
                }
                filtered_w = {w: insts for w, insts in filtered_w.items() if insts}
                filtered_p = {
                    ph: [inst for inst in insts if inst.get("speaker_id") == effective_speaker and inst.get("is_eligible", True)]
                    for ph, insts in phones_source.items()
                    if isinstance(insts, list)
                }
                filtered_p = {ph: insts for ph, insts in filtered_p.items() if insts}
                legacy_units_dict = {
                    "words": filtered_w,
                    "words_inventory": filtered_w,
                    "phones": filtered_p,
                    "phones_inventory": filtered_p,
                }
            else:
                legacy_units_dict = {
                    "words": words_source,
                    "words_inventory": words_source,
                    "phones": phones_source,
                    "phones_inventory": phones_source,
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
                    output_resolution=getattr(args, "resolution", None),
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

                # Subtitle generation & optional burning
                staged_subtitles = None
                subtitles_p = None
                if getattr(args, "subtitles", False) or getattr(args, "subtitles_path", None) or getattr(args, "burn_subtitles", False):
                    sub_ext = ".vtt" if (args.subtitles_path and Path(args.subtitles_path).suffix.lower() == ".vtt") else ".srt"
                    staged_subtitles = attempt_dir / f"{render_p.stem}{sub_ext}"
                    export_subtitles(
                        units,
                        staged_subtitles,
                        cue_mode=getattr(args, "subtitle_mode", "phrase"),
                    )
                    if getattr(args, "subtitles_path", None):
                        subtitles_p = Path(args.subtitles_path).resolve()
                    elif getattr(args, "subtitles", False):
                        subtitles_p = render_p.with_suffix(sub_ext)

                if getattr(args, "burn_subtitles", False):
                    if not staged_subtitles or not staged_subtitles.exists():
                        raise RuntimeError("Cannot burn subtitles: subtitle file was not generated.")
                    for mp4_f in list(attempt_dir.glob("*.mp4")):
                        if mp4_f.name.endswith("_burned.mp4"):
                            continue
                        burned_temp = attempt_dir / f"{mp4_f.stem}_burned.mp4"
                        burn_subtitles_to_video(
                            mp4_f,
                            staged_subtitles,
                            burned_temp,
                            style=getattr(args, "subtitle_style", "classic"),
                        )
                        shutil.move(str(burned_temp), str(mp4_f))
                    if not getattr(args, "subtitles", False) and not getattr(args, "subtitles_path", None):
                        staged_subtitles.unlink(missing_ok=True)
                        staged_subtitles = None

                def _compute_sha256(p: Path) -> str:
                    h = hashlib.sha256()
                    with open(p, "rb") as f:
                        while chunk := f.read(65536):
                            h.update(chunk)
                    return h.hexdigest()

                def _artifact_key(staged_file: Path) -> str:
                    stem = staged_file.stem
                    suffix = staged_file.suffix.lower()
                    if suffix in (".srt", ".vtt"):
                        return "subtitles"
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
                if staged_subtitles and subtitles_p:
                    file_mappings[staged_subtitles] = subtitles_p

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

                print(f"[+] Successfully published legacy mix to: {format_file_url(render_p)}")
                if xml_p:
                    print(f"[+] Saved Final Cut Pro XML: {format_file_url(xml_p)}")
                if subtitles_p:
                    print(f"[+] Saved Subtitles: {format_file_url(subtitles_p)}")
            finally:
                shutil.rmtree(attempt_dir, ignore_errors=True)

    elif args.command == "ingest":
        has_input = bool(
            getattr(args, "source", None)
            or getattr(args, "folder", None)
            or getattr(args, "manifest", None)
            or getattr(args, "video", None)
        )
        if not has_input:
            parser.error("Please provide source path(s) (video, folder, manifest) or --video/--folder/--manifest for ingestion.")

        if getattr(args, "watch", False):
            watched_folders: List[Path] = []
            if getattr(args, "source", None):
                s_list = args.source if isinstance(args.source, list) else [args.source]
                for s_entry in s_list:
                    for s_item in s_entry.split(","):
                        s_clean = s_item.strip()
                        if s_clean and Path(s_clean).is_dir():
                            watched_folders.append(Path(s_clean).resolve())
            if getattr(args, "folder", None):
                f_list = args.folder if isinstance(args.folder, list) else [args.folder]
                for f_entry in f_list:
                    for f_item in f_entry.split(","):
                        f_clean = f_item.strip()
                        if f_clean:
                            watched_folders.append(Path(f_clean).resolve())

            run_watch_loop(
                folders=watched_folders,
                manifest=getattr(args, "manifest", None),
                args=args,
                cache_dir=CACHE_BASE_DIR,
            )
            return

        try:
            source_items = collect_source_items(args)
            args.resolved_ranges = resolve_time_ranges(
                getattr(args, "range", None), getattr(args, "scene", None), [i["video"] for i in source_items]
            )
        except (ValueError, FileNotFoundError) as err:
            parser.error(str(err))

        if not source_items:
            parser.error("No valid video sources found to ingest.")

        try:
            summary = ingest_sources(source_items, args, cache_dir=CACHE_BASE_DIR)
        except ValueError as err:
            sys.exit(f"[!] Ingestion error: {err}")

        print_ingestion_summary_report(summary)

    elif args.command == "preprocess":
        vocal_stem_path = None
        if getattr(args, "isolate_vocals", False):
            from vocal_isolation import isolate_vocals as run_isolate_vocals
            v_p = Path(args.video).resolve()
            source_id = get_source_id(str(v_p))
            vocal_stem_path = run_isolate_vocals(
                v_p,
                output_dir=CACHE_BASE_DIR / "vocals" / source_id,
                pre_extract_audio=True,
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
            asr_tail_enrichment=args.asr_fill_tails,
        )

    elif args.command == "align":
        dict_to_use = args.dict
        if dict_to_use and Path(dict_to_use).resolve() == (SCRIPT_DIR / "dictionaries" / "merged_dictionary.dict").resolve():
            from lexicon import stage_mfa_dictionary
            staged_dict_dir = CACHE_BASE_DIR / "mfa"
            staged_dict_path, _ = stage_mfa_dictionary(
                base_dict_path=SCRIPT_DIR / "dictionaries" / "merged_dictionary.dict",
                custom_dict_path=SCRIPT_DIR / "dictionaries" / "custom_pronunciations.txt",
                output_path=staged_dict_dir / "staged_dictionary.dict",
            )
            dict_to_use = str(staged_dict_path)
        run_mfa_align(args.corpus, args.output, dict_to_use)

    elif args.command == "build-inventory":
        build_units_inventory(
            args.aligned,
            args.manifest,
            args.output,
            speaker_id=args.speaker,
            single_speaker_verified=args.assume_single_speaker,
        )

    elif args.command == "scenes":
        from movie_commands import cmd_scenes
        sys.exit(cmd_scenes(Path(args.source).resolve(), load_movie_scenes(args.source)))

    elif args.command == "speakers":
        from movie_commands import cmd_speakers
        sys.exit(cmd_speakers([Path(s).resolve() for s in (args.source or [])], CACHE_BASE_DIR))

    elif args.command == "review-speakers":
        from movie_commands import cmd_review_speakers
        try:
            rc = cmd_review_speakers(
                Path(args.source).resolve(),
                CACHE_BASE_DIR,
                only=args.only,
                ranges=args.range,
                limit=args.limit,
                play=not args.no_play,
                audio=args.audio,
            )
        except (FileNotFoundError, ValueError) as err:
            parser.error(str(err))
        sys.exit(rc)

    elif args.command == "clean":
        from cache import clean_all_source_caches, clean_source_cache, format_bytes

        targets: List[str] = []
        if getattr(args, "source", None):
            s_list = args.source if isinstance(args.source, list) else [args.source]
            for s in s_list:
                for item in s.split(","):
                    clean_item = item.strip()
                    if clean_item and clean_item not in targets:
                        targets.append(clean_item)
        if getattr(args, "video", None):
            v_list = args.video if isinstance(args.video, list) else [args.video]
            for v in v_list:
                for item in v.split(","):
                    clean_item = item.strip()
                    if clean_item and clean_item not in targets:
                        targets.append(clean_item)

        if not targets and not getattr(args, "all", False):
            parser.error("Specify one or more source(s) to clean (e.g. 'run_mixer.py clean input/crisis.mp4' or 'run_mixer.py clean crisis') or use --all to clean all sources.")

        dry_run = bool(getattr(args, "dry_run", False))
        clean_speakers = not bool(getattr(args, "keep_speakers", False))
        reg_path = getattr(args, "speaker_registry", "config/speaker_registry.json")
        cache_dir = CACHE_BASE_DIR

        if getattr(args, "all", False):
            mode_str = "[*] Dry run - scanning" if dry_run else "[*] Cleaning"
            print(f"{mode_str} all source cache files under {cache_dir}...")
            res = clean_all_source_caches(
                cache_dir=cache_dir,
                dry_run=dry_run,
                clean_speakers=clean_speakers,
                registry_path=reg_path,
            )
            num_dirs = len(res["deleted_dirs"])
            num_files = len(res["deleted_files"])
            size_str = format_bytes(res["freed_bytes"])
            action_verb = "Would free" if dry_run else "Freed"
            print(f"[+] {action_verb} {size_str} across {num_dirs} directories and {num_files} files.")
            if res.get("removed_speakers"):
                spk_verb = "Would clear" if dry_run else "Cleared"
                print(f"[+] {spk_verb} {len(res['removed_speakers'])} speaker profile(s) from registry ({', '.join(res['removed_speakers'])}).")
            return

        for t in targets:
            res = clean_source_cache(
                t,
                cache_dir=cache_dir,
                dry_run=dry_run,
                clean_speakers=clean_speakers,
                registry_path=reg_path,
            )
            total_items = len(res["deleted_dirs"]) + len(res["deleted_files"])
            if total_items == 0 and not res["removed_media_keys"] and not res.get("removed_speakers"):
                print(f"[-] No cached artifacts found for source '{t}' under {cache_dir}.")
                continue

            action_verb = "Would remove" if dry_run else "Cleaned"
            sid_info = f" (matched IDs: {', '.join(res['matched_source_ids'])})" if res["matched_source_ids"] else ""
            print(f"\n[+] {action_verb} cache for source '{t}'{sid_info}:")
            for d in res["deleted_dirs"]:
                rel_d = Path(d)
                try:
                    rel_d = rel_d.relative_to(SCRIPT_DIR)
                except ValueError:
                    pass
                print(f"    - Removed directory: {rel_d}")
            for f in res["deleted_files"]:
                rel_f = Path(f)
                try:
                    rel_f = rel_f.relative_to(SCRIPT_DIR)
                except ValueError:
                    pass
                print(f"    - Removed file:      {rel_f}")
            if res["removed_media_keys"]:
                print(f"    - Invalidated {len(res['removed_media_keys'])} entry from .media_hash_index.json")
            if res.get("removed_speakers"):
                for spk in res["removed_speakers"]:
                    print(f"    - Removed speaker profile from registry: '{spk}'")

            size_str = format_bytes(res["freed_bytes"])
            verb_past = "would be freed" if dry_run else "freed"
            print(f"    Total: {size_str} {verb_past} across {len(res['deleted_dirs'])} directories and {len(res['deleted_files'])} files.")


if __name__ == "__main__":
    main()
