#!/usr/bin/env python3
"""
Vocal Separation & Speech Isolation Pre-processing Module.
Integrates audio-separator CLI (UVR-MDX-NET / VR models) with Demucs CLI fallback.
Caches isolated vocal stems under .build/vocals/<source_id>/vocals.wav to prevent
redundant computation across runs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Union


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    """Compute sha256 hex digest of a file."""
    p = Path(file_path).resolve()
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def get_source_id(source_path: Union[str, Path]) -> str:
    """
    Generate a clean filesystem-safe source ID with content hash for caching.
    Matches the naming convention used across sentence_mixer.
    """
    p = Path(source_path).resolve()
    clean_stem = re.sub(r"[^\w\-_]", "_", p.stem)[:40]
    path_hash = hashlib.sha256(p.as_posix().encode("utf-8")).hexdigest()[:8]
    return f"{clean_stem}_{path_hash}"


def _find_vocal_candidates(directory: Path) -> List[Path]:
    """Find any generated vocal audio files in directory."""
    candidates: List[Path] = []
    if not directory.exists():
        return candidates
    negative_terms = (
        "no_vocals",
        "no-vocals",
        "novocals",
        "non_vocal",
        "non-vocal",
        "instrumental",
        "accompaniment",
        "other",
        "drums",
        "bass",
    )
    for f in directory.rglob("*"):
        if f.is_file():
            name_lower = f.name.lower()
            if any(neg in name_lower for neg in negative_terms):
                continue
            if "vocal" in name_lower and f.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a"}:
                candidates.append(f)

    # Prioritize exact "vocals.wav" or "(vocals).wav", then sort by mtime
    def _vocal_rank(p: Path) -> tuple[int, float]:
        name_stem = p.stem.lower()
        is_exact = 0 if name_stem in {"vocals", "vocal"} or name_stem.endswith("(vocals)") else 1
        mtime = p.stat().st_mtime if p.exists() else 0.0
        return (is_exact, -mtime)

    candidates.sort(key=_vocal_rank)
    return candidates


def _find_instrumental_candidates(directory: Path) -> List[Path]:
    """Find any generated instrumental/background audio files in directory."""
    candidates: List[Path] = []
    if not directory.exists():
        return candidates
    inst_terms = (
        "instrumental",
        "no_vocals",
        "no-vocals",
        "novocals",
        "non_vocal",
        "non-vocal",
        "accompaniment",
        "other",
    )
    for f in directory.rglob("*"):
        if f.is_file():
            name_lower = f.name.lower()
            if "vocal" in name_lower and not any(
                neg in name_lower for neg in ("no_vocals", "no-vocals", "novocals", "non_vocal", "non-vocal")
            ):
                continue
            if any(term in name_lower for term in inst_terms) and f.suffix.lower() in {".wav", ".flac", ".mp3", ".m4a"}:
                candidates.append(f)

    def _inst_rank(p: Path) -> tuple[int, float]:
        name_stem = p.stem.lower()
        is_exact = 0 if name_stem in {"instrumental", "no_vocals", "accompaniment"} or name_stem.endswith("(instrumental)") else 1
        mtime = p.stat().st_mtime if p.exists() else 0.0
        return (is_exact, -mtime)

    candidates.sort(key=_inst_rank)
    return candidates


def normalize_vocal_stem(
    wav_path: Union[str, Path],
    target_lufs: float = -18.0,
    peak_limit_db: float = -1.0,
) -> Path:
    """
    Applies static EBU R128 loudness normalization to an isolated vocal stem.
    Ensures speech signals reach nominal energy levels for accurate forced alignment
    without clipping or dynamic compression.
    """
    p = Path(wav_path).resolve()
    if not p.exists() or p.stat().st_size == 0:
        return p

    try:
        from video_assembler import normalize_audio_ebur128
        normalize_audio_ebur128(
            input_wav=p,
            output_wav=p,
            target_i=target_lufs,
            target_tp=peak_limit_db,
        )
    except Exception as e:
        print(f"[!] Warning: Could not normalize vocal stem {p}: {e}")

    return p


def isolate_vocals(
    source_path: Union[str, Path],
    output_dir: Union[str, Path] = Path(".build/vocals"),
    model_name: str = "UVR-MDX-NET-Inst_HQ_4.onnx",
    fallback_to_demucs: bool = True,
    force: bool = False,
) -> Path:
    """
    Isolate vocals from a video or audio file.

    1. Checks cache at <output_dir>/<source_id>/vocals.wav (or <output_dir>/vocals.wav).
       If existing and non-empty, immediately returns cached path (unless force=True).
    2. Runs audio-separator CLI with the requested model.
    3. If audio-separator fails or is not found, falls back to Demucs CLI (if fallback_to_demucs=True).
    4. Caches resulting vocal stem as vocals.wav and writes metadata.json provenance.
    5. Returns Path to the isolated vocals.wav.
    """
    src_p = Path(source_path).resolve()
    if not src_p.exists():
        raise FileNotFoundError(f"Source file not found: {src_p}")

    out_base = Path(output_dir).resolve()
    source_id = get_source_id(src_p)

    # Determine stem directory
    # If out_base is already named after source_id, or contains vocals.wav directly:
    if out_base.name == source_id or (out_base / "vocals.wav").exists():
        target_dir = out_base
    else:
        target_dir = out_base / source_id

    stem_path = target_dir / "vocals.wav"
    metadata_path = target_dir / "metadata.json"

    # Fast cache check with source content hash validation (R04)
    if not force:
        def _is_cache_valid(vocal_file: Path, meta_file: Path) -> bool:
            if not vocal_file.exists() or vocal_file.stat().st_size == 0:
                return False
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                    if meta.get("role") and meta.get("role") != "vocals":
                        return False
                    cached_hash = meta.get("source_sha256")
                    if cached_hash:
                        current_hash = compute_file_sha256(src_p)
                        if cached_hash != current_hash:
                            print(
                                f"[*] Source hash mismatch for {src_p} "
                                f"({cached_hash} != {current_hash}). Stale vocal stem invalidated."
                            )
                            return False
                except Exception as e:
                    print(f"[*] Corrupt stem metadata {meta_file} ({e}). Stale vocal stem invalidated.")
                    return False
            return True

        if _is_cache_valid(stem_path, metadata_path):


            print(f"[+] Using cached vocal stem: {stem_path}")
            return stem_path
        direct_stem = out_base / "vocals.wav"
        direct_meta = out_base / "metadata.json"
        if direct_stem != stem_path and _is_cache_valid(direct_stem, direct_meta):
            print(f"[+] Using cached vocal stem: {direct_stem}")
            return direct_stem

    # Forced rebuild or invalidated cache: clean up all stems and metadata (R04)
    for to_remove in [stem_path, target_dir / "instrumental.wav", metadata_path]:
        if to_remove.exists():
            try:
                to_remove.unlink()
            except Exception:
                pass

    target_dir.mkdir(parents=True, exist_ok=True)
    tool_used = ""
    resolved_model = model_name

    # Step 1: Try audio-separator CLI
    audio_sep_exe = shutil.which("audio-separator")
    if not audio_sep_exe:
        py_dir = Path(sys.executable).parent
        cand = py_dir / ("audio-separator.exe" if sys.platform == "win32" else "audio-separator")
        if cand.exists():
            audio_sep_exe = str(cand)
        else:
            audio_sep_exe = "audio-separator"

    audio_sep_cmd = [
        audio_sep_exe,
        str(src_p),
        "--model_filename",
        model_name,
        "--output_dir",
        str(target_dir),
        "--output_format",
        "WAV",
    ]

    audio_sep_success = False
    try:
        print(f"[*] Running audio-separator: {' '.join(audio_sep_cmd)}")
        res = subprocess.run(
            audio_sep_cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0:
            candidates = _find_vocal_candidates(target_dir)
            other_candidates = [c for c in candidates if c != stem_path]
            if other_candidates:
                shutil.copy2(other_candidates[0], stem_path)
                audio_sep_success = True
                tool_used = "audio-separator"
            elif stem_path.exists() and stem_path.stat().st_size > 0:
                audio_sep_success = True
                tool_used = "audio-separator"
            else:
                print(f"[!] audio-separator succeeded with code 0 but no vocal stem was located in {target_dir}")
        else:
            print(f"[!] audio-separator failed with return code {res.returncode}: {res.stderr.strip() or res.stdout.strip()}")
    except FileNotFoundError:
        print("[!] audio-separator CLI not found in PATH")
    except Exception as e:
        print(f"[!] Unexpected error running audio-separator: {e}")

    # Step 2: Try Demucs fallback if audio-separator was unsuccessful
    if not audio_sep_success and fallback_to_demucs:
        demucs_exe = shutil.which("demucs")
        if not demucs_exe:
            py_dir = Path(sys.executable).parent
            cand = py_dir / ("demucs.exe" if sys.platform == "win32" else "demucs")
            demucs_exe = str(cand) if cand.exists() else "demucs"

        demucs_cmd = [
            demucs_exe,
            "--two-stems=vocals",
            "-n",
            "htdemucs",
            str(src_p),
            "-o",
            str(target_dir),
        ]
        try:
            print(f"[*] Falling back to Demucs: {' '.join(demucs_cmd)}")
            res = subprocess.run(
                demucs_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0:
                candidates = _find_vocal_candidates(target_dir)
                other_candidates = [c for c in candidates if c != stem_path]
                if other_candidates:
                    shutil.copy2(other_candidates[0], stem_path)
                    audio_sep_success = True
                    tool_used = "demucs"
                    resolved_model = "htdemucs"
                elif stem_path.exists() and stem_path.stat().st_size > 0:
                    audio_sep_success = True
                    tool_used = "demucs"
                    resolved_model = "htdemucs"
                else:
                    print(f"[!] demucs succeeded with code 0 but no vocal stem was located in {target_dir}")
            else:
                print(f"[!] demucs failed with return code {res.returncode}: {res.stderr.strip() or res.stdout.strip()}")
        except FileNotFoundError:
            print("[!] demucs CLI not found in PATH")
        except Exception as e:
            print(f"[!] Unexpected error running demucs: {e}")

    # Step 3: Check if stem was successfully produced
    if not audio_sep_success or not stem_path.exists() or stem_path.stat().st_size == 0:
        raise RuntimeError(
            f"Vocal separation failed for '{src_p}': neither 'audio-separator' nor 'demucs' "
            "was available or succeeded. Ensure 'audio-separator' or 'demucs' is installed and accessible."
        )

    # Copy instrumental stem if produced by separator
    inst_stem_path = target_dir / "instrumental.wav"
    inst_candidates = _find_instrumental_candidates(target_dir)
    other_inst = [c for c in inst_candidates if c != inst_stem_path]
    if other_inst and not inst_stem_path.exists():
        try:
            shutil.copy2(other_inst[0], inst_stem_path)
        except Exception as e:
            print(f"[!] Warning: Failed to copy instrumental stem: {e}")

    # Save provenance metadata
    try:
        file_hash = compute_file_sha256(src_p)
    except Exception:
        file_hash = ""

    metadata: Dict[str, Any] = {
        "source_id": source_id,
        "source_path": str(src_p),
        "source_sha256": file_hash,
        "separator_tool": tool_used,
        "model_name": resolved_model,
        "isolated_vocals_path": str(stem_path),
        "isolated_instrumental_path": str(inst_stem_path) if inst_stem_path.exists() else "",
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # Normalize vocal stem for optimal forced alignment energy
    normalize_vocal_stem(stem_path)

    print(f"[+] Vocal isolation complete: {stem_path} (via {tool_used})")
    return stem_path


def main():
    parser = argparse.ArgumentParser(description="Isolate vocal track from video/audio using audio-separator or Demucs.")
    parser.add_argument("--input", "-i", required=True, help="Input video or audio file")
    parser.add_argument("--output-dir", "-o", default=".build/vocals", help="Output directory for vocal stems")
    parser.add_argument("--model", default="UVR-MDX-NET-Inst_HQ_4.onnx", help="Model filename for audio-separator")
    parser.add_argument("--no-fallback", action="store_true", help="Do not fallback to Demucs if audio-separator fails")
    parser.add_argument("--force", action="store_true", help="Force re-separation even if cached stem exists")
    args = parser.parse_args()

    vocal_stem = isolate_vocals(
        source_path=Path(args.input),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        fallback_to_demucs=not args.no_fallback,
        force=args.force,
    )
    print(str(vocal_stem))


if __name__ == "__main__":
    main()
