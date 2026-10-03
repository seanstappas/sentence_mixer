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
from typing import List, Optional, Union


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


from cache import compute_media_hash


def get_source_id(source_path: Union[str, Path]) -> str:
    """
    Generate a clean filesystem-safe source ID with content hash for caching.
    Matches the naming convention used across sentence_mixer.
    """
    p = Path(source_path).resolve()
    clean_stem = re.sub(r"[^\w\-_]", "_", p.stem)[:40]
    content_hash = compute_media_hash(p)[:8]
    return f"{clean_stem}_{content_hash}"


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


_real_subprocess_run = subprocess.run


def _run_with_progress(
    cmd: List[str],
    description: str = "Separating vocal track",
    model_name: str = "",
) -> subprocess.CompletedProcess:
    """
    Executes a separation CLI command with a live Rich progress bar tracking chunk completion.
    If subprocess.run is monkeypatched (e.g. by test suites), delegates directly to subprocess.run.
    """
    if subprocess.run is not _real_subprocess_run:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    use_rich = False
    try:
        from rich.console import Console
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TaskProgressColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )
        use_rich = True
    except ImportError:
        pass

    captured_output: List[str] = []

    if use_rich:
        console = Console(safe_box=True)
        tqdm_re = re.compile(r"(\d+)/(\d+)\s+\[([^\]]+)\]")
        pct_re = re.compile(r"(\d+)%")

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]{task.description}"),
            BarColumn(bar_width=28),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            console=console,
            transient=True,
        ) as progress:
            model_label = f" ({model_name})" if model_name else ""
            task = progress.add_task(f"Initializing {description}{model_label}...", total=100)

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in proc.stdout:
                captured_output.append(line)
                line_clean = line.strip()

                if "Loading model" in line_clean:
                    progress.update(task, description=f"Loading model{model_label}...", completed=5)
                elif "Starting separation" in line_clean:
                    progress.update(task, description=f"Separating audio chunks{model_label}...", completed=10)
                elif "Saving Vocals" in line_clean or "Saving Instrumental" in line_clean:
                    progress.update(task, description=f"Exporting vocal stem{model_label}...", completed=95)
                elif "Separation complete" in line_clean:
                    progress.update(task, description="Vocal isolation complete!", completed=100)
                else:
                    t_m = tqdm_re.search(line_clean)
                    p_m = pct_re.search(line_clean)
                    if t_m and p_m:
                        curr_it = int(t_m.group(1))
                        tot_it = int(t_m.group(2))
                        pct = int(p_m.group(1))
                        mapped_pct = min(95, 10 + int(pct * 0.85))
                        progress.update(
                            task,
                            total=100,
                            completed=mapped_pct,
                            description=f"Separating chunks ({curr_it}/{tot_it}){model_label}...",
                        )

            proc.wait()
            ret = proc.returncode
    else:
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    all_text = "".join(captured_output)
    if ret == 0:
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=all_text, stderr="")
    else:
        return subprocess.CompletedProcess(args=cmd, returncode=ret, stdout="", stderr=all_text)


def isolate_vocals(
    source_path: Union[str, Path],
    output_dir: Union[str, Path] = Path(".build/vocals"),
    model_name: str = "Kim_Vocal_2.onnx",
    fallback_to_demucs: bool = True,
    force: bool = False,
    normalize: bool = False,
    enable_denoise: bool = False,
    pre_extract_audio: bool = False,
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

    # Pre-extract clean lossless PCM audio for video containers if requested to eliminate
    # libsndfile container warnings and bypass Windows Media Foundation audioread artifacts
    video_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".flv", ".m4v"}
    input_file_for_separator = src_p
    extracted_wav_to_clean: Optional[Path] = None

    if pre_extract_audio and src_p.suffix.lower() in video_exts and shutil.which("ffmpeg"):
        extracted_wav = target_dir / f"{src_p.stem}_extracted.wav"
        try:
            cmd_extract = [
                "ffmpeg", "-y", "-nostats", "-loglevel", "error",
                "-i", str(src_p),
                "-vn",
                "-acodec", "pcm_s16le",
                "-ar", "44100",
                str(extracted_wav),
            ]
            res_ext = subprocess.run(cmd_extract, capture_output=True, check=False)
            if res_ext.returncode == 0 and extracted_wav.exists() and extracted_wav.stat().st_size > 0:
                input_file_for_separator = extracted_wav
                extracted_wav_to_clean = extracted_wav
        except Exception as e:
            print(f"[*] Note: pre-extraction skipped: {e}")
            input_file_for_separator = src_p

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
        str(input_file_for_separator),
        "--model_filename",
        model_name,
        "--output_dir",
        str(target_dir),
        "--output_format",
        "WAV",
        "--use_soundfile",
    ]
    is_mdx = any(k in model_name.lower() for k in ("mdx", "kim", "inst", "vocal", "hq"))
    if is_mdx:
        audio_sep_cmd.extend([
            "--mdx_segment_size",
            "256",
            "--mdx_overlap",
            "0.25",
        ])
        if enable_denoise:
            audio_sep_cmd.append("--mdx_enable_denoise")

    uvr_mdx_dir = Path("D:/Program Files/Ultimate Vocal Remover/models/MDX_Net_Models")
    tmp_mdx_dir = Path("/tmp/audio-separator-models")
    if not (tmp_mdx_dir / model_name).exists() and (uvr_mdx_dir / model_name).exists():
        audio_sep_cmd.extend(["--model_file_dir", str(uvr_mdx_dir)])

    audio_sep_success = False
    try:
        print(f"[*] Running audio-separator: {' '.join(audio_sep_cmd)}")
        res = _run_with_progress(
            audio_sep_cmd,
            description="audio-separator",
            model_name=model_name,
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
            res = _run_with_progress(
                demucs_cmd,
                description="Demucs",
                model_name="htdemucs",
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

    metadata = {
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

    # Clean up temporary extracted audio file if one was generated
    if extracted_wav_to_clean and extracted_wav_to_clean.exists():
        try:
            extracted_wav_to_clean.unlink()
        except Exception:
            pass

    # Optional loudness normalization (disabled by default to preserve raw clean separation)
    if normalize:
        normalize_vocal_stem(stem_path)

    print(f"[+] Vocal isolation complete: {stem_path} (via {tool_used})")
    return stem_path


def main():
    parser = argparse.ArgumentParser(description="Isolate vocal track from video/audio using audio-separator or Demucs.")
    parser.add_argument("--input", "-i", required=True, help="Input video or audio file")
    parser.add_argument("--output-dir", "-o", default=".build/vocals", help="Output directory for vocal stems")
    parser.add_argument("--model", default="Kim_Vocal_2.onnx", help="Model filename for audio-separator")
    parser.add_argument("--normalize", action="store_true", default=False, help="Apply static EBU R128 loudness normalization to vocal stem")
    parser.add_argument("--denoise", action="store_true", default=False, help="Enable dual-pass STFT denoising (disabled by default to avoid phase cancellation)")
    parser.add_argument("--pre-extract-audio", action="store_true", default=True, help="Pre-extract lossless PCM WAV for video containers")
    parser.add_argument("--no-fallback", action="store_true", help="Do not fallback to Demucs if audio-separator fails")
    parser.add_argument("--force", action="store_true", help="Force re-separation even if cached stem exists")
    args = parser.parse_args()

    vocal_stem = isolate_vocals(
        source_path=Path(args.input),
        output_dir=Path(args.output_dir),
        model_name=args.model,
        fallback_to_demucs=not args.no_fallback,
        force=args.force,
        normalize=args.normalize,
        enable_denoise=args.denoise,
        pre_extract_audio=args.pre_extract_audio,
    )
    print(str(vocal_stem))


if __name__ == "__main__":
    main()
