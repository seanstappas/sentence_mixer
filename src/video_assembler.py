#!/usr/bin/env python3
"""
Video & Audio Assembler for Sentence Mixing
Renders synchronized video cuts and seamless click-free audio from selected units using FFmpeg.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from typing import List, Dict, Any

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def render_sentence_mix(
    units: List[Dict[str, Any]],
    output_mp4_path: str,
    target_fps: float = 29.97,
    audio_sample_rate: int = 48000,
    keep_temp: bool = False
) -> str:
    """
    Renders video and audio for the selected units into a single MP4 file.
    Standardizes framerate, resolution, and audio sample rate across all cuts.
    """
    if not units:
        raise ValueError("Cannot render an empty unit list!")

    os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)
    temp_dir = tempfile.mkdtemp(prefix="ytp_mix_")
    print(f"[*] Rendering {len(units)} units via FFmpeg (temp: {temp_dir})...")

    slice_files = []
    concat_txt_path = os.path.join(temp_dir, "concat.txt")

    for idx, unit in enumerate(units):
        src_path = os.path.abspath(unit["source_video"])
        start_sec = max(0.0, float(unit["start_sec"]))
        end_sec = max(start_sec + 0.033, float(unit["end_sec"]))
        dur = end_sec - start_sec
        label = unit.get("label", f"slice_{idx}")

        slice_mp4 = os.path.join(temp_dir, f"slice_{idx:04d}.mp4")

        # Extract slice with exact timing, standardized 1080p, 29.97fps, and 48kHz stereo AAC
        # Add tiny 5ms audio fade in/out to prevent boundary popping/clicking
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{start_sec:.4f}",
            "-t", f"{dur:.4f}",
            "-i", src_path,
            "-vf", f"fps={target_fps},scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
            "-af", f"afade=t=in:ss=0:d=0.005,afade=t=out:st={max(0, dur - 0.005):.4f}:d=0.005",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "18",
            "-c:a", "aac",
            "-b:a", "192k",
            "-ar", str(audio_sample_rate),
            "-ac", "2",
            slice_mp4
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0 or not os.path.exists(slice_mp4):
            print(f"[!] Warning: Failed to render slice {idx} ({label})")
            continue

        slice_files.append(slice_mp4)

    if not slice_files:
        raise RuntimeError("No slices were successfully rendered!")

    # Write concat manifest
    with open(concat_txt_path, "w", encoding="utf-8") as cf:
        for sf in slice_files:
            # Escape path for ffmpeg concat
            escaped = sf.replace("\\", "/")
            cf.write(f"file '{escaped}'\n")

    # Concat slices into final MP4
    concat_cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", concat_txt_path,
        "-c", "copy",
        output_mp4_path
    ]
    subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Also render an audio-only master WAV alongside MP4
    audio_wav_path = os.path.splitext(output_mp4_path)[0] + ".wav"
    wav_cmd = [
        "ffmpeg", "-y",
        "-i", output_mp4_path,
        "-vn",
        "-c:a", "pcm_s16le",
        "-ar", str(audio_sample_rate),
        "-ac", "2",
        audio_wav_path
    ]
    subprocess.run(wav_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if not keep_temp:
        shutil.rmtree(temp_dir, ignore_errors=True)

    print(f"[+] Direct render complete:")
    print(f"    - Video: {output_mp4_path}")
    print(f"    - Audio: {audio_wav_path}")
    return output_mp4_path
