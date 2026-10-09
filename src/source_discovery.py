#!/usr/bin/env python3
"""
Source Discovery and Subtitle Pairing Module.

Discovers video files (.mp4, .mkv, .mov, .avi, .webm) and pairs them
with matching subtitle files (.en.vtt, .vtt, .en.srt, .srt, etc.)
using deterministic stem matching rules while ignoring build and hidden dirs.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union

SUPPORTED_VIDEO_EXTENSIONS: Set[str] = {
    ".mp4",
    ".mkv",
    ".mov",
    ".avi",
    ".webm",
}

EXCLUDED_DIR_NAMES: Set[str] = {
    ".build",
    ".mamba",
    ".venv",
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
    ".mypy_cache",
}


def should_skip_dir(dir_name: str) -> bool:
    """Check if a directory name should be skipped during folder traversal."""
    return dir_name in EXCLUDED_DIR_NAMES or dir_name.startswith(".")


def find_associated_subtitles(video_path: Union[str, Path]) -> Optional[str]:
    """
    Locate the matching subtitle file for a video file according to priority:
      1. {stem}.en.vtt
      2. {stem}.vtt
      3. {stem}.en.srt
      4. {stem}.srt
      5. {stem}.en-US.vtt
      6. {stem}.en-US.srt
      7. Any matching {stem}*.vtt, then {stem}*.srt (alphabetical)

    Returns the absolute path string of the matching subtitle, or None if not found.
    """
    v_path = Path(video_path).resolve()
    parent = v_path.parent
    if not parent.exists() or not parent.is_dir():
        return None

    stem = v_path.stem

    priority_candidates = [
        f"{stem}.en.vtt",
        f"{stem}.vtt",
        f"{stem}.en.srt",
        f"{stem}.srt",
        f"{stem}.en-US.vtt",
        f"{stem}.en-US.srt",
    ]

    # Pre-index directory entries for fast and case-insensitive resolution
    try:
        entries = list(parent.iterdir())
    except OSError:
        return None

    files_map: Dict[str, Path] = {}
    for entry in entries:
        if entry.is_file():
            files_map[entry.name.lower()] = entry.resolve()

    # 1. Exact or case-insensitive priority match
    for cand in priority_candidates:
        cand_lower = cand.lower()
        if cand_lower in files_map:
            return str(files_map[cand_lower])

    # 2. Priority 7 fallback: any {stem}*.vtt, then {stem}*.srt
    stem_lower = stem.lower()
    fallback_vtt: List[Path] = []
    fallback_srt: List[Path] = []

    for name_lower, resolved_p in files_map.items():
        if name_lower.startswith(stem_lower):
            if name_lower.endswith(".vtt"):
                fallback_vtt.append(resolved_p)
            elif name_lower.endswith(".srt"):
                fallback_srt.append(resolved_p)

    if fallback_vtt:
        fallback_vtt.sort(key=lambda p: p.name.lower())
        return str(fallback_vtt[0])

    if fallback_srt:
        fallback_srt.sort(key=lambda p: p.name.lower())
        return str(fallback_srt[0])

    return None


def discover_folder_sources(folder_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Recursively discover video files within folder_path and pair each with matching subtitles.
    Explicitly ignores build/hidden directories (.build, .mamba, .venv, .git, __pycache__).

    Returns a list of dicts:
      {
          "video": str,
          "vtt": Optional[str],
          "speaker_id": None,
          "single_speaker_verified": False,
          "recording_id": "",
          "speaker_intervals": None,
      }
    """
    f_path = Path(folder_path).resolve()
    if not f_path.exists():
        raise FileNotFoundError(f"Folder not found: {f_path}")
    if not f_path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {f_path}")

    discovered: List[Dict[str, Any]] = []

    for root, dirnames, filenames in os.walk(f_path):
        # Prune excluded directories in-place so os.walk does not descend into them
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]

        for fname in sorted(filenames):
            p = Path(root) / fname
            if p.suffix.lower() in SUPPORTED_VIDEO_EXTENSIONS:
                sub_match = find_associated_subtitles(p)
                discovered.append({
                    "video": str(p.resolve()),
                    "vtt": sub_match,
                    "speaker_id": None,
                    "single_speaker_verified": False,
                    "recording_id": "",
                    "speaker_intervals": None,
                })

    # Sort discovered items deterministically by video path
    discovered.sort(key=lambda item: item["video"])
    return discovered
