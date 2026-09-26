"""
Content-aware caching for sentence mixing pipeline.
Computes deterministic fingerprints for stages (sources, alignments, inventories, features, plans).
Ensures atomic publishing and invalidation.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Union


def hash_string(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_dict(d: Dict[str, Any]) -> str:
    serialized = json.dumps(d, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_file_content(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_media_hash(file_path: Union[str, Path], cache_dir: Optional[Union[str, Path]] = None) -> str:
    """
    Streams a full SHA-256 of the media file.
    Memoizes results by resolved path, file size, and st_mtime_ns in an index under cache root
    so unchanged large files are not re-hashed on every run.
    Raises FileNotFoundError / IOError on read failures loudly (no path-hash fallback).
    """
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Media file not found for hashing: {p}")

    stat = p.stat()
    size = stat.st_size
    mtime_ns = stat.st_mtime_ns

    index_dir = Path(cache_dir).resolve() if cache_dir else Path(".build").resolve()
    index_file = index_dir / ".media_hash_index.json"

    index: Dict[str, Any] = {}
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = {}

    posix_path = p.as_posix()
    cached_entry = index.get(posix_path)
    if (
        isinstance(cached_entry, dict)
        and cached_entry.get("size") == size
        and cached_entry.get("mtime_ns") == mtime_ns
        and isinstance(cached_entry.get("sha256"), str)
    ):
        return cached_entry["sha256"]

    # Full SHA-256 computation
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    full_sha = h.hexdigest()

    # Update index
    index[posix_path] = {
        "size": size,
        "mtime_ns": mtime_ns,
        "sha256": full_sha,
    }
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        tmp_idx = index_file.with_suffix(".tmp")
        with open(tmp_idx, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        os.replace(tmp_idx, index_file)
    except Exception:
        pass

    return full_sha


def compute_alignment_cache_key(
    media_hash: str,
    selected_audio_stream: int,
    selected_video_stream: int,
    transcript_hash: str,
    dict_hash: str,
    acoustic_model: str = "english_us_arpa",
    mfa_options: Optional[Dict[str, Any]] = None,
    preprocessor_version: str = "2.0",
) -> str:
    key_dict = {
        "media_hash": media_hash,
        "selected_audio_stream": selected_audio_stream,
        "selected_video_stream": selected_video_stream,
        "transcript_hash": transcript_hash,
        "dict_hash": dict_hash,
        "acoustic_model": acoustic_model,
        "mfa_options": mfa_options or {},
        "preprocessor_version": preprocessor_version,
    }
    return hash_dict(key_dict)


def compute_labeled_inventory_cache_key(
    alignment_key: str,
    speaker_id: Optional[str] = None,
    single_speaker_verified: bool = False,
    assume_single_speaker: bool = False,
    speaker_intervals_hash: Optional[str] = None,
    diarization_config: Optional[Dict[str, Any]] = None,
    registry_version: Optional[str] = None,
) -> str:
    key_dict = {
        "alignment_key": alignment_key,
        "speaker_id": speaker_id or "",
        "single_speaker_verified": bool(single_speaker_verified),
        "assume_single_speaker": bool(assume_single_speaker),
        "speaker_intervals_hash": speaker_intervals_hash or "",
        "diarization_config": diarization_config or {},
        "registry_version": registry_version or "",
    }
    return hash_dict(key_dict)


def compute_stem_cache_key(
    media_hash: str,
    separator_model: str = "UVR-MDX-NET-Inst_HQ_3",
    options: Optional[Dict[str, Any]] = None,
) -> str:
    key_dict = {
        "media_hash": media_hash,
        "separator_model": separator_model,
        "options": options or {},
    }
    return hash_dict(key_dict)


class CacheManager:
    """
    Manages structured project caches under .build/
    """
    def __init__(self, build_dir: Union[str, Path] = ".build"):
        self.build_dir = Path(build_dir).resolve()
        self.sources_dir = self.build_dir / "sources"
        self.alignment_dir = self.build_dir / "alignment"
        self.inventory_dir = self.build_dir / "inventory"
        self.features_dir = self.build_dir / "features"
        self.runs_dir = self.build_dir / "runs"

        for d in [self.sources_dir, self.alignment_dir, self.inventory_dir, self.features_dir, self.runs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_source_cache_dir(self, content_id: str) -> Path:
        d = self.sources_dir / content_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def compute_alignment_fingerprint(
        self,
        audio_hashes: Dict[str, str],
        transcript_content: str,
        dict_hashes: Dict[str, str],
        acoustic_model: str,
        mfa_options: Dict[str, Any],
    ) -> str:
        key_data = {
            "audio_hashes": audio_hashes,
            "transcript_content": transcript_content,
            "dict_hashes": dict_hashes,
            "acoustic_model": acoustic_model,
            "mfa_options": mfa_options,
        }
        return hash_dict(key_data)[:16]

    def compute_inventory_fingerprint(
        self,
        alignment_fingerprint: str,
        lexicon_fingerprint: str,
        schema_version: str,
    ) -> str:
        key_data = {
            "alignment_fingerprint": alignment_fingerprint,
            "lexicon_fingerprint": lexicon_fingerprint,
            "schema_version": schema_version,
        }
        return hash_dict(key_data)[:16]

    def compute_feature_fingerprint(
        self,
        source_audio_hash: str,
        feature_config: Dict[str, Any],
    ) -> str:
        key_data = {
            "source_audio_hash": source_audio_hash,
            "feature_config": feature_config,
        }
        return hash_dict(key_data)[:16]

    def atomic_publish_file(self, temp_src: Union[str, Path], target_dest: Union[str, Path]) -> Path:
        """Atomically moves/publishes a file to target_dest via a temporary rename."""
        temp_p = Path(temp_src).resolve()
        dest_p = Path(target_dest).resolve()
        dest_p.parent.mkdir(parents=True, exist_ok=True)

        temp_staging = dest_p.with_suffix(f"{dest_p.suffix}.tmp.{os.getpid()}")
        shutil.copy2(temp_p, temp_staging)
        os.replace(temp_staging, dest_p)
        return dest_p

    def atomic_publish_dir(self, temp_dir: Union[str, Path], target_dir: Union[str, Path]) -> Path:
        """Atomically replaces target_dir with temp_dir."""
        temp_d = Path(temp_dir).resolve()
        dest_d = Path(target_dir).resolve()
        dest_d.parent.mkdir(parents=True, exist_ok=True)

        if dest_d.exists():
            shutil.rmtree(dest_d)
        shutil.move(str(temp_d), str(dest_d))
        return dest_d
