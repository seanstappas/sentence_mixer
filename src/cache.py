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
from typing import Any, Dict, Union


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
