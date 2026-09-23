#!/usr/bin/env python3
"""
Speaker Registry for persistent speaker identity management across media sources.
Supports:
- Tracking named speaker profiles with acoustic centroid embeddings
- Cosine similarity matching for cross-clip identity classification
- Interactive TTY prompting for unassigned speaker clusters
- Atomic JSON persistence in config/speaker_registry.json
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class SpeakerProfile:
    speaker_id: str
    display_name: str
    embedding_centroid: List[float] = field(default_factory=list)
    sample_count: int = 0
    exemplar_audio_paths: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpeakerProfile":
        return cls(
            speaker_id=str(data["speaker_id"]),
            display_name=str(data.get("display_name", data["speaker_id"])),
            embedding_centroid=list(data.get("embedding_centroid", [])),
            sample_count=int(data.get("sample_count", 0)),
            exemplar_audio_paths=list(data.get("exemplar_audio_paths", [])),
        )

    def centroid_array(self) -> Optional[np.ndarray]:
        if not self.embedding_centroid:
            return None
        arr = np.array(self.embedding_centroid, dtype=np.float32)
        norm = np.linalg.norm(arr)
        if norm > 1e-8:
            arr /= norm
        return arr


class SpeakerRegistry:
    """
    Manages persistent speaker profiles and performs nearest-centroid speaker lookup.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        similarity_threshold: float = 0.76,
    ):
        self.config_path = Path(config_path or (Path(__file__).resolve().parent.parent / "config" / "speaker_registry.json"))
        self.similarity_threshold = similarity_threshold
        self.speakers: Dict[str, SpeakerProfile] = {}
        if self.config_path.exists():
            self.load()

    def load(self) -> None:
        """Loads speaker profiles from config JSON file."""
        if not self.config_path.exists():
            return
        with open(self.config_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.similarity_threshold = data.get("similarity_threshold", self.similarity_threshold)
        raw_speakers = data.get("speakers", {})
        self.speakers = {
            spk_id: SpeakerProfile.from_dict(spk_data)
            for spk_id, spk_data in raw_speakers.items()
        }

    def save(self) -> None:
        """Atomically saves speaker profiles to config JSON file."""
        self.config_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.config_path.with_suffix(".tmp")
        payload = {
            "version": "1.0",
            "similarity_threshold": self.similarity_threshold,
            "speakers": {spk_id: p.to_dict() for spk_id, p in self.speakers.items()},
        }
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, self.config_path)

    def get_speaker(self, speaker_id: str) -> Optional[SpeakerProfile]:
        return self.speakers.get(speaker_id)

    def match_cluster(
        self,
        cluster_centroid: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Compares a cluster centroid against known speaker profiles using cosine similarity.
        Returns (best_speaker_id, best_score) if score >= threshold, else (None, best_score).
        """
        thresh = threshold if threshold is not None else self.similarity_threshold
        if cluster_centroid.ndim > 1:
            cluster_centroid = cluster_centroid.flatten()
        norm = np.linalg.norm(cluster_centroid)
        if norm > 1e-8:
            cluster_centroid = cluster_centroid / norm

        best_id: Optional[str] = None
        best_score = -1.0

        for spk_id, profile in self.speakers.items():
            ref_arr = profile.centroid_array()
            if ref_arr is None or len(ref_arr) != len(cluster_centroid):
                continue
            sim = float(np.dot(cluster_centroid, ref_arr))
            if sim > best_score:
                best_score = sim
                best_id = spk_id

        if best_id is not None and best_score >= thresh:
            return best_id, best_score
        return None, best_score

    def register_or_update(
        self,
        speaker_id: str,
        display_name: Optional[str] = None,
        centroid: Optional[np.ndarray] = None,
        sample_count: int = 1,
        exemplar_path: Optional[str] = None,
    ) -> SpeakerProfile:
        """Registers a new speaker or updates an existing speaker's running centroid."""
        if centroid is not None:
            c_arr = centroid.flatten().astype(np.float32)
            c_norm = np.linalg.norm(c_arr)
            if c_norm > 1e-8:
                c_arr /= c_norm
        else:
            c_arr = None

        if speaker_id in self.speakers:
            prof = self.speakers[speaker_id]
            if display_name:
                prof.display_name = display_name
            if c_arr is not None:
                existing_arr = prof.centroid_array()
                if existing_arr is not None and len(existing_arr) == len(c_arr):
                    # Weighted running average
                    new_count = prof.sample_count + sample_count
                    updated = (existing_arr * prof.sample_count + c_arr * sample_count) / max(1, new_count)
                    norm = np.linalg.norm(updated)
                    if norm > 1e-8:
                        updated /= norm
                    prof.embedding_centroid = updated.tolist()
                    prof.sample_count = new_count
                else:
                    prof.embedding_centroid = c_arr.tolist()
                    prof.sample_count = sample_count
            if exemplar_path and exemplar_path not in prof.exemplar_audio_paths:
                prof.exemplar_audio_paths.append(exemplar_path)
        else:
            prof = SpeakerProfile(
                speaker_id=speaker_id,
                display_name=display_name or speaker_id,
                embedding_centroid=c_arr.tolist() if c_arr is not None else [],
                sample_count=sample_count,
                exemplar_audio_paths=[exemplar_path] if exemplar_path else [],
            )
            self.speakers[speaker_id] = prof

        self.save()
        return prof

    def prompt_or_register_cluster(
        self,
        cluster_id: str,
        centroid: np.ndarray,
        turn_count: int,
        duration_sec: float,
        interactive: bool = True,
        default_prefix: str = "speaker",
    ) -> str:
        """
        Attempts to match cluster against known profiles. If unrecognized:
        - In interactive mode (TTY), prompts the user for a speaker name.
        - In non-interactive mode, assigns an automatic speaker ID.
        """
        matched_id, score = self.match_cluster(centroid)
        if matched_id:
            print(f"[*] Cluster {cluster_id} matched known speaker: '{matched_id}' (cosine similarity: {score:.3f})")
            # Read-only match: do NOT update the centroid.  The stored profile
            # must not drift just because this cluster happened to be the nearest
            # match; that would contaminate long-term identity across recordings.
            return matched_id

        # Determine default ID
        default_id = cluster_id if cluster_id.startswith(default_prefix) else f"{default_prefix}_{len(self.speakers) + 1:02d}"

        assigned_id = default_id
        assigned_name = default_id

        is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        if interactive and is_tty:
            print("\n[?] Unrecognized speaker cluster detected:")
            print(f"    - Turns: {turn_count} speech turns")
            print(f"    - Total duration: {duration_sec:.1f}s")
            print(f"    - Best match: {matched_id or 'None'} (score: {score:.3f} < {self.similarity_threshold})")
            prompt_str = f"    Enter speaker identifier [default: {default_id}]: "
            try:
                user_val = input(prompt_str).strip()
                if user_val:
                    assigned_id = user_val.lower().replace(" ", "_")
                    assigned_name = user_val
            except (EOFError, KeyboardInterrupt):
                assigned_id = default_id

        print(f"[+] Registering speaker profile: '{assigned_id}'")
        self.register_or_update(
            speaker_id=assigned_id,
            display_name=assigned_name,
            centroid=centroid,
            sample_count=turn_count,
        )
        return assigned_id
