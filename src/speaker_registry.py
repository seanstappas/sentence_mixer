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

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np


@dataclass
class SpeakerProfile:
    speaker_id: str
    display_name: str
    embedding_centroid: List[float] = field(default_factory=list)
    sample_count: int = 0
    exemplar_audio_paths: List[str] = field(default_factory=list)
    source_ids: List[str] = field(default_factory=list)

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
            source_ids=list(data.get("source_ids", [])),
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

    def list_speakers(self) -> List[SpeakerProfile]:
        return list(self.speakers.values())

    def match_cluster(
        self,
        cluster_centroid: np.ndarray,
        threshold: Optional[float] = None,
        ambiguity_margin: float = 0.05,
    ) -> Tuple[Optional[str], float]:
        """
        Compares a cluster centroid against known speaker profiles using cosine similarity.
        Returns (best_speaker_id, best_score) if score >= threshold and (best_score - runner_up_score) >= ambiguity_margin.
        If the runner-up similarity is within ambiguity_margin, returns (None, best_score) to prevent ambiguous misclassification.
        """
        if cluster_centroid.ndim > 1:
            cluster_centroid = cluster_centroid.flatten()
        norm = np.linalg.norm(cluster_centroid)
        if norm > 1e-8:
            cluster_centroid = cluster_centroid / norm

        if threshold is not None:
            thresh = threshold
        elif len(cluster_centroid) == 512:
            # Calibrated threshold for 512-dim WavLM-SV speaker verification embeddings:
            # Intra-speaker similarity is >= 0.85; inter-speaker similarity for same-gender voices is ~0.80 - 0.81.
            # Using 0.84 prevents misclassifying distinct characters (e.g. Mas Amedda vs Bail Organa) as the same speaker.
            thresh = max(self.similarity_threshold, 0.84)
        else:
            thresh = self.similarity_threshold

        scores: List[Tuple[str, float]] = []
        for spk_id, profile in self.speakers.items():
            ref_arr = profile.centroid_array()
            if ref_arr is None or len(ref_arr) != len(cluster_centroid):
                continue
            sim = float(np.dot(cluster_centroid, ref_arr))
            scores.append((spk_id, sim))

        if not scores:
            return None, -1.0

        scores.sort(key=lambda x: x[1], reverse=True)
        best_id, best_score = scores[0]

        if best_score < thresh:
            return None, best_score

        if len(scores) > 1:
            runner_up_score = scores[1][1]
            if (best_score - runner_up_score) < ambiguity_margin:
                # Ambiguous tie: margin too narrow to safely attribute identity
                return None, best_score

        return best_id, best_score

    def register_or_update(
        self,
        speaker_id: str,
        display_name: Optional[str] = None,
        centroid: Optional[np.ndarray] = None,
        sample_count: int = 1,
        exemplar_path: Optional[str] = None,
        source_id: Optional[str] = None,
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
            if source_id and source_id not in prof.source_ids:
                prof.source_ids.append(source_id)
        else:
            prof = SpeakerProfile(
                speaker_id=speaker_id,
                display_name=display_name or speaker_id,
                embedding_centroid=c_arr.tolist() if c_arr is not None else [],
                sample_count=sample_count,
                exemplar_audio_paths=[exemplar_path] if exemplar_path else [],
                source_ids=[source_id] if source_id else [],
            )
            self.speakers[speaker_id] = prof

        self.save()
        return prof

    def rank_candidates(
        self,
        cluster_centroid: np.ndarray,
    ) -> List[Tuple[str, float]]:
        """
        Ranks all enrolled speaker profiles in self.speakers by cosine similarity
        against cluster_centroid in descending order of similarity.
        """
        if cluster_centroid.ndim > 1:
            cluster_centroid = cluster_centroid.flatten()
        norm = np.linalg.norm(cluster_centroid)
        if norm > 1e-8:
            cluster_centroid = cluster_centroid / norm

        scores: List[Tuple[str, float]] = []
        for spk_id, profile in self.speakers.items():
            ref_arr = profile.centroid_array()
            if ref_arr is None or len(ref_arr) != len(cluster_centroid):
                continue
            sim = float(np.dot(cluster_centroid, ref_arr))
            scores.append((spk_id, sim))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def prompt_or_register_cluster(
        self,
        cluster_id: str,
        centroid: Optional[np.ndarray] = None,
        turn_count: int = 1,
        duration_sec: float = 0.0,
        audio_preview_path: Optional[Union[str, Path]] = None,
        dialogue_snippet: Optional[str] = None,
        interactive: bool = True,
        default_prefix: str = "speaker",
        *,
        cluster_embedding: Optional[np.ndarray] = None,
        group_label: Optional[str] = None,
        clips: Optional[List[Dict[str, Any]]] = None,
    ) -> str:
        """
        Attempts to match cluster against known profiles. If unrecognized:
        - In interactive mode (TTY), displays turn metrics, audio preview path,
          dialogue snippet, and ranked candidate profiles, prompting for selection or naming.
        - In non-interactive or non-TTY mode, cleanly falls back to scoped anonymous ID.
        """
        if centroid is None and cluster_embedding is not None:
            centroid = cluster_embedding
        if centroid is None:
            raise ValueError("Either centroid or cluster_embedding must be provided to prompt_or_register_cluster")

        matched_id, score = self.match_cluster(centroid)
        if matched_id:
            label_suffix = f" ({group_label})" if group_label else ""
            print(f"[*] Cluster {cluster_id}{label_suffix} matched known speaker: '{matched_id}' (cosine similarity: {score:.3f})")
            # Read-only match: do NOT update the centroid. The stored profile
            # must not drift just because this cluster happened to be the nearest
            # match; that would contaminate long-term identity across recordings.
            return matched_id

        # Determine default ID
        default_id = (
            cluster_id
            if cluster_id.startswith("anon:") or cluster_id.startswith(default_prefix)
            else f"{default_prefix}_{len(self.speakers) + 1:02d}"
        )

        ranked_candidates = self.rank_candidates(centroid)

        assigned_id = None
        assigned_name = None

        if interactive:
            label_suffix = f" [{group_label}]" if group_label else ""
            file_uri = None
            if audio_preview_path is not None:
                p_path = Path(audio_preview_path)
                try:
                    file_uri = f"file:///{p_path.resolve().as_posix()}"
                except Exception:
                    file_uri = str(p_path)

            use_rich = False
            try:
                from rich.console import Console
                from rich.panel import Panel
                from rich.table import Table
                from rich import box
                use_rich = True
            except ImportError:
                pass

            if use_rich:
                panel_lines = []
                panel_lines.append(
                    f"[bold white]Turns:[/] [bold cyan]{turn_count} speech turns[/]    [dim]|[/]    [bold white]Duration:[/] [bold green]{duration_sec:.1f}s[/]"
                )
                if file_uri:
                    panel_lines.append(f"[bold white]Audio Preview:[/] [bold bright_blue underline]{file_uri}[/]")

                if clips:
                    panel_lines.append("")
                    panel_lines.append(f"[bold bright_white]Speech clips in this video ({len(clips)}):[/]")
                    non_empty_diags = [c.get("dialogue") for c in clips if c.get("dialogue")]
                    all_same_diag = len(set(non_empty_diags)) <= 1 if non_empty_diags else False

                    display_clips = clips if len(clips) <= 10 else (clips[:8] + [None] + [clips[-1]])
                    for c in display_clips:
                        if c is None:
                            panel_lines.append(f"    [dim]... and {len(clips) - 9} more clip(s)[/]")
                            continue
                        m_s = int(c["start_sec"] // 60)
                        s_s = c["start_sec"] % 60
                        m_e = int(c["end_sec"] // 60)
                        s_e = c["end_sec"] % 60
                        ts_str = f"[{m_s:02d}:{s_s:04.1f} - {m_e:02d}:{s_e:04.1f}]"
                        dur_str = f"({c['duration_sec']:.1f}s)"
                        if c.get("dialogue"):
                            diag_str = f': [bold bright_yellow]"{c["dialogue"]}"[/]'
                        else:
                            diag_str = ": [dim italic][Uncaptioned speech][/]"
                        panel_lines.append(f"  [cyan]• Clip {c['index']}[/] [dim]{ts_str}[/] [green]{dur_str}[/]{diag_str}")

                    if dialogue_snippet:
                        diag_label = "Full dialogue" if (all_same_diag and len(clips) > 1) else "Dialogue"
                        panel_lines.append("")
                        panel_lines.append(f'[bold bright_white]{diag_label}:[/] [bold bright_yellow]"{dialogue_snippet}"[/]')
                elif dialogue_snippet:
                    panel_lines.append("")
                    panel_lines.append(f'[bold bright_white]Dialogue:[/] [bold bright_yellow]"{dialogue_snippet}"[/]')

                title_str = f"[bold bright_yellow]Unrecognized Speaker Cluster[/] [dim]•[/] [bold bright_cyan]{group_label or cluster_id}[/]"
                panel = Panel(
                    "\n".join(panel_lines),
                    title=title_str,
                    border_style="bright_cyan",
                    box=box.ROUNDED,
                    safe_box=True,
                    padding=(1, 2),
                )
                r_console = Console(safe_box=True)
                r_console.print(panel)

                if ranked_candidates:
                    cand_table = Table(
                        title="[bold bright_white]Candidate Registered Speakers[/]",
                        box=box.SIMPLE,
                        show_header=True,
                        header_style="bold bright_cyan",
                        safe_box=True,
                    )
                    cand_table.add_column("Key", style="bold cyan", justify="center")
                    cand_table.add_column("Speaker ID", style="bold white")
                    cand_table.add_column("Display Name", style="white")
                    cand_table.add_column("Match Score", justify="right")
                    cand_table.add_column("Origin / Sources", style="dim")

                    for idx, (cand_id, cand_score) in enumerate(ranked_candidates, start=1):
                        prof = self.speakers.get(cand_id)
                        disp = prof.display_name if (prof and prof.display_name) else cand_id
                        if cand_score >= 0.80:
                            score_style = f"[bold green]{cand_score:.2f} ({int(cand_score*100)}%)[/]"
                        elif cand_score >= 0.60:
                            score_style = f"[bold yellow]{cand_score:.2f} ({int(cand_score*100)}%)[/]"
                        else:
                            score_style = f"[dim]{cand_score:.2f} ({int(cand_score*100)}%)[/]"

                        src_summary = ""
                        if prof and getattr(prof, "source_ids", None):
                            src_summary = ", ".join(prof.source_ids[:2])
                            if len(prof.source_ids) > 2:
                                src_summary += f" +{len(prof.source_ids)-2}"

                        cand_table.add_row(f"[{idx}]", cand_id, disp, score_style, src_summary)

                    r_console.print(cand_table)
                    prompt_str = (
                        f"Select candidate [1-{len(ranked_candidates)}], enter new speaker name, "
                        f"or press Enter for anonymous [default: {default_id}]: "
                    )
                else:
                    prompt_str = (
                        f"Enter new speaker name, or press Enter for anonymous [default: {default_id}]: "
                    )
            else:
                print(f"\n[?] Unrecognized speaker cluster detected{label_suffix}:")
                print(f"    - Turns: {turn_count} speech turns")
                print(f"    - Total duration: {duration_sec:.1f}s")
                if file_uri:
                    print(f"    - Audio Preview: {file_uri}")

                if clips:
                    print(f"    - Speech clips in this video ({len(clips)}):")
                    non_empty_diags = [c.get("dialogue") for c in clips if c.get("dialogue")]
                    all_same_diag = len(set(non_empty_diags)) <= 1 if non_empty_diags else False

                    display_clips = clips if len(clips) <= 10 else (clips[:8] + [None] + [clips[-1]])
                    for c in display_clips:
                        if c is None:
                            print(f"        ... and {len(clips) - 9} more clip(s)")
                            continue
                        m_s = int(c["start_sec"] // 60)
                        s_s = c["start_sec"] % 60
                        m_e = int(c["end_sec"] // 60)
                        s_e = c["end_sec"] % 60
                        ts_str = f"[{m_s:02d}:{s_s:04.1f} - {m_e:02d}:{s_e:04.1f}]"
                        dur_str = f"({c['duration_sec']:.1f}s)"
                        diag_str = ""
                        if not all_same_diag and c.get("dialogue"):
                            diag_str = f': "{c["dialogue"]}"'
                        print(f"        • Clip {c['index']} {ts_str} {dur_str}{diag_str}")

                    if dialogue_snippet:
                        diag_label = "Full dialogue" if (all_same_diag and len(clips) > 1) else "Dialogue"
                        print(f'    - {diag_label}: "{dialogue_snippet}"')
                elif dialogue_snippet:
                    print(f'    - Dialogue: "{dialogue_snippet}"')

                if ranked_candidates:
                    print("    Candidate speakers:")
                    for idx, (cand_id, cand_score) in enumerate(ranked_candidates, start=1):
                        prof = self.speakers.get(cand_id)
                        details: List[str] = []
                        if prof:
                            if prof.display_name and prof.display_name.lower() != cand_id.lower():
                                details.append(f'"{prof.display_name}"')
                            if getattr(prof, "source_ids", None):
                                src_summary = ", ".join(prof.source_ids[:2])
                                if len(prof.source_ids) > 2:
                                    src_summary += f" +{len(prof.source_ids)-2}"
                                details.append(f"from {src_summary}")
                        extra = f" | {' '.join(details)}" if details else ""
                        print(f"      [{idx}] {cand_id} ({cand_score:.2f}){extra}")
                    prompt_str = (
                        f"    Select candidate [1-{len(ranked_candidates)}], enter new speaker name, "
                        f"or press Enter for anonymous [default: {default_id}]: "
                    )
                else:
                    prompt_str = (
                        f"    Enter new speaker name, or press Enter for anonymous [default: {default_id}]: "
                    )

            try:
                user_val = input(prompt_str).strip()
                if user_val:
                    if user_val.isdigit():
                        cand_idx = int(user_val)
                        if 1 <= cand_idx <= len(ranked_candidates):
                            selected_id = ranked_candidates[cand_idx - 1][0]
                            assigned_id = selected_id
                            assigned_name = self.speakers[selected_id].display_name
                        else:
                            assigned_id = user_val.lower().replace(" ", "_")
                            assigned_name = user_val
                    else:
                        assigned_id = user_val.lower().replace(" ", "_")
                        assigned_name = user_val
            except (EOFError, KeyboardInterrupt, OSError):
                pass

        if assigned_id:
            try:
                from rich.console import Console
                Console(safe_box=True).print(f"[bold green][+][/] Registering speaker profile: [bold bright_white]'{assigned_id}'[/]")
            except ImportError:
                print(f"[+] Registering speaker profile: '{assigned_id}'")
            exemplar_str = str(audio_preview_path) if audio_preview_path else None
            inferred_source_id = None
            if cluster_id.startswith("anon:"):
                parts = cluster_id.split(":")
                if len(parts) >= 2:
                    inferred_source_id = parts[1]
            self.register_or_update(
                speaker_id=assigned_id,
                display_name=assigned_name,
                centroid=centroid,
                sample_count=turn_count,
                exemplar_path=exemplar_str,
                source_id=inferred_source_id,
            )
            return assigned_id
        else:
            print(f"[+] Leaving cluster unassigned in registry; using scoped anonymous ID '{default_id}'")
            return default_id

    def remove_source_speakers(
        self,
        source_ids: Union[str, List[str], Set[str]],
        matched_stems: Optional[Union[str, List[str], Set[str]]] = None,
        dry_run: bool = False,
    ) -> List[str]:
        """
        Removes or updates speaker profiles associated with the given source(s).
        If a speaker only exists for this source (or its exemplars originate from this source),
        removes the profile entirely.
        If a speaker is shared across multiple sources, retains the speaker but removes
        the cleaned source from its source_ids and exemplar_audio_paths.
        Returns the list of completely removed speaker IDs.
        """
        if isinstance(source_ids, str):
            s_set = {source_ids.lower()}
        else:
            s_set = {s.lower() for s in source_ids}

        stems_set: Set[str] = set()
        if matched_stems:
            if isinstance(matched_stems, str):
                stems_set = {matched_stems.lower()}
            else:
                stems_set = {st.lower() for st in matched_stems}

        removed_speakers: List[str] = []
        for spk_id, prof in list(self.speakers.items()):
            # Infer any legacy source IDs from exemplar audio paths
            all_sources = set(prof.source_ids)
            for ex in prof.exemplar_audio_paths:
                ex_name = Path(ex).name.lower()
                for sid in s_set:
                    if sid in ex_name or sid.replace(":", "_") in ex_name:
                        all_sources.add(sid)
                for stem in stems_set:
                    if len(stem) >= 3 and f"_{stem}_" in f"_{ex_name}_":
                        all_sources.add(stem)

            # Check if this profile matches the target source(s)
            matching_sources = {
                src for src in all_sources
                if src.lower() in s_set or any(stem in src.lower() for stem in stems_set)
            }
            if not matching_sources:
                has_matching_exemplar = False
                for ex in prof.exemplar_audio_paths:
                    ex_name = Path(ex).name.lower()
                    if any(sid in ex_name or sid.replace(":", "_") in ex_name for sid in s_set):
                        has_matching_exemplar = True
                        break
                    if any(len(st) >= 3 and f"_{st}_" in f"_{ex_name}_" for st in stems_set):
                        has_matching_exemplar = True
                        break
                if not has_matching_exemplar:
                    continue

            # Filter out cleaned sources and exemplars
            remaining_sources = [
                s for s in prof.source_ids
                if s.lower() not in s_set and not any(st in s.lower() for st in stems_set)
            ]
            remaining_exemplars = []
            for ex in prof.exemplar_audio_paths:
                ex_name = Path(ex).name.lower()
                is_target = any(sid in ex_name or sid.replace(":", "_") in ex_name for sid in s_set)
                if not is_target and stems_set:
                    is_target = any(len(st) >= 3 and f"_{st}_" in f"_{ex_name}_" for st in stems_set)
                if not is_target:
                    remaining_exemplars.append(ex)

            # If no remaining sources or exemplars, remove the speaker profile completely
            if not remaining_sources and not remaining_exemplars:
                removed_speakers.append(spk_id)
                if not dry_run:
                    del self.speakers[spk_id]
            else:
                if not dry_run:
                    prof.source_ids = remaining_sources
                    prof.exemplar_audio_paths = remaining_exemplars

        if removed_speakers and not dry_run:
            self.save()
        return removed_speakers

    def clear(self, dry_run: bool = False) -> List[str]:
        """Clears all speaker profiles from the registry and saves."""
        removed = list(self.speakers.keys())
        if not dry_run:
            self.speakers.clear()
            self.save()
        return removed

