"""
Human Prosody Guide Analysis (Milestone M5 / B10, D7).
Extracts pitch contour and relative duration guidance from a guide recording.
Strict Safety Invariant: The guide waveform is analysis-only and CANNOT enter
the output render or edit plan.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

from contracts import EditPlan
from features import ContinuousFeatureExtractor
from lexicon import normalize_text_to_words


class ProsodyGuide:
    """
    Extracts relative prosodic contours from a human guide audio file.
    Evaluates candidate divergence in relative pitch level and slope against the guide.
    """
    def __init__(self, guide_audio_path: Union[str, Path], guide_transcript: Optional[str] = None):
        self.guide_path = Path(guide_audio_path).resolve()
        if not self.guide_path.exists():
            raise FileNotFoundError(f"Guide audio file not found: {self.guide_path}")
        self.guide_transcript = guide_transcript
        self.feature_extractor = ContinuousFeatureExtractor(sr=16000)
        self.tracks = self.feature_extractor.extract_tracks(self.guide_path)
        self._cached_candidate_costs: Dict[str, float] = {}

    def get_guide_hash(self) -> str:
        """Returns SHA-256 of the guide audio file."""
        h = hashlib.sha256()
        with open(self.guide_path, "rb") as f:
            while chunk := f.read(65536):
                h.update(chunk)
        return h.hexdigest()

    def validate_transcript(self, target_text: str) -> bool:
        """
        Validates that the guide transcript words match the target sentence text words.
        Mismatches return False.
        """
        if not self.guide_transcript:
            return False
        norm_guide = [t[1] for t in normalize_text_to_words(self.guide_transcript) if t[1] not in ".,!?;:"]
        norm_target = [t[1] for t in normalize_text_to_words(target_text) if t[1] not in ".,!?;:"]
        return bool(norm_guide and norm_guide == norm_target)

    def get_relative_pitch_contour(self) -> List[Optional[float]]:
        """Returns sequence of relative F0 semitones across analysis frames."""
        f0_track = self.tracks["f0"]
        vf_track = self.tracks["voiced_flag"]
        voiced_vals = [f for f, vf in zip(f0_track, vf_track) if vf and f is not None and f > 0]

        if not voiced_vals:
            return []

        speaker_median_f0 = float(np.median(voiced_vals))
        contour: List[Optional[float]] = []
        for f, vf in zip(f0_track, vf_track):
            if vf and f is not None and f > 0:
                semitones = 12.0 * np.log2(f / speaker_median_f0)
                contour.append(float(semitones))
            else:
                contour.append(None)
        return contour

    def get_candidate_cost(
        self,
        cand: Any,
        graph: Optional[Any] = None,
        selector: Optional[Any] = None,
    ) -> float:
        """
        Computes candidate cost divergence based on the guide's relative pitch contour and slope.
        Rising guide rewards rising candidates (f0_slope > 0); falling guide rewards falling candidates.
        """
        cid = getattr(cand, "candidate_id", str(id(cand)))
        if cid in self._cached_candidate_costs:
            return self._cached_candidate_costs[cid]

        contour = self.get_relative_pitch_contour()
        if not contour:
            return 0.0

        # Determine target progress window for this candidate
        if graph and getattr(graph, "nodes", None) and len(graph.nodes) > 1:
            start_nid = getattr(cand, "target_node_start", 0)
            end_nid = getattr(cand, "target_node_end", start_nid)
            prog_start = start_nid / float(len(graph.nodes))
            prog_end = (end_nid + 1) / float(len(graph.nodes))
        else:
            prog_start, prog_end = 0.0, 1.0

        f_start = int(round(prog_start * (len(contour) - 1)))
        f_end = int(round(prog_end * (len(contour) - 1)))
        f_end = max(f_start + 1, min(len(contour), f_end + 1))

        sub_frames = [c for c in contour[f_start:f_end] if c is not None]
        if not sub_frames:
            sub_frames = [c for c in contour if c is not None]
        if not sub_frames:
            return 0.0

        guide_median_semitones = float(np.median(sub_frames))
        guide_slope = float(sub_frames[-1] - sub_frames[0]) if len(sub_frames) >= 2 else 0.0

        # Extract candidate features: f0_slope and pitch_semitones
        cand_slope = getattr(cand, "f0_slope", None)
        if cand_slope is None and selector is not None and hasattr(selector, "_get_candidate_acoustic_features"):
            try:
                s_feat, _ = selector._get_candidate_acoustic_features(
                    cand.source_content_id, cand.start_sample, cand.end_sample
                )
                cand_slope = float(s_feat)
            except Exception:
                cand_slope = 0.0
        elif cand_slope is None:
            cand_slope = 0.0

        cand_pitch = getattr(cand, "pitch_semitones", None)

        # Slope divergence cost
        # Sigmoid-shaped cost penalizing opposing slope directions
        if guide_slope > 0.5:
            # Guide is rising: reward rising slope, penalize falling slope
            cost_slope = 2.0 / (1.0 + np.exp(cand_slope / 20.0))
        elif guide_slope < -0.5:
            # Guide is falling: reward falling slope, penalize rising slope
            cost_slope = 2.0 / (1.0 + np.exp(-cand_slope / 20.0))
        else:
            cost_slope = 0.0

        # Pitch level divergence cost (if candidate explicit pitch semitones available)
        cost_pitch = 0.0
        if cand_pitch is not None:
            cost_pitch = 0.1 * abs(cand_pitch - guide_median_semitones)

        total_cost = float(cost_slope + cost_pitch)
        self._cached_candidate_costs[cid] = total_cost
        return total_cost


def assert_no_guide_audio_in_plan(plan: EditPlan, guide_path: Union[str, Path]) -> None:
    """
    Safety invariant assertion:
    Verifies that no segment in the EditPlan references the guide audio file.
    """
    guide_p = str(Path(guide_path).resolve()).lower()
    for seg in plan.segments:
        if seg.is_silence:
            continue
        seg_src = str(Path(seg.source_video_path).resolve()).lower()
        if seg_src == guide_p:
            raise AssertionError(f"Safety Violation: Guide audio leaked into PlanSegment {seg.segment_index}!")
        if seg.source_content_id.lower() == "guide":
            raise AssertionError(f"Safety Violation: Guide source_content_id found in PlanSegment {seg.segment_index}!")
