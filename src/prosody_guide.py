"""
Optional Human Prosody Guide Analysis (Milestone M6).
Extracts pitch contour and relative duration guidance from a guide recording.
Strict Safety Invariant: The guide waveform is analysis-only and CANNOT enter
the output render or edit plan.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Union

import numpy as np

from contracts import EditPlan
from features import ContinuousFeatureExtractor
from lexicon import normalize_text_to_words


class ProsodyGuide:
    """
    Extracts relative prosodic contours from a human guide audio file.
    """
    def __init__(self, guide_audio_path: Union[str, Path], guide_transcript: Optional[str] = None):
        self.guide_path = Path(guide_audio_path).resolve()
        if not self.guide_path.exists():
            raise FileNotFoundError(f"Guide audio file not found: {self.guide_path}")
        self.guide_transcript = guide_transcript
        self.feature_extractor = ContinuousFeatureExtractor(sr=16000)
        self.tracks = self.feature_extractor.extract_tracks(self.guide_path)

    def validate_transcript(self, target_text: str) -> bool:
        """
        Validates that the guide transcript matches the target sentence text.
        """
        if not self.guide_transcript:
            return True
        norm_guide = [t[1] for t in normalize_text_to_words(self.guide_transcript) if t[1] not in ".,!?;:"]
        norm_target = [t[1] for t in normalize_text_to_words(target_text) if t[1] not in ".,!?;:"]
        return norm_guide == norm_target

    def get_relative_pitch_contour(self) -> List[Optional[float]]:
        """Returns sequence of median F0 values across analysis frames."""
        f0_track = self.tracks["f0"]
        vf_track = self.tracks["voiced_flag"]
        voiced_vals = [f for f, vf in zip(f0_track, vf_track) if vf and f is not None and f > 0]

        if not voiced_vals:
            return []

        speaker_median_f0 = float(np.median(voiced_vals))
        # Relative semitones compared to guide speaker's median
        contour: List[Optional[float]] = []
        for f, vf in zip(f0_track, vf_track):
            if vf and f is not None and f > 0:
                semitones = 12.0 * np.log2(f / speaker_median_f0)
                contour.append(float(semitones))
            else:
                contour.append(None)
        return contour


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
