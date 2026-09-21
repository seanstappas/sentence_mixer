"""
Continuous acoustic feature extraction and join cost calculation.
Extracts spectral, energy, voicing, and F0 tracks over continuous analysis audio.
Preserves missing/unvoiced observations without fabrication.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import numpy as np
import soundfile as sf
import librosa

from contracts import FeatureRecord


class ContinuousFeatureExtractor:
    """
    Extracts continuous acoustic features across an entire analysis WAV file in one pass.
    """
    def __init__(
        self,
        sr: int = 16000,
        hop_length: int = 160,    # 10 ms at 16 kHz
        frame_length: int = 512,  # 32 ms at 16 kHz (cleaner pitch tracking)
        fmin: float = 80.0,       # E2
        fmax: float = 500.0,      # B4
    ):
        self.sr = sr
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.fmin = fmin
        self.fmax = fmax

    def extract_tracks(self, audio_path: Union[str, Path]) -> Dict[str, Any]:
        p = Path(audio_path).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Audio file not found: {p}")

        y, sr = sf.read(str(p))
        if y.ndim > 1:
            y = np.mean(y, axis=1)
        if sr != self.sr:
            y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)

        # 1. RMS Energy
        rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]

        # 2. Spectral Centroid
        centroid = librosa.feature.spectral_centroid(
            y=y, sr=self.sr, n_fft=self.frame_length, hop_length=self.hop_length
        )[0]

        # 3. MFCC (13 coefficients)
        mfcc = librosa.feature.mfcc(
            y=y, sr=self.sr, n_mfcc=13, n_fft=self.frame_length, hop_length=self.hop_length
        )  # shape (13, n_frames)

        # 4. F0 and Voicing using pyin
        # pyin returns f0, voiced_flag, voiced_probs
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y,
            fmin=self.fmin,
            fmax=self.fmax,
            sr=self.sr,
            frame_length=self.frame_length,
            hop_length=self.hop_length,
            fill_na=None,
        )

        n_frames = min(len(rms), len(centroid), mfcc.shape[1], len(f0))
        times = librosa.frames_to_time(np.arange(n_frames), sr=self.sr, hop_length=self.hop_length)

        return {
            "sr": self.sr,
            "hop_length": self.hop_length,
            "frame_length": self.frame_length,
            "n_frames": n_frames,
            "times": times,
            "rms": rms[:n_frames],
            "centroid": centroid[:n_frames],
            "mfcc": mfcc[:, :n_frames],
            "f0": f0[:n_frames],
            "voiced_flag": voiced_flag[:n_frames],
            "voiced_probs": voiced_probs[:n_frames],
        }

    def sample_to_frame_index(self, sample_idx: int, native_sr: int) -> int:
        """Map native sample index to analysis frame index."""
        sec = float(sample_idx) / float(native_sr)
        return int(round(sec * self.sr / self.hop_length))

    def get_window_features(
        self,
        tracks: Dict[str, Any],
        start_sample: int,
        end_sample: int,
        native_sr: int,
        source_content_id: str = "",
    ) -> FeatureRecord:
        """Extracts aggregated features and entry/exit descriptors for an interval."""
        start_frame = max(0, self.sample_to_frame_index(start_sample, native_sr))
        end_frame = min(tracks["n_frames"], self.sample_to_frame_index(end_sample, native_sr))

        if end_frame <= start_frame:
            end_frame = min(tracks["n_frames"], start_frame + 1)

        f0_slice = tracks["f0"][start_frame:end_frame]
        vf_slice = tracks["voiced_flag"][start_frame:end_frame]
        vp_slice = tracks["voiced_probs"][start_frame:end_frame]

        valid_f0 = [
            float(f) for f, vf, vp in zip(f0_slice, vf_slice, vp_slice)
            if vf and vp > 0.4 and f is not None and not np.isnan(f) and f > 0
        ]

        voicing_state: Literal["voiced", "unvoiced", "silence"]
        if valid_f0:
            avg_f0: Optional[float] = float(np.median(valid_f0))
            voicing_state = "voiced"
            confidence = float(np.mean(vp_slice)) if len(vp_slice) > 0 else 0.0
        else:
            avg_f0 = None
            voicing_state = "unvoiced"
            confidence = 0.0

        rms_slice = tracks["rms"][start_frame:end_frame]
        avg_rms = float(np.mean(rms_slice)) if len(rms_slice) > 0 else 0.0

        centroid_slice = tracks["centroid"][start_frame:end_frame]
        avg_centroid = float(np.mean(centroid_slice)) if len(centroid_slice) > 0 else 0.0

        mfcc_slice = tracks["mfcc"][:, start_frame:end_frame]
        avg_mfcc = [float(x) for x in np.mean(mfcc_slice, axis=1)] if mfcc_slice.shape[1] > 0 else [0.0] * 13

        rec = FeatureRecord(
            source_content_id=source_content_id,
            sample_rate=native_sr,
            start_sample=start_sample,
            end_sample=end_sample,
            rms_energy=avg_rms,
            f0_hz=avg_f0,
            f0_confidence=confidence,
            voicing_state=voicing_state,
            spectral_centroid=avg_centroid,
            mfcc_13=avg_mfcc,
        )
        rec.validate()
        return rec


def compute_join_cost(
    feat_exit: Optional[FeatureRecord],
    feat_entry: Optional[FeatureRecord],
    is_natural_continuation: bool = False,
) -> float:
    """
    Computes acoustic concatenation cost between candidate exit and candidate entry.
    Natural original continuations receive zero join cost.
    Handles unvoiced F0 without fabricating 120 Hz.
    """
    if is_natural_continuation:
        return 0.0

    if feat_exit is None or feat_entry is None:
        # Penalize unavailable features rather than treating as zero-cost
        return 1.5

    cost = 0.0

    # 1. Energy jump cost
    e1 = max(1e-5, feat_exit.rms_energy)
    e2 = max(1e-5, feat_entry.rms_energy)
    energy_diff = abs(math.log(e1) - math.log(e2))
    cost += 0.3 * min(energy_diff, 3.0)

    # 2. Spectral (MFCC) distance
    if feat_exit.mfcc_13 and feat_entry.mfcc_13:
        m1 = np.array(feat_exit.mfcc_13[1:])  # drop C0 (energy)
        m2 = np.array(feat_entry.mfcc_13[1:])
        dist = float(np.linalg.norm(m1 - m2) / (np.linalg.norm(m1) + np.linalg.norm(m2) + 1e-6))
        cost += 0.5 * dist

    # 3. Pitch (F0) continuity
    if feat_exit.f0_hz is not None and feat_entry.f0_hz is not None:
        # Both voiced: compare semitones
        f1 = feat_exit.f0_hz
        f2 = feat_entry.f0_hz
        semitone_diff = abs(12.0 * math.log2(f2 / f1))
        cost += 0.2 * min(semitone_diff / 12.0, 2.0)
    elif feat_exit.voicing_state != feat_entry.voicing_state:
        # Transition between voiced and unvoiced: moderate transition penalty
        cost += 0.25

    return cost
