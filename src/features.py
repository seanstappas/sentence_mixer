"""
Continuous acoustic feature extraction and join cost calculation.
Extracts spectral, energy, voicing, and F0 tracks over continuous analysis audio.
Preserves missing/unvoiced observations without fabrication.
"""

from __future__ import annotations

import math
from pathlib import Path
import time
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import librosa

from contracts import FeatureRecord

SPAN_FEATURES_VERSION = 1


def span_features_path(analysis_path: Union[str, Path]) -> Path:
    p = Path(analysis_path)
    return p.with_name(p.stem + ".span_features.npz")


def _span_features_worker(job: Tuple[str, float, float, int, int, int, float, float]) -> Tuple[float, float, int, Dict[str, np.ndarray]]:
    """Computes rms/centroid/mfcc/pYIN tracks for one span (top-level for process pools)."""
    path, start, end, sr, hop, frame_length, fmin, fmax = job
    file_sr = sf.info(path).samplerate
    first_frame = int(math.floor(start * sr / hop))
    a = int(round(first_frame * hop * file_sr / sr))
    b = int(round(end * file_sr))
    y, ysr = sf.read(path, start=a, stop=b, dtype="float32")
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    if ysr != sr:
        y = librosa.resample(y, orig_sr=ysr, target_sr=sr)
    if len(y) < frame_length:
        y = np.pad(y, (0, frame_length - len(y)))
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop)[0]
    cen = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=frame_length, hop_length=hop)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_length, hop_length=hop)
    f0, vf, vp = librosa.pyin(y, fmin=fmin, fmax=fmax, sr=sr, frame_length=frame_length, hop_length=hop, fill_na=None)
    n = min(len(rms), len(cen), mfcc.shape[1], len(f0))
    return start, end, first_frame, {
        "rms": rms[:n], "centroid": cen[:n], "mfcc": mfcc[:, :n], "f0": f0[:n], "voiced_flag": vf[:n], "voiced_probs": vp[:n],
    }


def precompute_span_features(
    analysis_path: Union[str, Path],
    spans: List[Tuple[float, float]],
    sr: int = 16000,
    hop_length: int = 160,
    frame_length: int = 512,
    fmin: float = 80.0,
    fmax: float = 500.0,
    workers: Optional[int] = None,
    log: Any = print,
) -> Path:
    """
    Precomputes continuous feature tracks over speech spans of a long analysis file and
    stores them as full-length frame arrays (frames outside spans: rms 0, f0 NaN,
    unvoiced). Incremental: spans already computed are skipped.
    """
    import os
    from concurrent.futures import ProcessPoolExecutor

    p = Path(analysis_path).resolve()
    out = span_features_path(p)
    duration = sf.info(str(p)).duration
    n_total = int(math.ceil(duration * sr / hop_length)) + 1

    arrays: Dict[str, np.ndarray]
    done: set = set()
    if out.exists():
        try:
            with np.load(str(out)) as data:
                if int(data.get("version", 0)) == SPAN_FEATURES_VERSION and int(data["n_frames"]) == n_total:
                    arrays = {k: data[k].copy() for k in ("rms", "centroid", "mfcc", "f0", "voiced_flag", "voiced_probs")}
                    done = {(round(float(a), 3), round(float(b), 3)) for a, b in data["spans_done"]}
                else:
                    arrays = {}
        except Exception:
            arrays = {}
    else:
        arrays = {}
    if not arrays:
        arrays = {
            "rms": np.zeros(n_total, dtype=np.float32),
            "centroid": np.zeros(n_total, dtype=np.float32),
            "mfcc": np.zeros((13, n_total), dtype=np.float32),
            "f0": np.full(n_total, np.nan, dtype=np.float32),
            "voiced_flag": np.zeros(n_total, dtype=bool),
            "voiced_probs": np.zeros(n_total, dtype=np.float32),
        }
    todo = sorted({(round(a, 3), round(b, 3)) for a, b in spans if b > a} - done)
    if not todo:
        return out

    workers = workers or max(1, min(12, (os.cpu_count() or 4) // 2))
    log(f"[*] Precomputing join features for {len(todo)} speech spans ({sum(b - a for a, b in todo) / 60:.0f} min of audio, {workers} workers)")
    jobs = [(str(p), a, b, sr, hop_length, frame_length, fmin, fmax) for a, b in todo]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for n, (a, b, f0_frame, tr) in enumerate(pool.map(_span_features_worker, jobs, chunksize=4), 1):
            m = min(len(tr["rms"]), n_total - f0_frame)
            if m <= 0:
                continue
            sl = slice(f0_frame, f0_frame + m)
            arrays["rms"][sl] = tr["rms"][:m]
            arrays["centroid"][sl] = tr["centroid"][:m]
            arrays["mfcc"][:, sl] = tr["mfcc"][:, :m]
            arrays["f0"][sl] = tr["f0"][:m]
            arrays["voiced_flag"][sl] = tr["voiced_flag"][:m]
            arrays["voiced_probs"][sl] = tr["voiced_probs"][:m]
            done.add((a, b))
            if n % 200 == 0:
                log(f"    {n}/{len(todo)} spans")
    tmp = out.with_name(out.stem + ".partial.npz")
    np.savez(
        str(tmp),
        version=SPAN_FEATURES_VERSION,
        sr=sr,
        hop_length=hop_length,
        frame_length=frame_length,
        n_frames=n_total,
        times=(np.arange(n_total) * hop_length / sr).astype(np.float32),
        spans_done=np.array(sorted(done), dtype=np.float64).reshape(-1, 2),
        **arrays,  # type: ignore[arg-type]
    )
    try:
        os.replace(tmp, out)
    except PermissionError:
        time.sleep(0.5)
        if out.exists():
            out.unlink(missing_ok=True)
        os.replace(tmp, out)
    return out


class ContinuousFeatureExtractor:
    """
    Extracts continuous acoustic features across an entire analysis WAV file in one pass.
    Files longer than LAZY_THRESHOLD_SEC are handled lazily per window.
    """

    LAZY_THRESHOLD_SEC = 1200.0
    LAZY_CONTEXT_SEC = 0.10

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

        # Long sources (whole movies): never decode or pYIN the whole file. Use tracks
        # precomputed over speech spans at ingest (precompute_span_features) when present,
        # else compute features per requested window (see get_window_features).
        try:
            info = sf.info(str(p))
            if info.duration > self.LAZY_THRESHOLD_SEC:
                span_cache = span_features_path(p)
                if span_cache.exists():
                    try:
                        with np.load(str(span_cache)) as data:
                            if int(data["hop_length"]) == self.hop_length and int(data["sr"]) == self.sr:
                                return {k: (data[k].copy() if data[k].ndim else data[k].item()) for k in data.files if k != "spans_done"}
                    except Exception:
                        pass
                return {"lazy": True, "path": str(p), "file_sr": int(info.samplerate), "n_frames": 0}
        except Exception:
            pass

        cache_path = p.with_suffix(".features.npz")
        if cache_path.exists():
            try:
                with np.load(str(cache_path), allow_pickle=True) as data:
                    return {
                        "sr": int(data["sr"]),
                        "hop_length": int(data["hop_length"]),
                        "frame_length": int(data["frame_length"]),
                        "n_frames": int(data["n_frames"]),
                        "times": data["times"].copy(),
                        "rms": data["rms"].copy(),
                        "centroid": data["centroid"].copy(),
                        "mfcc": data["mfcc"].copy(),
                        "f0": data["f0"].copy(),
                        "voiced_flag": data["voiced_flag"].copy(),
                        "voiced_probs": data["voiced_probs"].copy(),
                    }
            except Exception:
                pass

        try:
            info = sf.info(str(p))
            dur_str = f" ({info.duration:.1f}s audio)"
        except Exception:
            dur_str = ""

        print(f"[*] Extracting continuous acoustic tracks for '{p.name}'{dur_str}...")
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
        print("    Estimating continuous F0 pitch & voicing via pYIN...")
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

        tracks = {
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

        try:
            np.savez_compressed(
                str(cache_path),
                sr=tracks["sr"],
                hop_length=tracks["hop_length"],
                frame_length=tracks["frame_length"],
                n_frames=tracks["n_frames"],
                times=tracks["times"],
                rms=tracks["rms"],
                centroid=tracks["centroid"],
                mfcc=tracks["mfcc"],
                f0=tracks["f0"],
                voiced_flag=tracks["voiced_flag"],
                voiced_probs=tracks["voiced_probs"],
            )
            print(f"[+] Cached acoustic feature tracks to {cache_path.name}")
        except Exception:
            pass

        return tracks

    def sample_to_frame_index(self, sample_idx: int, native_sr: int) -> int:
        """Map native sample index to analysis frame index."""
        sec = float(sample_idx) / float(native_sr)
        return int(round(sec * self.sr / self.hop_length))

    def _window_tracks(self, lazy: Dict[str, Any], start_sample: int, end_sample: int, native_sr: int) -> Dict[str, Any]:
        """Computes feature tracks for a short window of a long file (memoized)."""
        if not hasattr(self, "_lazy_cache"):
            self._lazy_cache: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
        file_sr = int(lazy["file_sr"])
        s_sec = start_sample / float(native_sr)
        e_sec = max(end_sample / float(native_sr), s_sec + 0.01)
        ctx = self.LAZY_CONTEXT_SEC
        a = max(0, int((s_sec - ctx) * file_sr))
        b = int((e_sec + ctx) * file_sr)
        key = (lazy["path"], a, b)
        if key not in self._lazy_cache:
            y, sr = sf.read(lazy["path"], start=a, stop=b, dtype="float32")
            if y.ndim > 1:
                y = np.mean(y, axis=1)
            if sr != self.sr:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sr)
            if len(y) < self.frame_length:
                y = np.pad(y, (0, self.frame_length - len(y)))
            rms = librosa.feature.rms(y=y, frame_length=self.frame_length, hop_length=self.hop_length)[0]
            centroid = librosa.feature.spectral_centroid(y=y, sr=self.sr, n_fft=self.frame_length, hop_length=self.hop_length)[0]
            mfcc = librosa.feature.mfcc(y=y, sr=self.sr, n_mfcc=13, n_fft=self.frame_length, hop_length=self.hop_length)
            f0, vf, vp = librosa.pyin(
                y, fmin=self.fmin, fmax=self.fmax, sr=self.sr,
                frame_length=self.frame_length, hop_length=self.hop_length, fill_na=None,
            )
            n = min(len(rms), len(centroid), mfcc.shape[1], len(f0))
            self._lazy_cache[key] = {
                "n_frames": n, "rms": rms[:n], "centroid": centroid[:n], "mfcc": mfcc[:, :n],
                "f0": f0[:n], "voiced_flag": vf[:n], "voiced_probs": vp[:n],
                "origin_16k": int(round(a * self.sr / file_sr)),
            }
        tr = dict(self._lazy_cache[key])
        origin = tr["origin_16k"]
        tr["win_start"] = int(round(s_sec * self.sr)) - origin
        tr["win_end"] = int(round(e_sec * self.sr)) - origin
        return tr

    def get_window_features(
        self,
        tracks: Dict[str, Any],
        start_sample: int,
        end_sample: int,
        native_sr: int,
        source_content_id: str = "",
    ) -> FeatureRecord:
        """Extracts aggregated features and entry/exit descriptors for an interval."""
        orig_start, orig_end, orig_sr = start_sample, end_sample, native_sr
        if tracks.get("lazy"):
            tracks = self._window_tracks(tracks, start_sample, end_sample, native_sr)
            start_sample, end_sample, native_sr = tracks["win_start"], tracks["win_end"], self.sr
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
            sample_rate=orig_sr,
            start_sample=orig_start,
            end_sample=orig_end,
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
    cost += 0.15 * min(energy_diff, 3.0)

    # 2. Spectral (MFCC) distance
    if feat_exit.mfcc_13 and feat_entry.mfcc_13:
        m1 = np.array(feat_exit.mfcc_13[1:])  # drop C0 (energy)
        m2 = np.array(feat_entry.mfcc_13[1:])
        dist = float(np.linalg.norm(m1 - m2) / (np.linalg.norm(m1) + np.linalg.norm(m2) + 1e-6))
        cost += 0.5 * dist

    # 3. Pitch (F0) continuity
    if feat_exit.f0_hz is not None and feat_entry.f0_hz is not None:
        # Both voiced: compare semitones with steep penalty for larger melodic jumps
        f1 = feat_exit.f0_hz
        f2 = feat_entry.f0_hz
        semitone_diff = abs(12.0 * math.log2(f2 / f1))
        if semitone_diff <= 2.0:
            cost += 0.15 * (semitone_diff / 2.0)
        else:
            cost += 0.15 + 0.4 * min(semitone_diff - 2.0, 10.0)
    elif feat_exit.voicing_state != feat_entry.voicing_state:
        # Transition between voiced and unvoiced: moderate transition penalty
        cost += 0.25

    return cost
