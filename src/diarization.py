#!/usr/bin/env python3
"""
Acoustic Speaker Diarization & Turn Classification (Milestone M2 / R1).
Discovers, clusters, and separates speech turns from clean vocal audio.
Features:
- Operates on clean vocal stems (vocals.wav) or master audio
- Extracts normalized speaker embeddings (WavLM neural representations or acoustic MFCC fallbacks)
- Unsupervised turn clustering (AgglomerativeClustering with cosine distance)
- Persistent identity resolution through SpeakerRegistry
- Overlap detection and strict word-level single-speaker reconciliation
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

from speaker_registry import SpeakerRegistry


@dataclass
class DiarizationInterval:
    start_sec: float
    end_sec: float
    speaker_id: str
    confidence: float = 1.0
    is_overlap: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DiarizationInterval":
        return cls(
            start_sec=float(data["start_sec"]),
            end_sec=float(data["end_sec"]),
            speaker_id=str(data["speaker_id"]),
            confidence=float(data.get("confidence", 1.0)),
            is_overlap=bool(data.get("is_overlap", False)),
        )


class SpeakerDiarizer:
    """
    Performs acoustic speaker turn segmentation, embedding extraction,
    unsupervised clustering, and registry-backed identity mapping.
    """

    def __init__(
        self,
        registry: Optional[SpeakerRegistry] = None,
        distance_threshold: float = 0.28,
        min_turn_duration: float = 0.35,
        wavlm_model_id: str = "microsoft/wavlm-base",
    ):
        self.registry = registry or SpeakerRegistry()
        self.distance_threshold = distance_threshold
        self.min_turn_duration = min_turn_duration
        self.wavlm_model_id = wavlm_model_id
        self._wavlm_backend = None

    def _get_wavlm_backend(self):
        if self._wavlm_backend is None:
            try:
                from wavlm_backend import WavLMBackend, WavLMConfig, has_wavlm_dependencies
                if has_wavlm_dependencies():
                    cfg = WavLMConfig(model_id=self.wavlm_model_id)
                    self._wavlm_backend = WavLMBackend(cfg)
            except Exception as e:
                print(f"[!] Warning: Could not initialize WavLM backend for diarization: {e}. Falling back to acoustic features.")
                self._wavlm_backend = None
        return self._wavlm_backend

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """Extracts a normalized speaker embedding vector for an audio slice."""
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        backend = self._get_wavlm_backend()
        if backend is not None:
            dense = backend.extract_dense_embeddings(audio, sample_rate=sample_rate)
            vec = dense.mean(axis=0)
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
            return vec.astype(np.float32)

        # Fallback: MFCC + Delta + Spectral features via librosa/numpy
        try:
            import librosa
            if sample_rate != 16000:
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                sample_rate = 16000
            mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
            delta = librosa.feature.delta(mfcc)
            cent = librosa.feature.spectral_centroid(y=audio, sr=sample_rate)
            zcr = librosa.feature.zero_crossing_rate(y=audio)
            vec = np.concatenate([
                np.mean(mfcc, axis=1), np.std(mfcc, axis=1),
                np.mean(delta, axis=1), np.std(delta, axis=1),
                [float(np.mean(cent)), float(np.std(cent)), float(np.mean(zcr)), float(np.std(zcr))]
            ])
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
            return vec.astype(np.float32)
        except Exception:
            # Deterministic pseudo-embedding for testing when ML deps unavailable
            fft_mag = np.abs(np.fft.rfft(audio[:min(len(audio), 4096)]))
            vec = fft_mag[:64]
            norm = np.linalg.norm(vec)
            if norm > 1e-8:
                vec = vec / norm
            return vec.astype(np.float32)

    def detect_speech_turns(
        self,
        audio: np.ndarray,
        sample_rate: int,
        frame_len_sec: float = 0.05,
        energy_threshold: float = 0.015,
        min_silence_sec: float = 0.35,
    ) -> List[Tuple[float, float]]:
        """
        Segments audio into active speech turns using frame-level RMS energy and silence grouping.
        Returns list of (start_sec, end_sec) intervals.
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=1)

        hop_length = int(sample_rate * frame_len_sec)
        num_frames = max(1, len(audio) // hop_length)

        # Compute RMS energy per frame
        frames = np.array_split(audio[:num_frames * hop_length], num_frames)
        rms = np.array([float(np.sqrt(np.mean(f ** 2))) for f in frames])

        # Dynamic energy thresholding
        p25 = float(np.percentile(rms, 25))
        active_thresh = max(energy_threshold, p25 + 0.005)
        is_speech = rms >= active_thresh

        intervals: List[Tuple[float, float]] = []
        in_turn = False
        start_frame = 0
        silence_frame_count = 0
        silence_thresh_frames = int(min_silence_sec / frame_len_sec)

        for i, speech in enumerate(is_speech):
            if speech:
                if not in_turn:
                    in_turn = True
                    start_frame = i
                silence_frame_count = 0
            else:
                if in_turn:
                    silence_frame_count += 1
                    if silence_frame_count >= silence_thresh_frames:
                        end_frame = i - silence_frame_count
                        dur = (end_frame - start_frame) * frame_len_sec
                        if dur >= self.min_turn_duration:
                            intervals.append((start_frame * frame_len_sec, end_frame * frame_len_sec))
                        in_turn = False
                        silence_frame_count = 0

        if in_turn:
            end_frame = len(is_speech) - silence_frame_count
            dur = (end_frame - start_frame) * frame_len_sec
            if dur >= self.min_turn_duration:
                intervals.append((start_frame * frame_len_sec, end_frame * frame_len_sec))

        # Guard against pure silence
        if len(audio) == 0 or float(np.max(np.abs(audio))) < 1e-4:
            return []

        # Fallback: If energy threshold didn't capture segments, only partition if audio has substantial energy
        if not intervals:
            if float(np.max(np.abs(audio))) >= 0.01:
                total_dur = len(audio) / sample_rate
                if total_dur >= self.min_turn_duration:
                    intervals.append((0.0, total_dur))

        return intervals

    def diarize(
        self,
        audio_path: Union[str, Path],
        known_speaker: Optional[str] = None,
        candidate_turns: Optional[List[Tuple[float, float]]] = None,
        interactive: bool = True,
        content_id: Optional[str] = None,
    ) -> List[DiarizationInterval]:
        """
        Diarizes an audio file (typically clean vocals.wav).
        - candidate_turns: optional pre-extracted speech intervals (e.g. from VTT cues).
        - known_speaker: if supplied, forces all single-speaker turns to this speaker_id.
        """
        a_path = Path(audio_path).resolve()
        data, sr = sf.read(str(a_path))
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if content_id is None:
            try:
                from cache import compute_media_hash
                content_id = compute_media_hash(a_path)[:12]
            except Exception:
                import hashlib
                content_id = hashlib.sha256(a_path.as_posix().encode("utf-8")).hexdigest()[:12]

        # 1. Obtain turn boundaries
        turns = candidate_turns or self.detect_speech_turns(data, sr)
        if not turns:
            return []

        # 2. Extract embeddings per turn
        turn_embeddings: List[np.ndarray] = []
        valid_turns: List[Tuple[float, float]] = []

        for start_sec, end_sec in turns:
            s_idx = max(0, int(start_sec * sr))
            e_idx = min(len(data), int(end_sec * sr))
            slice_audio = data[s_idx:e_idx]
            if len(slice_audio) < int(self.min_turn_duration * sr):
                continue
            emb = self.extract_embedding(slice_audio, sample_rate=sr)
            turn_embeddings.append(emb)
            valid_turns.append((start_sec, end_sec))

        if not valid_turns:
            return []

        # 3. Unsupervised Clustering
        num_turns = len(valid_turns)
        if known_speaker is not None:
            # Caller has positively identified the speaker — register and return.
            spk_id = known_speaker
            mean_vec = np.mean(turn_embeddings, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-8:
                mean_vec /= norm
            self.registry.register_or_update(spk_id, centroid=mean_vec, sample_count=num_turns)
            return [
                DiarizationInterval(start_sec=s, end_sec=e, speaker_id=spk_id, confidence=1.0)
                for s, e in valid_turns
            ]

        if num_turns == 1:
            # Single-turn recording: run through registry matcher and prompt if unrecognized
            mean_vec = np.mean(turn_embeddings, axis=0)
            norm = np.linalg.norm(mean_vec)
            if norm > 1e-8:
                mean_vec /= norm
            matched_id, score = self.registry.match_cluster(mean_vec)
            if matched_id:
                print(
                    f"[*] Single-turn recording matched known speaker: '{matched_id}'"
                    f" (cosine similarity: {score:.3f})"
                )
                return [
                    DiarizationInterval(
                        start_sec=s, end_sec=e, speaker_id=matched_id, confidence=score
                    )
                    for s, e in valid_turns
                ]
            # No registry match -> prompt in TTY or assign anonymous ID D4
            s, e = valid_turns[0]
            cluster_id = f"anon:{content_id}:0"
            assigned_id = self.registry.prompt_or_register_cluster(
                cluster_id=cluster_id,
                centroid=mean_vec,
                turn_count=1,
                duration_sec=e - s,
                interactive=interactive,
            )
            conf = 1.0 if not assigned_id.startswith("anon:") else 0.0
            print(f"[+] Single-turn, unrecognized speaker -> assigned ID '{assigned_id}' (confidence: {conf})")
            return [
                DiarizationInterval(
                    start_sec=s, end_sec=e, speaker_id=assigned_id, confidence=conf
                )
                for s, e in valid_turns
            ]

        # Use AgglomerativeClustering on turn embeddings with cosine distance
        from sklearn.cluster import AgglomerativeClustering

        emb_matrix = np.array(turn_embeddings, dtype=np.float32)
        # Normalize each row
        row_norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        row_norms[row_norms < 1e-8] = 1.0
        emb_matrix /= row_norms

        clustering = AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=self.distance_threshold,
            n_clusters=None,
        )
        labels = clustering.fit_predict(emb_matrix)

        # 4. Map clusters to speaker identities via registry
        unique_labels = sorted(list(set(labels)))
        cluster_to_speaker: Dict[int, str] = {}

        for c_label in unique_labels:
            indices = np.where(labels == c_label)[0]
            cluster_embs = emb_matrix[indices]
            centroid = np.mean(cluster_embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-8:
                centroid /= norm

            c_turns = len(indices)
            c_dur = sum(valid_turns[idx][1] - valid_turns[idx][0] for idx in indices)

            cluster_id = f"anon:{content_id}:{c_label}"
            c_spk_id = self.registry.prompt_or_register_cluster(
                cluster_id=cluster_id,
                centroid=centroid,
                turn_count=c_turns,
                duration_sec=c_dur,
                interactive=interactive,
            )
            cluster_to_speaker[c_label] = c_spk_id

        # 5. Formulate final intervals with overlap and confidence checks
        cluster_centroids: Dict[int, np.ndarray] = {}
        for c_l in unique_labels:
            c_indices = np.where(labels == c_l)[0]
            c_embs = emb_matrix[c_indices]
            c_mean = np.mean(c_embs, axis=0)
            c_norm = np.linalg.norm(c_mean)
            if c_norm > 1e-8:
                c_mean /= c_norm
            cluster_centroids[c_l] = c_mean

        diarized_intervals: List[DiarizationInterval] = []
        for idx, (s, e) in enumerate(valid_turns):
            c_label = labels[idx]
            spk_id = cluster_to_speaker[c_label]

            # Compute margin to 2nd closest cluster (confidence)
            emb = emb_matrix[idx]
            sims = [float(np.dot(emb, cluster_centroids[c_l])) for c_l in unique_labels]
            sorted_sims = sorted(sims, reverse=True)
            conf = 1.0
            if len(sorted_sims) > 1:
                margin = sorted_sims[0] - sorted_sims[1]
                conf = max(0.5, min(1.0, 0.5 + margin * 2.0))
            if spk_id.startswith("anon:"):
                conf = 0.0

            # Check if this turn overlaps with any other turn in time
            is_ov = False
            for j, (os_sec, oe_sec) in enumerate(valid_turns):
                if j != idx and max(s, os_sec) < min(e, oe_sec):
                    is_ov = True
                    break

            diarized_intervals.append(
                DiarizationInterval(
                    start_sec=s,
                    end_sec=e,
                    speaker_id=spk_id,
                    confidence=conf,
                    is_overlap=is_ov,
                )
            )

        return diarized_intervals


def reconcile_word_intervals_with_speakers(
    diarized_intervals: List[DiarizationInterval],
    word_intervals: List[Dict[str, Any]],
    silence_intervals: Optional[List[Tuple[float, float]]] = None,
    collar_sec: float = 0.040,
) -> List[Dict[str, Any]]:
    """
    Strictly aligns word/phone boundaries against diarized speaker intervals (D5).
    - Word is eligible only if a trusted, non-overlap interval covers the whole word,
      except for an explicit boundary collar (default 40 ms) whose uncovered part
      lies in aligned silence and intersects no other speaker or overlap interval.
    - Reason codes: 'partial_coverage', 'overlap', 'unassigned', 'unreviewed_identity'.
    """
    reconciled = []
    for w in word_intervals:
        w_start = float(w["start_sec"])
        w_end = float(w["end_sec"])

        out_w = dict(w)

        # Find intersecting intervals
        matching_ivs = []
        matching_spks = set()
        has_overlap = False

        for iv in diarized_intervals:
            intersect_start = max(w_start, iv.start_sec)
            intersect_end = min(w_end, iv.end_sec)
            if intersect_start < intersect_end:
                matching_ivs.append(iv)
                matching_spks.add(iv.speaker_id)
                if iv.is_overlap or iv.speaker_id == "OVERLAP":
                    has_overlap = True

        if not matching_ivs:
            out_w["speaker_id"] = "UNKNOWN"
            out_w["is_eligible"] = False
            out_w["speaker_eligible"] = False
            out_w["eligibility_reason"] = "unassigned"
            reconciled.append(out_w)
            continue

        if has_overlap or len(matching_spks) > 1:
            out_w["speaker_id"] = "OVERLAP"
            out_w["is_eligible"] = False
            out_w["speaker_eligible"] = False
            out_w["eligibility_reason"] = "overlap"
            reconciled.append(out_w)
            continue

        # Single speaker candidate
        spk = next(iter(matching_spks))
        iv_start = min(iv.start_sec for iv in matching_ivs)
        iv_end = max(iv.end_sec for iv in matching_ivs)
        is_anon_or_unreviewed = any(
            iv.confidence == 0.0 or iv.speaker_id.startswith("anon:") for iv in matching_ivs
        )

        left_uncovered = max(0.0, iv_start - w_start)
        right_uncovered = max(0.0, w_end - iv_end)

        if left_uncovered == 0.0 and right_uncovered == 0.0:
            out_w["speaker_id"] = spk
            if is_anon_or_unreviewed:
                out_w["is_eligible"] = False
                out_w["speaker_eligible"] = False
                out_w["eligibility_reason"] = "unreviewed_identity"
            else:
                out_w["is_eligible"] = True
                out_w["speaker_eligible"] = True
                out_w["eligibility_reason"] = ""
            reconciled.append(out_w)
            continue

        # Check collar threshold
        if left_uncovered > collar_sec or right_uncovered > collar_sec:
            out_w["speaker_id"] = spk
            out_w["is_eligible"] = False
            out_w["speaker_eligible"] = False
            out_w["eligibility_reason"] = "partial_coverage"
            reconciled.append(out_w)
            continue

        # Uncovered edge within collar -> check if uncovered part lies in aligned silence
        def span_in_silence(s_start: float, s_end: float) -> bool:
            if s_end <= s_start:
                return True
            if w.get("uncovered_in_silence") or w.get("aligned_silence_collar"):
                return True
            if not silence_intervals:
                return False
            dur = s_end - s_start
            cov = 0.0
            for sil_s, sil_e in silence_intervals:
                c_s = max(s_start, sil_s)
                c_e = min(s_end, sil_e)
                if c_s < c_e:
                    cov += (c_e - c_s)
            return cov >= (dur - 1e-4)

        left_ok = span_in_silence(w_start, iv_start)
        right_ok = span_in_silence(iv_end, w_end)

        if left_ok and right_ok:
            out_w["speaker_id"] = spk
            if is_anon_or_unreviewed:
                out_w["is_eligible"] = False
                out_w["speaker_eligible"] = False
                out_w["eligibility_reason"] = "unreviewed_identity"
            else:
                out_w["is_eligible"] = True
                out_w["speaker_eligible"] = True
                out_w["eligibility_reason"] = ""
        else:
            out_w["speaker_id"] = spk
            out_w["is_eligible"] = False
            out_w["speaker_eligible"] = False
            out_w["eligibility_reason"] = "partial_coverage"

        reconciled.append(out_w)

    return reconciled
