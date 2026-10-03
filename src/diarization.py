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

import subprocess
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
        distance_threshold: float = 0.45,
        min_turn_duration: float = 0.35,
        wavlm_model_id: str = "microsoft/wavlm-base",
        sv_model_id: str = "microsoft/wavlm-base-plus-sv",
    ):
        self.registry = registry or SpeakerRegistry()
        self.distance_threshold = distance_threshold
        self.min_turn_duration = min_turn_duration
        self.wavlm_model_id = wavlm_model_id
        self.sv_model_id = sv_model_id
        self._wavlm_backend = None
        self._sv_model = None
        self._sv_feature_extractor = None
        self._sv_device = None
        self._sv_initialized = False

    def _get_sv_backend(self):
        """
        Lazily loads WavLMForXVector (speaker verification model, e.g. microsoft/wavlm-base-plus-sv).
        Returns (model, feature_extractor, device) or (None, None, None).
        """
        if not self._sv_initialized:
            self._sv_initialized = True
            try:
                import torch
                from transformers import AutoFeatureExtractor, WavLMForXVector
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model_name = self.sv_model_id
                try:
                    fe = AutoFeatureExtractor.from_pretrained(model_name, local_files_only=True)
                    model = WavLMForXVector.from_pretrained(model_name, local_files_only=True).to(device)
                except Exception:
                    fe = AutoFeatureExtractor.from_pretrained(model_name)
                    model = WavLMForXVector.from_pretrained(model_name).to(device)
                model.eval()
                self._sv_model = model
                self._sv_feature_extractor = fe
                self._sv_device = device
                print(f"[*] Loaded WavLM speaker verification model: {model_name} on {device}")
            except Exception:
                self._sv_model = None
                self._sv_feature_extractor = None
                self._sv_device = None
        return self._sv_model, self._sv_feature_extractor, self._sv_device

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

        if sample_rate != 16000:
            try:
                import soxr
                audio = soxr.resample(audio, sample_rate, 16000)
            except Exception:
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=16000)
                except Exception:
                    import scipy.signal
                    num_target_samples = int(round(len(audio) * 16000 / sample_rate))
                    audio = scipy.signal.resample(audio, num_target_samples)
            sample_rate = 16000

        # Preferred: Dedicated Speaker Verification model (WavLM-SV / WavLMForXVector)
        sv_model, sv_fe, sv_dev = self._get_sv_backend()
        if sv_model is not None and sv_fe is not None:
            import torch
            chunk = audio
            chunk = (chunk - np.mean(chunk)) / (np.std(chunk) + 1e-7)
            if len(chunk) < 1600:
                chunk = np.pad(chunk, (0, 1600 - len(chunk)))
            inp = sv_fe(chunk, sampling_rate=16000, return_tensors="pt").input_values.to(sv_dev)
            with torch.no_grad():
                out = sv_model(inp)
                vec = out.embeddings.squeeze(0).cpu().numpy()
                norm = np.linalg.norm(vec)
                if norm > 1e-8:
                    vec /= norm
                return vec.astype(np.float32)

        # Fallback 1: Raw WavLM backend
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
        p75 = float(np.percentile(rms, 75))
        p90 = float(np.percentile(rms, 90))
        if p90 > p25 + 0.003:
            # Dynamically adapt threshold between noise floor (p25) and active speech
            adaptive_thresh = p25 + (p75 - p25) * 0.35
            active_thresh = max(0.002, min(energy_threshold, adaptive_thresh))
        else:
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
        subtitle_path: Optional[Union[str, Path]] = None,
        cluster_preview_dir: Optional[Union[str, Path]] = None,
    ) -> List[DiarizationInterval]:
        """
        Diarizes an audio file (typically clean vocals.wav).
        - candidate_turns: optional pre-extracted speech intervals (e.g. from VTT cues).
        - known_speaker: if supplied, forces all single-speaker turns to this speaker_id.
        - subtitle_path: optional path to VTT/SRT subtitle file used to extract dialogue
          snippets shown during interactive speaker review.
        - cluster_preview_dir: optional directory where per-cluster preview WAVs are
          written. Defaults to <audio_dir>/.build/clusters/ when not supplied.
        """
        a_path = Path(audio_path).resolve()
        video_exts = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v"}
        read_path = a_path
        if a_path.suffix.lower() in video_exts:
            cache_audio_dir = Path(".build") / "audio_extracts"
            cache_audio_dir.mkdir(parents=True, exist_ok=True)
            wav_path = cache_audio_dir / f"{a_path.stem}.wav"
            if not wav_path.exists() or wav_path.stat().st_mtime < a_path.stat().st_mtime:
                cmd = [
                    "ffmpeg", "-y", "-i", str(a_path),
                    "-vn", "-ac", "1", "-ar", "16000",
                    str(wav_path)
                ]
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            read_path = wav_path

        data, sr = sf.read(str(read_path))
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        if sr != 16000:
            try:
                import soxr
                data = soxr.resample(data, sr, 16000)
            except Exception:
                try:
                    import librosa
                    data = librosa.resample(data, orig_sr=sr, target_sr=16000)
                except Exception:
                    import scipy.signal
                    num_target_samples = int(round(len(data) * 16000 / sr))
                    data = scipy.signal.resample(data, num_target_samples)
            sr = 16000

        if content_id is None:
            try:
                from cache import compute_media_hash
                content_id = compute_media_hash(a_path)[:12]
            except Exception:
                import hashlib
                content_id = hashlib.sha256(a_path.as_posix().encode("utf-8")).hexdigest()[:12]

        # Resolve cluster preview output directory
        if cluster_preview_dir is not None:
            _preview_dir = Path(cluster_preview_dir)
        else:
            _preview_dir = a_path.parent / ".build" / "clusters"

        # Load subtitle cues for dialogue snippet extraction (best-effort)
        _sub_cues: List[Tuple[float, float, str, Optional[List[Dict[str, Any]]]]] = []
        if subtitle_path is not None:
            try:
                from vtt_preprocessor import load_accurate_vtt_cues, parse_vtt_cues
                try:
                    _raw_cues = load_accurate_vtt_cues(str(Path(subtitle_path).resolve()))
                except Exception:
                    _raw_cues = parse_vtt_cues(str(Path(subtitle_path).resolve()))
                _sub_cues = [
                    (c["start_sec"], c["end_sec"], c.get("text", "").strip(), c.get("words"))
                    for c in _raw_cues
                    if c.get("text", "").strip()
                ]
            except Exception:
                pass  # Dialogue snippets are best-effort; don't abort diarization

        def _dialogue_for_interval(t_start: float, t_end: float) -> Optional[str]:
            """Return subtitle dialogue text matching a specific speech turn interval."""
            if not _sub_cues:
                return None
            t_dur = max(0.01, t_end - t_start)
            matched: List[str] = []
            for c_start, c_end, c_text, c_words in _sub_cues:
                c_dur = max(0.01, c_end - c_start)
                overlap = min(t_end, c_end) - max(t_start, c_start)
                if overlap > 0:
                    frac_cue = overlap / c_dur
                    frac_turn = overlap / t_dur
                    if (overlap >= 0.20 and (frac_cue >= 0.15 or frac_turn >= 0.20)) or overlap >= 0.5:
                        if c_words:
                            turn_words = []
                            for w_info in c_words:
                                ws = float(w_info.get("start_sec", c_start))
                                we = float(w_info.get("end_sec", c_end))
                                if min(t_end + 0.15, we) > max(t_start - 0.15, ws):
                                    turn_words.append(w_info.get("text", ""))
                            sub_text = " ".join(turn_words).strip()
                            if sub_text and sub_text not in matched:
                                matched.append(sub_text)
                        elif c_text and c_text not in matched:
                            matched.append(c_text)
            # Only snap if the turn is short (< 0.8s) and separated by a tiny gap (<= 0.15s)
            if not matched and t_dur < 0.8:
                best_cue = None
                best_dist = 0.15
                for c_start, c_end, c_text, _ in _sub_cues:
                    raw_gap = max(t_start - c_end, c_start - t_end)
                    if 0.0 <= raw_gap <= best_dist:
                        best_dist = raw_gap
                        best_cue = c_text
                if best_cue and best_cue not in matched:
                    matched.append(best_cue)
            return " … ".join(matched) if matched else None

        def _get_clips_info(turn_list: List[Tuple[float, float]]) -> List[Dict[str, Any]]:
            """Extract structured clip metadata for every turn in a cluster."""
            chronological = sorted(turn_list, key=lambda t: t[0])
            clips = []
            for idx, (t_start, t_end) in enumerate(chronological, start=1):
                clips.append({
                    "index": idx,
                    "start_sec": t_start,
                    "end_sec": t_end,
                    "duration_sec": t_end - t_start,
                    "dialogue": _dialogue_for_interval(t_start, t_end),
                })
            return clips

        def _dialogue_snippet_for_turns(turn_list: List[Tuple[float, float]], max_chars: int = 240) -> Optional[str]:
            """Return subtitle text that overlaps with any of the speech turns (best-effort)."""
            if not _sub_cues:
                return None
            parts: List[str] = []
            total = 0
            chronological = sorted(turn_list, key=lambda t: t[0])
            for t_start, t_end in chronological:
                d = _dialogue_for_interval(t_start, t_end)
                if d and d not in parts:
                    parts.append(d)
                    total += len(d)
                if total >= max_chars:
                    break
            if not parts:
                return None
            snippet = " … ".join(parts)
            if len(snippet) > max_chars:
                snippet = snippet[: max_chars - 1] + "…"
            return snippet

        def _write_cluster_preview(
            cluster_safe_id: str,
            turn_list: List[Tuple[float, float]],
            target_duration: float = 15.0,
            max_turn_snippet: float = 2.5,
        ) -> Optional[Path]:
            """
            Write a representative preview WAV for a cluster by concatenating
            audio from ALL speech turns in chronological order with short pauses.
            Ensures the user hears all clips associated with this speaker.
            """
            try:
                _preview_dir.mkdir(parents=True, exist_ok=True)
                preview_path = _preview_dir / f"{cluster_safe_id}_preview.wav"
                chronological = sorted(turn_list, key=lambda t: t[0])
                chunks: List[np.ndarray] = []
                silence_gap = np.zeros(int(0.15 * sr), dtype=np.float32)
                accumulated = 0.0

                for t_start, t_end in chronological:
                    if accumulated >= target_duration:
                        break
                    remain = target_duration - accumulated
                    turn_dur = t_end - t_start
                    take_dur = min(turn_dur, max_turn_snippet, remain)
                    s_idx = max(0, int(t_start * sr))
                    e_idx = min(len(data), int((t_start + take_dur) * sr))
                    chunk = data[s_idx:e_idx]
                    if len(chunk) > 0:
                        if chunks:
                            chunks.append(silence_gap)
                        chunks.append(chunk)
                        accumulated += take_dur

                if not chunks:
                    return None
                preview_audio = np.concatenate(chunks).astype(np.float32)
                sf.write(str(preview_path), preview_audio, sr)
                return preview_path
            except Exception:
                return None

        # 1. Obtain turn boundaries
        turns = candidate_turns
        if not turns and _sub_cues:
            turns = [
                (c_start, c_end)
                for c_start, c_end, c_text in _sub_cues
                if (c_end - c_start) >= self.min_turn_duration
            ]
        if not turns:
            turns = self.detect_speech_turns(data, sr)
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
            cluster_safe_id = cluster_id.replace(":", "_")
            _preview_path = _write_cluster_preview(cluster_safe_id, [(s, e)])
            _snippet = _dialogue_snippet_for_turns([(s, e)])
            _clips_info = _get_clips_info([(s, e)])
            assigned_id = self.registry.prompt_or_register_cluster(
                cluster_id=cluster_id,
                centroid=mean_vec,
                turn_count=1,
                duration_sec=e - s,
                interactive=interactive,
                audio_preview_path=_preview_path,
                dialogue_snippet=_snippet,
                clips=_clips_info,
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

        # Determine calibrated distance threshold:
        # WavLM-SV x-vectors have intra-distance ~0.13 and inter-distance ~0.24, so threshold 0.18 is optimal.
        # Fallback raw WavLM / MFCC representations use threshold 0.45.
        effective_thresh = self.distance_threshold
        if getattr(self, "_sv_model", None) is not None:
            if effective_thresh is None or np.isclose(effective_thresh, 0.45):
                effective_thresh = 0.18
        elif effective_thresh is None:
            effective_thresh = 0.45

        clustering = AgglomerativeClustering(
            metric="cosine",
            linkage="average",
            distance_threshold=effective_thresh,
            n_clusters=None,
        )
        labels = clustering.fit_predict(emb_matrix)

        # 4. Map clusters to speaker identities via registry
        unique_labels = sorted(list(set(labels)))
        cluster_to_speaker: Dict[int, str] = {}

        total_speech_sec = sum(e - s for s, e in valid_turns)
        header_text = f"Diarization analyzed entire input: discovered {len(unique_labels)} speaker group(s) across {len(valid_turns)} speech turns ({total_speech_sec:.1f}s speech)"

        use_rich_summary = False
        try:
            from rich.console import Console
            from rich.table import Table
            from rich import box
            use_rich_summary = True
        except ImportError:
            pass

        if use_rich_summary:
            summary_table = Table(
                title=f"[bold bright_cyan]{header_text}[/]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold bright_white on blue",
                border_style="bright_blue",
                safe_box=True,
            )
            summary_table.add_column("Group", style="bold cyan", justify="center")
            summary_table.add_column("Turns", justify="right")
            summary_table.add_column("Duration", justify="right", style="green")
            summary_table.add_column("Dialogue Excerpt", style="bold bright_yellow")

            for g_idx, c_label in enumerate(unique_labels, start=1):
                indices = np.where(labels == c_label)[0]
                c_turns = len(indices)
                c_dur = sum(valid_turns[idx][1] - valid_turns[idx][0] for idx in indices)
                c_turn_list = [valid_turns[idx] for idx in indices]
                g_snip = _dialogue_snippet_for_turns(c_turn_list, max_chars=80)
                snip_cell = f'"{g_snip}"' if g_snip else "[dim italic][Uncaptioned speech][/]"
                summary_table.add_row(f"{g_idx} / {len(unique_labels)}", f"{c_turns} turn(s)", f"{c_dur:.1f}s", snip_cell)

            Console(safe_box=True).print(summary_table)
            print()
        else:
            print(f"\n[+] {header_text}:")
            for g_idx, c_label in enumerate(unique_labels, start=1):
                indices = np.where(labels == c_label)[0]
                c_turns = len(indices)
                c_dur = sum(valid_turns[idx][1] - valid_turns[idx][0] for idx in indices)
                c_turn_list = [valid_turns[idx] for idx in indices]
                g_snip = _dialogue_snippet_for_turns(c_turn_list, max_chars=80)
                snip_str = f' | Dialogue: "{g_snip}"' if g_snip else ""
                print(f"    - Group {g_idx}/{len(unique_labels)}: {c_turns} turn(s), {c_dur:.1f}s total{snip_str}")
            print()

        for g_idx, c_label in enumerate(unique_labels, start=1):
            indices = np.where(labels == c_label)[0]
            cluster_embs = emb_matrix[indices]
            centroid = np.mean(cluster_embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 1e-8:
                centroid /= norm

            c_turns = len(indices)
            c_dur = sum(valid_turns[idx][1] - valid_turns[idx][0] for idx in indices)

            cluster_id = f"anon:{content_id}:{c_label}"
            cluster_safe_id = cluster_id.replace(":", "_")
            c_turn_list = [valid_turns[idx] for idx in indices]
            _preview_path = _write_cluster_preview(cluster_safe_id, c_turn_list)
            _snippet = _dialogue_snippet_for_turns(c_turn_list)
            _clips_info = _get_clips_info(c_turn_list)
            c_spk_id = self.registry.prompt_or_register_cluster(
                cluster_id=cluster_id,
                centroid=centroid,
                turn_count=c_turns,
                duration_sec=c_dur,
                interactive=interactive,
                audio_preview_path=_preview_path,
                dialogue_snippet=_snippet,
                group_label=f"Group {g_idx} of {len(unique_labels)}",
                clips=_clips_info,
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

        total_audio_sec = len(data) / sr if sr > 0 else 0.0
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

            # Extend interval into surrounding silence (hangover collar) to ensure trailing word phones are covered
            prev_e = valid_turns[idx - 1][1] if idx > 0 else 0.0
            next_s = valid_turns[idx + 1][0] if idx + 1 < len(valid_turns) else total_audio_sec
            pad_left = min(0.20, max(0.0, (s - prev_e) / 2.0))
            pad_right = min(0.35, max(0.0, (next_s - e) / 2.0))
            padded_s = max(0.0, s - pad_left)
            padded_e = min(total_audio_sec, e + pad_right)

            # Check if this turn overlaps with any other turn in time
            is_ov = False
            for j, (os_sec, oe_sec) in enumerate(valid_turns):
                if j != idx and max(s, os_sec) < min(e, oe_sec):
                    is_ov = True
                    break

            diarized_intervals.append(
                DiarizationInterval(
                    start_sec=padded_s,
                    end_sec=padded_e,
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
