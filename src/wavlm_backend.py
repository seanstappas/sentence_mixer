"""
Optional WavLM neural feature extraction and embedding reranker (Milestone M6).
Lazily imports torch/transformers. Basic pipeline works fully without Torch.
Extracts intermediate hidden layer representations (layer 6) and computes
cosine similarity distances that participate in candidate ranking.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np


def has_wavlm_dependencies() -> bool:
    """Checks if torch and transformers are available in the current environment."""
    import importlib.util
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("transformers") is not None
    )


class WavLMConfig:
    def __init__(
        self,
        model_id: str = "microsoft/wavlm-base-plus",
        revision: str = "main",
        layer: int = 6,
        feature_dim: int = 768,
        sample_rate: int = 16000,
    ):
        self.model_id = model_id
        self.revision = revision
        self.layer = layer
        self.feature_dim = feature_dim
        self.sample_rate = sample_rate

    @classmethod
    def load(cls, config_path: Union[str, Path]) -> "WavLMConfig":
        p = Path(config_path).resolve()
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(
            model_id=data.get("model_id", "microsoft/wavlm-base-plus"),
            revision=data.get("revision", "main"),
            layer=data.get("layer", 6),
            feature_dim=data.get("feature_dim", 768),
            sample_rate=data.get("sample_rate", 16000),
        )

    def save(self, config_path: Union[str, Path]) -> None:
        p = Path(config_path).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump({
                "model_id": self.model_id,
                "revision": self.revision,
                "layer": self.layer,
                "feature_dim": self.feature_dim,
                "sample_rate": self.sample_rate,
            }, f, indent=2)


class WavLMBackend:
    """
    WavLM analysis backend for candidate feature extraction and reranking.
    """
    def __init__(self, config: Optional[WavLMConfig] = None):
        if not has_wavlm_dependencies():
            raise ImportError(
                "WavLM backend requires 'torch' and 'transformers'. "
                "Install them via 'pip install .[wavlm]'"
            )

        import torch
        from transformers import AutoFeatureExtractor, WavLMModel

        self.config = config or WavLMConfig()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[*] Loading WavLM model: {self.config.model_id} (revision: {self.config.revision}, layer: {self.config.layer}) on {self.device}")
        try:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                trust_remote_code=False,
                local_files_only=True,
            )
            raw_model: Any = WavLMModel.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                trust_remote_code=False,
                local_files_only=True,
            )
        except Exception:
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                trust_remote_code=False,
            )
            raw_model = WavLMModel.from_pretrained(
                self.config.model_id,
                revision=self.config.revision,
                trust_remote_code=False,
            )
        self.model = raw_model.to(self.device)
        self.model.eval()

    def extract_dense_embeddings(self, audio_array: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        Extracts hidden state embeddings at configured layer for the given audio.
        Returns shape (n_frames, feature_dim).
        """
        import torch

        if audio_array.ndim > 1:
            audio_array = np.mean(audio_array, axis=1)

        if sample_rate != 16000:
            try:
                import librosa
                audio_array = librosa.resample(audio_array, orig_sr=sample_rate, target_sr=16000)
            except Exception:
                import scipy.signal
                num_target_samples = int(round(len(audio_array) * 16000 / sample_rate))
                audio_array = scipy.signal.resample(audio_array, num_target_samples)
            sample_rate = 16000

        min_samples = 1600
        if len(audio_array) < min_samples:
            audio_array = np.pad(audio_array, (0, min_samples - len(audio_array)))

        inputs = self.feature_extractor(
            audio_array,
            sampling_rate=16000,
            return_tensors="pt",
        )
        input_values = inputs.input_values.to(self.device)

        with torch.no_grad():
            outputs = self.model(input_values, output_hidden_states=True)
            # Hidden states: tuple of tensor shape (1, n_frames, 768)
            hidden_states = outputs.hidden_states
            layer_states = hidden_states[self.config.layer].squeeze(0).cpu().numpy()

        return layer_states

    def get_span_embedding(
        self,
        layer_states: np.ndarray,
        start_sample: int,
        end_sample: int,
        native_sr: int = 16000,
    ) -> np.ndarray:
        """
        Pools layer states over the interval [start_sample, end_sample).
        WavLM feature stride is ~20ms (320 samples at 16kHz).
        """
        stride_samples = int(0.02 * native_sr)
        start_frame = max(0, start_sample // stride_samples)
        end_frame = min(len(layer_states), max(start_frame + 1, end_sample // stride_samples))

        span_vectors = layer_states[start_frame:end_frame]
        pooled = np.mean(span_vectors, axis=0)
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm
        return pooled

    @staticmethod
    def compute_cosine_distance(emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Computes cosine distance 1.0 - cos_sim(emb1, emb2)."""
        dot = float(np.dot(emb1, emb2))
        norm1 = float(np.linalg.norm(emb1))
        norm2 = float(np.linalg.norm(emb2))
        if norm1 == 0 or norm2 == 0:
            return 1.0
        cos_sim = dot / (norm1 * norm2)
        return float(max(0.0, 1.0 - cos_sim))
