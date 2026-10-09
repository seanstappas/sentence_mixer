"""
Unit tests for WavLM config, backend, and embedding distance functions (M6).
The basic path works without Torch.
"""

import numpy as np
import pytest
from pathlib import Path

from src.wavlm_backend import WavLMConfig, WavLMBackend, has_wavlm_dependencies


def test_wavlm_config_save_load(tmp_path: Path):
    cfg = WavLMConfig(
        model_id="microsoft/wavlm-base-plus",
        revision="main",
        layer=6,
        feature_dim=768,
        sample_rate=16000,
    )
    save_path = tmp_path / "wavlm.json"
    cfg.save(save_path)

    assert save_path.exists()
    loaded = WavLMConfig.load(save_path)
    assert loaded.model_id == "microsoft/wavlm-base-plus"
    assert loaded.revision == "main"
    assert loaded.layer == 6
    assert loaded.feature_dim == 768
    assert loaded.sample_rate == 16000


def test_wavlm_cosine_distance():
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([1.0, 0.0, 0.0])
    # Identical vectors -> 0 distance
    assert abs(WavLMBackend.compute_cosine_distance(v1, v2)) < 1e-6

    # Orthogonal vectors -> 1.0 distance
    v3 = np.array([0.0, 1.0, 0.0])
    assert abs(WavLMBackend.compute_cosine_distance(v1, v3) - 1.0) < 1e-6

    # Opposite vectors -> 2.0 distance
    v4 = np.array([-1.0, 0.0, 0.0])
    assert abs(WavLMBackend.compute_cosine_distance(v1, v4) - 2.0) < 1e-6

    # Zero vector -> fallback to 1.0
    v0 = np.array([0.0, 0.0, 0.0])
    assert WavLMBackend.compute_cosine_distance(v0, v1) == 1.0


def test_wavlm_span_embedding_pooling():
    # Test pooling logic without needing torch
    # Mock layer_states: 10 frames of 4-dim features
    layer_states = np.zeros((10, 4), dtype=np.float32)
    layer_states[0:5, :] = 1.0
    layer_states[5:10, :] = 2.0

    class DummyBackend:
        get_span_embedding = WavLMBackend.get_span_embedding

    dummy = DummyBackend()
    # 16000 sr, stride is 320 samples (20ms)
    # interval 0 to 640 samples = frames 0, 1
    emb = dummy.get_span_embedding(layer_states, start_sample=0, end_sample=640, native_sr=16000)
    assert emb.shape == (4,)
    # Norm should be 1.0
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-5


def test_wavlm_backend_missing_dependencies():
    # If dependencies are missing, instantiate should raise ImportError
    if not has_wavlm_dependencies():
        with pytest.raises(ImportError, match="requires 'torch' and 'transformers'"):
            WavLMBackend()
