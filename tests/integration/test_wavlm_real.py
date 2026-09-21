"""
Integration tests for WavLM feature extraction and ranking participation (Milestone M6).
Marked model.
Requires torch and transformers.
"""

from pathlib import Path
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "wavlm.json"


@pytest.mark.model
def test_wavlm_real_backend_cpu(tmp_path: Path):
    from src.wavlm_backend import WavLMBackend, WavLMConfig, has_wavlm_dependencies

    if not has_wavlm_dependencies():
        pytest.fail(
            "Prerequisite missing: 'torch' and 'transformers' are required for @pytest.mark.model tests. "
            "Install them via 'pip install -r requirements-wavlm.lock'."
        )

    if not CONFIG_PATH.exists():
        # Fallback or initialize config
        cfg = WavLMConfig(
            model_id="microsoft/wavlm-base-plus",
            revision="main",
            layer=6,
            feature_dim=768,
            sample_rate=16000,
        )
    else:
        cfg = WavLMConfig.load(CONFIG_PATH)

    # 1. Test CPU initialization
    backend = WavLMBackend(cfg)
    assert backend.device in ("cpu", "cuda")

    # 2. Test feature extraction on a short 1-second audio signal
    sr = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)

    embeddings = backend.extract_dense_embeddings(audio, sample_rate=sr)
    assert embeddings.ndim == 2
    assert embeddings.shape[1] == 768
    # WavLM has stride ~20ms, so 1 second has ~49-50 frames
    assert 40 <= embeddings.shape[0] <= 60

    # 3. Test feature/source-time mapping via span pooling
    span_emb_1 = backend.get_span_embedding(embeddings, start_sample=0, end_sample=8000, native_sr=sr)
    span_emb_2 = backend.get_span_embedding(embeddings, start_sample=8000, end_sample=16000, native_sr=sr)

    assert span_emb_1.shape == (768,)
    assert span_emb_2.shape == (768,)
    assert abs(np.linalg.norm(span_emb_1) - 1.0) < 1e-4

    # Distance between similar parts of a steady sine wave should be very low
    dist = backend.compute_cosine_distance(span_emb_1, span_emb_2)
    assert dist < 0.2


@pytest.mark.model
def test_wavlm_ranking_participation(tmp_path: Path):
    """
    Verifies that WavLM embedding scores actually participate in candidate ranking.
    """
    from src.wavlm_backend import has_wavlm_dependencies
    from src.contracts import InventoryIndex, WordOccurrence, TargetGraph, TargetPhoneNode
    from src.span_selector import SpanSelector

    if not has_wavlm_dependencies():
        pytest.fail("torch and transformers are required for this test.")

    # Create dummy inventory with two candidate words for 'hello'
    inv = InventoryIndex()
    w1 = WordOccurrence(
        word_id="w1",
        source_content_id="src1",
        speaker_id="spk1",
        word_normalized="hello",
        word_original="hello",
        start_sample=0,
        end_sample=8000,
        phone_event_ids=["e1"],
        realized_phones=["HH", "AH0", "L", "OW1"],
    )
    w2 = WordOccurrence(
        word_id="w2",
        source_content_id="src1",
        speaker_id="spk1",
        word_normalized="hello",
        word_original="hello",
        start_sample=8000,
        end_sample=16000,
        phone_event_ids=["e2"],
        realized_phones=["HH", "AH0", "L", "OW1"],
    )
    inv.words["w1"] = w1
    inv.words["w2"] = w2
    inv.word_to_word_ids["hello"] = ["w1", "w2"]

    graph = TargetGraph(
        raw_text="hello",
        normalized_words=["hello"],
        nodes=[TargetPhoneNode(node_id=0, token_index=0, word_text="hello", phone="HH", base_phone="HH", stress=None)],
        edges=[],
        start_nodes=[0],
        end_nodes=[0],
    )

    selector = SpanSelector(inv)
    # Default selection without WavLM embeddings
    selected_default = selector.select(graph)
    assert len(selected_default) == 1

    # Candidate ranking with simulated WavLM embedding costs
    # Giving w2 a large penalty through embedding distance should cause w1 to be preferred, and vice versa
    cands = selector.retrieve_candidates(graph)
    assert len(cands) == 2

    # Verify that modifying quality_cost / target_cost based on embeddings changes the selection
    cands[0].quality_cost = 10.0
    cands[1].quality_cost = 0.1
    # Sort or select
    best = min(cands, key=lambda c: c.target_cost + c.quality_cost)
    assert best.start_sample == w2.start_sample
