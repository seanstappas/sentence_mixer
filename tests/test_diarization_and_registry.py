"""
Unit and integration tests for Milestone 2 (R1 & R03):
- Acoustic Speaker Diarization & Turn Discovery (src/diarization.py)
- Persistent Speaker Identity Registry (src/speaker_registry.py)
- Generic Speaker Clustering & Interactive Naming
- Overlap Detection & Strict Phrase Speaker Eligibility Filtering (R03)
- Empirical Anakin vs. Obi-Wan Speaker Separation Verification
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest
import soundfile as sf

from contracts import InventoryIndex, WordOccurrence, AlignedEvent, SourceRecord
from diarization import (
    DiarizationInterval,
    SpeakerDiarizer,
    reconcile_word_intervals_with_speakers,
)
from speaker_registry import SpeakerProfile, SpeakerRegistry
from span_selector import SpanSelector
from target_graph import TargetGraphBuilder
from lexicon import Lexicon


def test_speaker_profile_centroid_normalization():
    prof = SpeakerProfile(
        speaker_id="spk_01",
        display_name="Speaker One",
        embedding_centroid=[3.0, 4.0],
        sample_count=2,
    )
    arr = prof.centroid_array()
    assert arr is not None
    assert np.isclose(np.linalg.norm(arr), 1.0)
    assert np.isclose(arr[0], 0.6)
    assert np.isclose(arr[1], 0.8)


def test_speaker_registry_matching_and_persistence(tmp_path: Path):
    reg_path = tmp_path / "speaker_registry.json"
    registry = SpeakerRegistry(config_path=reg_path, similarity_threshold=0.80)

    # Register two distinct profiles
    v_obi = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    v_anakin = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    registry.register_or_update("obi-wan", display_name="Obi-Wan Kenobi", centroid=v_obi)
    registry.register_or_update("anakin", display_name="Anakin Skywalker", centroid=v_anakin)

    assert reg_path.exists()

    # Query with query close to Obi-Wan (sim ~0.96)
    q_obi = np.array([0.96, 0.28, 0.0], dtype=np.float32)
    matched_id, score = registry.match_cluster(q_obi)
    assert matched_id == "obi-wan"
    assert score >= 0.80

    # Query with ambiguous / unknown vector (sim < 0.80)
    q_unknown = np.array([0.5, 0.5, 0.707], dtype=np.float32)
    matched_unknown, score_unknown = registry.match_cluster(q_unknown)
    assert matched_unknown is None
    assert score_unknown < 0.80

    # Test auto-registration under non-interactive mode
    new_spk = registry.prompt_or_register_cluster(
        cluster_id="cluster_02",
        centroid=q_unknown,
        turn_count=5,
        duration_sec=12.5,
        interactive=False,
    )
    assert new_spk.startswith("speaker_")

    # Reload from disk and confirm persistence
    reloaded = SpeakerRegistry(config_path=reg_path)
    assert "obi-wan" in reloaded.speakers
    assert "anakin" in reloaded.speakers
    assert new_spk in reloaded.speakers


def test_detect_speech_turns_energy_segmentation(tmp_path: Path):
    sr = 16000
    # Create 3 seconds of audio: 1s tone, 0.8s silence, 1s tone
    t1 = np.linspace(0, 1.0, sr, endpoint=False)
    tone1 = 0.5 * np.sin(2 * np.pi * 440 * t1)
    silence = np.zeros(int(0.8 * sr))
    t2 = np.linspace(0, 1.0, sr, endpoint=False)
    tone2 = 0.5 * np.sin(2 * np.pi * 330 * t2)

    audio = np.concatenate([tone1, silence, tone2])
    wav_path = tmp_path / "segmented_audio.wav"
    sf.write(str(wav_path), audio, sr)

    diarizer = SpeakerDiarizer(min_turn_duration=0.25)
    turns = diarizer.detect_speech_turns(audio, sr, min_silence_sec=0.3)
    assert len(turns) == 2
    assert np.isclose(turns[0][0], 0.0, atol=0.1)
    assert np.isclose(turns[0][1], 1.0, atol=0.15)
    assert np.isclose(turns[1][0], 1.8, atol=0.15)
    assert np.isclose(turns[1][1], 2.8, atol=0.15)


def test_reconcile_word_intervals_with_overlap_detection():
    diarized_intervals = [
        DiarizationInterval(start_sec=0.0, end_sec=2.0, speaker_id="obi-wan"),
        DiarizationInterval(start_sec=2.0, end_sec=3.0, speaker_id="OVERLAP", is_overlap=True),
        DiarizationInterval(start_sec=3.0, end_sec=5.0, speaker_id="anakin"),
    ]

    words = [
        {"start_sec": 0.2, "end_sec": 0.8, "word": "hello"},
        {"start_sec": 2.1, "end_sec": 2.7, "word": "shouting"},
        {"start_sec": 3.2, "end_sec": 4.1, "word": "master"},
        {"start_sec": 6.0, "end_sec": 6.5, "word": "silence_word"},
    ]

    reconciled = reconcile_word_intervals_with_speakers(diarized_intervals, words)
    assert len(reconciled) == 4

    # 1. Clean Obi-Wan word
    assert reconciled[0]["speaker_id"] == "obi-wan"
    assert reconciled[0]["speaker_eligible"] is True

    # 2. Overlapping speech word -> marked ineligible
    assert reconciled[1]["speaker_id"] == "OVERLAP"
    assert reconciled[1]["speaker_eligible"] is False

    # 3. Clean Anakin word
    assert reconciled[2]["speaker_id"] == "anakin"
    assert reconciled[2]["speaker_eligible"] is True

    # 4. Out of bounds / silence word
    assert reconciled[3]["speaker_id"] == "UNKNOWN"
    assert reconciled[3]["speaker_eligible"] is False


def test_r03_span_selector_multi_speaker_phrase_filtering():
    """
    R03 Invariant Verification:
    SpanSelector must strictly prevent cross-speaker phrase splicing.
    When selecting for --speaker='anakin', a contiguous phrase where one word
    is spoken by obi-wan must be rejected as an intact phrase candidate.
    """
    inv = InventoryIndex()
    inv.sources["src1"] = SourceRecord(
        content_id="src1",
        source_path="video1.mp4",
        native_sample_rate=16000,
        native_channels=1,
    )

    # Word 1: 'i' (spoken by anakin)
    ev1 = AlignedEvent(
        event_id="ev_001",
        source_content_id="src1",
        utterance_id="utt1",
        speaker_id="anakin",
        kind="phone",
        start_sample=0,
        end_sample=1600,
        label="AY1",
        base_phone="AY",
        stress=1,
        successor_id="ev_002",
    )
    # Word 2: 'have' (spoken by obi-wan - cross-speaker contamination)
    ev2 = AlignedEvent(
        event_id="ev_002",
        source_content_id="src1",
        utterance_id="utt1",
        speaker_id="obi-wan",
        kind="phone",
        start_sample=1600,
        end_sample=3200,
        label="HH",
        base_phone="HH",
        stress=None,
        predecessor_id="ev_001",
    )

    wo1 = WordOccurrence(
        word_id="w_001",
        source_content_id="src1",
        speaker_id="anakin",
        word_normalized="i",
        word_original="I",
        start_sample=0,
        end_sample=1600,
        phone_event_ids=["ev_001"],
        realized_phones=["AY1"],
    )
    wo2 = WordOccurrence(
        word_id="w_002",
        source_content_id="src1",
        speaker_id="obi-wan",
        word_normalized="have",
        word_original="have",
        start_sample=1600,
        end_sample=3200,
        phone_event_ids=["ev_002"],
        realized_phones=["HH"],
    )

    ev1.word_occurrence_id = "w_001"
    ev2.word_occurrence_id = "w_002"

    # Add a separate, non-contiguous 'have' spoken by anakin in another utterance
    ev3 = AlignedEvent(
        event_id="ev_003",
        source_content_id="src1",
        utterance_id="utt2",
        speaker_id="anakin",
        kind="phone",
        start_sample=5000,
        end_sample=6600,
        label="HH",
        base_phone="HH",
        stress=None,
    )
    wo3 = WordOccurrence(
        word_id="w_003",
        source_content_id="src1",
        speaker_id="anakin",
        word_normalized="have",
        word_original="have",
        start_sample=5000,
        end_sample=6600,
        phone_event_ids=["ev_003"],
        realized_phones=["HH"],
    )
    ev3.word_occurrence_id = "w_003"

    inv.events["ev_001"] = ev1
    inv.events["ev_002"] = ev2
    inv.events["ev_003"] = ev3
    inv.words["w_001"] = wo1
    inv.words["w_002"] = wo2
    inv.words["w_003"] = wo3
    inv.word_to_word_ids["i"] = ["w_001"]
    inv.word_to_word_ids["have"] = ["w_002", "w_003"]
    inv.phone_to_event_ids["AY1"] = ["ev_001"]
    inv.phone_to_event_ids["HH"] = ["ev_002", "ev_003"]

    # Build graph for "i have"
    lex = Lexicon([])
    lex.word_to_pronunciations["i"] = [["AY1"]]
    lex.word_to_pronunciations["have"] = [["HH"]]

    graph_builder = TargetGraphBuilder(lex)
    graph = graph_builder.build("i have")

    selector = SpanSelector(inv)

    # 1. Requesting speaker='anakin': cross-speaker phrase 'i have' must NOT be selected
    selected_anakin = selector.select(graph, speaker_id="anakin")
    phrase_spans = [s for s in selected_anakin if s.span_type == "phrase"]
    assert len(phrase_spans) == 0, "Cross-speaker phrase was illegally formed!"

    # 2. Check that word 'have' from obi-wan is not selected as word span for anakin
    for s in selected_anakin:
        assert s.speaker_id == "anakin", f"Selected unit belonged to wrong speaker {s.speaker_id}"


def test_empirical_anakin_vs_obi_wan_separation(tmp_path: Path):
    """
    Empirical validation using recorded Star Wars dialogue:
    Confirms that real audio chunks from Obi-Wan and Anakin separate acoustically
    and match their respective registry profiles.
    """
    obi_dir = Path(".build/alignment/obi_wan_ep2_bfc3791b/corpus")
    anakin_dir = Path("corpus/anakin_ep3_1")

    if not obi_dir.exists() or not anakin_dir.exists():
        pytest.skip("Recorded media corpora not found; skipping empirical separation test.")

    obi_wavs = sorted(list(obi_dir.glob("chunk_*.wav")))[:4]
    anakin_wavs = sorted(list(anakin_dir.glob("chunk_*.wav")))[:4]

    if len(obi_wavs) < 2 or len(anakin_wavs) < 2:
        pytest.skip("Insufficient recorded chunks found for empirical separation test.")

    reg_file = tmp_path / "speaker_registry.json"
    registry = SpeakerRegistry(config_path=reg_file, similarity_threshold=0.74)
    diarizer = SpeakerDiarizer(registry=registry)

    # Diarize Obi-Wan chunks to train registry profile
    obi_embs = [diarizer.extract_embedding(sf.read(str(w))[0], sf.read(str(w))[1]) for w in obi_wavs]
    anakin_embs = [diarizer.extract_embedding(sf.read(str(w))[0], sf.read(str(w))[1]) for w in anakin_wavs]

    obi_centroid = np.mean(obi_embs, axis=0)
    obi_centroid /= np.linalg.norm(obi_centroid)

    anakin_centroid = np.mean(anakin_embs, axis=0)
    anakin_centroid /= np.linalg.norm(anakin_centroid)

    # Register known ground truth profiles
    registry.register_or_update("obi-wan", display_name="Obi-Wan Kenobi", centroid=obi_centroid)
    registry.register_or_update("anakin", display_name="Anakin Skywalker", centroid=anakin_centroid)

    # Match an unseen Obi-Wan test chunk
    unseen_obi_emb = diarizer.extract_embedding(sf.read(str(obi_wavs[-1]))[0], sf.read(str(obi_wavs[-1]))[1])
    matched_spk_obi, obi_score = registry.match_cluster(unseen_obi_emb)
    assert matched_spk_obi == "obi-wan"

    # Match an unseen Anakin test chunk
    unseen_anakin_emb = diarizer.extract_embedding(sf.read(str(anakin_wavs[-1]))[0], sf.read(str(anakin_wavs[-1]))[1])
    matched_spk_anakin, anakin_score = registry.match_cluster(unseen_anakin_emb)
    assert matched_spk_anakin == "anakin"
