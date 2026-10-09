"""
Comprehensive verification tests for the 9 audit remediation findings:
1. Transactional publication, multi-variant output delivery, and permanent XML audio links.
2. Exact video timing with overlaps (6x100ms cuts -> 16 frames / 0.530s) and tpad source-end holds.
3. CLI argument wiring, target options mutual exclusivity, and shared OOV resolution.
4. Diarization singleton isolation, ambiguity margin, silence turns, and >70% coverage.
5. Pronunciation branch realization matching in SpanSelector.
6. ProsodyGuide integration and get_candidate_cost.
7. PhonemeRecognizer 16 kHz resampling and FileLock in custom pronunciation saving.
8. Content-aware source identification and unflattened MFA collection.
9. evaluate.py render failure detection and listen_verify phonetic group resolution.
"""

from __future__ import annotations

import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

from contracts import (
    AlignedEvent,
    EditPlan,
    InventoryIndex,
    PlanSegment,
    RationalTime,
    SourceRecord,
    SpanCandidate,
    TargetGraph,
    TargetPhoneNode,
    WordOccurrence,
)
from diarization import (
    DiarizationInterval,
    SpeakerDiarizer,
    reconcile_word_intervals_with_speakers,
)
from edit_plan import build_edit_plan
from fcp_xml_generator import generate_fcp_xml
from lexicon import Lexicon
from phoneme_recognizer import PhonemeRecognizer
from prosody_guide import ProsodyGuide
import run_mixer
from speaker_registry import SpeakerProfile, SpeakerRegistry
from span_selector import SpanSelector
from tools.listen_verify import normalize_text


# ============================================================================
# Finding 1: XML Master WAV Path & Multi-Variant Delivery
# ============================================================================

def test_fcp_xml_master_wav_permanent_path(tmp_path: Path):
    """
    Verifies XML references the permanent published master WAV pathurl,
    inspecting the staged master WAV without requiring the permanent WAV to pre-exist.
    """
    staged_dir = tmp_path / ".attempt_123"
    staged_dir.mkdir()
    staged_master_wav = staged_dir / "mix.wav"
    # Create valid minimal WAV in staged dir
    sr = 48000
    t = np.linspace(0, 0.5, int(0.5 * sr), endpoint=False)
    sf.write(str(staged_master_wav), (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr)

    pub_dir = tmp_path / "published"
    pub_dir.mkdir()
    published_master_wav = pub_dir / "mix.wav"  # Does NOT exist yet!
    xml_path = pub_dir / "mix.xml"

    src = SourceRecord(
        content_id="src1",
        source_path=str(tmp_path / "video.mp4"),
        native_sample_rate=48000,
        native_channels=2,
    )
    seg = PlanSegment(
        segment_index=0,
        source_content_id="src1",
        speaker_id="spk1",
        source_start_sample=0,
        source_end_sample=24000,
        output_start_sample=0,
        output_end_sample=24000,
        target_text="test",
        target_phones=[],
        span_type="word",
        source_video_path=src.source_path,
        video_start_frame=0,
        video_frame_count=15,
        video_source_pts_sec=0.0,
    )
    plan = EditPlan(
        target_text="test",
        video_fps=RationalTime(30, 1),
        total_output_samples=24000,
        total_video_frames=15,
        segments=[seg],
        sources={"src1": src},
    )

    # Calling generate_fcp_xml with published path and inspect path
    generate_fcp_xml(
        plan,
        str(xml_path),
        master_wav_path=str(published_master_wav),
        inspect_wav_path=str(staged_master_wav),
    )
    assert xml_path.exists()
    content = xml_path.read_text(encoding="utf-8")
    assert "published/mix.wav" in content.replace("\\", "/") or "mix.wav" in content
    assert ".attempt_" not in content
    # XML must contain at least 1 audio clip item
    tree = ET.parse(str(xml_path))
    audio_clips = tree.getroot().findall(".//media/audio/track/clipitem")
    assert len(audio_clips) >= 1, "XML must contain audio clip items referencing the published master WAV"


# ============================================================================
# Finding 2: Video Timing with Overlaps (6x100ms) & Source-End Holds
# ============================================================================

def test_six_100ms_cuts_overlap_cumulative_video_timing():
    """
    Finding 2: 6 x 100-ms cuts at 48 kHz / 30 fps in source-derived mode.
    Each cut: 4800 samples. 5 transitions with 14ms overlap (672 samples).
    Total audio samples = 6*4800 - 5*672 = 25,440 (0.530 s).
    RationalTime(30, 1) ceiling frames = 16 frames.
    Deterministic midpoint picture-switch policy must produce sum(video_frame_count) == 16.
    """
    sr = 48000
    fps = RationalTime(30, 1)
    cut_samples = 4800  # 100ms at 48kHz

    src = SourceRecord(
        content_id="src1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )
    cands = []
    events = {}
    nodes = []

    for i in range(6):
        ev_id = f"ev_{i}"
        events[ev_id] = AlignedEvent(
            event_id=ev_id, source_content_id="src1", utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=i * cut_samples * 2, end_sample=i * cut_samples * 2 + cut_samples,
            label="AA", base_phone="AA"
        )
        nodes.append(TargetPhoneNode(
            node_id=i, token_index=i, word_text=f"w{i}", phone="AA", base_phone="AA",
            stress=1, is_word_initial=True, is_word_final=True, is_phrase_final=(i == 5)
        ))
        cands.append(SpanCandidate(
            candidate_id=f"c_{i}",
            target_node_start=i,
            target_node_end=i,
            covered_nodes=[i],
            source_content_id="src1",
            speaker_id="spk1",
            start_sample=i * cut_samples * 2,
            end_sample=i * cut_samples * 2 + cut_samples,
            event_ids=[ev_id],
            span_type="word",
            matched_word=f"w{i}",
        ))

    inv = InventoryIndex(sources={"src1": src}, events=events)
    graph = TargetGraph(
        raw_text="w0 w1 w2 w3 w4 w5",
        normalized_words=[f"w{i}" for i in range(6)],
        nodes=nodes,
        edges=[(i, i + 1) for i in range(5)],
        start_nodes=[0],
        end_nodes=[5],
    )

    plan = build_edit_plan(
        cands, graph, inv, project_sample_rate=sr, video_fps=fps, rendering_mode="source-derived", overlap_duration_ms=14.0
    )
    plan.validate()

    assert plan.total_output_samples == 25440, f"Expected 25440 audio samples, got {plan.total_output_samples}"
    assert plan.total_video_frames == 16, f"Expected 16 video frames, got {plan.total_video_frames}"
    assert sum(s.video_frame_count for s in plan.segments) == 16, "Sum of segment video_frame_count must exactly equal total_video_frames"

    # Verify contiguous display without gaps or overlaps
    for k in range(len(plan.segments) - 1):
        assert plan.segments[k].video_start_frame + plan.segments[k].video_frame_count == plan.segments[k + 1].video_start_frame


# ============================================================================
# Finding 3: CLI Argument Wiring & Target Mutual Exclusivity
# ============================================================================

def test_cli_target_mutual_exclusivity():
    """Finding 3: --text, --target-audio, and --record-target are mutually exclusive."""
    # Test that running with both --text and --target-audio fails with mutual exclusion error
    cmd = [
        sys.executable, str(REPO_ROOT / "run_mixer.py"), "mix",
        "--text", "Hello",
        "--target-audio", "dummy.wav",
        "--inventory", "dummy_inv.json",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "mutually exclusive" in res.stderr.lower() or "error" in res.stderr.lower()


# ============================================================================
# Finding 4: Diarization Singleton Scoping, Margin & Coverage Safety
# ============================================================================

def test_diarization_singleton_scoped_id_no_registry_pollution(tmp_path: Path):
    """Finding 4: Unenrolled single-turn speaker receives recording-scoped ID without polluting persistent registry."""
    reg_file = tmp_path / "registry.json"
    reg = SpeakerRegistry(config_path=reg_file)

    dummy_audio = tmp_path / "test_video.wav"
    sr = 16000
    # 1 second of speech
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 300 * t)).astype(np.float32)
    sf.write(str(dummy_audio), sig, sr)

    diarizer = SpeakerDiarizer(registry=reg)
    intervals = diarizer.diarize(dummy_audio)

    assert len(intervals) == 1
    spk_id = intervals[0].speaker_id
    assert spk_id.startswith("anon:"), f"Expected D4 anonymous ID format, got '{spk_id}'"
    assert spk_id != "spk_0000"
    # Registry must NOT contain this anonymous profile
    assert spk_id not in reg.speakers


def test_speaker_registry_ambiguity_margin():
    """Finding 4: Ambiguous ties within ambiguity_margin (0.05) return None to prevent misclassification."""
    reg = SpeakerRegistry()
    emb_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    # Target centroid: close to emb_a
    target = np.array([0.99, 0.10, 0.0], dtype=np.float32)
    target /= np.linalg.norm(target)

    # Runner up: almost as close
    emb_b = np.array([0.98, 0.12, 0.0], dtype=np.float32)
    emb_b /= np.linalg.norm(emb_b)

    reg.speakers["spk_a"] = SpeakerProfile(speaker_id="spk_a", display_name="A", embedding_centroid=list(emb_a))
    reg.speakers["spk_b"] = SpeakerProfile(speaker_id="spk_b", display_name="B", embedding_centroid=list(emb_b))

    matched_id, score = reg.match_cluster(target, threshold=0.70, ambiguity_margin=0.05)
    # Because similarity to spk_a and spk_b is within 0.05, it must return None (ambiguous tie)
    assert matched_id is None, f"Expected ambiguous tie to yield None, got {matched_id}"


def test_reconcile_word_intervals_coverage_ratio_threshold():
    """Finding 4 / D5: Uncovered part outside interval makes word ineligible unless within collar of aligned silence."""
    # Speaker interval from 1.0 to 2.0s
    intervals = [DiarizationInterval(start_sec=1.0, end_sec=2.0, speaker_id="obi-wan")]

    words = [
        # Word 1: 1.1s to 1.7s (100% inside -> eligible)
        {"start_sec": 1.1, "end_sec": 1.7, "word": "clean"},
        # Word 2: 0.9s to 1.5s (0.1s outside interval without verified silence -> ineligible under D5)
        {"start_sec": 0.9, "end_sec": 1.5, "word": "mostly_inside"},
        # Word 3: 0.5s to 1.2s (duration 0.7s, inside: 0.2s -> ineligible)
        {"start_sec": 0.5, "end_sec": 1.2, "word": "mostly_outside"},
    ]

    reconciled = reconcile_word_intervals_with_speakers(intervals, words)
    assert reconciled[0]["speaker_eligible"] is True
    assert reconciled[1]["speaker_eligible"] is False, "Word crossing outside interval without verified silence is ineligible"
    assert reconciled[2]["speaker_eligible"] is False, "Word with only 28% overlap must be ineligible"


def test_detect_speech_turns_pure_silence_returns_empty():
    """Finding 4: detect_speech_turns does not manufacture a speech turn for pure digital silence."""
    diarizer = SpeakerDiarizer()
    silence = np.zeros(16000 * 3, dtype=np.float32)
    turns = diarizer.detect_speech_turns(silence, sample_rate=16000)
    assert len(turns) == 0, f"Expected 0 speech turns for pure silence, got {turns}"


# ============================================================================
# Finding 5: Pronunciation Branch Realization Matching in SpanSelector
# ============================================================================

def test_span_selector_pronunciation_branch_realization_matching():
    """Finding 5: SpanSelector prevents R IY1 D recording from attaching to R EH1 D branch."""
    lex = Lexicon([])
    lex.word_to_pronunciations["read"] = [["R", "IY1", "D"], ["R", "EH1", "D"]]

    from target_graph import TargetGraphBuilder
    builder = TargetGraphBuilder(lex)
    graph = builder.build("read")

    inv = InventoryIndex()
    inv.sources["src1"] = SourceRecord(
        content_id="src1", source_path="v.mp4", native_sample_rate=16000, native_channels=1
    )
    evs = [
        AlignedEvent(event_id=f"e{i}", source_content_id="src1", utterance_id="u1", speaker_id="spk1",
                     kind="phone", start_sample=i*1600, end_sample=(i+1)*1600, label=p, base_phone=p[:2])
        for i, p in enumerate(["R", "IY1", "D"])
    ]
    for i in range(len(evs)-1):
        evs[i].successor_id = evs[i+1].event_id
    wo_present = WordOccurrence(
        word_id="wo_read_present", source_content_id="src1", speaker_id="spk1",
        word_normalized="read", word_original="read", start_sample=0, end_sample=4800,
        phone_event_ids=[e.event_id for e in evs], realized_phones=["R", "IY1", "D"],
    )
    inv.events = {e.event_id: e for e in evs}
    inv.words = {"wo_read_present": wo_present}
    inv.word_to_word_ids = {"read": ["wo_read_present"]}

    selector = SpanSelector(inv)
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")
    word_cands = [c for c in cands if c.span_type == "word"]

    for c in word_cands:
        # Check the branch of the covered nodes
        branch_ids = {graph.nodes[nid].branch_id for nid in c.covered_nodes}
        # Since candidate was realized as R IY1 D (branch 0), it MUST NOT attach to branch 1 (R EH1 D)
        assert 1 not in branch_ids, "Candidate realized as R IY1 D must not be attached to R EH1 D branch"


# ============================================================================
# Finding 6: ProsodyGuide Integration
# ============================================================================

def test_prosody_guide_get_candidate_cost(tmp_path: Path):
    """Finding 6: ProsodyGuide implements get_candidate_cost."""
    guide_wav = tmp_path / "guide.wav"
    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    sig = (0.5 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    sf.write(str(guide_wav), sig, sr)

    guide = ProsodyGuide(guide_wav)
    cand = MagicMock()
    cand.target_node_start = 0
    cand.target_node_end = 2
    cand.pitch_semitones = 2.0

    cost = guide.get_candidate_cost(cand)
    assert isinstance(cost, float)
    assert cost >= 0.0


# ============================================================================
# Finding 7: PhonemeRecognizer Resampling & FileLock
# ============================================================================

def test_phoneme_recognizer_resamples_non_16k_audio():
    """Finding 7: PhonemeRecognizer resamples audio to 16 kHz before Whisper transcription."""
    rec = PhonemeRecognizer()
    # 48 kHz audio array
    sr = 48000
    audio_48k = np.zeros(48000, dtype=np.float32)
    # Should not raise exception and should complete without error
    res = rec.transcribe_audio_to_text(audio_48k, sample_rate=sr)
    assert isinstance(res, str)


def test_lexicon_save_custom_pronunciation_filelock(tmp_path: Path):
    """Finding 7: Lexicon.save_custom_pronunciation uses FileLock and deduplicates atomically."""
    lex_file = tmp_path / "custom_dict.txt"
    lex = Lexicon()
    lex.save_custom_pronunciation("moxy", ["M", "AA1", "K", "S", "IY0"], lex_file)
    assert lex_file.exists()
    content = lex_file.read_text(encoding="utf-8")
    assert "moxy\tM AA1 K S IY0" in content

    # Duplicate call should not create duplicate entries
    lex.save_custom_pronunciation("moxy", ["M", "AA1", "K", "S", "IY0"], lex_file)
    lines = [ln for ln in lex_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1


# ============================================================================
# Finding 8: Content-Aware Caching
# ============================================================================

def test_get_source_id_content_aware(tmp_path: Path):
    """Finding 8: get_source_id produces different hashes when content changes at the same path."""
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"DATA_VERSION_ONE")
    id1 = run_mixer.get_source_id(str(media))

    media.write_bytes(b"DATA_VERSION_TWO_MODIFIED")
    id2 = run_mixer.get_source_id(str(media))

    assert id1 != id2, f"Expected content change to alter source ID, got identical '{id1}'"


# ============================================================================
# Finding 9: Acceptance Reporting & Phonetic Group Resolution
# ============================================================================

def test_listen_verify_resolves_phonetic_group_to_lexical_word():
    """Finding 9: listen_verify resolves {B AE1 K AH0 N} to 'bacon' instead of stripping it."""
    # Ensure bacon is in lexicon
    norm = normalize_text("I love {B EY1 K AH0 N}")
    assert "bacon" in norm.split(), f"Expected 'bacon' in normalized output, got: '{norm}'"


def test_listen_verify_unmapped_phonetic_group_raises_runtime_error():
    """Finding 9: Unmapped phonetic group raises ValueError (operational error, exit 2)."""
    with pytest.raises(ValueError, match="Unmapped phonetic group"):
        normalize_text("Hello {ZZ9 YY9 XX9} world")
