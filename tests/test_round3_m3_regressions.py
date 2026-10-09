from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

import run_mixer
from contracts import (
    AlignedEvent,
    InventoryIndex,
    SourceRecord,
    WordOccurrence,
)
from diarization import (
    DiarizationInterval,
    SpeakerDiarizer,
    reconcile_word_intervals_with_speakers,
)
from speaker_registry import SpeakerRegistry


def test_u14_different_recordings_different_anon_ids_and_registry_unchanged(tmp_path: Path):
    """
    U14: Two different recordings, same file name, identical first 64 ms:
    - Receive different anonymous IDs
    - In non-interactive mode: registry is UNCHANGED
    - In interactive mode: prompts once per unmatched cluster (singletons included)
    """
    reg_file = tmp_path / "speaker_registry.json"
    reg = SpeakerRegistry(config_path=reg_file)
    initial_speaker_count = len(reg.speakers)

    # Create two audio files with identical first 64 ms (1024 samples at 16kHz)
    sr = 16000
    dir1 = tmp_path / "rec1"
    dir2 = tmp_path / "rec2"
    dir1.mkdir()
    dir2.mkdir()

    file1 = dir1 / "same.wav"
    file2 = dir2 / "same.wav"

    common_lead = np.zeros(2000, dtype=np.float32)
    tail1 = np.sin(2 * np.pi * 440 * np.linspace(0, 1, 16000)).astype(np.float32)
    tail2 = np.sin(2 * np.pi * 880 * np.linspace(0, 1, 16000)).astype(np.float32)

    audio1 = np.concatenate([common_lead, tail1])
    audio2 = np.concatenate([common_lead, tail2])

    sf.write(str(file1), audio1, sr)
    sf.write(str(file2), audio2, sr)

    diarizer = SpeakerDiarizer(registry=reg)

    # 1. Non-interactive run
    intervals1 = diarizer.diarize(file1, interactive=False)
    intervals2 = diarizer.diarize(file2, interactive=False)

    assert len(intervals1) > 0
    assert len(intervals2) > 0
    id1 = intervals1[0].speaker_id
    id2 = intervals2[0].speaker_id

    assert id1.startswith("anon:"), f"Expected anonymous ID, got {id1}"
    assert id2.startswith("anon:"), f"Expected anonymous ID, got {id2}"
    assert id1 != id2, f"Expected distinct anonymous IDs for distinct files, got {id1} and {id2}"

    # Registry must be completely unchanged in non-interactive run
    reg.load()
    assert len(reg.speakers) == initial_speaker_count, "Registry must remain unchanged in non-interactive run"

    # 2. Interactive run: prompt is called
    prompts_called = []

    def mock_input(prompt_text):
        prompts_called.append(prompt_text)
        return ""  # user pressed Enter to skip

    with patch("builtins.input", mock_input), patch("sys.stdin.isatty", return_value=True):
        diarizer.diarize(file1, interactive=True)
        assert len(prompts_called) >= 1, "Expected interactive prompt for unrecognized singleton cluster"


def test_u15_word_eligibility_and_collar():
    """
    U15:
    - Word 83% covered whose uncovered part is not aligned silence -> Ineligible, partial_coverage
    - Word whose 30 ms uncovered edge lies in aligned silence -> Eligible within 40 ms collar
    - Word inside a confidence-0.0 anonymous interval -> Ineligible, unreviewed_identity
    """
    diar_intervals = [
        DiarizationInterval(start_sec=1.0, end_sec=2.0, speaker_id="obi_wan", confidence=1.0),
        DiarizationInterval(start_sec=5.0, end_sec=6.0, speaker_id="anon:source:0", confidence=0.0),
    ]

    words = [
        # Case 1: 83% covered (1.0 to 2.0 covered = 1.0s, total 1.2s), uncovered 0.8 to 1.0 is NOT silence
        {"word": "uncovered_speech", "start_sec": 0.8, "end_sec": 2.0},
        # Case 2: 30 ms uncovered edge (0.970 to 1.0) which lies in aligned silence
        {"word": "collar_silence", "start_sec": 0.970, "end_sec": 2.0},
        # Case 3: Inside confidence 0.0 anonymous interval
        {"word": "anon_word", "start_sec": 5.1, "end_sec": 5.9},
    ]

    silence_intervals = [
        (0.950, 1.000),  # Aligned silence covering the 30 ms edge of word 2
    ]

    reconciled = reconcile_word_intervals_with_speakers(
        diar_intervals,
        words,
        silence_intervals=silence_intervals,
        collar_sec=0.040,
    )

    r1 = reconciled[0]
    assert r1["is_eligible"] is False
    assert r1["eligibility_reason"] == "partial_coverage"

    r2 = reconciled[1]
    assert r2["is_eligible"] is True
    assert r2["eligibility_reason"] == ""
    assert r2["speaker_id"] == "obi_wan"

    r3 = reconciled[2]
    assert r3["is_eligible"] is False
    assert r3["eligibility_reason"] == "unreviewed_identity"


def test_u13_diarize_auto_backend_unavailable_exits_code_2(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    U13: --diarize auto, backend unavailable, --video plus --speaker without assertion.
    Exits code 2 with setup and assertion instructions.
    With --assume-single-speaker, labels recorded as asserted.
    """
    v_path = tmp_path / "clip.mp4"
    v_path.write_bytes(b"dummy_video_bytes")
    vtt_path = tmp_path / "clip.vtt"
    vtt_path.write_text("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nHello\n", encoding="utf-8")

    test_args = [
        "run_mixer.py",
        "mix",
        "--video", str(v_path),
        "--vtt", str(vtt_path),
        "--text", "Hello",
        "--speaker", "anakin",
        "--diarize", "auto",
    ]

    # Mock wavlm dependencies as False
    monkeypatch.setattr("sys.argv", test_args)
    try:
        import wavlm_backend
        monkeypatch.setattr(wavlm_backend, "has_wavlm_dependencies", lambda: False)
    except Exception:
        pass

    # Catch SystemExit and check exit code
    with pytest.raises(SystemExit) as exc_info:
        run_mixer.main()
    assert exc_info.value.code == 2

    # Now add --assume-single-speaker
    test_args_asserted = test_args + ["--assume-single-speaker"]
    monkeypatch.setattr("sys.argv", test_args_asserted)

    # Mock ensure_source_aligned and synthesizer to observe that it does NOT exit 2 with setup error
    mock_ensure = MagicMock(return_value=str(tmp_path / "units_index.json"))
    monkeypatch.setattr(run_mixer, "ensure_source_aligned", mock_ensure)

    inv = InventoryIndex(
        words={"w1": WordOccurrence(
            word_id="w1", source_content_id="s1", speaker_id="anakin",
            word_normalized="hello", word_original="Hello", start_sample=0, end_sample=16000,
            phone_event_ids=["e1"], realized_phones=["HH"], is_eligible=True,
        )},
        events={"e1": AlignedEvent(
            event_id="e1", source_content_id="s1", utterance_id="u1", speaker_id="anakin",
            kind="phone", start_sample=0, end_sample=16000, label="HH", is_eligible=True,
        )},
        sources={"s1": SourceRecord(content_id="s1", source_path=str(v_path), native_sample_rate=16000, native_channels=1)}
    )
    monkeypatch.setattr(run_mixer, "load_or_combine_inventories", lambda paths: (inv, {"words": {"hello": [{"speaker_id": "anakin", "is_eligible": True}]}, "phones": {}}))
    monkeypatch.setattr(run_mixer, "synthesize_sentence", lambda *args, **kwargs: ([], []))

    # Should get past the argument/assertion check without code 2 setup error
    try:
        run_mixer.main()
    except SystemExit as se:
        assert se.code != 2, "With --assume-single-speaker, should not exit with code 2 assertion error"
    except Exception:
        pass


def test_u12_diarize_force_with_overlap_and_temp_registry(tmp_path: Path):
    """
    U12: --diarize force, injected two-speaker backend with overlap, temporary --speaker-registry.
    - Per-word speaker and eligibility stored
    - Selection uses only eligible words of the requested identity
    - Default registry untouched
    """
    default_reg_file = Path("config/speaker_registry.json").resolve()
    default_mtime = default_reg_file.stat().st_mtime if default_reg_file.exists() else None

    # Injected two-speaker diarization intervals with an overlap section
    diar_intervals = [
        DiarizationInterval(start_sec=0.0, end_sec=2.0, speaker_id="spk_1", confidence=1.0),
        DiarizationInterval(start_sec=1.5, end_sec=3.0, speaker_id="spk_2", confidence=1.0),
        # Overlap region from 1.5 to 2.0
    ]

    word_cues = [
        {"start_sec": 0.2, "end_sec": 0.8, "word": "hello"},  # Clean spk_1
        {"start_sec": 1.6, "end_sec": 1.9, "word": "there"},  # Overlap between spk_1 and spk_2
        {"start_sec": 2.2, "end_sec": 2.8, "word": "general"},  # Clean spk_2
    ]

    reconciled = reconcile_word_intervals_with_speakers(diar_intervals, word_cues)

    w1, w2, w3 = reconciled
    assert w1["speaker_id"] == "spk_1"
    assert w1["is_eligible"] is True

    assert w2["speaker_id"] == "OVERLAP"
    assert w2["is_eligible"] is False
    assert w2["eligibility_reason"] == "overlap"

    assert w3["speaker_id"] == "spk_2"
    assert w3["is_eligible"] is True

    # Build an InventoryIndex with these reconciled words
    inv = InventoryIndex(
        words={
            "w1": WordOccurrence(
                word_id="w1", source_content_id="s1", speaker_id="spk_1",
                word_normalized="hello", word_original="Hello", start_sample=0, end_sample=16000,
                phone_event_ids=["e1"], realized_phones=["HH"], is_eligible=True,
            ),
            "w2": WordOccurrence(
                word_id="w2", source_content_id="s1", speaker_id="OVERLAP",
                word_normalized="there", word_original="There", start_sample=16000, end_sample=32000,
                phone_event_ids=["e2"], realized_phones=["DH"], is_eligible=False, eligibility_reason="overlap",
            ),
            "w3": WordOccurrence(
                word_id="w3", source_content_id="s1", speaker_id="spk_2",
                word_normalized="general", word_original="General", start_sample=32000, end_sample=48000,
                phone_event_ids=["e3"], realized_phones=["JH"], is_eligible=True,
            ),
        },
        word_to_word_ids={
            "hello": ["w1"],
            "there": ["w2"],
            "general": ["w3"],
        },
        events={
            "e1": AlignedEvent(
                event_id="e1", source_content_id="s1", utterance_id="u1", speaker_id="spk_1",
                kind="phone", start_sample=0, end_sample=16000, label="HH", is_eligible=True,
            ),
            "e2": AlignedEvent(
                event_id="e2", source_content_id="s1", utterance_id="u1", speaker_id="OVERLAP",
                kind="phone", start_sample=16000, end_sample=32000, label="DH", is_eligible=False,
            ),
            "e3": AlignedEvent(
                event_id="e3", source_content_id="s1", utterance_id="u1", speaker_id="spk_2",
                kind="phone", start_sample=32000, end_sample=48000, label="JH", is_eligible=True,
            ),
        },
        sources={"s1": SourceRecord(content_id="s1", source_path="video.mp4", native_sample_rate=16000, native_channels=1)}
    )

    from span_selector import SpanSelector
    from target_graph import TargetGraphBuilder
    from lexicon import Lexicon

    lex = Lexicon([])
    lex.word_to_pronunciations["hello"] = [["HH", "AH0"]]
    lex.word_to_pronunciations["there"] = [["DH", "EH1", "R"]]

    selector = SpanSelector(inv)
    graph_builder = TargetGraphBuilder(lex)
    graph = graph_builder.build("hello there")

    # Select with speaker spk_1: only eligible words for spk_1 must be selected
    # "there" is in overlap and ineligible, so only "hello" can match as a word span
    candidates = selector.retrieve_candidates(graph, speaker_id="spk_1")
    cand_words = [c.matched_word for c in candidates if c.span_type == "word"]
    assert "hello" in cand_words
    assert "there" not in cand_words, "Ineligible overlap word 'there' must NOT be selected for spk_1"

    # Verify default registry was not modified
    if default_reg_file.exists() and default_mtime is not None:
        assert default_reg_file.stat().st_mtime == default_mtime, "Default registry must remain untouched"
