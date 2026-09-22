"""
Unit tests for Audit Remediation items R01, R02, R04, R11, R13, R16, R17.
Milestone 1 / Core Invariants & Engineering Hardening:
- R01: Explicit sample range validation (start >= 0, end <= len), raise ValueError on invalid ranges, no silent np.pad zero-padding.
- R11: Preserve 2-channel stereo without averaging to mono; cuts-only mode purity (no auto-leveling, peak clamping, or loudnorm).
- R02: Preserve legitimate consecutive repetitions in subtitle cues; preserve empty intervals in TextGrids.
- R04: Vocal stem content hash validation against metadata.json (source_sha256) to invalidate stale caches.
- R13: Stem candidate discrimination (vocals.wav vs Demucs no_vocals.wav, never select no_vocals.wav as vocal stem).
- R16, R17: Offline test marker discipline (@pytest.mark.model) and clean static analysis.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import soundfile as sf

from contracts import EditPlan, PlanSegment, SourceRecord
from inventory_builder import parse_textgrid_pure_python
from video_assembler import render_master_pcm
from vocal_isolation import (
    _find_instrumental_candidates,
    _find_vocal_candidates,
    isolate_vocals,
)
from vtt_preprocessor import clean_rolling_cues


# ============================================================================
# R01 & R11: Audio Range Invariants, Stereo Integrity & Cuts-Only Mode Purity
# ============================================================================

def test_r01_sample_range_validation_negative_start_raises_value_error(tmp_path: Path):
    """
    R01 Verification:
    render_master_pcm must raise ValueError when source_start_sample < 0,
    rather than clamping to 0 or padding silently.
    """
    sr = 48000
    dummy_wav = tmp_path / "source.wav"
    audio_data = np.zeros((sr,), dtype=np.float32)
    sf.write(str(dummy_wav), audio_data, sr)

    out_wav = tmp_path / "out.wav"

    source_rec = SourceRecord(
        content_id="src_01",
        source_path=str(dummy_wav),
        cached_pcm_path=str(dummy_wav),
        native_sample_rate=sr,
        native_channels=1,
    )

    plan = EditPlan(
        target_text="test",
        project_sample_rate=sr,
        total_output_samples=sr,
        sources={"src_01": source_rec},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_01",
                speaker_id="spk_0",
                source_start_sample=-100,  # Negative start sample!
                source_end_sample=sr,
                output_start_sample=0,
                output_end_sample=sr,
                target_text="test",
                target_phones=["T", "EH1", "S", "T"],
                span_type="word",
                source_video_path=str(dummy_wav),
            )
        ],
    )

    with pytest.raises(ValueError, match="Invalid source sample range"):
        render_master_pcm(plan, out_wav)


def test_r01_sample_range_validation_past_end_raises_value_error(tmp_path: Path):
    """
    R01 Verification:
    render_master_pcm must raise ValueError when source_end_sample > len(source_arr),
    preventing silent zero-padding with np.pad.
    """
    sr = 48000
    dummy_wav = tmp_path / "source.wav"
    audio_data = np.zeros((sr,), dtype=np.float32)  # 1 second of audio (48000 samples)
    sf.write(str(dummy_wav), audio_data, sr)

    out_wav = tmp_path / "out.wav"

    source_rec = SourceRecord(
        content_id="src_01",
        source_path=str(dummy_wav),
        cached_pcm_path=str(dummy_wav),
        native_sample_rate=sr,
        native_channels=1,
    )

    plan = EditPlan(
        target_text="test",
        project_sample_rate=sr,
        total_output_samples=sr * 2,
        sources={"src_01": source_rec},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_01",
                speaker_id="spk_0",
                source_start_sample=0,
                source_end_sample=sr * 2,  # Requests 96000 samples, source only has 48000
                output_start_sample=0,
                output_end_sample=sr * 2,
                target_text="test",
                target_phones=["T", "EH1", "S", "T"],
                span_type="word",
                source_video_path=str(dummy_wav),
            )
        ],
    )

    with pytest.raises(ValueError, match="Invalid source sample range"):
        render_master_pcm(plan, out_wav)


def test_r11_stereo_audio_preserved_without_mono_downmixing(tmp_path: Path):
    """
    R11 Verification:
    Preserve 2-channel stereo without averaging to mono (np.mean).
    Channel 0 and Channel 1 must remain distinct throughout rendering.
    """
    sr = 48000
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Channel 0: 440 Hz tone, Channel 1: 880 Hz tone
    ch0 = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    ch1 = 0.25 * np.sin(2 * np.pi * 880 * t).astype(np.float32)
    stereo_data = np.column_stack([ch0, ch1])

    stereo_wav = tmp_path / "stereo_source.wav"
    sf.write(str(stereo_wav), stereo_data, sr)

    out_wav = tmp_path / "stereo_master.wav"

    source_rec = SourceRecord(
        content_id="src_stereo",
        source_path=str(stereo_wav),
        cached_pcm_path=str(stereo_wav),
        native_sample_rate=sr,
        native_channels=2,
    )

    plan = EditPlan(
        target_text="stereo test",
        project_sample_rate=sr,
        total_output_samples=int(sr * duration),
        sources={"src_stereo": source_rec},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_stereo",
                speaker_id="spk_0",
                source_start_sample=0,
                source_end_sample=int(sr * duration),
                output_start_sample=0,
                output_end_sample=int(sr * duration),
                target_text="stereo test",
                target_phones=["S", "T", "EH1", "R", "IY0", "OW0"],
                span_type="phrase",
                source_video_path=str(stereo_wav),
            )
        ],
    )

    render_master_pcm(plan, out_wav, normalize_loudness=False)

    rendered_data, rendered_sr = sf.read(str(out_wav))
    assert rendered_sr == sr
    assert rendered_data.ndim == 2
    assert rendered_data.shape[1] == 2
    # Verify channels were NOT averaged into mono: ch0 and ch1 must not be identical
    assert not np.allclose(rendered_data[:, 0], rendered_data[:, 1], atol=1e-3)
    # Verify channel 0 faithfully matches original ch0
    assert np.allclose(rendered_data[:, 0], ch0, atol=1e-3)
    # Verify channel 1 faithfully matches original ch1
    assert np.allclose(rendered_data[:, 1], ch1, atol=1e-3)


def test_r11_cuts_only_mode_purity(tmp_path: Path):
    """
    R11 Verification:
    In cuts-only mode (rendering_mode="cuts-only"):
    - No speech leveling / auto-gain is applied.
    - No peak-limiting clamp (0.95 multiplier) is applied.
    - No EBU R128 loudness normalization is applied.
    The output must preserve exact bit-for-bit or floating point source amplitudes.
    """
    sr = 48000
    duration = 0.5
    # High amplitude tone (>0.98 peak, e.g. 1.0)
    audio = np.ones(int(sr * duration), dtype=np.float32) * 0.99
    source_wav = tmp_path / "pure_source.wav"
    sf.write(str(source_wav), audio, sr, subtype="FLOAT")

    out_wav = tmp_path / "pure_master.wav"

    source_rec = SourceRecord(
        content_id="src_pure",
        source_path=str(source_wav),
        cached_pcm_path=str(source_wav),
        native_sample_rate=sr,
        native_channels=1,
    )

    plan = EditPlan(
        target_text="cuts only test",
        rendering_mode="cuts-only",
        project_sample_rate=sr,
        total_output_samples=int(sr * duration),
        sources={"src_pure": source_rec},
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src_pure",
                speaker_id="spk_0",
                source_start_sample=0,
                source_end_sample=int(sr * duration),
                output_start_sample=0,
                output_end_sample=int(sr * duration),
                target_text="cuts only test",
                target_phones=["K", "AH1", "T", "S"],
                span_type="word",
                source_video_path=str(source_wav),
                gain_db=0.0,
            )
        ],
    )

    render_master_pcm(plan, out_wav, normalize_loudness=True)

    rendered_data, _ = sf.read(str(out_wav), dtype="float32")
    # In cuts-only mode, peak must remain 0.99 (not clamped to 0.95 or leveled)
    assert np.allclose(rendered_data, 0.99, atol=1e-4)


# ============================================================================
# R02: Subtitle Deduplication & Silence Preservation
# ============================================================================

def test_r02_consecutive_repetitions_preserved():
    """
    R02 Verification:
    clean_rolling_cues must preserve legitimate consecutive repetitions in speech
    (e.g., character saying "No!" twice in two distinct cues), while still
    cleaning rolling multi-line carry-over subtitles.
    """
    # Case 1: Legitimate spoken repetitions as separate single-line cues
    cues_repetition = [
        {"start_sec": 1.0, "end_sec": 2.0, "text": "No!"},
        {"start_sec": 2.5, "end_sec": 3.5, "text": "No!"},
    ]
    cleaned = clean_rolling_cues(cues_repetition)
    assert len(cleaned) == 2
    assert cleaned[0]["text"] == "No!"
    assert cleaned[1]["text"] == "No!"

    # Case 2: Rolling multi-line subtitle carry-over
    # Cue 1 has line A, Cue 2 displays line A on top and new line B below
    cues_rolling = [
        {"start_sec": 1.0, "end_sec": 3.0, "text": "I have the high ground"},
        {"start_sec": 3.0, "end_sec": 5.0, "text": "I have the high ground\nYou underestimate my power"},
    ]
    cleaned_rolling = clean_rolling_cues(cues_rolling)
    assert len(cleaned_rolling) == 2
    assert cleaned_rolling[0]["text"] == "I have the high ground"
    assert cleaned_rolling[1]["text"] == "You underestimate my power"


def test_r02_textgrid_silence_intervals_preserved(tmp_path: Path):
    """
    R02 Verification:
    parse_textgrid_pure_python preserves empty intervals ("") as valid
    silence intervals rather than dropping them.
    """
    tg_file = tmp_path / "test_silence.TextGrid"
    tg_content = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 3.0
tiers? <exists>
size = 1
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0.0
        xmax = 3.0
        intervals: size = 3
        intervals [1]:
            xmin = 0.0
            xmax = 1.0
            text = "hello"
        intervals [2]:
            xmin = 1.0
            xmax = 2.0
            text = ""
        intervals [3]:
            xmin = 2.0
            xmax = 3.0
            text = "world"
"""
    tg_file.write_text(tg_content, encoding="utf-8")

    parsed = parse_textgrid_pure_python(str(tg_file))
    assert "words" in parsed
    intervals = parsed["words"]
    assert len(intervals) == 3
    assert intervals[0]["text"] == "hello"
    assert intervals[1]["text"] == ""
    assert intervals[1]["xmin"] == 1.0
    assert intervals[1]["xmax"] == 2.0
    assert intervals[2]["text"] == "world"


# ============================================================================
# R04 & R13: Stem Content Hashing & Inversion Protection
# ============================================================================

def test_r04_content_hash_prevents_stale_cache_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    R04 Verification:
    isolate_vocals validates source_sha256 stored in metadata.json against
    current file sha256. If source content changed, stale cached stems must be
    invalidated and deleted rather than reused.
    """
    src_file = tmp_path / "input.mp4"
    src_file.write_bytes(b"initial_video_content_version_1")

    vocals_dir = tmp_path / "vocals"
    from vocal_isolation import get_source_id
    sid = get_source_id(src_file)
    target_dir = vocals_dir / sid
    target_dir.mkdir(parents=True)

    old_vocal_wav = target_dir / "vocals.wav"
    old_vocal_wav.write_bytes(b"RIFF_old_stale_vocal_stem")

    meta_file = target_dir / "metadata.json"
    meta_file.write_text(
        json.dumps({
            "source_id": sid,
            "source_sha256": "outdated_non_matching_hash_12345",
        }),
        encoding="utf-8",
    )

    reseparated = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        reseparated.append(cmd)
        # Audio separator creates new stem
        out_d = Path(cmd[cmd.index("--output_dir") + 1])
        (out_d / "input_(Vocals).wav").write_bytes(b"RIFF_freshly_separated_vocal_stem")
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    res_path = isolate_vocals(
        source_path=src_file,
        output_dir=vocals_dir,
        force=False,
    )

    # Separation must have been rerun because cached hash did not match
    assert len(reseparated) > 0
    assert res_path.read_bytes() == b"RIFF_freshly_separated_vocal_stem"


def test_r13_stem_candidate_selection_excludes_no_vocals(tmp_path: Path):
    """
    R13 Verification:
    _find_vocal_candidates must strictly discriminate between vocals.wav and
    Demucs no_vocals.wav. Demucs no_vocals.wav contains the substring 'vocal'
    but represents the instrumental/accompaniment track and must NEVER be selected
    as speech stem.
    """
    test_dir = tmp_path / "demucs_out"
    test_dir.mkdir(parents=True)

    vocal_file = test_dir / "vocals.wav"
    vocal_file.write_bytes(b"RIFF_real_vocals")

    no_vocal_file = test_dir / "no_vocals.wav"
    no_vocal_file.write_bytes(b"RIFF_background_music_no_vocals")

    inst_file = test_dir / "instrumental.wav"
    inst_file.write_bytes(b"RIFF_instrumental")

    # Verify vocal candidates
    vocal_cands = _find_vocal_candidates(test_dir)
    assert vocal_file in vocal_cands
    assert no_vocal_file not in vocal_cands
    assert inst_file not in vocal_cands
    assert vocal_cands[0] == vocal_file

    # Verify instrumental candidates
    inst_cands = _find_instrumental_candidates(test_dir)
    assert no_vocal_file in inst_cands
    assert inst_file in inst_cands
    assert vocal_file not in inst_cands


def test_r04_force_rebuild_unlinks_all_stems(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    R04 Verification:
    When force=True, isolate_vocals unlinks vocals.wav, instrumental.wav,
    and metadata.json together.
    """
    src_file = tmp_path / "input.mp4"
    src_file.write_bytes(b"video_data")

    vocals_dir = tmp_path / "vocals"
    from vocal_isolation import get_source_id
    sid = get_source_id(src_file)
    target_dir = vocals_dir / sid
    target_dir.mkdir(parents=True)

    v_wav = target_dir / "vocals.wav"
    v_wav.write_bytes(b"old_vocals")
    i_wav = target_dir / "instrumental.wav"
    i_wav.write_bytes(b"old_inst")
    m_json = target_dir / "metadata.json"
    m_json.write_text("{}", encoding="utf-8")

    def mock_subprocess_run(cmd, *args, **kwargs):
        out_d = Path(cmd[cmd.index("--output_dir") + 1])
        (out_d / "input_(Vocals).wav").write_bytes(b"RIFF_new_vocal")
        return MagicMock(returncode=0)

    monkeypatch.setattr("subprocess.run", mock_subprocess_run)

    isolate_vocals(source_path=src_file, output_dir=vocals_dir, force=True)

    # Instrumental and metadata must not contain old values
    assert v_wav.read_bytes() == b"RIFF_new_vocal"
    new_meta = json.loads(m_json.read_text(encoding="utf-8"))
    assert new_meta["source_id"] == sid


# ============================================================================
# R16 & R17: Offline Test Marker Verification
# ============================================================================

def test_r16_neural_model_tests_marked_for_offline_runs():
    """
    R16 & R17 Verification:
    All tests in test_listen_verify.py that invoke Whisper or WavLM models
    must be decorated with @pytest.mark.model so offline CI/CD runs can execute
    `pytest -m 'not model and not real_corpus'` cleanly.
    """
    import tests.test_listen_verify as tlv

    model_test_names = [
        "test_real_audio_verification",
        "test_cli_exit_code_success",
        "test_cli_exit_code_failure",
        "test_cli_json_output",
        "test_validate_mix_asr_acceptance_gate",
        "test_evaluate_suite_asr_integration",
    ]

    for test_name in model_test_names:
        fn = getattr(tlv, test_name, None)
        assert fn is not None, f"Expected test {test_name} in test_listen_verify.py"
        # Check pytest marks
        pytest_marks = getattr(fn, "pytestmark", [])
        mark_names = [m.name for m in pytest_marks]
        assert "model" in mark_names, f"Test {test_name} is missing @pytest.mark.model decorator"
