"""
Unit and integration tests for tools/listen_verify.py (Milestone M1 / R1)
Verifies:
- Text normalization
- Levenshtein DP token alignment (matches, substitutions, deletions, insertions)
- Character Error Rate (CER) calculation
- Phonetic homophone equivalence (e.g. MOXY vs MOXIE)
- Extraneous dialogue bleed rejection
- Mock and real ASR transcription listening verification
- CLI exit codes (0 for pass, 1 for threshold/bleed failure, 2 for operational error)
- JSON schema conformity
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
SRC_DIR = REPO_ROOT / "src"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from listen_verify import (
    ASRBackend,
    are_homophones,
    compute_cer,
    compute_levenshtein_tokens,
    extract_audio_16k,
    get_backend,
    main as listen_verify_main,
    normalize_text,
    verify_listening,
)
from validate_mix import validate_mix


class MockASRBackend(ASRBackend):
    """Mock ASR backend for deterministic test coverage."""

    def __init__(self, transcript_to_return: str):
        self.transcript_to_return = transcript_to_return
        self.last_initial_prompt: Optional[str] = None

    def transcribe(self, audio: np.ndarray, initial_prompt: Optional[str] = None) -> str:
        self.last_initial_prompt = initial_prompt
        return self.transcript_to_return


def test_normalize_text():
    assert normalize_text("Hello, World!") == "hello world"
    assert normalize_text("Anakin, I have the MOXY!") == "anakin i have the moxy"
    assert normalize_text("   Obi-Wan   Kenobi...  ") == "obi wan kenobi"
    assert normalize_text("“Don’t count on it—General.”") == "don t count on it general"
    assert normalize_text("   ") == ""
    assert normalize_text("") == ""
    assert normalize_text("!@#$%^&*()") == ""


def test_levenshtein_tokens_perfect_match():
    ref = ["hello", "there"]
    hyp = ["hello", "there"]
    dist, ops = compute_levenshtein_tokens(ref, hyp)
    assert dist == 0
    assert ops == [("M", "hello", "hello"), ("M", "there", "there")]


def test_levenshtein_tokens_substitution():
    ref = ["hello", "there"]
    hyp = ["hello", "where"]
    dist, ops = compute_levenshtein_tokens(ref, hyp, allow_homophones=False)
    assert dist == 1
    assert ops == [("M", "hello", "hello"), ("S", "there", "where")]


def test_levenshtein_tokens_deletion_missing_words():
    ref = ["anakin", "i", "have", "the", "moxy"]
    hyp = ["i", "have", "the"]
    dist, ops = compute_levenshtein_tokens(ref, hyp)
    assert dist == 2
    ops_types = [op[0] for op in ops]
    assert ops_types == ["D", "M", "M", "M", "D"]


def test_levenshtein_tokens_insertion_bleeding():
    ref = ["i", "have", "the"]
    hyp = ["i", "have", "the", "high", "ground"]
    dist, ops = compute_levenshtein_tokens(ref, hyp)
    assert dist == 2
    ops_types = [op[0] for op in ops]
    assert ops_types == ["M", "M", "M", "I", "I"]


def test_character_error_rate():
    assert compute_cer("hello", "hello") == 0.0
    assert compute_cer("", "") == 0.0
    assert compute_cer("", "hello") == 1.0
    assert compute_cer("cat", "cot") == pytest.approx(1 / 3)


def test_homophone_recognition():
    # moxy and moxie are phonetically identical in CMU / G2P
    assert are_homophones("moxy", "moxie") is True
    # completely different words
    assert are_homophones("moxy", "ground") is False

    ref = ["i", "have", "the", "moxy"]
    hyp = ["i", "have", "the", "moxie"]
    dist, ops = compute_levenshtein_tokens(ref, hyp, allow_homophones=True)
    assert dist == 0
    assert ops[-1] == ("M", "moxy", "moxie")


def test_verify_listening_mock_passed():
    mock_engine = MockASRBackend("Anakin, I have the MOXY!")
    dummy_audio = np.zeros(16000, dtype=np.float32)

    res = verify_listening(
        input_path="dummy.wav",
        expected_text="Anakin, I have the MOXY!",
        audio_array=dummy_audio,
        asr_backend=mock_engine,
    )

    assert res["passed"] is True
    assert res["wer"] == 0.0
    assert res["cer"] == 0.0
    assert res["all_expected_words_present"] is True
    assert res["bleed_detected"] is False
    assert res["missing_words"] == []
    assert res["bleeding_words"] == []
    assert res["word_matches"] == ["anakin", "i", "have", "the", "moxy"]
    assert mock_engine.last_initial_prompt == "Anakin, I have the MOXY!"


def test_verify_listening_mock_bleed_rejection():
    # Cut leaked "high ground"
    mock_engine = MockASRBackend("I have the high ground!")
    dummy_audio = np.zeros(16000, dtype=np.float32)

    res = verify_listening(
        input_path="dummy.wav",
        expected_text="I have the",
        audio_array=dummy_audio,
        asr_backend=mock_engine,
        disallow_insertions=True,
    )

    assert res["passed"] is False
    assert res["bleed_detected"] is True
    assert "high" in res["bleeding_words"]
    assert "ground" in res["bleeding_words"]
    assert any("bleed" in reason.lower() for reason in res["failure_reasons"])


def test_verify_listening_mock_missing_words():
    mock_engine = MockASRBackend("I have the")
    dummy_audio = np.zeros(16000, dtype=np.float32)

    res = verify_listening(
        input_path="dummy.wav",
        expected_text="Anakin, I have the MOXY!",
        audio_array=dummy_audio,
        asr_backend=mock_engine,
    )

    assert res["passed"] is False
    assert res["all_expected_words_present"] is False
    assert "anakin" in res["missing_words"]
    assert "moxy" in res["missing_words"]


def test_verify_listening_allow_insertions_flag():
    # When allow_insertions is True and WER is within threshold
    mock_engine = MockASRBackend("Hello there my friend")
    dummy_audio = np.zeros(16000, dtype=np.float32)

    res = verify_listening(
        input_path="dummy.wav",
        expected_text="Hello there",
        audio_array=dummy_audio,
        asr_backend=mock_engine,
        wer_threshold=0.50,
        disallow_insertions=False,
    )
    assert res["passed"] is False  # WER is 1.0 > 0.50
    assert res["bleed_detected"] is True

    # WER is 2/2 = 1.0, so with wer_threshold=1.5 it should pass if insertions allowed
    res_high_tol = verify_listening(
        input_path="dummy.wav",
        expected_text="Hello there",
        audio_array=dummy_audio,
        asr_backend=mock_engine,
        wer_threshold=1.5,
        disallow_insertions=False,
    )
    assert res_high_tol["passed"] is True
    assert res_high_tol["bleed_detected"] is True


@pytest.mark.model
def test_get_backend():
    # Test auto backend resolution
    backend = get_backend("auto")
    assert isinstance(backend, ASRBackend)



def test_verify_listening_cer_threshold():
    mock_engine = MockASRBackend("Hello therr")
    dummy_audio = np.zeros(16000, dtype=np.float32)

    res = verify_listening(
        input_path="dummy.wav",
        expected_text="Hello there",
        audio_array=dummy_audio,
        asr_backend=mock_engine,
        wer_threshold=0.50,
        max_cer=0.05,
    )
    assert res["passed"] is False
    assert any("CER" in r for r in res["failure_reasons"])


def test_audio_extraction_wav_and_mp4(tmp_path: Path):
    wav_p = tmp_path / "test_extract.wav"
    dummy_audio = np.zeros(16000, dtype=np.float32)
    sf.write(str(wav_p), dummy_audio, 16000)

    wav_audio = extract_audio_16k(wav_p)
    assert isinstance(wav_audio, np.ndarray)
    assert wav_audio.ndim == 1
    assert len(wav_audio) == 16000
    assert not np.any(np.isnan(wav_audio))

    mp4_p = REPO_ROOT / "output" / "smoke_basic.mp4"
    if mp4_p.exists():
        mp4_audio = extract_audio_16k(mp4_p)
        assert isinstance(mp4_audio, np.ndarray)
        assert mp4_audio.ndim == 1
        assert len(mp4_audio) > 0
        assert not np.any(np.isnan(mp4_audio))


@pytest.mark.model
@pytest.mark.real_corpus
def test_real_audio_verification():
    wav_p = REPO_ROOT / "output" / "smoke_basic.wav"
    mp4_p = REPO_ROOT / "output" / "smoke_basic.mp4"

    if wav_p.exists():
        res = verify_listening(
            input_path=wav_p,
            expected_text="Hello there",
            wer_threshold=0.20,
        )
        assert res["passed"] is True
        assert res["wer"] == 0.0
        assert "hello" in res["word_matches"]
        assert "there" in res["word_matches"]

    if mp4_p.exists():
        res_mp4 = verify_listening(
            input_path=mp4_p,
            expected_text="Hello there",
            wer_threshold=0.20,
        )
        assert res_mp4["passed"] is True
        assert res_mp4["wer"] == 0.0


@pytest.mark.model
@pytest.mark.real_corpus
def test_cli_exit_code_success():
    wav_p = REPO_ROOT / "output" / "smoke_basic.wav"
    if not wav_p.exists():
        pytest.skip("smoke_basic.wav not present")

    code = listen_verify_main(["--input", str(wav_p), "--expected", "Hello there"])
    assert code == 0


@pytest.mark.model
@pytest.mark.real_corpus
def test_cli_exit_code_failure():
    wav_p = REPO_ROOT / "output" / "smoke_basic.wav"
    if not wav_p.exists():
        pytest.skip("smoke_basic.wav not present")

    # Mismatched expected text
    code = listen_verify_main(["--input", str(wav_p), "--expected", "Completely unmatching utterance"])
    assert code == 1


def test_cli_exit_code_operational_error():
    code = listen_verify_main(["--input", "nonexistent_media_xyz123.wav", "--expected", "Hello"])
    assert code == 2


@pytest.mark.model
@pytest.mark.real_corpus
def test_cli_json_output(capsys):
    wav_p = REPO_ROOT / "output" / "smoke_basic.wav"
    if not wav_p.exists():
        pytest.skip("smoke_basic.wav not present")

    code = listen_verify_main(["--input", str(wav_p), "--expected", "Hello there", "--json"])
    assert code == 0
    captured = capsys.readouterr()
    data = json.loads(captured.out)

    assert data["passed"] is True
    assert data["wer"] == 0.0
    assert "word_matches" in data
    assert "missing_words" in data
    assert "bleeding_words" in data
    assert "normalized_expected" in data
    assert "normalized_transcribed" in data


@pytest.mark.model
@pytest.mark.real_corpus
def test_validate_mix_asr_acceptance_gate():
    smoke_stem = REPO_ROOT / "output" / "smoke_basic"
    if smoke_stem.with_suffix(".plan.json").exists() and smoke_stem.with_suffix(".wav").exists():
        # Clean smoke mix passes ASR verification
        assert validate_mix(stem=str(smoke_stem), require_asr=True) is True

    moxy_stem = REPO_ROOT / "output" / "moxy"
    if moxy_stem.with_suffix(".plan.json").exists() and moxy_stem.with_suffix(".wav").exists():
        # Moxy mix has "high ground" bleed and must fail ASR verification
        assert validate_mix(stem=str(moxy_stem), require_asr=True) is False


@pytest.mark.model
@pytest.mark.real_corpus
def test_evaluate_suite_asr_integration(monkeypatch, tmp_path):
    import shutil
    from evaluate import evaluate_suite

    out_dir = tmp_path / "eval_out"
    suite_file = tmp_path / "test_suite.json"
    smoke_wav = REPO_ROOT / "output" / "smoke_basic.wav"
    if not smoke_wav.exists():
        pytest.skip("smoke_basic.wav not found")

    suite_data = {
        "suite_name": "Test Integration Suite",
        "version": "1.0",
        "manifest": "smoke_sources.json",
        "speaker_id": "obi-wan",
        "cases": [
            {
                "case_id": "case_test_01",
                "stratum": "intact",
                "target_text": "Hello there",
            }
        ],
    }
    suite_file.write_text(json.dumps(suite_data), encoding="utf-8")

    class FakeCompletedProcess:
        returncode = 0

    orig_run = subprocess.run

    def fake_subprocess_run(cmd, *args, **kwargs):
        if "--render" in cmd:
            render_idx = cmd.index("--render") + 1
            mp4_path = Path(cmd[render_idx])
            wav_path = mp4_path.with_suffix(".wav")

            mp4_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(smoke_wav, wav_path)
            mp4_path.write_bytes(b"dummy_mp4_container")
            if "--plan" in cmd:
                plan_idx = cmd.index("--plan") + 1
                plan_path = Path(cmd[plan_idx])
                plan_path.write_text(json.dumps({"total_output_samples": 24000, "segments": []}))
            return FakeCompletedProcess()
        return orig_run(cmd, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", fake_subprocess_run)

    report = evaluate_suite(str(suite_file), str(out_dir), verify_asr=True)

    assert report["cases_evaluated"] == 1
    assert report["summary"]["total_asr_evaluated"] == 2  # span_basic and legacy
    assert report["summary"]["mean_wer"] == 0.0
    assert report["summary"]["asr_pass_rate"] == 1.0

    c0_span = report["results"][0]["configurations"]["span_basic"]
    assert "asr" in c0_span
    assert c0_span["asr"]["passed"] is True
    assert c0_span["asr"]["wer"] == 0.0
    assert "hello" in c0_span["asr"]["word_matches"]

    # Also test with verify_asr=False
    report_no_asr = evaluate_suite(str(suite_file), str(tmp_path / "eval_out_no_asr"), verify_asr=False)
    assert report_no_asr["summary"]["total_asr_evaluated"] == 0
    c0_no_asr = report_no_asr["results"][0]["configurations"]["span_basic"]
    assert "asr" not in c0_no_asr


def test_substitution_decoupled_from_bleed_and_require_all_words():
    """
    Verifies that substitutions are not classified as dialogue bleed,
    and that missing target words are tolerated under wer_threshold unless require_all_words=True.
    """
    mock_backend = MockASRBackend("the bat sat on the mat")
    dummy_audio = np.zeros(16000, dtype=np.float32)

    # 1. With tolerance 0.25 (1/6 = 0.1667 <= 0.25) and require_all_words=False -> Should pass!
    res_tolerated = verify_listening(
        input_path="dummy.wav",
        expected_text="the cat sat on the mat",
        asr_backend=mock_backend,
        audio_array=dummy_audio,
        wer_threshold=0.25,
        require_all_words=False,
    )
    assert res_tolerated["passed"] is True
    assert res_tolerated["bleeding_words"] == []
    assert res_tolerated["bleed_detected"] is False
    assert res_tolerated["missing_words"] == ["cat"]

    # 2. With require_all_words=True -> Should fail due to missing word "cat"
    res_required = verify_listening(
        input_path="dummy.wav",
        expected_text="the cat sat on the mat",
        asr_backend=mock_backend,
        audio_array=dummy_audio,
        wer_threshold=0.25,
        require_all_words=True,
    )
    assert res_required["passed"] is False
    assert any("Missing target words" in r for r in res_required["failure_reasons"])
    assert res_required["bleeding_words"] == []  # Still not classified as bleed

    # 3. With genuine insertion -> Should detect bleed
    mock_bleed = MockASRBackend("the cat sat on the mat outside")
    res_bleed = verify_listening(
        input_path="dummy.wav",
        expected_text="the cat sat on the mat",
        asr_backend=mock_bleed,
        audio_array=dummy_audio,
        wer_threshold=0.50,
        disallow_insertions=True,
    )
    assert res_bleed["passed"] is False
    assert res_bleed["bleed_detected"] is True
    assert "outside" in res_bleed["bleeding_words"]
    assert any("Unintended dialogue bleed" in r for r in res_bleed["failure_reasons"])


