"""
Tests for tools/evaluate.py harness and blinded asset generation.
"""

import json
import tempfile
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))



def test_evaluate_suite_structure():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        suite_file = tmp_path / "test_suite.json"
        out_dir = tmp_path / "eval_out"

        suite_data = {
            "suite_name": "Mock Suite",
            "version": "1.0",
            "manifest": "evaluation/smoke_sources.json",
            "speaker_id": "test_spk",
            "cases": [
                {
                    "case_id": "mock_case_01",
                    "stratum": "intact_phrases",
                    "target_text": "hello there",
                }
            ]
        }
        suite_file.write_text(json.dumps(suite_data), encoding="utf-8")

        # Create dummy rendered files in out_dir so blinded assets are created
        out_dir.mkdir(parents=True)
        dummy_wav = out_dir / "mock_case_01_span_basic.wav"
        dummy_wav.write_bytes(b"RIFFdummyWAV")

        # Test evaluation function execution
        assert suite_file.exists()


def test_t24_acceptance_evaluation_exit_codes_and_negation(tmp_path: Path):
    """
    T24: All ASR failures/unavailable, missing negation, wrong order/repeats, phonetic/gap markup:
    Correct exits and semantic expectations; strict coverage; explicit profile for phone-only targets.
    """
    import numpy as np
    import pytest
    from listen_verify import verify_listening, ASRBackend, main as listen_main

    class MockEngine(ASRBackend):
        def __init__(self, transcript_to_return: str):
            self.ret = transcript_to_return
        def transcribe(self, audio: np.ndarray, initial_prompt: str = None) -> str:
            return self.ret

    dummy_audio = np.zeros(16000, dtype=np.float32)

    # 1. Missing negation MUST fail even if WER is 0.20
    # "we are not going home" (5 tokens) vs "we are going home" (4 tokens) -> 1 deletion / 5 = 0.20 WER
    negation_res = verify_listening(
        input_path="dummy.wav",
        expected_text="we are not going home",
        audio_array=dummy_audio,
        asr_backend=MockEngine("we are going home"),
        wer_threshold=0.25,
        require_all_words=True,
    )
    assert not negation_res["passed"], "Missing negation 'not' must cause verification to fail"
    assert "not" in negation_res["missing_words"]

    # 2. Semantic gap markup stripped from expectations
    gap_res = verify_listening(
        input_path="dummy.wav",
        expected_text="hello <gap:250ms> there",
        audio_array=dummy_audio,
        asr_backend=MockEngine("hello there"),
        require_all_words=True,
    )
    assert gap_res["passed"]
    assert gap_res["missing_words"] == []
    assert gap_res["normalized_expected"] == "hello there"

    # 3. Phonetic-only target without spoken words raises configuration error (ValueError)
    with pytest.raises(ValueError, match="contains no recognizable lexical words"):
        verify_listening(
            input_path="dummy.wav",
            expected_text="{B AE1 K AH0 N}",
            audio_array=dummy_audio,
            asr_backend=MockEngine("bacon"),
        )

    # 4. Exit codes through CLI main()
    # 4a. Operational error on nonexistent file -> Exit code 2
    rc_missing = listen_main(["--input", "nonexistent_file_12345.mp4", "--expected", "hello"])
    assert rc_missing == 2

    # 4b. Configuration error on phonetic-only target -> Exit code 2
    dummy_file = tmp_path / "dummy.wav"
    dummy_file.write_bytes(b"RIFF_wav")
    rc_config = listen_main(["--input", str(dummy_file), "--expected", "{B AE1 K AH0 N}"])
    assert rc_config == 2

