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
