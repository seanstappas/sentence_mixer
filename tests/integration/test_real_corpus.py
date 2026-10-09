"""
Real-corpus integration test for actual MFA alignment and render validation.
Marked real_corpus.
Fails with an actionable prerequisite error if verified inputs are missing.
"""

import json
import pytest
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
PYTHON = sys.executable
RUN_MIXER = REPO_ROOT / "run_mixer.py"
SMOKE_SOURCES = REPO_ROOT / "evaluation" / "smoke_sources.json"


@pytest.mark.real_corpus
def test_real_corpus_smoke_manifest(tmp_path: Path):
    if not SMOKE_SOURCES.exists():
        pytest.fail(f"Prerequisite missing: {SMOKE_SOURCES} does not exist.")

    with open(SMOKE_SOURCES, "r", encoding="utf-8") as f:
        sources = json.load(f)

    if not sources:
        pytest.fail(f"Prerequisite missing: {SMOKE_SOURCES} contains no sources.")

    for item in sources:
        video_rel = item.get("video")
        v_path = (SMOKE_SOURCES.parent / video_rel).resolve()
        if not v_path.exists():
            pytest.fail(f"Prerequisite missing: Source video file not found at {v_path}")

    # Run smoke mix in isolated tmp_path
    smoke_mp4 = tmp_path / "smoke_basic.mp4"
    smoke_plan = tmp_path / "smoke_basic.plan.json"
    smoke_xml = tmp_path / "smoke_basic.xml"

    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--manifest", str(SMOKE_SOURCES),
        "--speaker", "obi-wan",
        "--text", "Hello there",
        "--selector", "span",
        "--features", "basic",
        "--mode", "source-derived",
        "--fps", "30000/1001",
        "--render", str(smoke_mp4),
        "--plan", str(smoke_plan),
        "--xml", str(smoke_xml),
    ]

    res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
    assert res.returncode == 0, f"run_mixer.py mix failed:\n{res.stderr}\n{res.stdout}"
    assert smoke_mp4.exists()
    assert smoke_plan.exists()
