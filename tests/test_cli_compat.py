"""
Tests for CLI interface compatibility and entry point retention.
Verifies that all existing subcommands, arguments, and standalone tools work.
"""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RUN_MIXER = ROOT_DIR / "run_mixer.py"
SRC_DIR = ROOT_DIR / "src"


def run_cmd(args):
    cmd = [PYTHON] + args
    res = subprocess.run(cmd, cwd=str(ROOT_DIR), capture_output=True, text=True)
    return res


def test_cli_help():
    res = run_cmd([str(RUN_MIXER), "--help"])
    assert res.returncode == 0
    help_out = res.stdout.lower()
    assert "mix" in help_out
    assert "auto" in help_out
    assert "preprocess" in help_out
    assert "align" in help_out
    assert "build-inventory" in help_out


def test_subcommand_helps():
    for sub in ["mix", "auto", "preprocess", "align", "build-inventory"]:
        res = run_cmd([str(RUN_MIXER), sub, "--help"])
        assert res.returncode == 0, f"run_mixer.py {sub} --help failed: {res.stderr}"
        assert f"usage: run_mixer.py {sub}" in res.stdout or sub in res.stdout


def test_standalone_tool_helps():
    tools = [
        SRC_DIR / "vtt_preprocessor.py",
        SRC_DIR / "inventory_builder.py",
        SRC_DIR / "unit_selector.py",
        SRC_DIR / "inspect_alignment.py",
    ]
    for tool in tools:
        res = run_cmd([str(tool), "--help"])
        assert res.returncode == 0, f"{tool.name} --help failed: {res.stderr}"


def test_slugify_text():
    sys.path.insert(0, str(ROOT_DIR))
    from run_mixer import slugify_text

    assert slugify_text("Hello World!") == "hello_world"
    assert slugify_text("Anakin, Chancellor Anakin is evil!!") == "anakin_chancellor_anakin_is_evil"

    # Test word boundary truncation and max length limit
    long_phrase = "You have allowed this Sith lord to twist your mind until now you have become the very thing you swore to destroy"
    slug = slugify_text(long_phrase, max_words=8, max_length=40)
    assert len(slug) <= 40
    assert not slug.endswith("_")
    # All parts must be full words
    for word in slug.split("_"):
        assert word in long_phrase.lower()

    # Empty and special character strings
    assert slugify_text("!?!?") == "mix"
    assert slugify_text("") == "mix"


def test_audio_track_cli_argument():
    res = run_cmd([str(RUN_MIXER), "mix", "--help"])
    assert res.returncode == 0
    assert "--audio-track" in res.stdout
    assert "combined" in res.stdout
    assert "isolated" in res.stdout
    assert "both" in res.stdout

