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


def test_t20_cli_option_combinations_and_speaker_enforcement():
    """
    T20: Legacy/default wrong speaker, guide/features, cuts-only, requested plan:
    Enforce shared behavior or reject unsupported combinations explicitly.
    """
    # 1. Cuts-only mode with explicit --normalize-loudness must fail with clear conflict error
    res1 = run_cmd([str(RUN_MIXER), "mix", "--video", "dummy.mp4", "--vtt", "dummy.vtt", "--text", "hello", "--mode", "cuts-only", "--normalize-loudness"])
    assert res1.returncode != 0
    assert "cuts-only" in res1.stderr.lower() or "cuts-only" in res1.stdout.lower()

    # 2. Legacy selector with --guide must fail
    res2 = run_cmd([str(RUN_MIXER), "mix", "--video", "dummy.mp4", "--vtt", "dummy.vtt", "--text", "hello", "--selector", "legacy", "--guide", "guide.wav"])
    assert res2.returncode != 0
    assert "legacy selector does not support" in res2.stderr.lower() or "legacy selector does not support" in res2.stdout.lower()

    # 3. Legacy selector with --plan must fail
    res3 = run_cmd([str(RUN_MIXER), "mix", "--video", "dummy.mp4", "--vtt", "dummy.vtt", "--text", "hello", "--selector", "legacy", "--plan", "plan.json"])
    assert res3.returncode != 0
    assert "legacy selector does not generate editplan" in res3.stderr.lower() or "legacy selector does not generate editplan" in res3.stdout.lower()


def test_t23_atomic_publication_and_failure_isolation(tmp_path: Path):
    """
    T23: Invalid source/plan, missing video, failure after each artifact stage:
    Explicit error; prior complete generation preserved; no fake completed output.
    """
    out_dir = tmp_path / "mix_output"
    out_dir.mkdir(parents=True)
    out_render = out_dir / "target_mix.mp4"
    out_wav = out_dir / "target_mix.wav"
    out_plan = out_dir / "target_mix.plan.json"

    # Write existing prior generation
    out_render.write_bytes(b"prior_mp4_content_12345")
    out_wav.write_bytes(b"prior_wav_content_67890")
    out_plan.write_text('{"prior": "plan"}', encoding="utf-8")

    # Run mix targeting this output with a nonexistent / invalid input video
    res = run_cmd([
        str(RUN_MIXER), "mix",
        "--video", str(tmp_path / "nonexistent_source.mp4"),
        "--text", "hello there",
        "--render", str(out_render),
        "--plan", str(out_plan),
        "--selector", "span",
        "--non-interactive",
    ])

    assert res.returncode != 0, f"Expected non-zero exit code on missing source, got {res.returncode}"
    # Assert prior complete generation is strictly preserved and not overwritten or corrupted
    assert out_render.read_bytes() == b"prior_mp4_content_12345"
    assert out_wav.read_bytes() == b"prior_wav_content_67890"
    assert out_plan.read_text(encoding="utf-8") == '{"prior": "plan"}'

    # Assert no stray attempt directory left behind
    attempt_dirs = list(out_dir.glob(".attempt_*"))
    assert len(attempt_dirs) == 0


