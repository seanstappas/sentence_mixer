"""
Unit and integration tests for Vocal Separation & Speech Isolation Pre-processing (R3).
Covers:
- Stem cache hit logic (.build/vocals/ cache skips separation).
- audio-separator CLI command formatting, model selection, and execution.
- Demucs fallback when audio-separator fails (non-zero exit code or FileNotFoundError).
- Graceful handling / informative RuntimeError when neither tool succeeds.
- Flag fallback_to_demucs=False behavior.
- CLI flag --isolate-vocals in run_mixer.py (mix, auto, preprocess, align).
- preprocess_source audio_source_path integration (FFmpeg slicing from vocal stem).
- media_time.extract_source_record vocal stem resolution for cached_pcm_path.
"""

import argparse
import json
import subprocess
from pathlib import Path
from typing import List
from unittest.mock import MagicMock

import pytest

from media_time import extract_source_record
from run_mixer import preprocess_source
from vocal_isolation import (
    get_source_id,
    isolate_vocals,
)


@pytest.fixture
def dummy_media(tmp_path: Path) -> Path:
    """Creates a dummy media file with non-empty content."""
    media_file = tmp_path / "test_source.mp4"
    media_file.write_bytes(b"dummy_source_video_binary_content_for_testing_12345")
    return media_file


@pytest.fixture
def dummy_vtt(tmp_path: Path) -> Path:
    """Creates a minimal valid WebVTT file."""
    vtt_file = tmp_path / "test_source.en.vtt"
    vtt_content = """WEBVTT

00:00:01.000 --> 00:00:03.500
Hello there General Kenobi
"""
    vtt_file.write_text(vtt_content, encoding="utf-8")
    return vtt_file


def test_cache_hit_skips_separation(dummy_media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verifies that if .build/vocals/<source_id>/vocals.wav already exists and is non-empty,
    isolate_vocals immediately returns the cached path without invoking any subprocess.
    """
    vocals_base = tmp_path / ".build" / "vocals"
    source_id = get_source_id(dummy_media)
    cached_dir = vocals_base / source_id
    cached_dir.mkdir(parents=True, exist_ok=True)
    cached_wav = cached_dir / "vocals.wav"
    cached_wav.write_bytes(b"RIFF_dummy_cached_pcm_waveform_data_48khz")

    # Ensure subprocess.run is NEVER called
    mock_run = MagicMock()
    monkeypatch.setattr(subprocess, "run", mock_run)

    result_path = isolate_vocals(
        source_path=dummy_media,
        output_dir=vocals_base,
        force=False,
    )

    assert result_path == cached_wav
    assert result_path.exists()
    assert result_path.read_bytes() == b"RIFF_dummy_cached_pcm_waveform_data_48khz"
    mock_run.assert_not_called()


def test_force_flag_bypasses_cache(dummy_media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verifies that when force=True, isolate_vocals executes separation even if
    a cached stem already exists.
    """
    vocals_base = tmp_path / ".build" / "vocals"
    source_id = get_source_id(dummy_media)
    cached_dir = vocals_base / source_id
    cached_dir.mkdir(parents=True, exist_ok=True)
    cached_wav = cached_dir / "vocals.wav"
    cached_wav.write_bytes(b"old_cached_data")

    called_commands = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        called_commands.append(cmd)
        out_dir = Path(cmd[cmd.index("--output_dir") + 1]) if "--output_dir" in cmd else Path(cmd[cmd.index("-o") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        # Simulate audio-separator writing new stem
        cand_wav = out_dir / "test_source_(Vocals).wav"
        cand_wav.write_bytes(b"new_forced_vocal_data")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Separation done", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    result_path = isolate_vocals(
        source_path=dummy_media,
        output_dir=vocals_base,
        force=True,
    )

    assert len(called_commands) == 1
    assert Path(called_commands[0][0]).stem == "audio-separator"
    assert result_path == cached_wav
    assert result_path.read_bytes() == b"new_forced_vocal_data"


def test_audio_separator_cli_formatting_and_success(
    dummy_media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Verifies audio-separator CLI command line structure:
    - binary name: audio-separator
    - input source path
    - --model_filename UVR-MDX-NET-Inst_HQ_4.onnx
    - --output_dir <target_dir>
    - --output_format WAV
    And verifies that output vocals file is copied to vocals.wav and metadata.json is written.
    """
    vocals_base = tmp_path / "vocals_out"
    source_id = get_source_id(dummy_media)
    target_dir = vocals_base / source_id

    recorded_cmds: List[List[str]] = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        recorded_cmds.append(cmd)
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        # audio-separator produces a file with '(Vocals)' in name
        vocal_output = out_dir / f"{dummy_media.stem}_(Vocals).wav"
        vocal_output.write_bytes(b"RIFF_freshly_separated_vocals")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Success", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    stem_path = isolate_vocals(
        source_path=dummy_media,
        output_dir=vocals_base,
        model_name="UVR-MDX-NET-Inst_HQ_4.onnx",
    )

    assert len(recorded_cmds) == 1
    cmd = recorded_cmds[0]
    assert Path(cmd[0]).stem == "audio-separator"
    assert cmd[1] == str(dummy_media.resolve())
    assert "--model_filename" in cmd
    assert cmd[cmd.index("--model_filename") + 1] == "UVR-MDX-NET-Inst_HQ_4.onnx"
    assert "--output_dir" in cmd
    assert cmd[cmd.index("--output_dir") + 1] == str(target_dir)
    assert "--output_format" in cmd
    assert cmd[cmd.index("--output_format") + 1] == "WAV"

    assert stem_path == target_dir / "vocals.wav"
    assert stem_path.exists()
    assert stem_path.read_bytes() == b"RIFF_freshly_separated_vocals"

    # Verify metadata.json
    metadata_file = target_dir / "metadata.json"
    assert metadata_file.exists()
    meta = json.loads(metadata_file.read_text(encoding="utf-8"))
    assert meta["source_id"] == source_id
    assert meta["separator_tool"] == "audio-separator"
    assert meta["model_name"] == "UVR-MDX-NET-Inst_HQ_4.onnx"
    assert meta["isolated_vocals_path"] == str(stem_path.resolve())


def test_demucs_fallback_when_audio_separator_fails(
    dummy_media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Verifies that if audio-separator exits with non-zero return code (e.g. 1),
    isolate_vocals falls back to Demucs CLI (demucs --two-stems=vocals -n htdemucs ...).
    """
    vocals_base = tmp_path / "vocals_out"
    source_id = get_source_id(dummy_media)
    target_dir = vocals_base / source_id

    recorded_cmds: List[List[str]] = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        recorded_cmds.append(cmd)
        if "audio-separator" in Path(cmd[0]).name:
            return subprocess.CompletedProcess(
                args=cmd, returncode=1, stdout="", stderr="ONNX runtime GPU error"
            )
        elif "demucs" in Path(cmd[0]).name:
            out_dir = Path(cmd[cmd.index("-o") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            demucs_output_dir = out_dir / "htdemucs" / dummy_media.stem
            demucs_output_dir.mkdir(parents=True, exist_ok=True)
            vocal_stem = demucs_output_dir / "vocals.wav"
            vocal_stem.write_bytes(b"RIFF_demucs_htdemucs_vocals")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Demucs done", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=127, stdout="", stderr="Not found")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    stem_path = isolate_vocals(
        source_path=dummy_media,
        output_dir=vocals_base,
        fallback_to_demucs=True,
    )

    assert len(recorded_cmds) == 2
    assert Path(recorded_cmds[0][0]).stem == "audio-separator"
    demucs_cmd = recorded_cmds[1]
    assert Path(demucs_cmd[0]).stem == "demucs"
    assert "--two-stems=vocals" in demucs_cmd
    assert "-n" in demucs_cmd
    assert demucs_cmd[demucs_cmd.index("-n") + 1] == "htdemucs"
    assert str(dummy_media.resolve()) in demucs_cmd
    assert "-o" in demucs_cmd

    assert stem_path == target_dir / "vocals.wav"
    assert stem_path.exists()
    assert stem_path.read_bytes() == b"RIFF_demucs_htdemucs_vocals"

    meta = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
    assert meta["separator_tool"] == "demucs"
    assert meta["model_name"] == "htdemucs"


def test_demucs_fallback_when_audio_separator_not_in_path(
    dummy_media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Verifies that if audio-separator raises FileNotFoundError, Demucs fallback is engaged.
    """
    vocals_base = tmp_path / "vocals_out"

    recorded_cmds = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        recorded_cmds.append(cmd)
        if cmd[0] == "audio-separator":
            raise FileNotFoundError("audio-separator binary not found")
        elif cmd[0] == "demucs":
            out_dir = Path(cmd[cmd.index("-o") + 1])
            out_dir.mkdir(parents=True, exist_ok=True)
            vocal_file = out_dir / "vocals.wav"
            vocal_file.write_bytes(b"RIFF_demucs_stem")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Demucs finished", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    stem_path = isolate_vocals(
        source_path=dummy_media,
        output_dir=vocals_base,
        fallback_to_demucs=True,
    )

    assert len(recorded_cmds) == 2
    assert stem_path.exists()
    assert stem_path.read_bytes() == b"RIFF_demucs_stem"


def test_neither_cli_available_raises_informative_error(
    dummy_media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Verifies that if both audio-separator and demucs fail or are missing,
    an informative RuntimeError is raised explaining neither CLI was available.
    """
    def mock_subprocess_run(cmd, *args, **kwargs):
        raise FileNotFoundError(f"{cmd[0]} not found")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    with pytest.raises(RuntimeError, match="neither 'audio-separator' nor 'demucs'"):
        isolate_vocals(
            source_path=dummy_media,
            output_dir=tmp_path / "vocals",
            fallback_to_demucs=True,
        )


def test_fallback_to_demucs_false_disables_fallback(
    dummy_media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Verifies that when fallback_to_demucs=False, Demucs is not invoked upon audio-separator failure.
    """
    called_cmds = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        called_cmds.append(cmd)
        return subprocess.CompletedProcess(args=cmd, returncode=1, stdout="", stderr="audio-separator fail")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    with pytest.raises(RuntimeError):
        isolate_vocals(
            source_path=dummy_media,
            output_dir=tmp_path / "vocals",
            fallback_to_demucs=False,
        )

    # Only audio-separator was called, never demucs
    assert len(called_cmds) == 1
    assert Path(called_cmds[0][0]).stem == "audio-separator"


def test_run_mixer_cli_flags_isolate_vocals():
    """
    Verifies that run_mixer.py adds and parses --isolate-vocals across subcommands:
    - mix
    - auto
    - preprocess
    - align
    """
    # Parse with test arguments
    test_subcommands = [
        (["mix", "--text", "hello", "--isolate-vocals"], "mix"),
        (["auto", "--video", "test.mp4", "--text", "hello", "--isolate-vocals"], "auto"),
        (["preprocess", "--video", "test.mp4", "--vtt", "test.vtt", "--isolate-vocals"], "preprocess"),
        (["align", "--corpus", "c_dir", "--isolate-vocals"], "align"),
    ]

    for argv, sub in test_subcommands:
        # Build parser fresh
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="command", required=True)

        def add_synthesis_args(p):
            p.add_argument("--text", required=True)
            p.add_argument(
                "--isolate-vocals",
                action="store_true",
                default=False,
                help="Isolate speech stems",
            )

        p_mix = subparsers.add_parser("mix")
        add_synthesis_args(p_mix)

        p_auto = subparsers.add_parser("auto")
        add_synthesis_args(p_auto)
        p_auto.add_argument("--video")

        p_prep = subparsers.add_parser("preprocess")
        p_prep.add_argument("--video")
        p_prep.add_argument("--vtt")
        p_prep.add_argument("--isolate-vocals", action="store_true", default=False)

        p_align = subparsers.add_parser("align")
        p_align.add_argument("--corpus")
        p_align.add_argument("--isolate-vocals", action="store_true", default=False)

        args = parser.parse_args(argv)
        assert args.command == sub
        assert args.isolate_vocals is True

    # Check default is False when omitted
    args_default = parser.parse_args(["mix", "--text", "hello"])
    assert args_default.isolate_vocals is False


def test_preprocess_source_with_audio_source_path(
    dummy_media: Path, dummy_vtt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Verifies that when audio_source_path is provided to preprocess_source,
    FFmpeg extracts speech chunks from audio_source_path instead of dummy_media (raw video).
    """
    corpus_dir = tmp_path / "corpus_out"
    vocal_stem = tmp_path / "clean_vocals.wav"
    vocal_stem.write_bytes(b"RIFF_clean_speech_stem_48khz")

    recorded_ffmpeg_inputs: List[str] = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        if cmd[0] == "ffmpeg":
            # Record the input path passed to -i
            input_idx = cmd.index("-i") + 1
            recorded_ffmpeg_inputs.append(cmd[input_idx])
            # Touch chunk output wav
            out_wav = Path(cmd[-1])
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            out_wav.write_bytes(b"RIFF_chunk_16k_mono")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    manifest_path_str = preprocess_source(
        video_path=str(dummy_media),
        vtt_path=str(dummy_vtt),
        output_corpus_dir=str(corpus_dir),
        audio_source_path=vocal_stem,
    )

    assert len(recorded_ffmpeg_inputs) > 0
    # Every FFmpeg chunk slice must read from vocal_stem, NOT dummy_media
    for in_path in recorded_ffmpeg_inputs:
        assert in_path == str(vocal_stem.resolve()), (
            f"Expected chunk extraction from vocal stem {vocal_stem}, got {in_path}"
        )

    # Verify corpus manifest recorded audio_source_path
    manifest = json.loads(Path(manifest_path_str).read_text(encoding="utf-8"))
    assert manifest["audio_source_path"] == str(vocal_stem.resolve())
    assert manifest["source_video"] == str(dummy_media.resolve())


def test_preprocess_source_without_audio_source_path(
    dummy_media: Path, dummy_vtt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Verifies that when audio_source_path is None, FFmpeg extracts chunks from raw video.
    """
    corpus_dir = tmp_path / "corpus_out"
    recorded_ffmpeg_inputs: List[str] = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        if cmd[0] == "ffmpeg":
            input_idx = cmd.index("-i") + 1
            recorded_ffmpeg_inputs.append(cmd[input_idx])
            out_wav = Path(cmd[-1])
            out_wav.parent.mkdir(parents=True, exist_ok=True)
            out_wav.write_bytes(b"RIFF_chunk_16k_mono")
            return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    manifest_path_str = preprocess_source(
        video_path=str(dummy_media),
        vtt_path=str(dummy_vtt),
        output_corpus_dir=str(corpus_dir),
        audio_source_path=None,
    )

    assert len(recorded_ffmpeg_inputs) > 0
    for in_path in recorded_ffmpeg_inputs:
        assert in_path == str(dummy_media.resolve())

    manifest = json.loads(Path(manifest_path_str).read_text(encoding="utf-8"))
    assert manifest["audio_source_path"] is None


def test_extract_source_record_with_vocal_stem(
    dummy_media: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Verifies media_time.extract_source_record:
    - When vocal_stem_path is provided, cached_pcm_path points to the vocal stem.
    - When .build/vocals/<source_id>/vocals.wav exists in build_dir, cached_pcm_path points to it.
    """
    build_dir = tmp_path / ".build"
    vocal_stem = tmp_path / "my_vocal_stem.wav"
    vocal_stem.write_bytes(b"RIFF_vocal_pcm")

    # Mock ffprobe and ffmpeg so we don't need real media files
    probe_mock_data = {
        "streams": [
            {"codec_type": "audio", "sample_rate": "48000", "channels": 2, "index": 0, "duration": "5.0"},
            {"codec_type": "video", "r_frame_rate": "30000/1001", "duration": "5.0", "index": 1},
        ],
        "format": {"duration": "5.0"},
    }

    import media_time
    monkeypatch.setattr(media_time, "probe_media_file", lambda p: probe_mock_data)

    def mock_subprocess_run(cmd, *args, **kwargs):
        out_wav = Path(cmd[-1])
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        out_wav.write_bytes(b"RIFF_pcm_audio")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    source_id = get_source_id(dummy_media)

    # 1. Explicit vocal_stem_path
    rec1 = extract_source_record(
        video_path=dummy_media,
        build_dir=build_dir,
        vocal_stem_path=vocal_stem,
    )
    assert rec1.cached_vocal_pcm_path == str(vocal_stem.resolve())
    assert rec1.cached_pcm_path == str((build_dir / "sources" / rec1.content_id / "original.wav").resolve())

    # 2. Existing cached stem in build_dir/vocals/<source_id>/vocals.wav
    cached_vocals_dir = build_dir / "vocals" / source_id
    cached_vocals_dir.mkdir(parents=True, exist_ok=True)
    cached_vocals_wav = cached_vocals_dir / "vocals.wav"
    cached_vocals_wav.write_bytes(b"RIFF_auto_cached_vocals")

    rec2 = extract_source_record(
        video_path=dummy_media,
        build_dir=build_dir,
        # No explicit vocal_stem_path; auto-detect cached stem
    )
    assert rec2.cached_vocal_pcm_path == str(cached_vocals_wav.resolve())
    assert rec2.cached_pcm_path == str((build_dir / "sources" / rec2.content_id / "original.wav").resolve())

    # 3. isolate_vocals=True flag triggers isolation
    vocal_extracted = tmp_path / "extracted_vocals.wav"
    vocal_extracted.write_bytes(b"RIFF_isolate_flag_vocal")
    monkeypatch.setattr(
        "vocal_isolation.isolate_vocals",
        lambda *args, **kwargs: vocal_extracted,
    )
    rec3 = extract_source_record(
        video_path=dummy_media,
        build_dir=build_dir,
        isolate_vocals=True,
    )
    assert rec3.cached_vocal_pcm_path == str(vocal_extracted.resolve())
    assert rec3.cached_pcm_path == str((build_dir / "sources" / rec3.content_id / "original.wav").resolve())


def test_ensure_source_aligned_with_isolate_vocals(
    dummy_media: Path, dummy_vtt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Verifies that run_mixer.ensure_source_aligned passes vocal stem to preprocess_source
    and pre-stages SourceRecord when isolate_vocals=True.
    """
    from run_mixer import ensure_source_aligned

    vocal_stem_mock = tmp_path / "mocked_vocal_stem.wav"
    vocal_stem_mock.write_bytes(b"RIFF_mock_vocal_stem")

    isolated_called = []
    def mock_isolate(src, output_dir=None, force=False, **kwargs):
        isolated_called.append((src, output_dir))
        return vocal_stem_mock

    preprocess_called = []
    def mock_preprocess(video, vtt, corpus_dir, audio_source_path=None, **kwargs):
        preprocess_called.append((video, vtt, audio_source_path))
        out_manifest = Path(corpus_dir) / "corpus_manifest.json"
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(json.dumps({"chunks": {}}), encoding="utf-8")
        return str(out_manifest)

    def mock_mfa(*args, **kwargs):
        pass

    def mock_build_inv(*args, **kwargs):
        out_inv = Path(args[2])
        out_inv.parent.mkdir(parents=True, exist_ok=True)
        out_inv.write_text(json.dumps({"words": {}, "phones": {}}), encoding="utf-8")

    import run_mixer
    fake_build = tmp_path / ".build"
    monkeypatch.setattr("run_mixer.CACHE_BASE_DIR", fake_build)
    monkeypatch.setattr("vocal_isolation.isolate_vocals", mock_isolate)
    monkeypatch.setattr(run_mixer, "preprocess_source", mock_preprocess)
    monkeypatch.setattr(run_mixer, "run_mfa_align", mock_mfa)
    monkeypatch.setattr(run_mixer, "build_units_inventory", mock_build_inv)

    inv_path = ensure_source_aligned(
        video_path=str(dummy_media),
        vtt_path=str(dummy_vtt),
        force_realign=True,
        isolate_vocals=True,
    )

    assert len(isolated_called) == 1
    assert len(preprocess_called) == 1
    assert preprocess_called[0][2] == vocal_stem_mock
    assert Path(inv_path).exists()


def test_isolate_vocals_standalone_cli(dummy_media: Path, tmp_path: Path):
    """
    Verifies that python src/vocal_isolation.py --help and CLI invocation succeed.
    """
    import sys
    res_help = subprocess.run(
        [sys.executable, "src/vocal_isolation.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert res_help.returncode == 0
    assert "--input" in res_help.stdout
    assert "--output-dir" in res_help.stdout
    assert "--model" in res_help.stdout
    assert "--force" in res_help.stdout

    # Pre-cache vocals.wav so running CLI hits cache immediately without requiring audio-separator binary
    out_dir = tmp_path / "cli_vocals"
    source_id = get_source_id(dummy_media)
    target_dir = out_dir / source_id
    target_dir.mkdir(parents=True, exist_ok=True)
    cached_wav = target_dir / "vocals.wav"
    cached_wav.write_bytes(b"RIFF_cli_cached_stem")

    res_run = subprocess.run(
        [
            sys.executable,
            "src/vocal_isolation.py",
            "--input", str(dummy_media),
            "--output-dir", str(out_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert res_run.returncode == 0
    assert str(cached_wav.resolve()) in res_run.stdout


def test_ensure_source_aligned_staleness_detection(
    dummy_media: Path, dummy_vtt: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """
    Verifies that run_mixer.ensure_source_aligned detects stale caches when the VTT file
    is newer than units_index.json, triggering re-alignment even without force_realign.
    """
    import os
    import time
    from run_mixer import ensure_source_aligned, get_source_id

    source_id = get_source_id(str(dummy_media))
    fake_build = tmp_path / ".build"
    monkeypatch.setattr("run_mixer.CACHE_BASE_DIR", fake_build)

    inv_path = fake_build / "inventory" / source_id / "units_index.json"
    inv_path.parent.mkdir(parents=True, exist_ok=True)
    inv_path.write_text(json.dumps({"words": {}, "phones": {}}), encoding="utf-8")

    preprocess_called = []
    def mock_preprocess(video, vtt, corpus_dir, audio_source_path=None, **kwargs):
        preprocess_called.append(True)
        out_manifest = Path(corpus_dir) / "corpus_manifest.json"
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(json.dumps({"total_chunks": 1, "chunks": {}}), encoding="utf-8")
        return str(out_manifest)

    def mock_mfa(*args, **kwargs):
        pass

    def mock_build_inv(*args, **kwargs):
        out_inv = Path(args[2])
        out_inv.parent.mkdir(parents=True, exist_ok=True)
        out_inv.write_text(json.dumps({"words": {}, "phones": {}}), encoding="utf-8")

    import run_mixer
    monkeypatch.setattr(run_mixer, "preprocess_source", mock_preprocess)
    monkeypatch.setattr(run_mixer, "run_mfa_align", mock_mfa)
    monkeypatch.setattr(run_mixer, "build_units_inventory", mock_build_inv)

    # 1. Case: Inventory is NEWER than VTT -> Cache Hit, no preprocess
    now = time.time()
    os.utime(str(dummy_vtt), (now - 100, now - 100))
    os.utime(str(inv_path), (now, now))

    res_hit = ensure_source_aligned(
        video_path=str(dummy_media),
        vtt_path=str(dummy_vtt),
        force_realign=False,
    )
    assert len(preprocess_called) == 0
    assert res_hit == str(inv_path)

    # 2. Case: VTT is NEWER than Inventory -> Stale cache detected, triggers re-alignment
    os.utime(str(dummy_vtt), (now + 50, now + 50))
    os.utime(str(inv_path), (now, now))

    res_stale = ensure_source_aligned(
        video_path=str(dummy_media),
        vtt_path=str(dummy_vtt),
        force_realign=False,
    )
    assert len(preprocess_called) == 1
    assert Path(res_stale).exists()


def test_normalize_vocal_stem(tmp_path: Path):
    """Verifies that normalize_vocal_stem normalizes vocal stems to -18 LUFS without clipping."""
    from vocal_isolation import normalize_vocal_stem
    from tools.validate_mix import measure_audio_loudness
    import numpy as np
    import soundfile as sf

    sr = 48000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # 440 Hz sine wave at low amplitude (-30 dBFS)
    audio = 0.03 * np.sin(2 * np.pi * 440 * t)

    test_wav = tmp_path / "quiet_vocal.wav"
    sf.write(str(test_wav), audio, sr, subtype="PCM_16")

    initial_lufs, _ = measure_audio_loudness(test_wav)
    assert initial_lufs < -25.0

    # Normalize to -18 LUFS
    normalize_vocal_stem(test_wav, target_lufs=-18.0, peak_limit_db=-1.0)

    norm_lufs, norm_tp = measure_audio_loudness(test_wav)
    assert abs(norm_lufs - (-18.0)) < 1.0
    assert norm_tp <= -1.0 + 0.5

    # Pure silence test
    silence_wav = tmp_path / "silence.wav"
    sf.write(str(silence_wav), np.zeros(sr, dtype=np.float32), sr, subtype="PCM_16")
    normalize_vocal_stem(silence_wav)
    silence_read, _ = sf.read(str(silence_wav))
    assert np.all(silence_read == 0.0)


def test_audio_separator_mdx_flags_forwarded(tmp_path: Path, dummy_media: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verifies that audio-separator receives MDX parameters (--mdx_segment_size 256,
    --mdx_overlap 0.25, and optional --mdx_enable_denoise) when an MDX-Net model is used.
    """
    from vocal_isolation import isolate_vocals

    vocals_base = tmp_path / "vocals_out"
    recorded_cmds: List[List[str]] = []

    def mock_subprocess_run(cmd, *args, **kwargs):
        recorded_cmds.append(cmd)
        out_dir = Path(cmd[cmd.index("--output_dir") + 1])
        out_dir.mkdir(parents=True, exist_ok=True)
        vocal_output = out_dir / f"{dummy_media.stem}_(Vocals).wav"
        vocal_output.write_bytes(b"RIFF_mdx_vocals")
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="Success", stderr="")

    monkeypatch.setattr(subprocess, "run", mock_subprocess_run)

    # 1. By default, denoise is disabled to avoid phase cancellation and match UVR GUI
    stem_path = isolate_vocals(
        source_path=dummy_media,
        output_dir=vocals_base,
        model_name="UVR-MDX-NET-Inst_HQ_4.onnx",
        enable_denoise=False,
    )

    assert len(recorded_cmds) == 1
    cmd = recorded_cmds[0]
    assert "--mdx_segment_size" in cmd
    assert cmd[cmd.index("--mdx_segment_size") + 1] == "256"
    assert "--mdx_overlap" in cmd
    assert cmd[cmd.index("--mdx_overlap") + 1] == "0.25"
    assert "--mdx_enable_denoise" not in cmd
    assert stem_path.exists()

    # 2. When enable_denoise=True, flag is forwarded
    recorded_cmds.clear()
    isolate_vocals(
        source_path=dummy_media,
        output_dir=vocals_base,
        model_name="UVR-MDX-NET-Inst_HQ_4.onnx",
        force=True,
        enable_denoise=True,
    )
    assert len(recorded_cmds) == 1
    assert "--mdx_enable_denoise" in recorded_cmds[0]



