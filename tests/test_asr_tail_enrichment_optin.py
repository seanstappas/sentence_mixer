"""Whisper tail enrichment is opt-in: off by default, enabled with --asr-fill-tails."""

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import run_mixer
import vtt_preprocessor
from cache import compute_alignment_cache_key


def _make_long_cue_source(tmp_path: Path):
    # A 3 s chunk with one caption word triggers the duration/word-count heuristic.
    media = tmp_path / "clip.wav"
    sf.write(media, np.zeros(16000 * 4, dtype=np.float32), 16000)
    vtt = tmp_path / "clip.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.500 --> 00:00:03.500\nhello\n", encoding="utf-8")
    return media, vtt


@pytest.mark.parametrize("enabled", [False, True])
def test_preprocess_enrichment_only_when_enabled(tmp_path, monkeypatch, enabled):
    media, vtt = _make_long_cue_source(tmp_path)
    calls = []

    def fake_enrich(wav_path, current_text):
        calls.append(current_text)
        return current_text + " extra"

    monkeypatch.setattr(vtt_preprocessor, "enrich_chunk_transcript_with_asr", fake_enrich)

    out_dir = tmp_path / "corpus"
    manifest_path = vtt_preprocessor.preprocess_source(
        str(media), str(vtt), str(out_dir), asr_tail_enrichment=enabled
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    chunk_text = next(iter(manifest["chunks"].values()))["text"]

    assert manifest["asr_tail_enrichment"] is enabled
    if enabled:
        assert calls == ["hello"]
        assert chunk_text == "hello extra"
    else:
        assert calls == []
        assert chunk_text == "hello"


def test_preprocess_default_does_not_enrich(tmp_path, monkeypatch):
    media, vtt = _make_long_cue_source(tmp_path)

    def must_not_run(*args, **kwargs):
        raise AssertionError("Whisper enrichment must be opt-in")

    monkeypatch.setattr(vtt_preprocessor, "enrich_chunk_transcript_with_asr", must_not_run)
    manifest_path = vtt_preprocessor.preprocess_source(str(media), str(vtt), str(tmp_path / "corpus"))
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["asr_tail_enrichment"] is False


@pytest.mark.parametrize("command", [["mix", "--text", "hello"], ["ingest"], ["auto", "--video", "v.mp4", "--text", "hello"]])
def test_cli_flag_defaults_off(command):
    parser = run_mixer.build_cli_parser()
    assert parser.parse_args(command).asr_fill_tails is False
    assert parser.parse_args(command + ["--asr-fill-tails"]).asr_fill_tails is True


def test_preprocess_subcommand_flag_defaults_off():
    parser = run_mixer.build_cli_parser()
    base = ["preprocess", "--video", "v.mp4", "--vtt", "v.vtt"]
    assert parser.parse_args(base).asr_fill_tails is False
    assert parser.parse_args(base + ["--asr-fill-tails"]).asr_fill_tails is True


def test_enrichment_setting_changes_alignment_key():
    common = dict(
        media_hash="m",
        selected_audio_stream=0,
        selected_video_stream=0,
        transcript_hash="t",
        dict_hash="d",
    )
    off = compute_alignment_cache_key(**common, mfa_options={"asr_tail_enrichment": False})
    on = compute_alignment_cache_key(**common, mfa_options={"asr_tail_enrichment": True})
    assert off != on


@pytest.mark.parametrize("enabled", [False, True])
def test_ensure_source_aligned_passes_flag_and_records_it(tmp_path, monkeypatch, enabled):
    media = tmp_path / "clip.mp4"
    media.write_bytes(b"fake media bytes")
    vtt = tmp_path / "clip.vtt"
    vtt.write_text("WEBVTT\n\n00:00:00.500 --> 00:00:01.500\nhello\n", encoding="utf-8")

    monkeypatch.setattr(run_mixer, "CACHE_BASE_DIR", tmp_path / "cache")
    seen = {}

    def fake_preprocess(v_abs, vtt_path, out_dir, **kwargs):
        seen["asr_tail_enrichment"] = kwargs.get("asr_tail_enrichment")
        p = Path(out_dir)
        p.mkdir(parents=True, exist_ok=True)
        (p / "corpus_manifest.json").write_text(json.dumps({"chunks": {}}), encoding="utf-8")

    monkeypatch.setattr(run_mixer, "preprocess_source", fake_preprocess)
    monkeypatch.setattr(run_mixer, "run_mfa_align", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_mixer, "build_units_inventory", lambda *args, **kwargs: None)

    run_mixer.ensure_source_aligned(str(media), str(vtt), asr_fill_tails=enabled)

    assert seen["asr_tail_enrichment"] is enabled
    meta_files = list((tmp_path / "cache" / "alignment").glob("*/alignment_metadata.json"))
    assert len(meta_files) == 1
    meta = json.loads(meta_files[0].read_text(encoding="utf-8"))
    assert meta["mfa_options"] == {"asr_tail_enrichment": enabled}
