"""
Milestone 4 — Rich Interactive Speaker Review UX Tests

Tests for:
- Audio preview WAV generation for unrecognized speaker clusters
- Subtitle dialogue snippet extraction for cluster turns
- Ranked candidate presentation in prompt_or_register_cluster
- Numeric index selection from candidate list
- Full diarize() integration with subtitle_path and cluster_preview_dir
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from speaker_registry import SpeakerRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry(tmp_path: Path, threshold: float = 0.75) -> SpeakerRegistry:
    return SpeakerRegistry(config_path=tmp_path / "reg.json", similarity_threshold=threshold)


def _sine_wav(
    tmp_path: Path,
    name: str = "audio.wav",
    sr: int = 16000,
    duration: float = 3.0,
    freq: float = 440.0,
) -> Path:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    p = tmp_path / name
    sf.write(str(p), audio, sr)
    return p


def _fake_vtt(tmp_path: Path, cues: List[Tuple[str, str, str]]) -> Path:
    """Build a minimal VTT file from (start, end, text) tuples like ('00:00:01.000', '00:00:03.000', 'Hello')."""
    lines = ["WEBVTT\n\n"]
    for start, end, text in cues:
        lines.append(f"{start} --> {end}\n{text}\n\n")
    p = tmp_path / "subs.vtt"
    p.write_text("".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# SpeakerRegistry.rank_candidates
# ---------------------------------------------------------------------------

class TestRankCandidates:
    def test_empty_registry_returns_empty(self, tmp_path):
        reg = _make_registry(tmp_path)
        result = reg.rank_candidates(np.array([1.0, 0.0]))
        assert result == []

    def test_single_speaker_ranked(self, tmp_path):
        reg = _make_registry(tmp_path)
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        reg.register_or_update("obi-wan", centroid=v, sample_count=5)
        # Query identical vector → similarity ≈ 1.0
        result = reg.rank_candidates(v)
        assert len(result) == 1
        assert result[0][0] == "obi-wan"
        assert result[0][1] == pytest.approx(1.0, abs=1e-5)

    def test_ordering_by_similarity(self, tmp_path):
        reg = _make_registry(tmp_path)
        v_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        reg.register_or_update("anakin", centroid=v_a, sample_count=3)
        reg.register_or_update("obi-wan", centroid=v_b, sample_count=3)
        # Query close to anakin
        query = np.array([0.9, 0.1, 0.0], dtype=np.float32)
        result = reg.rank_candidates(query)
        assert result[0][0] == "anakin"
        assert result[0][1] > result[1][1]

    def test_unnormalized_query_is_handled(self, tmp_path):
        reg = _make_registry(tmp_path)
        v = np.array([3.0, 4.0], dtype=np.float32)
        reg.register_or_update("windu", centroid=np.array([0.6, 0.8], dtype=np.float32), sample_count=1)
        # Un-normalized: should still produce similarity ~1.0 after normalization
        result = reg.rank_candidates(v)
        assert result[0][1] == pytest.approx(1.0, abs=1e-4)

    def test_mismatched_dimension_profiles_skipped(self, tmp_path):
        reg = _make_registry(tmp_path)
        # Register 3-D profile but query with 2-D vector
        reg.register_or_update("ghost", centroid=np.array([1.0, 0.0, 0.0], dtype=np.float32), sample_count=1)
        result = reg.rank_candidates(np.array([1.0, 0.0], dtype=np.float32))
        assert result == []


# ---------------------------------------------------------------------------
# prompt_or_register_cluster — audio preview display & numeric selection
# ---------------------------------------------------------------------------

class TestPromptOrRegisterCluster:
    def _cluster_vec(self) -> np.ndarray:
        return np.array([0.5, 0.5, 0.0], dtype=np.float32)

    def test_no_match_noninteractive_returns_anon(self, tmp_path):
        reg = _make_registry(tmp_path)
        result = reg.prompt_or_register_cluster(
            cluster_id="anon:abc:0",
            centroid=self._cluster_vec(),
            turn_count=3,
            duration_sec=9.0,
            interactive=False,
        )
        assert result.startswith("anon:")

    def test_matched_profile_returned_without_prompt(self, tmp_path):
        reg = _make_registry(tmp_path, threshold=0.5)
        v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        reg.register_or_update("yoda", centroid=v, sample_count=2)
        result = reg.prompt_or_register_cluster(
            cluster_id="anon:xyz:1",
            centroid=v,
            turn_count=1,
            duration_sec=2.0,
            interactive=True,
        )
        assert result == "yoda"

    def test_interactive_numeric_selection_picks_ranked_candidate(self, tmp_path):
        reg = _make_registry(tmp_path, threshold=2.0)  # Threshold > 1.0 → impossible to auto-match
        v_a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v_b = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        reg.register_or_update("anakin", centroid=v_a, sample_count=3)
        reg.register_or_update("obi-wan", centroid=v_b, sample_count=3)
        # Cluster centroid close to anakin
        cluster_v = np.array([0.95, 0.05, 0.0], dtype=np.float32)
        # User types "1" — selects top-ranked candidate (anakin)
        with patch("builtins.input", return_value="1"):
            result = reg.prompt_or_register_cluster(
                cluster_id="anon:zzz:0",
                centroid=cluster_v,
                turn_count=2,
                duration_sec=5.0,
                interactive=True,
            )
        assert result == "anakin"

    def test_interactive_text_entry_registers_new_speaker(self, tmp_path):
        reg = _make_registry(tmp_path)
        cluster_v = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        with patch("builtins.input", return_value="mace windu"):
            result = reg.prompt_or_register_cluster(
                cluster_id="anon:qqq:0",
                centroid=cluster_v,
                turn_count=4,
                duration_sec=12.0,
                interactive=True,
            )
        assert result == "mace_windu"
        assert "mace_windu" in reg.speakers

    def test_interactive_enter_leaves_anonymous(self, tmp_path):
        reg = _make_registry(tmp_path)
        cluster_v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        with patch("builtins.input", return_value=""):
            result = reg.prompt_or_register_cluster(
                cluster_id="anon:ttt:0",
                centroid=cluster_v,
                turn_count=1,
                duration_sec=3.0,
                interactive=True,
            )
        assert result.startswith("anon:")

    def test_audio_preview_path_displayed_in_output(self, tmp_path, capsys):
        reg = _make_registry(tmp_path)
        preview_wav = tmp_path / "preview.wav"
        sf.write(str(preview_wav), np.zeros(100, dtype=np.float32), 16000)
        cluster_v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        with patch("builtins.input", return_value=""):
            reg.prompt_or_register_cluster(
                cluster_id="anon:ppp:0",
                centroid=cluster_v,
                turn_count=1,
                duration_sec=2.0,
                interactive=True,
                audio_preview_path=preview_wav,
            )
        captured = capsys.readouterr()
        assert "Audio Preview" in captured.out
        assert "preview.wav" in captured.out

    def test_dialogue_snippet_displayed_in_output(self, tmp_path, capsys):
        reg = _make_registry(tmp_path)
        cluster_v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        with patch("builtins.input", return_value=""):
            reg.prompt_or_register_cluster(
                cluster_id="anon:sss:0",
                centroid=cluster_v,
                turn_count=1,
                duration_sec=2.0,
                interactive=True,
                dialogue_snippet="You were the chosen one!",
            )
        captured = capsys.readouterr()
        assert "You were the chosen one!" in captured.out

    def test_all_speech_clips_displayed_in_output(self, tmp_path, capsys):
        reg = _make_registry(tmp_path)
        cluster_v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        clips = [
            {"index": 1, "start_sec": 6.3, "end_sec": 7.0, "duration_sec": 0.7, "dialogue": "them unfortunately"},
            {"index": 2, "start_sec": 7.4, "end_sec": 8.4, "duration_sec": 1.0, "dialogue": "the debate is not"},
            {"index": 3, "start_sec": 9.2, "end_sec": 9.6, "duration_sec": 0.4, "dialogue": "over"},
        ]
        with patch("builtins.input", return_value=""):
            reg.prompt_or_register_cluster(
                cluster_id="anon:multi:0",
                centroid=cluster_v,
                turn_count=3,
                duration_sec=2.1,
                interactive=True,
                dialogue_snippet="them unfortunately the debate is not over",
                clips=clips,
            )
        captured = capsys.readouterr()
        assert "Speech clips in this video (3):" in captured.out
        assert "Clip 1" in captured.out
        assert "Clip 2" in captured.out
        assert "Clip 3" in captured.out
        assert "[00:06.3 - 00:07.0]" in captured.out
        assert "[00:07.4 - 00:08.4]" in captured.out
        assert "[00:09.2 - 00:09.6]" in captured.out
        assert "them unfortunately" in captured.out
        assert "the debate is not" in captured.out
        assert "over" in captured.out

    def test_ranked_candidates_shown_in_output(self, tmp_path, capsys):
        reg = _make_registry(tmp_path, threshold=2.0)  # Threshold > 1.0 → impossible to auto-match
        reg.register_or_update("anakin", centroid=np.array([1.0, 0.0], dtype=np.float32), sample_count=2)
        reg.register_or_update("obi-wan", centroid=np.array([0.0, 1.0], dtype=np.float32), sample_count=2)
        cluster_v = np.array([0.9, 0.1], dtype=np.float32)
        with patch("builtins.input", return_value=""):
            reg.prompt_or_register_cluster(
                cluster_id="anon:ccc:0",
                centroid=cluster_v,
                turn_count=2,
                duration_sec=5.0,
                interactive=True,
            )
        captured = capsys.readouterr()
        assert "[1]" in captured.out
        assert "anakin" in captured.out
        assert "[2]" in captured.out
        assert "obi-wan" in captured.out

    def test_out_of_range_numeric_treated_as_new_name(self, tmp_path):
        reg = _make_registry(tmp_path, threshold=2.0)  # Threshold > 1.0 → impossible to auto-match
        reg.register_or_update("anakin", centroid=np.array([1.0, 0.0], dtype=np.float32), sample_count=2)
        cluster_v = np.array([0.9, 0.1], dtype=np.float32)
        # "99" is out of range → treated as new speaker name "99"
        with patch("builtins.input", return_value="99"):
            result = reg.prompt_or_register_cluster(
                cluster_id="anon:oob:0",
                centroid=cluster_v,
                turn_count=1,
                duration_sec=2.0,
                interactive=True,
            )
        # Out-of-range digit should be kept as a literal name
        assert result == "99"


# ---------------------------------------------------------------------------
# _write_cluster_preview — WAV generation inside diarize()
# ---------------------------------------------------------------------------

class TestClusterPreviewWavGeneration:
    """Integration test: diarize() with subtitle_path and cluster_preview_dir generates preview WAV."""

    def test_preview_wav_written_for_unrecognized_cluster(self, tmp_path):
        """When diarize() encounters an unrecognized single-turn cluster in non-interactive mode,
        it should still write a cluster preview WAV to cluster_preview_dir."""
        from diarization import SpeakerDiarizer

        sr = 16000
        duration = 4.0
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        audio_path = tmp_path / "audio.wav"
        sf.write(str(audio_path), audio, sr)

        preview_dir = tmp_path / "clusters"
        registry = SpeakerRegistry(config_path=tmp_path / "reg.json")
        diarizer = SpeakerDiarizer(registry=registry)

        # Inject a stub embedding backend so we don't need real WavLM
        diarizer.extract_embedding = lambda audio_chunk, sample_rate=16000: np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)  # type: ignore[method-assign]
        # Inject single speech turn spanning the whole file
        candidate_turns = [(0.0, duration)]

        diarizer.diarize(
            audio_path,
            candidate_turns=candidate_turns,
            interactive=False,
            content_id="testcid",
            cluster_preview_dir=preview_dir,
        )

        # Preview WAV should be written
        preview_files = list(preview_dir.glob("*_preview.wav"))
        assert len(preview_files) == 1, f"Expected 1 preview WAV, got: {preview_files}"
        wav_data, wav_sr = sf.read(str(preview_files[0]))
        assert wav_sr == sr
        assert len(wav_data) > 0

    def test_preview_wav_duration_capped_at_target(self, tmp_path):
        """Preview WAV should not exceed ~target_duration seconds even for long turns."""
        from diarization import SpeakerDiarizer

        sr = 16000
        duration = 10.0
        audio = np.random.default_rng(42).uniform(-0.3, 0.3, int(sr * duration)).astype(np.float32)
        audio_path = tmp_path / "audio.wav"
        sf.write(str(audio_path), audio, sr)

        preview_dir = tmp_path / "clusters"
        registry = SpeakerRegistry(config_path=tmp_path / "reg.json")
        diarizer = SpeakerDiarizer(registry=registry)
        diarizer.extract_embedding = lambda a, sample_rate=16000: np.array([0.5, 0.5], dtype=np.float32)  # type: ignore[method-assign]

        diarizer.diarize(
            audio_path,
            candidate_turns=[(0.0, duration)],
            interactive=False,
            content_id="longcid",
            cluster_preview_dir=preview_dir,
        )

        preview_files = list(preview_dir.glob("*_preview.wav"))
        assert preview_files, "No preview WAV generated"
        wav_data, wav_sr = sf.read(str(preview_files[0]))
        actual_duration = len(wav_data) / wav_sr
        assert actual_duration <= 3.0, f"Preview too long: {actual_duration:.2f}s"


# ---------------------------------------------------------------------------
# Dialogue snippet extraction
# ---------------------------------------------------------------------------

class TestDialogueSnippetExtraction:
    """Test that subtitle cues overlapping a cluster's turns are extracted as dialogue snippets."""

    def test_snippet_extracted_for_overlapping_cue(self, tmp_path, capsys):
        """diarize() with a subtitle file should show dialogue snippet during interactive UX."""
        from diarization import SpeakerDiarizer

        sr = 16000
        duration = 3.0
        audio = (0.3 * np.sin(2 * np.pi * 220 * np.linspace(0, duration, int(sr * duration)))).astype(np.float32)
        audio_path = tmp_path / "audio.wav"
        sf.write(str(audio_path), audio, sr)

        # VTT covering the first 3 seconds
        vtt_path = _fake_vtt(tmp_path, [("00:00:00.500", "00:00:02.500", "I have the high ground")])

        registry = SpeakerRegistry(config_path=tmp_path / "reg.json")
        diarizer = SpeakerDiarizer(registry=registry)
        diarizer.extract_embedding = lambda a, sample_rate=16000: np.array([0.2, 0.8], dtype=np.float32)  # type: ignore[method-assign]

        with patch("builtins.input", return_value=""):
            diarizer.diarize(
                audio_path,
                candidate_turns=[(0.0, duration)],
                interactive=True,
                content_id="snip_test",
                subtitle_path=vtt_path,
                cluster_preview_dir=tmp_path / "clusters",
            )

        out = capsys.readouterr().out
        assert "I have the high ground" in out, f"Expected dialogue snippet in output, got:\n{out}"

    def test_no_subtitle_path_no_snippet(self, tmp_path, capsys):
        """When subtitle_path is None, no dialogue snippet should appear."""
        from diarization import SpeakerDiarizer

        sr = 16000
        audio = np.zeros(int(sr * 2.0), dtype=np.float32)
        audio_path = tmp_path / "audio.wav"
        sf.write(str(audio_path), audio, sr)

        registry = SpeakerRegistry(config_path=tmp_path / "reg.json")
        diarizer = SpeakerDiarizer(registry=registry)
        diarizer.extract_embedding = lambda a, sample_rate=16000: np.array([0.1, 0.9], dtype=np.float32)  # type: ignore[method-assign]

        with patch("builtins.input", return_value=""):
            diarizer.diarize(
                audio_path,
                candidate_turns=[(0.0, 2.0)],
                interactive=True,
                content_id="nosnip",
                subtitle_path=None,
                cluster_preview_dir=tmp_path / "clusters",
            )

        out = capsys.readouterr().out
        assert "Dialogue" not in out

    def test_dialogue_does_not_bleed_across_adjacent_non_overlapping_turns(self, tmp_path):
        """Adjacent speech turns separated by a short gap must not bleed into each other's dialogue."""
        from diarization import SpeakerDiarizer

        sr = 16000
        duration = 5.0
        audio = (0.2 * np.sin(2 * np.pi * 300 * np.linspace(0, duration, int(sr * duration)))).astype(np.float32)
        audio_path = tmp_path / "audio.wav"
        sf.write(str(audio_path), audio, sr)

        vtt_path = _fake_vtt(tmp_path, [
            ("00:00:00.000", "00:00:01.900", "Count Dooku must have made a treaty with"),
            ("00:00:02.100", "00:00:04.500", "them unfortunately the debate is not"),
        ])

        registry = SpeakerRegistry(config_path=tmp_path / "reg.json")
        diarizer = SpeakerDiarizer(registry=registry)
        diarizer.extract_embedding = lambda a, sample_rate=16000: np.array([0.5, 0.5], dtype=np.float32)  # type: ignore[method-assign]

        # Use non-interactive diarize to inspect the assigned intervals
        intervals = diarizer.diarize(
            audio_path,
            candidate_turns=[(0.0, 1.9), (2.1, 4.5)],
            interactive=False,
            content_id="bleed_test",
            subtitle_path=vtt_path,
        )
        assert len(intervals) == 2

    def test_auto_derives_candidate_turns_from_subtitles_when_none(self, tmp_path):
        """When candidate_turns is None, diarize() automatically uses cues from subtitle_path."""
        from diarization import SpeakerDiarizer

        sr = 16000
        duration = 4.0
        audio = (0.3 * np.sin(2 * np.pi * 350 * np.linspace(0, duration, int(sr * duration)))).astype(np.float32)
        audio_path = tmp_path / "audio.wav"
        sf.write(str(audio_path), audio, sr)

        vtt_path = _fake_vtt(tmp_path, [
            ("00:00:00.500", "00:00:01.500", "First cue"),
            ("00:00:02.000", "00:00:03.500", "Second cue"),
        ])

        registry = SpeakerRegistry(config_path=tmp_path / "reg.json")
        diarizer = SpeakerDiarizer(registry=registry)
        diarizer.extract_embedding = lambda a, sample_rate=16000: np.array([0.5, 0.5], dtype=np.float32)  # type: ignore[method-assign]

        intervals = diarizer.diarize(
            audio_path,
            candidate_turns=None,
            interactive=False,
            content_id="auto_turns_test",
            subtitle_path=vtt_path,
        )
        assert len(intervals) == 2
        assert np.isclose(intervals[0].start_sec, 0.5, atol=0.05)
        assert np.isclose(intervals[0].end_sec, 1.5, atol=0.05)
        assert np.isclose(intervals[1].start_sec, 2.0, atol=0.05)
        assert np.isclose(intervals[1].end_sec, 3.5, atol=0.05)

    def test_adaptive_speech_detection_for_quiet_vocal_stems(self):
        """detect_speech_turns detects speech with RMS ~ 0.006 when background noise is ~ 0.0002."""
        from diarization import SpeakerDiarizer

        sr = 16000
        # 1s quiet speech (amplitude ~ 0.008, RMS ~ 0.0057), 1s silence, 1s quiet speech
        t1 = np.linspace(0, 1.0, sr, endpoint=False)
        tone1 = (0.008 * np.sin(2 * np.pi * 440 * t1)).astype(np.float32)
        silence = np.zeros(sr, dtype=np.float32)
        t2 = np.linspace(0, 1.0, sr, endpoint=False)
        tone2 = (0.008 * np.sin(2 * np.pi * 330 * t2)).astype(np.float32)

        audio = np.concatenate([tone1, silence, tone2])
        diarizer = SpeakerDiarizer(min_turn_duration=0.25)
        turns = diarizer.detect_speech_turns(audio, sr, min_silence_sec=0.3)

        assert len(turns) == 2, f"Expected 2 turns detected with adaptive threshold, got: {turns}"
        assert np.isclose(turns[0][0], 0.0, atol=0.1)
        assert np.isclose(turns[0][1], 1.0, atol=0.15)
        assert np.isclose(turns[1][0], 2.0, atol=0.15)
        assert np.isclose(turns[1][1], 3.0, atol=0.15)

