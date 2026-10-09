"""Tests for word release tails and punctuation pause styles (freeze / continue / off)."""

import numpy as np
import pytest
import soundfile as sf

from contracts import AlignedEvent, InventoryIndex, SourceRecord, SpanCandidate
from edit_plan import CONTINUE_SPAN_TYPE, build_edit_plan, release_tail_samples
from lexicon import Lexicon
from target_graph import TargetGraphBuilder

SR = 48000


def _inventory(analysis_path=None, gap_after_second_word=1.0):
    """'hello' (0.0-0.5 s) then aligned silence, 'there' at 2.0-2.5 s, more speech after."""
    src = SourceRecord(content_id="src_1", source_path="video.mp4", native_sample_rate=SR, native_channels=2,
                       audio_duration_sec=10.0, cached_analysis_pcm_path=str(analysis_path or ""))
    ev = {
        "e1": AlignedEvent(event_id="e1", source_content_id="src_1", utterance_id="u1", speaker_id="spk", kind="phone",
                           start_sample=0, end_sample=int(0.5 * SR), label="HH", base_phone="HH", successor_id="s1"),
        "s1": AlignedEvent(event_id="s1", source_content_id="src_1", utterance_id="u1", speaker_id="spk", kind="silence",
                           start_sample=int(0.5 * SR), end_sample=int(2.0 * SR), label="sil", base_phone="sil"),
        "e2": AlignedEvent(event_id="e2", source_content_id="src_1", utterance_id="u2", speaker_id="spk", kind="phone",
                           start_sample=int(2.0 * SR), end_sample=int(2.5 * SR), label="DH", base_phone="DH", successor_id="s2"),
        "s2": AlignedEvent(event_id="s2", source_content_id="src_1", utterance_id="u2", speaker_id="spk", kind="silence",
                           start_sample=int(2.5 * SR), end_sample=int((2.5 + gap_after_second_word) * SR), label="sil", base_phone="sil"),
        "e3": AlignedEvent(event_id="e3", source_content_id="src_1", utterance_id="u3", speaker_id="other", kind="phone",
                           start_sample=int((2.5 + gap_after_second_word) * SR), end_sample=int((3.0 + gap_after_second_word) * SR),
                           label="AA1", base_phone="AA"),
    }
    return InventoryIndex(sources={"src_1": src}, events=ev)


def _graph(text):
    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH"]]
    lex.word_to_pronunciations["there"] = [["DH"]]
    return TargetGraphBuilder(lex).build(text)


def _spans():
    c1 = SpanCandidate(candidate_id="c1", target_node_start=0, target_node_end=0, covered_nodes=[0], source_content_id="src_1",
                       speaker_id="spk", start_sample=0, end_sample=int(0.5 * SR), event_ids=["e1"], span_type="word", matched_word="hello")
    c2 = SpanCandidate(candidate_id="c2", target_node_start=1, target_node_end=1, covered_nodes=[1], source_content_id="src_1",
                       speaker_id="spk", start_sample=int(2.0 * SR), end_sample=int(2.5 * SR), event_ids=["e2"], span_type="word",
                       matched_word="there")
    return [c1, c2]


def test_release_tail_extends_into_silence_and_stops_before_speech():
    inv = _inventory(gap_after_second_word=0.05)
    c1, c2 = _spans()
    assert release_tail_samples(c1, inv, 0.100) == int(0.100 * SR)
    # Only 50 ms of silence before other speech: tail stops 30 ms short of it
    assert release_tail_samples(c2, inv, 0.100) == pytest.approx(0.020 * SR, abs=2)
    assert release_tail_samples(c1, inv, 0.0) == 0


def test_release_tail_lengthens_segment_and_fades_its_end():
    inv = _inventory()
    plan = build_edit_plan(_spans(), _graph("hello there"), inv, project_sample_rate=SR, release_tail_sec=0.1)
    words = [s for s in plan.segments if not s.is_silence]
    assert words[0].source_end_sample == int(0.6 * SR)
    assert words[0].fade_out_samples == int(0.010 * SR)
    plan.validate()


def test_pause_style_freeze_and_off():
    inv = _inventory()
    freeze = build_edit_plan(_spans(), _graph("Hello, there."), inv, project_sample_rate=SR, pause_style="freeze")
    assert [s.span_type for s in freeze.segments] == ["word", "pause", "word", "pause"]
    assert all(s.freeze_video for s in freeze.segments if s.span_type == "pause")
    # Freeze frame is taken right after the word, not at its start
    assert freeze.segments[1].video_source_pts_sec == pytest.approx(0.5, abs=1e-3)

    off = build_edit_plan(_spans(), _graph("Hello, there."), inv, project_sample_rate=SR, pause_style="off")
    assert [s.span_type for s in off.segments] == ["word", "word"]
    # Explicit gap tags are still honored with pauses off
    gap = build_edit_plan(_spans(), _graph("Hello <gap:300ms> there"), inv, project_sample_rate=SR, pause_style="off")
    assert [s.span_type for s in gap.segments] == ["word", "pause", "word"]


def test_pause_style_continue_plays_on_until_speech(tmp_path):
    # Analysis track: speech-level tone during words, quiet room tone between them
    t = np.arange(10 * 16000) / 16000
    y = 0.002 * np.sin(2 * np.pi * 90 * t)
    for a, b in ((0.0, 0.5), (2.0, 2.5), (2.75, 3.25)):
        y[int(a * 16000):int(b * 16000)] = 0.3 * np.sin(2 * np.pi * 200 * t[int(a * 16000):int(b * 16000)])
    path = tmp_path / "analysis.wav"
    sf.write(str(path), y.astype(np.float32), 16000)
    inv = _inventory(analysis_path=path, gap_after_second_word=0.25)

    plan = build_edit_plan(_spans(), _graph("Hello, there."), inv, project_sample_rate=SR, pause_style="continue")
    plan.validate()
    types = [s.span_type for s in plan.segments]
    # Comma pause (0.25 s) after 'hello': 1.5 s of quiet follows, so the clip plays on fully
    assert types[:3] == ["word", CONTINUE_SPAN_TYPE, "word"]
    cont = plan.segments[1]
    assert not cont.is_silence and not cont.freeze_video
    assert cont.source_start_sample == plan.segments[0].source_end_sample
    assert cont.duration_samples == int(0.25 * SR)
    # Final period (0.45 s): speech resumes 0.25 s after 'there', so play on ~0.21 s then hold
    assert types[3:] == [CONTINUE_SPAN_TYPE, "pause"]
    tail_cont, hold = plan.segments[3], plan.segments[4]
    assert 0.15 * SR < tail_cont.duration_samples < 0.25 * SR
    assert hold.freeze_video and hold.video_source_pts_sec == pytest.approx(tail_cont.source_end_sample / SR, abs=1e-3)
    assert tail_cont.duration_samples + hold.duration_samples == int(0.45 * SR)
    # Continued pauses take the gain of the word they continue
    assert cont.gain_db == plan.segments[0].gain_db
    assert plan.transitions[0].transition_type == "continuation"
