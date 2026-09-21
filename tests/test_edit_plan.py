"""
Tests for canonical EditPlan construction and timing invariants.
"""

from contracts import AlignedEvent, InventoryIndex, SourceRecord, SpanCandidate
from edit_plan import build_edit_plan
from lexicon import Lexicon
from target_graph import TargetGraphBuilder


def test_build_edit_plan_timing_and_pauses():
    sr = 48000
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )

    ev1 = AlignedEvent(
        event_id="ev1", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=24000, label="HH", base_phone="HH"
    )
    ev2 = AlignedEvent(
        event_id="ev2", source_content_id="src_1", utterance_id="u2", speaker_id="spk1",
        kind="phone", start_sample=48000, end_sample=72000, label="DH", base_phone="DH"
    )

    inv = InventoryIndex(
        sources={"src_1": src},
        events={"ev1": ev1, "ev2": ev2},
    )

    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH"]]
    lex.word_to_pronunciations["there"] = [["DH"]]

    builder = TargetGraphBuilder(lex)
    # Target text with comma pause: 'Hello, there'
    graph = builder.build("Hello, there")

    cand1 = SpanCandidate(
        candidate_id="c1",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="src_1",
        speaker_id="spk1",
        start_sample=0,
        end_sample=24000,
        event_ids=["ev1"],
        span_type="word",
        matched_word="hello",
    )

    cand2 = SpanCandidate(
        candidate_id="c2",
        target_node_start=1,
        target_node_end=1,
        covered_nodes=[1],
        source_content_id="src_1",
        speaker_id="spk1",
        start_sample=48000,
        end_sample=72000,
        event_ids=["ev2"],
        span_type="word",
        matched_word="there",
    )

    plan = build_edit_plan([cand1, cand2], graph, inv, project_sample_rate=sr)
    plan.validate()

    # Verify pause insertion: comma creates pause segment between word 0 and word 1
    pause_segs = [s for s in plan.segments if s.is_silence]
    assert len(pause_segs) == 1, "Punctuation comma must schedule an explicit silence pause segment"
    assert pause_segs[0].target_text == "[pause]"

    # Invariant: sample count equation
    seg_samples = sum(s.duration_samples for s in plan.segments)
    overlap_samples = sum(t.overlap_samples for t in plan.transitions if t.transition_type == "overlap")
    assert plan.total_output_samples == seg_samples - overlap_samples
