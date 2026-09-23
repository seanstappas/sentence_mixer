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


def test_t01_six_50ms_cuts_cumulative_frames():
    """T01: 6 x 50-ms cuts, 48 kHz, 30 fps, cuts-only -> 14,400 master samples; 9 actual video frames."""
    from contracts import RationalTime, TargetGraph, TargetPhoneNode

    sr = 48000
    fps = RationalTime(30, 1)
    cut_samples = int(round(0.050 * sr))  # 2400
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )
    cands = []
    events = {}
    nodes = []
    for i in range(6):
        ev_id = f"ev_{i}"
        events[ev_id] = AlignedEvent(
            event_id=ev_id, source_content_id="src_1", utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=i * cut_samples, end_sample=(i + 1) * cut_samples,
            label=f"P_{i}", base_phone="AA"
        )
        nodes.append(TargetPhoneNode(
            node_id=i, token_index=i, word_text=f"w{i}", phone=f"P_{i}", base_phone="AA",
            stress=1, is_word_initial=True, is_word_final=True, is_phrase_final=(i == 5)
        ))
        cands.append(SpanCandidate(
            candidate_id=f"c_{i}",
            target_node_start=i,
            target_node_end=i,
            covered_nodes=[i],
            source_content_id="src_1",
            speaker_id="spk1",
            start_sample=i * cut_samples,
            end_sample=(i + 1) * cut_samples,
            event_ids=[ev_id],
            span_type="word",
            matched_word=f"w{i}",
        ))

    inv = InventoryIndex(sources={"src_1": src}, events=events)
    graph = TargetGraph(
        raw_text="w0 w1 w2 w3 w4 w5",
        normalized_words=[f"w{i}" for i in range(6)],
        nodes=nodes,
        edges=[(i, i + 1) for i in range(5)],
        start_nodes=[0],
        end_nodes=[5],
    )
    plan = build_edit_plan(cands, graph, inv, project_sample_rate=sr, video_fps=fps, rendering_mode="cuts-only")
    plan.validate()
    assert plan.total_output_samples == 14400
    assert plan.total_video_frames == 9
    assert sum(s.video_frame_count for s in plan.segments) == 9


def test_t02_1000_cuts_cumulative_frames():
    """T02: 1,000 x 50-ms cuts -> 2,400,000 samples; 1,500 actual frames; bounded switch error."""
    from contracts import RationalTime, TargetGraph, TargetPhoneNode

    sr = 48000
    fps = RationalTime(30, 1)
    cut_samples = int(round(0.050 * sr))  # 2400
    src = SourceRecord(
        content_id="src_1",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=2,
    )
    N = 1000
    cands = []
    events = {}
    nodes = []
    for i in range(N):
        ev_id = f"ev_{i}"
        events[ev_id] = AlignedEvent(
            event_id=ev_id, source_content_id="src_1", utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=i * cut_samples, end_sample=(i + 1) * cut_samples,
            label="AA", base_phone="AA"
        )
        nodes.append(TargetPhoneNode(
            node_id=i, token_index=i, word_text=f"w{i}", phone="AA", base_phone="AA",
            stress=1, is_word_initial=True, is_word_final=True, is_phrase_final=(i == N - 1)
        ))
        cands.append(SpanCandidate(
            candidate_id=f"c_{i}",
            target_node_start=i,
            target_node_end=i,
            covered_nodes=[i],
            source_content_id="src_1",
            speaker_id="spk1",
            start_sample=i * cut_samples,
            end_sample=(i + 1) * cut_samples,
            event_ids=[ev_id],
            span_type="word",
            matched_word=f"w{i}",
        ))

    inv = InventoryIndex(sources={"src_1": src}, events=events)
    graph = TargetGraph(
        raw_text=" ".join(f"w{i}" for i in range(N)),
        normalized_words=[f"w{i}" for i in range(N)],
        nodes=nodes,
        edges=[(i, i + 1) for i in range(N - 1)],
        start_nodes=[0],
        end_nodes=[N - 1],
    )
    plan = build_edit_plan(cands, graph, inv, project_sample_rate=sr, video_fps=fps, rendering_mode="cuts-only")
    plan.validate()
    assert plan.total_output_samples == 2400000
    assert plan.total_video_frames == 1500
    assert sum(s.video_frame_count for s in plan.segments) == 1500


def test_t03_subframe_cuts_and_fractional_fps():
    """T03: Contributions shorter than a frame; fractional FPS. Audio preserved; zero-frame picture spans allowed."""
    from contracts import RationalTime, TargetGraph, TargetPhoneNode
    from media_time import cumulative_frame_ceiling

    sr = 48000
    fps = RationalTime(30000, 1001)  # 29.97 fps NTSC
    # 5ms cuts = 240 samples each (each frame is ~1601.6 samples)
    cut_samples = 240
    N = 20
    src = SourceRecord(
        content_id="src_sub",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=1,
    )
    cands = []
    events = {}
    nodes = []
    for i in range(N):
        ev_id = f"ev_sub_{i}"
        events[ev_id] = AlignedEvent(
            event_id=ev_id, source_content_id="src_sub", utterance_id=f"u_{i}", speaker_id="spk1",
            kind="phone", start_sample=i * cut_samples, end_sample=(i + 1) * cut_samples,
            label="AA", base_phone="AA"
        )
        nodes.append(TargetPhoneNode(
            node_id=i, token_index=i, word_text=f"w{i}", phone="AA", base_phone="AA",
            stress=1, is_word_initial=True, is_word_final=True, is_phrase_final=(i == N - 1)
        ))
        cands.append(SpanCandidate(
            candidate_id=f"c_{i}",
            target_node_start=i,
            target_node_end=i,
            covered_nodes=[i],
            source_content_id="src_sub",
            speaker_id="spk1",
            start_sample=i * cut_samples,
            end_sample=(i + 1) * cut_samples,
            event_ids=[ev_id],
            span_type="word",
            matched_word=f"w{i}",
        ))

    inv = InventoryIndex(sources={"src_sub": src}, events=events)
    graph = TargetGraph(
        raw_text=" ".join(f"w{i}" for i in range(N)),
        normalized_words=[f"w{i}" for i in range(N)],
        nodes=nodes,
        edges=[(i, i + 1) for i in range(N - 1)],
        start_nodes=[0],
        end_nodes=[N - 1],
    )
    plan = build_edit_plan(cands, graph, inv, project_sample_rate=sr, video_fps=fps, rendering_mode="cuts-only")
    plan.validate()

    assert plan.total_output_samples == N * cut_samples  # 4800
    expected_frames = cumulative_frame_ceiling(plan.total_output_samples, sr, fps)
    assert plan.total_video_frames == expected_frames
    assert sum(s.video_frame_count for s in plan.segments) == expected_frames
    # Subframe segments must contain zero-frame spans without error
    zero_frame_segs = [s for s in plan.segments if s.video_frame_count == 0]
    assert len(zero_frame_segs) > 0, "Subframe cuts must produce valid zero-frame picture spans"


def test_t04_exact_gap_250ms_samples_and_phrase_blocking():
    """T04: 'hello <gap:250ms> there' with contiguous source phrase available -> exactly 12,000 gap samples; no swallowed gap."""
    from contracts import WordOccurrence
    from span_selector import SpanSelector

    sr = 48000
    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH", "EH1", "L", "OW0"]]
    lex.word_to_pronunciations["there"] = [["DH", "EH1", "R"]]

    builder = TargetGraphBuilder(lex)
    graph = builder.build("hello <gap:250ms> there")

    # Verify explicit_gaps is populated on graph
    assert graph.explicit_gaps.get(0) == 0.250

    src = SourceRecord(
        content_id="src_phrase",
        source_path="video.mp4",
        native_sample_rate=sr,
        native_channels=1,
    )
    # Source has consecutive words "hello" and "there"
    ev1 = AlignedEvent(
        event_id="ev_h", source_content_id="src_phrase", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=24000, label="HH", base_phone="HH",
        word_occurrence_id="wo_h", successor_id="ev_t"
    )
    ev2 = AlignedEvent(
        event_id="ev_t", source_content_id="src_phrase", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=24000, end_sample=48000, label="DH", base_phone="DH",
        word_occurrence_id="wo_t"
    )
    wo1 = WordOccurrence(
        word_id="wo_h", source_content_id="src_phrase", speaker_id="spk1",
        word_normalized="hello", word_original="hello", start_sample=0, end_sample=24000,
        phone_event_ids=["ev_h"], realized_phones=["HH"]
    )
    wo2 = WordOccurrence(
        word_id="wo_t", source_content_id="src_phrase", speaker_id="spk1",
        word_normalized="there", word_original="there", start_sample=24000, end_sample=48000,
        phone_event_ids=["ev_t"], realized_phones=["DH"]
    )


    inv = InventoryIndex(
        sources={"src_phrase": src},
        events={"ev_h": ev1, "ev_t": ev2},
        words={"wo_h": wo1, "wo_t": wo2},
        word_to_word_ids={"hello": ["wo_h"], "there": ["wo_t"]},
    )

    selector = SpanSelector(inv)
    cands = selector.retrieve_candidates(graph, speaker_id="spk1")


    # SpanSelector MUST NOT create a single phrase candidate covering both words across <gap:250ms>
    phrase_cands = [c for c in cands if c.span_type == "phrase"]
    assert len(phrase_cands) == 0, "Phrase candidates must not span across explicit gaps"

    # Select word candidates
    c1 = next(c for c in cands if c.matched_word == "hello")
    c2 = next(c for c in cands if c.matched_word == "there")

    plan = build_edit_plan([c1, c2], graph, inv, project_sample_rate=sr, rendering_mode="cuts-only")
    plan.validate()

    pause_segs = [s for s in plan.segments if s.is_silence]
    assert len(pause_segs) == 1
    # Exactly 250ms at 48000 Hz = 12,000 samples
    assert pause_segs[0].duration_samples == 12000


def test_t05_zero_leading_trailing_consecutive_gaps():
    """T05: Zero, leading, trailing, consecutive gaps; cuts-only."""
    sr = 48000
    lex = Lexicon()
    lex.word_to_pronunciations["hello"] = [["HH"]]
    lex.word_to_pronunciations["there"] = [["DH"]]

    builder = TargetGraphBuilder(lex)

    # 1. Zero gap: <gap:0ms> should have 0 samples
    g_zero = builder.build("hello <gap:0ms> there")
    assert g_zero.explicit_gaps.get(0) == 0.0

    # 2. Consecutive gaps: <gap:100ms> <gap:150ms> add together to 250ms = 12000 samples
    g_consec = builder.build("hello <gap:100ms> <gap:150ms> there")
    assert abs(g_consec.explicit_gaps.get(0) - 0.250) < 1e-5

    # 3. Leading and trailing gaps
    g_flanked = builder.build("<gap:100ms> hello <gap:200ms> there <gap:300ms>")
    assert abs(g_flanked.explicit_gaps.get(-1) - 0.100) < 1e-5
    assert abs(g_flanked.explicit_gaps.get(0) - 0.200) < 1e-5
    assert abs(g_flanked.explicit_gaps.get(1) - 0.300) < 1e-5

    # Build plan for flanked target
    src = SourceRecord(
        content_id="src_1", source_path="video.mp4", native_sample_rate=sr, native_channels=1
    )
    ev1 = AlignedEvent(
        event_id="ev1", source_content_id="src_1", utterance_id="u1", speaker_id="spk1",
        kind="phone", start_sample=0, end_sample=24000, label="HH", base_phone="HH"
    )
    ev2 = AlignedEvent(
        event_id="ev2", source_content_id="src_1", utterance_id="u2", speaker_id="spk1",
        kind="phone", start_sample=48000, end_sample=72000, label="DH", base_phone="DH"
    )
    inv = InventoryIndex(sources={"src_1": src}, events={"ev1": ev1, "ev2": ev2})
    cand1 = SpanCandidate(
        candidate_id="c1", target_node_start=0, target_node_end=0, covered_nodes=[0],
        source_content_id="src_1", speaker_id="spk1", start_sample=0, end_sample=24000,
        event_ids=["ev1"], span_type="word", matched_word="hello",
    )
    cand2 = SpanCandidate(
        candidate_id="c2", target_node_start=1, target_node_end=1, covered_nodes=[1],
        source_content_id="src_1", speaker_id="spk1", start_sample=48000, end_sample=72000,
        event_ids=["ev2"], span_type="word", matched_word="there",
    )

    plan = build_edit_plan([cand1, cand2], g_flanked, inv, project_sample_rate=sr, rendering_mode="cuts-only")
    plan.validate()

    pauses = [s for s in plan.segments if s.is_silence]
    assert len(pauses) == 3, "Leading, internal, and trailing gaps must produce 3 pause segments"
    assert pauses[0].duration_samples == int(round(0.100 * sr))  # 4800
    assert pauses[1].duration_samples == int(round(0.200 * sr))  # 9600
    assert pauses[2].duration_samples == int(round(0.300 * sr))  # 14400

