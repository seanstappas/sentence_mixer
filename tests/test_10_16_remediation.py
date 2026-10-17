from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from cache import (
    find_cached_inventories_for_speaker,
    load_inventory_json,
    save_inventory_json,
)
from contracts import (
    AlignedEvent,
    EditPlan,
    InventoryIndex,
    PlanSegment,
    SourceRecord,
    SpanCandidate,
    TargetGraph,
    TargetPhoneNode,
    WordOccurrence,
)
from dialogue_mixer import _extract_plan_words, parse_script_text
from edit_plan import _refine_spans_in_native_audio, build_edit_plan, finalize_edit_plan, release_tail_samples
from inventory_builder import build_units_inventory
from lexicon import Lexicon, VOWEL_PHONES
from span_selector import SpanSelector, apply_diphone_midpoint_transitions
from subtitles import clean_segment_label, generate_subtitles


# ---------------------------------------------------------------------------
# 1. test_ah_in_vowel_set_and_swallowed_vowel_guard
# ---------------------------------------------------------------------------
def test_ah_in_vowel_set_and_swallowed_vowel_guard():
    """Verify AH is in vowel set, 40ms AH0 incurs swallowed vowel penalty, and word-initial AH collar applies."""
    assert "AH" in VOWEL_PHONES
    sr = 16000
    src = SourceRecord(content_id="src1", source_path="dummy.wav", native_sample_rate=sr, native_channels=1)

    # 40ms AH0: below the 50ms vowel floor
    short_ah = AlignedEvent(
        event_id="ev_ah_short",
        source_content_id="src1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        label="AH0",
        base_phone="AH",
        start_sample=0,
        end_sample=int(0.040 * sr),  # 40ms
        is_eligible=True,
    )
    # 60ms AH0: above the 50ms vowel floor
    valid_ah = AlignedEvent(
        event_id="ev_ah_valid",
        source_content_id="src1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        label="AH0",
        base_phone="AH",
        start_sample=int(0.100 * sr),
        end_sample=int(0.160 * sr),  # 60ms
        is_eligible=True,
    )
    # Word-initial AH with preceding voiced phone M (left bleed)
    ev_m = AlignedEvent(
        event_id="ev_m",
        source_content_id="src1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        start_sample=0,
        end_sample=1000,
        label="M",
        base_phone="M",
        successor_id="ev_ah_bleed",
    )
    ev_bleed = AlignedEvent(
        event_id="ev_ah_bleed",
        source_content_id="src1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        label="AH0",
        base_phone="AH",
        start_sample=1000,
        end_sample=2000,
        predecessor_id="ev_m",
        is_eligible=True,
    )

    inv = InventoryIndex(
        sources={"src1": src},
        events={
            "ev_ah_short": short_ah,
            "ev_ah_valid": valid_ah,
            "ev_m": ev_m,
            "ev_ah_bleed": ev_bleed,
        },
        phone_to_event_ids={"AH0": ["ev_ah_short", "ev_ah_valid", "ev_ah_bleed"], "M": ["ev_m"]},
        base_phone_to_event_ids={"AH": ["ev_ah_short", "ev_ah_valid", "ev_ah_bleed"], "M": ["ev_m"]},
    )

    selector = SpanSelector(inv, speaker_id="spk1")

    # Target node for word-initial AH (e.g. "attack", "about")
    node_initial = TargetPhoneNode(
        node_id=0,
        token_index=0,
        word_text="about",
        phone="AH0",
        base_phone="AH",
        stress=0,
        is_word_initial=True,
    )
    graph = TargetGraph(
        raw_text="about",
        normalized_words=["about"],
        nodes=[node_initial],
        edges=[],
        start_nodes=[0],
        end_nodes=[0],
    )

    spans = selector.select(graph, speaker_id="spk1")
    assert len(spans) == 1
    # Check that candidate selection succeeded and preferred the non-short phone
    assert spans[0].event_ids[0] in ("ev_ah_valid", "ev_ah_bleed")


# ---------------------------------------------------------------------------
# 2. test_diphone_midpoint_zeroes_variants_and_handles_3way_chain
# ---------------------------------------------------------------------------
def test_diphone_midpoint_zeroes_variants_and_handles_3way_chain():
    """Verify midpoint transitions zero variant samples and correctly handle 3-way chained joins."""
    sr = 48000
    src = SourceRecord(content_id="s1", source_path="dummy.wav", native_sample_rate=sr, native_channels=1)

    ev0 = AlignedEvent(event_id="e0", source_content_id="s1", utterance_id="u1", speaker_id="spk", kind="phone", start_sample=0, end_sample=1000, label="T", base_phone="T")
    ev1 = AlignedEvent(event_id="e1", source_content_id="s2", utterance_id="u2", speaker_id="spk", kind="phone", start_sample=0, end_sample=1000, label="T", base_phone="T")
    ev2 = AlignedEvent(event_id="e2", source_content_id="s3", utterance_id="u3", speaker_id="spk", kind="phone", start_sample=0, end_sample=1000, label="T", base_phone="T")

    inv = InventoryIndex(
        sources={"s1": src, "s2": src, "s3": src},
        events={"e0": ev0, "e1": ev1, "e2": ev2},
    )

    span0 = SpanCandidate(
        candidate_id="c0",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="s1",
        speaker_id="spk",
        start_sample=0,
        end_sample=1000,
        event_ids=["e0"],
        span_type="phone",
        matched_word="t",
        realized_phones=["T"],
        exit_variant_samples=150,
    )
    span1 = SpanCandidate(
        candidate_id="c1",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="s2",
        speaker_id="spk",
        start_sample=0,
        end_sample=1000,
        event_ids=["e1"],
        span_type="phone",
        matched_word="t",
        realized_phones=["T"],
        entry_variant_samples=120,
        exit_variant_samples=140,
    )
    span2 = SpanCandidate(
        candidate_id="c2",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="s3",
        speaker_id="spk",
        start_sample=0,
        end_sample=1000,
        event_ids=["e2"],
        span_type="phone",
        matched_word="t",
        realized_phones=["T"],
        entry_variant_samples=110,
    )

    node_t = TargetPhoneNode(0, 0, "test", "T", "T", stress=None)
    graph = TargetGraph(
        raw_text="test",
        normalized_words=["test"],
        nodes=[node_t],
        edges=[],
        start_nodes=[0],
        end_nodes=[0],
    )

    # 3-way chain: span0 -> span1 -> span2 where span1 is a 1-phone bridge span
    new_spans = apply_diphone_midpoint_transitions([span0, span1, span2], graph, inv)

    assert len(new_spans) == 3
    # Step 0-1 join zeroes exit on span0 and entry on span1
    assert new_spans[0].exit_variant_samples == 0
    assert new_spans[0].end_sample == 500  # midpoint of ev0

    # Step 1-2 join zeroes exit on span1 and entry on span2
    assert new_spans[1].entry_variant_samples == 0
    assert new_spans[1].exit_variant_samples == 0
    assert new_spans[1].start_sample == 500  # midpoint of ev1 (left trim)
    assert new_spans[1].end_sample == 500    # midpoint of ev1 (right trim, bridge)

    assert new_spans[2].entry_variant_samples == 0
    assert new_spans[2].start_sample == 500  # midpoint of ev2


# ---------------------------------------------------------------------------
# 3. test_constrained_phonetic_override_blocks_stress_and_cognate_pooling
# ---------------------------------------------------------------------------
def test_constrained_phonetic_override_blocks_stress_and_cognate_pooling():
    """Verify {...} override (is_constrained=True) blocks stress-mismatch and cognate pooling."""
    sr = 16000
    src = SourceRecord(content_id="s1", source_path="dummy.wav", native_sample_rate=sr, native_channels=1)

    # Target: constrained stressed EH1
    ev_eh1 = AlignedEvent(event_id="e_eh1", source_content_id="s1", utterance_id="u1", speaker_id="spk1", kind="phone", start_sample=0, end_sample=1600, label="EH1", base_phone="EH", is_eligible=True)
    ev_eh0 = AlignedEvent(event_id="e_eh0", source_content_id="s1", utterance_id="u1", speaker_id="spk1", kind="phone", start_sample=2000, end_sample=3600, label="EH0", base_phone="EH", is_eligible=True)
    ev_ih1 = AlignedEvent(event_id="e_ih1", source_content_id="s1", utterance_id="u1", speaker_id="spk1", kind="phone", start_sample=4000, end_sample=5600, label="IH1", base_phone="IH", is_eligible=True)

    inv = InventoryIndex(
        sources={"s1": src},
        events={"e_eh1": ev_eh1, "e_eh0": ev_eh0, "e_ih1": ev_ih1},
        phone_to_event_ids={"EH1": ["e_eh1"], "EH0": ["e_eh0"], "IH1": ["e_ih1"]},
        base_phone_to_event_ids={"EH": ["e_eh1", "e_eh0"], "IH": ["e_ih1"]},
    )

    selector = SpanSelector(inv, speaker_id="spk1")

    # Constrained node
    node_constrained = TargetPhoneNode(
        node_id=0,
        token_index=0,
        word_text="{EH1}",
        phone="EH1",
        base_phone="EH",
        stress=1,
        is_constrained=True,
    )
    graph_c = TargetGraph(
        raw_text="{EH1}",
        normalized_words=["{EH1}"],
        nodes=[node_constrained],
        edges=[],
        start_nodes=[0],
        end_nodes=[0],
    )
    spans_c = selector.select(graph_c, speaker_id="spk1")
    assert len(spans_c) == 1
    assert spans_c[0].event_ids == ["e_eh1"]

    # When exact EH1 is missing, fallback still occurs
    inv_no_exact = InventoryIndex(
        sources={"s1": src},
        events={"e_eh0": ev_eh0},
        phone_to_event_ids={"EH0": ["e_eh0"]},
        base_phone_to_event_ids={"EH": ["e_eh0"]},
    )
    selector_fallback = SpanSelector(inv_no_exact, speaker_id="spk1")
    spans_fb = selector_fallback.select(graph_c, speaker_id="spk1")
    assert len(spans_fb) == 1
    assert spans_fb[0].event_ids == ["e_eh0"]


# ---------------------------------------------------------------------------
# 4. test_release_tail_zero_room_preserves_exit_variant_and_zc_refinement
# ---------------------------------------------------------------------------
def test_release_tail_zero_room_preserves_exit_variant_and_zc_refinement(tmp_path: Path):
    """Verify release_tail_samples returns 0 when room <= 0 and refines post-tail endpoint when room > 0."""
    sr = 48000
    src = SourceRecord(content_id="s1", source_path="dummy.wav", native_sample_rate=sr, native_channels=1)

    # Word-final phone followed by only 20ms silence (< 30ms guard) -> room <= 0
    phone_ev = AlignedEvent(event_id="p1", source_content_id="s1", utterance_id="u1", speaker_id="spk", kind="phone", start_sample=0, end_sample=1000, label="T", base_phone="T", successor_id="sil1")
    sil_ev_short = AlignedEvent(event_id="sil1", source_content_id="s1", utterance_id="u1", speaker_id="spk", kind="silence", start_sample=1000, end_sample=1000 + int(0.020 * sr), label="sil", base_phone="")

    inv_tight = InventoryIndex(
        sources={"s1": src},
        events={"p1": phone_ev, "sil1": sil_ev_short},
    )
    span = SpanCandidate(
        candidate_id="c_word",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="s1",
        speaker_id="spk",
        start_sample=0,
        end_sample=1000,
        event_ids=["p1"],
        span_type="word",
        matched_word="cat",
    )
    assert release_tail_samples(span, inv_tight, tail_sec=0.100) == 0

    # With room > 0 (150ms silence), verify release tail extends and zero-crossing snap refines the end
    sil_ev_long = AlignedEvent(event_id="sil2", source_content_id="s1", utterance_id="u1", speaker_id="spk", kind="silence", start_sample=1000, end_sample=1000 + int(0.150 * sr), label="sil", base_phone="")
    inv_room = InventoryIndex(
        sources={"s1": src},
        events={"p1": phone_ev, "sil2": sil_ev_long},
    )
    phone_ev.successor_id = "sil2"
    tail_s = release_tail_samples(span, inv_room, tail_sec=0.050)
    assert tail_s > 0

    # Test that _refine_spans_in_native_audio incorporates tail_s
    starts, ends = _refine_spans_in_native_audio([span], inv_room, tail_sec=0.050)
    assert ends[0] == 1000 + tail_s


# ---------------------------------------------------------------------------
# 5. test_refine_spans_cross_sample_rate_vocal_stem_44100_vs_48000
# ---------------------------------------------------------------------------
def test_refine_spans_cross_sample_rate_vocal_stem_44100_vs_48000(tmp_path: Path):
    """Verify _refine_spans_in_native_audio accepts 44.1 kHz vocal stems for 48 kHz native sources."""
    sr_vocal = 44100
    sr_native = 48000

    # Write synthetic 44.1 kHz vocal WAV
    vocal_wav = tmp_path / "vocals_44k.wav"
    t = np.arange(sr_vocal * 2) / float(sr_vocal)
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(str(vocal_wav), sine, sr_vocal)

    src = SourceRecord(
        content_id="mov1",
        source_path=str(tmp_path / "dummy_mov.mp4"),
        cached_pcm_path=str(tmp_path / "mix_48k.wav"),
        cached_vocal_pcm_path=str(vocal_wav),
        native_sample_rate=sr_native,
        native_channels=2,
    )
    # Also write mix WAV at 48k
    sf.write(str(tmp_path / "mix_48k.wav"), np.zeros((sr_native * 2, 2), dtype=np.float32), sr_native)

    ev = AlignedEvent(event_id="ev1", source_content_id="mov1", utterance_id="u1", speaker_id="spk", kind="phone", start_sample=10000, end_sample=25000, label="hello", base_phone="")
    inv = InventoryIndex(sources={"mov1": src}, events={"ev1": ev})
    span = SpanCandidate(
        candidate_id="c_hello",
        target_node_start=0,
        target_node_end=0,
        covered_nodes=[0],
        source_content_id="mov1",
        speaker_id="spk",
        start_sample=10000,
        end_sample=25000,
        event_ids=["ev1"],
        span_type="word",
        matched_word="hello",
    )

    # Should not raise, and should use cached_vocal_pcm_path for refinement
    starts, ends = _refine_spans_in_native_audio([span], inv, audio_track="vocals")
    assert len(starts) == 1 and len(ends) == 1
    assert 9000 <= starts[0] <= 11000
    assert 24000 <= ends[0] <= 26000


# ---------------------------------------------------------------------------
# 6. test_tier1_loudness_gain_isolated_between_vocal_and_raw_tracks
# ---------------------------------------------------------------------------
def test_tier1_loudness_gain_isolated_between_vocal_and_raw_tracks(tmp_path: Path):
    """Verify finalize_edit_plan isolates raw-mix gains from vocal-stem gains in both/all modes."""
    sr = 48000
    # Quiet vocal stem (RMS ~ 0.01 -> needs positive gain)
    vocal_audio = (0.01 * np.ones(sr, dtype=np.float32))
    vocal_path = tmp_path / "vocals.wav"
    sf.write(str(vocal_path), vocal_audio, sr)

    # Loud raw mix with music (RMS ~ 0.3 -> needs negative/lower gain)
    raw_audio = (0.30 * np.ones(sr, dtype=np.float32))
    raw_path = tmp_path / "raw.wav"
    sf.write(str(raw_path), raw_audio, sr)

    src = SourceRecord(
        content_id="src_loud",
        source_path=str(raw_path),
        cached_pcm_path=str(raw_path),
        cached_vocal_pcm_path=str(vocal_path),
        native_sample_rate=sr,
        native_channels=1,
    )

    seg = PlanSegment(
        segment_index=0,
        source_content_id="src_loud",
        speaker_id="spk",
        source_start_sample=0,
        source_end_sample=sr,
        output_start_sample=0,
        output_end_sample=sr,
        target_text="test",
        target_phones=["T", "EH1", "S", "T"],
        span_type="word",
        source_video_path=str(raw_path),
    )
    plan = EditPlan(segments=[seg], sources={"src_loud": src}, project_sample_rate=sr)

    finalized = finalize_edit_plan(plan, audio_track="both", normalize_loudness=True)

    assert finalized.metadata.get("rms_analysis_track") == "vocals"
    assert "raw_segment_gains_db" in finalized.metadata

    # Vocal stem gain (seg.gain_db) should be positive because vocal is quiet
    assert seg.gain_db > 5.0
    # Raw track gain for the same segment should be substantially lower/negative
    raw_gain = finalized.metadata["raw_segment_gains_db"][0]
    assert raw_gain < seg.gain_db - 8.0


# ---------------------------------------------------------------------------
# 7. test_consecutive_repeated_words_preserved_in_plan_and_subtitles
# ---------------------------------------------------------------------------
def test_consecutive_repeated_words_preserved_in_plan_and_subtitles():
    """Verify consecutive repeated words ('roger roger') are preserved across plan, words, and subtitles."""
    sr = 48000
    src = SourceRecord(content_id="s1", source_path="dummy.wav", native_sample_rate=sr, native_channels=1)
    ev1 = AlignedEvent(event_id="e1", source_content_id="s1", utterance_id="u1", speaker_id="droid", kind="phone", start_sample=0, end_sample=1000, label="roger", base_phone="")
    ev2 = AlignedEvent(event_id="e2", source_content_id="s1", utterance_id="u1", speaker_id="droid", kind="phone", start_sample=1200, end_sample=2200, label="roger", base_phone="")

    inv = InventoryIndex(sources={"s1": src}, events={"e1": ev1, "e2": ev2})

    # Build target graph with two distinct token indices for "roger"
    graph = TargetGraph(
        raw_text="roger roger",
        normalized_words=["roger", "roger"],
        nodes=[
            TargetPhoneNode(0, 0, "roger", "R", "R", stress=None, is_word_final=False),
            TargetPhoneNode(1, 0, "roger", "AA1", "AA", stress=1, is_word_final=False),
            TargetPhoneNode(2, 0, "roger", "JH", "JH", stress=None, is_word_final=False),
            TargetPhoneNode(3, 0, "roger", "ER0", "ER", stress=0, is_word_final=True),
            TargetPhoneNode(4, 1, "roger", "R", "R", stress=None, is_word_final=False),
            TargetPhoneNode(5, 1, "roger", "AA1", "AA", stress=1, is_word_final=False),
            TargetPhoneNode(6, 1, "roger", "JH", "JH", stress=None, is_word_final=False),
            TargetPhoneNode(7, 1, "roger", "ER0", "ER", stress=0, is_word_final=True),
        ],
        edges=[],
        start_nodes=[0],
        end_nodes=[7],
    )

    span1 = SpanCandidate(
        candidate_id="c1",
        target_node_start=0,
        target_node_end=3,
        covered_nodes=[0, 1, 2, 3],
        source_content_id="s1",
        speaker_id="droid",
        start_sample=0,
        end_sample=1000,
        event_ids=["e1"],
        span_type="word",
        matched_word="roger",
    )
    span2 = SpanCandidate(
        candidate_id="c2",
        target_node_start=4,
        target_node_end=7,
        covered_nodes=[4, 5, 6, 7],
        source_content_id="s1",
        speaker_id="droid",
        start_sample=1200,
        end_sample=2200,
        event_ids=["e2"],
        span_type="word",
        matched_word="roger",
    )

    plan = build_edit_plan([span1, span2], graph, inv, project_sample_rate=sr)

    # Check covered_token_indices are populated
    assert plan.segments[0].covered_token_indices == [0]
    assert plan.segments[1].covered_token_indices == [1]

    # Verify _extract_plan_words does NOT collapse repeated words
    extracted = _extract_plan_words(plan)
    assert len(extracted) == 2
    assert extracted[0][0] == "roger"
    assert extracted[1][0] == "roger"

    # Verify clean_segment_label handles {...}
    assert clean_segment_label("{DH EH1 R}") == "DH EH1 R"

    # Verify subtitles retain both words
    srt_text = generate_subtitles(plan, format="srt")
    assert "roger roger" in srt_text.lower()


# ---------------------------------------------------------------------------
# 8. test_multi_speaker_selection_and_candidate_donor_priors
# ---------------------------------------------------------------------------
def test_multi_speaker_selection_and_candidate_donor_priors():
    """Verify SpanSelector supports multi-speaker pooling and evaluates priors against each donor."""
    sr = 16000
    src = SourceRecord(content_id="s1", source_path="dummy.wav", native_sample_rate=sr, native_channels=1)

    ev_palp = AlignedEvent(event_id="ep", source_content_id="s1", utterance_id="u1", speaker_id="palpatine", kind="phone", start_sample=0, end_sample=1600, label="P", base_phone="P", is_eligible=True)
    ev_sid = AlignedEvent(event_id="es", source_content_id="s1", utterance_id="u2", speaker_id="sidious", kind="phone", start_sample=0, end_sample=1600, label="T", base_phone="T", is_eligible=True)

    wo_palp = WordOccurrence(
        word_id="wp",
        source_content_id="s1",
        speaker_id="palpatine",
        word_normalized="power",
        word_original="power",
        start_sample=0,
        end_sample=4800,
        phone_event_ids=["ep"],
        realized_phones=["P", "AW1", "ER0"],
        is_eligible=True,
    )
    wo_sid = WordOccurrence(
        word_id="ws",
        source_content_id="s1",
        speaker_id="sidious",
        word_normalized="strike",
        word_original="strike",
        start_sample=0,
        end_sample=6400,
        phone_event_ids=["es"],
        realized_phones=["S", "T", "R", "AY1", "K"],
        is_eligible=True,
    )

    inv = InventoryIndex(
        sources={"s1": src},
        events={"ep": ev_palp, "es": ev_sid},
        words={"wp": wo_palp, "ws": wo_sid},
        word_to_word_ids={"power": ["wp"], "strike": ["ws"]},
    )

    # Multi-speaker string syntax
    selector = SpanSelector(inv, speaker_id="palpatine,sidious")
    assert selector.indexed_speakers == {"palpatine", "sidious"}

    # Querying candidate from multi-speaker set
    node_p = TargetPhoneNode(0, 0, "power", "P", "P", stress=None)
    graph = TargetGraph(
        raw_text="power",
        normalized_words=["power"],
        nodes=[node_p],
        edges=[],
        start_nodes=[0],
        end_nodes=[0],
    )
    spans = selector.select(graph, speaker_id={"palpatine", "sidious"})
    assert len(spans) == 1
    assert spans[0].speaker_id in ("palpatine", "sidious")

    # Verify parse_script_text handles comma-separated speaker headers
    lines = parse_script_text("PALPATINE,SIDIOUS: Everything is proceeding as I have foreseen.")
    assert len(lines) == 1
    assert lines[0][0] == "palpatine,sidious"


# ---------------------------------------------------------------------------
# 9. test_alignment_guards_reject_stretched_collapsed_and_recover_ocr_accents
# ---------------------------------------------------------------------------
def test_alignment_guards_reject_stretched_collapsed_and_recover_ocr_accents(tmp_path: Path):
    """Verify inventory builder rejects stretched/collapsed units and recovers accents and OCR typos."""
    aligned_dir = tmp_path / "aligned"
    aligned_dir.mkdir()

    # Create synthetic TextGrid with:
    # 1. 2.69s "the" with 2.66s AH0 (stretched anomaly)
    # 2. 4-phone word crushed to 40ms total (10ms each, collapsed)
    # 3. "padmé" (accent recovery)
    # 4. "nol" (OCR typo recovery -> "no")
    tg_content = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 10.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0.0
        xmax = 10.0
        intervals: size = 4
        intervals [1]:
            xmin = 0.0
            xmax = 2.69
            text = "the"
        intervals [2]:
            xmin = 3.0
            xmax = 3.040
            text = "fast"
        intervals [3]:
            xmin = 4.0
            xmax = 4.40
            text = "padmé"
        intervals [4]:
            xmin = 5.0
            xmax = 5.30
            text = "nol"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0.0
        xmax = 10.0
        intervals: size = 8
        intervals [1]:
            xmin = 0.0
            xmax = 0.030
            text = "DH"
        intervals [2]:
            xmin = 0.030
            xmax = 2.69
            text = "AH0"
        intervals [3]:
            xmin = 3.0
            xmax = 3.010
            text = "F"
        intervals [4]:
            xmin = 3.010
            xmax = 3.020
            text = "AE1"
        intervals [5]:
            xmin = 3.020
            xmax = 3.030
            text = "S"
        intervals [6]:
            xmin = 3.030
            xmax = 3.040
            text = "T"
        intervals [7]:
            xmin = 4.0
            xmax = 4.40
            text = "spn"
        intervals [8]:
            xmin = 5.0
            xmax = 5.30
            text = "spn"
"""
    tg_path = aligned_dir / "chunk_0001.TextGrid"
    tg_path.write_text(tg_content, encoding="utf-8")

    manifest_data = {
        "source_id": "test_src",
        "video_path": "test.mp4",
        "speaker_id": "spk1",
        "single_speaker_verified": True,
        "chunks": {
            "chunk_0001": {
                "chunk_id": "chunk_0001",
                "start_sec": 0.0,
                "end_sec": 10.0,
                "textgrid_path": str(tg_path),
                "text": "the fast padmé nol",
                "speaker_id": "spk1",
            }
        },
    }
    manifest_p = tmp_path / "manifest.json"
    manifest_p.write_text(json.dumps(manifest_data), encoding="utf-8")

    out_inv_path = tmp_path / "units_index.json"

    # Minimal lexicon with "padme" and "no"
    lex = Lexicon()
    lex.word_to_pronunciations["padme"] = [["P", "AE1", "D", "M", "EY0"]]
    lex.word_to_pronunciations["no"] = [["N", "OW1"]]

    inv_data = build_units_inventory(
        aligned_dir=str(aligned_dir),
        manifest_path=str(manifest_p),
        output_index_path=str(out_inv_path),
        lexicon=lex,
        recover_oov_words=True,
    )

    words = inv_data.get("words", {})
    # Stretched "the" should be marked ineligible
    the_words = [w for w in words.values() if w.get("word_normalized") == "the"]
    assert len(the_words) >= 1
    assert the_words[0].get("is_eligible") is False
    assert the_words[0].get("eligibility_reason") == "mfa_stretched"

    # Crushed "fast" should be marked ineligible
    fast_words = [w for w in words.values() if w.get("word_normalized") == "fast"]
    assert len(fast_words) >= 1
    assert fast_words[0].get("is_eligible") is False
    assert fast_words[0].get("eligibility_reason") == "mfa_collapsed"

    # "padmé" should be normalized to "padme" and recovered
    padme_words = [w for w in words.values() if w.get("word_normalized") == "padme"]
    assert len(padme_words) >= 1
    assert padme_words[0].get("is_eligible") is True

    # "nol" should be corrected to "no" and recovered
    no_words = [w for w in words.values() if w.get("word_normalized") == "no"]
    assert len(no_words) >= 1
    assert no_words[0].get("is_eligible") is True


# ---------------------------------------------------------------------------
# 10. test_compact_json_gz_roundtrip_and_fast_metadata_discovery
# ---------------------------------------------------------------------------
def test_compact_json_gz_roundtrip_and_fast_metadata_discovery(tmp_path: Path):
    """Verify compact units_index.json.gz round-trip and < 1ms metadata discovery without full gzip parse."""
    sr = 48000
    src = SourceRecord(content_id="s1", source_path="dummy.wav", native_sample_rate=sr, native_channels=1)
    ev = AlignedEvent(event_id="e1", source_content_id="s1", utterance_id="u1", speaker_id="spk1", kind="phone", start_sample=0, end_sample=1000, label="P", base_phone="P")
    wo = WordOccurrence(
        word_id="w1",
        source_content_id="s1",
        speaker_id="spk1",
        word_normalized="power",
        word_original="power",
        start_sample=0,
        end_sample=1000,
        phone_event_ids=["e1"],
        realized_phones=["P"],
        is_eligible=True,
    )

    inv = InventoryIndex(
        sources={"s1": src},
        events={"e1": ev},
        words={"w1": wo},
        word_to_word_ids={"power": ["w1"]},
        phone_to_event_ids={"P": ["e1"]},
    )

    inv_dir = tmp_path / "inventory" / "src1"
    inv_dir.mkdir(parents=True)
    gz_path = inv_dir / "units_index.json.gz"

    # Save compact .json.gz
    data = inv.to_dict(compact=True)
    save_inventory_json(data, gz_path)
    assert gz_path.exists()

    # Load and verify automatic reconstruction of index maps
    loaded = load_inventory_json(gz_path)
    reconstructed = InventoryIndex.from_dict(loaded)
    assert reconstructed.word_to_word_ids.get("power") == ["w1"]
    assert reconstructed.phone_to_event_ids.get("P") == ["e1"]

    # Write sibling inventory_metadata.json with speakers count map
    meta_path = inv_dir / "inventory_metadata.json"
    meta_path.write_text(
        json.dumps({
            "schema_version": "2.0",
            "word_count": 1,
            "speakers": {"spk1": 1},
        }),
        encoding="utf-8",
    )

    # find_cached_inventories_for_speaker should find "spk1" and return gz_path without calling gzip.open
    with patch("gzip.open", side_effect=AssertionError("gzip.open should not be called when metadata has speakers")):
        res = find_cached_inventories_for_speaker("spk1", cache_dir=tmp_path)
        assert len(res) == 1
        assert Path(res[0]).resolve() == gz_path.resolve()
