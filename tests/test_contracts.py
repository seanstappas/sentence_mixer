"""
Tests for shared data contracts and schema validation.
Verifies invariants from Milestone M0.
"""

import pytest
from contracts import (
    RationalTime,
    SourceRecord,
    AlignedEvent,
    WordOccurrence,
    InventoryIndex,
    EditPlan,
    PlanSegment,
    PlanTransition,
    INVENTORY_SCHEMA_VERSION,
    EDIT_PLAN_SCHEMA_VERSION,
)


def test_rational_time_parsing():
    r1 = RationalTime.from_str("30000/1001")
    assert r1.numerator == 30000
    assert r1.denominator == 1001
    assert abs(r1.as_float() - 29.97002997) < 1e-5

    # Decimal 29.97 alias
    r2 = RationalTime.from_str("29.97")
    assert r2.numerator == 30000
    assert r2.denominator == 1001

    # Decimal 23.976 alias
    r3 = RationalTime.from_str("23.976")
    assert r3.numerator == 24000
    assert r3.denominator == 1001

    # Standard integer frame rate
    r4 = RationalTime.from_number(24)
    assert r4.numerator == 24
    assert r4.denominator == 1


def test_source_record_validation():
    rec = SourceRecord(
        content_id="test_content_1",
        source_path="input/test.mp4",
        native_sample_rate=48000,
        native_channels=2,
    )
    rec.validate()

    with pytest.raises(ValueError, match="content_id"):
        invalid = SourceRecord(
            content_id="",
            source_path="input/test.mp4",
            native_sample_rate=48000,
            native_channels=2,
        )
        invalid.validate()

    with pytest.raises(ValueError, match="Invalid sample rate"):
        invalid = SourceRecord(
            content_id="id",
            source_path="input/test.mp4",
            native_sample_rate=-1,
            native_channels=2,
        )
        invalid.validate()


def test_aligned_event_and_word_occurrence():
    ev1 = AlignedEvent(
        event_id="ev_0",
        source_content_id="src_1",
        utterance_id="utt_1",
        speaker_id="spk_1",
        kind="phone",
        start_sample=0,
        end_sample=2400,
        label="HH",
        base_phone="HH",
    )
    ev1.validate()

    ev2 = AlignedEvent(
        event_id="ev_1",
        source_content_id="src_1",
        utterance_id="utt_1",
        speaker_id="spk_1",
        kind="phone",
        start_sample=2400,
        end_sample=4800,
        label="AH0",
        base_phone="AH",
    )
    ev2.validate()

    # Invalid bounds check: start > end
    with pytest.raises(ValueError, match="bounds"):
        bad_ev = AlignedEvent(
            event_id="bad",
            source_content_id="src_1",
            utterance_id="utt_1",
            speaker_id="spk_1",
            kind="phone",
            start_sample=5000,
            end_sample=4800,
            label="AH0",
        )
        bad_ev.validate()

    word = WordOccurrence(
        word_id="w_0",
        source_content_id="src_1",
        speaker_id="spk_1",
        word_normalized="hello",
        word_original="Hello",
        start_sample=0,
        end_sample=4800,
        phone_event_ids=["ev_0", "ev_1"],
        realized_phones=["HH", "AH0"],
    )
    word.validate()


def test_inventory_roundtrip():
    inv = InventoryIndex(
        schema_version=INVENTORY_SCHEMA_VERSION,
        sources={
            "src_1": SourceRecord(
                content_id="src_1",
                source_path="video.mp4",
                native_sample_rate=48000,
                native_channels=2,
            )
        },
        events={
            "ev_0": AlignedEvent(
                event_id="ev_0",
                source_content_id="src_1",
                utterance_id="utt_1",
                speaker_id="spk_1",
                kind="phone",
                start_sample=0,
                end_sample=2400,
                label="HH",
            )
        },
    )
    inv.validate()

    d = inv.to_dict()
    roundtrip = InventoryIndex.from_dict(d)
    assert roundtrip.schema_version == INVENTORY_SCHEMA_VERSION
    assert "src_1" in roundtrip.sources
    assert roundtrip.sources["src_1"].native_sample_rate == 48000
    assert "ev_0" in roundtrip.events

    # Version mismatch rejects
    bad_dict = dict(d)
    bad_dict["schema_version"] = "1.0.0"
    with pytest.raises(ValueError, match="Incompatible schema version"):
        InventoryIndex.from_dict(bad_dict).validate()


def test_edit_plan_timing_equation():
    seg1 = PlanSegment(
        segment_index=0,
        source_content_id="src_1",
        speaker_id="spk_1",
        source_start_sample=1000,
        source_end_sample=25000,  # 24000 samples (0.5s @ 48kHz)
        output_start_sample=0,
        output_end_sample=24000,
        target_text="hello",
        target_phones=["HH", "AH0", "L", "OW1"],
        span_type="word",
        source_video_path="input/src1.mp4",
    )

    seg2 = PlanSegment(
        segment_index=1,
        source_content_id="src_1",
        speaker_id="spk_1",
        source_start_sample=30000,
        source_end_sample=54000,  # 24000 samples
        output_start_sample=23760,  # overlap of 240 samples (5ms)
        output_end_sample=47760,
        target_text="there",
        target_phones=["DH", "EH1", "R"],
        span_type="word",
        source_video_path="input/src1.mp4",
    )

    trans = PlanTransition(
        transition_index=0,
        from_segment_index=0,
        to_segment_index=1,
        transition_type="overlap",
        overlap_samples=240,
        join_cost=0.1,
    )

    # Total expected output samples = 24000 + 24000 - 240 = 47760
    plan = EditPlan(
        schema_version=EDIT_PLAN_SCHEMA_VERSION,
        plan_id="plan_test",
        target_text="hello there",
        speaker_id="spk_1",
        project_sample_rate=48000,
        total_output_samples=47760,
        segments=[seg1, seg2],
        transitions=[trans],
    )
    plan.validate()

    # If declared total output samples differs, validation must fail
    plan.total_output_samples = 48000
    with pytest.raises(ValueError, match="EditPlan sample count mismatch"):
        plan.validate()
