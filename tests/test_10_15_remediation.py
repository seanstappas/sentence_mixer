from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from contracts import (
    AlignedEvent,
    EditPlan,
    InventoryIndex,
    PlanSegment,
    RationalTime,
    SourceRecord,
    SpanCandidate,
)
from dialogue_mixer import _extract_plan_words
from edit_plan import release_tail_samples
from movie_ingest import SegmentRecord, run_voice_checks
from span_selector import apply_diphone_midpoint_transitions


# ---------------------------------------------------------------------------
# 1. release_tail_samples negative boundary refinement
# ---------------------------------------------------------------------------
def test_release_tail_samples_with_negative_refinement():
    """Verify release_tail_samples extends into succ silence even when nominal_end is shifted backward."""
    sr = 48000
    src = SourceRecord(
        content_id="src1",
        source_path="dummy.wav",
        native_sample_rate=sr,
        native_channels=2,
    )
    last_phone = AlignedEvent(
        event_id="phone_1",
        source_content_id="src1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        label="T",
        base_phone="T",
        start_sample=1000,
        end_sample=2000,
        successor_id="sil_1",
    )
    succ_sil = AlignedEvent(
        event_id="sil_1",
        source_content_id="src1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="silence",
        label="sil",
        start_sample=2000,
        end_sample=10000,
    )
    inv = InventoryIndex(
        sources={"src1": src},
        events={"phone_1": last_phone, "sil_1": succ_sil},
    )
    span = SpanCandidate(
        candidate_id="c1",
        target_node_start=0,
        target_node_end=1,
        covered_nodes=[0],
        speaker_id="spk1",
        span_type="word",
        matched_word="test",
        source_content_id="src1",
        start_sample=1000,
        end_sample=2000,
        event_ids=["phone_1"],
    )

    # When nominal_end is shifted back 100 samples (e.g. by _refine_spans_in_native_audio)
    # The extension should bridge the 100-sample gap AND add the tail duration
    tail_sec = 0.100  # 4800 samples
    tail_samples = release_tail_samples(span, inv, tail_sec, nominal_end=span.end_sample - 100)
    expected_tail = 100 + int(round(tail_sec * sr))  # 100 + 4800 = 4900
    assert tail_samples == expected_tail


# ---------------------------------------------------------------------------
# 2. Multi-line minor speaker contamination (movie_ingest.py)
# ---------------------------------------------------------------------------
class _FakeEmbedder:
    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors

    def embed(self, start: float, end: float) -> np.ndarray:
        return self.vectors[f"{start:.3f}_{end:.3f}"]

    def embed_segments(self, wav_path: Path, segments: list[tuple[float, float]]) -> dict[float, np.ndarray]:
        return {s: self.vectors[f"{s:.3f}_{e:.3f}"] for s, e in segments if f"{s:.3f}_{e:.3f}" in self.vectors}

    def save(self) -> None:
        pass


def test_minor_speaker_thin_evidence_requires_unanimous_major():
    """Minor character lines (< 5 lines) require unanimous match across all lines to be flagged as voice_mismatch; non-unanimous contamination is preserved under thin-evidence safeguard."""
    rng = np.random.default_rng(123)
    dim = 16

    # 3 major characters (mira, tobin, osric)
    v_mira = np.zeros(dim, dtype=np.float32)
    v_mira[0] = 1.0
    v_tobin = np.zeros(dim, dtype=np.float32)
    v_tobin[1] = 1.0
    v_osric = np.zeros(dim, dtype=np.float32)
    v_osric[2] = 1.0

    centers = {"mira": v_mira, "tobin": v_tobin, "osric": v_osric}
    records: list[SegmentRecord] = []
    vectors: dict[str, np.ndarray] = {}
    t = 0.0

    def add_rec(label: str, vec: np.ndarray) -> SegmentRecord:
        nonlocal t
        v = (vec / np.linalg.norm(vec)).astype(np.float32)
        st = round(t, 3)
        en = round(st + 2.0, 3)
        key = f"{st:.3f}_{en:.3f}"
        r = SegmentRecord(
            key=key,
            chunk_id="c",
            seg_index=len(records),
            cue_start=st,
            start=st,
            end=en,
            label=label,
            label_source="subtitle_tag",
            text_hash="h",
        )
        r.speaker, r.status, r.eligible = label, "label", True
        vectors[key] = v
        records.append(r)
        t += 3.0
        return r

    # Add 15 lines for each major character
    for name in ("mira", "tobin", "osric"):
        for _ in range(15):
            noise = rng.normal(scale=0.01, size=dim).astype(np.float32)
            add_rec(name, centers[name] + noise)

    # Add 3 lines for minor character "trooper":
    # Lines 1 & 2 spoken by trooper (orthogonal to tobin)
    # Line 3 spoken by tobin (contaminated!)
    ortho = np.zeros(dim, dtype=np.float32)
    ortho[3] = 1.0
    t1 = add_rec("trooper", ortho + rng.normal(scale=0.01, size=dim).astype(np.float32))
    t2 = add_rec("trooper", ortho + rng.normal(scale=0.01, size=dim).astype(np.float32))
    t3 = add_rec("trooper", v_tobin + rng.normal(scale=0.01, size=dim).astype(np.float32))
    # Guard: 2 lines matching tobin unanimously are flagged as voice_mismatch
    g1 = add_rec("guard", v_tobin + rng.normal(scale=0.01, size=dim).astype(np.float32))
    g2 = add_rec("guard", v_tobin + rng.normal(scale=0.01, size=dim).astype(np.float32))

    _ = run_voice_checks(records, _FakeEmbedder(vectors), log=lambda m: None)

    # trooper lines are preserved under thin-evidence safeguard (not unanimous)
    assert t1.status in ("label_voice_ok", "label_voice_unclear")
    assert t1.eligible is True
    assert t2.status in ("label_voice_ok", "label_voice_unclear")
    assert t2.eligible is True
    assert t3.status in ("label_voice_ok", "label_voice_unclear")
    assert t3.eligible is True

    # guard lines (all matching tobin) are flagged voice_mismatch
    assert g1.status == "voice_mismatch"
    assert g1.eligible is False
    assert g2.status == "voice_mismatch"
    assert g2.eligible is False


# ---------------------------------------------------------------------------
# 3. Cross-track EBU R128 isolation in both / all modes
# ---------------------------------------------------------------------------
def test_ebur128_analysis_track_isolation(tmp_path: Path):
    """Verify frozen_stats from vocals analysis is not reused for raw audio track."""
    plan = EditPlan(
        schema_version="2.0.0",
        plan_id="test_plan",
        target_text="test",
        speaker_id="spk1",
        selector_mode="span",
        rendering_mode="source-derived",
        project_sample_rate=48000,
        video_fps=RationalTime(24, 1),
        total_output_samples=48000,
        total_video_frames=24,
        segments=[],
        metadata={
            "ebur128_analysis": {"input_i": -18.5, "input_tp": -2.0, "input_lra": 5.0, "input_thresh": -29.0},
            "ebur128_analysis_track": "vocals",
            "ebur128_target": {"target_i": -16.0, "target_tp": -1.0, "target_lra": 11.0},
        },
    )

    out_wav = tmp_path / "master.wav"
    sf.write(str(out_wav), np.zeros((48000, 2), dtype=np.float32), 48000)

    from video_assembler import render_master_pcm

    with patch("video_assembler.normalize_audio_ebur128") as mock_norm:
        # When audio_track is "raw", frozen_stats measured on "vocals" must be dropped
        render_master_pcm(plan, str(out_wav), normalize_loudness=True, audio_track="raw")
        assert mock_norm.called
        assert mock_norm.call_args[1].get("premeasured_stats") is None

        # When audio_track is "vocals", frozen_stats SHOULD be reused
        render_master_pcm(plan, str(out_wav), normalize_loudness=True, audio_track="vocals")
        assert mock_norm.call_args[1].get("premeasured_stats") == plan.metadata["ebur128_analysis"]


# ---------------------------------------------------------------------------
# 4. Dialogue _extract_plan_words with span_type == continue
# ---------------------------------------------------------------------------
def test_extract_plan_words_skips_continue_pauses():
    """Verify _extract_plan_words ignores span_type == 'continue' and '[pause]' text."""
    plan = EditPlan(
        schema_version="2.0.0",
        plan_id="plan_dialogue",
        target_text="hello world",
        speaker_id="spk1",
        selector_mode="span",
        rendering_mode="source-derived",
        project_sample_rate=48000,
        video_fps=RationalTime(24, 1),
        total_output_samples=144000,
        total_video_frames=72,
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src1",
                speaker_id="spk1",
                source_start_sample=0,
                source_end_sample=48000,
                output_start_sample=0,
                output_end_sample=48000,
                video_frame_count=24,
                span_type="word",
                target_text="hello",
                target_phones=["HH", "AH0", "L", "OW1"],
                source_video_path="",
            ),
            PlanSegment(
                segment_index=1,
                source_content_id="src1",
                speaker_id="spk1",
                source_start_sample=48000,
                source_end_sample=96000,
                output_start_sample=48000,
                output_end_sample=96000,
                video_frame_count=24,
                span_type="continue",
                is_silence=False,
                target_text="[pause]",
                target_phones=[],
                source_video_path="",
            ),
            PlanSegment(
                segment_index=2,
                source_content_id="src1",
                speaker_id="spk1",
                source_start_sample=96000,
                source_end_sample=144000,
                output_start_sample=96000,
                output_end_sample=144000,
                video_frame_count=24,
                span_type="word",
                target_text="world",
                target_phones=["W", "ER1", "L", "D"],
                source_video_path="",
            ),
        ],
    )

    words = _extract_plan_words(plan)
    assert len(words) == 2
    assert words[0][0] == "hello"
    assert words[1][0] == "world"
    assert all("[pause]" not in w[0] for w in words)


# ---------------------------------------------------------------------------
# 5. Diphone midpoint half-splice & idempotency
# ---------------------------------------------------------------------------
def test_diphone_midpoint_transitions_atomic_and_idempotent():
    """Verify apply_diphone_midpoint_transitions deduplicates only when both sides trim, and is idempotent."""
    sr = 48000
    src = SourceRecord(
        content_id="src1",
        source_path="dummy.wav",
        native_sample_rate=sr,
        native_channels=2,
    )
    ev_c = AlignedEvent(
        event_id="phone_c",
        source_content_id="src1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        label="T",
        base_phone="T",
        start_sample=1000,
        end_sample=2000,
    )
    ev_s = AlignedEvent(
        event_id="phone_s",
        source_content_id="src1",
        utterance_id="u1",
        speaker_id="spk1",
        kind="phone",
        label="T",
        base_phone="T",
        start_sample=3000,
        end_sample=4000,
    )
    inv = InventoryIndex(
        sources={"src1": src},
        events={"phone_c": ev_c, "phone_s": ev_s},
    )

    from contracts import TargetGraph, TargetPhoneNode

    nodes = [
        TargetPhoneNode(node_id=i, token_index=0, word_text="abt", phone="T", base_phone="T", stress=0)
        for i in range(8)
    ]
    graph = TargetGraph(
        raw_text="abt",
        normalized_words=["abt"],
        nodes=nodes,
        edges=[],
        start_nodes=[0],
        end_nodes=[7],
        punctuation_pauses={},
        terminal_punctuation="",
        explicit_gaps={},
    )

    # Both spans share node 5 at boundary
    span1 = SpanCandidate(
        candidate_id="c1",
        speaker_id="spk1",
        span_type="subword",
        matched_word="ab",
        source_content_id="src1",
        start_sample=500,
        end_sample=2000,
        event_ids=["phone_c"],
        covered_nodes=[4, 5],
        target_node_start=4,
        target_node_end=5,
        realized_phones=["AE", "T"],
    )
    span2 = SpanCandidate(
        candidate_id="c2",
        speaker_id="spk1",
        span_type="subword",
        matched_word="cd",
        source_content_id="src1",
        start_sample=3000,
        end_sample=4500,
        event_ids=["phone_s"],
        covered_nodes=[5, 6],
        target_node_start=5,
        target_node_end=7,
        realized_phones=["T", "D"],
    )

    # First pass: should trim at midpoints and deduplicate shared node 5 from span1
    res1 = apply_diphone_midpoint_transitions([span1, span2], graph, inv)
    assert len(res1) == 2
    assert res1[0].end_sample == 1500
    assert res1[1].start_sample == 3500
    assert res1[0].covered_nodes == [4]
    assert res1[0].realized_phones == ["AE"]
    assert res1[1].covered_nodes == [5, 6]
    assert res1[1].realized_phones == ["T", "D"]

    # Second pass: must be idempotent (no further popping of covered_nodes or realized_phones)
    res2 = apply_diphone_midpoint_transitions(res1, graph, inv)
    assert res2[0].covered_nodes == [4]
    assert res2[0].realized_phones == ["AE"]
    assert res2[1].covered_nodes == [5, 6]
    assert res2[1].realized_phones == ["T", "D"]


# ---------------------------------------------------------------------------
# 6. Clip-EOF freeze-frame extraction fallback
# ---------------------------------------------------------------------------
def test_clip_eof_freeze_frame_extraction(tmp_path: Path):
    """Verify that when extraction at video_source_pts_sec fails at EOF, -sseof -0.1 is executed."""
    import subprocess
    from contracts import PlanSegment, EditPlan, RationalTime

    seg = PlanSegment(
        segment_index=0,
        source_content_id="src1",
        speaker_id="spk1",
        source_start_sample=48000,
        source_end_sample=96000,
        output_start_sample=0,
        output_end_sample=48000,
        video_frame_count=24,
        span_type="pause",
        target_text="[pause]",
        target_phones=[],
        source_video_path=str(tmp_path / "dummy.mp4"),
        video_source_pts_sec=10.0,
        freeze_video=True,
    )
    (tmp_path / "dummy.mp4").write_text("video")

    plan = EditPlan(
        schema_version="2.0.0",
        plan_id="plan_freeze",
        target_text="test",
        speaker_id="spk1",
        selector_mode="span",
        rendering_mode="source-derived",
        project_sample_rate=48000,
        video_fps=RationalTime(24, 1),
        total_output_samples=48000,
        total_video_frames=24,
        segments=[seg],
    )

    executed_cmds: list[list[str]] = []

    def fake_subprocess_run(cmd, *args, **kwargs):
        executed_cmds.append(list(cmd))
        if "-sseof" in cmd:
            out_img = Path(cmd[-1])
            out_img.write_text("PNG DATA")
            return subprocess.CompletedProcess(cmd, 0)
        elif "-ss" in cmd and str(seg.source_video_path) in cmd:
            # Simulate EOF failure for normal seek
            return subprocess.CompletedProcess(cmd, 1)
        if len(cmd) > 1 and (cmd[-1].endswith(".mp4") or cmd[-1].endswith(".png")):
            Path(cmd[-1]).parent.mkdir(parents=True, exist_ok=True)
            Path(cmd[-1]).write_text("dummy")
        return subprocess.CompletedProcess(cmd, 0)

    with patch("subprocess.run", side_effect=fake_subprocess_run), \
         patch("video_assembler._has_video_stream", return_value=True), \
         patch("video_assembler.render_master_pcm"):
        from video_assembler import render_sentence_mix
        render_sentence_mix(plan, str(tmp_path / "out.mp4"), keep_temp=True)

    # Verify that -sseof -0.1 command was executed as fallback
    sseof_cmds = [c for c in executed_cmds if "-sseof" in c]
    assert len(sseof_cmds) > 0
    assert "-0.1" in sseof_cmds[0]

