"""
Canonical edit decision list builder.
Schedules integer project audio samples, rational video frame indices,
overlap crossfades, pauses, and loudness normalization into an immutable EditPlan.
"""

from __future__ import annotations

import bisect
import json
import math
import re
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import soundfile as sf

from contracts import (
    EDIT_PLAN_SCHEMA_VERSION,
    EditPlan,
    InventoryIndex,
    PlanSegment,
    PlanTransition,
    RationalTime,
    SourceRecord,
    SpanCandidate,
    TargetGraph,
)
from media_time import cumulative_frame_ceiling, samples_to_seconds

CONTINUE_SPAN_TYPE = "continue"
PAUSE_STYLES = ("freeze", "continue", "off")
RELEASE_GUARD_SEC = 0.030
SEGMENT_END_FADE_SEC = 0.010
CONTINUE_SPEECH_GUARD_SEC = 0.040
CONTINUE_MIN_SEC = 0.080
CONTINUE_SPEECH_DROP_DB = 12.0


def release_tail_samples(
    span: SpanCandidate,
    inventory: InventoryIndex,
    tail_sec: float,
    nominal_end: Optional[int] = None,
) -> int:
    """
    Native samples to extend a cut past its last aligned phone when that phone is followed
    by aligned silence. Forced alignment tends to end word-final consonants (f, s, th, t)
    early, so a hard cut at the boundary clips their release. The tail stays a guard short
    of whatever follows the silence.
    """
    if tail_sec <= 0 or not span.event_ids:
        return 0
    last = inventory.events.get(span.event_ids[-1])
    end_s = nominal_end if nominal_end is not None else span.end_sample
    if last is None or last.kind != "phone" or end_s < last.end_sample - 1:
        return 0
    succ = inventory.events.get(last.successor_id) if last.successor_id else None
    if succ is None or succ.kind != "silence":
        return 0
    src = inventory.sources.get(span.source_content_id)
    sr = src.native_sample_rate if src and src.native_sample_rate else 48000
    room = succ.end_sample - max(succ.start_sample, end_s) - int(round(RELEASE_GUARD_SEC * sr))
    return max(0, min(int(round(tail_sec * sr)), room))


class _SpeechIndex:
    """Aligned phone intervals per source (all speakers), for finding where speech resumes."""

    def __init__(self, inventory: InventoryIndex):
        per: Dict[str, List[Tuple[int, int]]] = {}
        for ev in inventory.events.values():
            if ev.kind == "phone":
                per.setdefault(ev.source_content_id, []).append((ev.start_sample, ev.end_sample))
        self.starts: Dict[str, List[int]] = {}
        self.ends: Dict[str, List[int]] = {}
        for sid, items in per.items():
            items.sort()
            self.starts[sid] = [a for a, _ in items]
            self.ends[sid] = [b for _, b in items]

    def free_after(self, source_id: str, start: int, limit: int) -> int:
        """Samples after `start` (up to `limit`) before the next aligned speech in the source."""
        starts = self.starts.get(source_id)
        if not starts:
            return limit
        ends = self.ends[source_id]
        i = bisect.bisect_right(starts, start)
        for j in range(max(0, i - 30), i):
            if ends[j] > start:
                return 0
        if i < len(starts):
            return max(0, min(limit, starts[i] - start))
        return limit


def _voice_free_samples(src_rec, start: int, limit: int, ref_start: int, ref_end: int) -> int:
    """
    Samples after `start` (up to `limit`) before the dialogue track gets loud again, relative to
    the level of the word just spoken. Catches speech the subtitles never aligned.
    """
    path = getattr(src_rec, "cached_analysis_pcm_path", "") or ""
    if not path or not Path(path).exists() or limit <= 0:
        return limit
    try:
        sr_native = src_rec.native_sample_rate or 48000
        info = sf.info(path)
        fsr = info.samplerate
        to_file = lambda n: int(round(n * fsr / sr_native))  # noqa: E731
        ref, _ = sf.read(path, start=to_file(ref_start), stop=max(to_file(ref_end), to_file(ref_start) + 1), dtype="float32", always_2d=True)
        ref_db = 20 * np.log10(np.sqrt(np.mean(ref.mean(axis=1) ** 2)) + 1e-9)
        y, _ = sf.read(path, start=to_file(start), stop=to_file(start + limit), dtype="float32", always_2d=True)
        y = y.mean(axis=1)
        hop = max(1, int(0.020 * fsr))
        for k in range(0, len(y), hop):
            frame = y[k:k + hop]
            if len(frame) == 0:
                break
            if 20 * np.log10(np.sqrt(np.mean(frame ** 2)) + 1e-9) > ref_db - CONTINUE_SPEECH_DROP_DB:
                return int(round(k * sr_native / fsr))
        return limit
    except Exception:
        return limit


def refine_splice_boundary(
    out_pcm: np.ndarray,
    in_pcm: np.ndarray,
    sr: int,
    max_shift_ms: float = 15.0,
    is_unvoiced: bool = False,
) -> Tuple[int, int]:
    """
    Non-destructively refines boundary alignment between outgoing and incoming audio.

    Voiced splices: Normalized cross-correlation lag search (+/- 15ms) maximizes pitch phase alignment.
    Unvoiced splices / edges: Snaps cut index to nearest upward-going zero-crossing (<= 2ms)
    inside the lowest-energy valley.

    Returns: (shift_out, shift_in) in integer samples.
    """
    if out_pcm is None or in_pcm is None or sr <= 0:
        return (0, 0)

    out_1d = np.asarray(out_pcm, dtype=np.float32)
    in_1d = np.asarray(in_pcm, dtype=np.float32)
    if out_1d.ndim > 1:
        out_1d = np.mean(out_1d, axis=-1)
    if in_1d.ndim > 1:
        in_1d = np.mean(in_1d, axis=-1)
    out_1d = np.atleast_1d(np.squeeze(out_1d))
    in_1d = np.atleast_1d(np.squeeze(in_1d))

    if out_1d.size < 2 or in_1d.size < 2:
        return (0, 0)

    if float(np.max(np.abs(out_1d))) < 1e-7 or float(np.max(np.abs(in_1d))) < 1e-7:
        return (0, 0)

    if is_unvoiced:
        # Snap cut indices to nearest upward-going zero crossing within <= 2 ms
        max_snap = max(1, int(round(0.002 * sr)))  # <= 2 ms (96 samples at 48 kHz)
        energy_win = max(1, int(round(0.005 * sr)))  # 5 ms valley window

        # 1. Outgoing signal: nominal exit cut is at the end of out_1d
        n_out = len(out_1d)
        start_search_out = max(1, n_out - max_snap)
        best_out_idx: Optional[int] = None
        min_out_energy = float("inf")

        for i in range(n_out - 1, start_search_out - 1, -1):
            if out_1d[i - 1] <= 0.0 < out_1d[i]:
                w_st = max(0, i - energy_win // 2)
                w_en = min(n_out, i + energy_win // 2 + 1)
                loc_e = float(np.mean(out_1d[w_st:w_en] ** 2))
                dist = abs(i - (n_out - 1))
                score = loc_e + 1e-4 * (dist / float(max_snap))
                if score < min_out_energy:
                    min_out_energy = score
                    best_out_idx = i

        if best_out_idx is not None:
            shift_out = best_out_idx - n_out
        else:
            fallback_range = out_1d[start_search_out:n_out]
            if len(fallback_range) > 0:
                if np.all(np.abs(fallback_range) == np.abs(fallback_range[-1])):
                    shift_out = 0
                else:
                    dists = np.arange(len(fallback_range) - 1, -1, -1)
                    scores = np.abs(fallback_range) + 1e-4 * (dists / float(max_snap))
                    best_sub = int(np.argmin(scores))
                    shift_out = (start_search_out + best_sub) - n_out
            else:
                shift_out = 0

        # 2. Incoming signal: nominal entry cut is at index 0 of in_1d
        n_in = len(in_1d)
        end_search_in = min(n_in - 1, max_snap)
        best_in_idx: Optional[int] = None
        min_in_energy = float("inf")

        for j in range(1, end_search_in + 1):
            if in_1d[j - 1] <= 0.0 < in_1d[j]:
                w_st = max(0, j - energy_win // 2)
                w_en = min(n_in, j + energy_win // 2 + 1)
                loc_e = float(np.mean(in_1d[w_st:w_en] ** 2))
                dist = j
                score = loc_e + 1e-4 * (dist / float(max_snap))
                if score < min_in_energy:
                    min_in_energy = score
                    best_in_idx = j

        if best_in_idx is not None:
            shift_in = best_in_idx
        else:
            fallback_range_in = in_1d[:end_search_in + 1]
            if len(fallback_range_in) > 0:
                if np.all(np.abs(fallback_range_in) == np.abs(fallback_range_in[0])):
                    shift_in = 0
                else:
                    dists_in = np.arange(len(fallback_range_in))
                    scores_in = np.abs(fallback_range_in) + 1e-4 * (dists_in / float(max_snap))
                    shift_in = int(np.argmin(scores_in))
            else:
                shift_in = 0

        shift_out = max(-max_snap, min(0, shift_out))
        shift_in = max(0, min(max_snap, shift_in))
        return (int(shift_out), int(shift_in))

    else:
        # Voiced mode: Normalized cross-correlation lag search on incoming audio (WSOLA-style optimal coupling)
        # Search non-negative lag offsets tau in [0, max_lag] aligning in_1d to outgoing reference out_1d.
        max_shift = int(round(max_shift_ms * sr / 1000.0))
        n_out = len(out_1d)
        n_in = len(in_1d)

        max_lag = min(max_shift, max(0, n_in - 2))
        if max_lag < 1:
            return (0, 0)

        # Snap outgoing cut to nearest upward zero-crossing within <= 2ms of nominal end
        max_snap = max(1, int(round(0.002 * sr)))
        shift_out = 0
        for i in range(n_out - 1, max(1, n_out - max_snap) - 1, -1):
            if out_1d[i - 1] <= 0.0 < out_1d[i]:
                shift_out = i - n_out  # <= 0
                break
        out_trimmed = out_1d[: n_out + shift_out] if shift_out < 0 else out_1d
        n_out_trimmed = len(out_trimmed)

        l_eval = min(n_out_trimmed, n_in - max_lag)
        if l_eval < 16:
            l_eval = min(n_out_trimmed, n_in, max(8, min(n_out_trimmed, n_in) // 2))
            max_lag = min(max_lag, n_in - l_eval)

        if max_lag < 1 or l_eval < 4:
            return (int(shift_out), 0)

        # Align incoming phase over crossfade window l_eval against terminal window of outgoing audio
        ref_out = out_trimmed[-l_eval:]
        ref_in = in_1d[:l_eval]
        denom_0 = float(np.sqrt((np.sum(ref_out ** 2) * np.sum(ref_in ** 2)) + 1e-12))
        r_0 = float(np.sum(ref_out * ref_in) / denom_0) if denom_0 > 0 else 0.0

        best_r = r_0
        best_lag = 0

        for lag in range(0, max_lag + 1):
            u = ref_out
            v = in_1d[lag : lag + l_eval]

            if len(u) != l_eval or len(v) != l_eval:
                continue

            dot = float(np.sum(u * v))
            denom = float(np.sqrt((np.sum(u ** 2) * np.sum(v ** 2)) + 1e-12))
            r = dot / denom if denom > 0 else -1.0

            if r > best_r:
                best_r = r
                best_lag = lag

        shift_in = best_lag
        return (int(shift_out), int(shift_in))


def _source_sample_to_video_pts(sample_idx: int, src_rec: SourceRecord) -> float:
    """
    Computes sample-accurate video PTS accounting for container codec delay and stream origins.
    """
    sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else 48000
    codec_delay = getattr(src_rec, "codec_delay_samples", 0) or 0
    effective_sample = max(0, sample_idx - codec_delay)
    audio_sec = samples_to_seconds(effective_sample, sr) + float(getattr(src_rec, "decoded_audio_origin_sec", 0.0) or 0.0)
    video_origin = float(getattr(src_rec, "decoded_video_origin_sec", 0.0) or 0.0)
    return max(0.0, audio_sec - video_origin)


def _refine_spans_in_native_audio(
    selected_spans: List[SpanCandidate],
    inventory: InventoryIndex,
    tail_sec: float = 0.0,
    graph: Optional[TargetGraph] = None,
    rendering_mode: str = "source-derived",
    overlap_duration_ms: float = 14.0,
) -> Tuple[List[int], List[int]]:
    """
    Refines boundary sample offsets in native source audio prior to EditPlan construction.
    Gracefully skips refinement if source media or cached PCM files do not exist on disk (mock-safe).
    """
    span_starts = [s.start_sample for s in selected_spans]
    span_ends = [s.end_sample for s in selected_spans]

    # Quick check if files exist on disk for all referenced sources
    can_refine = True
    src_paths: Dict[str, str] = {}
    for span in selected_spans:
        src_rec = inventory.sources.get(span.source_content_id)
        if not src_rec:
            can_refine = False
            break
        p = src_rec.cached_pcm_path or src_rec.source_path
        if not p or not Path(p).exists():
            can_refine = False
            break
        src_paths[span.source_content_id] = p

    if not can_refine or len(selected_spans) == 0:
        return span_starts, span_ends

    # Apply variant samples when flagged during candidate retrieval
    for i, span in enumerate(selected_spans):
        src_rec = inventory.sources[span.source_content_id]
        native_sr = src_rec.native_sample_rate or 48000
        min_len = int(round(0.030 * native_sr))

        entry_v = getattr(span, "entry_variant_samples", 0) or 0
        exit_v = getattr(span, "exit_variant_samples", 0) or 0

        # Only apply entry/exit collar trimming on non-continuations
        is_cont_pred = False
        if i > 0:
            pred = selected_spans[i - 1]
            if pred.source_content_id == span.source_content_id and pred.end_sample == span.start_sample:
                if not (pred.event_ids and span.event_ids):
                    is_cont_pred = True
                else:
                    p_last_ev = inventory.events.get(pred.event_ids[-1])
                    s_first_ev = inventory.events.get(span.event_ids[0])
                    if p_last_ev and s_first_ev and p_last_ev.successor_id == s_first_ev.event_id:
                        is_cont_pred = True

        is_cont_succ = False
        if i < len(selected_spans) - 1:
            succ = selected_spans[i + 1]
            if succ.source_content_id == span.source_content_id and span.end_sample == succ.start_sample:
                if not (span.event_ids and succ.event_ids):
                    is_cont_succ = True
                else:
                    c_last_ev = inventory.events.get(span.event_ids[-1])
                    s_first_ev = inventory.events.get(succ.event_ids[0])
                    if c_last_ev and s_first_ev and c_last_ev.successor_id == s_first_ev.event_id:
                        is_cont_succ = True

        new_st = span_starts[i] + (entry_v if not is_cont_pred else 0)
        new_en = span_ends[i] - (exit_v if not is_cont_succ else 0)
        if new_en - new_st >= min_len:
            span_starts[i] = new_st
            span_ends[i] = new_en

    # Refine outer utterance leading edge (span_starts[0]) to upward zero-crossing
    first_span = selected_spans[0]
    p_first = src_paths.get(first_span.source_content_id)
    if p_first:
        try:
            first_sr = inventory.sources[first_span.source_content_id].native_sample_rate or 48000
            st_0 = span_starts[0]
            info_first = sf.info(p_first)
            en_0 = min(info_first.frames, span_ends[0], st_0 + int(round(0.025 * first_sr)))
            if en_0 > st_0:
                head_audio, _ = sf.read(p_first, start=st_0, stop=en_0)
                if len(head_audio) >= 16:
                    _, shift_head = refine_splice_boundary(
                        np.zeros(len(head_audio), dtype=np.float32), head_audio, sr=first_sr, max_shift_ms=2.0, is_unvoiced=True
                    )
                    min_len_0 = int(round(0.030 * first_sr))
                    if shift_head > 0 and (span_ends[0] - (span_starts[0] + shift_head)) >= min_len_0:
                        span_starts[0] += shift_head
        except Exception:
            pass

    # Perform boundary refinement between adjacent cuts
    stop_fric = ("P", "T", "K", "B", "D", "G", "S", "SH", "F", "TH", "CH")
    for i in range(len(selected_spans) - 1):
        curr_span = selected_spans[i]
        next_span = selected_spans[i + 1]

        # Natural continuations require no boundary modification
        if curr_span.source_content_id == next_span.source_content_id and span_ends[i] == span_starts[i + 1]:
            if not (curr_span.event_ids and next_span.event_ids):
                continue
            c_ev = inventory.events.get(curr_span.event_ids[-1])
            n_ev = inventory.events.get(next_span.event_ids[0])
            if c_ev and n_ev and c_ev.successor_id == n_ev.event_id:
                continue

        p_curr = src_paths.get(curr_span.source_content_id)
        p_next = src_paths.get(next_span.source_content_id)
        if not p_curr or not p_next:
            continue

        try:
            curr_sr = inventory.sources[curr_span.source_content_id].native_sample_rate or 48000
            next_sr = inventory.sources[next_span.source_content_id].native_sample_rate or 48000

            # Detect whether this boundary is a pause or cuts-only splice
            has_pause = False
            if graph and hasattr(graph, "nodes") and curr_span.covered_nodes and next_span.covered_nodes:
                c_last_node = graph.nodes[curr_span.covered_nodes[-1]]
                n_first_node = graph.nodes[next_span.covered_nodes[0]]
                if c_last_node.token_index != n_first_node.token_index and c_last_node.is_word_final:
                    explicit = c_last_node.token_index in getattr(graph, "explicit_gaps", {})
                    p_pause = graph.punctuation_pauses.get(c_last_node.token_index, 0.0)
                    if explicit or p_pause > 0.0:
                        has_pause = True

            # If separated by pause or rendering cuts-only, use unvoiced zero-crossing snaps on both edges
            is_unv = (rendering_mode == "cuts-only") or has_pause
            if not is_unv:
                last_p = curr_span.realized_phones[-1].rstrip("012") if getattr(curr_span, "realized_phones", None) else ""
                first_p = next_span.realized_phones[0].rstrip("012") if getattr(next_span, "realized_phones", None) else ""
                if not last_p and graph and curr_span.covered_nodes and curr_span.covered_nodes[-1] < len(graph.nodes):
                    last_p = getattr(graph.nodes[curr_span.covered_nodes[-1]], "base_phone", "") or graph.nodes[curr_span.covered_nodes[-1]].phone.rstrip("012")
                if not first_p and graph and next_span.covered_nodes and next_span.covered_nodes[0] < len(graph.nodes):
                    first_p = getattr(graph.nodes[next_span.covered_nodes[0]], "base_phone", "") or graph.nodes[next_span.covered_nodes[0]].phone.rstrip("012")
                if last_p in stop_fric or last_p == "sil" or first_p in stop_fric or first_p == "sil":
                    is_unv = True

            # Window lengths: align evaluation window to crossfade duration + lag range
            win_curr = int(round(max(0.016, (overlap_duration_ms + 2.0) / 1000.0) * curr_sr))
            win_next = int(round(max(0.029, (overlap_duration_ms + 15.0) / 1000.0) * next_sr))

            curr_en = span_ends[i]
            next_st = span_starts[i + 1]

            info_next = sf.info(p_next)
            read_st_curr = max(span_starts[i], curr_en - win_curr)
            read_en_next = min(info_next.frames, span_ends[i + 1], next_st + win_next)

            if curr_en <= read_st_curr or read_en_next <= next_st:
                continue

            audio_curr, _ = sf.read(p_curr, start=read_st_curr, stop=curr_en)
            audio_next, _ = sf.read(p_next, start=next_st, stop=read_en_next)

            if len(audio_curr) < 16 or len(audio_next) < 16:
                continue

            # Resample audio_next if native sample rates differ
            if curr_sr != next_sr:
                import scipy.signal
                g_sr = math.gcd(curr_sr, next_sr)
                audio_next_resampled = scipy.signal.resample_poly(audio_next, curr_sr // g_sr, next_sr // g_sr, axis=0)
                shift_out, shift_in_resampled = refine_splice_boundary(
                    audio_curr, audio_next_resampled, sr=curr_sr, max_shift_ms=15.0, is_unvoiced=is_unv
                )
                shift_in = int(round(shift_in_resampled * float(next_sr) / float(curr_sr)))
            else:
                shift_out, shift_in = refine_splice_boundary(
                    audio_curr, audio_next, sr=curr_sr, max_shift_ms=15.0, is_unvoiced=is_unv
                )

            # Apply shifts while respecting minimum segment lengths
            min_len_curr = int(round(0.030 * curr_sr))
            min_len_next = int(round(0.030 * next_sr))

            if shift_out < 0 and (span_ends[i] + shift_out - span_starts[i]) >= min_len_curr:
                span_ends[i] += shift_out
            if shift_in > 0 and (span_ends[i + 1] - (span_starts[i + 1] + shift_in)) >= min_len_next:
                span_starts[i + 1] += shift_in
        except Exception:
            continue

    # Refine outer utterance trailing edge (span_ends[-1]) to upward zero-crossing
    last_span = selected_spans[-1]
    p_last = src_paths.get(last_span.source_content_id)
    if p_last:
        try:
            last_sr = inventory.sources[last_span.source_content_id].native_sample_rate or 48000
            en_last = span_ends[-1]
            st_last = max(span_starts[-1], en_last - int(round(0.025 * last_sr)))
            if en_last > st_last:
                tail_audio, _ = sf.read(p_last, start=st_last, stop=en_last)
                if len(tail_audio) >= 16:
                    shift_tail, _ = refine_splice_boundary(
                        tail_audio, np.zeros(len(tail_audio), dtype=np.float32), sr=last_sr, max_shift_ms=2.0, is_unvoiced=True
                    )
                    min_len_last = int(round(0.030 * last_sr))
                    if shift_tail < 0 and (span_ends[-1] + shift_tail - span_starts[-1]) >= min_len_last:
                        span_ends[-1] += shift_tail
        except Exception:
            pass

    return span_starts, span_ends


def build_edit_plan(
    selected_spans: List[SpanCandidate],
    graph: TargetGraph,
    inventory: InventoryIndex,
    project_sample_rate: int = 48000,
    video_fps: Optional[RationalTime] = None,
    rendering_mode: Literal["source-derived", "cuts-only"] = "source-derived",
    overlap_duration_ms: float = 14.0,
    lead_in_sec: float = 0.0,
    pause_style: str = "freeze",
    release_tail_sec: float = 0.0,
    finalize: bool = True,
    audio_track: str = "recombined",
) -> EditPlan:
    """
    Constructs an immutable, sample-accurate EditPlan from selected spans.

    pause_style controls punctuation pauses: "freeze" holds the last frame over silence,
    "continue" lets the source clip play on (picture and its background audio) while the
    source has no speech, then holds the frame for any remainder, "off" drops punctuation
    pauses (explicit <gap:...> tags are kept, frozen). release_tail_sec extends a cut into
    following aligned silence so word-final consonants are not clipped (0 disables).
    """
    if pause_style not in PAUSE_STYLES:
        raise ValueError(f"pause_style must be one of {PAUSE_STYLES}, got {pause_style!r}")
    if video_fps is None:
        video_fps = RationalTime(30000, 1001)

    if not selected_spans:
        raise ValueError("Cannot build EditPlan with empty spans")

    plan_id = f"plan_{uuid.uuid4().hex[:12]}"
    segments: List[PlanSegment] = []
    transitions: List[PlanTransition] = []

    current_output_sample = 0
    overlap_samples_const = int(round(overlap_duration_ms * project_sample_rate / 1000.0))

    seg_idx = 0

    tail_sec = release_tail_sec if rendering_mode != "cuts-only" else 0.0

    # Step 0: Plan-Time Boundary Refinement across native audio windows (mock-safe)
    refined_starts, refined_ends = _refine_spans_in_native_audio(
        selected_spans,
        inventory,
        tail_sec=tail_sec,
        graph=graph,
        rendering_mode=rendering_mode,
        overlap_duration_ms=overlap_duration_ms,
    )

    # Check for leading gap (boundary index -1)
    leading_gap = getattr(graph, "explicit_gaps", {}).get(-1, lead_in_sec)
    if leading_gap > 0.0 and selected_spans:
        first_span = selected_spans[0]
        first_src_rec = inventory.sources[first_span.source_content_id]
        pause_samples = int(round(leading_gap * project_sample_rate))
        p_start = current_output_sample
        p_end = p_start + pause_samples

        p_start_frame = cumulative_frame_ceiling(p_start, project_sample_rate, video_fps)
        p_end_frame = cumulative_frame_ceiling(p_end, project_sample_rate, video_fps)
        p_source_pts = _source_sample_to_video_pts(refined_starts[0], first_src_rec)

        pause_seg = PlanSegment(
            segment_index=seg_idx,
            source_content_id="silence",
            speaker_id=first_span.speaker_id,
            source_start_sample=0,
            source_end_sample=pause_samples,
            output_start_sample=p_start,
            output_end_sample=p_end,
            target_text="[pause]",
            target_phones=["sil"],
            span_type="pause",
            source_video_path=first_src_rec.source_path,
            is_silence=True,
            freeze_video=True,
            video_start_frame=p_start_frame,
            video_frame_count=p_end_frame - p_start_frame,
            video_source_pts_sec=p_source_pts,
        )
        segments.append(pause_seg)
        seg_idx += 1
        current_output_sample = p_end

        transitions.append(PlanTransition(
            transition_index=len(transitions),
            from_segment_index=pause_seg.segment_index,
            to_segment_index=seg_idx,
            transition_type="hard_splice",
            overlap_samples=0,
        ))

    speech_index: Optional[_SpeechIndex] = None
    end_fade = int(round(SEGMENT_END_FADE_SEC * project_sample_rate))

    def pause_seconds(token_index: int) -> float:
        explicit_gaps = getattr(graph, "explicit_gaps", {})
        if token_index in explicit_gaps:
            return explicit_gaps[token_index]
        if rendering_mode == "cuts-only" or pause_style == "off":
            return 0.0
        return graph.punctuation_pauses.get(token_index, 0.0)

    def emit_pause(prev_seg: PlanSegment, pause_sec: float, explicit: bool) -> PlanSegment:
        """Appends pause segment(s) after prev_seg; returns the last one."""
        nonlocal seg_idx, current_output_sample, speech_index
        pause_samples = int(round(pause_sec * project_sample_rate))
        last_seg = prev_seg

        src_rec = inventory.sources.get(prev_seg.source_content_id)
        if prev_seg.is_silence:
            freeze_pts = prev_seg.video_source_pts_sec
        elif src_rec is not None:
            freeze_pts = _source_sample_to_video_pts(prev_seg.source_end_sample, src_rec)
        else:
            freeze_pts = prev_seg.video_source_pts_sec + (float(prev_seg.duration_samples) / float(project_sample_rate))

        if pause_style == "continue" and not explicit and not prev_seg.is_silence and src_rec is not None:
            sr = src_rec.native_sample_rate or project_sample_rate
            want = int(round(pause_sec * sr))
            start = prev_seg.source_end_sample
            if speech_index is None:
                speech_index = _SpeechIndex(inventory)
            guard = int(round(CONTINUE_SPEECH_GUARD_SEC * sr))
            free = max(0, speech_index.free_after(prev_seg.source_content_id, start, want + guard) - guard)
            free = min(free, _voice_free_samples(src_rec, start, want, prev_seg.source_start_sample, prev_seg.source_end_sample))
            total_src = int(round((src_rec.audio_duration_sec or 0.0) * sr))
            if total_src > 0:
                free = min(free, max(0, total_src - start))
            cont = min(free, want)
            if cont >= int(round(CONTINUE_MIN_SEC * sr)):
                cont_proj = int(round(cont * project_sample_rate / float(sr)))
                c_start = current_output_sample
                c_end = c_start + cont_proj
                c_seg = PlanSegment(
                    segment_index=seg_idx,
                    source_content_id=prev_seg.source_content_id,
                    speaker_id=prev_seg.speaker_id,
                    source_start_sample=start,
                    source_end_sample=start + cont,
                    output_start_sample=c_start,
                    output_end_sample=c_end,
                    target_text="[pause]",
                    target_phones=["sil"],
                    span_type=CONTINUE_SPAN_TYPE,
                    source_video_path=prev_seg.source_video_path,
                    fade_out_samples=min(end_fade, cont_proj),
                    video_start_frame=cumulative_frame_ceiling(c_start, project_sample_rate, video_fps),
                    video_frame_count=0,
                    video_source_pts_sec=_source_sample_to_video_pts(start, src_rec),
                )
                segments.append(c_seg)
                transitions.append(PlanTransition(
                    transition_index=len(transitions),
                    from_segment_index=prev_seg.segment_index,
                    to_segment_index=c_seg.segment_index,
                    transition_type="continuation",
                    overlap_samples=0,
                ))
                seg_idx += 1
                current_output_sample = c_end
                last_seg = c_seg
                pause_samples = max(0, pause_samples - cont_proj)
                freeze_pts = _source_sample_to_video_pts(start + cont, src_rec)

        if pause_samples <= 0 and last_seg is not prev_seg:
            return last_seg
        p_start = current_output_sample
        p_end = p_start + pause_samples
        p_start_frame = cumulative_frame_ceiling(p_start, project_sample_rate, video_fps)
        p_end_frame = cumulative_frame_ceiling(p_end, project_sample_rate, video_fps)
        pause_seg = PlanSegment(
            segment_index=seg_idx,
            source_content_id="silence",
            speaker_id=prev_seg.speaker_id,
            source_start_sample=0,
            source_end_sample=pause_samples,
            output_start_sample=p_start,
            output_end_sample=p_end,
            target_text="[pause]",
            target_phones=["sil"],
            span_type="pause",
            source_video_path=prev_seg.source_video_path,
            is_silence=True,
            freeze_video=True,
            video_start_frame=p_start_frame,
            video_frame_count=p_end_frame - p_start_frame,  # may be 0 for <gap:0ms>
            video_source_pts_sec=freeze_pts,
        )
        segments.append(pause_seg)
        transitions.append(PlanTransition(
            transition_index=len(transitions),
            from_segment_index=last_seg.segment_index,
            to_segment_index=pause_seg.segment_index,
            transition_type="pause",
            overlap_samples=0,
        ))
        seg_idx += 1
        current_output_sample = p_end
        return pause_seg

    for i, span in enumerate(selected_spans):
        src_rec = inventory.sources[span.source_content_id]
        native_sr = src_rec.native_sample_rate
        next_span = selected_spans[i + 1] if i < len(selected_spans) - 1 else None

        span_start = refined_starts[i]
        span_nominal_end = refined_ends[i]

        # Determine if the next span is a natural continuation of this one
        is_cont = False
        if next_span is not None and span.source_content_id == next_span.source_content_id:
            if span.end_sample == next_span.start_sample and span_nominal_end == refined_starts[i + 1]:
                if not (span.event_ids and next_span.event_ids):
                    is_cont = True
                else:
                    last_ev = inventory.events.get(span.event_ids[-1])
                    first_ev = inventory.events.get(next_span.event_ids[0])
                    if last_ev and first_ev and last_ev.successor_id == first_ev.event_id:
                        is_cont = True

        tail = 0 if is_cont else release_tail_samples(span, inventory, tail_sec, nominal_end=span_nominal_end)
        span_end = span_nominal_end + tail

        # Duration in native samples
        native_duration = span_end - span_start
        # Resampled duration at project sample rate
        resampled_duration = int(round(float(native_duration) * project_sample_rate / float(native_sr)))

        seg_start_sample = current_output_sample
        seg_end_sample = seg_start_sample + resampled_duration

        # Compute video frame bounds using cumulative ceiling quantizer (drift-free)
        start_frame = cumulative_frame_ceiling(seg_start_sample, project_sample_rate, video_fps)
        end_frame = cumulative_frame_ceiling(seg_end_sample, project_sample_rate, video_fps)
        frame_count = end_frame - start_frame  # may be 0 for very short clips
        source_pts = _source_sample_to_video_pts(span_start, src_rec)

        target_phones = [graph.nodes[nid].phone for nid in span.covered_nodes if nid < len(graph.nodes)]

        realized = getattr(span, "realized_phones", None)
        if not realized and inventory is not None:
            realized = [inventory.events[eid].label for eid in span.event_ids if eid in inventory.events and inventory.events[eid].kind == "phone"]
        if not realized:
            realized = list(target_phones)

        target_words_covered: List[str] = []
        if graph and hasattr(graph, "nodes"):
            for nid in span.covered_nodes:
                if nid < len(graph.nodes):
                    w_txt = getattr(graph.nodes[nid], "word_text", "")
                    if w_txt and (not target_words_covered or target_words_covered[-1] != w_txt):
                        target_words_covered.append(w_txt)
        target_label = " ".join(target_words_covered) if target_words_covered else span.matched_word

        seg = PlanSegment(
            segment_index=seg_idx,
            source_content_id=span.source_content_id,
            speaker_id=span.speaker_id,
            source_start_sample=span_start,
            source_end_sample=span_end,
            output_start_sample=seg_start_sample,
            output_end_sample=seg_end_sample,
            target_text=target_label,
            target_phones=target_phones,
            span_type=span.span_type,
            source_video_path=src_rec.source_path,
            fade_out_samples=min(end_fade, resampled_duration // 4) if tail > 0 else 0,
            video_start_frame=start_frame,
            video_frame_count=frame_count,
            video_source_pts_sec=source_pts,
            realized_phones=realized,
            entry_variant_samples=getattr(span, "entry_variant_samples", 0),
            exit_variant_samples=getattr(span, "exit_variant_samples", 0),
        )
        segments.append(seg)
        seg_idx += 1

        # Advance output timeline
        current_output_sample = seg_end_sample

        # Check transition to next segment
        if next_span is not None:
            # Check if there is an explicit pause between these words
            last_node = graph.nodes[span.covered_nodes[-1]]
            next_first_node = graph.nodes[next_span.covered_nodes[0]]

            pause_sec = 0.0
            explicit = False
            if last_node.token_index != next_first_node.token_index and last_node.is_word_final:
                explicit = last_node.token_index in getattr(graph, "explicit_gaps", {})
                pause_sec = pause_seconds(last_node.token_index)

            if pause_sec > 0.0:
                last_pause = emit_pause(seg, pause_sec, explicit)
                transitions.append(PlanTransition(
                    transition_index=len(transitions),
                    from_segment_index=last_pause.segment_index,
                    to_segment_index=seg_idx,  # points to next segment
                    transition_type="hard_splice",
                    overlap_samples=0,
                ))
            else:
                if is_cont:
                    transitions.append(PlanTransition(
                        transition_index=len(transitions),
                        from_segment_index=seg.segment_index,
                        to_segment_index=seg_idx,
                        transition_type="continuation",
                        overlap_samples=0,
                    ))
                else:
                    if rendering_mode == "source-derived":
                        next_src_rec = inventory.sources.get(next_span.source_content_id)
                        next_sr = next_src_rec.native_sample_rate if next_src_rec and next_src_rec.native_sample_rate > 0 else project_sample_rate
                        next_resampled_dur = int(round(float(refined_ends[i + 1] - refined_starts[i + 1]) * project_sample_rate / float(next_sr)))
                        max_overlap = min(resampled_duration // 5, next_resampled_dur // 5)
                        overlap = min(overlap_samples_const, max_overlap)
                        current_output_sample -= overlap
                        transitions.append(PlanTransition(
                            transition_index=len(transitions),
                            from_segment_index=seg.segment_index,
                            to_segment_index=seg_idx,
                            transition_type="overlap",
                            overlap_samples=overlap,
                            fade_samples=overlap,
                        ))
                    else:  # cuts-only mode
                        transitions.append(PlanTransition(
                            transition_index=len(transitions),
                            from_segment_index=seg.segment_index,
                            to_segment_index=seg_idx,
                            transition_type="hard_splice",
                            overlap_samples=0,
                        ))

    # Check for trailing gap (boundary after last token)
    if selected_spans and segments:
        last_node = graph.nodes[selected_spans[-1].covered_nodes[-1]]
        trailing_gap = pause_seconds(last_node.token_index)
        if trailing_gap > 0.0:
            explicit = last_node.token_index in getattr(graph, "explicit_gaps", {})
            emit_pause(segments[-1], trailing_gap, explicit)

    total_output_samples = current_output_sample
    total_video_frames = cumulative_frame_ceiling(total_output_samples, project_sample_rate, video_fps)

    # Deterministic picture-switch policy during audio overlap:
    # Partition cumulative timeline at crossfade midpoint so sum(video_frame_count) == total_video_frames
    if segments:
        overlap_by_from_seg = {}
        for t in transitions:
            if t.transition_type == "overlap":
                overlap_by_from_seg[t.from_segment_index] = t.overlap_samples

        v_display_starts = [0] * len(segments)
        v_display_ends = [0] * len(segments)

        v_display_starts[0] = 0
        for k in range(len(segments) - 1):
            curr_seg = segments[k]
            next_seg = segments[k + 1]
            ov = overlap_by_from_seg.get(curr_seg.segment_index, 0)
            if ov > 0:
                midpoint = next_seg.output_start_sample + (ov // 2)
                v_display_ends[k] = midpoint
                v_display_starts[k + 1] = midpoint
            else:
                v_display_ends[k] = curr_seg.output_end_sample
                v_display_starts[k + 1] = next_seg.output_start_sample

        v_display_ends[-1] = total_output_samples

        for k, seg in enumerate(segments):
            v_start = v_display_starts[k]
            v_end = v_display_ends[k]
            start_f = cumulative_frame_ceiling(v_start, project_sample_rate, video_fps)
            end_f = cumulative_frame_ceiling(v_end, project_sample_rate, video_fps)
            seg.video_start_frame = start_f
            seg.video_frame_count = max(0, end_f - start_f)

            # Adjust video_source_pts_sec for audio overlap offset if not silence
            if not seg.is_silence:
                offset_samples = v_start - seg.output_start_sample
                if offset_samples > 0:
                    seg.video_source_pts_sec += (offset_samples / float(project_sample_rate))

    plan = EditPlan(
        schema_version=EDIT_PLAN_SCHEMA_VERSION,
        plan_id=plan_id,
        target_text=graph.raw_text,
        speaker_id=selected_spans[0].speaker_id,
        selector_mode="span",
        rendering_mode="source-derived" if rendering_mode == "source-derived" else "cuts-only",
        project_sample_rate=project_sample_rate,
        video_fps=video_fps,
        total_output_samples=total_output_samples,
        total_video_frames=total_video_frames,
        segments=segments,
        transitions=transitions,
        sources=inventory.sources,
        metadata={"num_cuts": len(transitions)},
    )
    plan.validate()
    if finalize:
        finalize_edit_plan(plan, inventory, normalize_loudness=(rendering_mode != "cuts-only"), audio_track=audio_track)
    return plan


def finalize_edit_plan(
    plan: EditPlan,
    inventory: Optional[InventoryIndex] = None,
    normalize_loudness: bool = True,
    target_i: float = -16.0,
    target_tp: float = -1.0,
    target_lra: float = 11.0,
    audio_track: str = "recombined",
) -> EditPlan:
    """
    Freezes all DSP decisions into the EditPlan before rendering:
    1. Computes float gain_db for all non-silence segments over consolidated natural runs.
       In cuts-only mode or if normalize_loudness is False, gain_db is 0.0.
    2. Measures EBU R128 loudness on an unpublished analysis master and records the
       analysis stats and target parameters in plan metadata.
    3. Guarantees that rendering will not need to make DSP decisions or mutate the plan.
    """
    is_cuts_only = (plan.rendering_mode == "cuts-only")

    # Step 1: Consolidate natural runs and compute gain_db for every segment
    runs: List[List[PlanSegment]] = []
    curr_run: List[PlanSegment] = []

    for seg in plan.segments:
        if seg.is_silence or seg.span_type == CONTINUE_SPAN_TYPE:
            # Continued pauses (the clip playing on after a word) are not speech: they take
            # the gain of the word they continue (set below), never their own leveling.
            seg.gain_db = 0.0
            if curr_run:
                runs.append(curr_run)
                curr_run = []
            continue

        if not curr_run:
            curr_run.append(seg)
        else:
            prev_seg = curr_run[-1]
            if (
                prev_seg.source_content_id == seg.source_content_id
                and prev_seg.source_end_sample == seg.source_start_sample
            ):
                curr_run.append(seg)
            else:
                runs.append(curr_run)
                curr_run = [seg]

    if curr_run:
        runs.append(curr_run)

    segment_rms_db: Dict[int, float] = {}

    for run in runs:
        if is_cuts_only or not normalize_loudness:
            for s in run:
                s.gain_db = 0.0
            continue

        first_seg = run[0]
        src_rec = plan.sources.get(first_seg.source_content_id) if plan.sources else None
        src_path = None
        if src_rec:
            # When audio_track is vocals, recombined, both, or all, prefer vocal stem if available
            effective_rms_track = "vocals" if audio_track in ("both", "all") else audio_track
            if effective_rms_track in ("vocals", "isolated", "recombined") and getattr(src_rec, "cached_vocal_pcm_path", None):
                if Path(src_rec.cached_vocal_pcm_path).exists():
                    src_path = src_rec.cached_vocal_pcm_path
            if not src_path:
                if src_rec.cached_pcm_path and Path(src_rec.cached_pcm_path).exists():
                    src_path = src_rec.cached_pcm_path
                elif src_rec.source_path and Path(src_rec.source_path).exists():
                    src_path = src_rec.source_path
        elif first_seg.source_video_path and Path(first_seg.source_video_path).exists():
            src_path = first_seg.source_video_path

        run_audio_chunks = []
        if src_path:
            try:
                native_rate = int(sf.info(str(src_path)).samplerate)
                n_frames = int(sf.info(str(src_path)).frames)
                rec_sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else native_rate
                for s in run:
                    if native_rate != rec_sr:
                        st = int(round(s.source_start_sample * native_rate / rec_sr))
                        en = int(round(s.source_end_sample * native_rate / rec_sr))
                    else:
                        st = s.source_start_sample
                        en = s.source_end_sample
                    st = max(0, min(n_frames, st))
                    en = max(0, min(n_frames, en))
                    # Ranged read: long sources (whole movies) are never loaded entirely
                    chunk, _ = sf.read(str(src_path), start=st, stop=en) if en > st else (np.zeros(0), native_rate)
                    if native_rate != plan.project_sample_rate and len(chunk) > 0:
                        import math
                        import scipy.signal
                        g = math.gcd(int(plan.project_sample_rate), int(native_rate))
                        up = int(plan.project_sample_rate // g)
                        down = int(native_rate // g)
                        chunk = scipy.signal.resample_poly(chunk, up, down, axis=0)
                    run_audio_chunks.append(chunk)
            except Exception:
                pass

        if run_audio_chunks:
            run_audio = np.concatenate(run_audio_chunks, axis=0)
            if run_audio.ndim > 1:
                run_audio = np.mean(run_audio, axis=1)
            active_mask = np.abs(run_audio) > 1e-4
            if np.sum(active_mask) > 50:
                rms = float(np.sqrt(np.mean(run_audio[active_mask] ** 2)))
            else:
                rms = float(np.sqrt(np.mean(run_audio ** 2))) if len(run_audio) > 0 else 0.0

            if rms > 1e-4:
                rms_db = 20.0 * np.log10(rms)
                auto_gain_db = -20.0 - rms_db
                is_subword_or_phone = any(s.span_type in ("subword", "phone") for s in run)
                limit_db = 12.0 if is_subword_or_phone else 9.0
                gain_val = float(max(-limit_db, min(limit_db, auto_gain_db)))
            else:
                rms_db = -20.0
                gain_val = 0.0
        else:
            rms_db = -20.0
            gain_val = 0.0

        for s in run:
            s.gain_db = gain_val
            segment_rms_db[s.segment_index] = rms_db

    # Ensure all segments have a valid float gain_db
    for s in plan.segments:
        if s.gain_db is None:
            s.gain_db = 0.0

    # Step 1b: Dialogue acoustic loudness smoothing and intra-word consistency
    # Limit rapid loudness transitions between consecutive spliced phonetic fragments (subwords/phones)
    # to <= 3.0 dB to prevent jarring volume jumps between consonants and vowels, while allowing
    # full multi-clip leveling range between long independent dialogue phrases.
    if not is_cuts_only and normalize_loudness:
        speech_indices = [idx for idx, s in enumerate(plan.segments) if not s.is_silence and s.span_type != CONTINUE_SPAN_TYPE]
        threshold_samples = int(round(0.250 * plan.project_sample_rate))
        if len(speech_indices) > 1:
            for i in range(len(speech_indices) - 1):
                idx_curr = speech_indices[i]
                idx_next = speech_indices[i + 1]
                s_c = plan.segments[idx_curr]
                s_n = plan.segments[idx_next]
                is_fragment = (
                    s_c.target_text == s_n.target_text
                    or s_c.span_type in ("subword", "phone")
                    or s_n.span_type in ("subword", "phone")
                    or s_c.duration_samples < threshold_samples
                    or s_n.duration_samples < threshold_samples
                )
                max_jump = 3.0 if (is_fragment and idx_next == idx_curr + 1) else 18.0
                level_curr = segment_rms_db.get(s_c.segment_index, -20.0) + (s_c.gain_db or 0.0)
                level_next = segment_rms_db.get(s_n.segment_index, -20.0) + (s_n.gain_db or 0.0)
                if level_next > level_curr + max_jump:
                    excess = level_next - (level_curr + max_jump)
                    s_n.gain_db = float(max(-12.0, (s_n.gain_db or 0.0) - excess))
                elif level_next < level_curr - max_jump:
                    deficit = (level_curr - max_jump) - level_next
                    s_n.gain_db = float(min(12.0, (s_n.gain_db or 0.0) + deficit))

            for i in range(len(speech_indices) - 1, 0, -1):
                idx_curr = speech_indices[i]
                idx_prev = speech_indices[i - 1]
                s_c = plan.segments[idx_curr]
                s_p = plan.segments[idx_prev]
                is_fragment = (
                    s_c.target_text == s_p.target_text
                    or s_c.span_type in ("subword", "phone")
                    or s_p.span_type in ("subword", "phone")
                    or s_c.duration_samples < threshold_samples
                    or s_p.duration_samples < threshold_samples
                )
                max_jump = 3.0 if (is_fragment and idx_curr == idx_prev + 1) else 18.0
                level_curr = segment_rms_db.get(s_c.segment_index, -20.0) + (s_c.gain_db or 0.0)
                level_prev = segment_rms_db.get(s_p.segment_index, -20.0) + (s_p.gain_db or 0.0)
                if level_prev > level_curr + max_jump:
                    excess = level_prev - (level_curr + max_jump)
                    s_p.gain_db = float(max(-12.0, (s_p.gain_db or 0.0) - excess))
                elif level_prev < level_curr - max_jump:
                    deficit = (level_curr - max_jump) - level_prev
                    s_p.gain_db = float(min(12.0, (s_p.gain_db or 0.0) + deficit))

        # Step 1c: Sentence-level speech leveling anchor
        # Prevent any speech segment inside a sentence from dropping > 3.5 dB below the median
        # sentence speech level to prevent quiet words or suffixes (e.g. 'pudding') from being swallowed.
        speech_segs = [s for s in plan.segments if not s.is_silence and s.span_type != CONTINUE_SPAN_TYPE]
        if speech_segs:
            speech_levels = [segment_rms_db.get(s.segment_index, -20.0) + (s.gain_db or 0.0) for s in speech_segs]
            median_level = float(np.median(speech_levels))
            min_allowed_level = median_level - 3.5
            for s in speech_segs:
                current_level = segment_rms_db.get(s.segment_index, -20.0) + (s.gain_db or 0.0)
                if current_level < min_allowed_level:
                    deficit = min_allowed_level - current_level
                    is_subword_or_phone = s.span_type in ("subword", "phone")
                    limit_db = 12.0 if is_subword_or_phone else 9.0
                    s.gain_db = float(min(limit_db, (s.gain_db or 0.0) + deficit))

    # Continued pauses inherit the gain of the segment they continue
    for k, s in enumerate(plan.segments):
        if s.span_type == CONTINUE_SPAN_TYPE and k > 0:
            s.gain_db = plan.segments[k - 1].gain_db or 0.0

    # Step 2: Measure loudness on an unpublished analysis master and record metadata
    if not is_cuts_only and normalize_loudness:
        try:
            from video_assembler import render_master_pcm
            with tempfile.TemporaryDirectory(prefix="sm_loudness_analysis_") as tmp_d:
                analysis_wav = Path(tmp_d) / "analysis_master.wav"
                effective_analysis_track = "vocals" if audio_track in ("both", "all") else audio_track
                render_master_pcm(plan, str(analysis_wav), normalize_loudness=False, audio_track=effective_analysis_track)
                if analysis_wav.exists() and analysis_wav.stat().st_size > 0:
                    cmd1 = [
                        "ffmpeg", "-nostats",
                        "-i", str(analysis_wav),
                        "-af", f"loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA={target_lra:.1f}:print_format=json",
                        "-f", "null", "-"
                    ]
                    proc1 = subprocess.run(cmd1, capture_output=True, text=True)
                    if proc1.returncode == 0:
                        match = re.search(r"\{\s*\"input_i\"\s*:.*?\}", proc1.stderr, re.DOTALL)
                        if match:
                            stats = json.loads(match.group(0))
                            plan.metadata["ebur128_analysis"] = stats
                            plan.metadata["ebur128_target"] = {
                                "target_i": target_i,
                                "target_tp": target_tp,
                                "target_lra": target_lra,
                            }
        except Exception:
            pass

    plan.metadata["normalize_loudness"] = normalize_loudness
    return plan
