"""
Variable-span unit selector and global dynamic programming search.
Implements the Phrase -> Word -> Whole Phone hierarchy, protects natural source runs,
enforces legal contiguity, and performs deterministic shortest-path sequence optimization.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np

from contracts import InventoryIndex, SpanCandidate, TargetGraph, WordOccurrence
from features import compute_join_cost, compute_manner_scale
from lexicon import strip_phone_stress, VOWEL_PHONES


class AcousticMetrics(tuple):
    """
    Backward-compatible 2-tuple (f0_slope, rms_energy) that also exposes
    f0_mean, cand_f0_hz, and dictionary-like / attribute access (Fix P3.4).
    """
    f0_slope: float
    rms_energy: float
    f0_mean: float

    def __new__(cls, f0_slope: float, rms_energy: float, f0_mean: float = 0.0):
        obj = super().__new__(cls, (float(f0_slope), float(rms_energy)))
        obj.f0_slope = float(f0_slope)
        obj.rms_energy = float(rms_energy)
        obj.f0_mean = float(f0_mean)
        return obj

    def __getitem__(self, item: Any) -> Any:
        if isinstance(item, str):
            if item in ("f0_slope", "slope"):
                return self.f0_slope
            if item in ("rms", "rms_energy"):
                return self.rms_energy
            if item in ("f0_mean", "cand_f0_hz", "f0"):
                return self.f0_mean
            raise KeyError(item)
        return super().__getitem__(item)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default


def phone_edit_distance(seq1: Tuple[str, ...], seq2: Tuple[str, ...]) -> int:
    """Computes Levenshtein edit distance between two phone tuples."""
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]


def apply_diphone_midpoint_transitions(
    spans: List[SpanCandidate],
    graph: TargetGraph,
    inventory: InventoryIndex,
) -> List[SpanCandidate]:
    """
    Applies diphone midpoint transitions for subword and phone joins inside synthesized words.
    When two subword or phone spans join inside the same synthesized word and share a boundary phone,
    positions the splice at the stationary formant midpoint (start_sample + end_sample) // 2.
    """
    if len(spans) < 2:
        return spans

    new_spans = list(spans)
    for k in range(len(new_spans) - 1):
        curr = new_spans[k]
        succ = new_spans[k + 1]

        if curr.target_node_end != succ.target_node_start:
            continue

        if curr.span_type not in ("subword", "phone") or succ.span_type not in ("subword", "phone"):
            continue

        node_curr = graph.nodes[curr.target_node_end] if curr.target_node_end < len(graph.nodes) else None
        node_succ = graph.nodes[succ.target_node_start] if succ.target_node_start < len(graph.nodes) else None
        if node_curr is None or node_succ is None:
            continue
        if node_curr.token_index != node_succ.token_index:
            continue

        p_curr = curr.realized_phones[-1] if curr.realized_phones else node_curr.phone
        p_succ = succ.realized_phones[0] if succ.realized_phones else node_succ.phone
        base_curr = strip_phone_stress(p_curr)
        base_succ = strip_phone_stress(p_succ)
        if base_curr != base_succ:
            continue

        ev_curr = inventory.events.get(curr.event_ids[-1]) if curr.event_ids else None
        ev_succ = inventory.events.get(succ.event_ids[0]) if succ.event_ids else None

        if ev_curr and ev_succ:
            mid_curr = (ev_curr.start_sample + ev_curr.end_sample) // 2
            mid_succ = (ev_succ.start_sample + ev_succ.end_sample) // 2
            can_trim_curr = curr.start_sample < mid_curr <= ev_curr.end_sample
            can_trim_succ = ev_succ.start_sample <= mid_succ < succ.end_sample

            # Deduplicate shared boundary node across the diphone join only when both sides
            # trim at the midpoint and the shared node is still present on both sides (idempotent)
            curr_cov = list(new_spans[k].covered_nodes)
            succ_cov = list(new_spans[k + 1].covered_nodes)
            curr_rp = list(new_spans[k].realized_phones) if new_spans[k].realized_phones else []
            succ_rp = list(new_spans[k + 1].realized_phones) if new_spans[k + 1].realized_phones else []
            if (
                can_trim_curr
                and can_trim_succ
                and curr.target_node_end == succ.target_node_start
                and curr_cov
                and succ_cov
                and curr_cov[-1] == succ_cov[0]
            ):
                if len(curr_cov) > 1:
                    curr_cov = curr_cov[:-1]
                    if len(curr_rp) > 1:
                        curr_rp = curr_rp[:-1]
                elif len(succ_cov) > 1:
                    succ_cov = succ_cov[1:]
                    if len(succ_rp) > 1:
                        succ_rp = succ_rp[1:]

            if can_trim_curr:
                new_spans[k] = dataclasses.replace(
                    new_spans[k],
                    end_sample=mid_curr,
                    covered_nodes=curr_cov,
                    realized_phones=curr_rp,
                )

            if can_trim_succ:
                new_spans[k + 1] = dataclasses.replace(
                    new_spans[k + 1],
                    start_sample=mid_succ,
                    covered_nodes=succ_cov,
                    realized_phones=succ_rp,
                )

    return new_spans


class SpanSelector:
    """
    Selects optimal variable-length units (Phrases, Words, Phones) to cover a TargetGraph.
    """
    def __init__(
        self,
        inventory: InventoryIndex,
        edit_penalty: float = 0.5,
        max_candidates_per_span: int = 50,
    ):
        self.inv = inventory
        self.edit_penalty = edit_penalty
        self.max_candidates_per_span = max_candidates_per_span
        self._wavlm_cache: Dict[Any, np.ndarray] = {}
        self._acoustic_metrics_cache: Dict[Tuple[str, int, int], AcousticMetrics] = {}

        # Reverse pronunciation index from realized phones to intact recorded WordOccurrences
        self.pron_to_words: Dict[Tuple[str, ...], List[WordOccurrence]] = {}
        self.base_pron_to_words: Dict[Tuple[str, ...], List[WordOccurrence]] = {}
        for wo in self.inv.words.values():
            if not getattr(wo, "is_eligible", True):
                continue
            if wo.realized_phones:
                key = tuple(wo.realized_phones)
                self.pron_to_words.setdefault(key, []).append(wo)
                base_key = tuple(strip_phone_stress(p) for p in wo.realized_phones)
                self.base_pron_to_words.setdefault(base_key, []).append(wo)

        # Reverse subword (n-phone) index for contiguous multi-phone slices within words
        self.subword_index: Dict[Tuple[str, ...], List[Tuple[WordOccurrence, int, int]]] = {}
        self.base_subword_index: Dict[Tuple[str, ...], List[Tuple[WordOccurrence, int, int]]] = {}
        for wo in self.inv.words.values():
            if not wo.phone_event_ids or not getattr(wo, "is_eligible", True):
                continue
            evs = [self.inv.events[eid] for eid in wo.phone_event_ids if eid in self.inv.events and self.inv.events[eid].kind == "phone"]
            if len(evs) < 2:
                continue
            labels = tuple(e.label for e in evs)
            base_labels = tuple(strip_phone_stress(p) for p in labels)
            for length in range(2, min(len(evs) + 1, 9)):
                for start_i in range(len(evs) - length + 1):
                    end_i = start_i + length
                    sub_key = labels[start_i:end_i]
                    sub_base_key = base_labels[start_i:end_i]
                    self.subword_index.setdefault(sub_key, []).append((wo, start_i, end_i))
                    self.base_subword_index.setdefault(sub_base_key, []).append((wo, start_i, end_i))

        # Speaker and global median duration modeling (Fix P3.2)
        speaker_word_durations: Dict[Tuple[Optional[str], str], List[float]] = {}
        global_word_durations: Dict[str, List[float]] = {}
        for wo in self.inv.words.values():
            if not getattr(wo, "is_eligible", True):
                continue
            srec = self.inv.sources.get(wo.source_content_id)
            sr = srec.native_sample_rate if srec and srec.native_sample_rate > 0 else 44100
            dur = (wo.end_sample - wo.start_sample) / float(sr)
            w_norm = wo.word_normalized.lower()
            speaker_word_durations.setdefault((wo.speaker_id, w_norm), []).append(dur)
            global_word_durations.setdefault(w_norm, []).append(dur)

        self.speaker_word_median: Dict[Tuple[Optional[str], str], float] = {
            k: float(np.median(v)) for k, v in speaker_word_durations.items() if v
        }
        self.global_word_median: Dict[str, float] = {
            k: float(np.median(v)) for k, v in global_word_durations.items() if v
        }

        # Speaker and global median duration modeling for phones
        speaker_phone_durations: Dict[Tuple[Optional[str], str], List[float]] = {}
        global_phone_durations: Dict[str, List[float]] = {}
        for ev in self.inv.events.values():
            if getattr(ev, "kind", "") != "phone" or not getattr(ev, "is_eligible", True):
                continue
            srec = self.inv.sources.get(ev.source_content_id)
            sr = srec.native_sample_rate if srec and srec.native_sample_rate > 0 else 44100
            dur = (ev.end_sample - ev.start_sample) / float(sr)
            base_p = getattr(ev, "base_phone", "") or strip_phone_stress(ev.label)
            speaker_phone_durations.setdefault((ev.speaker_id, ev.label), []).append(dur)
            speaker_phone_durations.setdefault((ev.speaker_id, base_p), []).append(dur)
            global_phone_durations.setdefault(ev.label, []).append(dur)
            global_phone_durations.setdefault(base_p, []).append(dur)

        self.speaker_phone_median: Dict[Tuple[Optional[str], str], float] = {
            k: float(np.median(v)) for k, v in speaker_phone_durations.items() if v
        }
        self.global_phone_median: Dict[str, float] = {
            k: float(np.median(v)) for k, v in global_phone_durations.items() if v
        }

        # Preloaded feature tracks for zero-disk-seek candidate acoustic evaluation
        self._preloaded_tracks: Dict[str, Dict[str, Any]] = {}
        spk_f0_vals: Dict[Optional[str], List[float]] = {}
        all_f0_vals: List[float] = []

        for src_id, srec in self.inv.sources.items():
            pcm_path = srec.cached_analysis_pcm_path or srec.cached_pcm_path or srec.source_path
            if pcm_path:
                feat_path = Path(pcm_path).parent / f"{Path(pcm_path).stem}.span_features.npz"
                if not feat_path.exists():
                    feat_path = Path(pcm_path).parent / f"{Path(pcm_path).stem}.features.npz"
                if feat_path.exists():
                    try:
                        npz_data = np.load(str(feat_path), allow_pickle=True)
                        self._preloaded_tracks[src_id] = {k: npz_data[k] for k in npz_data.files}
                    except Exception:
                        pass

        for wo in self.inv.words.values():
            if not getattr(wo, "is_eligible", True):
                continue
            tracks = self._preloaded_tracks.get(wo.source_content_id)
            if not tracks or "f0" not in tracks or "voiced_flag" not in tracks:
                continue
            hop = int(tracks.get("hop_length", 160))
            tr_sr = int(tracks.get("sr", 16000))
            srec = self.inv.sources.get(wo.source_content_id)
            native_sr = srec.native_sample_rate if srec and srec.native_sample_rate > 0 else 48000
            s_frame = min(max(0, int(round((wo.start_sample / float(native_sr)) * tr_sr / hop))), max(0, int(tracks["n_frames"]) - 1))
            e_frame = min(int(tracks["n_frames"]), max(s_frame + 1, int(round((wo.end_sample / float(native_sr)) * tr_sr / hop))))
            if e_frame > s_frame:
                f0_sl = tracks["f0"][s_frame:e_frame]
                vf_sl = tracks["voiced_flag"][s_frame:e_frame]
                vp_sl = tracks["voiced_probs"][s_frame:e_frame] if "voiced_probs" in tracks else np.ones_like(vf_sl, dtype=float)
                for f, vf, vp in zip(f0_sl, vf_sl, vp_sl):
                    if vf and vp > 0.5 and f is not None and not np.isnan(f) and 80.0 <= f <= 500.0:
                        spk_f0_vals.setdefault(wo.speaker_id, []).append(float(f))
                        all_f0_vals.append(float(f))

        self.speaker_median_f0: Dict[Optional[str], float] = {
            spk: float(np.median(vals)) for spk, vals in spk_f0_vals.items() if vals
        }
        self.global_median_f0: float = float(np.median(all_f0_vals)) if all_f0_vals else 150.0

    @staticmethod
    def _compute_quadratic_duration_penalty(actual_dur: float, median_dur: float) -> float:
        """
        Computes quadratic duration penalty (Fix P3.2):
        If ratio < 0.75: 0.60 * ((0.75 - ratio) / 0.75) ** 2
        If ratio > 1.50: 0.30 * min(2.0, (ratio - 1.50) / 1.0) ** 2
        Else: 0.0
        where ratio = actual_dur / max(median_dur, 0.04).
        """
        ref_dur = max(median_dur, 0.04)
        ratio = actual_dur / ref_dur
        if ratio < 0.75:
            return 0.60 * ((0.75 - ratio) / 0.75) ** 2
        elif ratio > 1.50:
            return 0.30 * min(2.0, (ratio - 1.50) / 1.0) ** 2
        return 0.0

    def _get_speaker_median_duration(
        self, speaker_id: Optional[str], word_norm: str, fallback: float
    ) -> float:
        w_norm = word_norm.lower()
        if (speaker_id, w_norm) in self.speaker_word_median:
            return self.speaker_word_median[(speaker_id, w_norm)]
        if w_norm in self.global_word_median:
            return self.global_word_median[w_norm]
        return fallback

    def _get_speaker_phone_median_duration(
        self, speaker_id: Optional[str], phone_label: str, fallback: float
    ) -> float:
        base_p = strip_phone_stress(phone_label)
        if (speaker_id, phone_label) in self.speaker_phone_median:
            return self.speaker_phone_median[(speaker_id, phone_label)]
        if (speaker_id, base_p) in self.speaker_phone_median:
            return self.speaker_phone_median[(speaker_id, base_p)]
        if phone_label in self.global_phone_median:
            return self.global_phone_median[phone_label]
        if base_p in self.global_phone_median:
            return self.global_phone_median[base_p]
        return fallback

    def retrieve_candidates(self, graph: TargetGraph, speaker_id: Optional[str] = None) -> List[SpanCandidate]:
        """
        Retrieves Phrase, Word, and Phone candidates covering contiguous segments of the target graph.
        """
        candidates: List[SpanCandidate] = []
        cand_id_counter = 0

        span_pools: List[List[SpanCandidate]] = []

        # Map word_normalized to matching word occurrences in inventory
        # Filter by speaker if requested
        eligible_words: Dict[str, List[WordOccurrence]] = {}
        for w_norm in set(graph.normalized_words):
            word_ids = self.inv.word_to_word_ids.get(w_norm, [])
            matches = [self.inv.words[wid] for wid in word_ids if wid in self.inv.words]
            if speaker_id is not None:
                matches = [wo for wo in matches if wo.speaker_id == speaker_id and getattr(wo, "is_eligible", True)]
            else:
                matches = [wo for wo in matches if getattr(wo, "is_eligible", True)]
            if matches:
                eligible_words[w_norm] = matches

        # Group graph nodes by word / token_index
        token_to_nodes: Dict[int, List[int]] = {}
        for n in graph.nodes:
            token_to_nodes.setdefault(n.token_index, []).append(n.node_id)

        # Group graph nodes by (token_index, branch_id) for branch-isolated candidates
        token_to_branch_nodes: Dict[tuple, List[int]] = {}
        for n in graph.nodes:
            key = (n.token_index, n.branch_id)
            token_to_branch_nodes.setdefault(key, []).append(n.node_id)

        # Per-token ordered list of (branch_id, [node_ids]) for word candidates
        token_branches: Dict[int, List[tuple]] = {}
        for t_idx in set(n.token_index for n in graph.nodes):
            branch_ids = sorted(set(
                n.branch_id for n in graph.nodes if n.token_index == t_idx
            ))
            token_branches[t_idx] = [
                (bid, token_to_branch_nodes[(t_idx, bid)]) for bid in branch_ids
            ]

        n_tokens = len(graph.normalized_words)

        # 1. Multi-word phrase candidates (Level 1)
        for span_len in range(min(n_tokens, 32), 1, -1):
            for start_tok in range(n_tokens - span_len + 1):
                end_tok = start_tok + span_len - 1
                phrase_tokens = graph.normalized_words[start_tok:end_tok + 1]

                # Explicit gap boundaries cannot be swallowed by a multi-word phrase candidate
                if any(t in getattr(graph, "explicit_gaps", {}) for t in range(start_tok, end_tok)):
                    continue

                # Find source words matching phrase_tokens[0]
                first_w_candidates = eligible_words.get(phrase_tokens[0], [])
                for first_wo in first_w_candidates:
                    # Check if subsequent words in source continue consecutively
                    curr_wo = first_wo
                    matched_wos = [first_wo]
                    is_full_phrase = True

                    for next_tok in phrase_tokens[1:]:
                        # Find if curr_wo's last event successor is the start of a next_tok word
                        last_ev_id = curr_wo.phone_event_ids[-1]
                        last_ev = self.inv.events.get(last_ev_id)
                        if not last_ev or not last_ev.successor_id:
                            is_full_phrase = False
                            break

                        succ_ev = self.inv.events.get(last_ev.successor_id)
                        # Skip silences between words in phrase if short
                        while succ_ev and succ_ev.kind == "silence":
                            if not succ_ev.successor_id:
                                break
                            succ_ev = self.inv.events.get(succ_ev.successor_id)

                        if not succ_ev or not succ_ev.word_occurrence_id:
                            is_full_phrase = False
                            break

                        next_wo = self.inv.words.get(succ_ev.word_occurrence_id)
                        if next_wo and getattr(next_wo, "is_eligible", True) and next_wo.word_normalized == next_tok and (speaker_id is None or next_wo.speaker_id == speaker_id):
                            matched_wos.append(next_wo)
                            curr_wo = next_wo
                        else:
                            is_full_phrase = False
                            break

                    if is_full_phrase and len(matched_wos) == span_len:
                        # Attach each word in phrase to its matching pronunciation branch (B09)
                        phrase_compatible = True
                        covered: List[int] = []
                        total_realized_phones: List[str] = []
                        phrase_realization_cost = 0.0

                        for tok_offset, wo in enumerate(matched_wos):
                            t_idx = start_tok + tok_offset
                            branches = token_branches.get(t_idx, [(0, token_to_nodes.get(t_idx, []))])
                            wo_p = tuple(wo.realized_phones) if wo.realized_phones else ()
                            if not wo_p:
                                wo_p = tuple(self.inv.events[eid].label for eid in wo.phone_event_ids if eid in self.inv.events and self.inv.events[eid].kind == "phone")
                            wo_base = tuple(strip_phone_stress(p) for p in wo_p)
                            total_realized_phones.extend(wo_p)

                            # Find best matching branch for this word occurrence
                            best_b_nodes = branches[0][1] if branches else []
                            best_match_quality = 999
                            found_match = False

                            for _bid, b_nodes in branches:
                                if not b_nodes:
                                    continue
                                b_p = tuple(graph.nodes[nid].phone for nid in b_nodes)
                                b_base = tuple(graph.nodes[nid].base_phone for nid in b_nodes)

                                if wo_p == b_p:
                                    best_b_nodes = b_nodes
                                    best_match_quality = 0
                                    found_match = True
                                    break
                                elif wo_base == b_base and best_match_quality > 1:
                                    best_b_nodes = b_nodes
                                    best_match_quality = 1
                                    found_match = True

                            if not found_match:
                                # Check if token is constrained
                                is_tok_constrained = any(graph.nodes[nid].is_constrained for nid in (branches[0][1] if branches else []))
                                if is_tok_constrained:
                                    phrase_compatible = False
                                    break
                                else:
                                    # Unconstrained token: attach to branch 0 with penalty
                                    best_b_nodes = branches[0][1] if branches else []
                                    b0_base = tuple(graph.nodes[nid].base_phone for nid in best_b_nodes)
                                    dist = phone_edit_distance(wo_base, b0_base)
                                    phrase_realization_cost += 0.25 + 0.5 * (dist / max(len(b0_base), 1))
                            else:
                                if best_match_quality == 1:
                                    phrase_realization_cost += 0.05

                            covered.extend(best_b_nodes)

                        if not phrase_compatible or not covered:
                            continue

                        all_events = []
                        for wo in matched_wos:
                            all_events.extend(wo.phone_event_ids)

                        src_rec = self.inv.sources.get(first_wo.source_content_id)
                        s_rate = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else 44100
                        phrase_dur = (matched_wos[-1].end_sample - matched_wos[0].start_sample) / float(s_rate)

                        expected_phrase_dur = sum(
                            self._get_speaker_median_duration(speaker_id or first_wo.speaker_id, tok, 0.25)
                            for tok in phrase_tokens
                        )
                        rush_penalty = 0.0
                        if expected_phrase_dur > 0 and phrase_dur < 0.55 * expected_phrase_dur:
                            rush_penalty = 0.75 * (0.55 * expected_phrase_dur - phrase_dur) / (0.55 * expected_phrase_dur)

                        swallowed_penalty = self._compute_swallowed_penalty(all_events, s_rate)

                        enunciation_guard_cost = 0.0
                        function_words = {
                            "to", "the", "a", "an", "of", "in", "on", "for", "at", "and", "or", "but", "so",
                            "it", "is", "as", "by", "are", "was", "were", "be", "been", "has", "have", "had",
                            "do", "does", "did", "my", "your", "his", "her", "our", "their", "we", "you", "he", "she", "they", "i", "not"
                        }
                        for wo_idx, wo in enumerate(matched_wos):
                            w_dur = (wo.end_sample - wo.start_sample) / float(s_rate)
                            is_func = wo.word_normalized.lower() in function_words
                            min_dur = 0.025 if is_func else 0.080
                            if w_dur < min_dur:
                                enunciation_guard_cost += 3.0 * (min_dur - w_dur) / min_dur
                            elif not is_func and (wo_idx == 0 or wo_idx == len(matched_wos) - 1) and w_dur < 0.100:
                                enunciation_guard_cost += 1.5 * (0.100 - w_dur) / 0.100

                        phrase_end_sample = matched_wos[-1].end_sample
                        phrase_start_sample = matched_wos[0].start_sample

                        exit_variant = 0
                        entry_variant = 0

                        end_node = graph.nodes[covered[-1]]
                        end_r_set = getattr(end_node, "right_context_phones", None) or ({end_node.right_context_phone} if end_node.right_context_phone else set())
                        if matched_wos[-1].phone_event_ids:
                            last_ev = self.inv.events.get(matched_wos[-1].phone_event_ids[-1])
                            if last_ev and last_ev.successor_id:
                                succ_ev = self.inv.events.get(last_ev.successor_id)
                                if succ_ev and succ_ev.kind == "phone":
                                    succ_base = getattr(succ_ev, "base_phone", "") or strip_phone_stress(succ_ev.label)
                                    if succ_base in ("P", "T", "K", "B", "D", "G"):
                                        exit_variant = max(exit_variant, int(round(0.012 * s_rate)))

                        # Guard 1: Pause / Gap & Phrase-Final Right Bleed Guard
                        phrase_right_bleed_cost = 0.0
                        if end_node.is_phrase_final or "sil" in end_r_set or end_node.right_context_phone == "sil":
                            last_wo = matched_wos[-1]
                            if last_wo.phone_event_ids:
                                last_ev = self.inv.events.get(last_wo.phone_event_ids[-1])
                                if last_ev and last_ev.successor_id:
                                    succ_ev = self.inv.events.get(last_ev.successor_id)
                                    if succ_ev and getattr(succ_ev, "kind", "") == "phone":
                                        phrase_right_bleed_cost = 0.35
                                        exit_variant = max(exit_variant, int(round(0.015 * s_rate)))

                        # Guard 2: Left Boundary Consonant / Nasal / Glide Bleed Guard
                        phrase_left_bleed_cost = 0.0
                        if first_wo.phone_event_ids:
                            first_ev = self.inv.events.get(first_wo.phone_event_ids[0])
                            if first_ev and first_ev.predecessor_id:
                                pred_ev = self.inv.events.get(first_ev.predecessor_id)
                                if pred_ev and getattr(pred_ev, "kind", "") == "phone":
                                    pred_base = getattr(pred_ev, "base_phone", "") or strip_phone_stress(pred_ev.label)
                                    first_node = graph.nodes[covered[0]]
                                    first_l_set = getattr(first_node, "left_context_phones", None) or ({first_node.left_context_phone} if first_node.left_context_phone else set())
                                    phrase_l_bases = {strip_phone_stress(p) for p in first_l_set if p}
                                    if pred_base in ("P", "T", "K", "B", "D", "G") and not (phrase_l_bases & {"P", "T", "K", "B", "D", "G"}):
                                        entry_variant = max(entry_variant, int(round(0.012 * s_rate)))
                                    if first_node.base_phone in VOWEL_PHONES:
                                        if pred_base == "V":
                                            phrase_left_bleed_cost = 0.80
                                            entry_variant = max(entry_variant, int(round(0.020 * s_rate)))
                                        elif pred_base in ("Z", "ZH", "DH"):
                                            phrase_left_bleed_cost = 0.20
                                        elif pred_base in ("N", "M", "NG", "Y", "R", "W", "L", "ER") or pred_base in VOWEL_PHONES:
                                            phrase_left_bleed_cost = 0.50
                                            entry_variant = max(entry_variant, int(round(0.020 * s_rate)))
                                    elif pred_base in ("N", "M", "NG", "Y", "R", "W", "L", "ER") or pred_base in VOWEL_PHONES:
                                        phrase_left_bleed_cost = 0.50
                                        entry_variant = max(entry_variant, int(round(0.020 * s_rate)))

                        bias_adj = self._compute_prosodic_bias_adjustment(
                            covered, first_wo.source_content_id, phrase_start_sample, phrase_end_sample, graph
                        )
                        p_feat = self._get_candidate_acoustic_features(first_wo.source_content_id, phrase_start_sample, phrase_end_sample)
                        p_f0 = p_feat.f0_mean if p_feat.f0_mean > 0 else None
                        ref_f0 = self.speaker_median_f0.get(first_wo.speaker_id, self.global_median_f0)
                        p_pitch = float(12.0 * np.log2(p_f0 / max(50.0, ref_f0))) if p_f0 else None
                        c = SpanCandidate(
                            candidate_id=f"cand_phrase_{cand_id_counter:05d}",
                            target_node_start=covered[0],
                            target_node_end=covered[-1],
                            covered_nodes=covered,
                            source_content_id=first_wo.source_content_id,
                            speaker_id=first_wo.speaker_id,
                            start_sample=phrase_start_sample,
                            end_sample=phrase_end_sample,
                            event_ids=all_events,
                            span_type="phrase",
                            matched_word=" ".join(phrase_tokens),
                            entry_variant_samples=entry_variant,
                            exit_variant_samples=exit_variant,
                            target_cost=0.0 + bias_adj + phrase_realization_cost + enunciation_guard_cost + rush_penalty + swallowed_penalty + phrase_right_bleed_cost + phrase_left_bleed_cost,
                            quality_cost=0.0,
                            realized_phones=total_realized_phones,
                            cand_f0_hz=p_f0,
                            pitch_semitones=p_pitch,
                        )
                        candidates.append(c)
                        cand_id_counter += 1

        # 2. Whole word candidates (Level 2)
        for t_idx in range(n_tokens):
            w_norm = graph.normalized_words[t_idx]
            word_candidates_base = list(eligible_words.get(w_norm, []))

            # Generate one set of candidates per pronunciation branch
            for _bid, branch_nodes in token_branches.get(t_idx, [(0, token_to_nodes.get(t_idx, []))]):
                if not branch_nodes:
                    continue
                branch_phones = tuple(graph.nodes[nid].phone for nid in branch_nodes)
                branch_base_phones = tuple(graph.nodes[nid].base_phone for nid in branch_nodes)
                is_constrained = any(graph.nodes[nid].is_constrained for nid in branch_nodes)

                word_candidates = list(word_candidates_base)

                # Reverse pronunciation lookup for intact recorded words matching this branch's phone sequence
                pron_matches = self.pron_to_words.get(branch_phones, [])
                if not pron_matches:
                    pron_matches = self.base_pron_to_words.get(branch_base_phones, [])

                for p_wo in pron_matches:
                    if speaker_id is not None and p_wo.speaker_id != speaker_id:
                        continue
                    if p_wo not in word_candidates:
                        word_candidates.append(p_wo)

                # Filter word_candidates by realized_phones and apply D6 constraints
                valid_branch_candidates_and_costs: List[Tuple[WordOccurrence, float, List[str]]] = []
                for wo in word_candidates:
                    wo_phones = tuple(wo.realized_phones) if wo.realized_phones else ()
                    if not wo_phones:
                        wo_phones = tuple(self.inv.events[eid].label for eid in wo.phone_event_ids if eid in self.inv.events and self.inv.events[eid].kind == "phone")
                    wo_base_phones = tuple(strip_phone_stress(p) for p in wo_phones)

                    realization_cost = 0.0
                    if wo_phones:
                        if wo_phones == branch_phones:
                            realization_cost = 0.0
                        elif wo_base_phones == branch_base_phones:
                            realization_cost = 0.05  # stress-only difference
                        else:
                            # Check if this realization matches another branch of the token
                            matches_other_branch = False
                            for other_bid, other_nodes in token_branches.get(t_idx, []):
                                if other_bid != _bid and other_nodes:
                                    other_phones = tuple(graph.nodes[nid].phone for nid in other_nodes)
                                    other_base = tuple(graph.nodes[nid].base_phone for nid in other_nodes)
                                    if wo_phones == other_phones or wo_base_phones == other_base:
                                        matches_other_branch = True
                                        break
                            if matches_other_branch:
                                continue

                            # Mismatched realization
                            if is_constrained:
                                # D6: Constrained tokens reject mismatched realization
                                continue
                            else:
                                dist = phone_edit_distance(wo_base_phones, branch_base_phones)
                                realization_cost = 0.25 + 0.5 * (dist / max(len(branch_base_phones), 1))

                    valid_branch_candidates_and_costs.append((wo, realization_cost, list(wo_phones)))

                priors_sum = sum(
                    graph.nodes[nid].duration_prior_sec or 0.10
                    for nid in branch_nodes if nid < len(graph.nodes)
                )
                target_word_text = graph.nodes[branch_nodes[0]].word_text.lower()
                target_word_in_inv = target_word_text in self.inv.word_to_word_ids
                expected_dur = self._get_speaker_median_duration(speaker_id, target_word_text, priors_sum)

                end_node = graph.nodes[branch_nodes[-1]]
                start_node = graph.nodes[branch_nodes[0]]
                end_r_phones = getattr(end_node, "right_context_phones", None) or {end_node.right_context_phone}

                def _compute_word_bleed(wo_it: WordOccurrence, srate: int) -> Tuple[float, float]:
                    r_bleed = 0.0
                    l_bleed = 0.0
                    if (end_node.is_phrase_final or "sil" in end_r_phones or end_node.right_context_phone == "sil") and wo_it.phone_event_ids:
                        l_ev = self.inv.events.get(wo_it.phone_event_ids[-1])
                        if l_ev and l_ev.successor_id:
                            s_ev = self.inv.events.get(l_ev.successor_id)
                            if s_ev and getattr(s_ev, "kind", "") == "phone":
                                r_bleed = 0.35
                    if wo_it.phone_event_ids:
                        f_ev = self.inv.events.get(wo_it.phone_event_ids[0])
                        if f_ev and f_ev.predecessor_id:
                            p_ev = self.inv.events.get(f_ev.predecessor_id)
                            if p_ev and getattr(p_ev, "kind", "") == "phone":
                                p_base = getattr(p_ev, "base_phone", "") or strip_phone_stress(p_ev.label)
                                if start_node.base_phone in VOWEL_PHONES:
                                    if p_base == "V":
                                        l_bleed = 0.80
                                    elif p_base in ("Z", "ZH", "DH"):
                                        l_bleed = 0.20
                                    elif p_base in ("N", "M", "NG", "Y", "R", "W", "L", "ER") or p_base in VOWEL_PHONES:
                                        l_bleed = 0.60
                                elif p_base in ("N", "M", "NG", "Y", "R", "W", "L", "ER") or p_base in VOWEL_PHONES:
                                    l_bleed = 0.50
                    return r_bleed, l_bleed

                def _word_candidate_sort_key(item: Tuple[WordOccurrence, float, List[str]]) -> float:
                    wo_it, r_c, _ = item
                    srec = self.inv.sources.get(wo_it.source_content_id)
                    srate = srec.native_sample_rate if srec and srec.native_sample_rate > 0 else 44100
                    w_dur = (wo_it.end_sample - wo_it.start_sample) / float(srate)
                    d_cost = self._compute_quadratic_duration_penalty(w_dur, expected_dur)
                    swallow_c = self._compute_swallowed_penalty(wo_it.phone_event_ids, srate)
                    lex_c = 0.0
                    if wo_it.word_normalized.lower() != target_word_text:
                        lex_c = 0.35 if target_word_in_inv else 0.05
                    r_bl, l_bl = _compute_word_bleed(wo_it, srate)
                    heur_c = min(0.50, r_bl + l_bl + swallow_c)
                    return r_c + d_cost + heur_c + lex_c

                valid_branch_candidates_and_costs.sort(key=_word_candidate_sort_key)
                word_branch_cands: List[SpanCandidate] = []
                target_l_phones = getattr(start_node, "left_context_phones", None) or ({start_node.left_context_phone} if start_node.left_context_phone else set())
                target_l_bases = {strip_phone_stress(p) for p in target_l_phones if p}
                for wo, r_c, r_phones in valid_branch_candidates_and_costs:
                    srec = self.inv.sources.get(wo.source_content_id)
                    srate = srec.native_sample_rate if srec and srec.native_sample_rate > 0 else 44100
                    w_dur = (wo.end_sample - wo.start_sample) / float(srate)
                    d_cost = self._compute_quadratic_duration_penalty(w_dur, expected_dur)
                    swallow_c = self._compute_swallowed_penalty(wo.phone_event_ids, srate)
                    lex_c = 0.0
                    if wo.word_normalized.lower() != target_word_text:
                        lex_c = 0.35 if target_word_in_inv else 0.05
                    w_end_sample = wo.end_sample
                    w_start_sample = wo.start_sample
                    r_bl, l_bl = _compute_word_bleed(wo, srate)
                    heur_c = min(0.50, r_bl + l_bl + swallow_c)

                    exit_variant = 0
                    entry_variant = 0
                    if wo.phone_event_ids:
                        last_ev = self.inv.events.get(wo.phone_event_ids[-1])
                        if last_ev and last_ev.successor_id:
                            succ_ev = self.inv.events.get(last_ev.successor_id)
                            if succ_ev and getattr(succ_ev, "kind", "") == "phone":
                                succ_base = getattr(succ_ev, "base_phone", "") or strip_phone_stress(succ_ev.label)
                                if succ_base in ("P", "T", "K", "B", "D", "G"):
                                    exit_variant = max(exit_variant, int(round(0.012 * srate)))
                    if r_bl > 0.0:
                        exit_variant = max(exit_variant, int(round(0.015 * srate)))

                    if wo.phone_event_ids:
                        first_ev = self.inv.events.get(wo.phone_event_ids[0])
                        if first_ev and first_ev.predecessor_id:
                            pred_ev = self.inv.events.get(first_ev.predecessor_id)
                            if pred_ev and getattr(pred_ev, "kind", "") == "phone":
                                pred_base = getattr(pred_ev, "base_phone", "") or strip_phone_stress(pred_ev.label)
                                if pred_base in ("P", "T", "K", "B", "D", "G") and not (target_l_bases & {"P", "T", "K", "B", "D", "G"}):
                                    entry_variant = max(entry_variant, int(round(0.012 * srate)))
                    if l_bl > 0.0:
                        collar_sec = 0.045 if l_bl >= 0.80 else 0.020
                        entry_variant = max(entry_variant, int(round(collar_sec * srate)))

                    bias_adj = self._compute_prosodic_bias_adjustment(
                        branch_nodes, wo.source_content_id, w_start_sample, w_end_sample, graph
                    )

                    c = SpanCandidate(
                        candidate_id=f"cand_word_{cand_id_counter:05d}",
                        target_node_start=branch_nodes[0],
                        target_node_end=branch_nodes[-1],
                        covered_nodes=branch_nodes,
                        source_content_id=wo.source_content_id,
                        speaker_id=wo.speaker_id,
                        start_sample=w_start_sample,
                        end_sample=w_end_sample,
                        event_ids=wo.phone_event_ids,
                        span_type="word",
                        matched_word=wo.word_normalized,
                        entry_variant_samples=entry_variant,
                        exit_variant_samples=exit_variant,
                        target_cost=0.05 + bias_adj + r_c + d_cost + heur_c + lex_c,
                        quality_cost=0.0,
                        realized_phones=r_phones,
                    )
                    cand_id_counter += 1
                    word_branch_cands.append(c)

                if word_branch_cands:
                    span_pools.append(word_branch_cands)

        # 2.5 Sub-word multi-phone candidates (Level 2.5 - diphones, triphones, syllables)
        for t_idx, branches in token_branches.items():
            for _bid, branch_nodes in branches:
                if len(branch_nodes) < 2:
                    continue
                is_constrained = any(graph.nodes[nid].is_constrained for nid in branch_nodes)
                # Subwords cover contiguous n-phones from donor words (up to len(branch_nodes))
                for length in range(min(len(branch_nodes), 8), 1, -1):
                    for start_idx in range(len(branch_nodes) - length + 1):
                        sub_nodes = branch_nodes[start_idx:start_idx + length]
                        sub_phones = tuple(graph.nodes[nid].phone for nid in sub_nodes)
                        sub_base = tuple(graph.nodes[nid].base_phone for nid in sub_nodes)

                        sub_matches = list(self.subword_index.get(sub_phones, []))
                        base_sub_matches = [m for m in self.base_subword_index.get(sub_base, []) if m not in sub_matches]

                        if speaker_id is not None:
                            sub_matches = [m for m in sub_matches if m[0].speaker_id == speaker_id]
                            base_sub_matches = [m for m in base_sub_matches if m[0].speaker_id == speaker_id]

                        # If subword covers the entire target word, it must be a strict sub-slice of a larger donor word
                        if length == len(branch_nodes):
                            sub_matches = [m for m in sub_matches if (m[2] - m[1]) < len(m[0].phone_event_ids)]
                            base_sub_matches = [m for m in base_sub_matches if (m[2] - m[1]) < len(m[0].phone_event_ids)]

                        all_matches: List[Tuple[Tuple[WordOccurrence, int, int], float]] = []
                        for m in sub_matches:
                            all_matches.append((m, 0.0))
                        for m in base_sub_matches:
                            if m not in sub_matches:
                                all_matches.append((m, 0.05))

                        if not all_matches:
                            continue

                        expected_dur = sum(
                            self._get_speaker_phone_median_duration(
                                speaker_id, graph.nodes[nid].phone, graph.nodes[nid].duration_prior_sec or 0.10
                            )
                            for nid in sub_nodes if nid < len(graph.nodes)
                        )
                        target_l_phone = getattr(graph.nodes[sub_nodes[0]], "left_context_phones", None) or graph.nodes[sub_nodes[0]].left_context_phone
                        target_r_phone = getattr(graph.nodes[sub_nodes[-1]], "right_context_phones", None) or graph.nodes[sub_nodes[-1]].right_context_phone
                        target_l_phones = getattr(graph.nodes[sub_nodes[0]], "left_context_phones", None) or ({graph.nodes[sub_nodes[0]].left_context_phone} if graph.nodes[sub_nodes[0]].left_context_phone else set())
                        target_l_bases = {strip_phone_stress(p) for p in target_l_phones if p}
                        end_node = graph.nodes[sub_nodes[-1]]
                        start_node = graph.nodes[sub_nodes[0]]
                        end_r_set = getattr(end_node, "right_context_phones", None) or ({end_node.right_context_phone} if end_node.right_context_phone else set())

                        def _compute_subword_bleed_and_guards(
                            wo_it: WordOccurrence, s_i_it: int, e_i_it: int, srate: int
                        ) -> Tuple[float, float, float, float]:
                            r_bleed = 0.0
                            l_bleed = 0.0
                            coda_pen = 0.0
                            franken_pen = 0.0

                            evs_all = [self.inv.events[eid] for eid in wo_it.phone_event_ids if eid in self.inv.events and self.inv.events[eid].kind == "phone"]
                            if not evs_all:
                                return 0.0, 0.0, 0.0, 0.0

                            # Guard 1: Right Bleed if subword ends at donor word boundary
                            if (end_node.is_phrase_final or "sil" in end_r_set or end_node.right_context_phone == "sil") and e_i_it == len(evs_all):
                                l_ev = evs_all[-1]
                                if l_ev and l_ev.successor_id:
                                    s_ev = self.inv.events.get(l_ev.successor_id)
                                    if s_ev and getattr(s_ev, "kind", "") == "phone":
                                        r_bleed = 0.35

                            # Guard 2: Left Bleed if subword starts at donor word boundary
                            if s_i_it == 0:
                                f_ev = evs_all[0]
                                if f_ev:
                                    if f_ev.predecessor_id is None and getattr(f_ev, "kind", "") == "phone":
                                        l_bleed = 0.20
                                    elif f_ev.predecessor_id:
                                        p_ev = self.inv.events.get(f_ev.predecessor_id)
                                        if p_ev and getattr(p_ev, "kind", "") == "phone":
                                            p_base = getattr(p_ev, "base_phone", "") or strip_phone_stress(p_ev.label)
                                            if start_node.base_phone in VOWEL_PHONES:
                                                if p_base == "V":
                                                    l_bleed = 0.80
                                                elif p_base in ("Z", "ZH", "DH"):
                                                    l_bleed = 0.20
                                                elif p_base in ("N", "M", "NG", "Y", "R", "W", "L", "ER") or p_base in VOWEL_PHONES:
                                                    l_bleed = 0.50
                                            elif p_base in ("N", "M", "NG", "Y", "R", "W", "L", "ER") or p_base in VOWEL_PHONES:
                                                l_bleed = 0.50

                            # Guard 3: Coda stop as onset stop penalty
                            if start_node.is_word_initial and start_node.base_phone in ("T", "P", "K", "B", "D", "G"):
                                if evs_all and evs_all[s_i_it].event_id == wo_it.phone_event_ids[-1]:
                                    coda_pen = 0.60

                            # Guard 4: Whole-word priority over accidental subword continuations
                            target_w = graph.nodes[sub_nodes[0]].word_text.lower()
                            has_whole_word = target_w in self.inv.word_to_word_ids
                            if length == len(branch_nodes):
                                if target_w in ("and", "to", "the", "my", "you", "why", "in", "on", "for", "of", "it", "is", "a", "i") or has_whole_word:
                                    franken_pen = 0.45
                            elif has_whole_word:
                                franken_pen = 0.35

                            return r_bleed, l_bleed, coda_pen, franken_pen

                        def _subword_sort_key(item: Tuple[Tuple[WordOccurrence, int, int], float]) -> float:
                            (wo_it, s_i, e_i), r_c = item
                            src_it = self.inv.sources.get(wo_it.source_content_id)
                            sr_it = src_it.native_sample_rate if src_it and src_it.native_sample_rate > 0 else 44100
                            evs_it = [self.inv.events[eid] for eid in wo_it.phone_event_ids if eid in self.inv.events and self.inv.events[eid].kind == "phone"]
                            dur_it = (evs_it[e_i - 1].end_sample - evs_it[s_i].start_sample) / float(sr_it)
                            d_cost = self._compute_quadratic_duration_penalty(dur_it, expected_dur)
                            first_ev = evs_it[s_i]
                            last_ev = evs_it[e_i - 1]
                            l_ev = self.inv.events.get(first_ev.predecessor_id) if first_ev.predecessor_id else None
                            r_ev = self.inv.events.get(last_ev.successor_id) if last_ev.successor_id else None
                            ctx_c = self._compute_context_affinity(target_l_phone, target_r_phone, l_ev, r_ev)
                            swallow_c = self._compute_swallowed_penalty([e.event_id for e in evs_it[s_i:e_i]], sr_it)
                            floor_pen = 3.0 * (0.040 - dur_it) / 0.040 if dur_it < 0.040 else 0.0
                            flap_c = 0.0
                            if start_node.is_word_initial and start_node.base_phone in ("T", "P", "K"):
                                following_unstressed = (len(sub_nodes) > 1 and graph.nodes[sub_nodes[1]].stress == 0)
                                if following_unstressed and l_ev and getattr(l_ev, "kind", "") == "phone":
                                    prev_base = getattr(l_ev, "base_phone", "") or strip_phone_stress(l_ev.label)
                                    if prev_base in VOWEL_PHONES:
                                        flap_c = 0.35
                            r_bl, l_bl, coda_p, franken_p = _compute_subword_bleed_and_guards(wo_it, s_i, e_i, sr_it)
                            heur_c = min(0.65, swallow_c + flap_c + r_bl + l_bl + coda_p)
                            return r_c + d_cost + ctx_c + floor_pen + heur_c + franken_p

                        all_matches.sort(key=_subword_sort_key)

                        subword_slice_cands: List[SpanCandidate] = []
                        for (wo, s_i, e_i), r_cost in all_matches:
                            evs = [self.inv.events[eid] for eid in wo.phone_event_ids if eid in self.inv.events and self.inv.events[eid].kind == "phone"]
                            sub_evs = evs[s_i:e_i]
                            src_rec = self.inv.sources.get(wo.source_content_id)
                            s_rate = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else 44100
                            actual_dur = (sub_evs[-1].end_sample - sub_evs[0].start_sample) / float(s_rate)
                            dur_cost = self._compute_quadratic_duration_penalty(actual_dur, expected_dur)
                            if actual_dur < 0.040:
                                dur_cost += 3.0 * (0.040 - actual_dur) / 0.040

                            first_ev = sub_evs[0]
                            last_ev = sub_evs[-1]
                            l_ev = self.inv.events.get(first_ev.predecessor_id) if first_ev.predecessor_id else None
                            r_ev = self.inv.events.get(last_ev.successor_id) if last_ev.successor_id else None
                            ctx_cost = self._compute_context_affinity(target_l_phone, target_r_phone, l_ev, r_ev)
                            swallow_cost = self._compute_swallowed_penalty([e.event_id for e in sub_evs], s_rate)

                            sub_start_sample = sub_evs[0].start_sample
                            sub_end_sample = sub_evs[-1].end_sample
                            r_bl_cost, l_bl_cost, coda_penalty, franken_penalty = _compute_subword_bleed_and_guards(wo, s_i, e_i, s_rate)

                            exit_variant = 0
                            entry_variant = 0

                            if sub_evs[-1].successor_id:
                                succ_ev = self.inv.events.get(sub_evs[-1].successor_id)
                                if succ_ev and succ_ev.kind == "phone":
                                    succ_base = getattr(succ_ev, "base_phone", "") or strip_phone_stress(succ_ev.label)
                                    if succ_base in ("P", "T", "K", "B", "D", "G"):
                                        exit_variant = max(exit_variant, int(round(0.012 * s_rate)))

                            if sub_evs[0].predecessor_id:
                                pred_ev = self.inv.events.get(sub_evs[0].predecessor_id)
                                if pred_ev and pred_ev.kind == "phone":
                                    pred_base = getattr(pred_ev, "base_phone", "") or strip_phone_stress(pred_ev.label)
                                    if pred_base in ("P", "T", "K", "B", "D", "G") and not (target_l_bases & {"P", "T", "K", "B", "D", "G"}):
                                        entry_variant = max(entry_variant, int(round(0.012 * s_rate)))

                            if r_bl_cost > 0.0:
                                exit_variant = max(exit_variant, int(round(0.015 * s_rate)))

                            if l_bl_cost > 0.0:
                                collar_sec = 0.045 if l_bl_cost >= 0.80 else 0.020
                                entry_variant = max(entry_variant, int(round(collar_sec * s_rate)))

                            bias_adj = self._compute_prosodic_bias_adjustment(
                                sub_nodes, wo.source_content_id, sub_start_sample, sub_end_sample, graph
                            )

                            flap_penalty = 0.0
                            if start_node.is_word_initial and start_node.base_phone in ("T", "P", "K"):
                                following_unstressed = (len(sub_nodes) > 1 and graph.nodes[sub_nodes[1]].stress == 0)
                                if following_unstressed and l_ev and getattr(l_ev, "kind", "") == "phone":
                                    prev_base = getattr(l_ev, "base_phone", "") or strip_phone_stress(l_ev.label)
                                    if prev_base in VOWEL_PHONES:
                                        flap_penalty = 0.35

                            matched_phones_str = "-".join(graph.nodes[nid].phone for nid in sub_nodes)
                            # Hierarchy: Phrase (0.0) < Word (0.05) < Subword (0.06..0.11) < Phone (0.15)
                            base_target_cost = max(0.06, 0.08 + 0.01 * (5 - length))

                            heur_sub = min(0.65, swallow_cost + flap_penalty + r_bl_cost + l_bl_cost + coda_penalty)
                            c = SpanCandidate(
                                candidate_id=f"cand_subword_{cand_id_counter:05d}",
                                target_node_start=sub_nodes[0],
                                target_node_end=sub_nodes[-1],
                                covered_nodes=sub_nodes,
                                source_content_id=wo.source_content_id,
                                speaker_id=wo.speaker_id,
                                start_sample=sub_start_sample,
                                end_sample=sub_end_sample,
                                event_ids=[e.event_id for e in sub_evs],
                                span_type="subword",
                                matched_word=f"{wo.word_normalized}[{matched_phones_str}]",
                                entry_variant_samples=entry_variant,
                                exit_variant_samples=exit_variant,
                                target_cost=base_target_cost + bias_adj + r_cost + dur_cost + ctx_cost + heur_sub + franken_penalty,
                                quality_cost=0.0,
                                realized_phones=[e.label for e in sub_evs],
                            )
                            cand_id_counter += 1
                            subword_slice_cands.append(c)

                        if subword_slice_cands:
                            span_pools.append(subword_slice_cands)


        # 3. Whole phone candidates (Level 3 - fallback)
        vowel_set = {"IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW", "ER", "AY", "AW", "OY"}
        for node in graph.nodes:
            event_id_candidates: List[Tuple[str, float]] = []
            seen_eids = set()
            for eid in self.inv.phone_to_event_ids.get(node.phone, []):
                event_id_candidates.append((eid, 0.0))
                seen_eids.add(eid)

            for eid in self.inv.base_phone_to_event_ids.get(node.base_phone, []):
                if eid not in seen_eids:
                    event_id_candidates.append((eid, 0.03))
                    seen_eids.add(eid)

            # Unstressed vowel reduction / schwa (stress 0): AH0 <-> IH0 <-> EH0
            if node.phone in ("AH0", "IH0", "EH0"):
                for reduced_p in ("AH0", "IH0", "EH0"):
                    if reduced_p != node.phone:
                        for eid in self.inv.phone_to_event_ids.get(reduced_p, []):
                            if eid not in seen_eids:
                                event_id_candidates.append((eid, 0.08))
                                seen_eids.add(eid)

            # Filter by speaker if applicable
            valid_events_and_costs: List[Tuple[Any, float]] = []
            for eid, stress_c in event_id_candidates:
                ev = self.inv.events.get(eid)
                if ev and getattr(ev, "is_eligible", True) and (speaker_id is None or ev.speaker_id == speaker_id):
                    valid_events_and_costs.append((ev, stress_c))

            # Acoustic cognate donor fallback for missing phonemes (e.g. G -> K, NG -> N, V -> F)
            if not valid_events_and_costs:
                from unit_selector import PHONETIC_COGNATE_FALLBACKS
                for cog in PHONETIC_COGNATE_FALLBACKS.get(node.base_phone, []):
                    cog_base = strip_phone_stress(cog)
                    cog_eids = list(self.inv.phone_to_event_ids.get(cog, []))
                    for ceid in self.inv.base_phone_to_event_ids.get(cog_base, []):
                        if ceid not in cog_eids:
                            cog_eids.append(ceid)
                    for ceid in cog_eids:
                        ev = self.inv.events.get(ceid)
                        if ev and getattr(ev, "is_eligible", True) and (speaker_id is None or ev.speaker_id == speaker_id):
                            valid_events_and_costs.append((ev, 0.35))
                    if valid_events_and_costs:
                        break

            target_l_phone = getattr(node, "left_context_phones", None) or node.left_context_phone
            target_r_phone = getattr(node, "right_context_phones", None) or node.right_context_phone
            target_l_phones = getattr(node, "left_context_phones", None) or ({node.left_context_phone} if node.left_context_phone else set())
            target_l_bases = {strip_phone_stress(p) for p in target_l_phones if p}
            node_r_set = getattr(node, "right_context_phones", None) or ({node.right_context_phone} if node.right_context_phone else set())

            def _compute_phone_guards(ev_it: Any, sr_it: int) -> Tuple[float, float, float]:
                coda_pen = 0.0
                r_bleed = 0.0
                l_bleed = 0.0
                if node.is_word_initial and node.base_phone in ("T", "P", "K", "B", "D", "G"):
                    donor_wid = getattr(ev_it, "word_occurrence_id", None)
                    if donor_wid:
                        donor_wo = self.inv.words.get(donor_wid)
                        if donor_wo and donor_wo.phone_event_ids:
                            if ev_it.event_id == donor_wo.phone_event_ids[-1]:
                                coda_pen = 0.60
                            elif ev_it.event_id == donor_wo.phone_event_ids[0]:
                                coda_pen = -0.05

                if node.is_phrase_final or "sil" in node_r_set or node.right_context_phone == "sil":
                    if ev_it.successor_id:
                        s_ev = self.inv.events.get(ev_it.successor_id)
                        if s_ev and getattr(s_ev, "kind", "") == "phone":
                            r_bleed = 0.35

                if node.is_word_initial and node.base_phone in vowel_set:
                    if ev_it.predecessor_id:
                        p_ev = self.inv.events.get(ev_it.predecessor_id)
                        if p_ev and getattr(p_ev, "kind", "") == "phone":
                            p_base = getattr(p_ev, "base_phone", "") or strip_phone_stress(p_ev.label)
                            if p_base == "V":
                                l_bleed = 0.80
                            elif p_base in ("Z", "ZH", "DH"):
                                l_bleed = 0.20
                            elif p_base in ("N", "M", "NG", "Y", "R", "W", "L", "ER") or p_base in vowel_set:
                                l_bleed = 0.50

                return coda_pen, r_bleed, l_bleed

            phone_median = self._get_speaker_phone_median_duration(speaker_id, node.phone, node.duration_prior_sec or 0.10)

            def _phone_sort_key(item: Tuple[Any, float]) -> float:
                ev_it, stress_c = item
                sr_it = self.inv.sources[ev_it.source_content_id].native_sample_rate
                dur_it = (ev_it.end_sample - ev_it.start_sample) / float(sr_it)
                d_cost = self._compute_quadratic_duration_penalty(dur_it, phone_median)
                l_ev = self.inv.events.get(ev_it.predecessor_id) if ev_it.predecessor_id else None
                r_ev = self.inv.events.get(ev_it.successor_id) if ev_it.successor_id else None
                ctx_c = self._compute_context_affinity(target_l_phone, target_r_phone, l_ev, r_ev)
                swallow_c = 0.0
                if node.base_phone in vowel_set:
                    if dur_it < 0.050:
                        swallow_c = 2.5 * (0.050 - dur_it) / 0.050
                else:
                    if dur_it < 0.035:
                        swallow_c = 2.5 * (0.035 - dur_it) / 0.035
                flap_c = 0.0
                if node.is_word_initial and node.base_phone in ("T", "P", "K"):
                    if l_ev and getattr(l_ev, "kind", "") == "phone":
                        prev_base = getattr(l_ev, "base_phone", "") or strip_phone_stress(l_ev.label)
                        if prev_base in VOWEL_PHONES:
                            flap_c = 0.35
                coda_p, r_bl, l_bl = _compute_phone_guards(ev_it, sr_it)
                return stress_c + d_cost + 2.0 * ctx_c + swallow_c + flap_c + coda_p + r_bl + l_bl

            # Rank by context affinity + duration closeness to prior and take top candidates
            ranked = sorted(valid_events_and_costs, key=_phone_sort_key)

            phone_node_cands: List[SpanCandidate] = []
            for ev, stress_cost in ranked:
                sr = self.inv.sources[ev.source_content_id].native_sample_rate
                dur = (ev.end_sample - ev.start_sample) / float(sr)
                dur_cost = self._compute_quadratic_duration_penalty(dur, phone_median)

                p_start_sample = ev.start_sample
                p_end_sample = ev.end_sample
                coda_penalty, r_bleed_cost, l_bleed_cost = _compute_phone_guards(ev, sr)

                exit_variant = 0
                entry_variant = 0

                if ev.successor_id:
                    succ_ev = self.inv.events.get(ev.successor_id)
                    if succ_ev and succ_ev.kind == "phone":
                        succ_base = getattr(succ_ev, "base_phone", "") or strip_phone_stress(succ_ev.label)
                        if succ_base in ("P", "T", "K", "B", "D", "G"):
                            exit_variant = max(exit_variant, int(round(0.012 * sr)))

                if r_bleed_cost > 0.0:
                    exit_variant = max(exit_variant, int(round(0.015 * sr)))

                if ev.predecessor_id:
                    pred_ev = self.inv.events.get(ev.predecessor_id)
                    if pred_ev and pred_ev.kind == "phone":
                        pred_base = getattr(pred_ev, "base_phone", "") or strip_phone_stress(pred_ev.label)
                        if pred_base in ("P", "T", "K", "B", "D", "G") and not (target_l_bases & {"P", "T", "K", "B", "D", "G"}):
                            entry_variant = max(entry_variant, int(round(0.012 * sr)))

                if l_bleed_cost > 0.0:
                    collar_sec = 0.045 if l_bleed_cost >= 0.80 else 0.020
                    entry_variant = max(entry_variant, int(round(collar_sec * sr)))

                bias_adj = self._compute_prosodic_bias_adjustment(
                    [node.node_id], ev.source_content_id, p_start_sample, p_end_sample, graph
                )

                l_ev = self.inv.events.get(ev.predecessor_id) if ev.predecessor_id else None
                r_ev = self.inv.events.get(ev.successor_id) if ev.successor_id else None
                ctx_cost = self._compute_context_affinity(target_l_phone, target_r_phone, l_ev, r_ev)

                flap_penalty = 0.0
                if node.is_word_initial and node.base_phone in ("T", "P", "K"):
                    if l_ev and getattr(l_ev, "kind", "") == "phone":
                        prev_base = getattr(l_ev, "base_phone", "") or strip_phone_stress(l_ev.label)
                        if prev_base in VOWEL_PHONES:
                            flap_penalty = 0.35

                swallow_cost = 0.0
                if node.base_phone in vowel_set:
                    if dur < 0.050:
                        swallow_cost = 2.5 * (0.050 - dur) / 0.050
                else:
                    if dur < 0.035:
                        swallow_cost = 2.5 * (0.035 - dur) / 0.035

                c = SpanCandidate(
                    candidate_id=f"cand_phone_{cand_id_counter:05d}",
                    target_node_start=node.node_id,
                    target_node_end=node.node_id,
                    covered_nodes=[node.node_id],
                    source_content_id=ev.source_content_id,
                    speaker_id=ev.speaker_id,
                    start_sample=p_start_sample,
                    end_sample=p_end_sample,
                    event_ids=[ev.event_id],
                    span_type="phone",
                    matched_word=node.word_text,
                    entry_variant_samples=entry_variant,
                    exit_variant_samples=exit_variant,
                    target_cost=0.15 + dur_cost + bias_adj + ctx_cost + stress_cost + swallow_cost + flap_penalty + coda_penalty + r_bleed_cost + l_bleed_cost,
                    quality_cost=0.0,
                    realized_phones=[ev.label],
                )
                cand_id_counter += 1
                phone_node_cands.append(c)

            if phone_node_cands:
                span_pools.append(phone_node_cands)

        # Fix P3.1: Pre-DP Continuation Linkage Audit across adjacent graph spans
        node_succs: Dict[int, List[int]] = {}
        for u, v in graph.edges:
            node_succs.setdefault(u, []).append(v)

        node_preds: Dict[int, List[int]] = {}
        for u, v in graph.edges:
            node_preds.setdefault(v, []).append(u)

        all_pre_dp_candidates = list(candidates)
        for pool in span_pools:
            all_pre_dp_candidates.extend(pool)

        exit_sample_map: Dict[Tuple[int, str, int], List[SpanCandidate]] = {}
        exit_event_map: Dict[Tuple[int, str, str], List[SpanCandidate]] = {}

        for c in all_pre_dp_candidates:
            u = c.target_node_end
            exit_sample_map.setdefault((u, c.source_content_id, c.end_sample), []).append(c)
            if c.event_ids:
                last_ev = self.inv.events.get(c.event_ids[-1])
                if last_ev and last_ev.successor_id:
                    exit_event_map.setdefault((u, c.source_content_id, last_ev.successor_id), []).append(c)

        continuation_candidates: Set[str] = set()
        for succ in all_pre_dp_candidates:
            v = succ.target_node_start
            possible_preds = list(node_preds.get(v, []))
            if succ.span_type in ("subword", "phone"):
                # Diphone joins allow transitions on identical boundary node
                possible_preds.append(v)

            for u in possible_preds:
                matching_preds = list(exit_sample_map.get((u, succ.source_content_id, succ.start_sample), []))
                if succ.event_ids:
                    matching_preds.extend(exit_event_map.get((u, succ.source_content_id, succ.event_ids[0]), []))

                for pred in matching_preds:
                    if pred.candidate_id == succ.candidate_id:
                        continue
                    if pred.start_sample == succ.start_sample and pred.end_sample == succ.end_sample:
                        continue
                    if pred.end_sample != succ.start_sample:
                        continue

                    # Intra-word fragmentation guard: subwords/phones carved from same donor word
                    # are penalized in DP rather than treated as 0-cost continuations
                    if (
                        pred.event_ids
                        and succ.event_ids
                        and (pred.span_type != "word" or succ.span_type != "word")
                    ):
                        p_last_ev = self.inv.events.get(pred.event_ids[-1])
                        s_first_ev = self.inv.events.get(succ.event_ids[0])
                        if (
                            p_last_ev
                            and s_first_ev
                            and p_last_ev.word_occurrence_id is not None
                            and p_last_ev.word_occurrence_id == s_first_ev.word_occurrence_id
                        ):
                            continue

                    continuation_candidates.add(pred.candidate_id)
                    continuation_candidates.add(succ.candidate_id)

        # For each span pool: keep all continuations, and top max_candidates_per_span of non-continuations
        for pool in span_pools:
            continuations = [c for c in pool if c.candidate_id in continuation_candidates]
            others = [c for c in pool if c.candidate_id not in continuation_candidates]
            surviving = continuations + others[:self.max_candidates_per_span]

            for c in surviving:
                feat = self._get_candidate_acoustic_features(c.source_content_id, c.start_sample, c.end_sample)
                if feat.f0_mean > 0:
                    c.cand_f0_hz = feat.f0_mean
                    ref_f0 = self.speaker_median_f0.get(c.speaker_id, self.global_median_f0)
                    c.pitch_semitones = float(12.0 * np.log2(feat.f0_mean / max(50.0, ref_f0)))

                if c.span_type == "phone":
                    node = graph.nodes[c.target_node_start]
                    base_p = node.base_phone
                    if base_p in ("Z", "S", "SH", "ZH", "V", "F", "TH", "DH", "CH", "JH") or base_p in vowel_set:
                        min_rms = 0.016
                        if 0.0 < feat.rms_energy < min_rms:
                            c.target_cost += 0.6 * (min_rms - feat.rms_energy) / min_rms

                candidates.append(c)

        return candidates

    def _get_candidate_acoustic_features(
        self,
        source_content_id: str,
        start_sample: int,
        end_sample: int,
    ) -> AcousticMetrics:
        """
        Returns AcousticMetrics (f0_slope, rms_energy, f0_mean) for the audio interval.
        Caches results by (source_content_id, start_sample, end_sample).
        Computes voiced F0 slope via degree-1 polynomial regression (np.polyfit)
        over voiced frames with >= 3 valid frames (confidence > 0.5); else 0.0 (Fix P3.4).
        """
        cache_key = (source_content_id, start_sample, end_sample)
        if cache_key in self._acoustic_metrics_cache:
            return self._acoustic_metrics_cache[cache_key]

        src_rec = self.inv.sources.get(source_content_id)
        if not src_rec:
            return AcousticMetrics(0.0, 0.0, 0.0)

        # Fast-path: read from preloaded feature tracks (zero disk seek)
        tracks = self._preloaded_tracks.get(source_content_id)
        if tracks and "rms" in tracks and "f0" in tracks and "n_frames" in tracks:
            hop = int(tracks.get("hop_length", 160))
            tr_sr = int(tracks.get("sr", 16000))
            native_sr = src_rec.native_sample_rate or 48000
            s_frame = min(max(0, int(round((start_sample / float(native_sr)) * tr_sr / hop))), max(0, int(tracks["n_frames"]) - 1))
            e_frame = min(int(tracks["n_frames"]), max(s_frame + 1, int(round((end_sample / float(native_sr)) * tr_sr / hop))))
            if e_frame > s_frame:
                rms_slice = tracks["rms"][s_frame:e_frame]
                f0_slice = tracks["f0"][s_frame:e_frame]
                vf_slice = tracks["voiced_flag"][s_frame:e_frame] if "voiced_flag" in tracks else np.ones_like(f0_slice, dtype=bool)
                vp_slice = tracks["voiced_probs"][s_frame:e_frame] if "voiced_probs" in tracks else np.ones_like(f0_slice, dtype=float)
                rms_val = float(np.sqrt(np.mean(rms_slice ** 2))) if len(rms_slice) > 0 else 0.0
                valid_pts = [
                    (idx * (hop / float(tr_sr)), float(f))
                    for idx, (vf, vp, f) in enumerate(zip(vf_slice, vp_slice, f0_slice))
                    if vf and vp > 0.5 and f is not None and not np.isnan(f) and 80.0 <= f <= 500.0
                ]
                if len(valid_pts) >= 3:
                    t_v = np.array([pt[0] for pt in valid_pts], dtype=float)
                    f_v = np.array([pt[1] for pt in valid_pts], dtype=float)
                    f0_slope = float(np.polyfit(t_v, f_v, 1)[0])
                    f0_mean = float(np.mean(f_v))
                elif len(valid_pts) > 0:
                    f0_slope = 0.0
                    f0_mean = float(np.mean([pt[1] for pt in valid_pts]))
                else:
                    f0_slope, f0_mean = 0.0, 0.0
                res = AcousticMetrics(f0_slope, rms_val, f0_mean)
                self._acoustic_metrics_cache[cache_key] = res
                return res

        pcm_path = None
        for cand_path in [src_rec.cached_analysis_pcm_path, src_rec.cached_pcm_path, src_rec.source_path]:
            if cand_path and Path(cand_path).exists():
                pcm_path = Path(cand_path)
                break

        if not pcm_path:
            return AcousticMetrics(0.0, 0.0, 0.0)

        try:
            import soundfile as sf
            import librosa

            info = sf.info(str(pcm_path))
            native_sr = src_rec.native_sample_rate or info.samplerate or 16000

            if info.samplerate != native_sr and native_sr > 0:
                s_file = int(round(start_sample * info.samplerate / native_sr))
                e_file = int(round(end_sample * info.samplerate / native_sr))
            else:
                s_file = start_sample
                e_file = end_sample

            s_file = max(0, min(info.frames, s_file))
            e_file = max(0, min(info.frames, e_file))
            if e_file <= s_file:
                return AcousticMetrics(0.0, 0.0, 0.0)

            audio_slice, file_sr = sf.read(str(pcm_path), start=s_file, stop=e_file, dtype="float32")
            if audio_slice.ndim > 1:
                audio_slice = np.mean(audio_slice, axis=1)

            if len(audio_slice) == 0:
                return AcousticMetrics(0.0, 0.0, 0.0)

            rms = float(np.sqrt(np.mean(audio_slice ** 2)))

            if file_sr != 16000 and len(audio_slice) > 10:
                audio_slice = librosa.resample(audio_slice, orig_sr=file_sr, target_sr=16000)

            f0_slope = 0.0
            f0_mean = 0.0
            if len(audio_slice) >= 320:
                try:
                    f0, voiced_flag, voiced_probs = librosa.pyin(
                        audio_slice, fmin=80, fmax=500, sr=16000, hop_length=160
                    )
                except Exception:
                    f0_yin = librosa.yin(audio_slice, fmin=80, fmax=500, sr=16000, hop_length=160)
                    voiced_flag = np.array([80 <= f <= 500 for f in f0_yin], dtype=bool)
                    voiced_probs = np.where(voiced_flag, 1.0, 0.0)
                    f0 = f0_yin

                valid_indices = [
                    i for i, (vf, vp, f) in enumerate(zip(voiced_flag, voiced_probs, f0))
                    if vf and vp > 0.5 and f is not None and not np.isnan(f) and 80 <= f <= 500
                ]

                if len(valid_indices) >= 3:
                    t_voiced = np.array([i * (160.0 / 16000.0) for i in valid_indices], dtype=float)
                    f0_voiced = np.array([float(f0[i]) for i in valid_indices], dtype=float)
                    f0_slope = float(np.polyfit(t_voiced, f0_voiced, 1)[0])
                    f0_mean = float(np.mean(f0_voiced))
                elif len(valid_indices) > 0:
                    f0_slope = 0.0
                    f0_mean = float(np.mean([float(f0[i]) for i in valid_indices]))
                else:
                    f0_slope = 0.0
                    f0_mean = 0.0

            res = AcousticMetrics(f0_slope, rms, f0_mean)
            self._acoustic_metrics_cache[cache_key] = res
            return res
        except Exception:
            return AcousticMetrics(0.0, 0.0, 0.0)

    def _compute_prosodic_bias_adjustment(
        self,
        covered_node_ids: List[int],
        source_content_id: str,
        start_sample: int,
        end_sample: int,
        graph: TargetGraph,
    ) -> float:
        """
        Computes Viterbi target cost adjustments based on punctuation-driven prosodic bias:
        - Question ('?'): rewards rising pitch slope (f0_slope > 0) and penalizes falling pitch.
        - Exclamation ('!'): rewards high RMS energy and primary lexical stress.
        """
        if not covered_node_ids:
            return 0.0

        last_node_idx = covered_node_ids[-1]
        if last_node_idx >= len(graph.nodes):
            return 0.0
        last_node = graph.nodes[last_node_idx]

        bias = last_node.prosodic_bias
        if not bias and last_node.is_phrase_final:
            if graph.terminal_punctuation == "?":
                bias = "question"
            elif graph.terminal_punctuation == "!":
                bias = "exclamation"

        if not bias:
            return 0.0

        # Prosodic pitch and RMS biases are word/clause-level phenomena.
        # Vowels in isolated subwords or phones have naturally higher RMS than whole words.
        # Subwords and phones must not receive artificial RMS or stress bonuses over intact words.
        first_node = graph.nodes[covered_node_ids[0]]
        is_partial_word = not (first_node.is_word_initial and last_node.is_word_final)
        if is_partial_word:
            return 0.0

        f0_slope, rms = self._get_candidate_acoustic_features(source_content_id, start_sample, end_sample)

        cost_adj = 0.0
        if bias == "question":
            if f0_slope > 20.0:
                cost_adj -= min(3.0, f0_slope * 0.015)
            elif f0_slope < -20.0:
                cost_adj += 2.0

        elif bias == "exclamation":
            has_primary_stress = any(
                graph.nodes[nid].stress == 1 for nid in covered_node_ids if nid < len(graph.nodes)
            )
            if has_primary_stress:
                cost_adj -= 1.5
            if rms > 0.08:
                cost_adj -= min(2.0, (rms - 0.08) * 15.0)
            elif 0.0 < rms < 0.03:
                cost_adj += 1.0

        return cost_adj

    def _compute_context_affinity(
        self,
        target_left_phone: Union[str, Set[str], Sequence[str]],
        target_right_phone: Union[str, Set[str], Sequence[str]],
        donor_left_ev: Optional[Any],
        donor_right_ev: Optional[Any],
    ) -> float:
        """
        Computes coarticulation context affinity between target phonetic environment and donor unit environment.
        Accepts strings, sets, or sequences of valid context phones across multi-branch lattices (Fix P3.4).
        Returns cost adjustment:
          - Negative (discount) if left and/or right phonetic context matches (triphone / diphone match).
          - Positive (penalty) if contexts are non-empty and mismatch.
        """
        target_left_set = {target_left_phone} if isinstance(target_left_phone, str) else (set(target_left_phone) if target_left_phone is not None else set())
        target_left_bases = {strip_phone_stress(p) for p in target_left_set if p}
        target_right_set = {target_right_phone} if isinstance(target_right_phone, str) else (set(target_right_phone) if target_right_phone is not None else set())
        target_right_bases = {strip_phone_stress(p) for p in target_right_set if p}

        left_match = False
        if target_left_bases and donor_left_ev is not None:
            if getattr(donor_left_ev, "kind", "") == "phone":
                d_left_base = getattr(donor_left_ev, "base_phone", "") or strip_phone_stress(getattr(donor_left_ev, "label", ""))
                left_match = (d_left_base in target_left_bases)
            elif getattr(donor_left_ev, "kind", "") == "silence":
                left_match = any(b in ("sil", "sp", "") for b in target_left_bases)

        right_match = False
        if target_right_bases and donor_right_ev is not None:
            if getattr(donor_right_ev, "kind", "") == "phone":
                d_right_base = getattr(donor_right_ev, "base_phone", "") or strip_phone_stress(getattr(donor_right_ev, "label", ""))
                right_match = (d_right_base in target_right_bases)
            elif getattr(donor_right_ev, "kind", "") == "silence":
                right_match = any(b in ("sil", "sp", "") for b in target_right_bases)

        stop_consonants = {"P", "T", "K", "B", "D", "G"}
        d_l_base = (getattr(donor_left_ev, "base_phone", "") or strip_phone_stress(getattr(donor_left_ev, "label", ""))) if donor_left_ev else ""
        severe_l_mismatch = (d_l_base in stop_consonants and not any(b in stop_consonants for b in target_left_bases))

        d_r_base = (getattr(donor_right_ev, "base_phone", "") or strip_phone_stress(getattr(donor_right_ev, "label", ""))) if donor_right_ev else ""
        severe_r_mismatch = (d_r_base in stop_consonants and not any(b in stop_consonants for b in target_right_bases))

        has_left = bool(target_left_bases)
        has_right = bool(target_right_bases)

        if has_left and has_right:
            if left_match and right_match:
                return -0.08  # Triphone match
            elif left_match or right_match:
                return -0.04  # Diphone match
            else:
                return 0.35 if (severe_r_mismatch or severe_l_mismatch) else 0.10   # Context mismatch
        elif has_left:
            if severe_l_mismatch:
                return 0.35
            return -0.04 if left_match else 0.05
        elif has_right:
            if severe_r_mismatch:
                return 0.35
            return -0.04 if right_match else 0.05
        elif severe_l_mismatch:
            return 0.35
        return 0.0

    def _compute_swallowed_penalty(self, event_ids: List[str], s_rate: int) -> float:
        """
        Penalizes heavily truncated or swallowed constituent phonemes:
        - Vowels < 50ms
        - Consonants < 35ms
        """
        penalty = 0.0
        phone_count = 0
        vowels = ("IY", "IH", "EY", "EH", "AE", "AA", "AO", "OW", "UH", "UW", "ER", "AY", "AW", "OY")
        for eid in event_ids:
            ev = self.inv.events.get(eid)
            if ev and ev.kind == "phone":
                phone_count += 1
                dur = (ev.end_sample - ev.start_sample) / float(s_rate)
                if ev.base_phone in vowels:
                    if dur < 0.050:
                        penalty += 2.5 * (0.050 - dur) / 0.050
                else:
                    if dur < 0.035:
                        penalty += 2.5 * (0.035 - dur) / 0.035
        if phone_count > 0:
            return penalty / phone_count
        return penalty

    def is_source_continuation(self, pred: SpanCandidate, curr: SpanCandidate) -> bool:
        """
        True only if curr immediately follows pred in the source recording without cuts.
        Requires both physical sample abutment and event successor linkage (when event_ids exist).
        """
        if pred.source_content_id != curr.source_content_id:
            return False

        # Invariant: reusing the identical source range is an edit, not a continuation!
        if pred.start_sample == curr.start_sample and pred.end_sample == curr.end_sample:
            return False

        # Physical sample abutment is mandatory
        if pred.end_sample != curr.start_sample:
            return False

        # Check event successor linkage when event IDs are present
        if pred.event_ids and curr.event_ids:
            last_ev = self.inv.events.get(pred.event_ids[-1])
            first_ev = self.inv.events.get(curr.event_ids[0])
            if not (last_ev and first_ev and last_ev.successor_id == first_ev.event_id):
                return False

        return True

    def _get_boundary_embedding(
        self,
        source_content_id: str,
        start_sample: int,
        end_sample: int,
        wavlm_backend: Any,
    ) -> Optional[np.ndarray]:
        """Extracts and caches pooled WavLM embedding for an audio interval on demand."""
        cache_key = (source_content_id, start_sample, end_sample)
        if cache_key in self._wavlm_cache:
            return self._wavlm_cache[cache_key]

        src_rec = self.inv.sources.get(source_content_id)
        if not src_rec or not src_rec.cached_analysis_pcm_path:
            return None

        p = Path(src_rec.cached_analysis_pcm_path)
        if not p.exists():
            return None

        native_sr = src_rec.native_sample_rate
        s_16k = int(round(start_sample * 16000 / native_sr))
        e_16k = int(round(end_sample * 16000 / native_sr))
        if e_16k <= s_16k:
            e_16k = s_16k + 1600

        import soundfile as sf
        try:
            audio_slice, sr = sf.read(str(p), start=max(0, s_16k), stop=e_16k, dtype="float32")
        except Exception:
            return None

        min_len = 1600
        if len(audio_slice) < min_len:
            audio_slice = np.pad(audio_slice, (0, min_len - len(audio_slice)))

        dense = wavlm_backend.extract_dense_embeddings(audio_slice, sample_rate=16000)
        pooled = np.mean(dense, axis=0)
        norm = float(np.linalg.norm(pooled))
        emb = (pooled / norm) if norm > 0 else pooled
        self._wavlm_cache[cache_key] = emb
        return emb

    def _get_candidate_boundary_feature(
        self,
        cand: SpanCandidate,
        is_exit: bool,
        feature_extractor: Optional[Any],
    ) -> Optional[Any]:
        """Returns boundary FeatureRecord for candidate exit or entry."""
        if feature_extractor is None:
            return None
        if isinstance(feature_extractor, dict):
            key = (cand.source_content_id, cand.end_sample if is_exit else cand.start_sample, is_exit)
            if key in feature_extractor:
                return feature_extractor[key]
            key_cand = (cand.candidate_id, "exit" if is_exit else "entry")
            if key_cand in feature_extractor:
                return feature_extractor[key_cand]
            return None
        if hasattr(feature_extractor, "get_boundary_features"):
            return feature_extractor.get_boundary_features(cand, is_exit=is_exit)
        if hasattr(feature_extractor, "extract_tracks") and hasattr(feature_extractor, "get_window_features"):
            src_rec = self.inv.sources.get(cand.source_content_id)
            if not src_rec:
                return None
            pcm_path = src_rec.cached_analysis_pcm_path or src_rec.source_path
            if not pcm_path or not Path(pcm_path).exists():
                return None
            if not hasattr(self, "_feature_tracks"):
                self._feature_tracks: Dict[str, Any] = {}
            if cand.source_content_id not in self._feature_tracks:
                try:
                    self._feature_tracks[cand.source_content_id] = feature_extractor.extract_tracks(pcm_path)
                except Exception:
                    return None
            tracks = self._feature_tracks.get(cand.source_content_id)
            if not tracks:
                return None
            window_samples = int(round(0.020 * src_rec.native_sample_rate))
            if is_exit:
                w_start = max(cand.start_sample, cand.end_sample - window_samples)
                w_end = cand.end_sample
            else:
                w_start = cand.start_sample
                w_end = min(cand.end_sample, cand.start_sample + window_samples)
            return feature_extractor.get_window_features(
                tracks, start_sample=w_start, end_sample=w_end, native_sr=src_rec.native_sample_rate, source_content_id=cand.source_content_id
            )
        return None

    def _compute_guide_prosody_cost(
        self,
        cand: SpanCandidate,
        graph: TargetGraph,
        prosody_guide: Optional[Any],
    ) -> float:
        if prosody_guide is None:
            return 0.0
        if hasattr(prosody_guide, "get_candidate_cost"):
            try:
                return float(prosody_guide.get_candidate_cost(cand, graph, selector=self))
            except TypeError:
                try:
                    return float(prosody_guide.get_candidate_cost(cand, graph))
                except TypeError:
                    return float(prosody_guide.get_candidate_cost(cand))
        if isinstance(prosody_guide, dict):
            return float(prosody_guide.get(cand.candidate_id, 0.0))
        return 0.0

    def select(
        self,
        graph: TargetGraph,
        speaker_id: Optional[str] = None,
        feature_extractor: Optional[Any] = None,
        wavlm_backend: Optional[Any] = None,
        prosody_guide: Optional[Any] = None,
    ) -> List[SpanCandidate]:
        """
        Finds the globally optimal sequence of candidates covering the TargetGraph
        using dynamic programming / Viterbi search with optional WavLM / prosody scoring.
        """
        candidates = self.retrieve_candidates(graph, speaker_id=speaker_id)
        if not candidates:
            raise ValueError(f"No unit candidates found to synthesize target sentence: '{graph.raw_text}'")

        phrase_cands = sum(1 for c in candidates if c.span_type == "phrase")
        word_cands = sum(1 for c in candidates if c.span_type == "word")
        subword_cands = sum(1 for c in candidates if c.span_type == "subword")
        phone_cands = sum(1 for c in candidates if c.span_type == "phone")
        print(f"[+] Retrieved {len(candidates)} candidate units (phrases: {phrase_cands}, words: {word_cands}, subwords: {subword_cands}, phones: {phone_cands})")
        print(f"[*] Optimizing path through {len(graph.nodes)} target phoneme nodes via Viterbi DP...")

        # Organize candidates by start node
        cands_by_start: Dict[int, List[SpanCandidate]] = {}
        for c in candidates:
            cands_by_start.setdefault(c.target_node_start, []).append(c)

        # Graph node successor map
        node_succs: Dict[int, List[int]] = {}
        for u, v in graph.edges:
            node_succs.setdefault(u, []).append(v)

        # dp[candidate_id] = (best_cumulative_cost, predecessor_candidate)
        dp: Dict[str, float] = {}
        parent: Dict[str, Optional[SpanCandidate]] = {}

        # Initialize with candidates starting at start_nodes
        start_set = set(graph.start_nodes)
        initial_candidates = [c for c in candidates if c.target_node_start in start_set]

        if not initial_candidates:
            raise ValueError("No candidates cover the start of the target utterance")

        for c in initial_candidates:
            guide_cost = self._compute_guide_prosody_cost(c, graph, prosody_guide)
            init_cost = c.target_cost + c.quality_cost + guide_cost
            dp[c.candidate_id] = init_cost
            parent[c.candidate_id] = None

        # Boundary feature and embedding memoization
        feat_exit_cache: Dict[str, Any] = {}
        feat_entry_cache: Dict[str, Any] = {}
        emb_exit_cache: Dict[str, Any] = {}
        emb_entry_cache: Dict[str, Any] = {}

        def _get_exit_feat(cand: SpanCandidate) -> Any:
            if cand.candidate_id not in feat_exit_cache:
                feat_exit_cache[cand.candidate_id] = self._get_candidate_boundary_feature(
                    cand, is_exit=True, feature_extractor=feature_extractor
                )
            return feat_exit_cache[cand.candidate_id]

        def _get_entry_feat(cand: SpanCandidate) -> Any:
            if cand.candidate_id not in feat_entry_cache:
                feat_entry_cache[cand.candidate_id] = self._get_candidate_boundary_feature(
                    cand, is_exit=False, feature_extractor=feature_extractor
                )
            return feat_entry_cache[cand.candidate_id]

        def _get_tail_emb(cand: SpanCandidate) -> Any:
            if cand.candidate_id not in emb_exit_cache:
                src_rec = self.inv.sources.get(cand.source_content_id)
                sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else 48000
                t_len = min(cand.end_sample - cand.start_sample, int(round(0.025 * sr)))
                emb_exit_cache[cand.candidate_id] = self._get_boundary_embedding(
                    cand.source_content_id, cand.end_sample - t_len, cand.end_sample, wavlm_backend
                )
            return emb_exit_cache[cand.candidate_id]

        def _get_head_emb(cand: SpanCandidate) -> Any:
            if cand.candidate_id not in emb_entry_cache:
                src_rec = self.inv.sources.get(cand.source_content_id)
                sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else 48000
                h_len = min(cand.end_sample - cand.start_sample, int(round(0.025 * sr)))
                emb_entry_cache[cand.candidate_id] = self._get_boundary_embedding(
                    cand.source_content_id, cand.start_sample, cand.start_sample + h_len, wavlm_backend
                )
            return emb_entry_cache[cand.candidate_id]

        # Topological DP forward pass across target phoneme nodes
        for node_id in range(len(graph.nodes)):
            curr_cands = cands_by_start.get(node_id, [])
            if not curr_cands:
                continue

            # Sort single-node candidates (target_node_end == target_node_start) BEFORE multi-node
            # candidates so phone [u, u] -> subword [u, u+k] diphone transitions at node u
            # populate dp[d_succ] before d_succ is expanded in this same node_id iteration.
            curr_cands_ordered = sorted(
                curr_cands,
                key=lambda c: (0 if c.target_node_end == c.target_node_start else 1, c.target_node_end),
            )

            for curr in curr_cands_ordered:
                if curr.candidate_id not in dp:
                    continue
                curr_cost = dp[curr.candidate_id]
                next_target_nodes = node_succs.get(curr.target_node_end, [])
                candidates_to_eval: List[Tuple[SpanCandidate, bool]] = []
                for nxt_node in next_target_nodes:
                    for succ in cands_by_start.get(nxt_node, []):
                        candidates_to_eval.append((succ, False))

                # Fix P2.2: Diphone joins inside synthesized words (subword<->subword, subword->phone, phone->subword)
                # Require at least one of (curr, d_succ) to span >1 node so [u, u] -> [u, u] self-loops cannot occur.
                if curr.span_type in ("subword", "phone"):
                    curr_is_multi = curr.target_node_end > curr.target_node_start
                    curr_node = graph.nodes[curr.target_node_end] if curr.target_node_end < len(graph.nodes) else None
                    p_curr_last = strip_phone_stress(curr.realized_phones[-1]) if curr.realized_phones else (curr_node.base_phone if curr_node else "")
                    for d_succ in cands_by_start.get(curr.target_node_end, []):
                        succ_is_multi = d_succ.target_node_end > d_succ.target_node_start
                        if (
                            d_succ.candidate_id != curr.candidate_id
                            and (curr_is_multi or succ_is_multi)
                            and d_succ.span_type in ("subword", "phone")
                        ):
                            d_node = graph.nodes[d_succ.target_node_start] if d_succ.target_node_start < len(graph.nodes) else None
                            if (
                                curr_node is not None
                                and d_node is not None
                                and curr_node.token_index == d_node.token_index
                            ):
                                p_succ_first = strip_phone_stress(d_succ.realized_phones[0]) if d_succ.realized_phones else d_node.base_phone
                                if p_curr_last and p_curr_last == p_succ_first:
                                    candidates_to_eval.append((d_succ, True))

                for succ, is_diphone in candidates_to_eval:
                    if self.is_source_continuation(curr, succ):
                        # Intra-word fragmentation guard: if curr and succ are subwords or phones
                        # carved from the same donor word, charging 0 transition cost rewards
                        # artificial fragmentation over keeping the multi-phone unit intact.
                        c_last_ev = self.inv.events.get(curr.event_ids[-1]) if curr.event_ids else None
                        s_first_ev = self.inv.events.get(succ.event_ids[0]) if succ.event_ids else None
                        same_donor_word = (
                            c_last_ev is not None
                            and s_first_ev is not None
                            and c_last_ev.word_occurrence_id is not None
                            and c_last_ev.word_occurrence_id == s_first_ev.word_occurrence_id
                        )
                        if same_donor_word and (curr.span_type != "word" or succ.span_type != "word"):
                            trans_cost = max(0.40, self.edit_penalty)
                        else:
                            trans_cost = 0.0
                    else:
                        feat_exit = _get_exit_feat(curr)
                        feat_entry = _get_entry_feat(succ)
                        curr_exit_p = curr.realized_phones[-1] if curr.realized_phones else (graph.nodes[curr.target_node_end].phone if curr.target_node_end < len(graph.nodes) else None)
                        succ_entry_p = succ.realized_phones[0] if succ.realized_phones else (graph.nodes[succ.target_node_start].phone if succ.target_node_start < len(graph.nodes) else None)
                        manner_scale = compute_manner_scale(curr_exit_p, succ_entry_p)
                        join_cost = compute_join_cost(
                            feat_exit,
                            feat_entry,
                            is_natural_continuation=False,
                            exit_phone=curr_exit_p,
                            entry_phone=succ_entry_p,
                            manner_scale=manner_scale,
                        )
                        if is_diphone:
                            trans_cost = (0.5 * self.edit_penalty) + (0.5 * join_cost)
                        else:
                            trans_cost = self.edit_penalty + join_cost
                        if wavlm_backend is not None:
                            tail_emb = _get_tail_emb(curr)
                            head_emb = _get_head_emb(succ)
                            if tail_emb is not None and head_emb is not None:
                                dist = wavlm_backend.compute_cosine_distance(tail_emb, head_emb)
                                trans_cost += 0.5 * manner_scale * dist

                        # Source continuity and surrounding scene coherence
                        src_bonus = 0.0
                        if curr.source_content_id == succ.source_content_id:
                            # Both cuts from same source video: background room acoustics,
                            # lighting, and microphone are continuous
                            src_bonus = 0.15
                            s_rec = self.inv.sources.get(curr.source_content_id)
                            s_rate = s_rec.native_sample_rate if s_rec and s_rec.native_sample_rate > 0 else 44100
                            if 0 <= succ.start_sample - curr.end_sample <= int(5.0 * s_rate):
                                src_bonus = 0.25
                        else:
                            # Sandwich splice: curr came from a different source,
                            # but predecessor was in succ's source in the same scene
                            prev_cand = parent.get(curr.candidate_id)
                            if prev_cand is not None and prev_cand.source_content_id == succ.source_content_id:
                                s_rec = self.inv.sources.get(succ.source_content_id)
                                s_rate = s_rec.native_sample_rate if s_rec and s_rec.native_sample_rate > 0 else 44100
                                if 0 <= succ.start_sample - prev_cand.end_sample <= int(5.0 * s_rate):
                                    src_bonus = 0.25

                        trans_cost = max(0.0, trans_cost - src_bonus)

                    # Unit reuse / source fatigue penalty: penalize repeating the exact same phone slice
                    reuse_penalty = 0.0
                    if succ.event_ids and (succ.span_type in ("phone", "subword")):
                        succ_eids = set(succ.event_ids)
                        p: Optional[SpanCandidate] = curr
                        depth = 0
                        while p is not None and depth < 6:
                            if p.event_ids and any(eid in succ_eids for eid in p.event_ids):
                                reuse_penalty += 0.25
                                break
                            p = parent.get(p.candidate_id)
                            depth += 1

                    guide_cost = self._compute_guide_prosody_cost(succ, graph, prosody_guide)
                    cand_cost = succ.target_cost + succ.quality_cost + guide_cost + reuse_penalty
                    new_cost = curr_cost + trans_cost + cand_cost

                    if succ.candidate_id not in dp or new_cost < dp[succ.candidate_id]:
                        dp[succ.candidate_id] = new_cost
                        parent[succ.candidate_id] = curr

        end_set = set(graph.end_nodes)
        terminal_candidates = [c for c in candidates if c.target_node_end in end_set and c.candidate_id in dp]
        if not terminal_candidates:
            raise ValueError(f"Could not find a continuous phonetic path covering target sentence: '{graph.raw_text}'")

        best_terminal_cand = min(terminal_candidates, key=lambda c: dp[c.candidate_id])

        # Backtrack optimal path
        path: List[SpanCandidate] = []
        curr_step: Optional[SpanCandidate] = best_terminal_cand
        while curr_step is not None:
            path.append(curr_step)
            curr_step = parent[curr_step.candidate_id]

        path.reverse()
        path = apply_diphone_midpoint_transitions(path, graph, self.inv)
        return path
