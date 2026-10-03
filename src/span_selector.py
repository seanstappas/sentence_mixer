"""
Variable-span unit selector and global dynamic programming search.
Implements the Phrase -> Word -> Whole Phone hierarchy, protects natural source runs,
enforces legal contiguity, and performs deterministic shortest-path sequence optimization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from contracts import InventoryIndex, SpanCandidate, TargetGraph, WordOccurrence
from features import compute_join_cost
from lexicon import strip_phone_stress, VOWEL_PHONES


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
        self._acoustic_metrics_cache: Dict[Tuple[str, int, int], Tuple[float, float]] = {}

        # Reverse pronunciation index from realized phones to intact recorded WordOccurrences
        self.pron_to_words: Dict[Tuple[str, ...], List[WordOccurrence]] = {}
        self.base_pron_to_words: Dict[Tuple[str, ...], List[WordOccurrence]] = {}
        for wo in self.inv.words.values():
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

    def retrieve_candidates(self, graph: TargetGraph, speaker_id: Optional[str] = None) -> List[SpanCandidate]:
        """
        Retrieves Phrase, Word, and Phone candidates covering contiguous segments of the target graph.
        """
        candidates: List[SpanCandidate] = []
        cand_id_counter = 0

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
                            graph.nodes[nid].duration_prior_sec or 0.10
                            for nid in covered if nid < len(graph.nodes)
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

                        if matched_wos[-1].phone_event_ids:
                            last_ev = self.inv.events.get(matched_wos[-1].phone_event_ids[-1])
                            if last_ev and last_ev.successor_id:
                                succ_ev = self.inv.events.get(last_ev.successor_id)
                                if succ_ev and succ_ev.kind == "phone":
                                    succ_base = getattr(succ_ev, "base_phone", "") or strip_phone_stress(succ_ev.label)
                                    if succ_base in ("P", "T", "K", "B", "D", "G"):
                                        collar = int(round(0.012 * s_rate))
                                        if phrase_end_sample - phrase_start_sample > 2 * collar:
                                            phrase_end_sample -= collar

                        # Guard 1: Pause / Gap & Phrase-Final Right Bleed Guard
                        phrase_right_bleed_cost = 0.0
                        end_node = graph.nodes[covered[-1]]
                        if end_node.is_phrase_final or end_node.right_context_phone == "sil":
                            last_wo = matched_wos[-1]
                            if last_wo.phone_event_ids:
                                last_ev = self.inv.events.get(last_wo.phone_event_ids[-1])
                                if last_ev and last_ev.successor_id:
                                    succ_ev = self.inv.events.get(last_ev.successor_id)
                                    if succ_ev and getattr(succ_ev, "kind", "") == "phone":
                                        phrase_right_bleed_cost = 0.35
                                        collar = int(round(0.015 * s_rate))
                                        if phrase_end_sample - phrase_start_sample > 2 * collar:
                                            phrase_end_sample -= collar

                        # Guard 2: Left Boundary Consonant / Nasal / Glide Bleed Guard
                        phrase_left_bleed_cost = 0.0
                        if first_wo.phone_event_ids:
                            first_ev = self.inv.events.get(first_wo.phone_event_ids[0])
                            if first_ev:
                                if first_ev.predecessor_id is None and getattr(first_ev, "kind", "") == "phone":
                                    phrase_left_bleed_cost = 0.50
                                    collar = int(round(0.020 * s_rate))
                                    if phrase_end_sample - phrase_start_sample > 2 * collar:
                                        phrase_start_sample += collar
                                elif first_ev.predecessor_id:
                                    pred_ev = self.inv.events.get(first_ev.predecessor_id)
                                    if pred_ev and getattr(pred_ev, "kind", "") == "phone":
                                        pred_base = getattr(pred_ev, "base_phone", "") or strip_phone_stress(pred_ev.label)
                                        first_node = graph.nodes[covered[0]]
                                        if first_node.base_phone in VOWEL_PHONES:
                                            if pred_base == "V":
                                                phrase_left_bleed_cost = 0.80
                                                collar = int(round(0.020 * s_rate))
                                                if phrase_end_sample - phrase_start_sample > 2 * collar:
                                                    phrase_start_sample += collar
                                            elif pred_base in ("Z", "ZH", "DH"):
                                                phrase_left_bleed_cost = 0.20
                                            elif pred_base in ("N", "M", "NG", "Y", "R", "W", "L", "ER") or pred_base in VOWEL_PHONES:
                                                phrase_left_bleed_cost = 0.50
                                                collar = int(round(0.020 * s_rate))
                                                if phrase_end_sample - phrase_start_sample > 2 * collar:
                                                    phrase_start_sample += collar
                                        elif pred_base in ("N", "M", "NG", "Y", "R", "W", "L", "ER") or pred_base in VOWEL_PHONES:
                                            phrase_left_bleed_cost = 0.50
                                            collar = int(round(0.020 * s_rate))
                                            if phrase_end_sample - phrase_start_sample > 2 * collar:
                                                phrase_start_sample += collar

                        bias_adj = self._compute_prosodic_bias_adjustment(
                            covered, first_wo.source_content_id, phrase_start_sample, phrase_end_sample, graph
                        )
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
                            target_cost=0.0 + bias_adj + phrase_realization_cost + enunciation_guard_cost + rush_penalty + swallowed_penalty + phrase_right_bleed_cost + phrase_left_bleed_cost,
                            quality_cost=0.0,
                            realized_phones=total_realized_phones,
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

                # Compute expected word duration from duration priors of constituent phone nodes
                expected_dur = sum(
                    graph.nodes[nid].duration_prior_sec or 0.10
                    for nid in branch_nodes if nid < len(graph.nodes)
                )

                target_word_text = graph.nodes[branch_nodes[0]].word_text.lower()
                target_word_in_inv = target_word_text in self.inv.word_to_word_ids

                end_node = graph.nodes[branch_nodes[-1]]
                start_node = graph.nodes[branch_nodes[0]]

                def _compute_word_bleed(wo_it: WordOccurrence, srate: int) -> Tuple[float, float]:
                    r_bleed = 0.0
                    l_bleed = 0.0
                    if (end_node.is_phrase_final or end_node.right_context_phone == "sil") and wo_it.phone_event_ids:
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
                    d_ratio = abs(w_dur - expected_dur) / max(expected_dur, 0.05)
                    swallow_c = self._compute_swallowed_penalty(wo_it.phone_event_ids, srate)
                    lex_c = 0.0
                    if wo_it.word_normalized.lower() != target_word_text:
                        lex_c = 0.35 if target_word_in_inv else 0.05
                    r_bl, l_bl = _compute_word_bleed(wo_it, srate)
                    return r_c + 0.35 * min(d_ratio, 2.0) + swallow_c + lex_c + r_bl + l_bl

                valid_branch_candidates_and_costs.sort(key=_word_candidate_sort_key)

                for wo, r_cost, r_phones in valid_branch_candidates_and_costs[:self.max_candidates_per_span]:
                    src_rec = self.inv.sources.get(wo.source_content_id)
                    s_rate = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else 44100
                    actual_dur = (wo.end_sample - wo.start_sample) / float(s_rate)
                    dur_cost = 0.35 * min(abs(actual_dur - expected_dur) / max(expected_dur, 0.05), 2.0)
                    swallow_cost = self._compute_swallowed_penalty(wo.phone_event_ids, s_rate)

                    lexical_sub_cost = 0.0
                    if wo.word_normalized.lower() != target_word_text:
                        lexical_sub_cost = 0.35 if target_word_in_inv else 0.05

                    w_end_sample = wo.end_sample
                    w_start_sample = wo.start_sample
                    r_bleed_cost, l_bleed_cost = _compute_word_bleed(wo, s_rate)

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
                        target_cost=0.05 + bias_adj + r_cost + dur_cost + swallow_cost + lexical_sub_cost + r_bleed_cost + l_bleed_cost,
                        quality_cost=0.0,
                        realized_phones=r_phones,
                    )
                    candidates.append(c)
                    cand_id_counter += 1

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
                            graph.nodes[nid].duration_prior_sec or 0.10
                            for nid in sub_nodes if nid < len(graph.nodes)
                        )
                        target_l_phone = graph.nodes[sub_nodes[0]].left_context_phone
                        target_r_phone = graph.nodes[sub_nodes[-1]].right_context_phone

                        end_node = graph.nodes[sub_nodes[-1]]
                        start_node = graph.nodes[sub_nodes[0]]

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
                            if (end_node.is_phrase_final or end_node.right_context_phone == "sil") and e_i_it == len(evs_all):
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
                            d_ratio = abs(dur_it - expected_dur) / max(expected_dur, 0.05)
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
                            return r_c + 0.25 * min(d_ratio, 2.0) + ctx_c + swallow_c + floor_pen + flap_c + r_bl + l_bl + coda_p + franken_p

                        all_matches.sort(key=_subword_sort_key)

                        for (wo, s_i, e_i), r_cost in all_matches[:self.max_candidates_per_span]:
                            evs = [self.inv.events[eid] for eid in wo.phone_event_ids if eid in self.inv.events and self.inv.events[eid].kind == "phone"]
                            sub_evs = evs[s_i:e_i]
                            src_rec = self.inv.sources.get(wo.source_content_id)
                            s_rate = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else 44100
                            actual_dur = (sub_evs[-1].end_sample - sub_evs[0].start_sample) / float(s_rate)
                            dur_cost = 0.25 * min(abs(actual_dur - expected_dur) / max(expected_dur, 0.05), 2.0)
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

                            if sub_evs[-1].successor_id:
                                succ_ev = self.inv.events.get(sub_evs[-1].successor_id)
                                if succ_ev and succ_ev.kind == "phone":
                                    succ_base = getattr(succ_ev, "base_phone", "") or strip_phone_stress(succ_ev.label)
                                    if succ_base in ("P", "T", "K", "B", "D", "G"):
                                        collar = int(round(0.012 * s_rate))
                                        if sub_end_sample - sub_start_sample > 2 * collar:
                                            sub_end_sample -= collar

                            if sub_evs[0].predecessor_id:
                                pred_ev = self.inv.events.get(sub_evs[0].predecessor_id)
                                if pred_ev and pred_ev.kind == "phone":
                                    pred_base = getattr(pred_ev, "base_phone", "") or strip_phone_stress(pred_ev.label)
                                    target_l_base = strip_phone_stress(target_l_phone) if target_l_phone else ""
                                    if pred_base in ("P", "T", "K", "B", "D", "G") and target_l_base not in ("P", "T", "K", "B", "D", "G"):
                                        collar = int(round(0.012 * s_rate))
                                        if sub_end_sample - sub_start_sample > 2 * collar:
                                            sub_start_sample += collar

                            if r_bl_cost > 0.0:
                                collar = int(round(0.015 * s_rate))
                                if sub_end_sample - sub_start_sample > 2 * collar:
                                    sub_end_sample -= collar

                            if l_bl_cost > 0.0:
                                collar_sec = 0.045 if l_bl_cost >= 0.80 else 0.020
                                collar = int(round(collar_sec * s_rate))
                                if sub_end_sample - sub_start_sample > 2 * collar:
                                    sub_start_sample += collar

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
                                target_cost=base_target_cost + bias_adj + r_cost + dur_cost + ctx_cost + swallow_cost + flap_penalty + r_bl_cost + l_bl_cost + coda_penalty + franken_penalty,
                                quality_cost=0.0,
                                realized_phones=[e.label for e in sub_evs],
                            )
                            candidates.append(c)
                            cand_id_counter += 1


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

            # Filter by speaker if applicable
            valid_events_and_costs: List[Tuple[Any, float]] = []
            for eid, stress_c in event_id_candidates:
                ev = self.inv.events.get(eid)
                if ev and getattr(ev, "is_eligible", True) and (speaker_id is None or ev.speaker_id == speaker_id):
                    valid_events_and_costs.append((ev, stress_c))

            target_l_phone = node.left_context_phone
            target_r_phone = node.right_context_phone

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

                if node.is_phrase_final or node.right_context_phone == "sil":
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

            def _phone_sort_key(item: Tuple[Any, float]) -> float:
                ev_it, stress_c = item
                sr_it = self.inv.sources[ev_it.source_content_id].native_sample_rate
                dur_it = (ev_it.end_sample - ev_it.start_sample) / float(sr_it)
                prior = node.duration_prior_sec or 0.10
                if node.base_phone in vowel_set:
                    if dur_it >= prior:
                        if dur_it <= 1.35 * prior:
                            d_cost = -0.05 if node.is_word_initial else 0.0
                        else:
                            d_cost = 0.5 * (dur_it - 1.35 * prior) / prior
                    else:
                        d_cost = 1.2 * (prior - dur_it) / prior
                else:
                    d_cost = abs(dur_it - prior) / prior
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

            for ev, stress_cost in ranked[:self.max_candidates_per_span]:
                sr = self.inv.sources[ev.source_content_id].native_sample_rate
                dur = (ev.end_sample - ev.start_sample) / float(sr)
                prior = node.duration_prior_sec or 0.10
                if node.base_phone in vowel_set:
                    if dur >= prior:
                        if dur <= 1.35 * prior:
                            dur_cost = -0.05 if node.is_word_initial else 0.0
                        else:
                            dur_cost = 0.15 * min((dur - 1.35 * prior) / prior, 2.0)
                    else:
                        dur_cost = 0.30 * min((prior - dur) / prior, 3.0)
                else:
                    dur_cost = 0.2 * min(abs(dur - prior) / prior, 3.0)

                p_start_sample = ev.start_sample
                p_end_sample = ev.end_sample
                coda_penalty, r_bleed_cost, l_bleed_cost = _compute_phone_guards(ev, sr)

                if ev.successor_id:
                    succ_ev = self.inv.events.get(ev.successor_id)
                    if succ_ev and succ_ev.kind == "phone":
                        succ_base = getattr(succ_ev, "base_phone", "") or strip_phone_stress(succ_ev.label)
                        if succ_base in ("P", "T", "K", "B", "D", "G"):
                            collar = int(round(0.012 * sr))
                            if p_end_sample - p_start_sample > 2 * collar:
                                p_end_sample -= collar

                if r_bleed_cost > 0.0:
                    collar = int(round(0.015 * sr))
                    if p_end_sample - p_start_sample > 2 * collar:
                        p_end_sample -= collar

                if ev.predecessor_id:
                    pred_ev = self.inv.events.get(ev.predecessor_id)
                    if pred_ev and pred_ev.kind == "phone":
                        pred_base = getattr(pred_ev, "base_phone", "") or strip_phone_stress(pred_ev.label)
                        target_l_base = strip_phone_stress(target_l_phone) if target_l_phone else ""
                        if pred_base in ("P", "T", "K", "B", "D", "G") and target_l_base not in ("P", "T", "K", "B", "D", "G"):
                            collar = int(round(0.012 * sr))
                            if p_end_sample - p_start_sample > 2 * collar:
                                p_start_sample += collar

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

                # Check phone audibility floor for voiced sounds and fricatives
                _, p_rms = self._get_candidate_acoustic_features(ev.source_content_id, p_start_sample, p_end_sample)
                p_energy_penalty = 0.0
                base_p = node.base_phone
                min_rms = 0.016
                if base_p in ("Z", "S", "SH", "ZH", "V", "F", "TH", "DH", "CH", "JH") or base_p in vowel_set:
                    if 0.0 < p_rms < min_rms:
                        p_energy_penalty = 0.6 * (min_rms - p_rms) / min_rms

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
                    target_cost=0.15 + dur_cost + bias_adj + p_energy_penalty + ctx_cost + stress_cost + swallow_cost + flap_penalty + coda_penalty + r_bleed_cost + l_bleed_cost,
                    quality_cost=0.0,
                    realized_phones=[ev.label],
                )
                candidates.append(c)
                cand_id_counter += 1

        return candidates

    def _get_candidate_acoustic_features(
        self,
        source_content_id: str,
        start_sample: int,
        end_sample: int,
    ) -> Tuple[float, float]:
        """
        Returns (f0_slope, rms_energy) for the audio interval.
        Caches results by (source_content_id, start_sample, end_sample).
        """
        cache_key = (source_content_id, start_sample, end_sample)
        if cache_key in self._acoustic_metrics_cache:
            return self._acoustic_metrics_cache[cache_key]

        src_rec = self.inv.sources.get(source_content_id)
        if not src_rec:
            return 0.0, 0.0

        pcm_path = None
        for cand_path in [src_rec.cached_analysis_pcm_path, src_rec.cached_pcm_path, src_rec.source_path]:
            if cand_path and Path(cand_path).exists():
                pcm_path = Path(cand_path)
                break

        if not pcm_path:
            return 0.0, 0.0

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
                return 0.0, 0.0

            audio_slice, file_sr = sf.read(str(pcm_path), start=s_file, stop=e_file, dtype="float32")
            if audio_slice.ndim > 1:
                audio_slice = np.mean(audio_slice, axis=1)

            if len(audio_slice) == 0:
                return 0.0, 0.0

            rms = float(np.sqrt(np.mean(audio_slice ** 2)))

            if file_sr != 16000 and len(audio_slice) > 10:
                audio_slice = librosa.resample(audio_slice, orig_sr=file_sr, target_sr=16000)

            f0_slope = 0.0
            if len(audio_slice) >= 320:
                f0 = librosa.yin(audio_slice, fmin=80, fmax=500, sr=16000)
                valid = [float(f) for f in f0 if f is not None and not np.isnan(f) and 80 <= f <= 500]
                if len(valid) >= 2:
                    dt = len(valid) * (len(audio_slice) / (16000.0 * len(f0)))
                    if dt > 0:
                        f0_slope = (valid[-1] - valid[0]) / dt

            self._acoustic_metrics_cache[cache_key] = (f0_slope, rms)
            return f0_slope, rms
        except Exception:
            return 0.0, 0.0

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
        target_left_phone: str,
        target_right_phone: str,
        donor_left_ev: Optional[Any],
        donor_right_ev: Optional[Any],
    ) -> float:
        """
        Computes coarticulation context affinity between target phonetic environment and donor unit environment.
        Returns cost adjustment:
          - Negative (discount) if left and/or right phonetic context matches (triphone / diphone match).
          - Positive (penalty) if contexts are non-empty and mismatch.
        """
        left_match = False
        if target_left_phone:
            t_left_base = strip_phone_stress(target_left_phone)
            if donor_left_ev is not None:
                if getattr(donor_left_ev, "kind", "") == "phone":
                    d_left_base = getattr(donor_left_ev, "base_phone", "") or strip_phone_stress(getattr(donor_left_ev, "label", ""))
                    left_match = (t_left_base == d_left_base)
                elif getattr(donor_left_ev, "kind", "") == "silence":
                    left_match = (t_left_base in ("sil", "sp", ""))

        right_match = False
        if target_right_phone:
            t_right_base = strip_phone_stress(target_right_phone)
            if donor_right_ev is not None:
                if getattr(donor_right_ev, "kind", "") == "phone":
                    d_right_base = getattr(donor_right_ev, "base_phone", "") or strip_phone_stress(getattr(donor_right_ev, "label", ""))
                    right_match = (t_right_base == d_right_base)
                elif getattr(donor_right_ev, "kind", "") == "silence":
                    right_match = (t_right_base in ("sil", "sp", ""))

        stop_consonants = {"P", "T", "K", "B", "D", "G"}
        d_l_base = getattr(donor_left_ev, "base_phone", "") if donor_left_ev else ""
        t_l_base = strip_phone_stress(target_left_phone) if target_left_phone else ""
        severe_l_mismatch = (d_l_base in stop_consonants and t_l_base not in stop_consonants)

        d_r_base = getattr(donor_right_ev, "base_phone", "") if donor_right_ev else ""
        t_r_base = strip_phone_stress(target_right_phone) if target_right_phone else ""
        severe_r_mismatch = (d_r_base in stop_consonants and t_r_base not in stop_consonants)

        if target_left_phone and target_right_phone:
            if left_match and right_match:
                return -0.08  # Triphone match
            elif left_match or right_match:
                return -0.04  # Diphone match
            else:
                return 0.35 if (severe_r_mismatch or severe_l_mismatch) else 0.10   # Context mismatch
        elif target_left_phone:
            if severe_l_mismatch:
                return 0.35
            return -0.04 if left_match else 0.05
        elif target_right_phone:
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
        Uses ordered event IDs and physical sample alignment.
        """
        if pred.source_content_id != curr.source_content_id:
            return False

        # Invariant: reusing the identical source range is an edit, not a continuation!
        if pred.start_sample == curr.start_sample and pred.end_sample == curr.end_sample:
            return False

        # Check physical sample adjacency
        if pred.end_sample == curr.start_sample:
            return True

        # Check event successor linkage
        if pred.event_ids and curr.event_ids:
            last_ev = self.inv.events.get(pred.event_ids[-1])
            first_ev = self.inv.events.get(curr.event_ids[0])
            if last_ev and first_ev and last_ev.successor_id == first_ev.event_id:
                return True

        return False

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
            window_samples = int(round(0.050 * src_rec.native_sample_rate))
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
                t_len = min(cand.end_sample - cand.start_sample, 4800)
                emb_exit_cache[cand.candidate_id] = self._get_boundary_embedding(
                    cand.source_content_id, cand.end_sample - t_len, cand.end_sample, wavlm_backend
                )
            return emb_exit_cache[cand.candidate_id]

        def _get_head_emb(cand: SpanCandidate) -> Any:
            if cand.candidate_id not in emb_entry_cache:
                h_len = min(cand.end_sample - cand.start_sample, 4800)
                emb_entry_cache[cand.candidate_id] = self._get_boundary_embedding(
                    cand.source_content_id, cand.start_sample, cand.start_sample + h_len, wavlm_backend
                )
            return emb_entry_cache[cand.candidate_id]

        # Topological DP forward pass across target phoneme nodes
        for node_id in range(len(graph.nodes)):
            curr_cands = cands_by_start.get(node_id, [])
            if not curr_cands:
                continue

            reachable = [c for c in curr_cands if c.candidate_id in dp]
            if not reachable:
                continue

            for curr in reachable:
                curr_cost = dp[curr.candidate_id]
                next_target_nodes = node_succs.get(curr.target_node_end, [])
                for nxt_node in next_target_nodes:
                    for succ in cands_by_start.get(nxt_node, []):
                        if self.is_source_continuation(curr, succ):
                            # Intra-word fragmentation guard: if curr and succ are subwords or phones
                            # carved from the same donor word, charging 0 transition cost rewards
                            # artificial fragmentation over keeping the multi-phone unit intact.
                            same_donor_word = (
                                bool(curr.event_ids and succ.event_ids)
                                and (self.inv.events.get(curr.event_ids[-1]).word_occurrence_id is not None)
                                and (self.inv.events.get(curr.event_ids[-1]).word_occurrence_id == self.inv.events.get(succ.event_ids[0]).word_occurrence_id)
                            )
                            if same_donor_word and (curr.span_type != "word" or succ.span_type != "word"):
                                trans_cost = max(0.40, self.edit_penalty)
                            else:
                                trans_cost = 0.0
                        else:
                            feat_exit = _get_exit_feat(curr)
                            feat_entry = _get_entry_feat(succ)
                            join_cost = compute_join_cost(feat_exit, feat_entry, is_natural_continuation=False)
                            trans_cost = self.edit_penalty + join_cost
                            if wavlm_backend is not None:
                                tail_emb = _get_tail_emb(curr)
                                head_emb = _get_head_emb(succ)
                                if tail_emb is not None and head_emb is not None:
                                    dist = wavlm_backend.compute_cosine_distance(tail_emb, head_emb)
                                    trans_cost += 0.5 * dist

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
        return path
