"""
Variable-span unit selector and global dynamic programming search.
Implements the Phrase -> Word -> Whole Phone hierarchy, protects natural source runs,
enforces legal contiguity, and performs deterministic shortest-path sequence optimization.
"""

from __future__ import annotations

import heapq
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from contracts import InventoryIndex, SpanCandidate, TargetGraph, WordOccurrence
from features import compute_join_cost


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

    def retrieve_candidates(self, graph: TargetGraph, speaker_id: Optional[str] = None) -> List[SpanCandidate]:
        """
        Retrieves Phrase, Word, and Phone candidates covering contiguous segments of the target graph.
        """
        candidates: List[SpanCandidate] = []
        cand_id_counter = 0

        # Map word_normalized to matching word occurrences in inventory
        # Filter by speaker if requested
        eligible_words: Dict[str, List[WordOccurrence]] = {}
        for w_norm, w_ids in self.inv.word_to_word_ids.items():
            matches = []
            for wid in w_ids:
                wo = self.inv.words.get(wid)
                if wo and (speaker_id is None or wo.speaker_id == speaker_id):
                    matches.append(wo)
            if matches:
                eligible_words[w_norm] = matches

        # Group graph nodes by word / token_index
        token_to_nodes: Dict[int, List[int]] = {}
        for n in graph.nodes:
            token_to_nodes.setdefault(n.token_index, []).append(n.node_id)

        n_tokens = len(graph.normalized_words)

        # 1. Multi-word phrase candidates (Level 1)
        for span_len in range(min(n_tokens, 4), 1, -1):
            for start_tok in range(n_tokens - span_len + 1):
                end_tok = start_tok + span_len - 1
                phrase_tokens = graph.normalized_words[start_tok:end_tok + 1]

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
                        if next_wo and next_wo.word_normalized == next_tok:
                            matched_wos.append(next_wo)
                            curr_wo = next_wo
                        else:
                            is_full_phrase = False
                            break

                    if is_full_phrase and len(matched_wos) == span_len:
                        # Collect all nodes across tokens start_tok..end_tok
                        covered = []
                        for t_idx in range(start_tok, end_tok + 1):
                            covered.extend(token_to_nodes[t_idx])

                        all_events = []
                        for wo in matched_wos:
                            all_events.extend(wo.phone_event_ids)

                        c = SpanCandidate(
                            candidate_id=f"cand_phrase_{cand_id_counter:05d}",
                            target_node_start=covered[0],
                            target_node_end=covered[-1],
                            covered_nodes=covered,
                            source_content_id=first_wo.source_content_id,
                            speaker_id=first_wo.speaker_id,
                            start_sample=matched_wos[0].start_sample,
                            end_sample=matched_wos[-1].end_sample,
                            event_ids=all_events,
                            span_type="phrase",
                            matched_word=" ".join(phrase_tokens),
                            target_cost=0.0,
                            quality_cost=0.0,
                        )
                        candidates.append(c)
                        cand_id_counter += 1

        # 2. Whole word candidates (Level 2)
        for t_idx in range(n_tokens):
            w_norm = graph.normalized_words[t_idx]
            target_nodes = token_to_nodes[t_idx]
            word_candidates = eligible_words.get(w_norm, [])

            for wo in word_candidates[:self.max_candidates_per_span]:
                c = SpanCandidate(
                    candidate_id=f"cand_word_{cand_id_counter:05d}",
                    target_node_start=target_nodes[0],
                    target_node_end=target_nodes[-1],
                    covered_nodes=target_nodes,
                    source_content_id=wo.source_content_id,
                    speaker_id=wo.speaker_id,
                    start_sample=wo.start_sample,
                    end_sample=wo.end_sample,
                    event_ids=wo.phone_event_ids,
                    span_type="word",
                    matched_word=w_norm,
                    target_cost=0.05,
                    quality_cost=0.0,
                )
                candidates.append(c)
                cand_id_counter += 1

        # 3. Whole phone candidates (Level 3 - fallback)
        for node in graph.nodes:
            matching_event_ids = self.inv.phone_to_event_ids.get(node.phone, [])
            if not matching_event_ids:
                matching_event_ids = self.inv.base_phone_to_event_ids.get(node.base_phone, [])

            # Filter by speaker if applicable
            valid_events = []
            for eid in matching_event_ids:
                ev = self.inv.events.get(eid)
                if ev and (speaker_id is None or ev.speaker_id == speaker_id):
                    valid_events.append(ev)

            # Rank by duration closeness to prior and take top candidates
            ranked = sorted(
                valid_events,
                key=lambda ev: abs((ev.end_sample - ev.start_sample) / self.inv.sources[ev.source_content_id].native_sample_rate - (node.duration_prior_sec or 0.1))
            )

            for ev in ranked[:self.max_candidates_per_span]:
                sr = self.inv.sources[ev.source_content_id].native_sample_rate
                dur = (ev.end_sample - ev.start_sample) / float(sr)
                prior = node.duration_prior_sec or 0.10
                dur_cost = 0.2 * min(abs(dur - prior) / prior, 3.0)

                c = SpanCandidate(
                    candidate_id=f"cand_phone_{cand_id_counter:05d}",
                    target_node_start=node.node_id,
                    target_node_end=node.node_id,
                    covered_nodes=[node.node_id],
                    source_content_id=ev.source_content_id,
                    speaker_id=ev.speaker_id,
                    start_sample=ev.start_sample,
                    end_sample=ev.end_sample,
                    event_ids=[ev.event_id],
                    span_type="phone",
                    matched_word=node.word_text,
                    target_cost=0.15 + dur_cost,
                    quality_cost=0.0,
                )
                candidates.append(c)
                cand_id_counter += 1

        return candidates

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

        # Priority queue for Dijkstra / topological DP: (cost, candidate_id)
        # Initialize with candidates starting at start_nodes
        start_set = set(graph.start_nodes)
        initial_candidates = [c for c in candidates if c.target_node_start in start_set]

        if not initial_candidates:
            raise ValueError("No candidates cover the start of the target utterance")

        pq: List[Tuple[float, str]] = []
        cand_map = {c.candidate_id: c for c in candidates}

        for c in initial_candidates:
            init_cost = c.target_cost + c.quality_cost
            dp[c.candidate_id] = init_cost
            parent[c.candidate_id] = None
            heapq.heappush(pq, (init_cost, c.candidate_id))

        best_terminal_cand: Optional[SpanCandidate] = None
        best_terminal_cost = float("inf")
        end_set = set(graph.end_nodes)

        while pq:
            cost, cid = heapq.heappop(pq)
            if cost > dp[cid]:
                continue

            curr = cand_map[cid]

            # If curr covers an end node of the target graph, consider as complete path
            if curr.target_node_end in end_set:
                if cost < best_terminal_cost:
                    best_terminal_cost = cost
                    best_terminal_cand = curr

            # Find next target nodes after curr.target_node_end
            next_target_nodes = node_succs.get(curr.target_node_end, [])
            for nxt_node in next_target_nodes:
                for succ in cands_by_start.get(nxt_node, []):
                    # Compute transition cost
                    is_cont = self.is_source_continuation(curr, succ)
                    if is_cont:
                        trans_cost = 0.0
                    else:
                        trans_cost = self.edit_penalty + compute_join_cost(None, None, is_natural_continuation=False)
                        if wavlm_backend is not None:
                            tail_len = min(curr.end_sample - curr.start_sample, 4800)
                            head_len = min(succ.end_sample - succ.start_sample, 4800)
                            tail_emb = self._get_boundary_embedding(
                                curr.source_content_id, curr.end_sample - tail_len, curr.end_sample, wavlm_backend
                            )
                            head_emb = self._get_boundary_embedding(
                                succ.source_content_id, succ.start_sample, succ.start_sample + head_len, wavlm_backend
                            )
                            if tail_emb is not None and head_emb is not None:
                                dist = wavlm_backend.compute_cosine_distance(tail_emb, head_emb)
                                trans_cost += 0.5 * dist

                    cand_cost = succ.target_cost + succ.quality_cost
                    new_cost = cost + trans_cost + cand_cost

                    if succ.candidate_id not in dp or new_cost < dp[succ.candidate_id]:
                        dp[succ.candidate_id] = new_cost
                        parent[succ.candidate_id] = curr
                        heapq.heappush(pq, (new_cost, succ.candidate_id))

        if best_terminal_cand is None:
            raise ValueError(f"Could not find a continuous phonetic path covering target sentence: '{graph.raw_text}'")

        # Backtrack optimal path
        path: List[SpanCandidate] = []
        curr_step: Optional[SpanCandidate] = best_terminal_cand
        while curr_step is not None:
            path.append(curr_step)
            curr_step = parent[curr_step.candidate_id]

        path.reverse()
        return path
