"""
Target pronunciation graph representation and construction.
Preserves sentence punctuation, word/phrase position, lexical stress,
alternative pronunciation realizations, and cross-word phonetic context.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

from contracts import TargetGraph, TargetPhoneNode
from lexicon import Lexicon, get_phone_stress, normalize_text_to_words, strip_phone_stress

# Robust duration priors by phone manner category (in seconds)
# Avoids any universal 90ms prior
DEFAULT_DURATION_PRIORS: Dict[str, float] = {
    # Vowels / Diphthongs
    "AA": 0.16, "AE": 0.17, "AH": 0.10, "AO": 0.16, "AW": 0.18, "AY": 0.18,
    "EH": 0.13, "ER": 0.14, "EY": 0.16, "IH": 0.09, "IY": 0.14, "OW": 0.17,
    "OY": 0.19, "UH": 0.10, "UW": 0.14,
    # Stops / Affricates
    "B": 0.06, "D": 0.06, "G": 0.06, "P": 0.08, "T": 0.07, "K": 0.08,
    "CH": 0.10, "JH": 0.09,
    # Fricatives
    "DH": 0.05, "F": 0.10, "S": 0.11, "SH": 0.11, "TH": 0.08, "V": 0.07,
    "Z": 0.09, "ZH": 0.09, "HH": 0.07,
    # Nasals / Liquids / Glides
    "M": 0.09, "N": 0.08, "NG": 0.09, "L": 0.09, "R": 0.09, "W": 0.08, "Y": 0.07,
}


class TargetGraphBuilder:
    """
    Builds a directed acyclic graph of legal phonetic paths for a target utterance.
    """
    def __init__(self, lexicon: Lexicon):
        self.lexicon = lexicon

    def build(self, target_text: str) -> TargetGraph:
        tokens_with_orig = normalize_text_to_words(target_text)
        if not tokens_with_orig:
            raise ValueError(f"No valid tokens found in target text: {target_text}")

        normalized_words: List[str] = []
        punctuation_pauses: Dict[int, float] = {}

        # Separate words from punctuation
        word_tokens: List[Tuple[str, str]] = []
        for orig_tok, norm_tok in tokens_with_orig:
            if norm_tok in ".,!?;:":
                # Assign pause to the preceding word
                w_idx = max(0, len(word_tokens) - 1)
                pause_sec = 0.45 if norm_tok in ".!?" else 0.25
                punctuation_pauses[w_idx] = pause_sec
            else:
                word_tokens.append((orig_tok, norm_tok))
                normalized_words.append(norm_tok)

        if not word_tokens:
            raise ValueError("Target text contains only punctuation")

        nodes: List[TargetPhoneNode] = []
        edges: List[Tuple[int, int]] = []

        # Track entry and exit node indices for each word
        # word_index -> List[Tuple[branch_start_node_id, branch_end_node_id]]
        word_branches: List[List[Tuple[int, int]]] = []

        for w_idx, (orig_w, norm_w) in enumerate(word_tokens):
            prons = self.lexicon.get_pronunciations(norm_w)
            if not prons:
                raise ValueError(f"No pronunciation could be determined for target word: '{norm_w}'")

            branches: List[Tuple[int, int]] = []
            for p_list in prons:
                branch_node_ids: List[int] = []
                for p_pos, phone in enumerate(p_list):
                    node_id = len(nodes)
                    base = strip_phone_stress(phone)
                    stress = get_phone_stress(phone)
                    prior_dur = DEFAULT_DURATION_PRIORS.get(base, 0.10)

                    node = TargetPhoneNode(
                        node_id=node_id,
                        token_index=w_idx,
                        word_text=norm_w,
                        phone=phone,
                        base_phone=base,
                        stress=stress,
                        is_word_initial=(p_pos == 0),
                        is_word_final=(p_pos == len(p_list) - 1),
                        is_phrase_final=(w_idx == len(word_tokens) - 1 and p_pos == len(p_list) - 1),
                        duration_prior_sec=prior_dur,
                    )
                    nodes.append(node)
                    branch_node_ids.append(node_id)

                    # Connect within branch
                    if p_pos > 0:
                        edges.append((branch_node_ids[p_pos - 1], node_id))

                if branch_node_ids:
                    branches.append((branch_node_ids[0], branch_node_ids[-1]))

            word_branches.append(branches)

        # Connect words across boundaries
        for w_idx in range(len(word_branches) - 1):
            curr_branches = word_branches[w_idx]
            next_branches = word_branches[w_idx + 1]

            for _, curr_exit in curr_branches:
                for next_entry, _ in next_branches:
                    edges.append((curr_exit, next_entry))

        # Assign context across nodes
        # Forward pass for right context, backward for left context
        edge_dict_fwd: Dict[int, List[int]] = {}
        edge_dict_bwd: Dict[int, List[int]] = {}
        for src, dst in edges:
            edge_dict_fwd.setdefault(src, []).append(dst)
            edge_dict_bwd.setdefault(dst, []).append(src)

        for node in nodes:
            # Right context
            succs = edge_dict_fwd.get(node.node_id, [])
            if succs:
                node.right_context_phone = nodes[succs[0]].phone
            # Left context
            preds = edge_dict_bwd.get(node.node_id, [])
            if preds:
                node.left_context_phone = nodes[preds[0]].phone

        start_nodes = [b[0] for b in word_branches[0]]
        end_nodes = [b[1] for b in word_branches[-1]]

        graph = TargetGraph(
            raw_text=target_text,
            normalized_words=normalized_words,
            nodes=nodes,
            edges=edges,
            start_nodes=start_nodes,
            end_nodes=end_nodes,
            punctuation_pauses=punctuation_pauses,
        )
        graph.validate()
        return graph
