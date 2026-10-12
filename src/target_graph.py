"""
Target pronunciation graph representation and construction.
Preserves sentence punctuation, word/phrase position, lexical stress,
alternative pronunciation realizations, and cross-word phonetic context.
Supports:
- Hybrid phonetic syntax: space-delimited ARPAbet tokens enclosed in braces (e.g. {B AE1 K AH0 N})
- Explicit cadence gaps: <gap:250ms> or <gap:1.5s>
- Punctuation-based cadence and intonation biasing (! and ?)
- Strict syntax validation (unclosed braces, invalid ARPAbet phones raise ValueError)
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from contracts import TargetGraph, TargetPhoneNode
from lexicon import Lexicon, get_phone_stress, strip_phone_stress, validate_phone, normalize_numerals
import math


# Robust duration priors by phone manner category (in seconds)
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


@dataclass
class HybridToken:
    kind: str  # "word", "phonetic", "gap", "punctuation"
    orig_text: str
    norm_text: str
    phones: Optional[List[str]] = None
    gap_seconds: Optional[float] = None


def tokenize_hybrid_syntax(text: str) -> List[HybridToken]:
    """
    Parses input text into typed tokens:
    - Words: 'Hello', 'Obi-Wan', "I'm"
    - Explicit phonetic groups: '{B AE1 K AH0 N}'
    - Explicit cadence gaps: '<gap:250ms>' or '<gap:1.5s>'
    - Punctuation marks: '.', ',', '!', '?', ';', ':'

    Raises ValueError on syntax violations (unmatched braces/brackets, invalid phones).
    """
    # Normalize typographical quotation marks
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    # Normalize unbracketed numerals to spoken words
    text = normalize_numerals(text)

    # Syntax boundary checks
    if text.count("{") != text.count("}"):
        raise ValueError(f"Unmatched braces in target text: '{text}'")
    if text.count("<") != text.count(">"):
        raise ValueError(f"Unmatched angle brackets in target text: '{text}'")

    # Scanner pattern matching discrete units in left-to-right order
    pattern = re.compile(
        r"(\{[^}]*\})|"                             # 1. Phonetic group: {B AE1 K AH0 N} or {}
        r"(<gap:[^>]+>)|"                           # 2. Explicit gap tag: <gap:250ms>
        r"([.,!?;:])|"                              # 3. Punctuation
        r"([A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*)|"     # 4. Standard lexical word
        r"(\S+)"                                    # 5. Any other non-whitespace (error candidate)
    )

    tokens: List[HybridToken] = []

    for m in pattern.finditer(text):
        phonetic, gap, punct, word, unrecognized = m.groups()

        if phonetic:
            content = phonetic[1:-1].strip()
            if not content:
                raise ValueError("Empty phonetic group: '{}'")
            raw_phones = content.split()
            valid_phones: List[str] = []
            for raw_p in raw_phones:
                p_up = raw_p.upper()
                try:
                    validate_phone(p_up)
                except ValueError as e:
                    raise ValueError(f"Invalid ARPAbet phone '{p_up}' in phonetic group '{phonetic}': {e}") from e
                valid_phones.append(p_up)

            norm_group = "{" + " ".join(valid_phones) + "}"
            tokens.append(
                HybridToken(
                    kind="phonetic",
                    orig_text=phonetic,
                    norm_text=norm_group,
                    phones=valid_phones,
                )
            )

        elif gap:
            gm = re.match(r"^<gap:\s*(-?\d+(?:\.\d+)?)\s*(ms|s)?>$", gap.lower())
            if not gm:
                raise ValueError(
                    f"Invalid gap format: '{gap}'. Expected format: <gap:250ms> or <gap:1.5s>"
                )
            val = float(gm.group(1))
            if not math.isfinite(val) or val < 0:
                raise ValueError(f"Invalid gap duration in '{gap}': gap duration must be finite and non-negative.")
            unit = gm.group(2)
            if unit == "ms":
                sec = val / 1000.0
            elif unit == "s":
                sec = val
            else:
                # Default to ms if >= 10, else seconds
                sec = val / 1000.0 if val >= 10.0 else val


            tokens.append(
                HybridToken(
                    kind="gap",
                    orig_text=gap,
                    norm_text=gap,
                    gap_seconds=sec,
                )
            )

        elif punct:
            tokens.append(
                HybridToken(
                    kind="punctuation",
                    orig_text=punct,
                    norm_text=punct,
                )
            )

        elif word:
            tokens.append(
                HybridToken(
                    kind="word",
                    orig_text=word,
                    norm_text=word.lower(),
                )
            )

        elif unrecognized:
            raise ValueError(f"Unrecognized token or malformed syntax: '{unrecognized}'")

    return tokens


class TargetGraphBuilder:
    """
    Builds a directed acyclic graph of legal phonetic paths for a target utterance,
    incorporating hybrid phonetic tokens, explicit gaps, and intonation inflection.
    """
    def __init__(self, lexicon: Lexicon):
        self.lexicon = lexicon

    def build(self, target_text: str, overrides: Optional[Dict[Any, Any]] = None) -> TargetGraph:
        tokens = tokenize_hybrid_syntax(target_text)
        if not tokens:
            raise ValueError(f"No valid tokens found in target text: {target_text}")

        normalized_words: List[str] = []
        punctuation_pauses: Dict[int, float] = {}
        explicit_gaps: Dict[int, float] = {}
        token_prosodic_bias: Dict[int, str] = {}

        # Separate speech tokens (words and phonetic groups) from gap tags and punctuation
        speech_tokens: List[HybridToken] = []
        terminal_punctuation: Optional[str] = None

        for tok in tokens:
            if tok.kind == "gap":
                b_idx = len(speech_tokens) - 1  # -1 for leading gap, 0..N-1 for internal/trailing
                gap_val = tok.gap_seconds if tok.gap_seconds is not None else 0.25
                explicit_gaps[b_idx] = explicit_gaps.get(b_idx, 0.0) + gap_val
                punctuation_pauses[b_idx] = explicit_gaps[b_idx]
            elif tok.kind == "punctuation":
                b_idx = max(0, len(speech_tokens) - 1)
                pause_sec = 0.45 if tok.norm_text in ".!?;:" else 0.25
                # Do not overwrite an explicit gap if one was already set
                if b_idx not in explicit_gaps:
                    punctuation_pauses[b_idx] = pause_sec
                terminal_punctuation = tok.norm_text
                if tok.norm_text == "?":
                    token_prosodic_bias[b_idx] = "question"
                elif tok.norm_text == "!":
                    token_prosodic_bias[b_idx] = "exclamation"
            else:
                speech_tokens.append(tok)
                normalized_words.append(tok.norm_text)

        if not speech_tokens:
            raise ValueError("Target text contains no spoken words or phonetic units")

        nodes: List[TargetPhoneNode] = []
        edges: List[Tuple[int, int]] = []
        word_branches: List[List[Tuple[int, int]]] = []

        for w_idx, tok in enumerate(speech_tokens):
            is_constrained = False
            if overrides and (w_idx in overrides or tok.norm_text in overrides):
                ov_val = overrides[w_idx] if w_idx in overrides else overrides[tok.norm_text]
                if hasattr(ov_val, "phones"):
                    prons = [ov_val.phones]
                elif isinstance(ov_val, (list, tuple)) and ov_val and isinstance(ov_val[0], str):
                    prons = [list(ov_val)]
                elif isinstance(ov_val, (list, tuple)) and ov_val and isinstance(ov_val[0], (list, tuple)):
                    prons = [list(p) for p in ov_val]
                else:
                    prons = [list(ov_val)]
                is_constrained = True
            elif tok.kind == "phonetic" and tok.phones:
                # Explicit phonetic group: deterministic pronunciation
                prons = [tok.phones]
                is_constrained = True
            else:
                # Ordinary lexical word: lookup in lexicon
                prons = self.lexicon.get_pronunciations(tok.norm_text)
                if not prons:
                    raise ValueError(f"No pronunciation could be determined for target word: '{tok.norm_text}'")
                is_constrained = False

            branches: List[Tuple[int, int]] = []
            for branch_idx, p_list in enumerate(prons):
                branch_node_ids: List[int] = []
                for p_pos, phone in enumerate(p_list):
                    node_id = len(nodes)
                    base = strip_phone_stress(phone)
                    stress = get_phone_stress(phone)
                    prior_dur = DEFAULT_DURATION_PRIORS.get(base, 0.10)

                    is_word_init = (p_pos == 0)
                    is_word_fin = (p_pos == len(p_list) - 1)
                    is_phrase_fin = (w_idx == len(speech_tokens) - 1 and is_word_fin)

                    node = TargetPhoneNode(
                        node_id=node_id,
                        token_index=w_idx,
                        word_text=tok.norm_text,
                        phone=phone,
                        base_phone=base,
                        stress=stress,
                        is_word_initial=is_word_init,
                        is_word_final=is_word_fin,
                        is_phrase_final=is_phrase_fin,
                        duration_prior_sec=prior_dur,
                        prosodic_bias=token_prosodic_bias.get(w_idx),
                        branch_id=branch_idx,
                        is_constrained=is_constrained,
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
        edge_dict_fwd: Dict[int, List[int]] = {}
        edge_dict_bwd: Dict[int, List[int]] = {}
        for src, dst in edges:
            edge_dict_fwd.setdefault(src, []).append(dst)
            edge_dict_bwd.setdefault(dst, []).append(src)

        for node in nodes:
            succs = edge_dict_fwd.get(node.node_id, [])
            if succs:
                node.right_context_phone = nodes[succs[0]].phone
                node.right_context_phones = {nodes[s].phone for s in succs}
            preds = edge_dict_bwd.get(node.node_id, [])
            if preds:
                node.left_context_phone = nodes[preds[0]].phone
                node.left_context_phones = {nodes[p].phone for p in preds}

        start_nodes = [b[0] for b in word_branches[0]]
        end_nodes = [b[1] for b in word_branches[-1]]

        for nid in start_nodes:
            nodes[nid].left_context_phone = "sil"
            nodes[nid].left_context_phones = {"sil"}

        for nid in end_nodes:
            nodes[nid].is_phrase_final = True
            nodes[nid].right_context_phone = "sil"
            nodes[nid].right_context_phones = {"sil"}

        # Apply pause/gap context: tokens followed by punctuation or explicit gaps
        for node in nodes:
            if node.token_index in punctuation_pauses or node.token_index in explicit_gaps:
                if node.is_word_final:
                    node.is_phrase_final = True
                    node.right_context_phone = "sil"
                    node.right_context_phones = {"sil"}
            if (node.token_index - 1) in punctuation_pauses or (node.token_index - 1) in explicit_gaps:
                if node.is_word_initial:
                    node.left_context_phone = "sil"
                    node.left_context_phones = {"sil"}

        graph = TargetGraph(
            raw_text=target_text,
            normalized_words=normalized_words,
            nodes=nodes,
            edges=edges,
            start_nodes=start_nodes,
            end_nodes=end_nodes,
            punctuation_pauses=punctuation_pauses,
            terminal_punctuation=terminal_punctuation or "",
            explicit_gaps=explicit_gaps,
        )

        graph.validate()
        return graph
