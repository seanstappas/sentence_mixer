#!/usr/bin/env python3
"""
Unit Selection Engine for Sentence Mixing
Synthesizes target phrases by prioritizing whole words from the aligned corpus inventory
and falling back to G2P phonemes when words are unavailable.
"""

import argparse
import json
import re
import sys
from typing import List, Dict, Any, Tuple, Optional

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


PHONETIC_COGNATE_FALLBACKS: Dict[str, List[str]] = {
    "W": ["UW1", "UW0", "UW", "OW1"],  # Labio-velar glide -> labio-velar vowel
    "Y": ["IY1", "IY0", "IY", "IH1"],  # Palatal glide -> high front vowel
    "NG": ["N", "M"],                   # Nasals
    "G": ["K"],                         # Velar plosive
    "B": ["P"],                         # Bilabial plosive
    "D": ["T"],                         # Alveolar plosive
    "V": ["F"],                         # Labiodental fricative
    "Z": ["S"],                         # Alveolar fricative
    "ZH": ["SH", "JH"],                 # Post-alveolar fricative
    "DH": ["TH", "D"],                  # Dental fricative
    "TH": ["T", "S"],                   # Dental fricative
    "AW": ["AA1", "UW1"],               # Diphthong fallback
    "OY": ["OW1", "IY1"],               # Diphthong fallback
}


def get_word_phonemes(word: str) -> List[str]:
    """Convert an unknown or OOV word into ARPA phonemes using g2p_en or basic phonetic rules."""
    try:
        from g2p_en import G2p
        g2p = G2p()
        phones = [p for p in g2p(word) if re.match(r"^[A-Z]", p)]
        if phones:
            return phones
    except Exception:
        pass

    # Built-in lightweight phonetic fallback if g2p_en is not available in current process
    fallback_map = {
        "a": ["AH0"], "b": ["B"], "c": ["K"], "d": ["D"], "e": ["EH1"],
        "f": ["F"], "g": ["G"], "h": ["HH"], "i": ["IH1"], "j": ["JH"],
        "k": ["K"], "l": ["L"], "m": ["M"], "n": ["N"], "o": ["OW1"],
        "p": ["P"], "q": ["K"], "r": ["R"], "s": ["S"], "t": ["T"],
        "u": ["AH1"], "v": ["V"], "w": ["W"], "x": ["K", "S"], "y": ["Y"],
        "z": ["Z"]
    }
    phones = []
    for char in word.lower():
        if char in fallback_map:
            phones.extend(fallback_map[char])
    return phones


def select_best_unit(candidates: List[Dict[str, Any]], target_type: str = "word") -> Dict[str, Any]:
    """
    Selects the most natural unit candidate.
    Filters out extreme duration outliers and selects near the median duration.
    """
    if not candidates:
        raise ValueError("No candidates provided")

    if len(candidates) == 1:
        return candidates[0]

    # Filter by sensible duration bounds
    if target_type == "word":
        min_dur, max_dur = 0.08, 1.8
    else:  # phoneme
        min_dur, max_dur = 0.03, 0.35

    filtered = [c for c in candidates if min_dur <= c["duration"] <= max_dur]
    pool = filtered if filtered else candidates

    # Sort by duration and pick median or near-median for steady tempo
    sorted_pool = sorted(pool, key=lambda x: x["duration"])
    mid_idx = len(sorted_pool) // 2
    return sorted_pool[mid_idx]


def _get_candidates_for_phone(
    p: str,
    phones_db: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], bool]:
    p_clean = p.upper()
    base_p = re.sub(r"\d", "", p_clean)

    # 1. Exact phone match
    candidates = phones_db.get(p_clean)

    # 2. If missing or micro-slice, pool all stress variants of base phone
    if not candidates or (re.search(r"\d", p_clean) and all(c.get("duration", 0) < 0.04 for c in candidates)):
        matching_keys = [k for k in phones_db if re.sub(r"\d", "", k) == base_p]
        all_cands = []
        for mk in matching_keys:
            all_cands.extend(phones_db[mk])
        if all_cands:
            candidates = all_cands

    # 3. If still missing, check acoustic cognate donors
    is_cognate = False
    if not candidates:
        for cog in PHONETIC_COGNATE_FALLBACKS.get(base_p, []):
            cog_base = re.sub(r"\d", "", cog)
            cog_keys = [k for k in phones_db if k == cog or re.sub(r"\d", "", k) == cog_base]
            cog_cands = []
            for ck in cog_keys:
                cog_cands.extend(phones_db[ck])
            if cog_cands:
                candidates = cog_cands
                is_cognate = True
                break

    return candidates or [], is_cognate


def _phone_matches(unit_phone: str, target_phone: str) -> bool:
    u = unit_phone.upper()
    t = target_phone.upper()
    if u == t:
        return True
    u_base = re.sub(r"\d", "", u)
    t_base = re.sub(r"\d", "", t)
    # Stress-insensitive match (e.g. EH1 matches EH2, AH0 matches AH1)
    if u_base == t_base:
        return True
    # Vowel reduction / schwa in unstressed positions (e.g. AH0 matches EH0 or IH0)
    if u in ("AH0", "IH0", "EH0") and t in ("AH0", "IH0", "EH0"):
        return True
    # Acoustic cognate donors
    if u_base in PHONETIC_COGNATE_FALLBACKS.get(t_base, []) or u in PHONETIC_COGNATE_FALLBACKS.get(t_base, []):
        return True
    if t_base in PHONETIC_COGNATE_FALLBACKS.get(u_base, []) or t in PHONETIC_COGNATE_FALLBACKS.get(u_base, []):
        return True
    return False


def _build_phone_successor_map(
    phones_db: Dict[str, List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, float], Dict[str, Any]]]:
    """
    Collects all phone candidates, sorts them chronologically per chunk,
    and builds a lookup mapping each phone unit to its immediate chronological successor.
    This enables generic phonetic sub-sequence matching within and across words.
    """
    all_units: List[Dict[str, Any]] = []
    for u_list in phones_db.values():
        for u in u_list:
            all_units.append(u)

    # Sort chronologically by chunk and start time
    all_units.sort(key=lambda x: (x.get("chunk_id", ""), x.get("start_sec", 0.0)))

    # Deduplicate in case multiple keys contained the same unit object
    unique_units: List[Dict[str, Any]] = []
    seen = set()
    for u in all_units:
        key = (u.get("chunk_id", ""), round(u.get("start_sec", 0.0), 4), round(u.get("end_sec", 0.0), 4))
        if key not in seen:
            seen.add(key)
            unique_units.append(u)

    successor_map: Dict[Tuple[str, float], Dict[str, Any]] = {}
    for i in range(len(unique_units) - 1):
        u1 = unique_units[i]
        u2 = unique_units[i + 1]
        if (
            u1.get("chunk_id") == u2.get("chunk_id")
            and abs(u2.get("start_sec", 0.0) - u1.get("end_sec", 0.0)) < 0.005
        ):
            successor_map[(u1.get("chunk_id", ""), round(u1.get("start_sec", 0.0), 3))] = u2

    return unique_units, successor_map


def _find_longest_contiguous_chain(
    start_idx: int,
    phone_tokens: List[str],
    unique_units: List[Dict[str, Any]],
    successor_map: Dict[Tuple[str, float], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    first_target = phone_tokens[start_idx]
    candidate_starts = [u for u in unique_units if _phone_matches(u.get("label", ""), first_target)]
    if not candidate_starts:
        return []

    best_chain: List[Dict[str, Any]] = []
    for start_u in candidate_starts:
        chain = [start_u]
        curr = start_u
        for next_p in phone_tokens[start_idx + 1:]:
            curr_key = (curr.get("chunk_id", ""), round(curr.get("start_sec", 0.0), 3))
            next_u = successor_map.get(curr_key)
            if next_u and _phone_matches(next_u.get("label", ""), next_p):
                chain.append(next_u)
                curr = next_u
            else:
                break
        if len(chain) > len(best_chain):
            best_chain = chain
            if len(best_chain) == len(phone_tokens) - start_idx:
                break
    return best_chain


def find_sub_word_candidate(
    token: str,
    words_db: Dict[str, List[Dict[str, Any]]],
    phones_db: Dict[str, List[Dict[str, Any]]],
    lexicon: Optional[Any] = None,
) -> Optional[Dict[str, Any]]:
    """
    Finds a clean sub-word audio slice when a target token is a prefix or suffix
    of an existing word in words_db (e.g. 'mendment' from 'amendment').
    """
    clean_token = token.lower()

    # 1. Suffix match: e.g. "mendment" in "amendment"
    for parent_word, insts in words_db.items():
        if not insts or len(parent_word) <= len(clean_token):
            continue
        if parent_word.endswith(clean_token):
            prefix = parent_word[:-len(clean_token)]
            prefix_phones: List[str] = []
            if lexicon is not None:
                prons = lexicon.get_pronunciations(prefix)
                if prons:
                    prefix_phones = prons[0]
            if not prefix_phones:
                prefix_phones = get_word_phonemes(prefix)

            for inst in insts:
                w_start = inst["start_sec"]
                w_end = inst["end_sec"]
                chunk_id = inst.get("chunk_id")
                w_phones: List[Dict[str, Any]] = []
                for ph_list in phones_db.values():
                    for p in ph_list:
                        if (
                            p.get("chunk_id") == chunk_id
                            and p["start_sec"] >= (w_start - 0.01)
                            and p["end_sec"] <= (w_end + 0.01)
                        ):
                            w_phones.append(p)
                w_phones.sort(key=lambda x: x["start_sec"])
                dedup_phones: List[Dict[str, Any]] = []
                for p in w_phones:
                    if not dedup_phones or abs(p["start_sec"] - dedup_phones[-1]["start_sec"]) > 0.005:
                        dedup_phones.append(p)

                num_prefix = len(prefix_phones)
                if 0 < num_prefix < len(dedup_phones):
                    sub_start = dedup_phones[num_prefix]["start_sec"]
                    sub_end = w_end
                    cand = dict(inst)
                    cand["start_sec"] = sub_start
                    cand["end_sec"] = sub_end
                    cand["duration"] = sub_end - sub_start
                    cand["type"] = "sub_word"
                    cand["token"] = token
                    cand["label"] = f"{token} (from '{parent_word}')"
                    if "sample_start" in cand and "sample_end" in cand:
                        sr_val = cand.get("native_sample_rate", 48000) or 48000
                        cand["sample_start"] = int(round(sub_start * sr_val))
                        cand["sample_end"] = int(round(sub_end * sr_val))
                    return cand

        # 2. Prefix match: e.g. "amend" in "amendment"
        if parent_word.startswith(clean_token):
            suffix = parent_word[len(clean_token):]
            suffix_phones: List[str] = []
            if lexicon is not None:
                prons = lexicon.get_pronunciations(suffix)
                if prons:
                    suffix_phones = prons[0]
            if not suffix_phones:
                suffix_phones = get_word_phonemes(suffix)

            for inst in insts:
                w_start = inst["start_sec"]
                w_end = inst["end_sec"]
                chunk_id = inst.get("chunk_id")
                w_phones = []
                for ph_list in phones_db.values():
                    for p in ph_list:
                        if (
                            p.get("chunk_id") == chunk_id
                            and p["start_sec"] >= (w_start - 0.01)
                            and p["end_sec"] <= (w_end + 0.01)
                        ):
                            w_phones.append(p)
                w_phones.sort(key=lambda x: x["start_sec"])
                dedup_phones = []
                for p in w_phones:
                    if not dedup_phones or abs(p["start_sec"] - dedup_phones[-1]["start_sec"]) > 0.005:
                        dedup_phones.append(p)

                num_suffix = len(suffix_phones)
                if 0 < num_suffix < len(dedup_phones):
                    sub_start = w_start
                    sub_end = dedup_phones[-num_suffix]["start_sec"]
                    cand = dict(inst)
                    cand["start_sec"] = sub_start
                    cand["end_sec"] = sub_end
                    cand["duration"] = sub_end - sub_start
                    cand["type"] = "sub_word"
                    cand["token"] = token
                    cand["label"] = f"{token} (from '{parent_word}')"
                    if "sample_start" in cand and "sample_end" in cand:
                        sr_val = cand.get("native_sample_rate", 48000) or 48000
                        cand["sample_start"] = int(round(sub_start * sr_val))
                        cand["sample_end"] = int(round(sub_end * sr_val))
                    return cand

    return None


def synthesize_sentence(
    target_text: str,
    inventory: Dict[str, Any],
    prefer_words: bool = True,
    lexicon: Optional[Any] = None,
    overrides: Optional[Dict[Any, Any]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Given a target phrase and a unit inventory, resolves the phrase into a sequence
    of video/audio slices prioritizing words and falling back to phonemes.
    """
    if re.search(r"<[^>]+>", target_text) or re.search(r"\{[^}]*\}", target_text):
        raise ValueError(
            "Legacy selector does not support markup (<gap:...>) or phonetic syntax ({...}). Use --selector span."
        )

    words_db = inventory.get("words", {})
    phones_db = inventory.get("phones", {})

    unique_units, successor_map = _build_phone_successor_map(phones_db)

    # Tokenize target text into clean words
    tokens = re.findall(r"[a-zA-Z0-9']+", target_text)
    selected_units: List[Dict[str, Any]] = []
    mix_plan: List[str] = []

    print(f"[*] Synthesizing phrase: \"{target_text}\"")
    print(f"[*] Tokens to match: {tokens}")

    for token in tokens:
        clean_token = token.lower()

        # Step 1: Check for whole word match
        if prefer_words and clean_token in words_db and len(words_db[clean_token]) > 0:
            chosen = select_best_unit(words_db[clean_token], target_type="word")
            unit_entry = dict(chosen)
            unit_entry["type"] = "word"
            unit_entry["token"] = token
            selected_units.append(unit_entry)
            mix_plan.append(f"[WORD] '{token}' -> {chosen['start_sec']:.3f}s..{chosen['end_sec']:.3f}s ({chosen['duration']:.2f}s)")
            continue

        # Step 1.5: Check for sub-word match (e.g. suffix/prefix of existing word)
        if prefer_words:
            sub_cand = find_sub_word_candidate(token, words_db, phones_db, lexicon=lexicon)
            if sub_cand:
                selected_units.append(sub_cand)
                mix_plan.append(f"[SUB-WORD] '{token}' -> {sub_cand['start_sec']:.3f}s..{sub_cand['end_sec']:.3f}s ({sub_cand['duration']:.2f}s, {sub_cand['label']})")
                continue

        # Step 2: Fall back to phonemes / contiguous sub-word chains
        print(f"[>] Word '{token}' not found in word corpus. Falling back to phonemes...")
        phone_tokens: List[str] = []
        if overrides and (token in overrides or clean_token in overrides):
            cand = overrides.get(token) or overrides.get(clean_token)
            if cand:
                phone_tokens = cand[0] if isinstance(cand[0], list) else cand
        if not phone_tokens and lexicon is not None:
            lex_prons = lexicon.get_pronunciations(clean_token)
            if lex_prons:
                phone_tokens = lex_prons[0]
        if not phone_tokens:
            phone_tokens = get_word_phonemes(token)
        print(f"    Phonetic breakdown for '{token}': {phone_tokens}")

        if not phone_tokens:
            print(f"[!] Warning: Could not generate phonemes for '{token}', skipping.")
            continue

        word_phone_units = []
        p_idx = 0
        while p_idx < len(phone_tokens):
            chain = _find_longest_contiguous_chain(p_idx, phone_tokens, unique_units, successor_map)
            if len(chain) >= 2:
                # Contiguous sub-word / multi-phone slice!
                s_start = chain[0]["start_sec"]
                s_end = chain[-1]["end_sec"]
                chain_labels = [c.get("label", "") for c in chain]
                sub_unit = dict(chain[0])
                sub_unit["type"] = "sub_word"
                sub_unit["label"] = "+".join(chain_labels)
                sub_unit["start_sec"] = s_start
                sub_unit["end_sec"] = s_end
                sub_unit["duration"] = s_end - s_start
                sub_unit["sample_start"] = chain[0].get("sample_start", 0)
                sub_unit["sample_end"] = chain[-1].get("sample_end", 0)
                sub_unit["token"] = f"{token}:{''.join(chain_labels)}"
                word_phone_units.append(sub_unit)
                p_idx += len(chain)
            else:
                p = phone_tokens[p_idx]
                p_clean = p.upper()
                base_p = re.sub(r"\d", "", p_clean)
                candidates, is_cognate = _get_candidates_for_phone(p, phones_db)
                if is_cognate:
                    print(f"    [*] Using acoustic donor for missing phoneme '{p}' in '{token}'")

                if candidates and len(candidates) > 0:
                    chosen_p = select_best_unit(candidates, target_type="phone")
                    p_entry = dict(chosen_p)
                    p_entry["type"] = "phone"
                    p_entry["token"] = f"{token}:{base_p}"
                    # If a vowel was borrowed as a consonant glide (e.g. UW for W), cap duration to a crisp glide
                    if is_cognate and base_p in ("W", "Y") and p_entry.get("duration", 0) > 0.12:
                        glide_dur = 0.12
                        p_entry["duration"] = glide_dur
                        p_entry["end_sec"] = p_entry["start_sec"] + glide_dur
                        if "sample_start" in p_entry:
                            sr_val = p_entry.get("native_sample_rate", 48000) or 48000
                            p_entry["sample_end"] = p_entry["sample_start"] + int(round(glide_dur * sr_val))
                    word_phone_units.append(p_entry)
                else:
                    print(f"[!] Warning: Phoneme '{p}' ({base_p}) not available in corpus!")
                p_idx += 1

        if word_phone_units:
            selected_units.extend(word_phone_units)
            if len(word_phone_units) == 1 and word_phone_units[0].get("type") == "sub_word":
                u0 = word_phone_units[0]
                mix_plan.append(f"[SUB-WORD] '{token}' -> {u0['start_sec']:.3f}s..{u0['end_sec']:.3f}s ({u0['duration']:.2f}s, phones: {u0['label']})")
            else:
                mix_plan.append(f"[PHONEME FALLBACK] '{token}' -> {len(word_phone_units)} units ({'+'.join([u['label'] for u in word_phone_units])})")

    return selected_units, mix_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unit selection engine for YTP sentence mixing.")
    parser.add_argument("--text", required=True, help="Target sentence to synthesize")
    parser.add_argument("--inventory", default="./inventory/units_index.json", help="Path to units_index.json")
    args = parser.parse_args()

    with open(args.inventory, "r", encoding="utf-8") as f:
        inv = json.load(f)

    units, plan = synthesize_sentence(args.text, inv)
    print("\n--- Mix Plan ---")
    for step in plan:
        print(step)
    print(f"\nTotal units selected: {len(units)}")
