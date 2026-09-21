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
from typing import List, Dict, Any, Tuple

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


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


def synthesize_sentence(
    target_text: str,
    inventory: Dict[str, Any],
    prefer_words: bool = True
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Given a target phrase and a unit inventory, resolves the phrase into a sequence
    of video/audio slices prioritizing words and falling back to phonemes.
    """
    words_db = inventory.get("words", {})
    phones_db = inventory.get("phones", {})

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

        # Step 2: Fall back to phonemes
        print(f"[>] Word '{token}' not found in word corpus. Falling back to phonemes...")
        phone_tokens = get_word_phonemes(token)
        print(f"    Phonetic breakdown for '{token}': {phone_tokens}")

        if not phone_tokens:
            print(f"[!] Warning: Could not generate phonemes for '{token}', skipping.")
            continue

        word_phone_units = []
        for p in phone_tokens:
            p_clean = p.upper()
            base_p = re.sub(r"\d", "", p_clean)

            # Check exact phone, then base phone without stress number
            candidates = phones_db.get(p_clean) or phones_db.get(base_p)

            if candidates and len(candidates) > 0:
                chosen_p = select_best_unit(candidates, target_type="phone")
                p_entry = dict(chosen_p)
                p_entry["type"] = "phone"
                p_entry["token"] = f"{token}:{base_p}"
                word_phone_units.append(p_entry)
            else:
                print(f"[!] Warning: Phoneme '{p}' ({base_p}) not available in corpus!")

        if word_phone_units:
            selected_units.extend(word_phone_units)
            mix_plan.append(f"[PHONEME FALLBACK] '{token}' -> {len(word_phone_units)} phones ({'+'.join([u['label'] for u in word_phone_units])})")

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
