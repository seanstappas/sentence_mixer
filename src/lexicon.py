"""
Shared lexicon service for MFA dictionary handling, custom pronunciations,
text normalization, and G2P fallback.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# Ensure NLTK data path does not touch locked BitLocker drives
DEFAULT_NLTK_DIR = Path(__file__).resolve().parent.parent / ".build" / "nltk"
if "NLTK_DATA" not in os.environ:
    os.environ["NLTK_DATA"] = str(DEFAULT_NLTK_DIR)

try:
    import nltk
    nltk.data.path = [os.environ["NLTK_DATA"]]
except ImportError:
    pass

_G2P_INSTANCE = None


def get_g2p():
    global _G2P_INSTANCE
    if _G2P_INSTANCE is None:
        try:
            import nltk
            nltk.data.path = [os.environ.get("NLTK_DATA", str(DEFAULT_NLTK_DIR))]
            from g2p_en import G2p
            _G2P_INSTANCE = G2p()
        except Exception:
            _G2P_INSTANCE = None
    return _G2P_INSTANCE


def normalize_text_to_words(text: str) -> List[Tuple[str, str]]:
    """
    Normalizes input text into a sequence of (original_token, normalized_token).
    Preserves hyphens in known compounds (e.g. OBI-WAN) and apostrophes in contractions.
    """
    # Replace curly quotes and typographic apostrophes
    text = text.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')

    # Tokenize preserving hyphenated words and words with apostrophes
    tokens = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*|[.,!?;:]", text)

    result: List[Tuple[str, str]] = []
    for tok in tokens:
        # Don't strip punctuation tokens
        if tok in ".,!?;:":
            result.append((tok, tok))
            continue
        clean = tok.strip().lower()
        if clean:
            result.append((tok, clean))
    return result


def is_numeric_prob(token: str) -> bool:
    try:
        float(token)
        return True
    except ValueError:
        return False


def strip_phone_stress(phone: str) -> str:
    """Strip trailing stress digits (e.g. OW1 -> OW, AH0 -> AH)."""
    return re.sub(r"\d+$", "", phone)


def get_phone_stress(phone: str) -> Optional[int]:
    """Return integer stress if present (e.g. OW1 -> 1), else None."""
    m = re.search(r"(\d+)$", phone)
    return int(m.group(1)) if m else None


class Lexicon:
    """
    Unified dictionary and pronunciation manager.
    Loads custom pronunciations first, then base dictionaries, with G2P fallback.
    """
    def __init__(self, dict_paths: Optional[List[Union[str, Path]]] = None):
        self.word_to_pronunciations: Dict[str, List[List[str]]] = {}
        self.loaded_files: List[str] = []

        if dict_paths:
            for p in dict_paths:
                self.load_dictionary_file(p)

    def load_dictionary_file(self, path: Union[str, Path]) -> None:
        p = Path(path).resolve()
        if not p.exists():
            return
        self.loaded_files.append(str(p))

        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                parts = line.split()
                if not parts:
                    continue

                word = parts[0].strip().lower()
                # Remove pronunciation variant counter like 'word(2)' if present
                word = re.sub(r"\(\d+\)$", "", word)

                # Collect phones, ignoring numeric probability columns from MFA dictionaries
                phones = []
                for tok in parts[1:]:
                    if is_numeric_prob(tok):
                        continue
                    # Clean up phone tokens
                    p_clean = tok.strip().upper()
                    if p_clean:
                        phones.append(p_clean)

                if phones:
                    if word not in self.word_to_pronunciations:
                        self.word_to_pronunciations[word] = []
                    if phones not in self.word_to_pronunciations[word]:
                        self.word_to_pronunciations[word].append(phones)

    def get_pronunciations(self, word: str) -> List[List[str]]:
        """Get all known pronunciations for a word, or generate via G2P."""
        w = word.strip().lower()
        if w in self.word_to_pronunciations:
            return self.word_to_pronunciations[w]

        # Try without hyphens or split by hyphen if compound
        if "-" in w:
            sub_words = w.split("-")
            sub_prons = [self.get_pronunciations(sw) for sw in sub_words]
            if all(sub_prons):
                combined = []
                for p in sub_prons[0]:
                    combined.extend(p)
                for sp_list in sub_prons[1:]:
                    combined.extend(sp_list[0])
                return [combined]

        # G2P fallback
        g2p = get_g2p()
        if g2p is not None:
            try:
                raw_phones = g2p(w)
                # Filter out spaces and punctuation from G2p output
                clean_phones = [p for p in raw_phones if p.strip() and re.match(r"^[A-Z]+\d*$", p)]
                if clean_phones:
                    return [clean_phones]
            except Exception:
                pass

        return []
