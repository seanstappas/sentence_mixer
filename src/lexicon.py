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

    def is_known(self, word: str) -> bool:
        """Returns True if word exists in loaded dictionaries, False if OOV."""
        w = word.strip().lower()
        return w in self.word_to_pronunciations

    def save_custom_pronunciation(
        self,
        word: str,
        phones: List[str],
        custom_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Persists an approved pronunciation to custom_pronunciations.txt and updates in-memory dictionary.
        """
        clean_word = word.strip().lower()
        clean_phones = [p.strip().upper() for p in phones if p.strip()]
        if not clean_phones:
            raise ValueError(f"Cannot save empty pronunciation for word '{word}'")

        if custom_path is None:
            # Default to repo root dictionaries/custom_pronunciations.txt
            repo_root = Path(__file__).resolve().parent.parent
            custom_path = repo_root / "dictionaries" / "custom_pronunciations.txt"
        else:
            custom_path = Path(custom_path).resolve()

        custom_path.parent.mkdir(parents=True, exist_ok=True)
        phones_str = " ".join(clean_phones)

        # Atomic append
        with open(custom_path, "a", encoding="utf-8") as f:
            f.write(f"\n{clean_word}\t{phones_str}\n")

        # Update in-memory entry
        if clean_word not in self.word_to_pronunciations:
            self.word_to_pronunciations[clean_word] = []
        if clean_phones not in self.word_to_pronunciations[clean_word]:
            self.word_to_pronunciations[clean_word].insert(0, clean_phones)


ARPABET_EXEMPLARS: Dict[str, str] = {
    "AA": "'a' in father / calm",
    "AE": "'a' in apple / bat",
    "AH": "'u' in cut / sofa",
    "AO": "'aw' in law / caught",
    "AW": "'ow' in cow / shout",
    "AY": "'i' in bite / ride",
    "B": "'b' in boy",
    "CH": "'ch' in cheese",
    "D": "'d' in dog",
    "DH": "'th' in they / breathe",
    "EH": "'e' in red / bed",
    "ER": "'ur' in bird / hurt",
    "EY": "'a' in say / make",
    "F": "'f' in food",
    "G": "'g' in go",
    "HH": "'h' in hat",
    "IH": "'i' in sit / big",
    "IY": "'ee' in see / feel",
    "JH": "'j' in jump",
    "K": "'k' in kite",
    "L": "'l' in look",
    "M": "'m' in man",
    "N": "'n' in no",
    "NG": "'ng' in sing",
    "OW": "'o' in go / boat",
    "OY": "'oy' in boy / coin",
    "P": "'p' in pet",
    "R": "'r' in run",
    "S": "'s' in sun",
    "SH": "'sh' in ship",
    "T": "'t' in top",
    "TH": "'th' in thin / math",
    "UH": "'oo' in book / put",
    "UW": "'oo' in moon / two",
    "V": "'v' in voice",
    "W": "'w' in wet",
    "Y": "'y' in yes",
    "Z": "'z' in zoo",
    "ZH": "'s' in vision / measure",
}


def format_phoneme_breakdown(phones: List[str]) -> str:
    """Formats ARPAbet phones into human-readable exemplars for interactive review."""
    out = []
    for p in phones:
        base = strip_phone_stress(p).upper()
        stress = " (stressed)" if "1" in p else (" (unstressed)" if "0" in p else "")
        desc = ARPABET_EXEMPLARS.get(base, base.lower())
        out.append(f"    {p:<4} -> {desc}{stress}")
    return "\n".join(out)
