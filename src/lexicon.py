"""
Shared lexicon service for MFA dictionary handling, custom pronunciations,
text normalization, and G2P fallback.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
_STAGED_DICT_CACHE: Dict[Tuple[str, str, str], Tuple[Any, str, str]] = {}
STAGING_FORMAT_VERSION = 2


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
    text = text.replace("â€™", "'").replace("â€˜", "'").replace("â€œ", '"').replace("â€", '"')

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


# Standard 39 ARPAbet base phones
VALID_BASE_PHONES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "B", "CH", "D", "DH", "EH", "ER", "EY",
    "F", "G", "HH", "IH", "IY", "JH", "K", "L", "M", "N", "NG", "OW", "OY",
    "P", "R", "S", "SH", "T", "TH", "UH", "UW", "V", "W", "Y", "Z", "ZH",
}

VOWEL_PHONES = {
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"
}


def strip_phone_stress(phone: str) -> str:
    """Strip trailing stress digits (e.g. OW1 -> OW, AH0 -> AH)."""
    return re.sub(r"\d+$", "", phone)


def get_phone_stress(phone: str) -> Optional[int]:
    """Return integer stress if present (e.g. OW1 -> 1), else None."""
    m = re.search(r"(\d+)$", phone)
    return int(m.group(1)) if m else None


def validate_phone(phone: str) -> None:
    """Validate a single ARPAbet phone token.
    Permits stress only as a single '0', '1', or '2' on a supported vowel.
    Rejects consonant stress, extra digits, unknown phones.
    """
    p_up = phone.strip().upper()
    base = strip_phone_stress(p_up)
    if base not in VALID_BASE_PHONES:
        raise ValueError(f"Invalid ARPAbet phone '{phone}': unknown base phone '{base}'")
    m = re.search(r"(\d+)$", p_up)
    if m:
        digits = m.group(1)
        if base not in VOWEL_PHONES:
            raise ValueError(f"Invalid consonant stress in phone '{phone}': consonants cannot take stress")
        if digits not in ("0", "1", "2"):
            raise ValueError(f"Invalid stress '{digits}' in phone '{phone}': stress must be 0, 1, or 2")


class Lexicon:

    """
    Unified dictionary and pronunciation manager.
    Loads custom pronunciations first, then base dictionaries, with G2P fallback.
    """
    def __init__(self, dict_paths: Optional[List[Union[str, Path]]] = None):
        self.word_to_pronunciations: Dict[str, List[List[str]]] = {}
        self.custom_words: set[str] = set()
        self.loaded_files: List[str] = []

        if dict_paths:
            for p in dict_paths:
                self.load_dictionary_file(p)

    def load_dictionary_file(self, path: Union[str, Path]) -> None:
        p = Path(path).resolve()
        if not p.exists():
            return
        self.loaded_files.append(str(p))
        is_custom = "custom_pronunciations" in p.name

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

                if not is_custom and word in self.custom_words:
                    continue

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
                    if is_custom:
                        self.custom_words.add(word)

                    # Filter out truncated single-phone variants when a full pronunciation exists (e.g. 'eat' -> ['IY1'] vs ['IY1', 'T'])
                    if len(phones) == 1 and word in self.word_to_pronunciations:
                        existing = self.word_to_pronunciations[word]
                        if any(len(p) >= 2 for p in existing):
                            continue

                    # Filter out coda stop-consonant omissions if a full stop-coda pronunciation exists (e.g. 'that' -> ['DH', 'AH0'] vs ['DH', 'AH0', 'T'])
                    if word in self.word_to_pronunciations:
                        existing = self.word_to_pronunciations[word]
                        if any(len(p) == len(phones) + 1 and p[:-1] == phones and strip_phone_stress(p[-1]) in ("T", "D", "K", "P", "B", "G") for p in existing):
                            continue

                    if word not in self.word_to_pronunciations:
                        self.word_to_pronunciations[word] = []

                    # If this is a full stop-coda pronunciation, remove any prior truncated prefix variant
                    if strip_phone_stress(phones[-1]) in ("T", "D", "K", "P", "B", "G"):
                        self.word_to_pronunciations[word] = [
                            p for p in self.word_to_pronunciations[word]
                            if not (len(p) == len(phones) - 1 and phones[:-1] == p)
                        ]

                    if len(phones) >= 2 and word in self.word_to_pronunciations:
                        self.word_to_pronunciations[word] = [p for p in self.word_to_pronunciations[word] if len(p) >= 2 or p == phones]
                    if phones not in self.word_to_pronunciations[word]:
                        self.word_to_pronunciations[word].append(phones)

    def get_pronunciations(self, word: str) -> List[List[str]]:
        """Get all known pronunciations for a word, or generate via G2P."""
        w = word.strip().lower()
        if w in self.word_to_pronunciations:
            return self.word_to_pronunciations[w]

        # Try without hyphens or split by hyphen if compound
        if "-" in w:
            sub_words = [sw for sw in w.split("-") if sw]
            sub_prons = [self.get_pronunciations(sw) for sw in sub_words]
            if all(sub_prons):
                import itertools
                results: List[List[str]] = []
                for prod in itertools.islice(itertools.product(*sub_prons), 16):
                    combo = []
                    for comp_phones in prod:
                        combo.extend(comp_phones)
                    results.append(combo)
                return results

        # Compound word decomposition without hyphens (e.g. bitcoin -> bit + coin)
        if len(w) >= 6:
            for split_idx in range(3, len(w) - 2):
                w1 = w[:split_idx]
                w2 = w[split_idx:]
                if w1 in self.word_to_pronunciations and w2 in self.word_to_pronunciations:
                    p1_list = self.word_to_pronunciations[w1]
                    p2_list = self.word_to_pronunciations[w2]
                    combined: List[List[str]] = []
                    for pr1 in p1_list:
                        for pr2 in p2_list:
                            pr2_demoted = [re.sub(r"1$", "2", p) if p.endswith("1") else p for p in pr2]
                            combined.append(pr1 + pr2_demoted)
                    if combined:
                        return combined


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

    def is_heteronym(self, word: str) -> bool:
        """Returns True if word has multiple distinct pronunciations in the lexicon."""
        prons = self.get_pronunciations(word)
        return len(prons) > 1

    def save_custom_pronunciation(
        self,
        word: str,
        phones: List[str],
        custom_path: Optional[Union[str, Path]] = None,
    ) -> None:
        """
        Persists an approved pronunciation to custom_pronunciations.txt with phone validation,
        deduplication on normalized (lowercase word, phone tuple), and atomic file replacement.
        Updates in-memory dictionary.
        """
        clean_word = word.strip().lower()
        clean_phones = [p.strip().upper() for p in phones if p.strip()]
        if not clean_phones:
            raise ValueError(f"Cannot save empty pronunciation for word '{word}'")

        # Validate every phone before saving (A05 / A10)
        for p in clean_phones:
            validate_phone(p)

        if custom_path is None:
            repo_root = Path(__file__).resolve().parent.parent
            custom_path = repo_root / "dictionaries" / "custom_pronunciations.txt"
        else:
            custom_path = Path(custom_path).resolve()

        custom_path.parent.mkdir(parents=True, exist_ok=True)
        phones_str = " ".join(clean_phones)
        entry_line = f"{clean_word}\t{phones_str}"
        target_norm = (clean_word, tuple(clean_phones))

        # Process-safe atomic deduplicated read/merge/write with FileLock
        lock_path = custom_path.with_suffix(".lock")
        from filelock import FileLock
        with FileLock(str(lock_path), timeout=10):
            existing_lines: List[str] = []
            existing_tuples = set()
            if custom_path.exists():
                for ln in custom_path.read_text(encoding="utf-8").splitlines():
                    ln_stripped = ln.strip()
                    if not ln_stripped:
                        continue
                    existing_lines.append(ln_stripped)
                    if not ln_stripped.startswith("#"):
                        parts = ln_stripped.split()
                        if parts:
                            w = parts[0].strip().lower()
                            phs = tuple(p.strip().upper() for p in parts[1:] if not is_numeric_prob(p))
                            existing_tuples.add((w, phs))

            if target_norm not in existing_tuples:
                existing_lines.append(entry_line)
                tmp_path = custom_path.with_suffix(f".tmp_{os.getpid()}")
                tmp_path.write_text("\n".join(existing_lines) + "\n", encoding="utf-8")
                tmp_path.replace(custom_path)

        # Update in-memory entry
        if clean_word not in self.word_to_pronunciations:
            self.word_to_pronunciations[clean_word] = []
        if clean_phones not in self.word_to_pronunciations[clean_word]:
            self.word_to_pronunciations[clean_word].insert(0, clean_phones)
        self.custom_words.add(clean_word)
        _STAGED_DICT_CACHE.clear()


def normalize_numerals(text: str) -> str:
    """
    Normalizes numeric digits outside phonetic {...} or tag <...> blocks to word equivalents using inflect.
    e.g. 'hello 1 world' -> 'hello one world'
    """
    if not any(c.isdigit() for c in text):
        return text

    engine = None
    try:
        import inflect
        engine = inflect.engine()
    except Exception:
        pass

    _DIGIT_WORDS = {
        "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
        "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
        "18": "eighteen", "19": "nineteen", "20": "twenty",
    }

    def _repl(m: re.Match) -> str:
        if m.group(1):  # {...} block
            return m.group(1)
        if m.group(2):  # <...> block
            return m.group(2)
        if m.group(3):  # digits
            num_str = m.group(3)
            if engine is not None:
                try:
                    words = engine.number_to_words(num_str)
                    return words.replace(",", "").replace("-", " ")
                except Exception:
                    pass
            if num_str in _DIGIT_WORDS:
                return _DIGIT_WORDS[num_str]
            return num_str
        return m.group(0)

    pattern = re.compile(r"(\{[^}]*\})|(<[^>]+>)|(\b\d+(?:\.\d+)?\b)")
    return pattern.sub(_repl, text)



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


def stage_mfa_dictionary(
    base_dict_path: Optional[Union[str, Path]] = None,
    custom_dict_path: Optional[Union[str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
) -> Tuple[Path, str]:
    """
    Dynamically combines base dictionary (merged_dictionary.dict) and custom pronunciations
    (custom_pronunciations.txt) into a single staged MFA dictionary (.build/mfa/staged_dictionary.dict).
    Validates tab-separated ARPAbet format (WORD\\tP H O N E S), handles case variants
    (both lowercase and uppercase), deduplicates entries, saves atomically with FileLock,
    and returns the staged dictionary path and its SHA-256 content hash.
    """
    repo_root = Path(__file__).resolve().parent.parent

    b_path = (Path(base_dict_path) if base_dict_path is not None else (repo_root / "dictionaries" / "merged_dictionary.dict")).resolve()
    c_path = (Path(custom_dict_path) if custom_dict_path is not None else (repo_root / "dictionaries" / "custom_pronunciations.txt")).resolve()
    out_p = (Path(output_path) if output_path is not None else (repo_root / ".build" / "mfa" / "staged_dictionary.dict")).resolve()

    if not b_path.exists() and not c_path.exists():
        raise FileNotFoundError(
            f"Neither base dictionary ({b_path}) nor custom dictionary ({c_path}) found."
        )

    def _file_sha(path: Path) -> str:
        if not path.exists():
            return ""
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 20):
                h.update(chunk)
        return h.hexdigest()

    # Reuse is decided by input content, not timestamps: an edit made within the same
    # filesystem timestamp tick as the last staging must still invalidate the result.
    b_stat = (b_path.stat().st_mtime_ns, b_path.stat().st_size) if b_path.exists() else (0, 0)
    c_sha = _file_sha(c_path)
    cache_key = (str(b_path), str(c_path), str(out_p))
    fast_key = (b_stat, c_sha)

    # Fast in-memory check (base dictionary by size+mtime, custom file by content)
    if cache_key in _STAGED_DICT_CACHE:
        cached_fast, cached_fp, cached_hash = _STAGED_DICT_CACHE[cache_key]
        if cached_fast == fast_key and out_p.exists() and out_p.stat().st_size > 0:
            return out_p, cached_hash

    inputs_fp = hashlib.sha256(f"{_file_sha(b_path)}|{c_sha}|{STAGING_FORMAT_VERSION}".encode("utf-8")).hexdigest()
    meta_p = out_p.with_name(out_p.name + ".inputs.json")

    # Reuse the staged file on disk if it was built from identical inputs
    if out_p.exists() and out_p.stat().st_size > 0 and meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if meta.get("inputs") == inputs_fp and meta.get("content_hash"):
                _STAGED_DICT_CACHE[cache_key] = (fast_key, inputs_fp, meta["content_hash"])
                return out_p, meta["content_hash"]
        except (OSError, ValueError):
            pass

    out_p.parent.mkdir(parents=True, exist_ok=True)
    staged_lines: List[str] = []
    seen: set[str] = set()

    for p in (b_path, c_path):
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if not parts:
                    continue
                word = parts[0].strip()
                # Remove pronunciation variant counter like 'word(2)' if present
                word = re.sub(r"\(\d+\)$", "", word)

                # Check for probability tokens (e.g. Kaldi acoustic model entries)
                has_prob = any(is_numeric_prob(tok) for tok in parts[1:])
                if has_prob:
                    # Preserve probability columns for Kaldi/MFA
                    rest = line.split("\t", 1)[1].strip() if "\t" in line else " ".join(parts[1:]).strip()
                    entry = f"{word}\t{rest}"
                    if entry not in seen:
                        seen.add(entry)
                        staged_lines.append(entry)
                else:
                    phones = [tok.strip().upper() for tok in parts[1:] if tok.strip()]
                    if not phones:
                        continue
                    phones_str = " ".join(phones)
                    # For custom pronunciations or non-prob entries, generate both lowercase and uppercase variants
                    # to ensure robust matching regardless of subtitle cue casing.
                    for w_variant in (word.lower(), word.upper()):
                        entry = f"{w_variant}\t{phones_str}"
                        if entry not in seen:
                            seen.add(entry)
                            staged_lines.append(entry)

    content_bytes = ("\n".join(staged_lines) + "\n").encode("utf-8")
    content_hash = hashlib.sha256(content_bytes).hexdigest()

    lock_path = out_p.with_suffix(".lock")
    from filelock import FileLock
    with FileLock(str(lock_path), timeout=30):
        for target, data in (
            (out_p, content_bytes),
            (meta_p, json.dumps({"inputs": inputs_fp, "content_hash": content_hash}).encode("utf-8")),
        ):
            tmp_path = target.parent / f"{target.name}.tmp_{os.getpid()}_{uuid.uuid4().hex}"
            try:
                tmp_path.write_bytes(data)
                # Windows refuses to replace a file another thread/process is reading; retry briefly
                for attempt in range(20):
                    try:
                        tmp_path.replace(target)
                        break
                    except PermissionError:
                        if attempt == 19:
                            raise
                        time.sleep(0.05)
            finally:
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except OSError:
                        pass

    _STAGED_DICT_CACHE[cache_key] = (fast_key, inputs_fp, content_hash)
    return out_p, content_hash
