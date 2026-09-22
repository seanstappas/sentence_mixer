"""
Interactive Out-of-Vocabulary (OOV) Learning Assistant.
Scans target texts for unrecognized words prior to graph building.
Prompts users with sounds-like exemplars, supports manual ARPAbet entry,
and enables microphone capture, atomically saving approved pronunciations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, List, Optional, Union

from lexicon import (
    Lexicon,
    format_phoneme_breakdown,
    get_phone_stress,
    strip_phone_stress,
)
from target_graph import VALID_BASE_PHONES, VOWEL_PHONES, tokenize_hybrid_syntax


class OOVAssistant:
    """
    Manages OOV word detection, interactive user prompting with sounds-like guides,
    and persistent storage of custom pronunciations.
    """
    def __init__(
        self,
        lexicon: Lexicon,
        custom_dict_path: Optional[Union[str, Path]] = None,
    ):
        self.lexicon = lexicon
        if custom_dict_path is not None:
            self.custom_dict_path = Path(custom_dict_path).resolve()
        else:
            repo_root = Path(__file__).resolve().parent.parent
            self.custom_dict_path = repo_root / "dictionaries" / "custom_pronunciations.txt"

    def detect_oov_words(self, text: str) -> List[str]:
        """
        Identifies unique out-of-vocabulary lexical words in input text.
        Skips explicit phonetic groups {...}, gap tags <gap:...>, and punctuation.
        """
        tokens = tokenize_hybrid_syntax(text)
        oov: List[str] = []
        seen = set()

        for tok in tokens:
            if tok.kind == "word":
                w = tok.norm_text
                if not self.lexicon.is_known(w) and w not in seen:
                    oov.append(w)
                    seen.add(w)

        return oov

    def resolve_oov_word(
        self,
        word: str,
        interactive: bool = True,
        input_fn: Callable[[str], str] = input,
        recognizer: Optional[Any] = None,
    ) -> List[str]:
        """
        Resolves an unrecognized word either interactively or via non-interactive G2P fallback.
        """
        clean_word = word.strip().lower()
        g2p_prons = self.lexicon.get_pronunciations(clean_word)
        suggestion = g2p_prons[0] if g2p_prons else ["AH0"]

        if not interactive:
            return suggestion

        print("=" * 60)
        print(f"[?] Unrecognized Word: \"{clean_word}\"")
        print("=" * 60)
        print(f"G2P Suggestion: {' '.join(suggestion)}")
        print("Pronunciation Guide:")
        print(format_phoneme_breakdown(suggestion))
        print("\nChoose an action:")
        print(f"  [1] Accept suggestion ({' '.join(suggestion)})")
        print("  [2] Enter custom ARPAbet phonemes manually")
        print("  [3] Speak pronunciation into microphone")
        print("  [4] Ignore and proceed")

        choice = input_fn("> ").strip()

        if choice == "1":
            self.lexicon.save_custom_pronunciation(clean_word, suggestion, self.custom_dict_path)
            print(f"[+] Saved '{clean_word}' -> {' '.join(suggestion)} to {self.custom_dict_path.name}")
            return suggestion

        elif choice == "2":
            user_input = input_fn("Enter ARPAbet phonemes (e.g. B AE1 K T AH0): ").strip()
            raw_tokens = user_input.split()
            valid_phones: List[str] = []
            for rt in raw_tokens:
                p_up = rt.upper()
                base = strip_phone_stress(p_up)
                if base not in VALID_BASE_PHONES:
                    print(f"[!] Warning: Invalid ARPAbet phone '{p_up}'. Falling back to G2P suggestion.")
                    return suggestion
                stress = get_phone_stress(p_up)
                if stress is not None and base not in VOWEL_PHONES:
                    p_up = base
                valid_phones.append(p_up)

            if valid_phones:
                self.lexicon.save_custom_pronunciation(clean_word, valid_phones, self.custom_dict_path)
                print(f"[+] Saved '{clean_word}' -> {' '.join(valid_phones)} to {self.custom_dict_path.name}")
                return valid_phones
            else:
                return suggestion

        elif choice == "3":
            try:
                if recognizer is None:
                    from phoneme_recognizer import PhonemeRecognizer
                    recognizer = PhonemeRecognizer()
                print(f"[*] Recording speech snippet for '{clean_word}'...")
                audio = recognizer.record_audio_snippet(duration=2.5)
                rec_phones = recognizer.audio_to_arpabet(audio)
                if rec_phones:
                    print(f"[*] Interpreted phones: {' '.join(rec_phones)}")
                    print(format_phoneme_breakdown(rec_phones))
                    confirm = input_fn("Accept this pronunciation? [Y/n]: ").strip().lower()
                    if confirm in ("", "y", "yes"):
                        self.lexicon.save_custom_pronunciation(clean_word, rec_phones, self.custom_dict_path)
                        print(f"[+] Saved '{clean_word}' -> {' '.join(rec_phones)} to {self.custom_dict_path.name}")
                        return rec_phones
            except Exception as e:
                print(f"[!] Microphone ingestion failed: {e}. Falling back to suggestion.")
            return suggestion

        else:  # Choice 4 or any other input
            print(f"[*] Proceeding with '{clean_word}' without saving.")
            return suggestion

    def resolve_all_in_text(
        self,
        text: str,
        interactive: bool = True,
        input_fn: Callable[[str], str] = input,
        recognizer: Optional[Any] = None,
    ) -> List[str]:
        """
        Scans text for all OOV words and prompts user resolution for each.
        """
        oov_words = self.detect_oov_words(text)
        resolved = []
        for word in oov_words:
            self.resolve_oov_word(
                word,
                interactive=interactive,
                input_fn=input_fn,
                recognizer=recognizer,
            )
            resolved.append(word)
        return resolved
