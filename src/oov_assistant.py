"""
Interactive Out-of-Vocabulary (OOV) Learning Assistant & Pronunciation Reviewer.
Scans target texts for unrecognized words and heteronyms prior to graph building.
Prompts users with sounds-like exemplars, supports manual ARPAbet entry with validation,
enables microphone capture, supports per-occurrence overrides ('use once' vs 'saved'),
and atomically persists approved pronunciations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

from lexicon import (
    Lexicon,
    format_phoneme_breakdown,
    validate_phone,
)
from target_graph import tokenize_hybrid_syntax


@dataclass
class PronunciationChoice:
    token_index: int
    text: str
    phones: List[str]
    source: str  # "dictionary", "custom", "g2p", "acoustic", "manual"
    scope: str   # "once", "saved"
    confidence: float = 1.0


class OOVAssistant:
    """
    Manages OOV word detection, interactive user review with sounds-like guides,
    per-occurrence overrides, and persistent storage of custom pronunciations.
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

    def review_token_occurrence(
        self,
        token_index: int,
        word: str,
        interactive: bool = True,
        input_fn: Callable[[str], str] = input,
        recognizer: Optional[Any] = None,
    ) -> PronunciationChoice:
        """
        Performs structured pronunciation review for a specific token occurrence.
        Supports heteronym variant choice, manual entry with re-prompting on error,
        microphone capture, and 'use once' vs 'saved' scope.
        """
        clean_word = word.strip().lower()
        known_prons = self.lexicon.get_pronunciations(clean_word)
        is_known = self.lexicon.is_known(clean_word)

        if not interactive:
            if known_prons:
                return PronunciationChoice(
                    token_index=token_index,
                    text=clean_word,
                    phones=known_prons[0],
                    source="dictionary" if is_known else "g2p",
                    scope="once",
                    confidence=1.0 if is_known else 0.7,
                )
            else:
                # No pronunciation obtainable in headless mode; do not invent AH0
                raise RuntimeError(
                    f"No pronunciation could be determined for word '{clean_word}' in headless mode. "
                    "Provide explicit phonetic syntax (e.g. {P H O N E S}) or run interactively."
                )

        print("=" * 60)
        print(f"[?] Pronunciation Review: \"{clean_word}\" (token #{token_index + 1})")
        print("=" * 60)

        chosen_phones: List[str] = []
        choice_source = "dictionary"
        choice_scope = "once"
        choice_confidence = 1.0

        if is_known and len(known_prons) > 1:
            print(f"Known word with {len(known_prons)} pronunciation variants:")
            for v_idx, pr in enumerate(known_prons, 1):
                print(f"  [{v_idx}] {' '.join(pr)}")
                print(format_phoneme_breakdown(pr))
            print("  [m] Enter custom ARPAbet phonemes manually")
            print("  [r] Speak pronunciation into microphone")
            print(f"  [s] Skip / accept default ({' '.join(known_prons[0])})")

            choice = input_fn("> ").strip().lower()

            if choice.isdigit() and 1 <= int(choice) <= len(known_prons):
                chosen_phones = list(known_prons[int(choice) - 1])
                choice_source = "dictionary"
            elif choice == "m":
                chosen_phones, choice_source = self._prompt_manual_entry(clean_word, input_fn)
            elif choice == "r":
                chosen_phones, choice_source = self._prompt_microphone_entry(clean_word, input_fn, recognizer)
            else:
                chosen_phones = list(known_prons[0])
                choice_source = "dictionary"

        elif not is_known:
            suggestion = known_prons[0] if known_prons else None
            if suggestion:
                print(f"Unrecognized Word. G2P Suggestion: {' '.join(suggestion)}")
                print("Pronunciation Guide:")
                print(format_phoneme_breakdown(suggestion))
            else:
                print("Unrecognized Word. No G2P suggestion available.")

            print("\nChoose an action:")
            if suggestion:
                print(f"  [1] Accept suggestion ({' '.join(suggestion)})")
                print("  [2] Enter custom ARPAbet phonemes manually")
                print("  [3] Speak pronunciation into microphone")
                print("  [4] Ignore and proceed")
            else:
                print("  [1] Enter custom ARPAbet phonemes manually")
                print("  [2] Speak pronunciation into microphone")
                print("  [3] Ignore and proceed")

            choice = input_fn("> ").strip().lower()

            if suggestion and choice == "1":
                chosen_phones = list(suggestion)
                choice_source = "g2p"
            elif (suggestion and choice == "2") or (not suggestion and choice == "1"):
                chosen_phones, choice_source = self._prompt_manual_entry(clean_word, input_fn)
            elif (suggestion and choice == "3") or (not suggestion and choice == "2"):
                chosen_phones, choice_source = self._prompt_microphone_entry(clean_word, input_fn, recognizer)
            else:
                if suggestion:
                    chosen_phones = list(suggestion)
                    choice_source = "g2p"
                else:
                    raise RuntimeError(f"No pronunciation could be determined for OOV word '{clean_word}'")

        else:
            # Single variant known word
            return PronunciationChoice(
                token_index=token_index,
                text=clean_word,
                phones=known_prons[0],
                source="dictionary",
                scope="once",
                confidence=1.0,
            )

        # Prompt for scope (use once vs save) if manual, acoustic, or variant override
        print("\nChoose scope:")
        print("  [1] Use once (this occurrence only)")
        print(f"  [2] Save for future (persist to {self.custom_dict_path.name})")
        scope_choice = input_fn("> ").strip()
        if scope_choice == "2":
            choice_scope = "saved"
            self.lexicon.save_custom_pronunciation(clean_word, chosen_phones, self.custom_dict_path)
            print(f"[+] Saved '{clean_word}' -> {' '.join(chosen_phones)} to {self.custom_dict_path.name}")
        else:
            choice_scope = "once"
            print("[*] Applying for this occurrence only (not persisted).")

        return PronunciationChoice(
            token_index=token_index,
            text=clean_word,
            phones=chosen_phones,
            source=choice_source,
            scope=choice_scope,
            confidence=choice_confidence,
        )

    def _prompt_manual_entry(self, word: str, input_fn: Callable[[str], str]) -> Tuple[List[str], str]:
        """Prompts for ARPAbet phonemes and re-prompts on error until valid."""
        while True:
            user_input = input_fn("Enter ARPAbet phonemes (e.g. B AE1 K T AH0): ").strip()
            raw_tokens = user_input.split()
            if not raw_tokens:
                print("[!] Input cannot be empty. Please enter phonemes.")
                continue

            valid_phones: List[str] = []
            has_error = False
            for rt in raw_tokens:
                p_up = rt.upper()
                try:
                    validate_phone(p_up)
                    valid_phones.append(p_up)
                except ValueError as e:
                    print(f"[!] Validation error: {e}. Please re-enter valid ARPAbet phonemes.")
                    has_error = True
                    break

            if not has_error and valid_phones:
                return valid_phones, "manual"

    def _prompt_microphone_entry(
        self,
        word: str,
        input_fn: Callable[[str], str],
        recognizer: Optional[Any],
    ) -> Tuple[List[str], str]:
        """Records from microphone with retry on failure."""
        while True:
            try:
                if recognizer is None:
                    from phoneme_recognizer import PhonemeRecognizer
                    recognizer = PhonemeRecognizer(lexicon=self.lexicon)
                print(f"[*] Recording speech snippet for '{word}'...")
                audio = recognizer.record_audio_snippet(duration=2.5)
                rec_phones = recognizer.audio_to_arpabet(audio)
                if rec_phones:
                    print(f"[*] Interpreted phones: {' '.join(rec_phones)}")
                    print(format_phoneme_breakdown(rec_phones))
                    confirm = input_fn("Accept this pronunciation? [Y/n/retry]: ").strip().lower()
                    if confirm in ("", "y", "yes"):
                        return rec_phones, "acoustic"
                    elif confirm in ("r", "retry"):
                        continue
                    else:
                        break
                else:
                    print("[!] No speech detected. Retry recording? [Y/n]: ")
                    retry = input_fn("> ").strip().lower()
                    if retry in ("", "y", "yes"):
                        continue
                    break
            except Exception as e:
                print(f"[!] Microphone ingestion failed: {e}")
                retry = input_fn("Retry microphone? [y/N]: ").strip().lower()
                if retry in ("y", "yes"):
                    continue
                break

        # Fallback to manual
        return self._prompt_manual_entry(word, input_fn)

    def review_all_in_text(
        self,
        text: str,
        interactive: bool = True,
        input_fn: Callable[[str], str] = input,
        recognizer: Optional[Any] = None,
    ) -> List[PronunciationChoice]:
        """
        Reviews all speech tokens in text. Reviews OOV words and heteronyms per occurrence.
        """
        tokens = tokenize_hybrid_syntax(text)
        choices: List[PronunciationChoice] = []
        speech_token_idx = 0

        for tok in tokens:
            if tok.kind in ("gap", "punctuation"):
                continue

            if tok.kind == "word":
                w = tok.norm_text
                is_oov = not self.lexicon.is_known(w)
                is_heteronym = self.lexicon.is_heteronym(w)

                if is_oov or is_heteronym:
                    choice = self.review_token_occurrence(
                        token_index=speech_token_idx,
                        word=w,
                        interactive=interactive,
                        input_fn=input_fn,
                        recognizer=recognizer,
                    )
                    choices.append(choice)

            speech_token_idx += 1

        return choices

    def resolve_oov_word(
        self,
        word: str,
        interactive: bool = True,
        input_fn: Callable[[str], str] = input,
        recognizer: Optional[Any] = None,
    ) -> List[str]:
        """
        Legacy compatibility method for resolving an OOV word.
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
            valid_phones, _ = self._prompt_manual_entry(clean_word, input_fn)
            self.lexicon.save_custom_pronunciation(clean_word, valid_phones, self.custom_dict_path)
            print(f"[+] Saved '{clean_word}' -> {' '.join(valid_phones)} to {self.custom_dict_path.name}")
            return valid_phones

        elif choice == "3":
            try:
                if recognizer is None:
                    from phoneme_recognizer import PhonemeRecognizer
                    recognizer = PhonemeRecognizer(lexicon=self.lexicon)
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

        else:
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
        Legacy compatibility method: scans text for all OOV words and prompts user resolution.
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
