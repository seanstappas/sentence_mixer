#!/usr/bin/env python3
"""
Microphone-to-Phoneme Ingestion & Acoustic Phoneme Recognizer (B07, D8).
Captures spoken speech (or WAV audio) and translates it into candidate ARPAbet phonetic strings
with stress markers per word (e.g. {B AE1 K AH0 N} {DH EH1 R}) for hybrid syntax synthesis.

Safety Invariants:
- Analysis-only: captured microphone audio is NEVER multiplexed into master PCM output.
- Supports interactive terminal review/edit with automated headless bypass and confidence gating.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import soundfile as sf

from lexicon import Lexicon, get_g2p, validate_phone


@dataclass
class RecognizedWordTarget:
    """Structured representation of a recognized word target."""
    word: str
    phones: List[str]
    tier: str  # "mfa_tier1", "ctc_tier2", "spelling_fallback"
    alternatives: List[List[str]] = field(default_factory=list)
    confidence: float = 1.0
    start_sec: float = 0.0
    end_sec: float = 0.0


class PhonemeRecognizer:
    """
    Acoustic phoneme recognition and microphone audio transcription engine.
    Supports Tier 1 MFA alignment against pronunciation variants, Tier 2 CTC/acoustic,
    and spelling-only fallback while strictly preserving word boundaries.
    """

    def __init__(
        self,
        lexicon: Optional[Lexicon] = None,
        aligner: Optional[Any] = None,
    ):
        self.lexicon = lexicon or Lexicon()
        self.aligner = aligner

    def check_audio_quality(self, audio_data: np.ndarray) -> None:
        """
        Validates audio quality: checks for empty or digital silence, and warns on clipping.
        """
        if len(audio_data) == 0:
            raise RuntimeError("Recorded audio is empty")
        rms = float(np.sqrt(np.mean(np.square(audio_data.astype(np.float32)))))
        if rms < 1e-4:
            raise RuntimeError("Recorded audio is silent: retry with microphone closer or increase input volume")
        clipped = float(np.mean(np.abs(audio_data) >= 0.99))
        if clipped > 0.05:
            print("[!] Warning: Audio appears clipped; consider lowering input gain.")

    def transcribe_audio_to_text(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribes speech audio array to text using faster-whisper without priming."""
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 16000 and len(audio_data) > 0:
            import librosa
            audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        try:
            from faster_whisper import WhisperModel
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                f"Whisper package ('faster-whisper') not installed: {e}. Please install faster-whisper or use text target."
            ) from e

        try:
            model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            # Unprimed transcription (D10 / B11)
            segments, _ = model.transcribe(audio_data, language="en", beam_size=5)
            text = " ".join(seg.text for seg in segments).strip()
            return text
        except RuntimeError:
            raise
        except Exception:
            return ""

    def transcribe_audio_to_words(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[Dict[str, Any]]:
        """
        Transcribes speech audio array to word entries with timestamps and probabilities.
        """
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        if sample_rate != 16000 and len(audio_data) > 0:
            import librosa
            audio_data = librosa.resample(audio_data.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
            sample_rate = 16000

        try:
            from faster_whisper import WhisperModel
        except (ImportError, ModuleNotFoundError) as e:
            raise RuntimeError(
                f"Whisper package ('faster-whisper') not installed: {e}. Please install faster-whisper or use text target."
            ) from e

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, _ = model.transcribe(audio_data, language="en", beam_size=5, word_timestamps=True)
        word_list: List[Dict[str, Any]] = []
        for seg in segments:
            if hasattr(seg, "words") and seg.words:
                for w in seg.words:
                    word_list.append({
                        "word": w.word.strip(),
                        "start": w.start,
                        "end": w.end,
                        "probability": getattr(w, "probability", 1.0),
                    })
            else:
                # Fallback splitting segment text
                tokens = seg.text.strip().split()
                if tokens:
                    dur_per = (seg.end - seg.start) / len(tokens)
                    for i, t in enumerate(tokens):
                        word_list.append({
                            "word": t,
                            "start": seg.start + i * dur_per,
                            "end": seg.start + (i + 1) * dur_per,
                            "probability": 0.9,
                        })
        return word_list

    def text_to_arpabet(self, text: str) -> List[str]:
        """Converts English text into CMU ARPAbet phone tokens with stress."""
        clean_words = [re.sub(r"[^\w']", "", w).lower() for w in text.split() if w.strip()]
        result_phones: List[str] = []
        g2p = get_g2p()

        for w in clean_words:
            if not w:
                continue
            prons = self.lexicon.get_pronunciations(w)
            if prons:
                result_phones.extend(prons[0])
            elif g2p is not None:
                phones = g2p(w)
                for p in phones:
                    p_clean = p.strip().upper()
                    if p_clean and p_clean not in " .,!?;:":
                        result_phones.append(p_clean)
            else:
                result_phones.append(w.upper())

        return result_phones

    def record_audio_snippet(
        self,
        duration: float = 2.5,
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """
        Records a short audio snippet from the microphone.
        Returns a 1D float32 numpy array normalized to [-1.0, 1.0].
        Raises RuntimeError if audio recording device or library is unavailable.
        """
        try:
            import sounddevice as sd  # type: ignore[import-not-found]

            num_frames = int(duration * sample_rate)
            rec = sd.rec(num_frames, samplerate=sample_rate, channels=1, dtype="float32")
            sd.wait()
            arr = np.squeeze(rec).astype(np.float32)
            self.check_audio_quality(arr)
            return arr
        except Exception as e:
            if "silent" in str(e) or "empty" in str(e):
                raise
            raise RuntimeError(f"Microphone recording unavailable: {e}") from e

    def audio_to_arpabet(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> List[str]:
        """
        Transcribes speech audio array and converts recognized text to ARPAbet phone tokens.
        """
        text = self.transcribe_audio_to_text(audio_data, sample_rate=sample_rate)
        if not text:
            return []
        return self.text_to_arpabet(text)

    def recognize_structured_words(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
        text_override: Optional[str] = None,
    ) -> List[RecognizedWordTarget]:
        """
        Recognizes words and resolves realized pronunciation variants using Tier 1 MFA
        or spelling fallback with all variants offered.
        """
        if text_override:
            clean_tokens = [re.sub(r"[^\w']", "", w).lower() for w in text_override.split() if w.strip()]
            word_entries = [{"word": t, "start": 0.0, "end": 0.0, "probability": 1.0} for t in clean_tokens if t]
        else:
            text = self.transcribe_audio_to_text(audio_data, sample_rate=sample_rate)
            if not text:
                return []
            try:
                word_entries = self.transcribe_audio_to_words(audio_data, sample_rate=sample_rate)
                if not word_entries:
                    word_entries = [{"word": t, "start": 0.0, "end": 0.0, "probability": 1.0} for t in text.split() if t.strip()]
            except Exception:
                word_entries = [{"word": t, "start": 0.0, "end": 0.0, "probability": 1.0} for t in text.split() if t.strip()]

        if not word_entries:
            return []

        results: List[RecognizedWordTarget] = []
        for we in word_entries:
            raw_w = str(we["word"])
            clean_w = re.sub(r"[^\w']", "", raw_w).lower()
            if not clean_w:
                continue

            variants = self.lexicon.get_pronunciations(clean_w)
            tier = "spelling_fallback"
            chosen_phones: List[str] = []
            alts: List[List[str]] = []

            # Tier 1: Forced alignment with aligner if available
            if self.aligner is not None and hasattr(self.aligner, "align_word") and variants:
                # Injected aligner chooses among listed variants
                tier = "mfa_tier1"
                chosen_phones = self.aligner.align_word(audio_data, clean_w, variants, float(str(we.get("start", 0.0))), float(str(we.get("end", 0.0))))
                alts = [v for v in variants if v != chosen_phones]
            elif variants:
                chosen_phones = list(variants[0])
                alts = list(variants[1:])
            else:
                # OOV fallback
                g2p = get_g2p()
                if g2p is not None:
                    raw_p = g2p(clean_w)
                    chosen_phones = [p.strip().upper() for p in raw_p if p.strip() and re.match(r"^[A-Z]+\d*$", p)]
                if not chosen_phones:
                    chosen_phones = [clean_w.upper()]

            results.append(
                RecognizedWordTarget(
                    word=clean_w,
                    phones=chosen_phones,
                    tier=tier,
                    alternatives=alts,
                    confidence=float(str(we.get("probability", 1.0))),
                    start_sec=float(str(we.get("start", 0.0))),
                    end_sec=float(str(we.get("end", 0.0))),
                )
            )

        return results

    def recognize_phonetic_string(
        self,
        audio_source: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        interactive: bool = True,
        min_confidence: float = 0.6,
        input_fn: Optional[Callable[[str], str]] = None,
    ) -> str:
        """
        Processes audio input into validated hybrid phonetic tokens per word:
        e.g. '{HH AH0 L OW1} {DH EH1 R}' (preserving word boundaries).
        Validates edited phones and re-prompts on error.
        In headless mode, rejects results below min_confidence.
        """
        if input_fn is None:
            input_fn = input
        if isinstance(audio_source, (str, Path)):
            audio_path = Path(audio_source).resolve()
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            data, sr = sf.read(str(audio_path))
        else:
            data = audio_source
            sr = sample_rate

        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Silence / quality check
        if np.all(data == 0) or np.sqrt(np.mean(np.square(data.astype(np.float32)))) < 1e-4:
            raise RuntimeError("Empty audio recognition result: recorded audio is silent or empty. Retry with microphone closer or increase volume.")

        rec_words = self.recognize_structured_words(data, sample_rate=sr)

        if not rec_words:
            raise RuntimeError(
                "Empty audio recognition result: retry with clearer audio or supply explicit phonemes."
            )

        is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()

        # Headless mode confidence gating
        if not (interactive and is_tty):
            low_conf_words = [w.word for w in rec_words if w.confidence < min_confidence]
            if low_conf_words:
                raise RuntimeError(
                    f"Low confidence audio recognition in headless mode for words: {', '.join(low_conf_words)}. "
                    "Run interactively or supply explicit text."
                )

        # Interactive per-word review
        if interactive and is_tty:
            print("\n[Microphone Audio Captured]")
            print(f"  Recognized utterance: '{' '.join(w.word for w in rec_words)}'")
            for idx, rw in enumerate(rec_words):
                w_disp = rw.word
                p_disp = " ".join(rw.phones)
                print(f"\n  Word [{idx + 1}/{len(rec_words)}]: \"{w_disp}\"")
                print(f"    Tier: {rw.tier} (confidence: {rw.confidence:.2f})")
                print(f"    Recognized Phones: {{{p_disp}}}")
                if rw.alternatives:
                    print(f"    Other Dictionary Variants: {[' '.join(a) for a in rw.alternatives]}")

                while True:
                    prompt = "    Accept [Enter] or enter modified phonemes (e.g. B AE1 K AH0 N): "
                    try:
                        user_val = input_fn(prompt).strip()
                    except (EOFError, KeyboardInterrupt):
                        user_val = ""

                    if not user_val:
                        break

                    clean_val = user_val.strip("{} ")
                    tokens = clean_val.split()
                    val_error = False
                    parsed_phones: List[str] = []
                    for t in tokens:
                        p_up = t.upper()
                        try:
                            validate_phone(p_up)
                            parsed_phones.append(p_up)
                        except ValueError as e:
                            print(f"    [!] Error: {e}. Please re-enter valid phones.")
                            val_error = True
                            break

                    if not val_error and parsed_phones:
                        rw.phones = parsed_phones
                        break

        # Output phonetic string with preserved word boundaries
        return " ".join("{" + " ".join(w.phones) + "}" for w in rec_words)


def assert_no_mic_audio_in_plan(plan: Any) -> None:
    """Safety check verifying that no temporary microphone analysis audio leaked into the plan."""
    segments = getattr(plan, "segments", [])

    for s in segments:
        v_path = getattr(s, "source_video_path", "")
        c_id = getattr(s, "source_content_id", "")
        if "mic_" in v_path.lower() or "mic_audio" in c_id.lower() or "recorded_mic" in v_path.lower():
            raise AssertionError(f"Safety violation: microphone analysis audio detected in output edit plan segment: {v_path}")
