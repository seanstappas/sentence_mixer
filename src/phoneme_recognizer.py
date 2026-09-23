#!/usr/bin/env python3
"""
Microphone-to-Phoneme Ingestion & Acoustic Phoneme Recognizer (Milestone M3 / R2).
Captures spoken speech (or WAV audio) and translates it into candidate ARPAbet phonetic strings
with stress markers (e.g. {B AE1 K AH0 N}) for hybrid syntax synthesis.

Safety Invariants:
- Analysis-only: captured microphone audio is NEVER multiplexed into master PCM output.
- Supports interactive terminal review/edit with automated headless bypass.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import soundfile as sf

from lexicon import Lexicon, get_g2p


class PhonemeRecognizer:
    """
    Acoustic phoneme recognition and microphone audio transcription engine.
    """

    def __init__(self, lexicon: Optional[Lexicon] = None):
        self.lexicon = lexicon or Lexicon()

    def transcribe_audio_to_text(
        self,
        audio_data: np.ndarray,
        sample_rate: int = 16000,
    ) -> str:
        """Transcribes speech audio array to text using faster-whisper."""
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        try:
            from faster_whisper import WhisperModel
            model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
            segments, _ = model.transcribe(audio_data, language="en", beam_size=5)
            text = " ".join(seg.text for seg in segments).strip()
            return text
        except Exception:
            # Fallback if faster-whisper not available or audio empty
            return ""

    def text_to_arpabet(self, text: str) -> List[str]:
        """Converts English text into CMU ARPAbet phone tokens with stress."""
        clean_words = [w.strip().lower() for w in text.split() if w.strip()]
        result_phones: List[str] = []

        g2p = get_g2p()

        for w in clean_words:
            # Check lexicon first for canonical pronunciation
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
                # Basic phonetic fallback
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
            return np.squeeze(rec).astype(np.float32)
        except Exception as e:
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

    def recognize_phonetic_string(
        self,
        audio_source: Union[str, Path, np.ndarray],
        sample_rate: int = 16000,
        interactive: bool = True,
    ) -> str:
        """
        Processes audio input into a validated hybrid phonetic string, e.g. '{B AE1 K AH0 N}'.
        Supports interactive TTY review and editing.
        """
        if isinstance(audio_source, (str, Path)):
            audio_path = Path(audio_source).resolve()
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio file not found: {audio_path}")
            data, sr = sf.read(str(audio_path))
        else:
            data = audio_source
            sr = sample_rate

        text = self.transcribe_audio_to_text(data, sample_rate=sr)
        phones = self.text_to_arpabet(text)

        if not phones:
            raise RuntimeError(
                "Empty audio recognition result: retry with clearer audio or supply explicit phonemes."
            )

        candidate_str = "{" + " ".join(phones) + "}"

        is_tty = hasattr(sys.stdin, "isatty") and sys.stdin.isatty()
        if interactive and is_tty:
            print("\n[Microphone Audio Captured]")
            print(f"  Transcript: '{text}'")
            print(f"  Recognized Phonemes: {candidate_str}")
            prompt = "  Accept [Enter] or enter modified phonemes: "
            try:
                user_val = input(prompt).strip()
                if user_val:
                    if not user_val.startswith("{"):
                        user_val = "{" + user_val
                    if not user_val.endswith("}"):
                        user_val = user_val + "}"
                    candidate_str = user_val
            except (EOFError, KeyboardInterrupt):
                pass

        return candidate_str


def assert_no_mic_audio_in_plan(plan: Any) -> None:
    """Safety check verifying that no temporary microphone analysis audio leaked into the plan."""
    segments = getattr(plan, "segments", [])

    for s in segments:
        v_path = getattr(s, "source_video_path", "")
        c_id = getattr(s, "source_content_id", "")
        if "mic_" in v_path.lower() or "mic_audio" in c_id.lower() or "recorded_mic" in v_path.lower():
            raise AssertionError(f"Safety violation: microphone analysis audio detected in output edit plan segment: {v_path}")
