#!/usr/bin/env python3
"""
Independent ASR Listening Verification Tool (Milestone M1 / R1)
Transcribes rendered MP4 video or WAV audio using Whisper / faster-whisper,
conditions language decoding with initial_prompt for target vocabulary priming,
computes Word Error Rate (WER) and Character Error Rate (CER) via dynamic programming,
and detects missing target words or unintended dialogue bleeding (insertions).

Exit codes:
  0: Transcription matches expected target text within tolerance
  1: Verification failed (WER/CER exceeded, missing words, or dialogue bleed detected)
  2: Operational error (missing file, invalid arguments, model load failure)
"""

from __future__ import annotations

import argparse
import io
import json
import re
import subprocess
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add repo root and src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Lazy-loaded Lexicon for phonetic homophone checking
_LEXICON_INSTANCE = None


def get_lexicon():
    """Lazily load lexicon for phonetic homophone checking."""
    global _LEXICON_INSTANCE
    if _LEXICON_INSTANCE is None:
        try:
            from lexicon import Lexicon
            dict_paths = [
                REPO_ROOT / "dictionaries" / "custom_pronunciations.txt",
                REPO_ROOT / "dictionaries" / "merged_dictionary.dict",
            ]
            valid_paths = [p for p in dict_paths if p.exists()]
            _LEXICON_INSTANCE = Lexicon(dict_paths=valid_paths)
        except Exception:
            _LEXICON_INSTANCE = None
    return _LEXICON_INSTANCE


@lru_cache(maxsize=1024)
def are_homophones(word1: str, word2: str) -> bool:
    """
    Check if two words share an identical phonetic pronunciation
    (stress-stripped) using Lexicon / G2P.
    """
    w1 = word1.strip().lower()
    w2 = word2.strip().lower()
    if w1 == w2:
        return True

    lex = get_lexicon()
    if lex is None:
        return False

    try:
        from lexicon import strip_phone_stress

        prons1 = lex.get_pronunciations(w1)
        prons2 = lex.get_pronunciations(w2)
        if not prons1 or not prons2:
            return False

        stripped_p1 = {tuple(strip_phone_stress(p) for p in pr) for pr in prons1}
        stripped_p2 = {tuple(strip_phone_stress(p) for p in pr) for pr in prons2}
        return bool(stripped_p1 & stripped_p2)
    except Exception:
        return False


def normalize_text(text: str) -> str:
    """
    Normalize text for speech recognition comparison:
    - Strips explicit <gap:...> cadence instructions
    - Strips explicit phonetic markup {...}
    - Lowercases text
    - Replaces typographic quotes and hyphens with spaces
    - Removes punctuation
    - Collapses multiple whitespace characters
    """
    if not text:
        return ""

    # Strip explicit gaps
    t = re.sub(r"<gap:[^>]+>", " ", text)
    # Strip explicit phone groups { ... }
    t = re.sub(r"\{[^}]*\}", " ", t)

    # Replace typographical quotes, dashes, hyphens
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = t.replace("-", " ").replace("—", " ").replace("_", " ")

    # Extract alphanumeric word tokens
    tokens = re.findall(r"[a-zA-Z0-9]+", t.lower())
    return " ".join(tokens)


def compute_levenshtein_tokens(
    ref_tokens: List[str],
    hyp_tokens: List[str],
    allow_homophones: bool = True,
) -> Tuple[int, List[Tuple[str, str, str]]]:
    """
    Computes Levenshtein edit distance and backtracks optimal alignment.
    Operations returned as (op, ref_token, hyp_token) where op in {'M', 'S', 'D', 'I'}:
      'M': Match (cost 0)
      'S': Substitution (cost 1, or 0 if homophones allowed)
      'D': Deletion / missing from hypothesis (cost 1)
      'I': Insertion / bleeding into hypothesis (cost 1)
    """
    m = len(ref_tokens)
    n = len(hyp_tokens)

    # dp[i][j] = (cost, op_type)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        r = ref_tokens[i - 1]
        for j in range(1, n + 1):
            h = hyp_tokens[j - 1]

            is_match = (r == h)
            is_homo = allow_homophones and are_homophones(r, h)
            sub_cost = 0 if (is_match or is_homo) else 1

            cost_match_sub = dp[i - 1][j - 1] + sub_cost
            cost_del = dp[i - 1][j] + 1  # reference word deleted
            cost_ins = dp[i][j - 1] + 1  # hypothesis word inserted

            dp[i][j] = min(cost_match_sub, cost_del, cost_ins)

    # Backtracking
    i = m
    j = n
    ops: List[Tuple[str, str, str]] = []

    while i > 0 or j > 0:
        if i > 0 and j > 0:
            r = ref_tokens[i - 1]
            h = hyp_tokens[j - 1]
            is_match = (r == h)
            is_homo = allow_homophones and are_homophones(r, h)
            sub_cost = 0 if (is_match or is_homo) else 1

            if dp[i][j] == dp[i - 1][j - 1] + sub_cost:
                op = "M" if sub_cost == 0 else "S"
                ops.append((op, r, h))
                i -= 1
                j -= 1
                continue

        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(("D", ref_tokens[i - 1], ""))
            i -= 1
            continue

        if j > 0 and dp[i][j] == dp[i][j - 1] + 1:
            ops.append(("I", "", hyp_tokens[j - 1]))
            j -= 1
            continue

        # Fallback if edge reached
        if i > 0:
            ops.append(("D", ref_tokens[i - 1], ""))
            i -= 1
        elif j > 0:
            ops.append(("I", "", hyp_tokens[j - 1]))
            j -= 1

    ops.reverse()
    return dp[m][n], ops


def compute_cer(ref_str: str, hyp_str: str) -> float:
    """
    Computes Character Error Rate (CER) between normalized strings.
    """
    r_chars = list(ref_str)
    h_chars = list(hyp_str)
    m = len(r_chars)
    n = len(h_chars)

    if m == 0:
        return 0.0 if n == 0 else 1.0

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        dp[i][0] = i
    for j in range(1, n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if r_chars[i - 1] == h_chars[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j - 1] + cost,
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
            )

    return float(dp[m][n]) / float(m)


def extract_audio_16k(input_path: Union[str, Path]) -> np.ndarray:
    """
    Extracts 16kHz mono float32 audio from MP4 video or WAV audio.
    Prefers FFmpeg stream extraction to avoid video decoding overhead.
    Falls back to soundfile for WAV files if FFmpeg is unavailable.
    """
    path = Path(input_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Media file not found: {path}")

    # Try FFmpeg extraction first
    try:
        cmd = [
            "ffmpeg", "-nostats", "-loglevel", "error",
            "-i", str(path),
            "-vn", "-c:a", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-f", "wav", "-"
        ]
        proc = subprocess.run(cmd, capture_output=True, check=True)
        data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=-1)
        return data.astype(np.float32)
    except Exception as ffmpeg_err:
        # Fallback to soundfile for audio files
        if path.suffix.lower() in [".wav", ".flac", ".ogg"]:
            try:
                data, sr = sf.read(str(path), dtype="float32")
                if data.ndim > 1:
                    data = data.mean(axis=-1)
                if sr != 16000:
                    try:
                        from scipy.signal import resample_poly
                        import math
                        gcd = math.gcd(sr, 16000)
                        data = resample_poly(data, 16000 // gcd, sr // gcd)
                    except ImportError:
                        pass
                return data.astype(np.float32)
            except Exception as sf_err:
                raise RuntimeError(
                    f"Failed to read audio from {path} via FFmpeg ({ffmpeg_err}) and soundfile ({sf_err})"
                )
        raise RuntimeError(f"FFmpeg failed to extract audio from {path}: {ffmpeg_err}")


class ASRBackend:
    """Base class for speech recognition backends."""

    def transcribe(self, audio: np.ndarray, initial_prompt: Optional[str] = None) -> str:
        raise NotImplementedError


class FasterWhisperBackend(ASRBackend):
    """Transcription backend using faster-whisper (CTranslate2)."""

    def __init__(self, model_name: str = "tiny.en", device: str = "cpu"):
        from faster_whisper import WhisperModel

        # Map common huggingface model names to faster-whisper IDs
        m_name = model_name
        if "whisper-tiny.en" in m_name:
            m_name = "tiny.en"
        elif "whisper-base.en" in m_name:
            m_name = "base.en"
        elif "whisper-small.en" in m_name:
            m_name = "small.en"
        elif "whisper-tiny" in m_name:
            m_name = "tiny"
        elif "whisper-base" in m_name:
            m_name = "base"

        compute_type = "int8" if device == "cpu" else "float16"
        self.model = WhisperModel(m_name, device=device, compute_type=compute_type)

    def transcribe(self, audio: np.ndarray, initial_prompt: Optional[str] = None) -> str:
        if len(audio) == 0:
            return ""
        segments, _ = self.model.transcribe(
            audio,
            initial_prompt=initial_prompt,
            language="en",
            beam_size=5,
        )
        return " ".join(seg.text for seg in segments).strip()


class TransformersBackend(ASRBackend):
    """Transcription backend using HuggingFace transformers pipeline."""

    def __init__(self, model_name: str = "openai/whisper-tiny.en", device: str = "cpu"):
        from transformers import pipeline

        m_name = model_name
        if m_name in ["tiny.en", "tiny"]:
            m_name = "openai/whisper-tiny.en"
        elif m_name in ["base.en", "base"]:
            m_name = "openai/whisper-base.en"

        device_arg = 0 if (device.startswith("cuda") or device == "gpu") else -1
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=m_name,
            device=device_arg,
        )

    def transcribe(self, audio: np.ndarray, initial_prompt: Optional[str] = None) -> str:
        if len(audio) == 0:
            return ""

        generate_kwargs: Dict[str, Any] = {}
        if initial_prompt:
            try:
                p_ids = self.pipe.tokenizer.get_prompt_ids(initial_prompt, return_tensors="pt")
                generate_kwargs["prompt_ids"] = p_ids
            except Exception:
                pass

        try:
            res = self.pipe(audio, generate_kwargs=generate_kwargs if generate_kwargs else None)
        except Exception:
            # Fallback without generate_kwargs if unsupported
            res = self.pipe(audio)

        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], dict):
            return str(res[0].get("text", "")).strip()
        elif isinstance(res, dict):
            return str(res.get("text", "")).strip()
        return ""


_BACKEND_CACHE: Dict[str, ASRBackend] = {}


def get_backend(backend: str = "auto", model_name: str = "openai/whisper-tiny.en", device: str = "cpu") -> ASRBackend:
    """
    Resolves and caches an ASR backend instance.
    'auto' prefers faster-whisper, falling back to transformers.
    """
    cache_key = f"{backend}:{model_name}:{device}"
    if cache_key in _BACKEND_CACHE:
        return _BACKEND_CACHE[cache_key]

    chosen: Optional[ASRBackend] = None

    if backend in ["faster-whisper", "faster_whisper"]:
        chosen = FasterWhisperBackend(model_name=model_name, device=device)
    elif backend in ["transformers", "hf"]:
        chosen = TransformersBackend(model_name=model_name, device=device)
    elif backend == "auto":
        try:
            chosen = FasterWhisperBackend(model_name=model_name, device=device)
        except Exception:
            chosen = TransformersBackend(model_name=model_name, device=device)
    else:
        raise ValueError(f"Unknown ASR backend: '{backend}'. Expected 'auto', 'faster-whisper', or 'transformers'.")

    _BACKEND_CACHE[cache_key] = chosen
    return chosen


def verify_listening(
    input_path: Union[str, Path],
    expected_text: str,
    model_name: str = "openai/whisper-tiny.en",
    backend: str = "auto",
    wer_threshold: float = 0.20,
    disallow_insertions: bool = True,
    device: str = "cpu",
    allow_homophones: bool = True,
    max_cer: Optional[float] = None,
    tolerance: Optional[float] = None,
    audio_array: Optional[np.ndarray] = None,
    asr_backend: Optional[ASRBackend] = None,
    require_all_words: bool = True,
) -> Dict[str, Any]:
    """
    Transcribes audio/video media and verifies transcription against expected text.

    Args:
        input_path: Path to rendered MP4 video or WAV audio.
        expected_text: Target text expected to be spoken.
        model_name: Whisper model identifier.
        backend: 'auto', 'faster-whisper', or 'transformers'.
        wer_threshold: Maximum allowed Word Error Rate (default: 0.20).
        disallow_insertions: If True, reject any extra spoken words (dialogue bleed).
        device: 'cpu' or 'cuda'.
        allow_homophones: If True, treat homophones (e.g. MOXY/MOXIE) as matches.
        max_cer: Maximum allowed Character Error Rate (optional).
        tolerance: Alias for wer_threshold.
        audio_array: Optional pre-loaded 16kHz float32 audio array.
        asr_backend: Optional pre-initialized ASRBackend instance.

    Returns:
        Structured dictionary conforming to M1 interface contracts.
    """
    if tolerance is not None:
        wer_threshold = tolerance

    # 1. Ingest audio
    if audio_array is not None:
        audio = audio_array
    else:
        audio = extract_audio_16k(input_path)

    # 2. Transcribe with initial_prompt conditioning
    if asr_backend is not None:
        engine = asr_backend
    else:
        engine = get_backend(backend=backend, model_name=model_name, device=device)

    transcribed_text = engine.transcribe(audio, initial_prompt=expected_text)

    # 3. Normalize text
    norm_expected = normalize_text(expected_text)
    norm_transcribed = normalize_text(transcribed_text)

    ref_tokens = norm_expected.split() if norm_expected else []
    hyp_tokens = norm_transcribed.split() if norm_transcribed else []

    if not ref_tokens:
        raise ValueError(
            f"Expected text '{expected_text}' contains no recognizable lexical words for ASR verification. Explicit transcript is required for phonetic-only targets."
        )

    # 4. Levenshtein token alignment
    dist, ops = compute_levenshtein_tokens(ref_tokens, hyp_tokens, allow_homophones=allow_homophones)

    word_matches: List[str] = []
    substitutions: List[Tuple[str, str]] = []
    deletions: List[str] = []
    insertions: List[str] = []

    missing_words: List[str] = []
    bleeding_words: List[str] = []

    for op, r_w, h_w in ops:
        if op == "M":
            word_matches.append(r_w)
        elif op == "S":
            substitutions.append((r_w, h_w))
            missing_words.append(r_w)
            # Substitutions are phonetic / recognition mismatches, NOT dialogue bleed
        elif op == "D":
            deletions.append(r_w)
            missing_words.append(r_w)
        elif op == "I":
            insertions.append(h_w)
            # Only genuine hypothesis insertions constitute dialogue bleed
            bleeding_words.append(h_w)

    ref_len = len(ref_tokens)
    num_errs = len(substitutions) + len(deletions) + len(insertions)
    wer = float(num_errs) / float(max(1, ref_len)) if ref_len > 0 else (0.0 if not hyp_tokens else 1.0)
    cer = compute_cer(norm_expected, norm_transcribed)

    # 5. Word presence
    word_presence = {w: (w in word_matches) for w in ref_tokens}
    all_expected_words_present = (len(missing_words) == 0 and ref_len > 0) if ref_len > 0 else True
    bleed_detected = len(bleeding_words) > 0

    # 6. Evaluate acceptance criteria
    failure_reasons: List[str] = []

    if wer > wer_threshold:
        failure_reasons.append(f"WER {wer:.2f} exceeds threshold {wer_threshold:.2f}")

    if max_cer is not None and cer > max_cer:
        failure_reasons.append(f"CER {cer:.2f} exceeds threshold {max_cer:.2f}")

    if (require_all_words or wer > wer_threshold) and not all_expected_words_present:
        failure_reasons.append(f"Missing target words: {missing_words}")

    if disallow_insertions and bleed_detected:
        failure_reasons.append(f"Unintended dialogue bleed detected: {bleeding_words}")

    passed = (len(failure_reasons) == 0)

    # Return unified dictionary covering both PROJECT.md and analysis schemas
    return {
        "input_path": str(input_path),
        "expected_text": expected_text,
        "transcribed_text": transcribed_text,
        "normalized_expected": norm_expected,
        "normalized_transcribed": norm_transcribed,
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "word_matches": word_matches,
        "missing_words": missing_words,
        "bleeding_words": bleeding_words,
        "extraneous_words": bleeding_words,
        "bleed_detected": bleed_detected,
        "all_expected_words_present": all_expected_words_present,
        "word_presence": word_presence,
        "passed": passed,
        "failure_reasons": failure_reasons,
        "backend": type(engine).__name__,
        "model": model_name,
        "metrics": {
            "wer": round(wer, 4),
            "cer": round(cer, 4),
            "substitutions_count": len(substitutions),
            "deletions_count": len(deletions),
            "insertions_count": len(insertions),
            "substitutions": substitutions,
            "deletions": deletions,
            "insertions": insertions,
        },
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify synthesized speech audio/video against target text via Whisper ASR."
    )
    parser.add_argument("--input", "-i", required=True, help="Path to rendered MP4 video or master WAV audio")
    parser.add_argument("--expected", "-e", required=True, help="Expected target text")
    parser.add_argument(
        "--model", "--model-size",
        dest="model",
        default="openai/whisper-tiny.en",
        help="Whisper model name/size (default: openai/whisper-tiny.en)",
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "faster-whisper", "transformers"],
        default="auto",
        help="ASR backend (default: auto)",
    )
    parser.add_argument(
        "--tolerance", "--max-wer",
        dest="tolerance",
        type=float,
        default=0.20,
        help="Maximum allowed Word Error Rate (default: 0.20)",
    )
    parser.add_argument(
        "--max-cer",
        type=float,
        default=None,
        help="Maximum allowed Character Error Rate (optional)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="Inference device (default: cpu)",
    )
    parser.add_argument(
        "--allow-insertions", "--allow-extraneous",
        dest="allow_insertions",
        action="store_true",
        help="Allow extra spoken words without flagging bleeding dialogue",
    )
    parser.add_argument(
        "--no-homophones",
        dest="allow_homophones",
        action="store_false",
        default=True,
        help="Disable phonetic homophone matching",
    )
    parser.add_argument(
        "--require-all-words",
        action="store_true",
        default=False,
        help="Require 100% of target words even if WER is within tolerance",
    )
    parser.add_argument("--json", action="store_true", help="Output results in JSON format")
    parser.add_argument("--output-json", help="Path to write JSON results file")

    args = parser.parse_args(argv)

    input_file = Path(args.input)
    if not input_file.exists():
        print(f"[!] Error: Input file does not exist: {input_file}", file=sys.stderr)
        return 2

    try:
        result = verify_listening(
            input_path=input_file,
            expected_text=args.expected,
            model_name=args.model,
            backend=args.backend,
            wer_threshold=args.tolerance,
            disallow_insertions=not args.allow_insertions,
            device=args.device,
            allow_homophones=args.allow_homophones,
            max_cer=args.max_cer,
            require_all_words=args.require_all_words,
        )
    except Exception as exc:
        print(f"[!] Operational error during ASR verification: {exc}", file=sys.stderr)
        return 2

    if args.output_json:
        out_p = Path(args.output_json)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        status_str = "PASSED" if result["passed"] else "FAILED"
        print(f"[*] ASR Verification: {status_str}")
        print(f"    Input:       {result['input_path']}")
        print(f"    Expected:    \"{result['expected_text']}\"")
        print(f"    Recognized:  \"{result['transcribed_text']}\"")
        print(f"    Normalized:  \"{result['normalized_transcribed']}\"")
        print(f"    WER:         {result['wer']:.2f} (tolerance: {args.tolerance:.2f})")
        print(f"    CER:         {result['cer']:.2f}")
        print(f"    Matches:     {result['word_matches']}")
        if result["missing_words"]:
            print(f"    Missing:     {result['missing_words']}")
        if result["bleeding_words"]:
            print(f"    Bleeding:    {result['bleeding_words']}")
        if not result["passed"]:
            print(f"    Reasons:     {', '.join(result['failure_reasons'])}")

    return 0 if result["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
