#!/usr/bin/env python3
"""
VTT Preprocessor for Montreal Forced Aligner (MFA)
Extracts discrete, clean speech intervals from video files using WebVTT subtitle files.
Handles ordinary multiline subtitles and YouTube rolling captions correctly.
Preserves genuine repetitions, punctuation, speaker tags, and source-time sample mappings.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def timestamp_to_seconds(ts_str: str) -> float:
    """Convert HH:MM:SS.mmm, HH:MM:SS,mmm, MM:SS.mmm, or MM:SS,mmm string to seconds float."""
    ts_str = ts_str.strip().replace(",", ".")
    parts = ts_str.split(":")
    if len(parts) == 3:
        h, m, s = parts
        return int(h) * 3600 + int(m) * 60 + float(s)
    elif len(parts) == 2:
        m, s = parts
        return int(m) * 60 + float(s)
    return float(ts_str)


def seconds_to_timestamp(sec: float) -> str:
    """Convert float seconds to HH:MM:SS.mmm string."""
    h = int(sec // 3600)
    sec %= 3600
    m = int(sec // 60)
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


def clean_vtt_text(text: str) -> str:
    """Remove HTML/VTT tags, bracketed markers, while preserving words, apostrophes, and hyphens."""
    # Remove tags like <c>, </c>, <v Speaker>, <00:00:01.000>
    text = re.sub(r"<[^>]+>", "", text)
    # Remove bracketed cues like [Music], [Applause]
    text = re.sub(r"\[.*?\]", "", text)
    # Remove YouTube marker artifacts
    text = re.sub(r">>|\"", "", text)
    # Normalize typographic apostrophes
    text = text.replace("’", "'").replace("‘", "'")
    # Replace non-word, non-apostrophe, non-hyphen, non-punctuation with space
    text = re.sub(r"[^\w\s\'-.,!?]", " ", text)
    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def clean_rolling_cues(cues: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Strips repeated carry-over text from consecutive rolling cues if cue time gap is <= 0.5s
    or overlapping. Ensures cue text strictly represents newly spoken dialogue.
    Handles cues containing either 'lines' (List[str]) or 'text' (str).
    """
    processed: List[Dict[str, Any]] = []

    for cue in cues:
        start_sec = float(cue["start_sec"])
        end_sec = float(cue["end_sec"])
        dur = end_sec - start_sec

        # Skip zero or negative duration glitches
        if dur <= 0.0:
            continue

        if "lines" in cue:
            cue_lines = cue["lines"]
            cleaned_lines = [clean_vtt_text(line_str) for line_str in cue_lines if clean_vtt_text(line_str)]
        elif "text" in cue:
            cue_lines = str(cue["text"]).split("\n")
            cleaned_lines = [clean_vtt_text(line_str) for line_str in cue_lines if clean_vtt_text(line_str)]
        else:
            continue

        if not cleaned_lines:
            continue

        # Check if this cue is a rolling update of the previous cue
        if processed:
            prev = processed[-1]
            gap = start_sec - prev["end_sec"]

            # In rolling displays (e.g. 2-line broadcast rolling captions),
            # previous lines are carried over into multi-line cues.
            # Only strip carry-over lines if the cue contains multiple lines (len(cleaned_lines) > 1),
            # or if it strictly overlaps in time (gap < -0.05) as an exact redundant redraw.
            # Genuine consecutive repetitions (e.g. "No!", "No!" with gap >= 0) must be preserved!
            if gap <= 0.5:
                prev_text = prev["text"].strip().lower()
                norm_prev = re.sub(r"[^\w\s\']", "", prev_text).strip()

                # Multi-line rolling carry-over: strip leading line only if remaining lines exist
                while len(cleaned_lines) > 1:
                    first_line_clean = cleaned_lines[0].strip().lower()
                    norm_first = re.sub(r"[^\w\s\']", "", first_line_clean).strip()

                    is_duplicate = False
                    if norm_first:
                        if norm_first == norm_prev or norm_prev.endswith(" " + norm_first) or norm_prev.endswith(norm_first):
                            is_duplicate = True
                    if not is_duplicate and first_line_clean:
                        if first_line_clean == prev_text or prev_text.endswith(" " + first_line_clean) or prev_text.endswith(first_line_clean):
                            is_duplicate = True

                    if is_duplicate:
                        cleaned_lines = cleaned_lines[1:]
                        if gap < 0:
                            start_sec = max(start_sec, prev["end_sec"])
                        dur = end_sec - start_sec
                    else:
                        break

                # For single-line cues, treat as duplicate if strongly overlapping in time (redraw glitch)
                # or if extremely short (<= 50ms) spacer/redraw carrying over identical text from prev
                if len(cleaned_lines) == 1:
                    first_line_clean = cleaned_lines[0].strip().lower()
                    norm_first = re.sub(r"[^\w\s\']", "", first_line_clean).strip()
                    is_dup_single = False
                    if norm_first and (norm_first == norm_prev or norm_prev.endswith(" " + norm_first) or norm_prev.endswith(norm_first)):
                        is_dup_single = True
                    if not is_dup_single and first_line_clean and (first_line_clean == prev_text or prev_text.endswith(" " + first_line_clean) or prev_text.endswith(first_line_clean)):
                        is_dup_single = True

                    if is_dup_single and (gap < -0.05 or dur <= 0.05):
                        cleaned_lines = []

        if not cleaned_lines or dur <= 0.0:
            continue

        combined_text = " ".join(cleaned_lines).strip()
        # Clean extra spaces before punctuation
        combined_text = re.sub(r"\s+([.,!?])", r"\1", combined_text)

        processed.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration": dur,
            "text": combined_text,
        })

    return processed


def parse_vtt_cues(vtt_path: str) -> List[Dict[str, Any]]:
    """
    Parse a WebVTT or SRT file and extract clean, non-duplicate speech intervals.
    Correctly handles:
      1. Ordinary multiline cues (preserves all lines in order).
      2. Rolling auto-generated captions with temporal overlap or <= 0.5s spacer gaps.
      3. Genuine repetitions that are temporally separated (> 0.5s).
      4. SubRip (.srt) comma millisecond separators and cue index headers.
    """
    with open(vtt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read().replace("\r\n", "\n").replace("\r", "\n")

    # Pattern for cues: start --> end [settings] \n [body]
    cue_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})[^\n]*\n([\s\S]*?)(?=\n\s*(?:\d{1,2}:\d{2}|\d+\n|\Z))"
    )

    raw_matches = cue_pattern.findall(content)
    raw_cues: List[Dict[str, Any]] = []
    for start_str, end_str, body in raw_matches:
        start_sec = timestamp_to_seconds(start_str)
        end_sec = timestamp_to_seconds(end_str)
        raw_lines = [line_str.strip() for line_str in body.split("\n") if line_str.strip()]
        if not raw_lines:
            continue
        raw_cues.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "lines": raw_lines
        })

    processed = clean_rolling_cues(raw_cues)

    # Merge small consecutive fragments with tiny gap (< 0.25s) from same cue grouping
    merged: List[Dict[str, Any]] = []
    for chunk in processed:
        if not merged:
            merged.append(chunk)
            continue
        prev = merged[-1]
        gap = chunk["start_sec"] - prev["end_sec"]
        if 0.0 <= gap < 0.20 and prev["duration"] < 1.0 and (prev["duration"] + chunk["duration"] + gap) < 12.0:
            prev["end_sec"] = chunk["end_sec"]
            prev["duration"] = prev["end_sec"] - prev["start_sec"]
            prev["text"] = f"{prev['text']} {chunk['text']}".strip()
        else:
            merged.append(chunk)

    return merged


def load_accurate_vtt_cues(vtt_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Loads subtitle cues from WebVTT or SRT with enhanced timestamp accuracy.
    If the file contains inline word timestamps (e.g. YouTube auto-captions),
    it refines cue boundaries and splits cues across uncaptioned gaps (>1.0s)
    to prevent uncaptioned speech from falsely inheriting neighboring dialogue.
    """
    vtt_p = Path(vtt_path).resolve()
    if not vtt_p.exists():
        return []

    try:
        with open(vtt_p, "r", encoding="utf-8", errors="replace") as f:
            content = f.read().replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return []

    cue_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}[.,]\d{3}|\d{1,2}:\d{2}[.,]\d{3})[^\n]*\n([\s\S]*?)(?=\n\s*(?:\d{1,2}:\d{2}|\d+\n|\Z))"
    )

    raw_matches = cue_pattern.findall(content)
    raw_cues: List[Dict[str, Any]] = []
    for s_str, e_str, body in raw_matches:
        raw_cues.append({
            "start_sec": timestamp_to_seconds(s_str),
            "end_sec": timestamp_to_seconds(e_str),
            "lines": [line_str.strip() for line_str in body.split("\n") if line_str.strip()],
        })

    cleaned = clean_rolling_cues(raw_cues)

    # Check for inline timestamps in raw lines
    has_inline_ts = bool(re.search(r"<\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3}>", content))
    if not has_inline_ts:
        return cleaned

    refined_cues: List[Dict[str, Any]] = []
    for c in cleaned:
        c_start = c["start_sec"]
        c_end = c["end_sec"]
        text_raw = c["text"]

        # Find matching raw cue to check for inline timestamps
        matching_raw_lines: List[str] = []
        for rc in raw_cues:
            if abs(rc["start_sec"] - c_start) < 0.1 and abs(rc["end_sec"] - c_end) < 0.1:
                matching_raw_lines = rc["lines"]
                break

        line_with_ts: Optional[str] = None
        for rl in matching_raw_lines:
            if re.search(r"<\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3}>", rl):
                clean_rl = clean_vtt_text(rl)
                if clean_rl and (clean_rl in text_raw or text_raw in clean_rl):
                    line_with_ts = rl
                    break

        if line_with_ts:
            parts = re.split(r"<(\d{1,2}:\d{2}(?::\d{2})?[.,]\d{3})>", line_with_ts)
            if len(parts) >= 3:
                # Extract fine-grained word timings
                word_timings: List[Dict[str, Any]] = []
                w_curr = clean_vtt_text(parts[0])
                t_curr = c_start

                for idx in range(1, len(parts), 2):
                    next_t = timestamp_to_seconds(parts[idx])
                    if w_curr:
                        est_w_dur = max(0.20, min(1.2, 0.08 * len(w_curr) + 0.15))
                        w_end = min(next_t, t_curr + est_w_dur)
                        word_timings.append({
                            "text": w_curr,
                            "start_sec": t_curr,
                            "end_sec": max(w_end, next_t - 0.02),
                        })
                    w_curr = clean_vtt_text(parts[idx + 1]) if idx + 1 < len(parts) else ""
                    t_curr = next_t

                if w_curr:
                    est_last_dur = max(0.35, min(1.2, 0.08 * len(w_curr) + 0.20))
                    last_end = min(c_end, t_curr + est_last_dur)
                    word_timings.append({
                        "text": w_curr,
                        "start_sec": t_curr,
                        "end_sec": last_end,
                    })

                # Sub-segment splitting across pauses (>0.8s)
                sub_segments: List[Tuple[float, float, str, List[Dict[str, Any]]]] = []
                seg_words: List[Dict[str, Any]] = []

                for wt in word_timings:
                    if seg_words:
                        prev_wt = seg_words[-1]
                        pause = wt["start_sec"] - prev_wt["end_sec"]
                        if pause > 0.65:
                            s_start = seg_words[0]["start_sec"]
                            s_end = seg_words[-1]["end_sec"]
                            s_text = " ".join(item["text"] for item in seg_words)
                            sub_segments.append((s_start, s_end, s_text, list(seg_words)))
                            seg_words = []
                    seg_words.append(wt)

                if seg_words:
                    s_start = seg_words[0]["start_sec"]
                    s_end = seg_words[-1]["end_sec"]
                    s_text = " ".join(item["text"] for item in seg_words)
                    sub_segments.append((s_start, s_end, s_text, list(seg_words)))

                if len(sub_segments) > 1:
                    for s, e, txt, w_list in sub_segments:
                        refined_cues.append({
                            "start_sec": s,
                            "end_sec": e,
                            "duration": e - s,
                            "text": txt,
                            "words": w_list,
                        })
                    continue
                elif sub_segments:
                    s, e, txt, w_list = sub_segments[0]
                    refined_cues.append({
                        "start_sec": s,
                        "end_sec": e,
                        "duration": e - s,
                        "text": txt,
                        "words": w_list,
                    })
                    continue

        refined_cues.append(c)

    return refined_cues


def preprocess_source(
    video_path: str,
    vtt_path: str,
    output_corpus_dir: str,
    speaker_id: Optional[str] = None,
    single_speaker_verified: bool = False,
    assume_single_speaker: bool = False,
    speaker_intervals: Optional[List[Dict[str, Any]]] = None,
    audio_source_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Extracts 16kHz mono WAV chunks and clean text files for Montreal Forced Aligner.
    Saves corpus_manifest.json mapping chunk IDs to global video offsets, sample bounds,
    and verified speaker identities. If audio_source_path is provided, chunks are sliced
    from the isolated vocal audio instead of the source video.
    """
    out_dir = Path(output_corpus_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    v_path = Path(video_path).resolve()
    vtt_p = Path(vtt_path).resolve()

    if not v_path.exists():
        raise FileNotFoundError(f"Video file not found: {v_path}")
    if not vtt_p.exists():
        raise FileNotFoundError(f"VTT file not found: {vtt_p}")

    chunks = parse_vtt_cues(str(vtt_p))

    # Resolve speaker eligibility
    is_verified = bool(single_speaker_verified or assume_single_speaker)
    effective_speaker = speaker_id if is_verified else None

    resolved_audio_src = str(Path(audio_source_path).resolve()) if audio_source_path and Path(audio_source_path).exists() else None

    manifest: Dict[str, Any] = {
        "source_video": str(v_path),
        "source_vtt": str(vtt_p),
        "audio_source_path": resolved_audio_src,
        "speaker_id": effective_speaker,
        "single_speaker_verified": is_verified,
        "assume_single_speaker": assume_single_speaker,
        "total_chunks": len(chunks),
        "chunks": {},
    }

    print(f"[*] Found {len(chunks)} speech intervals in {vtt_p.name}")

    for idx, c in enumerate(chunks):
        chunk_id = f"chunk_{idx:04d}"
        wav_path = out_dir / f"{chunk_id}.wav"
        txt_path = out_dir / f"{chunk_id}.txt"

        # Check speaker interval filtering if provided
        if speaker_intervals:
            in_interval = False
            for span in speaker_intervals:
                if c["start_sec"] >= span["start_sec"] and c["end_sec"] <= span["end_sec"]:
                    in_interval = True
                    break
            if not in_interval:
                continue

        # Save transcription for MFA
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(c["text"] + "\n")

        # Extract 16kHz mono 16-bit PCM WAV using ffmpeg
        start_ts = seconds_to_timestamp(c["start_sec"])
        dur = c["duration"]

        audio_input = resolved_audio_src if resolved_audio_src else str(v_path)
        cmd = [
            "ffmpeg", "-y",
            "-ss", start_ts,
            "-t", f"{dur:.3f}",
            "-i", audio_input,
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            str(wav_path)
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            print(f"[!] Warning: failed to extract audio for {chunk_id}")

        manifest["chunks"][chunk_id] = {
            "start_sec": c["start_sec"],
            "end_sec": c["end_sec"],
            "duration": c["duration"],
            "text": c["text"],
            "speaker_id": effective_speaker,
            "wav_path": str(wav_path),
            "txt_path": str(txt_path),
        }

    manifest_path = out_dir / "corpus_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    total_speech_sec = sum(c["duration"] for c in chunks)
    print(f"[+] Successfully generated {len(manifest['chunks'])} chunks ({total_speech_sec:.1f}s speech) in {out_dir}")
    print(f"[+] Manifest saved to {manifest_path}")
    return str(manifest_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video & VTT for MFA forced alignment.")
    parser.add_argument("--video", required=True, help="Path to input video file (.mp4, .webm, etc.)")
    parser.add_argument("--vtt", required=True, help="Path to WebVTT subtitles file (.vtt)")
    parser.add_argument("--output", default="./corpus", help="Output corpus directory")
    parser.add_argument("--speaker", default=None, help="Speaker identity")
    parser.add_argument("--assume-single-speaker", action="store_true", help="Explicitly assume single speaker")
    parser.add_argument("--audio-source", default=None, help="Path to isolated vocal stem or alternate audio source")
    args = parser.parse_args()

    preprocess_source(
        args.video,
        args.vtt,
        args.output,
        speaker_id=args.speaker,
        assume_single_speaker=args.assume_single_speaker,
        audio_source_path=args.audio_source,
    )
