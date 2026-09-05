#!/usr/bin/env python3
"""
VTT Preprocessor for Montreal Forced Aligner (MFA)
Extracts discrete, clean speech intervals from video files using WebVTT subtitle files.
Handles YouTube rolling captions, removes non-speech tokens, and exports 16kHz mono WAV chunks.
"""

import argparse
import json
import os
import re
import subprocess
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def timestamp_to_seconds(ts_str: str) -> float:
    """Convert HH:MM:SS.mmm or MM:SS.mmm string to seconds float."""
    parts = ts_str.strip().split(":")
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


def clean_vtt_line(line: str) -> str:
    """Remove HTML/VTT tags, speaker tags, bracketed markers, and special chars."""
    # Remove timestamps like <00:00:00.400> and tags like <c>, </c>, <v Rick>
    text = re.sub(r"<[^>]+>", "", line)
    # Remove bracketed cues like [Music], [Applause], [laughter]
    text = re.sub(r"\[.*?\]", "", text)
    # Remove YouTube marker artifacts
    text = re.sub(r">>|\"", "", text)
    # Normalize curly apostrophes
    text = text.replace("’", "'").replace("‘", "'")
    # Replace anything that isn't word characters, apostrophe, or whitespace with space
    text = re.sub(r"[^\w\s\']", " ", text)
    # Normalize multiple whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_vtt_cues(vtt_path: str):
    """
    Parse a WebVTT file and extract clean, non-duplicate speech intervals.
    Accounts for YouTube's auto-generated rolling caption structure.
    """
    with open(vtt_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    # Match standard VTT cues: start --> end [settings] \n [cue text]
    cue_pattern = re.compile(
        r"(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3})\s*-->\s*(\d{1,2}:\d{2}:\d{2}\.\d{3}|\d{1,2}:\d{2}\.\d{3})[^\n]*\n([\s\S]*?)(?=\n\s*(?:\d{1,2}:\d{2}|\d+\n|\Z))"
    )

    raw_matches = cue_pattern.findall(content)
    raw_cues = []

    for start_str, end_str, body in raw_matches:
        start_sec = timestamp_to_seconds(start_str)
        end_sec = timestamp_to_seconds(end_str)
        lines = [l.strip() for l in body.split("\n") if l.strip()]
        if not lines:
            continue
        raw_cues.append((start_sec, end_sec, lines))

    processed_chunks = []
    seen_texts = set()

    for idx, (start_sec, end_sec, lines) in enumerate(raw_cues):
        dur = end_sec - start_sec
        # Skip micro flash cues common in rolling captions (often 10ms)
        if dur < 0.15:
            continue

        # In YouTube rolling captions, lines often contain:
        # Line 0: previously shown text
        # Line 1: new incoming text with inline word timestamps <00:00:01.200><c> word</c>
        tag_lines = [l for l in lines if "<c>" in l or "<" in l]
        chosen_line = tag_lines[-1] if tag_lines else lines[-1]

        cleaned_text = clean_vtt_line(chosen_line)
        if not cleaned_text or len(cleaned_text.split()) < 1:
            continue

        # Avoid exact duplicate text blocks in immediately adjacent cues
        text_key = cleaned_text.lower()
        if text_key in seen_texts and len(processed_chunks) > 0:
            last_text = processed_chunks[-1]["text"].lower()
            if text_key == last_text:
                continue

        seen_texts.add(text_key)
        # Keep set reasonable size
        if len(seen_texts) > 50:
            seen_texts.clear()

        processed_chunks.append({
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration": dur,
            "text": cleaned_text
        })

    # Merge extremely short adjacent chunks if their text is small and gap is minimal (< 0.2s)
    merged_chunks = []
    for chunk in processed_chunks:
        if not merged_chunks:
            merged_chunks.append(chunk)
            continue
        prev = merged_chunks[-1]
        gap = chunk["start_sec"] - prev["end_sec"]
        # Merge if previous chunk is very short (< 1.2s) and gap is small
        if gap >= 0 and gap < 0.25 and prev["duration"] < 1.2 and (prev["duration"] + chunk["duration"] + gap) < 15.0:
            prev["end_sec"] = chunk["end_sec"]
            prev["duration"] = prev["end_sec"] - prev["start_sec"]
            prev["text"] = f"{prev['text']} {chunk['text']}".strip()
        else:
            merged_chunks.append(chunk)

    return merged_chunks


def preprocess_source(video_path: str, vtt_path: str, output_corpus_dir: str):
    """
    Extracts 16kHz mono WAV chunks and clean text files for Montreal Forced Aligner.
    Saves corpus_manifest.json mapping chunk IDs to global video offsets.
    """
    os.makedirs(output_corpus_dir, exist_ok=True)
    chunks = parse_vtt_cues(vtt_path)
    manifest = {
        "source_video": os.path.abspath(video_path),
        "source_vtt": os.path.abspath(vtt_path),
        "total_chunks": len(chunks),
        "chunks": {}
    }

    print(f"[*] Found {len(chunks)} speech intervals in {os.path.basename(vtt_path)}")

    for idx, c in enumerate(chunks):
        chunk_id = f"chunk_{idx:04d}"
        wav_path = os.path.join(output_corpus_dir, f"{chunk_id}.wav")
        txt_path = os.path.join(output_corpus_dir, f"{chunk_id}.txt")

        # Save transcription for MFA
        with open(txt_path, "w", encoding="utf-8") as tf:
            tf.write(c["text"] + "\n")

        # Extract 16kHz mono 16-bit PCM WAV using ffmpeg
        start_ts = seconds_to_timestamp(c["start_sec"])
        dur = c["duration"]

        cmd = [
            "ffmpeg", "-y",
            "-ss", start_ts,
            "-t", f"{dur:.3f}",
            "-i", video_path,
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            wav_path
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if res.returncode != 0:
            print(f"[!] Warning: failed to extract audio for {chunk_id}")

        manifest["chunks"][chunk_id] = {
            "start_sec": c["start_sec"],
            "end_sec": c["end_sec"],
            "duration": c["duration"],
            "text": c["text"],
            "wav_path": os.path.abspath(wav_path),
            "txt_path": os.path.abspath(txt_path)
        }

    manifest_path = os.path.join(output_corpus_dir, "corpus_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as mf:
        json.dump(manifest, mf, indent=2)

    total_speech_sec = sum(c["duration"] for c in chunks)
    print(f"[+] Successfully generated {len(chunks)} chunks ({total_speech_sec:.1f}s speech) in {output_corpus_dir}")
    print(f"[+] Manifest saved to {manifest_path}")
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess video & VTT for MFA forced alignment.")
    parser.add_argument("--video", required=True, help="Path to input video file (.mp4, .webm, etc.)")
    parser.add_argument("--vtt", required=True, help="Path to WebVTT subtitles file (.vtt)")
    parser.add_argument("--output", default="./corpus", help="Output corpus directory")
    args = parser.parse_args()

    preprocess_source(args.video, args.vtt, args.output)
