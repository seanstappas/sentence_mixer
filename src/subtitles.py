"""
Subtitle Generation, Formatting, and Video Burning Service.
Generates standard SubRip (.srt) and WebVTT (.vtt) subtitle files from EditPlan
timelines, with optional FFmpeg hardsub burning.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from contracts import EditPlan, PlanSegment


def format_timestamp_srt(seconds: float) -> str:
    """Format seconds into SubRip timestamp: HH:MM:SS,mmm"""
    total_ms = int(round(max(0.0, seconds) * 1000.0))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def format_timestamp_vtt(seconds: float) -> str:
    """Format seconds into WebVTT timestamp: HH:MM:SS.mmm"""
    total_ms = int(round(max(0.0, seconds) * 1000.0))
    hours = total_ms // 3600000
    minutes = (total_ms % 3600000) // 60000
    secs = (total_ms % 60000) // 1000
    millis = total_ms % 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"


def clean_segment_label(text: str) -> str:
    """
    Cleans segment target_text annotations into readable display text:
    - 'investigation[IH0-N-V-EH1-S-T]' -> 'invest'
    - '[pause]' -> ''
    - '<gap:...>' -> ''
    - '{...}' -> ''
    """
    if not text or text.startswith("[pause]") or text.startswith("<gap:") or text.startswith("{"):
        return ""
    # Strip subword phone brackets e.g. 'investigation[IH0-N-V-EH1-S-T]'
    m = re.match(r"^([^\[]+)\[", text)
    if m:
        donor = m.group(1).strip()
        # If donor word has known sub-root (e.g. investigation -> invest, intoxicating -> talk/tok)
        return donor
    return text.strip()


class SubtitleCue:
    def __init__(self, index: int, start_sec: float, end_sec: float, text: str):
        self.index = index
        self.start_sec = max(0.0, start_sec)
        self.end_sec = max(self.start_sec, end_sec)
        self.text = text.strip()


def extract_cues_from_plan(
    plan: Union[EditPlan, List[Dict[str, Any]]],
    cue_mode: str = "phrase",
) -> List[SubtitleCue]:
    """
    Extracts subtitle cues from an EditPlan or legacy unit list.
    - 'phrase': groups speech segments between pauses/gaps into natural clauses.
    - 'word': creates an individual cue for each word/segment.
    """
    if isinstance(plan, list):
        cues_legacy: List[SubtitleCue] = []
        cue_idx = 1
        curr_time = 0.0
        if cue_mode == "word":
            for u in plan:
                dur = max(0.033, float(u.get("end_sec", 0.0)) - float(u.get("start_sec", 0.0)))
                lbl = clean_segment_label(u.get("label", ""))
                if lbl and not u.get("is_silence", False):
                    cues_legacy.append(SubtitleCue(cue_idx, curr_time, curr_time + dur, lbl))
                    cue_idx += 1
                curr_time += dur
            return cues_legacy

        words: List[str] = []
        phrase_start = 0.0
        phrase_end = 0.0
        for u in plan:
            dur = max(0.033, float(u.get("end_sec", 0.0)) - float(u.get("start_sec", 0.0)))
            lbl = clean_segment_label(u.get("label", ""))
            is_sil = u.get("is_silence", False) or not lbl
            if is_sil:
                if words:
                    cues_legacy.append(SubtitleCue(cue_idx, phrase_start, phrase_end, " ".join(words)))
                    cue_idx += 1
                    words = []
            else:
                if not words:
                    phrase_start = curr_time
                words.append(lbl)
                phrase_end = curr_time + dur
            curr_time += dur
        if words:
            cues_legacy.append(SubtitleCue(cue_idx, phrase_start, phrase_end, " ".join(words)))
        return cues_legacy

    sr = float(plan.project_sample_rate)
    cues: List[SubtitleCue] = []
    cue_idx = 1

    if cue_mode == "word":
        for seg in plan.segments:
            if seg.is_silence:
                continue
            txt = clean_segment_label(seg.target_text)
            if not txt:
                continue
            start_s = seg.output_start_sample / sr
            end_s = seg.output_end_sample / sr
            if end_s > start_s:
                cues.append(SubtitleCue(cue_idx, start_s, end_s, txt))
                cue_idx += 1
        return cues

    # Default 'phrase' mode: group contiguous non-silence segments
    curr_group: List[PlanSegment] = []

    def _flush_group():
        nonlocal cue_idx, curr_group
        if not curr_group:
            return
        start_s = curr_group[0].output_start_sample / sr
        end_s = curr_group[-1].output_end_sample / sr
        words = []
        for s in curr_group:
            txt = clean_segment_label(s.target_text)
            if txt and (not words or words[-1] != txt):
                words.append(txt)
        cue_text = " ".join(words)
        if cue_text and end_s > start_s:
            cues.append(SubtitleCue(cue_idx, start_s, end_s, cue_text))
            cue_idx += 1
        curr_group = []

    for seg in plan.segments:
        # Continued pauses (clip playing on after a word) end a phrase like silence does
        if seg.is_silence or seg.span_type == "continue":
            _flush_group()
        else:
            curr_group.append(seg)
    _flush_group()

    return cues


def generate_subtitles(
    plan_or_units: Union[EditPlan, List[Dict[str, Any]]],
    format: str = "srt",
    cue_mode: str = "phrase",
) -> str:
    """Generates SRT or VTT subtitle string from an EditPlan or legacy unit list."""
    fmt = format.lower().strip()
    cues = extract_cues_from_plan(plan_or_units, cue_mode=cue_mode)

    if fmt == "vtt":
        lines = ["WEBVTT", ""]
        for cue in cues:
            t_start = format_timestamp_vtt(cue.start_sec)
            t_end = format_timestamp_vtt(cue.end_sec)
            lines.append(f"{t_start} --> {t_end}")
            lines.append(cue.text)
            lines.append("")
        return "\n".join(lines).rstrip() + "\n"

    # Default SRT format
    lines = []
    for cue in cues:
        t_start = format_timestamp_srt(cue.start_sec)
        t_end = format_timestamp_srt(cue.end_sec)
        lines.append(str(cue.index))
        lines.append(f"{t_start} --> {t_end}")
        lines.append(cue.text)
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_subtitles(
    plan_or_units: Union[EditPlan, List[Dict[str, Any]]],
    output_path: Union[str, Path],
    format: Optional[str] = None,
    cue_mode: str = "phrase",
) -> Path:
    """Writes subtitle file to disk matching the given EditPlan or legacy unit list."""
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if not format:
        format = "vtt" if out_p.suffix.lower() == ".vtt" else "srt"

    content = generate_subtitles(plan_or_units, format=format, cue_mode=cue_mode)
    out_p.write_text(content, encoding="utf-8")
    return out_p


def burn_subtitles_to_video(
    input_video_path: Union[str, Path],
    subtitle_path: Union[str, Path],
    output_video_path: Union[str, Path],
    style: str = "classic",
) -> Path:
    """
    Burns subtitles into video using FFmpeg.
    Style presets:
    - 'classic': Yellow font, bold black outline, centered at bottom (classic YouTube Poop style).
    - 'modern': Crisp white font, clean subtle outline.
    """
    in_vid = Path(input_video_path).resolve()
    sub_p = Path(subtitle_path).resolve()
    out_vid = Path(output_video_path).resolve()
    out_vid.parent.mkdir(parents=True, exist_ok=True)

    if not in_vid.exists():
        raise FileNotFoundError(f"Input video not found: {in_vid}")
    if not sub_p.exists():
        raise FileNotFoundError(f"Subtitle file not found: {sub_p}")

    # FFmpeg subtitles filter requires escaping backslashes and colons on Windows
    sub_escaped = str(sub_p).replace("\\", "/").replace(":", "\\:")

    if style == "classic":
        # Yellow (&H0000FFFF in BGR), black border, bold
        force_style = "Fontname=Arial,Fontsize=22,PrimaryColour=&H0000FFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=2.5,Shadow=1,MarginV=25,Bold=1"
    else:
        # Modern: White, subtle dark border
        force_style = "Fontname=Arial,Fontsize=20,PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,BorderStyle=1,Outline=1.8,Shadow=0,MarginV=25"

    vf = f"subtitles='{sub_escaped}':force_style='{force_style}'"

    cmd = [
        "ffmpeg", "-y", "-nostats", "-loglevel", "error",
        "-i", str(in_vid),
        "-vf", vf,
        "-c:v", "libx264", "-crf", "18", "-preset", "fast",
        "-c:a", "copy",
        str(out_vid),
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed to burn subtitles: {proc.stderr}")

    return out_vid
