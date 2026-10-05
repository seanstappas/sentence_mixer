"""
Movie subtitle loading for the ingest pipeline.

Reads a subtitle file into alignment cues that keep per-segment speaker labels:

- Labeled WebVTT (from tools/label_subtitles.py): each cue line is one segment; a
  standard voice tag <v speaker-id>...</v> carries the speaker. Untagged lines are
  segments with no speaker.
- Plain SRT/VTT (for example an embedded SDH track): dialogue-dash cues are split into
  segments, "Name:" labels are resolved through the character list, and sound-only or
  non-English cues are dropped.

Each returned cue becomes one MFA chunk. Word speakers are later assigned from segment
token order (not from estimated times), so two-speaker dash cues stay exact.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from subtitle_parsing import (
    BRACKET_TAG_RE,
    CharacterResolver,
    _normalize_tokens,
    parse_subtitle_text,
    split_segments,
)

VOICE_TAG_RE = re.compile(r"<v(?:\.[\w.-]+)?\s+([^>]+)>(.*?)(?:</v>|$)", re.IGNORECASE)
MARKUP_RE = re.compile(r"</?[^>]+>")

DEFAULT_PAD_BEFORE = 0.15
DEFAULT_PAD_AFTER = 0.25


def load_character_resolver(path: Optional[Union[str, Path]]) -> Optional[CharacterResolver]:
    if not path:
        return None
    p = Path(path)
    if not p.exists():
        return None
    return CharacterResolver(json.loads(p.read_text(encoding="utf-8")))


def _spoken_text(text: str) -> str:
    text = MARKUP_RE.sub("", text)
    text = BRACKET_TAG_RE.sub(" ", text)
    text = text.replace("\u266a", " ")
    return re.sub(r"\s+", " ", text).strip()


def _segment(text: str, speaker: Optional[str], source: str) -> Optional[Dict[str, Any]]:
    spoken = _spoken_text(text)
    tokens = _normalize_tokens(spoken)
    if not tokens:
        return None
    return {"speaker_id": speaker, "text": spoken, "tokens": tokens, "label_source": source}


def is_labeled_vtt(raw: str) -> bool:
    return bool(re.search(r"<v[\s.][^>]*>", raw, re.IGNORECASE))


def load_movie_cues(
    subtitle_path: Union[str, Path],
    resolver: Optional[CharacterResolver] = None,
    pad_before: float = DEFAULT_PAD_BEFORE,
    pad_after: float = DEFAULT_PAD_AFTER,
) -> Dict[str, Any]:
    """
    Returns {"cues": [...], "stats": {...}}. Each cue has start_sec/end_sec (padded chunk
    bounds), cue_start_sec/cue_end_sec (original), text, and segments
    [{speaker_id, text, tokens, label_source}].
    """
    p = Path(subtitle_path)
    raw = p.read_text(encoding="utf-8", errors="replace")
    labeled = is_labeled_vtt(raw)
    cues, skipped_empty = parse_subtitle_text(raw)

    stats: Dict[str, Any] = {
        "format": "labeled_vtt" if labeled else "plain",
        "cue_blocks": len(cues),
        "skipped_empty_blocks": skipped_empty,
        "zero_duration": 0,
        "sound_only": 0,
        "non_english": 0,
        "segments": 0,
        "labeled_segments": 0,
    }
    out: List[Dict[str, Any]] = []
    for cue in cues:
        if cue.end <= cue.start:
            stats["zero_duration"] += 1
            continue
        segments: List[Dict[str, Any]] = []
        drop = False
        if labeled:
            for line in cue.lines:
                m = VOICE_TAG_RE.search(line)
                if m:
                    seg = _segment(m.group(2), m.group(1).strip(), "subtitle_tag")
                else:
                    seg = _segment(line, None, "none")
                if seg:
                    segments.append(seg)
        else:
            for s in split_segments(cue):
                if s.non_english:
                    drop = True
                    break
                if s.sound_only:
                    continue
                speaker = None
                source = "none"
                if s.raw_label and resolver is not None:
                    speaker, kind = resolver.resolve(s.raw_label)
                    source = "subtitle_label" if speaker else "none"
                seg = _segment(s.text, speaker, source)
                if seg:
                    segments.append(seg)
        if drop:
            stats["non_english"] += 1
            continue
        if not segments:
            stats["sound_only"] += 1
            continue
        stats["segments"] += len(segments)
        stats["labeled_segments"] += sum(1 for s in segments if s["speaker_id"])
        out.append({
            "cue_start_sec": cue.start,
            "cue_end_sec": cue.end,
            "text": " ".join(s["text"] for s in segments),
            "segments": segments,
        })

    out.sort(key=lambda c: c["cue_start_sec"])
    for i, c in enumerate(out):
        prev_end = out[i - 1]["cue_end_sec"] if i > 0 else 0.0
        next_start = out[i + 1]["cue_start_sec"] if i + 1 < len(out) else c["cue_end_sec"] + pad_after
        start = max(prev_end, c["cue_start_sec"] - pad_before, 0.0)
        end = min(max(next_start, c["cue_end_sec"]), c["cue_end_sec"] + pad_after)
        start = min(start, c["cue_start_sec"])
        c["start_sec"] = round(start, 3)
        c["end_sec"] = round(end, 3)
        c["duration"] = round(end - start, 3)
    stats["cues"] = len(out)
    return {"cues": out, "stats": stats}


def assign_word_speakers(
    mfa_words: List[str],
    segments: List[Dict[str, Any]],
) -> List[Optional[Dict[str, Any]]]:
    """
    Maps each MFA word (in order) to the segment it came from by aligning normalized
    tokens. Returns, per word, the segment dict or None when the word cannot be placed
    unambiguously (unmatched words between two different speakers' segments).
    """
    import difflib

    seg_tokens: List[str] = []
    token_seg: List[int] = []
    for si, seg in enumerate(segments):
        for t in seg.get("tokens") or _normalize_tokens(seg.get("text", "")):
            seg_tokens.append(t)
            token_seg.append(si)

    word_tokens: List[str] = []
    token_word: List[int] = []
    for wi, w in enumerate(mfa_words):
        for t in _normalize_tokens(w):
            word_tokens.append(t)
            token_word.append(wi)

    word_seg: List[Optional[int]] = [None] * len(mfa_words)
    votes: List[Dict[int, int]] = [dict() for _ in mfa_words]
    sm = difflib.SequenceMatcher(a=word_tokens, b=seg_tokens, autojunk=False)
    for block in sm.get_matching_blocks():
        for k in range(block.size):
            wi = token_word[block.a + k]
            si = token_seg[block.b + k]
            votes[wi][si] = votes[wi].get(si, 0) + 1
    for wi, v in enumerate(votes):
        if len(v) == 1:
            word_seg[wi] = next(iter(v))

    # Fill unmatched words that sit between words of the same segment
    if len(segments) == 1:
        word_seg = [0 if _normalize_tokens(w) else None for w in mfa_words]
    else:
        for wi in range(len(mfa_words)):
            if word_seg[wi] is not None or votes[wi]:
                continue
            left = next((word_seg[j] for j in range(wi - 1, -1, -1) if word_seg[j] is not None), None)
            right = next((word_seg[j] for j in range(wi + 1, len(mfa_words)) if word_seg[j] is not None), None)
            if left is not None and left == right:
                word_seg[wi] = left
            elif left is None and right == 0:
                word_seg[wi] = 0
            elif right is None and left == len(segments) - 1:
                word_seg[wi] = left
    return [segments[s] if s is not None else None for s in word_seg]
