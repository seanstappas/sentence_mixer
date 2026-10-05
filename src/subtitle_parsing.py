"""
Subtitle parsing primitives shared by the movie ingest pipeline and tools/label_subtitles.py.

Parses SRT/WebVTT cues, splits dialogue-dash cues into per-speaker segments, reads
"Name:" labels and bracket tags, and resolves raw speaker names to stable character IDs
(case, accent, OCR-digit and small-typo tolerant).
"""

from __future__ import annotations

import collections
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

TEXT_SUBTITLE_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
TIMING_RE = re.compile(
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})\s*-->\s*"
    r"(\d{1,2}:\d{2}:\d{2}[.,]\d{1,3}|\d{1,2}:\d{2}[.,]\d{1,3})"
)
SUBTITLE_LABEL_RE = re.compile(r"^([A-Za-z][A-Za-z0-9 .'\-]{0,30}?)\s*:\s+(?=\S)")
SCRIPT_LABEL_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 .'&\-]{0,40}?)\s*(\([^)]*\))?\s*:\s*(.*)$")
SCENE_HEADING_RE = re.compile(r"^\s*\d*\s*(INT\.|EXT\.|INT/EXT|I/E\.)", re.IGNORECASE)
BRACKET_TAG_RE = re.compile(r"\[([^\]]*)\]|\(([^)]*)\)")
OCR_DIGIT_MAP = str.maketrans({"0": "o", "1": "l", "5": "s"})


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class Cue:
    index: int
    start: float
    end: float
    lines: List[str]


@dataclass
class Segment:
    cue_index: int
    segment_index: int
    start: float
    end: float
    text: str
    raw_label: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    sound_only: bool = False
    non_english: bool = False
    tokens: List[str] = field(default_factory=list)
    speaker: Optional[str] = None
    source: str = "none"
    confidence: float = 0.0
    evidence: Dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Character names
# ---------------------------------------------------------------------------

def _name_key(name: str) -> str:
    folded = unicodedata.normalize("NFKD", name)
    folded = "".join(ch for ch in folded if not unicodedata.combining(ch))
    folded = folded.replace("\ufffd", "e")  # cp1252-decoded accented e that failed to map
    return re.sub(r"[^a-z0-9]", "", folded.lower())


def _edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


class CharacterResolver:
    """Maps raw speaker names (with case, OCR, and spelling variants) to stable IDs."""

    def __init__(self, config: dict):
        self.display: Dict[str, str] = {}
        self.alias_to_id: Dict[str, str] = {}
        for speaker_id, info in config.get("speakers", {}).items():
            self.display[speaker_id] = info.get("display", speaker_id)
            for alias in [speaker_id, *info.get("aliases", [])]:
                self.alias_to_id[_name_key(alias)] = speaker_id
        for role_id, aliases in config.get("roles", {}).items():
            self.display[role_id] = role_id
            for alias in aliases:
                self.alias_to_id[_name_key(alias)] = role_id
        self.groups = {_name_key(g) for g in config.get("groups", [])}
        self.ignore = [re.compile(p, re.IGNORECASE) for p in config.get("ignore_patterns", [])]
        self.unknown: collections.Counter = collections.Counter()

    def resolve(self, raw: Optional[str]) -> Tuple[Optional[str], str]:
        """Returns (speaker_id or None, kind) where kind is speaker, role, group, ignored, or unknown."""
        if not raw:
            return None, "unknown"
        cleaned = raw.strip().strip(".")
        if any(p.search(cleaned) for p in self.ignore):
            return None, "ignored"
        key = _name_key(cleaned)
        if not key:
            return None, "ignored"
        if key in self.groups:
            return None, "group"
        for candidate in (key, key.translate(OCR_DIGIT_MAP)):
            if candidate in self.alias_to_id:
                return self._kind(self.alias_to_id[candidate])
        best_id, best_dist, tie = None, 99, False
        probe = key.translate(OCR_DIGIT_MAP)
        for alias_key, speaker_id in self.alias_to_id.items():
            dist = _edit_distance(probe, alias_key)
            if dist < best_dist:
                best_id, best_dist, tie = speaker_id, dist, False
            elif dist == best_dist and speaker_id != best_id:
                tie = True
        limit = 1 if len(probe) < 6 else 2
        if best_id is not None and best_dist <= limit and not tie:
            return self._kind(best_id)
        self.unknown[cleaned] += 1
        return None, "unknown"

    @staticmethod
    def _kind(speaker_id: str) -> Tuple[str, str]:
        return speaker_id, ("role" if speaker_id.startswith("role-") else "speaker")


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------

def _to_seconds(stamp: str) -> float:
    stamp = stamp.replace(",", ".")
    parts = stamp.split(":")
    if len(parts) == 2:
        parts = ["0", *parts]
    hours, minutes, seconds = parts
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_subtitle_text(raw: str) -> Tuple[List[Cue], int]:
    """Parses SRT or WebVTT text into cues. Returns (cues, skipped_block_count)."""
    raw = raw.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    cues: List[Cue] = []
    skipped = 0
    for block in re.split(r"\n\s*\n", raw):
        lines = block.split("\n")
        timing_at = next((i for i, ln in enumerate(lines) if TIMING_RE.search(ln)), None)
        if timing_at is None:
            continue
        match = TIMING_RE.search(lines[timing_at])
        assert match is not None
        text_lines = [ln for ln in lines[timing_at + 1:] if ln.strip()]
        if not text_lines:
            skipped += 1
            continue
        cues.append(Cue(len(cues), _to_seconds(match.group(1)), _to_seconds(match.group(2)), text_lines))
    return cues, skipped


def _normalize_tokens(text: str) -> List[str]:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch)).lower()
    text = text.replace("\u2019", "'").replace("'", "")
    return re.findall(r"[a-z0-9]+", text)


def _is_non_english_tag(tag: str) -> bool:
    upper = tag.upper()
    if "ENGLISH" in upper:
        return False
    return bool(re.search(r"\bSPEAKING\b", upper) or re.match(r"^IN [A-Z ]+$", upper.strip()))


def split_segments(cue: Cue) -> List[Segment]:
    """Splits a cue into speaker segments (dialogue dashes) and extracts labels and tags."""
    lines = [re.sub(r"</?[^>]+>", "", ln).strip() for ln in cue.lines]
    lines = [ln for ln in lines if ln]
    groups: List[List[str]] = []
    has_dashes = len(lines) > 1 and sum(1 for ln in lines if re.match(r"^-\s*\S", ln)) >= 1
    for ln in lines:
        if has_dashes and re.match(r"^-\s*\S", ln):
            groups.append([re.sub(r"^-\s*", "", ln)])
        elif groups:
            groups[-1].append(ln)
        else:
            groups.append([re.sub(r"^-\s*", "", ln)])

    total_chars = sum(len(" ".join(g)) for g in groups) or 1
    segments: List[Segment] = []
    cursor = cue.start
    for seg_index, group in enumerate(groups):
        text = " ".join(group).strip()
        raw_label = None
        label_match = SUBTITLE_LABEL_RE.match(text)
        if label_match and len(label_match.group(1).split()) <= 4:
            raw_label = label_match.group(1).strip()
            text = text[label_match.end():].strip()
        tags = [(a or b).strip() for a, b in BRACKET_TAG_RE.findall(text)]
        spoken = BRACKET_TAG_RE.sub(" ", text)
        tokens = _normalize_tokens(spoken)
        share = len(" ".join(group)) / total_chars
        seg_end = cue.end if seg_index == len(groups) - 1 else cursor + (cue.end - cue.start) * share
        segments.append(Segment(
            cue_index=cue.index,
            segment_index=seg_index,
            start=round(cursor, 3),
            end=round(seg_end, 3),
            text=text,
            raw_label=raw_label,
            tags=tags,
            sound_only=not tokens,
            non_english=any(_is_non_english_tag(t) for t in tags),
            tokens=tokens,
        ))
        cursor = seg_end
    return segments
