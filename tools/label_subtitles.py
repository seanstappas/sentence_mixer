#!/usr/bin/env python3
"""
Generate a speaker-labeled WebVTT subtitle file for a movie.

Inputs:
  - The movie's own subtitles: an embedded text track (SDH preferred) or a subtitle file.
  - Local transcripts ("Name: line") and screenplays (classic bold-cue HTML or inline
    "NAME: line"), saved by the user.
  - A character alias list (config/characters/*.json).

Output:
  - WebVTT with the subtitle's own timing and text, and a standard voice tag
    (<v speaker-id>...</v>) on each line whose speaker could be determined.
    Unlabeled lines have no tag. Sound-only and non-English lines are omitted.
  - A JSON report with per-line evidence, confidence, and coverage statistics.

Labels come from, in priority order: speaker labels inside the subtitles, transcript
alignment, then screenplay alignment (fallback only). Conflicting evidence leaves a
line unlabeled. Nothing is guessed from voice; that is a later ingest step.

Both outputs contain or describe copyrighted subtitle text. Write them only to
git-ignored locations and never commit them. This tool prints counts and names only.
"""

from __future__ import annotations

import argparse
import bisect
import collections
import datetime
import difflib
import hashlib
import html
import json
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHARACTERS = REPO_ROOT / "config" / "characters" / "star_wars.json"

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


@dataclass
class Speech:
    speaker_raw: Optional[str]
    text: str
    scene_index: Optional[int] = None


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


def probe_subtitle_streams(video: Path) -> List[dict]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "s",
         "-show_entries", "stream=index,codec_name:stream_tags=language,title:stream_disposition=hearing_impaired,forced",
         "-of", "json", str(video)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout).get("streams", [])


def choose_subtitle_stream(streams: Sequence[dict], preference: str = "auto") -> Tuple[int, dict]:
    """Returns (position among subtitle streams, stream info). Prefers English SDH text tracks."""
    text_streams = [(pos, s) for pos, s in enumerate(streams) if s.get("codec_name") in TEXT_SUBTITLE_CODECS]
    if not text_streams:
        raise ValueError("No text subtitle tracks found (image subtitles such as PGS need OCR and are unsupported).")
    if preference not in ("auto", "sdh", "plain"):
        wanted = int(preference)
        for pos, s in text_streams:
            if s.get("index") == wanted:
                return pos, s
        raise ValueError(f"Subtitle stream index {wanted} is not a text subtitle track.")

    def is_english(s: dict) -> bool:
        return (s.get("tags", {}).get("language") or "eng").lower() in ("eng", "en", "english")

    def is_sdh(s: dict) -> bool:
        title = (s.get("tags", {}).get("title") or "").upper()
        return "SDH" in title or bool(s.get("disposition", {}).get("hearing_impaired"))

    def is_forced(s: dict) -> bool:
        return bool(s.get("disposition", {}).get("forced")) or "FORCED" in (s.get("tags", {}).get("title") or "").upper()

    english = [(p, s) for p, s in text_streams if is_english(s) and not is_forced(s)] or text_streams
    if preference in ("auto", "sdh"):
        sdh = [(p, s) for p, s in english if is_sdh(s)]
        if sdh:
            return sdh[0]
    plain = [(p, s) for p, s in english if not is_sdh(s)]
    return (plain or english)[0]


def extract_subtitle_stream(video: Path, position: int, out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(video), "-map", f"0:s:{position}", "-f", "srt", str(out_path)],
        check=True,
    )
    return out_path


# ---------------------------------------------------------------------------
# Transcripts and screenplays
# ---------------------------------------------------------------------------

def _strip_directions(text: str) -> str:
    return re.sub(r"\s+", " ", BRACKET_TAG_RE.sub(" ", text)).strip()


def parse_inline_script(raw: str) -> Tuple[List[Speech], List[str]]:
    """Parses "Name: line" transcripts and inline screenplays into speeches and scene headings."""
    speeches: List[Speech] = []
    headings: List[str] = []
    for paragraph in re.split(r"\n\s*\n", raw.replace("\r\n", "\n")):
        lines = [ln for ln in paragraph.split("\n") if ln.strip()]
        if not lines:
            continue
        first = lines[0].strip()
        if SCENE_HEADING_RE.match(first):
            headings.append(first)
            lines = lines[1:]
            if not lines:
                continue
        current: Optional[Speech] = None
        for ln in lines:
            match = SCRIPT_LABEL_RE.match(ln)
            stripped = ln.strip()
            if match and len(match.group(1).split()) <= 5:
                current = Speech(match.group(1).strip(), _strip_directions(match.group(3)), len(headings) - 1 if headings else None)
                speeches.append(current)
            elif re.match(r"^[\[(].*[\])]$", stripped):
                current = None
            elif current is not None:
                current.text = (current.text + " " + _strip_directions(stripped)).strip()
            else:
                speeches.append(Speech(None, _strip_directions(stripped), len(headings) - 1 if headings else None))
    return [s for s in speeches if s.text], headings


def parse_classic_screenplay(raw_html: str) -> Tuple[List[Speech], List[str]]:
    """Parses a screenplay whose character cues are bold lines above indented dialogue."""
    marked = re.sub(r"<b>", "\x01", raw_html, flags=re.IGNORECASE)
    marked = re.sub(r"</b>", "\x02", marked, flags=re.IGNORECASE)
    marked = html.unescape(re.sub(r"<[^>]+>", "", marked))
    speeches: List[Speech] = []
    headings: List[str] = []
    parts = re.split(r"\x01(.*?)\x02", marked, flags=re.DOTALL)
    # parts alternates: text, bold, text, bold, ...
    for i in range(1, len(parts), 2):
        bold = parts[i].strip()
        after = parts[i + 1] if i + 1 < len(parts) else ""
        if not bold:
            continue
        if SCENE_HEADING_RE.match(bold):
            headings.append(bold)
            continue
        name = re.sub(r"\((CONT'D|CONTD|CONTINUED|V\.O\.|O\.S\.|O\.C\.|OS|VO)\)", "", bold, flags=re.IGNORECASE).strip()
        if not (name.isupper() and len(name) <= 30 and len(name.split()) <= 4):
            continue
        dialogue = re.split(r"\n\s*\n", after.lstrip("\n"), maxsplit=1)[0]
        text = _strip_directions(" ".join(ln.strip() for ln in dialogue.split("\n")))
        if text:
            speeches.append(Speech(name, text, len(headings) - 1 if headings else None))
    return speeches, headings


def load_script(path: Path) -> Tuple[List[Speech], List[str], str]:
    raw_bytes = path.read_bytes()
    try:
        raw = raw_bytes.decode("utf-8")
    except UnicodeDecodeError:
        raw = raw_bytes.decode("cp1252", errors="replace")
    if path.suffix.lower() in (".html", ".htm"):
        if len(re.findall(r"<b>\s*[A-Z][A-Z0-9 .'\-]{1,30}\s*</b>", raw)) > 50:
            speeches, headings = parse_classic_screenplay(raw)
            return speeches, headings, "screenplay-classic"
        raw = html.unescape(re.sub(r"<[^>]+>", "", raw))
    speeches, headings = parse_inline_script(raw)
    return speeches, headings, "inline"


# ---------------------------------------------------------------------------
# Alignment
# ---------------------------------------------------------------------------

def _anchor_pairs(a: Sequence[str], b: Sequence[str], n: int = 4) -> List[Tuple[int, int]]:
    def grams(tokens: Sequence[str]) -> Dict[Tuple[str, ...], List[int]]:
        out: Dict[Tuple[str, ...], List[int]] = collections.defaultdict(list)
        for i in range(len(tokens) - n + 1):
            out[tuple(tokens[i:i + n])].append(i)
        return out

    ga, gb = grams(a), grams(b)
    pairs = sorted((pa[0], gb[g][0]) for g, pa in ga.items() if len(pa) == 1 and len(gb.get(g, [])) == 1)
    # Longest chain increasing in both sequences (patience sorting on b).
    tails: List[int] = []
    tail_idx: List[int] = []
    prev: List[int] = [-1] * len(pairs)
    for k, (_, pb) in enumerate(pairs):
        pos = bisect.bisect_left(tails, pb)
        if pos == len(tails):
            tails.append(pb)
            tail_idx.append(k)
        else:
            tails[pos] = pb
            tail_idx[pos] = k
        prev[k] = tail_idx[pos - 1] if pos > 0 else -1
    chain: List[Tuple[int, int]] = []
    k = tail_idx[-1] if tail_idx else -1
    while k >= 0:
        chain.append(pairs[k])
        k = prev[k]
    return list(reversed(chain))


def align_tokens(a: Sequence[str], b: Sequence[str], max_gap: int = 3000) -> Dict[int, int]:
    """Monotonic word alignment from a to b: unique 4-gram anchors, then local matching between anchors."""
    mapping: Dict[int, int] = {}
    last_a, last_b = -1, -1

    def add(i: int, j: int) -> None:
        nonlocal last_a, last_b
        if i > last_a and j > last_b and a[i] == b[j]:
            mapping[i] = j
            last_a, last_b = i, j

    def fill(a0: int, a1: int, b0: int, b1: int) -> None:
        if a1 <= a0 or b1 <= b0 or (a1 - a0) > max_gap or (b1 - b0) > max_gap:
            return
        matcher = difflib.SequenceMatcher(None, a[a0:a1], b[b0:b1], autojunk=False)
        for block in matcher.get_matching_blocks():
            for k in range(block.size):
                add(a0 + block.a + k, b0 + block.b + k)

    anchors = _anchor_pairs(a, b)
    for ia, ib in anchors:
        if ia <= last_a or ib <= last_b:
            continue
        fill(last_a + 1, ia, last_b + 1, ib)
        for k in range(4):
            if ia + k < len(a) and ib + k < len(b):
                add(ia + k, ib + k)
    fill(last_a + 1, len(a), last_b + 1, len(b))
    return mapping


def vote_speakers(
    segments: Sequence[Segment],
    speeches: Sequence[Speech],
    resolver: CharacterResolver,
) -> Dict[Tuple[int, int], dict]:
    """Aligns segment words to script words and returns per-segment speaker votes."""
    a_tokens: List[str] = []
    a_owner: List[int] = []
    for idx, seg in enumerate(segments):
        for tok in seg.tokens:
            a_tokens.append(tok)
            a_owner.append(idx)
    b_tokens: List[str] = []
    b_speaker: List[Optional[str]] = []
    b_kind: List[str] = []
    for speech in speeches:
        speaker_id, kind = resolver.resolve(speech.speaker_raw) if speech.speaker_raw else (None, "unlabeled")
        for tok in _normalize_tokens(speech.text):
            b_tokens.append(tok)
            b_speaker.append(speaker_id)
            b_kind.append(kind)

    mapping = align_tokens(a_tokens, b_tokens)
    per_segment: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    matched: Dict[int, int] = collections.Counter()
    for ia, ib in mapping.items():
        owner = a_owner[ia]
        matched[owner] += 1
        if b_speaker[ib] is not None:
            per_segment[owner][b_speaker[ib]] += 1
        else:
            per_segment[owner]["<" + b_kind[ib] + ">"] += 1

    results: Dict[Tuple[int, int], dict] = {}
    for idx, seg in enumerate(segments):
        votes = per_segment.get(idx, collections.Counter())
        named = {k: v for k, v in votes.items() if not k.startswith("<")}
        n_tokens = len(seg.tokens)
        best, best_votes = (max(named.items(), key=lambda kv: kv[1]) if named else (None, 0))
        labeled_total = sum(named.values())
        share_of_labeled = best_votes / labeled_total if labeled_total else 0.0
        coverage = best_votes / n_tokens if n_tokens else 0.0
        accept = False
        if best is not None and share_of_labeled >= 0.8:
            if n_tokens <= 2:
                accept = best_votes == n_tokens
            else:
                accept = best_votes >= 2 and coverage >= 0.5
        results[(seg.cue_index, seg.segment_index)] = {
            "speaker": best if accept else None,
            "votes": dict(votes),
            "matched_words": matched.get(idx, 0),
            "coverage": round(coverage, 3),
        }
    return results


# ---------------------------------------------------------------------------
# Labeling and output
# ---------------------------------------------------------------------------

def label_segments(
    segments: List[Segment],
    resolver: CharacterResolver,
    transcripts: Sequence[Sequence[Speech]],
    screenplays: Sequence[Sequence[Speech]],
) -> Dict[str, object]:
    dialogue = [s for s in segments if not s.sound_only and not s.non_english]
    transcript_votes = [vote_speakers(dialogue, speeches, resolver) for speeches in transcripts]
    screenplay_votes = [vote_speakers(dialogue, speeches, resolver) for speeches in screenplays]

    stats: collections.Counter = collections.Counter()
    for seg in dialogue:
        key = (seg.cue_index, seg.segment_index)
        label_id, label_kind = resolver.resolve(seg.raw_label) if seg.raw_label else (None, "none")
        t_ids = {v[key]["speaker"] for v in transcript_votes if v[key]["speaker"]}
        s_ids = {v[key]["speaker"] for v in screenplay_votes if v[key]["speaker"]}
        seg.evidence = {
            "subtitle_label": label_id,
            "subtitle_label_kind": label_kind if seg.raw_label else None,
            "transcripts": [v[key] for v in transcript_votes],
            "screenplays": [v[key] for v in screenplay_votes],
        }
        if seg.raw_label and label_kind == "group":
            seg.source, seg.speaker = "group", None
            stats["group"] += 1
            continue
        if label_id:
            if t_ids and label_id not in t_ids:
                named_t = {t for t in t_ids if not t.startswith("role-")}
                if label_kind == "role" and len(named_t) == 1:
                    # Subtitle has only a generic role ("Man"); the transcript names the character.
                    seg.speaker, seg.source, seg.confidence = next(iter(named_t)), "transcript_over_role", 0.8
                    stats["role_resolved_by_transcript"] += 1
                    continue
                if label_kind == "speaker" and not named_t:
                    # Transcript has only a generic role; the official subtitle names the character.
                    seg.speaker, seg.source, seg.confidence = label_id, "subtitle_over_role", 0.9
                    stats["role_resolved_by_subtitle"] += 1
                    continue
                seg.source = "conflict"
                stats["conflict_subtitle_vs_transcript"] += 1
                continue
            seg.speaker, seg.source, seg.confidence = label_id, "subtitle_label", 1.0
            stats["agree_subtitle_transcript" if label_id in t_ids else "subtitle_label_only"] += 1
            continue
        if len(t_ids) > 1:
            seg.source = "conflict"
            stats["conflict_between_transcripts"] += 1
            continue
        if t_ids:
            seg.speaker = next(iter(t_ids))
            seg.source = "transcript"
            seg.confidence = max(v[key]["coverage"] for v in transcript_votes if v[key]["speaker"])
            stats["transcript"] += 1
            continue
        if len(s_ids) == 1:
            seg.speaker = next(iter(s_ids))
            seg.source = "screenplay"
            seg.confidence = 0.5 * max(v[key]["coverage"] for v in screenplay_votes if v[key]["speaker"])
            stats["screenplay"] += 1
            continue
        if len(s_ids) > 1:
            seg.source = "conflict"
            stats["conflict_between_screenplays"] += 1
            continue
        stats["unlabeled"] += 1
    stats["dialogue_lines"] = len(dialogue)
    stats["sound_only_lines"] = sum(1 for s in segments if s.sound_only)
    stats["non_english_lines"] = sum(1 for s in segments if s.non_english and not s.sound_only)
    return dict(stats)


def _vtt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}"


def write_vtt(cues: Sequence[Cue], segments: Sequence[Segment], out_path: Path, note: str) -> int:
    by_cue: Dict[int, List[Segment]] = collections.defaultdict(list)
    for seg in segments:
        by_cue[seg.cue_index].append(seg)
    lines = ["WEBVTT", "", "NOTE", note, ""]
    written = 0
    for cue in cues:
        if cue.end <= cue.start:
            continue
        kept = [s for s in by_cue.get(cue.index, []) if not s.sound_only and not s.non_english and s.text]
        if not kept:
            continue
        written += 1
        lines.append(str(written))
        lines.append(f"{_vtt_time(cue.start)} --> {_vtt_time(cue.end)}")
        for seg in kept:
            text = seg.text.replace("-->", "->")
            lines.append(f"<v {seg.speaker}>{text}</v>" if seg.speaker else text)
        lines.append("")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return written


def build_report(
    segments: Sequence[Segment],
    stats: Dict[str, object],
    resolver: CharacterResolver,
    meta: Dict[str, object],
) -> dict:
    per_speaker: Dict[str, Dict[str, float]] = collections.defaultdict(lambda: {"lines": 0, "seconds": 0.0})
    for seg in segments:
        if seg.speaker:
            per_speaker[seg.speaker]["lines"] += 1
            per_speaker[seg.speaker]["seconds"] = round(per_speaker[seg.speaker]["seconds"] + (seg.end - seg.start), 2)
    return {
        **meta,
        "summary": stats,
        "speakers": dict(sorted(per_speaker.items(), key=lambda kv: -kv[1]["lines"])),
        "unknown_names": dict(resolver.unknown.most_common()),
        "segments": [
            {
                "cue_index": s.cue_index,
                "segment_index": s.segment_index,
                "start": s.start,
                "end": s.end,
                "speaker": s.speaker,
                "source": s.source,
                "confidence": round(s.confidence, 3),
                "sound_only": s.sound_only,
                "non_english": s.non_english,
                "words": len(s.tokens),
                "text_sha1": hashlib.sha1(s.text.encode("utf-8")).hexdigest(),
                "evidence": s.evidence,
            }
            for s in segments
        ],
    }


def run(
    subtitle_path: Path,
    transcript_paths: Iterable[Path],
    screenplay_paths: Iterable[Path],
    characters_path: Path,
    out_vtt: Path,
    out_report: Path,
    meta: Optional[Dict[str, object]] = None,
) -> dict:
    resolver = CharacterResolver(json.loads(characters_path.read_text(encoding="utf-8")))
    raw = subtitle_path.read_bytes().decode("utf-8", errors="replace")
    cues, skipped = parse_subtitle_text(raw)
    segments = [seg for cue in cues for seg in split_segments(cue)]

    transcripts: List[List[Speech]] = []
    screenplays: List[List[Speech]] = []
    script_meta = []
    for path, bucket, role in [(p, transcripts, "transcript") for p in transcript_paths] + [
        (p, screenplays, "screenplay") for p in screenplay_paths
    ]:
        speeches, headings, fmt = load_script(path)
        bucket.append(speeches)
        script_meta.append({
            "role": role,
            "file": path.name,
            "format": fmt,
            "speeches": len(speeches),
            "labeled_speeches": sum(1 for s in speeches if s.speaker_raw),
            "scene_headings": len(headings),
        })

    stats = label_segments(segments, resolver, transcripts, screenplays)
    stats["cues"] = len(cues)
    stats["skipped_empty_blocks"] = skipped
    stats["zero_duration_cues_omitted"] = sum(1 for c in cues if c.end <= c.start)
    stats["cues_with_multiple_segments"] = sum(1 for c in cues if sum(1 for s in segments if s.cue_index == c.index) > 1)
    dialogue = [s for s in segments if not s.sound_only and not s.non_english]
    labeled = sum(1 for s in dialogue if s.speaker)
    stats["labeled_lines"] = labeled
    stats["labeled_share"] = round(labeled / len(dialogue), 3) if dialogue else 0.0

    note = (
        "Speaker-labeled subtitles generated by tools/label_subtitles.py on "
        f"{datetime.date.today().isoformat()}. Labels: subtitle labels, then transcript "
        "alignment, then screenplay alignment; unlabeled lines have no voice tag. "
        "Local use only; do not commit or distribute."
    )
    written = write_vtt(cues, segments, out_vtt, note)
    stats["vtt_cues_written"] = written
    report = build_report(segments, stats, resolver, {
        **(meta or {}),
        "subtitle_file": subtitle_path.name,
        "scripts": script_meta,
        "output_vtt": str(out_vtt),
    })
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path, help="Movie file with embedded text subtitles")
    source.add_argument("--subtitles", type=Path, help="Subtitle file (.srt or .vtt) instead of an embedded track")
    parser.add_argument("--subtitle-stream", default="auto", help="auto (English SDH first), sdh, plain, or an ffprobe stream index")
    parser.add_argument("--transcript", type=Path, action="append", default=[], help="Transcript file (Name: line); repeatable")
    parser.add_argument("--screenplay", type=Path, action="append", default=[], help="Screenplay file (HTML or text); fallback only; repeatable")
    parser.add_argument("--characters", type=Path, default=DEFAULT_CHARACTERS, help="Character alias JSON")
    parser.add_argument("--output", type=Path, required=True, help="Output WebVTT path (keep it in a git-ignored folder)")
    parser.add_argument("--report", type=Path, help="Output JSON report path (default: <output>.report.json)")
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / ".build" / "subtitles" / "source", help="Where extracted subtitle tracks are written")
    args = parser.parse_args(argv)

    meta: Dict[str, object] = {}
    if args.video:
        streams = probe_subtitle_streams(args.video)
        position, info = choose_subtitle_stream(streams, args.subtitle_stream)
        subtitle_path = args.work_dir / f"{args.video.stem}.s{position}.srt"
        extract_subtitle_stream(args.video, position, subtitle_path)
        meta = {
            "video": args.video.name,
            "subtitle_stream": {"position": position, "index": info.get("index"), "codec": info.get("codec_name"), "title": info.get("tags", {}).get("title")},
        }
    else:
        subtitle_path = args.subtitles

    report_path = args.report or args.output.with_suffix(args.output.suffix + ".report.json")
    report = run(subtitle_path, args.transcript, args.screenplay, args.characters, args.output, report_path, meta)

    summary = report["summary"]
    print(f"[+] Wrote {summary['vtt_cues_written']} cues to {args.output}")
    print(f"[+] Report: {report_path}")
    print("[*] Summary: " + ", ".join(f"{k}={v}" for k, v in sorted(summary.items())))
    top = list(report["speakers"].items())[:15]
    print("[*] Top speakers: " + ", ".join(f"{k} ({v['lines']} lines, {v['seconds']:.0f}s)" for k, v in top))
    if report["unknown_names"]:
        print("[!] Unrecognized names (add to the character list): " + ", ".join(f"{k} x{v}" for k, v in report["unknown_names"].items()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
