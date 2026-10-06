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
    Unlabeled lines have no tag. Sound-only, non-English, and zero-duration cues are
    omitted. Write it to .build/subtitles/labeled/<movie file stem>.en.vtt and
    `run_mixer.py ingest` picks it up automatically (no --vtt needed).
  - A JSON report (<output>.report.json) with per-line evidence, confidence, and a text
    hash (no text), per-speaker totals, unrecognized names, and "scenes": screenplay
    scene headings mapped to time ranges (used by `run_mixer.py scenes` and --scene).

Labels come from, in priority order: speaker labels inside the subtitles, transcript
alignment, then screenplay alignment (fallback only). Conflicting evidence leaves a
line unlabeled, except that a named character beats a generic role. Nothing is guessed
from voice here; at ingest, voice checks exclude lines whose voice clearly belongs to
another character and may label unlabeled lines, and `run_mixer.py review-speakers`
records manual corrections. Relabeling does not require re-alignment.

Shared parsing (cues, dialogue dashes, Name: labels, character resolution) lives in
src/subtitle_parsing.py and is also used by the ingest pipeline.

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
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHARACTERS = REPO_ROOT / "config" / "characters" / "star_wars.json"

sys.path.insert(0, str(REPO_ROOT / "src"))

from subtitle_parsing import (  # noqa: E402
    TEXT_SUBTITLE_CODECS,
    BRACKET_TAG_RE,
    Cue,
    Segment,
    CharacterResolver,
    parse_subtitle_text,
    _normalize_tokens,
    split_segments,
)

SCRIPT_LABEL_RE = re.compile(r"^\s*([A-Z][A-Za-z0-9 .'&\-]{0,40}?)\s*(\([^)]*\))?\s*:\s*(.*)$")
SCENE_HEADING_RE = re.compile(r"^\s*\d*\s*(INT\.|EXT\.|INT/EXT|I/E\.)", re.IGNORECASE)


@dataclass
class Speech:
    speaker_raw: Optional[str]
    text: str
    scene_index: Optional[int] = None



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


def map_scenes(
    segments: Sequence[Segment],
    speeches: Sequence[Speech],
    headings: Sequence[str],
    min_lines: int = 2,
) -> List[Dict[str, object]]:
    """
    Maps script scene headings to subtitle time: each subtitle line takes the scene of
    the script words it aligns to; scenes with enough lines, in increasing time order,
    become contiguous time ranges (boundaries halfway between scenes' dialogue).
    """
    dialogue = [s for s in segments if not s.sound_only and not s.non_english and s.tokens]
    a_tokens: List[str] = []
    a_owner: List[int] = []
    for idx, seg in enumerate(dialogue):
        for tok in seg.tokens:
            a_tokens.append(tok)
            a_owner.append(idx)
    b_tokens: List[str] = []
    b_scene: List[Optional[int]] = []
    for speech in speeches:
        for tok in _normalize_tokens(speech.text):
            b_tokens.append(tok)
            b_scene.append(speech.scene_index)
    mapping = align_tokens(a_tokens, b_tokens)
    votes: Dict[int, collections.Counter] = collections.defaultdict(collections.Counter)
    for ia, ib in mapping.items():
        if b_scene[ib] is not None:
            votes[a_owner[ia]][b_scene[ib]] += 1
    by_scene: Dict[int, List[Segment]] = collections.defaultdict(list)
    for idx, counter in votes.items():
        scene, n = counter.most_common(1)[0]
        if n >= 2 and n / sum(counter.values()) >= 0.6:
            by_scene[scene].append(dialogue[idx])

    kept: List[Tuple[int, float, float, int]] = []
    for scene in sorted(by_scene):
        lines = sorted(by_scene[scene], key=lambda s: s.start)
        if len(lines) < min_lines:
            continue
        # Robust span: drop lines far from the scene's median time (stray matches)
        mid = lines[len(lines) // 2].start
        lines = [s for s in lines if abs(s.start - mid) <= 900]
        if len(lines) < min_lines:
            continue
        first, last = lines[0].start, max(s.end for s in lines)
        if kept and first < kept[-1][1]:
            continue
        kept.append((scene, first, last, len(lines)))

    scenes: List[Dict[str, object]] = []
    for k, (scene, first, last, n) in enumerate(kept):
        if k == 0:
            start = max(0.0, first - 5.0)
        else:
            prev_last = kept[k - 1][2]
            start = (prev_last + first) / 2 if first > prev_last else first
        if k + 1 < len(kept):
            nxt = kept[k + 1][1]
            end = (last + nxt) / 2 if nxt > last else nxt
        else:
            end = last + 5.0
        scenes.append({
            "index": k + 1,
            "script_scene": scene,
            "heading": headings[scene] if 0 <= scene < len(headings) else "",
            "start": round(start, 3),
            "end": round(end, 3),
            "first_line": round(first, 3),
            "lines": n,
        })
    return scenes


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
    scene_sources: List[Tuple[str, List[Speech], List[str]]] = []
    for path, bucket, role in [(p, transcripts, "transcript") for p in transcript_paths] + [
        (p, screenplays, "screenplay") for p in screenplay_paths
    ]:
        speeches, headings, fmt = load_script(path)
        bucket.append(speeches)
        if headings:
            scene_sources.append((role, speeches, headings))
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
    # Scenes: prefer the script whose headings map to the most scenes
    scenes: List[Dict[str, object]] = []
    for _role, speeches, headings in scene_sources:
        candidate = map_scenes(segments, speeches, headings)
        if len(candidate) > len(scenes):
            scenes = candidate
    stats["scenes"] = len(scenes)
    report = build_report(segments, stats, resolver, {
        **(meta or {}),
        "subtitle_file": subtitle_path.name,
        "scripts": script_meta,
        "output_vtt": str(out_vtt),
    })
    report["scenes"] = scenes
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
