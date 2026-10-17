"""
Whole-movie ingest: block-cached alignment, subtitle speaker labels, voice checks.

Pipeline for a long source (see plans/10_04_movie_support_plan.md, M01-M08):

1. Stage seekable FLAC audio once (movie_media.ensure_movie_audio).
2. Resolve subtitles: explicit file, labeled VTT from tools/label_subtitles.py, sidecar
   file, or the embedded text track (SDH preferred).
3. Load cues with per-segment speakers (movie_subtitles.load_movie_cues); each cue is one
   MFA chunk sliced from the analysis track (center channel for 5.1 sources).
4. Align per 10-minute block, cached by a key over the block's chunks, dictionary,
   model, and preprocessing version. Range ingests only touch overlapping blocks; later
   ingests reuse finished blocks.
5. Assign words to segments by token order, then to speakers.
6. Voice checks (WavLM-SV embeddings on the analysis track): flag labeled lines whose
   voice clearly matches another speaker better (made ineligible), and propagate labels
   to unlabeled lines only above a threshold calibrated on the labeled lines.
7. Apply user overrides (config/speaker_labels/<media hash>.json, written by the
   review-speakers command; stores times, IDs and text hashes only).
8. Build one inventory from all finished blocks.

Speaker files and caches never store subtitle text except the local, git-ignored
alignment corpus that MFA needs.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import soundfile as sf

from cache import compute_media_hash, hash_dict, hash_file_content
from movie_media import (
    DEFAULT_BLOCK_SECONDS,
    MovieAudio,
    blocks_overlapping,
    ensure_movie_audio,
    ensure_movie_stems,
    ensure_vocal_analysis,
    extract_subtitle_stream,
    labeled_subtitle_candidates,
    list_subtitle_streams,
    plan_blocks,
    read_stems_meta,
    select_subtitle_stream,
)
from movie_subtitles import assign_word_speakers, load_character_resolver, load_movie_cues

MOVIE_PREPROCESSOR_VERSION = "movie-1"
VOICE_VERSION = "voice-2"
MIN_VOICE_SECONDS = 1.0
MIN_CENTROID_SEGMENTS = 5
TARGET_PROPAGATION_PRECISION = 0.95
MISMATCH_MARGIN = 0.10
SEPARATION_DROP_DB = 15.0
MIN_CONFIDENT_MARGIN = 0.05
MIN_CONFIDENT_SIMILARITY = 0.50

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CHARACTERS = REPO_ROOT / "config" / "characters" / "star_wars.json"
OVERRIDES_DIR = REPO_ROOT / "config" / "speaker_labels"


def text_sha1(text: str) -> str:
    return hashlib.sha1(" ".join(text.split()).lower().encode("utf-8")).hexdigest()[:16]


def segment_key(cue_start: float, seg_index: int, text: str) -> str:
    return f"{cue_start:.3f}|{seg_index}|{text_sha1(text)}"


def parse_time(value: str) -> float:
    """Parses HH:MM:SS(.ms), MM:SS, or seconds."""
    value = value.strip()
    if not value:
        raise ValueError("empty time")
    parts = value.split(":")
    if len(parts) > 3:
        raise ValueError(f"Invalid time: {value}")
    total = 0.0
    for p in parts:
        total = total * 60 + float(p)
    return total


def parse_ranges(values: Optional[Sequence[str]]) -> List[Tuple[float, float]]:
    """Parses ["01:30:00-01:35:00", "600-900"] into sorted (start, end) seconds."""
    out: List[Tuple[float, float]] = []
    for v in values or []:
        for part in str(v).split(","):
            part = part.strip()
            if not part:
                continue
            if "-" not in part:
                raise ValueError(f"Range must be START-END: {part}")
            a, b = part.split("-", 1)
            s, e = parse_time(a), parse_time(b)
            if e <= s:
                raise ValueError(f"Range end must be after start: {part}")
            out.append((s, e))
    return sorted(out)


def format_time(sec: float) -> str:
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = sec - h * 3600 - m * 60
    return f"{h:02d}:{m:02d}:{s:06.3f}"


# ---------------------------------------------------------------------------
# Subtitles
# ---------------------------------------------------------------------------


def resolve_movie_subtitles(
    video: Path,
    media_hash: str,
    build_dir: Path,
    probe: Dict[str, Any],
    explicit: Optional[str] = None,
    stream_choice: Optional[str] = None,
    log: Callable[[str], None] = print,
) -> Tuple[Path, Dict[str, Any]]:
    if explicit:
        p = Path(explicit).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Subtitle file not found: {p}")
        return p, {"kind": "file", "path": str(p)}
    if not stream_choice:
        labeled = labeled_subtitle_candidates(video, build_dir)
        if labeled:
            return labeled[0], {"kind": "labeled_vtt", "path": str(labeled[0])}
        from source_discovery import find_associated_subtitles

        side = find_associated_subtitles(video)
        if side:
            return Path(side), {"kind": "sidecar", "path": side}
    streams = list_subtitle_streams(probe)
    chosen = select_subtitle_stream(streams, stream_choice or "auto")
    if chosen is None:
        images = [s for s in streams if s["image_based"]]
        hint = " (only image-based tracks found; OCR is not supported)" if images else ""
        raise FileNotFoundError(f"No usable subtitles for {video.name}{hint}. Pass --vtt <file>.")
    path = extract_subtitle_stream(video, chosen, media_hash, build_dir)
    log(f"[*] Using embedded subtitle stream {chosen['index']} ({chosen['codec']}, '{chosen['title']}')")
    return path, {"kind": "embedded", "path": str(path), "stream": chosen}


# ---------------------------------------------------------------------------
# Voice checks (M06)
# ---------------------------------------------------------------------------


@dataclass
class SegmentRecord:
    key: str
    chunk_id: str
    seg_index: int
    cue_start: float
    start: float
    end: float
    label: Optional[str]
    label_source: str
    text_hash: str
    speaker: Optional[str] = None
    status: str = "unlabeled"
    eligible: bool = False
    voice_best: Optional[str] = None
    voice_sim: Optional[float] = None
    voice_own_sim: Optional[float] = None
    voice_margin: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k != "extra"}
        d.update(self.extra)
        return d


class VoiceEmbedder:
    """Speaker embeddings per line, read by sample range from a mono or stereo audio file."""

    def __init__(self, cache_path: Path, audio_path: Path, log: Callable[[str], None] = print):
        self.cache_path = cache_path
        self.audio_path = audio_path
        self.sample_rate = int(sf.info(str(audio_path)).samplerate)
        self.log = log
        self._diarizer: Any = None
        self.cache: Dict[str, np.ndarray] = {}
        if cache_path.exists():
            try:
                with np.load(str(cache_path)) as data:
                    self.cache = {k: data[k].copy() for k in data.files}
            except Exception:
                self.cache = {}
        self.dirty = False

    def available(self) -> bool:
        try:
            from diarization import SpeakerDiarizer

            if self._diarizer is None:
                self._diarizer = SpeakerDiarizer()
            model, _, _ = self._diarizer._get_sv_backend()
            return model is not None
        except Exception:
            return False

    def embed(self, start: float, end: float) -> np.ndarray:
        key = f"{start:.3f}_{end:.3f}"
        if key in self.cache:
            return self.cache[key]
        sr = self.sample_rate
        audio, file_sr = sf.read(str(self.audio_path), start=int(start * sr), stop=int(end * sr), dtype="float32")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        vec = self._diarizer.extract_embedding(audio, sample_rate=file_sr)
        self.cache[key] = vec
        self.dirty = True
        return vec

    def save(self) -> None:
        if self.dirty:
            tmp = self.cache_path.with_name(self.cache_path.stem + ".partial.npz")
            np.savez(str(tmp), **self.cache)  # type: ignore[arg-type]
            os.replace(tmp, self.cache_path)
            self.dirty = False


def _centroids(vectors: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
    out = {}
    for spk, vecs in vectors.items():
        if len(vecs) < MIN_CENTROID_SEGMENTS:
            continue
        m = np.mean(np.stack(vecs), axis=0)
        # One trimming pass: drop the 10% least similar, then recompute
        sims = np.stack(vecs) @ (m / (np.linalg.norm(m) + 1e-9))
        keep = sims >= np.quantile(sims, 0.10)
        m = np.mean(np.stack(vecs)[keep], axis=0)
        out[spk] = m / (np.linalg.norm(m) + 1e-9)
    return out


def separation_dropped_chunks(
    chunks: Dict[str, Dict[str, Any]],
    vocal_analysis: Path,
    center_analysis: Path,
    margin_db: float = SEPARATION_DROP_DB,
) -> Tuple[set, Dict[str, Any]]:
    """
    Chunks whose separated vocals are far quieter than usual relative to the center channel
    (the separation model moved the voice into the background stem, e.g. radio-filtered or
    whispered lines). Words from them would render silent from the vocal stem.
    """
    def level(path: Path, a: float, b: float) -> float:
        sr = sf.info(str(path)).samplerate
        y, _ = sf.read(str(path), start=int(a * sr), stop=int(b * sr), dtype="float32", always_2d=True)
        return float(20 * np.log10(np.sqrt(np.mean(y.mean(axis=1) ** 2)) + 1e-9)) if len(y) else -120.0

    diffs = {cid: level(vocal_analysis, c["start_sec"], c["end_sec"]) - level(center_analysis, c["start_sec"], c["end_sec"])
             for cid, c in chunks.items()}
    if not diffs:
        return set(), {}
    typical = float(np.median(list(diffs.values())))
    dropped = {cid for cid, d in diffs.items() if d < typical - margin_db}
    return dropped, {"typical_vocals_vs_center_db": round(typical, 1), "threshold_db": round(typical - margin_db, 1),
                     "dropped_chunks": len(dropped)}


def run_voice_checks(
    records: List[SegmentRecord],
    embedder: VoiceEmbedder,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Verifies labeled segments and propagates labels to unlabeled ones (see module doc)."""
    usable = [r for r in records if (r.end - r.start) >= MIN_VOICE_SECONDS and r.status != "separation_dropped"]
    t0 = time.time()
    emb: Dict[str, np.ndarray] = {}
    for n, r in enumerate(usable, 1):
        emb[r.key] = embedder.embed(r.start, r.end)
        if n % 250 == 0:
            log(f"    voice embeddings {n}/{len(usable)}")
            embedder.save()
    embedder.save()

    named = [r for r in usable if r.label and not r.label.startswith("role-") and r.status != "user"]
    by_spk: Dict[str, List[np.ndarray]] = {}
    for r in named:
        if r.label:
            by_spk.setdefault(r.label, []).append(emb[r.key])
    cents = _centroids(by_spk)

    # Leave-one-out nearest-centroid scoring of labeled lines (also the calibration set)
    sums = {s: np.sum(np.stack(v), axis=0) for s, v in by_spk.items() if s in cents}
    counts = {s: len(by_spk[s]) for s in sums}
    calib: List[Tuple[float, bool]] = []
    scored: List[Tuple[SegmentRecord, str, float, float, float]] = []
    for r in named:
        if not cents:
            continue
        e = emb[r.key]
        if r.label in cents:
            sims = {}
            for s, c in cents.items():
                if s == r.label and counts[s] > 1:
                    loo = sums[s] - e
                    c = loo / (np.linalg.norm(loo) + 1e-9)
                sims[s] = float(e @ c)
            ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
            best, best_sim = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else -1.0
            r.voice_best = best
            r.voice_sim = round(best_sim, 4)
            r.voice_own_sim = round(sims[r.label], 4)
            r.voice_margin = round(best_sim - second, 4)
            calib.append((best_sim - second, best == r.label))
            scored.append((r, best, best_sim, sims[r.label], best_sim - second))
        else:
            # Minor character (< MIN_CENTROID_SEGMENTS lines): evaluate against major-character centroids
            sims = {s: float(e @ c) for s, c in cents.items()}
            ranked = sorted(sims.items(), key=lambda kv: kv[1], reverse=True)
            best, best_sim = ranked[0]
            second = ranked[1][1] if len(ranked) > 1 else -1.0
            r.voice_best = best
            r.voice_sim = round(best_sim, 4)
            r.voice_margin = round(best_sim - second, 4)
            if r.label in cents:
                other_vecs = [emb[rec.key] for rec in named if rec.label == r.label and rec.key != r.key]
                if other_vecs:
                    m_other = np.mean(np.stack(other_vecs), axis=0)
                    own_sim = float(e @ (m_other / (np.linalg.norm(m_other) + 1e-9)))
                else:
                    own_sim = 0.0
            else:
                own_sim = 0.0
            r.voice_own_sim = round(own_sim, 4)
            scored.append((r, best, best_sim, own_sim, best_sim - second))

    # Calibrate a margin threshold with leave-one-out precision >= target
    threshold = None
    if calib:
        ordered = sorted(calib, key=lambda x: x[0], reverse=True)
        correct = 0
        for i, (margin, ok) in enumerate(ordered, 1):
            correct += int(ok)
            if i >= 20 and correct / i >= TARGET_PROPAGATION_PRECISION:
                threshold = margin
        loo_acc = sum(ok for _, ok in calib) / len(calib)
    else:
        loo_acc = 0.0
    if threshold is not None:
        threshold = max(threshold, MIN_CONFIDENT_MARGIN)

    # Text labels (subtitles/transcripts) agree with each other 97-99% of the time, so a
    # voice can only overrule one when the evidence is strong on every count:
    #   - the line is clearly closer to another speaker (gap >= MISMATCH_MARGIN),
    #   - that speaker clearly beats every other candidate (calibrated margin), and
    #   - the labeled speaker's profile is built from enough other lines to be trusted.
    mismatches = 0
    for r, best, best_sim, own_sim, margin in scored:
        if r.label in cents:
            own_lines = counts.get(r.label or "", 0) - 1
            strong = (
                best != r.label
                and threshold is not None
                and best_sim - own_sim >= MISMATCH_MARGIN
                and margin >= threshold
                and own_lines >= MIN_CENTROID_SEGMENTS
            )
        else:
            # Minor character (< MIN_CENTROID_SEGMENTS lines) evaluated against major-character centroids
            spk_lines = [rec for rec in named if rec.label == r.label]
            req_margin = min(threshold, MIN_CONFIDENT_MARGIN) if threshold is not None else MIN_CONFIDENT_MARGIN

            if len(spk_lines) <= 1:
                strong = (
                    best != r.label
                    and threshold is not None
                    and best_sim >= MIN_CONFIDENT_SIMILARITY
                    and margin >= req_margin
                )
            else:
                confident_matches = [
                    getattr(rec, "voice_best", None)
                    for rec in spk_lines
                    if getattr(rec, "voice_sim", 0.0) >= MIN_CONFIDENT_SIMILARITY
                    and getattr(rec, "voice_margin", 0.0) >= req_margin
                ]
                unanimous_major = (
                    len(confident_matches) == len(spk_lines)
                    and len(set(confident_matches)) == 1
                    and confident_matches[0] == best
                )

                strong = (
                    best != r.label
                    and threshold is not None
                    and best_sim >= MIN_CONFIDENT_SIMILARITY
                    and margin >= req_margin
                    and unanimous_major
                )
        if strong:
            r.status = "voice_mismatch"
            r.eligible = False
            mismatches += 1
        elif r.status in ("label",):
            r.status = "label_voice_ok" if best == r.label else "label_voice_unclear"

    propagated = 0
    if threshold is not None:
        for r in usable:
            if r.label or r.status == "user":
                continue
            e = emb[r.key]
            ranked = sorted(((s, float(e @ c)) for s, c in cents.items()), key=lambda kv: kv[1], reverse=True)
            if not ranked:
                continue
            best, best_sim = ranked[0]
            margin = best_sim - (ranked[1][1] if len(ranked) > 1 else -1.0)
            r.voice_best, r.voice_sim, r.voice_margin = best, round(best_sim, 4), round(margin, 4)
            if margin >= threshold:
                r.speaker = best
                r.status = "voice_propagated"
                r.eligible = True
                propagated += 1

    stats = {
        "voice_version": VOICE_VERSION,
        "segments_embedded": len(usable),
        "centroid_speakers": sorted(cents),
        "loo_accuracy": round(loo_acc, 4),
        "calibration_lines": len(calib),
        "propagation_margin_threshold": None if threshold is None else round(threshold, 4),
        "propagated": propagated,
        "voice_mismatch": mismatches,
        "seconds": round(time.time() - t0, 1),
    }
    return stats


# ---------------------------------------------------------------------------
# User overrides (M07)
# ---------------------------------------------------------------------------


def overrides_path(media_hash: str, base: Optional[Path] = None) -> Path:
    return (base or OVERRIDES_DIR) / f"{media_hash[:16]}.json"


def load_overrides(media_hash: str, base: Optional[Path] = None) -> Dict[str, Dict[str, Any]]:
    p = overrides_path(media_hash, base)
    if not p.exists():
        return {}
    data = json.loads(p.read_text(encoding="utf-8"))
    return {o["key"]: o for o in data.get("overrides", [])}


def save_overrides(media_hash: str, source_name: str, overrides: Dict[str, Dict[str, Any]], base: Optional[Path] = None) -> Path:
    p = overrides_path(media_hash, base)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "media_hash": media_hash,
        "source_name": source_name,
        "note": "Speaker label overrides from review-speakers. Keys are cue start|segment|text hash; no subtitle text.",
        "overrides": sorted(overrides.values(), key=lambda o: o["key"]),
    }
    tmp = p.with_name(f"{p.name}.tmp_{os.getpid()}")
    tmp.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    # Atomic replace: an interrupt mid-save never leaves a truncated label file
    os.replace(tmp, p)
    return p


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------


def _textgrid_words(tg_path: Path) -> List[Dict[str, Any]]:
    from inventory_builder import parse_textgrid

    tiers = parse_textgrid(str(tg_path))
    out = []
    for w in tiers.get("words", []):
        t = w["text"].strip()
        if not t or t.lower() in {"sil", "sp", "spn", "<eps>", "eps"}:
            continue
        if float(w["xmax"]) <= float(w["xmin"]):
            continue
        out.append({"text": t, "xmin": float(w["xmin"]), "xmax": float(w["xmax"])})
    return out


def ensure_movie_aligned(
    video_path: Union[str, Path],
    source_id: str,
    build_dir: Union[str, Path],
    run_mfa: Callable[[str, str], None],
    dict_hash: str,
    acoustic_model: str = "english_us_arpa",
    subtitle_path: Optional[str] = None,
    subtitle_stream: Optional[str] = None,
    characters_path: Optional[Union[str, Path]] = None,
    ranges: Optional[List[Tuple[float, float]]] = None,
    separate_stems: bool = True,
    voice_checks: bool = True,
    force_realign: bool = False,
    overrides_dir: Optional[Path] = None,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Ingests (or reuses) a movie and returns {'inventory_path', 'source_id', ...}."""
    from contracts import RationalTime, SourceRecord
    from inventory_builder import build_units_inventory

    v = Path(video_path).resolve()
    build = Path(build_dir).resolve()
    t_start = time.time()
    media_hash = compute_media_hash(v, cache_dir=build)
    audio: MovieAudio = ensure_movie_audio(v, media_hash, build, log=log)

    sub_path, sub_info = resolve_movie_subtitles(
        v, media_hash, build, audio.probe, explicit=subtitle_path, stream_choice=subtitle_stream, log=log
    )
    chars = Path(characters_path) if characters_path else DEFAULT_CHARACTERS
    resolver = load_character_resolver(chars)
    loaded = load_movie_cues(sub_path, resolver)
    cues = loaded["cues"]
    log(f"[*] Subtitles: {sub_path.name} ({sub_info['kind']}): {len(cues)} dialogue cues, "
        f"{loaded['stats']['labeled_segments']}/{loaded['stats']['segments']} segments labeled")

    blocks = plan_blocks(audio.duration_sec, DEFAULT_BLOCK_SECONDS)
    wanted_blocks = blocks_overlapping(blocks, ranges)

    # Dialogue isolation comes first: every analysis step (alignment, voice checks, join
    # features) reads the separated vocals for blocks that have them. Measured on Episode III
    # block 4: ASR recall of aligned words 0.77 vs 0.72 (single words) and 0.77 vs 0.76
    # (3-word windows) for separated vocals vs the raw center channel.
    stems: Dict[str, Any] = {}
    vocal_analysis: Optional[Path] = None
    stems_done: set = set()
    if separate_stems:
        stems = ensure_movie_stems(audio, block_indices=wanted_blocks, log=log)
        vocal_analysis = ensure_vocal_analysis(audio, log=log)
        if vocal_analysis is not None:
            stems_done = set(read_stems_meta(audio).get("blocks_done") or [])
        missing = [b for b in wanted_blocks if b not in stems_done]
        if missing:
            log(f"[!] Warning: {len(missing)} block(s) have no separated vocals; they are analyzed from the center channel")

    def block_analysis(b: int) -> Tuple[str, Path]:
        if vocal_analysis is not None and b in stems_done:
            return "vocals", vocal_analysis
        return audio.analysis_channel, audio.analysis_path

    sub_hash = hash_file_content(str(sub_path))
    align_root = build / "alignment" / source_id
    blocks_root = align_root / "blocks"
    block_cues: Dict[int, List[Tuple[str, Dict[str, Any]]]] = {}
    for idx, c in enumerate(cues):
        b = min(int(c["cue_start_sec"] // DEFAULT_BLOCK_SECONDS), len(blocks) - 1)
        block_cues.setdefault(b, []).append((f"b{b:03d}_c{idx:05d}", c))

    def block_key(b: int) -> str:
        items = [(cid, c["start_sec"], c["end_sec"], c["text"]) for cid, c in block_cues.get(b, [])]
        return hash_dict({
            "media_hash": media_hash,
            "block": list(blocks[b]),
            "chunks": items,
            "dict_hash": dict_hash,
            "acoustic_model": acoustic_model,
            "analysis": block_analysis(b)[0],
            "version": MOVIE_PREPROCESSOR_VERSION,
        })

    def block_done(b: int) -> bool:
        meta = blocks_root / f"block_{b:04d}" / "block_meta.json"
        if not meta.exists():
            return False
        try:
            return json.loads(meta.read_text(encoding="utf-8")).get("key") == block_key(b)
        except Exception:
            return False

    aligned_now = 0
    for n, b in enumerate(wanted_blocks, 1):
        if not block_cues.get(b):
            continue
        if block_done(b) and not force_realign:
            continue
        bdir = blocks_root / f"block_{b:04d}"
        corpus = bdir / "corpus"
        aligned = bdir / "aligned"
        shutil.rmtree(bdir, ignore_errors=True)
        corpus.mkdir(parents=True, exist_ok=True)
        aligned.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        chunk_list = block_cues[b]
        analysis_name, analysis_file = block_analysis(b)
        log(f"[*] Aligning block {b + 1}/{len(blocks)} ({blocks[b][0] / 60:.0f}-{blocks[b][1] / 60:.0f} min): "
            f"{len(chunk_list)} chunks from {'separated vocals' if analysis_name == 'vocals' else 'center channel'}")
        with sf.SoundFile(str(analysis_file)) as f:
            sr = f.samplerate
            for cid, c in chunk_list:
                f.seek(int(c["start_sec"] * sr))
                data = f.read(int((c["end_sec"] - c["start_sec"]) * sr), dtype="int16")
                sf.write(str(corpus / f"{cid}.wav"), data, sr, subtype="PCM_16")
                (corpus / f"{cid}.txt").write_text(c["text"] + "\n", encoding="utf-8")
        run_mfa(str(corpus), str(aligned))
        n_tg = len(list(aligned.glob("*.TextGrid")))
        (bdir / "block_meta.json").write_text(json.dumps({
            "key": block_key(b), "block": list(blocks[b]), "chunks": len(chunk_list), "textgrids": n_tg,
            "seconds": round(time.time() - t0, 1),
        }, indent=2), encoding="utf-8")
        # The 16 kHz chunk WAVs are only MFA input; keep the transcripts for inspection
        for w in corpus.glob("*.wav"):
            w.unlink()
        aligned_now += 1
        log(f"    {n_tg}/{len(chunk_list)} chunks aligned in {time.time() - t0:.0f}s")

    done_blocks = [b for b in range(len(blocks)) if block_cues.get(b) and block_done(b)]
    if not done_blocks:
        raise RuntimeError(f"No aligned blocks for {v.name}")

    # Fallback: chunks MFA could not align from separated vocals (separation occasionally
    # removes too much of a line) are retried once from the raw center channel.
    for b in done_blocks:
        if block_analysis(b)[0] != "vocals":
            continue
        bdir = blocks_root / f"block_{b:04d}"
        meta_p = bdir / "block_meta.json"
        bmeta = json.loads(meta_p.read_text(encoding="utf-8"))
        if bmeta.get("fallback_tried"):
            continue
        aligned = bdir / "aligned"
        missing_chunks = [(cid, c) for cid, c in block_cues[b] if not (aligned / f"{cid}.TextGrid").exists()]
        if missing_chunks:
            log(f"[*] Block {b + 1}: retrying {len(missing_chunks)} chunk(s) from the center channel")
            fb_corpus = bdir / "fallback_corpus"
            fb_aligned = bdir / "fallback_aligned"
            shutil.rmtree(fb_corpus, ignore_errors=True)
            shutil.rmtree(fb_aligned, ignore_errors=True)
            fb_corpus.mkdir(parents=True)
            fb_aligned.mkdir(parents=True)
            with sf.SoundFile(str(audio.analysis_path)) as f:
                sr = f.samplerate
                for cid, c in missing_chunks:
                    f.seek(int(c["start_sec"] * sr))
                    data = f.read(int((c["end_sec"] - c["start_sec"]) * sr), dtype="int16")
                    sf.write(str(fb_corpus / f"{cid}.wav"), data, sr, subtype="PCM_16")
                    (fb_corpus / f"{cid}.txt").write_text(c["text"] + "\n", encoding="utf-8")
            try:
                run_mfa(str(fb_corpus), str(fb_aligned))
            except Exception as err:
                log(f"    fallback alignment failed: {err}")
            recovered = []
            for tg in fb_aligned.glob("*.TextGrid"):
                shutil.move(str(tg), str(aligned / tg.name))
                recovered.append(tg.stem)
            shutil.rmtree(fb_corpus, ignore_errors=True)
            shutil.rmtree(fb_aligned, ignore_errors=True)
            bmeta["fallback_center_chunks"] = sorted(recovered)
            bmeta["textgrids"] = len(list(aligned.glob("*.TextGrid")))
            log(f"    recovered {len(recovered)}/{len(missing_chunks)}")
        bmeta["fallback_tried"] = True
        meta_p.write_text(json.dumps(bmeta, indent=2), encoding="utf-8")

    def block_textgrids(b: int) -> int:
        try:
            return int(json.loads((blocks_root / f"block_{b:04d}" / "block_meta.json").read_text(encoding="utf-8")).get("textgrids", 0))
        except Exception:
            return 0

    # Line-level analysis (join features, voice checks, selector F0 reads) uses one track for
    # the whole movie: separated vocals when every aligned block has them, else center channel.
    lines_on_vocals = vocal_analysis is not None and all(b in stems_done for b in done_blocks)
    line_analysis = vocal_analysis if lines_on_vocals and vocal_analysis is not None else audio.analysis_path
    voice_audio = "vocals" if lines_on_vocals else "center"

    # Join-cost features over speech spans (pYIN over a whole movie would take ~30 min per mix)
    from features import precompute_span_features
    precompute_span_features(
        line_analysis,
        [(c["start_sec"], c["end_sec"]) for b in done_blocks for _, c in block_cues[b]],
        log=log,
    )

    # Segment records with word-level timing from TextGrids
    overrides = load_overrides(media_hash, overrides_dir)
    records: List[SegmentRecord] = []
    chunks_meta: Dict[str, Any] = {}
    for b in done_blocks:
        aligned = blocks_root / f"block_{b:04d}" / "aligned"
        for cid, c in block_cues[b]:
            tg = aligned / f"{cid}.TextGrid"
            seg_entries = []
            words = _textgrid_words(tg) if tg.exists() else []
            assigned = assign_word_speakers([w["text"] for w in words], c["segments"]) if words else []
            for si, seg in enumerate(c["segments"]):
                key = segment_key(c["cue_start_sec"], si, seg["text"])
                w_idx = [i for i, a in enumerate(assigned) if a is seg]
                if w_idx:
                    s = c["start_sec"] + words[w_idx[0]]["xmin"]
                    e = c["start_sec"] + words[w_idx[-1]]["xmax"]
                else:
                    s, e = c["cue_start_sec"], c["cue_end_sec"]
                rec = SegmentRecord(
                    key=key, chunk_id=cid, seg_index=si, cue_start=c["cue_start_sec"],
                    start=round(s, 3), end=round(e, 3), label=seg["speaker_id"],
                    label_source=seg.get("label_source", "none"), text_hash=text_sha1(seg["text"]),
                )
                if seg["speaker_id"]:
                    rec.speaker, rec.status, rec.eligible = seg["speaker_id"], "label", True
                ov = overrides.get(key)
                if ov is not None:
                    rec.speaker = ov.get("speaker_id")
                    rec.label = rec.speaker
                    rec.status = "user"
                    rec.eligible = bool(rec.speaker)
                rec.extra["aligned_words"] = len(w_idx)
                records.append(rec)
                seg_entries.append(rec)
            chunks_meta[cid] = {
                "start_sec": c["start_sec"],
                "end_sec": c["end_sec"],
                "duration": c["duration"],
                "text": c["text"],
                "speaker_id": None,
                "segments": [{"key": r.key, "tokens": c["segments"][r.seg_index]["tokens"]} for r in seg_entries],
            }

    labels_path = audio.root / "speaker_labels.json"
    labels_key = hash_dict({
        "blocks": {str(b): block_key(b) for b in done_blocks},
        "textgrids": {str(b): block_textgrids(b) for b in done_blocks},
        "overrides": sorted((k, o.get("speaker_id")) for k, o in overrides.items()),
        "voice_checks": bool(voice_checks),
        "voice_version": VOICE_VERSION,
        "voice_audio": voice_audio,
        "separation_check_db": SEPARATION_DROP_DB,
        "subtitles": sub_hash,
    })
    cached_doc: Optional[Dict[str, Any]] = None
    if labels_path.exists() and not force_realign:
        try:
            doc = json.loads(labels_path.read_text(encoding="utf-8"))
            if doc.get("labels_key") == labels_key:
                cached_doc = doc
        except Exception:
            cached_doc = None

    voice_stats: Dict[str, Any] = {"skipped": "disabled"}
    separation_stats: Dict[str, Any] = {}
    dropped_chunks: set = set()
    if voice_audio == "vocals" and vocal_analysis is not None and cached_doc is None:
        dropped_chunks, separation_stats = separation_dropped_chunks(
            {cid: cm for cid, cm in chunks_meta.items()}, vocal_analysis, audio.analysis_path,
        )
        for r in records:
            if r.chunk_id in dropped_chunks:
                r.status = "separation_dropped"
                r.eligible = False
        if dropped_chunks:
            log(f"[!] {len(dropped_chunks)} line(s) lost their voice in vocal separation; excluded from mixes "
                "(their words would render silent from the vocal stem)")
    if cached_doc is not None:
        saved = {s["key"]: s for s in cached_doc.get("segments", [])}
        for r in records:
            s = saved.get(r.key)
            if s:
                for attr in ("speaker", "status", "eligible", "voice_best", "voice_sim", "voice_own_sim", "voice_margin"):
                    setattr(r, attr, s.get(attr))
        voice_stats = cached_doc.get("voice", voice_stats)
        separation_stats = cached_doc.get("separation", {})
    elif voice_checks:
        emb_cache = audio.root / ("voice_embeddings_vocals.npz" if voice_audio == "vocals" else "voice_embeddings.npz")
        embedder = VoiceEmbedder(emb_cache, line_analysis, log=log)
        if embedder.available():
            log(f"[*] Voice checks ({'separated vocals' if voice_audio == 'vocals' else 'center channel'}): "
                "verifying labeled lines and propagating labels to unlabeled lines")
            voice_stats = run_voice_checks(records, embedder, log=log)
            voice_stats["audio"] = voice_audio
        else:
            voice_stats = {"skipped": "speaker verification model unavailable"}
            log("[!] Warning: WavLM-SV model unavailable; skipping voice checks (labels are unverified)")

    rec_by_key = {r.key: r for r in records}
    for cid, cm in chunks_meta.items():
        segs = []
        for s in cm["segments"]:
            r = rec_by_key[s["key"]]
            segs.append({
                "key": r.key,
                "tokens": s["tokens"],
                "speaker_id": r.speaker,
                "is_eligible": bool(r.speaker) and r.eligible,
                "eligibility_reason": "" if (r.speaker and r.eligible) else (
                    r.status if r.status in ("voice_mismatch", "separation_dropped") else "unassigned"
                ),
                "identity_status": r.status,
            })
        cm["segments"] = segs

    # Labels file (no text)
    status_counts: Dict[str, int] = {}
    speaker_seconds: Dict[str, float] = {}
    for r in records:
        status_counts[r.status] = status_counts.get(r.status, 0) + 1
        if r.speaker and r.eligible:
            speaker_seconds[r.speaker] = speaker_seconds.get(r.speaker, 0.0) + (r.end - r.start)
    labels_doc = {
        "labels_key": labels_key,
        "media_hash": media_hash,
        "source_id": source_id,
        "subtitles": {k: v for k, v in sub_info.items() if k != "stream"} | {"sha256": sub_hash},
        "blocks_aligned": done_blocks,
        "blocks_total": len(blocks),
        "status_counts": status_counts,
        "voice": voice_stats,
        "separation": separation_stats,
        "speaker_seconds": {k: round(v, 1) for k, v in sorted(speaker_seconds.items(), key=lambda kv: -kv[1])},
        "segments": [r.to_dict() for r in records],
    }
    labels_path.write_text(json.dumps(labels_doc, indent=1), encoding="utf-8")

    manifest = {
        "source_video": str(v),
        "source_vtt": str(sub_path),
        "audio_source_path": None,
        "speaker_id": None,
        "single_speaker_verified": False,
        "movie": True,
        "chunks": chunks_meta,
        "total_chunks": len(chunks_meta),
    }
    align_root.mkdir(parents=True, exist_ok=True)
    manifest_path = align_root / "movie_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")

    stems_ok = bool(stems.get("complete")) if separate_stems else False
    if not separate_stems:
        meta_p = audio.root / "stems" / "stems_meta.json"
        if meta_p.exists():
            stems_ok = bool(json.loads(meta_p.read_text(encoding="utf-8")).get("complete"))
    stem_dir = audio.root / "stems"
    video: Dict[str, Any] = next((s for s in audio.probe.get("streams", []) if s.get("codec_type") == "video"), {})
    audio_stream: Dict[str, Any] = next((s for s in audio.probe.get("streams", []) if s.get("codec_type") == "audio"), {})
    fps = None
    avg_fps_str = video.get("avg_frame_rate", "")
    if avg_fps_str and avg_fps_str != "0/0":
        try:
            cand_fps = RationalTime.from_str(avg_fps_str)
            if cand_fps.numerator > 0 and cand_fps.denominator > 0:
                fps = cand_fps
        except Exception:
            pass
    if fps is None:
        r_fps_str = video.get("r_frame_rate", "24000/1001")
        if r_fps_str and r_fps_str != "0/0":
            try:
                cand_r = RationalTime.from_str(r_fps_str)
                if cand_r.numerator > 0 and cand_r.denominator > 0:
                    fps = cand_r
            except Exception:
                pass
    if fps is None:
        fps = RationalTime(24000, 1001)

    pad_val = audio_stream.get("initial_padding") or audio_stream.get("start_pad") or 0
    try:
        codec_delay = int(pad_val)
    except (ValueError, TypeError):
        codec_delay = 0

    def _safe_flt(val: Any) -> float:
        if val is None or val == "N/A" or val == "":
            return 0.0
        try:
            import math
            res = float(val)
            return res if math.isfinite(res) else 0.0
        except (ValueError, TypeError):
            return 0.0

    src_rec = SourceRecord(
        content_id=source_id,
        source_path=str(v),
        native_sample_rate=48000,
        native_channels=2,
        selected_audio_stream=int(audio_stream.get("index", 1)),
        selected_video_stream=int(video.get("index", 0)),
        decoded_audio_origin_sec=_safe_flt(audio_stream.get("start_time")),
        decoded_video_origin_sec=_safe_flt(video.get("start_time")),
        codec_delay_samples=codec_delay,
        video_fps=fps.to_dict(),
        video_duration_sec=audio.duration_sec,
        audio_duration_sec=audio.duration_sec,
        cached_pcm_path=str(audio.program_path),
        cached_vocal_pcm_path=str(stem_dir / "vocals.flac") if stems_ok else "",
        cached_instrumental_pcm_path=str(stem_dir / "instrumental.flac") if stems_ok else "",
        cached_analysis_pcm_path=str(line_analysis),
    )

    inv_dir = build / "inventory" / source_id
    from cache import resolve_inventory_path
    inv_path = inv_dir / "units_index.json.gz"
    resolved_inv = resolve_inventory_path(inv_dir)
    inv_meta_path = inv_dir / "inventory_metadata.json"
    inv_key = hash_dict({
        "blocks": {b: block_key(b) for b in done_blocks},
        "textgrids": {b: block_textgrids(b) for b in done_blocks},
        "labels": hash_dict({"segments": [(r.key, r.speaker, r.eligible, r.status) for r in records]}),
        "stems_ok": stems_ok,
        "line_analysis": voice_audio,
        "version": MOVIE_PREPROCESSOR_VERSION,
    })
    hit = False
    if resolved_inv.exists() and inv_meta_path.exists() and not force_realign:
        try:
            hit = json.loads(inv_meta_path.read_text(encoding="utf-8")).get("inventory_key") == inv_key
            inv_path = resolved_inv
        except Exception:
            hit = False
    if not hit:
        log(f"[*] Building movie inventory from {len(done_blocks)}/{len(blocks)} aligned blocks")
        inv_dir.mkdir(parents=True, exist_ok=True)
        build_units_inventory(
            str(blocks_root), str(manifest_path), str(inv_path), source_record=src_rec,
            recover_oov_words=True,
        )
        meta_dict = {}
        if inv_meta_path.exists():
            try:
                meta_dict = json.loads(inv_meta_path.read_text(encoding="utf-8"))
            except Exception:
                meta_dict = {}
        meta_dict.update({
            "inventory_key": inv_key, "movie": True, "media_hash": media_hash,
            "blocks_aligned": done_blocks, "blocks_total": len(blocks),
        })
        inv_meta_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")
    else:
        log(f"[+] Using cached movie inventory: {inv_path}")

    return {
        "inventory_path": str(inv_path),
        "source_id": source_id,
        "media_hash": media_hash,
        "labels_path": str(labels_path),
        "blocks_aligned": done_blocks,
        "blocks_total": len(blocks),
        "blocks_aligned_this_run": aligned_now,
        "stems_complete": stems_ok,
        "status_counts": status_counts,
        "voice": voice_stats,
        "speaker_seconds": labels_doc["speaker_seconds"],
        "cache_hit": hit and aligned_now == 0,
        "seconds": round(time.time() - t_start, 1),
    }
