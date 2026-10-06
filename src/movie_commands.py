"""
CLI commands for ingested movies: scenes, speakers, review-speakers.

Review answers are saved to config/speaker_labels/<media hash>.json as overrides keyed by
cue start, segment index and a text hash (no subtitle text), and are applied on the next
ingest/mix (alignment is cached, so only labels and the inventory are rebuilt).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

from cache import compute_media_hash
from movie_ingest import format_time, load_overrides, parse_ranges, save_overrides
from movie_media import movie_dir


def _labels_for(video: Path, build_dir: Path) -> Tuple[str, Dict[str, Any]]:
    media_hash = compute_media_hash(video, cache_dir=build_dir)
    lp = movie_dir(build_dir, media_hash) / "speaker_labels.json"
    if not lp.exists():
        raise FileNotFoundError(f"{video.name} has not been ingested yet. Run: run_mixer.py ingest \"{video}\"")
    return media_hash, json.loads(lp.read_text(encoding="utf-8"))


def cmd_scenes(video: Path, scenes: List[Dict[str, Any]], out: Callable[[str], None] = print) -> int:
    if not scenes:
        out(f"[!] No scenes for {video.name}. Scenes come from a screenplay passed to tools/label_subtitles.py --screenplay.")
        return 1
    out(f"[+] {len(scenes)} scenes in {video.name} (use --scene <number> or --scene <keyword>):")
    for sc in scenes:
        out(f"  {sc['index']:>4}  {format_time(sc['start'])} - {format_time(sc['end'])}  {sc.get('heading', '')}")
    return 0


def cmd_speakers(videos: List[Path], build_dir: Path, out: Callable[[str], None] = print) -> int:
    docs: List[Tuple[str, Dict[str, Any]]] = []
    if videos:
        for v in videos:
            _, doc = _labels_for(v, build_dir)
            docs.append((v.name, doc))
    else:
        for lp in sorted((build_dir / "movies").glob("*/speaker_labels.json")):
            doc = json.loads(lp.read_text(encoding="utf-8"))
            docs.append((doc.get("source_id", lp.parent.name), doc))
    if not docs:
        out("[!] No ingested movies found.")
        return 1
    for name, doc in docs:
        out(f"\n[+] {name}: {len(doc.get('blocks_aligned', []))}/{doc.get('blocks_total')} blocks aligned")
        counts = doc.get("status_counts", {})
        out("    lines: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
        voice = doc.get("voice", {})
        if voice and "skipped" not in voice:
            out(f"    voice: leave-one-out accuracy {voice.get('loo_accuracy')}, flagged {voice.get('voice_mismatch')}, "
                f"propagated {voice.get('propagated')}")
        for spk, secs in doc.get("speaker_seconds", {}).items():
            out(f"    {spk:<22} {secs:7.1f}s")
    return 0


def _segment_texts(build_dir: Path, source_id: str) -> Dict[str, str]:
    mp = build_dir / "alignment" / source_id / "movie_manifest.json"
    if not mp.exists():
        return {}
    data = json.loads(mp.read_text(encoding="utf-8"))
    out = {}
    for c in data.get("chunks", {}).values():
        for s in c.get("segments", []):
            out[s["key"]] = " ".join(s.get("tokens", []))
    return out


def _play(path: Path) -> None:
    """Starts playing a WAV without blocking, so the user can answer while it plays."""
    if sys.platform == "win32":
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT)
        except Exception:
            pass


def _stop_playback() -> None:
    if sys.platform == "win32":
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass


REVIEW_AUDIO_CHOICES = ("mix", "center", "vocals")


def _review_audio_path(root: Path, audio: str) -> Path:
    """mix: the film's stereo mix; center: raw 5.1 center channel; vocals: separated dialogue
    stem (music and effects removed; what alignment and voice checks analyze)."""
    if audio == "center":
        return root / "audio" / "analysis_16k.flac"
    if audio == "vocals":
        stem = root / "stems" / "vocals.flac"
        if stem.exists():
            return stem
    return root / "audio" / "program_stereo_48k.flac"


def cmd_review_speakers(
    video: Path,
    build_dir: Path,
    only: str = "flagged",
    ranges: Optional[List[str]] = None,
    limit: int = 50,
    play: bool = True,
    input_fn: Callable[[str], str] = input,
    out: Callable[[str], None] = print,
    overrides_dir: Optional[Path] = None,
    audio: str = "mix",
) -> int:
    media_hash, doc = _labels_for(video, build_dir)
    rngs = parse_ranges(ranges)
    segs = doc.get("segments", [])

    def wanted(s: Dict[str, Any]) -> bool:
        if rngs and not any(a <= s["start"] and s["end"] <= b for a, b in rngs):
            return False
        st = s.get("status")
        if only == "all":
            return True
        if only == "unlabeled":
            return st == "unlabeled"
        if only == "voice":
            return st == "voice_propagated"
        return st in ("unlabeled", "voice_mismatch")

    todo = [s for s in segs if wanted(s)]
    todo.sort(key=lambda s: (s.get("status") != "voice_mismatch", -(s["end"] - s["start"])))
    todo = todo[:limit]
    if not todo:
        out("[+] Nothing to review.")
        return 0

    overrides = load_overrides(media_hash, overrides_dir)
    texts = _segment_texts(build_dir, doc["source_id"])
    known = sorted(doc.get("speaker_seconds", {}).keys())
    root = movie_dir(build_dir, media_hash)
    source_audio = _review_audio_path(root, audio)
    if audio == "vocals" and source_audio.name != "vocals.flac":
        out("[!] Separated vocals not available for this movie yet; playing the stereo mix.")
    preview_dir = root / "review"
    preview_dir.mkdir(parents=True, exist_ok=True)

    out(f"[*] Reviewing {len(todo)} lines of {video.name} ({only}), playing {audio}. Known speakers: {', '.join(known)}")
    out("    Type while the clip plays (it stops when you answer). Enter = accept suggestion (keeps the text label "
        "when there is one), <id> = set speaker, x = not usable, s = skip, r = replay, q = quit")
    changed = 0
    saved_path: Optional[Path] = None
    try:
        for n, s in enumerate(todo, 1):
            preview = preview_dir / f"{audio}_{s['start']:.3f}.wav"
            if play and source_audio.exists() and not preview.exists():
                sr = sf.info(str(source_audio)).samplerate
                a = max(0.0, s["start"] - 0.3)
                data, sr = sf.read(str(source_audio), start=int(a * sr), stop=int((s["end"] + 0.3) * sr), dtype="float32")
                sf.write(str(preview), np.asarray(data), sr, subtype="PCM_16")
            # Text labels (subtitles/transcripts) are more reliable than voice on short film
            # lines, so Enter keeps an existing label; the voice match is shown for reference.
            suggestion = s.get("label") or s.get("speaker") or s.get("voice_best")
            voice_note = ""
            if s.get("voice_best"):
                voice_note = (f" | closest voice={s.get('voice_best')} (similarity {s.get('voice_sim')}, "
                              f"lead over next voice {s.get('voice_margin')}; under ~0.05 means the voice model is unsure)")
            out(f"\n[{n}/{len(todo)}] {format_time(s['start'])} ({s['end'] - s['start']:.1f}s) status={s.get('status')} "
                f"text label={s.get('label')}{voice_note}")
            out(f"    \"{texts.get(s['key'], '')}\"")
            while True:
                if play and preview.exists():
                    _play(preview)
                ans = input_fn(f"    speaker [{suggestion or '-'}]: ").strip().lower()
                _stop_playback()
                if ans == "r":
                    continue
                break
            if ans == "q":
                break
            if ans == "s":
                continue
            if ans == "" and not suggestion:
                continue
            speaker: Optional[str]
            if ans == "x":
                speaker = None
            elif ans == "":
                speaker = suggestion
            else:
                if not re.match(r"^[a-z0-9][a-z0-9_\-]*$", ans):
                    out("    (invalid id; skipped)")
                    continue
                speaker = ans
            overrides[s["key"]] = {"key": s["key"], "start": s["start"], "end": s["end"], "speaker_id": speaker}
            # Save after every answer so Ctrl+C or a closed window never loses work
            saved_path = save_overrides(media_hash, video.name, overrides, overrides_dir)
            changed += 1
            out(f"    saved ({speaker or 'not usable'})")
    except (KeyboardInterrupt, EOFError):
        out("\n[*] Review stopped. All answers given so far are already saved.")
    finally:
        _stop_playback()
    if changed:
        out(f"\n[+] Saved {changed} label(s) to {saved_path}. They apply on the next ingest or mix of this movie (alignment is reused).")
    return 0
