"""
Large-file (movie) media support.

Long sources such as 2-hour 4K Blu-ray rips cannot be decoded into memory or written to
a single WAV (6 ch x 48 kHz x 2.4 h exceeds the 4 GB WAV limit). This module stages
compact, seekable FLAC files once per media hash and processes vocal separation in
resumable fixed-length blocks:

    .build/movies/<media hash 16>/
        probe.json
        audio/program_stereo_48k.flac   stereo downmix (raw output track, separation input)
        audio/analysis_16k.flac         center channel for 5.1/7.1, else mono downmix
        stems/blocks/block_NNNN_{vocals,instrumental}.flac
        stems/vocals.flac, stems/instrumental.flac   stitched full-length stems
        stems/stems_meta.json

It also selects and extracts embedded text subtitle tracks (SubRip/ASS/WebVTT/mov_text).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

LONG_SOURCE_SECONDS = 1200.0
DEFAULT_BLOCK_SECONDS = 600.0
BLOCK_CONTEXT_SECONDS = 5.0
SEPARATION_SAMPLE_RATE = 44100
STEMS_VERSION = "1"

TEXT_SUBTITLE_CODECS = {"subrip", "srt", "ass", "ssa", "webvtt", "mov_text", "text"}
IMAGE_SUBTITLE_CODECS = {"hdmv_pgs_subtitle", "dvd_subtitle", "dvb_subtitle", "xsub"}


def _run(cmd: List[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")


def probe_media(path: Union[str, Path]) -> Dict[str, Any]:
    res = _run([
        "ffprobe", "-v", "error", "-show_streams", "-show_format", "-show_chapters",
        "-of", "json", str(Path(path).resolve()),
    ])
    if res.returncode != 0:
        raise RuntimeError(f"ffprobe failed for {path}: {res.stderr.strip()}")
    return json.loads(res.stdout)


def media_duration(probe: Dict[str, Any]) -> float:
    try:
        return float(probe.get("format", {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def first_audio_stream(probe: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    for s in probe.get("streams", []):
        if s.get("codec_type") == "audio":
            return s
    return None


def is_long_source(path: Union[str, Path], probe: Optional[Dict[str, Any]] = None) -> bool:
    """True for sources that need the movie path: long runtime or surround audio."""
    p = Path(path)
    if p.suffix.lower() in {".wav", ".flac", ".ogg", ".mp3"}:
        return False
    try:
        probe = probe or probe_media(p)
    except Exception:
        return False
    audio = first_audio_stream(probe) or {}
    channels = int(audio.get("channels") or 2)
    return media_duration(probe) > LONG_SOURCE_SECONDS or channels > 2


def movie_dir(build_dir: Union[str, Path], media_hash: str) -> Path:
    return Path(build_dir).resolve() / "movies" / media_hash[:16]


def _center_channel_filter(audio: Dict[str, Any]) -> Optional[str]:
    layout = str(audio.get("channel_layout") or "")
    channels = int(audio.get("channels") or 2)
    if channels >= 6 or any(k in layout for k in ("5.1", "7.1", "6.1")):
        return "pan=mono|c0=FC"
    return None


def _atomic_ffmpeg(cmd_without_output: List[str], output: Path) -> None:
    # Per-process temp name: concurrent ingests of the same movie must not collide
    tmp = output.with_name(f"{output.stem}.partial-{os.getpid()}{output.suffix}")
    if tmp.exists():
        tmp.unlink()
    res = _run(cmd_without_output + [str(tmp)])
    if res.returncode != 0 or not tmp.exists() or tmp.stat().st_size == 0:
        if tmp.exists():
            tmp.unlink()
        if output.exists() and output.stat().st_size > 0:
            return
        raise RuntimeError(f"ffmpeg failed writing {output.name}: {res.stderr.strip()[-800:]}")
    if output.exists() and output.stat().st_size > 0:
        tmp.unlink()
        return
    os.replace(tmp, output)


@dataclass
class MovieAudio:
    root: Path
    probe: Dict[str, Any]
    duration_sec: float
    native_sample_rate: int
    native_channels: int
    program_path: Path
    analysis_path: Path
    analysis_channel: str


def ensure_movie_audio(
    video_path: Union[str, Path],
    media_hash: str,
    build_dir: Union[str, Path],
    log: Callable[[str], None] = print,
) -> MovieAudio:
    """Extract the program (stereo 48 kHz) and analysis (16 kHz mono) FLAC files once."""
    v = Path(video_path).resolve()
    root = movie_dir(build_dir, media_hash)
    audio_dir = root / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    probe_path = root / "probe.json"
    if probe_path.exists():
        probe = json.loads(probe_path.read_text(encoding="utf-8"))
    else:
        probe = probe_media(v)
        probe_path.write_text(json.dumps(probe, indent=2), encoding="utf-8")

    audio = first_audio_stream(probe)
    if audio is None:
        raise ValueError(f"No audio stream in {v}")
    a_idx = int(audio.get("index", 0))
    native_sr = int(audio.get("sample_rate") or 48000)
    native_ch = int(audio.get("channels") or 2)

    program = audio_dir / "program_stereo_48k.flac"
    if not program.exists():
        log(f"[*] Extracting stereo program audio once: {v.name}")
        t0 = time.time()
        _atomic_ffmpeg([
            "ffmpeg", "-y", "-v", "error", "-i", str(v), "-map", f"0:{a_idx}", "-vn",
            "-ac", "2", "-ar", "48000", "-sample_fmt", "s16", "-c:a", "flac",
        ], program)
        log(f"    done in {time.time() - t0:.0f}s ({program.stat().st_size / 1e6:.0f} MB)")

    center = _center_channel_filter(audio)
    analysis = audio_dir / "analysis_16k.flac"
    if not analysis.exists():
        log(f"[*] Extracting analysis audio ({'center channel' if center else 'mono downmix'})")
        t0 = time.time()
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", str(v), "-map", f"0:{a_idx}", "-vn"]
        if center:
            cmd += ["-af", center]
        cmd += ["-ac", "1", "-ar", "16000", "-sample_fmt", "s16", "-c:a", "flac"]
        _atomic_ffmpeg(cmd, analysis)
        log(f"    done in {time.time() - t0:.0f}s")

    return MovieAudio(
        root=root,
        probe=probe,
        duration_sec=media_duration(probe),
        native_sample_rate=native_sr,
        native_channels=native_ch,
        program_path=program,
        analysis_path=analysis,
        analysis_channel="center" if center else "downmix",
    )


def plan_blocks(duration_sec: float, block_sec: float = DEFAULT_BLOCK_SECONDS) -> List[Tuple[float, float]]:
    if duration_sec <= 0:
        return []
    blocks = []
    t = 0.0
    while t < duration_sec - 1e-6:
        blocks.append((t, min(duration_sec, t + block_sec)))
        t += block_sec
    return blocks


def blocks_overlapping(blocks: List[Tuple[float, float]], ranges: Optional[List[Tuple[float, float]]]) -> List[int]:
    if not ranges:
        return list(range(len(blocks)))
    out = []
    for i, (bs, be) in enumerate(blocks):
        if any(rs < be and re_ > bs for rs, re_ in ranges):
            out.append(i)
    return out


class _Separator:
    """Lazily loaded in-process audio-separator model (loaded once for all blocks)."""

    def __init__(self, model_name: str, output_dir: Path):
        self.model_name = model_name
        self.output_dir = output_dir
        self._sep: Any = None

    def _load(self) -> Any:
        if self._sep is None:
            import logging
            from audio_separator.separator import Separator

            kwargs: Dict[str, Any] = {
                "output_dir": str(self.output_dir),
                "output_format": "WAV",
                "log_level": logging.WARNING,
                "use_soundfile": True,
                "mdx_params": {
                    "hop_length": 1024, "segment_size": 256, "overlap": 0.25,
                    "batch_size": 1, "enable_denoise": False,
                },
            }
            uvr_dir = Path("D:/Program Files/Ultimate Vocal Remover/models/MDX_Net_Models")
            if (uvr_dir / self.model_name).exists():
                kwargs["model_file_dir"] = str(uvr_dir)
            sep = Separator(**kwargs)
            sep.load_model(model_filename=self.model_name)
            self._sep = sep
        return self._sep

    def separate(self, wav_path: Path) -> Tuple[Path, Path]:
        sep = self._load()
        outputs = sep.separate(str(wav_path))
        vocals = inst = None
        for name in outputs:
            p = Path(name)
            if not p.is_absolute():
                p = self.output_dir / p
            low = p.name.lower()
            if "(vocals)" in low:
                vocals = p
            elif "(instrumental)" in low:
                inst = p
        if vocals is None or inst is None:
            raise RuntimeError(f"Separation did not produce both stems for {wav_path.name}: {outputs}")
        return vocals, inst


def ensure_movie_stems(
    audio: MovieAudio,
    block_indices: Optional[List[int]] = None,
    model_name: str = "Kim_Vocal_2.onnx",
    block_sec: float = DEFAULT_BLOCK_SECONDS,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """
    Separate vocals/instrumental per block (with context padding, trimmed after) and stitch
    finished blocks into full-length FLAC stems. Resumable: finished blocks are skipped.
    Blocks that are not processed are left silent in the stitched stems and recorded in
    stems_meta.json, so renders can refuse to use them.
    """
    stems_dir = audio.root / "stems"
    block_dir = stems_dir / "blocks"
    work_dir = stems_dir / "work"
    block_dir.mkdir(parents=True, exist_ok=True)

    blocks = plan_blocks(audio.duration_sec, block_sec)
    wanted = block_indices if block_indices is not None else list(range(len(blocks)))
    meta_path = stems_dir / "stems_meta.json"
    sr = SEPARATION_SAMPLE_RATE

    def block_files(i: int) -> Tuple[Path, Path]:
        return block_dir / f"block_{i:04d}_vocals.flac", block_dir / f"block_{i:04d}_instrumental.flac"

    todo = [i for i in wanted if not all(p.exists() for p in block_files(i))]
    if todo:
        work_dir.mkdir(parents=True, exist_ok=True)
        sep = _Separator(model_name, work_dir)
        for n, i in enumerate(todo, 1):
            bs, be = blocks[i]
            cs = max(0.0, bs - BLOCK_CONTEXT_SECONDS)
            ce = min(audio.duration_sec, be + BLOCK_CONTEXT_SECONDS)
            t0 = time.time()
            log(f"[*] Vocal separation block {i + 1}/{len(blocks)} ({bs / 60:.0f}-{be / 60:.0f} min) [{n}/{len(todo)} this run]")
            in_wav = work_dir / f"block_{i:04d}_input.wav"
            _atomic_ffmpeg([
                "ffmpeg", "-y", "-v", "error", "-ss", f"{cs:.3f}", "-t", f"{ce - cs:.3f}",
                "-i", str(audio.program_path), "-ac", "2", "-ar", str(sr), "-c:a", "pcm_s16le",
            ], in_wav)
            voc_wav, inst_wav = sep.separate(in_wav)
            trim_start = int(round((bs - cs) * sr))
            n_keep = int(round((be - bs) * sr))
            for src, dst in zip((voc_wav, inst_wav), block_files(i)):
                data, file_sr = sf.read(str(src), dtype="float32", always_2d=True)
                if file_sr != sr:
                    raise RuntimeError(f"Unexpected separator sample rate {file_sr}")
                piece = data[trim_start:trim_start + n_keep]
                if len(piece) < n_keep:
                    piece = np.pad(piece, ((0, n_keep - len(piece)), (0, 0)))
                tmp = dst.with_name(f"{dst.stem}.partial-{os.getpid()}.flac")
                sf.write(str(tmp), piece, sr, subtype="PCM_16", format="FLAC")
                os.replace(tmp, dst)
            for p in (in_wav, voc_wav, inst_wav):
                try:
                    p.unlink()
                except OSError:
                    pass
            log(f"    block done in {time.time() - t0:.0f}s")
        shutil.rmtree(work_dir, ignore_errors=True)

    done = [i for i in range(len(blocks)) if all(p.exists() for p in block_files(i))]
    prev_meta: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            prev_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            prev_meta = {}

    vocals_full = stems_dir / "vocals.flac"
    inst_full = stems_dir / "instrumental.flac"
    needs_stitch = (
        prev_meta.get("blocks_done") != done
        or prev_meta.get("version") != STEMS_VERSION
        or not vocals_full.exists()
        or not inst_full.exists()
    )
    if needs_stitch and done:
        log(f"[*] Stitching {len(done)}/{len(blocks)} separated blocks into full-length stems")
        total = int(round(audio.duration_sec * sr))
        for role, full in (("vocals", vocals_full), ("instrumental", inst_full)):
            tmp = full.with_name(f"{full.stem}.partial-{os.getpid()}.flac")
            with sf.SoundFile(str(tmp), "w", samplerate=sr, channels=2, subtype="PCM_16", format="FLAC") as out:
                written = 0
                for i, (bs, be) in enumerate(blocks):
                    n = min(int(round(be * sr)), total) - written
                    if n <= 0:
                        continue
                    path = block_files(i)[0 if role == "vocals" else 1]
                    if i in done:
                        data, _ = sf.read(str(path), dtype="float32", always_2d=True)
                        if len(data) < n:
                            data = np.pad(data, ((0, n - len(data)), (0, 0)))
                        out.write(data[:n])
                    else:
                        out.write(np.zeros((n, 2), dtype=np.float32))
                    written += n
            os.replace(tmp, full)
        meta = {
            "version": STEMS_VERSION,
            "model": model_name,
            "block_sec": block_sec,
            "sample_rate": sr,
            "blocks": [list(b) for b in blocks],
            "blocks_done": done,
            "complete": len(done) == len(blocks),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        prev_meta = meta

    return {
        "vocals": vocals_full if vocals_full.exists() else None,
        "instrumental": inst_full if inst_full.exists() else None,
        "blocks": blocks,
        "blocks_done": done,
        "complete": bool(done) and len(done) == len(blocks),
        "meta": prev_meta,
    }


def separated_ranges(stems_meta: Dict[str, Any]) -> List[Tuple[float, float]]:
    blocks = stems_meta.get("blocks") or []
    done = set(stems_meta.get("blocks_done") or [])
    return [tuple(b) for i, b in enumerate(blocks) if i in done]  # type: ignore[misc]


def read_stems_meta(audio: MovieAudio) -> Dict[str, Any]:
    p = audio.root / "stems" / "stems_meta.json"
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def ensure_vocal_analysis(audio: MovieAudio, log: Callable[[str], None] = print) -> Optional[Path]:
    """
    16 kHz mono analysis track from the separated vocal stem (silence in blocks not yet
    separated). All dialogue analysis (alignment, voice checks, join features) uses it for
    separated blocks. Rebuilt whenever the set of separated blocks changes.
    """
    meta = read_stems_meta(audio)
    vocals = audio.root / "stems" / "vocals.flac"
    done = meta.get("blocks_done") or []
    if not done or not vocals.exists():
        return None
    out = audio.root / "audio" / "analysis_vocals_16k.flac"
    marker = out.with_name(out.name + ".json")
    want = {"blocks_done": done, "stems_version": meta.get("version"), "model": meta.get("model")}
    if out.exists() and marker.exists():
        try:
            if json.loads(marker.read_text(encoding="utf-8")) == want:
                return out
        except Exception:
            pass
    log(f"[*] Building 16 kHz analysis track from separated vocals ({len(done)} blocks)")
    if out.exists():
        out.unlink()
    _atomic_ffmpeg([
        "ffmpeg", "-y", "-v", "error", "-i", str(vocals), "-ac", "1", "-ar", "16000",
        "-sample_fmt", "s16", "-c:a", "flac",
    ], out)
    if out.exists():
        marker.write_text(json.dumps(want), encoding="utf-8")
    return out


# ---------------------------------------------------------------------------
# Embedded subtitles (M02)
# ---------------------------------------------------------------------------


def list_subtitle_streams(probe: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    position = 0
    for s in probe.get("streams", []):
        if s.get("codec_type") != "subtitle":
            continue
        tags = {str(k).lower(): v for k, v in (s.get("tags") or {}).items()}
        disp = s.get("disposition") or {}
        codec = str(s.get("codec_name") or "").lower()
        title = str(tags.get("title") or "")
        out.append({
            "position": position,
            "index": int(s.get("index", 0)),
            "codec": codec,
            "language": str(tags.get("language") or "").lower(),
            "title": title,
            "text_based": codec in TEXT_SUBTITLE_CODECS,
            "image_based": codec in IMAGE_SUBTITLE_CODECS,
            "sdh": "sdh" in title.lower() or bool(disp.get("hearing_impaired")),
            "forced": bool(disp.get("forced")) or "forced" in title.lower(),
        })
        position += 1
    return out


def select_subtitle_stream(streams: List[Dict[str, Any]], choice: str = "auto") -> Optional[Dict[str, Any]]:
    """auto: English SDH text track, then English non-forced text, then any text track."""
    text = [s for s in streams if s["text_based"]]
    if not text:
        return None
    choice = (choice or "auto").lower()
    if choice not in {"auto", "sdh", "plain"}:
        try:
            idx = int(choice)
        except ValueError:
            raise ValueError(f"Invalid subtitle stream choice: {choice}")
        for s in streams:
            if s["index"] == idx:
                if not s["text_based"]:
                    raise ValueError(f"Subtitle stream {idx} is image-based ({s['codec']}); OCR is not supported")
                return s
        raise ValueError(f"No subtitle stream with index {idx}")

    def is_en(s: Dict[str, Any]) -> bool:
        return s["language"] in {"eng", "en", ""} or "english" in s["title"].lower()

    en = [s for s in text if is_en(s) and not s["forced"]]
    if choice == "sdh":
        return next((s for s in en if s["sdh"]), None)
    if choice == "plain":
        return next((s for s in en if not s["sdh"]), None)
    return next((s for s in en if s["sdh"]), None) or (en[0] if en else text[0])


def extract_subtitle_stream(
    video_path: Union[str, Path],
    stream: Dict[str, Any],
    media_hash: str,
    build_dir: Union[str, Path],
) -> Path:
    out_dir = Path(build_dir).resolve() / "subtitles" / media_hash[:16]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"stream_{stream['index']}.srt"
    if not out.exists() or out.stat().st_size == 0:
        _atomic_ffmpeg([
            "ffmpeg", "-y", "-v", "error", "-i", str(Path(video_path).resolve()),
            "-map", f"0:{stream['index']}", "-c:s", "srt", "-f", "srt",
        ], out)
    return out


def labeled_subtitle_candidates(video_path: Union[str, Path], build_dir: Union[str, Path]) -> List[Path]:
    """Labeled VTTs produced by tools/label_subtitles.py for this movie (by file stem)."""
    stem = Path(video_path).stem
    base = Path(build_dir).resolve() / "subtitles" / "labeled"
    return [p for p in (base / f"{stem}.en.vtt", base / f"{stem}.vtt") if p.exists()]
