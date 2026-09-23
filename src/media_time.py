"""
Media probing, original-rate PCM extraction, and exact rational time mapping.
Implements media contracts and avoids floating-point timing drift.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional, Union

from contracts import RationalTime, SourceRecord


def compute_file_sha256(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def probe_media_file(file_path: Union[str, Path]) -> Dict[str, Any]:
    """Run ffprobe to extract stream information."""
    p = str(Path(file_path).resolve())
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_streams",
        "-show_format",
        "-of", "json",
        p,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def extract_source_record(
    video_path: Union[str, Path],
    build_dir: Union[str, Path] = ".build",
    content_id: Optional[str] = None,
    recording_id: str = "",
    vocal_stem_path: Optional[Union[str, Path]] = None,
    isolate_vocals: bool = False,
) -> SourceRecord:
    """
    Probes media, extracts native sample rate, channels, rational FPS,
    and stages decoded PCM audio under .build/sources/<content_id>/.
    Supports using isolated vocal stem for cached_pcm_path when isolate_vocals
    is active or when cached vocals exist under .build/vocals/<source_id>/vocals.wav.
    """
    v_path = Path(video_path).resolve()
    if not v_path.exists():
        raise FileNotFoundError(f"Source video not found: {v_path}")

    if not content_id:
        file_hash = compute_file_sha256(v_path)[:16]
        content_id = f"{v_path.stem}_{file_hash}"

    probe_data = probe_media_file(v_path)
    streams = probe_data.get("streams", [])
    format_info = probe_data.get("format", {})

    audio_stream = None
    video_stream = None
    for s in streams:
        if s.get("codec_type") == "audio" and audio_stream is None:
            audio_stream = s
        elif s.get("codec_type") == "video" and video_stream is None:
            video_stream = s

    if audio_stream is None:
        raise ValueError(f"No audio stream found in {v_path}")

    native_sample_rate = int(audio_stream.get("sample_rate", 48000))
    native_channels = int(audio_stream.get("channels", 2))

    # Determine rational frame rate
    if video_stream:
        r_fps_str = video_stream.get("r_frame_rate", "30000/1001")
        fps = RationalTime.from_str(r_fps_str)
        video_duration = float(video_stream.get("duration", format_info.get("duration", 0.0)))
    else:
        fps = RationalTime(30000, 1001)
        video_duration = 0.0

    audio_duration = float(audio_stream.get("duration", format_info.get("duration", 0.0)))
    audio_origin_sec = float(audio_stream.get("start_time", 0.0))

    # Stage decoded audio
    sources_dir = Path(build_dir) / "sources" / content_id
    sources_dir.mkdir(parents=True, exist_ok=True)

    original_pcm = sources_dir / "original.wav"
    analysis_pcm = sources_dir / "analysis.wav"

    # Resolve isolated vocal stem if available
    vocal_stem: Optional[Path] = None
    if vocal_stem_path and Path(vocal_stem_path).exists() and Path(vocal_stem_path).stat().st_size > 0:
        vocal_stem = Path(vocal_stem_path).resolve()
    elif isolate_vocals:
        from vocal_isolation import isolate_vocals as run_isolate_vocals
        vocals_base = Path(build_dir) / "vocals"
        vocal_stem = run_isolate_vocals(v_path, output_dir=vocals_base)
    else:
        vocals_base = Path(build_dir) / "vocals"
        if vocals_base.exists():
            p1 = vocals_base / content_id / "vocals.wav"
            if p1.exists() and p1.stat().st_size > 0:
                vocal_stem = p1.resolve()
            else:
                clean_stem = re.sub(r"[^\w\-_]", "_", v_path.stem)[:40]
                path_hash = hashlib.sha256(v_path.as_posix().encode("utf-8")).hexdigest()[:8]
                p2 = vocals_base / f"{clean_stem}_{path_hash}" / "vocals.wav"
                if p2.exists() and p2.stat().st_size > 0:
                    vocal_stem = p2.resolve()
                else:
                    matches = [m for m in vocals_base.glob(f"{clean_stem}*/vocals.wav") if m.stat().st_size > 0]
                    if matches:
                        vocal_stem = matches[0].resolve()

    # Resolve instrumental stem if available
    inst_stem: Optional[Path] = None
    if vocal_stem:
        vocal_dir = vocal_stem.parent
        inst_direct = vocal_dir / "instrumental.wav"
        if inst_direct.exists() and inst_direct.stat().st_size > 0:
            inst_stem = inst_direct.resolve()
        else:
            inst_matches = [
                m for m in vocal_dir.glob("*")
                if m.is_file() and ("instrumental" in m.name.lower() or "no_vocals" in m.name.lower() or "other" in m.name.lower())
                and m.stat().st_size > 0
            ]
            if inst_matches:
                inst_stem = inst_matches[0].resolve()

    # Decode original rate PCM if not already present
    if not original_pcm.exists() or original_pcm.stat().st_size == 0:
        audio_stream_idx = int(audio_stream.get("index", 0))
        cmd_orig = [
            "ffmpeg",
            "-y",
            "-i", str(v_path),
            "-map", f"0:{audio_stream_idx}",
            "-vn",
            "-c:a", "pcm_s16le",
            "-ar", str(native_sample_rate),
            str(original_pcm),
        ]
        subprocess.run(cmd_orig, capture_output=True, check=True)

    # 16kHz mono analysis audio for MFA & feature extraction
    if not analysis_pcm.exists() or analysis_pcm.stat().st_size == 0:
        analysis_input = str(vocal_stem) if vocal_stem else str(v_path)
        cmd_analysis = [
            "ffmpeg",
            "-y",
            "-i", analysis_input,
        ]
        if not vocal_stem:
            cmd_analysis.extend(["-map", f"0:{int(audio_stream.get('index', 0))}"])
        cmd_analysis.extend([
            "-vn",
            "-c:a", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            str(analysis_pcm),
        ])
        subprocess.run(cmd_analysis, capture_output=True, check=True)


    rec = SourceRecord(
        content_id=content_id,
        source_path=str(v_path),
        native_sample_rate=native_sample_rate,
        native_channels=native_channels,
        selected_audio_stream=int(audio_stream.get("index", 0)),
        selected_video_stream=int(video_stream.get("index", 0)) if video_stream else 0,
        decoded_audio_origin_sec=audio_origin_sec,
        video_fps=fps.to_dict(),
        video_duration_sec=video_duration,
        audio_duration_sec=audio_duration,
        recording_id=recording_id,
        cached_pcm_path=str(original_pcm),
        cached_vocal_pcm_path=str(vocal_stem) if vocal_stem else "",
        cached_instrumental_pcm_path=str(inst_stem) if inst_stem else "",
        cached_analysis_pcm_path=str(analysis_pcm),
    )
    rec.validate()

    # Save media.json
    media_json = sources_dir / "media.json"
    with open(media_json, "w", encoding="utf-8") as f:
        json.dump(asdict(rec), f, indent=2)

    return rec


def samples_to_seconds(samples: int, sample_rate: int) -> float:
    return float(samples) / float(sample_rate)


def seconds_to_samples(seconds: float, sample_rate: int) -> int:
    return int(round(seconds * sample_rate))


def samples_to_rational_frames(samples: int, sample_rate: int, fps: RationalTime) -> RationalTime:
    # frames = (samples / sample_rate) * (fps_num / fps_den)
    frac = Fraction(samples * fps.numerator, sample_rate * fps.denominator)
    return RationalTime(frac.numerator, frac.denominator)


def samples_to_frame_index(samples: int, sample_rate: int, fps: RationalTime) -> int:
    # Cumulative frame index calculation
    num = samples * fps.numerator
    den = sample_rate * fps.denominator
    return num // den


def frame_index_to_samples(frame_index: int, fps: RationalTime, sample_rate: int) -> int:
    # samples = frame_index * (fps_den / fps_num) * sample_rate
    num = frame_index * fps.denominator * sample_rate
    den = fps.numerator
    return (num + den // 2) // den


def cumulative_frame_ceiling(n_samples: int, sample_rate: int, fps: RationalTime) -> int:
    """Ceiling quantizer Q(n) = (n*p + S*q - 1) // (S*q).

    Maps a cumulative sample offset to a cumulative frame index using ceiling
    division, giving drift-free frame counts when differences are taken:
        frame_count = Q(end_sample) - Q(start_sample)

    Allows 0-frame clips (e.g. for very short pauses) — callers must NOT apply
    max(1, ...) on the result.

    Args:
        n_samples:   Cumulative sample offset from the start of the timeline.
        sample_rate: Project sample rate (S).
        fps:         Output video frame rate (p/q as RationalTime).

    Returns:
        Cumulative frame index (ceiling).
    """
    if n_samples <= 0:
        return 0
    p = fps.numerator
    q = fps.denominator
    S = sample_rate
    return (n_samples * p + S * q - 1) // (S * q)
