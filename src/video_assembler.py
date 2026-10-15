#!/usr/bin/env python3
"""
Video & Audio Assembler for Sentence Mixing
Renders PCM master audio first, schedules frame-accurate CFR video slices (-crf 18),
and multiplexes delivery MP4 with single-pass audio encoding from the master WAV.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import soundfile as sf
import librosa

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from contracts import EditPlan, PlanSegment, RationalTime

COLOR_TAGS = ["-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709"]
OUTPUT_RESOLUTIONS = {"1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160), "2160p": (3840, 2160)}


def resolve_output_dims(plan: Any, output_resolution: str = "1080p") -> Tuple[int, int]:
    """Returns (width, height). 'source' uses the first video source's size (even dims)."""
    key = (output_resolution or "1080p").lower()
    if key in OUTPUT_RESOLUTIONS:
        return OUTPUT_RESOLUTIONS[key]
    if key == "source":
        for seg in getattr(plan, "segments", []):
            path = getattr(seg, "source_video_path", "")
            if not path or not Path(path).exists():
                continue
            try:
                res = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                     "stream=width,height", "-of", "csv=p=0", str(path)],
                    capture_output=True, text=True, check=True,
                )
                w, h = (int(x) for x in res.stdout.strip().split(",")[:2])
                return w - (w % 2), h - (h % 2)
            except Exception:
                continue
        return OUTPUT_RESOLUTIONS["1080p"]
    m = re.match(r"^(\d+)x(\d+)$", key)
    if m:
        return int(m.group(1)), int(m.group(2))
    raise ValueError(f"Unknown output resolution: {output_resolution}")


class LazyAudio:
    """
    Seekable view of an audio file (WAV/FLAC) that reads only the requested sample
    range. Supports len(), ndim, shape and [start:stop] slicing like a numpy array, so
    long sources (whole movies and their stems) are never decoded into memory.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = str(path)
        info = sf.info(self.path)
        self.samplerate = int(info.samplerate)
        self.frames = int(info.frames)
        self.channels = int(info.channels)

    def __len__(self) -> int:
        return self.frames

    @property
    def ndim(self) -> int:
        return 2 if self.channels > 1 else 1

    @property
    def shape(self) -> Tuple[int, ...]:
        return (self.frames, self.channels) if self.channels > 1 else (self.frames,)

    def __getitem__(self, key: slice) -> np.ndarray:
        if not isinstance(key, slice) or key.step not in (None, 1):
            raise TypeError("LazyAudio supports contiguous slices only")
        start, stop, _ = key.indices(self.frames)
        if stop <= start:
            return np.zeros((0, self.channels) if self.channels > 1 else (0,), dtype=np.float32)
        data, _ = sf.read(self.path, start=start, stop=stop, dtype="float32", always_2d=self.channels > 1)
        return data


def _apply_segment_fades(slice_audio: np.ndarray, seg: Any) -> np.ndarray:
    """Applies the segment's own fade-in/out ramps (project-rate samples), e.g. the end of a
    word release tail or a continued pause, so it never stops on a click."""
    n = len(slice_audio)
    fi = min(int(getattr(seg, "fade_in_samples", 0) or 0), n)
    fo = min(int(getattr(seg, "fade_out_samples", 0) or 0), n)
    if fi <= 0 and fo <= 0:
        return slice_audio
    out = np.array(slice_audio, dtype=np.float32, copy=True)
    if fi > 0:
        ramp = np.linspace(0.0, 1.0, fi, dtype=np.float32)
        out[:fi] *= ramp[:, np.newaxis] if out.ndim > 1 else ramp
    if fo > 0:
        ramp = np.linspace(1.0, 0.0, fo, dtype=np.float32)
        out[n - fo:] *= ramp[:, np.newaxis] if out.ndim > 1 else ramp
    return out


def _open_source_audio(path: Union[str, Path]) -> Tuple[Any, int]:
    """Opens an audio file for ranged reads (falls back to a full read if unsupported)."""
    try:
        lazy = LazyAudio(path)
        return lazy, lazy.samplerate
    except Exception:
        arr, rate = sf.read(str(path))
        return arr, rate


def to_file_uri(p: Union[str, Path, None]) -> str:
    """Format path as clickable file:/// URL."""
    if not p:
        return ""
    path = Path(p).resolve()
    try:
        return path.as_uri()
    except Exception:
        as_posix = path.as_posix()
        if not as_posix.startswith("/"):
            as_posix = "/" + as_posix
        return f"file://{as_posix}"


def normalize_audio_ebur128(
    input_wav: Union[str, Path],
    output_wav: Union[str, Path],
    target_i: float = -16.0,
    target_tp: float = -1.0,
    target_lra: float = 11.0,
    sample_rate: int = 48000,
    premeasured_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Applies two-pass FFmpeg loudnorm filter according to EBU R128 / ITU-R BS.1770
    while preserving the exact sample count with -ar <sample_rate>.
    Guards against pure silence without error.
    """
    in_p = Path(input_wav).resolve()
    out_p = Path(output_wav).resolve()
    if not in_p.exists():
        raise FileNotFoundError(f"Input WAV not found: {in_p}")
    out_p.parent.mkdir(parents=True, exist_ok=True)

    info_in = sf.info(str(in_p))
    total_samples = int(round(info_in.frames * (sample_rate / info_in.samplerate)))
    data, _ = sf.read(str(in_p), dtype="float32")
    max_peak = float(np.max(np.abs(data))) if len(data) > 0 else 0.0

    # Guard against pure silence (max_peak < 1e-4)
    if max_peak < 1e-4 or total_samples == 0:
        if in_p != out_p:
            if len(data) == 0:
                sf.write(str(out_p), np.zeros(total_samples, dtype=np.float32), sample_rate, subtype="PCM_16")
            else:
                sf.write(str(out_p), data, sample_rate, subtype="PCM_16")
        return {
            "input_i": -70.0,
            "input_tp": -70.0,
            "input_lra": 0.0,
            "input_thresh": -70.0,
            "output_i": -70.0,
            "output_tp": -70.0,
            "output_lra": 0.0,
            "output_thresh": -70.0,
            "normalization_type": "passthrough",
            "target_offset": 0.0,
            "sample_diff": 0,
            "silence": True,
            "target_i": target_i,
            "target_tp": target_tp,
        }

    if premeasured_stats is not None and "input_i" in premeasured_stats:
        stats = dict(premeasured_stats)
    else:
        # Pass 1: Measure loudness metrics with loudnorm
        cmd1 = [
            "ffmpeg", "-nostats",
            "-i", str(in_p),
            "-af", f"loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA={target_lra:.1f}:print_format=json",
            "-f", "null", "-"
        ]
        proc1 = subprocess.run(cmd1, capture_output=True, text=True)
        if proc1.returncode != 0:
            raise RuntimeError(f"FFmpeg loudnorm pass 1 failed (code {proc1.returncode}): {proc1.stderr}")

        match = re.search(r"\{\s*\"input_i\"\s*:.*?\}", proc1.stderr, re.DOTALL)
        if not match:
            raise RuntimeError(f"FFmpeg loudnorm pass 1 did not output JSON metrics: {proc1.stderr}")

        stats = json.loads(match.group(0))
    measured_i = stats.get("input_i", "-70.0")
    if measured_i == "-inf" or float(measured_i) <= -70.0:
        if in_p != out_p:
            sf.write(str(out_p), data, sample_rate, subtype="PCM_16")
        stats["sample_diff"] = 0
        stats["silence"] = True
        return stats

    # Pass 2: Linear loudnorm pass with sample rate preservation (-ar <sample_rate>)
    temp_out = out_p.with_suffix(".norm_tmp.wav") if in_p == out_p else out_p
    af_pass2 = (
        f"loudnorm=I={target_i:.1f}:TP={target_tp:.1f}:LRA={target_lra:.1f}:"
        f"measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
        f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:"
        f"offset={stats['target_offset']}:linear=true"
    )
    cmd2 = [
        "ffmpeg", "-y", "-nostats",
        "-i", str(in_p),
        "-af", af_pass2,
        "-ar", str(sample_rate),
        "-c:a", "pcm_s16le",
        str(temp_out)
    ]
    proc2 = subprocess.run(cmd2, capture_output=True, text=True)
    if proc2.returncode != 0:
        raise RuntimeError(f"FFmpeg loudnorm pass 2 failed (code {proc2.returncode}): {proc2.stderr}")

    # Ensure EXACT sample count invariant (diff == 0)
    out_info = sf.info(str(temp_out))
    if out_info.frames != total_samples:
        out_data, _ = sf.read(str(temp_out), dtype="float32")
        pad_len = total_samples - len(out_data)
        if pad_len > 0:
            pad_width: Any
            if out_data.ndim > 1:
                pad_width = ((0, pad_len),) + tuple((0, 0) for _ in range(out_data.ndim - 1))
            else:
                pad_width = (0, pad_len)
            out_data = np.pad(out_data, pad_width)
        elif len(out_data) > total_samples:
            out_data = out_data[:total_samples]
        sf.write(str(temp_out), out_data, sample_rate, subtype="PCM_16")

    if temp_out != out_p:
        shutil.move(str(temp_out), str(out_p))

    final_info = sf.info(str(out_p))
    sample_diff = final_info.frames - total_samples
    if sample_diff != 0:
        raise RuntimeError(
            f"Sample count mismatch after normalization: expected {total_samples}, got {final_info.frames}"
        )

    stats["sample_diff"] = sample_diff
    stats["total_samples"] = final_info.frames
    stats["output_i_target"] = target_i
    return stats


def render_master_pcm(
    plan: EditPlan,
    output_wav_path: Union[str, Path],
    cache_base_dir: Optional[Union[str, Path]] = None,
    normalize_loudness: Optional[bool] = None,
    audio_track: str = "combined",
) -> Path:
    """
    Renders the lossless PCM master WAV at project_sample_rate directly from source PCM.
    Applies Tier 1 per-segment speech leveling, scheduled crossfades,
    optional Tier 2 master EBU R128 loudness normalization, and verifies exact sample count.
    Supports audio_track: 'combined' (default, original full audio) or 'isolated' (vocal stem).
    """
    out_wav = Path(output_wav_path).resolve()
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sr = plan.project_sample_rate

    if normalize_loudness is None:
        normalize_loudness = bool(getattr(plan, "metadata", {}).get("normalize_loudness", False))

    # Determine channel layout (preserve 2-channel stereo without averaging to mono)
    is_stereo = False
    for s in plan.sources.values():
        if getattr(s, "native_channels", 1) >= 2:
            is_stereo = True
            break
    if not is_stereo:
        for seg in plan.segments:
            if seg.is_silence:
                continue
            src_p = Path(seg.source_video_path)
            if src_p.exists() and src_p.suffix.lower() in (".wav", ".flac", ".ogg"):
                try:
                    if sf.info(str(src_p)).channels >= 2:
                        is_stereo = True
                        break
                except Exception:
                    pass

    total_samples = plan.total_output_samples
    master_audio: Any
    if is_stereo:
        master_audio = np.zeros((total_samples, 2), dtype=np.float32)
    else:
        master_audio = np.zeros(total_samples, dtype=np.float32)

    # Cache of decoded source audio arrays
    decoded_sources: Dict[str, Tuple[np.ndarray, int]] = {}
    segment_slices: Dict[int, np.ndarray] = {}

    for seg in plan.segments:
        if seg.is_silence:
            continue

        src_rec = plan.sources.get(seg.source_content_id)
        src_path = src_rec.cached_pcm_path if src_rec and src_rec.cached_pcm_path else None
        src_key = seg.source_content_id
        if audio_track in ("isolated", "vocals"):
            vocal_path = getattr(src_rec, "cached_vocal_pcm_path", "") if src_rec else ""
            if not vocal_path or not Path(vocal_path).exists():
                content_id = getattr(src_rec, "content_id", seg.source_content_id)
                base = Path(cache_base_dir) if cache_base_dir else Path(".build")
                cand = base / "vocals" / content_id / "vocals.wav"
                if cand.exists():
                    vocal_path = str(cand)
                else:
                    clean_prefix = re.sub(r"[^\w\-_]", "_", content_id.split("_")[0])[:30]
                    v_base = base / "vocals"
                    if v_base.exists():
                        matches = [m for m in v_base.glob(f"{clean_prefix}*/vocals.wav") if m.stat().st_size > 0]
                        if matches:
                            vocal_path = str(matches[0])
            if vocal_path and Path(vocal_path).exists():
                src_path = vocal_path
                src_key = f"{seg.source_content_id}_vocals"
            else:
                raise RuntimeError(
                    f"Cannot render isolated audio track: No isolated vocal stem found for source '{seg.source_content_id}'. "
                    f"Please run with --isolate-vocals or verify .build/vocals/{seg.source_content_id}/vocals.wav exists."
                )
        elif audio_track == "recombined":
            # Recombined: Level/normalize vocals ONLY, then add untouched background instrumental stem
            vocal_path = getattr(src_rec, "cached_vocal_pcm_path", "") if src_rec else ""
            inst_path = getattr(src_rec, "cached_instrumental_pcm_path", "") if src_rec else ""
            content_id = getattr(src_rec, "content_id", seg.source_content_id)
            base = Path(cache_base_dir) if cache_base_dir else Path(".build")
            clean_prefix = re.sub(r"[^\w\-_]", "_", content_id.split("_")[0])[:30]

            if not vocal_path or not Path(vocal_path).exists():
                cand = base / "vocals" / content_id / "vocals.wav"
                if cand.exists():
                    vocal_path = str(cand)
                else:
                    v_base = base / "vocals"
                    if v_base.exists():
                        matches = [m for m in v_base.glob(f"{clean_prefix}*/vocals.wav") if m.stat().st_size > 0]
                        if matches:
                            vocal_path = str(matches[0])

            if not inst_path or not Path(inst_path).exists():
                cand_inst = base / "vocals" / content_id / "instrumental.wav"
                if cand_inst.exists():
                    inst_path = str(cand_inst)
                else:
                    v_base = base / "vocals"
                    if v_base.exists():
                        inst_matches = [
                            m for m in v_base.glob(f"{clean_prefix}*/*")
                            if m.is_file() and ("instrumental" in m.name.lower() or "no_vocals" in m.name.lower() or "other" in m.name.lower())
                            and m.stat().st_size > 0
                        ]
                        if inst_matches:
                            inst_path = str(inst_matches[0])

            if vocal_path and Path(vocal_path).exists() and inst_path and Path(inst_path).exists():
                # Load vocal stem
                v_key = f"{seg.source_content_id}_vocals"
                if v_key not in decoded_sources:
                    v_arr, v_rate = _open_source_audio(vocal_path)
                    decoded_sources[v_key] = (v_arr, v_rate)
                vocal_arr, v_rate = decoded_sources[v_key]

                # Load instrumental stem
                i_key = f"{seg.source_content_id}_instrumental"
                if i_key not in decoded_sources:
                    i_arr, i_rate = _open_source_audio(inst_path)
                    decoded_sources[i_key] = (i_arr, i_rate)
                inst_arr, i_rate = decoded_sources[i_key]

                if not is_stereo and ((vocal_arr.ndim > 1 and vocal_arr.shape[1] >= 2) or (inst_arr.ndim > 1 and inst_arr.shape[1] >= 2)):
                    is_stereo = True
                    master_audio = np.column_stack([master_audio, master_audio])

                rec_sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else v_rate
                if v_rate != rec_sr:
                    v_start = int(round(seg.source_start_sample * v_rate / rec_sr))
                    v_end = int(round(seg.source_end_sample * v_rate / rec_sr))
                else:
                    v_start = seg.source_start_sample
                    v_end = seg.source_end_sample

                if v_start < 0 or v_end > len(vocal_arr) or v_end < v_start:
                    raise ValueError(
                        f"Invalid source sample range [{seg.source_start_sample}, {seg.source_end_sample}] "
                        f"for vocal stem of length {len(vocal_arr)}"
                    )

                if i_rate != rec_sr:
                    i_start = int(round(seg.source_start_sample * i_rate / rec_sr))
                    i_end = int(round(seg.source_end_sample * i_rate / rec_sr))
                else:
                    i_start = seg.source_start_sample
                    i_end = seg.source_end_sample

                if i_start < 0 or i_end > len(inst_arr) or i_end < i_start:
                    raise ValueError(
                        f"Invalid source sample range [{seg.source_start_sample}, {seg.source_end_sample}] "
                        f"for instrumental stem of length {len(inst_arr)}"
                    )

                vocal_slice = vocal_arr[v_start:v_end].astype(np.float32)
                inst_slice = inst_arr[i_start:i_end].astype(np.float32)

                # Resample both to project sample rate if needed
                if v_rate != sr and len(vocal_slice) > 0:
                    if vocal_slice.ndim > 1:
                        vocal_slice = librosa.resample(vocal_slice, orig_sr=v_rate, target_sr=sr, axis=0)
                    else:
                        vocal_slice = librosa.resample(vocal_slice, orig_sr=v_rate, target_sr=sr)
                if i_rate != sr and len(inst_slice) > 0:
                    if inst_slice.ndim > 1:
                        inst_slice = librosa.resample(inst_slice, orig_sr=i_rate, target_sr=sr, axis=0)
                    else:
                        inst_slice = librosa.resample(inst_slice, orig_sr=i_rate, target_sr=sr)

                expected_dur = seg.duration_samples
                for stem_name, s_arr, stem_sr in [("vocal", vocal_slice, v_rate), ("instrumental", inst_slice, i_rate)]:
                    if len(s_arr) < expected_dur:
                        diff = expected_dur - len(s_arr)
                        pad_shape = ((0, diff), (0, 0)) if s_arr.ndim > 1 else (0, diff)
                        if stem_name == "vocal":
                            vocal_slice = np.pad(vocal_slice, pad_shape, mode="edge")
                        else:
                            inst_slice = np.pad(inst_slice, pad_shape, mode="edge")
                    elif len(s_arr) > expected_dur:
                        if stem_name == "vocal":
                            vocal_slice = vocal_slice[:expected_dur]
                        else:
                            inst_slice = inst_slice[:expected_dur]

                # Level / gain vocal slice ONLY using frozen plan gain_db
                if not seg.is_silence:
                    if seg.gain_db is None:
                        raise ValueError(
                            f"Undecided gain_db (None) on segment {seg.segment_index} ('{seg.target_text}'). "
                            "All non-silence segments must have a frozen float gain_db in the EditPlan before rendering."
                        )
                    v_gain_db = seg.gain_db
                else:
                    v_gain_db = 0.0

                if v_gain_db != 0.0:
                    vocal_slice = vocal_slice * float(10.0 ** (v_gain_db / 20.0))

                # Channel alignment between vocal and instrumental
                if vocal_slice.ndim != inst_slice.ndim:
                    if vocal_slice.ndim == 1 and inst_slice.ndim == 2:
                        vocal_slice = np.column_stack([vocal_slice, vocal_slice])
                    elif vocal_slice.ndim == 2 and inst_slice.ndim == 1:
                        inst_slice = np.column_stack([inst_slice, inst_slice])

                # Sum normalized vocal slice + original background instrumental slice
                slice_audio = vocal_slice + inst_slice

                if is_stereo and slice_audio.ndim == 1:
                    slice_audio = np.column_stack([slice_audio, slice_audio])
                elif not is_stereo and slice_audio.ndim > 1:
                    slice_audio = np.mean(slice_audio, axis=1)

                slice_audio = _apply_segment_fades(slice_audio, seg)

                # Splice into master buffer
                out_start = seg.output_start_sample
                out_end = min(total_samples, out_start + expected_dur)
                actual_len = out_end - out_start
                if actual_len > 0:
                    segment_slices[seg.segment_index] = slice_audio[:actual_len]
                    master_audio[out_start:out_end] += slice_audio[:actual_len]
                continue

        if not src_path or not Path(src_path).exists():
            # Fallback to source video extraction if cached PCM not found
            src_video = seg.source_video_path
            if src_key not in decoded_sources:
                extract_sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else sr
                extract_ch = src_rec.native_channels if src_rec and src_rec.native_channels > 0 else (2 if is_stereo else 1)
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(src_video),
                    "-vn",
                    "-c:a", "pcm_s16le",
                    "-ar", str(extract_sr),
                    "-ac", str(extract_ch),
                    "-f", "wav",
                    "-",
                ]
                proc = subprocess.run(cmd, capture_output=True, check=True)
                import io
                audio_arr, native_rate = sf.read(io.BytesIO(proc.stdout))
                decoded_sources[src_key] = (audio_arr, native_rate)
            source_arr, native_rate = decoded_sources[src_key]
        else:
            if src_key not in decoded_sources:
                arr, native_rate = _open_source_audio(src_path)
                decoded_sources[src_key] = (arr, native_rate)
            source_arr, native_rate = decoded_sources[src_key]

        if not is_stereo and (source_arr.ndim > 1 and source_arr.shape[1] >= 2):
            is_stereo = True
            master_audio = np.column_stack([master_audio, master_audio])

        # Extract sample slice, scaling sample indices if source array sample rate differs from plan source native rate
        rec_sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else native_rate
        if native_rate != rec_sr:
            start_s = int(round(seg.source_start_sample * native_rate / rec_sr))
            end_s = int(round(seg.source_end_sample * native_rate / rec_sr))
        else:
            start_s = seg.source_start_sample
            end_s = seg.source_end_sample

        if start_s < 0 or end_s > len(source_arr) or end_s < start_s:
            raise ValueError(
                f"Invalid source sample range [{seg.source_start_sample}, {seg.source_end_sample}] "
                f"for source '{seg.source_content_id}' of length {len(source_arr)}"
            )
        slice_audio = source_arr[start_s:end_s].astype(np.float32)

        # Resample to project sample rate if native rate differs
        if native_rate != sr and len(slice_audio) > 0:
            if slice_audio.ndim > 1:
                slice_audio = librosa.resample(slice_audio, orig_sr=native_rate, target_sr=sr, axis=0)
            else:
                slice_audio = librosa.resample(slice_audio, orig_sr=native_rate, target_sr=sr)

        expected_dur = seg.duration_samples
        if len(slice_audio) < expected_dur:
            if native_rate == sr:
                raise ValueError(
                    f"Invalid source slice: length {len(slice_audio)} is less than expected duration {expected_dur}"
                )
            elif expected_dur - len(slice_audio) > 2:
                raise ValueError(
                    f"Resampled slice length {len(slice_audio)} is less than expected duration {expected_dur}"
                )
            else:
                diff = expected_dur - len(slice_audio)
                pad_shape = ((0, diff), (0, 0)) if slice_audio.ndim > 1 else (0, diff)
                slice_audio = np.pad(slice_audio, pad_shape, mode="edge")
        elif len(slice_audio) > expected_dur:
            slice_audio = slice_audio[:expected_dur]

        # Tier 1 Speech Leveling across disparate cuts using frozen plan gain_db
        if not seg.is_silence:
            if seg.gain_db is None:
                raise ValueError(
                    f"Segment {seg.segment_index} ('{seg.target_text}') has undecided gain_db (None). "
                    "All non-silence segments must have a frozen float gain_db in the EditPlan before rendering."
                )
            gain_to_apply_db = seg.gain_db
        else:
            gain_to_apply_db = 0.0

        if gain_to_apply_db != 0.0:
            scale = float(10.0 ** (gain_to_apply_db / 20.0))
            slice_audio = slice_audio * scale

        if is_stereo and slice_audio.ndim == 1:
            slice_audio = np.column_stack([slice_audio, slice_audio])
        elif not is_stereo and slice_audio.ndim > 1:
            slice_audio = np.mean(slice_audio, axis=1)

        slice_audio = _apply_segment_fades(slice_audio, seg)

        # Splice into master buffer
        out_start = seg.output_start_sample
        out_end = min(total_samples, out_start + expected_dur)
        actual_len = out_end - out_start

        if actual_len > 0:
            segment_slices[seg.segment_index] = slice_audio[:actual_len]
            master_audio[out_start:out_end] += slice_audio[:actual_len]

    # Handle transitions: apply crossfades where overlaps exist
    for trans in plan.transitions:
        if trans.transition_type == "overlap" and trans.overlap_samples > 0:
            from_seg = plan.segments[trans.from_segment_index]
            to_seg = plan.segments[trans.to_segment_index]
            fade_start = max(0, to_seg.output_start_sample)
            fade_end = min(total_samples, from_seg.output_end_sample)
            fade_len = fade_end - fade_start
            if fade_len > 0:
                prev_slice = segment_slices.get(trans.from_segment_index)
                next_slice = segment_slices.get(trans.to_segment_index)
                if prev_slice is not None and next_slice is not None:
                    p_tail = prev_slice[-fade_len:]
                    n_head = next_slice[:fade_len]

                    if is_stereo:
                        prev_tail = p_tail if len(p_tail) >= fade_len else np.pad(p_tail, ((fade_len - len(p_tail), 0), (0, 0)))
                        next_head = n_head if len(n_head) >= fade_len else np.pad(n_head, ((0, fade_len - len(n_head)), (0, 0)))
                    else:
                        prev_tail = p_tail if len(p_tail) >= fade_len else np.pad(p_tail, (fade_len - len(p_tail), 0))
                        next_head = n_head if len(n_head) >= fade_len else np.pad(n_head, (0, fade_len - len(n_head)))
                    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                    if is_stereo:
                        fade_out = fade_out[:, np.newaxis]
                        fade_in = fade_in[:, np.newaxis]
                    master_audio[fade_start:fade_end] = (prev_tail * fade_out) + (next_head * fade_in)

    # Normalize audio to prevent clipping if summed overlaps peak (disabled in cuts-only mode)
    is_cuts_only = (getattr(plan, "rendering_mode", "source-derived") == "cuts-only")
    if not is_cuts_only:
        max_peak = float(np.max(np.abs(master_audio))) if len(master_audio) > 0 else 0.0
        if max_peak > 0.98:
            master_audio = (master_audio * (0.95 / max_peak)).astype(np.float32)  # type: ignore[assignment]

    # Write 16-bit PCM WAV
    sf.write(str(out_wav), master_audio, sr, subtype="PCM_16")

    # Tier 2: Master EBU R128 Loudness Normalization (disabled in cuts-only mode)
    if not is_cuts_only and normalize_loudness:
        frozen_stats = getattr(plan, "metadata", {}).get("ebur128_analysis")
        target_meta = getattr(plan, "metadata", {}).get("ebur128_target", {})
        t_i = target_meta.get("target_i", -16.0)
        t_tp = target_meta.get("target_tp", -1.0)
        t_lra = target_meta.get("target_lra", 11.0)
        normalize_audio_ebur128(
            input_wav=out_wav,
            output_wav=out_wav,
            target_i=t_i,
            target_tp=t_tp,
            target_lra=t_lra,
            sample_rate=sr,
            premeasured_stats=frozen_stats,
        )
        # Re-zero silence segments after loudnorm to guarantee pure digital silence without dither bleed
        silence_segs = [seg for seg in plan.segments if seg.is_silence]
        if silence_segs:
            data, out_sr = sf.read(str(out_wav), dtype="float32")
            ramp_len = int(round(0.003 * out_sr))
            for idx_sil, sil_seg in enumerate(silence_segs):
                s_start = max(0, min(len(data), sil_seg.output_start_sample))
                s_end = max(0, min(len(data), sil_seg.output_end_sample))
                if s_end <= s_start:
                    continue  # Skip 0-sample <gap:0ms> markers so speech is not notched
                prev_sil_end = silence_segs[idx_sil - 1].output_end_sample if idx_sil > 0 else 0
                next_sil_start = silence_segs[idx_sil + 1].output_start_sample if idx_sil + 1 < len(silence_segs) else len(data)
                if s_start > 0 and ramp_len > 0:
                    r_start = max(0, prev_sil_end, s_start - ramp_len)
                    l_in = s_start - r_start
                    if l_in > 0:
                        fade = 0.5 * (1.0 + np.cos(np.linspace(0, np.pi, l_in, dtype=np.float32)))
                        if data.ndim > 1:
                            fade = fade[:, np.newaxis]
                        data[r_start:s_start] *= fade
                data[s_start:s_end] = 0.0
                if s_end < len(data) and ramp_len > 0:
                    r_end = min(len(data), next_sil_start, s_end + ramp_len)
                    l_out = r_end - s_end
                    if l_out > 0:
                        fade = 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, l_out, dtype=np.float32)))
                        if data.ndim > 1:
                            fade = fade[:, np.newaxis]
                        data[s_end:r_end] *= fade
            sf.write(str(out_wav), data, out_sr, subtype="PCM_16")

    # Assert exact sample count
    info = sf.info(str(out_wav))
    if info.frames != total_samples:
        raise RuntimeError(
            f"Master WAV sample count mismatch: expected {total_samples}, wrote {info.frames}"
        )

    return out_wav


def _get_media_native_sample_rate(media_path: Union[str, Path], fallback: int = 48000) -> int:
    """Probe media file for native audio sample rate using ffprobe or soundfile."""
    try:
        from media_time import probe_media_file
        data = probe_media_file(media_path)
        for s in data.get("streams", []):
            if s.get("codec_type") == "audio":
                sr_val = int(s.get("sample_rate", 0))
                if sr_val > 0:
                    return sr_val
    except Exception:
        pass
    try:
        info = sf.info(str(media_path))
        if info.samplerate > 0:
            return int(info.samplerate)
    except Exception:
        pass
    return fallback


def _has_video_stream(media_path: Union[str, Path]) -> bool:
    """Check if media file contains at least one video stream."""
    p = Path(media_path)
    if p.suffix.lower() in [".wav", ".mp3", ".flac", ".ogg", ".aac", ".m4a"]:
        return False
    try:
        from media_time import probe_media_file
        data = probe_media_file(media_path)
        return any(s.get("codec_type") == "video" for s in data.get("streams", []))
    except Exception:
        return False


def render_sentence_mix_legacy(
    units: List[Dict[str, Any]],
    output_mp4_path: str,
    target_fps: Optional[Union[float, str, RationalTime]] = None,
    audio_sample_rate: int = 48000,
    keep_temp: bool = False,
    normalize_loudness: bool = True,
    audio_track: str = "combined",
) -> str:
    """
    Renders video and audio from legacy units list.
    Uses source native sample rate rather than hardcoded 48000 when converting seconds to samples.
    """
    if not units:
        raise ValueError("Cannot render empty units list")

    from contracts import RationalTime, SourceRecord

    if isinstance(target_fps, RationalTime):
        fps = target_fps
    elif isinstance(target_fps, (int, float)):
        fps = RationalTime.from_number(target_fps)
    elif isinstance(target_fps, str):
        fps = RationalTime.from_str(target_fps)
    else:
        fps = RationalTime(30000, 1001)

    segments: List[PlanSegment] = []
    sources: Dict[str, SourceRecord] = {}
    native_sr_cache: Dict[str, int] = {}
    curr_sample = 0

    for idx, u in enumerate(units):
        src_path = str(Path(u["source_video"]).resolve())
        content_id = Path(src_path).stem
        if src_path not in native_sr_cache:
            native_sr_cache[src_path] = _get_media_native_sample_rate(src_path, fallback=audio_sample_rate)
        native_sr = native_sr_cache[src_path]

        if content_id not in sources:
            sources[content_id] = SourceRecord(
                content_id=content_id,
                source_path=src_path,
                native_sample_rate=native_sr,
                native_channels=1,
            )

        start_s = float(u["start_sec"])
        end_s = float(u["end_sec"])
        dur_s = max(0.001, end_s - start_s)
        dur_samples = int(round(dur_s * audio_sample_rate))

        p_fps = fps.numerator
        q_fps = fps.denominator
        S_sr = audio_sample_rate

        def Q_frame(n_sample: int) -> int:
            return (n_sample * p_fps + S_sr * q_fps - 1) // (S_sr * q_fps) if n_sample > 0 else 0

        start_frame = Q_frame(curr_sample)
        dur_frames = max(0, Q_frame(curr_sample + dur_samples) - start_frame)

        seg = PlanSegment(
            segment_index=idx,
            source_content_id=content_id,
            speaker_id="speaker",
            source_start_sample=int(round(start_s * native_sr)),
            source_end_sample=int(round(end_s * native_sr)),
            output_start_sample=curr_sample,
            output_end_sample=curr_sample + dur_samples,
            target_text=u.get("label", ""),
            target_phones=[],
            span_type=u.get("type", "word"),
            source_video_path=src_path,
            video_start_frame=start_frame,
            video_frame_count=dur_frames,
            video_source_pts_sec=start_s,
        )
        segments.append(seg)
        curr_sample += dur_samples

    has_video = any(Path(u["source_video"]).exists() and _has_video_stream(u["source_video"]) for u in units)
    metadata = {}
    if not has_video:
        metadata["visual_policy"] = "audio-only"

    total_video_frames = sum(s.video_frame_count for s in segments)
    plan = EditPlan(
        plan_id="legacy_plan",
        target_text="",
        speaker_id="speaker",
        project_sample_rate=audio_sample_rate,
        video_fps=fps,
        total_output_samples=curr_sample,
        total_video_frames=total_video_frames,
        segments=segments,
        sources=sources,
        metadata=metadata,
    )
    from edit_plan import finalize_edit_plan
    finalize_edit_plan(plan, normalize_loudness=normalize_loudness, audio_track=audio_track)
    return render_sentence_mix(
        plan,
        output_mp4_path,
        target_fps=fps,
        audio_sample_rate=audio_sample_rate,
        keep_temp=keep_temp,
        normalize_loudness=normalize_loudness,
        audio_track=audio_track,
    )


def render_sentence_mix(
    plan_or_units: Union[EditPlan, List[Dict[str, Any]]],
    output_mp4_path: str,
    target_fps: Optional[Union[float, str, RationalTime]] = None,
    audio_sample_rate: int = 48000,
    keep_temp: bool = False,
    normalize_loudness: bool = True,
    audio_track: str = "combined",
    output_resolution: Optional[str] = None,
) -> str:
    """
    Renders video and audio according to an EditPlan (or legacy units list).
    Produces authoritative master WAV(s), re-encodes CFR video slices (-crf 18),
    and multiplexes with delivery AAC audio encoded once from the master WAV.
    Supports audio_track: 'combined' (default), 'isolated' (speech stem), or 'both' (renders both MP4s).
    output_resolution: '1080p' (default), '1440p', '4k', 'source', or 'WxH'; letterboxed to fit.
    """
    out_mp4 = Path(output_mp4_path).resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    out_wav = out_mp4.with_suffix(".wav")

    # Delegate legacy unit dictionaries to render_sentence_mix_legacy
    if not isinstance(plan_or_units, EditPlan):
        return render_sentence_mix_legacy(
            plan_or_units,
            output_mp4_path,
            target_fps=target_fps,
            audio_sample_rate=audio_sample_rate,
            keep_temp=keep_temp,
            normalize_loudness=normalize_loudness,
            audio_track=audio_track,
        )

    plan = plan_or_units

    fps = plan.video_fps
    fps_str = f"{fps.numerator}/{fps.denominator}"
    out_w, out_h = resolve_output_dims(
        plan, output_resolution or getattr(plan, "metadata", {}).get("output_resolution") or "1080p"
    )
    scale_pad = (
        f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease:flags=lanczos,"
        f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2,setsar=1"
    )

    # 1. Render authoritative Master PCM WAV(s)
    is_both = audio_track == "both"
    is_all = audio_track == "all"
    out_wav_vocals: Optional[Path] = None
    out_wav_recombined: Optional[Path] = None

    if is_all:
        print(f"[*] Rendering Triple Master PCM WAVs (raw, vocals & recombined): {plan.total_output_samples} samples")
        render_master_pcm(plan, out_wav, normalize_loudness=normalize_loudness, audio_track="raw")
        out_wav_vocals = out_mp4.parent / f"{out_mp4.stem}_vocals.wav"
        render_master_pcm(plan, out_wav_vocals, normalize_loudness=normalize_loudness, audio_track="vocals")
        out_wav_recombined = out_mp4.parent / f"{out_mp4.stem}_recombined.wav"
        render_master_pcm(plan, out_wav_recombined, normalize_loudness=normalize_loudness, audio_track="recombined")
    elif is_both:
        print(f"[*] Rendering Dual Master PCM WAVs (raw & vocals): {plan.total_output_samples} samples")
        render_master_pcm(plan, out_wav, normalize_loudness=normalize_loudness, audio_track="raw")
        out_wav_vocals = out_mp4.parent / f"{out_mp4.stem}_vocals.wav"
        render_master_pcm(plan, out_wav_vocals, normalize_loudness=normalize_loudness, audio_track="vocals")
    elif audio_track in ("isolated", "vocals"):
        print(f"[*] Rendering Isolated Vocal Master PCM WAV: {to_file_uri(out_wav)} ({plan.total_output_samples} samples)")
        render_master_pcm(plan, out_wav, normalize_loudness=normalize_loudness, audio_track="vocals")
    elif audio_track == "recombined":
        print(f"[*] Rendering Recombined (Normalized Vocals + Original Instrumental) Master PCM WAV: {to_file_uri(out_wav)} ({plan.total_output_samples} samples)")
        render_master_pcm(plan, out_wav, normalize_loudness=normalize_loudness, audio_track="recombined")
    else:
        print(f"[*] Rendering Master PCM WAV: {to_file_uri(out_wav)} ({plan.total_output_samples} samples)")
        render_master_pcm(plan, out_wav, normalize_loudness=normalize_loudness, audio_track="raw")

    # 2. Render and re-encode video slices to temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="sm_render_"))
    slice_files: List[Path] = []
    concat_txt_path = temp_dir / "concat.txt"
    temp_concat_video = temp_dir / "concat_video.mp4"

    try:
        batched_success = False
        active_segments = [(idx, seg) for idx, seg in enumerate(plan.segments) if seg.video_frame_count > 0]

        if active_segments:
            try:
                holds_dir = temp_dir / "holds"
                holds_dir.mkdir(parents=True, exist_ok=True)

                BATCH_CHUNK_SIZE = 40
                segment_batches = [
                    active_segments[i : i + BATCH_CHUNK_SIZE]
                    for i in range(0, len(active_segments), BATCH_CHUNK_SIZE)
                ]
                batch_outputs: List[Path] = []
                can_batch = True

                for b_idx, batch_segs in enumerate(segment_batches):
                    input_args: List[str] = []
                    filter_chains: List[str] = []
                    out_labels: List[str] = []
                    current_in_idx = 0

                    for k, (idx, seg) in enumerate(batch_segs):
                        frame_count = seg.video_frame_count
                        dur_sec = frame_count * (fps.denominator / float(fps.numerator))
                        is_freeze = getattr(seg, "freeze_video", False) or seg.is_silence or seg.span_type == "pause"
                        has_valid_source = bool(
                            seg.source_video_path
                            and Path(seg.source_video_path).exists()
                            and _has_video_stream(seg.source_video_path)
                        )

                        if is_freeze:
                            freeze_img = holds_dir / f"hold_{idx:04d}.png"
                            extracted = False
                            if freeze_img.exists() and freeze_img.stat().st_size > 0:
                                extracted = True
                            elif has_valid_source:
                                cmd_extract = [
                                    "ffmpeg", "-y",
                                    "-ss", f"{seg.video_source_pts_sec:.6f}",
                                    "-i", str(seg.source_video_path),
                                    "-frames:v", "1",
                                    "-q:v", "1",
                                    str(freeze_img),
                                ]
                                res = subprocess.run(cmd_extract, capture_output=True)
                                if res.returncode == 0 and freeze_img.exists() and freeze_img.stat().st_size > 0:
                                    extracted = True
                            if not extracted:
                                can_batch = False
                                break
                            input_args.extend(["-loop", "1", "-framerate", fps_str, "-t", f"{dur_sec:.6f}", "-i", str(freeze_img)])
                            filter_chains.append(f"[{current_in_idx}:v]fps={fps_str},{scale_pad},trim=end_frame={frame_count},format=yuv420p[v{k}]")
                            out_labels.append(f"[v{k}]")
                            current_in_idx += 1
                        elif has_valid_source:
                            start_pts = seg.video_source_pts_sec
                            # Dedicated per-cut seek input to prevent memory buffer queue overflow in FFmpeg
                            input_args.extend(["-ss", f"{start_pts:.6f}", "-t", f"{dur_sec + 0.25:.6f}", "-i", str(seg.source_video_path)])
                            filter_chains.append(
                                f"[{current_in_idx}:v]fps={fps_str},trim=end_frame={frame_count},"
                                f"setpts=PTS-STARTPTS,{scale_pad},tpad=stop_mode=clone:stop=-1,trim=end_frame={frame_count},format=yuv420p[v{k}]"
                            )
                            out_labels.append(f"[v{k}]")
                            current_in_idx += 1
                        else:
                            is_audio_only = (
                                not seg.source_video_path
                                or getattr(plan, "metadata", {}).get("visual_policy") == "audio-only"
                            )
                            if is_audio_only:
                                input_args.extend(["-f", "lavfi", "-t", f"{dur_sec:.6f}", "-i", f"color=c=black:s={out_w}x{out_h}:r={fps_str}"])
                                filter_chains.append(f"[{current_in_idx}:v]trim=end_frame={frame_count},format=yuv420p[v{k}]")
                                out_labels.append(f"[v{k}]")
                                current_in_idx += 1
                            else:
                                can_batch = False
                                break

                    if not can_batch or not out_labels:
                        can_batch = False
                        break

                    batch_out = temp_concat_video if len(segment_batches) == 1 else temp_dir / f"batch_{b_idx:04d}.mp4"
                    concat_filter = "".join(out_labels) + f"concat=n={len(out_labels)}:v=1:a=0[outv]"
                    full_filtergraph = ";".join(filter_chains) + ";" + concat_filter
                    cmd_batch = [
                        "ffmpeg", "-y",
                        *input_args,
                        "-filter_complex", full_filtergraph,
                        "-map", "[outv]",
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "18",
                        *COLOR_TAGS,
                        "-r", fps_str,
                        "-an",
                        str(batch_out),
                    ]
                    res_batch = subprocess.run(cmd_batch, capture_output=True, timeout=300)
                    if res_batch.returncode == 0 and batch_out.exists() and batch_out.stat().st_size > 0:
                        batch_outputs.append(batch_out)
                    else:
                        can_batch = False
                        break

                if can_batch and len(batch_outputs) == len(segment_batches):
                    if len(batch_outputs) == 1:
                        batched_success = True
                    else:
                        batch_concat_txt = temp_dir / "batch_concat.txt"
                        with open(batch_concat_txt, "w", encoding="utf-8") as bcf:
                            for b_p in batch_outputs:
                                esc_p = str(b_p).replace("\\", "/")
                                bcf.write(f"file '{esc_p}'\n")
                        cmd_batch_concat = [
                            "ffmpeg", "-y",
                            "-f", "concat",
                            "-safe", "0",
                            "-i", str(batch_concat_txt),
                            "-c", "copy",
                            str(temp_concat_video),
                        ]
                        res_cc = subprocess.run(cmd_batch_concat, capture_output=True)
                        if res_cc.returncode == 0 and temp_concat_video.exists() and temp_concat_video.stat().st_size > 0:
                            batched_success = True
            except Exception:
                batched_success = False

        if not batched_success:
            print(f"[*] Cutting {len(plan.segments)} CFR video slices (-crf 18) to {to_file_uri(temp_dir)}...")
            holds_dir = temp_dir / "holds"
            holds_dir.mkdir(parents=True, exist_ok=True)
            for idx, seg in enumerate(plan.segments):
                slice_mp4 = temp_dir / f"slice_{idx:04d}.mp4"
                frame_count = seg.video_frame_count

                # Skip zero-frame segments (e.g. <gap:0ms> pauses shorter than one video frame)
                if frame_count <= 0:
                    continue

                # Check if this segment is a pause/freeze-frame cut
                is_freeze = getattr(seg, "freeze_video", False) or seg.is_silence or seg.span_type == "pause"
                if is_freeze:
                    freeze_img = holds_dir / f"hold_{idx:04d}.png"
                    extracted = False

                    if freeze_img.exists() and freeze_img.stat().st_size > 0:
                        extracted = True
                    elif seg.source_video_path and Path(seg.source_video_path).exists() and _has_video_stream(seg.source_video_path):
                        cmd_extract = [
                            "ffmpeg", "-y",
                            "-ss", f"{seg.video_source_pts_sec:.6f}",
                            "-i", str(seg.source_video_path),
                            "-frames:v", "1",
                            "-q:v", "1",
                            str(freeze_img),
                        ]
                        res = subprocess.run(cmd_extract, capture_output=True)
                        if res.returncode == 0 and freeze_img.exists() and freeze_img.stat().st_size > 0:
                            extracted = True
                    elif idx > 0 and len(slice_files) > 0:
                        cmd_extract = [
                            "ffmpeg", "-y",
                            "-sseof", "-0.05",
                            "-i", str(slice_files[-1]),
                            "-update", "1",
                            "-q:v", "1",
                            str(freeze_img),
                        ]
                        res = subprocess.run(cmd_extract, capture_output=True)
                        if res.returncode == 0 and freeze_img.exists() and freeze_img.stat().st_size > 0:
                            extracted = True

                    if extracted:
                        cmd = [
                            "ffmpeg", "-y",
                            "-loop", "1",
                            "-framerate", fps_str,
                            "-i", str(freeze_img),
                            "-frames:v", str(frame_count),
                            "-vf", f"fps={fps_str},{scale_pad},format=yuv420p",
                            "-c:v", "libx264",
                            "-preset", "veryfast",
                            "-crf", "18",
                            *COLOR_TAGS,
                            "-r", fps_str,
                            "-an",
                            str(slice_mp4),
                        ]
                        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        slice_files.append(slice_mp4)
                        continue

                has_valid_source = bool(
                    seg.source_video_path
                    and Path(seg.source_video_path).exists()
                    and _has_video_stream(seg.source_video_path)
                )
                if has_valid_source:
                    start_pts = seg.video_source_pts_sec
                    cmd = [
                        "ffmpeg", "-y",
                        "-ss", f"{start_pts:.6f}",
                        "-i", str(seg.source_video_path),
                        "-frames:v", str(frame_count),
                        "-vf", f"fps={fps_str},{scale_pad},tpad=stop_mode=clone:stop=-1,format=yuv420p",
                        "-c:v", "libx264",
                        "-preset", "veryfast",
                        "-crf", "18",
                        *COLOR_TAGS,
                        "-r", fps_str,
                        "-an",
                        str(slice_mp4)
                    ]
                else:
                    is_audio_only = (
                        not seg.source_video_path
                        or getattr(plan, "metadata", {}).get("visual_policy") == "audio-only"
                    )
                    if is_audio_only:
                        # Fallback to color frame if source video does not exist or has no video stream only when explicit audio-only visual policy requested
                        cmd = [
                            "ffmpeg", "-y",
                            "-f", "lavfi",
                            "-i", f"color=c=black:s={out_w}x{out_h}:r={fps_str}",
                            "-frames:v", str(frame_count),
                            "-c:v", "libx264",
                            "-preset", "veryfast",
                            "-crf", "18",
                            *COLOR_TAGS,
                            "-r", fps_str,
                            "-an",
                            str(slice_mp4)
                        ]
                    else:
                        if not Path(seg.source_video_path).exists():
                            raise FileNotFoundError(
                                f"Missing source video for segment {seg.segment_index}: '{seg.source_video_path}'. "
                                "Audio-only visual policy not requested."
                            )
                        else:
                            raise ValueError(
                                f"Unreadable source video (no video stream found) for segment {seg.segment_index}: '{seg.source_video_path}'. "
                                "Audio-only visual policy not requested."
                            )

                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                slice_files.append(slice_mp4)

            # 3. Concatenate video slices
            with open(concat_txt_path, "w", encoding="utf-8") as cf:
                for sf in slice_files:
                    escaped = str(sf).replace("\\", "/")
                    cf.write(f"file '{escaped}'\n")

            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_txt_path),
                "-c", "copy",
                str(temp_concat_video)
            ]
            subprocess.run(concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        # 4. Final multiplexing of concatenated video with master PCM WAV
        print(f"[*] Multiplexing video with master WAV into final delivery MP4: {to_file_uri(out_mp4)}")
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", str(temp_concat_video),
            "-i", str(out_wav),
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            str(out_mp4)
        ]
        subprocess.run(mux_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if (is_both or is_all) and out_wav_vocals is not None:
            out_mp4_vocals = out_mp4.parent / f"{out_mp4.stem}_vocals.mp4"
            print(f"[*] Multiplexing isolated vocal MP4: {to_file_uri(out_mp4_vocals)}")
            mux_vocals_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_concat_video),
                "-i", str(out_wav_vocals),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                str(out_mp4_vocals)
            ]
            subprocess.run(mux_vocals_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if is_all and out_wav_recombined is not None:
            out_mp4_recombined = out_mp4.parent / f"{out_mp4.stem}_recombined.mp4"
            print(f"[*] Multiplexing recombined MP4: {to_file_uri(out_mp4_recombined)}")
            mux_recombined_cmd = [
                "ffmpeg", "-y",
                "-i", str(temp_concat_video),
                "-i", str(out_wav_recombined),
                "-c:v", "copy",
                "-c:a", "aac",
                "-b:a", "192k",
                str(out_mp4_recombined)
            ]
            subprocess.run(mux_recombined_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    finally:
        if not keep_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)

    if is_all:
        print(f"[+] Render complete: {to_file_uri(out_mp4)}, {to_file_uri(out_mp4.parent / f'{out_mp4.stem}_vocals.mp4')}, and {to_file_uri(out_mp4.parent / f'{out_mp4.stem}_recombined.mp4')}")
    elif is_both:
        print(f"[+] Render complete: {to_file_uri(out_mp4)} and {to_file_uri(out_mp4.parent / f'{out_mp4.stem}_vocals.mp4')}")
    else:
        print(f"[+] Render complete: {to_file_uri(out_mp4)}")
    return str(out_mp4)
