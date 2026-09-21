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


def normalize_audio_ebur128(
    input_wav: Union[str, Path],
    output_wav: Union[str, Path],
    target_i: float = -16.0,
    target_tp: float = -1.0,
    target_lra: float = 11.0,
    sample_rate: int = 48000,
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

    stats: Dict[str, Any] = json.loads(match.group(0))
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

    # Master audio buffer initialized to zeros
    total_samples = plan.total_output_samples
    master_audio = np.zeros(total_samples, dtype=np.float32)

    # Cache of decoded source audio arrays
    decoded_sources: Dict[str, Tuple[np.ndarray, int]] = {}
    segment_slices: Dict[int, np.ndarray] = {}

    for seg in plan.segments:
        if seg.is_silence:
            continue

        src_rec = plan.sources.get(seg.source_content_id)
        if audio_track == "isolated":
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

            if not vocal_path or not Path(vocal_path).exists() or not inst_path or not Path(inst_path).exists():
                raise RuntimeError(
                    f"Cannot render recombined audio track: Missing vocal or instrumental stem for source '{seg.source_content_id}'. "
                    f"Vocal: {vocal_path}, Instrumental: {inst_path}. Run with --isolate-vocals to produce stems."
                )

            # Load vocal stem
            v_key = f"{seg.source_content_id}_vocals"
            if v_key not in decoded_sources:
                v_arr, v_rate = sf.read(vocal_path)
                if v_arr.ndim > 1:
                    v_arr = np.mean(v_arr, axis=1)
                decoded_sources[v_key] = (v_arr, v_rate)
            vocal_arr, v_rate = decoded_sources[v_key]

            # Load instrumental stem
            i_key = f"{seg.source_content_id}_instrumental"
            if i_key not in decoded_sources:
                i_arr, i_rate = sf.read(inst_path)
                if i_arr.ndim > 1:
                    i_arr = np.mean(i_arr, axis=1)
                decoded_sources[i_key] = (i_arr, i_rate)
            inst_arr, i_rate = decoded_sources[i_key]

            rec_sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else v_rate
            if v_rate != rec_sr:
                v_start = max(0, int(round(seg.source_start_sample * v_rate / rec_sr)))
                v_end = min(len(vocal_arr), int(round(seg.source_end_sample * v_rate / rec_sr)))
            else:
                v_start = max(0, seg.source_start_sample)
                v_end = min(len(vocal_arr), seg.source_end_sample)
            vocal_slice = vocal_arr[v_start:v_end].astype(np.float32)

            if i_rate != rec_sr:
                i_start = max(0, int(round(seg.source_start_sample * i_rate / rec_sr)))
                i_end = min(len(inst_arr), int(round(seg.source_end_sample * i_rate / rec_sr)))
            else:
                i_start = max(0, seg.source_start_sample)
                i_end = min(len(inst_arr), seg.source_end_sample)
            inst_slice = inst_arr[i_start:i_end].astype(np.float32)

            # Resample both to project sample rate if needed
            if v_rate != sr and len(vocal_slice) > 0:
                vocal_slice = librosa.resample(vocal_slice, orig_sr=v_rate, target_sr=sr)
            if i_rate != sr and len(inst_slice) > 0:
                inst_slice = librosa.resample(inst_slice, orig_sr=i_rate, target_sr=sr)

            expected_dur = seg.duration_samples
            # Pad / trim vocal
            if len(vocal_slice) < expected_dur:
                vocal_slice = np.pad(vocal_slice, (0, expected_dur - len(vocal_slice)))
            elif len(vocal_slice) > expected_dur:
                vocal_slice = vocal_slice[:expected_dur]

            # Pad / trim instrumental
            if len(inst_slice) < expected_dur:
                inst_slice = np.pad(inst_slice, (0, expected_dur - len(inst_slice)))
            elif len(inst_slice) > expected_dur:
                inst_slice = inst_slice[:expected_dur]

            # Level / gain vocal slice ONLY
            v_gain_db = seg.gain_db
            if v_gain_db == 0.0 and normalize_loudness and len(vocal_slice) > 0:
                active_mask = np.abs(vocal_slice) > 1e-4
                if np.sum(active_mask) > 50:
                    rms = float(np.sqrt(np.mean(vocal_slice[active_mask] ** 2)))
                else:
                    rms = float(np.sqrt(np.mean(vocal_slice ** 2)))
                if rms > 1e-4:
                    rms_db = 20.0 * np.log10(rms)
                    auto_gain_db = -20.0 - rms_db
                    v_gain_db = max(-9.0, min(9.0, auto_gain_db))

            if v_gain_db != 0.0:
                vocal_slice = vocal_slice * float(10.0 ** (v_gain_db / 20.0))

            # Sum normalized vocal slice + original background instrumental slice
            slice_audio = vocal_slice + inst_slice

            # Splice into master buffer
            out_start = seg.output_start_sample
            out_end = min(total_samples, out_start + expected_dur)
            actual_len = out_end - out_start
            if actual_len > 0:
                segment_slices[seg.segment_index] = slice_audio[:actual_len]
                master_audio[out_start:out_end] += slice_audio[:actual_len]
            continue
        else:
            src_path = src_rec.cached_pcm_path if src_rec and src_rec.cached_pcm_path else None
            src_key = seg.source_content_id

        if not src_path or not Path(src_path).exists():
            # Fallback to source video extraction if cached PCM not found
            src_video = seg.source_video_path
            if src_key not in decoded_sources:
                extract_sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else sr
                cmd = [
                    "ffmpeg", "-y",
                    "-i", str(src_video),
                    "-vn",
                    "-c:a", "pcm_s16le",
                    "-ar", str(extract_sr),
                    "-ac", "1",
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
                arr, native_rate = sf.read(src_path)
                if arr.ndim > 1:
                    arr = np.mean(arr, axis=1)
                decoded_sources[src_key] = (arr, native_rate)
            source_arr, native_rate = decoded_sources[src_key]

        # Extract sample slice, scaling sample indices if source array sample rate differs from plan source native rate
        rec_sr = src_rec.native_sample_rate if src_rec and src_rec.native_sample_rate > 0 else native_rate
        if native_rate != rec_sr:
            start_s = max(0, int(round(seg.source_start_sample * native_rate / rec_sr)))
            end_s = min(len(source_arr), int(round(seg.source_end_sample * native_rate / rec_sr)))
        else:
            start_s = max(0, seg.source_start_sample)
            end_s = min(len(source_arr), seg.source_end_sample)
        slice_audio = source_arr[start_s:end_s].astype(np.float32)

        # Resample to project sample rate if native rate differs
        if native_rate != sr and len(slice_audio) > 0:
            slice_audio = librosa.resample(slice_audio, orig_sr=native_rate, target_sr=sr)

        expected_dur = seg.duration_samples
        if len(slice_audio) < expected_dur:
            # Pad slightly if boundary truncation occurred
            slice_audio = np.pad(slice_audio, (0, expected_dur - len(slice_audio)))
        elif len(slice_audio) > expected_dur:
            slice_audio = slice_audio[:expected_dur]

        # Tier 1 Speech Leveling across disparate cuts:
        # Use PlanSegment.gain_db if non-zero; otherwise auto-level if normalize_loudness is True.
        gain_to_apply_db = seg.gain_db
        if gain_to_apply_db == 0.0 and normalize_loudness and len(slice_audio) > 0:
            active_mask = np.abs(slice_audio) > 1e-4
            if np.sum(active_mask) > 50:
                rms = float(np.sqrt(np.mean(slice_audio[active_mask] ** 2)))
            else:
                rms = float(np.sqrt(np.mean(slice_audio ** 2)))

            if rms > 1e-4:
                rms_db = 20.0 * np.log10(rms)
                # Nominal speech target ~ -20 dBFS
                auto_gain_db = -20.0 - rms_db
                # Clamp to [-9.0, +9.0] dB to prevent blowing up noise floor or crushing shouting
                gain_to_apply_db = max(-9.0, min(9.0, auto_gain_db))

        if gain_to_apply_db != 0.0:
            scale = float(10.0 ** (gain_to_apply_db / 20.0))
            slice_audio = slice_audio * scale

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
            # Overlap region is at [to_seg.output_start_sample, from_seg.output_end_sample]
            fade_start = max(0, to_seg.output_start_sample)
            fade_end = min(total_samples, from_seg.output_end_sample)
            fade_len = fade_end - fade_start
            if fade_len > 0:
                prev_slice = segment_slices.get(trans.from_segment_index)
                next_slice = segment_slices.get(trans.to_segment_index)
                if prev_slice is not None and next_slice is not None:
                    prev_tail = prev_slice[-fade_len:] if len(prev_slice) >= fade_len else np.pad(prev_slice, (fade_len - len(prev_slice), 0))
                    next_head = next_slice[:fade_len] if len(next_slice) >= fade_len else np.pad(next_slice, (0, fade_len - len(next_slice)))
                    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
                    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
                    master_audio[fade_start:fade_end] = (prev_tail * fade_out) + (next_head * fade_in)

    # Normalize audio to prevent clipping if summed overlaps peak
    max_peak = float(np.max(np.abs(master_audio))) if len(master_audio) > 0 else 0.0
    if max_peak > 0.98:
        master_audio = (master_audio * (0.95 / max_peak)).astype(np.float32)  # type: ignore[assignment]

    # Write 16-bit PCM WAV
    sf.write(str(out_wav), master_audio, sr, subtype="PCM_16")

    # Tier 2: Master EBU R128 Loudness Normalization
    if normalize_loudness:
        normalize_audio_ebur128(
            input_wav=out_wav,
            output_wav=out_wav,
            target_i=-16.0,
            target_tp=-1.0,
            target_lra=11.0,
            sample_rate=sr,
        )

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
        dur_s = max(0.033, end_s - start_s)
        dur_samples = int(round(dur_s * audio_sample_rate))
        start_frame = int(round(curr_sample * fps.numerator / (audio_sample_rate * fps.denominator)))
        dur_frames = max(1, int(round(dur_samples * fps.numerator / (audio_sample_rate * fps.denominator))))

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
    )
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
) -> str:
    """
    Renders video and audio according to an EditPlan (or legacy units list).
    Produces authoritative master WAV(s), re-encodes CFR video slices (-crf 18),
    and multiplexes with delivery AAC audio encoded once from the master WAV.
    Supports audio_track: 'combined' (default), 'isolated' (speech stem), or 'both' (renders both MP4s).
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

    # 1. Render authoritative Master PCM WAV(s)
    is_both = audio_track == "both"
    is_all = audio_track == "all"
    out_wav_vocals: Optional[Path] = None
    out_wav_recombined: Optional[Path] = None

    if is_all:
        print(f"[*] Rendering Triple Master PCM WAVs (combined, isolated & recombined): {plan.total_output_samples} samples")
        render_master_pcm(plan, out_wav, normalize_loudness=normalize_loudness, audio_track="combined")
        out_wav_vocals = out_mp4.parent / f"{out_mp4.stem}_vocals.wav"
        render_master_pcm(plan, out_wav_vocals, normalize_loudness=normalize_loudness, audio_track="isolated")
        out_wav_recombined = out_mp4.parent / f"{out_mp4.stem}_recombined.wav"
        render_master_pcm(plan, out_wav_recombined, normalize_loudness=normalize_loudness, audio_track="recombined")
    elif is_both:
        print(f"[*] Rendering Dual Master PCM WAVs (combined & isolated): {plan.total_output_samples} samples")
        render_master_pcm(plan, out_wav, normalize_loudness=normalize_loudness, audio_track="combined")
        out_wav_vocals = out_mp4.parent / f"{out_mp4.stem}_vocals.wav"
        render_master_pcm(plan, out_wav_vocals, normalize_loudness=normalize_loudness, audio_track="isolated")
    elif audio_track == "isolated":
        print(f"[*] Rendering Isolated Vocal Master PCM WAV: {out_wav} ({plan.total_output_samples} samples)")
        render_master_pcm(plan, out_wav, normalize_loudness=normalize_loudness, audio_track="isolated")
    elif audio_track == "recombined":
        print(f"[*] Rendering Recombined (Normalized Vocals + Original Instrumental) Master PCM WAV: {out_wav} ({plan.total_output_samples} samples)")
        render_master_pcm(plan, out_wav, normalize_loudness=normalize_loudness, audio_track="recombined")
    else:
        print(f"[*] Rendering Master PCM WAV: {out_wav} ({plan.total_output_samples} samples)")
        render_master_pcm(plan, out_wav, normalize_loudness=normalize_loudness, audio_track="combined")

    # 2. Render and re-encode video slices to temporary directory
    temp_dir = Path(tempfile.mkdtemp(prefix="sm_render_"))
    slice_files: List[Path] = []
    concat_txt_path = temp_dir / "concat.txt"

    print(f"[*] Cutting {len(plan.segments)} CFR video slices (-crf 18) to {temp_dir}...")
    try:
        for idx, seg in enumerate(plan.segments):
            slice_mp4 = temp_dir / f"slice_{idx:04d}.mp4"
            float(seg.duration_samples) / float(plan.project_sample_rate)

            frame_count = max(1, seg.video_frame_count)
            if seg.source_video_path and Path(seg.source_video_path).exists() and _has_video_stream(seg.source_video_path):
                start_pts = seg.video_source_pts_sec
                cmd = [
                    "ffmpeg", "-y",
                    "-ss", f"{start_pts:.6f}",
                    "-i", str(seg.source_video_path),
                    "-frames:v", str(frame_count),
                    "-vf", f"fps={fps_str},scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-r", fps_str,
                    "-an",
                    str(slice_mp4)
                ]
            else:
                # Fallback to color frame if source video does not exist or has no video stream
                cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi",
                    "-i", f"color=c=black:s=1920x1080:r={fps_str}",
                    "-frames:v", str(frame_count),
                    "-c:v", "libx264",
                    "-preset", "veryfast",
                    "-crf", "18",
                    "-r", fps_str,
                    "-an",
                    str(slice_mp4)
                ]

            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            slice_files.append(slice_mp4)

        # 3. Concatenate video slices
        with open(concat_txt_path, "w", encoding="utf-8") as cf:
            for sf in slice_files:
                escaped = str(sf).replace("\\", "/")
                cf.write(f"file '{escaped}'\n")

        temp_concat_video = temp_dir / "concat_video.mp4"
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
        print(f"[*] Multiplexing video with master WAV into final delivery MP4: {out_mp4}")
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
            print(f"[*] Multiplexing isolated vocal MP4: {out_mp4_vocals}")
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
            print(f"[*] Multiplexing recombined MP4: {out_mp4_recombined}")
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
        print(f"[+] Render complete: {out_mp4}, {out_mp4.parent / f'{out_mp4.stem}_vocals.mp4'}, and {out_mp4.parent / f'{out_mp4.stem}_recombined.mp4'}")
    elif is_both:
        print(f"[+] Render complete: {out_mp4} and {out_mp4.parent / f'{out_mp4.stem}_vocals.mp4'}")
    else:
        print(f"[+] Render complete: {out_mp4}")
    return str(out_mp4)
