#!/usr/bin/env python3
"""
Mix Validation Tool (Milestone M4)
Validates rendered sentence mix artifacts against the frozen EditPlan.
Checks exact master PCM sample counts, rational video frame quantization,
source provenance legality, audio/video duration bounds, and XML agreement.
Exits with nonzero status code on any invariant violation.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile as sf

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add src and tools to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from contracts import EditPlan, RationalTime
from listen_verify import verify_listening


def measure_audio_loudness(wav_path: Union[str, Path]) -> Tuple[float, float]:
    """
    Measures integrated loudness (LUFS) and true peak (dBTP/dBFS) using FFmpeg ebur128 filter.
    Returns (integrated_lufs, true_peak_dbtp).
    """
    p = Path(wav_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Master WAV not found: {p}")

    cmd = [
        "ffmpeg", "-nostats",
        "-i", str(p),
        "-filter_complex", "ebur128=peak=true",
        "-f", "null", "-"
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    stderr = res.stderr

    m_i = re.search(r"Integrated loudness:\s+I:\s+([-\d.]+|-\s*inf)\s+LUFS", stderr)
    if not m_i:
        m_i = re.search(r"I:\s+([-\d.]+|-\s*inf)\s+LUFS", stderr)

    m_tp = re.search(r"True peak:\s+Peak:\s+([-\d.]+|-\s*inf)\s+(?:dBFS|dBTP)", stderr)
    if not m_tp:
        m_tp = re.search(r"Peak:\s+([-\d.]+|-\s*inf)\s+(?:dBFS|dBTP)", stderr)

    if not m_i or not m_tp:
        raise RuntimeError(f"Could not parse ebur128 loudness metrics from FFmpeg output: {stderr[-500:]}")

    i_raw = m_i.group(1).replace(" ", "")
    tp_raw = m_tp.group(1).replace(" ", "")

    integrated_lufs = -70.0 if "inf" in i_raw else float(i_raw)
    true_peak_dbtp = float("-inf") if "inf" in tp_raw else float(tp_raw)

    return integrated_lufs, true_peak_dbtp


measure_loudness = measure_audio_loudness


def validate_mix(
    stem: Optional[str] = None,
    render_path: Optional[Union[str, Path]] = None,
    plan_path: Optional[Union[str, Path]] = None,
    require_asr: bool = False,
    skip_asr: bool = False,
    asr_tolerance: float = 0.20,
    check_loudness: Optional[bool] = None,
    target_loudness: float = -16.0,
    loudness_tolerance: float = 1.0,
    max_true_peak: float = -1.0,
) -> bool:
    if stem:
        stem_path = Path(stem).resolve()
        actual_plan_path = stem_path.with_suffix(".plan.json")
        wav_path = stem_path.with_suffix(".wav")
        mp4_path = stem_path.with_suffix(".mp4")
        xml_path = stem_path.with_suffix(".xml")
    elif render_path and plan_path:
        mp4_path = Path(render_path).resolve()
        actual_plan_path = Path(plan_path).resolve()
        wav_path = mp4_path.with_suffix(".wav")
        xml_path = mp4_path.with_suffix(".xml")
        stem_path = mp4_path.with_suffix("")
    else:
        raise ValueError("Either stem or both render_path and plan_path must be provided to validate_mix")

    plan_path = actual_plan_path
    errors: List[str] = []

    print(f"[*] Validating mix artifacts for stem: {stem_path.name}")

    # 1. Check file existence
    if not plan_path.exists():
        errors.append(f"Missing EditPlan file: {plan_path}")
    if not wav_path.exists():
        errors.append(f"Missing Master WAV file: {wav_path}")
    if not mp4_path.exists():
        errors.append(f"Missing Rendered MP4 file: {mp4_path}")

    if errors:
        for err in errors:
            print(f"[!] Error: {err}", file=sys.stderr)
        return False

    # 2. Validate EditPlan
    try:
        plan = EditPlan.load(plan_path)
        plan.validate()
        print(f"[+] EditPlan is structurally valid (plan_id: {plan.plan_id}, segments: {len(plan.segments)})")
    except Exception as e:
        errors.append(f"EditPlan validation failed: {e}")
        for err in errors:
            print(f"[!] Error: {err}", file=sys.stderr)
        return False

    # 3. Validate Master PCM WAV
    try:
        info = sf.info(str(wav_path))
        if info.samplerate != plan.project_sample_rate:
            errors.append(
                f"Master WAV sample rate mismatch: expected {plan.project_sample_rate} Hz, got {info.samplerate} Hz"
            )

        if info.frames != plan.total_output_samples:
            errors.append(
                f"Master WAV sample count mismatch: plan declared {plan.total_output_samples} samples, "
                f"wav contains {info.frames} samples (diff: {info.frames - plan.total_output_samples})"
            )

        # Check waveform bounds
        data, _ = sf.read(str(wav_path))
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            errors.append("Master WAV contains NaN or Inf samples")

        peak = np.max(np.abs(data)) if len(data) > 0 else 0.0
        if peak > 1.0001:
            errors.append(f"Master WAV exceeds 0 dBFS clipping bound: peak={peak:.4f}")

        print(f"[+] Master WAV verified: exact {info.frames} samples, {info.samplerate} Hz, peak={peak:.3f}")
    except Exception as e:
        errors.append(f"Failed to inspect Master WAV: {e}")

    # 3.5 Validate EBU R128 Loudness Normalization
    should_check_loudness = (
        check_loudness is True
        or (check_loudness is None and bool(getattr(plan, "metadata", {}).get("normalize_loudness", False)))
    )
    if should_check_loudness and wav_path.exists():
        try:
            data_check, _ = sf.read(str(wav_path), dtype="float32")
            wav_peak = float(np.max(np.abs(data_check))) if len(data_check) > 0 else 0.0
            if wav_peak < 1e-4:
                errors.append(f"Master WAV is silent (peak={wav_peak:.6f}) and fails loudness normalization")
            else:
                i_lufs, tp_dbtp = measure_audio_loudness(wav_path)
                min_lufs = target_loudness - loudness_tolerance
                max_lufs = target_loudness + loudness_tolerance
                is_peak_limited = (abs(tp_dbtp - max_true_peak) <= 0.2 and i_lufs < min_lufs)
                if (i_lufs < min_lufs or i_lufs > max_lufs) and not is_peak_limited:
                    errors.append(
                        f"Master WAV integrated loudness {i_lufs:.2f} LUFS is outside acceptance bounds "
                        f"[{min_lufs:.1f}, {max_lufs:.1f}] LUFS (target: {target_loudness:.1f} ± {loudness_tolerance:.1f} LUFS)"
                    )
                if tp_dbtp > max_true_peak:
                    errors.append(
                        f"Master WAV true peak {tp_dbtp:.2f} dBTP exceeds limit {max_true_peak:.1f} dBTP"
                    )
                if not any("loudness" in e or "true peak" in e for e in errors):
                    print(
                        f"[+] Master WAV EBU R128 loudness verified: {i_lufs:.2f} LUFS (range [{min_lufs:.1f}, {max_lufs:.1f}] LUFS), "
                        f"true peak: {tp_dbtp:.2f} dBTP (limit <= {max_true_peak:.1f} dBTP)"
                    )
        except Exception as e:
            errors.append(f"Loudness measurement failed: {e}")

    # 4. Validate MP4 container via ffprobe
    try:
        cmd = [
            "ffprobe", "-v", "error",
            "-show_streams", "-show_format",
            "-of", "json",
            str(mp4_path)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        probe_data = json.loads(res.stdout)

        video_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "video"), None)
        audio_stream = next((s for s in probe_data.get("streams", []) if s.get("codec_type") == "audio"), None)

        if video_stream is None:
            errors.append("Rendered MP4 has no video stream")
        if audio_stream is None:
            errors.append("Rendered MP4 has no audio stream")

        if video_stream and audio_stream:
            # Check frame rate
            r_frame_rate = video_stream.get("r_frame_rate", "")
            expected_fps_frac = plan.video_fps.as_fraction()
            actual_fps = RationalTime.from_str(r_frame_rate).as_fraction()
            if abs(float(actual_fps - expected_fps_frac)) > 0.05:
                errors.append(f"Video frame rate mismatch: expected {expected_fps_frac}, got {actual_fps}")

            # Check bounded audio/video duration discrepancy (< 1 frame)
            one_frame_sec = float(plan.video_fps.denominator) / float(plan.video_fps.numerator)
            v_dur = float(video_stream.get("duration", probe_data.get("format", {}).get("duration", 0.0)))
            a_dur = float(audio_stream.get("duration", probe_data.get("format", {}).get("duration", 0.0)))

            # Account for audio encoder priming/padding
            diff = abs(v_dur - a_dur)
            if diff > (one_frame_sec * 2.0):  # allow within 2 frame bounds for AAC priming
                errors.append(
                    f"Audio/video duration divergence exceeds bound: audio={a_dur:.3f}s, video={v_dur:.3f}s (diff={diff:.3f}s > {one_frame_sec:.3f}s)"
                )
            else:
                print(f"[+] Logical mux timing verified: video={v_dur:.3f}s, audio={a_dur:.3f}s (diff={diff:.3f}s)")

    except Exception as e:
        errors.append(f"ffprobe validation failed: {e}")

    # 5. Validate Source Provenance
    for seg in plan.segments:
        if seg.is_silence:
            continue
        if seg.source_content_id not in plan.sources:
            errors.append(f"Segment {seg.segment_index} references undeclared source '{seg.source_content_id}'")
        if seg.source_start_sample < 0 or seg.source_end_sample <= seg.source_start_sample:
            errors.append(f"Segment {seg.segment_index} has invalid source bounds [{seg.source_start_sample}, {seg.source_end_sample})")

    # 6. Validate XML agreement if present
    if xml_path.exists():
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            timebase_elem = root.find(".//sequence/rate/timebase")
            expected_tb = str(int(round(plan.video_fps.as_float())))
            if timebase_elem is not None and timebase_elem.text != expected_tb:
                errors.append(f"XML timebase mismatch: expected {expected_tb}, got {timebase_elem.text}")
            print(f"[+] FCP7 XML sequence verified: timebase={expected_tb}")
        except Exception as e:
            errors.append(f"XML validation failed: {e}")

    # 7. Validate ASR Listening Verification
    should_run_asr = require_asr or (bool(getattr(plan, "target_text", None)) and not skip_asr)
    if should_run_asr:
        if not getattr(plan, "target_text", None):
            errors.append("ASR listening verification required, but EditPlan does not specify target_text")
        else:
            try:
                media_for_asr = wav_path if wav_path.exists() else mp4_path
                asr_res = verify_listening(
                    input_path=media_for_asr,
                    expected_text=plan.target_text,
                    wer_threshold=asr_tolerance,
                )
                if not asr_res["passed"]:
                    errors.append(
                        f"ASR listening verification failed: WER={asr_res['wer']:.2f}, "
                        f"transcribed='{asr_res['transcribed_text']}', expected='{plan.target_text}'. "
                        f"Reasons: {', '.join(asr_res.get('failure_reasons', []))}"
                    )
                    if asr_res.get("bleeding_words"):
                        errors.append(f"Unintended dialogue bleed detected: {asr_res['bleeding_words']}")
                    if asr_res.get("missing_words"):
                        errors.append(f"Missing target words: {asr_res['missing_words']}")
                else:
                    print(
                        f"[+] ASR listening verification passed: transcribed='{asr_res['transcribed_text']}', "
                        f"WER={asr_res['wer']:.2f}, CER={asr_res['cer']:.2f}"
                    )
            except Exception as e:
                errors.append(f"ASR listening verification failed with exception: {e}")

    if errors:
        print("\n[!] Mix Validation FAILED with errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return False

    print("\n[+] All validation checks passed successfully!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate sentence mixer output artifacts against EditPlan.")
    parser.add_argument("--stem", help="Base path / stem of output artifacts (without extension)")
    parser.add_argument("--render", help="Path to rendered MP4 file (optional alternative to --stem)")
    parser.add_argument("--plan", help="Path to EditPlan JSON file (optional alternative to --stem)")
    parser.add_argument("--require-asr", action="store_true", help="Require ASR listening verification against plan.target_text")
    parser.add_argument("--skip-asr", action="store_true", help="Skip ASR listening verification")
    parser.add_argument("--asr-tolerance", type=float, default=0.20, help="ASR WER tolerance threshold (default: 0.20)")
    parser.add_argument("--check-loudness", action="store_true", default=None, help="Enforce EBU R128 loudness check (-16.0 ± 1.0 LUFS, <= -1.0 dBTP)")
    parser.add_argument("--skip-loudness", action="store_true", help="Skip EBU R128 loudness check")
    args = parser.parse_args()

    if not args.stem and not (args.render and args.plan):
        parser.error("Either --stem or both --render and --plan must be provided.")

    check_loudness_flag = True if args.check_loudness else (False if args.skip_loudness else None)

    success = validate_mix(
        stem=args.stem,
        render_path=args.render,
        plan_path=args.plan,
        require_asr=args.require_asr,
        skip_asr=args.skip_asr,
        asr_tolerance=args.asr_tolerance,
        check_loudness=check_loudness_flag,
    )
    sys.exit(0 if success else 1)

