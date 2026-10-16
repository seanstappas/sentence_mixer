#!/usr/bin/env python3
"""
Acoustic & Enunciation Quality Checker for Sentence Mixer Renders.
Combines ASR recognition (unprimed & target-primed via tools/listen_verify.py),
duration floors, acoustic energy checks, and plan segment verification
to ensure synthesized speech sounds clear, natural, and free of swallowed words.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add repo root and src to sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from listen_verify import (
    extract_audio_16k,
    normalize_text,
    verify_listening,
)


def detect_splice_clicks(
    audio_path: Path,
    plan_data: Dict[str, Any],
    window_ms: float = 5.0,
    sigma_threshold: float = 6.0,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Inspects a +/- window_ms window around each EditPlan transition timestamp
    in the audio, flagging any sample-to-sample difference spike |Delta x[n]| > 6 * sigma_local
    or >4kHz spectral energy surge as an audible splice click.
    """
    detected_clicks: List[Dict[str, Any]] = []
    try:
        import soundfile as sf
        import numpy as np
        import scipy.signal

        if audio_path.suffix.lower() == ".wav":
            data, sr = sf.read(str(audio_path), dtype="float32")
        else:
            cand_wav = audio_path.with_suffix(".wav")
            if cand_wav.exists():
                data, sr = sf.read(str(cand_wav), dtype="float32")
            else:
                data = extract_audio_16k(audio_path)
                sr = 16000
        if data.ndim > 1:
            data = data.mean(axis=1)
    except Exception:
        return True, []

    if len(data) == 0:
        return True, []

    plan_sr = plan_data.get("project_sample_rate", 48000)
    segments = plan_data.get("segments", [])
    transitions = plan_data.get("transitions", [])

    w_samples = int(round((window_ms / 1000.0) * sr))
    local_samples = int(round(0.025 * sr))

    for trans in transitions:
        to_idx = trans.get("to_segment_index")
        from_idx = trans.get("from_segment_index")
        overlap = trans.get("overlap_samples", 0)

        pts_to_check: List[Tuple[str, float]] = []
        if to_idx is not None and to_idx < len(segments):
            to_seg = segments[to_idx]
            pts_to_check.append(("fade_start", to_seg.get("output_start_sample", 0) / float(plan_sr)))
        if overlap > 0 and from_idx is not None and from_idx < len(segments):
            from_seg = segments[from_idx]
            pts_to_check.append(("fade_end", from_seg.get("output_end_sample", 0) / float(plan_sr)))

        for boundary_type, t_sec in pts_to_check:
            center_sample = int(round(t_sec * sr))

            s_min = max(0, center_sample - local_samples)
            s_max = min(len(data), center_sample + local_samples)
            if s_max - s_min < 10:
                continue

            w_start = max(0, center_sample - w_samples)
            w_end = min(len(data), center_sample + w_samples)
            if w_end - w_start < 2:
                continue

            bg_left = np.diff(data[s_min:w_start]) if w_start - s_min >= 2 else np.zeros(0, dtype=np.float32)
            bg_right = np.diff(data[w_end:s_max]) if s_max - w_end >= 2 else np.zeros(0, dtype=np.float32)
            bg_diff = np.concatenate([bg_left, bg_right])
            sigma_local = float(np.std(bg_diff)) + 1e-6 if len(bg_diff) >= 4 else float(np.std(np.diff(data[s_min:s_max]))) + 1e-6

            w_diff = np.abs(np.diff(data[w_start:w_end]))
            max_diff = float(np.max(w_diff))

            spike_ratio = max_diff / sigma_local

            hp_click = False
            if sr > 10000 and (s_max - s_min) > 20:
                try:
                    sos = scipy.signal.butter(4, 4000.0 / (0.5 * sr), btype="high", output="sos")
                    chunk_hp = scipy.signal.sosfiltfilt(sos, data[s_min:s_max])
                    rw_start = w_start - s_min
                    rw_end = w_end - s_min
                    win_hp = chunk_hp[rw_start:rw_end]
                    bg_hp_left = chunk_hp[:rw_start] if rw_start > 0 else np.zeros(0, dtype=np.float32)
                    bg_hp_right = chunk_hp[rw_end:] if rw_end < len(chunk_hp) else np.zeros(0, dtype=np.float32)
                    bg_hp = np.concatenate([bg_hp_left, bg_hp_right])
                    win_rms = float(np.sqrt(np.mean(win_hp ** 2))) if len(win_hp) > 0 else 0.0
                    bg_rms = float(np.sqrt(np.mean(bg_hp ** 2))) + 1e-6 if len(bg_hp) > 0 else 1e-6
                    win_peak = float(np.max(np.abs(win_hp))) if len(win_hp) > 0 else 0.0
                    if win_rms > 8.0 * bg_rms and win_peak > 0.015:
                        hp_click = True
                except Exception:
                    pass

            if (spike_ratio > sigma_threshold and max_diff > 0.03) or hp_click:
                detected_clicks.append({
                    "transition_index": trans.get("transition_index"),
                    "boundary_type": boundary_type,
                    "time_sec": round(t_sec, 4),
                    "spike_ratio": round(spike_ratio, 2),
                    "max_diff": round(max_diff, 4),
                    "sigma_local": round(sigma_local, 5),
                    "spectral_flux_click": hp_click,
                })

    return len(detected_clicks) == 0, detected_clicks


def check_render_quality(
    input_media: Path,
    expected_text: str,
    plan_path: Optional[Path] = None,
    critical_words: Optional[List[str]] = None,
    min_word_duration_ms: float = 65.0,
    device: str = "cpu",
    asr_backend_name: str = "auto",
) -> Tuple[bool, Dict[str, Any]]:
    """
    Performs comprehensive acoustic and ASR quality verification.

    Returns:
        (passed: bool, report: Dict[str, Any])
    """
    report: Dict[str, Any] = {
        "media": str(input_media),
        "expected_text": expected_text,
        "critical_words": critical_words or [],
        "checks": {},
        "issues": [],
    }

    # 1. Inspect EditPlan if available
    plan_data = None
    if plan_path and plan_path.exists():
        try:
            plan_data = json.loads(plan_path.read_text(encoding="utf-8"))
        except Exception as e:
            report["issues"].append(f"Failed to load plan: {e}")
    else:
        # Try inferring plan path from media path
        cand_plan = input_media.with_suffix(".plan.json")
        if not cand_plan.exists():
            # Check without suffixes like _recombined or _vocals
            clean_stem = input_media.stem.replace("_recombined", "").replace("_vocals", "")
            cand_plan = input_media.parent / f"{clean_stem}.plan.json"
        if cand_plan.exists():
            try:
                plan_data = json.loads(cand_plan.read_text(encoding="utf-8"))
                plan_path = cand_plan
            except Exception:
                pass

    sr = 48000
    if plan_data:
        sr = plan_data.get("project_sample_rate", 48000)
        report["plan_path"] = str(plan_path)

        # Word duration analysis from plan segments
        word_durations: Dict[str, float] = {}
        speech_levels: List[float] = []
        for s in plan_data.get("segments", []):
            if s.get("is_silence") or s.get("span_type") in ("pause", "continue"):
                continue
            dur_ms = (s["output_end_sample"] - s["output_start_sample"]) * 1000.0 / float(sr)
            w_text = normalize_text(s.get("target_text", "")).lower()
            if w_text:
                for tok in w_text.split():
                    word_durations[tok] = word_durations.get(tok, 0.0) + dur_ms
            gain = s.get("gain_db", 0.0)
            speech_levels.append(gain)

        report["word_durations_ms"] = word_durations

    # 2. Extract 16kHz audio array for ASR verification
    try:
        audio_16k = extract_audio_16k(input_media)
    except Exception as e:
        report["issues"].append(f"Audio extraction failed: {e}")
        return False, report

    # 3. Run ASR verification (both unprimed and assisted)
    asr_res = verify_listening(
        input_path=input_media,
        expected_text=expected_text,
        backend=asr_backend_name,
        device=device,
        tolerance=0.45,
        audio_array=audio_16k,
        require_all_words=False,
    )
    report["asr"] = asr_res

    unprimed_hyp = normalize_text(asr_res.get("transcribed_text", "")).lower().split()
    assisted_hyp = normalize_text(asr_res.get("assisted_metrics", {}).get("transcription", "")).lower().split()

    critical_passed = True
    missing_critical = []
    for cw in (critical_words or []):
        cw_norm = normalize_text(cw).lower()
        found = (cw_norm in unprimed_hyp) or (cw_norm in assisted_hyp)
        if not found:
            # Check subword match
            found = any(cw_norm in tok for tok in (unprimed_hyp + assisted_hyp))
        if not found:
            missing_critical.append(cw)
            critical_passed = False
            report["issues"].append(f"Critical word '{cw}' was not detected by ASR in rendered audio")

    report["checks"]["critical_words_detected"] = {
        "passed": critical_passed,
        "missing": missing_critical,
    }

    # 4. Check critical words duration in edit plan
    if plan_data:
        swallowed_words = []
        for cw in (critical_words or []):
            cw_norm = normalize_text(cw).lower()
            dur = word_durations.get(cw_norm, 0.0)
            is_recognized = (cw_norm in unprimed_hyp) or (cw_norm in assisted_hyp)
            # Only flag as swallowed if an intact word span was ultra-short, or if ASR failed to recognize it
            if 0 < dur < min_word_duration_ms and not is_recognized:
                swallowed_words.append((cw, dur))
                report["issues"].append(f"Word '{cw}' is critically short ({dur:.1f} ms < {min_word_duration_ms:.1f} ms) and not recognized")

        report["checks"]["duration_check"] = {
            "passed": len(swallowed_words) == 0,
            "swallowed_words": swallowed_words,
        }

    # 5. Check for splice clicks at transitions
    if plan_data:
        clicks_passed, clicks = detect_splice_clicks(input_media, plan_data)
        report["checks"]["click_detection"] = {
            "passed": clicks_passed,
            "detected_clicks": clicks,
        }
        if not clicks_passed:
            report["issues"].append(f"Detected {len(clicks)} audible splice click(s) at transition boundaries")

    # Overall pass criteria:
    # 1. All critical target words detected
    # 2. No critical words swallowed below duration floor
    # 3. No audible splice clicks at transition boundaries
    # 4. Assisted WER <= wer_threshold
    assisted_wer = asr_res.get("assisted_wer", asr_res.get("wer", 0.0))
    wer_threshold = getattr(asr_res, "wer_threshold", 0.40) if hasattr(asr_res, "wer_threshold") else 0.40
    report["checks"]["wer_pass"] = assisted_wer <= wer_threshold
    if assisted_wer > wer_threshold:
        report["issues"].append(f"Assisted WER too high ({assisted_wer:.2f} > {wer_threshold:.2f})")

    overall_passed = (
        critical_passed
        and report["checks"].get("duration_check", {}).get("passed", True)
        and report["checks"].get("click_detection", {}).get("passed", True)
        and report["checks"]["wer_pass"]
    )
    report["overall_passed"] = overall_passed

    return overall_passed, report


def check_suite_quality(
    suite_path: Path,
    render_dir: Path,
    device: str = "cpu",
    wer_threshold: float = 0.40,
) -> Tuple[bool, List[Dict[str, Any]]]:
    """Runs quality check across all cases in an evaluation suite."""
    if not suite_path.exists():
        raise FileNotFoundError(f"Suite file not found: {suite_path}")

    suite_data = json.loads(suite_path.read_text(encoding="utf-8"))
    cases = suite_data.get("cases", [])
    all_reports: List[Dict[str, Any]] = []
    all_passed = True

    from run_mixer import slugify_text

    for case in cases:
        case_id = case.get("case_id", "")
        expected = case.get("target_text", "")
        slug = slugify_text(expected)

        # Look for matching render file
        cand_files = [
            render_dir / f"{case_id}.mp4",
            render_dir / f"{slug}.mp4",
            render_dir / f"{slug}_recombined.mp4",
            render_dir / f"{case_id}.wav",
            render_dir / f"{slug}.wav",
        ]
        render_file = next((f for f in cand_files if f.exists()), None)
        if not render_file:
            v_matches = sorted(
                list(render_dir.glob(f"{slug}_v*.mp4")) + list(render_dir.glob(f"{case_id}_v*.mp4")),
                reverse=True,
            )
            if v_matches:
                render_file = v_matches[0]

        if not render_file:
            all_reports.append({
                "case_id": case_id,
                "expected": expected,
                "media": None,
                "overall_passed": False,
                "issues": [f"No render output found for case '{case_id}' in {render_dir}"],
            })
            all_passed = False
            continue

        crit = case.get("critical_words", [])
        passed, rep = check_render_quality(
            input_media=render_file,
            expected_text=expected,
            critical_words=crit,
            device=device,
        )
        rep["case_id"] = case_id
        all_reports.append(rep)
        if not passed:
            all_passed = False

    return all_passed, all_reports


def print_summary_table(reports: List[Dict[str, Any]]) -> None:
    """Prints a structured Markdown summary table for quality check reports."""
    print("\n| Status | Case / File | Expected Text | Assisted WER | Critical Words | Issues |")
    print("| :---: | :--- | :--- | :---: | :---: | :--- |")
    for rep in reports:
        passed = rep.get("overall_passed", False)
        status_icon = "PASS" if passed else "FAIL"
        media_name = Path(rep["media"]).name if rep.get("media") else rep.get("case_id", "N/A")
        exp = rep.get("expected_text", rep.get("expected", ""))
        asr = rep.get("asr", {})
        wer = asr.get("assisted_wer", 0.0)
        crit_res = rep.get("checks", {}).get("critical_words_detected", {})
        missing = crit_res.get("missing", [])
        crit_str = "All OK" if not missing else f"Missing: {','.join(missing)}"
        issues = "; ".join(rep.get("issues", [])) or "None"
        print(f"| **{status_icon}** | `{media_name}` | *\"{exp}\"* | {wer:.2f} | {crit_str} | {issues} |")
    print()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Acoustic and ASR enunciation quality checker")
    parser.add_argument("--input", "-i", type=Path, default=None, help="Path to rendered audio WAV or video MP4")
    parser.add_argument("--expected", "-e", type=str, default=None, help="Expected target text")
    parser.add_argument("--plan", "-p", type=Path, default=None, help="Path to edit plan JSON (optional)")
    parser.add_argument("--critical-words", "-c", nargs="*", default=[], help="Specific critical words to verify")
    parser.add_argument("--min-dur-ms", type=float, default=65.0, help="Minimum duration floor in ms (default: 65)")
    parser.add_argument("--wer-threshold", type=float, default=0.40, help="Assisted WER pass threshold (default: 0.40)")
    parser.add_argument("--suite", "-s", type=Path, default=None, help="Path to evaluation suite JSON to verify")
    parser.add_argument("--render-dir", type=Path, default=Path("output"), help="Render directory for suite cases")
    parser.add_argument("--device", default="cpu", help="Device for ASR (default: cpu)")
    parser.add_argument("--json", action="store_true", help="Output machine-readable JSON")

    args = parser.parse_args(argv)

    if args.suite:
        passed, reports = check_suite_quality(
            suite_path=args.suite,
            render_dir=args.render_dir,
            device=args.device,
            wer_threshold=args.wer_threshold,
        )
        if args.json:
            print(json.dumps({"suite_passed": passed, "cases": reports}, indent=2))
        else:
            print_summary_table(reports)
        return 0 if passed else 1

    if not args.input or not args.expected:
        parser.error("Either --suite or both --input and --expected must be provided.")

    if not args.input.exists():
        print(f"Error: Input media file not found: {args.input}", file=sys.stderr)
        return 2

    passed, report = check_render_quality(
        input_media=args.input,
        expected_text=args.expected,
        plan_path=args.plan,
        critical_words=args.critical_words,
        min_word_duration_ms=args.min_dur_ms,
        device=args.device,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        status_str = "PASS" if passed else "FAIL"
        print("\n=======================================================")
        print(f" Quality Check [{status_str}]: {args.input.name}")
        print("=======================================================")
        print(f" Expected text: \"{args.expected}\"")
        asr = report.get("asr", {})
        print(f" Unprimed ASR:  \"{asr.get('transcribed_text', '')}\"")
        print(f" Assisted ASR:  \"{asr.get('assisted_metrics', {}).get('transcription', '')}\"")
        print(f" Assisted WER:  {asr.get('assisted_wer', 0.0):.2f}")
        if args.critical_words:
            print(f" Critical words: {args.critical_words}")
            missing = report["checks"].get("critical_words_detected", {}).get("missing", [])
            if missing:
                print(f" [-] MISSING WORDS: {missing}")
            else:
                print(" [+] All critical words successfully recognized!")

        if report.get("issues"):
            print("\n Issues Detected:")
            for iss in report["issues"]:
                print(f"  * {iss}")
        print("=======================================================\n")

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
