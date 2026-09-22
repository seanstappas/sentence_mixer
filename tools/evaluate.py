#!/usr/bin/env python3
"""
Evaluation Harness for Sentence Mixing (Milestones M5 & M6)
Runs repeatable test suites, computes objective cut and timing metrics,
exports blinded listening assets, and generates an unblinded assignment key.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RUN_MIXER = REPO_ROOT / "run_mixer.py"
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from listen_verify import verify_listening


def evaluate_suite(
    suite_path: str,
    output_dir: str,
    verify_asr: bool = True,
    asr_tolerance: float = 0.20,
) -> Dict[str, Any]:
    s_path = Path(suite_path).resolve()
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    blinded_dir = out_dir / "blinded_audio"
    blinded_dir.mkdir(parents=True, exist_ok=True)

    with open(s_path, "r", encoding="utf-8") as f:
        suite = json.load(f)

    manifest_rel = suite.get("manifest", "smoke_sources.json")
    if Path(manifest_rel).is_absolute():
        manifest_abs = Path(manifest_rel)
    elif (s_path.parent / manifest_rel).exists():
        manifest_abs = (s_path.parent / manifest_rel).resolve()
    elif (REPO_ROOT / manifest_rel).exists():
        manifest_abs = (REPO_ROOT / manifest_rel).resolve()
    else:
        manifest_abs = (s_path.parent / Path(manifest_rel).name).resolve()
    speaker_id = suite.get("speaker_id", "obi-wan")

    cases = suite.get("cases", [])
    print(f"[*] Running Evaluation Suite: '{suite.get('suite_name', 'Suite')}' ({len(cases)} cases)")
    print(f"    Manifest: {manifest_abs}")
    print(f"    Output Dir: {out_dir}")

    results: List[Dict[str, Any]] = []
    assignment_key: Dict[str, Any] = {}

    for case in cases:
        case_id = case["case_id"]
        target_text = case["target_text"]
        stratum = case.get("stratum", "general")
        print(f"\n[Case: {case_id}] [{stratum}] Target: \"{target_text}\"")

        case_results = {
            "case_id": case_id,
            "stratum": stratum,
            "target_text": target_text,
            "configurations": {},
        }

        # Evaluate across configurations: span_basic and legacy
        for config_name, selector in [("span_basic", "span"), ("legacy", "legacy")]:
            render_stem = out_dir / f"{case_id}_{config_name}"
            render_mp4 = render_stem.with_suffix(".mp4")
            render_wav = render_stem.with_suffix(".wav")
            render_plan = render_stem.with_suffix(".plan.json")

            cmd = [
                PYTHON, str(RUN_MIXER), "mix",
                "--manifest", str(manifest_abs),
                "--speaker", speaker_id,
                "--text", target_text,
                "--selector", selector,
                "--features", "basic",
                "--mode", "source-derived",
                "--render", str(render_mp4),
                "--plan", str(render_plan),
            ]

            res = subprocess.run(cmd, cwd=str(REPO_ROOT), capture_output=True, text=True)
            success = (res.returncode == 0 and render_mp4.exists())

            config_metrics: Dict[str, Any] = {
                "success": success,
                "selector": selector,
                "features": "basic",
                "render_mp4": str(render_mp4) if success else None,
                "render_wav": str(render_wav) if render_wav.exists() else None,
            }

            if success and render_plan.exists():
                try:
                    with open(render_plan, "r", encoding="utf-8") as pf:
                        pdata = json.load(pf)
                    config_metrics["total_output_samples"] = pdata.get("total_output_samples", 0)
                    config_metrics["num_segments"] = len(pdata.get("segments", []))
                    config_metrics["num_transitions"] = len(pdata.get("transitions", []))
                    span_types = [s.get("span_type") for s in pdata.get("segments", [])]
                    config_metrics["span_type_distribution"] = {
                        st: span_types.count(st) for st in set(span_types)
                    }
                except Exception:
                    pass

            # Generate blinded asset for listening comparison
            if render_wav.exists():
                blind_token = uuid.uuid4().hex[:10]
                blinded_wav = blinded_dir / f"blind_{blind_token}.wav"
                shutil.copy2(render_wav, blinded_wav)
                assignment_key[f"blind_{blind_token}"] = {
                    "case_id": case_id,
                    "target_text": target_text,
                    "configuration": config_name,
                    "selector": selector,
                }
                config_metrics["blinded_asset"] = f"blind_{blind_token}.wav"

            # Run ASR listening verification
            if verify_asr and success and (render_wav.exists() or render_mp4.exists()):
                media_path = render_wav if render_wav.exists() else render_mp4
                try:
                    asr_res = verify_listening(
                        input_path=media_path,
                        expected_text=target_text,
                        wer_threshold=asr_tolerance,
                    )
                    config_metrics["asr"] = {
                        "transcribed_text": asr_res.get("transcribed_text", ""),
                        "normalized_transcribed": asr_res.get("normalized_transcribed", ""),
                        "wer": asr_res.get("wer", 1.0),
                        "cer": asr_res.get("cer", 1.0),
                        "passed": asr_res.get("passed", False),
                        "all_expected_words_present": asr_res.get("all_expected_words_present", False),
                        "word_matches": asr_res.get("word_matches", []),
                        "missing_words": asr_res.get("missing_words", []),
                        "bleeding_words": asr_res.get("bleeding_words", []),
                    }
                except Exception as asr_err:
                    config_metrics["asr"] = {
                        "error": str(asr_err),
                        "passed": False,
                        "wer": 1.0,
                    }

            case_results["configurations"][config_name] = config_metrics
            asr_info = ""
            if "asr" in config_metrics and isinstance(config_metrics["asr"], dict):
                asr_data: Dict[str, Any] = config_metrics["asr"]
                asr_tag = "PASS" if asr_data.get("passed") else "FAIL"
                asr_wer = float(asr_data.get("wer", 0.0))
                asr_info = f" (ASR: {asr_tag}, WER={asr_wer:.2f})"
            print(f"    - {config_name}: {'PASS' if success else 'FAIL'}{asr_info}")

        results.append(case_results)

    # Compute aggregate ASR metrics across evaluated cases
    all_wers = []
    all_cers = []
    asr_passes = 0
    total_asr_evaluated = 0

    for r in results:
        for cname, cdata in r.get("configurations", {}).items():
            if "asr" in cdata and "wer" in cdata["asr"]:
                total_asr_evaluated += 1
                all_wers.append(cdata["asr"]["wer"])
                if "cer" in cdata["asr"]:
                    all_cers.append(cdata["asr"]["cer"])
                if cdata["asr"].get("passed"):
                    asr_passes += 1

    mean_wer = round(float(sum(all_wers) / len(all_wers)), 4) if all_wers else None
    mean_cer = round(float(sum(all_cers) / len(all_cers)), 4) if all_cers else None
    asr_pass_rate = round(float(asr_passes / total_asr_evaluated), 4) if total_asr_evaluated > 0 else None

    # Save outputs
    eval_report = {
        "suite_name": suite.get("suite_name"),
        "version": suite.get("version"),
        "manifest": str(manifest_abs),
        "cases_evaluated": len(results),
        "summary": {
            "cases_evaluated": len(results),
            "total_asr_evaluated": total_asr_evaluated,
            "mean_wer": mean_wer,
            "mean_cer": mean_cer,
            "asr_pass_rate": asr_pass_rate,
        },
        "results": results,
    }

    report_path = out_dir / "evaluation_report.json"
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(eval_report, rf, indent=2)

    key_path = out_dir / "assignment_key.json"
    with open(key_path, "w", encoding="utf-8") as kf:
        json.dump(assignment_key, kf, indent=2)

    print("\n[+] Evaluation complete:")
    print(f"    - Report: {report_path}")
    print(f"    - Blinded Assets: {blinded_dir}")
    print(f"    - Assignment Key: {key_path}")
    if mean_wer is not None:
        print(f"    - Mean WER: {mean_wer:.3f}")
    if mean_cer is not None:
        print(f"    - Mean CER: {mean_cer:.3f}")
    if asr_pass_rate is not None:
        print(f"    - ASR Pass Rate: {asr_pass_rate * 100:.1f}% ({asr_passes}/{total_asr_evaluated})")

    return eval_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate sentence mixer test suite.")
    parser.add_argument("--suite", required=True, help="Path to evaluation suite JSON")
    parser.add_argument("--output-dir", default="./output/evaluation", help="Directory for evaluation results")
    parser.add_argument("--skip-asr", action="store_true", help="Skip ASR listening verification")
    parser.add_argument("--asr-tolerance", type=float, default=0.20, help="ASR WER tolerance threshold (default: 0.20)")
    args = parser.parse_args()

    evaluate_suite(
        suite_path=args.suite,
        output_dir=args.output_dir,
        verify_asr=not args.skip_asr,
        asr_tolerance=args.asr_tolerance,
    )
