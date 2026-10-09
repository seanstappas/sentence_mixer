#!/usr/bin/env python3
"""
Batch render 10 diverse test lines across Anakin and Obi-Wan with burned subtitles,
followed by automated ASR listening verification.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent.resolve()
PYTHON = sys.executable

TEST_LINES = [
    {
        "id": "test_line_01",
        "name": "test_line_01_obiwan_lightsaber",
        "speaker": "obi-wan",
        "manifest": "evaluation/smoke_sources.json",
        "text": "Hello there, have you seen my lightsaber?",
    },
    {
        "id": "test_line_02",
        "name": "test_line_02_anakin_sand_chocolate",
        "speaker": "anakin",
        "manifest": "evaluation/anakin_sources.json",
        "text": "I do not like sand, but I love chocolate.",
    },
    {
        "id": "test_line_03",
        "name": "test_line_03_obiwan_master_of_evil",
        "speaker": "obi-wan",
        "manifest": "evaluation/smoke_sources.json",
        "text": "The Chancellor is a master of evil.",
    },
    {
        "id": "test_line_04",
        "name": "test_line_04_anakin_underestimate_power",
        "speaker": "anakin",
        "manifest": "evaluation/anakin_sources.json",
        "text": "You underestimate my power, Master Yoda.",
    },
    {
        "id": "test_line_05",
        "name": "test_line_05_obiwan_high_ground",
        "speaker": "obi-wan",
        "manifest": "evaluation/smoke_sources.json",
        "text": "Don't try it, Anakin, I have the high ground.",
    },
    {
        "id": "test_line_06",
        "name": "test_line_06_anakin_jedi_are_evil",
        "speaker": "anakin",
        "manifest": "evaluation/anakin_sources.json",
        "text": "From my point of view, the Jedi are evil.",
    },
    {
        "id": "test_line_07",
        "name": "test_line_07_obiwan_death_of_me",
        "speaker": "obi-wan",
        "manifest": "evaluation/smoke_sources.json",
        "text": "Why do I get the feeling you are going to be the death of me?",
    },
    {
        "id": "test_line_08",
        "name": "test_line_08_anakin_browser_history",
        "speaker": "anakin",
        "manifest": "evaluation/anakin_sources.json",
        "text": "Master, I will delete your browser history.",
    },
    {
        "id": "test_line_09",
        "name": "test_line_09_obiwan_council_party",
        "speaker": "obi-wan",
        "manifest": "evaluation/smoke_sources.json",
        "text": "Anakin, the council does not approve of this party.",
    },
    {
        "id": "test_line_10",
        "name": "test_line_10_anakin_peace_and_pizza",
        "speaker": "anakin",
        "manifest": "evaluation/anakin_sources.json",
        "text": "I have brought peace, freedom, justice, and pizza to my new empire.",
    },
]


def run_batch():
    output_dir = ROOT / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    results = []

    print(f"[+] Starting batch rendering of {len(TEST_LINES)} diverse test lines...")
    start_total = time.time()

    for item in TEST_LINES:
        print("\n=======================================================")
        print(f"[+] Rendering {item['id']}: [{item['speaker']}] \"{item['text']}\"")
        print("=======================================================")

        mp4_path = output_dir / f"{item['name']}.mp4"
        cmd_render = [
            PYTHON,
            str(ROOT / "run_mixer.py"),
            "mix",
            "--manifest", str(ROOT / item["manifest"]),
            "--speaker", item["speaker"],
            "--text", item["text"],
            "--selector", "span",
            "--render", str(mp4_path),
            "--subtitles",
            "--burn-subtitles",
            "--subtitle-style", "classic",
            "--non-interactive",
        ]

        t0 = time.time()
        p = subprocess.run(cmd_render, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
        render_dur = time.time() - t0

        if p.returncode != 0:
            print(f"[-] FAILED rendering {item['id']}:")
            print(p.stderr)
            results.append({
                "id": item["id"],
                "name": item["name"],
                "speaker": item["speaker"],
                "text": item["text"],
                "status": "render_failed",
                "error": p.stderr,
                "render_seconds": render_dur,
            })
            continue

        print(f"[+] Rendered in {render_dur:.1f}s: {mp4_path}")

        # Run ASR listening verification
        cmd_verify = [
            PYTHON,
            str(ROOT / "tools" / "listen_verify.py"),
            "--input", str(mp4_path),
            "--expected", item["text"],
        ]
        pv = subprocess.run(cmd_verify, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8")
        print(pv.stdout.strip())

        # Parse output for metrics
        recognized = ""
        wer = 1.0
        for line in pv.stdout.splitlines():
            if "Recognized:" in line:
                recognized = line.split("Recognized:", 1)[1].strip().strip('"')
            elif "WER:" in line:
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        wer = float(parts[1])
                    except ValueError:
                        pass

        results.append({
            "id": item["id"],
            "name": item["name"],
            "speaker": item["speaker"],
            "text": item["text"],
            "status": "passed" if pv.returncode == 0 else "divergent",
            "render_seconds": round(render_dur, 2),
            "mp4": str(mp4_path),
            "recognized": recognized,
            "wer": wer,
            "verify_exit_code": pv.returncode,
        })

    total_dur = time.time() - start_total
    report_path = output_dir / "batch_test_lines_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({"total_seconds": round(total_dur, 2), "results": results}, f, indent=2)

    print(f"\n[+] Batch rendering finished in {total_dur:.1f}s. Report written to: {report_path}")
    print("\nSummary Table:")
    print(f"{'ID':<14} | {'Speaker':<8} | {'WER':<5} | {'Recognized Dialogue'}")
    print("-" * 80)
    for r in results:
        wer_str = f"{r.get('wer', 1.0):.2f}"
        rec = r.get("recognized", "")
        print(f"{r['id']:<14} | {r['speaker']:<8} | {wer_str:<5} | {rec}")


if __name__ == "__main__":
    run_batch()
