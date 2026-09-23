#!/usr/bin/env python3
"""
Empirical Speaker Diarization & Recognition Test: Anakin vs. Obi-Wan in input/anakin_ep3_1.mp4.
Enrolls Obi-Wan from verified single-speaker corpus, enrolls Anakin from anakin_ep2,
and diarizes the multi-speaker space battle in anakin_ep3_1.mp4.
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from diarization import SpeakerDiarizer
from speaker_registry import SpeakerRegistry
from vtt_preprocessor import parse_vtt_cues


def extract_16k_mono(media_path: Path, output_wav: Path) -> Path:
    output_wav.parent.mkdir(parents=True, exist_ok=True)
    if not output_wav.exists() or output_wav.stat().st_size == 0:
        cmd = [
            "ffmpeg", "-y", "-i", str(media_path),
            "-vn", "-ar", "16000", "-ac", "1",
            str(output_wav)
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg extraction failed: {res.stderr}")
    return output_wav


def main():
    print("[*] Starting Speaker Diarization & Recognition Test on anakin_ep3_1.mp4")
    build_dir = REPO_ROOT / ".build" / "speaker_test"
    build_dir.mkdir(parents=True, exist_ok=True)

    # 1. Prepare 16kHz mono audio tracks
    obi_media = REPO_ROOT / "input" / "obi_wan_ep3.mp4"
    anakin_media = REPO_ROOT / "input" / "anakin_ep2_1.mp4"
    test_media = REPO_ROOT / "input" / "anakin_ep3_1.mp4"
    test_vtt = REPO_ROOT / "input" / "anakin_ep3_1.en.vtt"

    obi_wav = extract_16k_mono(obi_media, build_dir / "obi_wan_enrollment.wav")
    anakin_wav = extract_16k_mono(anakin_media, build_dir / "anakin_enrollment.wav")
    test_wav = extract_16k_mono(test_media, build_dir / "anakin_ep3_1.wav")

    # 2. Setup Speaker Registry
    registry_file = build_dir / "speaker_registry.json"
    registry = SpeakerRegistry(config_path=registry_file, similarity_threshold=0.72)
    diarizer = SpeakerDiarizer(registry=registry)

    # 3. Enroll Obi-Wan
    print("[+] Enrolling Obi-Wan Kenobi from obi_wan_ep3...")
    obi_audio, obi_sr = sf.read(str(obi_wav))
    # Extract turns from the first 60 seconds
    obi_turns = diarizer.detect_speech_turns(obi_audio[: 60 * obi_sr], obi_sr)
    obi_embs = []
    for s, e in obi_turns[:10]:
        chunk = obi_audio[int(s * obi_sr) : int(e * obi_sr)]
        if len(chunk) >= int(0.4 * obi_sr):
            obi_embs.append(diarizer.extract_embedding(chunk, sample_rate=obi_sr))

    obi_centroid = np.mean(obi_embs, axis=0)
    obi_centroid /= np.linalg.norm(obi_centroid)
    registry.register_or_update("obi-wan", display_name="Obi-Wan Kenobi", centroid=obi_centroid)
    print(f"    Enrolled Obi-Wan with {len(obi_embs)} speech turns.")

    # 4. Enroll Anakin
    print("[+] Enrolling Anakin Skywalker from anakin_ep2_1...")
    anakin_audio, anakin_sr = sf.read(str(anakin_wav))
    anakin_turns = diarizer.detect_speech_turns(anakin_audio[: 60 * anakin_sr], anakin_sr)
    anakin_embs = []
    for s, e in anakin_turns[:10]:
        chunk = anakin_audio[int(s * anakin_sr) : int(e * anakin_sr)]
        if len(chunk) >= int(0.4 * anakin_sr):
            anakin_embs.append(diarizer.extract_embedding(chunk, sample_rate=anakin_sr))

    anakin_centroid = np.mean(anakin_embs, axis=0)
    anakin_centroid /= np.linalg.norm(anakin_centroid)
    registry.register_or_update("anakin", display_name="Anakin Skywalker", centroid=anakin_centroid)
    print(f"    Enrolled Anakin with {len(anakin_embs)} speech turns.")

    # Check acoustic separation between the two centroids
    cos_sim = float(np.dot(obi_centroid, anakin_centroid))
    print(f"[*] Inter-speaker cosine similarity between Obi-Wan and Anakin: {cos_sim:.3f} (distance = {1.0 - cos_sim:.3f})")

    # 5. Diarize anakin_ep3_1.mp4
    print("[+] Diarizing anakin_ep3_1.mp4 using candidate VTT speech cues...")
    vtt_cues = parse_vtt_cues(str(test_vtt))
    candidate_turns: List[Tuple[float, float]] = [
        (c["start_sec"], c["end_sec"]) for c in vtt_cues if (c["end_sec"] - c["start_sec"]) >= 0.4
    ]
    print(f"    Found {len(candidate_turns)} eligible dialogue cues in VTT.")

    # Run diarization
    intervals = diarizer.diarize(
        test_wav,
        candidate_turns=candidate_turns,
        interactive=False,
    )

    # 6. Analyze and summarize diarization distribution
    speaker_counts = {}
    speaker_durations = {}
    sample_classifications = []

    for idx, iv in enumerate(intervals):
        dur = iv.end_sec - iv.start_sec
        speaker_counts[iv.speaker_id] = speaker_counts.get(iv.speaker_id, 0) + 1
        speaker_durations[iv.speaker_id] = speaker_durations.get(iv.speaker_id, 0.0) + dur

        # Match with VTT cue text for qualitative review
        cue_text = ""
        for c in vtt_cues:
            if abs(c["start_sec"] - iv.start_sec) < 0.1:
                cue_text = c["text"]
                break

        if idx < 20 or iv.speaker_id in ("obi-wan", "anakin"):
            sample_classifications.append({
                "interval": f"{iv.start_sec:.2f}s - {iv.end_sec:.2f}s",
                "speaker": iv.speaker_id,
                "confidence": round(iv.confidence, 3),
                "text": cue_text,
            })

    report = {
        "media_file": str(test_media),
        "total_intervals": len(intervals),
        "inter_speaker_centroid_similarity": round(cos_sim, 4),
        "speaker_turn_distribution": speaker_counts,
        "speaker_duration_seconds": {k: round(v, 2) for k, v in speaker_durations.items()},
        "sample_classifications": sample_classifications[:30],
    }

    report_path = build_dir / "diarization_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n[+] Diarization & Speaker Recognition Results:")
    print(f"    Total Speech Turns Evaluated: {len(intervals)}")
    print(f"    Obi-Wan Turns: {speaker_counts.get('obi-wan', 0)} ({speaker_durations.get('obi-wan', 0.0):.1f}s)")
    print(f"    Anakin Turns:  {speaker_counts.get('anakin', 0)} ({speaker_durations.get('anakin', 0.0):.1f}s)")
    for spk, count in sorted(speaker_counts.items()):
        if spk not in ("obi-wan", "anakin"):
            print(f"    Other / Unresolved Cluster '{spk}': {count} turns ({speaker_durations.get(spk, 0.0):.1f}s)")

    print(f"\n[+] Detailed report saved to: {report_path}")


if __name__ == "__main__":
    main()
