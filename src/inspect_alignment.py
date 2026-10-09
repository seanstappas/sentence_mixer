#!/usr/bin/env python3
"""
Alignment Inspector for Sentence Mixing
Inspects phonetic and word intervals in Praat TextGrids via terminal table or waveform/spectrogram plots.
"""

import argparse
import json
import os
import sys

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


from typing import Optional


def inspect_chunk(
    chunk_id: str,
    aligned_dir: str,
    corpus_dir: Optional[str] = None,
    plot_path: Optional[str] = None
):
    from praatio import textgrid

    # Normalize chunk id (e.g. 0 -> chunk_0000)
    if chunk_id.isdigit():
        chunk_name = f"chunk_{int(chunk_id):04d}"
    else:
        chunk_name = os.path.splitext(os.path.basename(chunk_id))[0]

    tg_path = os.path.join(aligned_dir, f"{chunk_name}.TextGrid")
    if not os.path.exists(tg_path):
        sys.exit(f"Error: TextGrid not found at {tg_path}")

    tg = textgrid.openTextgrid(tg_path, False)

    # Read manifest offset if available
    offset_sec = 0.0
    text_content = ""
    if corpus_dir:
        manifest_p = os.path.join(corpus_dir, "corpus_manifest.json")
        if os.path.exists(manifest_p):
            with open(manifest_p, "r", encoding="utf-8") as mf:
                mdata = json.load(mf)
                if chunk_name in mdata.get("chunks", {}):
                    offset_sec = mdata["chunks"][chunk_name]["start_sec"]
                    text_content = mdata["chunks"][chunk_name]["text"]

    print("=" * 70)
    print(f" ALIGNMENT INSPECTION: {chunk_name}")
    if text_content:
        print(f" Utterance text: \"{text_content}\"")
    if offset_sec > 0:
        print(f" Global video time offset: {offset_sec:.3f}s")
    print("=" * 70)

    words_tier = tg.getTier("words")
    phones_tier = tg.getTier("phones")

    print("\n--- WORDS TIER ---")
    print(f" {'Word':<18} | {'Local Start':<11} | {'Local End':<11} | {'Global Time':<16} | {'Duration'}")
    print("-" * 75)
    for entry in words_tier.entries:
        if not entry.label.strip():
            continue
        dur = entry.end - entry.start
        g_start = offset_sec + entry.start
        g_end = offset_sec + entry.end
        print(f" {entry.label:<18} | {entry.start:7.3f}s   | {entry.end:7.3f}s   | {g_start:7.3f}s..{g_end:7.3f}s | {dur:6.3f}s")

    print("\n--- PHONES TIER (Acoustic Boundaries) ---")
    print(f" {'Phone':<18} | {'Local Start':<11} | {'Local End':<11} | {'Global Time':<16} | {'Duration'}")
    print("-" * 75)
    for entry in phones_tier.entries:
        if not entry.label.strip():
            continue
        dur = entry.end - entry.start
        g_start = offset_sec + entry.start
        g_end = offset_sec + entry.end
        print(f" {entry.label:<18} | {entry.start:7.3f}s   | {entry.end:7.3f}s   | {g_start:7.3f}s..{g_end:7.3f}s | {dur:6.3f}s")

    if plot_path and corpus_dir:
        wav_path = os.path.join(corpus_dir, f"{chunk_name}.wav")
        if os.path.exists(wav_path):
            plot_alignment(wav_path, tg, plot_path, title=f"Alignment: {chunk_name} ({text_content})")


def plot_alignment(wav_path: str, tg, output_png: str, title: str = "Alignment"):
    """Generates a waveform and spectrogram plot with overlaid word and phoneme boundary markers."""
    import matplotlib.pyplot as plt
    import numpy as np
    import soundfile as sf

    audio, sr = sf.read(wav_path)
    if len(audio.shape) > 1:
        audio = audio[:, 0]
    duration = len(audio) / sr
    time_axis = np.linspace(0, duration, len(audio))

    fig, (ax_wave, ax_spec) = plt.subplots(2, 1, figsize=(14, 7), sharex=True, gridspec_kw={"height_ratios": [1, 2]})

    # Waveform
    ax_wave.plot(time_axis, audio, color="#2b5c8f", lw=0.8)
    ax_wave.set_ylabel("Amplitude")
    ax_wave.set_title(title, fontsize=13, fontweight="bold")
    ax_wave.grid(True, alpha=0.3)

    # Spectrogram
    Pxx, freqs, bins, im = ax_spec.specgram(audio, NFFT=512, Fs=sr, noverlap=384, cmap="inferno")
    ax_spec.set_ylabel("Frequency (Hz)")
    ax_spec.set_xlabel("Time (seconds)")
    ax_spec.set_ylim(0, 8000)

    # Draw word boundaries on waveform
    words_tier = tg.getTier("words")
    for entry in words_tier.entries:
        if not entry.label.strip():
            continue
        ax_wave.axvline(entry.start, color="red", ls="--", alpha=0.7, lw=1.2)
        ax_wave.axvline(entry.end, color="red", ls="--", alpha=0.7, lw=1.2)
        mid_time = (entry.start + entry.end) / 2
        ax_wave.text(mid_time, ax_wave.get_ylim()[1] * 0.75, entry.label,
                     ha="center", va="center", color="darkred", fontweight="bold", fontsize=11,
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="red"))

    # Draw phoneme boundaries on spectrogram
    phones_tier = tg.getTier("phones")
    for entry in phones_tier.entries:
        if not entry.label.strip():
            continue
        ax_spec.axvline(entry.start, color="cyan", ls=":", alpha=0.8, lw=1.0)
        ax_spec.axvline(entry.end, color="cyan", ls=":", alpha=0.8, lw=1.0)
        mid_time = (entry.start + entry.end) / 2
        ax_spec.text(mid_time, 7000, entry.label,
                     ha="center", va="center", color="white", fontweight="bold", fontsize=9,
                     bbox=dict(boxstyle="square,pad=0.15", facecolor="#111111", alpha=0.75, edgecolor="cyan"))

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_png)), exist_ok=True)
    plt.savefig(output_png, dpi=180)
    plt.close()
    print(f"[+] Saved visual alignment spectrogram to: {output_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inspect MFA phonetic and word boundaries.")
    parser.add_argument("--chunk", default="0", help="Chunk ID or index (e.g. 0, chunk_0000)")
    parser.add_argument("--aligned", default="./aligned", help="Aligned TextGrids directory")
    parser.add_argument("--corpus", default="./corpus", help="Corpus directory containing WAVs and manifest")
    parser.add_argument("--plot", help="Path to save visual spectrogram image (e.g. inspect.png)")
    args = parser.parse_args()

    inspect_chunk(args.chunk, args.aligned, args.corpus, args.plot)
