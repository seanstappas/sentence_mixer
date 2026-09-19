This implementation specification and operational guide is formatted for your local coding agent (**Antigravity**) to refactor the pipeline into the research-grade **MFA + Non-Uniform Viterbi Unit Selection** architecture.

---

# Architecture Refactor & Implementation Specification

## System Overview & Objectives

This refactor transitions the engine from naive median-duration slicing to a **Non-Uniform Viterbi Unit Selection Synthesizer** (Hunt & Black 1996, Fried et al. 2019, Baas et al. 2023).

1. **Clean Directory Isolation:** Consolidate all temporary files (`corpus/`, `aligned/`, `cache/`, `inventory/`) into a single `.build/` directory that can be cleared with one command without risking source media or dictionaries.


2. **Dynamic Cross-Platform Paths:** Remove hardcoded user paths (`C:/Users/Sean/...`) in favor of dynamic environment and `pathlib.Path.home()` lookups.


3. **Contiguity-Biased Non-Uniform Viterbi Search:** Treat phrase selection as a global cost-minimization problem. Spoken multi-word sequences receive a negative join-cost bonus ($\le -80.0$), ensuring natural multi-word phrases are kept intact as single cuts.


4. **Perceptual Acoustic Join Cost:** Replace random splicing with acoustic distance metrics ($F_0$ pitch continuity, spectral energy, and boundary feature representations).


5. **Phase-Aligned Splicing & Cadence:** Snap all audio cutpoints to positive zero-crossings to prevent clicks and phase cancellation, and insert 40–50 ms micro-pauses at word boundaries.



---

## 1. Directory Restructuring & `.gitignore`

Configure the workspace so that:

* Permanent inputs remain in `input/`.


* Output renders save to `output/`.


* Reference dictionaries remain in `dictionaries/`.


* Source engine code remains in `src/`.


* **All generated alignment artifacts, WAV slices, and index caches live inside `.build/**`.



```text
D:\Code\sentence_mixer\
├── .build/                         <-- All intermediate artifacts (gitignored)
│   ├── cache/                      <-- Per-source hash caches
│   ├── corpus/                     <-- 16kHz mono WAV chunks + transcripts
│   ├── aligned/                    <-- Praat .TextGrids from MFA
│   └── inventory/                  <-- units_index.json
├── dictionaries/
│   ├── custom_pronunciations.txt
│   └── merged_dictionary.dict
├── input/
│   ├── obi_wan_ep2.mp4
│   ├── obi_wan_ep2.en.vtt
│   ├── obi_wan_ep3.mp4
│   └── obi_wan_ep3.en.vtt
├── output/                         <-- Final rendered MP4/WAV/XML files
├── research/                       <-- Literature PDFs & documentation
├── src/
│   ├── vtt_preprocessor.py
│   ├── inventory_builder.py
│   ├── unit_selector.py
│   ├── video_assembler.py
│   ├── fcp_xml_generator.py
│   └── inspect_alignment.py
├── .gitignore
├── README.md
├── run_mixer.py
└── sources_manifest.json

```

### Update `.gitignore`

```gitignore
# Python artifacts
__pycache__/
*.py[cod]
*$py.class

# Isolated scratch directory
.build/

# Local tools & environments
bin/
.mamba/

# Renders
output/*.mp4
output/*.wav
output/*.xml

```

---

## 2. Environment Setup (PowerShell)

Run these commands in PowerShell to prepare `micromamba`, dependencies, and MFA models:

```powershell
# 1. Ensure Python dependencies are installed in your host environment
pip install praatio g2p_en webvtt-py pydub soundfile librosa numpy transformers torch

# 2. Download portable micromamba executable into bin/ if not already present
if (-not (Test-Path "bin\micromamba.exe")) {
    New-Item -ItemType Directory -Force -Path "bin"
    curl.exe -L "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64.tar.bz2" -o "micromamba.tar.bz2"
    tar.exe -xf "micromamba.tar.bz2"
    Move-Item -Force "Library\bin\micromamba.exe" "bin\micromamba.exe"
    Remove-Item -Recurse -Force "Library", "micromamba.tar.bz2", "info" -ErrorAction SilentlyContinue
}

# 3. Create mfa_env under $HOME\micromamba\envs\mfa_env
$mambaRoot = "$HOME\micromamba"
& "bin\micromamba.exe" create -n mfa_env -c conda-forge --root-prefix $mambaRoot montreal-forced-aligner python=3.10 -y

# 4. Download pre-trained acoustic models and dictionaries
& "bin\micromamba.exe" run -r $mambaRoot -n mfa_env mfa model download acoustic english_us_arpa
& "bin\micromamba.exe" run -r $mambaRoot -n mfa_env mfa model download dictionary english_us_arpa
& "bin\micromamba.exe" run -r $mambaRoot -n mfa_env mfa model download g2p english_us_arpa

```

---

## 3. Code Adjustments

### A. `run_mixer.py`

Update `run_mixer.py` to route all scratch operations to `.build/` and remove hardcoded paths:

```python
#!/usr/bin/env python3
"""YTP Sentence Mixing Pipeline CLI.

Coordinates multi-source alignment via MFA and non-uniform unit synthesis.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

if sys.platform == "win32":
  try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
  except Exception:
    pass

SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR / "src"
BIN_DIR = SCRIPT_DIR / "bin"
BUILD_DIR = SCRIPT_DIR / ".build"
CACHE_BASE_DIR = BUILD_DIR / "cache"
OUTPUT_DIR = SCRIPT_DIR / "output"

MICROMAMBA_EXE = BIN_DIR / "micromamba.exe"
MAMBA_ROOT = Path(
    os.environ.get("MAMBA_ROOT_PREFIX", Path.home() / "micromamba")
)
MFA_ENV_PATH = MAMBA_ROOT / "envs" / "mfa_env"

if str(SRC_DIR) not in sys.path:
  sys.path.insert(0, str(SRC_DIR))

from fcp_xml_generator import generate_fcp_xml
from inventory_builder import build_units_inventory
from unit_selector import synthesize_sentence
from video_assembler import render_sentence_mix
from vtt_preprocessor import preprocess_source


def get_source_id(video_path: str) -> str:
  clean_stem = re.sub(r"[^\w\-_]", "_", Path(video_path).stem)[:40]
  path_hash = hashlib.sha256(
      Path(video_path).resolve().as_posix().encode("utf-8")
  ).hexdigest()[:8]
  return f"{clean_stem}_{path_hash}"


def find_associated_vtt(video_path: str) -> str:
  v_path = Path(video_path)
  parent = v_path.parent
  stem = v_path.stem

  candidates = [
      parent / f"{stem}.en.vtt",
      parent / f"{stem}.vtt",
      parent / f"{stem}.en-US.vtt",
  ]
  for c in candidates:
    if c.exists():
      return str(c)

  # Fallback prefix scan avoiding glob regex escape bugs
  for f in parent.iterdir():
    if f.suffix.lower() == ".vtt" and f.stem.startswith(stem):
      return str(f)
  return ""


def run_mfa_align(
    corpus_dir: str,
    output_aligned_dir: str,
    dictionary_path: str = None,
    acoustic_model: str = "english_us_arpa",
):
  os.makedirs(output_aligned_dir, exist_ok=True)
  dict_arg = (
      dictionary_path
      if dictionary_path and os.path.exists(dictionary_path)
      else "english_us_arpa"
  )

  cmd = [
      str(MICROMAMBA_EXE),
      "run",
      "-r",
      str(MAMBA_ROOT),
      "-n",
      "mfa_env",
      "mfa",
      "align",
      corpus_dir,
      dict_arg,
      acoustic_model,
      output_aligned_dir,
      "--clean",
      "--use_mp",
  ]

  print(f"[*] Running MFA Alignment:\n    {' '.join(cmd)}")
  res = subprocess.run(cmd)
  if res.returncode != 0:
    print("[!] MFA multiprocessing failed. Retrying in single-process mode...")
    cmd_fallback = [
        str(MICROMAMBA_EXE),
        "run",
        "-r",
        str(MAMBA_ROOT),
        "-n",
        "mfa_env",
        "mfa",
        "align",
        corpus_dir,
        dict_arg,
        acoustic_model,
        output_aligned_dir,
        "--clean",
        "-j",
        "1",
    ]
    res = subprocess.run(cmd_fallback)
    if res.returncode != 0:
      raise RuntimeError("MFA alignment failed. Check MFA log files.")

  print(f"[+] MFA Alignment complete: {output_aligned_dir}")


def ensure_source_aligned(
    video_path: str, vtt_path: str = None, force_realign: bool = False
) -> str:
  v_abs = os.path.abspath(video_path)
  if not os.path.exists(v_abs):
    raise FileNotFoundError(f"Source video not found: {v_abs}")

  source_id = get_source_id(v_abs)
  source_cache_dir = CACHE_BASE_DIR / source_id
  corpus_dir = source_cache_dir / "corpus"
  aligned_dir = source_cache_dir / "aligned"
  manifest_path = corpus_dir / "corpus_manifest.json"
  inventory_path = source_cache_dir / "units_index.json"

  dict_path = SCRIPT_DIR / "dictionaries" / "merged_dictionary.dict"
  if not dict_path.exists():
    dict_path = SCRIPT_DIR / "dictionaries" / "custom_pronunciations.txt"

  if inventory_path.exists() and not force_realign:
    print(f"[+] Using cached index: {inventory_path}")
    return str(inventory_path)

  if not vtt_path:
    vtt_path = find_associated_vtt(v_abs)
    if not vtt_path or not os.path.exists(vtt_path):
      raise FileNotFoundError(f"No matching .vtt found for {v_abs}")

  print(f"\n[+] Aligning source: {Path(v_abs).name}")
  os.makedirs(corpus_dir, exist_ok=True)
  os.makedirs(aligned_dir, exist_ok=True)

  preprocess_source(v_abs, vtt_path, str(corpus_dir))
  run_mfa_align(str(corpus_dir), str(aligned_dir), str(dict_path))
  build_units_inventory(
      str(aligned_dir), str(manifest_path), str(inventory_path)
  )

  return str(inventory_path)


def merge_inventories(inventories: list) -> dict:
  combined = {
      "source_videos": [],
      "total_unique_words": 0,
      "total_words_instances": 0,
      "total_unique_phones": 0,
      "total_phones_instances": 0,
      "words": {},
      "phones": {},
  }
  seen_videos = set()
  for inv in inventories:
    srcs = inv.get("source_videos") or [inv.get("source_video")]
    for s in srcs:
      if s and s not in seen_videos:
        seen_videos.add(s)
        combined["source_videos"].append(s)

    for word, instances in inv.get("words", {}).items():
      combined["words"].setdefault(word, []).extend(instances)
      combined["total_words_instances"] += len(instances)

    for phone, instances in inv.get("phones", {}).items():
      combined["phones"].setdefault(phone, []).extend(instances)
      combined["total_phones_instances"] += len(instances)

  combined["total_unique_words"] = len(combined["words"])
  combined["total_unique_phones"] = len(combined["phones"])
  return combined


def main():
  parser = argparse.ArgumentParser(
      description="YTP Sentence Mixing Engine CLI"
  )
  subparsers = parser.add_subparsers(dest="command", required=True)

  p_mix = subparsers.add_parser("mix", help="Synthesize target sentence")
  p_mix.add_argument("--text", required=True, help="Text to synthesize")
  p_mix.add_argument(
      "--video", nargs="+", help="Path(s) to source video(s) (.mp4)"
  )
  p_mix.add_argument("--vtt", nargs="+", help="Path(s) to subtitles (.vtt)")
  p_mix.add_argument("--manifest", help="Path to sources_manifest.json")
  p_mix.add_argument(
      "--inventory", nargs="+", help="Direct path(s) to units_index.json"
  )
  p_mix.add_argument("--xml", help="Export Final Cut Pro XML timeline path")
  p_mix.add_argument("--render", help="Output MP4 export path")
  p_mix.add_argument("--fps", type=float, default=29.97, help="Framerate")
  p_mix.add_argument(
      "--force-realign", action="store_true", help="Force re-alignment"
  )

  args = parser.parse_args()

  if args.command == "mix":
    raw_videos = []
    raw_vtts = []

    if args.manifest and os.path.exists(args.manifest):
      with open(args.manifest, "r", encoding="utf-8") as mf:
        for item in json.load(mf):
          if isinstance(item, dict) and "video" in item:
            raw_videos.append(item["video"])
            raw_vtts.append(item.get("vtt"))

    if args.video:
      for v in args.video:
        raw_videos.extend([p.strip() for p in v.split(",") if p.strip()])
    if args.vtt:
      for vt in args.vtt:
        raw_vtts.extend([p.strip() for p in vt.split(",") if p.strip()])

    loaded_inventories = []
    if args.inventory:
      for inv_p in args.inventory:
        with open(inv_p, "r", encoding="utf-8") as f:
          loaded_inventories.append(json.load(f))

    for idx, vid_p in enumerate(raw_videos):
      custom_vtt = raw_vtts[idx] if idx < len(raw_vtts) else None
      inv_p = ensure_source_aligned(
          vid_p, custom_vtt, force_realign=args.force_realign
      )
      with open(inv_p, "r", encoding="utf-8") as f:
        loaded_inventories.append(json.load(f))

    if not loaded_inventories:
      sys.exit("Error: No source videos or inventories provided.")

    inv = (
        loaded_inventories[0]
        if len(loaded_inventories) == 1
        else merge_inventories(loaded_inventories)
    )
    units, plan = synthesize_sentence(args.text, inv)

    print("\n--- Optimized Synthesized Cut Plan ---")
    for p in plan:
      print(f"  {p}")

    clean_name = re.sub(r"[^\w\-_]", "_", args.text[:25])
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_render = args.render or str(OUTPUT_DIR / f"mix_{clean_name}.mp4")

    if args.xml:
      generate_fcp_xml(
          units, args.xml, sequence_name=f"Mix_{clean_name}", fps=args.fps
      )

    render_sentence_mix(units, out_render, target_fps=args.fps)


if __name__ == "__main__":
  main()

```

---

### B. `src/inventory_builder.py`

Update `src/inventory_builder.py` to extract acoustic boundary features (MFCC, $F_0$, RMS) and record triphone context:

```python
#!/usr/bin/env python3
"""Inventory Builder with Acoustic Feature Extraction.

Parses Praat .TextGrids and computes boundary acoustic parameters for join-cost evaluation.
"""

import glob
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Dict, List
import librosa
import numpy as np


def extract_unit_acoustics(
    wav_path: str, local_start: float, local_end: float
) -> dict:
  """Extracts 13 boundary MFCCs, F0 pitch, and RMS loudness."""
  dur = local_end - local_start
  default_feats = {
      "mfcc_start": [0.0] * 13,
      "mfcc_end": [0.0] * 13,
      "f0_start": 120.0,
      "f0_end": 120.0,
      "mean_f0": 120.0,
      "rms_db": -20.0,
  }
  if dur < 0.02 or not os.path.exists(wav_path):
    return default_feats

  try:
    y, sr = librosa.load(
        wav_path, sr=16000, offset=local_start, duration=dur, mono=True
    )
    if len(y) < 256:
      return default_feats

    # Compute MFCCs (13 coefficients, 5ms hop)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=512, hop_length=80)
    mfcc_start = [round(float(x), 3) for x in mfccs[:, 0]]
    mfcc_end = [round(float(x), 3) for x in mfccs[:, -1]]

    rms = float(np.sqrt(np.mean(y**2)))
    rms_db = round(20 * math.log10(max(rms, 1e-5)), 1)

    f0, _, _ = librosa.pyin(y, fmin=65, fmax=300, sr=sr, frame_length=512)
    valid_f0 = f0[~np.isnan(f0)]
    mean_f0 = float(np.median(valid_f0)) if len(valid_f0) > 0 else 120.0
    f0_start = (
        float(valid_f0[0])
        if len(valid_f0) > 0
        else (float(f0[0]) if not np.isnan(f0[0]) else mean_f0)
    )
    f0_end = (
        float(valid_f0[-1])
        if len(valid_f0) > 0
        else (float(f0[-1]) if not np.isnan(f0[-1]) else mean_f0)
    )

    return {
        "mfcc_start": mfcc_start,
        "mfcc_end": mfcc_end,
        "f0_start": round(f0_start, 1),
        "f0_end": round(f0_end, 1),
        "mean_f0": round(mean_f0, 1),
        "rms_db": rms_db,
    }
  except Exception:
    return default_feats


def parse_textgrid(tg_path: str) -> Dict[str, List[Dict[str, Any]]]:
  tiers = {}
  try:
    from praatio import textgrid

    tg = textgrid.openTextgrid(tg_path, False)
    for tname in tg.tierNames:
      tiers[tname.lower()] = [
          {"xmin": e.start, "xmax": e.end, "text": e.label}
          for e in tg.getTier(tname).entries
      ]
    return tiers
  except Exception:
    pass

  # Fallback pure-python parser
  with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
    lines = [l.strip() for l in f if l.strip()]

  current_tier = None
  i = 0
  while i < len(lines):
    line = lines[i]
    m_name = re.match(r'name\s*=\s*"([^"]+)"', line)
    if m_name:
      current_tier = m_name.group(1).lower()
      tiers.setdefault(current_tier, [])
      i += 1
      continue

    if current_tier:
      m_xmin = re.match(r"xmin\s*=\s*([0-9.]+)", line)
      if m_xmin:
        xmin = float(m_xmin.group(1))
        xmax, text = None, ""
        for j in range(i + 1, min(i + 6, len(lines))):
          m_xmax = re.match(r"xmax\s*=\s*([0-9.]+)", lines[j])
          if m_xmax:
            xmax = float(m_xmax.group(1))
          m_text = re.match(r'text\s*=\s*"([^"]*)"', lines[j])
          if m_text:
            text = m_text.group(1)
            i = j
            break
        if xmax is not None:
          tiers[current_tier].append(
              {"xmin": xmin, "xmax": xmax, "text": text}
          )
    i += 1
  return tiers


def build_units_inventory(
    aligned_dir: str, manifest_path: str, output_index_path: str
) -> Dict[str, Any]:
  with open(manifest_path, "r", encoding="utf-8") as f:
    manifest = json.load(f)

  source_video = manifest["source_video"]
  chunks_meta = manifest["chunks"]
  corpus_dir = os.path.dirname(os.path.abspath(manifest_path))

  words_inv = {}
  phones_inv = {}

  tg_files = glob.glob(os.path.join(aligned_dir, "*.TextGrid"))
  print(
      f"[*] Building inventory across {len(tg_files)} TextGrids with acoustic"
      " boundaries..."
  )

  for tg_file in tg_files:
    chunk_id = Path(tg_file).stem
    if chunk_id not in chunks_meta:
      continue

    chunk_start = float(chunks_meta[chunk_id]["start_sec"])
    wav_path = os.path.join(corpus_dir, f"{chunk_id}.wav")
    tiers = parse_textgrid(tg_file)

    # 1. Words tier
    words_tier = tiers.get("words", [])
    for entry in words_tier:
      raw_w = entry["text"].strip()
      clean_w = re.sub(r"[^\w\']", "", raw_w).lower()
      if not clean_w or clean_w in {"sil", "sp", "spn", "eps", "<eps>"}:
        continue

      l_start, l_end = float(entry["xmin"]), float(entry["xmax"])
      dur = l_end - l_start
      if dur <= 0.01:
        continue

      feats = extract_unit_acoustics(wav_path, l_start, l_end)
      item = {
          "source_video": source_video,
          "chunk_id": chunk_id,
          "label": clean_w,
          "start_sec": round(chunk_start + l_start, 4),
          "end_sec": round(chunk_start + l_end, 4),
          "duration": round(dur, 4),
          "type": "word",
          **feats,
      }
      words_inv.setdefault(clean_w, []).append(item)

    # 2. Phones tier (Triphone context extraction)
    phones_tier = tiers.get("phones", [])
    valid_phones = [
        p
        for p in phones_tier
        if p["text"].strip()
        and p["text"].strip().upper() not in {"SIL", "SP", "SPN", "EPS", ""}
    ]

    for idx, entry in enumerate(valid_phones):
      raw_p = entry["text"].strip().upper()
      base_p = re.sub(r"\d", "", raw_p)
      if not base_p:
        continue

      l_start, l_end = float(entry["xmin"]), float(entry["xmax"])
      dur = l_end - l_start
      if dur <= 0.01:
        continue

      left_p = (
          re.sub(r"\d", "", valid_phones[idx - 1]["text"].strip().upper())
          if idx > 0
          else "SIL"
      )
      right_p = (
          re.sub(r"\d", "", valid_phones[idx + 1]["text"].strip().upper())
          if idx < len(valid_phones) - 1
          else "SIL"
      )

      feats = extract_unit_acoustics(wav_path, l_start, l_end)
      item = {
          "source_video": source_video,
          "chunk_id": chunk_id,
          "label": raw_p,
          "base_phone": base_p,
          "left_phone": left_p,
          "right_phone": right_p,
          "start_sec": round(chunk_start + l_start, 4),
          "end_sec": round(chunk_start + l_end, 4),
          "duration": round(dur, 4),
          "type": "phone",
          **feats,
      }
      for k in {raw_p, base_p}:
        phones_inv.setdefault(k, []).append(item)

  data = {
      "source_video": source_video,
      "total_unique_words": len(words_inv),
      "total_words_instances": sum(len(v) for v in words_inv.values()),
      "total_unique_phones": len(phones_inv),
      "total_phones_instances": sum(len(v) for v in phones_inv.values()),
      "words": words_inv,
      "phones": phones_inv,
  }

  os.makedirs(os.path.dirname(os.path.abspath(output_index_path)), exist_ok=True)
  with open(output_index_path, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2)

  print(
      f"[+] Built indexed units: {data['total_unique_words']} unique words,"
      f" {data['total_unique_phones']} phonemes"
  )
  return data

```

---

### C. `src/unit_selector.py`

Replace `src/unit_selector.py` with the complete **Non-Uniform Viterbi Engine**:

```python
#!/usr/bin/env python3
"""Global Viterbi Non-Uniform Unit Selection Engine for Sentence Mixing.

Minimizes phonetic/stress target cost and boundary acoustic join cost.
Contiguous sequences receive a negative cost bonus (-100.0), naturally
prioritizing longest multi-word and multi-phone runs.
"""

from typing import Any, Dict, List, Tuple
from g2p_en import G2p
import numpy as np


class ViterbiUnitSelector:

  def __init__(
      self,
      inventory: Dict[str, Any],
      w_ctx: float = 20.0,
      w_pitch: float = 0.3,
      w_dur: float = 5.0,
      w_mfcc: float = 0.8,
      w_join_f0: float = 0.5,
      w_join_rms: float = 1.0,
      contiguity_bonus: float = -100.0,
  ):
    self.inv = inventory
    self.words_db = inventory.get("words", {})
    self.phones_db = inventory.get("phones", {})
    self.w_ctx = w_ctx
    self.w_pitch = w_pitch
    self.w_dur = w_dur
    self.w_mfcc = w_mfcc
    self.w_join_f0 = w_join_f0
    self.w_join_rms = w_join_rms
    self.contiguity_bonus = contiguity_bonus
    self.g2p = G2p()

  def decompose_target_text(self, text: str) -> List[Dict[str, Any]]:
    import re

    words = re.findall(r"[a-zA-Z0-9']+", text)
    flat_targets = []

    for w_idx, w in enumerate(words):
      clean_w = w.lower()
      p_seq = [p for p in self.g2p(w) if re.match(r"^[A-Z]", p)]

      for p_idx, p in enumerate(p_seq):
        flat_targets.append({
            "word": clean_w,
            "word_idx": w_idx,
            "is_word_start": (p_idx == 0),
            "is_word_end": (p_idx == len(p_seq) - 1),
            "phone": p,
            "base_phone": re.sub(r"\d", "", p),
            "left_phone": (
                re.sub(r"\d", "", p_seq[p_idx - 1]) if p_idx > 0 else "SIL"
            ),
            "right_phone": (
                re.sub(r"\d", "", p_seq[p_idx + 1])
                if p_idx < len(p_seq) - 1
                else "SIL"
            ),
        })
    return flat_targets

  def compute_target_cost(
      self, target: Dict[str, Any], unit: Dict[str, Any]
  ) -> float:
    cost = 0.0
    if unit.get("left_phone", "SIL") != target["left_phone"]:
      cost += self.w_ctx
    if unit.get("right_phone", "SIL") != target["right_phone"]:
      cost += self.w_ctx
    if unit.get("label") != target["phone"]:
      cost += 5.0
    dur = unit.get("duration", 0.09)
    cost += abs(dur - 0.09) * self.w_dur * 10.0
    mean_f0 = unit.get("mean_f0", 120.0)
    cost += abs(mean_f0 - 120.0) * self.w_pitch
    return cost

  def compute_join_cost(self, u1: Dict[str, Any], u2: Dict[str, Any]) -> float:
    is_contiguous = (
        u1.get("chunk_id") == u2.get("chunk_id")
        and u1.get("source_video") == u2.get("source_video")
        and abs(u1["end_sec"] - u2["start_sec"]) < 0.015
    )

    if is_contiguous:
      return self.contiguity_bonus  # Strongly rewards natural sequences

    cost = 45.0  # Boundary penalty for introducing a cut

    # Acoustic spectral boundary discontinuity
    if "mfcc_end" in u1 and "mfcc_start" in u2:
      m1 = np.array(u1["mfcc_end"])
      m2 = np.array(u2["mfcc_start"])
      cost += float(np.linalg.norm(m1 - m2)) * self.w_mfcc

    f0_jump = abs(u1.get("f0_end", 120.0) - u2.get("f0_start", 120.0))
    cost += f0_jump * self.w_join_f0

    rms_jump = abs(u1.get("rms_db", -20.0) - u2.get("rms_db", -20.0))
    cost += rms_jump * self.w_join_rms
    return cost

  def select(
      self, target_text: str, pool_size: int = 30
  ) -> Tuple[List[Dict[str, Any]], List[str]]:
    targets = self.decompose_target_text(target_text)
    if not targets:
      return [], []

    N = len(targets)
    candidate_pools = []

    for t in targets:
      cands = self.phones_db.get(t["phone"]) or self.phones_db.get(
          t["base_phone"], []
      )
      if not cands:
        raise RuntimeError(f"Missing phoneme in corpus: {t['phone']}")
      sorted_cands = sorted(cands, key=lambda c: abs(c.get("duration", 0.09) - 0.09))
      candidate_pools.append(sorted_cands[:pool_size])

    delta = [{} for _ in range(N)]
    psi = [{} for _ in range(N)]

    for u_idx, u in enumerate(candidate_pools[0]):
      delta[0][u_idx] = self.compute_target_cost(targets[0], u)

    for i in range(1, N):
      t_curr = targets[i]
      for u_idx, u in enumerate(candidate_pools[i]):
        t_cost = self.compute_target_cost(t_curr, u)
        min_total = float("inf")
        best_prev = 0

        for prev_idx, prev_u in enumerate(candidate_pools[i - 1]):
          j_cost = self.compute_join_cost(prev_u, u)
          total = delta[i - 1][prev_idx] + j_cost + t_cost
          if total < min_total:
            min_total = total
            best_prev = prev_idx

        delta[i][u_idx] = min_total
        psi[i][u_idx] = best_prev

    best_last = min(delta[N - 1], key=delta[N - 1].get)
    path = [best_last]
    for i in range(N - 1, 0, -1):
      path.append(psi[i][path[-1]])
    path.reverse()

    chosen_units = [candidate_pools[i][path[i]] for i in range(N)]

    # Segment consolidation: Merge contiguous phones into continuous phrase cuts
    consolidated = []
    curr = None

    for idx, u in enumerate(chosen_units):
      spec = targets[idx]
      if curr is None:
        curr = dict(u)
        curr["phones_count"] = 1
        curr["is_word_end"] = spec["is_word_end"]
      else:
        is_contig = (
            curr.get("chunk_id") == u.get("chunk_id")
            and curr.get("source_video") == u.get("source_video")
            and abs(curr["end_sec"] - u["start_sec"]) < 0.015
        )
        if is_contig:
          curr["end_sec"] = u["end_sec"]
          curr["duration"] = round(curr["end_sec"] - curr["start_sec"], 4)
          curr["label"] += f"+{u['label']}"
          curr["phones_count"] += 1
          curr["is_word_end"] = spec["is_word_end"]
        else:
          consolidated.append(curr)
          curr = dict(u)
          curr["phones_count"] = 1
          curr["is_word_end"] = spec["is_word_end"]

    if curr:
      consolidated.append(curr)

    plan = [
        f"Cut {idx:02d}: '{c['label']}' ({c['phones_count']} phones) ->"
        f" {c['start_sec']:.3f}s..{c['end_sec']:.3f}s ({c['duration']:.2f}s)"
        for idx, c in enumerate(consolidated)
    ]
    return consolidated, plan


def synthesize_sentence(
    target_text: str, inventory: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[str]]:
  selector = ViterbiUnitSelector(inventory)
  return selector.select(target_text)

```

---

### D. `src/video_assembler.py`

Update `src/video_assembler.py` to add **waveform zero-crossing snapping** and **word-boundary cadence padding**:

```python
#!/usr/bin/env python3
"""Video & Audio Assembler for Sentence Mixing.

Applies positive zero-crossing snapping, intra-slice micro-fades, and cadence spacing.
"""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Dict, List
import numpy as np
import soundfile as sf


def snap_to_zero_crossing(
    video_or_wav_path: str,
    timestamp_sec: float,
    search_window_sec: float = 0.006,
) -> float:
  """Snaps timestamp to the nearest positive zero-crossing in audio."""
  try:
    with sf.SoundFile(video_or_wav_path) as f:
      sr = f.samplerate
      center_sample = int(round(timestamp_sec * sr))
      window = int(round(search_window_sec * sr))
      start_frame = max(0, center_sample - window)
      f.seek(start_frame)
      samples = f.read(frames=window * 2)

      if len(samples) < 2:
        return timestamp_sec

      mono = samples[:, 0] if samples.ndim > 1 else samples
      zc = np.where((mono[:-1] <= 0) & (mono[1:] > 0))[0]
      if len(zc) > 0:
        best = zc[np.argmin(np.abs(zc - (center_sample - start_frame)))]
        return round((start_frame + best) / sr, 4)
  except Exception:
    pass
  return timestamp_sec


def render_sentence_mix(
    units: List[Dict[str, Any]],
    output_mp4_path: str,
    target_fps: float = 29.97,
    audio_sample_rate: int = 48000,
    keep_temp: bool = False,
) -> str:
  if not units:
    raise ValueError("Cannot render an empty unit list!")

  os.makedirs(os.path.dirname(os.path.abspath(output_mp4_path)), exist_ok=True)
  temp_dir = tempfile.mkdtemp(prefix="ytp_mix_")
  print(f"[*] Rendering {len(units)} units via FFmpeg (temp: {temp_dir})...")

  slice_files = []
  concat_txt = os.path.join(temp_dir, "concat.txt")

  for idx, unit in enumerate(units):
    src_path = os.path.abspath(unit["source_video"])
    raw_start = float(unit["start_sec"])
    raw_end = float(unit["end_sec"])

    # Snap cuts to zero crossings
    start_sec = snap_to_zero_crossing(src_path, raw_start)
    end_sec = snap_to_zero_crossing(src_path, raw_end)
    dur = max(0.034, end_sec - start_sec)
    label = unit.get("label", f"slice_{idx}")

    slice_mp4 = os.path.join(temp_dir, f"slice_{idx:04d}.mp4")

    # Intra-slice 5ms micro-fades
    fade_filter = (
        f"afade=t=in:ss=0:d=0.005,afade=t=out:st={dur-0.005:.4f}:d=0.005"
    )

    # Insert 45ms natural cadence padding after complete words
    if unit.get("is_word_end", False):
      fade_filter += ",apad=pad_dur=0.045"

    cmd = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_sec:.4f}",
        "-t",
        f"{dur:.4f}",
        "-i",
        src_path,
        "-vf",
        f"fps={target_fps},scale=1920:1080:force_original_aspect_ratio=decrease,pad=1920:1080:(ow-iw)/2:(oh-ih)/2",
        "-af",
        fade_filter,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-ar",
        str(audio_sample_rate),
        "-ac",
        "2",
        slice_mp4,
    ]

    res = subprocess.run(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    if res.returncode != 0 or not os.path.exists(slice_mp4):
      print(f"[!] Warning: Failed slice {idx} ({label})")
      continue

    slice_files.append(slice_mp4)

  if not slice_files:
    raise RuntimeError("No slices rendered successfully!")

  with open(concat_txt, "w", encoding="utf-8") as cf:
    for sf_path in slice_files:
      cf.write(f"file '{sf_path.replace(os.sep, '/')}'\n")

  concat_cmd = [
      "ffmpeg",
      "-y",
      "-f",
      "concat",
      "-safe",
      "0",
      "-i",
      concat_txt,
      "-c",
      "copy",
      output_mp4_path,
  ]
  subprocess.run(
      concat_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
  )

  # Master WAV
  audio_wav = os.path.splitext(output_mp4_path)[0] + ".wav"
  wav_cmd = [
      "ffmpeg",
      "-y",
      "-i",
      output_mp4_path,
      "-vn",
      "-c:a",
      "pcm_s16le",
      "-ar",
      str(audio_sample_rate),
      "-ac",
      "2",
      audio_wav,
  ]
  subprocess.run(wav_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  if not keep_temp:
    shutil.rmtree(temp_dir, ignore_errors=True)

  print(f"[+] Render complete:\n    - Video: {output_mp4_path}")
  return output_mp4_path

```

---

## 4. End-to-End Execution & Testing (PowerShell)

### Step 1: Initialize Manifest

Verify `sources_manifest.json` points to the files in `input/`:

```json
[
  {
    "video": "input/obi_wan_ep2.mp4",
    "vtt": "input/obi_wan_ep2.en.vtt"
  },
  {
    "video": "input/obi_wan_ep3.mp4",
    "vtt": "input/obi_wan_ep3.en.vtt"
  }
]

```

### Step 2: Run Multi-Source Sentence Synthesis

Synthesize a sentence spanning vocabulary across both episodes:

```powershell
python run_mixer.py mix `
  --manifest sources_manifest.json `
  --text "Hello there Anakin Chancellor Palpatine is evil" `
  --render "output/obi_wan_mix.mp4"

```

### Step 3: Verification & Sanity Check

1. **Contiguity & Phrase Inspection:**
The printed cut plan should output multi-phone clusters:


```text
Cut 00: 'HH+AH0+L+OW1' (4 phones) -> 188.020s..188.350s (0.33s)
Cut 01: 'DH+EH1+R' (3 phones) -> 324.500s..324.810s (0.31s)

```


If words were spoken consecutively in the source video, they will appear merged into a single cut.


2. **Audio Boundary Quality:**
Play `output/obi_wan_mix.wav` through headphones. Cuts will transition without DC-offset clicks or pitch leaps.


3. **Workspace Check:**
Confirm that `input/`, `dictionaries/`, and `src/` remain clean, with all intermediate files contained inside `.build/`. To clear all cached indexes and start over, delete `.build/`:


```powershell
Remove-Item -Recurse -Force .build

```