---
name: sentence-mixer
description: >-
  Operational cheatsheet and runbook for the MFA sentence mixing engine.
  Use when running sentence mixes, ingesting video corpora, setting up continuous watch mode,
  debugging unit selection errors, or adding custom phonetic pronunciations.
---

# Sentence Mixer Operational Guide

## Quick CLI Cheatsheet

### 1. Mixing Dialogue
```powershell
# Mix using folder discovery and default span selector
.venv\Scripts\python.exe run_mixer.py mix `
  --folder "vids/" `
  --speaker "character_name" `
  --text "Your sentence to mix" `
  --render "output/result.mp4"

# Mix across all cached sources in library
.venv\Scripts\python.exe run_mixer.py mix `
  --all-sources `
  --speaker "character_name" `
  --text "Your sentence to mix" `
  --render "output/result.mp4"
```

### 2. Ingestion & Continuous Watch Mode
```powershell
# Continuous watch mode: monitors folder and auto-ingests new clips
.venv\Scripts\python.exe run_mixer.py ingest `
  --folder "vids/" `
  --watch `
  --poll-interval 5.0
```

### 3. Adding Proper Nouns & Custom Pronunciations
Add custom phonetic pronunciations to `dictionaries/custom_pronunciations.txt`:
```text
windu	W IH1 N D UW0
yoda	Y OW1 D AH0
```
Then invalidate affected alignment with `--force-realign`.

## Troubleshooting Common Issues
- **`ValueError: Could not find a continuous phonetic path`**:
  Occurs when the speaker corpus lacks a required phoneme and cognate fallback was disabled or missing. Ensure Level 3 cognate fallbacks (`G` <-> `K`, `NG` <-> `N`, `V` <-> `F`, unstressed `AH0` <-> `IH0` <-> `EH0`) are permitted.
- **Dropped Subtitle Words**:
  Check if subtitle duration is shorter than chunk audio. Prefer fixing the subtitle file (or using uploader-written subtitles instead of YouTube auto-captions). Whisper tail enrichment is opt-in with `--asr-fill-tails`; its words are unverified and can include hallucinations or other speakers, so use it sparingly. Changing the setting re-aligns affected sources automatically on the next ingest or mix.
- **Robotic / Fragmented Speech**:
  The target sentence contains too many OOV words. Check `EditPlan` (`.plan.json`) to inspect whether selections are `[PHRASE]`, `[WORD]`, `[SUBWORD]`, or `[PHONE]`.
