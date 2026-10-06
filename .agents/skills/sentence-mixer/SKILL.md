---
name: sentence-mixer
description: >-
  Operational cheatsheet and runbook for the MFA sentence mixing engine.
  Use when running sentence mixes, ingesting video corpora or whole movies, labeling movie
  subtitle speakers, setting up continuous watch mode, cleaning caches, debugging unit
  selection errors, or adding custom phonetic pronunciations.
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

### 3. Whole Movies (long MKV sources)
```powershell
# (Recommended, once per movie) Speaker-label the movie's own subtitles from local
# transcripts/screenplays. Ingest picks the result up automatically from this path.
.venv\Scripts\python.exe tools\label_subtitles.py --video "input\movie.mkv" `
  --transcript input\movie_transcript.txt --screenplay .build\scripts\movie_screenplay.html `
  --output ".build\subtitles\labeled\<movie file stem>.en.vtt"

# Ingest a whole movie (cached per 10-minute block; resumable)
.venv\Scripts\python.exe run_mixer.py ingest "input\movie.mkv"
# Only part of a movie
.venv\Scripts\python.exe run_mixer.py ingest "input\movie.mkv" --range 01:30:00-01:35:00

# Inspect / fix labels
.venv\Scripts\python.exe run_mixer.py speakers
.venv\Scripts\python.exe run_mixer.py scenes "input\movie.mkv"
.venv\Scripts\python.exe run_mixer.py review-speakers "input\movie.mkv" --only flagged

# Mix in 4K, optionally restricted to a scene or range
.venv\Scripts\python.exe run_mixer.py mix --all-sources --speaker obi-wan `
  --text "Your sentence" --resolution 4k --scene 12 --render "output\obiwan.mp4"
```
- Speaker IDs come from `config/characters/star_wars.json` (e.g. `anakin`, `obi-wan`, `padme`, `palpatine`; `sidious` and `vader` are separate voices). The labeling tool prints unrecognized names: add them as aliases there and rerun it (about 2 s).
- Relabeling (rerunning the tool, or `review-speakers` answers) rebuilds labels and the inventory on the next ingest/mix but reuses the MFA alignment.
- Never print or commit labeled VTT/report/transcript/screenplay text; they are copyrighted and git-ignored.

### 4. Cleaning Caches
```powershell
.venv\Scripts\python.exe run_mixer.py clean "input\movie.mkv" --dry-run   # always dry-run first
.venv\Scripts\python.exe run_mixer.py clean "input\movie.mkv" --keep-speakers
```
`clean` matches only the exact source (by file, stem, or source ID) and removes its alignment, inventory, vocals, and movie caches (`.build/movies/<hash>/`, about 2 GB per movie; re-ingest takes about 1.5 h). It keeps labeled VTTs and reviewed labels.

### 5. Adding Proper Nouns & Custom Pronunciations
Add custom phonetic pronunciations to `dictionaries/custom_pronunciations.txt`:
```text
windu	W IH1 N D UW0
yoda	Y OW1 D AH0
```
The staged dictionary is rebuilt from file content, so affected sources re-align automatically on the next ingest or mix (no `--force-realign` needed). For movies this re-aligns every block (about 30 min per movie).

## Troubleshooting Common Issues
- **`ValueError: Could not find a continuous phonetic path`**:
  Occurs when the speaker corpus lacks a required phoneme and cognate fallback was disabled or missing. Ensure Level 3 cognate fallbacks (`G` <-> `K`, `NG` <-> `N`, `V` <-> `F`, unstressed `AH0` <-> `IH0` <-> `EH0`) are permitted.
- **Dropped Subtitle Words**:
  Check if subtitle duration is shorter than chunk audio. Prefer fixing the subtitle file (or using uploader-written subtitles instead of YouTube auto-captions). Whisper tail enrichment is opt-in with `--asr-fill-tails`; its words are unverified and can include hallucinations or other speakers, so use it sparingly. Changing the setting re-aligns affected sources automatically on the next ingest or mix.
- **Robotic / Fragmented Speech**:
  The target sentence contains too many OOV words. Check `EditPlan` (`.plan.json`) to inspect whether selections are `[PHRASE]`, `[WORD]`, `[SUBWORD]`, or `[PHONE]`.
- **Movie: speaker has little usable speech / wrong voice in a mix**:
  Run `run_mixer.py speakers` to see usable seconds per character and label counts. Lines flagged `voice_mismatch` are excluded automatically; fix labels with `review-speakers` (answers apply on the next mix without re-aligning).
- **Movie: `recombined` audio unavailable**:
  Stems are separated per block during ingest; `speakers`/ingest output says whether stems are complete. Re-run ingest (it resumes), or use `--audio-track raw`.
