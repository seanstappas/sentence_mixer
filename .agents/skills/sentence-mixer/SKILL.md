---
name: sentence-mixer
description: >-
  Operational cheatsheet and runbook for the MFA sentence mixing engine.
  Use when running sentence mixes, ingesting video corpora or whole movies, labeling movie
  subtitle speakers, setting up continuous watch mode, cleaning caches, debugging unit
  selection errors, or adding custom phonetic pronunciations.
---

# Sentence Mixer Operational Guide

Taking over the project? Read `plans/10_05_handoff.md` first (state, measured decisions, traps, open work).

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

Useful mix options:
- `--resolution 4k` (default 1080p); `--fps auto` (default) keeps the source frame rate.
- `--pause-style freeze|continue|off`: punctuation pauses hold the frame (default), let the clip play on while nobody speaks next in the source, or are removed.
- `--release-tail-ms 100` (default): cuts extend into following silence so word endings (f, s, t) are not clipped; `0` disables.
- `--audio-track recombined` (default, keeps the film's music and effects), `raw`, or `vocals`.

Verify a render by ear and with ASR (whisper-tiny WER is noisy; compare several sentences):
```powershell
.venv\Scripts\python.exe tools\listen_verify.py -i output\result.mp4 -e "Your sentence to mix"
```

### 2. Multi-Speaker Dialogue Scenes & Active Word Highlighting
Render conversations between multiple speakers with frame-accurate edit plans and burned-in word-by-word highlighted subtitles:
```powershell
# Run the dialogue runner script for pre-configured comic scenes
.venv\Scripts\python.exe tools\render_dialogue_clips.py --clip 1       # Episode III: Anakin & Mace Windu
.venv\Scripts\python.exe tools\render_dialogue_clips.py --clip 2       # Episode VI: Luke, Palpatine & Vader
.venv\Scripts\python.exe tools\render_dialogue_clips.py --clip all --resolution 1080p
```
Or use the Python API in `src/dialogue_mixer.py`:
```python
from dialogue_mixer import parse_script_text, synthesize_and_render_dialogue
from run_mixer import load_or_combine_inventories

script = """
luke: You've failed, Your Highness. I'm a Jedi Master, like my father before me.
palpatine: Actually, Anakin wasn't a Jedi Master. He was only a Jedi Knight.
"""
lines = parse_script_text(script)
inv, _ = load_or_combine_inventories([path_to_inventory])
synthesize_and_render_dialogue(
    lines=lines,
    inventory_index=inv,
    output_video_path=Path("output/scene.mp4"),
    burn_subtitles=True,
    highlight_words=True,
)
```
- Subtitles are generated in ASS format with words dynamically highlighted in yellow (`&H0000FFFF&`) as they are spoken.
- Conversational pauses (`pause_between_sec=0.45`) are inserted between turns without audio dropouts.

### 3. Ingestion & Continuous Watch Mode
```powershell
# Continuous watch mode: monitors folder and auto-ingests new clips
.venv\Scripts\python.exe run_mixer.py ingest `
  --folder "vids/" `
  --watch `
  --poll-interval 5.0
```

### 4. Whole Movies (long MKV sources)
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
.venv\Scripts\python.exe run_mixer.py review-speakers "input\movie.mkv" --only flagged --audio vocals

# Mix in 4K, optionally restricted to a scene or range
.venv\Scripts\python.exe run_mixer.py mix --all-sources --speaker obi-wan `
  --text "Your sentence" --resolution 4k --scene 12 --render "output\obiwan.mp4"
```
- Speaker IDs come from `config/characters/star_wars.json` (e.g. `anakin`, `obi-wan`, `padme`, `palpatine`; `sidious` and `vader` are separate voices). The labeling tool prints unrecognized names: add them as aliases there and rerun it (about 2 s).
- Relabeling (rerunning the tool, or `review-speakers` answers) rebuilds labels and the inventory on the next ingest/mix but reuses the MFA alignment.
- `review-speakers` plays each line without blocking (type while it plays), saves every answer immediately, and Enter keeps the text label. `--audio mix|center|vocals` picks what is played. A voice lead under about 0.05 means the voice model is unsure.
- Movie analysis (alignment, voice checks, join features) uses the separated dialogue track; the center channel is only a fallback. Lines whose voice the separation removed are marked `separation_dropped` and excluded.
- Never print or commit labeled VTT/report/transcript/screenplay text; they are copyrighted and git-ignored.

### 5. Cleaning Caches
```powershell
.venv\Scripts\python.exe run_mixer.py clean "input\movie.mkv" --dry-run   # always dry-run first
.venv\Scripts\python.exe run_mixer.py clean "input\movie.mkv" --keep-speakers
```
`clean` matches only the exact source (by file, stem, or source ID) and removes its alignment, inventory, vocals, and movie caches (`.build/movies/<hash>/`, about 2 GB per movie; re-ingest takes about 1.5 h). It keeps labeled VTTs and reviewed labels.

### 6. Adding Proper Nouns & Custom Pronunciations
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
- **Word endings sound clipped**:
  Forced alignment often ends final consonants early. The release tail (`--release-tail-ms`, default 100) extends cuts into following silence; raise it for slow speakers, or check the `.plan.json` `source_end_sample` of the word against the source.
- **Frozen frame during pauses is distracting**:
  Use `--pause-style continue` (clip plays on when no speech follows) or `--pause-style off`.
- **Movie: a correct text label was flagged by voice**:
  Text labels win unless voice evidence is strong on every count; if a flag is still wrong, answer it in `review-speakers` (Enter keeps the text label).
- **Movie: `recombined` audio unavailable**:
  Stems are separated per block during ingest; `speakers`/ingest output says whether stems are complete. Re-run ingest (it resumes), or use `--audio-track raw`.
