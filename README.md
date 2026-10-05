# Montreal Forced Aligner (MFA) Sentence Mixing Toolkit

An automated, unit-selection sentence mixing engine for video and audio remixing. Uses Montreal Forced Aligner (MFA) to derive acoustic boundaries for words and phonemes from source video and WebVTT subtitle tracks. Prioritizes whole-word and phrase unit selection with automatic phonetic fallback for missing words.

Outputs synchronized **rendered MP4 video / master WAV audio** (and optional Premiere Pro / Final Cut Pro 7 XML timelines).

> [!IMPORTANT]
> **Source-Preserving Invariant**: Analysis inputs (Diarization, ASR, WavLM, phone recognizers, microphone capture, and prosody guides) NEVER donate audio or video to output renders. Output speech is derived strictly from user source recordings.

---

## Overview & Principles

The pipeline selects and edits recorded speech and video so a speaker appears to say target text as naturally as the available recordings allow. No TTS, voice cloning, generated mouth movement, or generative inpainting:
- **Any video format**: MP4, WebM, MKV, MOV, etc.
- **Any English speech content**: Talks, lectures, interviews, gaming streams, movies, or commentary.
- **Any WebVTT subtitles**: Automatically parses standard WebVTT or YouTube auto-generated captions, de-duplicates rolling lines, discards non-speech bracketed markers (`[Music]`, `[Applause]`), and extracts 16kHz mono audio chunks.
- **Content-Aware Caching**: Three-layer content hashing (audio content bytes, MFA alignment, and speaker-labeled inventory) eliminates redundant computation while ensuring immediate cache invalidation on edits.
- **Phonetic Fallback for Any Vocabulary**: If a word in your target sentence was never spoken in the source video, the engine automatically breaks the word down into ARPAbet phonemes using CMU/G2P (`g2p_en`) and splices together matching phonetic intervals from the speaker's vocal track.

---

## System Prerequisites & Setup (Fresh Installation Guide)

Follow these steps if deploying to a new machine or environment.

### 1. System Requirements
- **OS**: Windows 10/11, macOS, or Linux
- **Python**: 3.10+ (for host orchestration and unit selection)
- **FFmpeg**: Must be installed and available in system `PATH` (`ffmpeg -version`)

### 2. Python Packages (Host Environment)
Install the required helper libraries into your primary Python environment:
```bash
pip install praatio g2p_en webvtt-py pydub matplotlib soundfile
```
*(On first run, `g2p_en` downloads standard NLTK tagger data. If prompted, run `python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng')"`).*

### 3. MFA Environment Setup (via Micromamba or Conda)
Montreal Forced Aligner relies on Kaldi, OpenFST, and C++ bindings (`kalpy`, `pynini`) distributed via `conda-forge`.

#### Option A: Portable Setup via Micromamba (Recommended on Windows — No Admin Required)
On Windows PowerShell:
```powershell
# 1. Download portable micromamba executable
curl.exe -L "https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-win-64.tar.bz2" -o "micromamba.tar.bz2"
tar.exe -xf "micromamba.tar.bz2"
Move-Item -Force "Library\bin\micromamba.exe" "bin\micromamba.exe"
Remove-Item -Recurse -Force "Library", "micromamba.tar.bz2", "info" -ErrorAction SilentlyContinue

# 2. Create the isolated mfa_env environment
& "bin\micromamba.exe" create -n mfa_env -c conda-forge --root-prefix "C:\Users\Sean\micromamba" montreal-forced-aligner python=3.10 -y

# 3. Download pre-trained ARPA acoustic, dictionary, and G2P models
& "bin\micromamba.exe" run -r "C:\Users\Sean\micromamba" -n mfa_env mfa model download acoustic english_us_arpa
& "bin\micromamba.exe" run -r "C:\Users\Sean\micromamba" -n mfa_env mfa model download dictionary english_us_arpa
& "bin\micromamba.exe" run -r "C:\Users\Sean\micromamba" -n mfa_env mfa model download g2p english_us_arpa
```

#### Option B: Standard Conda / Mamba (If already installed on system)
```bash
conda create -n mfa_env -c conda-forge montreal-forced-aligner python=3.10 -y
conda activate mfa_env
mfa model download acoustic english_us_arpa
mfa model download dictionary english_us_arpa
mfa model download g2p english_us_arpa
```

---

## Instructions for AI Agents & Developers

Agents and developers can execute `run_mixer.py` using system Python or the virtual environment (`.venv\Scripts\python.exe` on Windows).

### 1. Basic Sentence Mixing (Using Any Source Video)

The primary synthesis workflow uses `run_mixer.py mix`:
- If the video has already been aligned, alignment and units are loaded from cache in `< 0.05s`.
- If the video is new, the toolkit automatically detects the adjacent `.vtt` or `.srt` file, runs vocal isolation, preprocessing, MFA alignment, and caches the inventory before synthesizing.

```bash
python run_mixer.py mix \
  --video "path/to/source_video.mp4" \
  --speaker "speaker_name" \
  --text "Target sentence to synthesize" \
  --render "output/mix.mp4"
```
*(XML sequence export is optional; add `--xml "output/mix.xml"` if an FCP7 / Premiere Pro XML timeline is needed).*

### 2. Multi-Source Sentence Mixing & Folder Discovery

Expand the speaker's available vocabulary across multiple recordings:

**Explicit videos or comma-separated lists:**
```bash
python run_mixer.py mix \
  --video "vids/clip1.mp4" "vids/clip2.mp4" "vids/clip3.mp4" \
  --speaker obi-wan \
  --text "Hello there" \
  --render "output/multisource_mix.mp4"
```

**Folder Auto-Discovery (`--folder`):**
Scans one or more directories, automatically pairing media files (`.mp4`, `.mkv`, `.mov`, `.avi`, `.webm`) with matching subtitle files (`.en.vtt`, `.vtt`, `.srt`):
```bash
python run_mixer.py mix \
  --folder "vids/" \
  --speaker obi-wan \
  --text "I have the high ground" \
  --render "output/folder_mix.mp4"
```

**JSON Manifest (`--manifest`):**
```bash
python run_mixer.py mix \
  --manifest "evaluation/anakin_sources.json" \
  --speaker anakin \
  --text "This is where the fun begins" \
  --render "output/manifest_mix.mp4"
```

**Corpus-Wide Pooling (`--all-sources`):**
Mix across all previously ingested and cached sources in `.build/` without re-specifying file paths:
```bash
python run_mixer.py mix \
  --all-sources \
  --speaker obi-wan \
  --text "You were the chosen one" \
  --render "output/corpus_mix.mp4"
```

### 3. Subtitles & Burning

Generate synchronized subtitles alongside the render, or burn them into the video frames:
```bash
# Export matching .srt subtitles
python run_mixer.py mix \
  --video "vids/clip.mp4" \
  --speaker obi-wan \
  --text "Hello there" \
  --subtitles \
  --render "output/mix.mp4"

# Burn subtitles directly into video with classic YTP styling
python run_mixer.py mix \
  --video "vids/clip.mp4" \
  --speaker obi-wan \
  --text "Hello there" \
  --burn-subtitles \
  --subtitle-style classic \
  --subtitle-mode phrase \
  --render "output/burned_mix.mp4"
```

---

## Corpus Ingestion & Continuous Watch Mode (`ingest`)

For processing large video libraries or maintaining a continuous ingest pipeline, use the dedicated `ingest` subcommand. `ingest` performs vocal isolation, diarization, MFA alignment, and inventory indexing without requiring dummy synthesis text.

### Batch Ingestion
```bash
# Ingest an entire directory of clips
python run_mixer.py ingest --folder "vids/"

# Ingest via JSON manifest
python run_mixer.py ingest --manifest "sources.json"
```

### Continuous Folder Monitoring (`--watch`)
Monitors target folders indefinitely, automatically processing new or modified video/subtitle pairs as they are added:
```bash
python run_mixer.py ingest --folder "vids/" --watch --poll-interval 5.0
```
- **Sub-Second Delta Alignment**: Unchanged media files hit the multi-layer content hash cache in milliseconds. Only new or modified files trigger processing.
- **Graceful Termination**: Cleanly stops on `Ctrl+C` without corrupting cache indexes.

### Ingestion Pipeline Highlights
1. **Vocal Stem Isolation (`--isolate-vocals`)**: Uses `audio-separator` (UVR MDX-Net / Demucs) to extract clean speech stems into `.build/vocals/`, removing orchestral scores and background noise before alignment.
2. **Subtitle Cleaning**: De-duplicates rolling captions and strips non-speech markers. Optional Whisper tail enrichment (`--asr-fill-tails`, off by default) appends words Whisper hears after a cue ends; its output is unverified and can add hallucinated or other speakers' words, so use it only for sources whose captions are known to drop trailing words. The setting is part of the alignment cache key, so turning it on or off re-aligns affected sources automatically.
3. **Diarization & Speaker Verification**: Identifies speech turns using WavLM speaker verification (`microsoft/wavlm-base-plus-sv`).
4. **Interactive Speaker Clustering (`--interactive-clusters`)**: In interactive mode, unrecognized speaker clusters generate short 2-3 second `.wav` previews (`.build/clusters/<id>_preview.wav`) and present ranked candidate profiles for quick assignment.
5. **Dynamic Custom Dictionary Staging**: Automatically merges `dictionaries/custom_pronunciations.txt` into `dictionaries/merged_dictionary.dict` at `.build/mfa/staged_dictionary.dict` so custom proper nouns and names align as intact words rather than OOV noise (`spn`).

---

## Whole Movies (Long Sources)

Sources longer than 20 minutes or with surround audio (for example 2-hour 4K Blu-ray MKVs) use a dedicated, memory-safe pipeline (`src/movie_media.py`, `src/movie_ingest.py`). Nothing is decoded whole into memory, and every stage is cached and resumable.

```powershell
# 1. (Optional, recommended) Label speakers from the movie's own subtitles plus local
#    transcripts/screenplays. Writes a git-ignored labeled VTT the ingest picks up automatically.
python tools\label_subtitles.py --video "input\movie.mkv" --transcript input\ep3_transcript.txt `
  --screenplay .build\scripts\ep3_imsdb.html --output ".build\subtitles\labeled\<movie file stem>.en.vtt"

# 2. Ingest the whole movie (or only part of it with --range / --scene)
python run_mixer.py ingest "input\movie.mkv"
python run_mixer.py ingest "input\movie.mkv" --range 01:30:00-01:35:00

# 3. Inspect speakers and scenes; review uncertain lines (plays each line, saves labels)
python run_mixer.py speakers
python run_mixer.py scenes "input\movie.mkv"
python run_mixer.py review-speakers "input\movie.mkv" --only flagged

# 4. Mix, optionally restricted to a scene or time range, rendered in 4K
python run_mixer.py mix "input\movie.mkv" --speaker obi-wan --text "Your sentence" --resolution 4k
python run_mixer.py mix --all-sources --speaker anakin --scene 42 --text "Your sentence" --resolution 4k
```

How it works:
1. **Audio staged once as FLAC** under `.build/movies/<media hash>/audio/`: a stereo 48 kHz program downmix (raw track and separation input) and a 16 kHz analysis track taken from the **center channel** of 5.1 mixes (where film dialogue lives).
2. **Vocal separation in 10-minute blocks** (with 5 s context, trimmed), stitched into full-length `vocals.flac` / `instrumental.flac` for the default `recombined` track. Resumable block by block.
3. **Subtitles**: a labeled VTT from `tools/label_subtitles.py` if present, else a sidecar file, else the embedded text track (English SDH preferred; `--subtitle-stream auto|sdh|plain|<index>`). Image (PGS) tracks are not supported. Each cue is one MFA chunk (padded slightly, never into the neighboring cue).
4. **Alignment per 10-minute block**, cached by a key over that block's chunks, dictionary and model. `--range` aligns only overlapping blocks; later ingests reuse them.
5. **Speakers**: words take the speaker of the subtitle line they came from, matched by word order (exact for two-speaker dialogue-dash cues). `Name:` labels resolve through `config/characters/*.json` (aliases, OCR typos).
6. **Voice checks** (`--voice-checks`, default on): WavLM-SV embeddings verify labeled lines (a line whose voice clearly matches another character is made ineligible) and label unlabeled lines only above a margin calibrated to 95% leave-one-out precision on the labeled lines.
7. **Review**: `review-speakers` saves answers to `config/speaker_labels/<media hash>.json` (times, IDs and text hashes only, no subtitle text); they apply on the next ingest or mix without re-aligning.
8. **Mixing**: `--range` / `--scene` restrict candidate words to that part of the movie. The timeline frame rate defaults to the source's (`--fps auto`, 23.976 for films), and `--resolution 4k` renders 3840x2160 (letterboxed scope picture), 8-bit BT.709 H.264.

Copyright note: labeled VTTs, reports, screenplays, and the alignment corpus contain subtitle or script text and stay in git-ignored folders. Tracked files store only times, IDs, scores and text hashes.

---

## Unit Selection Architecture: Variable-Span (`span`, Default) vs Legacy

The sentence mixing toolkit defaults to **Variable-Span Unit Selection (`--selector span`)**, a global Viterbi optimization algorithm that dramatically outperforms greedy word-matching.

### Multi-Tier Selection Hierarchy

The variable-span engine optimizes cuts across four tiers:
1. **Tier 1: Intact Phrase Spans**: Continuous multi-word phrases directly from the source video. Preserves natural coarticulation, F0 pitch contours, and timing with zero edit splices.
2. **Tier 2: Whole Word Units**: Intact word tokens chosen near median duration to prevent anomalous outliers.
3. **Tier 2.5: Subword / Morpheme Spans**: Contiguous acoustic sequences within words (e.g. `can` + `not`, prefix/suffix segments) discovered across adjacent phonemes.
4. **Tier 3: Phoneme Spans with Acoustic Cognate Fallbacks**: Individual phone units linked via Viterbi shortest-path.

#### Acoustic Cognate & Vowel Reduction Fallbacks
Small speaker corpora (<10 seconds of speech) often lack specific phonemes (e.g., voiced velar stop `G`, velar nasal `NG`, voiced labiodental `V`). Instead of failing with a path disconnect:
- **Consonant Cognates**: Substitutes phonetically compatible sounds at small acoustic cost (0.35):
  - `G` $\leftrightarrow$ `K` (velar stops)
  - `NG` $\leftrightarrow$ `N` / `M` (nasals)
  - `V` $\leftrightarrow$ `F` (labiodental fricatives)
  - `Z` $\leftrightarrow$ `S` / `JH` $\leftrightarrow$ `CH`
- **Weak Vowel Reduction**: Unstressed vowels (`AH0` $\leftrightarrow$ `IH0` $\leftrightarrow$ `EH0`) reduce interchangeably at low cost (0.15).

### Global Viterbi Dynamic Programming Search
The selector minimizes total path cost across all target phoneme nodes:

$$C_{\text{total}} = \sum C_{\text{target}}(u_i) + \sum C_{\text{join}}(u_{i-1}, u_i)$$

- **Target Cost**: Penalizes phonetic mismatch, unnatural duration stretch, and non-realized pronunciation branches.
- **Join / Concatenation Cost**: Evaluates physical boundary compatibility using measured energy differences, spectral flux, voicing continuity, and F0 pitch steps.

### Cadence Tags & Span Markup
- **Cadence Pauses (`<gap:Xms>`)**: Insert hard timeline gaps without audio bridging (e.g. `Hello <gap:350ms> there`).
- **Explicit Pronunciation Branches (`{...}`)**: Disambiguate homographs or force phonetic realizations (e.g. `{R EH1 D}` vs `{R IY1 D}`).
- **Prosody Guidance (`--guide <audio>` + `--guide-transcript <path>`)**: Biases candidate selection toward a human speaker's pitch contour and rhythm without donating guide audio.

### Legacy Selector (`--selector legacy`)
The legacy engine uses a greedy word-first lookup with phoneme fallback splicing. It is maintained strictly for baseline backwards compatibility. It does not support phrase spans, cadence tags (`<gap:...>`), pronunciation brackets, or prosody guides.

---

## Audio Rendering: Recombination & EBU R128 Normalization

By default (`--audio-track recombined`), the rendering pipeline uses a high-fidelity two-stage audio mix:
1. **Vocal Isolation & Loudness Normalization**: Dialogue is spliced from isolated vocal stems and normalized according to EBU R128 / ITU-R BS.1770 (-16 LUFS, -1.0 dBTP true peak). This eliminates jarring volume differences between quiet whispers and shouting.
2. **Instrumental Recombination**: The normalized vocal cuts are recombined with the original background audio and music from the source video clips. Dialogue cuts remain clean and clear while preventing abrupt dead-silence drops in the background track.

**Available Audio Tracks (`--audio-track`):**
- `recombined` (default): Normalized vocals + original background instrumental.
- `raw` (or `combined`): Direct unseparated audio cuts from source video.
- `vocals` (or `isolated`): Clean isolated speech stems without background music.
- `both`: Renders both `raw` and `vocals` tracks.
- `all`: Renders `raw`, `vocals`, and `recombined` tracks.

---

## CLI Options Reference

### `run_mixer.py mix` (Synthesis)

| Option | Description | Default |
| --- | --- | --- |
| `--text <string>` | Target sentence / text to synthesize | `None` |
| `--selector span\|legacy` | Unit selection backend (`span` Viterbi shortest-path or `legacy` word-first) | `span` |
| `--speaker <id>` | Filter inventory to verified speaker identity (strictly a filter) | `None` |
| `--assume-single-speaker` | Explicit assertion of single-speaker status for single-source input | `False` |
| `--video <path...>` | Path(s) to source video(s) | `None` |
| `--vtt <path...>` | Path(s) to WebVTT / SRT subtitle files | `None` |
| `--folder <dir...>` | Path(s) to folder(s) containing source videos and subtitles | `None` |
| `--manifest <path>` | Path to JSON source manifest | `None` |
| `--all-sources` | Mix across all ingested and cached sources in `.build/` | `False` |
| `--render <path>` | Output rendered MP4 video path | `None` |
| `--plan <path>` | Output canonical `EditPlan` JSON path | `<stem>.plan.json` |
| `--xml <path>` | Optional Final Cut Pro 7 / Premiere Pro XML timeline | `None` |
| `--audio-track <track>` | Audio track variant: `recombined`, `raw`, `vocals`, `both`, `all` | `recombined` |
| `--normalize-loudness` | Enable EBU R128 (-16 LUFS, -1.0 dBTP) loudness normalization | Enabled (`True`) |
| `--isolate-vocals` | Isolate speech stems using audio-separator/Demucs before alignment | Enabled (`True`) |
| `--features basic\|wavlm` | Acoustic feature backend (`basic` continuous acoustics or `wavlm` neural) | `basic` |
| `--diarize auto\|off\|force` | Diarization mode (`auto`, `off`, `force`) | `auto` |
| `--diarize-threshold <val>` | Cosine distance threshold for speaker clustering | `0.18` (WavLM) |
| `--speaker-registry <path>` | Path to persistent speaker registry JSON | `config/speaker_registry.json` |
| `--subtitles` | Export matching `.srt` subtitle file alongside output video | `False` |
| `--burn-subtitles` | Hardcode subtitles directly into rendered video frames | `False` |
| `--subtitle-style classic\|modern`| Subtitle styling (`classic` YTP yellow, `modern` clean white) | `classic` |
| `--subtitle-mode phrase\|word` | Subtitle timing mode (`phrase` clauses, `word` highlights) | `phrase` |
| `--fps <rate>` | Timeline framerate: `auto` (the sources' own rate) or e.g. `30000/1001`, `24` | `auto` |
| `--resolution <size>` | Output video size: `1080p`, `1440p`, `4k` (3840x2160), `source`, or `WxH` (letterboxed) | `1080p` |
| `--range <START-END>` | Movies: only use words inside this time range (repeatable), e.g. `01:30:00-01:35:00` | `None` |
| `--scene <n\|keyword>` | Movies: only use words inside a scene (see `scenes` command) | `None` |
| `--subtitle-stream <s>` | Movies: embedded subtitle track (`auto`, `sdh`, `plain`, or stream index) | labeled VTT, else `auto` |
| `--characters <path>` | Movies: character alias list for subtitle speaker labels | `config/characters/star_wars.json` |
| `--voice-checks` | Movies: verify labels by voice and label unlabeled lines when confident | Enabled (`True`) |
| `--guide <audio>` | Human prosody guide audio (analysis only) | `None` |
| `--guide-transcript <path>` | Transcript matching prosody guide target words | `None` |
| `--target-audio <path>` | Reference audio recording for target pronunciation recognition | `None` |
| `--record-target` | Record spoken target audio interactively via microphone | `False` |
| `--review-pronunciations` | Interactively review pronunciations for OOV words | `False` |
| `--force-realign` | Invalidate cache and force fresh MFA alignment | `False` |
| `--asr-fill-tails` | Opt-in Whisper tail enrichment of subtitle text before alignment (unverified words) | `False` |

### `run_mixer.py ingest` (Preprocessing)

| Option | Description | Default |
| --- | --- | --- |
| `--folder <dir...>` | Path(s) to folder(s) containing source media and subtitles | `None` |
| `--manifest <path>` | Path to JSON source manifest | `None` |
| `--video <path...>` | Path(s) to source video(s) | `None` |
| `--vtt <path...>` | Path(s) to subtitle files | `None` |
| `--watch` | Continuously monitor folders for new media and ingest incrementally | `False` |
| `--poll-interval <sec>` | Polling interval in seconds for continuous watch mode | `5.0` |
| `--speaker <id>` | Speaker identity filter or assignment | `None` |
| `--isolate-vocals` | Isolate speech stems before alignment | Enabled (`True`) |
| `--diarize auto\|off\|force` | Diarization mode | `auto` |
| `--interactive-clusters` | Interactively review and assign unrecognized speaker clusters | `False` |
| `--asr-fill-tails` | Opt-in Whisper tail enrichment of subtitle text before alignment (unverified words) | `False` |
| `--range`, `--scene`, `--subtitle-stream`, `--characters`, `--voice-checks` | Movie options (see `mix`); `--range` aligns only overlapping 10-minute blocks | |

### Movie commands

| Command | Description |
| --- | --- |
| `run_mixer.py speakers [movie...]` | Speakers and usable seconds per ingested movie, label and voice-check counts |
| `run_mixer.py scenes <movie>` | Scene numbers, times and headings (from a screenplay given to `tools/label_subtitles.py`) |
| `run_mixer.py review-speakers <movie> [--only flagged\|unlabeled\|voice\|all] [--range ...] [--limit N]` | Play and label uncertain lines; saves to `config/speaker_labels/` |

```text
sentence_mixer/
├── bin/
│   └── micromamba.exe              # Self-contained package manager for MFA / Kaldi
├── config/
│   ├── characters/star_wars.json   # Character IDs, aliases, roles for subtitle speaker labels
│   ├── speaker_labels/             # Reviewed movie line labels (times, IDs, text hashes only)
│   └── speaker_registry.json       # Enrolled speaker acoustic profiles
├── dictionaries/
│   ├── custom_pronunciations.txt   # User-defined proper nouns & custom phonetic spellings
│   └── merged_dictionary.dict      # Full acoustic dictionary (208k entries)
├── evaluation/
│   ├── anakin_sources.json         # Evaluation source manifests
│   └── smoke_sources.json
├── src/
│   ├── diarization.py              # WavLM speaker verification & turn clustering
│   ├── fcp_xml_generator.py        # Final Cut Pro 7 / Premiere Pro XML timeline generator
│   ├── inspect_alignment.py        # Spectrogram & terminal alignment inspector
│   ├── inventory_builder.py        # TextGrid parser and unit indexer
│   ├── lexicon.py                  # ARPAbet lexicon, compound handling, and validation
│   ├── loudness.py                 # EBU R128 loudness normalization
│   ├── movie_commands.py           # speakers / scenes / review-speakers commands
│   ├── movie_ingest.py             # Whole-movie block alignment, speaker labels, voice checks
│   ├── movie_media.py              # FLAC staging, block vocal separation, embedded subtitles
│   ├── movie_subtitles.py          # Labeled/plain subtitle cues with per-line speakers
│   ├── oov_assistant.py            # Out-of-vocabulary pronunciation resolution
│   ├── prosody.py                  # Prosody extraction and F0 contour matching
│   ├── span_selector.py            # Variable-span Viterbi DP unit selection engine
│   ├── subtitle_parsing.py         # Shared SRT/VTT cue parsing and character name resolution
│   ├── unit_selector.py            # Legacy word-first matcher with phone fallback
│   ├── video_assembler.py          # Frame-accurate FFmpeg cuts & audio recombination
│   ├── vocal_isolation.py          # Demucs / UVR vocal separation
│   └── vtt_preprocessor.py         # Subtitle parsing & optional Whisper tail enrichment
├── tools/
│   ├── evaluate.py                 # Automated multi-stratum test evaluation harness
│   ├── label_subtitles.py          # Speaker-labeled movie subtitles from transcripts/screenplays
│   ├── listen_verify.py            # Unprimed ASR listening verification gate
│   └── validate_mix.py             # Timing, sample count, and manifest validator
├── README.md                       # Comprehensive documentation & setup guide
└── run_mixer.py                    # Unified CLI entry point (`mix`, `ingest`, `preprocess`)
```
