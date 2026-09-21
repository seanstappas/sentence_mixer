# Montreal Forced Aligner (MFA) Sentence Mixing Toolkit

An automated, unit-selection sentence mixing engine for video and audio remixing. Uses Montreal Forced Aligner (MFA) to derive sub-millisecond acoustic boundaries for words and phonemes from any source video and WebVTT subtitle track. Prioritizes whole-word unit selection with automatic grapheme-to-phoneme (G2P) phonetic fallback for missing words.

Outputs synchronized **rendered MP4 video / master WAV audio** (and optional Premiere Pro FCP7 XML timelines).

---

## Overview & Capabilities

The pipeline is completely source-agnostic and works with any video, speaker, or speech domain:
- **Any video format**: MP4, WebM, MKV, MOV, etc.
- **Any English speech content**: Talks, lectures, interviews, gaming streams, movies, or commentary.
- **Any WebVTT subtitles**: Automatically parses standard WebVTT or YouTube auto-generated rolling captions, de-duplicates rolling lines, discards non-speech bracketed markers (`[Music]`, `[Applause]`), and extracts 16kHz mono audio chunks.
- **Automatic Caching**: Generates a unique hash from the source video path. Once a video is aligned, its inventory is permanently cached under `tools/sentence_mixer/cache/<source_id>/` and reloaded in `~0.05s`.
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
Move-Item -Force "Library\bin\micromamba.exe" "tools\sentence_mixer\bin\micromamba.exe"
Remove-Item -Recurse -Force "Library", "micromamba.tar.bz2", "info" -ErrorAction SilentlyContinue

# 2. Create the isolated mfa_env environment
& "tools\sentence_mixer\bin\micromamba.exe" create -n mfa_env -c conda-forge --root-prefix "C:\Users\Sean\micromamba" montreal-forced-aligner python=3.10 -y

# 3. Download pre-trained ARPA acoustic, dictionary, and G2P models
& "tools\sentence_mixer\bin\micromamba.exe" run -r "C:\Users\Sean\micromamba" -n mfa_env mfa model download acoustic english_us_arpa
& "tools\sentence_mixer\bin\micromamba.exe" run -r "C:\Users\Sean\micromamba" -n mfa_env mfa model download dictionary english_us_arpa
& "tools\sentence_mixer\bin\micromamba.exe" run -r "C:\Users\Sean\micromamba" -n mfa_env mfa model download g2p english_us_arpa
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

Agents can invoke `tools/sentence_mixer/run_mixer.py` using system Python.

### 1. Basic Sentence Mixing (Using Any Source Video)
Pass `--video` with your source file:
- If the video has already been aligned, alignment is loaded from cache in `~0.05s`.
- If the video is new, the toolkit automatically detects the adjacent `.vtt` file, runs preprocessing and MFA alignment, caches the index, and synthesizes the sentence.

```bash
python "tools/sentence_mixer/run_mixer.py" mix \
  --video "path/to/source_video.mp4" \
  --text "Target sentence to synthesize" \
  --render "path/to/output.mp4"
```
*(XML output is omitted by default. If an FCP7/Premiere Pro sequence timeline is specifically desired, add `--xml "path/to/output.xml"`).*

### 2. Multi-Source Sentence Mixing (Expanding Vocabulary & Corpus)
You can provide multiple video sources by separating them with spaces or commas:

```bash
python "tools/sentence_mixer/run_mixer.py" mix \
  --video "vids/source1.mp4" "vids/source2.mp4" "vids/source3.mp4" \
  --text "Your sentence to synthesize across multiple sources" \
  --render "output/multisource_mix.mp4"
```

Or pass a JSON manifest file listing all video and VTT paths:
```bash
python "tools/sentence_mixer/run_mixer.py" mix \
  --manifest "tools/sentence_mixer/sources_manifest.json" \
  --text "Your sentence to synthesize across entire library" \
  --render "output/library_mix.mp4"
```

Manifest format (`sources_manifest.json`):
```json
[
  {
    "video": "path/to/video1.mp4",
    "vtt": "path/to/video1.en.vtt"
  },
  {
    "video": "path/to/video2.mp4"
  }
]
```

**How Caching & Pooling Works:**
- Each video is checked against its own independent cache (`cache/<source_id>/`).
- Previously aligned videos load instantly. Any new video is aligned once and saved to cache.
- The unit inventories (all words and phonemes) are merged seamlessly in memory into a single combined dictionary.
- Each selected unit retains its own `source_video` path, so FFmpeg cuts directly from whichever video provides the word or phoneme.

### 3. Auto-Detecting Subtitles vs Explicit `--vtt`
If the subtitle file is named after the video (e.g. `source.mp4` and `source.en.vtt` or `source.vtt` in the same directory), `--vtt` is auto-detected. If the subtitle file is located elsewhere or named differently, pass `--vtt`:

```bash
python "tools/sentence_mixer/run_mixer.py" mix \
  --video "path/to/source_video.mp4" \
  --vtt "path/to/custom_subtitles.vtt" \
  --text "Your sentence to synthesize" \
  --render "output/mix.mp4"
```

### 3. Advanced Variable-Span Unit Selection & Features (Milestones M0-M6)

The engine now supports variable-span selection (Phrase -> Word -> Whole Phone), canonical edit planning, master PCM-first rendering, and optional WavLM neural feature reranking:

```bash
# Variable-span unit selection with basic acoustic features
python run_mixer.py mix \
  --manifest evaluation/smoke_sources.json \
  --speaker obi-wan \
  --text "Hello there" \
  --selector span \
  --features basic \
  --mode source-derived \
  --fps 30000/1001 \
  --render output/mix.mp4 \
  --plan output/mix.plan.json \
  --xml output/mix.xml

# Variable-span unit selection with neural WavLM feature reranking
python run_mixer.py mix \
  --manifest evaluation/smoke_sources.json \
  --speaker obi-wan \
  --text "Hello there" \
  --selector span \
  --features wavlm \
  --mode source-derived \
  --render output/mix_wavlm.mp4

# Validate output timing, sample count, provenance, and XML agreement
python tools/validate_mix.py --stem output/mix

# Run the automated multi-stratum evaluation suite
python tools/evaluate.py --suite evaluation/suite.json --output-dir output/evaluation
```

#### Supported CLI Options

| Option | Description | Default |
| --- | --- | --- |
| `--selector legacy\|span` | Unit selection backend (`legacy` word-first or `span` variable-span) | `legacy` |
| `--speaker <id>` | Filter inventory to verified speaker identity | `None` (or auto-resolved if single verified speaker) |
| `--assume-single-speaker` | Explicit assertion of single-speaker status | `False` |
| `--mode source-derived\|cuts-only` | Join smoothing mode | `source-derived` |
| `--features basic\|wavlm` | Acoustic feature backend (`basic` continuous acoustics or `wavlm` neural) | `basic` |
| `--fps <rate>` | Rational framerate (e.g. `30000/1001`, `24`, `25`, `29.97`) | `30000/1001` |
| `--plan <path>` | Path to output canonical `EditPlan` JSON | `<render_stem>.plan.json` |
| `--xml <path>` | Optional Final Cut Pro 7 / Premiere Pro XML timeline | `None` |
| `--guide <audio>` | Optional human prosody guide audio (analysis-only) | `None` |
| `--force-realign` | Invalidate cache and force fresh MFA alignment | `False` |

### 4. Force Re-alignment
If you modified subtitles or added custom dictionary entries:
```bash
python run_mixer.py mix \
  --video "path/to/source_video.mp4" \
  --text "Your sentence" \
  --force-realign
```

### 5. Inspecting Alignment Quality & Phoneme Boundaries
To verify boundary accuracy on any speech chunk:

```bash
# Terminal table inspection of words and phonemes
python "tools/sentence_mixer/src/inspect_alignment.py" --chunk 0 --aligned "tools/sentence_mixer/aligned" --corpus "tools/sentence_mixer/corpus"

# Generate a visual waveform + spectrogram plot with overlaid boundaries
python "tools/sentence_mixer/src/inspect_alignment.py" --chunk 0 --aligned "tools/sentence_mixer/aligned" --corpus "tools/sentence_mixer/corpus" --plot "output/inspect.png"
```

---

## Directory Structure

```text
tools/sentence_mixer/
├── bin/
│   └── micromamba.exe              # Self-contained conda package manager for MFA / Kaldi
├── cache/                          # Automatic alignment cache per source video ID
├── corpus/                         # Speech intervals (16kHz WAV + TXT)
├── aligned/                        # Aligned Praat .TextGrid files
├── dictionaries/
│   ├── custom_pronunciations.txt   # User-defined proper nouns & slang terms
│   └── merged_dictionary.dict      # Full acoustic dictionary (208k entries)
├── inventory/
│   └── units_index.json            # Indexed database of all words & phonemes
├── output/                         # Rendered MP4s, WAVs, and XML timelines
├── src/
│   ├── vtt_preprocessor.py         # Rolling caption de-duplicator & audio chunker
│   ├── inventory_builder.py        # TextGrid parser and unit indexer
│   ├── unit_selector.py            # Word-first matcher with G2P phoneme fallback
│   ├── fcp_xml_generator.py        # Final Cut Pro 7 XML sequence generator
│   ├── video_assembler.py          # FFmpeg frame-accurate cuts with micro-crossfades
│   └── inspect_alignment.py        # Spectrogram & terminal alignment inspector
├── README.md                       # Documentation & setup guide
└── run_mixer.py                    # Unified CLI entry point
```

---

## How the Unit Selection Engine Works

1. **Tokenization & Normalization**: Strips punctuation and splits the target text into word tokens.
2. **Whole Word Lookup**: Looks up each token in the aligned inventory. If found, picks a clean occurrence near median duration to prevent outliers.
3. **Phonetic Fallback**: If a word does not exist in the source video corpus, `g2p_en` decomposes the word into ARPA phonemes (e.g. `['HH', 'EY1', 'T']`).
4. **Phoneme Splicing**: The engine extracts matching phoneme intervals from the corpus, selecting healthy duration candidates (50ms - 200ms) with smooth transitions.
5. **Assembly**: Uses FFmpeg to extract standardized 1080p clips, adds 5ms audio micro-fades to eliminate boundary pops, and concatenates them into seamless video (`.mp4`) and audio (`.wav`) masters.
