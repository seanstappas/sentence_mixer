# Sentence Mixer Project Guidelines & Guardrails

## Core Invariants
- **Zero-Commit Invariant**: Never execute `git commit`. Keep the working tree uncommitted.
- **Source-Preserving Invariant**: Never edit or mutate source files in `input/` or user media folders.
- **No Generative Audio Injection**: Output audio must be derived strictly from source recordings. Never inject TTS, voice cloning, or generated speech into render outputs.

## Unit Selection Standards
- Always use the **Variable-Span Selector** (`--selector span`, default). Do NOT default to `--selector legacy`.
- The variable-span engine optimizes across 4 tiers: Phrases -> Words -> Subwords -> Phonemes.
- When working with small speaker corpora (<30s speech), ensure acoustic cognate fallbacks are active to avoid broken Viterbi paths on missing consonants (`G` -> `K`, `NG` -> `N`, `V` -> `F`).

## Audio Quality & Mixing Heuristics
- Default to `--audio-track recombined` to preserve cinematic background atmosphere underneath EBU R128 (-16 LUFS) normalized vocals.
- **Synthesis Realism**: Target sentences sound natural only when constructed primarily from words or phrases already in the speaker's vocabulary. If a target is >70% out-of-vocabulary (OOV), advise the user to either expand the corpus (`run_mixer.py ingest`) or rephrase using existing dialogue.
- Whisper tail enrichment is opt-in (`--asr-fill-tails`, default off). Its words are unverified and get force-aligned as if spoken, so never enable it by default or treat its output as ground truth. If expected words are missing from captions, prefer fixing the subtitle file.

## Whole Movies
- Long sources (>20 min or surround audio) go through `src/movie_ingest.py`: never decode a whole movie into memory or write >4 GB WAVs; read audio by sample range (`LazyAudio`, `sf.read(start=, stop=)`).
- Isolate dialogue before any analysis: alignment, voice checks, and join features read the separated vocals (`analysis_vocals_16k.flac`); the raw center channel is only a fallback. Output audio keeps everything by default (`--audio-track recombined`).
- Speaker labels come from subtitles/transcripts (`tools/label_subtitles.py`), voice checks, and user review (`config/speaker_labels/`). Do not run clustering diarization on movies.
- Movie caches are expensive to rebuild (about 1.5 h per movie). Run `run_mixer.py clean <source> --dry-run` before any real `clean`, and never delete `.build/subtitles/labeled/` or `config/speaker_labels/` (user inputs).
- Labeled VTTs, label reports, screenplays/transcripts, and the alignment corpus contain copyrighted text: keep them in git-ignored folders and never copy their text into tracked files, tests, or chat output. Tests use invented dialogue.
