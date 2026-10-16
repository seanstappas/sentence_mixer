# Sentence Mixer Project Guidelines & Guardrails

Handoff and current state: `plans/10_05_handoff.md` (read before making changes).

## Core Invariants
- **Zero-Commit Invariant**: Never execute `git commit`. Keep the working tree uncommitted.
- **Source-Preserving Invariant**: Never edit or mutate source files in `input/` or user media folders.
- **No Generative Audio Injection**: Output audio must be derived strictly from source recordings. Never inject TTS, voice cloning, or generated speech into render outputs.

## Unit Selection Standards
- Always use the **Variable-Span Selector** (`--selector span`, default). Do NOT default to `--selector legacy`.
- The variable-span engine optimizes across 4 tiers: Phrases -> Words -> Subwords -> Phonemes.
- When working with small speaker corpora (<30s speech), ensure acoustic cognate fallbacks are active to avoid broken Viterbi paths on missing consonants (`G` -> `K`, `NG` -> `N`, `V` -> `F`).
- **Subtitle-Anchored OOV Recovery**: MFA labels words missing from its static dictionary as `spn` (spoken noise) even when the word boundaries `[xmin, xmax]` are accurately aligned on the TextGrid `words` tier. Movie ingestion (`src/movie_ingest.py`) enables `recover_oov_words=True` in `build_units_inventory` (`src/inventory_builder.py`) to synthesize constituent phones from `Lexicon` (`custom_pronunciations.txt` -> CMUdict -> G2P) proportional to duration priors, preserving intact spoken words (e.g. *Alderaan*, *droid*, *Skywalker*).

## Audio Quality & Mixing Heuristics
- Default to `--audio-track recombined` to preserve cinematic background atmosphere underneath EBU R128 (-16 LUFS) normalized vocals.
- **Synthesis Realism**: Target sentences sound natural only when constructed primarily from words or phrases already in the speaker's vocabulary. If a target is >70% out-of-vocabulary (OOV), advise the user to either expand the corpus (`run_mixer.py ingest`) or rephrase using existing dialogue.
- Whisper tail enrichment is opt-in (`--asr-fill-tails`, default off). Its words are unverified and get force-aligned as if spoken, so never enable it by default or treat its output as ground truth. If expected words are missing from captions, prefer fixing the subtitle file.

## Engineering Practice
- Measure before switching approaches: compare renders with `tools/listen_verify.py` (WER) and inventories with an ASR recall check on word windows; several plausible changes measured worse. Report numbers, not impressions.
- Run the gates on the final tree (Ruff, mypy, offline pytest with `PYTHONDONTWRITEBYTECODE=1`, `NLTK_DATA=.build\nltk`, `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`); timing tests use best-of-N.
- Restart long-running ingests after code changes; running processes keep the old code.

## Whole Movies
- Long sources (>20 min or surround audio) go through `src/movie_ingest.py`: never decode a whole movie into memory or write >4 GB WAVs; read audio by sample range (`LazyAudio`, `sf.read(start=, stop=)`).
- Isolate dialogue before any analysis: alignment, voice checks, and join features read the separated vocals (`analysis_vocals_16k.flac`); the raw center channel is only a fallback. Output audio keeps everything by default (`--audio-track recombined`).
- Speaker labels come from subtitles/transcripts (`tools/label_subtitles.py`), voice checks, and user review (`config/speaker_labels/`). Do not run clustering diarization on movies.
- Text speaker labels are authoritative; voice (WavLM-SV, 0.70-0.86 accurate on short film lines) may only exclude a labeled line on strong calibrated evidence, and review defaults keep the text label.
- Movie caches are expensive to rebuild (about 1.5 h per movie). Run `run_mixer.py clean <source> --dry-run` before any real `clean`, and never delete `.build/subtitles/labeled/` or `config/speaker_labels/` (user inputs).
- When adding new entries to `dictionaries/custom_pronunciations.txt` or updating inventory logic for already-ingested movies, run `.venv\Scripts\python.exe tools\rebuild_movie_inventories.py` (~3 s per movie) to rebuild `units_index.json` in-place from cached `.TextGrid` files without re-running MFA alignment.
- Labeled VTTs, label reports, screenplays/transcripts, and the alignment corpus contain copyrighted text: keep them in git-ignored folders and never copy their text into tracked files, tests, or chat output. Tests use invented dialogue.
