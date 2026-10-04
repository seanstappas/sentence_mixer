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
