# Handoff: State of the Sentence Mixer (2026-10-05)

Read this first if you are taking over. It summarizes what was built, why, what was measured,
what is still open, and the traps we hit. Details live in the linked files.

Required reading, in order:

1. [GEMINI.md](../GEMINI.md): project rules and guardrails (invariants, copyright, movies).
2. [.agents/skills/sentence-mixer/SKILL.md](../.agents/skills/sentence-mixer/SKILL.md): command cheatsheet and troubleshooting.
3. [README.md](../README.md): full user documentation (Whole Movies, labeling, cache layout, options).
4. [10_04_movie_support_plan.md](10_04_movie_support_plan.md): movie design, sections 2A/2B/2C are the status log.
5. [10_04_antigravity_remediation_plan.md](10_04_antigravity_remediation_plan.md): older remediation items (C01 done; others open).

## 1. What the system does now

A source-preserving YTP sentence mixer. Given source videos plus subtitles and target text, it
force-aligns the dialogue (MFA), indexes words and phones per speaker, picks the best sequence
of phrases/words/subwords/phones (Viterbi, `--selector span`), and renders a frame-accurate
MP4 (up to 4K) from the original audio and video. No generated speech, ever.

Two kinds of sources:

| | Short clips (YouTube etc.) | Whole movies (> 20 min or 5.1 audio) |
| --- | --- | --- |
| Code path | `ensure_source_aligned` in `run_mixer.py`, `vtt_preprocessor.py` | `src/movie_ingest.py` (dispatched automatically by `is_movie_source`) |
| Subtitles | sidecar VTT/SRT | labeled VTT from `tools/label_subtitles.py`, else sidecar, else embedded SDH text track |
| Speaker identity | WavLM-SV diarization, or `--assume-single-speaker` / manifest intervals | subtitle `<v>` labels, voice checks, `review-speakers` overrides (no diarization) |
| Vocal isolation | whole clip, audio-separator (Kim_Vocal_2) | 10-minute blocks, stitched FLAC stems, resumable |
| Analysis audio | isolated vocals | `analysis_vocals_16k.flac` (separated dialogue); center channel only as fallback |
| Caching | per source | per 10-minute block (alignment), per movie (stems, labels, features) |

Ingested so far (local, git-ignored caches): Star Wars Episode II, III, and VI (4K Blu-ray MKVs in
`input/`). All three fully aligned: Episode II (1184 dialogue cues, 7593 words), Episode III (1185 cues,
7163 words), Episode VI (930 cues, 5764 words).
Movie hashes (cache folder names): Episode II `56214896cf2afcf7`, Episode III `a39114e3f7ca0eae`,
Episode VI `3d083b52d216f9f0`.

## 2. Key commands

```powershell
Set-Location 'D:\Code\sentence_mixer'
$ep2 = (Get-ChildItem -LiteralPath .\input -Filter 'Star.Wars.Episode.II.-*.mkv').FullName
$ep3 = (Get-ChildItem -LiteralPath .\input -Filter 'Star.Wars.Episode.III.-*.mkv').FullName
$ep6 = (Get-ChildItem -LiteralPath .\input -Filter 'Star.Wars.Episode.VI.-*.mkv').FullName

.venv\Scripts\python.exe run_mixer.py speakers                      # speakers + usable seconds
.venv\Scripts\python.exe run_mixer.py scenes $ep3                   # scene numbers/times
.venv\Scripts\python.exe run_mixer.py mix $ep2 $ep3 --speaker obi-wan --text "..." --resolution 4k
.venv\Scripts\python.exe run_mixer.py mix $ep3 --scene 32 --speaker obi-wan --text "..." --pause-style continue
.venv\Scripts\python.exe run_mixer.py review-speakers $ep2 --only flagged --audio vocals
.venv\Scripts\python.exe tools\listen_verify.py -i output\x.mp4 -e "expected text"   # ASR check (WER)
.venv\Scripts\python.exe tools\render_dialogue_clips.py --clip all  # multi-speaker scene rendering with word-highlighted ASS subtitles
.venv\Scripts\python.exe run_mixer.py clean $ep3 --dry-run          # ALWAYS dry-run first
```

Quality gates (all must pass; current: 503 passed, 1 skipped; Ruff and mypy clean):

```powershell
$env:PYTHONDONTWRITEBYTECODE='1'; $env:NLTK_DATA='.build\nltk'; $env:HF_HUB_OFFLINE='1'; $env:TRANSFORMERS_OFFLINE='1'
.venv\Scripts\python.exe -m ruff check .
.venv\Scripts\python.exe -m mypy run_mixer.py src
.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider
```

## 3. Movie pipeline, end to end

1. **Label subtitles offline** (`tools/label_subtitles.py`, about 2 s per movie). Inputs: the
   movie's SDH text track, the user's transcripts (`input/ep2_transcript.txt`,
   `input/ep3_transcript.txt`), IMSDb screenplays (`.build/scripts/ep{2,3}_imsdb.html`,
   fallback only), and `config/characters/star_wars.json`. Output:
   `.build/subtitles/labeled/<movie stem>.en.vtt` (`<v speaker-id>` tags) plus a report with
   per-line evidence (text hashes only) and scene time ranges. Coverage: Episode II 72% of
   lines labeled, Episode III 94%. Text sources agree with each other 97-99%.
2. **Ingest** (`run_mixer.py ingest <movie>`): FLAC staging (stereo program, center channel),
   block vocal separation (about 45 min per movie on CPU), 16 kHz separated-dialogue track,
   per-block MFA alignment (about 15 min per movie), center-channel retry for chunks MFA
   cannot align from vocals, separation-loss check, join features over speech spans, voice
   checks, user overrides, inventory. Whole re-ingest from scratch: about 1.5 h per movie.
3. **Mix**: span selection over the movie inventory; `--range` / `--scene` restrict words by
   time; `--fps auto` keeps 23.976; `--resolution 4k` renders 3840x2160 letterboxed; default
   audio `recombined` = leveled dialogue over the film's own music and effects.

Line status values in `.build/movies/<hash>/speaker_labels.json` (no text stored):
`label` / `label_voice_ok` / `label_voice_unclear` (usable), `voice_propagated` (unlabeled line
labeled by voice; usable), `voice_mismatch` (excluded), `separation_dropped` (excluded: vocal
separation removed the voice), `unlabeled` (excluded), `user` (from review-speakers).

Current counts: Episode II: 588 voice-consistent, 96 unclear, 179 label-only, 145 propagated,
8 flagged, 1 separation-dropped, 201 unlabeled, 4 user. Episode III: 624 / 263 / 254 / 16 / 7 / 4 / 57.

## 4. Decisions and the measurements behind them

| Decision | Evidence |
| --- | --- |
| Analyze separated vocals, not the 5.1 center channel | One Episode III block aligned 3 ways, ASR recall of words cut at each alignment's times: separated stereo vocals 0.77 single words / 0.77 3-word windows; raw center 0.72 / 0.76; separated center 0.73 / 0.74. Test mixes: mean WER 0.45 -> 0.29. |
| Keep text labels authoritative over voice | WavLM-SV nearest-speaker accuracy on short film lines is only 0.70-0.86. A Gunray line was flagged as Mace Windu with a 0.01 lead and a 4-line profile. Flags now need: >= 0.10 closer to another speaker, a calibrated lead (95% leave-one-out precision, >= 0.05), and >= 5 other lines for the labeled speaker. Flags fell 21 -> 8 (II) and 42 -> 7 (III). |
| Center-channel retry for unalignable chunks | 14 chunks failed on vocals; 12 had normal vocal levels (MFA failed on fast lines), all recovered from the center channel. |
| Separation-loss check | Separation keeps about -10.7 dB of the center level typically; lines more than 15 dB below that lost their voice (radio/processed voices). 5 lines excluded; 3 had been force-aligned to near-silence. |
| Release tail (100 ms into following silence) | MFA ended word-final fricatives early (an "f" got 50 ms; its hiss lasted 80-100 ms more). In one test mix, ASR missed that word 4/4 times before and 0/4 after; WER 0.38 -> 0.13. |
| Precompute join features over speech spans | Lazy per-window pYIN took about 140 s per mix; precomputed, about 15 s. |
| Center channel for nothing but fallback | Earlier we used it for everything; measurements above reversed that. |
| Scenes from screenplay headings | IMSDb headings mapped to subtitle time by word alignment: 44 scenes (II), 90 (III). Boundaries are midpoints between scenes' dialogue, not picture cuts. |

## 5. Lessons and traps (read before changing things)

- **Measure, then switch.** Several plausible changes (vocals for voice checks, center channel
  for alignment) went the other way when measured. Use `tools/listen_verify.py` WER on renders
  and the 4-word-window ASR recall check (see section 4) on inventories. Whisper-tiny WER is
  noisy: judge on several sentences, and listen.
- **Default mix audio must keep everything**: `--audio-track recombined`. `vocals` is opt-in.
- **Never print or commit copyrighted text**: subtitles, transcripts, screenplays, labeled VTTs,
  reports, alignment corpora. Print counts, IDs, times, hashes. Tests use invented dialogue.
- **`clean` is destructive and slow to undo** (1.5 h per movie). Always `--dry-run` first. It
  used to delete other sources sharing a name prefix (fixed; regression test exists).
- **Windows specifics**: PowerShell 5 (no `&&`); `winsound` for non-blocking playback; file
  replaces can hit sharing violations (retry or per-process temp names); MFA must run with
  `-j 1` and single-threaded BLAS; Python writes can add a UTF-8 BOM if done via PowerShell
  `Set-Content` (strip it).
- **Caches are keyed by content**: dictionary staging by file content (not mtimes); blocks by
  chunk list + dictionary + model + analysis audio; labels by blocks, overrides, voice
  settings, subtitles. Changing any of these triggers exactly the needed rebuild.
- **Concurrency**: two ingests of the same movie can run at once (per-process temp names),
  but a running process keeps old code; restart it after code changes.
- **Plan files** in `plans/` are ASCII with CRLF line endings; keep them that way.
- **Timing-sensitive tests**: use best-of-N timing, not single measurements.
- **Interrupted renders** leave `.attempt_<id>` staging folders next to the output (published files are written atomically from them); they are safe to delete.

## 6. Open work (prioritized)

1. **Selection quality on long inventories** (biggest remaining quality gap). Mixes that use
   single-phone pieces or very short word instances are often misheard (8 words in 1.4 s).
   Ideas: penalize instances much shorter than the speaker's median for that word, prefer
   phrase-level prosody continuity, cap phone-level splicing per word.
2. **Voice checks** remain weak (leave-one-out 0.70 Episode III). Options: a stronger speaker
   model (ECAPA-TDNN), longer context windows per line, or per-scene profiles.
3. **Unknown voices**: clustering leftover unlabeled lines into anonymous groups for review
   (movie plan M06) is not done.
4. **OCR typo check** comparing SDH and plain subtitle tracks (movie plan M03 step 8).
5. **Scene boundaries** from shot detection (currently dialogue midpoints).
6. **GPU**: torch is CPU-only here (RTX 4090 present). A CUDA build would speed separation
   (about 45 min per movie) and voice embeddings a lot.
7. Older remediation items still open in the round-4 plan (speaker-change detection in
   diarization for short clips, etc.).

## 7. Where things are

| What | Where |
| --- | --- |
| Movie code | `src/movie_media.py`, `src/movie_ingest.py`, `src/movie_subtitles.py`, `src/movie_commands.py`, `src/subtitle_parsing.py` |
| Edit plan (tails, pause styles) | `src/edit_plan.py` (`release_tail_samples`, `emit_pause`, `CONTINUE_SPAN_TYPE`) |
| Renderer (lazy audio reads, fades, 4K) | `src/video_assembler.py` (`LazyAudio`, `_apply_segment_fades`, `resolve_output_dims`) |
| Labeling tool | `tools/label_subtitles.py`, `config/characters/star_wars.json` |
| Reviewed labels (tracked) | `config/speaker_labels/<hash16>.json` |
| Movie caches (git-ignored) | `.build/movies/<hash16>/`, `.build/alignment/<source id>/blocks/`, `.build/inventory/<source id>/` |
| Labeled VTTs / reports (git-ignored, copyrighted) | `.build/subtitles/labeled/` |
| Dialogue scene mixer | `src/dialogue_mixer.py`, `tools/render_dialogue_clips.py` (multi-speaker script parsing, EditPlan stitching, word-highlighted ASS subtitles) |
| Test renders | `output/movie_tests/` (center-channel era), `output/movie_tests_vocals/` (current), `output/episode_3_anakin_mace.mp4`, `output/episode_6_luke_palpatine_vader.mp4` |
| Tests for this work | `tests/test_movie_support.py`, `tests/test_label_subtitles.py`, `tests/test_pause_styles.py`, `tests/test_dialogue_mixer.py` |
