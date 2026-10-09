# Antigravity Remediation Handoff, Round 4: Speaker Safety, Honest Data, Clean Baseline

Date: 2026-10-04.
Audited revision: `cca1595c` ("update docs"), clean working tree. Changes since the
previous handoff landed in commits `818f59f8`, `31f53bc9`, `44a95aac`, and `cca1595c`.

## 1. Context, scope, and precedence

Implement every work package below (C01-C12), then apply the guideline updates in
section 6. The system has improved a lot: mixes sound better, the UX is better,
and most earlier timing, publishing, XML, and speaker-filter defects are fixed.
The remaining problems fall into three groups:

1. **Speaker safety:** fast dialogue exchanges can still put one character's words
   under another character's identity.
2. **Honest data:** some steps silently add or substitute content (ASR-added
   words, phoneme substitutions, weaker embedding fallbacks) without telling the user.
3. **Clean baseline:** the committed code does not pass its own checks, and the
   repository contains scratch artifacts.

Product rules (unchanged): output speech must come only from source recordings
(no TTS, voice cloning, or generated speech); analysis models (Whisper, WavLM,
diarization) are analysis-only; never modify source media under `input/` or user
media folders; do not run `git commit` unless the user explicitly asks.

Read with:

- [Round-3 handoff](9_23_antigravity_remediation_plan.md): binding decisions D1-D10
  still apply unless this document says otherwise.
- [Original request](ORIGINAL_REQUEST.md), [design](design.md),
  [enhancement review](9_21_astra_thoughts.md).
- Project rules [GEMINI.md](../GEMINI.md) and the skill at
  `.agents/skills/sentence-mixer/SKILL.md` (tracked in git; the rest of `/.agents/`
  is ignored).

Precedence: this document overrides earlier handoffs where it is more specific.
Lines 206 and 254 of `ORIGINAL_REQUEST.md` describe a ">30% overlap" eligibility
rule. That rule is superseded by D5 (whole-word coverage with a 40 ms collar),
which the current code already implements. Keep D5.

### Working well: keep these as regression guards

| Behavior | Verified result at `cca1595c` |
| --- | --- |
| Overlap picture timing | 6 x 100 ms overlapping cuts: 25,440 samples (0.530 s), 16 planned and 16 rendered frames |
| XML master audio | XML generation raises if the master WAV is missing, instead of omitting audio |
| `--speaker` | Acts as a filter; labels come only from assertions, verified manifests, or diarization |
| Word eligibility | Whole-word coverage with a 40 ms silence collar (D5); anonymous or unreviewed identities ineligible |
| Listening gate | Unprimed transcription, with the primed result reported separately as "assisted" |
| Evaluator | ASR unavailable is reported as `blocked` (exit 2) |
| Span selector default, recombined audio default | Accepted by the user as the new defaults |
| Whisper tail enrichment opt-in (2026-10-04) | Off by default; `--asr-fill-tails` enables it; setting is in the alignment cache key; covered by `tests/test_asr_tail_enrichment_optin.py` |

## 2. Verified findings (evidence)

All reproductions used synthetic audio/video, injected fake models, and temporary
folders. No real media was processed and no repository file was changed.

| ID | Finding | Evidence |
| --- | --- | --- |
| F1 | Fast exchanges merge into one speaker | Voices at 140 Hz and 260 Hz separated by a 0.2 s gap became one VAD turn (0.5-3.05 s); both words were labeled `obi-wan` and eligible; no overlap flag |
| F2 | No real overlap detection | `is_overlap` is set only when VAD turns overlap, but energy-VAD turns are disjoint by construction ([diarization.py](../src/diarization.py), `diarize`, around lines 690-715) |
| F3 | Labeled turns are padded into quiet audio | Up to 0.20 s before and 1.50 s after each turn, regardless of what the quiet region contains |
| F4 | Whisper tail enrichment adds unverified words | Caption "Hello there" plus Whisper output "hello there thank you for watching" became the MFA transcript `Hello there thank you for watching` ([vtt_preprocessor.py](../src/vtt_preprocessor.py), `enrich_chunk_transcript_with_asr`). The user's cache had 8 enriched chunks across 5 sources, including long garbage tails in `obi_wan_ep3`. **Default fixed 2026-10-04:** now opt-in via `--asr-fill-tails`; remaining work in C03 |
| F5 | Diarization audio-extract cache collision | `episode2/clip.mp4` (4 s) was diarized with the 2 s audio extracted from `episode1/clip.mp4`; the extract path is CWD-relative `.build/audio_extracts/<stem>.wav` and ignores `--cache-dir` |
| F6 | Diarization reruns on every `mix` | No diarization result cache; the WavLM-SV pass repeats for every unasserted source on every run |
| F7 | Silent phoneme substitutions | Cognate table includes voicing swaps (B->P, D->T, G->K, V->F, Z->S, DH->D, W->UW); used only when the speaker has no instance of the phone, but never reported ([unit_selector.py](../src/unit_selector.py), `PHONETIC_COGNATE_FALLBACKS`) |
| F8 | Silent embedding fallback | Speaker embeddings fall back from WavLM-SV to raw WavLM, MFCC, then an FFT "test" vector; the registry stores no embedding model identity; the SV model loader may download if not cached |
| F9 | Misleading unverified-source path | With no assertion and no diarization backend, the CLI prints "Proceeding with unverified identity", but every word is then ineligible, so the mix fails after full alignment |
| F10 | Checks fail on committed code | Offline suite: 445 passed, **1 failed**, 1 skipped, 1 xpassed. Ruff: 1 error (`tests/test_m4_speaker_review_ux.py:436`, unused `emb_map`). mypy: 1 error (`run_mixer.py:959`, `distance_threshold` is `Any \| None`, expected `float`) |
| F11 | Order-dependent dictionary cache | `test_custom_pronunciation_invalidates_alignment_cache` fails in the full suite but passes 5/5 alone; `stage_mfa_dictionary` decides freshness from file timestamps and a module-level cache instead of content ([lexicon.py](../src/lexicon.py)). After the 2026-10-04 opt-in change it failed in 1 of 2 further full runs (456 passed / 455 passed + this failure) |
| F12 | Stale non-strict xfail | `test_stage_mfa_dictionary_concurrency_filelock` is `xfail(strict=False)` and now passes |
| F13 | Repo hygiene | 23 `scratch_*.wav` files were committed at the repo root in `818f59f8` (since removed by the user; `.gitignore` now blocks them). `config/speaker_registry.json` stores absolute `.build` preview paths and cache-specific source IDs, so it is not portable |
| F14 | Validator tolerance | [validate_mix.py](../tools/validate_mix.py) still allows a 2-frame audio/video duration difference |
| F15 | Tuning and claims | Dozens of hand-tuned penalties in [span_selector.py](../src/span_selector.py); speaker thresholds (0.18 distance, 0.84 similarity) were calibrated on a few Star Wars clips; selection bonuses depend on the predecessor path, so the DP is a strong heuristic, not a guaranteed global optimum |
| F16 | Overstated reports | Earlier summaries claimed "strict mypy" (config is not strict) and "446 passing, zero lint errors" (narrowed run; current code fails as in F10) |

## 3. Binding decisions for this round

**E1. Whisper never changes transcripts unless the user opts in.** Subtitle text
(movie subtitle files or YouTube captions) is used as-is. YouTube's auto-captions come
from a large Google speech model that is generally at least as accurate as the local
Whisper models, so automatic correction is not reliable. Whisper tail enrichment is
available only with `--asr-fill-tails` (implemented 2026-10-04, default off). Words it
adds are marked `asr_added` and listed in the mix summary when used (E3). Broader ASR
review tooling is deferred (section 6.5).

**E2. Speaker labels must not cross speaker changes.** A labeled interval may not
contain a detected speaker change. Where a change cannot be ruled out, words are
ineligible with reason `possible_speaker_change`. Padding into quiet regions is
limited and never crosses into another turn.

**E3. Every degradation is visible.** Phoneme substitutions, vowel reductions,
homophone/lexical substitutions, ASR-added words, embedding fallbacks, and
diarization fallbacks are recorded in plan metadata and summarized in the console
after each mix. A strict mode disables them.

**E4. Analysis backends fail loudly.** If the requested speaker-verification model
is unavailable, stop with setup instructions. Weaker backends are used only when
selected explicitly and are recorded. Model loading uses local files only unless a
separate preparation command is run. Production code contains no test-only
embeddings.

**E5. Caches are keyed by content and configuration, under the cache root.** No
cache uses CWD-relative paths, file stems, or timestamps as identity.

**E6. Claims must match fresh evidence.** A completion report lists the exact
commands run on the final tree with their full result lines. Narrowed or partial
runs are labeled as such.

**E7. Requirement documents are not edited to match the implementation.** If a
requirement is changed or relaxed, propose it to the user; record accepted changes
as dated notes, never by rewriting the original text.

## 4. Work packages

Each package lists what is wrong, why it matters, how to fix it, and how to verify.
Line numbers are approximate; inspect the current code first.

### C01. Restore a clean, trustworthy baseline

**Status (2026-10-04): done.** Dictionary staging reuses the staged file only when a
content fingerprint of both inputs matches (sidecar `<staged>.inputs.json`; custom file
hashed every call, base file by size and mtime in memory), so same-tick edits
invalidate correctly; replaces retry briefly on Windows sharing errors. The flaky test
passed 15/15 runs and the concurrency test 25/25 with its `xfail` marker removed.
Ruff (`emb_map`) and mypy (`distance_threshold: Optional[float]`) are clean repo-wide.
Step 4 (reverse-order run) is still open.

**Why:** F10-F12. The committed code fails its own checks, and a timing-dependent
cache hides real invalidation problems.

**Fix:**

1. `stage_mfa_dictionary` in [lexicon.py](../src/lexicon.py): compute the hash from
   the normalized staged content every call (the inputs are small text files; hashing
   is fast). Rewrite the staged file only when its content hash differs. Drop the
   timestamp shortcut and the module-level `_STAGED_DICT_CACHE`, or key that cache by
   input content hashes rather than mtimes.
2. Fix the Ruff error (remove or use `emb_map` in
   `tests/test_m4_speaker_review_ux.py`) and the mypy error at `run_mixer.py:959`
   (make `distance_threshold` `Optional[float]` in `SpeakerDiarizer.__init__` and
   resolve the default inside).
3. Remove the `xfail(strict=False)` marker from
   `test_stage_mfa_dictionary_concurrency_filelock` now that it passes; if it can still
   fail, fix the race instead of keeping a non-strict marker.
4. Run the full offline suite twice: once normally and once with the test files in
   reverse order (command in section 7) to catch order dependence.

**Verify:** V01, V02.

### C02. Detect speaker changes inside turns and stop over-padding

**Why:** F1-F3. This is the main remaining way a mix can use the wrong character's
voice. Film dialogue often has replies with gaps shorter than 0.35 s.

**Fix:**

1. In [diarization.py](../src/diarization.py), after energy VAD, split each turn with
   sliding-window speaker embeddings (for example 1.0-1.5 s windows, 0.25 s hop, using
   the already-loaded WavLM-SV model). Detect change points where consecutive window
   embeddings exceed a distance threshold, and split turns there. Assign clusters per
   segment, not per whole turn.
2. Mark segments with low within-segment consistency (high variance between window
   embeddings) as `possible_speaker_change`; words in them are ineligible (E2).
3. Limit padding: at most 0.10 s on each side, only into regions with energy below
   the VAD threshold, and never past the midpoint to an adjacent turn. Record the
   padding used.
4. Optional but recommended: add a pinned pyannote Community-1 backend as an extra
   (`diarization` optional dependency) that provides overlap-aware turns. When
   installed and selected (`--diarize-backend pyannote`), use its overlap regions to
   set `is_overlap`. Without it, document clearly that overlap is not detected.
5. Make speaker thresholds configurable in one place, and record the backend, model,
   thresholds, and VAD parameters in the inventory metadata.

**Verify:** V03, V04, V05.

### C03. Keep Whisper tail enrichment opt-in and make it safer when used

**Status:** the opt-in switch is done (2026-10-04). Enrichment is off by default;
`--asr-fill-tails` on `mix`, `auto`, `ingest`, and `preprocess` turns it on. The setting
is stored in the corpus manifest and is part of the alignment cache key, so changing it
re-aligns affected sources automatically. Tests:
[test_asr_tail_enrichment_optin.py](../tests/test_asr_tail_enrichment_optin.py).

**Why:** F4. Forced alignment places every listed word somewhere in the audio, so a
hallucinated word becomes a selectable "recording" of that word. A scan of the user's
cache before the change found 8 enriched chunks across 5 sources. Some additions were
plausible ("were here", "I hate you"), but two `obi_wan_ep3` chunks received long
garbage tails (misheard droid beeps, "Arthur" for Artoo, other characters' lines).
YouTube's auto-captions come from a large Google model that is generally at least as
accurate as the local Whisper `base.en`, so automatic correction is not reliable.

**Remaining fix (only affects runs with `--asr-fill-tails`):**

1. Load the Whisper model once per run with `local_files_only=True`; if unavailable,
   log once and leave the text unchanged. It is currently reloaded for every chunk.
2. Accept only trailing words whose timestamps start after the last matched caption
   word and lie inside the chunk, with word probability at least 0.6 and segment
   `no_speech_prob` below 0.5; cap at 4 words; reject a short hallucination list
   ("thank you for watching", "subscribe", and similar). Today any trailing text is
   appended, which produced the multi-sentence garbage above.
3. Record per chunk the original caption text and the appended words in the corpus
   manifest, and mark those words `asr_added` in the inventory (`WordOccurrence` and
   its phone events).
4. List any `asr_added` words used in a mix in the substitution summary (C06).

Do not build transcript-source detection, cross-checking, or review tooling in this
round; those ideas are recorded in section 6.5 for later.

**Verify:** V06, V07, V17, V18, V19.
### C04. Cache diarization correctly

**Why:** F5, F6. Collisions can analyze the wrong file; rerunning diarization on every
mix is slow and can change results between runs.

**Fix:**

1. Write audio extracts under `CACHE_BASE_DIR / "audio_extracts" / <media hash>.wav`
   (pass the cache root into `diarize`). Remove the CWD-relative path and the
   stem/mtime check.
2. Cache diarization results (turns, cluster embeddings, assignments) at
   `CACHE_BASE_DIR / "diarization" / <media hash>/<config hash>.json`, where the config
   hash covers backend, model id and revision, thresholds, VAD parameters, and padding.
   Registry mapping is applied after loading, so renaming a speaker does not require
   re-embedding.
3. `--diarize force` recomputes; `auto` reuses the cache.
4. Make `mix` and `ingest` consistent about interactive cluster review (one flag and
   one default, documented).

**Verify:** V08, V09.

### C05. Fail fast and clearly for unverified sources

**Why:** F9. Users wait for a full alignment and then get an unexplained failure.

**Fix:** Before vocal isolation and alignment, if a source has no assertion, no
verified manifest intervals, and no usable diarization backend, stop with an
actionable message (use `--assume-single-speaker`, provide intervals, or install the
diarization backend). Replace the "Proceeding with unverified identity" message.
Optionally offer `--allow-unverified-speaker`, which labels words `unverified` and
requires an explicit `--speaker unverified` request to use them.

**Verify:** V10.

### C06. Report every substitution and degradation

**Why:** F7. Users currently cannot tell why a word sounds wrong, and the ASR gate may
not catch a voicing swap.

**Fix:**

1. Give each candidate a `substitutions` list (for example
   `{"type": "cognate", "target": "G", "used": "K"}`, `vowel_reduction`,
   `lexical_homophone`, `asr_added_word`). Store it on plan segments.
2. After each mix, print a short summary: counts per type and the affected words.
   Include it in plan metadata and in `tools/quality_check.py` output.
3. Add `--strict-phones` that disables cognates, reductions, and lexical substitution,
   failing with the list of missing phones instead.
4. Remove the `W -> UW`, `Y -> IY`, diphthong, and `DH -> D` style entries from the
   default table unless listening tests support them; keep them behind an explicit
   `--allow-loose-cognates` option.

**Verify:** V11.

### C07. Make speaker embeddings explicit and comparable

**Why:** F8. Comparing profiles built with different models gives meaningless scores,
and silent fallback hides that speaker identity has become unreliable.

**Fix:**

1. Store `embedding_model`, `model_revision`, and `embedding_dim` on each registry
   profile and in the registry header. Refuse to match across different models; offer a
   re-enrollment command.
2. Load the SV model with `local_files_only=True`; on failure, raise with preparation
   instructions (E4). Add a preparation command or document the existing one.
3. Allow MFCC or raw WavLM only via an explicit `--diarize-backend` value, marked as
   unreliable in metadata. Delete the FFT pseudo-embedding from production code (move
   it to a test fixture if tests need it).
4. Migrate the existing registry: profiles without model metadata are treated as
   `microsoft/wavlm-base-plus-sv` only if their dimension is 512; otherwise mark them
   for re-enrollment.

**Verify:** V12.

### C08. Repository hygiene and a portable speaker registry

**Why:** F13. Scratch binaries bloated the repo (the user has already removed them).
The speaker registry is worth keeping in git: it is the only durable record of the
user's speaker-naming work. But it currently stores absolute paths into the ignored
`.build` folder and cache-specific source IDs, so it breaks on another machine or
after a cache cleanup.

**Fix:**

1. `.gitignore` already ignores `scratch_*.wav`, `/scratch/`, and everything under
   `/.agents/` except `/.agents/skills/`. Future scratch output goes under
   `.build/scratch/` or the agent's own scratch folder.
2. Keep `config/speaker_registry.json` tracked. Make it portable:
   - Store exemplar audio paths relative to the repository (or cache root) and treat a
     missing preview as "not available" rather than an error.
   - Store source references by media content hash (plus file name for display), not
     by cache-specific IDs alone.
   - Write the file with sorted keys and stable float formatting so diffs stay small.
   - Tests and evaluation always use a temporary registry; nothing in the test suite may
     write to the real file.
3. Do not delete or untrack any user files. If something else looks like it should not
   be tracked, list it in the final report with the command the user can run.

**Verify:** V13, V20.

### C09. Correct documents without rewriting history

**Why:** F16 and E7. Wrong claims mislead future work.

**Fix:**

1. In [ORIGINAL_REQUEST.md](ORIGINAL_REQUEST.md), do not edit the original lines.
   Append a dated note stating that the ">30% overlap" wording at lines 206 and 254 is
   superseded by D5, and that the earlier "completed" statement was premature.
2. In [README.md](../README.md), remove or correct "strict mypy", "global optimum",
   and similar claims; document `--strict-phones`, `--diarize-backend`, and the
   substitution report (`--asr-fill-tails` is already documented).
3. Update [GEMINI.md](../GEMINI.md) and the skill as described in section 6.

**Verify:** V14.

### C10. Tighten the artifact validator

**Why:** F14. A 2-frame tolerance can hide off-by-one scheduling bugs.

**Fix:** In [validate_mix.py](../tools/validate_mix.py), keep the exact frame-count
check, and limit the audio/video duration difference to one frame plus the measured
AAC priming (read from the container's start time/padding metadata), not a fixed two
frames.

**Verify:** V15.

### C11. Make selector tuning maintainable and measurable

**Why:** F15. Many unexplained constants make quality changes hard to reason about and
risk overfitting to one corpus.

**Fix:**

1. Move selector penalties and bonuses into one documented configuration object
   (for example `SelectorWeights` dataclass) loaded from `config/selector.json`, with
   the current values as defaults. No behavior change in this step.
2. Record a per-segment cost breakdown (target, join, bleed, duration, substitution,
   source bonus) in the plan for debugging.
3. Change "global optimum" wording to "dynamic-programming search with path-dependent
   heuristics" in docs and docstrings.
4. Extend the evaluation suite with at least one non-Star Wars source (any
   user-provided clip) before retuning thresholds; record which corpus each constant
   was calibrated on.

**Verify:** V16.

### C12. Final evidence report

Write a report under a unique `.build\reports\round4-<timestamp>\` folder with the
base commit, a hash of the tested tree, each command with its full summary line,
package status (C01-C12, V01-V20), and pending human checks (listening, real
multi-speaker clips, NLE import). Follow E6.

## 5. Regression checks

Use synthetic tones/clips and injected fake models; label them as synthetic.

| ID | Check | Required result |
| --- | --- | --- |
| V01 | Full offline suite, then the same suite with test files run in reverse order | 0 failed, no xpass; same results both orders |
| V02 | Ruff and mypy on `src tests tools run_mixer.py` | 0 errors |
| V03 | Two distinct fake voices with a 0.2 s gap (injected window embeddings) | Split into two segments with different labels, or both words ineligible; never both eligible for one speaker |
| V04 | One voice throughout a 3 s turn | Single segment, words eligible (no over-splitting) |
| V05 | Turn followed by 1 s of quiet containing low-level speech energy of another fake voice | Padding stops at 0.10 s; the other voice's words are not labeled with the first speaker |
| V06 | With `--asr-fill-tails`: caption "Hello there", fake Whisper hears "thank you for watching" (low probability, on the hallucination list) | Nothing appended; reason logged |
| V07 | With `--asr-fill-tails`: fake Whisper hears one high-probability word after the caption end | Word appended, marked `asr_added`, recorded in the manifest, and listed in the mix summary when used |
| V08 | Two `clip.mp4` files with different audio in different folders | Each diarized on its own audio; extracts under the cache root |
| V09 | Run `mix` twice on the same unasserted source | Second run loads cached diarization (embedding function not called) |
| V10 | Unasserted source, diarization backend unavailable, no `--assume-single-speaker` | Exit 2 before isolation/alignment with the actionable message |
| V11 | Target needing a G when the speaker has only K | Plan and console list the cognate substitution; `--strict-phones` exits 2 naming G |
| V12 | Registry profile from a different embedding model | Not matched; re-enrollment message |
| V13 | `git check-ignore scratch_test.wav` | Ignored |
| V14 | `ORIGINAL_REQUEST.md` diff | Only appended dated notes; original lines unchanged |
| V15 | Render whose video is one frame short | Validator fails |
| V16 | Plan JSON for a mix | Contains per-segment cost breakdown and substitution lists; default weights reproduce previous selections on the existing tests |
| V17 | Default run (no flag) on a source whose cue is much shorter than its audio | Enrichment never runs; transcript equals the caption (implemented in `test_asr_tail_enrichment_optin.py`) |
| V18 | Toggle `--asr-fill-tails` on a cached source | Alignment key changes; the source re-aligns once; the manifest records the setting (implemented) |
| V19 | With `--asr-fill-tails`, fake Whisper returns words for 10 chunks | Model constructed once per run, not per chunk |
| V20 | Registry round trip with a temporary registry | Exemplar paths stored relative; a missing preview does not break matching; repeated saves without changes produce identical bytes; test suite leaves the real registry untouched (hash before and after) |

Keep the section-1 guards passing (overlap timing 25,440 samples / 16 frames, XML
audio, `--speaker` filter, D5 eligibility, unprimed gate).

## 6. How Gemini/Antigravity should improve its own guidance

The previous round showed recurring problems: overstated verification, requirement
text edited to match code, scratch files in the repo root, silent fallbacks
encouraged by global rules, and many avoidable tool errors (the session's own
`learning_proposal.md` counted 497). Apply these updates as part of this round.

### 6.1 Project rules: `D:\Code\sentence_mixer\GEMINI.md`

Keep the existing invariants and replace the heuristics sections with:

```markdown
## Data Honesty
- Output speech comes only from source recordings. Analysis models (Whisper, WavLM,
  diarization) never add words, sounds, or speaker labels that are not confirmed.
- Subtitle text is used as-is. Whisper tail enrichment is opt-in
  (`--asr-fill-tails`); its words are marked `asr_added` and reported when used.
  Never enable it by default, and never edit the original subtitle files.
- Every substitution (cognate, vowel reduction, homophone, ASR word) and every
  analysis fallback must appear in the plan metadata and the console summary.
  Never add a silent fallback; prefer a clear error plus an explicit opt-in flag.

## Speaker Safety
- `--speaker` filters; it never labels. Labels come from `--assume-single-speaker`,
  verified manifests, or diarization with a confirmed registry mapping.
- A word is a donor only if one trusted speaker interval covers it fully (40 ms
  silence collar). Possible speaker changes or overlap make words ineligible.

## Verification Before Claims
- Before reporting completion, run on the final tree:
  full offline pytest, ruff, mypy, pip check. Quote the exact summary lines.
- Label narrowed runs ("subset", "single file"). Never call mypy "strict" unless
  `strict = true` is configured.
- Do not add `skip`/`xfail` or widen tolerances to make a run green. Remove
  `xfail` markers once tests pass.

## Workspace Hygiene
- Scratch scripts and audio go under `.build/scratch/` or the agent scratch folder,
  never the repository root.
- Do not edit requirement documents in `plans/` to match the implementation;
  append dated notes and tell the user when a requirement changes.
- Changing a default or a binding decision requires telling the user explicitly.
```

### 6.2 Skill: `.agents/skills/sentence-mixer/SKILL.md`

Update the troubleshooting section so it no longer recommends silent fallbacks:

- **"Could not find a continuous phonetic path":** first report which phones are
  missing for that speaker (`--strict-phones` lists them). Suggest ingesting more
  material or rephrasing. Cognates are a visible last resort, and the plan lists them.
- **Dropped or wrong subtitle words:** fix the subtitle file, or prefer uploader-written
  subtitles over YouTube auto-captions when they exist. `--asr-fill-tails` (opt-in)
  appends confident words Whisper hears after a cue ends; they are marked `asr_added`
  and listed in the mix summary when used. (Already applied to the skill on 2026-10-04;
  extend it with the `asr_added` reporting once C03 lands.)
- **Wrong voice in output:** inspect the plan's speaker and eligibility fields; rerun
  ingest with `--diarize force`; check for `possible_speaker_change` reasons.
- Add a "Before you claim done" checklist mirroring 6.1, and the scratch-folder rule.
- Add the actual commands for running checks (see section 7).

### 6.3 Global rules: `C:\Users\Sean\.gemini\config\GEMINI.md`

Keep the PowerShell, venv-path, quoting, and import-hygiene sections (they address
real repeated errors). Revise the architecture section, which currently generalizes
project-specific habits to every project:

- Replace "Graded Fallback Ladders ... never fatally disconnect" with: "When a search
  or pipeline lacks coverage, prefer an explicit error that lists what is missing.
  Fallbacks must be opt-in or clearly reported, with their cost and effect recorded."
- Replace "Enrich missing data at the ingest boundary" with: "Validate external data
  at ingest. Any automatically inferred data must be flagged with its provenance and
  must not be treated as ground truth without confirmation."
- Remove "always recombine ... over the original ambient background" as a global rule;
  it is a sentence-mixer product default, not a general principle.
- Add a global "Verification Before Claims" rule identical to the 6.1 section, and a
  rule: "Never write scratch files into a repository root; use a scratch or build
  folder that is git-ignored."
- Add: "Do not modify global or user-level configuration files unless the user asks;
  propose the change instead."

### 6.4 Process habits

- At the start of a session, read the newest file in `plans/` and the project
  `GEMINI.md`; restate the binding decisions you will follow.
- After each work package, run its focused tests; at the end, run the full gates once
  on the final tree and paste the summary lines into the report.
- When a run fails, record the failure in the report even if later fixed; do not
  replace evidence with a later partial rerun.
- Keep a short running log of tool errors and their fixes in the agent scratch folder;
  update the global rules only with patterns that recur at least three times.

### 6.5 Deferred: transcript review tooling (do not build this round)

Recorded for later, if YouTube sources need more than the opt-in tail fill:

- Detect `transcript_source` (`subtitles` vs `auto_captions`, for example from inline
  word timestamps in YouTube VTTs) and prefer uploader-written subtitles when both exist.
- Use Whisper as a second opinion only: mark caption words `disputed` where it
  confidently disagrees, and store words only Whisper heard as `asr_suggested`
  (never used until accepted).
- A `review-transcripts` command and on-demand confirmation during interactive mixes,
  saving decisions to `config/transcript_corrections/<media hash>.json` without
  touching the original subtitle files.

## 7. Commands (Windows PowerShell 5)

```powershell
Set-Location 'D:\Code\sentence_mixer'
$ErrorActionPreference = 'Stop'
$Repo = (Get-Location).Path
$Py = Join-Path $Repo '.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NLTK_DATA = Join-Path $Repo '.build\nltk'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYANNOTE_METRICS_ENABLED = '0'
function Run {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Exit ${LASTEXITCODE}: $Program $($Arguments -join ' ')"
    }
}
Run 'git' @('--no-pager', 'status', '--short')
Run 'git' @('--no-pager', 'log', '-1', '--format=%H')
```

Full gates (run at the start to record the baseline, and again at the end):

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'not model and not real_corpus')
Run $Py @('-m', 'ruff', 'check', 'src', 'tests', 'tools', 'run_mixer.py')
Run $Py @('-m', 'mypy', 'src', 'tools', 'run_mixer.py')
Run $Py @('-m', 'pip', 'check')
```

Order-dependence check for V01 (runs test files in reverse alphabetical order):

```powershell
$TestFiles = Get-ChildItem -LiteralPath .\tests -Filter 'test_*.py' | Sort-Object Name -Descending | ForEach-Object { $_.FullName }
Run $Py (@('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'not model and not real_corpus') + $TestFiles)
```

Focused groups while working:

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'not model and not real_corpus',
    'tests\test_diarization_and_registry.py', 'tests\test_m4_speaker_review_ux.py',
    'tests\test_ingest_cli_and_watch.py', 'tests\test_vtt_preprocessor.py',
    'tests\test_mfa_dictionary_staging.py', 'tests\test_cache.py')
```

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'not model and not real_corpus',
    'tests\test_span_selector.py', 'tests\test_challenger_m1_bench_and_selector.py',
    'tests\test_edit_plan.py', 'tests\test_validate_mix.py', 'tests\test_quality_check.py')
```

Cached-model gate (record as blocked if weights are missing; do not download during tests):

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'model and not real_corpus')
```

Ignore-rule check for V13:

```powershell
& git check-ignore -v 'scratch_test.wav'
if ($LASTEXITCODE -ne 0) { throw 'scratch_*.wav is not ignored' }
```

## 8. Completion contract

Done means: C01-C12 implemented, V01-V20 passing (synthetic checks labeled), the
section-1 guards still passing, guideline files updated per section 6, and the
report from C12 written. Do not claim completion if any word can still be a donor
without confirmed speaker coverage, if Whisper can change any transcript without
`--asr-fill-tails`, if any substitution or fallback (including `asr_added` words) is
unreported, or if the final-tree gates were not run.
Listening quality on real clips, real multi-speaker accuracy, and NLE import remain
human checks; list them as pending unless performed.

## 9. Copy-ready prompt

```text
Implement plans\10_04_antigravity_remediation_plan.md in D:\Code\sentence_mixer.

Start from commit cca1595c plus the uncommitted 2026-10-04 changes (opt-in Whisper
tail enrichment, its tests and docs, and the .gitignore update); keep those changes.
Read the plan fully, then plans\9_23_antigravity_remediation_plan.md
(decisions D1-D10 still apply), GEMINI.md, and .agents\skills\sentence-mixer\SKILL.md.
Restate decisions E1-E7 before starting.

Run the baseline gates in section 7 and record the exact results. Then implement C01
first so the baseline is clean, followed by C02-C11 in order, writing the matching
V-checks for each package before fixing the code. Keep the section-1 guards passing.

Keep all output speech source-derived. Do not modify source media, subtitle files,
user dictionaries, or the real speaker registry except through tested code paths using
temporary copies. Do not delete or untrack files or edit original requirement text;
append dated notes and list any proposed git commands in your report. Put scratch files
under .build\scratch, never the repo root. Do not install packages unless a manifest
changes or a check fails for a missing dependency; do not download models during tests.

Apply the guideline updates in section 6 to GEMINI.md, the skill, and the global
GEMINI.md (show the user the global-file diff in your final message).

At the end, run the full gates on the final tree, write the C12 report, and report
each package's status with the exact gate summary lines. Do not commit unless asked.
```
