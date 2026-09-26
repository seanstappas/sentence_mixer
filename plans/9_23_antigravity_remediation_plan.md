# Antigravity Remediation Handoff, Round 3: Identity, Inputs, and Honest Delivery

Date: 2026-09-23.
Audited state: base commit `e9f8a4cdfe1c7e7d46c53c37583a3c9a5f6c614d`
("fixes from antigravity") plus the uncommitted round-3 working-tree changes in
`run_mixer.py`, `src/diarization.py`, `src/edit_plan.py`, `src/lexicon.py`,
`src/phoneme_recognizer.py`, `src/prosody_guide.py`, `src/span_selector.py`,
`src/speaker_registry.py`, `src/video_assembler.py`, `tools/evaluate.py`,
`tools/listen_verify.py`, several tests, and the new
`tests/test_audit_fixes_comprehensive.py`.

## 1. Assignment, context, and precedence

Implement every work package in this document. The round-3 changes fixed most of
the rendering and publication failures from the previous audit, and those fixes
must be kept. However, several requested capabilities are still disconnected or
unsafe, the default selector silently misreads the new inputs, and one new
regression was introduced. This is an implementation work order, not a
completion report.

Product context: a generic, source-preserving YTP sentence mixer. It selects
and edits recorded speech and video so a speaker appears to say target text as
naturally as the available recordings allow. No TTS, voice cloning, generated
mouth movement, or generative inpainting. Diarization, ASR, phone recognition,
WavLM, microphone recordings, and guide recordings are analysis inputs only;
they must never become output audio.

Read with:

- [Previous handoff](9_22_antigravity_remediation_plan.md): still the underlying
  specification (A01-A16, T01-T25) for anything not restated here.
- [Original request and follow-ups](ORIGINAL_REQUEST.md).
- [Architecture](design.md) and [roadmap](milestones.md).
- [Enhancement review](9_21_astra_thoughts.md) for speaker, phonetic, cadence,
  and interactive-review design reasoning.
- [Earlier R01-R18 remediation](audit-remediation-plan.md).

Precedence: this document governs where it is more specific. The summary that
was appended to `ORIGINAL_REQUEST.md` on 2026-09-22 describes word eligibility as
"exclude words with more than 30% overlap". That relaxed the handoff, which
requires complete trusted single-speaker coverage. Decision D5 below is the rule
to implement.

Working rules:

- Inspect the current tree first; line numbers below are approximate and refer
  to the audited state.
- Preserve unrelated changes, media, caches, exports, and registries. Do not
  commit unless asked.
- `dictionaries/custom_pronunciations.txt` has an uncommitted line `1<TAB>W AH1 N`
  of unconfirmed origin. Do not delete or commit it silently; report it (B14).
- Tests and evaluation must use private temporary output, cache, dictionary,
  and registry locations.
- Do not install packages unless you change dependency manifests or a check
  fails because a dependency is missing. Acquire models only in an explicit
  preparation step, pinned, then run offline.

### Verified fixed in round 3: keep these working

These were confirmed with real synthetic renders or deterministic probes. Keep
them as regression guards.

| Behavior | Verified result |
| --- | --- |
| Cuts-only cumulative timing | 6 x 50 ms cuts render 14,400 samples and 9 frames |
| Overlap picture schedule | 6 x 100 ms overlapping cuts render 25,440 samples (0.530 s) and 16 frames (0.533 s); previously 18 frames |
| Source-end hold | A clip crossing the end of a 1 s source video renders all 3 required frames (tpad clone); previously 1 |
| Explicit gaps | Leading 50 ms / internal 250 ms / trailing 100 ms compile to 2,400 / 12,000 / 4,800 samples at 48 kHz |
| Strict stress validation | `{AE9}`, `{AH10}`, `{T1}` rejected by the target parser |
| Output variants | `--audio-track all` now publishes `_vocals` and `_recombined` MP4/WAV files |
| Transactional publication | An injected failure mid-publication restores the prior MP4, WAV, plan, and manifest |
| Evaluator exit | All renders failing now exits 1 |
| Registry ambiguity | A 0.9 / 0.9 tie returns no identity; a match no longer mutates the stored centroid |
| Silence and supplied overlap | Pure silence yields no speech turns; supplied overlapping turns are flagged |
| Recognizer resampling | 0.5 s stereo WAVs at 16 / 44.1 / 48 kHz reach the ASR backend as 8,000 mono samples |
| Concurrent dictionary saves | Two processes saving at once both persist (file lock) |
| MFA result collection | TextGrids keep their relative directories |
| Mixed phone groups in ASR checks | `{...}` groups resolve to dictionary words instead of being discarded |
| Basic acoustic join scoring | Measured boundary features reach the selector (from round 2) |

### Audit evidence and limitations

| Check | Result |
| --- | --- |
| `pytest -m "not model and not real_corpus"` | 253 passed, 17 deselected |
| Ruff, mypy, `pip check` | All pass |
| `pytest -m "model and not real_corpus"` | 2 passed, 1 failed |
| Production-path probes through the real CLI with synthetic media | Confirmed every defect below |

The model-gate failure is `test_get_backend`: ASR weights are not cached in the
repository's offline Hugging Face cache. It is a prerequisite gap, not a
recognition-quality result. No new completion report was written for round 3;
the September 22 report overstates completion. Listening quality, held-out
speaker accuracy, and NLE import were not evaluated.

## 2. Binding decisions

Apply these consistently across all packages.

**D1. Speaker identity sources.** A source span receives a speaker label only
from: (a) a manifest item with `speaker_id` plus `single_speaker_verified: true`
or explicit `speaker_intervals`; (b) `--video` plus `--speaker S` plus
`--assume-single-speaker`, recorded as an explicit whole-source assertion; or
(c) diarization with a confirmed registry mapping (B05). `--speaker` by itself
is a selection filter. It never creates, changes, or overwrites labels.

**D2. Legacy selector scope.** Legacy remains the default for plain text only
until evaluation justifies a change. It must enforce speaker identity, use the
shared lexicon and review choices, and reject span-only inputs (phone groups,
gap tags, `--target-audio`, `--record-target`, guide, WavLM, plan output) with
an actionable error suggesting `--selector span`. It must never reinterpret
markup as words.

**D3. Diarization modes.** `--diarize auto` (default) reuses cached diarization
for the same media hash and configuration, or runs the configured backend for
sources that lack D1(a)/(b) identity evidence. If the backend is unavailable:
when a speaker is requested, stop with setup and assertion instructions; when no
speaker is requested, continue with the source as one unverified identity and
print a warning that different voices may be mixed. `off` uses only D1(a)/(b).
`force` recomputes diarization and reconciliation without re-running MFA.
`--speaker-registry PATH` selects the registry file; tests use a temporary path.

**D4. Anonymous clusters.** Unmatched clusters, including single-turn
recordings, receive `anon:<source content_id>:<cluster_index>` IDs and a
per-recording display label. In a TTY, every unmatched cluster goes through the
same naming/review prompt. Anonymous clusters are eligible only when that exact
ID is requested, never for a named request, and are never merged into named
registry profiles without confirmation.

**D5. Word eligibility.** A word is eligible for identity X only if a trusted,
non-overlap interval of X covers the whole word, except for an explicit
boundary collar (default 40 ms, configurable, recorded in metadata) whose
uncovered part lies in aligned silence and intersects no other speaker or
overlap interval. Record a reason code for every ineligible word:
`partial_coverage`, `overlap`, `unassigned`, `unreviewed_identity`.

**D6. Pronunciation constraints.** A token is *constrained* when its phones come
from an explicit `{...}` group, a per-occurrence review choice, or recognized
target audio. Constrained tokens accept only recordings whose realized phones
match (stress-only differences allowed at a small cost). Unconstrained
dictionary words may use a recording whose realization matches no listed
variant, with a mismatch cost and a recorded diagnostic. Per-occurrence choices
are passed to graph building; "use once" choices are not persisted.

**D7. Guide semantics.** The guide expresses relative per-word prosody
preferences (pitch level and slope, relative duration, pauses). Candidate costs
must be computed from each candidate's measured source audio. A guide requires
a readable transcript file matching the target words; mismatches and
unalignable guides fail instead of warning.

**D8. Spoken target recognition tiers.** Tier 1 (default when MFA is
available): unprimed Whisper word transcription, then MFA forced alignment of
the user's recording against those words with every listed pronunciation
variant, so MFA chooses the variant that matches the audio. Tier 2 (optional):
a pinned CTC phone recognizer with a tested symbol mapping for words or sounds
with no listed variant; stress unknown (vowel without a digit). A spelling-only
fallback must be labeled as such and must offer all dictionary variants for
review. Every tier preserves word boundaries.

**D9. Publication and XML.** XML is generated from the staged master WAV for
inspection and references the final published path. Requested audio is never
silently omitted. The completion manifest lists every published file under a
unique key with its hash.

**D10. Evidence honesty.** The ASR acceptance gate is unprimed. Frame counts
and sample counts are checked exactly against the plan, apart from measured
codec priming. Missing prerequisites are reported as blocked, never as passed or
as synthetic scores.

## 3. Execution order

| Phase | Packages | Exit requirement |
| --- | --- | --- |
| 1. Evidence first | B15 regressions for B01-B03 | New tests fail on the audited tree for the reproduced defects |
| 2. Delivery and CLI correctness | B01, B02, B03 | Fresh-folder XML audio, no relabeling, legacy enforced or rejected |
| 3. Cache foundations | B04 | Full fingerprints; alignment and labeled-inventory layers separated |
| 4. Speaker identity | B06, then B05 | Safe helper rules, then real pipeline wiring |
| 5. Pronunciation inputs | B08, B09, B07 | Structured choices, strict constrained matching, word-preserving recognition |
| 6. Guide | B10 | Guide changes the chosen recording |
| 7. Faithful outputs | B12, B13 | XML holds and native rates; frozen DSP decisions |
| 8. Acceptance and packaging | B11, B14, remaining B15 | Truthful gates, declared dependencies, docs, evidence report |

Wire production callers inside each package; do not defer integration to the
end. Agree on contract and schema changes (inventory eligibility fields,
pronunciation overrides, plan fields) before editing their consumers.

## 4. Work packages

### B01. Generate XML against the published master and record every artifact

**What is wrong / why.** [run_mixer.py](../run_mixer.py) (around line 839 for
span, 987 for legacy) passes the final master WAV path to `generate_fcp_xml`
while that WAV is still in the attempt directory.
[fcp_xml_generator.py](../src/fcp_xml_generator.py) (around line 161) adds
audio clip items only if that path already exists. On a first run into a fresh
folder the XML therefore has no audio track at all. A rerun into the same folder
appears to work only because the previous generation's WAV exists. The
completion manifest keys extra artifacts by file stem (around line 889), so
`mix_vocals.mp4` and `mix_vocals.wav` overwrite each other. The existing test
pre-creates the WAV, and the validator checks only the XML timebase, so neither
catches this.

**Evidence.** Real CLI, `--audio-track all`, custom XML name, fresh folder:
0 audio clip items. Second run into the same folder: 2. Manifest keys:
`master_wav, mix_recombined, mix_vocals, plan, render, xml` for 4 variant files.

**Primary surfaces.** [run_mixer.py](../run_mixer.py),
[fcp_xml_generator.py](../src/fcp_xml_generator.py),
[validate_mix.py](../tools/validate_mix.py).

**Implementation.**

1. Separate the file inspected from the path referenced: read duration,
   channels, and rate from the staged WAV; write the final published path into
   `pathurl`. Apply to both selectors.
2. If XML was requested and the staged master is missing, fail the attempt.
   Never emit an XML without its requested audio.
3. Write correct audio file metadata and explicit channel mapping (see B12).
4. Key manifest artifacts by role or full file name (for example
   `render`, `master_wav`, `vocals_render`, `vocals_wav`, `recombined_render`,
   `recombined_wav`, `plan`, `xml`). Fail if two files map to one key. List every
   published file with its SHA-256.
5. Write a completion manifest for the legacy branch as well.
6. B11 extends the validator to check XML audio and manifest completeness.

**Acceptance.** U01, U02.

### B02. Make `--speaker` a filter and stop relabeling cached sources

**What is wrong / why.** The new staleness rule in
[run_mixer.py](../run_mixer.py) (around line 304) marks a cached inventory stale
when it contains no words for the requested speaker. Re-alignment then labels
every chunk with the requested name, because preprocessing assigns the source
label to all chunks. A cache labeled `obi-wan` requested as `anakin` is rebuilt
as `anakin`, and the mix "succeeds" with the wrong voice. Before this change it
correctly failed with no candidates. It also forces a full MFA re-alignment even
though alignment does not depend on speaker labels. `--video` inputs already take
`--speaker` as a label without any assertion flag. This contradicts the README
("Filter inventory to verified speaker identity") and the design rule that a
request is not identity evidence. Pooled inventories with several identities
and no `--speaker` are still pooled silently.

**Evidence.** Cached words labeled `obi-wan`, `ensure_source_aligned(...,
speaker_id="anakin")`: preprocessing was called again with `speaker_id="anakin"`.

**Primary surfaces.** [run_mixer.py](../run_mixer.py),
[vtt_preprocessor.py](../src/vtt_preprocessor.py),
[inventory_builder.py](../src/inventory_builder.py), [README.md](../README.md).

**Implementation.**

1. Remove the relabeling staleness rule. Identity evidence becomes part of the
   labeled-inventory cache key (B04).
2. Apply D1. For `--video` inputs, only `--assume-single-speaker` (or
   diarization, B05) creates labels; record which evidence was used.
3. A requested speaker absent from the eligible pooled inventory exits 2 before
   selection, listing available identities. No re-alignment, no relabeling.
4. Several identities in the pooled inventory and no `--speaker`: exit 2 listing
   them, instead of pooling different voices.
5. The same rules apply to `mix` and `auto`, and to both selectors.
6. Update help text and README.

**Acceptance.** U03, U04, U05.

### B03. Enforce speaker identity and shared inputs in the legacy selector

**What is wrong / why.** Legacy is the default selector, so ordinary commands
use it.

- The new speaker filter at [run_mixer.py](../run_mixer.py) (around line 946)
  keeps entries whose `speaker_id` is missing. Legacy word and phone entries
  written by [inventory_builder.py](../src/inventory_builder.py) (around lines
  291 and 356) never contain `speaker_id`, so the filter never removes anything,
  and phone entries are not filtered at all.
- [unit_selector.py](../src/unit_selector.py) (around line 89) tokenizes with
  `[a-zA-Z0-9']+`. `hello <gap:250ms> hello` becomes the words
  `hello, gap, 250ms, hello`; `--target-audio` output `{HH AH0 L OW1}` becomes
  the words `HH, AH0, L, OW1`.
- Legacy fallback pronunciations come from G2P directly
  (`get_word_phonemes`, around line 23). Custom dictionary entries and review
  choices never affect legacy output, even though the shared review now runs
  before selector dispatch.

**Evidence.** With an inventory produced by the real builder (words labeled
`speaker-a` in the manifest), `--speaker speaker-b` selected speaker-a's
`hello`. Legacy received `hello <gap:250ms> hello` and `{HH AH0 L OW1}` as text.

**Primary surfaces.** [run_mixer.py](../run_mixer.py),
[unit_selector.py](../src/unit_selector.py),
[inventory_builder.py](../src/inventory_builder.py).

**Implementation.**

1. Derive the legacy word and phone views from `InventoryIndex` at load time
   (so labels, eligibility, and source identity always match the v2 inventory),
   or add `speaker_id`, eligibility, and source identity to both legacy entry
   types with a schema bump.
2. Filter words and phones by the effective speaker. Entries without a label are
   excluded when a speaker is requested. Unavailable speaker: B02.
3. Parse and validate the target before any source alignment. For legacy, reject
   phone groups, gap tags, `--target-audio`, and `--record-target` with exit 2 and
   a message suggesting `--selector span` (existing guide, WavLM, and plan
   rejections stay).
4. Add an optional lexicon and per-occurrence overrides parameter to
   `synthesize_sentence` for fallback pronunciations, keeping the old signature
   working for external callers.
5. Move all cheap validation (target syntax, option conflicts, speaker
   availability once inventories are loaded) ahead of expensive work.

**Acceptance.** U06, U07, U08, U09.

### B04. Fingerprint content fully and split cache layers

**What is wrong / why.** `get_source_id` in [run_mixer.py](../run_mixer.py)
(around line 75) hashes only the file size plus the first and last 64 KB, and
silently falls back to a path hash on errors. Speaker-interval changes, lexicon
changes (merged or custom dictionary used by MFA), MFA model and options, and
preprocessing version still do not invalidate caches. The new ID scheme also
orphans every existing cache, so the next run re-aligns and re-separates every
source without warning.

**Evidence.** Flipping byte 150,000 of a 300,000-byte file kept the same source
ID. Changing `speaker_intervals` on a cached source did not trigger a rebuild.

**Primary surfaces.** [run_mixer.py](../run_mixer.py),
[cache.py](../src/cache.py), [media_time.py](../src/media_time.py),
[vocal_isolation.py](../src/vocal_isolation.py).

**Implementation.**

1. Stream a full SHA-256 of each media file. Memoize by resolved path, size, and
   `st_mtime_ns` in an index under the cache root so unchanged large files are
   not re-hashed every run. Read errors fail loudly; no path-hash fallback.
2. Alignment key: media hash, selected streams, transcript hash, hash of the
   exact dictionary file given to MFA, MFA model identity and options,
   preprocessing schema version.
3. Labeled-inventory key: alignment key plus identity evidence (assertion,
   interval hash, diarization backend/model/revision/config, registry version,
   reviewed mappings). Identity changes rebuild only this layer, not MFA.
4. Stem key: media hash plus separator model and options.
5. Store keys in cache metadata; rebuild only the affected layer;
   `--force-realign` still forces.
6. Migration: detect old-scheme caches, reuse them only after verifying the full
   hash recorded in their metadata, otherwise report once that they need
   rebuilding. Never delete caches automatically; document manual cleanup.

**Acceptance.** U10, U11.

### B05. Wire diarization and the registry into ingestion

**What is wrong / why.** `--diarize` and `--speaker-registry` are defined in
[run_mixer.py](../run_mixer.py) (around lines 499-500) but never read. No
production code calls `SpeakerDiarizer` or
`reconcile_word_intervals_with_speakers`, so multi-speaker sources are still
labeled wholesale. The default registry path is relative to the current working
directory rather than the repository.

**Primary surfaces.** [run_mixer.py](../run_mixer.py),
[diarization.py](../src/diarization.py),
[speaker_registry.py](../src/speaker_registry.py),
[inventory_builder.py](../src/inventory_builder.py),
[contracts.py](../src/contracts.py), [span_selector.py](../src/span_selector.py).

**Implementation.**

1. Add an identity stage after alignment (reconciliation needs aligned words)
   that receives the diarization mode and a resolved registry path.
2. Implement D3 modes, with cached results keyed as in B04.
3. Define an injectable backend interface returning turns with overlap flags and
   per-cluster embeddings. Production backend: pinned
   [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   as an optional extra. The existing energy-VAD clustering may remain only as an
   explicitly selected, recorded fallback marked `overlap_detection: false`; it
   must not be used silently for eligibility on multi-speaker material.
4. Map clusters to registry identities with the existing margin rule. In a TTY,
   prompt for unmatched clusters with timestamps and transcript examples for
   audition; non-interactive runs use D4 anonymous IDs.
5. Reconcile words and phone events with D5 and store per-word speaker, identity
   status, eligibility, and reason in the inventory (schema bump for
   `WordOccurrence` / `AlignedEvent`).
6. Both selectors honor eligibility.
7. Registry writes are atomic and locked; tests never touch the default
   registry.

**Acceptance.** U12, U13.

### B06. Fix diarization helper identity and eligibility rules

**What is wrong / why.**

- Single-turn anonymous IDs in [diarization.py](../src/diarization.py) (around
  line 268) are `spk_<file stem>_<hash of the first 4,096 bytes of samples>`.
  Stems are often identical (every separated stem is `vocals.wav`) and many
  recordings start with digital silence, so different people share an ID. No
  naming prompt appears, even in a TTY.
- Eligibility (around line 403) accepts a word with at least 70% coverage, so up
  to 30% of a word may lie outside the speaker's interval.
- Unreviewed anonymous intervals (confidence 0.0) are still eligible.
- Overlap is detected only from supplied intervals; generated turns come from
  energy VAD without overlap detection.
- Console messages contain non-ASCII arrows, which raise `UnicodeEncodeError` on
  cp1252 consoles when the caller has not reconfigured stdout.

**Evidence.** Two different recordings named `same.wav` with identical leading
samples both became `spk_same_ad7facb2`, with zero prompts. A word 83% inside
its speaker interval was eligible. A word inside a confidence-0.0 anonymous
interval was eligible.

**Implementation.** Apply D4 and D5; take overlap from the B05 backend; use
ASCII in console output (or encode safely); add reason codes; keep the registry
margin fix.

**Acceptance.** U14, U15.

### B07. Recognize spoken targets as words with their actual pronunciation

**What is wrong / why.**

- `recognize_phonetic_string` in
  [phoneme_recognizer.py](../src/phoneme_recognizer.py) (around lines 113-160)
  wraps the phones of the whole utterance in one group. "hello there" becomes
  `{HH AH0 L OW1 DH EH1 R}`, a single token, so no whole-word or phrase candidate
  can match and selection falls back to individual phones (the least natural
  option).
- Recognition is still Whisper text converted to the first dictionary
  pronunciation (`text_to_arpabet`, around lines 56-78). "read" always becomes
  `R IY1 D`, whichever way it was spoken.
- The review prompt accepts edited phones without validation (`{T1}` accepted),
  so the error surfaces later as a crash.
- `transcribe_audio_to_text` returns an empty string on any exception (around
  line 54). A missing dependency or model is reported as "Empty audio
  recognition result".
- The Whisper model name is implicit and unpinned; recording length is fixed at
  2.5 s with no silence or clipping checks or retry.
- The CLI overwrites `args.text` with the raw phone string, losing the spoken
  words for display and ASR expectations.

**Evidence.** With transcription injected as "hello there": the recognizer
returned `{HH AH0 L OW1 DH EH1 R}`; selection produced 7 phone splices, while
`{HH AH0 L OW1} {DH EH1 R}` produced 2 whole-word spans on the same inventory.

**Primary surfaces.** [phoneme_recognizer.py](../src/phoneme_recognizer.py),
[run_mixer.py](../run_mixer.py), [target_graph.py](../src/target_graph.py),
[oov_assistant.py](../src/oov_assistant.py).

**Implementation.**

1. Implement D8 Tier 1: unprimed Whisper words (with word timestamps), then MFA
   alignment of the recording against those words using the merged lexicon with
   all variants; read each word's chosen variant from the phone tier. Document
   the limit: MFA can choose among listed variants, not discover unlisted sounds.
2. Implement Tier 2 as an optional extra with a pinned model revision, an
   explicit, tested mapping from its symbols to ARPAbet, unknown stress as a
   vowel without a digit, and alternatives with confidence.
3. Label the spelling-only fallback and offer every dictionary variant.
4. Return a structured target: per word, the display text, chosen phones,
   source tier, alternatives, and confidence. Build the graph from it (B08
   overrides) and store it in plan metadata. Do not replace the target with a
   raw phone string.
5. Review UI: per-word groups; edit, merge, or split groups; validate every phone
   with `validate_phone` and re-prompt on errors; retry recording; optional
   playback. Headless: accept only results above a documented confidence,
   otherwise exit 2 naming the uncertain words.
6. Report real causes (missing package, missing model, no audio device). Check
   recordings for silence and clipping; make recording length configurable.
7. Keep recordings analysis-only (existing `assert_no_mic_audio_in_plan` plus
   B11 provenance checks).

**Acceptance.** U16, U17, U18.

### B08. Make pronunciation review structured, per-occurrence, and effective

**What is wrong / why.**

- `detect_oov_words` in [oov_assistant.py](../src/oov_assistant.py) (around
  line 39) flags only words missing from the lexicon; known heteronyms such as
  "read read" are never reviewed.
- `resolve_all_in_text` (around line 138) discards each resolution and returns
  the words, and [run_mixer.py](../run_mixer.py) (around line 713) discards that.
  Only choices persisted to the dictionary affect output; "use once" is
  impossible, and choosing a heteronym reading would overwrite it globally.
- An invalid manual entry falls back to the G2P guess without re-prompting; a
  failed or empty microphone attempt falls back to the G2P guess without offering
  a retry; with no G2P available, the assistant invents `["AH0"]`.
- Deduplication in [lexicon.py](../src/lexicon.py) (around line 244) compares
  raw lines, so an existing `BACTA  B AE1 K T AH0` plus the same save creates a
  second, lowercase entry.
- Digits are words to the tokenizer: "1" is treated as an unknown word, which
  likely produced the uncommitted `1<TAB>W AH1 N` entry.
- Saved custom entries do not reach MFA while `merged_dictionary.dict` exists,
  because `ensure_source_aligned` passes only the merged dictionary.

**Primary surfaces.** [oov_assistant.py](../src/oov_assistant.py),
[lexicon.py](../src/lexicon.py), [target_graph.py](../src/target_graph.py),
[run_mixer.py](../run_mixer.py), [unit_selector.py](../src/unit_selector.py).

**Implementation.**

1. Return structured choices:
   `(token_index, text, phones, source, scope, confidence)` where `source` is
   dictionary, custom, g2p, acoustic, or manual, and `scope` is once or saved.
2. In a TTY (default), review OOV words, known words with more than one
   pronunciation, and low-confidence acoustic results.
3. Pass per-occurrence overrides to `TargetGraphBuilder.build` (and to legacy,
   B03). Overridden tokens are constrained (D6).
4. Menu: accept, choose another variant, edit (validated, re-prompt), record
   (retry on failure), skip. Offer "use once" and "save for future". Persist only
   saved choices.
5. Headless: no prompts; dictionary default or G2P guess, recorded as uncertain
   in plan metadata. No invented phones; if no pronunciation is obtainable, exit 2
   naming the word.
6. Deduplicate on parsed, normalized `(lowercase word, phone tuple)` and rewrite
   the file in a stable order while preserving comments.
7. Normalize numerals to words during target preparation (`inflect`, already a
   dependency of `g2p_en`).
8. Build the MFA dictionary deterministically from merged plus custom entries,
   hash it, and include it in the alignment key (B04).

**Acceptance.** U19, U20, U21.

### B09. Match recorded realizations to constrained pronunciations

**What is wrong / why.** The round-3 filter in
[span_selector.py](../src/span_selector.py) (around line 214) rejects a recording
only when its realization matches another branch, or when alternatives exist
and lengths are equal. When the target has a single pronunciation (for example
after choosing one reading), a mismatched recording is still used. Phrase
candidates check only branch 0 and are dropped when a word matches another
branch, losing intact phrases for non-default readings. Plan segments record the
target branch phones, not what was actually recorded.

**Evidence.** Lexicon `read = R EH1 D` only, recording realized `R IY1 D`: the
recording was selected as a whole-word span.

**Implementation.**

1. Compare each realization with every branch: exact match attaches to that
   branch; stress-only differences attach with a small cost; otherwise apply D6
   (reject for constrained tokens; allow unconstrained words with a normalized
   phone-edit-distance cost and a diagnostic).
2. For phrases, attach each word to its matching branch rather than branch 0;
   reject only on conflict with a constrained token.
3. Store realized phones in plan segments next to target phones.

**Acceptance.** U22.

### B10. Make the guide choose between recordings

**What is wrong / why.** `ProsodyGuide.get_candidate_cost` in
[prosody_guide.py](../src/prosody_guide.py) (around lines 62-90) reads
`cand.pitch_semitones`, which `SpanCandidate` does not have. The fallback cost
depends only on the guide contour at the candidate's proportional position in
the target, so every candidate covering the same words gets the same cost. The
proportional node-index mapping ignores durations and silences, and the contour
is recomputed on every call. In [run_mixer.py](../run_mixer.py) (around lines
757-765), a missing transcript defaults to the target text (validating the
target against itself), a nonexistent transcript path is treated as literal
transcript text, and a mismatch only prints a warning.

**Evidence.** Two candidates for the same word, rising versus falling guide:
identical guide costs (0.0 and 0.0) and the same chosen recording.

**Primary surfaces.** [prosody_guide.py](../src/prosody_guide.py),
[span_selector.py](../src/span_selector.py), [features.py](../src/features.py),
[run_mixer.py](../run_mixer.py).

**Implementation.**

1. Compute per-candidate prosody from source audio using the selector's cached
   feature tracks: confidence-gated voiced F0 median and slope in semitones
   relative to the source speaker's median. Missing data is neutral.
2. Align the guide with MFA using its transcript to get per-word intervals;
   compute per-word relative pitch level and slope, relative duration, and
   pauses. Unalignable guide: fail.
3. Compare per covered word, weighted by coverage so the cost does not depend on
   how many candidates a path uses; compute once per candidate (cached).
4. `--guide` requires `--guide-transcript` pointing to an existing UTF-8 file
   whose words match the target; otherwise exit 2.
5. Record the guide hash, transcript hash, alignment summary, and applied
   per-word preferences in plan metadata. The guide never contributes audio.

**Acceptance.** U23, U24.

### B11. Make acceptance tools truthful

**What is wrong / why.**

- [listen_verify.py](../tools/listen_verify.py) (around line 488) passes the
  expected text as the ASR prompt, so the gate is not independent.
- [validate_mix.py](../tools/validate_mix.py) is unchanged: a two-frame duration
  tolerance (around line 256), an XML check that reads only the timebase (around
  line 280, which is why B01 went unnoticed), and a peak-limited loudness waiver
  (around line 204). It does not count frames or check the manifest.
- [evaluate.py](../tools/evaluate.py) records unavailable ASR as WER 1.0 (around
  line 167), and its configurations are hard-coded to legacy and span basic
  (no WavLM ablation).
- Phone groups resolve to the first dictionary word with that pronunciation, so
  homophones depend on dictionary order.

**Implementation.**

1. Default the gate to unprimed transcription; report a primed "assisted"
   metric separately and label it.
2. Validator: exact WAV sample count; exact video frame count via
   `ffprobe -count_frames` equal to `plan.total_video_frames`; audio/video
   duration difference within one frame plus measured AAC priming; XML audio
   clip present, its master path existing and matching the manifest hash,
   sequence length equal to the plan, video clip positions matching segments with
   frames, freeze segments represented (B12); manifest completeness and hashes;
   loudness reported as unmet or inconclusive instead of waived.
3. Evaluator: unavailable ASR is `blocked` with null metrics and exit 2 when ASR
   is required; configurations come from the suite (legacy, span_basic,
   span_wavlm, and span_guide when a guide is supplied) with identical corpus and
   eligibility.
4. Phone-group expectations accept any word of the homophone set; nonce groups
   require a supplied transcript.

**Acceptance.** U25, U26, U27.

### B12. Represent holds and native rates faithfully in XML

**What is wrong / why.** [fcp_xml_generator.py](../src/fcp_xml_generator.py)
still exports pause and freeze segments as advancing source video, so an editor
shows different (possibly neighboring-dialogue) frames than the render. File
rates and in/out points use the project FPS instead of each source's native FPS.
Two audio tracks reference the same file without channel mapping. The renderer
still substitutes black video for missing sources (around line 849 of
[video_assembler.py](../src/video_assembler.py)), and freeze images come from
`-sseof -0.05` on the previous slice (around line 788) rather than the exact last
displayed frame.

**Implementation.**

1. Extract the exact held frame once, save it beside the XML (for example
   `<xml stem>_holds/hold_<n>.png`) with provenance in the manifest, and use the
   same image in the renderer and as a still-image clip item in the XML.
2. Use native source FPS for file rates and in/out points; the sequence uses the
   project FPS. Variable-frame-rate sources use a documented CFR proxy or fail
   explicitly.
3. Emit audio tracks matching the master channel layout (one stereo clip, or
   explicit source-track mapping per channel; mono master gives one track).
4. Missing or unreadable source video is an error unless an explicit audio-only
   visual policy is requested.
5. Importing into the target NLE remains a human gate.

**Acceptance.** U28.

### B13. Freeze DSP decisions in the plan

**What is wrong / why.** `PlanSegment.gain_db` of `None` still makes the renderer
compute per-segment gains at render time (around line 479 of
[video_assembler.py](../src/video_assembler.py)), after the plan is saved.
Normalization parameters are not stored, per-phone leveling can treat halves of
one natural run differently, and re-rendering a saved plan can change the audio.

**Implementation.**

1. Add a finalization pass that computes gains over consolidated natural runs
   with the renderer's read path and stores a float `gain_db` for every
   non-silence segment (0.0 means a decided zero; always 0.0 in cuts-only).
2. Measure loudness on an unpublished analysis master, store the measured
   parameters and target in plan metadata, and apply them in the final render.
3. The renderer raises if any gain is undecided and never chooses gains itself.
4. Rendering does not change the plan or its hash.

**Acceptance.** U29.

### B14. Declare dependencies, update docs, and tidy state

**What is wrong / why.**

- [lexicon.py](../src/lexicon.py) imports `filelock` when saving a
  pronunciation (around line 238), but `filelock` is declared nowhere; it is
  present only because the WavLM dependencies install it. A baseline or dev-only
  install fails when a user accepts a pronunciation.
- `sounddevice`, `faster-whisper`, and a diarization backend are undeclared; no
  optional extras or locks exist for them.
- [README.md](../README.md) and [pyproject.toml](../pyproject.toml) were not
  updated: sub-millisecond alignment claims, obsolete paths, and no
  documentation of the new flags, identity rules, or legacy limits.
- The uncommitted `1<TAB>W AH1 N` dictionary line and orphaned caches (B04) need
  explicit handling.

**Implementation.**

1. Add `filelock` to the baseline dependencies.
2. Add optional extras, for example `mic` (`sounddevice`, `faster-whisper`),
   `asr` (`faster-whisper`), and `diarization` (pinned backend), with compiled
   locks tested on Windows.
3. Update the README: flags, D1 identity rules, D2 legacy limits, recognition
   tiers, guide requirements, cache migration, and validation profiles. Remove
   sub-millisecond and naturalness guarantees.
4. Report the dictionary line to the user and let them decide; B08 numeral
   normalization prevents new digit entries.
5. Write the evidence report required in section 8.

**Acceptance.** U30.

### B15. Replace tests that do not exercise production behavior

Replace or strengthen these; do not delete coverage.

| Test | Problem | Replacement |
| --- | --- | --- |
| `test_fcp_xml_master_wav_permanent_path` in [test_audit_fixes_comprehensive.py](../tests/test_audit_fixes_comprehensive.py) | Pre-creates the WAV, hiding B01 | Real CLI fresh-folder run (U01) |
| `test_prosody_guide_get_candidate_cost` | MagicMock with a `pitch_semitones` attribute real candidates lack | Selector-level guide test (U23) |
| `test_phoneme_recognizer_resamples_non_16k_audio` | Asserts only that a string returns | Fake backend capturing array length and rate |
| `test_lexicon_save_custom_pronunciation_filelock` | Sequential saves | Two-process test with a barrier, plus normalized dedup |
| `test_get_source_id_content_aware` | Tiny files only | Middle-byte change in a file larger than 128 KB |
| `test_t11_*`, `test_t12_*` in [test_cache.py](../tests/test_cache.py) | Assert fixture timestamps and metadata only | Call `ensure_source_aligned` and stem validation |
| Empirical registry test near line 305 of [test_diarization_and_registry.py](../tests/test_diarization_and_registry.py) | Evaluates a clip included in enrollment | Held-out clip |
| T21 in [test_fcp_xml_generator.py](../tests/test_fcp_xml_generator.py) | Counts clip durations only | Assert hold representation (U28) |
| [test_speaker_recognition.py](../tools/test_speaker_recognition.py) | Writes a fixed `.build` registry, no reference labels, passes with zero intervals | Private registry, reference-label manifest, metrics with denominators, fail on zero intervals or identical centroids |

## 5. Regression matrix

Implement in existing test files where practical. Fixtures may use synthetic
tones, colored video, and injected fake backends; label them as such.

| ID | Fixture / operation | Required result |
| --- | --- | --- |
| U01 | Real CLI, fresh folder, `--xml` with a different basename; span and legacy | At least one audio clip item whose path resolves to the existing published master |
| U02 | `--audio-track all` | Manifest lists every published file under unique keys with matching SHA-256 |
| U03 | `--speaker speaker-b` against a speaker-a-only inventory, both selectors | Exit 2 listing available identities; no selection, alignment, or relabeling |
| U04 | Cache labeled `obi-wan`, request `anakin` | No preprocessing or MFA call; labels unchanged; exit 2 |
| U05 | Pooled inventory with two identities, no `--speaker` | Exit 2 listing identities |
| U06 | Legacy, multi-speaker inventory, `--speaker a` | Words and phone fallbacks only from `a` |
| U07 | Legacy with `hello <gap:250ms> hello` | Exit 2 suggesting `--selector span`, before source alignment; never tokens `gap` or `250ms` |
| U08 | Legacy with `{...}`, `--target-audio`, or `--record-target` | Exit 2 suggesting `--selector span` |
| U09 | Legacy OOV word with a saved custom pronunciation | Phone fallback uses the custom phones, not G2P |
| U10 | 300,000-byte media, middle byte changed | New media hash; alignment layer rebuilt |
| U11 | Change intervals, dictionary, MFA model or options, or identity assertion | Only the affected layer rebuilds; identity-only changes make no MFA call |
| U12 | `--diarize force`, injected two-speaker backend with overlap, temporary `--speaker-registry` | Per-word speaker and eligibility stored; selection uses only eligible words of the requested identity; default registry untouched |
| U13 | `--diarize auto`, backend unavailable, `--video` plus `--speaker` without assertion | Exit 2 with setup and assertion instructions; with `--assume-single-speaker`, labels recorded as asserted |
| U14 | Two different recordings, same file name, identical first 64 ms | Different anonymous IDs; TTY prompts once per unmatched cluster (singletons included); registry unchanged |
| U15 | Word 83% covered whose uncovered part is not aligned silence; word whose 30 ms uncovered edge lies in aligned silence; word inside a confidence-0.0 anonymous interval | Ineligible; eligible within the 40 ms collar; not eligible for any named request |
| U16 | `--target-audio` saying "hello there" (fake Whisper, injected aligner) | Two word groups; whole-word selection (2 word spans, not 7 phones) |
| U17 | Recording realized `R EH1 D` | Target `{R EH1 D}` for "read"; spelling-only fallback offers both variants |
| U18 | Edited `{T1}`; missing Whisper package; silent audio; low confidence headless | Rejected with re-prompt; error names the missing package; retry or exit 2; exit 2 naming uncertain words |
| U19 | `read read` in a TTY with different choices | One review per occurrence; graph constrained per occurrence; "use once" not persisted; choices in plan metadata |
| U20 | Save the same pronunciation as an existing `BACTA  B AE1 K T AH0` line | No duplicate; concurrent saves still both persist |
| U21 | Headless OOV with no G2P; target containing "1" | Exit 2 naming the word (no `AH0`); "1" normalized to "one" |
| U22 | Constrained `R EH1 D`, only an `R IY1 D` recording | Not selected; unconstrained reduced realizations still usable with a recorded cost; plan records realized phones |
| U23 | Rising- and falling-pitch candidates for one word | Rising guide chooses the rising candidate; falling guide the falling one; costs differ; works without WavLM |
| U24 | `--guide` without transcript; nonexistent transcript path; mismatched transcript | Each exits 2 |
| U25 | ASR gate with a prompt-recording fake backend | Gate call has no prompt; assisted metric labeled separately |
| U26 | XML without audio; missing manifest artifact; one-frame video deficit; peak-limited under-target loudness | Validator fails each with a specific message |
| U27 | Unavailable ASR; suite listing span_wavlm | `blocked`, null metrics, exit 2; configuration executed from the suite; all-renders-failed still exits 1 |
| U28 | Pause segment; 24 fps source in a 30 fps sequence; mono master | Hold still in XML matching the rendered frame; native in/out; one audio track |
| U29 | Finalized plan rendered twice | Every non-silence segment has a float `gain_db`; identical WAVs; plan hash unchanged; renderer rejects undecided gains |
| U30 | Fresh virtual environment with baseline dependencies only | `filelock` importable and a pronunciation save succeeds |

Keep the round-3 guards from section 1 passing (6 x 50 ms, 6 x 100 ms
overlaps, end hold, gaps, variants, rollback, evaluator exit, registry tie,
silence, resampling, concurrent saves).

## 6. Coverage of the round-3 audit findings

| Audit finding | Package |
| --- | --- |
| Fresh-folder XML has no audio; manifest key collisions | B01 |
| Legacy speaker filter no-op; markup read as words; review ignored | B03, B08 |
| Requesting a speaker relabels the cached source | B02, B04 |
| Diarization flags parse-only; helper IDs, coverage, confidence, overlap | B05, B06 |
| Guide cannot distinguish candidates; transcript handling | B10 |
| Spoken input collapses to one group, spelling-based, unvalidated edits | B07 |
| Heteronyms never reviewed; choices discarded; `AH0`; dedup | B08 |
| Single-pronunciation realization mismatch accepted | B09 |
| Partial fingerprint; intervals, lexicon, model not keyed; orphaned caches | B04 |
| Undeclared `filelock` and optional dependencies; stale docs | B14 |
| Primed ASR; weak validator; WER 1.0 for unavailable ASR; no ablation | B11 |
| XML holds, native rates, channel mapping; black fallback | B12 |
| Render-time gains | B13 |
| Tests that mock the behavior they claim to verify | B15 |

Still-open items from the previous handoff map as follows: A04 to B04, A05 to
B09, A07 to B10/B12, A08 to B05/B06, A09 to B07, A10 to B08, A11 to B10, A13 to
B13, A14 to B01/B12, A15 to B02/B03/B14, A16 to B11/B15.

## 7. Exact Windows PowerShell commands

Use Windows PowerShell 5 syntax (no `&&`, `||`, or PowerShell 7 operators). Run
the preamble in every fresh shell. `Run` stops on a nonzero exit; fix failures
instead of continuing.

### 7.1 Preamble and baseline

```powershell
Set-Location 'D:\Code\sentence_mixer'
$ErrorActionPreference = 'Stop'
$Repo = (Get-Location).Path
$Py = Join-Path $Repo '.venv\Scripts\python.exe'
$env:PYTHONDONTWRITEBYTECODE = '1'
$env:NLTK_DATA = Join-Path $Repo '.build\nltk'
$env:HF_HOME = Join-Path $Repo '.build\huggingface'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
$env:PYANNOTE_METRICS_ENABLED = '0'
$env:MAMBA_ROOT_PREFIX = Join-Path $Repo '.mamba'
$env:MFA_ROOT_DIR = Join-Path $Repo '.build\mfa'
function Run {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Exit ${LASTEXITCODE}: $Program $($Arguments -join ' ')"
    }
}
if (-not (Test-Path -LiteralPath $Py)) {
    throw 'Repository virtual environment missing. Follow the existing setup runbook first.'
}
Run 'git' @('--no-pager', 'status', '--short')
Run 'git' @('--no-pager', 'log', '-1', '--format=%H')
Get-Command ffmpeg.exe, ffprobe.exe -ErrorAction Stop | Out-Null
```

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'not model and not real_corpus')
Run $Py @('-m', 'ruff', 'check', 'src', 'tests', 'tools', 'run_mixer.py')
Run $Py @('-m', 'mypy', 'src', 'tools', 'run_mixer.py')
Run $Py @('-m', 'pip', 'check')
```

### 7.2 Focused test groups

Run the group matching your change; add new test files to the relevant group.

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'not model and not real_corpus',
    'tests\test_cli_compat.py', 'tests\test_cache.py', 'tests\test_gemini_bugs.py',
    'tests\test_challenger_m1_2.py', 'tests\test_audit_fixes_comprehensive.py',
    'tests\test_fcp_xml_generator.py', 'tests\test_validate_mix.py',
    'tests\test_evaluation.py', 'tests\test_listen_verify.py')
```

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'not model and not real_corpus',
    'tests\test_diarization_and_registry.py', 'tests\test_oov_assistant.py',
    'tests\test_phonetic_syntax_and_mic.py', 'tests\test_target_graph.py',
    'tests\test_span_selector.py', 'tests\test_inventory_builder.py')
```

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'not model and not real_corpus',
    'tests\test_edit_plan.py', 'tests\test_video_assembler.py', 'tests\test_cut_accuracy.py',
    'tests\test_freeze_frame_cadence.py', 'tests\test_loudness.py',
    'tests\test_prosody_guide.py', 'tests\test_features.py', 'tests\test_media_time.py')
```

### 7.3 Dependencies (only after editing `pyproject.toml`)

```powershell
Run $Py @('-m', 'piptools', 'compile', '--extra', 'dev', '--output-file', 'requirements-dev.lock', 'pyproject.toml')
Run $Py @('-m', 'pip', 'install', '-r', 'requirements-dev.lock')
Run $Py @('-m', 'pip', 'install', '--no-deps', '-e', '.')
Run $Py @('-m', 'pip', 'check')
```

After adding the optional extras, compile a lock for each one you define:

```powershell
Run $Py @('-m', 'piptools', 'compile', '--extra', 'mic', '--output-file', 'requirements-mic.lock', 'pyproject.toml')
Run $Py @('-m', 'piptools', 'compile', '--extra', 'diarization', '--output-file', 'requirements-diarization.lock', 'pyproject.toml')
```

U30 baseline-install check (downloads public packages into a throwaway
environment; run once after dependency changes):

```powershell
$BaseEnv = Join-Path $env:TEMP ('sm-baseline-' + [guid]::NewGuid().ToString('N'))
Run 'py.exe' @('-3.10', '-m', 'venv', $BaseEnv)
$BasePy = Join-Path $BaseEnv 'Scripts\python.exe'
Run $BasePy @('-m', 'pip', 'install', '-e', $Repo)
Run $BasePy @('-c', "import sys; sys.path.insert(0, r'$Repo\src'); import filelock, lexicon; print('baseline imports ok')")
Remove-Item -LiteralPath $BaseEnv -Recurse -Force
```

### 7.4 Full gates

```powershell
Run $Py @('-m', 'ruff', 'check', 'src', 'tests', 'tools', 'run_mixer.py')
Run $Py @('-m', 'mypy', 'src', 'tools', 'run_mixer.py')
Run $Py @('-m', 'pip', 'check')
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'not model and not real_corpus')
Run $Py @('.\run_mixer.py', 'mix', '--help')
Run $Py @('.\run_mixer.py', 'auto', '--help')
Run $Py @('.\tools\validate_mix.py', '--help')
Run $Py @('.\tools\evaluate.py', '--help')
```

After provisioning pinned models explicitly, run their gate offline. If weights
are missing, record the gate as blocked with the exact prerequisite:

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'model and not real_corpus')
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-p', 'no:cacheprovider', '-m', 'real_corpus')
```

### 7.5 Real smoke run with XML and frame checks

Run only after B01-B04 are implemented. Uses a unique run folder and private
cache, and a different XML basename to exercise B01.

```powershell
$RunId = 'round3-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N')
$RunRoot = Join-Path $Repo ('.build\reports\' + $RunId)
$RunCache = Join-Path $RunRoot 'cache'
$Stem = Join-Path $RunRoot 'mix_basic'
$Xml = Join-Path $RunRoot 'different_timeline_name.xml'
New-Item -ItemType Directory -Path $RunRoot -ErrorAction Stop | Out-Null
Run $Py @('.\run_mixer.py', 'mix',
    '--manifest', '.\evaluation\smoke_sources.json',
    '--speaker', 'obi-wan', '--text', 'Hello there',
    '--selector', 'span', '--features', 'basic',
    '--mode', 'source-derived', '--fps', '30000/1001',
    '--non-interactive', '--cache-dir', $RunCache,
    '--render', ($Stem + '.mp4'), '--plan', ($Stem + '.plan.json'),
    '--xml', $Xml)
[xml]$Timeline = Get-Content -LiteralPath $Xml -Raw
$AudioClips = @($Timeline.SelectNodes('//audio/track/clipitem'))
if ($AudioClips.Count -lt 1) { throw 'Exported XML contains no audio clip items.' }
$PathUrl = $Timeline.SelectSingleNode('//audio/track/clipitem/file/pathurl').InnerText
$MasterPath = ([Uri]::UnescapeDataString($PathUrl) -replace '^file://localhost/', '') -replace '/', '\'
if (-not (Test-Path -LiteralPath $MasterPath)) { throw "XML references a missing master WAV: $MasterPath" }
$Plan = Get-Content -LiteralPath ($Stem + '.plan.json') -Raw | ConvertFrom-Json
$Frames = & ffprobe.exe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of csv=p=0 ($Stem + '.mp4')
if ($LASTEXITCODE -ne 0) { throw 'ffprobe failed' }
if ([int]$Frames -ne [int]$Plan.total_video_frames) { throw "Rendered $Frames frames; plan requires $($Plan.total_video_frames)." }
Run $Py @('.\tools\validate_mix.py', '--render', ($Stem + '.mp4'), '--plan', ($Stem + '.plan.json'), '--require-asr')
```

If ASR weights are not provisioned, the last command fails for that reason;
record the ASR gate as blocked instead of removing `--require-asr`.

### 7.6 Evaluation run

Acceptance failures are evidence, so capture the exit code instead of stopping:

```powershell
$EvalRoot = Join-Path $Repo ('.build\reports\evaluation-' + [guid]::NewGuid().ToString('N'))
& $Py '.\tools\evaluate.py' '--suite' '.\evaluation\suite.json' '--output-dir' $EvalRoot
$EvalExit = $LASTEXITCODE
Write-Output "Evaluation exit code: $EvalExit (0 passed, 1 acceptance failures recorded, 2 operational error)"
```

### 7.7 Attended checks (human gates)

- Speaker identity: after B05/B06, run an interactive ingestion of a
  multi-speaker source with a temporary `--speaker-registry` and `--diarize force`,
  name clusters when prompted, then run a held-out recording against the frozen
  registry. Use a manually checked enrollment/evaluation manifest; report
  wrong-speaker acceptance, overlap acceptance, correct coverage, and unresolved
  counts with denominators.
- Spoken target: `--record-target --selector span` in a real terminal with a
  microphone; confirm word groups, variant choice, edit validation, and retry.
- NLE: import the smoke XML into the target editor; confirm audio, holds, and
  timing.

## 8. Completion and evidence contract

Write a machine-readable report and a short Markdown summary under a unique
`.build\reports\` run folder containing:

- Base commit plus a hash of the tested working tree.
- Status of B01-B15 and U01-U30 with links to tests and artifacts.
- Exact commands, exit codes, pass/fail/skip/deselect counts, timestamps,
  dependency and model versions.
- Measured samples, frames, XML audio references, manifest hashes, and speaker
  eligibility counts for the smoke run.
- Blocked gates with the exact missing prerequisite.
- Human gates (listening, speaker held-out accuracy, microphone, NLE) listed as
  pending unless actually performed.

Do not declare completion if any of these remain true: XML can omit requested
audio; `--speaker` can relabel or pool voices; legacy reads markup as words or
ignores speaker filtering; diarization flags are parse-only; the guide cannot
change the chosen recording; spoken targets lose word boundaries; a required
dependency is undeclared; or a gate can pass without the evidence it claims.

## 9. Copy-ready Antigravity prompt

```text
Implement plans\9_23_antigravity_remediation_plan.md end to end in
D:\Code\sentence_mixer.

The audited state is commit e9f8a4cdfe1c7e7d46c53c37583a3c9a5f6c614d plus the
uncommitted round-3 changes. Inspect the current tree first and preserve
unrelated changes. Read this handoff, then plans\9_22_antigravity_remediation_plan.md,
plans\ORIGINAL_REQUEST.md, plans\9_21_astra_thoughts.md, and plans\design.md.
Section 1 lists round-3 fixes that were verified; keep them working.

Follow the binding decisions D1-D10 exactly and implement B01-B15 in the
phase order of section 3. Write the U01-U30 regressions first where they
reproduce known defects, confirm they fail, then fix the production code and
wire every CLI path. Use the PowerShell commands in section 7, adapting them
only for documented interface changes.

Keep all output speech source-derived. Diarization, ASR, phone recognition,
WavLM, microphone and guide recordings are analysis-only. Do not overwrite
source media, user dictionaries, registries, existing exports, or unrelated
caches; tests and evaluation use private temporary locations. Leave the
uncommitted "1<TAB>W AH1 N" dictionary line alone and mention it in your report.
Install packages only after changing dependency manifests or when a check fails
for a missing dependency; acquire models explicitly, pin them, run offline.
Do not weaken tests, widen tolerances, mock the behavior under test, or claim
completion from file existence.

Write the evidence report described in section 8. If a model, credential,
microphone, listening panel, or NLE is unavailable, finish everything else and
record that gate as blocked or pending. Do not commit unless asked. At the end,
report the status of each package, actual test results, artifact locations,
and remaining human or external gates.
```
