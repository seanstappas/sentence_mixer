# Antigravity Remediation Handoff: Correctness and End-to-End Integration

Date: 2026-09-22.
Audited revision: `368fcd599f88329ac406baa2acbb4095090690a2`
(`Add comprehensive tests for Milestone 4 and bug fixes for Milestone 1`).

## 1. Assignment, context, and precedence

Implement the remaining fixes in this document. The latest commit made useful
improvements, but it did not complete the earlier remediation or the new user
workflows. Passing helper tests is not evidence that the production CLI uses
those helpers correctly. This is an implementation work order, not a completion
report.

The product is a generic, source-preserving YTP sentence mixer: select and edit
recorded speech/video to say a target phrase as naturally as the available
recordings allow. Do not replace missing source speech with TTS, voice cloning,
generative inpainting, or generated mouth movements. Diarization, ASR, phone
recognition, WavLM, and guide recordings are analysis tools, not output donors.

Read these documents in addition to the relevant current source:

- [Original user requirements and follow-up](ORIGINAL_REQUEST.md).
- [Earlier R01-R18 remediation specification](audit-remediation-plan.md).
- [Architecture](design.md) and [M0-M6 roadmap](milestones.md).
- [Review of the major enhancements](9_21_astra_thoughts.md).
- [Original agent runbook](agent-runbook.md), for environment/model background.
- [Project requirements](PROJECT.md) and [research notes](../research/README.md).
- [Gemini enhancement plan](9_21_gemini_major_improvements_plan.md),
  [design](9_21_gemini_major_improvements_design.md), and
  [bug list](9_21_gemini_bugs.md), as historical context rather than trusted code.

For current implementation status, this document supersedes the earlier audit.
The earlier R01-R18 acceptance requirements still apply unless explicitly
clarified here. Do not copy obsolete snippets, restore deleted document names,
or recreate parallel dictionaries/timelines. Inspect the real interfaces first:
the repository already has `Lexicon`, `TargetGraph`, `TargetPhoneNode`,
`SpanCandidate`, `SourceRecord`, `PlanSegment`, and `EditPlan`.

Preserve raw/vocals/recombined/both/all outputs, optional normalization, legacy
text mixing, and the opt-in WavLM/guide paths. Preserve unrelated dirty changes,
custom pronunciations, source media, registry data, model caches, and existing
exports. Do not commit unless separately requested. Keep source recordings,
transcripts, embeddings, and credentials local.

### What is already improved: retain it

- The PCM renderer rejects negative/out-of-source sample ranges.
- Stereo handling is improved. Explicit cuts-only plans suppress automatic
  leveling, peak clamping, and loudness normalization.
- Ordinary consecutive single-line repetitions and empty TextGrid intervals
  are better preserved.
- `no_vocals.wav` is no longer chosen as a vocal stem.
- XML file definitions are attached instead of being unattached references.
- Explicit `--speaker` takes precedence on the span branch.
- Effective isolation metadata is recorded.
- Recursive discovery, validated basic hybrid syntax, reverse word-pronunciation
  lookup, OOV prompting, punctuation scoring, and freeze rendering have useful
  starting implementations.
- The missing-CER evaluator exception is fixed.
- Ruff, mypy, and `pip check` pass at the audited revision.

Do not mark an entire repair complete because one of these narrower fixes works.

### Audit evidence and limitations

The audit used the existing repository virtual environment, offline model
settings, mocked dependencies where appropriate, and disposable synthetic media.
It did not modify application code or overwrite user exports.

| Check at the audited revision | Result |
| --- | --- |
| `pytest -m 'not model and not real_corpus'` | 209 passed, 8 failed, 9 deselected |
| Cached real WavLM tests | 2 passed |
| Ruff / mypy / `pip check` | All passed |
| Deterministic production-path probes | Confirmed the failures below |
| Subjective listening, held-out speaker quality, target-NLE approval | Not established |

All eight nominal-offline failures required unavailable ASR weights: seven
`TestASRListeningVerification` cases in
[test_challenger_r1_r2.py](../tests/test_challenger_r1_r2.py), plus `test_get_backend`
in [test_listen_verify.py](../tests/test_listen_verify.py). These are test
isolation/prerequisite failures, not measured recognition-quality failures.
Installing weights does not repair their incorrect offline classification.

The September 19 implementation-status report is historical evidence, not proof
that this revision satisfies the roadmap. Generate fresh evidence after repair;
do not reuse the counts above as final results.

## 2. Architecture and behavioral decisions

Implement one connected pipeline:

```text
source media + metadata
  -> content-identified assets and exact time maps
  -> overlap-aware speaker turns and reviewed identity mappings
  -> aligned, eligible word/phone inventory and continuous analysis features

text / explicit phones / reviewed microphone or WAV recognition
  -> shared target preparation and versioned pronunciation lookup
  -> target graph with legal alternatives and hard boundary instructions
  -> measured, source-preserving candidate selection and safe join refinement
  -> finalized audio plan and cumulative picture schedule
  -> WAV / MP4 / XML / validation report from the same frozen decisions
```

Use typed additions to existing contracts where necessary. The diagram is not
permission to create disconnected replacement models or an unused second API.

### Decisions that must be consistent across implementations

1. **Exact gaps are hard constraints; punctuation is a soft preference.**
   An explicit gap replaces implicit/punctuation spacing at that boundary.
   Retained, verified source silence counts toward its duration; do not add the
   same silence twice. Consecutive explicit gaps add. Zero means zero.
2. **Explicit gaps also work in cuts-only mode.** Cuts-only disables automatic
   DSP and automatic pause presets, not intentional target boundary commands.
3. **Leading/trailing gaps are supported.** Hold the first eligible selected
   frame for a leading gap and the last eligible selected frame for a trailing
   gap. Gap-only requests without speech need an explicit unsupported-input
   error unless a separate audio-only visual policy is deliberately implemented.
4. **A local cluster is not a global identity.** `speaker_01` may be a display
   label; its internal ID must be recording-scoped until deliberately mapped.
   A filename, subtitle label, or singleton cluster does not establish identity.
5. **Unknown or overlapping source speech is not automatically eligible.**
   Preserve evidence and exclusions. Do not turn uncertainty into confidence 1.0.
6. **Unknown stress is distinct from unstressed stress `0`.** Never fabricate
   primary stress on all vowels or treat arbitrary digits as valid stress.
7. **Headless mode never prompts.** Preserve the user's requested G2P fallback
   for ordinary OOV spelling, with recorded uncertainty. Missing source phones,
   failed audio recognition, and unresolved speaker requirements still fail
   explicitly; they must not remove target content or invent success.
8. **Keep legacy as the default until the existing quality gates justify a
   deliberate change.** Move shared preparation, speaker enforcement, and
   finalization outside selector branches. Reject genuinely unsupported
   legacy/feature combinations with actionable errors, not silent ignores.
   Document `--selector span` for advanced capabilities until supported otherwise.
9. **Missing measurements are not measurements of zero.** Distinguish valid
   unvoiced speech from absent F0, empty data, failed extraction, and missing files.
10. **Human/model-dependent gates are honest gates.** Complete all implementable
    work, but report unavailable credentials, models, microphone hardware,
    listening judgments, or NLE approval as blocked/pending, not verified.

### Proposed CLI additions, not existing flags

Implement and test these interfaces before using their examples below:

| Proposed interface | Required semantics |
| --- | --- |
| Mutually exclusive `--text`, `--target-audio PATH`, `--record-target` | All produce the same reviewed target representation |
| `--diarize auto\|off\|force` | Default `auto`; use trusted supplied eligibility or run configured local diarization; never download or invent labels silently |
| `--speaker-registry PATH` | Default `config\speaker_registry.json`; injectable private registry for tests/evaluation |
| `--guide-transcript PATH` | UTF-8 declared transcript, separately validated against the guide and target |
| `--cache-dir PATH` | Default current repository cache location; one injectable root for all generated pipeline intermediates |

`auto` diarization may reuse validated turn metadata. If speaker safety requires
diarization and the configured backend is unavailable, give preparation
instructions and stop that request. `off` requires trustworthy supplied intervals
or an explicit, recorded single-speaker assertion; it is not unrestricted pooling.
`force` recomputes the affected analysis without deleting unrelated caches.

Extend `mix`, `auto`, and applicable preprocessing commands consistently. An
inapplicable flag must be rejected. Existing isolation flags on `align` and
`build-inventory` also need honest behavior rather than parse-only support.

## 3. Execution order

| Phase | Packages | Exit requirement |
| --- | --- | --- |
| 1. Establish evidence | A01; establish A16 report/exit tests | Regressions expose real failures; tests do not overwrite user assets |
| 2. Repair shared foundations | A02-A04; shared CLI context from A15 | Valid identities, source bounds, time maps, cache provenance |
| 3. Connect inputs | A05, then A08-A10 | Shared phone validation/graph precedes speaker and target workflow integration |
| 4. Repair selection | A06-A07, A11 | Legal coverage and measured costs drive actual chosen paths |
| 5. Repair delivery | A12-A14 | One frozen audio/picture plan produces faithful outputs |
| 6. Close integration and evidence | A15-A16 | CLI matrix, offline suite, configured model/corpus gates, truthful docs |

Add tests and wire production callers within each package. A15 is not permission
to defer all integration until the end. Schema/identity/time decisions must be
agreed before editing their consumers. Avoid simultaneous competing edits to
contracts, the selector, renderer, or CLI.

## 4. Work packages

### A01. Make the test boundaries trustworthy

**What is wrong / why:** the nominal offline suite still attempts real ASR model
loads. Some new tests fabricate methods on mocks, exercise manually constructed
plans, or assert file existence rather than requested behavior. The real-corpus
smoke test writes fixed `output\smoke_basic.*` names.

**Primary surfaces:** [pytest configuration](../pyproject.toml),
[ASR challenger tests](../tests/test_challenger_r1_r2.py),
[OOV tests](../tests/test_oov_assistant.py),
[real-corpus smoke](../tests/integration/test_real_corpus.py),
[test infrastructure documentation](TEST_INFRA.md).

**Implementation:**

1. Mark every actual weight-loading test `model` and every user-media-dependent
   test `real_corpus`, including tests outside the integration directory.
   Collection and ordinary unit tests must not initialize models or download.
2. Keep deterministic mocked ASR tests in the offline suite. Real model tests
   remain real tests; moving a marker is not a substitute for running their gate.
3. Use real interfaces or autospecced dependencies. A test may inject recognized
   output, but must still call the actual capture/recognition adapter that
   production uses. Do not invent missing methods on a permissive mock.
4. Use `tmp_path` or unique run roots for output, registry, alignment, and cache
   fixtures. Refactor the fixed-name real-corpus smoke before running it.
5. Add the regression matrix in section 5. Demonstrate red before green for
   reproduced failures. For statically identified gaps, label added coverage
   honestly instead of claiming a reproduction that was never run.
6. Keep lint/type checks meaningful. Do not add broad exclusions, blanket
   `xfail`, wider tolerances, or `Any` casts to erase failures.

**Acceptance:** the full offline suite runs without weights, microphones, user
media, downloads, or shared-output mutation. Explicitly requested model/corpus
gates cannot silently skip prerequisites and then claim completion.

### A02. Validate contracts and preserve input identity/content

**What is wrong / why:** renderer bounds checks improved, but plan validation is
still mostly a duration equation. Inventory probing can invent source metadata
after a real failure. Caption handling still needs multiline/short-cue coverage.
Recursive MFA discovery feeds a flattening copier and basename-based identity.

**Reproduction:** two speaker directories containing 23 same-named chunks each
produced only 23 + 1 staged WAVs across two batches: 46 inputs became 24.

**Primary surfaces:** [contracts](../src/contracts.py),
[preprocessing](../src/vtt_preprocessor.py),
[inventory builder](../src/inventory_builder.py),
[alignment batching](../run_mixer.py).

**Implementation:**

1. Separate structural validation from validation against actual decoded assets.
   Validate finite values, positive rates, rational denominators, half-open
   bounds, source/utterance/word/event references, containment, and ordered links.
2. Validate selected legal graph coverage, speaker eligibility, output positions,
   transition endpoints, overlap geometry, picture schedule, and total duration.
   Reject negative indices, cyclic links, undeclared holes, and triple overlaps.
3. Remove invented 48-kHz/stereo source records after probe/decode failure.
   Synthetic tests must explicitly inject a valid source record or probe.
4. Preserve per-interval speaker assignments and verification provenance in
   preprocessing manifests. Do not retain only temporal inclusion and replace
   every interval's identity with the top-level speaker label.
5. Preserve relative speaker/source directories through MFA batch staging and
   result collection, or use collision-proof IDs with an explicit reverse map.
   Inventory IDs must not depend solely on `Path(...).stem`. Check expected
   input/output coverage and reject collisions or unexplained missing results.
6. Preserve genuine repeated multiline cues and short speech. Deduplicate only
   demonstrated rolling-display carry-over. Keep real silence/noise intervals
   and ordered adjacency; do not infer continuity by discarding them.
7. Version incompatible schemas. Provide explicit migration or targeted rebuild
   messages, not plausible defaults for missing provenance.

**Acceptance:** keep the fixed range/stereo/repetition tests green; add corrupt
contract cases, multiline repetitions, short cues, failed probes, and duplicate
basenames below, at, and above the 45-chunk batch threshold. Every submitted
eligible chunk must remain distinguishable and accounted for.

### A03. Use actual stream/sample/frame mappings

**What is wrong / why:** source records declare streams/origins that decode paths
do not consistently use. Native audio samples, analysis samples, container times,
and source video PTS are not interchangeable. Nominal FPS is not a VFR mapping.
The new diarization and recognition helpers also mishandle non-16-kHz input.

**Primary surfaces:** [media-time helpers](../src/media_time.py),
[preprocessing](../src/vtt_preprocessor.py),
[inventory building](../src/inventory_builder.py),
[renderer](../src/video_assembler.py), and analysis consumers.

**Implementation:**

1. Centralize rational/integer mappings between container/cue time, decoded
   native samples, chunk-local MFA time, analysis samples, and source-frame PTS.
   Apply nonzero origins and codec delays exactly once.
2. Explicitly `-map` the selected audio/video stream in all relevant FFmpeg
   commands. Persist actual stream identity, time base, decoded counts, channel
   layout, and conversion details.
3. Use actual frame presentation intervals for VFR, or a documented CFR proxy
   with a reversible original-frame map. Do not relabel the source with project FPS.
4. Reuse a tested analysis-resampling/channel utility. WavLM/CTC/Whisper adapters
   receive their required sample rate and valid mono arrays; passing a 48-kHz
   array to an API that assumes 16 kHz is not resampling.
5. Resample continuous tracks or consolidated source runs, not each arbitrary
   phone independently. Define endpoint rounding once to preserve partition
   invariance and exact output length.
6. Validate separated-stem length, delay, rate, and mapping before transferring
   stem alignment timestamps to original audio/video.

**Acceptance:** synthetic media with distinguishable audio streams, nonzero
origins, 44.1/48-kHz inputs, mixed native FPS, and VFR frame identities must select
the correct actual samples/frames. Equivalent audio at 16/44.1/48 kHz must reach
recognizers at equivalent time/duration, not just pass a metadata assertion.

### A04. Make cache and stem provenance govern the real pipeline

**What is wrong / why:** a cache hit in `ensure_source_aligned` returns before
new stem hashing is reached. Same-path media and changed speaker requirements
reuse stale inventories. Missing/corrupt stem metadata can still be trusted.

**Reproduction:** replacing media bytes and requesting another speaker reused
the prior inventory with isolation enabled and old stems present.

**Primary surfaces:** [CLI cache orchestration](../run_mixer.py),
[cache utilities](../src/cache.py), [media staging](../src/media_time.py),
[vocal isolation](../src/vocal_isolation.py).

**Implementation:**

1. Distinguish readable/path labels from immutable content identity. Validate
   dependency manifests before any pipeline cache return.
2. Fingerprint source bytes, selected streams, transcript bytes, speaker
   intervals/mappings, registry/profile versions, merged lexicon, alignment and
   diarization models/configs, preprocessing version, and relevant options.
   Feature caches also include backend settings.
3. Invalidate affected descendants when an upstream identity changes. An
   unchanged VTT timestamp or cue count is not sufficient evidence of freshness.
4. Cache hits require complete matching metadata. Treat missing/corrupt/old
   provenance as stale or an explicit incompatible-cache error, not a warning
   followed by successful reuse.
5. Retain exact vocal/instrumental role discrimination. Remove filename-prefix
   and newest-file guesses elsewhere in media staging as well.
6. Separate into a private per-attempt directory; verify source/model/options,
   role, rate, channels, length, and delay. Publish the matching stem set and
   metadata together. A force rebuild cannot pair new vocals with old background.
7. Pass an injectable cache context through the pipeline. Tests/evaluation use
   private generated caches; production can reuse validated existing assets.
   Rebuild narrowly rather than deleting the cache root.

**Acceptance:** changed media with preserved path/mtime/size, changed dictionary,
speaker intervals, model/layer/options, incomplete stems, corrupt metadata, and
failed rebuilds cannot return stale success. Identical dependencies still hit.
Test the CLI entry point, not only the isolation helper.

### A05. Compile strict phonetic input and legal pronunciation paths

**What is wrong / why:** whole-word/phrase candidates combine nodes from mutually
exclusive alternatives. Compound lexicon lookup combines alternatives rather
than choosing one per component. Validation accepts `AE9` and silently turns
consonant `T1` into `T`.

**Reproductions:** recorded `R IY1 D` claims `R IY1 D R EH1 D`; `read-off`
becomes `R IY1 D R EH1 D AO1 F`.

**Primary surfaces:** [target parser/graph](../src/target_graph.py),
[lexicon](../src/lexicon.py), [span selector](../src/span_selector.py),
[contracts](../src/contracts.py).

**Implementation:**

1. Parse the entire target with source locations and typed word, phone-group,
   punctuation, and gap boundaries. Report malformed input at its position;
   never reinterpret malformed syntax as ordinary words.
   Preserve original markup separately from the spoken lexical/phone expectations
   consumed by guide alignment and acceptance checks.
2. Use one inventory-aware phone validator in the parser, lexicon, OOV UI, and
   recognizer mapping. Permit stress only as a single `0`, `1`, or `2` on a
   supported vowel. Reject consonant stress, extra digits, unknown phones,
   nonfinite/negative gaps, and malformed/empty phone groups.
3. Represent each occurrence and pronunciation branch explicitly. A candidate
   must follow a connected branch and match compatible realized source phones.
   Check every internal edge and every transition to the next candidate.
4. Derive boundary context from the chosen path, not the first neighbor in an
   ambiguous graph. Count pronunciation priors once per word occurrence.
5. Form compound alternatives using deterministic bounded Cartesian products,
   choosing one pronunciation per component; record any pruning/cap.
6. Complete reverse retrieval for both recorded words and eligible contiguous
   phone runs. A braced group can be a nonce word. Prefer intact suitable source
   realizations without requiring a dictionary spelling.
7. Do not silently ignore explicitly requested stress through a blanket
   stressless fallback. Define and record any permitted equivalence; unknown
   stress may permit alternatives without pretending the user specified `0`.
8. Verify final coverage of each ordered target occurrence exactly once.

**Acceptance:** alternatives, consecutive heteronyms, repeated words, compounds,
explicit groups, nonce groups, incompatible source pronunciations, unknown
stress, and invalid stress syntax have deterministic tests. Disconnected node
lists fail even if their endpoint indices appear valid.

### A06. Connect measured scoring and protect natural source runs

**What is wrong / why:** basic joins still receive `None, None`; feature and
guide arguments can be passed without being used. Pruning can remove intact
runs. Continuation and per-span costs depend on arbitrary partitioning.

**Reproduction:** instrumented basic selection called the join scorer with
missing features at every edited transition and never accessed the supplied
feature/guide objects.

**Primary surfaces:** [feature extraction](../src/features.py),
[selector](../src/span_selector.py), [CLI preparation](../run_mixer.py).

**Implementation:**

1. Prepare a typed, versioned feature store from continuous source tracks and
   actually pass it through the CLI to selection. Reuse extraction across queries.
2. Supply measured boundary records for energy, spectral shape, voicing/F0
   confidence, duration/context, and available alignment/source-quality evidence.
   Missing data needs a reason, not fabricated zero vectors or confidence.
3. Store cost components so a selected path explains target, quality, join,
   pronunciation, guide, and optional embedding contributions.
4. Protect complete matching words/phrases and feasible continuation chains
   before ordinary pruning. Bound internal gaps by actual audio content/duration.
   Do not skip intervening speech, noise, or speaker changes by label alone.
5. Share a continuation predicate between selector and planner: source, speaker,
   physical coverage, ordered events, and retained gaps must agree. Reusing the
   same source range for another target occurrence is an edit.
6. Make suitability costs coverage-weighted/partition-invariant. Charge new-edit
   penalties only for introduced joins; consolidate natural runs before DSP.
   Never reward splitting one word into many phones to collect repeated bonuses.
7. Make caps/configuration/pruning visible, preserve deterministic ties, and
   distinguish truly absent source coverage from exhausted/pruned search.

**Acceptance:** swapping measured compatible/incompatible boundaries changes the
actual chosen path and cost trace. Verify a real CLI feature-store call, not only
scorer arithmetic. Test 31+ competing occurrences, source-order permutations,
long protected runs, true gaps, repeated 14-ms units, and independently enumerated
small exact searches including the real transition costs.

### A07. Plan safe joins, exact gaps, and restrained punctuation preferences

**What is wrong / why:** fixed overlaps are bounded only by the outgoing clip.
Gap handling occurs only between selected spans. Zero, leading, trailing, and
consecutive gaps are mishandled. Punctuation features use weakly validated raw
pitch/energy and can reward arbitrary fragmentation.

**Reproductions:** `hello <gap:250ms> there` becomes one phrase with zero inserted
pause; a trailing gap disappears; `<gap:0ms>` compiles to 250 ms.

**Primary surfaces:** [target graph](../src/target_graph.py),
[selection](../src/span_selector.py), [join planning](../src/edit_plan.py),
[cadence tests](../tests/test_freeze_frame_cadence.py).

**Implementation:**

1. Carry explicit boundary instructions through retrieval, selection, and plan
   validation. Candidates cannot hide a hard internal gap. Reuse suitable
   recorded silence or split legally; meet the exact total without double-counting.
2. Implement the gap decisions in section 2, including zero, leading/trailing,
   consecutive, and cuts-only cases. At 48 kHz, 250 ms means exactly 12,000 samples.
3. Generate bounded legal entry/exit variants and score the variants actually
   rendered. Validate any refinement against required phone content and identity.
4. Bound an overlap by both contributors, safe handles, and neighboring overlap
   geometry. Protect short stops/transients. Keep a legal hard-splice alternative;
   do not delete needed content to achieve a universal crossfade length.
5. Record selected bounds, fade curves, overlaps, and cost. Do not choose a path
   with one join and silently render another.
6. Apply question/exclamation preferences once at the relevant phrase/word
   boundary, using confidence-gated terminal voiced F0 and speaker/recording-
   normalized energy. Missing prosody is neutral, not a favorable measurement.
   Any stress preference must inspect the recorded realization, not award every
   candidate a bonus merely because its target node requests primary stress.
7. Retain the requested weak rising-pitch question preference, but do not claim
   all questions must rise. Lexical stress is not expressive emphasis. Guide or
   explicit reviewed delivery choices may override weak punctuation preferences.
8. Schedule pauses visually through A12. No arbitrary advance into other dialogue,
   black fallback, looping, or audio pitch/time stretching.

**Acceptance:** parser -> retrieval -> selection -> planner -> renderer tests
assert exact gap samples, protected content, and picture holds. Test conflicting
punctuation/gap instructions, retained source pauses, very short incoming phones,
anti-phase joins, unvoiced/missing F0, and equal-cost legal segment partitions.

### A08. Implement real overlap-aware diarization and safe identity mapping

**What is wrong / why:** the new helpers have no production callers. Singleton
turns bypass matching/naming and reuse `speaker_01`. Silence may become a full
speech turn. Overlap and partial coverage can become eligible through proximity.
Ambiguous registry matches update profiles without review. Cluster confidence
also reuses the last cluster's embedding collection.

**Reproductions:** two orthogonal singleton embeddings were merged into one
profile; a perfect existing identity match was bypassed; tied 0.9 similarities
selected the first profile without prompting; a nearby word became eligible
with identity `OVERLAP`.

**Primary surfaces:** [diarization](../src/diarization.py),
[registry](../src/speaker_registry.py),
[preprocessing](../src/vtt_preprocessor.py),
[inventory building](../src/inventory_builder.py), [CLI](../run_mixer.py).

**Implementation:**

1. Integrate a real local overlap-aware diarization backend behind a typed,
   testable adapter. Evaluate pinned
   [pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)
   as the baseline, not energy segmentation mislabeled as speaker diarization.
   Keep acquisition separate and document gating/licensing/Windows dependencies.
2. Preserve regular overlap-aware turns even if exclusive diarization is used
   to assist alignment. Vocal separation and exclusive labels do not remove an
   overlapping voice from donor audio. Compare raw versus stem analysis rather
   than requiring separation or asserting a guaranteed DER improvement.
3. Return no speech for validated silence. A VAD/backend failure is not evidence
   that the entire file is speech. Model input must use A03's rate/channel mapping.
4. Resolve every cluster, including a singleton, through the same identity path.
   Compare the actual cluster centroid to compatible enrolled profiles.
   Fix the reused-last-cluster confidence bug; retain score provenance and margin.
5. Require both calibrated similarity and ambiguity/margin handling. Ambiguous
   matches require review or remain unresolved. Do not contaminate a persistent
   centroid merely because it was the first/nearest match.
6. Support audition, naming, merge, split/reassignment, and rejection without
   discarding local turn evidence. Persist reviewed changes atomically. Record
   embedding model/revision, input processing, enrollment evidence, and profile
   version; incompatible embedding spaces cannot be compared.
7. Noninteractive unknown clusters receive unique recording-scoped IDs, not a
   shared global `speaker_01`. Auto-naming is not human verification.
8. Reconcile aligned words/phones with complete trusted single-speaker coverage.
   Crossing, overlapping, uncertain, partially covered, or out-of-interval
   contributions are excluded pending review. No nearest-interval rescue.
   Any collar policy must be explicit and cannot cross speech or overlap.
9. Check the complete selected source span and splice handles, not just the word
   label. Preserve per-interval IDs through preprocessing and inventory building.
10. Enforce requested identity before every selector. No missing-speaker fallback
    to the whole pool. Multiple possible speakers require a choice; singleton
    labels alone are insufficient evidence for automatic verified selection.

**Acceptance:** real CLI ingestion reaches the backend and reconciliation.
Deterministic fixtures cover singleton identity separation, known matches,
ambiguous ties, silence, overlap, partial/outside words, per-cluster confidence,
source-rate conversion, persistence/reload, and all candidate span sizes.
Then run the held-out real-media protocol in A16; helper tests alone do not
establish Anakin/Obi-Wan identification quality.

### A09. Provide a real microphone/WAV pronunciation workflow

**What is wrong / why:** the OOV mic action calls nonexistent
`record_audio_snippet` and `audio_to_arpabet` methods. The actual recognizer is
ASR text followed by the first dictionary pronunciation. Empty transcription
becomes `{AH0}`, and array sample rates are mishandled.

**Primary surfaces:** [recognizer](../src/phoneme_recognizer.py),
[OOV assistant](../src/oov_assistant.py), [CLI](../run_mixer.py),
[input tests](../tests/test_phonetic_syntax_and_mic.py).

**Implementation:**

1. Define one concrete capture/file ingestion interface with actual sample rate,
   channel layout, duration, and capture status. Connect both the OOV action and
   direct `--record-target`/`--target-audio` to it.
2. Validate silence, clipping, empty/invalid files, cancellation, device errors,
   and permissions. Preserve analysis recordings only under an explicit policy.
3. Implement acoustic phone recognition or clearly separate a weaker assisted
   spelling mode. A CTC phone model such as
   [wav2vec2-lv-60-espeak-cv-ft](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft)
   is a candidate, not a drop-in ARPAbet/stress transcriber.
4. Inspect the pinned model vocabulary and implement a tested supported-symbol
   mapping. Carry unsupported sounds, alternatives, and uncertainty to review.
   Use unknown stress or justified lexical alternatives; never assign all vowels
   stress `1`, silently drop sounds, or choose a heteronym from spelling alone.
5. Offer playback, retry, phone edits, and editable word grouping. Produce the
   same structured target as typed input. Recognized spelling may assist the
   review but must not silently override the spoken alternative.
6. Empty/failed recognition produces a retry/error, never a fabricated schwa.
   Headless mode requires usable supplied/recognized input and reports uncertainty.
7. Tag recording assets as analysis-only. Exclude them and aliases/proxies/stems
   derived from them from every renderable source role.

**Acceptance:** call the real concrete adapter with injected capture/model
dependencies; test cancellation, empty audio, clipping, unsupported labels,
44.1/48-kHz resampling, heteronyms, grouping, direct file input, and OOV reuse.
Assert no microphone/guide-derived samples can reach raw or processed output.
Run a local model integration separately; hardware capture needs an honest
manual/hardware gate, not a mock represented as a real recording.

### A10. Share interactive pronunciation review and validated persistence

**What is wrong / why:** prompts run only for OOV words inside the span branch,
after source alignment. Known alternatives are not reviewed. Invalid phones
can be saved; repeated saves append duplicate lines. The custom, in-memory, and
merged MFA dictionaries are not a single versioned source of truth.

**Primary surfaces:** [lexicon](../src/lexicon.py),
[OOV assistant](../src/oov_assistant.py), [CLI](../run_mixer.py),
[custom dictionary](../dictionaries/custom_pronunciations.txt),
[merged dictionary](../dictionaries/merged_dictionary.dict).

**Implementation:**

1. Introduce a shared pronunciation-resolution service used by target preparation
   and source-transcript OOV handling before MFA. Keep prompts at the UI boundary;
   library code returns structured alternatives and uncertainty.
2. Return lookup provenance: approved custom entry, dictionary alternative,
   G2P suggestion, acoustic proposal, or unresolved. A dictionary miss is not
   the same as a donor-corpus miss.
3. In TTY, default to review for OOV and uncertain known alternatives. Preserve
   per-occurrence choices, including different readings of repeated `read`.
   Present exemplars and allow accept/edit/mic/retry/cancel.
4. Distinguish use-once from saving a reusable variant. Do not globally overwrite
   a heteronym based on one request. Include uncertain source/join audition in
   the same review mechanism where alternatives are presented.
5. Validate through A05 before use or persistence. Deduplicate normalized entries,
   preserve variant semantics, and implement atomic locked read/merge/write or
   equivalent concurrency-safe persistence. Failed writes are visible.
6. Derive the target lexicon and MFA dictionary from the same deterministic,
   versioned merge. Refresh the affected merged artifact on confirmed changes
   and invalidate dependent alignment/selection caches through A04.
7. `--non-interactive` and non-TTY execution never read stdin. Ordinary OOV G2P
   guesses remain supported and reported, but are not silently persisted as
   approved. Cancellation/EOF never means omitting a requested word.

**Acceptance:** default legacy and explicit span CLI paths both reach shared
review for supported targets. Test known ambiguity, repeated occurrences, use-once,
save/reload, malformed phones/stress, duplicate/concurrent writes, interrupted
save, headless fallback, source OOV before MFA, and changed merged-lexicon identity.

### A11. Finish guide and contextual WavLM integration

**What is wrong / why:** the guide is ignored by selection and its transcript is
validated against the same target text assigned to it. WavLM runs, but short
isolated snippets, fixed native-sample windows, incomplete cache identity, and
silent missing-feature fallback remain.

**Primary surfaces:** [guide](../src/prosody_guide.py),
[WavLM backend](../src/wavlm_backend.py),
[model preparation](../tools/prepare_wavlm.py),
[selector](../src/span_selector.py), [CLI](../run_mixer.py).

**Implementation:**

1. Add a real guide transcript input/declared sidecar, compare target coverage,
   align the guide, and preserve independent verification/uncertainty. Forced
   alignment of supplied text is not proof it was actually spoken.
2. Extract target-aligned relative duration, pause, energy, and confidence-gated
   F0 preferences. Normalize appropriately and feed A06/A07; do not synthesize,
   pitch-shift, or time-stretch donor speech to match the guide.
3. Hash guide audio, transcript, alignment, and configuration. Enforce
   analysis-only provenance through every alias and output variant.
4. Keep WavLM optional, supplemental to the complete basic scorer, and analysis-
   only. It is neither free phoneme recognition nor a sample-accurate cut finder.
5. Extract contextual tracks in bounded overlapping batches. Define time-based
   context, stride/receptive-field mapping, padding, and retained center regions.
   Do not encode each candidate pair independently or use 4,800 native samples
   as a rate-independent duration.
6. Cache by audio identity, immutable checkpoint revision, layer, preprocessing,
   context/chunking, and schema. Validate finite dimensions/interval mappings.
   No silent fallback when an explicitly requested backend fails.
7. Preparation must resolve a pinned revision; never silently save mutable
   `main` as if it were a reproducible checkpoint.

**Acceptance:** changing only a guide or injected backend measurements changes
the actual selected eligible path. Basic-plus-guide works without WavLM. Real
cached inference must feed that same selector path. Cover rate equivalence,
chunk edges, model/layer cache invalidation, invalid guides, and analysis leakage.
Do not use a manual `quality_cost` edit as proof of model participation.

### A12. Schedule all video from the cumulative audio timeline

**What is wrong / why:** per-segment flooring/capping and forced minimum one frame
survive in both planner and renderer. The renderer concatenates these shortened
counts rather than honoring the intended cumulative timing.

**Reproductions at 48 kHz / 30 fps / cuts-only:** six 50-ms contributions render
0.300 s audio versus 0.200 s video; 1,000 contributions yield 2,400,000 audio
samples but 1,000 video frames instead of 1,500.

**Primary surfaces:** [planner](../src/edit_plan.py),
[renderer](../src/video_assembler.py), [contracts](../src/contracts.py).

**Implementation:**

1. Finalize absolute audio boundaries, pauses, and overlap geometry first.
   Define a deterministic picture-switch policy for overlaps.
2. For sample rate `S`, FPS `p/q`, and nonnegative boundary `n`, use one integer
   quantizer, for example `Q(n) = (n*p + S*q - 1) // (S*q)`.
   Total frames are `Q(total_samples)`; spans use differences of absolute `Q`s,
   not sums of independently rounded durations.
3. Permit audio contributions with zero separately displayed frames. Remove
   downstream `max(1, ...)` logic that reinstates extra frames.
4. Never cap scheduled duration by `floor(source_duration * fps)`. Use legal
   source-frame mapping from A03 and explicitly hold an eligible frame when
   required. Never advance into unselected neighboring speech to fill duration.
5. Add a canonical picture-span/frame-map contract if audio segments alone
   cannot express overlaps, zero-frame contributions, and holds cleanly.
   Renderer and exporter must consume the same map.
6. Freeze the actual final eligible selected frame for pauses, not an arbitrary
   `-sseof 0.05` guess. Leading/trailing holds follow section 2. Failed extraction
   is a contextual error, not advancing video and then black fallback.
7. Verify emitted frame counts/PTS before muxing. Do not hide divergence with
   audio truncation, `-shortest`, arbitrary black padding, or relaxed validation.

**Acceptance:** six cuts produce exactly 14,400 master samples and 9 frames;
1,000 produce exactly 2,400,000 samples and 1,500 frames. Render a low-resolution
1,000-contribution fixture and count frames, not just plan entries. Add a testable
project video-size setting if needed while preserving existing output defaults.
Test 24, 25, 30, and 30000/1001 fps, subframe clips, overlaps, gaps,
mixed source rates, switch identity,
and end-covering duration error strictly below one frame before codec padding.

### A13. Freeze DSP decisions and publish only validated output sets

**What is wrong / why:** rendering still treats zero gain as automatic, decides
processing after the plan is saved, and writes final-name artifacts before all
requested outputs pass. The "immutable" plan is mutable and weakly validated.

**Primary surfaces:** [contracts](../src/contracts.py),
[planner](../src/edit_plan.py), [renderer](../src/video_assembler.py),
[CLI finalization](../run_mixer.py).

**Implementation:**

1. Resolve channel layout, selected audio roles, gains, fades, overlap curves,
   resampling, normalization, and visual policy before freezing. Zero dB means
   a decided zero; represent auto/unresolved processing separately.
2. Measure normalization on an unpublished analysis master if needed, then save
   the resolved parameters and render the final output. Do not independently
   level halves of one natural source run.
3. Preserve the existing cuts-only DSP fixes. Distinguish default normalization
   from an explicitly conflicting request: cuts-only defaults it off; an explicit
   request for normalization with cuts-only receives a clear conflict error.
4. Preserve stereo/anti-phase content and explicitly handle or reject unsupported
   multichannel layouts. Mono analysis must never silently become output audio.
5. Store per-output-variant source/stem identity and processing while sharing
   one timing plan. Preserve raw/vocals/recombined/both/all behavior and aliases.
6. Validate a genuinely frozen snapshot, give it a stable content hash, and make
   render/export consume it without mutation or hidden processing decisions.
7. Stage artifacts in a unique attempt directory. Validate every requested
   WAV/MP4/plan/XML/report before publication. Publish a completion manifest with
   artifact hashes last; multiple renames alone are not an atomic transaction.
   Preserve the previous complete generation on failure.
8. Retain contextual stderr/errors and identify incomplete attempt artifacts.
   Never replace missing media/failed freeze extraction with silent success.

**Acceptance:** exact cuts-only sample comparisons at a common format/rate,
explicit 0-dB gain, anti-phase stereo, continuous-run partition invariance,
all audio variants, frozen-plan hash stability, and injected failures at each
publication stage. Failed attempts cannot masquerade as a completed new mix.

### A14. Make XML faithfully represent the rendered sequence

**What is wrong / why:** attached file definitions are fixed, but the exporter
guesses the master WAV from the XML basename, ignores freeze semantics, and
uses project FPS for source in/out calculations.

**Reproductions:** a custom XML basename produced zero audio clipitems despite a
rendered master WAV. A pause marked frozen became an advancing source interval
from frame 6 to 14 with no freeze representation.

**Primary surfaces:** [XML exporter](../src/fcp_xml_generator.py),
[CLI](../run_mixer.py), [artifact validator](../tools/validate_mix.py).

**Implementation:**

1. Pass the actual finalized master path/identity explicitly from orchestration;
   do not infer it from an independently selectable XML path.
2. Export the canonical A12 picture schedule, including holds and omitted
   zero-frame picture contributions. Use a supported freeze representation or
   explicit source-derived still/proxy assets with provenance, not advancing video.
3. Distinguish native source rates/time mappings from sequence rate. Preserve
   actual source mapping or use documented relinkable CFR proxies for unsupported
   NLE cases. Do not pretend VFR has one native frame duration.
4. Emit complete reachable media definitions, correct paths/durations/rates,
   valid IDs, explicit audio source-channel mappings, and correct mono/stereo
   track counts. Do not accidentally duplicate one channel as two contributions.
5. Reference the exact processed master and final plan. XML path customization
   must not change audio content, timing, or requested freeze behavior.
6. Expand machine validation beyond a timebase check. Validate source definitions,
   artifact identity, in/out bounds, timeline positions, holds, and audio mapping.

**Acceptance:** real tiny media fixtures cover independent render/XML directories
and basenames, pauses, subframe clips, mixed rates, mono/stereo, missing masters,
unreachable file IDs, and relinking. Prepare a target-NLE import check; do not
claim application-level approval from XML parsing alone.

### A15. Unify CLI behavior, dependency setup, and documentation

**What is wrong / why:** the default legacy branch bypasses new preparation and
ignores parsed controls. New helpers are not reachable workflows. Setup lacks
new optional dependencies, and documentation still claims sub-millisecond MFA
accuracy and uses obsolete paths.

**Reproduction:** default legacy accepted a nonexistent guide and wrong speaker,
passed an unfiltered inventory to synthesis, did not write the requested plan,
and did not propagate cuts-only rendering mode.

**Primary surfaces:** [CLI](../run_mixer.py),
[legacy selector](../src/unit_selector.py), [dependencies](../pyproject.toml),
[README](../README.md), [runbook](agent-runbook.md), [test docs](TEST_INFRA.md).

**Implementation:**

1. Build common source/target preparation, speaker enforcement, selection result
   adaptation, plan finalization, output naming, render/export, and validation.
   Legacy may retain its public API through a deliberate adapter, not a bypass.
2. Honor `auto --output-name`, explicit render files/directories, independent
   plan/XML destinations, audio variants, and effective mode settings.
3. For every option/selector/subcommand combination, either implement its effect
   or reject it clearly. Distinguish explicit flags from defaults where needed.
   Update evaluator legacy arguments rather than passing unsupported flags and
   pretending its configuration is equivalent.
4. Implement the proposed input/diarization/cache/registry/transcript interfaces
   in section 2 and test mutual exclusion, defaults, errors, and help text.
5. Add optional dependency groups/locks for the actual diarization, phone-model,
   and capture backends, with compatible pinned preparation configurations.
   Keep baseline operation lightweight and model loading lazy. Do not install
   packages unless manifests change or validation exposes a missing dependency.
6. Document explicit public-model acquisition, credentials/licensing where
   applicable, offline reuse, and Windows/device prerequisites. No credentials
   in source/config/logs and no implicit downloads during ordinary execution.
7. Update current usage examples and capability limitations, including native
   speaker safety, phonetic review, exact-gap rules, analysis-only provenance,
   cache/schema migration, normalization, and XML masters.
8. Remove sub-millisecond accuracy/naturalness guarantees and unsupported
   completion claims. Timestamp actual evidence and retain pending human gates.

**Acceptance:** a CLI matrix covers `mix`/`auto`, legacy/span, TTY/headless,
text/phones/file/mic, single/multiple/missing speakers, mode/normalization,
features/guide, custom output locations, and each audio variant. Unsupported
combinations fail before final output; none silently ignore user intent.

### A16. Make acceptance, evaluation, and empirical claims truthful

**What is wrong / why:** render success still becomes evaluator success even if
ASR fails. Default token acceptance tolerates missing negation. Unavailable ASR
becomes synthetic WER 1.0. XML/provenance/timing checks are too weak. Some speaker
"unseen" checks reuse enrollment recordings.

**Reproductions:** the evaluator returned normally with two successful renders
and 0% ASR pass rate; `we are going home` passed against `we are not going home`
at WER 0.20.

**Primary surfaces:** [listener](../tools/listen_verify.py),
[validator](../tools/validate_mix.py), [evaluator](../tools/evaluate.py),
[evaluation suite](../evaluation/suite.json),
[speaker tests](../tests/test_diarization_and_registry.py).

**Implementation:**

1. Separate render, structural, ASR/acoustic, and overall acceptance status.
   Preserve error/blocked/skipped reasons; missing metrics remain null, not 1.0.
   Keep the missing-CER fix. Empty suites and unavailable required checks fail.
2. Exit 0 for the requested profile passing, 1 for measured acceptance failure,
   and 2 for operational/configuration failure. Persist reports even on failure.
   Explicit structural-only runs cannot satisfy full release acceptance.
3. Strict mixer acceptance requires every ordered target token occurrence and
   no unintended insertions. Keep tolerant WER/CER as a labeled diagnostic mode,
   not a waiver for missing negations, repeated words, or wrong order.
   Derive expectations from A05's semantic target, not raw braces, ARPAbet token
   spellings, or `<gap:...>` markup. Phone-only/nonce groups need an explicit
   phonetic-coverage profile or a reviewed spoken transcript for word-ASR checks;
   do not invent a spelling or report an inapplicable word-ASR check as passed.
   Requiring word-ASR without the necessary expectation is a configuration error.
4. Use unprimed ASR for an independent recognition gate. Retain assisted/primed
   ASR as a separate labeled metric; passing the expected sentence as a prompt
   is not independent evidence. ASR does not replace legal source/target coverage.
5. Validate exact master sample counts, source hashes/roles/bounds, actual video
   frames/PTS, frozen-plan identity, and full XML agreement. Account for measured
   codec priming/padding, not arbitrary two-frame forgiveness.
6. When requested, measure -16 +/- 1 LUFS and <= -1 dBTP honestly. Unmeasurable
   short or peak-limited under-target material is inconclusive/unmet, not passed.
   Cuts-only is not subject to an unrequested loudness profile.
7. Make evaluation configurations explicit: legacy where applicable, basic span,
   basic+WavLM, and guide when supplied. Hold corpus, eligibility, renderer,
   processing, and unrelated weights constant for meaningful comparisons.
8. Exclude failed/stale artifacts from blinded assets. Use a fresh run directory,
   save assignment keys, stderr, configuration identities, and artifact hashes.
9. Build a manually checked speaker evaluation manifest using suitable existing
   Anakin/Obi-Wan and other available media. Separate enrollment from held-out
   recording/scene evidence; do not enroll and then recognize the same clip and
   call it unseen. Filename labels are not word-level ground truth.
10. Report wrong-speaker eligible-donor rate, overlap acceptance, usable coverage,
    unresolved/review rate, and available diarization metrics for raw versus stem
    analysis. Deterministic overlap fixtures require zero known-overlap donor
    acceptance. Publish real held-out counts and denominators; do not claim a
    universal DER or naturalness guarantee from a tiny fixture.
    A held-out acceptance set must have zero accepted known-wrong-speaker or
    known-overlap donors and nonzero correct eligible coverage for each named
    speaker's declared positive cases. Rejecting everything is not a pass.
    Freeze any additional empirical thresholds on development data before
    inspecting held-out results; report unresolved cases separately.
11. Prepare blinded listening examples for intelligibility, naturalness, joins,
    voice consistency, and visual cadence. Keep human listening and actual NLE
    approval separate from automated gates.

**Acceptance:** real CLI exit/report tests cover all-ASR-failed, unavailable model,
missing CER, empty suite, missing negation, repeated/misordered words, insertions,
stale exports, changed source identity, wrong frames/XML, and unmet loudness.
Complete the evidence/completion contract in section 8.

## 5. Required regression matrix

Implement these in existing relevant test files where practical. Add dedicated
files only when they improve organization. These are acceptance outcomes, not
instructions to hardcode the fixtures into production.

| ID | Fixture / operation | Required repaired outcome |
| --- | --- | --- |
| T01 | 6 x 50-ms cuts, 48 kHz, 30 fps, cuts-only | 14,400 master samples; 9 actual video frames |
| T02 | 1,000 x same cuts | 2,400,000 samples; 1,500 actual frames; bounded switch error |
| T03 | Contributions shorter than a frame; fractional FPS | Audio preserved; zero-frame picture spans allowed; no accumulating rounding |
| T04 | `hello <gap:250ms> there` with contiguous source phrase available | Exactly 12,000 gap samples; no swallowed/doubled gap; correct hold |
| T05 | Zero, leading, trailing, consecutive gaps; cuts-only | Declared semantics preserved through actual rendering |
| T06 | Two-pronunciation `read`, compounds, repeated ambiguous words | One connected pronunciation per occurrence/component |
| T07 | `{AE9}`, `{AH10}`, `{T1}`, malformed groups/gaps | Location-aware rejection; no silent repair/persistence |
| T08 | Controlled basic boundary features swapped | Actual selected path and cost trace change |
| T09 | Guide/backend preferences swapped | Same production selector consumes measured inputs; no manually altered candidate-cost shortcut |
| T10 | 31+ competing occurrences; reordered inventory; split natural run | Protected suitable runs survive; equivalent partitions preserve score and PCM |
| T11 | Changed source bytes, speaker metadata, lexicon/model; same path | Correct dependent invalidation before outer inventory cache return |
| T12 | Missing/corrupt stem metadata; newer `no_vocals`; failed force rebuild | No wrong-role or mixed-generation reuse |
| T13 | Two audio streams, nonzero origins, mixed rates, VFR | Assert the actual selected samples and source-frame identities |
| T14 | Nested duplicate chunk basenames across batching thresholds | Every submitted chunk remains distinct and accounted for |
| T15 | Two different singleton voices; known match; tied registry matches | No accidental identity merge/update; reviewed or unresolved mapping |
| T16 | Silence, overlap, boundary-crossing/outside words and splice handles | No manufactured speech or unsafe donor eligibility |
| T17 | Microphone action through actual adapter; WAV at 16/44.1/48 kHz | Real API is connected; correct rate/duration; editable result |
| T18 | Empty recognition, heteronym, unsupported sound, capture cancellation | Error/review/alternatives, never fabricated `{AH0}` or forced first spelling |
| T19 | TTY/non-TTY OOV; known ambiguity; use-once/save/reload/concurrent write | No blocking headless read; validated atomic deduplication; versioned shared lexicon |
| T20 | Legacy/default wrong speaker, guide/features, cuts-only, requested plan | Enforce shared behavior or reject unsupported combinations explicitly |
| T21 | XML basename/directory differs; pause/hold; mixed rates; stereo | Correct master, source mapping, timeline/holds, and channel assignments |
| T22 | Zero dB, anti-phase stereo, source-run partitioning, all output roles | Frozen processing preserved; no hidden changes or analysis-source leakage |
| T23 | Invalid source/plan, missing video, failure after each artifact stage | Explicit error; prior complete generation preserved; no fake completed output |
| T24 | All ASR failures/unavailable, missing negation, wrong order/repeats, phonetic/gap markup | Correct exits and semantic expectations; strict coverage; explicit profile for phone-only targets |
| T25 | Stale artifact, wrong frame count, invalid XML, unmet loudness | Full validator rejects; no existence-only or widened-tolerance pass |

Negative tests must assert diagnostics/status and absence of completed output,
not only an exception somewhere. Model-free tests may use synthetic tones,
colored/numbered video frames, and explicit fake backend results. They must not
be described as real speech, real microphone capture, or held-out model accuracy.

## 6. Original R01-R18 coverage checklist

Nothing from the earlier repair assignment disappears because this handoff uses
new package names.

| Earlier requirement | Current handoff | Status at audited revision |
| --- | --- | --- |
| R01 invalid data/contracts | A02, A13, A16 | Renderer bounds improved; wider validation/fallbacks incomplete |
| R02 captions/silence | A02 | Ordinary repetitions/empty intervals improved; remaining edge coverage |
| R03 speaker eligibility | A02, A08, A15 | Explicit span speaker precedence fixed; full enforcement missing |
| R04 content-aware caches | A04 | Stem hash helper partial; actual outer pipeline still stale |
| R05 real stream/time mappings | A03 | Incomplete |
| R06 legal pronunciation paths | A05, A10 | Still broken |
| R07 acoustic ranking | A06, A07 | Basic measurements still disconnected |
| R08 protected/partition-invariant runs | A06, A13 | Incomplete |
| R09 safe joins/pauses | A07, A12 | Fixed-overlap and missing-boundary behavior remains |
| R10 cumulative video scheduling | A12 | Both small and long drift reproduced |
| R11 frozen DSP/channels/publication | A13, A15 | Stereo/cuts-only improved; finalized-plan/publication gaps remain |
| R12 faithful XML | A14 | Attached definitions fixed; master/holds/rates incomplete |
| R13 separation provenance | A04 | `no_vocals` fix retained; complete provenance/rebuild needed |
| R14 real guide integration | A11 | Guide still ignored |
| R15 contextual WavLM | A03, A04, A11 | Real backend works; contextual integration/evidence incomplete |
| R16 honest acceptance | A16 | Missing-CER crash fixed; success/coverage gates still wrong |
| R17 tests/tooling | A01, A16 | Static checks pass; offline isolation/integration assertions incomplete |
| R18 CLI/documentation | A15 | Important flags/workflows still bypassed or ignored |

The four newer capabilities are principally A08 (speakers), A05/A09 (phonetic
and microphone input), A07/A12 (cadence), and A10/A15 (default-interactive review).

## 7. Exact Windows PowerShell execution sequence

Use Windows PowerShell-compatible syntax. Do not use Bash continuations or
PowerShell 7-only operators. Run the preamble in every fresh shell; dependent
commands in a block run in that same shell. `Run` deliberately stops on a
nonzero native exit. Fix the failure instead of continuing to declare success.

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
Get-Command ffmpeg.exe, ffprobe.exe -ErrorAction Stop
```

Capture the current baseline once. At the audited revision the first command is
expected to expose the eight unmarked model-dependent failures. Record that
failure, implement A01, and rerun; do not remove real coverage to make it green.

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'not model and not real_corpus')
Run $Py @('-m', 'ruff', 'check', 'src', 'tests', 'tools', 'run_mixer.py')
Run $Py @('-m', 'mypy', 'src', 'tools', 'run_mixer.py')
Run $Py @('-m', 'pip', 'check')
```

### 7.2 Focused development checks

Run the smallest relevant group while implementing; add newly created test
files to the appropriate command. These paths exist at the audited revision.
Do not repeatedly run the full corpus/model suite for local unit changes.

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'not model and not real_corpus',
    'tests\test_contracts.py', 'tests\test_media_time.py',
    'tests\test_inventory_builder.py', 'tests\test_vtt_preprocessor.py',
    'tests\test_cache.py', 'tests\test_vocal_isolation.py',
    'tests\test_gemini_bugs.py', 'tests\test_audit_remediation.py')
```

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'not model and not real_corpus',
    'tests\test_target_graph.py', 'tests\test_span_selector.py',
    'tests\test_features.py', 'tests\test_prosody_guide.py',
    'tests\test_wavlm_backend.py', 'tests\test_diarization_and_registry.py',
    'tests\test_phonetic_syntax_and_mic.py', 'tests\test_oov_assistant.py')
```

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'not model and not real_corpus',
    'tests\test_edit_plan.py', 'tests\test_video_assembler.py',
    'tests\test_cut_accuracy.py', 'tests\test_freeze_frame_cadence.py',
    'tests\test_fcp_xml_generator.py', 'tests\test_loudness.py',
    'tests\test_cli_compat.py', 'tests\test_listen_verify.py',
    'tests\test_validate_mix.py', 'tests\test_evaluation.py',
    'tests\test_challenger_r1_r2.py')
```

### 7.3 Dependencies and model preparation

Reuse the current environment/locks. Install or restore packages only after a
manifest change or a missing-dependency failure. If development requirements
were changed, resolve/review the lock and install intentionally:

```powershell
Run $Py @('-m', 'piptools', 'compile', '--extra', 'dev',
    '--output-file', 'requirements-dev.lock', 'pyproject.toml')
Run $Py @('-m', 'pip', 'install', '-r', 'requirements-dev.lock')
Run $Py @('-m', 'pip', 'install', '--no-deps', '-e', '.')
Run $Py @('-m', 'pip', 'check')
```

This block is conditional, not a mandatory baseline reinstall. Add and document
separate optional backend locks/preparation commands in A15 before invoking
them. Provision public weights only in an explicit preparation step with the
required authorization/license acceptance; never upload user media. Keep tokens
out of command transcripts and tracked files. Restore offline settings afterward.
Do not silently upgrade unrelated packages or guess a model revision.

### 7.4 Full offline gate and configured model/corpus gates

```powershell
Run $Py @('-m', 'ruff', 'check', 'src', 'tests', 'tools', 'run_mixer.py')
Run $Py @('-m', 'mypy', 'src', 'tools', 'run_mixer.py')
Run $Py @('-m', 'pip', 'check')
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'not model and not real_corpus')
Run $Py @('.\run_mixer.py', 'mix', '--help')
Run $Py @('.\run_mixer.py', 'auto', '--help')
Run $Py @('.\tools\validate_mix.py', '--help')
Run $Py @('.\tools\evaluate.py', '--help')
```

After provisioning the exact pinned models, run their gate offline. Preserve
real tests and inspect deselection/skip counts. Then run real-corpus tests only
after A01 has removed fixed output paths and isolated writable caches/registries.

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'model and not real_corpus')
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'real_corpus')
```

If a required model/corpus/device is unavailable, record the exact prerequisite
and gate as blocked. Continue the other packages. Never describe a deselected,
mocked, or skipped gate as empirically verified.

### 7.5 Repaired-pipeline smoke and artifact checks

**Run only after the packages above, including the proposed `--cache-dir`
interface, are implemented.** This deliberately uses a unique output/cache root
and a different XML basename to exercise the previously broken integration.
The source manifest and model prerequisites must be validated first.

```powershell
$RunId = 'remediation-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '-' + [guid]::NewGuid().ToString('N')
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
Run $Py @('.\tools\validate_mix.py', '--render', ($Stem + '.mp4'),
    '--plan', ($Stem + '.plan.json'), '--require-asr')
Run 'ffprobe.exe' @('-v', 'error', '-count_frames',
    '-show_entries', 'stream=codec_type,nb_read_frames,duration',
    '-of', 'json', ($Stem + '.mp4'))
```

A14/A16 must make the validator discover the explicitly recorded XML artifact
from the plan/completion manifest, even when its basename differs. If the final
interface needs an explicit XML argument instead, implement/document it and
update this command before claiming the custom-path case passes.

Run the evaluation suite under another unique directory after A16 forwards a
private generated cache/registry context to all subprocesses:

```powershell
$EvalId = 'evaluation-' + [guid]::NewGuid().ToString('N')
$EvalRoot = Join-Path $Repo ('.build\reports\' + $EvalId)
Run $Py @('.\tools\evaluate.py', '--suite', '.\evaluation\suite.json',
    '--output-dir', $EvalRoot)
```

The suite must declare implemented ablation configurations rather than silently
hardcoding only two. A failure is useful evidence; do not rewrite target phrases
or acceptance thresholds merely to obtain green output.

### 7.6 New input-workflow examples after implementation

These exercise new interfaces and are not baseline commands. The file example
requires a real user-supplied target recording; do not generate synthetic speech
to satisfy it. Recognition/coverage failure must remain visible.

```powershell
$TargetWav = Join-Path $Repo 'input\target_pronunciation.wav'
if (-not (Test-Path -LiteralPath $TargetWav)) {
    throw 'Supply a real target_pronunciation.wav before this manual integration check.'
}
$InputId = 'target-audio-' + [guid]::NewGuid().ToString('N')
$InputRoot = Join-Path $Repo ('.build\reports\' + $InputId)
New-Item -ItemType Directory -Path $InputRoot -ErrorAction Stop | Out-Null
Run $Py @('.\run_mixer.py', 'mix',
    '--manifest', '.\evaluation\smoke_sources.json',
    '--speaker', 'obi-wan', '--target-audio', $TargetWav,
    '--selector', 'span', '--non-interactive',
    '--cache-dir', (Join-Path $InputRoot 'cache'),
    '--render', (Join-Path $InputRoot 'target_audio.mp4'))
```

For an attended microphone/review check, use a fresh output directory:

```powershell
$MicRoot = Join-Path $Repo ('.build\reports\microphone-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $MicRoot -ErrorAction Stop | Out-Null
Run $Py @('.\run_mixer.py', 'mix',
    '--manifest', '.\evaluation\smoke_sources.json',
    '--speaker', 'obi-wan', '--record-target', '--selector', 'span',
    '--cache-dir', (Join-Path $MicRoot 'cache'),
    '--render', (Join-Path $MicRoot 'microphone_target.mp4'))
```

Use a private registry and enrollment/evaluation manifest for attended speaker
review. Test `--diarize auto`, `off`, and `force` according to section 2; do not
rename or merge the user's real registry profiles just to make a fixture pass.

## 8. Completion and evidence contract

Persist a fresh machine-readable report under a unique `.build\reports\` run
directory. Include:

- Base revision plus a diff/tree identity for the tested worktree.
- A01-A16 and T01-T25 status, linking each to tests/artifacts.
- Exact commands, exits, counts, skips, timestamps, environment/backend versions,
  and immutable model/lexicon/config identities.
- Expected versus actual samples, frames, gap durations, coverage, speaker
  exclusions, source hashes, XML references, and output manifest hashes.
- Separate offline, model, real-corpus, human-listening, and NLE gates.
- Blocked prerequisites with the remaining action, not fabricated pass metrics.

Final deliverables:

1. Working integrated code and regression tests for every package.
2. Updated dependency/setup/usage documentation and accurate capability limits.
3. Non-destructive schema/cache migration or explicit targeted rebuild support.
4. Verified output/evaluation artifacts in private run directories.
5. A concise completion report describing fixes, actual check results, remaining
   blockers, and human review still required.

Do not declare full completion if basic features or guide inputs are unused,
speaker safety can be bypassed, hard gaps disappear, frame drift remains,
microphone input only works on a permissive mock, an explicitly requested option
is ignored, or overall success can conceal failed required validation.

Do not claim subjective naturalness, held-out speaker recognition, or Premiere
import approval from synthetic/unit tests. Finish the code and available gates;
report external/human gates honestly. Do not stop after scaffolding or a passing
subset and describe the rest as an optional future enhancement.

## 9. Copy-ready Antigravity prompt

```text
Implement plans\9_22_antigravity_remediation_plan.md end to end in
D:\Code\sentence_mixer.

The baseline is commit 368fcd599f88329ac406baa2acbb4095090690a2. Inspect the
current worktree first and preserve unrelated changes. Read the handoff,
plans\audit-remediation-plan.md, plans\ORIGINAL_REQUEST.md,
plans\9_21_astra_thoughts.md, and the referenced design before changing code.
The handoff describes verified partial fixes as well as remaining failures.
Retain the valid fixes; repair root causes and production integration.

Complete A01-A16 and T01-T25 in dependency order. Add regressions that expose
the actual defects, then implement the fixes and wire every relevant CLI path.
Use the exact PowerShell checks in the handoff, adapting only for real,
documented interface changes. No silent flag ignores, fabricated recognition,
unsafe speaker pooling, disconnected helper implementations, or passing mock
substitutes for real backend integration.

Keep all output speech source-derived. WavLM, diarization, ASR, phone models,
and microphone/guide recordings are analysis-only. Preserve the existing
audio variants and correctly resolved cuts-only/source-derived behavior.
Use one legal target graph, one frozen edit plan, and one cumulative picture
schedule for render and XML.

Do not overwrite source media, user dictionaries/registry entries, existing
exports, or unrelated caches. Make tests and evaluation use private run roots.
Install dependencies only after manifest changes or missing-dependency
failures; acquire models explicitly, pin them, and keep inference offline.
Do not weaken tests, widen tolerances, rewrite expected phrases, or claim
completion from file existence.

Persist current evidence under a unique .build\reports run directory and
update directly related docs. If credentials, models, microphone hardware,
human listening, or NLE approval are unavailable, finish all other work and
record the precise gate as blocked/pending. Do not call it verified.

Continue through the entire implementation and available verification, not
only the first milestone. At the end report each package's status, actual
test results, artifact locations, and remaining external/human gates.
Do not commit unless separately requested.
```
