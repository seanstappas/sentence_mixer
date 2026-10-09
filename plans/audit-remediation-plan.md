# Implementation Audit: Agent Repair Plan

Date: 2026-09-21.
Audited revision: `05e9edc2` (`Improvements to sentence mixing.`).

## 1. Assignment and scope

Repair the implementation, not just the tests or completion report. The audit
found useful infrastructure, but several roadmap features are disconnected and
several output invariants are broken. This document supplies the context, intended
fix, implementation steps, and acceptance evidence for another agent.

Read this alongside:

- [Design](design.md) for architectural reasoning.
- [Milestones](milestones.md) for the M0-M6 contracts and release gates.
- [Original execution runbook](agent-runbook.md) for environment/model setup.
- [Later user requirements](ORIGINAL_REQUEST.md) and [project contracts](PROJECT.md)
  for ASR verification, loudness normalization, and vocal separation.

The later requirements are intentional additions. Do not remove vocal isolation,
ASR, loudness controls, or the existing combined/isolated/recombined output modes
merely because the older roadmap described a narrower baseline. Keep original
combined audio available and distinguish processed/separated outputs honestly.
Neither WavLM nor guide audio may synthesize output speech.

This is a repair specification, **not a claim that the repairs already exist**.
Inspect the current worktree before editing; paths and interfaces below describe
the audited revision. Preserve unrelated changes, source media, dictionaries,
model caches, and existing outputs. Do not commit unless asked.

### Evidence already collected

The audit was read-only and used synthetic fixtures plus existing local outputs:

- 36 focused core tests passed despite the reproduced defects below.
- The broader suite with `not model and not real_corpus`, run offline, yielded
  120 passed, 14 failed, and 3 deselected. Most failures involved unmarked ASR
  tests requiring uncached weights; this is not evidence of poor ASR accuracy.
  A separate evaluator exception was reproduced without any real model.
- Two cached WavLM model tests passed offline. Real inference works, but the
  purported ranking-participation test does not exercise backend-driven selection.
- Ruff reported seven errors; mypy reported five; `pip check` passed.
- The September 19 implementation-status report is stale. Its M0-M6
  `automated_verified` labels are not proof of current integration or correctness.

Do not reuse these historical counts as your final results. Rerun the relevant
checks against your repaired revision and record the actual commands and exits.

### Finding-to-fix index

| Audited finding | Repair sections |
| --- | --- |
| Basic acoustic scoring is disconnected; guide is a no-op | R07, R14 |
| Accumulating video drift | R05, R10 |
| Speaker restrictions can be bypassed | R03 |
| XML contains references without media definitions | R12 |
| Words cover incompatible pronunciation branches | R06 |
| Actual pipeline caches ignore content/dependency changes | R04, R13, R15 |
| Invalid source ranges become silent successful output | R01, R11, R16 |
| Demucs background can be selected as vocals | R13 |
| Subtitle deduplication removes real repetitions | R02 |
| Weak ASR/evaluation/validator acceptance | R16 |
| Inadequate tests, test markers, lint/type failures | R17 |
| WavLM integration and evidence are incomplete | R15 |
| Other coupled gaps: pruning, continuation, coupling, pauses | R08, R09 |
| Other coupled gaps: stereo, processing, finalized-plan fidelity | R11 |
| Ignored CLI options and overstated documentation/status | R18 |

### Recommended execution order

1. Establish R17's offline test boundaries and add failing regression fixtures.
   Keep the old passing tests; replace misleading assertions, not valid behavior.
2. Repair input integrity: R01-R05 and R13. Agree on shared schema/mapping changes
   before wiring the callers; avoid separate source-ID/time conversion schemes.
3. Repair legal selection and naturalness: R06-R09.
4. Repair final planning/rendering/export: R10-R12 and R18's CLI routing.
5. Complete the optional analysis integrations: R14-R15.
6. Finish R16's acceptance harness, run R17's complete verification, and update
   R18's documentation/status from the evidence.

Implement regression tests with each repair, not only at the end. Work can
overlap where independent, but avoid simultaneous competing edits to contracts,
the selector, renderer, or CLI.

## 2. Repair work packages

### R01. Reject invalid source data instead of manufacturing successful output

**Context / why:** [Contract validation](../src/contracts.py) checks only a small
subset of invariants. [PCM rendering](../src/video_assembler.py) clamps source
ranges and pads shortages. A 3.0-3.1 second speech request against a two-second
source produced 4,800 silent samples at 48 kHz without failing.
[Inventory building](../src/inventory_builder.py) also catches media-probe errors
and invents a 48 kHz/stereo source record for "synthetic fixtures."

**Fix:**

1. Separate cheap structural validation from validation against actual source
   assets. Run both before publication; run structural checks when loading plans
   and inventories. Do not make tests the reason for production fallback data.
2. Validate identities/references, positive rates, valid rational denominators,
   finite numbers, half-open source bounds, actual decoded frame counts, and
   word/phone/utterance containment. Validate ordered predecessor/successor links,
   including same-source/speaker consistency and cycles.
3. Validate output starts/ends, transition indices, declared overlaps, target
   coverage, and video schedule. Matching only the sum of durations is inadequate.
   Undeclared gaps, triple overlaps, negative indices, and missing assets fail.
4. Reject source truncation, missing speech, decode failure, and missing required
   video explicitly. Remove silent black-video fallback for broken video input.
   Audio-only sources require a separately declared visual policy, not an
   automatically invented successful video source.
5. Replace the inventory probe catch with a contextual error. Synthetic tests
   must inject an explicit valid `SourceRecord`/probe dependency.
6. Handle resampler length rounding with a declared rational sample mapping,
   not arbitrary zero padding. Any permitted endpoint adjustment must be small,
   documented, and verified; it must never conceal an out-of-source request.
7. Version incompatible schema changes. Old data needs an explicit migration or
   rebuild error; do not fill missing provenance with plausible defaults.

**Acceptance:** extend [contract tests](../tests/test_contracts.py),
[inventory tests](../tests/test_inventory_builder.py), and
[renderer tests](../tests/test_video_assembler.py). The out-of-range reproduction
must raise before a completed output appears. Corrupt paths/hashes, cyclic links,
wrong rates, invalid overlaps, and internally inconsistent timelines must fail.
A valid explicit pause still works. Include source/segment IDs in diagnostics.

### R02. Preserve genuine caption repetitions and ordered silence

**Context / why:** [VTT preprocessing](../src/vtt_preprocessor.py) treats matching
text within 0.5 seconds as rolling duplication. Ordinary adjacent "No!" / "No!"
cues became one. It also drops all cues at or below 50 ms. The
[TextGrid parser](../src/inventory_builder.py) opens praatio with empty intervals
excluded, jeopardizing the ordered silence context needed by selection.

**Fix:**

1. Separate ordinary cues from proven rolling-display carry-over. Use actual
   rolling structure/timestamps and retained-versus-new lines, not gap size or a
   suffix match alone. Preserve ordinary repeated words and repeated full cues.
   When evidence is ambiguous, preserve/flag it rather than deleting speech.
2. Keep the existing rolling-caption spacer-gap regressions, but add ordinary
   short-gap repetitions. Empty display spacers are not spoken text; meaningful
   short cues need a quality diagnostic rather than indiscriminate deletion.
3. Preserve multiline order, punctuation, cue-time mapping, and speaker tags as
   metadata. Log/record carry-over removals so a corpus can be audited.
4. Preserve empty-label TextGrid intervals as silence, and retain noise events.
   Verify equivalent semantics between praatio and the fallback parser. Do not
   catch malformed TextGrids broadly and silently reinterpret them.
5. Validate chunk-local to source-sample conversion, overlap deduplication, and
   real gaps. No removed silence/noise interval may become a false continuation.

**Acceptance:** extend [VTT tests](../tests/test_vtt_preprocessor.py),
[cut tests](../tests/test_cut_accuracy.py), and
[inventory tests](../tests/test_inventory_builder.py). Cover ordinary repetitions
at zero/10/500 ms gaps, genuine rolling lines with spacers, overlapping cues,
speaker tags, short meaningful cues, and phone-silence-phone TextGrid tiers.
Assert resulting text, sample ranges, and ordered event links, not just counts.

### R03. Enforce one eligible speaker across every selected contribution

**Context / why:** [Preprocessing](../src/vtt_preprocessor.py) uses interval
annotations only as a time filter, discarding interval speaker IDs.
[Inventory construction](../src/inventory_builder.py) applies one global ID.
[Phrase retrieval](../src/span_selector.py) checks only the first word.
[CLI resolution](../run_mixer.py) overrides an explicit speaker request when the
inventory contains one ID. A speaker-A request selected an A/B phrase in a fixture.

**Fix:**

1. Propagate each annotated interval's speaker ID and verification method into
   chunks, utterances, words, and events. Require containment in an eligible
   interval. A cue crossing speakers/overlapping speech needs corrected
   annotation; do not guess a text split or assign it wholly to the first speaker.
2. Centralize eligibility checks. Apply them to phrase, word, phone, continuation,
   direct-inventory, and legacy-adapter paths, then validate again in the plan.
3. An explicit `--speaker` always wins or produces "requested speaker unavailable."
   Auto-select only when exactly one verified eligible speaker exists. Multiple
   eligible speakers without a request are an error, not a pooled inventory.
4. Unverified/unknown speech is not eligible by default. The explicit
   `--assume-single-speaker` assertion requires an ID and is recorded as a user
   assertion, not presented as automatic diarization or independent verification.
5. Check every event/word in an extended phrase, including source identity and
   continuity. A matching next word alone is insufficient.

**Acceptance:** add A/B alternating speech and overlapping-speaker fixtures,
multi-source same-speaker success, wrong explicit speaker, ambiguous no-speaker,
unknown speaker, and asserted single-speaker cases to
[CLI](../tests/test_cli_compat.py), [inventory](../tests/test_inventory_builder.py),
and [selector](../tests/test_span_selector.py) tests. Assert source event speakers
in the resulting plan, not merely `plan.speaker_id`.

### R04. Make content-aware caching control the actual pipeline

**Context / why:** [CLI reuse decisions](../run_mixer.py) mostly use paths/mtimes,
not the new [cache helper](../src/cache.py). Replacing same-path media and changing
speaker metadata still reused the old inventory. Several derived audio files are
reused merely because they exist. Directory publication deletes the old directory
before moving the new one, which is not an atomic replacement.

**Fix:**

1. Use one source-content identity scheme throughout ingestion, alignment,
   features, stems, inventories, rendering, and export. Human-readable path IDs
   may label a cache, but are not proof that its contents match.
2. Persist dependency fingerprints and check them in actual CLI cache lookups:
   - Media bytes, selected streams, decode/time/channel configuration.
   - Transcript bytes, speaker annotations/assertions, preprocessing version.
   - Dictionary contents, phone inventory, MFA acoustic/G2P model identities and
     alignment options. A model alias alone is insufficient.
   - Alignment identity, lexicon/schema versions, inventory-building options.
   - Analysis audio/stem identity and basic/WavLM/guide extraction configurations.
3. Rebuild only affected descendants. Reuse on unchanged content; invalidate on
   changed content even with the same path or preserved mtime. Distinguish missing
   transcript inputs from intentionally self-contained validated inventories.
4. Do not auto-discover stems using loose filename prefixes. Require exact
   provenance matches; apply R13's model/content checks.
5. Stage builds in unique sibling directories. Publish immutable fingerprinted
   entries after validation, then atomically update a small pointer/manifest where
   needed. Coordinate parallel builders; never delete a currently readable cache
   directory before its replacement is ready.
6. Give old caches an explicit migration/rebuild path. Do not wipe the user's
   entire cache or force re-alignment on every run.

**Acceptance:** extend [cache tests](../tests/test_cache.py) and
[pipeline tests](../tests/integration/test_pipeline.py). Exercise the CLI reuse
path, not only hash utilities. Independently change media, VTT text, speaker
intervals, dictionary, model revision, isolation mode, and feature config. Verify
correct descendants invalidate, unchanged runs hit, interrupted builds do not
publish, and concurrent readers never see an incomplete entry.

### R05. Use real stream/time mappings, including offsets and VFR

**Context / why:** [Source probing](../src/media_time.py) records a selected audio
stream and audio start time, but decode commands omit explicit stream mapping
and downstream code treats samples/rates as source video PTS. Nominal
`r_frame_rate` alone does not describe variable-frame-rate material.

**Fix:**

1. Explicitly map the selected audio/video streams in every relevant FFmpeg path.
   Persist actual stream indices, time bases, decoded origins, and decoded sample
   counts. Do not let FFmpeg's automatic default differ from the recorded choice.
2. Centralize mappings between cue/container time, chunk-local alignment time,
   native decoded samples, analysis samples, and video presentation timestamps.
   Apply source start offsets exactly once. Use rational/integer calculations
   until serialization/FFmpeg argument boundaries.
3. Derive original-frame selection from actual frame PTS/presentation intervals
   for VFR. A documented CFR proxy is acceptable only with a persisted mapping to
   original frames; merely relabeling a VFR source as project FPS is not.
4. Record channel/sample-format conversions. Resample continuous source tracks
   or consolidated natural runs so artificial unit partitions do not restart
   resampling filters and alter preserved speech.
5. Validate separated-stem delay/length/rate mappings before using their alignment
   timestamps against original video. Recompute analysis when its input changes.

**Acceptance:** extend [media-time tests](../tests/test_media_time.py) and synthetic
FFmpeg integration tests. Include two distinguishable audio streams, a nonzero
audio/video start offset, 44.1/48 kHz sources, mixed native FPS, and a small VFR
source with known frame identities. Assert selected samples/frames, not just
probe metadata. Report an unsupported mapping explicitly rather than guessing;
do not mark the roadmap's media-coverage gate complete while it remains unsupported.

### R06. Cover exactly one legal pronunciation path

**Context / why:** [Word/phrase retrieval](../src/span_selector.py) combines every
node for a token across all pronunciation branches. A three-phone recording of
"read" claimed `R IY1 D R EH1 D`, crossing a nonexistent edge.
[Target context](../src/target_graph.py) picks the first predecessor/successor at
branching boundaries. [Hyphenated lexicon fallback](../src/lexicon.py) can also
concatenate all variants of the first component into one impossible pronunciation.

**Fix:**

1. Represent branch/pronunciation identity explicitly. Match recorded
   `realized_phones` to one compatible branch, retaining node order and branch ID.
   Use declared stress/reduction equivalences only; log any allowed equivalence.
2. Whole words and phrases must traverse one connected legal path, not all nodes
   sharing a token index. Internal and between-candidate graph edges must exist.
   Multiple compatible alternatives may yield separate candidates.
3. Compute cross-word phonetic context from the actual predecessor/successor
   choice in the path; do not collapse a branching graph to its first neighbor.
4. Form compound pronunciations by combining one variant per component, with
   deterministic bounded enumeration and diagnostics for any cap.
5. Verify each target token occurrence and pronunciation exactly once in the
   final path. Repeated words are distinct token occurrences. Charge a
   pronunciation prior once per word, not once per phone/span.

**Acceptance:** extend [target-graph](../tests/test_target_graph.py) and
[selector tests](../tests/test_span_selector.py), adding lexicon cases there or
in a new dedicated test file. Cover "read" alternatives, consecutive ambiguous
words, incompatible recorded realizations, repeated tokens, and compounds with
multiple component variants. Reject disconnected node lists even if their start
and end happen to look valid.

### R07. Connect measured acoustic/prosodic features to real selection

**Context / why:** [Selection](../src/span_selector.py) currently calls
`compute_join_cost(None, None, ...)` at every edited basic-path join.
[Feature extraction](../src/features.py) exists, but its observations do not
control the main selector. Word/phrase quality costs are always zero.

**Fix:**

1. Introduce/reuse a typed feature-store interface backed by continuous cached
   source tracks. The CLI must prepare it and the selector must actually query
   it for candidate suitability and entry/exit descriptors. Avoid `Any` plumbing
   that makes an unused dependency appear implemented.
2. Pass actual boundary records into join scoring. Retain explicit F0 confidence,
   voicing, energy, spectral, stress/context, duration, and source-quality terms.
   Unknown alignment confidence stays unknown; do not invent measured quality.
3. Distinguish unvoiced speech, silence, and unavailable observations. Empty
   windows currently become zero RMS/MFCC values; make them explicitly unavailable
   with a reason, not plausible measured data. Extraction/asset failures surface.
4. Use speaker/phone/context duration statistics where supported. Sparse-data
   priors must be declared fallbacks, not universal duration/pitch targets.
   Preserve legitimate expressive variation rather than flattening everything.
5. Scale and version score components; retain per-candidate/join explanations.
   Derive weights from development comparisons, not a claimed research guarantee.
   Proven original continuations incur no introduced-edit/acoustic-join penalty.

**Acceptance:** extend [feature tests](../tests/test_features.py) and
[selector tests](../tests/test_span_selector.py). Use equally eligible candidates
with controlled boundary observations: swapping the measured energy/spectrum/F0
compatibility must change the selected path and reported cost. Include a CLI
integration proving the real feature store is used. Keep low-confidence F0
missing, test high-F0/unvoiced material, and prove extraction is cached rather
than repeated once per possible transition.

### R08. Protect source runs and make selection independent of partitioning

**Context / why:** [Retrieval](../src/span_selector.py) truncates whole-word
occurrences by inventory order and phone occurrences by duration. Phrase length
is hardcoded, silence skipping is unbounded, and continuation accepts either
sample adjacency or a successor link. Fixed phrase/word/phone costs make identical
speech cost differently depending on its arbitrary segmentation.

**Fix:**

1. Retrieve/protect complete matching words/phrases and their feasible successor
   chains before ordinary pruning. Candidate budgets may not silently remove the
   only intact run. Make caps/configuration and pruning statistics explicit.
2. Bound phrase gaps by actual duration and retained silence/noise content. A
   source span includes everything audible between its endpoints; do not match
   labels while overlooking intervening speech, speaker changes, or long gaps.
3. Share one continuation predicate between search and planning. It requires
   same source/speaker, consistent event order, and retained physical sample
   coverage. A successor link cannot excuse a dropped gap. Reusing an identical
   source interval for repeated target speech remains an edit.
4. Make target/quality costs coverage-weighted and partition-invariant. Count
   introduced edits only where new joins exist. Prefer an intact phrase because
   of its measured suitability and avoided edits, not unconditional zero cost.
5. Consolidate verified natural runs before DSP while retaining their token/event
   provenance. Support longer runs via protected continuation edges rather than
   requiring an unbounded phrase candidate.
6. Retry a wider bounded search if pruning might have removed coverage. Distinguish
   exhausted search from truly absent vocabulary. Use stable deterministic ties.

**Acceptance:** strengthen [selector tests](../tests/test_span_selector.py):
create more than 30 genuinely competing occurrences for the rank-31 case; compare
exact search with independently enumerated paths including actual join terms;
test a repeated 14 ms interval, a real gap, a phrase longer than the span cap,
and inventory-order permutations. The same source run split into different legal
partitions must have equivalent cost and unchanged preserved PCM.

### R09. Plan content-safe joins and punctuation, not fixed post-selection cuts

**Context / why:** [Edit planning](../src/edit_plan.py) adds a fixed 14 ms overlap,
bounded only by the outgoing segment. Coupling-variant fields are unused. Pauses
are considered only between selected spans, so punctuation inside an intact
phrase can disappear. Selected costs do not describe the rendered join.

**Fix:**

1. Define bounded, source-legal entry/exit variants with protected required phone
   content. Initially retain whole-phone coverage; do not introduce unvalidated
   diphone/half-phone splitting to solve this repair.
2. Score/select the actual variants or rerank feasible paths after refinement.
   A final boundary change must revalidate coverage and update its cost. Keep a
   safe hard-splice/no-overlap option when no content-safe overlap exists.
3. Bound overlaps by both contributors and available safe handles; protect short
   transients, stop closures/releases, and already-overlapped neighboring content.
   Store chosen source bounds, curves, fades, and overlap in the plan.
4. Treat zero crossings as one observation, not proof of a good splice. Check
   channel-aware discontinuity, voicing, and transient risk without deleting
   required acoustic content.
5. Model punctuation/guide pauses at target boundaries before treating a phrase
   as indivisible. Preserve a suitable recorded internal pause; otherwise choose
   a legal split/alternative and explicitly schedule room tone or digital silence.
   No universal pause at every word boundary and no double-inserted source pause.
6. Plan the associated source-frame hold/continuation under R10, never arbitrary
   video after the selected word that could expose neighboring dialogue.

**Acceptance:** extend [edit-plan](../tests/test_edit_plan.py) and
[cut tests](../tests/test_cut_accuracy.py). Cover a long outgoing contribution
followed by a shorter-than-overlap phone, protected transients, anti-phase joins,
punctuation within a phrase, retained natural silence, and repeated 14 ms units.
Assert actual source coverage and overlap geometry, not only master length.

### R10. Schedule video from the cumulative audio clock without drift

**Context / why:** [Planning](../src/edit_plan.py) independently floors/caps each
source clip and forces at least one frame. [Rendering](../src/video_assembler.py)
concatenates those counts, ignoring the intended timeline positions.

Reproductions: 1,000 x 50 ms cuts at 30 fps produced 50 seconds of audio but only
1,000 frames (33.333 seconds) of video. A six-cut real FFmpeg render was 0.300
seconds audio versus 0.200 seconds video. An existing 77-segment output differed
by approximately 0.794 seconds.

**Fix:**

1. Finalize audio starts/ends, pauses, and overlaps first. Derive picture-switch
   boundaries from those absolute times using one documented overlap policy
   (for example, switch to incoming video at its scheduled audio start).
2. Quantize absolute boundaries, never each duration independently. For end-covering
   CFR, with `D` samples, rate `S`, and FPS `p/q`:
   `N = ceil(D * p / (S * q))`.
   Use integer arithmetic. For an interior boundary `b`,
   `Q(b) = ceil(b * p / (S * q))`; counts come from differences of ordered `Q`s.
3. Allow zero separately displayed frames for subframe contributions. Keep their
   audio and record the visual mapping. Remove every renderer/planner `max(1, ...)`
   that would turn these back into one frame each.
4. Do not cap the scheduled output duration by flooring source duration. Select
   legal source frames using R05's PTS mapping; when necessary, hold an already
   eligible frame rather than advance into neighboring footage. Record holds,
   including the final covering frame, explicitly.
5. Introduce a separate picture-span/frame-map representation if audio segments
   cannot express zero-frame spans, holds, and overlaps clearly. Both exporter
   and renderer must consume it; neither may invent a second schedule.
6. Re-encode video as required for accurate non-keyframe cuts, then verify emitted
   frame counts/PTS before concatenation/muxing. Skip zero-frame picture spans.
   Do not shorten audio, append arbitrary black frames, or use `-shortest` to hide
   disagreement.

**Acceptance:** extend [edit-plan](../tests/test_edit_plan.py) and
[renderer tests](../tests/test_video_assembler.py). At 24/25/30/30000:1001 fps,
assert exact planned frame counts and every interior quantization error below
one frame. For the 50 ms cuts-only fixtures at 48 kHz, with no added pauses or
overlaps, the 1,000-cut 30 fps case must have **1,500 frames and 2,400,000 samples**.
The six-cut 30 fps render must have **9 frames and 14,400 samples**. Also test
subframe contributions, pauses, overlaps, and mixed-source rates. Render at least one
1,000-contribution low-resolution fixture; count actual frames and inspect switch
positions, not just container duration. Re-encoding can be grouped without
changing the finalized frame map.

### R11. Make processing, channels, and publication faithful to a frozen plan

**Context / why:** [Rendering](../src/video_assembler.py) silently averages stereo
to mono, recalculates gains when a stored gain is zero, applies a global peak
clamp even without normalization, and can normalize cuts-only output.
[The CLI](../run_mixer.py) saves the plan before these decisions occur. Outputs
are written to final names before all requested artifacts validate.

**Fix:**

1. Represent project channel layout/sample format and all conversions explicitly.
   Preserve stereo when requested/available; no implicit averaging that cancels
   anti-phase content. A 16 kHz mono analysis track is not the output waveform.
2. Decide gains, fades, overlaps, selected audio variant, and normalization policy
   before freezing the plan. Zero dB is a legitimate decided value, not "auto."
   Calculate leveling over suitable consolidated speech runs, not each tiny phone;
   do not level two halves of an untouched continuation independently.
3. If master normalization needs measurement, render an unpublished analysis
   master, measure it, finalize processing parameters, then render/verify the
   deliverable. Store measured normalization parameters, algorithm/version, and
   actual processing mode. The renderer must not silently revise decisions.
4. Enforce cuts-only as the reference: no automatic overlap, fade, leveling,
   normalization, or peak clamp. Required resampling/format conversion stays
   explicit. Preserve source-derived mode's default normalization behavior.
   Resolve defaults before planning: cuts-only defaults normalization off;
   explicitly requesting normalization with cuts-only should produce an
   actionable conflict error, not silently ignore one option.
5. Preserve combined/isolated/recombined/both/all behavior. Record output-variant
   source/stem hashes and processing separately while sharing the same timing
   plan. Recombined gain policy must say which changes affect vocals/background.
6. Test true peak and loudness separately. Do not conceal an unattainable loudness
   target by silently adding dynamics processing, changing duration, or declaring
   a peak-limited result compliant. R16 defines truthful acceptance reporting.
7. Make finalized data immutable in practice (immutable nested values or a
   validated frozen snapshot), not only in its docstring. Record a stable plan
   identity/hash and make the renderer/exporter consume that exact snapshot.
8. Stage WAV/MP4/JSON/XML/reports under unique temporary names. Validate every
   requested artifact, then publish with a completion manifest containing hashes.
   Multiple renames alone are not an atomic multi-file transaction. Preserve
   prior completed output on failure; retain useful errors and identify partial
   new artifacts as incomplete. Capture FFmpeg stderr rather than discarding it.

**Acceptance:** extend [renderer](../tests/test_video_assembler.py),
[loudness](../tests/test_loudness.py), and
[pipeline tests](../tests/integration/test_pipeline.py). Compare actual sample
arrays in cuts-only mode at a common format/rate; test stereo anti-phase material,
0 dB gain, high peaks, partition invariance, all audio variants, and injected
failures after WAV/video/XML creation. No failed run may appear as a completed
new output, and rendering may not mutate the persisted plan.

### R12. Export complete, relinkable XML from the same timeline

**Context / why:** [FCP XML export](../src/fcp_xml_generator.py) constructs `<file>`
nodes with paths but never attaches them. A real existing export had 12 file
references and zero `<pathurl>` definitions. It also infers master audio from the
XML basename and confuses project/native source rates.

**Fix:**

1. Attach a complete media definition on first use of each file ID, with valid
   name/path URL, media/channel metadata, and native rate. Subsequent references
   may use the same ID. Apply this to both source video and master audio.
2. Pass the actual requested output variant's master-WAV path explicitly from
   orchestration. XML in another directory or with another basename must not
   silently omit audio.
3. Use R10's canonical picture schedule, source PTS/rate mapping, and holds.
   Keep project FPS and source FPS separate. For constructs FCP7 XML cannot
   faithfully represent, use a declared mapped proxy or fail that export mode;
   never silently approximate while claiming equivalence.
4. Make the authoritative master audio play once with correct channel routing.
   Do not accidentally double a mono master or leave source audio unmuted.
   Preserve audio-only subframe edits through the master and provenance plan.
5. Validate definitions/references, escaping, native/project rates, clip timing,
   source bounds, WAV identity, and channel mapping before publication.

**Acceptance:** strengthen [XML tests](../tests/test_fcp_xml_generator.py) and
[validator tests](../tests/test_validate_mix.py). Use real tiny WAV/video fixtures,
custom XML/WAV directories and basenames, repeated source IDs, mixed FPS, stereo,
holds, and subframe cuts. Assert every file reference resolves to one appropriate
definition and accessible matching media. Actual target-NLE import/playback is
a separate human gate; do not claim it from XML parsing alone.

### R13. Identify separator stems unambiguously and validate their provenance

**Context / why:** [Vocal discovery](../src/vocal_isolation.py) matches the
substring `vocal` and chooses the newest file. That includes `no_vocals.wav`.
A synthetic Demucs-output fixture reproduced the background being copied as
vocals. Stem reuse ignores recorded source/model hashes; force rebuild can also
leave an old instrumental stem.

**Fix:**

1. Use backend-specific output contracts: Demucs `vocals.wav` and `no_vocals.wav`
   have distinct roles. For audio-separator, use its output mapping or exact
   role-aware naming. Exclude instrumental/no-vocal outputs from vocal discovery.
2. Run separation in an isolated per-attempt directory. Accept only this
   invocation's verified results; do not select stale files by recency.
3. Validate stem content identity, actual model/tool revision and options,
   selected input stream, sample rate/channels, duration, and time mapping.
   Cache hits require matching metadata, not simply nonempty `vocals.wav`.
4. Publish a matching stem set and provenance together. A forced rebuild replaces
   all affected roles; do not pair new vocals with old background.
5. Distinguish raw separator stems from any normalized alignment derivative.
   Store processing/provenance so recombination and alignment know what they use.
   Fail when a requested role is missing/ambiguous; no filename-prefix fallback.

**Acceptance:** extend [isolation tests](../tests/test_vocal_isolation.py) using
mock separator processes and different known waveforms for vocals/background.
Make `no_vocals.wav` newer and verify it is never selected as speech. Test stale
extra files, changed source/model/options, force rebuild, missing/ambiguous roles,
duration mismatch, failed normalization, and failed separator fallback. Neural
downloads/inference are not required for these deterministic tests.

### R14. Make the human guide influence selection without supplying output audio

**Context / why:** [Guide construction](../src/prosody_guide.py) extracts features,
but [selection](../src/span_selector.py) ignores them. The CLI passes target text
as the guide transcript and then compares it with itself. There is no aligned
per-word/phone duration guidance.

**Fix:**

1. Add an explicit guide transcript input, such as `--guide-transcript <text-file>`,
   with documented UTF-8 semantics. Accept a declared sidecar or independently
   obtained transcript; do not assign `args.text` and call that validation.
2. Compare normalized guide tokens with the target, align guide audio, and validate
   bounds/coverage/alignment quality. Forced alignment alone is not proof that
   the speaker said the supplied text; use independent unprimed ASR as a check
   when configured, and retain uncertainty/manual-verification status explicitly.
   Empty, unrelated, or unalignable guides cannot silently pass.
3. Extract target-aligned relative durations, pauses, emphasis, voicing, and F0.
   Normalize pitch/rate to source-speaker statistics and make these soft target
   preferences, not waveform pitch/time modification.
4. Wire those preferences into R07's costs and R09's pause handling. Report what
   guidance was applied and where the corpus could not satisfy it.
5. Hash guide content/transcript/alignment settings into applicable caches and
   the plan. Exclude guide assets from all rendering source roles, including
   cached PCM/stem/proxy aliases, not just the visible source-video path.

**Acceptance:** extend [guide tests](../tests/test_prosody_guide.py) and
[pipeline tests](../tests/integration/test_pipeline.py). Controlled alternative
readings must select different eligible source candidates when only guide
preferences change. Test transcript/audio mismatch, failed alignment, no voiced
F0, guide cache invalidation, and attempted guide leakage through every audio
asset field. Basic-plus-guide must work without WavLM. A supplied real human
guide is a separate integration/listening fixture, not a requirement to generate
synthetic speech.

### R15. Finish contextual WavLM integration and test its actual contribution

**Context / why:** [The backend](../src/wavlm_backend.py) performs real inference
and its score participates in edited joins. However, [the selector](../src/span_selector.py)
extracts isolated short boundary snippets, uses 4,800 native samples regardless
of rate, silently omits unavailable embeddings, and caches without model/layer
identity. [Preparation](../tools/prepare_wavlm.py) can fall back to mutable `main`.

**Fix:**

1. Keep WavLM optional and analysis-only. Retain lexical/phonetic eligibility and
   the complete basic scorer; embeddings supplement rather than replace them.
2. Extract contextual source tracks in bounded overlapping chunks/batches.
   Specify context, padding, retained center regions, model stride/receptive-field
   mapping, and original-sample timestamps. Express windows in time, not a
   constant number of samples at unknown native rates.
3. Cache tracks/references with audio hash, immutable checkpoint revision, layer,
   configuration, chunking, normalization, and feature-schema identity. Reuse
   across queries; never compute an encoder pass for every candidate pair.
4. Validate dimensions, layer bounds, finite features, sample rate, and interval
   mapping. Requesting WavLM with missing assets/config/extraction must fail
   contextually; no silent basic fallback or fake embeddings.
5. Require a pinned revision. If preparation cannot resolve it, fail with an
   actionable offline/preparation message rather than writing `main`. Keep model
   acquisition separate from ordinary inference.
6. Add basic versus basic+WavLM configurations to R16's evaluation. Keep the same
   corpus, eligibility, renderer, and other weights for the relevant ablation.
   Record recall, runtime, memory, and listening outcomes; do not promote the
   backend on an embedding-similarity test alone.

**Acceptance:** strengthen [backend tests](../tests/test_wavlm_backend.py) and
[real-model tests](../tests/integration/test_wavlm_real.py). A deterministic
injected feature backend must change the selector's chosen path, and a real
cached-model integration must show measured embedding costs reaching that same
selection path. Manually editing `quality_cost` then calling `min()` is not such
a test. Check 44.1/48 kHz-equivalent windows, chunk edges, changed model/layer cache
keys, missing assets, and offline pinned-model reuse. Never claim a 20 ms encoder
feature is a sample-accurate splice locator.

### R16. Make validation and evaluation honest acceptance gates

**Context / why:** [Evaluation](../tools/evaluate.py) calls render success `PASS`
even when ASR fails, normally exits zero for failed cases, and crashes formatting
missing CER after ASR errors. [ASR verification](../tools/listen_verify.py)
accepted omission of "not" from a five-word target at WER 0.20.
[Validation](../tools/validate_mix.py) checks little source/XML provenance, allows
two-frame timing errors, and labels peak-limited below-target loudness as verified.

**Fix:**

1. Use typed results with separate render, structural, acoustic/ASR, and overall
   acceptance status. Preserve errors, skips, missing measurements, and reasons.
   Do not encode an unavailable ASR model as WER 1.0 or a passed/skipped check.
2. Persist a machine-readable report even on failure. Guard optional metrics:
   no formatting/multiplication of `None`. Aggregate available measurements
   separately from failed/unavailable counts; an empty suite is not a pass.
3. CLI exits: 0 only when the requested validation profile passes, 1 for failed
   acceptance criteria, and 2 for operational/configuration failure. Distinguish
   an explicitly requested structural-only/skip-ASR run from full acceptance in
   the report. A skipped check cannot satisfy a full-release gate.
4. At strict mixer acceptance gates require every target token occurrence and
   disallow unintended insertions. WER/CER tolerance can still support explicitly
   labeled diagnostic recognition checks; it cannot excuse a missing negation.
   Keep configurable documented pronunciation/homophone equivalences, not broad
   fuzzy acceptance or a set-membership test that loses order/repetitions.
5. Preserve the later vocabulary-priming feature as an assisted ASR mode, but
   label it. Use an unprimed pass for the independent recognition gate, optionally
   report both, and flag disagreement. Passing the full expected sentence into
   decoding is not independent evidence of exact speech coverage.
6. ASR is supplemental: retain deterministic legal target/source coverage checks.
   Recognition success cannot establish naturalness, boundary fidelity, or
   absence of every click. Recognition failure needs diagnosis/listening, not
   rewriting the expected text to force a pass.
7. Validate every source interval/hash/role and exact PCM sample count. Verify
   actual frame count and presentation schedule against R10; account for codec
   priming/padding from actual timing metadata. Remove arbitrary two-frame
   forgiveness rather than relabeling it as a one-frame bound.
8. Enforce the selected loudness profile truthfully: -16 +/- 1 LUFS and <= -1 dBTP
   when normalization is requested. An unmeasurable short clip or peak-limited
   under-target result is explicit unmet/inconclusive evidence, not compliance.
   Non-normalized cuts-only output must not be measured against a profile it did
   not request. State measurement precision/tolerance rather than hidden waivers.
9. Validate R12's XML media definitions, master identity, channels, and schedule.
   Timebase-only checks must not announce that the timeline is verified.
10. Drive evaluation configurations from the suite (legacy/basic/basic+WavLM,
    guide when supplied). Compare legacy/new selection under the same renderer.
    Persist subprocess stderr and configuration identities. Exclude failed/stale
    renders from blinded assets; use a new run directory and saved randomization.
    Do not overwrite existing successful user outputs during evaluation.

**Acceptance:** extend [ASR](../tests/test_listen_verify.py),
[evaluation](../tests/test_evaluation.py), and
[validator tests](../tests/test_validate_mix.py). Mock ASR for deterministic cases:
all failures, unavailable model, missing CER, empty suite, a missing negation,
repeated/misordered words, and insertions. Assert real CLI exit codes and JSON
schema. Include stale prior artifacts, malformed/missing XML definitions,
source-hash mismatch, wrong frame count, and peak-limited loudness.

### R17. Repair tests/tooling so passing results mean the requirements hold

**Context / why:** Several current tests exercise helpers rather than their
integration. The rank-31 fixture lacks 31 competitors; exact-search comparison
omits the acoustic term; WavLM ranking manually edits costs; XML only checks
structure; real-corpus smoke mostly checks file existence. ASR model tests are
unmarked and run in the supposedly offline suite. Current lint/type checks fail.

**Fix:**

1. Add the concrete regression cases in R01-R16 to the relevant existing files.
   Test actual output shapes, samples, graph paths, identities, and CLI exits.
   A mock is appropriate for failure injection, not as proof of real integration.
2. Mark every weight-loading test `model`; mark every user-corpus-dependent test
   `real_corpus`, including tests outside the integration directory. Prevent
   inference/downloads during collection. Keep separate mocked offline ASR tests.
3. Use `tmp_path`/unique per-run artifact directories. Change the real-corpus
   smoke test so it cannot replace `output\smoke_basic.*` or other user mixes.
   Reuse validated immutable source caches but isolate generated outputs.
4. Missing prerequisites may be reported during developer discovery, but explicitly
   requested model/corpus release checks must fail or be recorded blocked, never
   be silently skipped while declaring all milestones complete.
5. Fix current Ruff issues without unrelated rewrites. Fix real mypy type errors:
   array padding shape types in the renderer, ASR pipeline return handling, and
   evaluation result/optional-metric types. Do not suppress whole modules or use
   broad casts to hide the incorrect assumptions.
6. Run focused tests per change, then the full offline suite and explicitly
   provisioned model/corpus checks. Keep a current evidence report with revision,
   configs, command exits, pass/fail/skip counts, and remaining human gates.

**Acceptance:** all applicable checks in section 3 pass. The original regressions
must fail on the audited implementation and pass after repair, except additional
coverage checks explicitly introduced for statically identified gaps. No new
blanket `xfail`, disabled test, widened tolerance, or fabricated fixture result.

### R18. Route every CLI surface consistently and correct public claims

**Context / why:** [The CLI](../run_mixer.py) finalizes/render/exports the legacy
branch differently, ignores some parsed options there, and does not use
`auto --output-name` when choosing output names. The
[README](../README.md) still promises sub-millisecond MFA alignment and describes
verified speaker filtering as if enforced. Other status/docs overstate completion.

**Fix:**

1. Both selectors must return selection information adapted into one finalization
   path. `mix` and `auto` must share speaker validation, feature/mode handling,
   plan persistence, renderer, XML export, output variants, and failure behavior.
   Preserve public legacy return compatibility through an explicit adapter.
2. Honor `auto --output-name` when no explicit render destination overrides it.
   Test custom plan/XML directories and render-directory versus render-file input.
3. No ignored feature/guide/mode/isolation option. If a selector/subcommand cannot
   support an option meaningfully, reject it with an actionable explanation and
   document that limitation rather than pretending it ran.
4. Update the README and directly related plan/runbook/project status after the
   implementation is verified. Remove alignment-precision/naturalness guarantees,
   stale folder references, and unconditional completion claims. Document schema/
   cache migration, speaker assertions, cuts-only defaults, separation provenance,
   normalization limitations, XML master audio, and validation profiles.
5. Generate completion status from captured current evidence. Separate automated
   code verification, objective media checks, human listening, and target-NLE
   approval. Leave WavLM optional and keep the existing selector default until
   the documented release decision is supported.

**Acceptance:** extend [CLI](../tests/test_cli_compat.py) and
[pipeline tests](../tests/integration/test_pipeline.py). Exercise both selectors,
`mix`/`auto`, all output variants, explicit speaker failure, custom filenames,
cuts-only conflicts, and unsupported-option diagnostics. The persisted plan used
by renderer and XML must be the same; docs/help examples must match actual behavior.

## 3. Exact execution and verification commands

These commands are for the implementing agent. This documentation change does
not run them or install anything. The existing repository-local environment and
cached models should be reused. Install/restore only after an actual missing
dependency failure or a deliberate manifest change, using the existing lockfiles.
Keep source media/transcripts local.

### PowerShell preamble

Run this in each fresh shell invocation, then the applicable block below.
`Run` deliberately turns native nonzero exits into failures. A regression's
expected pre-fix failure is evidence to record, not a reason to weaken its test.

```powershell
Set-Location 'D:\Code\sentence_mixer'
$ErrorActionPreference = 'Stop'
$Repo = (Get-Location).Path
$Py = Join-Path $Repo '.venv\Scripts\python.exe'
$env:MAMBA_ROOT_PREFIX = Join-Path $Repo '.mamba'
$env:MFA_ROOT_DIR = Join-Path $Repo '.build\mfa'
$env:HF_HOME = Join-Path $Repo '.build\huggingface'
$env:NLTK_DATA = Join-Path $Repo '.build\nltk'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
if (-not (Test-Path -LiteralPath $Py)) {
    throw "Missing project Python: $Py. Restore the documented environment first."
}
function Run {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Exit ${LASTEXITCODE}: $Program $($Arguments -join ' ')"
    }
}
```

### Inspect and record the starting state

```powershell
Run 'git' @('--no-pager', 'status', '--short')
Run 'git' @('--no-pager', 'log', '-3', '--oneline')
Run 'git' @('--no-pager', 'diff', '--stat')
Get-Command ffmpeg.exe, ffprobe.exe -ErrorAction Stop
Run $Py @('-m', 'pip', 'check')
Run $Py @('.\run_mixer.py', 'mix', '--help')
Run $Py @('.\run_mixer.py', 'auto', '--help')
```

### Focused groups during repairs

Run the group covering the current changes, not all groups after every edit.
R17 must first make marker exclusions trustworthy. New tests should normally go
in these existing files; if adding files, include them in the applicable group.

```powershell
# Input, identity, source mapping, caches, separation.
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'not model and not real_corpus',
    'tests\test_contracts.py', 'tests\test_vtt_preprocessor.py',
    'tests\test_inventory_builder.py', 'tests\test_media_time.py',
    'tests\test_cache.py', 'tests\test_vocal_isolation.py')
```

```powershell
# Legal coverage, acoustic ranking, continuations, optional analysis plumbing.
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'not model and not real_corpus',
    'tests\test_target_graph.py', 'tests\test_span_selector.py',
    'tests\test_features.py', 'tests\test_prosody_guide.py',
    'tests\test_wavlm_backend.py')
```

```powershell
# Final scheduling, real synthetic media, XML, DSP, and acceptance/CLI integration.
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'not model and not real_corpus',
    'tests\test_edit_plan.py', 'tests\test_video_assembler.py',
    'tests\test_cut_accuracy.py', 'tests\test_fcp_xml_generator.py',
    'tests\test_loudness.py', 'tests\test_listen_verify.py',
    'tests\test_validate_mix.py', 'tests\test_evaluation.py',
    'tests\test_cli_compat.py', 'tests\integration\test_pipeline.py')
```

### Final offline and static checks

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'not model and not real_corpus')
Run $Py @('-m', 'ruff', 'check', 'src', 'tests', 'tools', 'run_mixer.py')
Run $Py @('-m', 'mypy', 'src', 'tools', 'run_mixer.py')
Run $Py @('-m', 'pip', 'check')
Run 'git' @('--no-pager', 'diff', '--check')
```

### Explicit model and corpus checks

Only run after repairing the tests' output isolation and provisioning their
documented prerequisites. WavLM weights were cached during the audit; do not
assume ASR weights, separators, or all corpus alignments are also available.
These commands remain offline; missing assets are a clear prerequisite failure.
Use the original runbook's explicit preparation steps for required public
downloads, removing offline flags only for that separate preparation invocation.

```powershell
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'model and not real_corpus')
Run $Py @('-m', 'pytest', '-q', '--tb=short', '-m', 'real_corpus')
```

### Fresh real-media smoke and evaluation

After all relevant repairs, use a unique destination. The checked-in
[smoke manifest](../evaluation/smoke_sources.json) declares `obi-wan`; do not
substitute another speaker and rely on the old override bug. Confirm its media
and annotations are valid locally. This example uses the existing real-media
target as an integration case, not as proof of general naturalness.

```powershell
$RunDir = Join-Path $Repo ('.build\remediation-runs\' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $RunDir | Out-Null
$Stem = Join-Path $RunDir 'basic'
Run $Py @('.\run_mixer.py', 'mix',
    '--manifest', '.\evaluation\smoke_sources.json', '--speaker', 'obi-wan',
    '--text', 'Anakin, I have the MOXY!',
    '--selector', 'span', '--features', 'basic', '--mode', 'source-derived',
    '--fps', '30000/1001', '--render', ($Stem + '.mp4'),
    '--plan', ($Stem + '.plan.json'), '--xml', ($Stem + '.xml'))
Run $Py @('.\tools\validate_mix.py', '--stem', $Stem)
Run $Py @('.\tools\listen_verify.py', '--input', ($Stem + '.wav'),
    '--expected', 'Anakin, I have the MOXY!', '--require-all-words', '--json')
Run $Py @('.\tools\evaluate.py', '--suite', '.\evaluation\suite.json',
    '--output-dir', (Join-Path $RunDir 'evaluation'))
```

The strict ASR gate above assumes R16's independent/strict behavior has been
implemented and the required model is available. Also run the repaired evaluation
suite's basic+WavLM configuration and any supplied guide cases; record missing
human-guide material as a pending real-guide gate, not as an invented pass.
Verify separated variants with their actual prerequisites as well as their
deterministic offline tests. Do not regenerate the user's existing mixes in place.

## 4. Completion contract

The final agent report must enumerate R01-R18 with status, changed files, actual
tests/commands and exits, and any limitations. Include measured long-timeline
sample/frame counts, source/speaker/coverage evidence, XML reference checks, and
strict evaluation results. Save machine-readable evidence under a new local
report/run directory; do not overwrite the old report as if it verified new code.

The implementation is not finished merely because helper tests pass or output
files exist. Every advertised option must affect the intended path or fail
explicitly. No silent source substitutions, widened tolerances, ignored requested
features, invented confidence, or success-shaped failures.

Human transcription/naturalness comparisons, review of source-boundary fidelity,
and import/playback in the actual target NLE remain separate release gates. Prepare
blinded materials and a usable export; never fabricate human approval. Do not
promote a default solely from automated WER or one favorite example.

### Copy-ready agent prompt

> Implement every repair R01-R18 in plans\audit-remediation-plan.md against the
> current repository. Use plans\design.md and plans\milestones.md for invariants,
> while preserving the later ASR, vocal-isolation, and loudness requirements in
> plans\ORIGINAL_REQUEST.md. First inspect the worktree and reproduce the defects
> with regression tests; then repair root causes across all callers, not just
> helper functions. Follow the dependency order and exact Windows PowerShell
> verification commands in the repair plan. Complete the optional WavLM and
> human-guide code paths as well as the basic path, without synthesizing output
> speech. Preserve unrelated changes and existing media/outputs; do not commit
> unless asked. Run focused checks during implementation and all applicable final
> checks. Return an R01-R18 evidence table, current commands/results, migration
> instructions, and honest pending human/model/corpus gates. Do not call a skipped,
> disconnected, untested, or merely scaffolded feature complete.
