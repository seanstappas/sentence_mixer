# Implementation Plan: Source-Preserving Sentence Mixing

Updated: 2026-09-19.

**This is a staged implementation specification, not completed functionality.**
It replaces the previous wholesale replacement-code recipe. Implement and verify
each milestone before proceeding; do not paste speculative replacement modules
over working interfaces.

Read the [design and rationale](design.md) and [research notes](../research/README.md)
alongside this plan. The intended result is MFA-based, variable-span unit
selection with a canonical audio/video timeline. WavLM remains an optional,
analysis-only enhancement after the baseline has been measured.

## 1. Scope and non-negotiable requirements

### Initial supported scope

- English target text and compatible pronunciation/acoustic resources.
- One explicitly identified speaker per output, drawn from one or more videos.
- Whole phrases and words when suitable, with shorter phonetic spans as fallback.
- Original recorded speech and original video frames; no speech generation,
  vocoder, neural waveform reconstruction, or synthesized mouth movements.
- Source-derived rendering with disclosed gain, fades, overlaps, and format
  conversion; a cuts-only reference mode for comparisons.
- Rendered MP4, a lossless PCM master WAV, a machine-readable edit/provenance plan,
  a quality report, and optional FCP7 XML.

Manual speaker/interval annotation is sufficient for the first release.
Automatic diarization, active-speaker detection, multilingual support, and
pitch/time modification are later features, not silently assumed capabilities.
Neural separation/denoising must not replace the original output waveform.

### Invariants

1. Cover every required target word and one complete legal pronunciation path.
   Missing content is an explicit failure with a coverage report.
2. Every speech contribution maps to a legal source sample interval and speaker.
   Silence/room tone, overlaps, gains, and frame holds have explicit records.
3. Preserve source-contiguous runs when their complete path wins the objective;
   never discard them solely because their phones are not near 90 ms.
4. A word boundary does not imply silence. No universal 45 ms word padding.
5. Alignment is an estimate. No sub-millisecond accuracy or artifact-free claims.
6. Missing acoustic observations stay missing; do not substitute 120 Hz, zero
   feature vectors, or other plausible-looking success values.
7. Selection, preview, rendering, and export use one finalized edit plan.
8. A failed slice, invalid source mapping, or unsuccessful master export fails the
   run. Never publish a shortened sentence as a completed result.
9. Keep neural analysis optional. Requesting an unavailable WavLM profile raises
   an actionable error; it does not silently run a different scorer.

## 2. Current surfaces and compatibility

Preserve these existing entry points while refactoring their internals:

| Surface | Required treatment |
| --- | --- |
| [run_mixer.py](../run_mixer.py) | Keep `mix`, `auto`, `preprocess`, `align`, and `build-inventory`, including their existing arguments. Make `auto` share the same orchestration path as `mix`. |
| [vtt_preprocessor.py](../src/vtt_preprocessor.py) | Repair ordinary/rolling subtitle handling and source-time mapping; retain standalone invocation. |
| [inventory_builder.py](../src/inventory_builder.py) | Produce versioned source/event/word records and feature references; retain the standalone inventory command. |
| [unit_selector.py](../src/unit_selector.py) | Keep the existing word-first heuristic as `legacy`; add the new span selector behind an explicit option before changing defaults. |
| [video_assembler.py](../src/video_assembler.py) | Route rendering through a validated canonical plan and PCM master, with compatibility adapters for existing callers. |
| [fcp_xml_generator.py](../src/fcp_xml_generator.py) | Consume that same plan; use authoritative master audio when source-only XML cannot represent the edit faithfully. |
| [inspect_alignment.py](../src/inspect_alignment.py) | Show source sample offsets, quality flags, context, proposed coupling regions, and original/refined boundaries. Update defaults without losing existing options. |

Introduce small shared modules for contracts, media/time mapping, feature
extraction, and edit planning where appropriate. Do not duplicate timing or
pronunciation logic in the CLI, renderer, and exporter.

Preserve the legacy `synthesize_sentence` return shape through an adapter for
existing callers. New typed planning APIs should not overload the meaning of an
old unstructured unit dictionary. The CLI must finalize once and pass the same
plan to rendering and XML, rather than invoking two independent finalizers.

Recognize legacy inventories explicitly:

- They may remain usable with the legacy selector and clear provenance/quality
  warnings, subject to media and content validation.
- The new selector requires the new schema or an explicit rebuild/migration.
- Do not infer unavailable confidence, speaker identity, or acoustic measurements
  from absent fields.
- Explain the required rebuild when a direct `--inventory` input is incompatible;
  do not quietly align unrelated media or fall back to a default inventory.

Stricter speaker/content validation is an intentional behavior change. Introduce
it with the opt-in span selector, document migration, and apply the default switch
only after the release gate. Preserve original user media and dictionaries.

## 3. Environment, workspace, and cache contract

### 3.1. Dependencies and executable discovery

Create a dependency manifest and a reproducible tested dependency lock/constraints
set during implementation. Separate baseline, development/test, and optional
WavLM dependencies. The baseline uses the existing Python ecosystem, FFmpeg,
Praatio, numerical/audio analysis libraries, and a validated pronunciation
backend; it does not require Torch/Transformers merely to concatenate clips.

Keep MFA in its supported isolated environment. Discover tools using explicit
configuration, the active environment/PATH, or a configured micromamba launcher.
Select the executable name for the platform; do not hardcode a Windows `.exe`
for macOS/Linux or a particular user's home directory.

Record and validate:

- FFmpeg/ffprobe and decoder configuration.
- MFA software version and acoustic-model name/version/checksum.
- Dictionary and G2P model identity, phone inventory, and custom-entry hashes.
- Feature library/settings versions and optional encoder checkpoint revision.

Do not assume a model named `english_us_arpa` has the same version as the MFA
software. Verify actual installed metadata and supported CLI flags. The 2026 MFA
benchmark is evidence for evaluating specific models, not a command to blindly
download a moving latest version.

Use subprocess argument lists, preserve useful stderr/log paths, and classify
failures. Retry only an identified retryable condition; do not treat every
alignment failure as a multiprocessing problem.

### 3.2. Workspace

```text
.build\
  sources\<source_content_id>\
    media.json
    original.wav
    analysis.wav
  alignment\<alignment_fingerprint>\
    corpus\
    aligned\
    validation.json
  inventory\<inventory_fingerprint>\
    units_index.json
  features\<feature_fingerprint>\
    feature_index.json
    feature_arrays.npz
  runs\<run_id>\
    plan_draft.json
    logs\
    render\
input\
dictionaries\
output\
```

Names above define responsibilities, not a requirement to load an entire corpus
or feature array into RAM. A compact versioned JSON index plus binary arrays is a
reasonable first implementation; optimize storage only after profiling.

Extend `.gitignore` rather than replacing it: preserve existing local exclusions,
ignore project build artifacts and generated deliverables, and leave source
assets under the user's control. Model caches/environments outside `.build` are
separately configured and are not cleanup targets.

Temporary project files belong in a unique run directory under `.build`, not an
untracked system-temp rendering tree. Preserve failure logs. Cleanup must resolve
and validate owned paths and never remove input, dictionaries, or broad user
directories.

### 3.3. Cache dependency graph

| Stage | Fingerprint inputs |
| --- | --- |
| Decoded/analysis audio | Media content identity, selected stream, source timing metadata, decoder/resampler versions and settings |
| Prepared corpus/alignment | Audio identity, transcript content, subtitle/normalization policy, utterance/speaker intervals, dictionary/G2P/acoustic model identities, MFA options |
| Basic features | PCM/analysis identity, feature configuration/library version, speaker/session grouping |
| Inventory | Alignment, shared lexicon, feature references, source mappings, schema version |
| WavLM features | Analysis audio, checkpoint revision, layer selection, chunking/context, normalization, encoder implementation |
| Edit plan | Inventory/feature identities, target/pronunciation graph, guide identity if any, selector and rendering policy versions |

Use content hashes for correctness. File stat information may accelerate
unchanged-file checks but must not make a source-path hash the validity key.
Changing only the optional encoder must not force unrelated MFA realignment.

Validate before publishing, write atomically, and handle concurrent builders with
stage/run isolation or locking. A force-rebuild uses a fresh stage directory so
obsolete chunks/TextGrids cannot leak into the new inventory.

## 4. Shared data contracts

Define proper typed records and serialized-schema validation before building the
selector. Use separate inventory, feature, and edit-plan schema versions.
Intervals are half-open: `[start_sample, end_sample)`. Seconds shown to users are
derived values, not competing timing authorities.

| Record | Required information |
| --- | --- |
| `SourceRecord` | Content ID, source path, selected audio/video streams, native sample rate/channels, decoded-audio time origin, rational stream time bases, video PTS/frame index, codec-delay/skip metadata, recording/session identity, cached PCM paths |
| `UtteranceRecord` | Source sample range, transcript/token mapping, speaker ID and verification method, chunk/padding offsets, alignment resources, quality/visibility flags |
| `AlignedEvent` | Stable ID, source/utterance/speaker IDs, kind (`phone`, `silence`, `noise`), source sample bounds, phone/stress where applicable, word occurrence, predecessor/successor IDs, available alignment measurements and their meaning |
| `WordOccurrence` | Normalized word, original text mapping, ordered realized phone IDs, pronunciation ID, source bounds and linguistic context |
| `FeatureRecord` | Source-time window, configuration ID, spectral/energy/voicing/F0 values, validity masks, measurement confidence; optional encoder-vector references |
| `TargetGraph` | Original/normalized text, token mapping, punctuation/phrase hints, legal pronunciation branches and priors, phone/stress/context labels, optional pause/guide constraints |
| `SpanCandidate` | Target start/end nodes and covered edges, one source range/event sequence, pronunciation compatibility, feature references, eligible entry/exit coupling variants |
| `EditPlan` | Frozen sources/fingerprints, selected target path, segments/transitions, output sample rate/positions, rational video rate/frame mappings, all processing, quality findings, expected output lengths |

Store each event once. Exact/base-phone and word indexes contain IDs, not duplicated
event dictionaries. Counts must count entities, not aliases in multiple indexes.

### Boundary uncertainty and confidence

A TextGrid does not automatically supply confidence. Store available measurements
with their definitions; otherwise mark them unavailable. Heuristic duration flags,
alignment likelihoods, model disagreement, and human-reviewed bounds are different
types of evidence, not interchangeable probabilities.

Represent an estimated boundary separately from any uncertainty interval and any
approved coupling region. An uncertainty interval is not permission to trim away
a consonant. Do not present interpolated or highly precise floating-point
timestamps as equally precise acoustic ground truth.

### Edit-plan timing fields

Each rendered contribution includes:

- Source ID, source sample interval, and any permitted contextual handles.
- Output start/end samples at the project rate, with an explicit resampling map.
- Selected target coverage and entry/exit coupling variant.
- Gain/fade/overlap parameters, including curve definitions and durations.
- Associated original video frames/PTS and any declared hold or proxy mapping.

Each transition identifies its type: preserved continuation, hard splice, faded
splice, overlap, or explicit pause. During overlap, record both contributors and
which source supplies the visible video. Preserve this distinction in provenance.

For the selected speech intervals and gap events, validate:

```text
output_samples =
    sum(resampled_segment_samples)
    - sum(planned_overlap_samples)
    + sum(explicit_gap_samples)
```

Do not also count a gap as a speech segment or subtract the same overlap twice.
Source-contiguous subdivisions must coalesce without changing the signal or
duration. Input/output sample rates may differ; use rational mappings and
cumulative scheduling, not independent per-phone duration rounding.

## 5. Milestone M0: characterization and test fixtures

Before changing production behavior:

1. Capture CLI help and exercise each existing command with small fixtures.
2. Save representative legacy word-first cut plans as comparison artifacts.
3. Add deterministic synthetic PCM/video, subtitle, TextGrid, and inventory
   fixtures. Use tiny media for CI; do not commit large source videos.
4. Add a focused test runner and document the targeted commands in the project
   dependency configuration. Do not require MFA downloads or GPU inference for
   ordinary unit tests.
5. Mark current incorrect behavior as characterized defects to fix, not golden
   behavior that the new implementation must preserve.

Mandatory regression fixtures from the design review:

| Fixture | Required new behavior |
| --- | --- |
| Ordinary two-line cue: `Hello there` / `General Kenobi` | Preserve both lines in order. |
| Rolling captions plus a genuinely repeated word | Remove display overlap without deleting repeated speech. |
| Good intact word whose phones rank 31st under the old 90 ms heuristic | Retain the complete candidate; when its configured score wins, produce one source run rather than three splices. |
| Target cross-word boundary `T` followed by `K` | Preserve that context; do not replace it with two `SIL` boundaries. |
| Source phones separated by silence | Keep the silence event and do not claim immediate phonetic adjacency. |
| A 14 ms source unit reused for two target occurrences | Treat reuse as two contributions with an edit, not a self-continuation merged into one occurrence. |
| Known word with an observed pronunciation differing from default G2P | Consider the legitimate recorded word/pronunciation rather than discarding it. |
| Missing feature, unvoiced F0, and feature-extraction exception | Distinguish all three; no invented measurements or hidden successful fallback. |
| A 100 ms clip and an explicitly requested 45 ms pause | Default has no pause; the requested-pause case plans both audio and video coverage. |
| Failed required slice/encoder process | Fail the run and preserve diagnostics; publish no complete-looking truncated result. |

**Exit gate:** the fixtures are reproducible, current entry points are documented,
and new schemas can reject structurally invalid records with useful errors.

## 6. Milestone M1: ingestion, speaker eligibility, and alignment

### Transcript and pronunciation preparation

- Parse ordinary multiline and rolling WebVTT deliberately. Preserve punctuation,
  speaker labels, timed-word anchors, and original cue/token references.
- Deduplicate only demonstrated temporal/display overlap. Do not drop a new word
  just because its rolling display cue is short.
- Normalize numbers, contractions, apostrophes, and proper names consistently
  between source transcripts and targets. Retain mappings to the user's text.
- Use one lexicon service for MFA preparation and target generation. Handle MFA
  dictionary pronunciation/silence-probability columns correctly; numeric
  probability fields are not phones.
- Load reviewed custom pronunciations and all supported variants. Use validated
  G2P for unknown words and report unsupported/ambiguous entries. No silent
  character-by-character phonetic fallback.

### Media, utterances, and speaker identity

- Probe real audio/video stream starts, rates, channels, duration, and frame PTS.
  Decode original-rate PCM once and make a separate 16 kHz analysis view.
- Verify offsets through decoding/resampling, including nonzero stream origins
  and codec delay. Never apply analysis-frame indices directly to source audio.
- Use speech/utterance preparation with context padding; keep exact source
  offsets and ownership so overlapping chunks do not duplicate inventory events.
- Require user-verified single-speaker inputs or annotated speaker intervals for
  the new selector. Keep recording/session IDs separate from speaker IDs.
- Preserve cross-word and real-silence context. Reject cross-speaker spans,
  overlapping speakers, and invalid ranges according to documented policy.
- Carry visibility/noise/music/clipping flags. Unknown visibility is not evidence
  of a speaking face; a strict visible-speaker option excludes unknown cases.

Support manifest-relative paths and explicit speaker metadata. This is a proposed
schema extension; preserve existing object/string manifest forms through adapters.
For example, after this milestone:

```json
[
  {
    "video": "input\\speaker_a_session_1.mp4",
    "vtt": "input\\speaker_a_session_1.en.vtt",
    "speaker_id": "speaker-a",
    "recording_id": "session-1",
    "single_speaker_verified": true
  },
  {
    "video": "input\\interview.mp4",
    "vtt": "input\\interview.en.vtt",
    "recording_id": "session-2",
    "speaker_intervals": [
      {"speaker_id": "speaker-a", "start_sec": 12.5, "end_sec": 18.2}
    ]
  }
]
```

Only annotated intervals in the second source are eligible; the rest of the video
must not inherit `speaker-a`. Validate overlapping/conflicting labels and bounds.
CLI paths remain relative to the calling directory; paths in a manifest are
relative to that manifest. Document this change and test explicit absolute paths.

### Alignment validation

Run MFA with compatible, versioned dictionaries/models. Check transcript coverage,
OOV handling, tier presence, source bounds, interval ordering, anomalous durations,
and phone-to-word association before publishing an inventory.

Keep genuine silence/noise events in the ordered stream even if they are excluded
from speech candidate lookup. Preserve observed pronunciations and context across
preprocessing chunks when the source evidence establishes continuity.

Do not substitute another parser/empty tier after an arbitrary parse exception.
Supported format fallback must be explicit and validated. Structural corruption
fails; reviewable uncertainty is reported without claiming false confidence.

**Exit gate:** correct multiline/rolling behavior, verified time mappings,
source/speaker eligibility, complete phone/word associations, and correct cache
invalidation. Manually audit representative phone boundaries and report error
distributions, not just the number of successful alignments.

## 7. Milestone M2: continuous basic features and target prosody

Compute acoustic tracks once over continuous analysis audio with adequate context.
Store frame/sample mappings and distinguish padded/unreliable edge windows.
Baseline features include spectral descriptors/trajectories, local energy,
voicing and F0, plus recording and alignment quality flags.

Requirements:

- Boundary features use local entry/exit windows, not whole-unit RMS masquerading
  as a boundary measurement.
- F0 is defined only in sufficiently reliable voiced regions. Use a configurable
  search range appropriate to the tested voices, not a fixed male-only range.
- Preserve missing/invalid measurements. Extraction exceptions are surfaced;
  a genuinely unvoiced frame is not an exception and not a default pitch.
- Define missing-data scoring explicitly: unavailable required features reject a
  candidate; expected undefined values such as unvoiced F0 gate only the
  inapplicable term; unavailable optional observations carry a validity/quality
  policy and calibrated score normalization. An empty observation set must not
  become a perfect zero-cost join.
- Derive robust duration expectations by phone, stress, and available context.
  Document sparse-data backoff through suitable pooled statistics or an
  explicitly unavailable term, never a hidden 90 ms universal prior.
- Use speaker-relative/log-pitch comparisons and soft phrase-position/emphasis
  preferences. Do not flatten speech toward a median.
- Calibrate feature scales and distinguish acoustic similarity from expected
  phonetic change. Persist normalizers, weights, and validity policies.

Build the target pronunciation graph from the shared lexicon, preserving sentence
punctuation, word/phrase position, stress, and alternative realizations. Target
phones connect across words unless an actual pause is selected. Cross-word
context must remain correct for whichever pronunciation branch is chosen.

Text-only prosody is intentionally a soft model. Missing emphasis information
must not be represented as a highly confident flat contour. Keep a future guide
interface separate from lexical correctness and from output audio.

**Exit gate:** reliable voiced/unvoiced handling, short/high-pitch test cases,
correct context/variant graphs, feature cache reuse, and no fabricated features.
Measure indexing time/memory and verify that extraction is not decoding each
individual phone or word.

## 8. Milestone M3: variable-span retrieval and global selection

### Candidate retrieval

1. Retrieve complete lexical phrases/words and matching realized phone sequences.
2. Add shorter contiguous source ranges and compatible context-bearing fallbacks.
   Matching must obey declared pronunciation equivalences, not arbitrary
   acoustic/viseme substitutions.
3. Associate a bounded set of legal entry/exit coupling variants with each range.
   Preserve required phone content and enforce valid source bounds/duration.
4. Keep complete-run candidates and source-successor continuations protected
   during pruning. Rank additional candidates by multiple relevant features.
5. Deduplicate by source identity/range/variant, not merely labels. For scalable
   retrieval, index token/phone n-grams and source successors before considering
   approximate embedding search.

Scope constraint for M3 baseline: Strictly implement the Phrase $\rightarrow$ Word $\rightarrow$ Whole Phone retrieval hierarchy first. Diphone-style midpoint splitting is deferred as an optional post-baseline experiment and must not delay shipping the whole-phone search. Begin with whole-phone target edges. If adding half-phone/diphone candidates,
first extend the graph with ordered coverage-part anchors for suitable phones.
A diphone-style source span can cover the right part of one phone and the left
part of its successor, preserving their recorded transition. A full target path
must still cover every required part exactly once; whole-word/phrase candidates
traverse those same anchors. Store source split/coupling locations separately from
MFA boundaries, prohibit invalid splits of short transients, and test for missing
or doubled parts. Do not claim phone coverage merely because a trimmed clip
retains its original phone label. Evaluate this granularity extension against the
whole-phone baseline before enabling it.

Search caps are configuration, not undocumented constants. Emit candidate/beam
statistics and retry with a wider bounded search when pruning may have removed
coverage. Report search exhaustion separately from genuinely absent vocabulary.

### Scoring and recurrence

Use the objective in the design: target/prosodic suitability and source quality,
plus an introduced-edit penalty and acoustic/visual transition costs. Set the
introduced-edit/join term to zero only for proven original continuations.
No magic negative `-100` bonus and no unconditional longest-match rule.

Ensure coverage-weighted costs are consistent when an identical source run is
represented as one span or several natural continuations. Count pronunciation
priors once per word and avoid accidental preferences for particular candidate
partitionings or branch lengths.

Conceptually, for a candidate including its coupling variant:

```text
best[candidate] =
    target_and_quality_cost(candidate)
    + min over compatible predecessors:
        best[predecessor] + transition_cost(predecessor, candidate)
```

Compatibility includes target-graph connectivity and pronunciation choices,
source/content legality, and feasible boundary/video constraints. The previous
candidate/variant is part of the state; a single best number per word position
cannot capture pair-dependent join costs.

Compare exact dynamic programming with brute-force enumeration on tiny graphs.
If beam pruning is used, label it approximate and provide an unpruned small-case
mode. Stable source IDs/positions provide deterministic tie-breaking.

### Contiguity, consolidation, and alternatives

Use event successor relationships plus physical source coverage, not a floating
`abs(end - start) < 0.015` shortcut. Do not skip a silence/noise event while
claiming that adjacent retained phone labels were originally back-to-back.

Consolidate only actual source continuations, preserving all target coverage,
internal pauses, and original timing. Never fade internal boundaries of a merged
run. Repeated use of the same recording for repeated target speech remains
separate unless it truly references distinct consecutive source events.

Return a ranked path and enough alternatives/cost explanations for audition.
Do not quietly change target words, reduce lexical coverage, or select another
speaker to improve a score.

**Exit gate:** complete-path coverage and pronunciation consistency, protected-run
and repeated-unit regressions, exact-search agreement, deterministic results, and
usable diagnostics. Compare phrase/word retention and audible outcomes with the
legacy selector, not just its numeric objective.

## 9. Milestone M4: final planning, rendering, and export

### Coupling and DSP

Refine only within validated coupling regions and score the actual chosen
boundaries. If subsequent refinement changes cost/feasibility, rerank/revalidate
before freezing the plan. Never move cuts independently after XML export.

Use cached PCM for waveform analysis. Zero-crossing proximity may inform a choice
but does not prove phase agreement or click-free output, especially for stereo.
Treat voiced and unvoiced regions appropriately and protect closures/releases.

Make fade curves, gain ramps, overlap durations, and handles explicit. A short
fade/overlap is allowed only when it does not violate protected content or source
bounds. Do not assume FFmpeg's default fade curve is a raised cosine. Ensure the
gain envelopes avoid unintended level jumps/clipping and test correlated,
anti-phase, and transient-rich material.

No pause is inserted merely because a cut ends a word. For a requested phrase
pause, prefer suitable recorded silence/room tone when available or label digital
silence explicitly. Plan the associated original-frame continuation/hold, or
reject it under a stricter visual policy.

### Canonical scheduling

The planner, not the encoder, decides clip lengths. Audio uses integer project
sample positions and source mappings; video uses rational rates and original PTS.
Normalize legacy decimal FPS aliases explicitly, for example `29.97` to
`30000/1001`, and also accept explicit rational input.

Calculate video boundaries from cumulative output time. Do not round each phone
duration separately or force audio to one frame/34 ms. A subframe audio segment
may have no separately displayed video frame: record that fact, choose a feasible
visual mapping, or select a different path if each edit must be visible.

For output audio duration `D` samples, sample rate `S`, and FPS `p/q`, an explicit
end-covering CFR policy uses `ceil(D * p / (S * q))` frames. Any final-frame hold is
recorded, and video overhang is less than one output frame. Interior quantization
error must remain within the declared bound without accumulating across cuts.

Handle audio/video source-start offsets, codec priming, VFR, and mixed source
rates. Do not mark every source as having the project frame rate.

### Output transaction

1. Freeze and validate the plan, including source hashes and all required assets.
2. Assemble the PCM master at the declared sample rate. Assert its exact sample
   count and validate waveform bounds/processing.
3. Assemble original-frame video from that plan: Individual video cut slices from compressed H.264 sources must be re-encoded to constant framerate (-c:v libx264 -preset veryfast -crf 18 -r 30000/1001 -an) before concatenation to avoid keyframe alignment freezes. Only the final multiplexing of the concatenated video with the master PCM WAV uses -c:v copy.
4. Encode delivery audio once from the PCM master and mux with video.
5. Verify stream presence, logical presentation durations, and frame scheduling,
   accounting for AAC priming/padding instead of ignoring it.
6. Produce the requested XML, provenance plan, and quality report from the same
   finalized state. Publish outputs only after all requested artifacts validate.

Use temporary output names and a completion manifest or equivalent validated
publication protocol; multiple files cannot be assumed atomically committed as
one merely because each rename is atomic. On failure, preserve logs and clearly
mark partial artifacts as incomplete.

Do not create a lossy AAC file for every tiny unit and stream-copy those packets
as the authoritative audio assembly. Do not derive the master WAV from the MP4.
If source decoding or a required contribution fails, stop rather than skip it.

### FCP7 XML fidelity

The authoritative editable timeline uses original/proxy video intervals and the
PCM master for audio when subframe cuts, fades, overlaps, or resampling cannot be
faithfully represented by the source-clip XML path. Optional source-audio reference
tracks must be muted/disabled to avoid doubled sound.

Keep the JSON plan alongside the export so source audio edits remain inspectable.
Use correct native/source rate metadata and project rate separately. If a requested
source-only export is approximate, report its limitations explicitly rather than
claiming render equivalence. Validate supported exports in an actual target NLE
before declaring that compatibility complete.

**Exit gate:** exact PCM timing, bounded nonaccumulating video quantization,
no unplanned padding or leaked neighboring speech, faithful master-audio export,
and explicit failure handling. Verify a long sequence as well as tiny unit cases.

## 10. Milestone M5: CLI integration and baseline release gate

Preserve existing commands/options and document new options as they land. The
following is the proposed interface, not a claim about today's CLI:

| Option | Contract |
| --- | --- |
| `--selector legacy\|span` | Initially opt in to `span`; switch the default only after the baseline release gate. |
| `--speaker <id>` | Select a verified speaker identity from source/interval metadata. |
| `--assume-single-speaker` | Explicit user assertion for otherwise unlabeled input; require a speaker ID and record this verification method. Never apply it silently. |
| `--mode source-derived\|cuts-only` | Disclosed join processing versus a cuts-only reference. No pitch/time modification in either initial mode. |
| `--features basic\|wavlm` | `basic` by default; `wavlm` becomes available only in M6 with validated optional dependencies/configuration. |
| `--plan <path>` | Persist the canonical plan; otherwise derive its path from the render basename. |
| `--fps <decimal-or-rational>` | Preserve documented decimal aliases and support rational rates. |
| `--guide <audio>` | Optional later analysis-only prosody guide, not an initial baseline requirement. |

Apply relevant planning options to both `mix` and `auto`. A single verified
speaker may be selected automatically; ambiguous multi-speaker input must not be
silently pooled. Preserve the existing video/VTT/inventory arguments and validate
their associations.

Example after M1-M5 are implemented, with the verified manifest above:

```powershell
python .\run_mixer.py mix `
  --manifest .\sources_manifest.json `
  --speaker speaker-a `
  --text "The target sentence goes here." `
  --selector span `
  --features basic `
  --mode source-derived `
  --fps "30000/1001" `
  --render .\output\demo.mp4 `
  --plan .\output\demo.plan.json `
  --xml .\output\demo.xml
```

Use the configured project Python environment; this example does not install
dependencies or download models. Preserve standalone preparation/alignment/
inventory/inspection commands for debugging and reuse.

The report should identify selected pronunciations, target coverage, source runs,
cut count, score components, boundary changes, uncertainty/quality warnings,
processing, model/configuration identities, search approximations, and expected/
observed output timings. Provide useful alternative candidates for bad joins.

Update the root usage README during implementation to distinguish implemented
features from experiments, remove alignment/naturalness guarantees, and document
cache/schema migration and XML audio behavior.

### Release measurements

Create separate test strata:

- Intact source phrases and words.
- New combinations of recorded words.
- Unseen words requiring subword assembly.
- Proper names, pronunciation variants, reductions, questions, and emphasis.
- Different voices, recording sessions, and challenging but supported media.

Freeze the corpus/target lists, development weights, comparison setup, and
acceptance margins before collecting held-out judgments. Use matched loudness
and randomized blinded playback. Compare legacy selection and the new selector
under the same renderer as well as complete end-to-end systems.

Record listener transcription accuracy, paired naturalness preference, audible
join ratings, visual distraction, and manual correction effort. ASR output and
feature distances are supplemental diagnostics, not naturalness ground truth.
Report uncertainty and per-stratum failures instead of only one aggregate score.

Promote the new default only after all hard invariants pass and held-out listening
supports an improvement without a predeclared intelligibility regression. If
evidence is insufficient, keep the new selector opt-in and report that outcome.

## 11. Milestone M6: optional WavLM and guide experiments

### WavLM decision

**Retain WavLM in the roadmap; do not make it the initial dependency or assume
that it is the best join metric.** Start with a pinned WavLM Base+ checkpoint as
a practical experiment. Compare a small declared set of layers and, if justified,
a larger checkpoint or HuBERT using the same evaluation.

Implementation requirements:

1. Add optional neural dependencies and explicit model acquisition only now.
   Pin revisions, record license/resource requirements, and permit offline reuse.
2. Extract contextual features in batches from analysis audio. Handle chunk
   overlap/edges explicitly and map vectors back to original source times.
3. Index feature references, not copies in every phone/word record. Include
   checkpoint/layer/chunking/normalization in cache keys.
4. Without guide audio, use lexical/phonetic matches to form candidate pools and
   embeddings only to rank compatible realizations or joins. WavLM has no native
   arbitrary-text query representation.
5. Keep protected phrase/continuation candidates and explicit pitch, voicing,
   context, energy, and quality terms. Do not let approximate nearest-neighbor
   retrieval eliminate a valid complete path without a diagnostic/retry.
6. Preserve source intervals through retrieval. Never average embeddings and
   vocode them; kNN-VC's synthesis stage is not allowed.
7. Keep PCM-based coupling refinement. Roughly 20 ms feature spacing does not
   identify a sample-accurate cut.
8. Missing models, incompatible feature schemas, or extraction errors are explicit
   failures when this backend is requested. No fake vectors or silent switch.

Evaluate independently:

- Basic features versus basic plus WavLM retrieval.
- Basic features versus basic plus WavLM join/path reranking.
- Selected layers/checkpoints, candidate recall, indexing cost, warm-query latency,
  peak memory, and blinded listening outcomes.

Hold rendering, target coverage, speaker eligibility, and test material constant
for the relevant ablations. Define a practical effect size and resource budget
before evaluation; do not select a default from an unblinded favorite example.
If there is no reliable benefit, keep WavLM experimental or disabled.

### Human prosody guide

As a separate optional experiment, align a user-recorded reading of the target and
extract durations, pauses, emphasis, and relative F0. Normalize pitch/rate guidance
for the selected source speaker; make it a soft preference when the corpus cannot
match it. Validate the guide transcript rather than assuming it matches.

Guide audio is analysis-only and may never leak into the render. Include its hash
and alignment configuration in plan/cache identity. A guide need not require
WavLM: conventional pitch/duration analysis is a valid independent baseline.

## 12. Consolidated verification checklist

Use the existing runner once established in M0; execute the smallest relevant
groups during development and the complete acceptance suite before release.

| Area | Required checks |
| --- | --- |
| Input | Multiline/rolling captions, repetitions, punctuation, names/numbers, missing transcripts, speaker tags/intervals, overlapping speech |
| Alignment/inventory | Bounds/order/tier validation, real silence context, phone-to-word mapping, variants, deduplication across chunks, unknown confidence, manual boundary audits |
| Features | Voiced/unvoiced and high-F0 cases, short phones, padded windows, stereo/energy behavior, explicit failures, content/config-aware cache reuse |
| Selection | Protected rank-31 run, repeated 14 ms unit, cross-word context, compatible pronunciation branches, missing-phone failure, exact versus brute-force paths, deterministic ties |
| Candidate limits | Whole-run preservation, bounded-span continuations, coverage after pruning, wider-search retry, distinction between exhaustion and true absence |
| Audio | No universal gaps, preserved natural runs, boundary-content protection, anti-phase/transient tests, exact planned sample count, no per-unit AAC masters |
| Video | 24, 25, `30000/1001`, and 30 fps; mixed-rate/VFR sources; stream offsets; subframe cuts; explicitly requested pause/overlap coverage |
| Long timelines | At least 1,000 synthetic short contributions; exact master samples and no cumulative drift beyond the declared one-frame presentation bound |
| Export | Same frozen plan for render/XML, proper source-rate metadata, master-audio playback agreement, supported-NLE validation |
| Reliability | Bad media, failed required subprocess, partial writes, parallel cache builders, stale schema/content/model hashes, paths with spaces/Unicode, safe cleanup |
| Compatibility | All existing CLI subcommands, direct inventories with explicit version handling, inspector behavior, intentional migration/default changes |
| Provenance | Every speech sample contribution references an eligible source range; guide/model waveforms absent; gaps, conversion, gains, overlaps, holds, and proxies recorded |
| Listening | Frozen held-out strata, blinded legacy/new comparisons, intelligibility and naturalness, uncertainty reporting, manual correction effort |

For exact preservation tests, use cuts-only mode, decoded PCM at a common rate,
and known fixtures. Compare sample arrays and counts; do not demand byte-identical
AAC/MP4 encodes. For source-derived mode, compare the declared deterministic
transforms and source mappings. Validate logical muxed timing after codec delay/
padding, separately from encoded packet counts.

## 13. Deferred work and completion criteria

Defer learned perceptual join costs, MAPS-style ensemble uncertainty, automatic
speaker/visibility labeling, additional languages, and optional WSOLA/PSOLA-like
time/pitch processing until the baseline and evaluation harness are reliable.
These are extensions, not reasons to postpone a useful original-clip editor.

Do not substitute a newer word aligner for a validated phone aligner based on
release date alone. WhisperX can assist transcription/word anchoring; its default
character alignment is not an ARPAbet phone inventory. Qwen3's reported 80 ms
timestamp bins likewise do not define fine splice points. See the research notes
for the scope and limits of each method.

The initial refactor is complete only when M0-M5 gates pass, the supported
source-to-output provenance and timing contracts are verified, existing workflows
have a documented migration path, and held-out evaluation supports the declared
quality claims. M6 is a separately reported experiment, not a prerequisite for
shipping the baseline and not a promise that WavLM will win.
