# Source-Preserving Sentence Mixer: Design and Rationale

Updated: 2026-09-19.

**Status: proposed architecture, not a description of completed features.** The
current code selects whole words independently, with phone fallback. The
[milestones](milestones.md) defines the staged migration and
acceptance tests. The [research notes](../research/README.md) distinguish findings
in the supplied papers from engineering recommendations.

## 1. Goal and constraints

Given source videos and target text, assemble the requested speech from the
speaker's original recordings, together with original source video frames.
Optimize intelligibility, sentence-level delivery, acoustic joins, and the
recognizable cut-based YTP editing style.

The output must not use generated speech, a neural vocoder, voice cloning, face
inpainting, or generated mouth movements. Neural models may analyze the sources
or rank candidates; they must not synthesize the delivered speech or frames.

Two explicitly identified rendering modes are useful:

| Mode | Allowed operations |
| --- | --- |
| Source-derived, the intended default | Original decoded audio and frames; disclosed format conversion, gain, short fades, and constrained audio overlaps. No pitch shifting or time stretching in the initial implementation. |
| Cuts-only reference | Original intervals with no join smoothing, gain adjustment, or speech retiming. Necessary format conversion remains recorded; this is not a claim of bit-identical container output. |

Every speech contribution must have a source interval and transformation record.
Requested pauses may use verified source room tone or explicitly recorded digital
silence; never substitute unrecorded speech. Later experiments with deterministic
pitch/time modification require a separate opt-in policy and corresponding video
timing. They are not part of the baseline.

Original video provenance does not prove that the visible person is speaking:
sources can contain reaction shots, narration, or off-screen dialogue. Visibility
is separate metadata and, when required, a candidate eligibility constraint.
Frame quantization and the brief two-source contribution during an audio overlap
must also be disclosed rather than described as perfect lip synchronization.

The first supported configuration is English speech, one explicitly selected
speaker per output, and one or more source recordings. Dictionaries, normalization,
and alignment backends should be replaceable, but multilingual quality is a
separate validation task. A file name is not a speaker identity.

### What cannot be guaranteed

An inventory containing every phone still may not contain the needed transition,
stress, intonation, or recording condition. No search algorithm can guarantee
natural arbitrary sentences from inadequate source material. Return useful
coverage diagnostics and alternatives instead of inventing speech or dropping
required words. Corpus curation and additional recordings can improve quality
more than a more complicated similarity model.

## 2. Architectural decision and evidence

Use **variable-span, source-contiguous unit selection**, with a segmental
shortest-path/Viterbi-style search. Keep acoustic analysis separate from the
original-media renderer.

| Approach | Decision |
| --- | --- |
| Independent median-duration word selection | Retain as an A/B baseline, not the desired final optimizer. It preserves words but ignores sentence-wide compatibility. |
| Phone lattice followed by merging | A valid intermediate baseline if continuation paths survive pruning; insufficient as a guarantee of whole-word or phrase preservation. |
| Variable-span selection | Chosen architecture: phrases, words, and shorter spans compete in one global search, with phone fallback and explicitly modeled half-phone/diphone extensions. |
| WavLM/HuBERT analysis | Optional later retrieval/ranking features. Their usefulness for original-waveform joins must be measured. |
| Generative TTS, neural codecs/vocoders, talking-head rendering | Outside the output contract, regardless of how natural their results sound. |

Hunt and Black (1996) already assign zero concatenation cost to consecutive
database units. Black and Taylor (1997) use uniform phone units and search movable
coupling points; they do not establish the previous plan's claimed guarantees.
VoCo (2017) contributes useful variable-length range selection and alternative
candidates, but also uses a synthetic guide and signal modification. Fried et al.
(2019) uses viseme retrieval, retiming, and neural rendering, so its audiovisual
realism is not evidence for an untouched-clip renderer.

The [2026 MFA comparison](https://arxiv.org/abs/2606.18466) supports retaining MFA
as a serious phone-alignment baseline, not replacing it solely because another
aligner is newer. Its reported mean phone-boundary errors for pretrained MFA
ARPA 3.0 on TIMIT and Buckeye are 12.11 and 13.87 ms, respectively. These are
benchmark-specific estimates, not universal guarantees; model/domain differences,
phone inventories, and training-data overlap matter. The work is a preprint.
Software version, acoustic-model version, and dictionary version are distinct.

Alignment timestamps are estimates, not sample-accurate ground truth. The
[2026 ensemble-boundary study](https://doi.org/10.1515/phon-2025-0039) motivates
retaining uncertainty information. Such intervals describe alignment uncertainty,
not a guarantee that every point inside them is a safe splice.

## 3. Pipeline and responsibilities

```text
Original media + transcript + speaker/interval metadata
  -> media probe and original-rate PCM cache
  -> transcript normalization, utterance preparation, and quality checks
  -> MFA word/phone alignment with pronunciation and source-time mappings
  -> ordered inventory + continuous acoustic features
  -> target pronunciation graph + variable-span candidate retrieval
  -> sequence search over compatible spans and bounded coupling variants
  -> validated, immutable edit plan
  -> PCM master + original-frame video + timeline export + provenance report

Optional later analysis:
  WavLM features / human prosody guide -> candidate scoring, never output synthesis
```

### 3.1. Trustworthy inputs and inventory

- Retain all lines in ordinary multiline subtitles. Deduplicate rolling captions
  using temporal overlap and repeated prefixes/suffixes, not global text removal.
  Preserve genuine repetitions, punctuation, speaker tags, and word-time anchors.
- Verify transcript coverage. Forced alignment can assign plausible-looking
  times to incorrect text; successful execution is not proof of a correct match.
- Use appropriately padded utterances rather than assuming subtitle cue edges
  are acoustic boundaries. Preserve exact chunk-to-source sample offsets.
- Require verified single-speaker material or speaker-labeled intervals initially.
  Diarization can later propose segments, but cross-file cluster IDs are not
  automatically the same speaker.
- Record speech overlap, clipping, noise/music, alignment anomalies, recording
  session, and visibility. Invalid mappings are errors; uncertain measurements
  are explicit flags, not fabricated feature values.
- Keep an ordered stream of phones, silence, and noise events. Index words by
  their actual aligned phone sequences and retain pronunciation variants.

Each source is identified by content and media metadata. Each event has a stable
ID, speaker/utterance association, source sample interval, and true predecessor
and successor. Chunk boundaries are implementation details, not linguistic
boundaries; overlapping chunk observations must not create duplicate events.

### 3.2. Target specification

Normalize text while retaining original-token mappings, punctuation, lexical
stress, and phrase structure. Use the same pronunciation resources and phone
inventory as alignment. Prefer reviewed dictionary entries, then supported
pronunciation alternatives, then validated G2P for unknown words. Never use a
letter-to-phone guess as an unnoticed fallback.

Represent alternative pronunciations as legal paths through a target graph.
Selection must cover one complete, consistent realization of the requested
words; it cannot switch pronunciations halfway through a word or skip a phone.
Carry context across word boundaries. A word edge is not a silence phone.

Text does not fully specify prosody. The baseline uses soft stress, phrase-position,
punctuation, speaking-rate, and speaker-conditioned preferences. An optional
human-recorded guide can later supply relative pitch, timing, and emphasis.
The guide's waveform never enters the output.

## 4. Candidate generation and sequence optimization

### 4.1. Variable-span candidates

A candidate covers a connected target range and one contiguous original source
range. Retrieve matching phrases and words as first-class candidates, then shorter
phone sequences and, when necessary, context-bearing phone units.
Internal source pauses are retained only when compatible with the target's pause
policy. Do not silently remove an interior sound while calling the result intact.

Diphone-style spans can preserve the transition between two sounds while placing
joins inside suitable stable phones. Add them only with explicit target coverage
anchors: a split phone has ordered left/right parts, and a complete path must
cover both exactly once. Whole-word candidates cover the same parts. Arbitrarily
cutting every phone in half is not equivalent; short closures/transients may be
unsuitable. Start with whole-phone coverage and evaluate this extension separately.

Preserve complete matching runs and their continuations during pruning. Rank
additional candidates using context, prosody, quality, and boundary suitability,
not closeness to one universal duration. Configurable search limits and their
effect on coverage belong in diagnostics. Bounded span lengths must still allow
longer natural runs to compose through zero-edit continuations.

Two candidates are source-contiguous only when ordered event IDs and physical
coverage prove that no intervening event is omitted. A small timestamp difference
alone is insufficient. Reusing the same source unit for repeated target speech
is an edited repetition, not a continuation that can be merged away.

### 4.2. Objective

For a complete compatible path $P$, minimize:

$$
C(P) = \sum_{c \in P} \bigl(C_{\mathrm{target}}(c) + C_{\mathrm{quality}}(c)\bigr)
     + \sum_{(p,c) \in P}
       \bigl(\lambda_{\mathrm{edit}} I_{\mathrm{new\ edit}}
             + C_{\mathrm{join}}(p,c)
             + C_{\mathrm{visual}}(p,c)\bigr).
$$

Content coverage, speaker identity, legal source bounds, and supported
pronunciations are eligibility constraints, not small penalties that can be
outweighed by a pleasant join.

Target and quality terms use consistent coverage weights so that splitting the
same source run into several candidate records does not change its intrinsic
score. A genuine source continuation has zero introduced-edit/join cost.
Retaining a long run is therefore attractive without an arbitrary large negative
bonus. It is not mandatory when another realization has substantially better
prosody or quality.

The state includes the previous candidate and its coupling variant; the next join
depends on them. Dynamic programming is exact over the retained compatible graph,
not over candidates that were pruned away. Beam search is an explicitly reported
approximation. Ties must be reproducible.

### 4.3. Acoustic and prosodic costs

| Term | Intended treatment |
| --- | --- |
| Phonetic context | Compare neighboring sounds, stress, syllable/word/phrase position, and actual source pauses. Context agreement helps but does not eliminate coarticulation differences. |
| Duration | Use robust phone/context-conditioned distributions with documented backoff. Do not drive all sounds toward 90 ms. |
| Pitch | Compare voiced, reliable estimates in log-frequency/semitones and relative speaker range. Preserve appropriate contours; do not pull all units toward 120 Hz or a flat median. |
| Energy | Compare local entry/exit envelopes and recording-level compatibility, not only whole-unit RMS. Account for any planned gain. |
| Spectrum | Use calibrated spectral features and local trajectories at compatible coupling regions. Adjacent different phones need not have identical spectra. |
| Quality/uncertainty | Penalize or exclude unreliable alignment, overlap, clipping, and incompatible recording conditions according to explicit policy. |
| Visual continuity | Optionally account for visibility, shot/pose changes, and excessive cuts; do not assume visually similar shots always make better edits. |

Extract features over continuous analysis chunks with sufficient context, then
associate windows with source intervals. Re-decoding every phone is wasteful and
introduces artificial analysis edges. Unvoiced pitch is undefined, not 120 Hz.
Store validity and measurement confidence separately from values. Extraction
failures must be reported; unavailable features must not earn artificially cheap
join scores.

Weights and thresholds are tuning parameters established on development material,
then evaluated on held-out recordings and sentences. No supplied paper establishes
universal weights for this application.

## 5. Boundary refinement and source-derived DSP

Phone boundaries often occur in rapidly changing acoustic regions. Retrieve safe
context and evaluate a small set of bounded entry/exit variants, potentially
including stable regions within suitable phones. Protect the required phonetic
content, especially stop closures/releases and short consonants. Duration and
ordering constraints apply to each variant.

Score the variants that will actually be rendered. Do not optimize one boundary
and independently move it afterward. If later refinement changes a path's
feasibility or cost, revalidate/rerank before freezing the edit plan.

Zero-crossing proximity is at most an auxiliary feature. It does not establish
pitch-phase agreement, spectral continuity, or simultaneous stereo-channel zeros.
SoundFile reads supported audio files, not ordinary MP4/AAC video containers;
decode source PCM with FFmpeg before waveform inspection.

Short, explicitly specified fades or overlaps can suppress discontinuities, but
must be tested for transient damage, cancellation, and level dips. Match channels
coherently. Preserve naturally continuous runs without internal fades. Longer
overlaps are not automatically better, and no fixed 5 ms rule is safe everywhere.

There is no automatic 45 ms gap after every word. Preserve suitable source pauses
and model requested punctuation/phrase breaks explicitly. A deliberately choppy
editing style can expose a separate pause/cut policy; it is not the natural-speech
default.

Loudness Consistency: During audio chunk pre-processing or master timeline staging, normalize dialogue clips across disparate recording sources (e.g. Episode II vs. Episode III mastering levels) using local RMS or EBU R128 loudness matching to prevent abrupt volume jumps across cuts.

## 6. One canonical timeline

Finalize selection, coupling, fades, overlaps, gains, pauses, and video mapping
into one versioned edit-decision list before rendering or exporting anything.

- Use integer source sample indices and output sample indices for audio.
- Retain rational source time bases, decoded-audio origins, video presentation
  timestamps, and codec-delay information. Source audio and video need not start
  at the same timestamp.
- Use a common project audio rate with recorded resampling mappings. Frame
  positions derive from cumulative output time, not separately rounded clip
  durations. Use rational rates such as `30000/1001`, not accumulated `29.97`
  approximations.
- Account for overlaps and pauses in output positions. Crossfading does not
  inherently cause drift; inconsistent scheduling does.
- Do not extend a short audio interval to 34 ms merely to obtain a video frame.
  Subframe audio cuts require an explicit video policy or a different selected
  path. Not every phonetic cut can be separately visible at 30 fps.
- Select original video frames by their timestamps. Explicitly plan any frame
  hold/repetition, pause coverage, and which source is visible during an overlap.
  Default frame quantization error is bounded by one output frame without
  cumulative drift; stricter visibility requirements can reject infeasible paths.

Render the lossless PCM master first, assemble video from the same plan, and
encode delivery audio once. A WAV decoded from an AAC render is not the master.
Verify logical playback timing after accounting for encoder priming/padding.

Preview, final output, and XML consume the same frozen plan. Where FCP7 XML cannot
faithfully express subframe audio edits or processing, export the PCM master as
the authoritative audio track, plus source-video edits and the provenance plan.
Do not present an unsmoothed, differently timed XML sequence as an equivalent
render. Mixed-rate/VFR media requires native timestamp handling or explicitly
mapped conformed proxies.

## 7. WavLM: retained, but optional and evidence-gated

**Yes, WavLM remains in the roadmap. It is not required for the first useful
system, and it is not established as the best splice-quality metric.**

First implement and measure a reliable classical-feature baseline. Then benchmark
WavLM as an analysis-only feature for candidate retrieval and join/path reranking.
WavLM Base+ is a practical first checkpoint to evaluate; a larger checkpoint or
another encoder must justify its cost on this task. Test layers rather than
assuming the layer used by kNN-VC is optimal for raw waveform joins.

Important boundaries of this experiment:

1. WavLM encodes audio, not arbitrary target text. Without an audio guide, use
   phonetic/lexical retrieval first and embeddings for compatible candidate
   comparisons; do not invent a text-to-WavLM nearest-neighbor query.
2. Retain source IDs and timestamps for every feature. Its roughly 20 ms feature
   stride is not sample-accurate splice timing.
3. Keep explicit context, voicing, pitch, energy, and quality features. Similar
   embeddings are not proof of compatible excitation or a good join.
4. Do not average retrieved features and pass them through a vocoder. kNN-VC's
   HiFi-GAN output violates this project's output contract.
5. Make model/checkpoint revision, layer, normalization, and chunking part of the
   feature cache key. Keep neural dependencies optional.
6. Enable the feature by default only after held-out blinded comparisons show a
   useful improvement without intelligibility/provenance regressions and within
   a declared latency/memory budget. Otherwise keep it experimental.

A learned perceptual join scorer and a human prosody guide are reasonable later
experiments. Neither should delay a correct source-preserving baseline.

## 8. Workspace, reliability, and compatibility

Project-owned generated artifacts belong under `.build`:

```text
.build\
  sources\<content_id>\       Original PCM, analysis audio, media metadata
  alignment\<fingerprint>\    Prepared corpus, TextGrids, validation results
  inventory\<fingerprint>\    Versioned event/word index and feature references
  features\<fingerprint>\     Basic features and optional encoder arrays
  runs\<run_id>\              Run-scoped temporary renders and diagnostics
input\                       User media and transcripts; never modified
dictionaries\                Reviewed pronunciation assets
output\                      Final MP4, PCM WAV, XML, edit plan, quality report
```

Third-party model caches and isolated MFA environments may live elsewhere under
explicit configuration; they are not implicitly owned by project cleanup.

Cache fingerprints include relevant media/transcript content, normalization and
dictionary versions, acoustic models, analysis settings, and schema/tool versions.
Recompute only the affected stages. Publish validated artifacts atomically; an
existing JSON file or a path hash alone is not cache validity.

Preserve `mix`, `auto`, `preprocess`, `align`, and `build-inventory`, the inspector,
and existing direct-inventory workflows through explicit adapters/migration.
Resolve executables per platform/configuration rather than assuming
`micromamba.exe` everywhere. Do not silently replace malformed inventories,
missing features, or failed slices with success-shaped output.

## 9. Quality gates and implementation order

Implement in this order:

1. Characterize current behavior and define schemas/provenance.
2. Fix transcript preparation, speaker eligibility, alignment validation, and
   content-aware caches.
3. Extract continuous basic features and shared pronunciation/target metadata.
4. Add protected variable-span retrieval and global selection.
5. Add coupling-aware planning, sample-accurate rendering, and consistent exports.
6. Run held-out listening and timing tests before changing the default selector.
7. Evaluate optional WavLM, guide prosody, and learned ranking independently.

Use intact-phrase, novel word-combination, and unseen-word/subword test sets
separately. Measure listener transcription accuracy, blinded naturalness
preference, audible joins, visual distraction, and manual correction effort.
ASR and feature distances are diagnostics, not substitutes for listening.

Automated gates cover full target coverage, preserved natural runs, legal
contiguity, uncertainty handling, exact PCM sample counts, bounded video timing,
XML/render agreement, cache invalidation, and explicit failure reporting. The
[milestones](milestones.md) specifies the fixtures and release
criteria. Do not promise artifact-free or fully natural output before measurement.
