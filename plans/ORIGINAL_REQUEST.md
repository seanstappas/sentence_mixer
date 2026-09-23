# Original User Request

## Initial Request — 2026-09-20T09:41:20-07:00

Enhance the sentence mixing engine in D:\Code\sentence_mixer with independent ASR listening verification, vocal separation pre-processing, audio loudness normalization, and rigorous cut accuracy.

Working directory: D:\Code\sentence_mixer
Integrity mode: development

## Requirements

### R1. Independent ASR Listening Verification Tool
Build a standalone verification script tools/listen_verify.py that operates independently from the mixing framework:
- Uses faster-whisper (or local Whisper) to transcribe any rendered output MP4 or master WAV audio.
- Compares recognized transcription against the target text to compute Word Error Rate (WER), Character Error Rate (CER), and word presence.
- Exits with a nonzero status code if transcription divergence exceeds a configurable threshold (e.g., missing critical target words or bleeding unintended dialogue).
- Integrate this verification check into tools/evaluate.py and tools/validate_mix.py as an automated acceptance gate.

### R2. Output Audio Loudness Normalization (Post-Processing)
Implement a loudness normalization post-processing stage:
- Normalize synthesized dialogue and master audio according to the EBU R128 / ITU-R BS.1770 standard (e.g. -16 LUFS for web media, -1.0 dBTP true peak limit).
- Balance loudness across disparate source video clips so cuts flow smoothly without jarring volume jumps between quiet whispers and shouting.
- Ensure normalization prevents digital clipping without introducing pumping or dynamic distortion.
- Wire into run_mixer.py with an opt-in/opt-out flag (e.g. --normalize-loudness, enabled by default).

### R3. Vocal Separation / Speech Isolation (Pre-Processing)
Integrate a CLI-based vocal separation workflow into the corpus ingestion and preprocessing pipeline:
- Use the audio-separator CLI (leveraging UVR5 MDX-Net / VR models) with optional Demucs fallback to strip background music, ambient hums, and sound effects from input videos.
- Controlled via a pre-processing CLI flag (e.g. --isolate-vocals).
- Isolated vocal stems serve as the speech source for MFA forced alignment and unit extraction, preventing background orchestral scores or sound effects from contaminating phoneme boundaries and synthesized cuts.
- Cache isolated stems under .build/vocals/ to avoid redundant separation on repeated runs.

### R4. Substring Cut Accuracy & Boundary Bleed Elimination
Diagnose and resolve cut boundary discrepancies in unit retrieval and rendering:
- Ensure that variable-span phrase and word cuts strictly begin and terminate on the exact sample/frame boundaries of the targeted words without bleeding into adjacent words (e.g. preventing "high ground" from leaking into "I have the").
- Verify that TextGrid sample time mappings and FFmpeg video PTS offsets match 1:1, preventing visual or acoustic bleed across edit points.

## Acceptance Criteria

### Independent ASR Listening Gate
- [ ] tools/listen_verify.py --input <render_path> --expected "<text>" executes independently, transcribes the audio, and outputs structured JSON metrics (transcription, WER, word matches).
- [ ] tools/evaluate.py runs ASR verification across all evaluation suite test cases and reports automated WER scores.
- [ ] Running ASR verification on the synthesis of "Anakin, I have the MOXY!" confirms that the recognized text matches the target without unintended words.

### EBU R128 Loudness Normalization
- [ ] Rendered master WAV audio measures within +/- 1.0 LUFS of the target (-16 LUFS) with true peak <= -1.0 dBTP across multi-clip mixes.
- [ ] Volume jumps between spliced segments from different scenes are perceptually balanced without audible clipping or abrupt gain pumping.

### UVR Vocal Isolation Pre-Processing
- [ ] Running ingestion with --isolate-vocals extracts clean speech tracks using audio-separator, caching the vocal stem in .build/vocals/.
- [ ] MFA alignment on isolated vocal stems produces valid TextGrids with equal or higher phonetic confidence than noisy originals.

### Exact Cut Fidelity & Verification
- [ ] Synthesizing "Anakin, I have the MOXY!" produces exact dialogue without audio or video bleed from neighboring dialogue lines ("high ground").
- [ ] tools/validate_mix.py and tools/listen_verify.py pass cleanly for both basic and WavLM render pipelines.
- [ ] All existing automated tests in tests/ pass with zero regressions.

## Follow-up — 2026-09-21T18:11:08-07:00

Upgrade and harden the Sentence Mixer engine in D:\Code\sentence_mixer by implementing acoustic multi-speaker diarization and classification, hybrid phonetic syntax with microphone input, cadence and intonation biasing, interactive out-of-vocabulary learning, the audit remediation gaps (R01–R18), and bug fixes.

Working directory: D:\Code\sentence_mixer
Integrity mode: development

## Reference Materials
- Major Capabilities Plan: plans/9_21_gemini_major_improvements_plan.md
- Major Capabilities Design & Rationale: plans/9_21_gemini_major_improvements_design.md
- Design Review & Refinements: plans/9_21_astra_thoughts.md
- Audit Remediation Work Packages: plans/audit-remediation-plan.md
- Bug Fixes & Edge Cases: plans/9_21_gemini_bugs.md

## Requirements

### R1. Acoustic Speaker Diarization & Persistent Identity Classification
Implement robust acoustic multi-speaker turn discovery and classification for multi-speaker source media:
- Integrate offline neural speaker diarization (e.g. pyannote/speaker-diarization-community-1 or 3.1) operating on clean vocal stems (vocals.wav) to discover and separate speakers directly from the audio (do not rely on subtitle markup).
- Preserve turn boundaries and overlap regions. Reconcile diarized intervals against aligned word/phone boundaries without applying naive chunk-wide majority voting (exclude multi-speaker overlapping speech from single-speaker candidate pools).
- Support a persistent speaker registry enabling review, naming, merging, and cross-video cluster mapping.

### R2. Hybrid Phonetic Syntax & Microphone-to-Phoneme Ingestion
Expand the input representation to support explicit phonetic control and microphone transcription:
- Parse space-delimited ARPAbet tokens enclosed in braces (e.g. {B AE1 K AH0 N}) representing individual word entities.
- Implement a reverse index from realized pronunciations to recorded word instances and contiguous phone spans in the unit selector, preserving whole-word and phrase candidates.
- Add microphone audio capture (and WAV file ingestion) with acoustic phoneme recognition (e.g. Wav2Vec2 CTC phone model) to convert spoken utterances into candidate ARPAbet strings with interactive review and editable grouping. Microphone audio is analysis-only and must never leak into the output waveform.

### R3. Cadence Control, Video Freeze-Frames & Intonation Biasing (!, ?)
Implement cadence control and expressive pitch/energy biasing:
- Support explicit gap syntax (<gap:250ms>) compiled through the canonical timeline.
- For inter-word pauses, hold/freeze the last video frame of the preceding cut or extend the tail of the source clip if the speaker is silent, preventing black frames, video looping, or audio pitch stretching.
- Apply soft Viterbi target cost biasing for punctuation marks:
  - Terminal ?: Reward positive terminal pitch slope (Delta F_0 / Delta t > 0).
  - Terminal !: Reward high RMS energy and primary lexical stress (1).
  - Fall back cleanly to neutral units without failing synthesis if ideal pitch/energy matches are absent.

### R4. Interactive Out-of-Vocabulary (OOV) Learning Assistant
Provide interactive assistance when encountering unknown words:
- Interactive by default in TTY terminals with a --non-interactive flag to bypass for automated headless pipelines and test suites.
- Present "sounds-like" phoneme exemplars (e.g., 'a' in apple, 'u' in cut) alongside ARPAbet tokens.
- Allow users to accept, edit, or enter pronunciations via text or microphone.
- Automatically persist confirmed pronunciations to dictionaries/custom_pronunciations.txt for immediate reuse across runs.

### R5. Audit Remediation (R01–R18) & Engine Hardening
Remediate all verified architectural gaps identified in plans/audit-remediation-plan.md:
- R01/R11: Reject invalid source ranges instead of padding silent synthetic output.
- R02: Fix subtitle deduplication to preserve legitimate repeated words/lines.
- R03: Enforce strict speaker eligibility filtering without silent fallback.
- R04/R13: Fingerprint dependencies, model configs, and audio stems to prevent stale cache reuse and stem inversion.
- R05/R10: Fix accumulating video drift and timeline sample/frame synchronization.
- R06/R08/R09: Enforce valid pronunciation branch coverage, prune paths soundly, and handle transitions/coupling variants cleanly.
- R07/R14/R15: Reconnect basic acoustic scoring and WavLM ranking into the selection objective with validated participation.
- R12: Ensure Premiere Pro / FCP7 XML exports contain full media definitions.
- R16–R18: Add rigorous offline test markers, clean lint/type errors, and verify CLI options.

### R6. Bug Fixes & Edge Cases
Resolve the specific defects documented in plans/9_21_gemini_bugs.md:
- Fix vocal stem cache validation loop in ensure_source_aligned by ensuring cached_vocal_pcm_path is serialized into inventory metadata.
- Change flat globbing to recursive globbing (rglob) in MFA alignment batching.
- Synchronize effective isolate_vocals state into plan.metadata.
- Do not silently overwrite explicit user-specified --speaker arguments.

## Acceptance Criteria

### Diarization & Speaker Classification
- [ ] Diarization intervals properly segment multi-speaker audio and tag individual aligned word events in units_index.json without relying on subtitle voice tags.
- [ ] Overlapping speech turns are marked and excluded from single-speaker candidate pools.

### Hybrid Phonetic Syntax & Microphone Ingestion
- [ ] Syntax such as {B AE1 K AH0 N} parses cleanly into a phonetic word entity.
- [ ] The unit selector matches whole recorded words corresponding to the phonetic sequence when available.
- [ ] Microphone/audio ingestion produces editable ARPAbet suggestions without inserting microphone audio into rendered outputs.

### Cadence, Freeze-Frames & Intonation Biasing
- [ ] <gap:250ms> inserts an exact 250 ms pause into the master audio and holds the preceding video frame.
- [ ] Synthesis of sentences ending with ? demonstrates lower target cost for rising F_0 units.
- [ ] Synthesis of sentences ending with ! demonstrates lower target cost for high RMS/stress-1 units.

### Interactive OOV Assistant
- [ ] Encountering an unknown word in interactive mode prompts the user with phoneme exemplars and options to edit or accept.
- [ ] In --non-interactive mode, OOV words fall back to G2P without blocking.
- [ ] New pronunciations are saved to dictionaries/custom_pronunciations.txt and reloaded automatically.

### Audit Remediation, Bug Fixes & Test Suite
- [ ] Missing or invalid source ranges trigger explicit errors rather than silent padded output.
- [ ] MFA batching uses rglob to discover nested chunk WAVs and TextGrids.
- [ ] Cache validation does not enter an infinite invalidation loop when vocal stems exist.
- [ ] User-specified --speaker flag is respected even in single-speaker inventories.
- [ ] All existing automated tests in tests/ pass with zero regressions.

## Follow-up — 2026-09-22T02:06:24Z

User Update & New Inputs for Milestone 2 (Acoustic Speaker Diarization & Registry):

1. Anakin Media Added in input/:
   - `input/anakin_ep2_1.mp4` (+ `.vtt`)
   - `input/anakin_ep2_2.mp4` (+ `.vtt`)
   - `input/anakin_ep3_1.mp4` (auto-transcribing to `anakin_ep3_1.en.vtt`)
   - `input/anakin_ep3_2.mp4` (+ `.vtt`)
   - Existing: `obi_wan_ep2.mp4`, `obi_wan_ep3.mp4`
   Use these files to empirically test and verify speaker recognition / separation between Anakin and Obi-Wan.

2. Generic Speaker Clustering & Interactive Naming:
   - For generic source media without pre-labeled speakers, the diarization/clustering tooling must acoustically group turns into speaker clusters.
   - Compare cluster embeddings against known speaker profiles in `config/speaker_registry.json`.
   - If a cluster does not match a known profile, prompt the user interactively in TTY for speaker name(s) (with fallback auto-naming `speaker_01`, etc. when `--non-interactive` is passed).
   - Persist confirmed speaker profiles to `config/speaker_registry.json`.

3. Alignment of anakin_ep3_1:
   - We are currently generating `input/anakin_ep3_1.en.vtt` via Whisper so it has subtitles and can be aligned cleanly.

Incorporate these requirements into Milestone 2 execution and verification.
## Follow-up — 2026-09-22T20:50:45Z

Implement the complete remediation handoff in `plans/9_22_antigravity_remediation_plan.md` end to end in `D:\Code\sentence_mixer`.

Working directory: D:\Code\sentence_mixer
Integrity mode: development

## Requirements

### R1. Cumulative Video Scheduling & Timeline Alignment (A12, T01-T03)
Schedule video frames strictly from the cumulative audio timeline rather than per-segment flooring or independent duration rounding:
- Use one unified integer quantizer: $Q(n) = (n \cdot p + S \cdot q - 1) // (S \cdot q)$ for sample rate $S$ and rational frame rate $p/q$.
- Calculate segment video frames as $Q(n_{\text{end}}) - Q(n_{\text{start}})$, permitting sub-frame audio cuts to have 0 separately displayed frames without truncating the video timeline.
- When an audio cut exceeds available source video PTS, hold/clone the final eligible source video frame (e.g. using `tpad=stop_mode=clone`) rather than advancing into unselected neighboring dialogue or truncating video.
- Eliminate audio/video timing drift so $N$ cuts match cumulative duration exactly (e.g., 6 x 50-ms cuts at 48 kHz / 30 fps render exactly 14,400 audio samples and 9 video frames; 1,000 cuts render exactly 2,400,000 audio samples and 1,500 video frames).

### R2. Hard Gap Constraints, Phonetic Validation & Branch Isolation (A05, A07, T04-T07)
Enforce strict syntax parsing, graph branching, and timeline gap preservation:
- Treat explicit `<gap:Xms>` as hard timeline constraints that phrase candidate retrieval cannot cross. `<gap:0ms>` strictly compiles to 0.0s (not defaulting to 250ms).
- Support leading, trailing, zero, and consecutive gaps in both standard and cuts-only modes, holding the appropriate source frame for video pause intervals.
- Tag target graph nodes with explicit pronunciation branch IDs so multi-word or whole-word candidates cannot span incompatible pronunciation variants (e.g. `READ = R IY1 D` vs `R EH1 D`).
- In `lexicon.py`, construct compound word alternatives via bounded Cartesian products choosing exactly one pronunciation per component.
- Strictly validate phonemes and stress (stress 0, 1, or 2 permitted only on vowels; reject consonant stress like `T1` or invalid stress like `AE9`).

### R3. Connected Measured Acoustic Join Scoring & Prosody Guidance (A06, A07, A11, T08-T10)
Connect actual measured acoustic features and prosody guide inputs to candidate selection:
- Extract and supply continuous boundary feature records (energy, spectral flux, voicing, F0) into `compute_join_cost(tail_feat, head_feat)` in `SpanSelector` instead of passing `(None, None)`.
- Connect `ProsodyGuide` relative pitch contour to target candidate scoring.
- Protect natural continuous source runs from pruning before DP selection, and charge new-edit penalties only for introduced joins.
- Swapping compatible/incompatible boundary features or guide preferences must measurably change the selected path and cost trace.

### R4. Overlap-Aware Diarization & Word Eligibility Safety (A08, T15-T16)
Implement safe speaker reconciliation and identity mapping:
- Eliminate the singleton auto-assignment shortcut (`speaker_01`). Require calibrated cosine threshold verification (> 0.75) and margin checks before assigning speech turns to enrolled registry profiles.
- Flag ambiguous matches as unresolved requiring review rather than silently contaminating persistent profiles.
- Exclude words with >30% overlap or out-of-interval speech from single-speaker candidate eligibility (`is_eligible = False`). Eliminate `speaker_id="OVERLAP"` with `is_eligible=True`.
- Noninteractive unknown clusters receive recording-scoped IDs, not global shared profiles.

### R5. Real Microphone / WAV Adapter & Unified Interactive Review (A09, A10, T17-T19)
Provide a concrete capture/audio file adapter and unified target preparation:
- Connect a concrete audio ingestion interface with explicit 16 kHz mono resampling.
- Implement acoustic phoneme recognition with error and uncertainty handling, eliminating invented `{AH0}` fallbacks on empty/unrecognized audio.
- Align `oov_assistant.py` method calls with actual `PhonemeRecognizer` signatures.
- Unify target preparation and pronunciation review before selector dispatch across both legacy and span modes.
- Implement atomic, concurrency-safe persistence for `custom_pronunciations.txt` with deduplication and phone validation.

### R6. Cache Invalidation, MFA Staging & Faithful Delivery (A02, A04, A13, A14, T11-T14, T21-T23)
Harden outer caching, batch staging, and output generation:
- Fingerprint media content bytes, selected stream, transcript bytes, speaker intervals, lexicon version, alignment model, and isolation options before accepting outer inventory cache hits in `run_mixer.py`.
- Preserve relative directory hierarchy during MFA batch chunk staging to prevent chunk filename collisions across multi-speaker inputs.
- Pass the finalized master WAV path explicitly to FCP XML generation (do not guess from XML basename). Export canonical picture schedules with holds/freezes and accurate channel mappings.
- Freeze all DSP decisions (gains, fades, normalization) in `EditPlan` before rendering, and stage deliverables in a unique attempt directory before atomic publication.

### R7. Truthful Acceptance Gates & CLI Unification (A15, A16, T20, T24-T25)
Unify CLI flags and enforce rigorous acceptance criteria:
- Wire `--text`, `--target-audio`, `--record-target`, `--diarize auto|off|force`, `--speaker-registry`, `--guide-transcript`, and `--cache-dir` with mutual exclusion and actionable error handling. Reject unsupported legacy option combinations.
- In `listen_verify.py` and `evaluate.py`, enforce strict token matching that catches missing negations (e.g., "we are not going home" vs "we are going home"), missing words, and unexpected dialogue bleed.
- Return exit code 0 for passing, 1 for acceptance failure, and 2 for operational/configuration error.
- All donor speech remains strictly source-derived; analysis tools (ASR, WavLM, diarization, guide) must never donate audio to rendered speech.
- Preserve the dirty working tree; do NOT run `git commit`.

---

## Acceptance Criteria

### Audio/Video Timeline & Exact Gaps (T01-T05)
- [ ] 6 x 50-ms cuts at 48 kHz / 30 fps in cuts-only mode produce exactly 14,400 master audio samples and exactly 9 video frames.
- [ ] 1,000 cuts produce exactly 2,400,000 master audio samples and exactly 1,500 video frames without cumulative drift.
- [ ] Ultra-short subframe audio cuts are preserved with 0-frame picture spans allowed.
- [ ] `hello <gap:250ms> there` produces exactly 12,000 silence samples at 48 kHz without phrase retrieval bridging across the gap.
- [ ] Zero, leading, trailing, and consecutive gaps compile and render faithfully with corresponding picture holds.

### Phonetic Correctness & Acoustic Scoring (T06-T10)
- [ ] Heteronym `read` (`R IY1 D` vs `R EH1 D`) is resolved to a single connected pronunciation branch per occurrence; compound words like `read-off` generate one pronunciation per component.
- [ ] Invalid phone/stress syntax (`{AE9}`, `{AH10}`, `{T1}`) is rejected with location-aware errors and not silently repaired or saved.
- [ ] Swapping measured basic boundary features changes the chosen DP path and transition cost trace.
- [ ] Prosody guide pitch preferences feed candidate costs in `SpanSelector`.
- [ ] Protected suitable natural source runs survive pruning and are preferred over over-fragmented phone splices.

### Provenance, Diarization & Word Safety (T11-T16)
- [ ] Modifying media bytes, speaker intervals, or lexicon invalidates outer inventory cache and triggers re-alignment.
- [ ] MFA batch staging preserves directory structure, preventing chunk collisions across multi-speaker inputs.
- [ ] Two distinct singleton speakers are not automatically merged into `speaker_01`.
- [ ] Speech intervals with >30% overlap or outside valid speech intervals are marked `is_eligible = False`.

### Microphone, XML & Acceptance Gates (T17-T25)
- [ ] Microphone/WAV adapter ingests audio at 16/44.1/48 kHz, resamples to 16 kHz mono, and handles empty/uncertain input without fake `{AH0}`.
- [ ] XML exporter receives explicit master WAV path and correctly renders freeze frames as hold clips regardless of XML basename.
- [ ] `listen_verify.py` rejects dropped negations ("we are not going home" vs "we are going home") with exit code 1.
- [ ] Full offline test suite (`pytest -m "not model and not real_corpus"`), `ruff`, and `mypy` pass with 100% success and 0 errors.
