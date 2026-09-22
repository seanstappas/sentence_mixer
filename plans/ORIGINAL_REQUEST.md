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


