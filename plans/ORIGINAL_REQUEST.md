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
