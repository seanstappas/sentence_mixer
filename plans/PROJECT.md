# Project: sentence_mixer

## Architecture
`sentence_mixer` is an automated, video-accurate sentence mixing engine that performs phonetic and phrase-level unit selection from aligned video corpora to synthesize novel speech utterances while maintaining 1:1 audiovisual frame synchronization.

### Pipeline Architecture
1. **Ingestion & Vocal Isolation**:
   - `src/vtt_preprocessor.py`: Parses VTT cues, strips rolling caption duplicates, and slices chunks for alignment.
   - `src/vocal_isolation.py`: Uses `audio-separator` CLI (UVR-MDX-NET-Inst_HQ_4.onnx) with Demucs fallback to extract clean speech stems cached under `.build/vocals/<source_id>/vocals.wav`.
2. **Phonetic Alignment & Unit Inventory**:
   - Montreal Forced Aligner (MFA 3.4.2 in `.mamba/envs/mfa_env`) generates Praat `.TextGrid` files from speech audio chunks.
   - `src/inventory_builder.py`: Extracts word and phone intervals into `units_index.json`.
3. **Unit Selection & Edit Planning**:
   - `src/span_selector.py`: Dynamic programming Viterbi search finds optimal candidate sequence (phrase -> word -> phone).
   - `src/edit_plan.py`: Maps selected candidates to 48kHz project audio sample timeline and 29.97fps rational video frames.
4. **Assembly, Post-Processing & Loudness Normalization**:
   - `src/video_assembler.py`: Slices PCM audio, applies crossfade ramps, performs segment leveling and two-pass EBU R128 master loudness normalization (-16 LUFS, -1.0 dBTP), extracts CFR video slices, and muxes final MP4/WAV.
5. **Acceptance Verification & ASR Gate**:
   - `tools/listen_verify.py`: Independent Whisper/faster-whisper transcription, DP Levenshtein WER/CER, insertion-based bleed detection.
   - `tools/validate_mix.py` and `tools/evaluate.py`: Container, sample count, loudness compliance, and ASR acceptance gates.

---

## Feature Inventory
| # | Feature | Description | Milestone | Source |
|---|---------|-------------|-----------|--------|
| 1 | Standalone ASR Listener CLI | `tools/listen_verify.py` transcribes MP4/WAV via faster-whisper/transformers, outputs JSON with WER, CER, word matches | M1 | R1 |
| 2 | ASR Gate in Validation & Eval | Wire `tools/listen_verify.py` into `tools/validate_mix.py` and `tools/evaluate.py` as automated acceptance gate | M1 | R1 |
| 3 | Target Text Vocabulary Priming | Use `initial_prompt` in Whisper to avoid phonetic mistranscriptions of target names (e.g. MOXY) | M1 | R1 |
| 4 | Insertion-Based Bleed Rejection | Non-zero exit code if extra spoken words (insertions) exceed tolerance, rejecting dialogue bleeds | M1 | R1 |
| 5 | Master EBU R128 Normalization | Two-pass `-af loudnorm=I=-16:TP=-1.0:LRA=11` filter preserving exact sample count with `-ar 48000` | M2 | R2 |
| 6 | Per-Segment Speech Leveling | Pre-gain adjustment on disparate cuts before crossfading to prevent volume jumps between scenes | M2 | R2 |
| 7 | Loudness CLI Option | Wire `--normalize-loudness` (default True via `BooleanOptionalAction`) into `run_mixer.py` | M2 | R2 |
| 8 | Loudness Validation Gate | Assert in `tools/validate_mix.py` that integrated loudness is -16.0 +/- 1.0 LUFS and true peak <= -1.0 dBTP | M2 | R2 |
| 9 | UVR Vocal Separation Module | `src/vocal_isolation.py` wrapping `audio-separator` CLI with Demucs fallback | M3 | R3 |
| 10 | Vocal Stem Caching | Cache isolated vocal audio under `.build/vocals/<source_id>/vocals.wav` with hash invalidation | M3 | R3 |
| 11 | Vocal Isolation CLI Option | Add `--isolate-vocals` to `run_mixer.py` (`mix`, `preprocess`, `auto`) | M3 | R3 |
| 12 | Clean Speech Alignment Feeding | Feed isolated vocal stems into `vtt_preprocessor.py` chunking and MFA forced alignment | M3 | R3 |
| 13 | VTT Rolling Deduplication Fix | Fix `vtt_preprocessor.py` to properly strip rolling subtitle carry-over lines regardless of spacer cues | M4 | R4 |
| 14 | Crossfade Ramp Implementation | Fix `video_assembler.py` lines 122-126 where `np.linspace` crossfade ramps were unassigned | M4 | R4 |
| 15 | Video PTS & Frame Count Alignment | Format PTS with 6 decimal places and bound video frame count to prevent visual overrun into adjacent words | M4 | R4 |
| 16 | Legacy Sample Rate Skew Fix | Use source native sample rate rather than hardcoded 48kHz in `video_assembler.py` legacy adapter | M4 | R4 |
| 17 | End-to-End Moxy Verification | Verify synthesis of "Anakin, I have the MOXY!" produces exact dialogue with zero "high ground" bleed | M5 | Acceptance |
| 18 | Regression Test Suite Pass | 100% pass across all existing and new automated tests in `tests/` | M5 | Acceptance |

---

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|--------|
| M1 | Independent ASR Listening Verification | Build `tools/listen_verify.py`, integrate into `tools/validate_mix.py` and `tools/evaluate.py`, unit test in `tests/test_listen_verify.py` | None | DONE |
| M2 | Output Audio Loudness Normalization | Implement master EBU R128 (-16 LUFS, -1.0 dBTP) & segment leveling in `src/video_assembler.py`, `--normalize-loudness` in `run_mixer.py`, loudness check in `tools/validate_mix.py`, unit tests | None | DONE |
| M3 | Vocal Separation Pre-processing | Implement `src/vocal_isolation.py` (audio-separator + Demucs fallback), `.build/vocals/` cache, `--isolate-vocals` CLI, integration with MFA alignment | None | DONE |
| M4 | Cut Accuracy & Bleed Elimination | Fix VTT rolling caption deduplication in `src/vtt_preprocessor.py`, crossfade ramps in `src/video_assembler.py`, frame count bounds in `src/edit_plan.py`, PTS formatting, and sample rate skew | None | DONE |
| M5 | Full Acceptance & Regression Verification | Re-align/re-build test units, run full test suite, verify "Anakin, I have the MOXY!", run `tools/evaluate.py`, `tools/validate_mix.py`, and `tools/listen_verify.py` | M1, M2, M3, M4 | IN_PROGRESS |

---

## Interface Contracts

### M1: `tools/listen_verify.py` CLI & Module Contract
- **CLI**:
  ```bash
  python tools/listen_verify.py --input <path_to_wav_or_mp4> --expected "<target_text>" [--model whisper-tiny.en] [--tolerance 0.0] [--json]
  ```
- **Return Code**: `0` on match within tolerance, `1` on mismatch / bleeding / missing critical words.
- **JSON Output Format**:
  ```json
  {
    "input_path": "output/moxy.mp4",
    "expected_text": "Anakin, I have the MOXY!",
    "transcribed_text": "Anakin, I have the MOXY!",
    "normalized_expected": "anakin i have the moxy",
    "normalized_transcribed": "anakin i have the moxy",
    "wer": 0.0,
    "cer": 0.0,
    "word_matches": ["anakin", "i", "have", "the", "moxy"],
    "missing_words": [],
    "bleeding_words": [],
    "passed": true
  }
  ```
- **Function Contract**:
  ```python
  def verify_listening(
      input_path: str | Path,
      expected_text: str,
      model_name: str = "openai/whisper-tiny.en",
      backend: str = "auto",
      wer_threshold: float = 0.20,
      disallow_insertions: bool = True
  ) -> dict:
      ...
  ```

### M2: Loudness Normalization Contract
- **CLI**: `--normalize-loudness` / `--no-normalize-loudness` in `run_mixer.py` (defaults to `True`).
- **Function Contract in `src/video_assembler.py`**:
  ```python
  def normalize_audio_ebur128(
      input_wav: Path,
      output_wav: Path,
      target_i: float = -16.0,
      target_tp: float = -1.0,
      target_lra: float = 11.0,
      sample_rate: int = 48000
  ) -> dict:
      """Applies two-pass FFmpeg loudnorm filter preserving exact sample count."""
  ```
- **Acceptance Metrics**:
  - Integrated Loudness: $-16.0 \pm 1.0\text{ LUFS}$
  - Maximum True Peak: $\le -1.0\text{ dBTP}$
  - Sample Count Delta: $\Delta = 0$ samples vs `plan.total_output_samples`.

### M3: Vocal Isolation Contract
- **CLI**: `--isolate-vocals` in `run_mixer.py` (`mix`, `preprocess`, `auto`).
- **Storage**: `.build/vocals/<source_id>/vocals.wav`.
- **Function Contract in `src/vocal_isolation.py`**:
  ```python
  def isolate_vocals(
      source_video_or_audio: Path,
      output_dir: Path = Path(".build/vocals"),
      model_name: str = "UVR-MDX-NET-Inst_HQ_4.onnx",
      force: bool = False
  ) -> Path:
      """Separates vocal stem using audio-separator CLI with Demucs fallback, returns cached vocals.wav."""
  ```

### M4: Cut Accuracy & Bleed Elimination Contract
- `src/vtt_preprocessor.py`:
  - `clean_rolling_cues(cues: list[dict]) -> list[dict]`: Strips repeated text from consecutive rolling cues if cue time gap is $< 0.5\text{s}$ or overlapping.
- `src/video_assembler.py`:
  - `render_master_pcm`: Applies computed `fade_out_ramp` and `fade_in_ramp` to overlapping PCM buffers:
    ```python
    fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
    master_audio[overlap_start:overlap_end] = (
        prev_tail * fade_out + next_head * fade_in
    )
    ```
  - PTS formatting uses `f"{start_pts:.6f}"` for microsecond precision.
- `src/edit_plan.py`:
  - Video frame count is strictly bounded by max candidate source duration to eliminate visual overrun into neighboring words.

---

## Code Layout
- `src/`
  - `contracts.py`: Dataclass models and schemas.
  - `vtt_preprocessor.py`: VTT parsing, rolling caption cleanup, audio chunking.
  - `vocal_isolation.py`: UVR5 / Demucs vocal separation pipeline.
  - `inventory_builder.py`: TextGrid parsing and units inventory generation.
  - `span_selector.py`: Viterbi unit selection algorithm.
  - `edit_plan.py`: Audio/video timeline and frame scheduling.
  - `video_assembler.py`: Audio slicing, crossfading, loudness normalization, video multiplexing.
- `tools/`
  - `listen_verify.py`: Independent ASR listening verification tool.
  - `validate_mix.py`: Acceptance validation gate for rendered mixes.
  - `evaluate.py`: Multi-test evaluation suite runner.
- `tests/`
  - `test_listen_verify.py`: Unit tests for ASR verification and bleed rejection.
  - `test_loudness.py`: Unit tests for EBU R128 loudness normalization and sample count preservation.
  - `test_vocal_isolation.py`: Unit tests for vocal separation, caching, and fallback.
  - `test_cut_accuracy.py`: Unit tests for VTT deduplication, crossfade ramps, and PTS alignment.
