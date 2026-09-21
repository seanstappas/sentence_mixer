# E2E Test Infra: sentence_mixer

## Test Philosophy
- Opaque-box, requirement-driven. Derived from ORIGINAL_REQUEST.md.
- Methodology: Category-Partition + Boundary Value Analysis + Pairwise Combinatorial + Real-World Workload Testing.

## Feature Inventory & Test Coverage
| # | Feature | Source (Requirement) | Tier 1 (Feature) | Tier 2 (Boundary) | Tier 3 (Cross-Feature) | Tier 4 (Real-World) |
|---|---------|----------------------|:----------------:|:-----------------:|:----------------------:|:-------------------:|
| F1 | Standalone ASR Listener (`tools/listen_verify.py`) | R1 (Lines 12–18) | 5 | 5 | ✓ | ✓ |
| F2 | ASR Gate Integration (`evaluate.py`, `validate_mix.py`) | R1 (Lines 17, 40–44) | 5 | 5 | ✓ | ✓ |
| F3 | Insertion Bleed Detection & Nonzero Exit | R1 (Lines 15–16) | 5 | 5 | ✓ | ✓ |
| F4 | Master EBU R128 Normalization (-16 LUFS, -1.0 dBTP) | R2 (Lines 19–25) | 5 | 5 | ✓ | ✓ |
| F5 | Sample Count Invariant Preservation (-ar 48000) | R2 (Lines 21–23, 46) | 5 | 5 | ✓ | ✓ |
| F6 | Normalization CLI Flag (`--normalize-loudness`) | R2 (Lines 24, 46–48) | 5 | 5 | ✓ | ✓ |
| F7 | Vocal Isolation Pre-processing (`src/vocal_isolation.py`) | R3 (Lines 26–32) | 5 | 5 | ✓ | ✓ |
| F8 | Vocal Stem Caching under `.build/vocals/` | R3 (Lines 31, 50) | 5 | 5 | ✓ | ✓ |
| F9 | Pre-processing CLI Flag (`--isolate-vocals`) | R3 (Lines 29, 50–52) | 5 | 5 | ✓ | ✓ |
| F10 | VTT Rolling Caption Deduplication | R4 (Lines 33–37) | 5 | 5 | ✓ | ✓ |
| F11 | Crossfade Envelope Ramp Implementation | R4 (Lines 34–36, 54) | 5 | 5 | ✓ | ✓ |
| F12 | Substring Cut Accuracy ("Anakin, I have the MOXY!") | R4 (Lines 35, 54–55) | 5 | 5 | ✓ | ✓ |

## Test Architecture
- **Test Runner**: `.venv\Scripts\pytest.exe -v`
- **Standalone Verification Tools**:
  - `python tools/listen_verify.py --input <path> --expected "<text>"`
  - `python tools/validate_mix.py --render <mp4_path> --plan <plan_path> [--require-asr]`
  - `python tools/evaluate.py --manifest evaluation/smoke_sources.json --test-suite evaluation/smoke_suite.json`
- **Pass/Fail Semantics**:
  - All pytest tests exit with code 0.
  - `listen_verify.py` exits 0 on match, 1 on divergence or bleed.
  - `validate_mix.py` exits 0 on container, sample, loudness, and ASR compliance.

## Real-World Application Scenarios (Tier 4)
| # | Scenario | Features Exercised | Target Text | Acceptance Criterion |
|---|----------|--------------------|-------------|----------------------|
| 1 | Moxy Synthesis ("Anakin, I have the MOXY!") | F1, F3, F10, F11, F12 | "Anakin, I have the MOXY!" | Exact match via ASR, zero "high ground" bleed |
| 2 | Loudness Balance across Multi-Clip Scenes | F4, F5, F6 | Spliced dialogue from Ep2 & Ep3 | -16.0 +/- 1.0 LUFS, true peak <= -1.0 dBTP, diff = 0 samples |
| 3 | Ingestion with Vocal Isolation | F7, F8, F9, F10 | Full ingestion & alignment | Clean vocal stem cached in `.build/vocals/`, valid TextGrids |
| 4 | Independent Acceptance Evaluation Suite | F1, F2, F3, F4, F6, F12 | `evaluation/smoke_suite.json` | All test cases pass automated ASR verification and reporting |
| 5 | Dual Render Pipeline Validation (Basic & WavLM) | F1, F2, F5, F12 | "Hello there." & "Anakin, I have the MOXY!" | `validate_mix.py` passes cleanly for both span and WavLM selector |

## Coverage Thresholds
- Tier 1: ≥5 per feature (60 tests)
- Tier 2: ≥5 per feature (60 tests)
- Tier 3: Pairwise coverage across feature interactions (≥12 tests)
- Tier 4: ≥5 realistic application scenarios
