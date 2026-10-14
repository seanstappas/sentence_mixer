# Accompanying Implementation Guide: 10_14 Improvements & Agent Guardrails

This companion guide complements [`10_14_improvements.md`](file:///d:/Code/sentence_mixer/plans/10_14_improvements.md) by providing exact code patterns, mock-safe fallback invariants, and edge case resolutions to ensure flawless execution by any autonomous agent.

---

## 1. Mock-Safe Boundary Refinement Invariant (`build_edit_plan`)

### Problem
In `tests/test_edit_plan.py` and other unit tests, `SourceRecord` objects often reference dummy/mock paths such as `"video.mp4"` or `"dummy.wav"` that do not exist on disk.
If `build_edit_plan` unconditionally tries to open audio files with `sf.SoundFile` or `_open_source_audio`, tests will fail with `FileNotFoundError`.

### Solution Pattern
In `src/edit_plan.py`, wrap native sample boundary refinement in a file-presence guard:

```python
def _refine_spans_in_native_audio(
    selected_spans: List[SpanCandidate],
    inventory: InventoryIndex,
    sr: int = 48000,
) -> Tuple[List[int], List[int]]:
    """
    Refines boundary sample offsets in native source audio prior to EditPlan construction.
    Gracefully skips refinement if source media or cached PCM files do not exist on disk (mock-safe).
    """
    span_starts = [s.start_sample for s in selected_spans]
    span_ends = [s.end_sample for s in selected_spans]

    # Check if all referenced source audio files exist
    can_refine = True
    for span in selected_spans:
        src_rec = inventory.sources.get(span.source_content_id)
        if not src_rec:
            can_refine = False
            break
        pcm_path = src_rec.cached_pcm_path or src_rec.source_path
        if not pcm_path or not Path(pcm_path).exists():
            can_refine = False
            break

    if not can_refine:
        return span_starts, span_ends

    # Perform two-sided refinement on real audio...
    ...
    return span_starts, span_ends
```

---

## 2. Minor Character Voice Check Calibration (`movie_ingest.py`)

### Problem
Adversarial tests in `tests/test_p4_adversarial.py` enforce:
1. `test_minor_character_inconsistent_evidence_protected`: A character with 2 lines where line 1 has `sim=0.51` (Tobin) and line 2 has `sim=0.49` must **not** be flagged `voice_mismatch`.
2. `test_minor_character_split_across_different_majors`: A character with 2 lines where line 1 matches Tobin and line 2 matches Mira must **not** be flagged `voice_mismatch`.

### Solution Pattern
In `src/movie_ingest.py:378-395`:

```python
else:
    # Minor character (< MIN_CENTROID_SEGMENTS lines) evaluated against major-character centroids
    spk_lines = [rec for rec in named if rec.label == r.label]
    req_margin = max(threshold, MIN_CONFIDENT_MARGIN) if threshold is not None else MIN_CONFIDENT_MARGIN

    # For 1-line characters: flag if best matches a major character with high similarity and margin
    if len(spk_lines) <= 1:
        strong = (
            best != r.label
            and threshold is not None
            and best_sim >= MIN_CONFIDENT_SIMILARITY
            and margin >= req_margin
        )
    else:
        # For multi-line minor characters (2-4 lines):
        # Require that ALL confident lines point to the SAME major character,
        # AND that the line is closer to that major character than to own lines by MISMATCH_MARGIN.
        confident_matches = [
            getattr(rec, "voice_best", None)
            for rec in spk_lines
            if getattr(rec, "voice_sim", 0.0) >= MIN_CONFIDENT_SIMILARITY
            and getattr(rec, "voice_margin", 0.0) >= req_margin
        ]
        unanimous_major = len(confident_matches) == len(spk_lines) and len(set(confident_matches)) == 1 and confident_matches[0] == best
        beats_own = (best_sim - own_sim) >= MISMATCH_MARGIN

        strong = (
            best != r.label
            and threshold is not None
            and best_sim >= MIN_CONFIDENT_SIMILARITY
            and margin >= req_margin
            and unanimous_major
            and beats_own
        )
```

---

## 3. Backwards Compatibility for Cached NPZ Files (`features.py`)

When loading `.features.npz` or `.span_features.npz` files that may have been generated prior to adding `delta_mfcc_13`, always guard array indexing:

```python
if "delta_mfcc" in tracks:
    delta_mfcc = [float(x) for x in tracks["delta_mfcc"][:, frame_idx]]
elif "mfcc" in tracks and tracks["mfcc"].shape[1] >= 2:
    # Compute on the fly if missing from old cache
    delta_mfcc = [float(x) for x in (tracks["mfcc"][:, min(end_frame, tracks["mfcc"].shape[1] - 1)] - tracks["mfcc"][:, max(0, start_frame)])]
else:
    delta_mfcc = [0.0] * 13
```

---

## 4. Preserving Repeated Whole Words in Subtitles (`dialogue_mixer.py`)

In `_extract_plan_words`:
```python
is_subword_or_phone = s.span_type in ("subword", "phone")
can_merge = is_subword_or_phone and (prev_span_type in ("subword", "phone"))

if " " in lbl:
    ...
elif lbl == curr_word and can_merge:
    curr_t1 = max(curr_t1, t1)
else:
    if curr_word:
        words.append((curr_word, time_offset_sec + curr_t0, time_offset_sec + curr_t1))
    curr_word = lbl
    curr_t0 = t0
    curr_t1 = t1
prev_span_type = s.span_type
```
This ensures repeated words like `"no no"` or `"that that"` are never accidentally collapsed into a single word.
