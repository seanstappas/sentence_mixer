### Issues & Edge Cases Identified

#### 1. Infinite Cache Invalidation Loop with Vocal Isolation

In `run_mixer.py` (`ensure_source_aligned`):

```python
# 4. Check if vocal isolation is requested but stem is missing from disk or inventory
if not is_stale and isolate_vocals:
    vocals_dir = CACHE_BASE_DIR / "vocals" / source_id
    stem_path = vocals_dir / "vocals.wav"
    if not stem_path.exists() or stem_path.stat().st_size == 0:
        is_stale = True
    else:
        try:
            with open(inventory_path, "r", encoding="utf-8") as f:
                inv_data = json.load(f)
            srcs = inv_data.get("sources", {})
            if not any(s.get("cached_vocal_pcm_path") for s in srcs.values()):
                is_stale = True
        except Exception:
            is_stale = True

```

* **The Problem:** If `inventory_builder.py` does not explicitly serialize `cached_vocal_pcm_path` into the `sources` dictionary of `units_index.json`, `any(s.get("cached_vocal_pcm_path") ...)` evaluates to `False` every single time.
* **Impact:** Every run with `--isolate-vocals` (or `--audio-track isolated/recombined`) marks the cache as stale, deleting the corpus and re-running MFA from scratch on every invocation.

#### 2. Non-Recursive Globbing in MFA Batching

In `run_mixer.py` (`run_mfa_align`):

```python
wav_files = sorted(list(c_path.glob("chunk_*.wav")))
...
for tg in b_aligned.glob("*.TextGrid"):
    shutil.move(str(tg), str(out_aligned / tg.name))

```

* **The Problem:** Both `glob` calls are flat (non-recursive). If `preprocess_source` nests chunks inside a speaker folder (e.g., `corpus/speaker_name/chunk_001.wav`), `wav_files` will be empty (`[]`), causing alignment to be skipped or fail. Similarly, when MFA outputs TextGrids into speaker subfolders, `b_aligned.glob("*.TextGrid")` will not discover them.
* **Fix:** Change `glob` to `rglob`:
```python
wav_files = sorted(list(c_path.rglob("chunk_*.wav")))
...
for tg in b_aligned.rglob("*.TextGrid"):
    shutil.move(str(tg), str(out_aligned / tg.name))

```



#### 3. Metadata Disconnect Between `--audio-track` and `plan.metadata`

In `run_mixer.py`:

```python
audio_track = getattr(args, "audio_track", "combined")
isolate_vocals_flag = getattr(args, "isolate_vocals", False) or (audio_track in ("isolated", "recombined", "both", "all"))
...
plan.metadata["isolate_vocals"] = getattr(args, "isolate_vocals", False)
plan.metadata["audio_track"] = getattr(args, "audio_track", "combined")

```

* **The Problem:** If a user runs `run_mixer.py mix --text "..." --audio-track isolated` (without explicitly typing `--isolate-vocals`), `isolate_vocals_flag` correctly evaluates to `True` during alignment, but `plan.metadata["isolate_vocals"]` is stored as `False`.
* **Impact:** If `video_assembler.py` checks `plan.metadata.get("isolate_vocals")` to determine whether to fetch the isolated stem, it will mistakenly bypass vocal stems.
* **Fix:** Store the effective flag:
```python
plan.metadata["isolate_vocals"] = isolate_vocals_flag

```



#### 4. Silent Override of User-Specified `--speaker`

In `run_mixer.py`:

```python
unique_speakers = {wo.speaker_id for wo in inventory_index.words.values() if wo.speaker_id}
if len(unique_speakers) == 1:
    effective_speaker = next(iter(unique_speakers))
else:
    effective_speaker = args.speaker

```

* **The Problem:** If an inventory contains exactly one speaker (`speakerA`), but the user explicitly runs `run_mixer.py mix --speaker speakerB`, `effective_speaker` silently overwrites the user's flag to `speakerA` instead of warning or filtering.
* **Fix:** Only default to the single speaker if `args.speaker is None`:
```python
if args.speaker is not None:
    effective_speaker = args.speaker
elif len(unique_speakers) == 1:
    effective_speaker = next(iter(unique_speakers))
else:
    effective_speaker = None

```



#### 5. Dead Expression Statement

At line 144 of `run_mixer.py`:

```python
source_id = get_source_id(str(v_abs))
CACHE_BASE_DIR / "sources" / source_id  # <--- Dead expression
corpus_dir = CACHE_BASE_DIR / "alignment" / source_id / "corpus"

```

This line evaluates a `Path` object without assigning it or creating the directory (`.mkdir(parents=True, exist_ok=True)`).