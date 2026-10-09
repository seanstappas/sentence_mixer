# Feature Plan: Full-Movie Sources, Speaker Labeling, and Scene Ranges

Date: 2026-10-04.
Base: commit `cca1595c` plus the uncommitted 2026-10-04 changes (opt-in Whisper tail
enrichment, `.gitignore` update, and the labeled-subtitle tool and outputs described in
section 2A). Companion to the
[round-4 remediation plan](10_04_antigravity_remediation_plan.md).

## 1. Goals

Make whole movies first-class sources, starting with the two Blu-ray rips in `input/`:

- `Star.Wars.Episode.II.-.Attack.Of.The.Clones...mkv`
- `Star.Wars.Episode.III.-.Revenge.Of.The.Sith...mkv`

Requested capabilities:

1. **Official subtitles:** use the subtitle tracks embedded in the MKV, no sidecar file needed.
2. **Speakers for the whole movie:** label who says each line, using subtitle labels,
   user-supplied transcripts/screenplays, and voice matching, with the CLI asking the
   user only about what remains uncertain.
3. **Subsections:** ingest and mix a time range of a movie (for example one scene)
   instead of the whole file; later, named scenes.

Product rules still apply: output speech comes only from the source recordings; never
modify files under `input/`; analysis models are analysis-only; no commits unless asked.

## 2. What the inputs actually look like (measured 2026-10-04)

| Property | Episode II | Episode III |
| --- | --- | --- |
| File size, duration | 6.46 GB, 2:22:30 | 6.35 GB, 2:20:08 |
| Video | HEVC 10-bit, 3840x1608 | same |
| Audio | AAC 5.1 (6 channels), one track | same |
| Text subtitle tracks | `English-SRT` (1,185 cues), `English-SDH-SRT` (1,274 cues) | `English-SRT` (1,187), `English-SDH-SRT` (1,396) |
| Image subtitle tracks | 4 x PGS (need OCR; out of scope) | same |
| Chapters | none | none |
| `Name:` speaker labels (SDH) | 68 lines (~5%) | 62 lines (~4%) |
| Two-speaker cues (dialogue dashes) | 44 | 84 |
| Sound/manner tags (SDH) | 114 cues, e.g. `[GRUNTS]`, `[R2-D2 BEEPING]`, `[SPEAKING HUTTESE]` | 281 cues |

Observations that drive the design:

- The SDH track is the better primary text: more complete, carries the few speaker
  labels, and marks non-dialogue sounds. Labels are mixed case (`Obi-Wan:`), and some
  contain OCR errors (`C-3p0o`, stray characters), so names need fuzzy normalization.
- Labels cover only ~5% of lines, so most labels must come from transcripts/screenplays
  and voice matching.
- 5.1 audio: film dialogue is mostly mixed into the **center channel**. Using the center
  channel for analysis and alignment gives cleaner speech than a stereo downmix, often
  without needing vocal separation at all.
- The fandom transcript site blocks automated fetching (HTTP 403). The IMSDb screenplays
  are earlier drafts (scene headings in the `INT./EXT. <LOCATION> - <TIME>` form,
  character names above indented dialogue), so their wording differs from the final film in places.
  Both sources are copyrighted: they must stay as local, user-provided files and never
  be committed or reproduced in outputs.

### Current code limits for 6 GB / 2.4-hour sources

| Area | Problem | Where |
| --- | --- | --- |
| Rendering | Decodes the **entire** source audio into memory on first use (the AAC track is small in the file, but decoded 6 ch x 48 kHz x 2.4 h is about 4.5 GB as 16-bit PCM and about 18 GB as the float64 array `sf.read` returns), and reads whole vocal/instrumental stems with `sf.read` | [video_assembler.py](../src/video_assembler.py), `render_master_pcm` (around lines 286-305 and 396-430) |
| Vocal isolation | Pre-extracts all 6 channels at 44.1 kHz to one WAV (~4.5 GB, beyond the 4 GB WAV limit) and separates the whole movie in one job | [vocal_isolation.py](../src/vocal_isolation.py), around line 349 |
| Diarization | Reads the whole audio file into memory and runs energy VAD over music and effects | [diarization.py](../src/diarization.py), `diarize` |
| Caching | One alignment/inventory per source file, so any range work invalidates or redoes the whole movie | [run_mixer.py](../run_mixer.py), `ensure_source_aligned` |
| Subtitles | Only sidecar `.vtt`/`.srt` files are discovered; embedded tracks are ignored | [source_discovery.py](../src/source_discovery.py) |
| Video | Each rendered clip decodes 4K HEVC 10-bit from the original (slow but correct) | [video_assembler.py](../src/video_assembler.py), slice rendering |

Full-file SHA-256 hashing of a 6 GB file takes about a minute once and is memoized by
size and mtime ([cache.py](../src/cache.py), `compute_media_hash`), so it is acceptable.

## 2A. Done on 2026-10-04: speaker-labeled subtitles for both movies

The labeling step (most of M03 parsing and the M05 alignment, as an offline tool) has
been done and run. Antigravity should build the ingest side on top of these files.

**New files (tracked, no copyrighted text):**

- [tools/label_subtitles.py](../tools/label_subtitles.py): extracts the embedded English
  SDH text track (falls back to plain English; `--subtitles` accepts a sidecar file),
  parses cues robustly, splits dialogue-dash cues into per-speaker lines (times split by
  text length), reads `Name:` labels, flags sound-only and non-English lines, aligns
  subtitle words to transcripts (unique 4-gram anchors plus local matching between
  anchors), and uses screenplays only as a fallback for lines still unlabeled. Labels
  come from subtitle labels first, then transcripts, then screenplays. Disagreements
  leave a line unlabeled, except that a named character beats a generic role (`Man`,
  `Clone Trooper`) from the other source. Prints counts and names only.
- [config/characters/star_wars.json](../config/characters/star_wars.json): speaker IDs,
  display names, aliases (including OCR and spelling variants), generic roles
  (`role-*`), group labels, and ignore patterns. IDs reuse existing registry IDs
  (`organa`, `amedda`, `aak`, `palpatine`).
- [tests/test_label_subtitles.py](../tests/test_label_subtitles.py): 6 tests with original
  synthetic dialogue (parsing, dash splitting, labels/tags, alias and OCR resolution,
  stream selection, script parsers, end-to-end VTT and report).

**Generated outputs (git-ignored, local only, contain subtitle text; never commit):**

- `.build/subtitles/labeled/<movie file stem>.en.vtt`: WebVTT with the movie's own cue
  timing and text. Labeled lines carry `<v speaker-id>...</v>`; unlabeled lines have no
  tag. Sound-only, non-English, and zero-duration cues are omitted. `Name:` prefixes are
  removed from the text.
- `.build/subtitles/labeled/<movie file stem>.en.vtt.report.json`: per-line start/end,
  speaker, source, confidence, evidence (votes and coverage per script), and a SHA-1 of
  the line text (no text), plus summary and per-speaker totals.
- Inputs used: `input/ep2_transcript.txt`, `input/ep3_transcript.txt` (user-saved), and
  IMSDb screenplays downloaded to `.build/scripts/ep2_imsdb.html` and `ep3_imsdb.html`
  (git-ignored). Extracted subtitle tracks are in `.build/subtitles/source/`.

Regenerate (about 2 seconds per movie):

```powershell
Set-Location 'D:\Code\sentence_mixer'
$Movie = (Get-ChildItem -LiteralPath .\input -Filter 'Star.Wars.Episode.III.-*.mkv').FullName
$Out = Join-Path (Get-Location) ('.build\subtitles\labeled\' + [IO.Path]::GetFileNameWithoutExtension($Movie) + '.en.vtt')
& .\.venv\Scripts\python.exe tools\label_subtitles.py --video $Movie --transcript input\ep3_transcript.txt --screenplay .build\scripts\ep3_imsdb.html --output $Out
```

**Results:**

| | Episode II | Episode III |
| --- | --- | --- |
| Subtitle track used | English-SDH-SRT (stream 3) | English-SDH-SRT (stream 3) |
| Cues / dialogue lines (after dash splits) | 1,274 / 1,222 | 1,397 / 1,226 |
| Labeled dialogue lines | **876 (71.7%)** | **1,151 (93.9%)** |
| From subtitle labels | 65 (24 confirmed by transcript, 41 subtitle-only) | 58 (36 confirmed, 17 subtitle-only, 5 role-vs-name resolved) |
| From transcript | 581 | 1,091 |
| From screenplay fallback | 230 | 2 |
| Unlabeled (no evidence or conflict) | 344 | 75 |
| Omitted: sound-only / non-English / zero-duration | 96 / 0 / 0 | 253 / 1 / 1 |
| Named speakers / generic roles | 29 / 5 | 26 / 10 |
| Largest speakers (lines / seconds) | anakin 221/527, obi-wan 141/318, padme 132/314, dooku 62/174, yoda 49/156, palpatine 42/126 | anakin 275/667, obi-wan 214/534, palpatine 203/645, padme 102/257, yoda 73/254, mace-windu 50/137 |
| Screenplay scene headings parsed (for M09) | 149 | 236 |

**Accuracy evidence (independent checks):**

- Official subtitle labels vs transcript alignment, on lines with both: Episode II 24 of
  24 agree; Episode III 36 of 41 agree, and 4 of the 5 differences are a generic role
  versus a named character (now resolved to the named one); 1 true conflict stays
  unlabeled.
- Screenplay vote vs trusted labels, on lines with both: 97.1% agreement (311 lines,
  Episode II) and 99.6% (1,121 lines, Episode III). Most differences are again role
  versus named character.
- The existing pipeline parses both VTTs with no tags or `Name:` prefixes left in the
  alignment text and no empty cues (1,184 and 1,185 cues).

**Known limitations to carry into M06/M07:**

- Episode II's transcript names speakers for only part of the film, so 344 lines remain
  unlabeled and 230 rely on the screenplay fallback (lower confidence in the report).
  Voice matching (M06) and review (M07) should focus there.
- Line times inside dialogue-dash cues are split by text length; MFA word times must
  replace them before building speaker intervals.
- `sidious` and `palpatine`, and `vader` and `anakin`, are separate IDs on purpose (see
  the character file notes). `role-*` labels are not single voice identities.
- Labels are not yet verified against the audio; a transcript can attribute a line the
  film gives to someone else.

## 2B. Implemented on 2026-10-04 (second pass): full-movie ingest and mixing

Both movies are ingested end to end and can be mixed by any labeled speaker, whole
movie, time range, or scene, rendered in 4K. Work packages:

| Package | Status | Where |
| --- | --- | --- |
| M01 large files | Done: FLAC staging (stereo 48 kHz program + 16 kHz center-channel analysis), 10-minute block vocal separation stitched into full-length FLAC stems (resumable), ranged reads everywhere (`LazyAudio` in the renderer, ranged gain reads in `edit_plan`), join features precomputed over speech spans only (parallel, incremental). Not done: `--proxy-video` (not needed: 4K slices cut in about 1 s each), `--isolation-input` (separation uses the stereo program). | [movie_media.py](../src/movie_media.py), [video_assembler.py](../src/video_assembler.py), [features.py](../src/features.py) |
| M02 embedded subtitles | Done: text-track listing, auto selection (English SDH, then plain, then any text), `--subtitle-stream`, extraction to `.build/subtitles/<hash>/`. Order: `--vtt`, labeled VTT, sidecar, embedded. | [movie_media.py](../src/movie_media.py), [movie_ingest.py](../src/movie_ingest.py) |
| M03 parsing | Done: parsing primitives moved to [subtitle_parsing.py](../src/subtitle_parsing.py) (the tool imports them); `<v>` tags read as speakers; cue padding (0.15 s before, 0.25 s after, never into the neighboring cue); skipped/zero-duration/sound-only/non-English counts reported. Not done: SDH-vs-plain OCR typo check (step 8). | [movie_subtitles.py](../src/movie_subtitles.py) |
| M05 integration | Done: scenes exported by the tool (`scenes` in the report: 44 for Episode II, 90 for Episode III). The tool stays the offline labeling step. | [label_subtitles.py](../tools/label_subtitles.py) |
| M06 voice | Done: WavLM-SV embeddings on the center channel per line (>= 1 s); per-speaker trimmed centroids; leave-one-out calibration; labeled lines are made ineligible only when the voice is >= 0.10 closer to another character, that character leads all others by the calibrated margin (>= 95% leave-one-out precision, at least 0.05), and the labeled character has >= 5 other lines; otherwise the text label stands (`label_voice_unclear`). Unlabeled lines are labeled only above the calibrated margin. Revised 2026-10-05 after a false flag (a Gunray line flagged as Mace Windu with a 0.01 lead and a 4-line Gunray profile). The separated vocal stem was tested as the embedding source and was slightly worse (leave-one-out 0.85/0.70 vs 0.86/0.73), so embeddings stay on the center channel. Not done: clustering leftover unknown voices into anonymous groups. | [movie_ingest.py](../src/movie_ingest.py) |
| M07 label store | Done: `.build/movies/<hash>/speaker_labels.json` (per line: times, label, status, voice scores, text hash; no text), `review-speakers` (plays each line, Enter accepts suggestion), overrides in `config/speaker_labels/<hash>.json`, `speakers` command. Words take the speaker of their subtitle line by word order (exact for dash cues). | [movie_commands.py](../src/movie_commands.py), [inventory_builder.py](../src/inventory_builder.py) |
| M08 ranges | Done: `--range` on ingest (aligns only overlapping blocks) and mix (out-of-range words made ineligible), block-level alignment cache. | [run_mixer.py](../run_mixer.py) |
| M09 scenes | Done: `scenes` command, `--scene <number or keyword>`. | [run_mixer.py](../run_mixer.py) |
| Rendering | Done: `--resolution` (1080p, 1440p, 4k, source, or WxH) (letterboxed, lanczos, 8-bit BT.709 H.264); `--fps auto` (default) uses the sources' frame rate (23.976 for the films). | [video_assembler.py](../src/video_assembler.py) |
| M10 docs | Done: README "Whole Movies", option tables, SKILL.md, GEMINI.md. | |
| Dialogue isolation first | Done 2026-10-05: all movie analysis (MFA alignment, voice checks, join features, selector F0 reads) reads `audio/analysis_vocals_16k.flac`, built from the separated vocal stem; the raw center channel is only a fallback for blocks without separated vocals. Experiment (Episode III block 4, same chunks and transcripts, ASR recall of words cut at each alignment's times): separated stereo vocals 0.77 / 0.77 (single words / 3-word windows), raw center channel 0.72 / 0.76, separated center channel 0.73 / 0.74; boundaries differ by a median of 5 ms. Separated vocals scored slightly lower as a voice-check input in an earlier comparison (leave-one-out 0.85/0.70 vs 0.86/0.73) but are used for consistency; the stricter flag guards limit the effect. Chunks MFA cannot align from the vocals are retried once from the center channel (5 in Episode III, 9 in Episode II, all recovered), so every chunk now aligns (1185/1185 and 1184/1184; previously 1182/1185). Of the 14 fallback chunks, 12 had normal vocal levels (MFA failed on fast or condensed lines); 2 had lost the voice in separation. A per-line check now marks lines whose vocals are >15 dB below the movie's typical vocals-vs-center level as `separation_dropped` (excluded; words would render silent from the vocal stem): 1 line in Episode II, 4 in Episode III, including 3 that MFA had force-aligned to near-silent vocals. Re-rendering the 11 test sentences: mean whisper-tiny WER 0.45 -> 0.29 (6 better, 1 same, 4 slightly worse); broad window check unchanged (word recall 0.80 Episode III, 0.82 Episode II). Output audio is unchanged: `recombined` (default) keeps the film's music and effects; `raw` and `vocals` remain options. | [movie_ingest.py](../src/movie_ingest.py), [movie_media.py](../src/movie_media.py) |
| Cache cleanup | Done: `clean <movie>` also removes `.build/movies/<hash>/` and extracted subtitle tracks (keeps labeled VTTs and reviewed labels); `clean --all` covers them too. Fixed a pre-existing matcher bug that also deleted other sources sharing a name prefix (cleaning Episode III removed Episode II; `anakin_ep2_1` removed `anakin_ep2_2`). | [cache.py](../src/cache.py) |

Tests: [test_movie_support.py](../tests/test_movie_support.py) (15 tests, invented dialogue).
Full offline suite after the main changes: 491 passed, 1 skipped, 1 xpassed (known flaky
dictionary-cache test deselected).

Results:

| | Episode II | Episode III |
| --- | --- | --- |
| Blocks aligned (last block is credits) | 14/15 | 14/15 |
| Alignment time (MFA, one call per block) | 31 min | 25 min |
| Vocal separation (CPU, 15 blocks) | 44 min | 47 min |
| Lines: labeled and voice-consistent / unclear (kept) | 777 / 91 | 911 / 233 |
| Lines flagged by voice (excluded; 21 / 42 before the 2026-10-05 guards) | 8 | 6 |
| Unlabeled lines labeled by voice | 152 | 17 |
| Still unlabeled | 194 | 58 |
| Voice leave-one-out accuracy | 0.86 | 0.73 |
| Top usable speech | anakin 382 s, obi-wan 314 s, padme 234 s, yoda 177 s, dooku 155 s | palpatine 475 s, anakin 462 s, obi-wan 402 s, yoda 195 s, padme 170 s |

Alignment quality check (ASR on 40 random 4-word windows cut at MFA word times, base.en):
Episode III word recall 0.79, in line with the existing short clips (0.76 to 0.84).
Mixing time after ingest: about 15 s per sentence in 4K (it was about 140 s before the
join-feature precompute).

Test mixes (4K 3840x2160, 23.976 fps, recombined audio; original sentences; unprimed
whisper-tiny WER from tools/listen_verify.py, in output/movie_tests/):

| Movie | Speaker | Words in the speaker's vocabulary | WER |
| --- | --- | --- | --- |
| II | padme | 6/7 | 0.00 |
| II | dooku | 5/6 | 0.17 |
| II | anakin | 8/8 | 0.38 |
| II | yoda | 5/8 | 0.38 |
| II | obi-wan | 6/8 | 0.75 |
| III | obi-wan | 5/7 | 0.00 |
| III | anakin | 8/9 | 0.56 |
| III | yoda | 4/7 | 0.57 |
| III | mace-windu | 5/6 | 0.83 |
| III | palpatine | mixed | 1.17 |
| II + III | obi-wan (both films pooled) | 9/10 | 0.10 |

High WER mostly comes from words assembled from single phones and from the selector
picking very short instances (8 words in 1.4 s); it is not movie-specific (alignment
accuracy matches the short clips). Improving selection for long inventories (prefer
natural-duration instances, phrase-level prosody) is the next quality step.

Known limitations and next steps:

- Voice checks are moderately reliable on 1-3 s film lines (leave-one-out 0.73 to 0.86);
  flags are conservative (a wrong voice in a mix is worse than a skipped line). Review
  with `run_mixer.py review-speakers <movie> --only flagged`.
- Mix quality depends on the sentence: lines built mostly from words the character says
  sound natural; words built from single phones are often misheard (see the realism
  guideline).
- Remaining items: OCR typo check (M03 step 8), anonymous grouping of unknown voices
  (M06), proxy video and isolation-input options (M01), center-vs-stereo separation
  comparison.

## 2C. Done on 2026-10-05: quality fixes after user testing

Full handoff (state, measurements, traps, open work): [10_05_handoff.md](10_05_handoff.md).

- Voice checks no longer overrule text labels on weak evidence (false Gunray-as-Mace-Windu
  flag): flags need a >= 0.10 gap, a calibrated lead over every other voice, and >= 5 other
  lines for the labeled speaker. Flags 21 -> 8 (Episode II), 42 -> 7 (Episode III).
- Dialogue is isolated before any analysis (alignment, voice checks, join features read
  `analysis_vocals_16k.flac`); center channel only as fallback, including a retry for chunks
  MFA cannot align from vocals (14 recovered). Both movies re-aligned; 100% of cues aligned.
- Lines whose voice was removed by separation are marked `separation_dropped` (5 lines).
- `review-speakers`: non-blocking playback, `--audio mix|center|vocals`, every answer saved
  immediately (atomic writes), Enter keeps the text label.
- Rendering: `--release-tail-ms` (default 100) extends cuts into following silence so
  word-final consonants are not clipped; `--pause-style freeze|continue|off`; per-segment
  end fades; freeze after the last word now shows the frame after it, not its first frame.

## 2D. Done on 2026-10-06: Episode VI ingest and multi-speaker dialogue scenes

- **Episode VI (*Return of the Jedi*) Ingested**:
  - Embedded English SDH subtitles extracted, transcripts and screenplays aligned.
  - All 13 blocks (131 min) aligned via MFA, stems separated (`vocals.flac` / `instrumental.flac`), join features precomputed, voice checks calibrated.
  - 29 speakers indexed into `.build/inventory/Star_Wars_Episode_VI_-_Return_Of_The_Jed_3d083b52/units_index.json` (5,764 words).
- **Multi-Speaker Dialogue Synthesizer (`src/dialogue_mixer.py`)**:
  - Script parser (`speaker: line`) with hyphen normalization and speaker aliasing (`sidious` -> `palpatine`).
  - Concatenates multiple `EditPlan` timelines with natural conversational pauses (`concat_edit_plans`).
  - Word-by-word highlighted ASS subtitle generation (`generate_highlighted_ass_subtitles`) with active yellow highlight (`&H0000FFFF&`) and intra-card continuity to prevent flickering during pauses.
  - Automated FFmpeg subtitle burning into CFR MP4 deliveries with recombined master audio (-16 LUFS).
- **CLI Scene Runner (`tools/render_dialogue_clips.py`)**:
  - Rendered `output/episode_3_anakin_mace.mp4` (26.9s, 1080p, 14.1 MB).
  - Rendered `output/episode_6_luke_palpatine_vader.mp4` (22.3s, 1080p, 9.8 MB).

## 3. Design overview

```text
movie.mkv
  -> probe (streams, duration)            -> .build/movies/<media hash>/probe.json
  -> extract audio once: center channel (analysis) + stereo downmix (output), FLAC
  -> extract text subtitle tracks         -> .build/subtitles/<media hash>/stream_<n>.srt
  -> parse cues into lines (labels, dashes, sound tags, OCR name fixes)
  -> [optional] align user transcripts/screenplays to lines -> speaker + scene candidates
  -> per time block (default 10 min, only blocks overlapping the requested ranges):
       vocal isolation (optional for center channel), MFA alignment, word timings
  -> voice embeddings per line -> verify labels, propagate to unlabeled lines,
     cluster leftovers -> CLI review -> config/speaker_labels/<media hash>.json
  -> labels become verified speaker intervals -> existing per-word eligibility (D5)
  -> mix with --range / --scene filters candidate words by source time
```

Key choices:

1. **Block-based processing.** The movie timeline is split into fixed blocks (default
   600 s). Isolation, chunk extraction, MFA, and inventory building are cached per block,
   keyed by media hash, block bounds, and the usual configuration. A range ingest
   processes only overlapping blocks; ingesting the whole movie later reuses them. The
   movie's inventory is the union of its ingested blocks.
2. **Speaker labels are data, not inference at mix time.** All label evidence (subtitle
   labels, script alignment, voice) is combined once into a reviewable label file. Mixes
   read the file. This plugs into the existing verified `speaker_intervals` path, so the
   per-word eligibility rule (whole-word coverage, 40 ms collar) keeps working.
3. **No copyrighted text in tracked files.** Label and scene files store times, speaker
   IDs, evidence scores, and a hash of the line text, not the subtitle or script text.
   Review tools read the text from the local subtitle/script files at runtime.

## 4. Work packages

Prerequisites from the round-4 plan: C01 (clean baseline), C04 (diarization caching and
cache-root paths), and C07 (embedding model provenance) should land first or together
with M01, because these packages build on them. C02 (speaker-change detection) remains
useful for sources without transcripts.

### M01. Large-file media foundations

**Why:** the current pipeline loads whole files into memory and writes a >4 GB WAV.

**Fix:**

1. On first use of a long source (default: longer than 20 minutes, or any MKV with
   5.1 audio), extract audio once into `.build/movies/<media hash>/audio/`:
   - `analysis_center_48k.flac`: center channel (`pan=mono|c0=FC`) for 5.1/7.1 layouts,
     or a mono downmix for stereo sources. Used for alignment, embeddings, and analysis.
   - `program_stereo_48k.flac`: standard stereo downmix, used as the `raw` output track
     and as input for background/instrumental separation.
   FLAC avoids the 4 GB WAV limit and keeps files small. Record the channel mapping in
   the source record.
2. Make every audio read sample-ranged: open with `soundfile.SoundFile`, seek, and read
   only the needed frames (plus fade handles). Apply this in `render_master_pcm` for the
   original, vocal, and instrumental roles, and in feature extraction. Remove the
   "decode the whole video audio to a pipe" path for long sources; fall back to
   `ffmpeg -ss <t> -t <d>` extraction of just the needed span if no cached file exists.
3. Vocal isolation per block: separate each block (with 5 s overlap on each side, trimmed
   after) from the cached FLAC, cache stems per block, and stitch only the requested
   spans. Add `--isolation-input center|stereo` (default `center` for 5.1 sources) and
   allow `--no-isolate-vocals` for center-channel sources where separation is not needed.
4. Diarization and VAD must never load the whole movie; they operate on block files or
   on line spans (M06).
5. Optional `--proxy-video`: build a 1080p H.264 proxy per ingested block for faster
   rendering; renders and XML still reference timing from the original. Default off.
6. Print progress and an estimate per block; make all steps resumable (a stopped ingest
   continues from the last finished block).

**Verify:** X01, X02, X03.

### M02. Embedded subtitle extraction and selection

**Fix:**

1. In [source_discovery.py](../src/source_discovery.py), when a video has no sidecar
   subtitle, probe its subtitle streams. Supported: text codecs (`subrip`, `ass`/`ssa`,
   `webvtt`, `mov_text`). Image codecs (`hdmv_pgs_subtitle`, `dvd_subtitle`) are reported
   as unsupported (OCR is out of scope).
2. Selection order for `--subtitle-stream auto`: English SDH text track (title contains
   `SDH`, or `hearing_impaired` disposition), then English non-forced text, then any
   text track. Allow `--subtitle-stream <index>|sdh|plain` and a manifest field.
   Sidecar files still win when present unless `--subtitle-stream` is given.
3. Extract with `ffmpeg -map 0:<index>` to `.build/subtitles/<media hash>/stream_<n>.srt`
   (never next to the movie). Cache by media hash and stream index.
4. Record the chosen stream, codec, title, and cue count in the corpus manifest and
   print them during ingest.

**Verify:** X04, X05.

### M03. Movie subtitle parsing

**Status (2026-10-04):** steps 1-5 are implemented in
[tools/label_subtitles.py](../tools/label_subtitles.py) (`parse_subtitle_text`,
`split_segments`, `CharacterResolver`) as an offline tool. Remaining: move or import that
logic into the pipeline (do not duplicate it), and do steps 6-9 there. The labeled VTTs
already work as `--vtt` sidecars because the current cleaner strips the `<v>` tags.

**Fix:** extend [vtt_preprocessor.py](../src/vtt_preprocessor.py) cue parsing into a
`SubtitleLine` model (cue id, line index, start/end, text, label, flags):

1. **Dialogue dashes:** split a cue whose lines start with `-` into separate lines
   (different speakers). Initial line times are proportional to text length; MFA word
   times refine them later (M07).
2. **Speaker labels:** detect `Name:` at the start of a line (any case), store the label,
   and remove it from the text sent to MFA.
3. **Sound and manner tags:** strip `[...]` and `(...)` tags from MFA text, keep them as
   flags. Lines that are only tags are non-dialogue. Tags like `[SPEAKING HUTTESE]` mark
   the line as non-English (excluded from donors). Tags like `[WEAKLY]` or `[SHOUTS]` are
   kept as delivery hints for later prosody work.
4. **Formatting:** strip `<i>` and other markup; keep an `italic` flag (often off-screen
   or voice-over) as weak evidence for M06.
5. **Name normalization:** map raw labels through the character list (M04) using
   case-insensitive fuzzy matching (edit distance plus common OCR confusions such as
   `0`/`o`, `1`/`l`), so `C-3p0o` resolves to `C-3PO`. Unknown names are kept and
   offered for review.
6. **Speaker tags in WebVTT:** read the standard voice tag (`<v Name>text`) as the line's
   speaker label. The existing cleaner already strips tags, so labeled VTT files work with
   the current pipeline; this step only adds reading the label. Never treat a `Name:`
   prefix as spoken text.
7. **Cue padding:** professional subtitles often start slightly late or end slightly
   early, and chunks are currently cut exactly at cue times. Add configurable padding
   (default 0.2 s each side, never crossing into the neighboring cue) when extracting MFA
   chunks, and keep word times relative to the true source timeline.
8. **OCR typo check:** the SDH text appears OCR-derived. When both SDH and plain English
   tracks exist, align them and report words that differ (likely OCR errors) for review;
   prefer the plain track's spelling when the words otherwise align one-to-one.
9. **Skipped-cue reporting:** an earlier note said the parser drops cues (Episode III SDH:
   1,398 blocks, 1,396 parsed). That was checked and is not a parser bug: one block is
   empty and one cue has zero duration (start equals end). Count and report skipped
   blocks and zero-duration cues with their reason instead of dropping them silently (the
   labeling tool reports `skipped_empty_blocks` and `zero_duration_cues_omitted`).

Movie SRT has cue-level timing only (unlike YouTube auto-caption VTT, which has per-word
timestamps). That is sufficient because MFA computes word timings.

**Verify:** X06, X07.

### M04. Character list and aliases

**Status (2026-10-04):** done for both movies as one shared file,
[config/characters/star_wars.json](../config/characters/star_wars.json). Remaining: load it
from the pipeline, and offer unknown names for review (M07). Deliberate choices:
`sidious` is separate from `palpatine` (processed voice), and `vader` is separate from
`anakin` (different performer); use `--speaker` with both IDs to pool them on purpose.

**Fix:** add `config/characters/<movie slug>.json` (tracked): a list of speaker IDs with
display names and aliases, for example `padme` with aliases `PADME`, `AMIDALA`,
`SENATOR AMIDALA`; `palpatine` with `CHANCELLOR`, `SUPREME CHANCELLOR`. Generic
roles (`MAN`, `MAN 1`, `ALIEN`, `CLONE TROOPER`, `BATTLE DROID`) map to role IDs that
are not merged with named characters. Build the first version automatically from
subtitle labels and script speaker names (M05), then let the user edit it. Speaker IDs
match the speaker registry so names stay consistent across movies and clips.

Notes for voice matching: one actor can voice several labels (for example Jango Fett and
the clone troopers), and processed voices (droids, Grievous) cluster poorly. Labels stay
separate even if voices are similar; M06 must not merge them automatically.

**Verify:** X08.

### M05. Transcript and screenplay alignment

**Why:** labels from subtitles cover ~5% of lines; a transcript or screenplay labels most
lines and also provides scene headings.

**Status (2026-10-04):** steps 1-3, 5 and 6 are done as the offline tool (section 2A),
including the exported labeled VTT and JSON report for both movies. Remaining: step 4
(scene candidates; the screenplay parsers already find 149 and 236 headings but do not
export them yet), exposing the tool through `run_mixer.py` (for example a
`label-speakers` subcommand that calls the same functions), and feeding the report into
the label store (M07).

**Fix:**

1. **Inputs:** `--script <path>` (repeatable) pointing to local files the user saved.
   Do not download them automatically. Available now (2026-10-04, saved by the user):
   - `input/ep2_transcript.txt`: transcript style; 351 `Name: line` speeches with 30
     speaker names (fewer speeches than subtitle cues, so some long speeches likely span
     several cues and some dialogue may be missing).
   - `input/ep3_transcript.txt`: transcript style; 1,124 speeches with 77 speaker names,
     plus about 300 stage-direction lines in brackets or parentheses (ignore them).
   Names vary in form and case (`Obi-Wan`, `OBI-WAN`, `Obi Wan Kenobi`,
   `Obi-Wan Kenobi`), so M04 alias normalization is required before alignment.
   Parsers:
   - Transcript style (`NAME: line`, plain text or saved HTML).
   - Screenplay style (character name on its own line above indented dialogue,
     parentheticals, `INT./EXT.` scene headings), from saved HTML or text.
   Output: ordered speeches `(speaker raw name, text, scene heading index)` and the scene
   heading list.
2. **Alignment:** normalize both word streams (lowercase, strip punctuation, numerals to
   words). Find unique word 4-grams shared by the subtitle stream and the script stream
   as anchors, keep the longest increasing chain of anchors (handles reordering and cuts
   in drafts), then align words between anchors with a banded edit-distance alignment.
   Each subtitle line gets a speaker vote from its aligned words; confidence is the share
   of the line's words matched to that speaker (require at least 3 matched words or 60%).
3. **Combining sources:** transcripts are the primary script source. Screenplays (for
   example the IMSDb drafts, saved locally if the user wants them) are a fallback: use
   them only for lines that subtitle labels and transcripts leave unlabeled, with lower
   confidence, and for scene headings. If two sources disagree, mark the line `conflict`
   for voice evidence or review. Earlier-draft screenplays will leave many lines
   unmatched; that is expected.
4. **Scenes:** map each aligned screenplay scene heading to the subtitle time of its
   first matched line, producing scene candidates for M09.
5. Report coverage per movie: share of lines labeled by subtitles, by each script, by
   agreement, and unmatched.
6. **Labeled subtitle export:** `run_mixer.py label-speakers --export-vtt <path>` writes
   the movie's own subtitle timing and text with `<v Speaker>` tags on labeled lines (no
   tag on unlabeled lines) plus a JSON report of confidence and unmatched lines. This is
   the "pre-labeled subtitle file" workflow: it can be produced right after M03 and M05,
   used immediately as a sidecar subtitle, and edited by hand. The export contains
   subtitle text, so it is written only to git-ignored locations (`input/` or `.build/`)
   and never committed.
7. Fandom transcript pages block automated fetching, so the user saves them locally;
   IMSDb screenplays are earlier drafts, so expect lower match rates from them than from
   the transcripts.

**Verify:** X09, X10.

### M06. Voice verification, propagation, and grouping

**Why:** voice evidence catches alignment mistakes and labels lines that subtitles and
scripts miss, and is what groups unknown speakers during ingest.

**Fix:**

1. After MFA, compute one speaker embedding per subtitle line, using the line's word span
   from MFA on the analysis audio (center channel or vocal stem). Lines shorter than
   0.8 s get a low weight; lines flagged as non-dialogue or two-speaker are skipped.
2. **Enrollment within the movie:** for each speaker ID with enough high-confidence lines
   (default at least 8 lines and 15 s), build a centroid from subtitle-labeled and
   script-labeled lines. Store centroids in the registry with model provenance (C07)
   and the movie as a source.
3. **Verification:** if a labeled line's embedding is far from its own centroid and close
   to another speaker's (with margin), mark it `conflict` for review instead of trusting
   the label.
4. **Propagation:** label unlabeled lines by nearest centroid only above a similarity
   threshold and margin (calibrated per movie, see step 6). Below that, leave them
   unlabeled.
5. **Grouping leftovers:** cluster the remaining lines and present each cluster in the
   existing interactive review (preview WAV, dialogue snippets, ranked candidates).
   This is the "likely groups" step during ingest. Non-interactive ingest keeps them as
   anonymous groups.
6. **Calibration you can trust:** run leave-one-out classification on the
   high-confidence lines and report accuracy per speaker. Pick the threshold that keeps
   estimated wrong-speaker rate under 2%; report the numbers in the ingest summary.

**Verify:** X11, X12.

### M07. Label store and integration with ingest

**Fix:**

1. `config/speaker_labels/<media hash>.json` (tracked): per line `start`, `end`, `cue_id`,
   `line_index`, `speaker_id` (or null), `status` (`confirmed`, `auto`, `conflict`,
   `unlabeled`), evidence (subtitle label, script votes, voice similarity and margin),
   and `text_sha1`. No subtitle or script text.
2. New commands:
   - `run_mixer.py label-speakers <movie> [--script ...] [--range ...]` runs M03-M06 for
     the given ranges and writes the label file. `ingest` runs it automatically when a
     label file is missing and scripts are given.
   - `run_mixer.py review-speakers <movie> [--range ...] [--status conflict,unlabeled]`
     walks lines with audio previews and lets the user confirm, change, or skip.
3. Turn confirmed and auto labels into verified `speaker_intervals` for
   `ensure_source_aligned`. Use MFA word times to set line bounds precisely, so
   dialogue-dash cues split into two intervals. Lines marked `conflict` or `unlabeled`
   produce no interval, so their words stay ineligible.
4. Include the label file hash in the labeled-inventory cache key (relabeling rebuilds
   only the inventory layer, not MFA).

**Verify:** X13.

### M08. Time ranges for ingest and mix

**Fix:**

1. Add `--range START-END` (repeatable) to `ingest`, `label-speakers`, `review-speakers`,
   and `mix`. Accept `HH:MM:SS(.mmm)`, `MM:SS`, or seconds. Validate against the
   duration; overlapping ranges merge.
2. `ingest --range`: process only blocks overlapping the ranges and only cues whose
   midpoint lies inside a range. Record ingested ranges in the movie's cache metadata.
   `ingest` without `--range` processes the whole movie block by block (resumable), so
   both full movies can be ingested once and then mixed for any speaker in any scene;
   `mix` can pool several movies (`--all-sources` or several sources) and still filter by
   `--range` or `--scene` per source.
3. `mix --range`: restrict candidate words, phones, and phrases to source spans fully
   inside the ranges. If a range has not been ingested, ingest it first (or fail with
   the command to run when `--non-interactive`).
4. Manifest support: a source item can carry `ranges` so a manifest can name several
   scenes from several movies.
5. Report per range: lines, labeled lines per speaker, and seconds of eligible speech for
   the requested speaker, so the user sees whether a scene has enough material.

**Verify:** X14, X15.

### M09. Named scenes (second phase)

**Fix:**

1. `config/scenes/<media hash>.json` (tracked): scene ID, start, end, name, and origin
   (`screenplay`, `detected`, or `user`).
2. Sources of scene boundaries, in order of preference: screenplay headings mapped to
   time (M05); detected boundaries from shot changes (`ffmpeg` scene score or
   PySceneDetect on a low-resolution decode) merged with dialogue gaps longer than a few
   seconds; and user edits.
3. Commands: `run_mixer.py list-scenes <movie>`, `run_mixer.py edit-scenes <movie>`
   (rename, merge, split by time). `--scene <name or id>` on `ingest` and `mix` resolves to
   a range.

**Verify:** X16.

### M10. Docs, skill, and evidence

Update the README (movie workflow, new commands and flags, copyright note about scripts),
the skill (`.agents/skills/sentence-mixer/SKILL.md`: movie quick start and
troubleshooting), and write an evidence report under `.build\reports\movies-<timestamp>\`
with timings and memory for a real range ingest (section 6.3). Follow the verification
rules in the round-4 plan (E6).

## 5. Regression checks

Use synthetic fixtures: a short MKV generated with ffmpeg containing 6-channel audio
with distinct tones per channel, two embedded SRT tracks (one titled SDH, with labels,
dashes, and tags), and fake embedding/ASR backends. Never use or commit copyrighted text
in fixtures; write original dialogue for test subtitles and scripts.

| ID | Check | Required result |
| --- | --- | --- |
| X01 | Long synthetic source (stretched to look like 2 h via a fake probe) rendered with a 3-segment plan | No full-file read: `soundfile.read` without frame bounds is never called on source or stem files; only requested frames are read |
| X02 | 5.1 fixture with a distinct tone only in the center channel | Analysis FLAC contains the center tone, not the others; stereo downmix contains all |
| X03 | Interrupt ingest after block 1 of 3, then rerun | Block 1 reused; blocks 2-3 processed; no duplicate work |
| X04 | MKV with SDH and plain text tracks and one PGS track, no sidecar | SDH track chosen; PGS reported unsupported; extracted file under `.build/subtitles/` |
| X05 | Same MKV with a sidecar `.srt` | Sidecar used unless `--subtitle-stream` is given |
| X06 | Cue with two dialogue-dash lines | Two lines with separate text; times split proportionally before MFA |
| X07 | Lines `Obi-wan: ...`, `C-3p0o: ...`, `[GRUNTS]`, `[SPEAKING HUTTESE] ...`, and a VTT line `<v Obi-Wan>...` | Labels `obi-wan`, `c-3po`; tag-only line non-dialogue; non-English line excluded from donors; labels and voice tags removed from MFA text; chunks padded 0.2 s without crossing neighbors; every cue block parsed |
| X08 | Script names `AMIDALA` and `PADME` with alias config | Both map to `padme`; `MAN 1` maps to a role ID, not a named character |
| X09 | Synthetic film subtitles plus a "draft" script with reordered and altered lines | At least 90% of matching lines get the correct speaker; altered lines stay unlabeled rather than mislabeled |
| X10 | Two scripts disagreeing on one line; then `--export-vtt` | Line marked `conflict`; exported VTT has `<v>` tags only on labeled lines, re-parses to the same labels, and is written only to an ignored path |
| X11 | Fake embeddings: two speakers well separated, one mislabeled line | Mislabeled line flagged `conflict`; unlabeled lines near a centroid with margin get `auto` |
| X12 | Fake embeddings with a third unknown voice | Its lines grouped into one cluster and offered for review; nothing assigned to a named speaker |
| X13 | Label file with one dialogue-dash cue split between two speakers | Two verified intervals from MFA word times; words of each half eligible only for its speaker; relabeling rebuilds the inventory layer without an MFA call |
| X14 | `ingest --range 00:00:10-00:00:20` on a 3-block fixture | Only the overlapping block processed; cues outside the range skipped |
| X15 | `mix --range` with a target word present inside and outside the range | Only the in-range occurrence is selectable; out-of-range ingestion requested or reported |
| X16 | Screenplay with two scene headings aligned to subtitles | Two scenes with correct start times; `--scene` resolves to the same range as the explicit times |

Keep the round-4 guards passing (overlap timing, XML audio, `--speaker` filter, D5
eligibility, unprimed listening gate, opt-in Whisper enrichment).

## 6. Commands (Windows PowerShell 5)

### 6.1 Preamble and gates

Use the preamble and gates from section 7 of the
[round-4 plan](10_04_antigravity_remediation_plan.md) at the start and end of the work.

### 6.2 Inspecting a movie

```powershell
Set-Location 'D:\Code\sentence_mixer'
$Movie = Get-ChildItem -LiteralPath .\input -Filter 'Star.Wars.Episode.III*.mkv' | Select-Object -First 1
& ffprobe -v error -show_entries 'format=duration:stream=index,codec_type,codec_name,channels,channel_layout:stream_tags=language,title' -of compact $Movie.FullName
```

### 6.3 Real-movie smoke test (after M01-M08)

Run on a five-minute range first. Save scripts locally under `input\scripts\` (they are
git-ignored). Record wall time and peak memory in the evidence report.

```powershell
Set-Location 'D:\Code\sentence_mixer'
$Py = Join-Path (Get-Location) '.venv\Scripts\python.exe'
$Movie = (Get-ChildItem -LiteralPath .\input -Filter 'Star.Wars.Episode.III*.mkv' | Select-Object -First 1).FullName
$Scripts = @(Get-ChildItem -LiteralPath .\input\scripts -Filter 'ep3_*' -ErrorAction SilentlyContinue | ForEach-Object { $_.FullName })
$ScriptArgs = @()
foreach ($S in $Scripts) { $ScriptArgs += @('--script', $S) }
$Timer = [Diagnostics.Stopwatch]::StartNew()
& $Py '.\run_mixer.py' 'ingest' $Movie '--range' '01:30:00-01:35:00' @ScriptArgs
if ($LASTEXITCODE -ne 0) { throw "ingest failed with exit $LASTEXITCODE" }
"Ingest took $([math]::Round($Timer.Elapsed.TotalMinutes, 1)) minutes"
& $Py '.\run_mixer.py' 'review-speakers' $Movie '--range' '01:30:00-01:35:00' '--status' 'conflict,unlabeled'
```

Then mix from that range only:

```powershell
& $Py '.\run_mixer.py' 'mix' '--source' $Movie '--range' '01:30:00-01:35:00' '--speaker' 'obi-wan' '--text' 'Hello there' '--render' '.\output\ep3_range_test.mp4'
if ($LASTEXITCODE -ne 0) { throw "mix failed with exit $LASTEXITCODE" }
```

The `--source` and `--range` flags are introduced by this plan; adjust if the final
interface names differ, and document the final names.

## 7. Suggested milestones

Already done (section 2A): labeled VTTs and reports for both movies, character list,
labeling tool and tests. Start from these.

1. **Ranges and subtitles first (M01, M02, M03, M08):** ingest and mix a five-minute
   range of Episode III using the labeled VTT from section 2A as the `--vtt` sidecar (or
   the embedded SDH track through M02), with whole-source assertion for a
   single-speaker test scene. Proves large-file handling.
2. **Labels (M03 step 6, M05 integration, M07):** read `<v>` tags and the report into
   the label store, refine line times with MFA word times, build verified speaker
   intervals, add the review command; mixes restricted to confirmed labels.
3. **Voice (M06):** verification, propagation (Episode II's 344 unlabeled lines first),
   grouping, calibration report; then whole movies block by block.
4. **Scenes (M09)** from the screenplay headings, and docs (M10).

Each milestone ends with the gates, the regression checks it covers, and an evidence
report. Report real-movie timings honestly (isolation of a whole movie on CPU may take
hours; ranges keep iteration fast).

## 8. Copy-ready prompt

```text
Implement plans\10_04_movie_support_plan.md in D:\Code\sentence_mixer, milestone by
milestone (section 7), after the round-4 prerequisites C01, C04, and C07 from
plans\10_04_antigravity_remediation_plan.md are in place.

Read both plans, GEMINI.md, and .agents\skills\sentence-mixer\SKILL.md first. Keep the
2026-10-04 uncommitted changes. Section 2A of the movie plan is already done: reuse
tools\label_subtitles.py, config\characters\star_wars.json, and the labeled VTTs and
reports in .build\subtitles\labeled\ rather than rewriting them; move shared logic into
src\ and import it from the tool. Write the X-checks for each package before the code.

Never modify files under input\. Extract subtitles and audio into .build only. Keep
transcripts and screenplays as local user files; never download them automatically,
never commit them, and never copy their text or subtitle text into tracked files,
fixtures, or reports (label and scene files store times, IDs, scores, and text hashes).
Use original synthetic dialogue in tests.

Output speech must come only from the source recordings. Do not install packages unless
a manifest changes or a check fails for a missing dependency; do not download models
during tests. Put scratch files under .build\scratch.

At the end of each milestone, run the full gates on the final tree, write the evidence
report with real timings and memory for the range smoke test, and report status per
package with exact gate summary lines. Do not commit unless asked.
```
