# Architectural Rationale & Design Document: Automated Sentence Mixing

## Executive Summary

This document details the engineering and algorithmic design of the sentence-mixing pipeline. It explains the design choices, the physical and mathematical reasons for those choices, and how this architecture compares to alternative approaches—such as naive heuristic slicing and modern generative neural text-to-speech (TTS).

---

## 1. The Core Engineering Challenge

Sentence mixing is a specialized subfield of **constrained, video-locked concatenative speech synthesis**. Unlike conventional text-to-speech, where a system can generate arbitrary audio waveforms from noise, sentence mixing imposes a strict physical constraint:

> **The output audio must be backed by matching, authentic video frames extracted from the original source media.**

Every speech segment selected by the engine corresponds to a physical segment of video footage ($t_{\text{start}}$ to $t_{\text{end}}$) showing the speaker's face and mouth moving at that precise moment.

Designing an automated system for this task introduces three competing constraints:

1. **Intelligibility:** Speech must be clear enough for listeners to understand the new sentence.
2. **Acoustic Continuity:** Cuts must transition smoothly without clicks, pops, pitch jumps, or energy drops.
3. **Aesthetic & Visual Provenance:** The resulting video cuts must retain the authentic, rhythmic editing style characteristic of sentence mixing, without synthetic audio hallucination or generative video warping.

---

## 2. Evaluation of Alternatives

Four primary architectural paradigms exist for synthesizing speech and synchronized video.

### Option A: Generative Neural TTS (ElevenLabs, XTTS, VITS, Bark)

Generative neural models take text and a speaker reference clip, passing them through an autoregressive transformer or diffusion backbone to synthesize a fresh audio waveform.

* **Why it fails for sentence mixing:** Generative models synthesize audio from scratch. They do not use source cuts, meaning there are no associated source video timecodes.
* **The failure mode:** You cannot construct an authentic video edit from a neural hallucination. Splicing random video clips over an AI-generated voice track loses the physical connection where the speaker's mouth matches the phonemes being spoken.

### Option B: Generative Audio + Neural Talking-Head Inpainting (Wav2Lip, SadTalker)

This approach generates audio using Neural TTS (Option A), and then runs a deep-learning face-morphing model (like Wav2Lip) to synthesize new mouth movements over an existing video frame.

* **Why it fails for sentence mixing:**
1. **Aesthetic destruction:** It produces an uncanny, smoothed-out face-morph rather than the kinetic, frame-accurate cut aesthetic of sentence mixing.
2. **Visual artifacts:** Neural face inpainters suffer from blurriness, resolution mismatch (typically downscaled to 256x256 or 512x512), and temporal flickering around the jaw and teeth.
3. **Heavy compute requirement:** Generating inpainting requires powerful GPUs, high latency, and complex neural inference pipelines.



### Option C: Naive Heuristic / First-Match Splicing (The Previous Implementation)

The system parses subtitle cues, grabs the first occurrence or median-duration instance of a word, and falls back to isolated single phonemes if a word is missing.

* **Why it fails for high-quality audio:**
1. **Intonation rollercoaster:** Words are selected in isolation without evaluating pitch context. A low-pitched, calm word might be spliced next to a high-pitched exclamation, causing wild frequency swings.


2. **Coarticulation destruction:** English phonemes morph their acoustic formants based on adjacent sounds. Splicing isolated monophones (e.g., `/d/` + `/ɑː/` + `/ɹ/` + `/θ/`) sounds like an unintelligible, buzzing robotic artifact.


3. **Rushed cadence (Machine-gun effect):** Concatenating discrete words with 0 ms padding makes the speech feel unnaturally rushed.


4. **Phase cancellation clicks:** Cutting audio without aligning to waveform phase boundaries creates loud impulse transients (pops and clicks) at the seams.





---

### Option D: Non-Uniform Viterbi Unit Selection (Our Chosen Design)

Rooted in the foundational concatenative synthesis work of **Hunt & Black (1996)** and the video-editing synthesis of **Fried et al. (SIGGRAPH 2019)**, this approach models sentence mixing as a **global cost-minimization search over a discrete candidate trellis**.

```
Candidate Lattice:
Word 1 (Phonemes)      Word 2 (Phonemes)      Word 3 (Phonemes)
[ u_1,1 ] ───────────► [ u_2,1 ] ───────────► [ u_3,1 ]
[ u_1,2 ] ───┐   ┌───► [ u_2,2 ] ───┐   ┌───► [ u_3,2 ]
[ u_1,3 ] ───┼───┼───► [ u_2,3 ] ───┼───┼───► [ u_3,3 ]
             └───┘                  └───┘
  (Each path evaluated: Target Cost + Join Cost via Dynamic Programming)

```

| Dimension | Option A: Generative TTS | Option B: Neural Inpainting | Option C: Naive Splicing | Option D: Non-Uniform Viterbi (Our Design) |
| --- | --- | --- | --- | --- |
| **Physical Video Provenance** | None (0%) | Artificial / Inpainted | Authentic (100%) | **Authentic (100%)** |
| **Phonetic Intelligibility** | High (Synthetic) | High (Synthetic) | Low / Broken | **High (Acoustically Guided)** |
| **Prosodic Smoothness** | High | High | Chaotic / Jumpy | **Optimized (Pitch/Energy Seams)** |
| **Multi-Word Preservation** | N/A | N/A | Accidental / Rare | **Guaranteed by Contiguity Bonus** |
| **Boundary Artifacts** | None | None | Severe (Clicks/Pops) | **Zero (Phase-Aligned Snapping)** |
| **Hardware Overhead** | High (CUDA/GPU) | Extreme (GPU Render) | Minimal (CPU) | **Moderate (Pure CPU/Lightweight)** |

---

## 3. The Mathematical & Algorithmic Framework

Our pipeline synthesizes sentences by solving an explicit objective function.

Given a target sequence of $N$ phonetic specifications $T = (t_1, t_2, \dots, t_N)$ and a database of candidate units $\mathcal{U}$, we seek the candidate sequence $U^* = (u_1, u_2, \dots, u_N)$ that minimizes the total sequence cost:

$$C(T, U) = \sum_{i=1}^{N} C_t(t_i, u_i) + \sum_{i=1}^{N-1} C_j(u_i, u_{i+1})$$

### 3.1. Target Cost $C_t(t_i, u_i)$

The target cost measures how well an individual candidate unit matches the desired linguistic target:

$$C_t(t_i, u_i) = w_{\text{ctx}} D_{\text{ctx}}(t_i, u_i) + w_{\text{stress}} D_{\text{stress}}(t_i, u_i) + w_{\text{dur}} D_{\text{dur}}(u_i) + w_{\text{pitch}} D_{\text{pitch}}(u_i)$$

1. **Triphone Context Match ($D_{\text{ctx}}$):** Evaluates the left and right phonemes of the candidate in its original utterance. If a candidate was recorded between the same adjacent sounds as the target, coarticulation distortion drops to near zero.
2. **Stress & Accent ($D_{\text{stress}}$):** Penalizes substituting an unstressed vowel (e.g., `AH0`) for a primary-stressed vowel (`AH1`).
3. **Duration Outlier Penalty ($D_{\text{dur}}$):** Penalizes candidates whose duration deviates excessively from the natural baseline duration of that phoneme (typically ~90 ms), preventing rushed or unnaturally elongated sounds.
4. **Prosodic Pitch Anchor ($D_{\text{pitch}}$):** Penalizes candidates whose fundamental frequency ($F_0$) deviates significantly from the speaker's natural median pitch, preventing unnatural pitch shifts.

---

### 3.2. Join Cost $C_j(u_i, u_{i+1})$

The join cost quantifies acoustic discontinuity at the splice point between two consecutive units:

$$C_j(u_i, u_{i+1}) = \begin{cases} C_{\text{bonus}}, & \text{if } u_i \text{ and } u_{i+1} \text{ are contiguous in the source media} \\ C_{\text{cut}} + C_{\text{spectral}} + C_{\Delta F_0} + C_{\Delta E}, & \text{otherwise} \end{cases}$$

1. **The Contiguity Bonus ($C_{\text{bonus}} \le -80.0$):**
If candidate $u_i$ and candidate $u_{i+1}$ were originally spoken back-to-back in the same video chunk, the transition is naturally smooth. Assigning a negative cost bonus causes the Viterbi path to favor multi-word phrases, whole words, and natural syllables over fragmented cuts.
2. **Spectral Discontinuity ($C_{\text{spectral}}$):**
Evaluates the Euclidean distance between the exit Mel-Frequency Cepstral Coefficients (MFCCs) of unit $u_i$ and the entry MFCCs of unit $u_{i+1}$:

$$C_{\text{spectral}} = w_{\text{mfcc}} \Vert{}\mathbf{m}_{\text{end}}(u_i) - \mathbf{m}_{\text{start}}(u_{i+1})\Vert{}_2$$



This ensures that the vocal tract shape and formant frequencies at the end of the first sound match the start of the next.
3. **Pitch Jump ($C_{\Delta F_0}$):**
Penalizes frequency steps across the boundary:

$$C_{\Delta F_0} = w_{\Delta F_0} \left\vert{} F_{0,\text{end}}(u_i) - F_{0,\text{start}}(u_{i+1}) \right\vert{}$$


4. **Loudness Discontinuity ($C_{\Delta E}$):**
Penalizes energy steps:

$$C_{\Delta E} = w_{\Delta E} \left\vert{} \text{RMS}_{\text{end}}(u_i) - \text{RMS}_{\text{start}}(u_{i+1}) \right\vert{}$$



---

## 4. Acoustic & Video Signal Processing Mechanics

### 4.1. Phase Alignment via Positive Zero-Crossing Snapping

Cutting an audio stream at an arbitrary timestamp creates a sharp step discontinuity in the waveform, resulting in an audible broadband click.

Furthermore, if unit $A$ ends during a positive wave cycle and unit $B$ begins during a negative cycle, crossfading them produces **destructive phase cancellation**, causing a hollow or muffled transient.

```
Waveform Phase Mismatch (Click / Cancellation):
Unit A:  ───/╲  (Ends high)
              ╲
Unit B:        ╲╱───  (Starts low)
               ▲ DISCONTINUITY / POP

Positive Zero-Crossing Snapped:
Unit A:  ───/╲
              ╲
Unit B:        ───/╲  (Both cleanly cross 0V going positive)
                  ▲ SEAMLESS PHASE TRANSITION

```

* **The Solution:** The assembler scans a $\pm 6\text{ ms}$ local window around each start and end boundary, snapping the cut to the nearest index where:

$$y[n-1] \le 0 \quad \text{and} \quad y[n] > 0$$



This ensures that every cut begins and ends at zero amplitude with a rising phase.

### 4.2. Intra-Slice Micro-Fading

Standard crossfading between two clips shortens the combined audio duration relative to the video, causing progressive audio-video drift across multiple cuts.

* **The Solution:** The assembler applies an **intra-slice 5 ms raised-cosine fade-in and fade-out** directly to the audio stream within each individual cut. This preserves a 1:1 match between audio packet duration and video frame duration while eliminating edge clicks.



### 4.3. Cadence & Inter-Word Micro-Pauses

In continuous speech, speakers naturally insert micro-pauses (40–80 ms) between distinct words. Without these pauses, concatenated speech sounds like an unnaturally hurried rush.

* **The Solution:** The Viterbi engine tracks word boundaries (`is_word_end`). When a cut ends a complete word, the assembler inserts a **45 ms trailing micro-pause** (`apad=pad_dur=0.045`). Slices within a multi-phone word or contiguous phrase transition with 0 ms delay to preserve coarticulation.



---

## 5. System Architecture & Workspace Design

### Directory Structure & Hygiene

Temporary build files are isolated from project source code to prevent repository bloat:

```text
sentence_mixer/
├── .build/                         <-- SCRATCH / BUILD ARTIFACTS (Ignored by Git)
│   ├── cache/                      <-- Per-source SHA-256 alignment caches
│   ├── corpus/                     <-- 16kHz mono audio chunks + transcripts
│   ├── aligned/                    <-- Aligned Praat .TextGrid files
│   └── inventory/                  <-- Indexed units_index.json database
├── dictionaries/                   <-- SOURCE ASSETS (Tracked in Git)
│   ├── custom_pronunciations.txt   <-- Domain-specific proper nouns (Star Wars names)
│   └── merged_dictionary.dict      <-- Expanded CMU/ARPA pronunciation lexicon
├── input/                          <-- SOURCE MEDIA (User-Provided)
│   ├── source1.mp4 & source1.en.vtt
│   └── source2.mp4 & source2.en.vtt
├── output/                         <-- RENDERS (User Deliverables)
│   ├── mix_output.mp4
│   └── mix_output.wav
├── src/                            <-- APPLICATION SOURCE CODE
│   ├── vtt_preprocessor.py         <-- Caption deduplication & audio chunking
│   ├── inventory_builder.py        <-- TextGrid parsing & acoustic boundary extraction
│   ├── unit_selector.py            <-- Viterbi non-uniform unit selection engine
│   ├── video_assembler.py          <-- Zero-crossing snapping & FFmpeg rendering
│   └── fcp_xml_generator.py        <-- Premiere Pro / FCP7 XML sequence exporter
├── README.md                       <-- Setup and execution guide
└── run_mixer.py                    <-- Unified CLI orchestrator

```

### Key Workspace Benefits

1. **Single-Command Cleanup:** Deleting `.build/` removes all temporary WAV chunks, TextGrids, and alignment caches without affecting source media or custom dictionaries.


2. **Deterministic Re-Alignment:** When source media or dictionary files change, clearing `.build/` forces a clean rebuild from scratch.


3. **Cross-Platform Portability:** All file and directory references are resolved dynamically via `pathlib.Path.home()`, eliminating hardcoded user paths.