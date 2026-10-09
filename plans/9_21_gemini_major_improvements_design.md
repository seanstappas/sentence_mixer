### 1. Speaker Diarization: Why Pyannote on Isolated Vocal Stems

#### Why Tiered (VTT Parsing First, Neural Diarization Second)

Running deep learning models across full video files is computationally expensive. Many production subtitles (such as official release VTTs) already contain speaker markup (`<v Obi-Wan>` or `OBI-WAN:`).

* Parsing existing text markup takes **< 1 ms** and achieves **100% ground-truth precision**.
* Only falling back to neural diarization when metadata is missing ensures the pipeline remains fast for standard media.

#### Why Diarize the Isolated Vocal Stem (`vocals.wav`) Instead of Raw Video Audio

In film and television dialogue (e.g., Star Wars), voices are almost never clean; they are layered over dynamic orchestral music, weapon sound effects, and spatial reverb.

* **The Failure Mode:** Speaker diarization models (like ECAPA-TDNN or x-vector embeddings) calculate acoustic similarity vectors across sliding windows. Loud background strings, brass swells, or continuous ambient rumble distort these embeddings, causing the clustering algorithm to falsely classify one speaker as multiple different people or confuse quiet speakers with background music.
* **The Solution:** Because the pipeline already includes Demucs stem separation (`vocal_isolation.py`), feeding the stripped `vocals.wav` into diarization isolates the vocal formant frequencies ($F_1$–$F_4$), reducing the Diarization Error Rate (DER) from roughly 25–40% on cinematic audio down to **< 8%**.

#### Why `pyannote/speaker-diarization-3.1`

* **Whisper is not a diarizer:** Whisper is an end-to-end speech-to-text model. It does not natively cluster or label distinct speaker identities across a timeline.
* **End-to-End Overlap Detection:** Older diarization tools (like Kaldi diarization or pure spectral clustering) struggle when characters speak over each other. Pyannote 3.1 integrates end-to-end neural segmentation that detects simultaneous speech turns natively.
* **Speed:** At roughly **0.02x–0.04x Real-Time Factor (RTF)** on a modern CUDA GPU, a 20-minute video episode diarizes in **under 45 seconds**.

---

### 2. Hybrid Phonetics: Why `{P H O N E}` Syntax and CTC Recognition

#### Why Space-Delimited ARPAbet (`{B AE1 K AH0 N}`)

* **Disambiguation of Multi-Character Phonemes:** ARPAbet contains both single-letter consonants (`B`, `T`, `D`) and two-letter consonant digraphs (`CH`, `SH`, `TH`, `DH`, `ZH`, `NG`). Without explicit spaces, tokens like `{THAE1T}` create parsing ambiguities (is `TH` the dental fricative /θ/ or a `T` followed by an aspirated `H`?). Spaces eliminate lookahead parsing ambiguities.
* **1:1 Alignment with Pipeline Internals:** Montreal Forced Aligner, CMUdict, and your `InventoryIndex` internally store phonemes as uppercase ARPAbet strings with lexical stress digits (`0` = unstressed, `1` = primary stress, `2` = secondary stress). Passing these tokens directly skips any lossy translation layer.

#### Why One Set of Braces per Word (`{...} {...}`)

* **Hierarchical Span Search Preservation:** Your selector (`span_selector.py`) operates hierarchically: it searches for whole words first, falling back to diphones and single phones only when necessary.
* If a user writes `{B AE1 K AH0 N} {P IH1 G}`, the engine treats `{B AE1 K AH0 N}` as a single word entity. It can query the inventory for any clip where that exact pronunciation was spoken as an intact word (e.g., matching the word *bacon*).
* If multi-word syntax were allowed (`{B AE1 K AH0 N P IH1 G}`), the inventory would never find a single 7-phone word, immediately forcing the engine to fragment the phrase into sub-word phonetic cuts.



#### Why a CTC Phoneme Recognizer Over Whisper for Mic Input

* **Whisper's Strong Language Model Bias:** Whisper is designed to output standard English orthography. If you speak nonsense syllables, character sound effects, exaggerated accents, or spliced sound sequences into the mic (e.g., *"doh"*, *"spoo-kuh"*, *"bap"*), Whisper forces them into standard dictionary words (*"dough"*, *"spooky"*, *"bath"*).
* **Acoustic Fidelity:** A Connectionist Temporal Classification (CTC) phone model (e.g., `wav2vec2-lv-60-espeak-cv-ft`) has no word-level language model constraint. It listens directly to the spectral transitions and emits the raw phoneme posteriors. What you utter is mapped purely to physical sound units.

---

### 3. Cadence & Prosody: Why Freeze-Frames and Soft Cost Biasing

#### Why Pause Gaps & Video Freeze-Frames

* **Acoustic Naturalness:** Concatenating speech clips without pauses sounds breathless and robotic. Human speech relies on inter-phrase silences (150–400 ms) for rhythmic cadence.
* **Visual Continuity:** Naive sentence-mixing scripts either leave video black during pauses or stretch the audio of the previous clip, causing pitch wobble. Holding/freezing the last frame of the preceding video cut across the gap mimics natural film reaction editing without corrupting audio pitch.

#### Why Soft Viterbi Biasing (Instead of Hard Pruning) for `?` and `!`

In sentence mixing, the source inventory is **finite and closed**—you can only use what the actor actually said on camera:

* **The Flaw of Hard Filtering:** If you strictly mandate that a word ending with `?` can *only* come from an interrogative source sentence, synthesis will fail whenever the actor never asked a question containing that word.
* **The Advantage of Soft Priors:** By applying a cost reduction in the Viterbi trellis:
* If a candidate unit has a rising pitch slope ($F_0$ slope $> 0$), its target cost drops significantly, causing the optimizer to pick it over flat alternatives.
* If no rising candidate exists, the optimizer still selects the cleanest available neutral unit, preventing synthesis failure.



#### Acoustic Rationale for $F_0$ Slope and RMS Energy

* **Question (`?`):** In General American and British English, terminal rising intonation is the primary acoustic correlate of questions and inquisitive remarks. Measuring the pitch derivative ($\Delta F_0 / \Delta t$) across the terminal vowel accurately identifies this contour.
* **Exclamation (`!`):** Emphatic delivery correlates directly with high Root Mean Square (RMS) energy and primary lexical stress (`1`). Rewarding high RMS energy and stress-1 vowels prevents the engine from splicing in quiet murmurs or trailing whispered syllables for an exclamation.

---

### 4. Interactive OOV Assistant: Why Auto-Learning and Exemplars

#### Why "Sounds-Like" Exemplars Over Raw IPA/ARPAbet

Most creators understand how they want a word to sound, but cannot distinguish ARPAbet vowels like `AA` vs `AE` vs `AH` on sight.

* Showing `B AE1 K T AH0` is abstract.
* Showing `AE1 -> 'a' in apple (stressed)` and `AH0 -> 'u' in cut / sofa (unstressed)` provides instant clarity, preventing incorrect pronunciations from entering the project.

#### Why Default-Interactive with a `--non-interactive` Bypass

* **Interactive by Default:** Out-of-vocabulary (OOV) words are the single biggest source of synthesis failure in unit selection. G2P models frequently mispronounce proper nouns, fantasy names (*Coruscant*, *Bacta*, *Tatooine*), and slang. Intercepting these words at the CLI level saves you from waiting for a full video render only to discover a syllable was mispronounced.
* **`--non-interactive` Guard:** Headless automated pipelines, regression test suites (`pytest`), and background workers cannot accept `stdin` inputs. Checking `sys.stdin.isatty()` alongside an explicit flag ensures automated workflows run uninterrupted using best-guess G2P.

#### Why Auto-Persisting to `custom_pronunciations.txt`

* **Zero Repeat Overhead:** Once you confirm how a word sounds once (e.g., adding `bacta  B AE1 K T AH0`), it is appended to disk. MFA and `Lexicon` read from this file on all future runs, meaning you never have to re-enter or re-verify that word again.