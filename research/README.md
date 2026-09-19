# Research Compendium & Literature Index

Evidence for a **clip-only sentence mixer**: select and reorder recorded speech and its corresponding video, without generating speech or faces. These notes distinguish findings in the supplied papers from engineering choices for this project; they are not claims that the current implementation reproduces those systems.

## Design decisions supported by this reading

- **First build MFA alignment, source-contiguous variable-span unit selection, and sample-accurate original-PCM rendering.** Preserve the source identity and audio/video time mapping of every selected interval. Sample-accurate rendering means executing chosen cuts accurately; it does not make estimated linguistic boundaries accurate.
- **Prefer natural runs softly, not greedily.** Longer phrases preserve internal coarticulation, rhythm, and prosody and require fewer artificial joins. A shorter alternative can nevertheless fit the target context better. Phrase-first is a quality/cut preference, not an unconditional longest-match rule.
- **Alignment is evidence, not ground truth or a safe-cut certificate.** Transcript, pronunciation, acoustic-model, and domain errors matter. Boundary uncertainty and diagnostic scores can prioritize review, rejection, and local refinement; they do not guarantee an inaudible cut.
- **Keep analysis separate from rendering.** WavLM is an optional later candidate-retrieval/reranking feature, admitted only after measured A/B quality gains over the baseline. It is neither a renderer nor an established best join metric. No vocoder or generated-face component belongs in the clip-only pipeline.
- **Disclose source-derived DSP.** The unmodified-PCM baseline is distinct from optional crossfades or gain changes: those alter source samples but do not generate new speech content. Any future time/pitch changes must also be explicit, optional, and recorded, rather than silently treated as unchanged clips.

No paper here guarantees natural arbitrary sentences from a limited corpus. Missing pronunciations, suitable coarticulation, or phrase-level prosody cannot always be repaired by selecting different cuts.

## Local papers and reading priorities

All links below resolve to supplied PDFs in this directory. References use **PDF page numbers**, except where explicitly labeled as printed thesis or proceedings pages; ACM article pages retain their `96:` or `68:` prefixes.

| Paper | Priority and project relevance |
| --- | --- |
| [McAuliffe et al. 2026 — current MFA comparison](<2026_Montreal Forced Aligner and the state of speech-to-text alignment in 2026_2606.18466v1.pdf>) | **Read first:** current alignment evidence, benchmark qualifications, model/domain choices; preprint |
| [McAuliffe et al. 2017 — original MFA](2017_Montreal_Forced_Aligner_Trainable_Text-Speech_Alig.pdf) | Historical architecture and measured alignment errors, not current-model performance |
| [Povey et al. 2011 — Kaldi](2011_The_Kaldi_speech_recognition_toolkit.pdf) | Background on the acoustic-model and decoding infrastructure |
| [Hunt & Black 1996 — unit selection](1996_HuntB96-speechsynth.pdf) | **Core:** target cost, join cost, contiguity, and sequence search |
| [Black & Taylor 1997 — clustered unit selection](1997_Black_1997_b.pdf) | **Core:** phonetic/prosodic context and movable join locations |
| [Vepa 2004 — perceptual join costs](2004_JithendraVepa_PhDthesis.pdf) | Selected methods/results: metric evaluation and its limits |
| [Jin et al. 2017 — VoCo](2017_Jin2017-VoCo-paper.pdf) | **Core transferable methods:** contextual range selection and alternatives; modified audio output |
| [Fried et al. 2019 — talking-head editing](<2019_Text-based Editing of Talking-head Video.pdf>) | Visual discontinuities and subsequence retrieval; **not** a clip-only renderer |
| [Chen et al. 2022 — WavLM](2022_WavLM.pdf) | Optional analysis-only representations; later experimental stage |
| [Baas et al. 2023 — kNN-VC](<2023_Voice Conversion With Just Nearest Neighbors_2305.18975v1.pdf>) | Feature-retrieval inspiration and important vocoder-dependent limitations |

## 1. Current MFA evidence: McAuliffe et al. (2026)

- **Title:** *Montreal Forced Aligner and the state of speech-to-text alignment in 2026*
- **Authors:** Michael McAuliffe, Kaylynn Gunter, Michael Wagner, Morgan Sonderegger.
- **Status of supplied copy:** **arXiv:2606.18466v1, 16 June 2026, preprint.** Do not infer conference acceptance or peer-review status from the filename or its associated benchmarking repository.

This is the most directly relevant current comparison in the collection. It documents MFA's updated HMM-GMM training, pronunciation modeling, pretrained models, adaptation, and evaluation utilities, and compares classical and neural aligners on English, Japanese, and Korean.

### Decisive results

The following are **mean phone-boundary errors and percentages within 10 ms**, not guaranteed precision for every boundary. These are the language-matched, unadapted pretrained-model rows, not the models trained on each benchmark.

| Benchmark | Model condition in the paper | Mean error (ms) | Boundaries within 10 ms |
| --- | --- | ---: | ---: |
| TIMIT, English read speech | MFA ARPA 3.0 | 12.11 | 61.85% |
| Buckeye, English spontaneous speech | MFA ARPA 3.0 | 13.87 | 62.93% |
| CSJ, Japanese spontaneous speech | MFA Japanese 3.0 | 10.82 | 63.44% |
| Seoul Corpus, Korean spontaneous speech | MFA Korean 3.0 | 14.78 | 61.11% |

**References:** §§4.1–4.2, p.5; §§5.1–5.2, p.6; Tables 5–7, pp.7–8.

- The newer models improve substantially over the original MFA models. However, **MFA is not the best system on every benchmark**: MAUS and MAPS have lower mean phone errors on TIMIT (11.26 and 11.46 ms).
- The headline “below 15 ms” must not be applied to word boundaries. MFA ARPA 3.0's **word-boundary** means are 19.93 ms on TIMIT and 21.75 ms on Buckeye (Table 4, p.6).
- **Model condition is not a software-version pin.** `MFA ARPA 3.0` and `MFA Global 3.0` identify different benchmark model conditions. Record the installed MFA software release separately from acoustic-model, dictionary, G2P, and phone-set versions. ARPA 3.0 uses approximately 982 hours of LibriSpeech, as did the original English model; Global 3.0 uses approximately 3,500 hours with broader dialect coverage (§§3.1, 4.2, pp.2–5). More training data is not a guarantee that Global is the better choice for a particular corpus.

### Qualifications that affect this project

- **Benchmark protocol matters.** The evaluation uses onset timestamps, phone-label mappings, and exclusions when reference/hypothesis manner categories disagree. Spontaneous-speech utterances of three words or fewer and certain problematic transcriptions are filtered out. These are not unconditional error measurements on arbitrary source videos (§4, pp.5–6).
- **Domain overlap is not the same as test-recording leakage.** The authors say the benchmark speech styles and varieties are well represented in MFA training. Separately, they flag MAPS's training on TIMIT as a possible inflation of its TIMIT performance. Rows trained or adapted on a benchmark should not be presented as performance on an unseen domain (§6, pp.7–8).
- **Adaptation is conditional.** Within-language adaptation gives small or mixed changes here; remapping and adapting an English model to Japanese/Korean yields larger gains. Phonological rules also help some conditions and hurt others (§§5.2.2–5.2.3, pp.6–7; §6, p.8).
- **Diagnostics are not certified cut confidence.** The paper describes per-utterance diagnostics and iterative transcript/dictionary correction, including the possibility of catastrophic misalignment from small input errors (§3.4, p.4). It does not establish a calibrated per-boundary probability of perceptually safe splicing.

**Transfer:** MFA is a defensible starting aligner, with model/dictionary selection and local boundary audits. Evaluate representative speakers, accents, background audio, and speaking styles before assigning confidence thresholds. Do not turn aggregate millisecond errors into fixed “safe” trimming margins.

## 2. Original MFA: McAuliffe et al. (2017)

- **Title:** *Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi*
- **Authors:** Michael McAuliffe, Michaela Socolof, Sarah Mihuc, Michael Wagner, Morgan Sonderegger.
- **Publication:** Interspeech 2017, pp.498–502.

**Evidence:** The original system combines context-dependent triphone GMM-HMM models with speaker adaptation. Its features use a 25-ms window and 10-ms frame shift. Historical mean phone-boundary errors are 17 ms on Buckeye and 25 ms on Phonsay; across evaluated boundary conditions, 2–5% differ from manual annotation by at least 100 ms. These are the **2017 models' results**, not measurements of today's installation.

**References:** §2, printed pp.498–499 (PDF pp.1–2); §3.3.1 and Tables 1–2, printed pp.500–501 (PDF pp.3–4).

**Transfer and limits:** Accurate transcripts, appropriate pronunciations, and domain fit are prerequisites. Triphone modeling improves contextual acoustic matching; it does not eliminate coarticulation or turn each aligned phone into an independently spliceable sound. Neither timestamp formatting nor a 10-ms frame grid implies exact-millisecond correctness.

## 3. Kaldi: Povey et al. (2011)

- **Title:** *The Kaldi Speech Recognition Toolkit*
- **Authors:** Daniel Povey et al.
- **Publication:** IEEE Automatic Speech Recognition and Understanding Workshop (ASRU), 2011.

**Evidence and transfer:** Provides feature extraction, context-dependent acoustic modeling, speaker adaptation, phonetic decision trees, and finite-state decoding infrastructure used by MFA. Relevant background for understanding model outputs and configuration, rather than a unit-selection or audio-rendering recipe.

**References:** §§III–V, p.2; §IX and Tables II–III, pp.3–4.

**Limit:** Its experiments report recognition word-error rates. Those results do not establish boundary accuracy, audible join quality, or preservation of original media.

## 4. Unit selection: Hunt & Black (1996)

- **Title:** *Unit Selection in a Concatenative Speech Synthesis System Using a Large Speech Database*
- **Authors:** Andrew J. Hunt and Alan W. Black.
- **Publication:** IEEE ICASSP 1996, pp.373–376.

The central distinction is between **target cost** (a unit's suitability for the intended phone, context, and prosody) and **concatenation cost** (compatibility with its neighbor). Omitting the paper's start/end silence terms, the objective is:

$$
C(T,U)=\sum_{i=1}^{N} C_t(t_i,u_i)
      +\sum_{i=2}^{N} C_j(u_{i-1},u_i).
$$

- **Contiguity is already explicit here:** consecutive units from the source database have zero concatenation cost, encouraging naturally continuous runs. This is an incentive, not a command to choose every available whole phrase.
- The target specification includes pitch, duration, power, and phonetic context. The paper assumes this specification exists; it does **not** solve natural target-prosody inference from arbitrary text.
- Viterbi gives the optimal path for the stated additive objective and candidate graph. Practical pruning reduces the search space; neither the cost model nor a pruned search guarantees perceptual optimality.
- The project can adapt this objective to source-contiguous variable-length spans. That is an engineering formulation, not a claim that this paper implements the entire audiovisual pipeline.

**References:** §§2–2.2, printed pp.373–375 (PDF pp.1–3); §4, printed p.376 (PDF p.4).

**Clip-only limit:** The paper's motivation is to reduce prosodic signal modification, not to prove that modification is never needed. Keep prosodic suitability in selection rather than assuming locally smooth joins yield natural sentence-level intonation.

## 5. Context and join location: Black & Taylor (1997)

- **Title:** *Automatically Clustering Similar Units for Unit Selection in Speech Synthesis*
- **Authors:** Alan W. Black and Paul Taylor.
- **Publication:** Eurospeech 1997.

This is the actual title of the supplied PDF. It is **not** *Automatically Selecting a Dictionary of Units for Concatenative Speech Synthesis*. It explicitly describes its own units as **uniform phones**, contrasting them with earlier non-uniform-unit work.

**Transferable methods:**

- Cluster alternatives using phonetic/prosodic context, including neighboring phones, stress, syllable/phrase position, pitch, and duration.
- Search over **both join location and join cost**. “Optimal coupling” can move a join toward a more stable phone region; phone-to-phone boundaries can be particularly unstable.
- Preserve context in comparing candidates. The join features include cepstral information, F0, power, and their deltas; phonetic identity alone is insufficient.

**References:** §§1–2.1, pp.1–2; §§2.2–2.3, p.2; §§3–4, pp.3–4.

**Clip-only limit:** The reported experiments used no post-selection signal modification, making them especially relevant. Nevertheless, the authors acknowledge missing prosodic forms, bad joins, and remaining low-quality examples. Movable cuts must preserve the required speech content; “move to the middle” is not a universal boundary rule.

## 6. Perceptual join metrics: Vepa (2004)

- **Title:** *Join Cost for Unit Selection Speech Synthesis*
- **Author:** Jithendra Vepa.
- **Publication/status:** University of Edinburgh PhD thesis, January 2004; not a short conference paper.

**Evidence:** The principal listening experiment varies mid-diphthong joins in ten sentences covering five stressed American-English diphthongs, using one male voice, controlled prosody, and a range of join qualities. It compares spectral representations and distance measures against listeners' ratings.

- The thesis explicitly focuses on **spectral cost**, omitting energy and F0 components from that investigation. It does **not** establish universal F0/energy/timbre weights for this project.
- Single-frame distances often correlate better than simple multi-frame averages. Learned weighting produces limited gains, with some closed-set results failing to generalize.
- LSF-based join costs receive more preferences than the compared MCA/Kalman alternatives, with qualified statistical significance. Linear smoothing is **not significantly preferred to no smoothing** in the reported comparison.

**References:** §§3.2.1–3.2.3, printed thesis pp.47–50; §§7.2–7.3, pp.151–156. Add **22** to a printed thesis page number to obtain the PDF page number.

**Transfer:** Calibrate join features with perceptual tests and held-out examples. Do not assume more frames, a fancier distance, or smoothing automatically improves joins; the study's speaker/phone coverage is limited.

**Clip-only limit:** The thesis also studies LSF smoothing and residual-excited LPC resynthesis (§§6.2.2–6.2.3, printed pp.129–132). Those operations modify/reconstruct waveforms. Their quality results cannot be claimed for unchanged PCM cuts, although their diagnostic features can inform selection.

## 7. Contextual range selection: VoCo (2017)

- **Title:** *VoCo: Text-based Insertion and Replacement in Audio Narration*
- **Authors:** Zeyu Jin, Gautham J. Mysore, Stephen DiVerdi, Jingwan Lu, Adam Finkelstein.
- **Publication:** ACM Transactions on Graphics 36(4), Article 96, SIGGRAPH 2017.

**Transferable methods:** Triphone-aware candidate selection; preferences for source continuity and fewer breaks; direct selection of variable-length frame ranges; joint similarity, duration, and concatenation costs; and multiple alternatives for a user to audition. Range selection can choose a join inside a phone, not only at an aligned boundary.

**References:** §§4.1–4.5, pp.96:4–7; §5.1, p.96:8. The “poison” example in §5.1 is a useful warning: an alternative with **more breaks** sounds more natural in context.

**Not unchanged-clip synthesis:** VoCo uses a synthetic TTS guide and a concatenative voice converter, with WSOLA/crossfading and optional manual pitch/duration changes (§§3.3–4.1, pp.96:4–5). This is not simply a neural vocoder, but its output is still not an untouched concatenation of source intervals. Its selection methods can be adapted without adopting its renderer.

**Evaluation limits:** The main study replaces individual words, using approximately 40 minutes of speech per voice (§5.2, p.96:8). The authors explicitly limit synthesis to words/short phrases and acknowledge that their hand-crafted energy is imperfectly correlated with perception and that natural long-range prosody remains unsolved (§6, pp.96:10–11).

## 8. Visual continuity, not a cutlist blueprint: Fried et al. (2019)

- **Title:** *Text-based Editing of Talking-head Video*
- **Authors:** Ohad Fried et al.
- **Publication:** ACM Transactions on Graphics 38(4), Article 68, SIGGRAPH 2019.

**Transferable ideas:** Prefer longer matching subsequences to reduce transitions; account for target timing; recognize that independently retrieved segments can have incompatible head pose, expression, illumination, and background.

**What the method actually does:** §3.3 uses phoneme/**viseme** sequence matching, edit-distance dynamic programming, and enumeration/memoization of target partitions, with a short-subsequence penalty. It is not a whole-word-first original-audio search. Visually interchangeable phones can be acoustically different, so its substitutions must not be copied into audio selection.

The renderer retimes and blends 3D face parameters and generates a new lower face with a neural renderer. Its output video is initially mute; audio is supplied using TTS, VoCo, or a separate performer recording. Most own-voice insertion results use a separate recording, while move/delete edits can use original audio.

**References:** §§3.3–3.5, pp.68:5–8; “Adding Audio,” p.68:8. **Section 4 is results, not the method blueprint.**

**Clip-only limit:** These are generated/retimed faces, not original-frame-only cuts. The evaluation and blending ablation therefore do not validate a clip-only renderer (§5.3, p.68:10; §5.7/Table 3, pp.68:11–12). The transferable lesson is to evaluate and budget visible discontinuities, not to assume neural-rendered smoothness will follow from the same retrieval strategy.

## 9. Optional analysis features: WavLM (2022)

- **Title:** *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*
- **Authors:** Sanyuan Chen et al.
- **Publication:** IEEE Journal of Selected Topics in Signal Processing, 2022. The supplied manuscript is **arXiv:2110.13900v5, 17 June 2022**.

**Evidence:** WavLM learns contextual speech representations through masked prediction and denoising. Its downstream results show layer-dependent speaker/content information, not one universal vector that guarantees preservation of pitch, loudness, room acoustics, or join quality. Base/Base+ use 768-dimensional hidden states; Large uses 1024. The feature stride is 20 ms, not sample-level boundary precision.

**References:** §IV.A, p.3; §§V.A–V.B.3, pp.5–6.

**Transfer and gate:** An encoder can analyze source audio without changing rendered samples. Later experiments may use its features for candidate retrieval or reranking, with model/layer/context choices recorded. Keep explicit contextual, voicing/F0, energy, and spectral baselines. Admit the feature only if blinded A/B listening and intelligibility checks show a repeatable benefit on held-out target edits at acceptable cost.

**Not established:** This paper neither validates a raw cosine boundary-join cost nor demonstrates that WavLM replaces hand-engineered join features for unmodified clips.

## 10. Feature matching with a generated waveform: kNN-VC (2023)

- **Title:** *Voice Conversion With Just Nearest Neighbors*
- **Authors:** **Matthew Baas, Benjamin van Niekerk, Herman Kamper.**
- **Publication:** Interspeech 2023; the supplied copy is **arXiv:2305.18975v1, 30 May 2023**.

**Actual system:** Encode source and reference speech with WavLM, replace each source feature with the mean of its nearest reference features, then **generate a waveform with an adapted HiFi-GAN vocoder**. The experiments use WavLM-Large layer 6, cosine matching, and typically four neighbors.

The matcher is non-parametric, but the renderer is trained. The authors explicitly describe inconsistent contexts between neighboring retrieved frames and train a **prematched vocoder** to reduce the resulting artifacts. Their main comparison uses that trained renderer.

**References:** §§3.1–3.4 and §4.1, pp.2–3; §5.2, p.4.

**Transfer:** Useful evidence that self-supervised features can retrieve acoustically relevant examples, and that layer selection matters for pitch/prosody and speaker similarity.

**Strict limitation:** It is not raw-waveform concatenation and supplies **no mathematical proof** that cosine distance between one clip's exit and another's entry measures audible join smoothness. Matching similar content to a query is a different task from evaluating transitions between successive, possibly different phones. Its naturalness scores cannot be transferred to a renderer with no vocoder.

## What still needs project-specific evidence

1. **Alignment and boundary refinement:** audit real source material, including transcription/pronunciation failures and difficult acoustic conditions. Report error tails and rejected cases, not just means. Independently listen to refined cuts; uncertainty alone is not a safety guarantee.
2. **Selection:** compare soft phrase/contiguity preferences with shorter alternatives, including stress, phrase position, neighboring phones, and target prosody. Keep enough candidate diversity to avoid locking in a poor long span.
3. **Join quality:** evaluate spectral, voicing/F0, energy, and contextual baselines with blinded listening. Add WavLM only as an ablation, not as an assumed upgrade. Word-error rate alone does not measure naturalness or visible continuity.
4. **Rendering and provenance:** verify selected PCM sample ranges and audio/video mappings independently of selection quality. Distinguish unchanged cuts from disclosed crossfade/gain processing and any explicitly enabled future time/pitch edits.

Recommended reading order: **2026 MFA → Hunt & Black → Black & Taylor → VoCo selection sections → Vepa's selected experiments**. Use the original MFA/Kaldi papers for background, Fried for visual failure modes, and WavLM/kNN-VC for the optional later analysis experiment.