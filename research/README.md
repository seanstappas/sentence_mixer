# Research Compendium & Literature Index

This document outlines the core scientific literature behind the **Montreal Forced Aligner (MFA) + Non-Uniform Viterbi Unit Selection** sentence mixing pipeline. It maps each paper to its specific algorithmic function in the codebase and highlights which papers serve as foundational theory, direct implementation blueprints, or modern drop-in replacements.

---

## Quick Reference: Pipeline Mapping

| File Name | Year | Topic / Focus | Role in Our Pipeline |
| --- | --- | --- | --- |
| `2011_The_Kaldi_speech_recognition_toolkit.pdf` | 2011 | ASR / WFST Framework | Engine backend for acoustic modeling & feature extraction

 |
| `2017_Montreal_Forced_Aligner_Trainable_Text-Speech_Alig.pdf` | 2017 | Forced Alignment | **Ground truth temporal segmentation** (word & phone intervals)

 |
| `1996_HuntB96-speechsynth.pdf` | 1996 | Unit Selection Formulation | **Core optimization math** (Target Cost + Join Cost via Viterbi) |
| `1997_Black_1997_b.pdf` | 1997 | Non-Uniform Unit Selection | Variable-length chunking (preferring words/phrases over phones) |
| `2004_JithendraVepa_PhDthesis.pdf` | 2004 | Perceptual Join Metrics | Mathematical evaluation of spectral boundary distance ($F_0$, MFCC, LSF) |
| `2017_Jin2017-VoCo-paper.pdf` | 2017 | Audio Voice Editing | Practical application of voice editing from personal speech databases |
| `2019_Text-based Editing of Talking-head Video.pdf` | 2019 | Video-Driven Sentence Mixing | **End-to-end architecture blueprint** connecting transcripts, cuts, and video |
| `2022_WavLM.pdf` | 2022 | Self-Supervised Speech Representations | Modern acoustic feature extractor (replaces manual MFCC/pitch math) |
| `2023_Voice Conversion With Just Nearest Neighbors_2305.18975v1.pdf` | 2023 | SSL Concatenation & Distance | Proves cosine distance in WavLM latent space predicts seamless joins |

---

## Detailed Paper Breakdowns

### 1. Forced Alignment & Temporal Ground Truth

#### `2011_The_Kaldi_speech_recognition_toolkit.pdf`

* **Title:** *The Kaldi Speech Recognition Toolkit*

* **Authors:** Daniel Povey, Arnab Ghoshal, Gilles Boulianne, Lukáš Burget, Ondřej Glembek, Nagendra Goel, Mirko Hannemann, Petr Motlíček, Yanmin Qian, Petr Schwarz, et al. (IEEE ASRU 2011)


* **What it is:** The foundational open-source C++ speech recognition toolkit built around Weighted Finite-State Transducers (WFSTs).


* **Why it matters:** Kaldi provides the lower-level binaries, triphone acoustic models, and Baum-Welch / Viterbi decoding routines that execute inside our `mfa_env`.


* **Reading Priority:** **Background only.** You do not need to study Kaldi's C++ internals; MFA wraps and simplifies its alignment capabilities.



#### `2017_Montreal_Forced_Aligner_Trainable_Text-Speech_Alig.pdf`

* **Title:** *Montreal Forced Aligner: Trainable Text-Speech Alignment Using Kaldi*

* **Authors:** Michael McAuliffe, Michaela Socolof, Sarah Mihuc, Michael Wagner, Morgan Sonderegger (Interspeech 2017)


* **What it is:** The paper introducing MFA, demonstrating how Kaldi's acoustic modeling can be structured into an accessible, command-line forced aligner.


* **Why it matters:** This explains **how our interval boundaries are derived**. MFA uses triphone acoustic models with speaker-adaptive training (fMLLR) rather than primitive monophones. This ensures that word and phoneme timestamps align to the exact millisecond, preventing boundary cutoffs.


* **Reading Priority:** **High.** Essential for understanding the alignment step and dictionary requirements.



---

### 2. The Algorithmic Engine: Unit Selection & Join Costs

#### `1996_HuntB96-speechsynth.pdf`

* **Title:** *Unit Selection in a Concatenative Speech Synthesis System Using a Large Speech Database*
* **Authors:** Andrew J. Hunt and Alan W. Black (IEEE ICASSP 1996)
* **What it is:** The foundational 4-page paper that invented modern concatenative unit selection.
* **Why it matters:** **This is the exact mathematical model implemented in `unit_selector.py`.** It defines the dual-cost equation:

$$C(T, U) = \sum_{i=1}^N C_t(t_i, u_i) + \sum_{i=1}^{N-1} C_j(u_i, u_{i+1})$$



It proves that sentence mixing is a graph-search problem solvable globally in polynomial time using the Viterbi algorithm.
* **Reading Priority:** **Must-read (Core Math).**

#### `1997_Black_1997_b.pdf`

* **Title:** *Automatically Selecting a Dictionary of Units for Concatenative Speech Synthesis*
* **Authors:** Alan W. Black and Paul A. Taylor (Eurospeech 1997)
* **What it is:** The follow-up to Hunt & Black extending unit selection to **variable-length non-uniform units**.
* **Why it matters:** Solves the "choppy monophone" problem by proving that candidate units should not be restricted to single letters. It formalized the **sub-sequence continuity bonus** ($C_j = 0$ for naturally contiguous phonemes), which forces the search trellis to pick multi-word phrases and whole words whenever they exist in the source video.
* **Reading Priority:** **Medium.** Explains why our dynamic programming engine consolidates contiguous intervals into longer cuts.

#### `2004_JithendraVepa_PhDthesis.pdf`

* **Title:** *Join Methods for Unit Selection Speech Synthesis*
* **Author:** Jithendra Vepa (PhD Thesis, University of Edinburgh, 2004)
* **What it is:** The most comprehensive academic study on acoustic join costs, evaluating spectral distance metrics, Line Spectral Frequencies (LSFs), MFCCs, and pitch jumps across concatenation boundaries.
* **Why it matters:** Provides the empirical justification for our signal-processing join metrics. It explains why simple waveform splicing fails without spectral cross-matching and establishes the baseline weights used when comparing boundary frames ($F_0$, energy, and timbre).
* **Reading Priority:** **Reference / Skim.** Read Chapter 1 (Introduction) and Chapter 3 (Distance Measures at Acoustic Joins).

---

### 3. Audio-Visual Sentence Mixing & Voice Editing

#### `2017_Jin2017-VoCo-paper.pdf`

* **Title:** *VoCo: Text-based Insertion and Replacement in Audio Narration*
* **Authors:** Zeyu Jin, Gautham J. Mysore, Stephen DiVerdi, Jingwan Lu, Adam Finkelstein (ACM Transactions on Graphics / SIGGRAPH 2017)
* **What it is:** Adobe's famous "Photoshop for Voice" research project.
* **Why it matters:** Bridges classical unit selection with modern desktop voice editing. It demonstrated how to take a personal audio corpus (e.g., 20 minutes of speech), align it via forced alignment, build an in-memory phonetic index, and dynamically synthesize new text by splicing words and diphones with pitch smoothing.
* **Reading Priority:** **High.** Serves as the direct conceptual prototype for automated sentence mixing.

#### `2019_Text-based Editing of Talking-head Video.pdf`

* **Title:** *Text-Based Editing of Talking-Head Video*
* **Authors:** Ohad Fried, Ayush Tewari, Michael Zollhöfer, Adam Finkelstein, Eli Shechtman, Dan B Goldman, Maneesh Agrawala, Christian Theobalt (ACM Transactions on Graphics / SIGGRAPH 2019)
* **What it is:** Stanford and Princeton's extension of Project VoCo directly to multi-modal video.
* **Why it matters:** **The complete engineering blueprint for our video pipeline.** Section 4 describes the dynamic programming search across aligned transcript intervals, how whole words are prioritized, how fallback phonemes are stitched, and how video cutlists are generated to match the synthesized audio track.
* **Reading Priority:** **Must-read (Video Blueprint).** Section 4 directly describes what our codebase implements.

---

### 4. Modern Self-Supervised Representations (The Latent Space Upgrade)

#### `2022_WavLM.pdf`

* **Title:** *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing*
* **Authors:** Sanyuan Chen, Chengyi Wang, Zhengyang Chen, Yu Wu, Shujie Liu, Zhuo Chen, Jinyu Li, Naoyuki Kanda, Takuya Yoshioka, Xiong Xiao, et al. (IEEE JSTSP 2022)
* **What it is:** Microsoft's pre-trained transformer model trained on 94,000 hours of unlabelled speech using masked speech denoising.
* **Why it matters:** **This replaces 1990s MFCC math with deep-learning embeddings.** Instead of manually balancing weights for pitch, loudness, and formants, extracting a 768-dimensional latent vector from WavLM encodes vocal tract geometry, speaker identity, and room acoustics in a unified vector space.
* **Reading Priority:** **Medium.** Read the Introduction and Model Architecture to understand why WavLM retains speaker timbre and acoustic environment.

#### `2023_Voice Conversion With Just Nearest Neighbors_2305.18975v1.pdf`

* **Title:** *Voice Conversion With Just Nearest Neighbors* (kNN-VC)
* **Authors:** Matthew Baas, Kevin Sisman, Herman Kamper (Interspeech 2023)
* **What it is:** A landmark paper demonstrating that pure concatenative synthesis using simple distance in WavLM latent space outperforms complex neural generative models.
* **Why it matters:** Proves mathematically that **cosine distance in WavLM representation space directly measures acoustic join smoothness**:

$$C_j(u_1, u_2) = 1 - \cos(\mathbf{z}_{\text{exit}}, \mathbf{z}_{\text{entry}})$$



It provides empirical proof that matching WavLM vectors yields natural transitions without needing pitch modification or vocoder retraining.
* **Reading Priority:** **High.** Short, direct, and validates using WavLM cosine similarity as our join cost function.

---

## Suggested Reading Order

1. **`1996_HuntB96-speechsynth.pdf`** — Read the 4 pages to master the $C_t + C_j$ Viterbi formulation.
2. **`2019_Text-based Editing of Talking-head Video.pdf`** — Read Section 4 to see how the Viterbi formulation controls video cuts and timeline assembly.
3. **`2023_Voice Conversion With Just Nearest Neighbors_2305.18975v1.pdf`** — Read Sections 1–3 to see how WavLM cosine distance modernizes the join cost metric.
4. **`2017_Montreal_Forced_Aligner_Trainable_Text-Speech_Alig.pdf`** — Skim to understand how MFA constructs the acoustic intervals that feed the entire system.