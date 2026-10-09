**The four improvements make sense and fit your goal. However, I would revise these documents before having an agent implement them.** The direction is good; several implementation details would produce incorrect speaker assignments, pronunciations, or timing.

I reviewed all three documents against the current code and checked the proposed models’ documentation. I also ran small deterministic checks of the proposed parser, scoring formula, inventory serialization, and MFA batching behavior. **No repository files were changed.**

## 1. Speaker detection: good goal, but distinguish clustering from identity

The best practical approach here is:

**Speaker metadata + local diarization → reviewable speaker clusters → persistent speaker identities → conservative word/span eligibility.**

### What needs changing

**Do not use the proposed 60% dominant-speaker rule.**

`The plan` assigns an entire chunk to its majority speaker. A chunk containing 60% person A and 40% person B would therefore contribute B’s words to A’s inventory. That directly contradicts the single-speaker requirement.

Instead:

- Preserve turn boundaries and overlap information.
- Reconcile those boundaries with aligned words/phones.
- Split into trustworthy single-speaker units where possible.
- Mark boundary-crossing or uncertain units ineligible until reviewed.
- Check the actual selected source interval, including any splice handles—not just its speaker label.

**Diarization does not automatically identify characters or match people across videos.**

`SPEAKER_00` means a cluster within a particular recording. It does not mean “Obi-Wan,” and another video’s `SPEAKER_00` may be someone else. Add a persistent speaker registry: audition representative clips, name/merge/split clusters, and save mappings. Speaker embeddings can suggest cross-video matches, but should support rejection and confirmation rather than silently merging voices.

**VTT labels are useful evidence, not guaranteed ground truth.**

Remove the `100%-precision claim`. Subtitles can be inaccurate, loosely timed, or contain multiple speakers. Also, the proposed regex requires `</v>`, but standard [WebVTT examples](https://www.w3.org/TR/webvtt1/#introduction-caption) omit that closing tag. I confirmed the proposed expression misses those cues.

**Do not require vocal separation before every diarization run.**

Compare original audio versus separated audio on representative material. Separation can help with music, but can also damage speaker cues. The `claimed improvement to below 8% DER` is not supported by evidence in the document.

Crucially, a vocal stem generally still contains **all talking people**. Neither vocal isolation nor diarization removes an overlapping speaker from the waveform. Overlapping speech should normally be excluded from clean original-audio donor clips.

### Better current model baseline

I would evaluate **[pyannote Community-1](https://huggingface.co/pyannote/speaker-diarization-community-1)** rather than start a new integration around 3.1. Its official documentation reports improved speaker counting/assignment and supports local, offline use.

Keep both its regular overlap-aware output and its “exclusive” assignment output: exclusive labels simplify transcript reconciliation but do **not** make overlapping audio clean.

Pin the model/library versions, test the Windows dependency stack, and keep acquisition explicit. Do not substitute the hosted Precision-2 service for a local pipeline. No model is guaranteed best for your cinematic material; measure wrong-speaker donor acceptance and usable coverage, not only overall DER.

## 2. Phonetic and microphone input: worthwhile, but the stress/mapping proposal is wrong

Explicit phonetic groups are a good addition. However, **accepting phonemes in the target parser is only half the implementation.**

### Whole-word retrieval needs a pronunciation index

The current `selector` retrieves whole words by their written spelling. Adding `{B AE1 K AH0 N}` to the graph will not automatically find an intact recorded word with those phones.

Add a reverse index from compatible realized pronunciations to recorded words and contiguous phone runs. Preserve whole-word/phrase candidates for phonetic input instead of unnecessarily falling back to individual phones. A phonetic group can also represent a nonce word; it need not correspond to an existing dictionary entry.

### The proposed tokenizer silently accepts malformed input

Executing the `proposed tokenizer` showed:

- `hello {B AE1` becomes ordinary word tokens instead of an unmatched-brace error.
- `{NOT_A_PHONE}` is accepted as phonetic input.
- `<gap:250ms>` becomes a normal lexical token rather than a gap.

Use one validated parser for words, phonetic groups, punctuation, and explicit gaps. It must consume the entire input or report the offending location. Validate phones and stress against the configured inventory’s phone alphabet.

### Do not assign primary stress to every recognized vowel

`This proposed step` should be removed. It would turn unstressed vowels into stressed ones and undermine the reason for microphone input.

The proposed [Wav2Vec2 phone recognizer](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft) emits multilingual phonetic labels—not CMU ARPAbet with reliable English lexical stress. Its [actual vocabulary](https://huggingface.co/facebook/wav2vec2-lv-60-espeak-cv-ft/raw/main/vocab.json) makes that distinction clear.

A better workflow is:

1. Capture audio, or accept an existing recording through the same interface.
2. Produce phone alternatives with uncertainty information.
3. Convert supported sounds through an explicit, tested mapping.
4. Preserve unknown stress; obtain it from a known word, a separate estimate, or user correction.
5. Show/play back the result for confirmation.

CTC avoids some spelling bias, but it is **not a perfect physical-sound transcriber**. It still has training, language, and accent biases, and an English phone inventory cannot represent every sound effect or vocalization.

Ordinary MFA forced alignment is also not interchangeable with unconstrained microphone recognition: it aligns a supplied transcript/pronunciation.

Finally, the execution steps only connect the microphone to the OOV dialog. Your broader request needs a direct **“record target pronunciation”** workflow too, with editable grouping, playback/retry, silence/clipping checks, and optional reuse as an analysis-only prosody guide. The microphone waveform must never become output speech.

## 3. Punctuation: separate exact timing controls from delivery preferences

Explicit gap controls and soft prosody preferences are sensible. The current proposal mixes them too tightly.

### Gaps

Use two distinct concepts:

- **Explicit gap:** an exact requested duration, such as `<gap:250ms>`.
- **Punctuation boundary:** a configurable preference for phrasing, pauses, or delivery.

The proposed 180/350/650 ms values can be presets, not universal rules. Define whether a requested gap replaces an existing source pause or adds to it; otherwise intact phrases can acquire doubled pauses.

Compile these through the existing canonical timeline. Do not maintain separate independently advanced sample/frame clocks or introduce a competing `EditCut` timeline. This must build on the scheduler repair in `the remediation plan`.

A freeze-frame is a useful option, not guaranteed natural-looking editing. Hold an eligible source frame explicitly; do not inadvertently advance into neighboring dialogue. Account for punctuation inside whole-phrase candidates, not only between selected clips.

### Questions and exclamations

**“Question = rising pitch” is too simplistic.** Yes/no questions often rise; ordinary wh-questions commonly fall. [Cambridge’s intonation reference](https://dictionary.cambridge.org/grammar/british-grammar/intonation) illustrates the distinction.

Use weak phrase-level preferences, with optional explicit rise/fall/emphasis controls or guide-derived contours. Compare confident voiced regions in relative semitones rather than raw Hz slopes across an arbitrary clip.

**Lexical stress is not expressive emphasis, and raw RMS is not a reliable shouting detector.** Recording gain, background music, compression, and microphone distance all affect RMS. Use speaker/recording-normalized energy alongside pitch range, duration, and voicing observations.

There is also a concrete problem in the `proposed energy formula`: measured RMS `0.1` adds a `1.35` penalty, while missing RMS defaults to `1.0` and incurs none. It can therefore prefer missing data. Missing observations must remain missing, and prosody costs must not reward arbitrary phone/span partitioning.

## 4. Interactive learning: yes, but broader than an OOV prompt

Default-interactive behavior is a good fit. I would distinguish four situations:

| Situation | Useful interaction |
|---|---|
| Unknown spelling/pronunciation | Accept, edit, or record pronunciation |
| Ambiguous known word, such as “read” | Choose the pronunciation for this occurrence |
| Uncertain speaker identity | Audition and name/merge/split speaker clusters |
| Uncertain source realization/join | Audition alternative recorded candidates |

A word absent from the dictionary is not the same as a sound absent from the source corpus. Saving a pronunciation cannot create missing recordings.

Important persistence changes:

- Return structured lookup provenance: dictionary, custom entry, G2P suggestion, or unresolved—not just a list of phones.
- Separate “use once” from “save as a reusable variant.” Do not globally overwrite a heteronym because of one sentence.
- Validate, deduplicate, and atomically save approved entries, including concurrent-write handling.
- Make target lookup and MFA consume the same versioned merged lexicon. Currently `alignment chooses the merged dictionary`; appending to the custom dictionary does **not** automatically update that alignment dictionary.
- In noninteractive mode, use an explicit documented guess/error policy and report unconfirmed choices. “Ignore” must never mean silently omit the target word.

Keep prompts at the CLI/UI boundary, with terminal detection and cancellation handling. The lower-level pipeline should remain usable without stdin.

## 5. The implementation snippets do not match the repository

This is another reason not to hand the plan to an agent unchanged.

It references `Lexicon.lookup`, `self.entries`, `GraphNode`, and `EditCut`, but the current implementation uses:

- `get_pronunciations`
- `word_to_pronunciations`
- `TargetPhoneNode`
- `PlanSegment`

Those are not cosmetic differences: blindly implementing the snippets could create parallel dictionaries and timelines that existing selection/rendering never use. The design also incorrectly describes diphone fallback as existing functionality; the current baseline is whole-phone fallback.

## 6. Verdict on Gemini’s five bug claims

| Claim | Assessment and appropriate fix |
|---|---|
| **1. Infinite isolation-cache invalidation** | **Not established as described.** The vocal-path field already exists and is serialized by `the contracts`; I verified that directly. Failed probing or incorrect stem discovery can still produce bad cache state, but that needs a reproduction. Fix dependency/provenance validation, not an allegedly missing serialization field. |
| **2. Nonrecursive MFA globbing** | **Real nested-directory integration gap, but the explanation and fix are incomplete.** With no flat WAV matches, `the current code` still invokes MFA once—it does not skip alignment. Changing to `rglob` while flattening filenames can overwrite same-named chunks and lose speaker structure. Preserve relative paths/stable IDs, and update `inventory discovery` too. |
| **3. Isolation metadata mismatch** | **Real provenance bug; claimed rendering impact is overstated.** Record requested/effective processing accurately. However, `the current renderer` selects stems through the explicit audio-track argument, so the false metadata flag is not presently what makes it bypass vocals. |
| **4. Explicit speaker overridden** | **Confirmed, important bug.** The suggested precedence fix is necessary, but insufficient: also reject unavailable requested speakers and ambiguous/unverified automatic choices. Falling back to unrestricted pooling is not acceptable. |
| **5. Dead Path expression** | **Confirmed, harmless cleanup.** Remove `the unused expression`; do not invent a directory-creation requirement merely because it is there. |

## Recommended direction

Use the existing `audit remediation` as the foundation, especially speaker eligibility, legal pronunciation coverage, real acoustic scoring, and cumulative timing.

Then build these enhancements around **one shared target representation**:

**Text / explicit phones / reviewed microphone recognition → pronunciation groups + boundary instructions → legal selection graph → one finalized edit plan.**

Speaker discovery should feed the same validated source inventory, and interactive review should serve all uncertain decisions rather than being a separate OOV-only feature.

**Bottom line:** keep all four capabilities. Replace the majority-speaker labeling, fabricated stress, universal punctuation/prosody assumptions, and disconnected persistence/API examples. The remaining work is mostly making the design truthful and consistent—not finding one newer neural model that solves everything.