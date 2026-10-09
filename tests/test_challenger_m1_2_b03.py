"""
Empirical Adversarial Challenge Suite: Milestone 1 (B03)
Author: teamwork_preview_challenger_m1_2 (Challenger 2)

Verifies:
1. Legacy selector with arbitrary gap markup (<gap:0ms>, <gap:250ms>, leading/trailing/consecutive/spaces/malformed):
   Confirm exit code 2 and that NO alignment or file reading is triggered.
2. Legacy selector with arbitrary phonetic brackets ({HH AH0 L OW1}, invalid brackets, mixed words):
   Confirm exit code 2 before alignment.
3. Legacy selector with --target-audio or --record-target:
   Confirm exit code 2 before alignment.
4. Legacy fallback with custom pronunciations:
   Verify custom pronunciation overrides are chosen over default G2P.
5. Strict speaker isolation for both phone and word units:
   Verify zero phone or word leakage when filtering by speaker in legacy selector.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
import pytest
import soundfile as sf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RUN_MIXER = REPO_ROOT / "run_mixer.py"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from lexicon import Lexicon
from unit_selector import synthesize_sentence


def create_synthetic_media(workspace: Path, filename: str = "source.mp4", duration_sec: float = 1.0, sr: int = 16000) -> Path:
    media_path = workspace / filename
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = workspace / f"{Path(filename).stem}.wav"
    sf.write(str(wav_path), audio, sr)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=blue:s=320x240:r=30:d={duration_sec}",
        "-i", str(wav_path),
        "-c:v", "libx264", "-c:a", "pcm_s16le",
        str(media_path),
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return media_path


# ============================================================================
# 1. Arbitrary Gap Markup on Legacy Selector
# ============================================================================

@pytest.mark.parametrize(
    "gap_text",
    [
        "<gap:0ms>",
        "<gap:250ms>",
        "<gap:1000ms>",
        "<gap:0s>",
        "<gap:1.5s>",
        "<gap: 250ms>",
        "<gap:250 ms>",
        "<gap:0.0ms>",
        "<gap:foo>",
        "<gap:-10ms>",
        "<gap:100ms> leading",
        "trailing <gap:100ms>",
        "hello <gap:250ms> world",
        "hello <gap:50ms><gap:50ms> world",
        "<gap:0ms><gap:0ms>",
    ],
)
def test_legacy_rejects_arbitrary_gap_markup_before_alignment(gap_text: str):
    """
    Arbitrary gap markup passed to legacy selector must exit code 2 and suggest
    --selector span WITHOUT reading files or attempting alignment.
    """
    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--video", "nonexistent_source_video_for_gap_test.mp4",
        "--text", gap_text,
        "--selector", "legacy",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, (
        f"Expected exit code 2 for gap '{gap_text}', got {res.returncode}.\n"
        f"stdout: {res.stdout}\nstderr: {res.stderr}"
    )
    combined = (res.stderr + res.stdout).lower()
    assert "--selector span" in combined, f"Missing '--selector span' in output for '{gap_text}'"
    assert "filenotfounderror" not in combined, (
        f"File reading was triggered before validation for '{gap_text}'!\n{res.stderr}"
    )
    assert "source video not found" not in combined, (
        f"Alignment was triggered before validation for '{gap_text}'!\n{res.stderr}"
    )


# ============================================================================
# 2. Arbitrary Phonetic Brackets on Legacy Selector
# ============================================================================

@pytest.mark.parametrize(
    "phonetic_text",
    [
        "{HH AH0 L OW1}",
        "{B AE1 K AH0 N}",
        "{K AE1 T}",
        "{}",
        "{invalid}",
        "{123}",
        "{T1}",
        "{AE9}",
        "{HH AH0",
        "mixed words hello {HH AH0} world",
        "multiple {B AE1} {K AH0 N}",
    ],
)
def test_legacy_rejects_arbitrary_phonetic_brackets_before_alignment(phonetic_text: str):
    """
    Arbitrary phonetic brackets passed to legacy selector must exit code 2 and suggest
    --selector span WITHOUT reading files or attempting alignment.
    """
    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--video", "nonexistent_source_video_for_bracket_test.mp4",
        "--text", phonetic_text,
        "--selector", "legacy",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, (
        f"Expected exit code 2 for phonetic text '{phonetic_text}', got {res.returncode}.\n"
        f"stdout: {res.stdout}\nstderr: {res.stderr}"
    )
    combined = (res.stderr + res.stdout).lower()
    assert "--selector span" in combined, f"Missing '--selector span' in output for '{phonetic_text}'"
    assert "filenotfounderror" not in combined, (
        f"File reading was triggered before validation for '{phonetic_text}'!\n{res.stderr}"
    )
    assert "source video not found" not in combined, (
        f"Alignment was triggered before validation for '{phonetic_text}'!\n{res.stderr}"
    )


# ============================================================================
# 3. Spoken Audio Targets on Legacy Selector
# ============================================================================

def test_legacy_rejects_target_audio_before_alignment():
    """
    --target-audio with legacy selector must exit code 2 and suggest --selector span
    without attempting alignment.
    """
    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--video", "nonexistent_source_video.mp4",
        "--target-audio", "dummy_ref.wav",
        "--selector", "legacy",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, f"Expected exit code 2, got {res.returncode}.\n{res.stderr}"
    combined = (res.stderr + res.stdout).lower()
    assert "--selector span" in combined
    assert "filenotfounderror" not in combined
    assert "source video not found" not in combined


def test_legacy_rejects_record_target_before_alignment():
    """
    --record-target with legacy selector must exit code 2 and suggest --selector span
    without attempting alignment.
    """
    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--video", "nonexistent_source_video.mp4",
        "--record-target",
        "--selector", "legacy",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, f"Expected exit code 2, got {res.returncode}.\n{res.stderr}"
    combined = (res.stderr + res.stdout).lower()
    assert "--selector span" in combined
    assert "filenotfounderror" not in combined
    assert "source video not found" not in combined


def test_legacy_rejects_multiple_target_representations():
    """
    Specifying both --text and --target-audio must exit code 2 (mutual exclusivity).
    """
    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--video", "nonexistent.mp4",
        "--text", "hello",
        "--target-audio", "dummy.wav",
        "--selector", "legacy",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, f"Expected exit code 2 for conflicting options, got {res.returncode}"


# ============================================================================
# 4. Custom Pronunciation Overrides in Legacy Fallback
# ============================================================================

def test_legacy_fallback_prioritizes_custom_pronunciations_over_default_g2p(tmp_path: Path):
    """
    When an unknown/OOV word is synthesized in legacy mode, custom pronunciations
    in Lexicon must take priority over default G2P phonemes.
    """
    media = create_synthetic_media(tmp_path, "synth_custom.mp4", duration_sec=1.0)

    # Word 'gundark' - default G2P would produce ['G', 'AH1', 'N', 'D', 'AA0', 'R', 'K']
    # Custom pronunciation override: ['G', 'OW1', 'R', 'D']
    phones_dict = {}
    for p in ["G", "OW1", "R", "D", "AH1", "N", "AA0", "K"]:
        phones_dict[p] = [{
            "source_video": str(media),
            "chunk_id": "c1",
            "label": p,
            "start_sec": 0.0,
            "end_sec": 0.1,
            "duration": 0.1,
            "sample_start": 0,
            "sample_end": 1600,
            "speaker_id": "speaker-a",
        }]

    inv = {
        "words": {},
        "phones": phones_dict,
    }

    # Lexicon with custom pronunciation
    lex = Lexicon([])
    lex.word_to_pronunciations["gundark"] = [["G", "OW1", "R", "D"]]

    units, plan = synthesize_sentence("gundark", inv, lexicon=lex)
    assert len(units) == 4, f"Expected 4 units from custom pronunciation, got {len(units)}"
    selected_labels = [u["label"] for u in units]
    assert selected_labels == ["G", "OW1", "R", "D"], (
        f"Expected custom pronunciation ['G', 'OW1', 'R', 'D'], got {selected_labels}"
    )


def test_legacy_fallback_prioritizes_explicit_overrides_over_lexicon_and_g2p(tmp_path: Path):
    """
    Explicit overrides dictionary passed to synthesize_sentence must take precedence
    over both Lexicon and G2P.
    """
    media = create_synthetic_media(tmp_path, "synth_override.mp4", duration_sec=1.0)
    phones_dict = {}
    for p in ["Z", "IY1", "P", "B", "AE1", "K"]:
        phones_dict[p] = [{
            "source_video": str(media),
            "chunk_id": "c1",
            "label": p,
            "start_sec": 0.0,
            "end_sec": 0.1,
            "duration": 0.1,
            "sample_start": 0,
            "sample_end": 1600,
            "speaker_id": "speaker-a",
        }]

    inv = {"words": {}, "phones": phones_dict}
    lex = Lexicon([])
    lex.word_to_pronunciations["testword"] = [["B", "AE1", "K"]]
    explicit_overrides = {"testword": [["Z", "IY1", "P"]]}

    units, plan = synthesize_sentence("testword", inv, lexicon=lex, overrides=explicit_overrides)
    selected_labels = [u["label"] for u in units]
    assert selected_labels == ["Z", "IY1", "P"], (
        f"Explicit override must take precedence, got {selected_labels}"
    )


# ============================================================================
# 5. Strict Speaker Isolation (Zero Leakage)
# ============================================================================

def test_legacy_strict_speaker_isolation_words_and_phones(tmp_path: Path):
    """
    Verify complete speaker isolation in legacy selector:
    - Multi-speaker inventory: speaker-a and speaker-b.
    - Filtering by speaker-a selects ONLY speaker-a words and phones.
    - Attempting to select a word that only speaker-b has must FAIL,
      and its phone fallback must NOT leak speaker-b phones.
    """
    media_a = create_synthetic_media(tmp_path, "voice_a.mp4", duration_sec=1.0)
    media_b = create_synthetic_media(tmp_path, "voice_b.mp4", duration_sec=1.0)

    # speaker-a has:
    #   word: "apple"
    #   phones: "AE1", "P", "L"
    # speaker-b has:
    #   word: "banana"
    #   phones: "B", "AH0", "N"
    # both have:
    #   word: "fruit" (different source_video and chunk_id)
    #   phone: "IH0" (different source_video and chunk_id)
    legacy_inv = {
        "words": {
            "apple": [{
                "source_video": str(media_a),
                "chunk_id": "chunk_a_apple",
                "label": "apple",
                "start_sec": 0.0,
                "end_sec": 0.3,
                "duration": 0.3,
                "sample_start": 0,
                "sample_end": 4800,
                "phones": ["AE1", "P", "L"],
                "speaker_id": "speaker-a",
            }],
            "banana": [{
                "source_video": str(media_b),
                "chunk_id": "chunk_b_banana",
                "label": "banana",
                "start_sec": 0.0,
                "end_sec": 0.4,
                "duration": 0.4,
                "sample_start": 0,
                "sample_end": 6400,
                "phones": ["B", "AH0", "N"],
                "speaker_id": "speaker-b",
            }],
            "fruit": [
                {
                    "source_video": str(media_a),
                    "chunk_id": "chunk_a_fruit",
                    "label": "fruit",
                    "start_sec": 0.3,
                    "end_sec": 0.6,
                    "duration": 0.3,
                    "sample_start": 4800,
                    "sample_end": 9600,
                    "phones": ["F", "R", "UW1", "T"],
                    "speaker_id": "speaker-a",
                },
                {
                    "source_video": str(media_b),
                    "chunk_id": "chunk_b_fruit",
                    "label": "fruit",
                    "start_sec": 0.4,
                    "end_sec": 0.7,
                    "duration": 0.3,
                    "sample_start": 6400,
                    "sample_end": 11200,
                    "phones": ["F", "R", "UW1", "T"],
                    "speaker_id": "speaker-b",
                },
            ],
        },
        "phones": {
            "AE1": [{
                "source_video": str(media_a),
                "chunk_id": "chunk_a_ae",
                "label": "AE1",
                "start_sec": 0.0,
                "end_sec": 0.1,
                "duration": 0.1,
                "sample_start": 0,
                "sample_end": 1600,
                "speaker_id": "speaker-a",
            }],
            "B": [{
                "source_video": str(media_b),
                "chunk_id": "chunk_b_b",
                "label": "B",
                "start_sec": 0.0,
                "end_sec": 0.1,
                "duration": 0.1,
                "sample_start": 0,
                "sample_end": 1600,
                "speaker_id": "speaker-b",
            }],
            "IH0": [
                {
                    "source_video": str(media_a),
                    "chunk_id": "chunk_a_ih",
                    "label": "IH0",
                    "start_sec": 0.6,
                    "end_sec": 0.7,
                    "duration": 0.1,
                    "sample_start": 9600,
                    "sample_end": 11200,
                    "speaker_id": "speaker-a",
                },
                {
                    "source_video": str(media_b),
                    "chunk_id": "chunk_b_ih",
                    "label": "IH0",
                    "start_sec": 0.7,
                    "end_sec": 0.8,
                    "duration": 0.1,
                    "sample_start": 11200,
                    "sample_end": 12800,
                    "speaker_id": "speaker-b",
                },
            ],
        },
    }

    inv_file = tmp_path / "multi_speaker_inv.json"
    full_inv = {
        "metadata": {"version": 2},
        "sources": {
            "src_a": {"content_id": "src_a", "source_path": str(media_a), "native_sample_rate": 16000, "native_channels": 1},
            "src_b": {"content_id": "src_b", "source_path": str(media_b), "native_sample_rate": 16000, "native_channels": 1},
        },
        "events": {
            "ev_a": {"event_id": "ev_a", "source_content_id": "src_a", "utterance_id": "u_a", "speaker_id": "speaker-a", "kind": "phone", "start_sample": 0, "end_sample": 4800, "label": "AE1", "base_phone": "AE"},
            "ev_b": {"event_id": "ev_b", "source_content_id": "src_b", "utterance_id": "u_b", "speaker_id": "speaker-b", "kind": "phone", "start_sample": 0, "end_sample": 6400, "label": "B", "base_phone": "B"},
        },
        "words": {
            "wo_a": {"word_id": "wo_a", "source_content_id": "src_a", "speaker_id": "speaker-a", "word_normalized": "apple", "word_original": "apple", "start_sample": 0, "end_sample": 4800, "phone_event_ids": ["ev_a"], "realized_phones": ["AE1"]},
            "wo_b": {"word_id": "wo_b", "source_content_id": "src_b", "speaker_id": "speaker-b", "word_normalized": "banana", "word_original": "banana", "start_sample": 0, "end_sample": 6400, "phone_event_ids": ["ev_b"], "realized_phones": ["B"]},
        },
        "legacy_words": legacy_inv["words"],
        "legacy_phones": legacy_inv["phones"],
    }
    inv_file.write_text(json.dumps(full_inv, indent=2), encoding="utf-8")

    # 1. Synthesize speaker-a's word: "apple"
    render_a = tmp_path / "out_apple.mp4"
    cmd1 = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-a",
        "--text", "apple",
        "--selector", "legacy",
        "--render", str(render_a),
        "--non-interactive",
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert res1.returncode == 0, f"mix failed: {res1.stderr}"
    assert "[WORD] 'apple' -> 0.000s..0.300s (0.30s)" in res1.stdout

    # 2. Synthesize shared word: "fruit"
    # speaker-a's fruit is at 0.300s..0.600s, speaker-b's fruit is at 0.400s..0.700s
    render_fruit = tmp_path / "out_fruit.mp4"
    cmd2 = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-a",
        "--text", "fruit",
        "--selector", "legacy",
        "--render", str(render_fruit),
        "--non-interactive",
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode == 0, f"mix failed: {res2.stderr}"
    # Must use speaker-a's cut (0.300s..0.600s), NEVER speaker-b's cut (0.400s..0.700s)
    assert "[WORD] 'fruit' -> 0.300s..0.600s (0.30s)" in res2.stdout, (
        f"Speaker-a cut 0.300s..0.600s was not selected for 'fruit'!\nstdout:\n{res2.stdout}"
    )
    assert "0.400s..0.700s" not in res2.stdout, "Speaker-b cut leaked into speaker-a mix!"

    # 3. Requesting speaker-b's word "banana" while filtering to speaker-a:
    # Speaker-a has phone AE1, but B and N exist ONLY in speaker-b.
    # Verify that speaker-b's phones B and N NEVER leak into speaker-a's mix!
    render_banana = tmp_path / "out_banana.mp4"
    cmd3 = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-a",
        "--text", "banana",
        "--selector", "legacy",
        "--render", str(render_banana),
        "--non-interactive",
    ]
    res3 = subprocess.run(cmd3, capture_output=True, text=True)
    # Speaker-b phones B and N must NOT be used:
    assert "[!] Warning: Phoneme 'B' (B) not available in corpus!" in res3.stdout
    assert "[!] Warning: Phoneme 'N' (N) not available in corpus!" in res3.stdout
    assert "chunk_b" not in res3.stdout, "Speaker-b unit leaked into speaker-a synthesis!"

    # 3b. Requesting word whose phones exist ONLY in speaker-b (e.g. "bob"):
    # Speaker-b has phone B, speaker-a has NO matching phones.
    # Mix for speaker-a MUST fail with non-zero exit code because speaker-b's B cannot leak!
    cmd3b = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-a",
        "--text", "bob",
        "--selector", "legacy",
        "--render", str(tmp_path / "out_bob.mp4"),
        "--non-interactive",
    ]
    res3b = subprocess.run(cmd3b, capture_output=True, text=True)
    assert res3b.returncode != 0, (
        f"Legacy mix should have failed when requested phones belong exclusively to speaker-b!\n"
        f"stdout: {res3b.stdout}"
    )
    assert "chunk_b" not in res3b.stdout, "Speaker-b unit leaked into speaker-a synthesis!"

    # 4. Verify unlabeled entries (speaker_id=None) are strictly excluded when speaker is requested
    legacy_inv_unlabeled = {
        "words": {
            "mystery": [{
                "source_video": str(media_a),
                "chunk_id": "chunk_unlabeled",
                "label": "mystery",
                "start_sec": 0.0,
                "end_sec": 0.3,
                "duration": 0.3,
                "sample_start": 0,
                "sample_end": 4800,
                "phones": ["M"],
                "speaker_id": None,  # Unlabeled
            }],
        },
        "phones": {},
    }
    unlabeled_inv_file = tmp_path / "unlabeled_inv.json"
    full_inv_unlabeled = {
        "metadata": {"version": 2},
        "sources": full_inv["sources"],
        "events": {},
        "words": {},
        "legacy_words": legacy_inv_unlabeled["words"],
        "legacy_phones": {},
    }
    unlabeled_inv_file.write_text(json.dumps(full_inv_unlabeled, indent=2), encoding="utf-8")

    cmd4 = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(unlabeled_inv_file),
        "--speaker", "speaker-a",
        "--text", "mystery",
        "--selector", "legacy",
        "--render", str(tmp_path / "out_unlabeled.mp4"),
        "--non-interactive",
    ]
    res4 = subprocess.run(cmd4, capture_output=True, text=True)
    # Since mystery has speaker_id=None, it must not be selected when speaker-a is requested
    assert res4.returncode != 0, "Unlabeled word unit must NOT be selected when a speaker filter is active!"
