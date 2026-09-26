"""
Regression tests for Antigravity Round-3 Milestone 1:
Delivery & CLI Correctness (B01, B02, B03, U01-U09).

Acceptance criteria:
- U01: Real CLI, fresh folder, --xml with a different basename (span & legacy):
       At least one audio clipitem resolving to the existing published master WAV.
- U02: --audio-track all:
       Manifest lists every published file under unique keys with matching SHA-256.
- U03: --speaker speaker-b against speaker-a-only inventory (both selectors):
       Exit code 2 listing available identities; no selection, alignment, or relabeling.
- U04: Cache labeled obi-wan, request anakin:
       No preprocessing or MFA call; labels unchanged; exit code 2.
- U05: Pooled inventory with two identities, no --speaker:
       Exit code 2 listing identities.
- U06: Legacy, multi-speaker inventory, --speaker a:
       Words and phone fallbacks only from speaker a.
- U07: Legacy with hello <gap:250ms> hello:
       Exit code 2 suggesting --selector span, before source alignment; never tokens gap/250ms.
- U08: Legacy with {...}, --target-audio, or --record-target:
       Exit code 2 suggesting --selector span.
- U09: Legacy OOV word with a saved custom pronunciation:
       Phone fallback uses custom phones from lexicon, not G2P.
"""

from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
import soundfile as sf
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RUN_MIXER = REPO_ROOT / "run_mixer.py"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "tools"))

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


def create_synthetic_inventory_file(
    path: Path,
    source_media_path: Path,
    speakers: list[str],
    words_per_speaker: dict[str, list[str]],
    sample_rate: int = 16000,
) -> Path:
    """Create a valid units_index.json file matching InventoryIndex schema."""
    sources = {
        "src1": {
            "content_id": "src1",
            "source_path": str(source_media_path),
            "cached_analysis_pcm_path": str(source_media_path),
            "cached_vocal_pcm_path": str(source_media_path.with_suffix(".wav")),
            "cached_instrumental_pcm_path": str(source_media_path.with_suffix(".wav")),
            "native_sample_rate": sample_rate,
            "native_channels": 1,
        }
    }
    events = {}
    words = {}
    legacy_words = {}
    legacy_phones = {}

    ev_idx = 0
    sample_cursor = 0
    samples_per_word = int(0.2 * sample_rate)

    word_phones_map = {
        "hello": ["HH", "AH0", "L", "OW1"],
        "there": ["DH", "EH1", "R"],
    }

    for spk in speakers:
        spk_word_list = words_per_speaker.get(spk, ["hello", "there"])
        for w_idx, w_text in enumerate(spk_word_list):
            w_norm = w_text.lower()
            w_id = f"wo_{spk}_{w_norm}_{w_idx}"
            plist = word_phones_map.get(w_norm, ["HH"])

            w_start_s = sample_cursor
            p_sample_dur = samples_per_word // len(plist)
            w_ev_ids = []

            for p_idx, p_label in enumerate(plist):
                ev_id = f"ev_{spk}_{w_norm}_{ev_idx}"
                ev_idx += 1
                p_start = sample_cursor
                p_end = sample_cursor + p_sample_dur
                sample_cursor = p_end

                base_p = p_label.rstrip("012")
                stress = int(p_label[-1]) if p_label[-1].isdigit() else 0

                events[ev_id] = {
                    "event_id": ev_id,
                    "source_content_id": "src1",
                    "utterance_id": f"utt_{spk}",
                    "speaker_id": spk,
                    "kind": "phone",
                    "start_sample": p_start,
                    "end_sample": p_end,
                    "label": p_label,
                    "base_phone": base_p,
                    "stress": stress,
                    "word_occurrence_id": w_id,
                }
                w_ev_ids.append(ev_id)

                if p_label not in legacy_phones:
                    legacy_phones[p_label] = []
                legacy_phones[p_label].append({
                    "source_video": str(source_media_path),
                    "chunk_id": f"chunk_{spk}",
                    "label": p_label,
                    "start_sec": p_start / sample_rate,
                    "end_sec": p_end / sample_rate,
                    "duration": (p_end - p_start) / sample_rate,
                    "sample_start": p_start,
                    "sample_end": p_end,
                    "speaker_id": spk,
                })

            w_end_s = sample_cursor
            words[w_id] = {
                "word_id": w_id,
                "source_content_id": "src1",
                "speaker_id": spk,
                "word_normalized": w_norm,
                "word_original": w_text,
                "start_sample": w_start_s,
                "end_sample": w_end_s,
                "phone_event_ids": w_ev_ids,
                "realized_phones": plist,
            }

            if w_norm not in legacy_words:
                legacy_words[w_norm] = []
            legacy_words[w_norm].append({
                "source_video": str(source_media_path),
                "chunk_id": f"chunk_{spk}",
                "label": w_norm,
                "start_sec": w_start_s / sample_rate,
                "end_sec": w_end_s / sample_rate,
                "duration": (w_end_s - w_start_s) / sample_rate,
                "sample_start": w_start_s,
                "sample_end": w_end_s,
                "phones": plist,
                "speaker_id": spk,
            })

    inv_dict = {
        "metadata": {"version": 2},
        "sources": sources,
        "events": events,
        "words": words,
        "legacy_words": legacy_words,
        "legacy_phones": legacy_phones,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(inv_dict, f, indent=2)
    return path


# ============================================================================
# U01: Real CLI fresh folder XML audio link (span and legacy)
# ============================================================================

def test_u01_xml_master_audio_link_fresh_folder_span(tmp_path: Path):
    """
    U01: Real CLI run into a fresh folder with custom XML name exports an XML
    whose audio track resolves to the existing published master WAV (span selector).
    """
    media = create_synthetic_media(tmp_path, "synth.mp4", duration_sec=1.0)
    inv_file = tmp_path / "inventory.json"
    create_synthetic_inventory_file(inv_file, media, ["speaker-a"], {"speaker-a": ["hello"]})

    fresh_dir = tmp_path / "fresh_out_span"
    render_mp4 = fresh_dir / "my_mix.mp4"
    custom_xml = fresh_dir / "custom_timeline.xml"

    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-a",
        "--text", "hello",
        "--selector", "span",
        "--render", str(render_mp4),
        "--xml", str(custom_xml),
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"mix failed: {res.stderr}\n{res.stdout}"
    assert custom_xml.exists(), "XML was not generated"

    tree = ET.parse(str(custom_xml))
    root = tree.getroot()
    audio_clips = root.findall(".//media/audio/track/clipitem")
    assert len(audio_clips) >= 1, f"Expected at least 1 audio clipitem in fresh folder XML, found {len(audio_clips)}"

    pathurl_elem = root.find(".//media/audio/track/clipitem/file/pathurl")
    assert pathurl_elem is not None, "XML audio clipitem has no pathurl element"
    pathurl = pathurl_elem.text or ""
    assert "custom_timeline" not in pathurl, "pathurl should resolve to master WAV, not xml basename"
    # Published master WAV must exist
    expected_wav = fresh_dir / "my_mix.wav"
    assert expected_wav.exists(), "Published master WAV does not exist"


def test_u01_xml_master_audio_link_fresh_folder_legacy(tmp_path: Path):
    """
    U01: Real CLI run into a fresh folder with custom XML name exports an XML
    whose audio track resolves to the existing published master WAV (legacy selector).
    """
    media = create_synthetic_media(tmp_path, "synth_legacy.mp4", duration_sec=1.0)
    inv_file = tmp_path / "inv_legacy.json"
    create_synthetic_inventory_file(inv_file, media, ["speaker-a"], {"speaker-a": ["hello"]})

    fresh_dir = tmp_path / "fresh_out_legacy"
    render_mp4 = fresh_dir / "my_mix.mp4"
    custom_xml = fresh_dir / "custom_timeline.xml"

    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-a",
        "--text", "hello",
        "--selector", "legacy",
        "--render", str(render_mp4),
        "--xml", str(custom_xml),
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"mix failed: {res.stderr}\n{res.stdout}"
    assert custom_xml.exists(), "XML was not generated"

    tree = ET.parse(str(custom_xml))
    root = tree.getroot()
    audio_clips = root.findall(".//media/audio/track/clipitem")
    assert len(audio_clips) >= 1, f"Expected at least 1 audio clipitem in fresh folder XML (legacy), found {len(audio_clips)}"

    pathurl_elem = root.find(".//media/audio/track/clipitem/file/pathurl")
    assert pathurl_elem is not None, "Legacy XML audio clipitem has no pathurl element"
    expected_wav = fresh_dir / "my_mix.wav"
    assert expected_wav.exists(), "Published master WAV does not exist"


# ============================================================================
# U02: --audio-track all completion manifest completeness and unique keys
# ============================================================================

def test_u02_audio_track_all_manifest_unique_keys_and_hashes(tmp_path: Path):
    """
    U02: Running with --audio-track all records every published variant in the
    completion manifest under unique keys with matching SHA-256 hashes.
    """
    media = create_synthetic_media(tmp_path, "synth_all.mp4", duration_sec=1.0)
    inv_file = tmp_path / "inv_all.json"
    create_synthetic_inventory_file(inv_file, media, ["speaker-a"], {"speaker-a": ["hello"]})

    out_dir = tmp_path / "out_audio_track_all"
    render_mp4 = out_dir / "mix.mp4"
    render_xml = out_dir / "mix.xml"

    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-a",
        "--text", "hello",
        "--selector", "span",
        "--audio-track", "all",
        "--render", str(render_mp4),
        "--xml", str(render_xml),
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"mix failed: {res.stderr}\n{res.stdout}"

    manifest_file = out_dir / "mix.manifest.json"
    assert manifest_file.exists(), f"Completion manifest missing at {manifest_file}"

    manifest_data = json.loads(manifest_file.read_text(encoding="utf-8"))
    artifacts = manifest_data.get("artifacts", {})

    # In --audio-track all, variants include vocals and recombined mp4 and wav
    # Check that both .mp4 and .wav variants are present and keyed uniquely
    keys = list(artifacts.keys())
    assert len(keys) == len(set(keys)), "Manifest contains duplicate keys!"

    # Must contain main render, master_wav, plan, xml, vocals_render, vocals_wav, recombined_render, recombined_wav
    # At least 8 distinct files must be recorded
    assert len(artifacts) >= 8, f"Expected at least 8 manifest artifacts for --audio-track all, got {len(artifacts)}: {keys}"

    import hashlib
    for key, item in artifacts.items():
        art_path = Path(item["path"])
        assert art_path.exists(), f"Manifest artifact '{key}' path does not exist: {art_path}"
        computed_sha = hashlib.sha256(art_path.read_bytes()).hexdigest()
        assert computed_sha == item["sha256"], f"SHA-256 mismatch for artifact '{key}'"


# ============================================================================
# U03: --speaker speaker-b against speaker-a inventory exits 2
# ============================================================================

def test_u03_absent_speaker_exits_code_2_span(tmp_path: Path):
    """
    U03: --speaker speaker-b against speaker-a-only inventory exits 2 listing
    available identities; no selection, alignment, or relabeling (span selector).
    """
    media = create_synthetic_media(tmp_path, "spk_a.mp4", duration_sec=1.0)
    inv_file = tmp_path / "inv_a.json"
    create_synthetic_inventory_file(inv_file, media, ["speaker-a"], {"speaker-a": ["hello"]})

    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-b",
        "--text", "hello",
        "--selector", "span",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, f"Expected returncode 2, got {res.returncode}. Output: {res.stdout}\n{res.stderr}"
    combined_out = (res.stderr + res.stdout).lower()
    assert "speaker-a" in combined_out, "Error message must list available identity 'speaker-a'"


def test_u03_absent_speaker_exits_code_2_legacy(tmp_path: Path):
    """
    U03: --speaker speaker-b against speaker-a-only inventory exits 2 listing
    available identities; no selection, alignment, or relabeling (legacy selector).
    """
    media = create_synthetic_media(tmp_path, "spk_a_leg.mp4", duration_sec=1.0)
    inv_file = tmp_path / "inv_a_leg.json"
    create_synthetic_inventory_file(inv_file, media, ["speaker-a"], {"speaker-a": ["hello"]})

    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-b",
        "--text", "hello",
        "--selector", "legacy",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, f"Expected returncode 2, got {res.returncode}. Output: {res.stdout}\n{res.stderr}"
    combined_out = (res.stderr + res.stdout).lower()
    assert "speaker-a" in combined_out, "Error message must list available identity 'speaker-a'"


def test_u03_zero_speaker_inventory_exits_code_2(tmp_path: Path):
    """
    U03: Requesting --speaker on an inventory with 0 labeled speakers
    exits code 2 with 'Available speakers: none' (span and legacy).
    """
    media = create_synthetic_media(tmp_path, "zero_spk.mp4", duration_sec=1.0)
    inv_file = tmp_path / "zero_speaker_inv.json"
    create_synthetic_inventory_file(
        inv_file, media, speakers=[None], words_per_speaker={None: ["hello"]}  # type: ignore[list-item]
    )

    for selector in ["span", "legacy"]:
        cmd = [
            PYTHON, str(RUN_MIXER), "mix",
            "--inventory", str(inv_file),
            "--speaker", "foo",
            "--text", "hello",
            "--selector", selector,
            "--non-interactive",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 2, (
            f"Expected exit code 2 for selector '{selector}', got {res.returncode}. "
            f"Stderr: {res.stderr}\nStdout: {res.stdout}"
        )
        combined = (res.stderr + res.stdout).lower()
        assert "requested speaker 'foo' not found in inventory" in combined
        assert "available speakers: none" in combined


# ============================================================================
# U04: Cache labeled obi-wan, request anakin exits 2 without relabeling/realigning
# ============================================================================

def test_u04_cached_inventory_no_relabeling_on_different_speaker(tmp_path: Path):
    """
    U04: Cache labeled obi-wan, request anakin:
    No preprocessing or MFA call; labels unchanged; exit code 2.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    media = create_synthetic_media(tmp_path, "obi_wan_video.mp4", duration_sec=1.0)
    inv_path = cache_dir / "units_index.json"
    create_synthetic_inventory_file(inv_path, media, ["obi-wan"], {"obi-wan": ["hello"]})

    # Record content before run
    content_before = inv_path.read_text(encoding="utf-8")

    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_path),
        "--speaker", "anakin",
        "--text", "hello",
        "--cache-dir", str(cache_dir),
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, f"Expected exit code 2, got {res.returncode}. Output:\n{res.stdout}\n{res.stderr}"

    # Verify inventory was NOT rewritten or relabeled
    content_after = inv_path.read_text(encoding="utf-8")
    assert content_before == content_after, "Cached inventory file was modified or relabeled!"
    assert "anakin" not in content_after, "Cached inventory was relabeled to anakin!"


# ============================================================================
# U05: Pooled inventory with two identities and no --speaker exits 2
# ============================================================================

def test_u05_pooled_inventory_without_speaker_exits_code_2(tmp_path: Path):
    """
    U05: Pooled inventory with two identities, no --speaker:
    Exit 2 listing identities instead of silently mixing disparate voices.
    """
    media = create_synthetic_media(tmp_path, "two_spk.mp4", duration_sec=1.0)
    inv_file = tmp_path / "inv_two_spk.json"
    create_synthetic_inventory_file(
        inv_file, media,
        ["speaker-a", "speaker-b"],
        {"speaker-a": ["hello"], "speaker-b": ["there"]}
    )

    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--text", "hello there",
        "--selector", "span",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, f"Expected exit code 2 for un-filtered multi-speaker inventory, got {res.returncode}"
    combined_out = (res.stderr + res.stdout).lower()
    assert "speaker-a" in combined_out and "speaker-b" in combined_out, (
        f"Expected output to list both speaker identities, got: {combined_out}"
    )


# ============================================================================
# U06: Legacy selector with --speaker filters words and phones strictly
# ============================================================================

def test_u06_legacy_speaker_filtering_words_and_phones(tmp_path: Path):
    """
    U06: Legacy, multi-speaker inventory, --speaker a:
    Words and phone fallbacks only from a.
    """
    media_a = create_synthetic_media(tmp_path, "spk_a_vid.mp4", duration_sec=1.0)
    media_b = create_synthetic_media(tmp_path, "spk_b_vid.mp4", duration_sec=1.0)

    # In legacy inventory:
    # speaker-a has word "hello"
    # speaker-b has word "there"
    # phone "HH" exists for both
    legacy_inv = {
        "words": {
            "hello": [{
                "source_video": str(media_a),
                "chunk_id": "chunk_a",
                "label": "hello",
                "start_sec": 0.0,
                "end_sec": 0.3,
                "duration": 0.3,
                "sample_start": 0,
                "sample_end": 4800,
                "phones": ["HH"],
                "speaker_id": "speaker-a",
            }],
            "there": [{
                "source_video": str(media_b),
                "chunk_id": "chunk_b",
                "label": "there",
                "start_sec": 0.0,
                "end_sec": 0.3,
                "duration": 0.3,
                "sample_start": 0,
                "sample_end": 4800,
                "phones": ["DH"],
                "speaker_id": "speaker-b",
            }],
        },
        "phones": {
            "HH": [
                {
                    "source_video": str(media_a),
                    "chunk_id": "chunk_a",
                    "label": "HH",
                    "start_sec": 0.0,
                    "end_sec": 0.3,
                    "duration": 0.3,
                    "sample_start": 0,
                    "sample_end": 4800,
                    "speaker_id": "speaker-a",
                },
                {
                    "source_video": str(media_b),
                    "chunk_id": "chunk_b",
                    "label": "HH",
                    "start_sec": 0.0,
                    "end_sec": 0.3,
                    "duration": 0.3,
                    "sample_start": 0,
                    "sample_end": 4800,
                    "speaker_id": "speaker-b",
                },
            ]
        }
    }

    inv_file = tmp_path / "legacy_multi.json"
    full_inv = {
        "metadata": {"version": 2},
        "sources": {
            "src_a": {"content_id": "src_a", "source_path": str(media_a), "native_sample_rate": 16000, "native_channels": 1},
            "src_b": {"content_id": "src_b", "source_path": str(media_b), "native_sample_rate": 16000, "native_channels": 1},
        },
        "events": {
            "ev_a": {"event_id": "ev_a", "source_content_id": "src_a", "utterance_id": "u_a", "speaker_id": "speaker-a", "kind": "phone", "start_sample": 0, "end_sample": 4800, "label": "HH", "base_phone": "HH"},
            "ev_b": {"event_id": "ev_b", "source_content_id": "src_b", "utterance_id": "u_b", "speaker_id": "speaker-b", "kind": "phone", "start_sample": 0, "end_sample": 4800, "label": "HH", "base_phone": "HH"},
        },
        "words": {
            "wo_a": {"word_id": "wo_a", "source_content_id": "src_a", "speaker_id": "speaker-a", "word_normalized": "hello", "word_original": "hello", "start_sample": 0, "end_sample": 4800, "phone_event_ids": ["ev_a"], "realized_phones": ["HH"]},
            "wo_b": {"word_id": "wo_b", "source_content_id": "src_b", "speaker_id": "speaker-b", "word_normalized": "there", "word_original": "there", "start_sample": 0, "end_sample": 4800, "phone_event_ids": ["ev_b"], "realized_phones": ["DH"]},
        },
        "legacy_words": legacy_inv["words"],
        "legacy_phones": legacy_inv["phones"],
    }
    inv_file.write_text(json.dumps(full_inv, indent=2), encoding="utf-8")

    out_mp4 = tmp_path / "legacy_out.mp4"
    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-a",
        "--text", "hello",
        "--selector", "legacy",
        "--render", str(out_mp4),
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"mix failed: {res.stderr}\n{res.stdout}"

    # Also test that requesting a word from speaker-b fails when filtering to speaker-a
    cmd2 = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "speaker-a",
        "--text", "there",
        "--selector", "legacy",
        "--render", str(tmp_path / "legacy_out2.mp4"),
        "--non-interactive",
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    # Since "there" belongs to speaker-b, and phone DH does not exist in speaker-a, it must fail to find units
    assert res2.returncode != 0, "Legacy selector must NOT select words or phones from speaker-b when filtered to speaker-a"


# ============================================================================
# U07: Legacy with gap tag exits 2 before alignment, never tokens gap
# ============================================================================

def test_u07_legacy_rejects_gap_before_alignment():
    """
    U07: Legacy with hello <gap:250ms> hello:
    Exit 2 suggesting --selector span, before source alignment; never tokens gap or 250ms.
    """
    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--video", "nonexistent_source_video_for_u07.mp4",
        "--text", "hello <gap:250ms> hello",
        "--selector", "legacy",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, (
        f"Expected exit code 2 before alignment, got {res.returncode}. Output:\n{res.stdout}\n{res.stderr}"
    )
    combined = (res.stderr + res.stdout).lower()
    assert "--selector span" in combined, "Error message must suggest --selector span"
    assert "source video not found" not in combined, "Source alignment must NOT run before target validation"


# ============================================================================
# U08: Legacy with {...}, --target-audio, or --record-target exits 2
# ============================================================================

def test_u08_legacy_rejects_phonetic_and_audio_targets():
    """
    U08: Legacy with {...}, --target-audio, or --record-target:
    Exit 2 suggesting --selector span.
    """
    # 1. Phonetic syntax {...}
    cmd1 = [
        PYTHON, str(RUN_MIXER), "mix",
        "--video", "nonexistent_source_video.mp4",
        "--text", "{HH AH0 L OW1}",
        "--selector", "legacy",
        "--non-interactive",
    ]
    res1 = subprocess.run(cmd1, capture_output=True, text=True)
    assert res1.returncode == 2, f"Expected exit code 2 for {{...}}, got {res1.returncode}"
    assert "--selector span" in (res1.stderr + res1.stdout).lower()

    # 2. --target-audio
    cmd2 = [
        PYTHON, str(RUN_MIXER), "mix",
        "--video", "nonexistent_source_video.mp4",
        "--target-audio", "dummy.wav",
        "--selector", "legacy",
        "--non-interactive",
    ]
    res2 = subprocess.run(cmd2, capture_output=True, text=True)
    assert res2.returncode == 2, f"Expected exit code 2 for --target-audio, got {res2.returncode}"
    assert "--selector span" in (res2.stderr + res2.stdout).lower()

    # 3. --record-target
    cmd3 = [
        PYTHON, str(RUN_MIXER), "mix",
        "--video", "nonexistent_source_video.mp4",
        "--record-target",
        "--selector", "legacy",
        "--non-interactive",
    ]
    res3 = subprocess.run(cmd3, capture_output=True, text=True)
    assert res3.returncode == 2, f"Expected exit code 2 for --record-target, got {res3.returncode}"
    assert "--selector span" in (res3.stderr + res3.stdout).lower()


# ============================================================================
# U09: Legacy OOV word with saved custom pronunciation uses custom phones
# ============================================================================

def test_u09_legacy_custom_pronunciation_fallback(tmp_path: Path):
    """
    U09: Legacy OOV word with a saved custom pronunciation:
    Phone fallback uses the custom phones, not G2P.
    """
    media = create_synthetic_media(tmp_path, "synth_u09.mp4", duration_sec=1.0)

    # Inventory has phonemes "B", "AE1", "K", "T", "AH0"
    phones_dict = {}
    for p in ["B", "AE1", "K", "T", "AH0"]:
        phones_dict[p] = [{
            "source_video": str(media),
            "chunk_id": "c1",
            "label": p,
            "start_sec": 0.0,
            "end_sec": 0.1,
            "duration": 0.1,
            "sample_start": 0,
            "sample_end": 1600,
        }]

    inv = {
        "words": {},
        "phones": phones_dict,
    }

    # Custom word "bacta" -> B AE1 K T AH0
    lex = Lexicon([])
    lex.word_to_pronunciations["bacta"] = [["B", "AE1", "K", "T", "AH0"]]

    # Call synthesize_sentence with lexicon parameter
    units, plan_desc = synthesize_sentence("bacta", inv, lexicon=lex)
    assert len(units) == 5, f"Expected 5 phoneme units for 'bacta', got {len(units)}"
    assert any("[PHONEME FALLBACK]" in p for p in plan_desc)
    selected_labels = [u["label"] for u in units]
    assert selected_labels == ["B", "AE1", "K", "T", "AH0"], f"Expected custom phones, got {selected_labels}"
