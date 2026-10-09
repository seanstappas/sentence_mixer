"""
Adversarial Challenge Test Suite for Milestone 1 (B01, B02, B03).
Author: Challenger M1-1 (teamwork_preview_challenger_m1_1)

Tests:
1. Adversarial XML export into deeply nested fresh folders with custom XML filenames
   and different directories from the render destination, verifying <pathurl> resolves
   to the published master WAV and audio clips exist.
2. Adversarial error path: generate_fcp_xml with missing inspect_wav_path raises FileNotFoundError.
3. Completion manifest integrity under both span and legacy selectors, with --audio-track all
   and standard tracks, including unusual filenames with spaces and symbols.
4. Zero duplicate keys and 100% SHA-256 byte-for-byte verification of all manifest artifacts.
5. Identity governance with single-speaker, multi-speaker (3+ speakers), and absent speaker
   requests across span and legacy selectors, strictly asserting exit code 2 and stderr content.
6. Strict legacy speaker filtering: multi-speaker inventory requested with speaker-a selects
   exclusively speaker-a candidates and 0 from other speakers.
7. Cache immutability: requesting an absent speaker NEVER invalidates, modifies, or relabels
   cached sources (verified via recursive directory SHA-256 snapshots).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import pytest
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
PYTHON = sys.executable
RUN_MIXER = REPO_ROOT / "run_mixer.py"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from fcp_xml_generator import generate_fcp_xml
from edit_plan import EditPlan, PlanSegment


def create_synthetic_media(workspace: Path, filename: str = "source.mp4", duration_sec: float = 1.0, sr: int = 16000) -> Path:
    media_path = workspace / filename
    media_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_sec, int(duration_sec * sr), endpoint=False)
    audio = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    wav_path = workspace / f"{Path(filename).stem}.wav"
    sf.write(str(wav_path), audio, sr)

    cmd = [
        "ffmpeg", "-y",
        "-f", "lavfi", "-i", f"color=c=magenta:s=320x240:r=30:d={duration_sec}",
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
        "world": ["W", "ER1", "L", "D"],
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


def compute_tree_snapshot(directory: Path) -> dict[str, tuple[int, int, str]]:
    """Recursively computes {rel_path: (size, mtime_ns, sha256)} for all files in dir."""
    snapshot = {}
    if not directory.exists():
        return snapshot
    for p in sorted(directory.rglob("*")):
        if p.is_file():
            rel = str(p.relative_to(directory))
            st = p.stat()
            h = hashlib.sha256(p.read_bytes()).hexdigest()
            snapshot[rel] = (st.st_size, st.st_mtime_ns, h)
    return snapshot


# ============================================================================
# Adversarial Challenge 1: XML Export into Deeply Nested Directories & Pathurl
# ============================================================================

@pytest.mark.parametrize("selector", ["span", "legacy"])
def test_adversarial_xml_deeply_nested_custom_name_pathurl(tmp_path: Path, selector: str):
    """
    Stress-test XML generation where:
    - Destination directory is brand new, deeply nested, and distinct from render folder.
    - Custom XML file has spaces and non-standard naming.
    - Verified that <pathurl> points to the published master WAV and the WAV actually exists.
    """
    media = create_synthetic_media(tmp_path / "media", "media_adv.mp4", duration_sec=1.0)
    inv_file = tmp_path / "inv" / "units.json"
    create_synthetic_inventory_file(inv_file, media, ["adv_speaker"], {"adv_speaker": ["hello", "there"]})

    render_dir = tmp_path / "renders" / "run_01"
    xml_dir = tmp_path / "nested" / "deep" / "exports" / "final project"
    render_mp4 = render_dir / "adversarial mix.mp4"
    custom_xml = xml_dir / "my sequence cut (v1.2).xml"

    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "adv_speaker",
        "--text", "hello there",
        "--selector", selector,
        "--render", str(render_mp4),
        "--xml", str(custom_xml),
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"CLI mix failed under {selector}: {res.stderr}\n{res.stdout}"
    assert custom_xml.exists(), f"Custom XML missing at {custom_xml}"

    tree = ET.parse(str(custom_xml))
    root = tree.getroot()

    audio_clips = root.findall(".//media/audio/track/clipitem")
    assert len(audio_clips) >= 1, f"No audio clipitem found in XML ({selector})"

    pathurl_elem = root.find(".//media/audio/track/clipitem/file/pathurl")
    assert pathurl_elem is not None, "pathurl element missing in audio clipitem"
    pathurl_raw = pathurl_elem.text or ""
    assert pathurl_raw.startswith("file://localhost/"), f"Malformed pathurl: {pathurl_raw}"

    # Extract target file path from pathurl
    # file://localhost/D:/path...
    url_path = pathurl_raw.replace("file://localhost/", "")
    parsed_path = Path(unquote(url_path))

    expected_master_wav = (render_dir / "adversarial mix.wav").resolve()
    assert parsed_path.resolve() == expected_master_wav, (
        f"pathurl does not match published master WAV!\nParsed:   {parsed_path.resolve()}\nExpected: {expected_master_wav}"
    )
    assert expected_master_wav.exists(), f"Published master WAV does not exist at {expected_master_wav}"
    assert expected_master_wav.stat().st_size > 0, "Published master WAV is empty"


def test_adversarial_xml_generator_missing_inspect_wav_raises_filenotfound(tmp_path: Path):
    """
    Directly challenge generate_fcp_xml with a non-existent inspect_wav_path.
    Must strictly raise FileNotFoundError and never emit audio tracks with fake/missing audio.
    """
    dummy_plan = EditPlan(
        target_text="hello",
        segments=[
            PlanSegment(
                segment_index=0,
                source_content_id="src1",
                speaker_id="spk",
                source_start_sample=0,
                source_end_sample=8000,
                output_start_sample=0,
                output_end_sample=8000,
                target_text="hello",
                target_phones=["HH", "AH0"],
                span_type="word",
                source_video_path="fake.mp4",
                gain_db=0.0,
            )
        ],
        total_output_samples=8000,
        total_video_frames=15,
        project_sample_rate=16000,
    )
    fake_staged_wav = tmp_path / "does_not_exist.wav"
    dest_xml = tmp_path / "out.xml"

    with pytest.raises(FileNotFoundError, match="Master audio file not found for XML inspection"):
        generate_fcp_xml(
            dummy_plan,
            output_xml_path=str(dest_xml),
            sequence_name="test_seq",
            master_wav_path=str(tmp_path / "published.wav"),
            inspect_wav_path=str(fake_staged_wav),
        )


# ============================================================================
# Adversarial Challenge 2: Completion Manifest Completeness & SHA-256 Hashes
# ============================================================================

@pytest.mark.parametrize("selector,track_mode", [
    ("span", "all"),
    ("span", "raw"),
    ("legacy", "raw"),
])
def test_adversarial_manifest_completeness_and_sha256(tmp_path: Path, selector: str, track_mode: str):
    """
    Adversarially challenge completion manifest generation:
    - Multiple track configurations: --audio-track all (span) and raw/standard (span, legacy)
    - Verify:
      1. Completion manifest exists.
      2. Zero duplicate keys in manifest artifacts.
      3. Every artifact key points to an existing file.
      4. SHA-256 hash matches 100% byte-for-byte with actual file on disk.
      5. For --audio-track all, both .mp4 and .wav stems exist without key collision.
    """
    media = create_synthetic_media(tmp_path / "media", f"media_{selector}_{track_mode}.mp4", duration_sec=1.0)
    inv_file = tmp_path / f"inv_{selector}_{track_mode}.json"
    create_synthetic_inventory_file(inv_file, media, ["spk_manifest"], {"spk_manifest": ["hello", "world"]})

    out_dir = tmp_path / f"deliverables_{selector}_{track_mode}"
    render_mp4 = out_dir / "complex mix_name #1.mp4"
    render_xml = out_dir / "complex mix_name #1.xml"

    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "spk_manifest",
        "--text", "hello world",
        "--selector", selector,
        "--audio-track", track_mode,
        "--render", str(render_mp4),
        "--xml", str(render_xml),
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 0, f"mix failed for {selector} with audio-track {track_mode}:\n{res.stderr}\n{res.stdout}"

    manifest_p = out_dir / "complex mix_name #1.manifest.json"
    assert manifest_p.exists(), f"Completion manifest missing at {manifest_p}"

    manifest_data = json.loads(manifest_p.read_text(encoding="utf-8"))
    assert "artifacts" in manifest_data, "Manifest missing 'artifacts' key"
    artifacts = manifest_data["artifacts"]

    keys = list(artifacts.keys())
    assert len(keys) == len(set(keys)), f"Duplicate artifact keys found: {keys}"

    # Verify primary keys
    assert "render" in artifacts
    assert "master_wav" in artifacts
    assert "xml" in artifacts
    if selector == "span":
        assert "plan" in artifacts, f"plan artifact missing in span mode: {keys}"

    if track_mode == "all":
        assert "vocals_render" in artifacts, f"vocals_render missing in {keys}"
        assert "vocals_wav" in artifacts, f"vocals_wav missing in {keys}"
        assert "recombined_render" in artifacts, f"recombined_render missing in {keys}"
        assert "recombined_wav" in artifacts, f"recombined_wav missing in {keys}"

    # Verify every file on disk matches its SHA-256 exactly
    for key, info in artifacts.items():
        art_path = Path(info["path"])
        assert art_path.exists(), f"Artifact '{key}' claims path '{art_path}' which does not exist"
        computed_sha = hashlib.sha256(art_path.read_bytes()).hexdigest()
        assert computed_sha == info["sha256"], f"SHA-256 hash mismatch for artifact key '{key}' at {art_path}"


# ============================================================================
# Adversarial Challenge 3: Identity Governance & Exit Code 2 Assertions
# ============================================================================

def test_adversarial_absent_speaker_single_inventory_exits_2(tmp_path: Path):
    """Absent speaker on single-speaker inventory must exit 2 with informative stderr."""
    media = create_synthetic_media(tmp_path, "spk_single.mp4", duration_sec=1.0)
    inv_file = tmp_path / "inv_single.json"
    create_synthetic_inventory_file(inv_file, media, ["only_luke"], {"only_luke": ["hello"]})

    for sel in ["span", "legacy"]:
        cmd = [
            PYTHON, str(RUN_MIXER), "mix",
            "--inventory", str(inv_file),
            "--speaker", "darth_vader",
            "--text", "hello",
            "--selector", sel,
            "--non-interactive",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        assert res.returncode == 2, f"Expected exit code 2, got {res.returncode} for selector {sel}"
        combined = (res.stderr + res.stdout).lower()
        assert "only_luke" in combined, f"Available speaker 'only_luke' not mentioned in error output ({sel})"
        assert "darth_vader" in combined, f"Requested speaker 'darth_vader' not mentioned in error output ({sel})"


def test_adversarial_multi_speaker_inventory_governance(tmp_path: Path):
    """
    Multi-speaker inventory with 3 speakers (alpha, beta, gamma):
    - Request without --speaker -> exits 2 listing all 3 speakers.
    - Request with absent speaker delta -> exits 2 listing all 3 speakers.
    - Request with valid speaker beta -> exits 0.
    """
    media = create_synthetic_media(tmp_path, "spk_trio.mp4", duration_sec=1.0)
    inv_file = tmp_path / "inv_trio.json"
    create_synthetic_inventory_file(
        inv_file,
        media,
        ["spk_alpha", "spk_beta", "spk_gamma"],
        {
            "spk_alpha": ["hello"],
            "spk_beta": ["hello", "world"],
            "spk_gamma": ["world"],
        },
    )

    # 1. Missing --speaker
    cmd_no_spk = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--text", "hello",
        "--selector", "span",
        "--non-interactive",
    ]
    res_no = subprocess.run(cmd_no_spk, capture_output=True, text=True)
    assert res_no.returncode == 2, f"Expected exit code 2 when --speaker omitted for multi-speaker, got {res_no.returncode}"
    comb_no = (res_no.stderr + res_no.stdout).lower()
    assert "multiple speakers found in inventory" in comb_no
    assert "spk_alpha" in comb_no and "spk_beta" in comb_no and "spk_gamma" in comb_no

    # 2. Absent speaker
    cmd_absent = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "spk_delta",
        "--text", "hello",
        "--selector", "span",
        "--non-interactive",
    ]
    res_ab = subprocess.run(cmd_absent, capture_output=True, text=True)
    assert res_ab.returncode == 2, f"Expected exit code 2 for absent speaker, got {res_ab.returncode}"
    comb_ab = (res_ab.stderr + res_ab.stdout).lower()
    assert "spk_delta" in comb_ab
    assert "spk_alpha" in comb_ab and "spk_beta" in comb_ab and "spk_gamma" in comb_ab

    # 3. Valid speaker succeeds
    out_valid = tmp_path / "valid_render.mp4"
    cmd_valid = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "spk_beta",
        "--text", "hello",
        "--selector", "span",
        "--render", str(out_valid),
        "--non-interactive",
    ]
    res_val = subprocess.run(cmd_valid, capture_output=True, text=True)
    assert res_val.returncode == 0, f"Valid speaker mix failed:\n{res_val.stderr}\n{res_val.stdout}"
    assert out_valid.exists()


def test_adversarial_legacy_speaker_isolation(tmp_path: Path):
    """
    Stress-test legacy selector with multi-speaker inventory:
    Ensure only requested speaker's words and phones are synthesized.
    """
    media = create_synthetic_media(tmp_path, "spk_legacy_iso.mp4", duration_sec=1.0)
    inv_file = tmp_path / "inv_legacy_iso.json"
    create_synthetic_inventory_file(
        inv_file,
        media,
        ["spk_one", "spk_two"],
        {
            "spk_one": ["hello"],
            "spk_two": ["there"],
        },
    )

    out_mp4 = tmp_path / "legacy_iso_fail.mp4"
    # Request word 'there' with speaker 'spk_one', who only has 'hello' (HH, AH0, L, OW1)
    # 'there' has phones (DH, EH1, R) - zero overlap with spk_one.
    # Legacy selector must not leak 'there' or phones from 'spk_two'.
    cmd_fail = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "spk_one",
        "--text", "there",
        "--selector", "legacy",
        "--render", str(out_mp4),
        "--non-interactive",
    ]
    res_fail = subprocess.run(cmd_fail, capture_output=True, text=True)
    # Must fail synthesis because spk_one has 0 matching words and 0 matching phones
    assert res_fail.returncode != 0, f"Legacy leaked another speaker's words/phones! Return code: {res_fail.returncode}"
    assert not out_mp4.exists(), "Output was erroneously rendered using leaked speech units from spk_two!"

    # Positive control: spk_one requesting 'hello' succeeds cleanly
    out_spk1 = tmp_path / "legacy_spk1.mp4"
    cmd_ok1 = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "spk_one",
        "--text", "hello",
        "--selector", "legacy",
        "--render", str(out_spk1),
        "--non-interactive",
    ]
    res_ok1 = subprocess.run(cmd_ok1, capture_output=True, text=True)
    assert res_ok1.returncode == 0, f"Valid speaker mix failed:\n{res_ok1.stderr}\n{res_ok1.stdout}"
    assert out_spk1.exists(), "spk_one render output missing"

    # Positive control: spk_two requesting 'there' succeeds cleanly
    out_spk2 = tmp_path / "legacy_spk2.mp4"
    cmd_ok2 = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "spk_two",
        "--text", "there",
        "--selector", "legacy",
        "--render", str(out_spk2),
        "--non-interactive",
    ]
    res_ok2 = subprocess.run(cmd_ok2, capture_output=True, text=True)
    assert res_ok2.returncode == 0, f"Valid speaker mix failed:\n{res_ok2.stderr}\n{res_ok2.stdout}"
    assert out_spk2.exists(), "spk_two render output missing"


# ============================================================================
# Adversarial Challenge 4: Cache Immutability Under Absent Speaker Requests
# ============================================================================

def test_adversarial_absent_speaker_never_invalidates_or_relabels_cache(tmp_path: Path):
    """
    Verify that requesting an absent speaker against an existing cached source
    leaves the entire cache directory byte-for-byte identical:
    - 0 files created
    - 0 files modified
    - 0 files deleted
    - 0 timestamp changes
    - 0 SHA-256 changes
    """
    cache_root = tmp_path / "cache_store"
    cache_root.mkdir(parents=True)
    source_dir = cache_root / "sources" / "src_test_cache"
    source_dir.mkdir(parents=True)

    media = create_synthetic_media(tmp_path, "cached_video.mp4", duration_sec=1.0)
    inv_file = source_dir / "units_index.json"
    create_synthetic_inventory_file(inv_file, media, ["original_speaker"], {"original_speaker": ["hello", "world"]})

    # Add mock alignment and chunk files to simulate real cached directory
    mock_aligned_dir = source_dir / "aligned"
    mock_aligned_dir.mkdir()
    (mock_aligned_dir / "chunks.txt").write_text("chunk_001\toriginal_speaker\n", encoding="utf-8")
    (mock_aligned_dir / "alignment_meta.json").write_text('{"status": "aligned", "model": "mfa"}', encoding="utf-8")

    # Record full directory snapshot before challenge
    snapshot_before = compute_tree_snapshot(cache_root)
    assert len(snapshot_before) >= 3, "Setup failed: snapshot before challenge is too small"

    # Attempt mix requesting absent speaker 'intruder_speaker'
    cmd = [
        PYTHON, str(RUN_MIXER), "mix",
        "--inventory", str(inv_file),
        "--speaker", "intruder_speaker",
        "--text", "hello",
        "--cache-dir", str(cache_root),
        "--selector", "span",
        "--non-interactive",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode == 2, f"Expected exit code 2, got {res.returncode}:\n{res.stdout}\n{res.stderr}"

    # Record full directory snapshot after challenge
    snapshot_after = compute_tree_snapshot(cache_root)

    # Compare snapshots
    added_files = set(snapshot_after.keys()) - set(snapshot_before.keys())
    removed_files = set(snapshot_before.keys()) - set(snapshot_after.keys())

    assert len(added_files) == 0, f"Cache directory was polluted with new files: {added_files}"
    assert len(removed_files) == 0, f"Cache directory had files deleted: {removed_files}"

    for rel_path, before_meta in snapshot_before.items():
        after_meta = snapshot_after[rel_path]
        b_size, b_mtime, b_sha = before_meta
        a_size, a_mtime, a_sha = after_meta

        assert b_size == a_size, f"File size changed for {rel_path}: {b_size} -> {a_size}"
        assert b_mtime == a_mtime, f"File mtime changed for {rel_path}: {b_mtime} -> {a_mtime}"
        assert b_sha == a_sha, f"File content SHA-256 changed for {rel_path}: {b_sha} -> {a_sha}"

    # Also check units_index.json text directly
    inv_text = inv_file.read_text(encoding="utf-8")
    assert "intruder_speaker" not in inv_text, "Cached inventory was relabeled to intruder_speaker!"
    assert "original_speaker" in inv_text, "Original speaker label was wiped!"
