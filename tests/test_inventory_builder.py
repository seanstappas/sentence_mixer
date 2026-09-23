"""
Tests for inventory building from TextGrids.
Verifies silence event retention, event chaining, and schema compliance.
"""

import json
import tempfile
from pathlib import Path
from inventory_builder import build_units_inventory
from contracts import INVENTORY_SCHEMA_VERSION, SourceRecord


def create_synthetic_textgrid(tg_path: Path):
    """Creates a sample Praat Long Format TextGrid with phone and word tiers."""
    content = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0
xmax = 1.5
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "words"
        xmin = 0
        xmax = 1.5
        intervals: size = 3
        intervals [1]:
            xmin = 0
            xmax = 0.2
            text = ""
        intervals [2]:
            xmin = 0.2
            xmax = 0.8
            text = "hello"
        intervals [3]:
            xmin = 0.8
            xmax = 1.5
            text = "there"
    item [2]:
        class = "IntervalTier"
        name = "phones"
        xmin = 0
        xmax = 1.5
        intervals: size = 6
        intervals [1]:
            xmin = 0
            xmax = 0.2
            text = "sil"
        intervals [2]:
            xmin = 0.2
            xmax = 0.4
            text = "HH"
        intervals [3]:
            xmin = 0.4
            xmax = 0.8
            text = "AH0"
        intervals [4]:
            xmin = 0.8
            xmax = 1.0
            text = "DH"
        intervals [5]:
            xmin = 1.0
            xmax = 1.4
            text = "EH1"
        intervals [6]:
            xmin = 1.4
            xmax = 1.5
            text = "sil"
"""
    tg_path.write_text(content, encoding="utf-8")


def test_build_inventory_silence_and_chaining():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        aligned_dir = tmp_path / "aligned"
        aligned_dir.mkdir()

        tg_file = aligned_dir / "chunk_0000.TextGrid"
        create_synthetic_textgrid(tg_file)

        manifest = {
            "source_video": "dummy.mp4",
            "speaker_id": "test_speaker",
            "single_speaker_verified": True,
            "chunks": {
                "chunk_0000": {
                    "start_sec": 10.0,
                    "end_sec": 11.5,
                    "duration": 1.5,
                    "text": "hello there",
                }
            }
        }
        manifest_file = tmp_path / "corpus_manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")

        output_index = tmp_path / "units_index.json"

        # Explicit synthetic source record to avoid invoking ffmpeg on non-existent dummy.mp4
        synth_source = SourceRecord(
            content_id="synth_source_1",
            source_path="dummy.mp4",
            native_sample_rate=48000,
            native_channels=2,
        )

        inv_data = build_units_inventory(
            str(aligned_dir),
            str(manifest_file),
            str(output_index),
            source_record=synth_source,
        )

        assert inv_data["schema_version"] == INVENTORY_SCHEMA_VERSION
        assert "events" in inv_data
        assert "words" in inv_data

        events = inv_data["events"]
        # Total 6 events from intervals [1] to [6]
        assert len(events) == 6

        # Invariant: silence events must be retained in ordered stream
        sil_events = [ev for ev in events.values() if ev["kind"] == "silence"]
        assert len(sil_events) == 2, "Both boundary silence events must be retained"

        # Invariant: event chaining (predecessor and successor IDs)
        event_list = list(events.values())
        for i in range(len(event_list) - 1):
            curr = event_list[i]
            next_ev = event_list[i + 1]
            assert curr["successor_id"] == next_ev["event_id"]
            assert next_ev["predecessor_id"] == curr["event_id"]

        # Check word occurrences
        words = inv_data["words"]
        assert len(words) == 2
        hello_w = next(w for w in words.values() if w["word_normalized"] == "hello")
        assert hello_w["realized_phones"] == ["HH", "AH0"]
        assert len(hello_w["phone_event_ids"]) == 2


def test_t14_nested_duplicate_chunk_basenames():
    """T14: Nested duplicate chunk basenames across batching thresholds -> Every submitted chunk remains distinct and accounted for."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        aligned_dir = tmp_path / "aligned"
        (aligned_dir / "spk_a").mkdir(parents=True)
        (aligned_dir / "spk_b").mkdir(parents=True)

        tg_a = aligned_dir / "spk_a" / "chunk_0001.TextGrid"
        tg_b = aligned_dir / "spk_b" / "chunk_0001.TextGrid"
        create_synthetic_textgrid(tg_a)
        create_synthetic_textgrid(tg_b)

        manifest = {
            "source_video": "dummy.mp4",
            "speaker_id": "multi",
            "chunks": {
                "spk_a/chunk_0001": {
                    "start_sec": 0.0,
                    "end_sec": 1.5,
                    "text": "hello there",
                    "speaker_id": "spk_a",
                },
                "spk_b/chunk_0001": {
                    "start_sec": 10.0,
                    "end_sec": 11.5,
                    "text": "hello there",
                    "speaker_id": "spk_b",
                },
            }
        }
        manifest_file = tmp_path / "corpus_manifest.json"
        manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
        output_index = tmp_path / "units_index.json"

        synth_source = SourceRecord(
            content_id="src_nested",
            source_path="dummy.mp4",
            native_sample_rate=48000,
            native_channels=2,
        )

        inv_data = build_units_inventory(
            str(aligned_dir),
            str(manifest_file),
            str(output_index),
            source_record=synth_source,
        )

        # Both chunks must be preserved as distinct utterances
        utterances = inv_data["utterances"]
        assert len(utterances) == 2, f"Expected 2 distinct utterances for nested duplicate basenames, got {len(utterances)}"
        assert "spk_a/chunk_0001" in utterances
        assert "spk_b/chunk_0001" in utterances

        # Words should be indexed from both chunks (2 words per chunk = 4 words total)
        assert len(inv_data["words"]) == 4, f"Expected 4 words across both utterances, got {len(inv_data['words'])}"

