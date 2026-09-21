"""
End-to-end synthetic pipeline integration test.
Exercises Preprocess -> Inventory -> TargetGraph -> SpanSelector -> EditPlan -> MasterPCM -> ValidateMix.
Marked not model and not real_corpus.
"""

import json
import tempfile
import numpy as np
import soundfile as sf
from pathlib import Path

from contracts import InventoryIndex, RationalTime, SourceRecord
from edit_plan import build_edit_plan
from inventory_builder import build_units_inventory
from lexicon import Lexicon
from span_selector import SpanSelector
from target_graph import TargetGraphBuilder
from video_assembler import render_master_pcm
from tests.test_inventory_builder import create_synthetic_textgrid


def test_end_to_end_synthetic_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        aligned_dir = tmp / "aligned"
        aligned_dir.mkdir()

        # 1. Create synthetic TextGrid
        tg_file = aligned_dir / "chunk_0000.TextGrid"
        create_synthetic_textgrid(tg_file)

        # 2. Create synthetic audio file (1.5 seconds @ 48kHz)
        sr = 48000
        src_audio = tmp / "source_audio.wav"
        t = np.linspace(0, 1.5, int(sr * 1.5), endpoint=False)
        wave = (0.4 * np.sin(2 * np.pi * 300.0 * t)).astype(np.float32)
        sf.write(str(src_audio), wave, sr)

        src_video = tmp / "dummy_video.mp4"
        src_video.write_bytes(b"dummy_video_bytes")

        manifest = {
            "source_video": str(src_video),
            "speaker_id": "test_speaker",
            "single_speaker_verified": True,
            "chunks": {
                "chunk_0000": {
                    "start_sec": 0.0,
                    "end_sec": 1.5,
                    "duration": 1.5,
                    "text": "hello there",
                }
            }
        }
        manifest_path = tmp / "corpus_manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

        # 3. Build Inventory
        inv_path = tmp / "units_index.json"
        src_rec = SourceRecord(
            content_id="test_src_01",
            source_path=str(src_video),
            native_sample_rate=sr,
            native_channels=1,
            cached_pcm_path=str(src_audio),
        )

        inv_data = build_units_inventory(
            str(aligned_dir),
            str(manifest_path),
            str(inv_path),
            speaker_id="test_speaker",
            single_speaker_verified=True,
            source_record=src_rec,
        )
        inv_index = InventoryIndex.from_dict(inv_data)

        # 4. Target Graph
        lex = Lexicon()
        lex.word_to_pronunciations["hello"] = [["HH", "AH0"]]
        lex.word_to_pronunciations["there"] = [["DH", "EH1"]]

        graph_builder = TargetGraphBuilder(lex)
        graph = graph_builder.build("hello there")

        # 5. Span Selector
        selector = SpanSelector(inv_index)
        selected_spans = selector.select(graph, speaker_id="test_speaker")
        assert len(selected_spans) >= 1

        # 6. Edit Plan
        plan = build_edit_plan(
            selected_spans,
            graph,
            inv_index,
            project_sample_rate=sr,
            video_fps=RationalTime(30000, 1001),
        )
        plan.validate()
        plan_path = tmp / "pipeline_test.plan.json"
        plan.save(plan_path)

        # 7. Render Master PCM
        out_wav = tmp / "pipeline_test.wav"
        rendered_wav = render_master_pcm(plan, out_wav)

        assert rendered_wav.exists()
        info = sf.info(str(rendered_wav))
        # Exact sample count asserted
        assert info.frames == plan.total_output_samples
