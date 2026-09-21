"""
Tests for Final Cut Pro 7 XML generation.
Verifies timeline structure, video clipitems, and master audio track references.
"""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from contracts import EditPlan, PlanSegment, RationalTime, SourceRecord
from fcp_xml_generator import generate_fcp_xml


def test_generate_fcp_xml_from_edit_plan():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        xml_path = tmp_path / "test_sequence.xml"
        fake_wav = tmp_path / "master.wav"
        fake_wav.write_text("fake audio header", encoding="utf-8")

        src_rec = SourceRecord(
            content_id="src_1",
            source_path="D:/Code/sentence_mixer/input/video1.mp4",
            native_sample_rate=48000,
            native_channels=2,
        )

        seg1 = PlanSegment(
            segment_index=0,
            source_content_id="src_1",
            speaker_id="spk1",
            source_start_sample=0,
            source_end_sample=24000,
            output_start_sample=0,
            output_end_sample=24000,
            target_text="hello",
            target_phones=[],
            span_type="word",
            source_video_path=src_rec.source_path,
            video_frame_count=15,
            video_source_pts_sec=0.0,
        )

        plan = EditPlan(
            plan_id="test_plan_xml",
            target_text="hello",
            speaker_id="spk1",
            video_fps=RationalTime(30000, 1001),
            total_output_samples=24000,
            total_video_frames=15,
            segments=[seg1],
            sources={"src_1": src_rec},
        )
        plan.validate()

        result_path = generate_fcp_xml(plan, str(xml_path), master_wav_path=str(fake_wav))
        assert Path(result_path).exists()

        tree = ET.parse(result_path)
        root = tree.getroot()
        assert root.tag == "xmeml"
        assert root.attrib.get("version") == "4"

        # Check sequence timebase
        timebase = root.find(".//sequence/rate/timebase")
        assert timebase is not None and timebase.text == "30"

        # Check video clipitem
        vclips = root.findall(".//media/video/track/clipitem")
        assert len(vclips) == 1
        assert "hello" in vclips[0].find("name").text

        # Check master audio clipitem
        aclips = root.findall(".//media/audio/track/clipitem")
        assert len(aclips) >= 1
        assert any("Master_Audio" in ac.find("name").text for ac in aclips)
