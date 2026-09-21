#!/usr/bin/env python3
"""
Final Cut Pro 7 XML (xmeml v4) Generator for Adobe Premiere Pro / FCP
Exports frame-accurate multi-cut sequence timelines referencing original source media
and uses authoritative master PCM audio for sample-accurate subframe fidelity.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom
from typing import Any, Dict, List, Optional, Tuple, Union

from contracts import EditPlan, RationalTime


def generate_fcp_xml(
    plan_or_units: Union[EditPlan, List[Dict[str, Any]]],
    output_xml_path: str,
    sequence_name: str = "YTP_Sentence_Mix",
    fps: Optional[Union[float, RationalTime]] = None,
    width: int = 1920,
    height: int = 1080,
    master_wav_path: Optional[str] = None,
) -> str:
    """
    Generate an FCP7 XML sequence containing video cuts and authoritative master audio.
    Consumes either an EditPlan or a legacy units list.
    """
    out_p = Path(output_xml_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(plan_or_units, EditPlan):
        plan = plan_or_units
        r_fps = plan.video_fps
        fps_float = r_fps.as_float()
        segments: List[Any] = list(plan.segments)
        # If not explicitly specified, derive master WAV from plan output stem
        if not master_wav_path:
            candidate_wav = out_p.with_suffix(".wav")
            if candidate_wav.exists():
                master_wav_path = str(candidate_wav)
    else:
        # Legacy units list
        units = plan_or_units
        if isinstance(fps, RationalTime):
            fps_float = float(fps.as_fraction())
        elif fps is not None:
            fps_float = float(fps)
        else:
            fps_float = 29.97
        r_fps = RationalTime.from_number(fps_float)
        segments = []
        for idx, u in enumerate(units):
            start_s = float(u["start_sec"])
            end_s = float(u["end_sec"])
            dur_s = max(1.0 / fps_float, end_s - start_s)
            frame_cnt = max(1, int(round(dur_s * fps_float)))
            segments.append({
                "source_video_path": str(Path(u["source_video"]).resolve()),
                "video_source_pts_sec": start_s,
                "video_frame_count": frame_cnt,
                "target_text": u.get("label", f"slice_{idx}"),
                "span_type": u.get("type", "word"),
                "is_silence": False,
            })

    timebase = int(round(fps_float))
    ntsc_flag = "TRUE" if abs(fps_float - 29.97) < 0.05 or abs(fps_float - 23.976) < 0.05 else "FALSE"

    root = ET.Element("xmeml", version="4")
    seq = ET.SubElement(root, "sequence", id="ytp-sequence-1")
    ET.SubElement(seq, "name").text = sequence_name

    seq_rate = ET.SubElement(seq, "rate")
    ET.SubElement(seq_rate, "timebase").text = str(timebase)
    ET.SubElement(seq_rate, "ntsc").text = ntsc_flag

    media = ET.SubElement(seq, "media")

    # Video track
    video = ET.SubElement(media, "video")
    v_format = ET.SubElement(video, "format")
    v_sample = ET.SubElement(v_format, "samplecharacteristics")
    ET.SubElement(v_sample, "width").text = str(width)
    ET.SubElement(v_sample, "height").text = str(height)
    v_rate = ET.SubElement(v_sample, "rate")
    ET.SubElement(v_rate, "timebase").text = str(timebase)
    ET.SubElement(v_rate, "ntsc").text = ntsc_flag

    vtrack = ET.SubElement(video, "track")

    # Audio tracks
    audio = ET.SubElement(media, "audio")
    atrack1 = ET.SubElement(audio, "track")
    atrack2 = ET.SubElement(audio, "track")

    # File cache to reuse file elements
    file_elements: Dict[str, Tuple[str, ET.Element]] = {}
    file_counter = 1

    curr_timeline_frame = 0

    for idx, seg in enumerate(segments):
        if isinstance(seg, dict):
            src_path = seg["source_video_path"]
            start_pts = seg["video_source_pts_sec"]
            duration_frames = seg["video_frame_count"]
            label = seg["target_text"]
            unit_type = seg["span_type"]
            seg["is_silence"]
        else:
            src_path = seg.source_video_path
            start_pts = seg.video_source_pts_sec
            duration_frames = seg.video_frame_count
            label = seg.target_text
            unit_type = seg.span_type

        in_frame = int(round(start_pts * fps_float))
        out_frame = in_frame + duration_frames

        # Register file node if not already seen
        if src_path not in file_elements:
            file_id = f"file-{file_counter}"
            file_counter += 1
            f_node = ET.Element("file", id=file_id)
            ET.SubElement(f_node, "name").text = os.path.basename(src_path)
            formatted_url = f"file://localhost/{src_path.replace(os.sep, '/')}"
            ET.SubElement(f_node, "pathurl").text = formatted_url
            f_rate = ET.SubElement(f_node, "rate")
            ET.SubElement(f_rate, "timebase").text = str(timebase)
            ET.SubElement(f_rate, "ntsc").text = ntsc_flag
            file_elements[src_path] = (file_id, f_node)

        file_id, f_def_node = file_elements[src_path]

        # Video clipitem
        vclip = ET.SubElement(vtrack, "clipitem", id=f"vclip-{idx}")
        ET.SubElement(vclip, "name").text = f"{idx:03d}_{label}_{unit_type}"
        ET.SubElement(vclip, "start").text = str(curr_timeline_frame)
        ET.SubElement(vclip, "end").text = str(curr_timeline_frame + duration_frames)
        ET.SubElement(vclip, "in").text = str(in_frame)
        ET.SubElement(vclip, "out").text = str(out_frame)
        ET.SubElement(vclip, "file", id=file_id)

        curr_timeline_frame += duration_frames

    # If master WAV is provided, place master WAV as the authoritative audio track
    if master_wav_path and Path(master_wav_path).exists():
        wav_path_abs = str(Path(master_wav_path).resolve())
        wav_id = "file-master-audio"
        wav_node = ET.Element("file", id=wav_id)
        ET.SubElement(wav_node, "name").text = os.path.basename(wav_path_abs)
        formatted_url = f"file://localhost/{wav_path_abs.replace(os.sep, '/')}"
        ET.SubElement(wav_node, "pathurl").text = formatted_url

        aclip1 = ET.SubElement(atrack1, "clipitem", id="aclip-master-1")
        ET.SubElement(aclip1, "name").text = "Master_Audio_L"
        ET.SubElement(aclip1, "start").text = "0"
        ET.SubElement(aclip1, "end").text = str(curr_timeline_frame)
        ET.SubElement(aclip1, "in").text = "0"
        ET.SubElement(aclip1, "out").text = str(curr_timeline_frame)
        ET.SubElement(aclip1, "file", id=wav_id)

        aclip2 = ET.SubElement(atrack2, "clipitem", id="aclip-master-2")
        ET.SubElement(aclip2, "name").text = "Master_Audio_R"
        ET.SubElement(aclip2, "start").text = "0"
        ET.SubElement(aclip2, "end").text = str(curr_timeline_frame)
        ET.SubElement(aclip2, "in").text = "0"
        ET.SubElement(aclip2, "out").text = str(curr_timeline_frame)
        ET.SubElement(aclip2, "file", id=wav_id)

    xml_str = ET.tostring(root, encoding="utf-8")
    parsed = minidom.parseString(xml_str)
    pretty_xml = parsed.toprettyxml(indent="  ")

    with open(out_p, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"[+] Exported FCP7 XML sequence to {out_p}")
    return str(out_p)
