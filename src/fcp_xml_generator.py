#!/usr/bin/env python3
"""
Final Cut Pro 7 XML (xmeml v4) Generator for Adobe Premiere Pro
Exports frame-accurate multi-cut sequence timelines referencing original source media.
"""

import os
import xml.etree.ElementTree as ET
from xml.dom import minidom
from typing import List, Dict, Any


def generate_fcp_xml(
    units: List[Dict[str, Any]],
    output_xml_path: str,
    sequence_name: str = "YTP_Sentence_Mix",
    fps: float = 29.97,
    width: int = 1920,
    height: int = 1080
) -> str:
    """
    Generate an FCP7 XML sequence containing video and audio tracks for the selected units.

    Each unit in `units` should be a dict:
      {
        "source_video": str (absolute path),
        "start_sec": float,
        "end_sec": float,
        "label": str,
        "type": "word" | "phone"
      }
    """
    timebase = int(round(fps))
    ntsc_flag = "TRUE" if abs(fps - 29.97) < 0.05 or abs(fps - 23.976) < 0.05 else "FALSE"

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

    # Audio tracks (L & R stereo)
    audio = ET.SubElement(media, "audio")
    atrack1 = ET.SubElement(audio, "track")
    atrack2 = ET.SubElement(audio, "track")

    # File cache to reuse file elements by absolute path
    file_elements = {}
    file_counter = 1

    curr_timeline_frame = 0

    for idx, unit in enumerate(units):
        src_path = os.path.abspath(unit["source_video"])
        start_sec = max(0.0, unit["start_sec"])
        end_sec = max(start_sec + (1.0 / fps), unit["end_sec"])
        label = unit.get("label", f"Unit_{idx}")
        unit_type = unit.get("type", "clip")

        in_frame = int(round(start_sec * fps))
        out_frame = max(in_frame + 1, int(round(end_sec * fps)))
        duration_frames = out_frame - in_frame

        # Register file node if not already seen
        if src_path not in file_elements:
            file_id = f"file-{file_counter}"
            file_counter += 1
            f_node = ET.Element("file", id=file_id)
            ET.SubElement(f_node, "name").text = os.path.basename(src_path)
            # Premiere Pro pathurl format: file://localhost/C:/...
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

        # Append file reference
        vclip.append(f_def_node)

        # Audio clipitem Track 1 (Left)
        aclip1 = ET.SubElement(atrack1, "clipitem", id=f"aclip1-{idx}")
        ET.SubElement(aclip1, "name").text = f"{idx:03d}_{label}_audio1"
        ET.SubElement(aclip1, "start").text = str(curr_timeline_frame)
        ET.SubElement(aclip1, "end").text = str(curr_timeline_frame + duration_frames)
        ET.SubElement(aclip1, "in").text = str(in_frame)
        ET.SubElement(aclip1, "out").text = str(out_frame)
        f_ref1 = ET.Element("file", id=file_id)
        aclip1.append(f_ref1)

        # Audio clipitem Track 2 (Right)
        aclip2 = ET.SubElement(atrack2, "clipitem", id=f"aclip2-{idx}")
        ET.SubElement(aclip2, "name").text = f"{idx:03d}_{label}_audio2"
        ET.SubElement(aclip2, "start").text = str(curr_timeline_frame)
        ET.SubElement(aclip2, "end").text = str(curr_timeline_frame + duration_frames)
        ET.SubElement(aclip2, "in").text = str(in_frame)
        ET.SubElement(aclip2, "out").text = str(out_frame)
        f_ref2 = ET.Element("file", id=file_id)
        aclip2.append(f_ref2)

        curr_timeline_frame += duration_frames

    ET.SubElement(seq, "duration").text = str(curr_timeline_frame)

    rough_string = ET.tostring(root, "utf-8")
    reparsed = minidom.parseString(rough_string)
    pretty_xml = reparsed.toprettyxml(indent="  ")

    os.makedirs(os.path.dirname(os.path.abspath(output_xml_path)), exist_ok=True)
    with open(output_xml_path, "w", encoding="utf-8") as f:
        f.write(pretty_xml)

    print(f"[+] Exported Premiere Pro XML ({len(units)} units, {curr_timeline_frame} frames) to: {output_xml_path}")
    return output_xml_path
