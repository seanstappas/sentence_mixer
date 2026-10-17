#!/usr/bin/env python3
"""
Final Cut Pro 7 XML (xmeml v4) Generator for Adobe Premiere Pro / FCP
Exports frame-accurate multi-cut sequence timelines referencing original source media
and uses authoritative master PCM audio for sample-accurate subframe fidelity.
"""

from __future__ import annotations

import os
import wave
import xml.etree.ElementTree as ET
from pathlib import Path
from xml.dom import minidom
from typing import Any, Dict, List, Optional, Union

from contracts import EditPlan, RationalTime


def generate_fcp_xml(
    plan_or_units: Union[EditPlan, List[Dict[str, Any]]],
    output_xml_path: str,
    sequence_name: str = "YTP_Sentence_Mix",
    fps: Optional[Union[float, RationalTime]] = None,
    width: int = 1920,
    height: int = 1080,
    master_wav_path: Optional[str] = None,
    inspect_wav_path: Optional[str] = None,
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
        # If not explicitly specified, derive master WAV from plan output stem.
        # This fallback is a last resort — callers should pass master_wav_path
        # explicitly so the WAV is always resolved relative to the render output,
        # not relative to the (possibly different) XML output directory.
        if not master_wav_path:
            candidate_wav = out_p.with_suffix(".wav")
            if candidate_wav.exists():
                import warnings
                warnings.warn(
                    f"[fcp_xml_generator] master_wav_path not supplied; "
                    f"falling back to '{candidate_wav}' (inferred from XML basename). "
                    "Pass master_wav_path explicitly to avoid path mismatches.",
                    stacklevel=3,
                )
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

    # File cache to reuse file elements
    file_elements: Dict[str, str] = {}
    file_counter = 1

    curr_timeline_frame = 0

    for idx, seg in enumerate(segments):
        if isinstance(seg, dict):
            src_path = seg["source_video_path"]
            start_pts = seg["video_source_pts_sec"]
            duration_frames = seg["video_frame_count"]
            label = seg["target_text"]
            unit_type = seg["span_type"]
            is_freeze = seg.get("freeze_video", False) or seg.get("is_silence", False) or unit_type == "pause"
            src_content_id = seg.get("source_content_id", "")
        else:
            src_path = seg.source_video_path
            start_pts = seg.video_source_pts_sec
            duration_frames = seg.video_frame_count
            label = seg.target_text
            unit_type = seg.span_type
            is_freeze = getattr(seg, "freeze_video", False) or getattr(seg, "is_silence", False) or unit_type == "pause"
            src_content_id = getattr(seg, "source_content_id", "")

        if duration_frames <= 0:
            continue

        # Handle freeze frame / pause holds
        if is_freeze:
            holds_dir = out_p.parent / f"{out_p.stem}_holds"
            holds_dir.mkdir(parents=True, exist_ok=True)
            hold_img = holds_dir / f"hold_{idx:04d}.png"
            if not hold_img.exists() or hold_img.stat().st_size == 0:
                if master_wav_path:
                    alt_hold = Path(master_wav_path).parent / f"{Path(master_wav_path).stem}_holds" / f"hold_{idx:04d}.png"
                    if alt_hold.exists() and alt_hold.stat().st_size > 0:
                        import shutil
                        shutil.copy2(alt_hold, hold_img)

            if not hold_img.exists() or hold_img.stat().st_size == 0:
                if src_path and Path(src_path).exists():
                    try:
                        import subprocess
                        cmd = [
                            "ffmpeg", "-y",
                            "-ss", f"{start_pts:.6f}",
                            "-i", str(src_path),
                            "-frames:v", "1",
                            "-q:v", "1",
                            str(hold_img),
                        ]
                        subprocess.run(cmd, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                        if not hold_img.exists() or hold_img.stat().st_size == 0:
                            cmd_eof = [
                                "ffmpeg", "-y",
                                "-sseof", "-0.1",
                                "-i", str(src_path),
                                "-frames:v", "1",
                                "-q:v", "1",
                                str(hold_img),
                            ]
                            subprocess.run(cmd_eof, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    except Exception:
                        pass

            if not hold_img.exists() or hold_img.stat().st_size == 0:
                try:
                    import subprocess
                    cmd = [
                        "ffmpeg", "-y",
                        "-f", "lavfi",
                        "-i", f"color=c=black:s={width}x{height}",
                        "-frames:v", "1",
                        str(hold_img),
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except Exception:
                    pass

            hold_url = f"file://localhost/{str(hold_img.resolve()).replace(os.sep, '/')}"
            vclip = ET.SubElement(vtrack, "clipitem", id=f"vclip-{idx}")
            ET.SubElement(vclip, "name").text = f"{idx:03d}_{label}_hold"
            ET.SubElement(vclip, "start").text = str(curr_timeline_frame)
            ET.SubElement(vclip, "end").text = str(curr_timeline_frame + duration_frames)
            ET.SubElement(vclip, "in").text = "0"
            ET.SubElement(vclip, "out").text = str(duration_frames)

            hold_id = f"file-hold-{idx}"
            f_node = ET.SubElement(vclip, "file", id=hold_id)
            ET.SubElement(f_node, "name").text = hold_img.name
            ET.SubElement(f_node, "pathurl").text = hold_url
            f_rate = ET.SubElement(f_node, "rate")
            ET.SubElement(f_rate, "timebase").text = str(timebase)
            ET.SubElement(f_rate, "ntsc").text = ntsc_flag
            ET.SubElement(f_node, "duration").text = str(duration_frames)
            f_media = ET.SubElement(f_node, "media")
            f_video = ET.SubElement(f_media, "video")
            f_sample = ET.SubElement(f_video, "samplecharacteristics")
            ET.SubElement(f_sample, "width").text = str(width)
            ET.SubElement(f_sample, "height").text = str(height)

            curr_timeline_frame += duration_frames
            continue

        # Determine source native video FPS
        src_fps_float = fps_float
        src_rec = None
        if isinstance(plan_or_units, EditPlan) and plan_or_units.sources:
            src_rec = plan_or_units.sources.get(src_content_id)
        if src_rec and hasattr(src_rec, "video_fps"):
            v_fps = src_rec.video_fps
            if isinstance(v_fps, dict):
                src_fps_float = float(v_fps.get("numerator", 30000)) / float(v_fps.get("denominator", 1001))
            elif isinstance(v_fps, RationalTime):
                src_fps_float = v_fps.as_float()
        elif src_path and Path(src_path).exists():
            try:
                from media_time import probe_media_file
                p_data = probe_media_file(src_path)
                for s in p_data.get("streams", []):
                    if s.get("codec_type") == "video":
                        r_fr = s.get("r_frame_rate", "")
                        if r_fr:
                            src_fps_float = RationalTime.from_str(r_fr).as_float()
                            break
            except Exception:
                pass

        src_timebase = int(round(src_fps_float))
        src_ntsc = "TRUE" if abs(src_fps_float - 29.97) < 0.05 or abs(src_fps_float - 23.976) < 0.05 else "FALSE"

        in_frame = int(round(start_pts * src_fps_float))
        dur_sec = float(duration_frames) / fps_float
        out_frame = in_frame + int(round(dur_sec * src_fps_float))

        # Video clipitem
        vclip = ET.SubElement(vtrack, "clipitem", id=f"vclip-{idx}")
        ET.SubElement(vclip, "name").text = f"{idx:03d}_{label}_{unit_type}"
        ET.SubElement(vclip, "start").text = str(curr_timeline_frame)
        ET.SubElement(vclip, "end").text = str(curr_timeline_frame + duration_frames)
        ET.SubElement(vclip, "in").text = str(in_frame)
        ET.SubElement(vclip, "out").text = str(out_frame)

        # Register file node if not already seen
        if src_path not in file_elements:
            file_id = f"file-{file_counter}"
            file_counter += 1
            f_node = ET.SubElement(vclip, "file", id=file_id)
            ET.SubElement(f_node, "name").text = os.path.basename(src_path)
            formatted_url = f"file://localhost/{src_path.replace(os.sep, '/')}"
            ET.SubElement(f_node, "pathurl").text = formatted_url
            f_rate = ET.SubElement(f_node, "rate")
            ET.SubElement(f_rate, "timebase").text = str(src_timebase)
            ET.SubElement(f_rate, "ntsc").text = src_ntsc
            file_elements[src_path] = file_id
        else:
            file_id = file_elements[src_path]
            ET.SubElement(vclip, "file", id=file_id)

        curr_timeline_frame += duration_frames

    # If master WAV is provided, place master WAV as the authoritative audio track
    if master_wav_path:
        inspect_target = Path(inspect_wav_path) if inspect_wav_path else Path(master_wav_path)
        if not inspect_target.exists():
            raise FileNotFoundError(
                f"Master audio file not found for XML inspection: '{inspect_target}'. "
                f"Cannot generate XML without valid master audio."
            )

        num_channels = 2
        try:
            with wave.open(str(inspect_target), "rb") as wf:
                num_channels = wf.getnchannels()
        except Exception:
            pass

        wav_path_abs = str(Path(master_wav_path).resolve())
        wav_id = "file-master-audio"
        formatted_url = f"file://localhost/{wav_path_abs.replace(os.sep, '/')}"

        if num_channels == 1:
            atrack1 = ET.SubElement(audio, "track")
            aclip1 = ET.SubElement(atrack1, "clipitem", id="aclip-master-1")
            ET.SubElement(aclip1, "name").text = "Master_Audio"
            ET.SubElement(aclip1, "start").text = "0"
            ET.SubElement(aclip1, "end").text = str(curr_timeline_frame)
            ET.SubElement(aclip1, "in").text = "0"
            ET.SubElement(aclip1, "out").text = str(curr_timeline_frame)
            wav_f1 = ET.SubElement(aclip1, "file", id=wav_id)
            ET.SubElement(wav_f1, "name").text = os.path.basename(wav_path_abs)
            ET.SubElement(wav_f1, "pathurl").text = formatted_url
            f_rate = ET.SubElement(wav_f1, "rate")
            ET.SubElement(f_rate, "timebase").text = str(timebase)
            ET.SubElement(f_rate, "ntsc").text = ntsc_flag
        else:
            atrack1 = ET.SubElement(audio, "track")
            aclip1 = ET.SubElement(atrack1, "clipitem", id="aclip-master-1")
            ET.SubElement(aclip1, "name").text = "Master_Audio_L"
            ET.SubElement(aclip1, "start").text = "0"
            ET.SubElement(aclip1, "end").text = str(curr_timeline_frame)
            ET.SubElement(aclip1, "in").text = "0"
            ET.SubElement(aclip1, "out").text = str(curr_timeline_frame)
            wav_f1 = ET.SubElement(aclip1, "file", id=wav_id)
            ET.SubElement(wav_f1, "name").text = os.path.basename(wav_path_abs)
            ET.SubElement(wav_f1, "pathurl").text = formatted_url
            f_rate = ET.SubElement(wav_f1, "rate")
            ET.SubElement(f_rate, "timebase").text = str(timebase)
            ET.SubElement(f_rate, "ntsc").text = ntsc_flag

            atrack2 = ET.SubElement(audio, "track")
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
