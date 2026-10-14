"""
Dialogue Scene Synthesizer and Word-Highlighted Subtitle Service.
Enables multi-character conversation synthesis from dialogue scripts with
word-by-word highlighted subtitle burning.
"""

from __future__ import annotations

import dataclasses
import re
import subprocess
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from contracts import EditPlan, PlanSegment, PlanTransition, RationalTime
from edit_plan import build_edit_plan, finalize_edit_plan
from features import ContinuousFeatureExtractor
from lexicon import Lexicon
from media_time import cumulative_frame_ceiling
from span_selector import SpanSelector
from target_graph import TargetGraphBuilder
from video_assembler import render_sentence_mix


def format_ass_time(sec: float) -> str:
    """Format seconds into ASS timestamp: H:MM:SS.cs"""
    cs = int(round(max(0.0, sec) * 100))
    h, cs = divmod(cs, 360000)
    m, cs = divmod(cs, 6000)
    s, cs = divmod(cs, 100)
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def parse_script_text(text: str) -> List[Tuple[str, str]]:
    """
    Parses dialogue script text into a list of (speaker_id, dialogue_text) turns.
    Expected format:
      speaker: line of dialogue
    """
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            spk, dia = line.split(":", 1)
            spk_clean = spk.strip().lower().replace(" ", "-")
            dia_clean = dia.strip()
            if dia_clean:
                # Replace lone hyphens with commas for clean tokenization
                dia_clean = re.sub(r"\s+-\s+", ", ", dia_clean)
                lines.append((spk_clean, dia_clean))
    return lines


def concat_edit_plans(
    plans: Sequence[EditPlan],
    pause_between_sec: float = 0.45,
) -> EditPlan:
    """
    Concatenates multiple individual EditPlans into a single continuous EditPlan,
    inserting natural conversational pause segments between speaker turns.
    """
    if not plans:
        raise ValueError("Cannot concatenate empty list of EditPlans")
    if len(plans) == 1:
        return plans[0]

    first = plans[0]
    sr = first.project_sample_rate
    fps = first.video_fps

    combined_segments: List[PlanSegment] = []
    combined_transitions: List[PlanTransition] = []
    combined_sources: Dict[str, Any] = {}
    combined_metadata: Dict[str, Any] = dict(first.metadata)

    current_sample = 0
    seg_idx = 0
    trans_idx = 0
    pause_samples = int(round(pause_between_sec * sr))

    for p_idx, plan in enumerate(plans):
        combined_sources.update(plan.sources)

        # Conversational pause between dialogue turns
        if p_idx > 0 and pause_samples > 0 and combined_segments:
            prev_seg = combined_segments[-1]
            p_start = current_sample
            p_end = p_start + pause_samples
            p_start_frame = cumulative_frame_ceiling(p_start, sr, fps)
            p_end_frame = cumulative_frame_ceiling(p_end, sr, fps)
            freeze_pts = prev_seg.video_source_pts_sec + (float(prev_seg.duration_samples) / float(sr))
            pause_seg = PlanSegment(
                segment_index=seg_idx,
                source_content_id="silence",
                speaker_id=prev_seg.speaker_id,
                source_start_sample=0,
                source_end_sample=pause_samples,
                output_start_sample=p_start,
                output_end_sample=p_end,
                target_text="[pause]",
                target_phones=["sil"],
                span_type="pause",
                source_video_path=prev_seg.source_video_path,
                gain_db=0.0,
                fade_in_samples=0,
                fade_out_samples=0,
                is_silence=True,
                video_start_frame=p_start_frame,
                video_frame_count=max(0, p_end_frame - p_start_frame),
                video_source_pts_sec=freeze_pts,
                freeze_video=True,
                realized_phones=[],
            )
            combined_segments.append(pause_seg)
            combined_transitions.append(PlanTransition(
                transition_index=trans_idx,
                from_segment_index=prev_seg.segment_index,
                to_segment_index=seg_idx,
                transition_type="pause",
                overlap_samples=0,
            ))
            seg_idx += 1
            trans_idx += 1
            current_sample = p_end

        # Map plan's segments with absolute sample offset
        seg_idx_map = {}
        plan_sample_offset = current_sample

        for s in plan.segments:
            old_idx = s.segment_index
            new_start = s.output_start_sample + plan_sample_offset
            new_end = s.output_end_sample + plan_sample_offset
            new_start_frame = cumulative_frame_ceiling(new_start, sr, fps)
            new_end_frame = cumulative_frame_ceiling(new_end, sr, fps)
            new_seg = dataclasses.replace(
                s,
                segment_index=seg_idx,
                output_start_sample=new_start,
                output_end_sample=new_end,
                video_start_frame=new_start_frame,
                video_frame_count=max(0, new_end_frame - new_start_frame),
            )
            seg_idx_map[old_idx] = seg_idx

            if combined_segments and len(seg_idx_map) == 1 and combined_segments[-1].segment_index != seg_idx:
                combined_transitions.append(PlanTransition(
                    transition_index=trans_idx,
                    from_segment_index=combined_segments[-1].segment_index,
                    to_segment_index=seg_idx,
                    transition_type="hard_splice",
                    overlap_samples=0,
                ))
                trans_idx += 1

            combined_segments.append(new_seg)
            seg_idx += 1

        current_sample = plan_sample_offset + plan.total_output_samples

        for t in plan.transitions:
            combined_transitions.append(PlanTransition(
                transition_index=trans_idx,
                from_segment_index=seg_idx_map[t.from_segment_index],
                to_segment_index=seg_idx_map[t.to_segment_index],
                transition_type=t.transition_type,
                overlap_samples=t.overlap_samples,
                fade_samples=t.fade_samples,
            ))
            trans_idx += 1

    # Re-apply video midpoint partitioning across combined segments and transitions
    # (matching src/edit_plan.py:454-485) to prevent double-counting video frames
    # across audio crossfades and ensure sum(video_frame_count) == total_video_frames.
    overlap_by_from_seg: Dict[int, int] = {}
    for t in combined_transitions:
        if t.transition_type == "overlap":
            overlap_by_from_seg[t.from_segment_index] = t.overlap_samples

    v_display_starts = [0] * len(combined_segments)
    v_display_ends = [0] * len(combined_segments)

    if combined_segments:
        v_display_starts[0] = 0
        for k in range(len(combined_segments) - 1):
            curr_seg = combined_segments[k]
            next_seg = combined_segments[k + 1]
            ov = overlap_by_from_seg.get(curr_seg.segment_index, 0)
            if ov > 0:
                midpoint = next_seg.output_start_sample + (ov // 2)
                v_display_ends[k] = midpoint
                v_display_starts[k + 1] = midpoint
            else:
                v_display_ends[k] = curr_seg.output_end_sample
                v_display_starts[k + 1] = next_seg.output_start_sample

        v_display_ends[-1] = current_sample

        for k, seg in enumerate(combined_segments):
            v_start = v_display_starts[k]
            v_end = v_display_ends[k]
            start_f = cumulative_frame_ceiling(v_start, sr, fps)
            end_f = cumulative_frame_ceiling(v_end, sr, fps)
            seg.video_start_frame = start_f
            seg.video_frame_count = max(0, end_f - start_f)

    total_frames = cumulative_frame_ceiling(current_sample, sr, fps)
    combined_target_text = "\n".join(f"{p.speaker_id}: {p.target_text}" for p in plans)
    combined_metadata["num_cuts"] = len(combined_transitions)

    combined_plan = EditPlan(
        plan_id=f"plan_multi_{uuid.uuid4().hex[:12]}",
        target_text=combined_target_text,
        speaker_id="multi",
        selector_mode="span",
        rendering_mode=first.rendering_mode,
        project_sample_rate=sr,
        video_fps=fps,
        total_output_samples=current_sample,
        total_video_frames=total_frames,
        segments=combined_segments,
        transitions=combined_transitions,
        sources=combined_sources,
        metadata=combined_metadata,
    )
    combined_plan.validate()
    return combined_plan


def _extract_plan_words(plan: EditPlan, time_offset_sec: float = 0.0) -> List[Tuple[str, float, float]]:
    """Extracts (word_text, start_sec, end_sec) from an EditPlan."""
    words: List[Tuple[str, float, float]] = []
    curr_word: Optional[str] = None
    curr_t0 = 0.0
    curr_t1 = 0.0

    prev_span_type = ""
    for s in plan.segments:
        if s.is_silence or s.span_type == "pause":
            if curr_word:
                words.append((curr_word, time_offset_sec + curr_t0, time_offset_sec + curr_t1))
                curr_word = None
            prev_span_type = s.span_type
            continue
        lbl = s.target_text.strip()
        t0 = s.output_start_sample / float(plan.project_sample_rate)
        t1 = s.output_end_sample / float(plan.project_sample_rate)

        can_merge = (s.span_type in ("subword", "phone")) and (prev_span_type in ("subword", "phone"))

        if " " in lbl:
            if curr_word:
                words.append((curr_word, time_offset_sec + curr_t0, time_offset_sec + curr_t1))
                curr_word = None
            subwords = lbl.split()
            dur_per = (t1 - t0) / max(1, len(subwords))
            for idx, sw in enumerate(subwords):
                words.append((sw, time_offset_sec + t0 + idx * dur_per, time_offset_sec + t0 + (idx + 1) * dur_per))
        elif lbl == curr_word and can_merge:
            curr_t1 = max(curr_t1, t1)
        else:
            if curr_word:
                words.append((curr_word, time_offset_sec + curr_t0, time_offset_sec + curr_t1))
            curr_word = lbl
            curr_t0 = t0
            curr_t1 = t1
        prev_span_type = s.span_type

    if curr_word:
        words.append((curr_word, time_offset_sec + curr_t0, time_offset_sec + curr_t1))
    return words


def _align_words_with_script_tokens(
    target_text: str,
    extracted_words: List[Tuple[str, float, float]],
) -> List[Tuple[str, float, float]]:
    """
    Aligns raw extracted words with the script target text to preserve
    original punctuation and capitalization.
    """
    raw_tokens = [tok for tok in target_text.split() if tok]
    if not extracted_words:
        return []
    if len(raw_tokens) == len(extracted_words):
        return [(tok, ew[1], ew[2]) for tok, ew in zip(raw_tokens, extracted_words)]

    # Fallback: if lengths differ (due to hyphen/dash splitting), map sequentially
    aligned: List[Tuple[str, float, float]] = []
    ew_idx = 0
    for tok in raw_tokens:
        if ew_idx < len(extracted_words):
            _, t0, t1 = extracted_words[ew_idx]
            aligned.append((tok, t0, t1))
            ew_idx += 1
        elif aligned:
            # Re-use last end time with tiny duration
            aligned.append((tok, aligned[-1][2], aligned[-1][2] + 0.2))
    return aligned


NAMED_HIGHLIGHT_COLORS: Dict[str, str] = {
    "yellow": "&H0000FFFF&",       # Classic subtitle yellow
    "purple": "&H00FF40D0&",       # Vibrant amethyst/purple (Mace Windu)
    "violet": "&H00FF40D0&",
    "amethyst": "&H00FF40D0&",
    "blue": "&H00FFB432&",         # Vibrant electric/lightsaber blue (Anakin, Vader)
    "green": "&H0050F032&",        # Vibrant lightsaber green (Luke)
    "red": "&H003C3CFF&",          # Vibrant Sith crimson red (Sidious)
    "crimson": "&H003C3CFF&",
    "orange": "&H001EA0FF&",
    "cyan": "&H00FFFF00&",
    "magenta": "&H00FF00FF&",
    "pink": "&H00CBC0FF&",
    "white": "&H00FFFFFF&",
}


def color_to_ass(color_val: str) -> str:
    """
    Converts a color string (named color, hex '#RRGGBB', or raw ASS '&H00BBGGRR&')
    to an ASS color tag string.
    """
    c = color_val.strip()
    if not c:
        return "&H0000FFFF&"
    if c.startswith("&H") and c.endswith("&"):
        return c
    c_lower = c.lower()
    if c_lower in NAMED_HIGHLIGHT_COLORS:
        return NAMED_HIGHLIGHT_COLORS[c_lower]
    # Check hex #RRGGBB or RRGGBB
    hex_match = re.match(r"^#?([0-9a-fA-F]{2})([0-9a-fA-F]{2})([0-9a-fA-F]{2})$", c)
    if hex_match:
        r = int(hex_match.group(1), 16)
        g = int(hex_match.group(2), 16)
        b = int(hex_match.group(3), 16)
        return f"&H00{b:02X}{g:02X}{r:02X}&"
    return "&H0000FFFF&"


def resolve_speaker_color(
    speaker: str,
    speaker_colors: Optional[Dict[str, str]] = None,
    default_color: str = "&H0000FFFF&",
) -> str:
    """
    Resolves the ASS highlight color for a speaker from a speaker_colors mapping.
    Supports normalized IDs, token/prefix matching, and common character aliases.
    """
    if not speaker_colors:
        return color_to_ass(default_color)

    spk_norm = speaker.strip().lower().replace("_", "-")
    # 1. Exact match
    if spk_norm in speaker_colors:
        return color_to_ass(speaker_colors[spk_norm])

    # 2. Match without hyphens
    spk_flat = spk_norm.replace("-", "")
    for k, v in speaker_colors.items():
        if k.strip().lower().replace("-", "").replace("_", "") == spk_flat:
            return color_to_ass(v)

    # 3. Match first token / prefix (e.g. 'mace' in 'mace-windu')
    spk_parts = [p for p in re.split(r"[-_\s]+", spk_norm) if p]
    for part in spk_parts:
        for k, v in speaker_colors.items():
            k_clean = k.strip().lower().replace("-", "").replace("_", "")
            if k_clean == part:
                return color_to_ass(v)

    # 4. Character aliases
    aliases = {
        "palpatine": ["sidious", "emperor"],
        "sidious": ["palpatine", "emperor"],
        "vader": ["darth-vader", "darth"],
        "mace-windu": ["mace", "windu"],
        "anakin": ["skywalker"],
    }
    for alias in aliases.get(spk_norm, []):
        for k, v in speaker_colors.items():
            if k.strip().lower() == alias:
                return color_to_ass(v)

    # 5. Substring containment
    for k, v in speaker_colors.items():
        k_norm = k.strip().lower()
        if k_norm in spk_norm or spk_norm in k_norm:
            return color_to_ass(v)

    return color_to_ass(default_color)


def generate_highlighted_ass_subtitles(
    turn_plans: Sequence[Tuple[str, str, EditPlan, float]],  # (speaker, text, plan, plan_start_sec)
    output_path: Path,
    video_width: int = 1920,
    video_height: int = 1080,
    max_words_per_card: int = 8,
    speaker_colors: Optional[Dict[str, str]] = None,
    default_highlight_color: str = "&H0000FFFF&",
) -> Path:
    """
    Generates an ASS subtitle file where each dialogue clause is shown on screen,
    and each word lights up in the speaker's highlight color as it is spoken.
    """
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        f"PlayResX: {video_width}",
        f"PlayResY: {video_height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        "Style: Default,Arial,42,&H00FFFFFF,&H00000000,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,2.8,1.2,2,40,40,55,1",
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    for spk, text, plan, offset_sec in turn_plans:
        raw_words = _extract_plan_words(plan, time_offset_sec=offset_sec)
        aligned_words = _align_words_with_script_tokens(text, raw_words)
        if not aligned_words:
            continue

        spk_highlight_color = resolve_speaker_color(
            spk,
            speaker_colors=speaker_colors,
            default_color=default_highlight_color,
        )

        # Split aligned words into subtitle cards (clauses)
        cards: List[List[Tuple[str, float, float]]] = []
        curr_card: List[Tuple[str, float, float]] = []

        for w_item in aligned_words:
            curr_card.append(w_item)
            w_text = w_item[0]
            # Clause boundary on punctuation or card length limit
            is_punct = any(w_text.endswith(p) for p in (".", "!", "?", ";", ","))
            if is_punct or len(curr_card) >= max_words_per_card:
                cards.append(curr_card)
                curr_card = []
        if curr_card:
            cards.append(curr_card)

        # Generate ASS events for each card
        for card_idx, card in enumerate(cards):
            card_end = card[-1][2]
            # Extend card display slightly into following pause if next card starts later
            next_start = (
                cards[card_idx + 1][0][1]
                if card_idx + 1 < len(cards)
                else (offset_sec + plan.total_output_samples / float(plan.project_sample_rate))
            )
            card_display_end = min(next_start, card_end + 0.35)

            for w_idx, (w_text, w_start, w_end) in enumerate(card):
                next_w_start = card[w_idx + 1][1] if w_idx + 1 < len(card) else card_end
                highlight_end = max(w_start + 0.04, min(next_w_start, max(w_end, w_start + 0.05)))

                t_s = format_ass_time(w_start)
                t_e = format_ass_time(highlight_end)
                # Build card text with active word highlighted in speaker color
                display_tokens = []
                for j, (other_w, _, _) in enumerate(card):
                    if j == w_idx:
                        display_tokens.append(f"{{\\c{spk_highlight_color}}}{other_w}{{\\c&H00FFFFFF&}}")
                    else:
                        display_tokens.append(other_w)
                card_str = " ".join(display_tokens)
                lines.append(f"Dialogue: 0,{t_s},{t_e},Default,,0,0,0,,{card_str}")

                # If there is a gap before the next word in this card, hold plain text
                if next_w_start > highlight_end + 0.05:
                    t_gap_s = format_ass_time(highlight_end)
                    t_gap_e = format_ass_time(next_w_start)
                    plain_str = " ".join(w[0] for w in card)
                    lines.append(f"Dialogue: 0,{t_gap_s},{t_gap_e},Default,,0,0,0,,{plain_str}")

            # Hold full card in white during remainder of card pause (if any)
            if card_display_end > card_end + 0.05:
                t_s = format_ass_time(card_end)
                t_e = format_ass_time(card_display_end)
                plain_str = " ".join(w[0] for w in card)
                lines.append(f"Dialogue: 0,{t_s},{t_e},Default,,0,0,0,,{plain_str}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def synthesize_and_render_dialogue(
    lines: Sequence[Tuple[str, str]],
    inventory_index: Any,
    output_video_path: Path,
    fps: RationalTime = RationalTime(24000, 1001),
    audio_track: str = "recombined",
    pause_between_sec: float = 0.45,
    resolution: Optional[str] = "1080p",
    burn_subtitles: bool = True,
    highlight_words: bool = True,
    speaker_colors: Optional[Dict[str, str]] = None,
    default_highlight_color: str = "&H0000FFFF&",
) -> Path:
    """
    High-level entry point:
    Synthesizes multi-character dialogue, concatenates into an EditPlan,
    renders video + master audio, generates highlighted ASS subtitles,
    and burns subtitles into the delivery MP4.
    """
    lex = Lexicon()
    graph_builder = TargetGraphBuilder(lex)
    selector = SpanSelector(inventory_index)
    feature_extractor = ContinuousFeatureExtractor(sr=16000)

    # Resolve speaker aliases against inventory if needed
    avail_speakers = set(w.speaker_id for w in inventory_index.words.values() if w.speaker_id)
    mapped_lines: List[Tuple[str, str]] = []
    for spk, text in lines:
        resolved = spk
        if resolved not in avail_speakers:
            if resolved == "sidious" and "palpatine" in avail_speakers:
                resolved = "palpatine"
            elif resolved == "palpatine" and "sidious" in avail_speakers:
                resolved = "sidious"
            elif resolved.replace("-", " ") in avail_speakers:
                resolved = resolved.replace("-", " ")
        mapped_lines.append((resolved, text))

    plans: List[EditPlan] = []
    turn_infos: List[Tuple[str, str, EditPlan, float]] = []

    print(f"\n[*] Synthesizing {len(mapped_lines)} dialogue turns...")
    curr_time_sec = 0.0

    for idx, (spk, text) in enumerate(mapped_lines, 1):
        print(f"  [Turn {idx}/{len(lines)}] {spk}: \"{text}\"")
        g = graph_builder.build(text)
        spans = selector.select(g, speaker_id=spk, feature_extractor=feature_extractor)
        plan = build_edit_plan(
            spans,
            g,
            inventory_index,
            project_sample_rate=48000,
            video_fps=fps,
            pause_style="continue",
            release_tail_sec=0.100,
            finalize=False,
        )
        plans.append(plan)
        turn_infos.append((spk, text, plan, curr_time_sec))
        curr_time_sec += (plan.total_output_samples / 48000.0) + pause_between_sec

    # Combine plans into one master timeline
    combined_plan = concat_edit_plans(plans, pause_between_sec=pause_between_sec)
    finalize_edit_plan(combined_plan, inventory_index, normalize_loudness=True, audio_track=audio_track)
    print(f"[+] Combined EditPlan created: {len(combined_plan.segments)} segments, duration: {combined_plan.total_output_samples/48000:.2f}s")

    # Staged rendering
    attempt_dir = output_video_path.parent / f".attempt_{uuid.uuid4().hex}"
    attempt_dir.mkdir(parents=True, exist_ok=True)

    import shutil
    try:
        raw_mp4 = attempt_dir / f"{output_video_path.stem}_raw.mp4"
        plan_json = attempt_dir / f"{output_video_path.stem}.plan.json"
        combined_plan.save(str(plan_json))

        print(f"[*] Rendering master video ({audio_track} audio, {resolution or 'source'})...")
        render_sentence_mix(
            combined_plan,
            str(raw_mp4),
            normalize_loudness=True,
            audio_track=audio_track,
            output_resolution=resolution,
        )

        final_delivery = raw_mp4
        if burn_subtitles:
            ass_path = attempt_dir / f"{output_video_path.stem}.ass"
            generate_highlighted_ass_subtitles(
                turn_infos,
                ass_path,
                speaker_colors=speaker_colors,
                default_highlight_color=default_highlight_color,
            )

            burned_mp4 = attempt_dir / output_video_path.name
            sub_escaped = str(ass_path.resolve()).replace("\\", "/").replace(":", r"\:")
            vf = f"subtitles='{sub_escaped}'"
            print("[*] Burning word-highlighted subtitles via FFmpeg...")
            cmd = [
                "ffmpeg", "-y", "-nostats", "-loglevel", "error",
                "-i", str(raw_mp4),
                "-vf", vf,
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-c:a", "copy",
                str(burned_mp4),
            ]
            subprocess.run(cmd, check=True)
            final_delivery = burned_mp4

        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(final_delivery), str(output_video_path))
        shutil.copy(str(plan_json), str(output_video_path.with_suffix(".plan.json")))
        raw_wav = raw_mp4.with_suffix(".wav")
        if raw_wav.exists():
            shutil.copy(str(raw_wav), str(output_video_path.with_suffix(".wav")))
        if burn_subtitles:
            ass_src = attempt_dir / f"{output_video_path.stem}.ass"
            if ass_src.exists():
                shutil.copy(str(ass_src), str(output_video_path.with_suffix(".ass")))
    finally:
        shutil.rmtree(attempt_dir, ignore_errors=True)

    print(f"[+] Successfully generated dialogue clip: {output_video_path} ({output_video_path.stat().st_size} bytes)")
    return output_video_path
