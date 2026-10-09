#!/usr/bin/env python3
"""
CLI runner for multi-speaker dialogue scenes with burned-in word-highlighted subtitles.
Renders Episode III (Anakin & Mace Windu) and Episode VI (Luke, Palpatine, Vader) scenes.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add src and project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from typing import Dict, Optional

from dialogue_mixer import parse_script_text, synthesize_and_render_dialogue
from run_mixer import load_or_combine_inventories

EP3_INVENTORY_PATH = Path(".build/inventory/Star_Wars_Episode_III_-_Revenge_Of_The_S_a39114e3/units_index.json")
EP6_INVENTORY_PATH = Path(".build/inventory/Star_Wars_Episode_VI_-_Return_Of_The_Jed_3d083b52/units_index.json")

EPISODE_3_SCRIPT = """
anakin: It's not the Jedi way, he must live!
mace-windu: Tell you what. If you let me violate the Jedi Code this one time, I'll let you violate the Code too. Anything you want, anytime. Sound good?
anakin: Will you let me publicly marry Senator Amidala and help me get her the proper medical care she needs to carry her child to term?
mace-windu: That's a suspiciously specific request, but okay.
"""

EPISODE_6_SCRIPT = """
luke: You've failed, Your Highness. I'm a Jedi Master, like my father before me.
palpatine: Actually, Anakin wasn't a Jedi Master. He was only a Jedi Knight.
luke: What? How could they do that? This is outrageous; it was unfair. How could he be on the Council and not be a Master?
vader: That's my boy!
"""

# Default character lightsaber / thematic highlight colors
EP3_DEFAULT_COLORS: Dict[str, str] = {
    "mace": "purple",
    "mace-windu": "purple",
    "anakin": "blue",
}

EP6_DEFAULT_COLORS: Dict[str, str] = {
    "luke": "green",
    "palpatine": "red",
    "sidious": "red",
    "vader": "blue",
}


def parse_colors_arg(arg: Optional[str]) -> Dict[str, str]:
    """Parses 'speaker=color,speaker2=color2' or JSON dictionary into a dict."""
    if not arg:
        return {}
    val = arg.strip()
    if val.startswith("{") and val.endswith("}"):
        import json
        return json.loads(val)
    parsed = {}
    for part in val.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" in part:
            k, v = part.split("=", 1)
            parsed[k.strip().lower()] = v.strip()
        elif ":" in part:
            k, v = part.split(":", 1)
            parsed[k.strip().lower()] = v.strip()
    return parsed


def render_episode_3(
    output_path: Path,
    resolution: str = "1080p",
    speaker_colors: Optional[Dict[str, str]] = None,
) -> Path:
    print("\n" + "=" * 60)
    print("RENDERING EPISODE III DIALOGUE: ANAKIN & MACE WINDU")
    print("=" * 60)
    lines = parse_script_text(EPISODE_3_SCRIPT)
    print(f"Loading inventory: {EP3_INVENTORY_PATH}")
    inv3, _ = load_or_combine_inventories([str(EP3_INVENTORY_PATH)])
    colors = dict(EP3_DEFAULT_COLORS)
    if speaker_colors:
        colors.update(speaker_colors)
    print(f"Speaker highlight colors: {colors}")
    return synthesize_and_render_dialogue(
        lines=lines,
        inventory_index=inv3,
        output_video_path=output_path,
        audio_track="recombined",
        pause_between_sec=0.45,
        resolution=resolution,
        burn_subtitles=True,
        highlight_words=True,
        speaker_colors=colors,
    )


def render_episode_6(
    output_path: Path,
    resolution: str = "1080p",
    speaker_colors: Optional[Dict[str, str]] = None,
) -> Path:
    print("\n" + "=" * 60)
    print("RENDERING EPISODE VI DIALOGUE: LUKE, PALPATINE & VADER")
    print("=" * 60)
    lines = parse_script_text(EPISODE_6_SCRIPT)
    print(f"Loading inventory: {EP6_INVENTORY_PATH}")
    inv6, _ = load_or_combine_inventories([str(EP6_INVENTORY_PATH)])
    colors = dict(EP6_DEFAULT_COLORS)
    if speaker_colors:
        colors.update(speaker_colors)
    print(f"Speaker highlight colors: {colors}")
    return synthesize_and_render_dialogue(
        lines=lines,
        inventory_index=inv6,
        output_video_path=output_path,
        audio_track="recombined",
        pause_between_sec=0.45,
        resolution=resolution,
        burn_subtitles=True,
        highlight_words=True,
        speaker_colors=colors,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Render multi-speaker dialogue scenes.")
    parser.add_argument(
        "--clip",
        choices=["1", "2", "all"],
        default="all",
        help="Which clip to render (1: Ep3, 2: Ep6, all: both)",
    )
    parser.add_argument(
        "--resolution",
        default="1080p",
        help="Target video resolution (1080p, 4k, source)",
    )
    parser.add_argument(
        "--colors",
        default=None,
        help="Custom speaker highlight colors, e.g. 'mace=purple,anakin=blue' or JSON",
    )
    args = parser.parse_args()

    out_dir = Path("output")
    out_dir.mkdir(parents=True, exist_ok=True)

    custom_colors = parse_colors_arg(args.colors)

    if args.clip in ("1", "all"):
        p3 = out_dir / "episode_3_anakin_mace.mp4"
        render_episode_3(p3, resolution=args.resolution, speaker_colors=custom_colors)

    if args.clip in ("2", "all"):
        p6 = out_dir / "episode_6_luke_palpatine_vader.mp4"
        render_episode_6(p6, resolution=args.resolution, speaker_colors=custom_colors)

    print("\n[+] All requested clips rendered successfully.")


if __name__ == "__main__":
    main()
