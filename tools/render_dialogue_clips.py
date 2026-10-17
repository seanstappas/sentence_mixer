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

from typing import Dict, List, Optional

from dialogue_mixer import parse_script_text, synthesize_and_render_dialogue
from run_mixer import load_or_combine_inventories
from cache import resolve_inventory_path

EP2_INVENTORY_PATH = resolve_inventory_path(".build/inventory/Star_Wars_Episode_II_-_Attack_Of_The_Clo_56214896")
EP3_INVENTORY_PATH = resolve_inventory_path(".build/inventory/Star_Wars_Episode_III_-_Revenge_Of_The_S_a39114e3")
EP4_INVENTORY_PATH = resolve_inventory_path(".build/inventory/Star_Wars_Episode_IV_-_A_New_Hope_1977_2_04865440")
EP6_INVENTORY_PATH = resolve_inventory_path(".build/inventory/Star_Wars_Episode_VI_-_Return_Of_The_Jed_3d083b52")

EPISODE_4_ALDERAAN_SCRIPT = """
obi-wan: Alderaan ought to be here, but it isn't.
han: Lost a planet, Master Obi-Wan has. How embarrassing.
obi-wan: WTF did you say?
han: I meant it's probably been blown away by the Empire.
"""

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

EPISODE_3_WEDDING_SCRIPT = """
obi-wan: I have seen a security recording of him killing younglings.
padme: I do not believe you. He promised me in our wedding vows he would not kill any more children!
obi-wan: What wedding? Wait, what do you mean by more children?
"""

EPISODE_3_COUNCIL_CHAMBERS_SCRIPT = """
mace-windu: If what you have told me is true, you will have gained my trust. Remain here. Wait in the Council Chambers until we return.
anakin: I will stay here until they return.
mace-windu: The Sith Lord has been destroyed. You have saved the Republic.
yoda: The rank of Master, grant you we do.
anakin: Thank you, Master!
padme: Ani, come here. There is someone I want you to meet. Luke, Leia, this is your father!
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

EP3_WEDDING_DEFAULT_COLORS: Dict[str, str] = {
    "obi-wan": "blue",
    "padme": "yellow",
}

EP3_COUNCIL_DEFAULT_COLORS: Dict[str, str] = {
    "mace-windu": "purple",
    "mace": "purple",
    "anakin": "blue",
    "yoda": "green",
    "padme": "yellow",
}

EP4_ALDERAAN_DEFAULT_COLORS: Dict[str, str] = {
    "obi-wan": "blue",
    "han": "orange",
}

EP4_PRONUNCIATION_OVERRIDES: Dict[str, List[List[str]]] = {
    "wtf": [["W", "AH1", "T", "DH", "AH0", "F", "AH1", "K"]],
}

EPISODE_3_TITLE_DROPS_SCRIPT = """
padme: Obi-Wan, tell me what's going on.
obi-wan: It's Palpatine. He's The Phantom Menace we've been looking for. He caused the Attack of the Clones on the Jedi in order to bring about the Revenge of the Sith.
padme: Oh no! That means that he's been behind these Star Wars the whole time!
"""

EP3_TITLE_DROPS_DEFAULT_COLORS: Dict[str, str] = {
    "padme": "magenta",
    "obi-wan": "cyan",
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


def render_episode_3_wedding_vows(
    output_path: Path,
    resolution: str = "1080p",
    speaker_colors: Optional[Dict[str, str]] = None,
) -> Path:
    print("\n" + "=" * 60)
    print("RENDERING EPISODE III MEME: THE WEDDING VOWS (OBI-WAN & PADME)")
    print("=" * 60)
    lines = parse_script_text(EPISODE_3_WEDDING_SCRIPT)
    print(f"Loading inventories: {EP2_INVENTORY_PATH}, {EP3_INVENTORY_PATH}")
    inv, _ = load_or_combine_inventories([str(EP2_INVENTORY_PATH), str(EP3_INVENTORY_PATH)])
    colors = dict(EP3_WEDDING_DEFAULT_COLORS)
    if speaker_colors:
        colors.update(speaker_colors)
    print(f"Speaker highlight colors: {colors}")
    return synthesize_and_render_dialogue(
        lines=lines,
        inventory_index=inv,
        output_video_path=output_path,
        audio_track="recombined",
        pause_between_sec=0.45,
        resolution=resolution,
        burn_subtitles=True,
        highlight_words=True,
        speaker_colors=colors,
    )


def render_episode_3_council_chambers(
    output_path: Path,
    resolution: str = "1080p",
    speaker_colors: Optional[Dict[str, str]] = None,
) -> Path:
    print("\n" + "=" * 60)
    print("RENDERING EPISODE III MEME: COUNCIL CHAMBERS (MACE, ANAKIN, YODA, PADME)")
    print("=" * 60)
    lines = parse_script_text(EPISODE_3_COUNCIL_CHAMBERS_SCRIPT)
    print(f"Loading inventories: {EP2_INVENTORY_PATH}, {EP3_INVENTORY_PATH}")
    inv, _ = load_or_combine_inventories([str(EP2_INVENTORY_PATH), str(EP3_INVENTORY_PATH)])
    colors = dict(EP3_COUNCIL_DEFAULT_COLORS)
    if speaker_colors:
        colors.update(speaker_colors)
    print(f"Speaker highlight colors: {colors}")
    return synthesize_and_render_dialogue(
        lines=lines,
        inventory_index=inv,
        output_video_path=output_path,
        audio_track="recombined",
        pause_between_sec=0.45,
        resolution=resolution,
        burn_subtitles=True,
        highlight_words=True,
        speaker_colors=colors,
    )


def render_episode_4_alderaan(
    output_path: Path,
    resolution: str = "1080p",
    speaker_colors: Optional[Dict[str, str]] = None,
) -> Path:
    print("\n" + "=" * 60)
    print("RENDERING EPISODE IV MEME: ALDERAAN / LOST A PLANET (OBI-WAN & HAN)")
    print("=" * 60)
    lines = parse_script_text(EPISODE_4_ALDERAAN_SCRIPT)
    print(f"Loading inventory: {EP4_INVENTORY_PATH}")
    inv4, _ = load_or_combine_inventories([str(EP4_INVENTORY_PATH)])
    colors = dict(EP4_ALDERAAN_DEFAULT_COLORS)
    if speaker_colors:
        colors.update(speaker_colors)
    print(f"Speaker highlight colors: {colors}")
    return synthesize_and_render_dialogue(
        lines=lines,
        inventory_index=inv4,
        output_video_path=output_path,
        audio_track="recombined",
        pause_between_sec=0.45,
        resolution=resolution,
        burn_subtitles=True,
        highlight_words=True,
        speaker_colors=colors,
        pronunciation_overrides=EP4_PRONUNCIATION_OVERRIDES,
    )


def render_episode_3_title_drops(
    output_path: Path,
    resolution: str = "1080p",
    speaker_colors: Optional[Dict[str, str]] = None,
) -> Path:
    print("\n" + "=" * 60)
    print("RENDERING EPISODE III MEME: TITLE DROPS (PADME & OBI-WAN)")
    print("=" * 60)
    lines = parse_script_text(EPISODE_3_TITLE_DROPS_SCRIPT)
    print(f"Loading inventory: {EP3_INVENTORY_PATH}")
    inv3, _ = load_or_combine_inventories([str(EP3_INVENTORY_PATH)])
    colors = dict(EP3_TITLE_DROPS_DEFAULT_COLORS)
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Render multi-speaker dialogue scenes.")
    parser.add_argument(
        "--clip",
        choices=["1", "2", "3", "4", "5", "6", "all"],
        default="all",
        help="Which clip to render (1: Ep3 Anakin/Mace, 2: Ep6 Luke/Palpatine/Vader, 3: Ep3 Meme Wedding Vows, 4: Ep3 Meme Council Chambers, 5: Ep4 Meme Alderaan, 6: Ep3 Meme Title Drops, all: all clips)",
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
        p1 = out_dir / "episode_3_anakin_mace.mp4"
        render_episode_3(p1, resolution=args.resolution, speaker_colors=custom_colors)

    if args.clip in ("2", "all"):
        p2 = out_dir / "episode_6_luke_palpatine_vader.mp4"
        render_episode_6(p2, resolution=args.resolution, speaker_colors=custom_colors)

    if args.clip in ("3", "all"):
        p3 = out_dir / "meme_wedding_vows.mp4"
        render_episode_3_wedding_vows(p3, resolution=args.resolution, speaker_colors=custom_colors)

    if args.clip in ("4", "all"):
        p4 = out_dir / "meme_council_chambers.mp4"
        render_episode_3_council_chambers(p4, resolution=args.resolution, speaker_colors=custom_colors)

    if args.clip in ("5", "all"):
        p5 = out_dir / "meme_alderaan_lost_planet.mp4"
        render_episode_4_alderaan(p5, resolution=args.resolution, speaker_colors=custom_colors)

    if args.clip in ("6", "all"):
        p6 = out_dir / "meme_star_wars_title_drops.mp4"
        render_episode_3_title_drops(p6, resolution=args.resolution, speaker_colors=custom_colors)

    print("\n[+] All requested clips rendered successfully.")


if __name__ == "__main__":
    main()
