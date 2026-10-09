"""Tests for tools/label_subtitles.py using original synthetic dialogue only."""

import json
import sys
from pathlib import Path

import pytest

from tools import label_subtitles as ls

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))
from vtt_preprocessor import parse_vtt_cues  # noqa: E402

CHARACTERS = {
    "speakers": {
        "mira": {"display": "Mira Vance", "aliases": ["mira", "mira vance", "captain vance"]},
        "tobin": {"display": "Tobin Hale", "aliases": ["tobin", "tobin hale"]},
        "k-9x": {"display": "K-9X", "aliases": ["k-9x", "k9x"]},
        "osric": {"display": "Osric", "aliases": ["osric"]},
    },
    "roles": {"role-guard": ["guard", "guard 1"]},
    "groups": ["all", "together"],
    "ignore_patterns": ["^page \\d+$"],
}

SRT = """1
00:00:01,000 --> 00:00:03,000
We should leave before the storm hits the valley.

2
00:00:03,500 --> 00:00:05,000
- Is the bridge still standing?
- Only the eastern half.

3
00:00:06,000 --> 00:00:07,000
[THUNDER RUMBLING]

4
00:00:08,000 --> 00:00:10,000
K-9x: Warning, structural integrity is failing.

5
00:00:10,500 --> 00:00:12,000
[SPEAKING MARTIAN] Bring me the lantern.

6
00:00:12,500 --> 00:00:12,500
Zero length cue.

7
00:00:13,000 --> 00:00:15,000
Then we cross at dawn and pray the rope holds.

8
00:00:15,500 --> 00:00:17,500
Guard: Halt, who goes there in the night?

9
00:00:18,000 --> 00:00:20,000
Nobody you need to worry about tonight, friend.

10
00:00:21,000 --> 00:00:23,000
Together: We march at first light.

"""

TRANSCRIPT = """Mira: We should leave before the storm hits the valley.

Tobin: Is the bridge still standing?

Mira: Only the eastern half.

K-9X: Warning, structural integrity is failing.

Tobin: Then we cross at dawn and pray the rope holds. [He looks away]

Osric: Halt, who goes there in the night?

Nobody you need to worry about tonight, friend.
"""

SCREENPLAY_HTML = """<html><body><pre>
<b>EXT. RIVER VALLEY - NIGHT</b>
Rain falls on the old bridge.

<b>				TOBIN</b>
			Nobody you need to worry about
			tonight, friend.

<b>				MIRA</b>
			(whispering)
			Something we never say on screen.
</pre></body></html>
""" + "\n".join(f"<b>FILLER{i}</b>" for i in range(60))


@pytest.fixture()
def resolver():
    return ls.CharacterResolver(CHARACTERS)


def test_parse_subtitle_text_counts_and_times():
    cues, skipped = ls.parse_subtitle_text("\ufeff" + SRT + "11\n00:00:30,000 --> 00:00:31,000\n\n")
    assert len(cues) == 10
    assert skipped == 1
    assert cues[0].start == pytest.approx(1.0) and cues[0].end == pytest.approx(3.0)


def test_split_segments_dashes_labels_tags():
    cues, _ = ls.parse_subtitle_text(SRT)
    dash = ls.split_segments(cues[1])
    assert len(dash) == 2
    assert dash[0].start == pytest.approx(3.5) and dash[1].end == pytest.approx(5.0)
    assert dash[0].end == pytest.approx(dash[1].start)
    labeled = ls.split_segments(cues[3])[0]
    assert labeled.raw_label == "K-9x"
    assert not labeled.text.lower().startswith("k-9x")
    assert ls.split_segments(cues[2])[0].sound_only
    assert ls.split_segments(cues[4])[0].non_english


def test_resolver_aliases_ocr_roles_groups(resolver):
    assert resolver.resolve("MIRA VANCE") == ("mira", "speaker")
    assert resolver.resolve("K-9x") == ("k-9x", "speaker")
    assert resolver.resolve("TOBlN") == ("tobin", "speaker")  # OCR l for I
    assert resolver.resolve("Guard 1") == ("role-guard", "role")
    assert resolver.resolve("Together") == (None, "group")
    assert resolver.resolve("PAGE 12") == (None, "ignored")
    assert resolver.resolve("Zebulon") == (None, "unknown")
    assert resolver.unknown["Zebulon"] == 1


def test_choose_subtitle_stream_prefers_english_sdh():
    streams = [
        {"index": 2, "codec_name": "subrip", "tags": {"language": "eng", "title": "English"}},
        {"index": 3, "codec_name": "subrip", "tags": {"language": "eng", "title": "English-SDH"}},
        {"index": 4, "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
    ]
    assert ls.choose_subtitle_stream(streams)[1]["index"] == 3
    assert ls.choose_subtitle_stream(streams, "plain")[1]["index"] == 2
    with pytest.raises(ValueError):
        ls.choose_subtitle_stream([streams[2]])


def test_script_parsers():
    speeches, headings = ls.parse_inline_script(TRANSCRIPT)
    assert [s.speaker_raw for s in speeches][:3] == ["Mira", "Tobin", "Mira"]
    assert speeches[-1].speaker_raw is None
    assert "looks away" not in speeches[4].text
    classic, classic_headings = ls.parse_classic_screenplay(SCREENPLAY_HTML)
    assert classic_headings == ["EXT. RIVER VALLEY - NIGHT"]
    assert [s.speaker_raw for s in classic] == ["TOBIN", "MIRA"]
    assert "whispering" not in classic[1].text
    text_sp = """1   EXT. RIVER VALLEY - NIGHT
A quiet night in the woods.

TOBIN (whispering)
Nobody you need to worry about tonight, friend.

MIRA
We cross at dawn.
"""
    sp_speeches, sp_headings = ls.parse_text_screenplay(text_sp)
    assert len(sp_headings) == 1
    assert [s.speaker_raw for s in sp_speeches] == ["TOBIN", "MIRA"]
    assert "whispering" not in sp_speeches[0].text


def test_end_to_end_labels_and_vtt(tmp_path):
    subs = tmp_path / "movie.srt"
    subs.write_text(SRT, encoding="utf-8")
    transcript = tmp_path / "transcript.txt"
    transcript.write_text(TRANSCRIPT, encoding="utf-8")
    screenplay = tmp_path / "screenplay.html"
    screenplay.write_text(SCREENPLAY_HTML, encoding="utf-8")
    characters = tmp_path / "characters.json"
    characters.write_text(json.dumps(CHARACTERS), encoding="utf-8")
    out_vtt = tmp_path / "labeled.en.vtt"
    out_report = tmp_path / "labeled.report.json"

    report = ls.run(subs, [transcript], [screenplay], characters, out_vtt, out_report)
    by_cue = {}
    for seg in report["segments"]:
        by_cue.setdefault(seg["cue_index"], []).append(seg)

    assert by_cue[0][0]["speaker"] == "mira" and by_cue[0][0]["source"] == "transcript"
    assert [s["speaker"] for s in by_cue[1]] == ["tobin", "mira"]
    assert by_cue[3][0]["speaker"] == "k-9x" and by_cue[3][0]["source"] == "subtitle_label"
    assert by_cue[6][0]["speaker"] == "tobin"
    # Subtitle has a generic role, transcript names the character.
    assert by_cue[7][0]["speaker"] == "osric" and by_cue[7][0]["source"] == "transcript_over_role"
    # Transcript has this line without a speaker; the screenplay fallback names it.
    assert by_cue[8][0]["speaker"] == "tobin" and by_cue[8][0]["source"] == "screenplay"
    assert by_cue[9][0]["speaker"] is None and by_cue[9][0]["source"] == "group"

    summary = report["summary"]
    assert summary["zero_duration_cues_omitted"] == 1
    assert summary["sound_only_lines"] == 1 and summary["non_english_lines"] == 1

    vtt = out_vtt.read_text(encoding="utf-8")
    assert vtt.startswith("WEBVTT")
    assert "<v mira>We should leave" in vtt
    assert "THUNDER" not in vtt and "lantern" not in vtt and "Zero length" not in vtt

    parsed = parse_vtt_cues(str(out_vtt))
    assert len(parsed) == 7
    assert all("<" not in c["text"] and not c["text"].lower().startswith(("k-9x:", "guard:")) for c in parsed)

    report_text = out_report.read_text(encoding="utf-8")
    assert "storm hits the valley" not in report_text  # report stores hashes, not text
