"""Tests for whole-movie support (large files, labeled subtitles, ranges, voice checks, scenes).

All dialogue and character names are invented for these tests.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from movie_subtitles import assign_word_speakers, load_movie_cues  # noqa: E402
from movie_ingest import (  # noqa: E402
    SegmentRecord,
    load_overrides,
    parse_ranges,
    parse_time,
    run_voice_checks,
    save_overrides,
    segment_key,
)
from movie_media import blocks_overlapping, list_subtitle_streams, plan_blocks, select_subtitle_stream  # noqa: E402
from subtitle_parsing import CharacterResolver  # noqa: E402

CHARACTERS = {
    "speakers": {
        "mira": {"display": "Mira", "aliases": ["MIRA VOSS"]},
        "tobin": {"display": "Tobin", "aliases": []},
    },
    "roles": {"role-guard": ["GUARD"]},
    "groups": ["CROWD"],
}

LABELED_VTT = """WEBVTT

NOTE
test

1
00:00:01.000 --> 00:00:03.000
<v mira>The lanterns are lit already.</v>

2
00:00:04.000 --> 00:00:06.500
<v mira>Bring the copper kettle.</v>
<v tobin>Not before supper.</v>

3
00:00:07.000 --> 00:00:08.000
Somebody hums quietly.
"""

PLAIN_SRT = """1
00:00:01,000 --> 00:00:02,000
[THUNDER RUMBLES]

2
00:00:02,500 --> 00:00:04,000
MIRA VOSS: Close the shutters.

3
00:00:04,500 --> 00:00:06,000
-Where is the ladder?
-Behind the barn.

4
00:00:07,000 --> 00:00:08,000
[SPEAKING OLD TONGUE]
Velka morra.

5
00:00:09,000 --> 00:00:09,000
Zero length.
"""


def test_load_labeled_vtt_segments_and_padding(tmp_path: Path):
    p = tmp_path / "movie.en.vtt"
    p.write_text(LABELED_VTT, encoding="utf-8")
    res = load_movie_cues(p)
    cues = res["cues"]
    assert res["stats"]["format"] == "labeled_vtt"
    assert len(cues) == 3
    assert [s["speaker_id"] for s in cues[1]["segments"]] == ["mira", "tobin"]
    assert cues[2]["segments"][0]["speaker_id"] is None
    assert "<v" not in cues[1]["text"]
    # Padding never crosses into the neighboring cue
    assert cues[0]["start_sec"] == pytest.approx(0.85)
    assert cues[0]["end_sec"] <= cues[1]["cue_start_sec"]
    assert cues[1]["start_sec"] >= cues[0]["cue_end_sec"]


def test_load_plain_srt_labels_dashes_and_drops(tmp_path: Path):
    p = tmp_path / "movie.srt"
    p.write_text(PLAIN_SRT, encoding="utf-8")
    res = load_movie_cues(p, CharacterResolver(CHARACTERS))
    st = res["stats"]
    assert st["sound_only"] == 1 and st["non_english"] == 1 and st["zero_duration"] == 1
    cues = res["cues"]
    assert len(cues) == 2
    assert cues[0]["segments"][0]["speaker_id"] == "mira"
    assert cues[0]["text"] == "Close the shutters."
    assert len(cues[1]["segments"]) == 2


def test_assign_word_speakers_by_token_order():
    segs = [
        {"speaker_id": "mira", "tokens": ["bring", "the", "copper", "kettle"]},
        {"speaker_id": "tobin", "tokens": ["not", "before", "supper"]},
    ]
    words = ["bring", "the", "copper", "kettle", "not", "before", "supper"]
    got = [s["speaker_id"] if s else None for s in assign_word_speakers(words, segs)]
    assert got == ["mira"] * 4 + ["tobin"] * 3
    # A word MFA spelled differently between two segments is left unresolved
    words2 = ["bring", "the", "copper", "kettles", "not", "before", "supper"]
    got2 = [s["speaker_id"] if s else None for s in assign_word_speakers(words2, segs)]
    assert got2[3] is None and got2[4] == "tobin"
    # Hyphenated MFA words still map
    segs3 = [{"speaker_id": "mira", "tokens": ["well", "met", "jo", "ann"]}]
    got3 = assign_word_speakers(["well", "met", "jo-ann"], segs3)
    assert all(s is not None for s in got3)


def test_parse_ranges_and_times():
    assert parse_time("01:30:00") == 5400
    assert parse_time("02:03.5") == 123.5
    assert parse_ranges(["01:30:00-01:35:00", "10-20,30-40"]) == [(10, 20), (30, 40), (5400, 5700)]
    with pytest.raises(ValueError):
        parse_ranges(["50-40"])
    with pytest.raises(ValueError):
        parse_ranges(["100"])


def test_blocks_and_overlap():
    blocks = plan_blocks(1500.0, 600.0)
    assert blocks == [(0.0, 600.0), (600.0, 1200.0), (1200.0, 1500.0)]
    assert blocks_overlapping(blocks, None) == [0, 1, 2]
    assert blocks_overlapping(blocks, [(590.0, 610.0)]) == [0, 1]
    assert blocks_overlapping(blocks, [(1300.0, 1400.0)]) == [2]


def test_subtitle_stream_selection_prefers_english_sdh_text():
    probe = {"streams": [
        {"index": 0, "codec_type": "video"},
        {"index": 2, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "eng", "title": "English-SRT"}},
        {"index": 3, "codec_type": "subtitle", "codec_name": "subrip", "tags": {"language": "eng", "title": "English-SDH-SRT"}},
        {"index": 4, "codec_type": "subtitle", "codec_name": "hdmv_pgs_subtitle", "tags": {"language": "eng"}},
    ]}
    streams = list_subtitle_streams(probe)
    assert select_subtitle_stream(streams)["index"] == 3
    assert select_subtitle_stream(streams, "plain")["index"] == 2
    assert select_subtitle_stream(streams, "2")["index"] == 2
    with pytest.raises(ValueError):
        select_subtitle_stream(streams, "4")
    assert select_subtitle_stream(list_subtitle_streams({"streams": [probe["streams"][3]]})) is None


def test_lazy_audio_matches_full_read(tmp_path: Path):
    from video_assembler import LazyAudio

    data = np.random.default_rng(0).uniform(-0.5, 0.5, size=(48000, 2)).astype(np.float32)
    p = tmp_path / "a.flac"
    sf.write(str(p), data, 48000, subtype="PCM_16")
    full, _ = sf.read(str(p), dtype="float32")
    lazy = LazyAudio(p)
    assert len(lazy) == 48000 and lazy.ndim == 2 and lazy.shape == (48000, 2)
    np.testing.assert_allclose(lazy[1000:2000], full[1000:2000])
    assert lazy[5:5].shape == (0, 2)


def test_lazy_features_for_long_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from features import ContinuousFeatureExtractor

    sr = 16000
    t = np.arange(sr * 4) / sr
    y = (0.3 * np.sin(2 * np.pi * 150 * t)).astype(np.float32)
    p = tmp_path / "analysis.flac"
    sf.write(str(p), y, sr)
    fx = ContinuousFeatureExtractor(sr=16000)
    monkeypatch.setattr(ContinuousFeatureExtractor, "LAZY_THRESHOLD_SEC", 1.0)
    tracks = fx.extract_tracks(p)
    assert tracks.get("lazy") is True
    rec = fx.get_window_features(tracks, start_sample=96000, end_sample=98400, native_sr=48000)
    assert rec.start_sample == 96000 and rec.sample_rate == 48000
    assert rec.f0_hz is not None and abs(rec.f0_hz - 150) < 10
    assert not (tmp_path / "analysis.features.npz").exists()


def test_precomputed_span_features_are_used(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from features import ContinuousFeatureExtractor, precompute_span_features, span_features_path

    sr = 16000
    t = np.arange(sr * 4) / sr
    y = (0.3 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    p = tmp_path / "analysis.flac"
    sf.write(str(p), y, sr)
    monkeypatch.setattr(ContinuousFeatureExtractor, "LAZY_THRESHOLD_SEC", 1.0)
    out = precompute_span_features(p, [(1.0, 2.0)], workers=1, log=lambda m: None)
    assert out == span_features_path(p) and out.exists()
    mtime = out.stat().st_mtime_ns
    precompute_span_features(p, [(1.0, 2.0)], workers=1, log=lambda m: None)
    assert out.stat().st_mtime_ns == mtime  # nothing new to compute
    fx = ContinuousFeatureExtractor(sr=16000)
    tracks = fx.extract_tracks(p)
    assert not tracks.get("lazy") and tracks["n_frames"] == int(np.ceil(4 * sr / 160)) + 1
    inside = fx.get_window_features(tracks, start_sample=int(1.4 * 48000), end_sample=int(1.5 * 48000), native_sr=48000)
    assert inside.f0_hz is not None and abs(inside.f0_hz - 200) < 10
    outside = fx.get_window_features(tracks, start_sample=int(3.0 * 48000), end_sample=int(3.1 * 48000), native_sr=48000)
    assert outside.f0_hz is None and outside.rms_energy == 0.0


def _textgrid(words, phones, xmax):
    def tier(name, items):
        out = [f'    item [{1 if name == "words" else 2}]:', '        class = "IntervalTier"', f'        name = "{name}"',
               '        xmin = 0.0', f'        xmax = {xmax}', f'        intervals: size = {len(items)}']
        for i, (a, b, txt) in enumerate(items, 1):
            out += [f'        intervals [{i}]:', f'            xmin = {a}', f'            xmax = {b}', f'            text = "{txt}"']
        return "\n".join(out)
    return "\n".join([
        'File type = "ooTextFile"', 'Object class = "TextGrid"', '', 'xmin = 0.0', f'xmax = {xmax}',
        'tiers? <exists>', 'size = 2', 'item []:', tier("words", words), tier("phones", phones), '',
    ])


def test_inventory_builder_assigns_segment_speakers(tmp_path: Path):
    from contracts import SourceRecord
    from inventory_builder import build_units_inventory

    aligned = tmp_path / "aligned"
    aligned.mkdir()
    words = [(0.0, 0.4, "bring"), (0.4, 0.8, "kettle"), (0.8, 1.0, ""), (1.0, 1.4, "not"), (1.4, 1.9, "supper")]
    phones = [(0.0, 0.2, "B"), (0.2, 0.4, "R"), (0.4, 0.6, "K"), (0.6, 0.8, "EH1"), (0.8, 1.0, ""),
              (1.0, 1.2, "N"), (1.2, 1.4, "AA1"), (1.4, 1.9, "S")]
    (aligned / "b000_c00000.TextGrid").write_text(_textgrid(words, phones, 1.9), encoding="utf-8")
    manifest = {
        "source_video": str(tmp_path / "movie.mkv"),
        "movie": True,
        "chunks": {"b000_c00000": {
            "start_sec": 10.0, "end_sec": 11.9, "duration": 1.9, "text": "bring kettle not supper", "speaker_id": None,
            "segments": [
                {"key": "a", "tokens": ["bring", "kettle"], "speaker_id": "mira", "is_eligible": True,
                 "eligibility_reason": "", "identity_status": "label_voice_ok"},
                {"key": "b", "tokens": ["not", "supper"], "speaker_id": "tobin", "is_eligible": False,
                 "eligibility_reason": "voice_mismatch", "identity_status": "voice_mismatch"},
            ],
        }},
    }
    mp = tmp_path / "movie_manifest.json"
    mp.write_text(json.dumps(manifest), encoding="utf-8")
    rec = SourceRecord(content_id="movie_x", source_path=str(tmp_path / "movie.mkv"), native_sample_rate=48000, native_channels=2)
    out = build_units_inventory(str(aligned), str(mp), str(tmp_path / "inv.json"), source_record=rec)
    words_out = {w["word_normalized"]: w for w in out["words"].values()}
    assert words_out["bring"]["speaker_id"] == "mira" and words_out["bring"]["is_eligible"]
    assert words_out["supper"]["speaker_id"] == "tobin" and not words_out["supper"]["is_eligible"]
    assert words_out["supper"]["eligibility_reason"] == "voice_mismatch"
    assert words_out["bring"]["start_sample"] == 480000
    # Compact JSON for movie inventories
    assert "\n" not in (tmp_path / "inv.json").read_text(encoding="utf-8")[:200]


class _FakeEmbedder:
    def __init__(self, vectors):
        self.vectors = vectors

    def embed(self, start, end):
        return self.vectors[round(start, 3)]

    def save(self):
        pass


def test_voice_checks_flag_mismatch_and_propagate():
    rng = np.random.default_rng(1)
    centers = {"mira": rng.normal(size=16), "tobin": rng.normal(size=16)}
    records, vectors = [], {}
    t = 0.0

    def add(label, voice, status="label"):
        nonlocal t
        v = centers[voice] + 0.15 * rng.normal(size=16)
        v = v / np.linalg.norm(v)
        r = SegmentRecord(key=f"k{t}", chunk_id="c", seg_index=0, cue_start=t, start=round(t, 3), end=round(t + 2.0, 3),
                          label=label, label_source="subtitle_tag", text_hash="h")
        if label:
            r.speaker, r.status, r.eligible = label, status, True
        vectors[round(t, 3)] = v
        records.append(r)
        t += 3.0
        return r

    for _ in range(15):
        add("mira", "mira")
        add("tobin", "tobin")
    wrong = add("mira", "tobin")
    unlabeled = add(None, "tobin")
    stats = run_voice_checks(records, _FakeEmbedder(vectors), log=lambda m: None)
    assert wrong.status == "voice_mismatch" and not wrong.eligible
    assert unlabeled.status == "voice_propagated" and unlabeled.speaker == "tobin"
    assert stats["loo_accuracy"] > 0.9


def test_restrict_inventory_to_ranges():
    import run_mixer
    from contracts import AlignedEvent, InventoryIndex, SourceRecord, WordOccurrence

    inv = InventoryIndex()
    inv.sources["s"] = SourceRecord(content_id="s", source_path="m.mkv", native_sample_rate=48000, native_channels=2)
    for i, sec in enumerate((5.0, 50.0)):
        ev = AlignedEvent(event_id=f"e{i}", source_content_id="s", utterance_id="u", speaker_id="mira", kind="phone",
                          start_sample=int(sec * 48000), end_sample=int((sec + 0.2) * 48000), label="AA1")
        inv.events[ev.event_id] = ev
        inv.words[f"w{i}"] = WordOccurrence(word_id=f"w{i}", source_content_id="s", speaker_id="mira", word_normalized="ah",
                                           word_original="ah", start_sample=ev.start_sample, end_sample=ev.end_sample,
                                           phone_event_ids=[ev.event_id], realized_phones=["AA1"])
    kept = run_mixer.restrict_inventory_to_ranges(inv, [(40.0, 60.0)])
    assert kept == 1
    assert not inv.words["w0"].is_eligible and inv.words["w0"].eligibility_reason == "outside_range"
    assert not inv.events["e0"].is_eligible and inv.events["e1"].is_eligible


def test_scene_selection_by_number_ignores_numbers_in_headings(monkeypatch: pytest.MonkeyPatch):
    import run_mixer

    scenes = [
        {"index": 3, "heading": "12 INT. MILL - DAY", "start": 100.0, "end": 200.0},
        {"index": 12, "heading": "40 EXT. RIVER - NIGHT", "start": 900.0, "end": 960.0},
    ]
    monkeypatch.setattr(run_mixer, "load_movie_scenes", lambda vp: scenes)
    assert run_mixer.resolve_time_ranges(None, ["12"], ["m.mkv"]) == [(900.0, 960.0)]
    assert run_mixer.resolve_time_ranges(None, ["mill"], ["m.mkv"]) == [(100.0, 200.0)]
    assert run_mixer.resolve_time_ranges(["00:00:10-00:00:20"], ["3"], ["m.mkv"]) == [(10.0, 20.0), (100.0, 200.0)]
    with pytest.raises(ValueError):
        run_mixer.resolve_time_ranges(None, ["99"], ["m.mkv"])


def test_map_scenes_orders_and_makes_ranges():
    from tools import label_subtitles as ls

    srt = "\n\n".join(
        f"{i + 1}\n00:{i:02d}:00,000 --> 00:{i:02d}:03,000\n{line}"
        for i, line in enumerate([
            "the orchard gate is open tonight",
            "fetch the lantern from the cellar",
            "the harbor bells are ringing again",
            "tell the ferryman to wait for us",
        ])
    )
    cues, _ = ls.parse_subtitle_text(srt)
    segments = [s for c in cues for s in ls.split_segments(c)]
    script = (
        "INT. ORCHARD - NIGHT\n\nMIRA: The orchard gate is open tonight.\n\nTOBIN: Fetch the lantern from the cellar.\n\n"
        "EXT. HARBOR - DAWN\n\nMIRA: The harbor bells are ringing again.\n\nTOBIN: Tell the ferryman to wait for us.\n"
    )
    speeches, headings = ls.parse_inline_script(script)
    scenes = ls.map_scenes(segments, speeches, headings)
    assert [s["heading"] for s in scenes] == ["INT. ORCHARD - NIGHT", "EXT. HARBOR - DAWN"]
    assert scenes[0]["end"] == scenes[1]["start"]
    assert scenes[0]["start"] <= 0.0 + 1e-6 and 63.0 < scenes[1]["start"] < 120.0


def test_overrides_roundtrip_and_review(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import movie_commands

    key = segment_key(12.5, 0, "Close the shutters.")
    save_overrides("abcdef0123456789ff", "movie.mkv", {key: {"key": key, "start": 12.5, "end": 13.4, "speaker_id": "mira"}}, base=tmp_path)
    assert load_overrides("abcdef0123456789ff", base=tmp_path)[key]["speaker_id"] == "mira"
    stored = (tmp_path / "abcdef0123456789.json").read_text(encoding="utf-8")
    assert "shutters" not in stored.lower()

    labels = {
        "source_id": "movie_x",
        "speaker_seconds": {"mira": 10.0, "tobin": 5.0},
        "segments": [
            {"key": "k1", "start": 1.0, "end": 2.0, "status": "unlabeled", "speaker": None, "label": None, "voice_best": "tobin"},
            {"key": "k2", "start": 3.0, "end": 4.0, "status": "voice_mismatch", "speaker": "mira", "label": "mira", "voice_best": "tobin"},
            {"key": "k3", "start": 5.0, "end": 6.0, "status": "label_voice_ok", "speaker": "mira", "label": "mira"},
        ],
    }
    monkeypatch.setattr(movie_commands, "_labels_for", lambda v, b: ("feedface00000000aa", labels))
    answers = iter(["", "x"])
    movie_commands.cmd_review_speakers(
        Path("movie.mkv"), tmp_path, play=False, input_fn=lambda prompt: next(answers),
        out=lambda m: None, overrides_dir=tmp_path,
    )
    saved = load_overrides("feedface00000000aa", base=tmp_path)
    assert saved["k2"]["speaker_id"] == "tobin"  # voice-flagged line comes first; Enter accepts the voice match
    assert saved["k1"]["speaker_id"] is None  # marked not usable
    assert "k3" not in saved
