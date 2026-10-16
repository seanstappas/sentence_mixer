#!/usr/bin/env python3
"""
Inventory Builder for Sentence Mixing
Parses Praat .TextGrid alignment files and source video manifests into a searchable,
versioned unit database (InventoryIndex) with sample-accurate intervals, speaker
eligibility, ordered event streams, and backward-compatible legacy views.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

# Ensure UTF-8 output on Windows consoles
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from contracts import (
    INVENTORY_SCHEMA_VERSION,
    AlignedEvent,
    InventoryIndex,
    SourceRecord,
    UtteranceRecord,
    WordOccurrence,
)
from lexicon import get_phone_stress, strip_phone_stress
from media_time import extract_source_record, seconds_to_samples


def parse_textgrid_pure_python(tg_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pure Python parser for Praat .TextGrid files (both short and long text formats).
    Returns dict mapping tier name -> list of intervals [{'xmin': float, 'xmax': float, 'text': str}].
    """
    tiers: Dict[str, List[Dict[str, Any]]] = {}
    with open(tg_path, "r", encoding="utf-8", errors="replace") as f:
        lines = [line.strip() for line in f if line.strip()]

    current_tier = None
    i = 0
    while i < len(lines):
        line = lines[i]

        # Match Long Format tier header: name = "words" or class = "IntervalTier"
        name_match = re.match(r'name\s*=\s*"([^"]+)"', line)
        if name_match:
            current_tier = name_match.group(1).lower()
            if current_tier not in tiers:
                tiers[current_tier] = []
            i += 1
            continue

        # Match Long Format intervals: xmin = ..., xmax = ..., text = "..."
        if current_tier:
            xmin_m = re.match(r'xmin\s*=\s*([0-9.]+)', line)
            if xmin_m:
                xmin = float(xmin_m.group(1))
                xmax = None
                text = ""
                found_text = False
                # Scan next lines for xmax and text
                j = i + 1
                while j < min(i + 5, len(lines)):
                    xmax_m = re.match(r'xmax\s*=\s*([0-9.]+)', lines[j])
                    if xmax_m:
                        xmax = float(xmax_m.group(1))
                    text_m = re.match(r'text\s*=\s*"([^"]*)"', lines[j])
                    if text_m:
                        text = text_m.group(1)
                        found_text = True
                        i = j
                        break
                    j += 1
                if xmax is not None and found_text:
                    tiers[current_tier].append({"xmin": xmin, "xmax": xmax, "text": text})

        i += 1

    return tiers


def parse_textgrid(tg_path: str) -> Dict[str, List[Dict[str, Any]]]:
    """Parse TextGrid using praatio if available, else pure python fallback."""
    try:
        from praatio import textgrid
        tg = textgrid.openTextgrid(tg_path, readEmpty=True)
        parsed_tiers: Dict[str, List[Dict[str, Any]]] = {}
        for tname in tg.tierNames:
            entries = []
            for entry in tg.getTier(tname).entries:
                entries.append({"xmin": entry.start, "xmax": entry.end, "text": entry.label})
            parsed_tiers[tname.lower()] = entries
        return parsed_tiers
    except Exception:
        return parse_textgrid_pure_python(tg_path)


def build_units_inventory(
    aligned_dir: str,
    manifest_path: str,
    output_index_path: str,
    speaker_id: Optional[str] = None,
    single_speaker_verified: bool = False,
    source_record: Optional[SourceRecord] = None,
    speaker_intervals: Optional[List[Any]] = None,
    lexicon: Optional[Any] = None,
    recover_oov_words: bool = False,
) -> Dict[str, Any]:
    """
    Builds a JSON unit inventory of all words and phonemes from aligned TextGrids.
    Produces an InventoryIndex conforming to the v2.0.0 schema while maintaining
    legacy words_inventory and phones_inventory dictionaries for backward compatibility.
    """
    manifest_p = Path(manifest_path).resolve()
    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    # Initialize lexicon for subtitle-anchored OOV word recovery if requested
    if (recover_oov_words or lexicon is not None) and lexicon is None:
        try:
            from lexicon import Lexicon
            repo_root = Path(__file__).resolve().parent.parent
            dict_files = [
                repo_root / "dictionaries" / "custom_pronunciations.txt",
                repo_root / "dictionaries" / "merged_dictionary.dict",
            ]
            lexicon = Lexicon([p for p in dict_files if p.exists()])
        except Exception:
            lexicon = None

    # Resolve source video path relative to manifest's parent directory
    rel_video = manifest.get("source_video", "")
    source_video_path = (manifest_p.parent / rel_video).resolve() if rel_video else manifest_p.parent

    # Determine speaker identity & verification
    is_verified = bool(
        single_speaker_verified
        or manifest.get("single_speaker_verified", False)
        or manifest.get("assume_single_speaker", False)
    )
    if is_verified:
        effective_speaker = speaker_id or manifest.get("speaker_id")
    else:
        effective_speaker = manifest.get("speaker_id")

    active_speaker_intervals = speaker_intervals or manifest.get("speaker_intervals")

    # Source Record & native sample rate
    audio_src = manifest.get("audio_source_path")
    vocal_stem_path = Path(audio_src) if audio_src and Path(audio_src).exists() else None
    is_isolated = bool(vocal_stem_path and "vocal" in vocal_stem_path.name.lower())

    if source_record is None:
        try:
            source_record = extract_source_record(
                source_video_path,
                vocal_stem_path=vocal_stem_path,
                isolate_vocals=is_isolated,
            )
        except Exception:
            # Fallback for synthetic test fixtures without video file
            source_record = SourceRecord(
                content_id=Path(source_video_path).stem or "synth_source",
                source_path=str(source_video_path),
                native_sample_rate=48000,
                native_channels=2,
            )
            if vocal_stem_path:
                source_record.cached_vocal_pcm_path = str(vocal_stem_path.resolve())
    elif not source_record.cached_vocal_pcm_path and vocal_stem_path:
        source_record.cached_vocal_pcm_path = str(vocal_stem_path.resolve())

    sr = source_record.native_sample_rate
    content_id = source_record.content_id

    chunks_meta = manifest.get("chunks", {})
    tg_files = [str(p) for p in sorted(Path(aligned_dir).rglob("*.TextGrid"))]
    print(f"[*] Found {len(tg_files)} .TextGrid files in {aligned_dir}")

    events_dict: Dict[str, AlignedEvent] = {}
    words_dict: Dict[str, WordOccurrence] = {}
    utterances_dict: Dict[str, UtteranceRecord] = {}

    word_to_word_ids: Dict[str, List[str]] = {}
    phone_to_event_ids: Dict[str, List[str]] = {}
    base_phone_to_event_ids: Dict[str, List[str]] = {}

    # Legacy inventory structures for backward compatibility
    legacy_words: Dict[str, List[Dict[str, Any]]] = {}
    legacy_phones: Dict[str, List[Dict[str, Any]]] = {}

    total_events_indexed = 0
    total_words_indexed = 0

    # Sort textgrid files for deterministic event ordering
    tg_files = sorted(tg_files)

    for tg_file in tg_files:
        rel_key = str(Path(tg_file).relative_to(aligned_dir).with_suffix("")).replace("\\", "/")
        stem_key = Path(tg_file).stem
        if rel_key in chunks_meta:
            chunk_id = rel_key
        elif stem_key in chunks_meta:
            chunk_id = stem_key
        else:
            continue


        chunk_info = chunks_meta[chunk_id]
        chunk_start_sec = float(chunk_info.get("start_sec", 0.0))
        chunk_end_sec = float(chunk_info.get("end_sec", 0.0))
        chunk_text = chunk_info.get("text", "")
        chunk_speaker = chunk_info.get("speaker_id") or effective_speaker

        utt_start_sample = seconds_to_samples(chunk_start_sec, sr)
        utt_end_sample = seconds_to_samples(chunk_end_sec, sr)

        utterances_dict[chunk_id] = UtteranceRecord(
            utterance_id=chunk_id,
            source_content_id=content_id,
            start_sample=utt_start_sample,
            end_sample=utt_end_sample,
            transcript=chunk_text,
            speaker_id=chunk_speaker,
            verification_method="verified_manifest" if is_verified else "unverified",
        )

        parsed_tiers = parse_textgrid(tg_file)
        phones_tier = parsed_tiers.get("phones", [])
        words_tier = parsed_tiers.get("words", [])

        # 1. Process phone tier into AlignedEvents
        chunk_event_ids: List[str] = []
        for p_idx, p_entry in enumerate(phones_tier):
            raw_label = p_entry["text"].strip()
            loc_xmin = float(p_entry["xmin"])
            loc_xmax = float(p_entry["xmax"])
            if loc_xmax <= loc_xmin:
                continue

            glob_start_sec = chunk_start_sec + loc_xmin
            glob_end_sec = chunk_start_sec + loc_xmax
            start_sample = seconds_to_samples(glob_start_sec, sr)
            end_sample = seconds_to_samples(glob_end_sec, sr)

            clean_label = raw_label.upper()
            kind: Literal["phone", "silence", "noise"]
            if not clean_label or clean_label in {"SIL", "SP", "<EPS>", ""}:
                kind = "silence"
                label = "sil"
                base_phone = "sil"
                stress = None
            elif clean_label in {"SPN", "NOISE", "<UNK>"}:
                kind = "noise"
                label = "spn"
                base_phone = "spn"
                stress = None
            else:
                kind = "phone"
                label = clean_label
                base_phone = strip_phone_stress(clean_label)
                stress = get_phone_stress(clean_label)

            event_id = f"ev_{content_id}_{chunk_id}_{p_idx:04d}"
            ev = AlignedEvent(
                event_id=event_id,
                source_content_id=content_id,
                utterance_id=chunk_id,
                speaker_id=chunk_speaker,
                kind=kind,
                start_sample=start_sample,
                end_sample=end_sample,
                label=label,
                base_phone=base_phone,
                stress=stress,
            )

            # Link predecessors and successors within the chunk
            if chunk_event_ids:
                prev_id = chunk_event_ids[-1]
                ev.predecessor_id = prev_id
                events_dict[prev_id].successor_id = event_id

            events_dict[event_id] = ev
            chunk_event_ids.append(event_id)
            total_events_indexed += 1

            if kind == "phone":
                if label not in phone_to_event_ids:
                    phone_to_event_ids[label] = []
                phone_to_event_ids[label].append(event_id)

                if base_phone not in base_phone_to_event_ids:
                    base_phone_to_event_ids[base_phone] = []
                base_phone_to_event_ids[base_phone].append(event_id)

                # Legacy phone entry
                if label not in legacy_phones:
                    legacy_phones[label] = []
                legacy_phones[label].append({
                    "source_video": str(source_video_path),
                    "chunk_id": chunk_id,
                    "label": label,
                    "start_sec": glob_start_sec,
                    "end_sec": glob_end_sec,
                    "duration": glob_end_sec - glob_start_sec,
                    "sample_start": start_sample,
                    "sample_end": end_sample,
                    "speaker_id": chunk_speaker,
                    "source_content_id": content_id,
                })

        # Collect silence intervals for this chunk
        chunk_silences: List[Tuple[float, float]] = []
        for p_entry in phones_tier:
            p_lbl = p_entry["text"].strip().upper()
            if not p_lbl or p_lbl in {"SIL", "SP", "<EPS>", ""}:
                chunk_silences.append((chunk_start_sec + float(p_entry["xmin"]), chunk_start_sec + float(p_entry["xmax"])))

        # Pre-collect words to reconcile against speaker intervals
        words_to_reconcile = []
        for w_idx, w_entry in enumerate(words_tier):
            raw_w = w_entry["text"].strip()
            clean_word = re.sub(r"[^\w\'-]", "", raw_w).lower()
            if not clean_word or clean_word in {"sil", "sp", "spn", "eps", "<eps>"}:
                continue
            w_loc_xmin = float(w_entry["xmin"])
            w_loc_xmax = float(w_entry["xmax"])
            if w_loc_xmax <= w_loc_xmin:
                continue
            words_to_reconcile.append({
                "w_idx": w_idx,
                "start_sec": chunk_start_sec + w_loc_xmin,
                "end_sec": chunk_start_sec + w_loc_xmax,
            })

        reconciled_by_idx: Dict[int, Dict[str, Any]] = {}
        chunk_segments = chunk_info.get("segments")
        segment_by_idx: Dict[int, Optional[Dict[str, Any]]] = {}
        if chunk_segments:
            # Movie chunks: words take the speaker of the subtitle segment they came from,
            # matched by token order (exact for two-speaker dialogue-dash cues).
            from movie_subtitles import assign_word_speakers
            seg_words = [w for w in words_tier if w["text"].strip()]
            assigned = assign_word_speakers([w["text"].strip() for w in seg_words], chunk_segments)
            pos_by_id = {id(w): i for i, w in enumerate(seg_words)}
            for w_idx, w_entry in enumerate(words_tier):
                pos = pos_by_id.get(id(w_entry))
                segment_by_idx[w_idx] = assigned[pos] if pos is not None else None
        elif active_speaker_intervals:
            from diarization import reconcile_word_intervals_with_speakers
            reconciled_list = reconcile_word_intervals_with_speakers(
                active_speaker_intervals,
                words_to_reconcile,
                silence_intervals=chunk_silences,
                collar_sec=0.040,
            )
            reconciled_by_idx = {rw["w_idx"]: rw for rw in reconciled_list}

        # 2. Process words tier and associate with constituent phone events
        for w_idx, w_entry in enumerate(words_tier):
            raw_w = w_entry["text"].strip()
            clean_word = re.sub(r"[^\w\'-]", "", raw_w).lower()
            if not clean_word or clean_word in {"sil", "sp", "spn", "eps", "<eps>"}:
                continue

            w_loc_xmin = float(w_entry["xmin"])
            w_loc_xmax = float(w_entry["xmax"])
            if w_loc_xmax <= w_loc_xmin:
                continue

            w_glob_start = chunk_start_sec + w_loc_xmin
            w_glob_end = chunk_start_sec + w_loc_xmax
            w_start_sample = seconds_to_samples(w_glob_start, sr)
            w_end_sample = seconds_to_samples(w_glob_end, sr)

            # Find phone events falling within this word's sample range
            constituent_ids: List[str] = []
            realized_phones: List[str] = []
            for ev_id in chunk_event_ids:
                ev = events_dict[ev_id]
                if ev.kind == "phone":
                    # Check overlap with word interval
                    if max(ev.start_sample, w_start_sample) < min(ev.end_sample, w_end_sample):
                        constituent_ids.append(ev_id)
                        realized_phones.append(ev.label)
                        ev.word_occurrence_id = f"w_{content_id}_{chunk_id}_{w_idx:04d}"

            if not constituent_ids:
                if lexicon is not None:
                    # Subtitle-Anchored OOV Recovery:
                    # When MFA assigns 'spn' to a valid subtitle word, synthesize phone events
                    # across the word interval from Lexicon / G2P so the word is not lost.
                    import unicodedata
                    norm_w = unicodedata.normalize("NFKD", clean_word).encode("ASCII", "ignore").decode("utf-8")
                    prons = lexicon.get_pronunciations(clean_word) or (lexicon.get_pronunciations(norm_w) if norm_w != clean_word else [])
                    if prons:
                        chosen_phones = prons[0]
                        num_p = len(chosen_phones)
                        from target_graph import DEFAULT_DURATION_PRIORS
                        priors = [DEFAULT_DURATION_PRIORS.get(strip_phone_stress(p), 0.10) for p in chosen_phones]
                        total_prior = sum(priors) if sum(priors) > 0 else 1.0
                        total_samples = max(num_p, w_end_sample - w_start_sample)

                        curr_sample = w_start_sample
                        synth_event_ids: List[str] = []
                        word_occ_id = f"w_{content_id}_{chunk_id}_{w_idx:04d}"

                        for p_i, ph in enumerate(chosen_phones):
                            ph_clean = ph.upper()
                            ph_dur_samples = int(round(total_samples * (priors[p_i] / total_prior)))
                            if p_i == num_p - 1:
                                next_sample = w_end_sample
                            else:
                                next_sample = min(w_end_sample, curr_sample + max(1, ph_dur_samples))

                            ev_id = f"ev_{content_id}_{chunk_id}_syn_{w_idx:04d}_{p_i:02d}"
                            base_p = strip_phone_stress(ph_clean)
                            stress_p = get_phone_stress(ph_clean)

                            ev = AlignedEvent(
                                event_id=ev_id,
                                source_content_id=content_id,
                                utterance_id=chunk_id,
                                speaker_id=chunk_speaker,
                                kind="phone",
                                start_sample=curr_sample,
                                end_sample=next_sample,
                                label=ph_clean,
                                base_phone=base_p,
                                stress=stress_p,
                                word_occurrence_id=word_occ_id,
                            )
                            if synth_event_ids:
                                prev_id = synth_event_ids[-1]
                                ev.predecessor_id = prev_id
                                events_dict[prev_id].successor_id = ev_id

                            events_dict[ev_id] = ev
                            synth_event_ids.append(ev_id)
                            constituent_ids.append(ev_id)
                            realized_phones.append(ph_clean)

                            if ph_clean not in phone_to_event_ids:
                                phone_to_event_ids[ph_clean] = []
                            phone_to_event_ids[ph_clean].append(ev_id)

                            if base_p not in base_phone_to_event_ids:
                                base_phone_to_event_ids[base_p] = []
                            base_phone_to_event_ids[base_p].append(ev_id)

                            # Legacy phone entry
                            if ph_clean not in legacy_phones:
                                legacy_phones[ph_clean] = []
                            legacy_phones[ph_clean].append({
                                "source_video": str(source_video_path),
                                "chunk_id": chunk_id,
                                "label": ph_clean,
                                "start_sec": curr_sample / float(sr),
                                "end_sec": next_sample / float(sr),
                                "duration": (next_sample - curr_sample) / float(sr),
                                "sample_start": curr_sample,
                                "sample_end": next_sample,
                                "speaker_id": chunk_speaker,
                                "source_content_id": content_id,
                            })

                            total_events_indexed += 1
                            curr_sample = next_sample

                        if chunk_event_ids and synth_event_ids:
                            prev_ch_id = chunk_event_ids[-1]
                            events_dict[prev_ch_id].successor_id = synth_event_ids[0]
                            events_dict[synth_event_ids[0]].predecessor_id = prev_ch_id
                        chunk_event_ids.extend(synth_event_ids)

            if not constituent_ids:
                continue

            if chunk_segments:
                seg = segment_by_idx.get(w_idx)
                if seg is not None:
                    w_spk = seg.get("speaker_id") or ""
                    w_eligible = bool(w_spk) and bool(seg.get("is_eligible", True))
                    w_reason = "" if w_eligible else (seg.get("eligibility_reason") or "unassigned")
                    w_status = seg.get("identity_status") or ("label" if w_eligible else "unassigned")
                else:
                    w_spk, w_eligible, w_reason, w_status = "", False, "segment_unresolved", "unassigned"
            elif active_speaker_intervals:
                rw = reconciled_by_idx.get(w_idx, {})
                w_spk = rw.get("speaker_id", "UNKNOWN")
                w_eligible = rw.get("is_eligible", False)
                w_reason = rw.get("eligibility_reason", "unassigned")
                w_status = "diarized" if w_eligible else ("unreviewed" if w_reason == "unreviewed_identity" else "ineligible")
            else:
                if is_verified and effective_speaker:
                    w_spk = effective_speaker
                    w_eligible = True
                    w_reason = ""
                    w_status = "asserted" if manifest.get("assume_single_speaker") else "verified"
                else:
                    w_spk = effective_speaker or ""
                    w_eligible = bool(is_verified and effective_speaker)
                    w_reason = "" if w_eligible else "unassigned"
                    w_status = "verified" if w_eligible else "unassigned"

            word_id = f"w_{content_id}_{chunk_id}_{w_idx:04d}"
            wo = WordOccurrence(
                word_id=word_id,
                source_content_id=content_id,
                speaker_id=w_spk,
                word_normalized=clean_word,
                word_original=raw_w,
                start_sample=w_start_sample,
                end_sample=w_end_sample,
                phone_event_ids=constituent_ids,
                realized_phones=realized_phones,
                is_eligible=w_eligible,
                eligibility_reason=w_reason,
                identity_status=w_status,
            )
            words_dict[word_id] = wo
            total_words_indexed += 1

            for ev_id in constituent_ids:
                ev = events_dict[ev_id]
                ev.speaker_id = w_spk
                ev.is_eligible = w_eligible
                ev.eligibility_reason = w_reason
                ev.identity_status = w_status

            if clean_word not in word_to_word_ids:
                word_to_word_ids[clean_word] = []
            word_to_word_ids[clean_word].append(word_id)

            # Legacy word entry
            if clean_word not in legacy_words:
                legacy_words[clean_word] = []
            legacy_words[clean_word].append({
                "source_video": str(source_video_path),
                "chunk_id": chunk_id,
                "label": clean_word,
                "start_sec": w_glob_start,
                "end_sec": w_glob_end,
                "duration": w_glob_end - w_glob_start,
                "sample_start": w_start_sample,
                "sample_end": w_end_sample,
                "phones": realized_phones,
                "speaker_id": w_spk,
                "source_content_id": content_id,
                "is_eligible": w_eligible,
                "eligibility_reason": w_reason,
            })

    inv = InventoryIndex(
        schema_version=INVENTORY_SCHEMA_VERSION,
        sources={content_id: source_record},
        utterances=utterances_dict,
        events=events_dict,
        words=words_dict,
        word_to_word_ids=word_to_word_ids,
        phone_to_event_ids=phone_to_event_ids,
        base_phone_to_event_ids=base_phone_to_event_ids,
    )
    inv.validate()

    reconciled_legacy_phones: Dict[str, List[Dict[str, Any]]] = {}
    for ev in events_dict.values():
        if ev.kind == "phone":
            reconciled_legacy_phones.setdefault(ev.label, []).append({
                "source_video": str(source_video_path),
                "chunk_id": ev.utterance_id,
                "label": ev.label,
                "start_sec": ev.start_sample / sr,
                "end_sec": ev.end_sample / sr,
                "duration": (ev.end_sample - ev.start_sample) / sr,
                "sample_start": ev.start_sample,
                "sample_end": ev.end_sample,
                "speaker_id": ev.speaker_id,
                "source_content_id": content_id,
                "is_eligible": getattr(ev, "is_eligible", True),
                "eligibility_reason": getattr(ev, "eligibility_reason", ""),
            })

    out_dict = inv.to_dict()
    # Add legacy compatibility fields
    out_dict["source_video"] = str(source_video_path)
    out_dict["words_inventory"] = legacy_words
    out_dict["phones_inventory"] = reconciled_legacy_phones
    out_dict["phones"] = reconciled_legacy_phones

    out_p = Path(output_index_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)
    with open(out_p, "w", encoding="utf-8") as f:
        if manifest.get("movie"):
            json.dump(out_dict, f, separators=(",", ":"))
        else:
            json.dump(out_dict, f, indent=2)

    uri_str = out_p.as_uri() if hasattr(out_p, "as_uri") else str(out_p)
    print(f"[+] Successfully indexed {total_words_indexed} words and {total_events_indexed} events to {uri_str}")
    return out_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build units inventory from aligned TextGrids.")
    parser.add_argument("--aligned-dir", required=True, help="Directory containing Praat .TextGrid files")
    parser.add_argument("--manifest", required=True, help="Path to corpus_manifest.json")
    parser.add_argument("--output", default="./units_index.json", help="Path to output units_index.json")
    parser.add_argument("--speaker", default=None, help="Speaker identity")
    parser.add_argument("--assume-single-speaker", action="store_true", help="Assume single speaker")
    parser.add_argument("--vocal-stem", default=None, help="Path to cached vocal stem WAV")
    args = parser.parse_args()

    custom_source_rec = None
    if args.vocal_stem:
        v_stem_p = Path(args.vocal_stem).resolve()
        with open(args.manifest, "r", encoding="utf-8") as mf:
            m_data = json.load(mf)
        v_path = Path(m_data.get("source_video", ""))
        try:
            custom_source_rec = extract_source_record(
                v_path,
                vocal_stem_path=v_stem_p,
                isolate_vocals=True,
            )
        except Exception:
            custom_source_rec = SourceRecord(
                content_id=v_path.stem or "source",
                source_path=str(v_path),
                native_sample_rate=48000,
                native_channels=2,
                cached_vocal_pcm_path=str(v_stem_p),
            )

    build_units_inventory(
        args.aligned_dir,
        args.manifest,
        args.output,
        speaker_id=args.speaker,
        single_speaker_verified=args.assume_single_speaker,
        source_record=custom_source_rec,
    )
