"""
Content-aware caching for sentence mixing pipeline.
Computes deterministic fingerprints for stages (sources, alignments, inventories, features, plans).
Ensures atomic publishing and invalidation.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union


def hash_string(data: str) -> str:
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def hash_dict(d: Dict[str, Any]) -> str:
    def _default_encoder(obj: Any) -> Any:
        if hasattr(obj, "to_dict") and callable(obj.to_dict):
            return obj.to_dict()
        if hasattr(obj, "__dict__"):
            return obj.__dict__
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    serialized = json.dumps(d, default=_default_encoder, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def hash_file_content(file_path: Union[str, Path], chunk_size: int = 65536) -> str:
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"File not found for hashing: {p}")
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_media_hash(file_path: Union[str, Path], cache_dir: Optional[Union[str, Path]] = None) -> str:
    """
    Streams a full SHA-256 of the media file.
    Memoizes results by resolved path, file size, and st_mtime_ns in an index under cache root
    so unchanged large files are not re-hashed on every run.
    Raises FileNotFoundError / IOError on read failures loudly (no path-hash fallback).
    """
    p = Path(file_path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"Media file not found for hashing: {p}")

    stat = p.stat()
    size = stat.st_size
    mtime_ns = stat.st_mtime_ns

    index_dir = Path(cache_dir).resolve() if cache_dir else Path(".build").resolve()
    index_file = index_dir / ".media_hash_index.json"

    index: Dict[str, Any] = {}
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                index = json.load(f)
        except Exception:
            index = {}

    posix_path = p.as_posix()
    cached_entry = index.get(posix_path)
    if (
        isinstance(cached_entry, dict)
        and cached_entry.get("size") == size
        and cached_entry.get("mtime_ns") == mtime_ns
        and isinstance(cached_entry.get("sha256"), str)
    ):
        return cached_entry["sha256"]

    # Full SHA-256 computation
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    full_sha = h.hexdigest()

    # Update index
    index[posix_path] = {
        "size": size,
        "mtime_ns": mtime_ns,
        "sha256": full_sha,
    }
    try:
        index_dir.mkdir(parents=True, exist_ok=True)
        tmp_idx = index_file.with_suffix(".tmp")
        with open(tmp_idx, "w", encoding="utf-8") as f:
            json.dump(index, f, indent=2)
        os.replace(tmp_idx, index_file)
    except Exception:
        pass

    return full_sha


def compute_alignment_cache_key(
    media_hash: str,
    selected_audio_stream: int,
    selected_video_stream: int,
    transcript_hash: str,
    dict_hash: str,
    acoustic_model: str = "english_us_arpa",
    mfa_options: Optional[Dict[str, Any]] = None,
    preprocessor_version: str = "2.0",
) -> str:
    key_dict = {
        "media_hash": media_hash,
        "selected_audio_stream": selected_audio_stream,
        "selected_video_stream": selected_video_stream,
        "transcript_hash": transcript_hash,
        "dict_hash": dict_hash,
        "acoustic_model": acoustic_model,
        "mfa_options": mfa_options or {},
        "preprocessor_version": preprocessor_version,
    }
    return hash_dict(key_dict)


def compute_labeled_inventory_cache_key(
    alignment_key: str,
    speaker_id: Optional[str] = None,
    single_speaker_verified: bool = False,
    assume_single_speaker: bool = False,
    speaker_intervals_hash: Optional[str] = None,
    diarization_config: Optional[Dict[str, Any]] = None,
    registry_version: Optional[str] = None,
) -> str:
    key_dict = {
        "alignment_key": alignment_key,
        "speaker_id": speaker_id or "",
        "single_speaker_verified": bool(single_speaker_verified),
        "assume_single_speaker": bool(assume_single_speaker),
        "speaker_intervals_hash": speaker_intervals_hash or "",
        "diarization_config": diarization_config or {},
        "registry_version": registry_version or "",
    }
    return hash_dict(key_dict)


def compute_stem_cache_key(
    media_hash: str,
    separator_model: str = "UVR-MDX-NET-Inst_HQ_3",
    options: Optional[Dict[str, Any]] = None,
) -> str:
    key_dict = {
        "media_hash": media_hash,
        "separator_model": separator_model,
        "options": options or {},
    }
    return hash_dict(key_dict)


class CacheManager:
    """
    Manages structured project caches under .build/
    """
    def __init__(self, build_dir: Union[str, Path] = ".build"):
        self.build_dir = Path(build_dir).resolve()
        self.sources_dir = self.build_dir / "sources"
        self.alignment_dir = self.build_dir / "alignment"
        self.inventory_dir = self.build_dir / "inventory"
        self.features_dir = self.build_dir / "features"
        self.runs_dir = self.build_dir / "runs"

        for d in [self.sources_dir, self.alignment_dir, self.inventory_dir, self.features_dir, self.runs_dir]:
            d.mkdir(parents=True, exist_ok=True)

    def get_source_cache_dir(self, content_id: str) -> Path:
        d = self.sources_dir / content_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def compute_alignment_fingerprint(
        self,
        audio_hashes: Dict[str, str],
        transcript_content: str,
        dict_hashes: Dict[str, str],
        acoustic_model: str,
        mfa_options: Dict[str, Any],
    ) -> str:
        key_data = {
            "audio_hashes": audio_hashes,
            "transcript_content": transcript_content,
            "dict_hashes": dict_hashes,
            "acoustic_model": acoustic_model,
            "mfa_options": mfa_options,
        }
        return hash_dict(key_data)[:16]

    def compute_inventory_fingerprint(
        self,
        alignment_fingerprint: str,
        lexicon_fingerprint: str,
        schema_version: str,
    ) -> str:
        key_data = {
            "alignment_fingerprint": alignment_fingerprint,
            "lexicon_fingerprint": lexicon_fingerprint,
            "schema_version": schema_version,
        }
        return hash_dict(key_data)[:16]

    def compute_feature_fingerprint(
        self,
        source_audio_hash: str,
        feature_config: Dict[str, Any],
    ) -> str:
        key_data = {
            "source_audio_hash": source_audio_hash,
            "feature_config": feature_config,
        }
        return hash_dict(key_data)[:16]

    def atomic_publish_file(self, temp_src: Union[str, Path], target_dest: Union[str, Path]) -> Path:
        """Atomically moves/publishes a file to target_dest via a temporary rename."""
        temp_p = Path(temp_src).resolve()
        dest_p = Path(target_dest).resolve()
        dest_p.parent.mkdir(parents=True, exist_ok=True)

        temp_staging = dest_p.with_suffix(f"{dest_p.suffix}.tmp.{os.getpid()}")
        shutil.copy2(temp_p, temp_staging)
        os.replace(temp_staging, dest_p)
        return dest_p

    def atomic_publish_dir(self, temp_dir: Union[str, Path], target_dir: Union[str, Path]) -> Path:
        """Atomically replaces target_dir with temp_dir."""
        temp_d = Path(temp_dir).resolve()
        dest_d = Path(target_dir).resolve()
        dest_d.parent.mkdir(parents=True, exist_ok=True)

        if dest_d.exists():
            shutil.rmtree(dest_d)
        shutil.move(str(temp_d), str(dest_d))
        return dest_d


def format_bytes(num_bytes: int) -> str:
    """Formats bytes into human-readable string (e.g. 1.2 MB)."""
    val = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(val) < 1024.0:
            return f"{val:3.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} TB"


def _get_path_size(p: Path) -> int:
    """Computes total bytes of a file or directory tree."""
    if not p.exists():
        return 0
    if p.is_file():
        try:
            return p.stat().st_size
        except OSError:
            return 0
    total = 0
    try:
        for root, _, files in os.walk(p):
            for f in files:
                fp = Path(root) / f
                try:
                    total += fp.stat().st_size
                except OSError:
                    pass
    except OSError:
        pass
    return total


def clean_source_cache(
    source_target: Union[str, Path],
    cache_dir: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    clean_speakers: bool = True,
    registry_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Finds and cleans all cached processing files and build artifacts for a specific source.
    Accepts:
      - Video file path (e.g. 'input/crisis.mp4')
      - Clean source stem (e.g. 'crisis')
      - Source ID (e.g. 'crisis_9c96ee02')
    Cleans:
      - .build/sources/<source_id>*
      - .build/alignment/<source_id>*
      - .build/inventory/<source_id>*
      - .build/vocals/<source_id>*
      - .build/audio_extracts/<clean_stem>*.wav
      - .build/clusters/*<clean_stem>* / *<source_id>*
      - .build/mfa/temp/*<source_id>*
      - .build/movies/<media hash 16>/ and .build/subtitles/<media hash 16>/ (whole-movie caches)
      - Removes entry from .build/.media_hash_index.json
    Keeps user inputs: labeled VTTs in .build/subtitles/labeled/ and reviewed labels in
    config/speaker_labels/.
    """
    base_dir = Path(cache_dir).resolve() if cache_dir else Path(".build").resolve()
    target_str = str(source_target).strip()
    target_p = Path(target_str)

    matched_source_ids: Set[str] = set()
    matched_stems: Set[str] = set()
    matched_media_keys: Set[str] = set()
    matched_hashes: Set[str] = set()

    # 1. Direct file check
    if target_p.exists() and target_p.is_file():
        abs_p = target_p.resolve()
        posix_p = abs_p.as_posix()
        matched_media_keys.add(posix_p)
        matched_media_keys.add(str(abs_p))
        clean_stem = re.sub(r"[^\w\-_]", "_", abs_p.stem)[:40]
        matched_stems.add(clean_stem)
        try:
            full_sha = compute_media_hash(abs_p, cache_dir=base_dir)
            matched_hashes.add(full_sha)
            short_sha = full_sha[:8]
            matched_source_ids.add(f"{clean_stem}_{short_sha}")
            matched_source_ids.add(f"{clean_stem}_{full_sha[:16]}")
        except Exception:
            pass
    else:
        # User passed a stem or source_id like "crisis" or "crisis_9c96ee02"
        clean_target = re.sub(r"[^\w\-_]", "_", target_p.stem if target_p.suffix else target_str)[:40]
        if clean_target:
            matched_stems.add(clean_target)
            matched_source_ids.add(clean_target)

    # 2. Match against .media_hash_index.json
    index_file = base_dir / ".media_hash_index.json"
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                media_index = json.load(f)
            for k, val in media_index.items():
                k_p = Path(k)
                k_stem = re.sub(r"[^\w\-_]", "_", k_p.stem)[:40]
                sha = val.get("sha256", "") if isinstance(val, dict) else ""
                short_sha = sha[:8]
                sid = f"{k_stem}_{short_sha}"

                matches = False
                if target_str in (k, k_p.name, k_p.stem, k_stem, sid):
                    matches = True
                elif target_str.lower() in (k.lower(), k_p.name.lower(), k_p.stem.lower(), k_stem.lower(), sid.lower()):
                    matches = True
                elif any(s.lower() in (k_stem.lower(), sid.lower()) for s in matched_stems if len(s) >= 3):
                    matches = True
                elif sha and (target_str.lower() == sha.lower() or sha.startswith(target_str.lower())):
                    matches = True

                if matches:
                    matched_media_keys.add(k)
                    matched_stems.add(k_stem)
                    if sha:
                        matched_hashes.add(sha)
                    if short_sha:
                        matched_source_ids.add(sid)
                        matched_source_ids.add(f"{k_stem}_{sha[:16]}")
        except Exception:
            pass

    # 3. Match against existing subdirectories in cache: sources, alignment, inventory, vocals.
    # A directory belongs to a stem only as "<stem>" or "<stem>_<content hash>"; looser prefix
    # matching would also delete other sources sharing a prefix (e.g. "Star_Wars_Episode_II"
    # when cleaning Episode III, or "anakin_ep2_2" when cleaning "anakin_ep2_1").
    def _owned_by(name: str, stem: str) -> bool:
        return bool(re.fullmatch(re.escape(stem.lower()) + r"(_[0-9a-f]{8,16})?", name.lower()))

    scan_subdirs = ["sources", "alignment", "inventory", "vocals"]
    for s_sub in scan_subdirs:
        sub_path = base_dir / s_sub
        if sub_path.exists():
            for child in sub_path.iterdir():
                if child.is_dir():
                    c_name = child.name
                    for stem in list(matched_stems):
                        if _owned_by(c_name, stem):
                            matched_source_ids.add(c_name)

    deleted_dirs: List[Path] = []
    deleted_files: List[Path] = []

    # Identify directories to delete
    for s_sub in ["sources", "alignment", "inventory", "vocals"]:
        sub_path = base_dir / s_sub
        if not sub_path.exists():
            continue
        for child in sub_path.iterdir():
            if not child.is_dir():
                continue
            should_delete = False
            if child.name in matched_source_ids:
                should_delete = True
            elif any(_owned_by(child.name, stem) for stem in matched_stems if len(stem) >= 3):
                should_delete = True

            if should_delete and child not in deleted_dirs:
                deleted_dirs.append(child)

    # Whole-movie caches keyed by media hash (audio FLACs, stems, features, labels, subtitles)
    for sha in matched_hashes:
        for movie_cache in (base_dir / "movies" / sha[:16], base_dir / "subtitles" / sha[:16]):
            if movie_cache.is_dir() and movie_cache not in deleted_dirs:
                deleted_dirs.append(movie_cache)

    # Audio extracts
    audio_ext_dir = base_dir / "audio_extracts"
    if audio_ext_dir.exists():
        for child in audio_ext_dir.iterdir():
            if child.is_file():
                for stem in matched_stems:
                    if child.stem.lower() == stem.lower() or child.stem.lower().startswith(f"{stem.lower()}_"):
                        if child not in deleted_files:
                            deleted_files.append(child)

    # Clusters
    for c_dir in [base_dir / "clusters", base_dir / ".build" / "clusters"]:
        if c_dir.exists():
            for child in c_dir.iterdir():
                if child.is_file():
                    c_lower = child.name.lower()
                    should_del = False
                    for sid in matched_source_ids:
                        if sid.lower() in c_lower or sid.replace(":", "_").lower() in c_lower:
                            should_del = True
                            break
                    if not should_del:
                        for stem in matched_stems:
                            if len(stem) >= 3 and f"_{stem.lower()}_" in f"_{c_lower}_":
                                should_del = True
                                break
                    if should_del and child not in deleted_files:
                        deleted_files.append(child)

    # MFA temp
    for mfa_temp in [base_dir / "mfa" / "temp", base_dir / "mfa"]:
        if mfa_temp.exists():
            for child in mfa_temp.iterdir():
                if child.is_dir() and child.name not in ("pretrained_models", "models", "extracted_models", "joblib_cache"):
                    for sid in matched_source_ids:
                        if sid.lower() in child.name.lower():
                            if child not in deleted_dirs:
                                deleted_dirs.append(child)

    # Calculate total size before deletion
    total_freed_bytes = sum(_get_path_size(p) for p in deleted_dirs) + sum(_get_path_size(p) for p in deleted_files)

    # Media keys to remove
    media_keys_to_remove: List[str] = []
    if index_file.exists():
        try:
            with open(index_file, "r", encoding="utf-8") as f:
                media_index = json.load(f)
            for k, val in media_index.items():
                if k in matched_media_keys:
                    media_keys_to_remove.append(k)
                    continue
                k_p = Path(k)
                k_stem = re.sub(r"[^\w\-_]", "_", k_p.stem)[:40]
                if any(k_stem.lower() == stem.lower() for stem in matched_stems):
                    media_keys_to_remove.append(k)
        except Exception:
            pass

    # Perform deletion if not dry-run
    if not dry_run:
        for f_file in deleted_files:
            try:
                if f_file.exists():
                    f_file.unlink()
            except OSError:
                pass
        for d in deleted_dirs:
            try:
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass

        if media_keys_to_remove and index_file.exists():
            try:
                with open(index_file, "r", encoding="utf-8") as f:
                    media_index = json.load(f)
                for k in media_keys_to_remove:
                    media_index.pop(k, None)
                tmp_idx = index_file.with_suffix(".tmp")
                with open(tmp_idx, "w", encoding="utf-8") as f:
                    json.dump(media_index, f, indent=2)
                os.replace(tmp_idx, index_file)
            except Exception:
                pass

    removed_speakers: List[str] = []
    if clean_speakers:
        reg_p = Path(registry_path) if registry_path else (base_dir.parent / "config" / "speaker_registry.json")
        if reg_p.exists():
            try:
                from speaker_registry import SpeakerRegistry
                reg = SpeakerRegistry(config_path=reg_p)
                removed_speakers = reg.remove_source_speakers(
                    source_ids=matched_source_ids,
                    matched_stems=matched_stems,
                    dry_run=dry_run,
                )
            except Exception:
                pass

    return {
        "target": target_str,
        "matched_source_ids": sorted(list(matched_source_ids)),
        "matched_stems": sorted(list(matched_stems)),
        "deleted_dirs": [str(d) for d in deleted_dirs],
        "deleted_files": [str(f) for f in deleted_files],
        "removed_media_keys": media_keys_to_remove,
        "removed_speakers": removed_speakers,
        "freed_bytes": total_freed_bytes,
        "dry_run": dry_run,
    }


def clean_all_source_caches(
    cache_dir: Optional[Union[str, Path]] = None,
    dry_run: bool = False,
    clean_speakers: bool = True,
    registry_path: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Cleans all cached sources, alignments, inventories, vocals, movie caches, extracted
    subtitle tracks, audio extracts, and clusters from cache_dir, while preserving models,
    staged dictionary, bootstrap files, and user inputs (labeled VTTs, reviewed labels).
    """
    base_dir = Path(cache_dir).resolve() if cache_dir else Path(".build").resolve()
    deleted_dirs: List[Path] = []
    deleted_files: List[Path] = []

    # Clean subdirectories under sources, alignment, inventory, vocals, movies
    for s_sub in ["sources", "alignment", "inventory", "vocals", "movies"]:
        sub_path = base_dir / s_sub
        if sub_path.exists():
            for child in sub_path.iterdir():
                if child.is_dir() and child not in deleted_dirs:
                    deleted_dirs.append(child)

    # Extracted embedded subtitle tracks (keep labeled VTTs and tool-extracted sources: user inputs)
    subs_dir = base_dir / "subtitles"
    if subs_dir.exists():
        for child in subs_dir.iterdir():
            if child.is_dir() and child.name not in ("labeled", "source") and child not in deleted_dirs:
                deleted_dirs.append(child)

    # Audio extracts
    audio_ext_dir = base_dir / "audio_extracts"
    if audio_ext_dir.exists():
        for child in audio_ext_dir.iterdir():
            if child.is_file() and child not in deleted_files:
                deleted_files.append(child)

    # Clusters
    for c_dir in [base_dir / "clusters", base_dir / ".build" / "clusters"]:
        if c_dir.exists():
            for child in c_dir.iterdir():
                if child.is_file() and child not in deleted_files:
                    deleted_files.append(child)

    # MFA temp
    mfa_temp = base_dir / "mfa" / "temp"
    if mfa_temp.exists():
        for child in mfa_temp.iterdir():
            if child.is_dir() and child not in deleted_dirs:
                deleted_dirs.append(child)

    total_freed_bytes = sum(_get_path_size(p) for p in deleted_dirs) + sum(_get_path_size(p) for p in deleted_files)

    index_file = base_dir / ".media_hash_index.json"
    cleared_index = False
    if index_file.exists():
        total_freed_bytes += _get_path_size(index_file)
        if not dry_run:
            try:
                index_file.unlink()
                cleared_index = True
            except OSError:
                pass
        else:
            cleared_index = True

    removed_speakers: List[str] = []
    if clean_speakers:
        reg_p = Path(registry_path) if registry_path else (base_dir.parent / "config" / "speaker_registry.json")
        if reg_p.exists():
            try:
                from speaker_registry import SpeakerRegistry
                reg = SpeakerRegistry(config_path=reg_p)
                removed_speakers = reg.clear(dry_run=dry_run)
            except Exception:
                pass

    if not dry_run:
        for f in deleted_files:
            try:
                if f.exists():
                    f.unlink()
            except OSError:
                pass
        for d in deleted_dirs:
            try:
                if d.exists():
                    shutil.rmtree(d, ignore_errors=True)
            except OSError:
                pass

    return {
        "deleted_dirs": [str(d) for d in deleted_dirs],
        "deleted_files": [str(f) for f in deleted_files],
        "cleared_media_index": cleared_index,
        "removed_speakers": removed_speakers,
        "freed_bytes": total_freed_bytes,
        "dry_run": dry_run,
    }


def _normalize_speaker_id(speaker: str) -> str:
    """Normalize a speaker name or ID for consistent matching."""
    return re.sub(r"[-\s]+", "_", speaker.strip().lower())


def resolve_inventory_path(path_or_dir: Union[str, Path]) -> Path:
    """Resolve an inventory directory or file path to units_index.json.gz (preferred) or units_index.json."""
    p = Path(path_or_dir)
    if p.is_dir():
        gz = p / "units_index.json.gz"
        return gz if gz.exists() else (p / "units_index.json")
    if p.exists():
        return p
    if p.name == "units_index.json":
        gz = p.with_name("units_index.json.gz")
        if gz.exists():
            return gz
    elif p.name == "units_index.json.gz":
        raw = p.with_name("units_index.json")
        if raw.exists():
            return raw
    return p


def load_inventory_json(path_or_dir: Union[str, Path]) -> Dict[str, Any]:
    """Load JSON dict transparently from units_index.json.gz or units_index.json."""
    p = resolve_inventory_path(path_or_dir)
    if p.name.endswith(".gz"):
        with gzip.open(p, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_inventory_json(
    data: Dict[str, Any],
    output_path: Union[str, Path],
    compress: Optional[bool] = None,
) -> Path:
    """Write compact JSON to .json.gz (if output_path ends with .gz or compress=True) or .json."""
    p = Path(output_path)
    do_compress = str(p).endswith(".gz") if compress is None else compress
    if do_compress:
        if not str(p).endswith(".gz"):
            p = p.with_name(p.name + ".gz")
        p.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(p, "wt", encoding="utf-8", compresslevel=6) as f:
            json.dump(data, f, separators=(",", ":"))
    else:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(data, f, separators=(",", ":"))
    return p


def is_valid_inventory(inventory_file: Union[str, Path]) -> bool:
    """Check if an inventory file exists, is readable, and contains actual units."""
    p = resolve_inventory_path(inventory_file)
    if not p.exists() or p.stat().st_size < 20:
        return False

    # Check sibling inventory_metadata.json for instant (< 1 ms) validation
    meta_p = p.parent / "inventory_metadata.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            if "word_count" in meta or "event_count" in meta or "speakers" in meta:
                return bool(meta.get("word_count", 0) > 0 or meta.get("event_count", 0) > 0 or meta.get("speakers"))
        except Exception:
            pass

    try:
        data = load_inventory_json(p)
    except Exception:
        return False

    words = data.get("words")
    if isinstance(words, dict) and len(words) > 0:
        return True
    events = data.get("events")
    if isinstance(events, dict) and len(events) > 0:
        return True
    phones = data.get("phones")
    if isinstance(phones, dict) and len(phones) > 0:
        return True
    return False


def inventory_has_speaker(inventory_file: Union[str, Path], target_speaker: str) -> bool:
    """Check if an inventory contains any eligible units for target_speaker."""
    p = resolve_inventory_path(inventory_file)
    if not p.exists() or p.stat().st_size < 20:
        return False

    norm_target = _normalize_speaker_id(target_speaker)

    # Check sibling inventory_metadata.json for instant (< 0.1 ms) speaker check
    meta_p = p.parent / "inventory_metadata.json"
    if meta_p.exists():
        try:
            meta = json.loads(meta_p.read_text(encoding="utf-8"))
            spks = meta.get("speakers")
            if isinstance(spks, dict):
                return any(_normalize_speaker_id(k) == norm_target and cnt > 0 for k, cnt in spks.items())
        except Exception:
            pass

    try:
        data = load_inventory_json(p)
    except Exception:
        return False

    # 1. Check words
    if "words" in data and isinstance(data["words"], dict):
        for item in data["words"].values():
            if isinstance(item, dict):
                spk = item.get("speaker_id")
                if spk and _normalize_speaker_id(spk) == norm_target:
                    if item.get("is_eligible", True):
                        return True
            elif isinstance(item, list):
                for inst in item:
                    if isinstance(inst, dict):
                        spk = inst.get("speaker_id")
                        if spk and _normalize_speaker_id(spk) == norm_target:
                            if inst.get("is_eligible", True):
                                return True

    # 2. Check events
    if "events" in data and isinstance(data["events"], dict):
        for ev in data["events"].values():
            if isinstance(ev, dict):
                spk = ev.get("speaker_id")
                if spk and _normalize_speaker_id(spk) == norm_target:
                    if ev.get("is_eligible", True):
                        return True

    # 3. Check phones
    if "phones" in data and isinstance(data["phones"], dict):
        for item in data["phones"].values():
            if isinstance(item, list):
                for inst in item:
                    if isinstance(inst, dict):
                        spk = inst.get("speaker_id")
                        if spk and _normalize_speaker_id(spk) == norm_target:
                            if inst.get("is_eligible", True):
                                return True

    return False


def find_cached_inventories_for_speaker(
    speaker_id: str,
    cache_dir: Union[str, Path],
    registry_path: Optional[Union[str, Path]] = None,
) -> List[str]:
    """
    Find all cached units_index paths (.json.gz or .json) that contain units for a given speaker.
    Checks SpeakerRegistry for associated source_ids first, and scans cache_dir/inventory.
    """
    base_cache = Path(cache_dir).resolve()
    inv_dir = base_cache / "inventory"
    if not inv_dir.exists():
        return []

    matched_paths: Set[str] = set()

    # 1. Check registry profile source_ids
    if registry_path:
        reg_p = Path(registry_path).resolve()
        if reg_p.exists():
            try:
                from speaker_registry import SpeakerRegistry
                reg = SpeakerRegistry(config_path=reg_p)
                norm_spk = _normalize_speaker_id(speaker_id)
                for spk_key, prof in reg.speakers.items():
                    if _normalize_speaker_id(spk_key) == norm_spk or _normalize_speaker_id(prof.display_name) == norm_spk:
                        for sid in getattr(prof, "source_ids", []):
                            cand_dir = inv_dir / sid
                            cand = resolve_inventory_path(cand_dir)
                            if cand.exists() and is_valid_inventory(cand) and inventory_has_speaker(cand, speaker_id):
                                matched_paths.add(str(cand.resolve()))
            except Exception:
                pass

    # 2. Scan all subdirectories in inv_dir
    for sub in inv_dir.iterdir():
        if sub.is_dir() and not sub.name.startswith("."):
            cand = resolve_inventory_path(sub)
            cand_resolved = str(cand.resolve())
            if cand_resolved not in matched_paths:
                if cand.exists() and is_valid_inventory(cand) and inventory_has_speaker(cand, speaker_id):
                    matched_paths.add(cand_resolved)

    return sorted(list(matched_paths))


def find_all_cached_inventories(cache_dir: Union[str, Path]) -> List[str]:
    """Find all valid cached units_index paths (.json.gz or .json) in cache_dir/inventory."""
    base_cache = Path(cache_dir).resolve()
    inv_dir = base_cache / "inventory"
    if not inv_dir.exists():
        return []

    valid_paths: List[str] = []
    for sub in sorted(inv_dir.iterdir()):
        if sub.is_dir() and not sub.name.startswith("."):
            cand = resolve_inventory_path(sub)
            if cand.exists() and is_valid_inventory(cand):
                valid_paths.append(str(cand.resolve()))

    return valid_paths

