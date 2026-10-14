"""
Shared typed data contracts and schema validation for sentence mixing pipeline.
Implements the contracts defined in design/design.md and design/milestones.md.
All intervals are half-open: [start_sample, end_sample).
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Set, Tuple, Union

INVENTORY_SCHEMA_VERSION = "2.0.0"
FEATURE_SCHEMA_VERSION = "2.0.0"
EDIT_PLAN_SCHEMA_VERSION = "2.0.0"


@dataclass
class RationalTime:
    """Rational representation of time / frame rates (e.g. 30000/1001)."""
    numerator: int
    denominator: int

    @classmethod
    def from_str(cls, val: str) -> "RationalTime":
        val = val.strip()
        if "/" in val:
            parts = val.split("/", 1)
            return cls(int(parts[0]), int(parts[1]))
        f = float(val)
        # Handle common broadcast frame rates
        if abs(f - 29.97) < 0.01:
            return cls(30000, 1001)
        if abs(f - 23.976) < 0.01 or abs(f - 23.98) < 0.01:
            return cls(24000, 1001)
        if abs(f - 59.94) < 0.01:
            return cls(60000, 1001)
        frac = Fraction(f).limit_denominator(1001)
        return cls(frac.numerator, frac.denominator)

    @classmethod
    def from_number(cls, val: Union[int, float, Fraction, "RationalTime"]) -> "RationalTime":
        if isinstance(val, RationalTime):
            return val
        if isinstance(val, Fraction):
            return cls(val.numerator, val.denominator)
        return cls.from_str(str(val))

    def as_fraction(self) -> Fraction:
        return Fraction(self.numerator, self.denominator)

    def as_float(self) -> float:
        return float(self.numerator) / float(self.denominator)

    def to_dict(self) -> Dict[str, int]:
        return {"numerator": self.numerator, "denominator": self.denominator}

    def validate(self) -> None:
        if self.denominator <= 0:
            raise ValueError(f"RationalTime denominator must be positive: {self.denominator}")
        if self.numerator < 0:
            raise ValueError(f"RationalTime numerator cannot be negative: {self.numerator}")


@dataclass
class SourceRecord:
    """Metadata for an original source media asset."""
    content_id: str
    source_path: str
    native_sample_rate: int
    native_channels: int
    selected_audio_stream: int = 0
    selected_video_stream: int = 0
    decoded_audio_origin_sec: float = 0.0
    decoded_video_origin_sec: float = 0.0
    rational_time_base: Dict[str, int] = field(default_factory=lambda: {"numerator": 1, "denominator": 1000})
    video_fps: Dict[str, int] = field(default_factory=lambda: {"numerator": 30000, "denominator": 1001})
    video_duration_sec: float = 0.0
    audio_duration_sec: float = 0.0
    codec_delay_samples: int = 0
    recording_id: str = ""
    cached_pcm_path: str = ""
    cached_vocal_pcm_path: str = ""
    cached_instrumental_pcm_path: str = ""
    cached_analysis_pcm_path: str = ""

    def validate(self) -> None:
        if not self.content_id:
            raise ValueError("SourceRecord must have a non-empty content_id")
        if self.native_sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {self.native_sample_rate}")
        if self.native_channels <= 0:
            raise ValueError(f"Invalid channels: {self.native_channels}")


@dataclass
class UtteranceRecord:
    """An utterance or chunk extracted from a source for alignment."""
    utterance_id: str
    source_content_id: str
    start_sample: int
    end_sample: int
    transcript: str
    speaker_id: str
    verification_method: str  # e.g. "verified_manifest", "assume_single_speaker", "annotated_interval"
    chunk_padding_samples: int = 0
    quality_flags: List[str] = field(default_factory=list)
    is_visible: Optional[bool] = None

    def validate(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError(f"Invalid utterance sample range: [{self.start_sample}, {self.end_sample})")


@dataclass
class AlignedEvent:
    """An individual aligned acoustic event (phone, silence, or noise)."""
    event_id: str
    source_content_id: str
    utterance_id: str
    speaker_id: str
    kind: Literal["phone", "silence", "noise"]
    start_sample: int
    end_sample: int
    label: str  # ARPAbet phone (with stress if phone), e.g. 'OW1', or 'sil', 'sp'
    base_phone: str = ""  # Base phone stripped of stress, e.g. 'OW'
    stress: Optional[int] = None
    word_occurrence_id: Optional[str] = None
    predecessor_id: Optional[str] = None
    successor_id: Optional[str] = None
    confidence: Optional[float] = None
    quality_flags: List[str] = field(default_factory=list)
    is_eligible: bool = True
    eligibility_reason: str = ""
    identity_status: str = "verified"

    def validate(self) -> None:
        if self.start_sample < 0 or self.end_sample < self.start_sample:
            raise ValueError(f"Invalid event sample bounds: [{self.start_sample}, {self.end_sample})")
        if self.kind not in ("phone", "silence", "noise"):
            raise ValueError(f"Invalid event kind: {self.kind}")


@dataclass
class WordOccurrence:
    """An aligned word occurrence linking to its constituent aligned events."""
    word_id: str
    source_content_id: str
    speaker_id: str
    word_normalized: str
    word_original: str
    start_sample: int
    end_sample: int
    phone_event_ids: List[str]
    realized_phones: List[str]
    pronunciation_id: str = ""
    preceding_silence_sample_duration: int = 0
    following_silence_sample_duration: int = 0
    is_eligible: bool = True
    eligibility_reason: str = ""
    identity_status: str = "verified"

    def validate(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError(f"Invalid word sample bounds: [{self.start_sample}, {self.end_sample})")
        if not self.phone_event_ids:
            raise ValueError("WordOccurrence must have at least one phone event")


@dataclass
class InventoryIndex:
    """Root database index of aligned sources, events, and words."""
    schema_version: str = INVENTORY_SCHEMA_VERSION
    sources: Dict[str, SourceRecord] = field(default_factory=dict)
    utterances: Dict[str, UtteranceRecord] = field(default_factory=dict)
    events: Dict[str, AlignedEvent] = field(default_factory=dict)
    words: Dict[str, WordOccurrence] = field(default_factory=dict)
    word_to_word_ids: Dict[str, List[str]] = field(default_factory=dict)
    phone_to_event_ids: Dict[str, List[str]] = field(default_factory=dict)
    base_phone_to_event_ids: Dict[str, List[str]] = field(default_factory=dict)
    lexicon_fingerprint: str = ""
    alignment_fingerprint: str = ""

    def validate(self) -> None:
        if self.schema_version != INVENTORY_SCHEMA_VERSION:
            raise ValueError(f"Incompatible schema version: {self.schema_version}, expected {INVENTORY_SCHEMA_VERSION}")
        for s in self.sources.values():
            s.validate()
        for e in self.events.values():
            e.validate()
        for w in self.words.values():
            w.validate()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InventoryIndex":
        sources = {k: SourceRecord(**v) for k, v in data.get("sources", {}).items()}
        utterances = {k: UtteranceRecord(**v) for k, v in data.get("utterances", {}).items()}
        events = {k: AlignedEvent(**v) for k, v in data.get("events", {}).items()}
        words = {k: WordOccurrence(**v) for k, v in data.get("words", {}).items()}
        return cls(
            schema_version=data.get("schema_version", INVENTORY_SCHEMA_VERSION),
            sources=sources,
            utterances=utterances,
            events=events,
            words=words,
            word_to_word_ids=data.get("word_to_word_ids", {}),
            phone_to_event_ids=data.get("phone_to_event_ids", {}),
            base_phone_to_event_ids=data.get("base_phone_to_event_ids", {}),
            lexicon_fingerprint=data.get("lexicon_fingerprint", ""),
            alignment_fingerprint=data.get("alignment_fingerprint", ""),
        )


@dataclass
class FeatureRecord:
    """Acoustic feature representation for a source audio window or event boundary."""
    source_content_id: str
    sample_rate: int
    start_sample: int
    end_sample: int
    rms_energy: float
    f0_hz: Optional[float]  # None if unvoiced
    f0_confidence: float
    voicing_state: Literal["voiced", "unvoiced", "silence"]
    spectral_centroid: float
    mfcc_13: List[float] = field(default_factory=list)
    delta_mfcc_13: List[float] = field(default_factory=list)
    wavlm_embedding_id: Optional[str] = None
    quality_flags: List[str] = field(default_factory=list)

    def validate(self) -> None:
        if self.start_sample < 0 or self.end_sample < self.start_sample:
            raise ValueError("Invalid sample bounds for FeatureRecord")
        if self.voicing_state not in ("voiced", "unvoiced", "silence"):
            raise ValueError(f"Invalid voicing state: {self.voicing_state}")


@dataclass
class TargetPhoneNode:
    """Node in the target pronunciation graph."""
    node_id: int
    token_index: int
    word_text: str
    phone: str
    base_phone: str
    stress: Optional[int]
    left_context_phone: str = ""
    right_context_phone: str = ""
    left_context_phones: Set[str] = field(default_factory=set)
    right_context_phones: Set[str] = field(default_factory=set)
    is_word_initial: bool = False
    is_word_final: bool = False
    is_phrase_final: bool = False
    duration_prior_sec: Optional[float] = None
    target_pitch_semitones: Optional[float] = None
    prosodic_bias: Optional[str] = None  # e.g. "question", "exclamation"
    branch_id: int = 0  # index into lexicon.get_pronunciations() alternatives for this token
    is_constrained: bool = False


@dataclass
class TargetGraph:
    """Graph of legal pronunciation branches for a target sentence."""
    raw_text: str
    normalized_words: List[str]
    nodes: List[TargetPhoneNode]
    edges: List[Tuple[int, int]]  # directed edges between node indices
    start_nodes: List[int]
    end_nodes: List[int]
    punctuation_pauses: Dict[int, float] = field(default_factory=dict)  # token index -> pause seconds
    terminal_punctuation: str = ""
    explicit_gaps: Dict[int, float] = field(default_factory=dict)  # boundary index -> gap seconds

    def validate(self) -> None:
        if not self.nodes:
            raise ValueError("TargetGraph must have at least one phone node")
        if not self.start_nodes or not self.end_nodes:
            raise ValueError("TargetGraph must define start and end nodes")


@dataclass
class SpanCandidate:
    """A variable-span candidate covering target graph nodes using contiguous source events."""
    candidate_id: str
    target_node_start: int
    target_node_end: int
    covered_nodes: List[int]
    source_content_id: str
    speaker_id: str
    start_sample: int
    end_sample: int
    event_ids: List[str]
    span_type: Literal["phrase", "word", "phone", "subword"]
    matched_word: str = ""
    entry_variant_samples: int = 0
    exit_variant_samples: int = 0
    target_cost: float = 0.0
    quality_cost: float = 0.0
    realized_phones: List[str] = field(default_factory=list)
    cand_f0_hz: Optional[float] = None
    pitch_semitones: Optional[float] = None

    def validate(self) -> None:
        if self.start_sample < 0 or self.end_sample <= self.start_sample:
            raise ValueError("SpanCandidate source sample bounds invalid")
        if not self.covered_nodes:
            raise ValueError("SpanCandidate must cover at least one target node")


@dataclass
class PlanSegment:
    """An audio/video segment to render in the canonical edit plan."""
    segment_index: int
    source_content_id: str
    speaker_id: str
    source_start_sample: int
    source_end_sample: int
    output_start_sample: int
    output_end_sample: int
    target_text: str
    target_phones: List[str]
    span_type: str
    source_video_path: str
    gain_db: Optional[float] = None
    fade_in_samples: int = 0
    fade_out_samples: int = 0
    is_silence: bool = False
    video_start_frame: int = 0
    video_frame_count: int = 0
    video_source_pts_sec: float = 0.0
    freeze_video: bool = False
    realized_phones: List[str] = field(default_factory=list)
    entry_variant_samples: int = 0
    exit_variant_samples: int = 0

    @property
    def duration_samples(self) -> int:
        return self.output_end_sample - self.output_start_sample

    def validate(self) -> None:
        if not math.isfinite(self.output_start_sample) or not math.isfinite(self.output_end_sample):
            raise ValueError(f"Segment {self.segment_index} output bounds must be finite")
        if self.output_start_sample < 0 or self.output_end_sample <= self.output_start_sample:
            raise ValueError(
                f"Invalid segment output sample bounds: [{self.output_start_sample}, {self.output_end_sample})"
            )
        if self.gain_db is not None and not math.isfinite(self.gain_db):
            raise ValueError(f"Segment {self.segment_index} gain_db must be finite: {self.gain_db}")
        if not math.isfinite(self.video_source_pts_sec):
            raise ValueError(f"Segment {self.segment_index} video_source_pts_sec must be finite: {self.video_source_pts_sec}")
        if self.video_start_frame < 0 or self.video_frame_count < 0:
            raise ValueError(f"Segment {self.segment_index} video frames must be non-negative")
        if self.fade_in_samples < 0 or self.fade_out_samples < 0:
            raise ValueError(f"Segment {self.segment_index} fades must be non-negative")
        if not self.is_silence:
            if not math.isfinite(self.source_start_sample) or not math.isfinite(self.source_end_sample):
                raise ValueError(f"Segment {self.segment_index} source bounds must be finite")
            if self.source_start_sample < 0 or self.source_end_sample <= self.source_start_sample:
                raise ValueError(
                    f"Invalid segment source sample bounds: [{self.source_start_sample}, {self.source_end_sample})"
                )


@dataclass
class PlanTransition:
    """Transition between two adjacent plan segments."""
    transition_index: int
    from_segment_index: int
    to_segment_index: int
    transition_type: Literal["continuation", "hard_splice", "faded_splice", "overlap", "pause"]
    overlap_samples: int = 0
    fade_samples: int = 0
    join_cost: float = 0.0
    notes: str = ""

    def validate(self) -> None:
        if self.from_segment_index < 0 or self.to_segment_index < 0:
            raise ValueError(f"Transition indices must be non-negative: {self.from_segment_index} -> {self.to_segment_index}")
        if self.from_segment_index >= self.to_segment_index:
            raise ValueError(
                f"Transition link must be strictly forward (no cycles): {self.from_segment_index} -> {self.to_segment_index}"
            )
        if self.overlap_samples < 0:
            raise ValueError(f"Transition overlap_samples cannot be negative: {self.overlap_samples}")
        if self.fade_samples < 0:
            raise ValueError(f"Transition fade_samples cannot be negative: {self.fade_samples}")
        if not math.isfinite(self.join_cost):
            raise ValueError(f"Transition join_cost must be finite: {self.join_cost}")


@dataclass
class EditPlan:
    """Immutable, finalized edit decision plan for rendering and export."""
    schema_version: str = EDIT_PLAN_SCHEMA_VERSION
    plan_id: str = ""
    target_text: str = ""
    speaker_id: str = ""
    selector_mode: str = "span"
    rendering_mode: Literal["source-derived", "cuts-only"] = "source-derived"
    project_sample_rate: int = 48000
    video_fps: RationalTime = field(default_factory=lambda: RationalTime(30000, 1001))
    total_output_samples: int = 0
    total_video_frames: int = 0
    segments: List[PlanSegment] = field(default_factory=list)
    transitions: List[PlanTransition] = field(default_factory=list)
    sources: Dict[str, SourceRecord] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != EDIT_PLAN_SCHEMA_VERSION:
            raise ValueError(f"Incompatible plan schema version: {self.schema_version}")
        if self.project_sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {self.project_sample_rate}")
        if not math.isfinite(self.total_output_samples) or self.total_output_samples < 0:
            raise ValueError(f"Invalid total_output_samples: {self.total_output_samples}")
        if not math.isfinite(self.total_video_frames) or self.total_video_frames < 0:
            raise ValueError(f"Invalid total_video_frames: {self.total_video_frames}")

        if isinstance(self.video_fps, RationalTime):
            self.video_fps.validate()
            if self.video_fps.numerator <= 0:
                raise ValueError(f"video_fps numerator must be positive: {self.video_fps.numerator}")
        else:
            raise ValueError("video_fps must be an instance of RationalTime")

        if not self.segments:
            raise ValueError("EditPlan contains no segments")

        # Validate each segment
        for seg in self.segments:
            seg.validate()
            if self.sources and not seg.is_silence:
                if seg.source_content_id not in self.sources:
                    raise ValueError(
                        f"Segment {seg.segment_index} references unknown source_content_id: '{seg.source_content_id}'"
                    )

        # Timeline monotonicity & start boundary
        if self.segments[0].output_start_sample != 0:
            raise ValueError(
                f"EditPlan timeline must start at sample 0, got {self.segments[0].output_start_sample}"
            )

        for k in range(len(self.segments) - 1):
            curr = self.segments[k]
            nxt = self.segments[k + 1]
            if nxt.output_start_sample < curr.output_start_sample:
                raise ValueError(
                    f"Timeline non-monotonic: segment {k+1} starts at {nxt.output_start_sample} "
                    f"before segment {k} starts at {curr.output_start_sample}"
                )
            # Absence of undeclared holes: segment k+1 cannot start after segment k ends
            if nxt.output_start_sample > curr.output_end_sample:
                raise ValueError(
                    f"Undeclared hole in timeline between segment {k} (ends at {curr.output_end_sample}) "
                    f"and segment {k+1} (starts at {nxt.output_start_sample}). Any pause must be an explicit silence segment."
                )

        # Absence of triple overlaps: at any time point, at most 2 segments can overlap
        for k in range(len(self.segments) - 2):
            if self.segments[k + 2].output_start_sample < self.segments[k].output_end_sample:
                raise ValueError(
                    f"Triple overlap detected: segment {k+2} (starts at {self.segments[k+2].output_start_sample}) "
                    f"overlaps with segment {k} (ends at {self.segments[k].output_end_sample})"
                )

        # Transition integrity
        for t in self.transitions:
            t.validate()
            if t.from_segment_index >= len(self.segments) or t.to_segment_index >= len(self.segments):
                raise ValueError(
                    f"Transition {t.transition_index} references out-of-range segment indices: "
                    f"{t.from_segment_index} -> {t.to_segment_index} (total segments: {len(self.segments)})"
                )
            from_seg = self.segments[t.from_segment_index]
            to_seg = self.segments[t.to_segment_index]
            expected_overlap = from_seg.output_end_sample - to_seg.output_start_sample
            if t.transition_type == "overlap":
                if t.overlap_samples <= 0:
                    raise ValueError(f"Transition {t.transition_index} has type 'overlap' but overlap_samples={t.overlap_samples} <= 0")
                if t.overlap_samples >= from_seg.duration_samples or t.overlap_samples >= to_seg.duration_samples:
                    raise ValueError(
                        f"Transition {t.transition_index} overlap {t.overlap_samples} exceeds segment duration "
                        f"({from_seg.duration_samples} or {to_seg.duration_samples})"
                    )
                if t.overlap_samples != expected_overlap:
                    raise ValueError(
                        f"Transition {t.transition_index} declared overlap {t.overlap_samples} "
                        f"does not match actual segment boundary overlap {expected_overlap}"
                    )
            else:
                if expected_overlap != 0:
                    raise ValueError(
                        f"Transition {t.transition_index} is '{t.transition_type}' but segments overlap by {expected_overlap} samples"
                    )

        # Validate non-accumulating output sample timing equation:
        # output_samples = sum(segment_samples) - sum(overlap_samples)
        # Note: explicit pauses are represented as silence segments with is_silence=True
        computed_samples = sum(seg.duration_samples for seg in self.segments)
        total_overlap = sum(t.overlap_samples for t in self.transitions if t.transition_type == "overlap")
        expected_total = computed_samples - total_overlap
        if self.total_output_samples != expected_total:
            raise ValueError(
                f"EditPlan sample count mismatch: declared total_output_samples={self.total_output_samples}, "
                f"computed={expected_total} (sum_segments={computed_samples} - overlaps={total_overlap})"
            )
        if self.segments[-1].output_end_sample != self.total_output_samples:
            raise ValueError(
                f"EditPlan last segment end sample {self.segments[-1].output_end_sample} does not equal total_output_samples {self.total_output_samples}"
            )

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["video_fps"] = self.video_fps.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EditPlan":
        fps_data = data.get("video_fps", {"numerator": 30000, "denominator": 1001})
        if isinstance(fps_data, dict):
            fps = RationalTime(fps_data["numerator"], fps_data["denominator"])
        else:
            fps = RationalTime.from_str(str(fps_data))

        sources = {k: SourceRecord(**v) for k, v in data.get("sources", {}).items()}
        segments = [PlanSegment(**s) for s in data.get("segments", [])]
        transitions = [PlanTransition(**t) for t in data.get("transitions", [])]

        return cls(
            schema_version=data.get("schema_version", EDIT_PLAN_SCHEMA_VERSION),
            plan_id=data.get("plan_id", ""),
            target_text=data.get("target_text", ""),
            speaker_id=data.get("speaker_id", ""),
            selector_mode=data.get("selector_mode", "span"),
            rendering_mode=data.get("rendering_mode", "source-derived"),
            project_sample_rate=data.get("project_sample_rate", 48000),
            video_fps=fps,
            total_output_samples=data.get("total_output_samples", 0),
            total_video_frames=data.get("total_video_frames", 0),
            segments=segments,
            transitions=transitions,
            sources=sources,
            metadata=data.get("metadata", {}),
        )

    def save(self, filepath: Union[str, Path]) -> None:
        p = Path(filepath).resolve()
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "EditPlan":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
