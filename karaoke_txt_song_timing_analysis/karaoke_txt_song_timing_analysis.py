#!/usr/bin/env python3
"""
Estimate UltraStar Deluxe #GAP values using WhisperX word timestamps.

Safety:
- Reads audio and .txt chart files only.
- Does not modify charts or audio unless a chart-writing option is enabled.
- Writes a text report only unless --write-gap or --write-bpm is enabled.

Example:
    poetry run python ultrastar_gap.py \
        --source "$HOME/Music/whisper-test" \
        --report "$HOME/Music/whisper-test/gap_report.txt" \
        --language en \
        --model medium \
        --device cpu \
        --match-threshold 0.55

The suggested GAP is estimated from:
    audio word start time - chart note time excluding the existing GAP

A positive correction means increase #GAP; a negative correction means
decrease #GAP.
"""

from __future__ import annotations

import argparse
from collections import deque
from collections.abc import Iterator
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import nullcontext, redirect_stderr, redirect_stdout
import difflib
from datetime import datetime, timedelta
import io
import math
import multiprocessing
import os
import re
import shlex
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import traceback

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from korean_romanizer import Romanizer


PROJECT_FILE = Path(__file__).resolve().parents[1] / "pyproject.toml"
with PROJECT_FILE.open("rb") as project_file:
    VERSION = tomllib.load(project_file)["project"]["version"]


NOTE_RE = re.compile(
    r"^([:*FRG])\s*(-?\d+)\s+(\d+)\s+(-?\d+)\s(.*)$",
    re.IGNORECASE,
)

# Structural note-line pattern used to flag unsupported note-type markers.
ANY_NOTE_RE = re.compile(r"^([^\s:#-])\s+-?\d+\s+\d+\s+-?\d+\s+.*$")

NUMBER_RE = r"[+-]?\d+(?:[.,]\d+)?"
GAP_RE = re.compile(rf"^\s*#GAP:\s*({NUMBER_RE})", re.I)
AUDIO_SUFFIXES = {
    ".mp3",
    ".m4a",
    ".wav",
    ".flac",
    ".ogg",
    ".aac",
    ".opus",
    ".mp4",
}
MIN_GAP_WRITE_CHANGE_MS = 100.0
MIN_BPM_WRITE_CHANGE_PERCENT = 0.5
GAP_BYTES_RE = re.compile(
    rb"^(\xef\xbb\xbf)?([ \t]*#GAP:[ \t]*)"
    rb"([+-]?\d+(?:[.,]\d+)?)([ \t]*(?:\r?\n)?)$",
    re.I,
)
BPM_BYTES_RE = re.compile(
    rb"^(\xef\xbb\xbf)?([ \t]*#BPM:[ \t]*)"
    rb"([+-]?\d+(?:[.,]\d+)?)([ \t]*(?:\r?\n)?)$",
    re.I,
)
BPM_RE = re.compile(rf"^\s*#BPM:\s*({NUMBER_RE})", re.I)
AUDIO_RE = re.compile(r"^\s*#AUDIO:\s*(.*?)\s*$", re.I)
MP3_RE = re.compile(r"^\s*#MP3:\s*(.*?)\s*$", re.I)
RELATIVE_RE = re.compile(r"^\s*#RELATIVE:\s*(.*?)\s*$", re.I)

TOKEN_RE = re.compile(r"[^\W_]+(?:'[^\W_]+)?", re.I)
HANGUL_RE = re.compile(
    r"[\u1100-\u11ff\u3130-\u318f\ua960-\ua97f\uac00-\ud7af\ud7b0-\ud7ff]"
)
GAP_WINDOW_FALLBACK_SECONDS = 90.0
MIN_USABLE_CHART_WORDS = 3
ETA_SAMPLE_SIZE = 10
MIN_GAP_DRIFT_MATCHES = 10
MIN_GAP_DRIFT_SPAN_SECONDS = 20.0
MIN_GAP_DRIFT_MS = 1000.0
MAX_GAP_DRIFT_RESIDUAL_MAD_MS = 1000.0
GAP_MATCH_CONTEXT_RADIUS_SECONDS = 8.0
GAP_MATCH_CONTEXT_TOLERANCE_MS = 600.0
MIN_BPM_STRONG_SAMPLE_MATCHES = 40
MAX_BPM_STRONG_SAMPLE_SPREAD_MS = 200.0


@dataclass
class ChartWord:
    text: str
    beat: float
    seconds: float
    index: int


@dataclass
class AudioWord:
    text: str
    start: float
    end: float
    index: int


@dataclass
class Match:
    chart: ChartWord
    audio: AudioWord
    similarity: float
    gap_ms: float

    @property
    def audio_time(self) -> float:
        return self.audio.start


@dataclass
class BpmSample:
    label: str
    chart_time_seconds: float
    offset_ms: float
    match_count: int
    spread_ms: float


@dataclass
class BpmSampleDiagnostic:
    label: str
    audio_start: float
    audio_end: float
    chart_words: list[ChartWord]
    transcription: str
    aligned_words: list[AudioWord]
    matches: list[Match]
    note: str = ""


@dataclass
class Analysis:
    chart_path: Path
    audio_path: Optional[Path]
    old_gap_ms: Optional[float]
    bpm: Optional[float]
    status: str
    transcription: str = ""
    matches: Optional[list[Match]] = None
    suggested_gap_ms: Optional[float] = None
    correction_ms: Optional[float] = None
    confidence: str = "Not assessed"
    reason: str = ""
    gap_drift_warning: str = ""
    audio_duration: Optional[float] = None
    audio_opening_seconds: Optional[float] = None
    chart_opening_seconds: Optional[float] = None
    chart_words: Optional[list[ChartWord]] = None
    unknown_note_types: Optional[list[tuple[str, int]]] = None
    bpm_samples: Optional[list[BpmSample]] = None
    suggested_bpm: Optional[float] = None
    bpm_correction: Optional[float] = None
    bpm_confidence: str = "Not assessed"
    bpm_reason: str = ""
    bpm_write_block_reason: str = ""
    bpm_sample_notes: list[str] = field(default_factory=list)
    bpm_sample_diagnostics: list[BpmSampleDiagnostic] = field(default_factory=list)
    gap_write_status: str = ""
    bpm_write_status: str = ""


@dataclass
class ChartAnalysisOutput:
    analysis: Analysis
    logs: str = ""


_ACTIVE_CHILD_PROCESSES: set[multiprocessing.Process] = set()
_ACTIVE_CHILD_PROCESSES_LOCK = threading.Lock()
_CHART_PROCESS_TERMINATION_REQUESTED = threading.Event()


def normalize_word(text: str) -> str:
    """Normalize a word for forgiving text comparisons."""
    return "".join(TOKEN_RE.findall(text.lower()))


def tokens(text: str) -> list[str]:
    return [normalize_word(x) for x in TOKEN_RE.findall(text) if normalize_word(x)]


def select_gap_opening_windows(
    chart_words: list[ChartWord],
    audio_opening_seconds: float,
    chart_opening_seconds: float,
    gap_ms: float = 0.0,
) -> tuple[float, float]:
    """Extend sparse or late-starting opening lyric windows to 90 seconds."""
    gap_seconds = gap_ms / 1000.0
    opening_words = [
        word
        for word in chart_words
        if word.seconds <= chart_opening_seconds
        and 0 <= word.seconds + gap_seconds <= audio_opening_seconds
    ]
    usable_words = sum(
        len(normalize_word(word.text)) >= 3 for word in opening_words
    )
    expected_first_lyric = min(
        (
            word.seconds + gap_seconds
            for word in chart_words
            if word.seconds + gap_seconds >= 0
        ),
        default=None,
    )
    starts_late = (
        expected_first_lyric is not None
        and expected_first_lyric >= audio_opening_seconds / 2.0
    )
    if usable_words >= MIN_USABLE_CHART_WORDS and not starts_late:
        return audio_opening_seconds, chart_opening_seconds

    return (
        max(audio_opening_seconds, GAP_WINDOW_FALLBACK_SECONDS),
        max(chart_opening_seconds, GAP_WINDOW_FALLBACK_SECONDS),
    )


def estimate_completion_time(
    song_durations: list[float],
    remaining_songs: int,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    """Estimate completion using the most recent completed song durations."""
    if not song_durations:
        return None

    average_duration = statistics.fmean(song_durations[-ETA_SAMPLE_SIZE:])
    return (now or datetime.now()) + timedelta(
        seconds=average_duration * remaining_songs
    )


def estimate_parallel_completion_time(
    elapsed_seconds: float,
    completed_songs: int,
    total_songs: int,
    now: Optional[datetime] = None,
) -> Optional[datetime]:
    """Estimate completion from the dual-device run's observed throughput."""
    if completed_songs <= 0 or total_songs < completed_songs:
        return None

    remaining_songs = total_songs - completed_songs
    average_seconds_per_completion = elapsed_seconds / completed_songs
    return (now or datetime.now()) + timedelta(
        seconds=average_seconds_per_completion * remaining_songs
    )


def format_estimated_end(
    estimated_end: datetime,
    now: Optional[datetime] = None,
) -> str:
    """Format an estimated end time with a day offset when needed."""
    current_date = (now or datetime.now()).date()
    days_ahead = (estimated_end.date() - current_date).days
    if days_ahead == 0:
        return estimated_end.strftime("%H:%M")
    day_label = "day" if days_ahead == 1 else "days"
    return f"+{days_ahead} {day_label} {estimated_end:%H:%M}"


def prepare_audio_words_for_matching(
    audio_words: list[AudioWord],
    chart_words: list[ChartWord],
    detected_language: Optional[str],
) -> list[AudioWord]:
    """Romanize Korean audio words when chart lyrics use Latin characters."""
    language_code = (detected_language or "").casefold().split("-", maxsplit=1)[0]
    if language_code != "ko" or any(
        HANGUL_RE.search(word.text) for word in chart_words
    ):
        return audio_words

    return [
        AudioWord(
            text=Romanizer(word.text).romanize(),
            start=word.start,
            end=word.end,
            index=word.index,
        )
        for word in audio_words
    ]


def read_chart_text(path: Path) -> str:
    """Read UTF-8 charts, falling back to legacy Windows-1252 encoding."""
    data = path.read_bytes()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        try:
            return data.decode("cp1252")
        except UnicodeDecodeError:
            return data.decode("latin-1")


def word_similarity(a: str, b: str) -> float:
    """
    Return a permissive similarity score between 0 and 1.

    Exact matches score 1.0. Similar words such as 'antichrist' and
    'anti-coaster' receive partial credit. Short common words are handled
    conservatively by the sequence matcher, not treated as strong evidence
    on their own.
    """
    a = normalize_word(a)
    b = normalize_word(b)

    if not a or not b:
        return 0.0

    if a == b:
        return 1.0

    score = difflib.SequenceMatcher(None, a, b).ratio()

    # A shared prefix is useful when Whisper changes the end of a sung word.
    shorter = min(len(a), len(b))
    if shorter >= 4 and a[:4] == b[:4]:
        score = max(score, 0.68)

    # Common prefixes in the example "anarchist"/"anarchoaster".
    if shorter >= 5 and a[:5] == b[:5]:
        score = max(score, 0.72)

    return min(score, 1.0)


def detect_unknown_note_types(path: Path) -> list[tuple[str, int]]:
    """Find note-shaped lines whose note-type marker is not supported by NOTE_RE.

    Returns (marker, one-based line number) pairs. Player markers such as P1/P2
    are not treated as note lines because they do not have the note-line fields.
    """
    unknown: list[tuple[str, int]] = []
    try:
        for line_number, line in enumerate(
            read_chart_text(path).splitlines(),
            start=1,
        ):
            if ANY_NOTE_RE.match(line) and not NOTE_RE.match(line):
                marker = line[0]
                unknown.append((marker, line_number))
    except OSError:
        pass
    return unknown


def parse_chart(path: Path) -> tuple[Optional[float], Optional[float], list[ChartWord]]:
    """Read BPM, GAP and reconstruct complete lyric words from note fragments.

    UltraStar note text may be a syllable rather than a whole word. A trailing
    space marks the end of a lyric word, so that whitespace must be inspected
    before stripping or normalizing the text. Phrase-break lines also terminate
    any pending word. The word's timing is the beat of its first fragment.
    """
    bpm = None
    gap_ms = None
    raw_words: list[tuple[str, float]] = []
    pending_text = ""
    pending_beat: Optional[float] = None
    relative_timing = False
    previous_start_beat = 0.0

    def parse_number(value: str) -> float:
        """Parse UltraStar numeric fields with either decimal separator."""
        return float(value.replace(",", "."))

    def flush_pending_word() -> None:
        nonlocal pending_text, pending_beat
        word = normalize_word(pending_text)
        if word and pending_beat is not None:
            raw_words.append((word, pending_beat))
        pending_text = ""
        pending_beat = None

    for line in read_chart_text(path).splitlines():
        # Do not strip lyric trailing spaces: they carry word-boundary data.

        relative_match = RELATIVE_RE.match(line)
        if relative_match:
            relative_timing = relative_match.group(1).strip().casefold() == "yes"
            continue

        gap_match = GAP_RE.match(line)
        if gap_match:
            gap_ms = parse_number(gap_match.group(1))
            continue

        bpm_match = BPM_RE.match(line)
        if bpm_match:
            bpm = parse_number(bpm_match.group(1))
            continue

        # A phrase break ends any unfinished word. It does not replace the
        # previous note as the reference for subsequent relative note beats.
        phrase_match = re.match(r"^\s*-\s*(-?\d+)(?:\s+-?\d+)*\s*$", line)
        if phrase_match:
            flush_pending_word()
            continue

        note_match = NOTE_RE.match(line)
        if not note_match:
            continue

        encoded_beat = float(int(note_match.group(2)))
        beat = previous_start_beat + encoded_beat if relative_timing else encoded_beat
        previous_start_beat = beat
        lyric = note_match.group(5)

        # A tilde is a note-extension marker, not a sung word. It also
        # safely closes any pending lyric word before the extension.
        if lyric.strip() == "~":
            flush_pending_word()
            continue

        # UltraStar uses an extra leading space in the lyric field to
        # mark the start of a new word. NOTE_RE consumes only the single
        # field-separator space, so any additional leading space survives.
        # Trailing spaces can also mark a word boundary in some charts.
        starts_word = bool(lyric) and lyric[0].isspace()
        ends_word = bool(lyric) and lyric[-1].isspace()

        if starts_word:
            flush_pending_word()

        fragment = lyric.strip()
        if fragment:
            if pending_beat is None:
                pending_beat = float(beat)
            pending_text += fragment

        if ends_word:
            flush_pending_word()

    # Don't lose the final word if the file ends without a trailing space.
    flush_pending_word()

    if bpm is None or bpm <= 0:
        return gap_ms, bpm, []

    words = [
        ChartWord(
            text=text,
            beat=beat,
            seconds=beat * 60.0 / (bpm * 4.0),
            index=i,
        )
        for i, (text, beat) in enumerate(raw_words)
    ]

    return gap_ms, bpm, words


def read_audio_reference(chart_path: Path) -> tuple[Optional[str], Optional[str]]:
    """Read #AUDIO preferentially, with legacy #MP3 support."""
    audio_value: Optional[str] = None
    mp3_value: Optional[str] = None
    for line in read_chart_text(chart_path).splitlines():
        match = AUDIO_RE.match(line)
        if match and audio_value is None:
            value = match.group(1).strip().strip('"').strip("'").strip()
            audio_value = value or None
            continue
        match = MP3_RE.match(line)
        if match and mp3_value is None:
            value = match.group(1).strip().strip('"').strip("'").strip()
            mp3_value = value or None
    if audio_value:
        return "#AUDIO", audio_value
    if mp3_value:
        return "#MP3", mp3_value
    return None, None


def read_mp3_reference(chart_path: Path) -> Optional[str]:
    """Compatibility wrapper returning the selected audio filename."""
    _tag, value = read_audio_reference(chart_path)
    return value


def resolve_case_insensitive(path: Path) -> Optional[Path]:
    """Resolve a path exactly first, then case-insensitively for Linux systems."""
    if path.exists() and path.is_file():
        return path

    # Walk each path component so references with different filename casing
    # still resolve on case-sensitive filesystems.
    if path.is_absolute():
        current = Path(path.anchor)
        parts = path.parts[1:]
    else:
        current = Path.cwd()
        parts = path.parts

    try:
        for part in parts:
            if part in {"", "."}:
                continue
            if part == "..":
                current = current.parent
                continue
            if not current.is_dir():
                return None
            match = next(
                (child for child in current.iterdir() if child.name.casefold() == part.casefold()),
                None,
            )
            if match is None:
                return None
            current = match
        return current if current.is_file() else None
    except OSError:
        return None


def find_audio_for_chart(chart_path: Path) -> Optional[Path]:
    """Find audio from #AUDIO or legacy #MP3; otherwise use same-basename lookup."""
    _reference_tag, audio_reference = read_audio_reference(chart_path)

    if audio_reference:
        # Prefer #AUDIO; legacy #MP3 is selected only when #AUDIO is absent.
        reference = Path(audio_reference.replace("\\", "/")).expanduser()
        candidate = reference if reference.is_absolute() else chart_path.parent / reference
        resolved = resolve_case_insensitive(candidate)
        if resolved and resolved.suffix.casefold() in AUDIO_SUFFIXES:
            return resolved
        return None

    for candidate in chart_path.parent.iterdir():
        if (
            candidate.is_file()
            and candidate.stem.casefold() == chart_path.stem.casefold()
            and candidate.suffix.casefold() in AUDIO_SUFFIXES
        ):
            return candidate

    return None


def audio_lookup_failure_reason(chart_path: Path) -> str:
    """Explain whether a referenced or same-basename audio file is missing or unsupported."""
    reference_tag, audio_reference = read_audio_reference(chart_path)
    if audio_reference:
        reference = Path(audio_reference.replace("\\", "/")).expanduser()
        candidate = reference if reference.is_absolute() else chart_path.parent / reference
        resolved = resolve_case_insensitive(candidate)
        if resolved is not None:
            return (
                f"Audio file referenced by {reference_tag} ({audio_reference!r}) "
                f"has unsupported file type {resolved.suffix!r}."
            )
        return (
            f"No audio file referenced by {reference_tag} "
            f"({audio_reference!r}) was found."
        )

    try:
        unsupported_match = next(
            (
                candidate
                for candidate in chart_path.parent.iterdir()
                if candidate.is_file()
                and candidate.stem.casefold() == chart_path.stem.casefold()
            ),
            None,
        )
    except OSError:
        unsupported_match = None
    if unsupported_match is not None:
        return (
            f"Same-basename audio file {unsupported_match.name!r} has "
            f"unsupported file type {unsupported_match.suffix!r}."
        )
    return "No #AUDIO/#MP3 reference or same-basename audio file was found."


def is_existing_ultrastar_chart(path: Path) -> bool:
    """Protect a chart from being accidentally overwritten as the report."""
    if not path.is_file() or path.suffix.casefold() != ".txt":
        return False
    has_bpm = False
    has_note = False
    try:
        for line in read_chart_text(path).splitlines():
            if BPM_RE.match(line):
                has_bpm = True
            if NOTE_RE.match(line):
                has_note = True
            if has_bpm and has_note:
                return True
    except OSError:
        return False
    return False


def transcribe_audio_windows(
    audio_path: Path,
    model_name: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    batch_size: int,
    opening_seconds: float,
    extra_windows: list[tuple[str, float, float]],
    model: Optional[Any] = None,
) -> tuple[
    list[AudioWord],
    float,
    str,
    Optional[str],
    dict[str, list[AudioWord]],
    dict[str, str],
    Any,
]:
    """Transcribe windows, optionally reusing a model loaded by an earlier call.

    Extra windows are (label, start_seconds, duration_seconds). Their timestamps
    are converted back to absolute audio time so they can be compared directly
    with chart note times. This samples the song without transcribing the whole
    audio file, keeping the additional BPM check relatively inexpensive.
    """
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX is not installed in this Python environment."
        ) from exc

    sample_rate = 16000
    audio = whisperx.load_audio(str(audio_path))
    full_duration = len(audio) / sample_rate

    windows: list[tuple[str, float, object]] = []
    opening_end = min(full_duration, opening_seconds)
    if opening_end > 0:
        windows.append(("opening", 0.0, audio[:int(opening_end * sample_rate)]))

    for label, requested_start, requested_duration in extra_windows:
        if requested_duration <= 0 or full_duration <= 0:
            continue
        start = max(0.0, requested_start)
        end = min(full_duration, start + requested_duration)
        if end - start < min(3.0, requested_duration * 0.6):
            continue
        windows.append(
            (
                label,
                start,
                audio[int(start * sample_rate):int(end * sample_rate)],
            )
        )

    if model is None:
        model = whisperx.load_model(
            model_name,
            device,
            compute_type=compute_type,
            language=language,
        )

    raw_windows: list[tuple[str, float, object, dict]] = []
    detected_language = language
    opening_text = ""
    for index, (label, start, chunk) in enumerate(windows):
        result = model.transcribe(
            chunk,
            batch_size=batch_size,
            language=language,
        )
        if not isinstance(result, dict):
            raise TypeError(
                "WhisperX transcription returned "
                f"{type(result).__name__}, expected a dictionary."
            )
        if index == 0:
            detected_language = result.get("language") or language
        segments = result.get("segments", [])
        text = str(result.get("text", "") or "")
        if not text:
            text = " ".join(
                str(seg.get("text", ""))
                for seg in segments
                if isinstance(seg, dict)
            ).strip()
        if label == "opening":
            opening_text = text
        raw_windows.append((label, start, chunk, {"segments": segments, "text": text}))

    align_model = None
    metadata = None
    if any(data["segments"] for _, _, _, data in raw_windows):
        if not detected_language:
            raise RuntimeError(
                "WhisperX did not report a language, so the transcription "
                "cannot be aligned."
            )
        align_model, metadata = whisperx.load_align_model(
            language_code=detected_language,
            device=device,
        )

    aligned_words: dict[str, list[AudioWord]] = {}
    for label, offset, chunk, data in raw_windows:
        audio_words: list[AudioWord] = []
        segments = data["segments"]
        if segments and align_model is not None:
            aligned_result = whisperx.align(
                segments,
                align_model,
                metadata,
                chunk,
                device,
                return_char_alignments=False,
            )
            # Some versions return (aligned_result, metadata); others return only it.
            if isinstance(aligned_result, tuple):
                aligned_result = aligned_result[0]
            if not isinstance(aligned_result, dict):
                raise TypeError(
                    "WhisperX alignment returned "
                    f"{type(aligned_result).__name__}, expected a dictionary."
                )

            for segment in aligned_result.get("segments", []):
                if not isinstance(segment, dict):
                    continue
                for item in segment.get("words", []):
                    if not isinstance(item, dict):
                        continue
                    word = str(item.get("word", "")).strip()
                    start = item.get("start")
                    end = item.get("end")
                    if not word or start is None or end is None:
                        continue
                    try:
                        start = float(start) + offset
                        end = float(end) + offset
                    except (TypeError, ValueError):
                        continue
                    if not math.isfinite(start) or not math.isfinite(end):
                        continue
                    audio_words.append(
                        AudioWord(
                            text=word,
                            start=start,
                            end=end,
                            index=len(audio_words),
                        )
                    )
        aligned_words[label] = audio_words

    return (
        aligned_words.get("opening", []),
        full_duration,
        opening_text,
        detected_language,
        aligned_words,
        {
            label: data["text"]
            for label, _, _, data in raw_windows
        },
        model,
    )


def estimate_bpm_from_samples(
    samples: list[BpmSample],
    current_bpm: float,
) -> tuple[Optional[float], Optional[float], str, str]:
    """Robustly estimate BPM from the slope of audio-vs-chart timing offsets."""
    usable = [
        sample for sample in samples
        if sample.match_count >= 3 and sample.spread_ms <= 1200
    ]
    strong_samples = [
        sample for sample in usable
        if (
            sample.match_count >= MIN_BPM_STRONG_SAMPLE_MATCHES
            and sample.spread_ms <= MAX_BPM_STRONG_SAMPLE_SPREAD_MS
        )
    ]
    use_strong_samples = len(strong_samples) >= 2
    estimation_samples = strong_samples if use_strong_samples else usable
    if len(estimation_samples) < 2:
        return (
            None, None, "Low",
            f"Only {len(estimation_samples)} reliable timing samples; "
            "at least 2 are needed.",
        )

    span = max(s.chart_time_seconds for s in estimation_samples) - min(
        s.chart_time_seconds for s in estimation_samples
    )
    if span < 30.0:
        return (
            None, None, "Low",
            f"Reliable samples span only {span:.1f}s; at least 30s are needed.",
        )

    # Theil-Sen slope: median of all pairwise offset slopes, robust to one bad sample.
    slopes = []
    for i, first in enumerate(estimation_samples):
        for second in estimation_samples[i + 1:]:
            dt = second.chart_time_seconds - first.chart_time_seconds
            if abs(dt) >= 10.0:
                slopes.append((second.offset_ms - first.offset_ms) / (dt * 1000.0))
    if not slopes:
        return None, None, "Low", "Could not estimate timing drift from the samples."

    slope = statistics.median(slopes)  # seconds of offset drift per chart second
    denominator = 1.0 + slope
    if denominator <= 0:
        return None, None, "Low", "The estimated timing drift is not physically plausible."

    suggested = current_bpm / denominator
    correction = suggested - current_bpm
    relative_change = abs(correction) / current_bpm
    intercept = statistics.median(
        sample.offset_ms - slope * sample.chart_time_seconds * 1000.0
        for sample in estimation_samples
    )
    residuals = [
        sample.offset_ms - (intercept + slope * sample.chart_time_seconds * 1000.0)
        for sample in estimation_samples
    ]
    residual_mad = median_absolute_deviation(residuals)
    max_residual = max(abs(residual) for residual in residuals)
    sample_description = (
        f"{len(estimation_samples)} strong samples"
        if use_strong_samples
        else f"{len(estimation_samples)} reliable samples"
    )
    sample_summary = (
        f"{sample_description} over {span:.1f}s; "
        f"timing-drift residual MAD {residual_mad:.0f} ms."
    )
    if use_strong_samples:
        excluded_samples = [
            sample for sample in usable if sample not in strong_samples
        ]
        if excluded_samples:
            excluded_labels = ", ".join(sample.label for sample in excluded_samples)
            sample_summary += (
                f" Excluded weaker sample(s) from the estimate: {excluded_labels}."
            )

    if relative_change > 0.15:
        confidence = "Low"
        reason = (
            f"Estimated BPM change {relative_change * 100:.1f}% exceeds the "
            f"15% safety limit; review sample matches. {sample_summary}"
        )
    elif use_strong_samples and len(strong_samples) == 2:
        confidence = "High"
        reason = (
            "At least 2 sections each have at least "
            f"{MIN_BPM_STRONG_SAMPLE_MATCHES} matches, offset MAD no greater "
            f"than {MAX_BPM_STRONG_SAMPLE_SPREAD_MS:.0f} ms, and span at least "
            f"30s. {sample_summary}"
        )
    elif len(estimation_samples) == 2:
        confidence = "Low"
        reason = (
            f"Only 2 reliable samples over {span:.1f}s; at least 3 are needed "
            f"for a stronger confidence assessment. {sample_summary}"
        )
    elif (
        len(estimation_samples) >= 3
        and all(sample.match_count >= 4 for sample in estimation_samples)
        and residual_mad <= 500
        and (not use_strong_samples or max_residual <= 500)
        and all(sample.spread_ms <= 700 for sample in estimation_samples)
    ):
        confidence = "High"
        reason = sample_summary
    elif (
        residual_mad <= 1000
        and (not use_strong_samples or max_residual <= 1000)
    ):
        confidence = "Medium"
        reason = sample_summary
    else:
        confidence = "Low"
        reason = (
            f"The timing-drift trend is inconsistent. {sample_summary}"
        )

    return suggested, correction, confidence, reason

def local_alignment(
    chart_words: list[ChartWord],
    audio_words: list[AudioWord],
    threshold: float,
    skip_penalty: float = 0.34,
) -> list[Match]:
    """
    Align chart and ASR words in order using local dynamic programming.

    The algorithm permits insertions, deletions and imperfect word matches.
    It seeks a coherent phrase instead of matching a word found anywhere
    in the full song.
    """
    if not chart_words or not audio_words:
        return []

    n = len(chart_words)
    m = len(audio_words)

    # Local alignment: a new candidate sequence may begin at any position.
    scores = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]

    # Directions: 1=diagonal match, 2=skip chart word, 3=skip audio word.
    for i in range(1, n + 1):
        cw = chart_words[i - 1]

        for j in range(1, m + 1):
            aw = audio_words[j - 1]
            similarity = word_similarity(cw.text, aw.text)

            # Similarity below threshold is not a positive match, but it can
            # still be crossed by the alignment to reach later words.
            match_score = (similarity - threshold) * 2.5
            diagonal = scores[i - 1][j - 1] + match_score
            skip_chart = scores[i - 1][j] - skip_penalty
            skip_audio = scores[i][j - 1] - skip_penalty

            best = max(0.0, diagonal, skip_chart, skip_audio)
            scores[i][j] = best

            if best == 0.0:
                trace[i][j] = 0
            elif best == diagonal:
                trace[i][j] = 1
            elif best == skip_chart:
                trace[i][j] = 2
            else:
                trace[i][j] = 3

    # Trace back from the best local endpoint.
    best_i, best_j, best_score = 0, 0, 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if scores[i][j] > best_score:
                best_i, best_j, best_score = i, j, scores[i][j]

    pairs: list[tuple[ChartWord, AudioWord, float]] = []
    i, j = best_i, best_j

    while i > 0 and j > 0 and scores[i][j] > 0:
        direction = trace[i][j]

        if direction == 1:
            cw = chart_words[i - 1]
            aw = audio_words[j - 1]
            sim = word_similarity(cw.text, aw.text)

            if sim >= threshold:
                pairs.append((cw, aw, sim))

            i -= 1
            j -= 1
        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            break

    pairs.reverse()

    # A second, deliberately forgiving pass can rescue short opening phrases
    # where local alignment is broken by a few poor ASR words.
    #
    # It greedily searches only forward in audio, preserving word order.
    if len(pairs) < 4:
        pairs = greedy_ordered_alignment(chart_words, audio_words, threshold)

    # Remove duplicate or non-monotonic pairs.
    clean: list[tuple[ChartWord, AudioWord, float]] = []
    last_chart = -1
    last_audio = -1

    for cw, aw, sim in pairs:
        if cw.index > last_chart and aw.index > last_audio:
            clean.append((cw, aw, sim))
            last_chart = cw.index
            last_audio = aw.index

    return [
        Match(
            chart=cw,
            audio=aw,
            similarity=sim,
            gap_ms=(aw.start - cw.seconds) * 1000.0,
        )
        for cw, aw, sim in clean
    ]


def greedy_ordered_alignment(
    chart_words: list[ChartWord],
    audio_words: list[AudioWord],
    threshold: float,
) -> list[tuple[ChartWord, AudioWord, float]]:
    """
    Forgiving ordered fallback.

    For each chart word, search a small forward window in the ASR words and
    select the most similar candidate. Short words such as 'I' and 'am' are
    accepted as supporting matches but are not sufficient on their own.
    """
    result = []
    audio_cursor = 0

    for cw in chart_words:
        best_index = None
        best_similarity = 0.0

        # Keep the search local so repeated words later in the song cannot
        # become an opening match.
        search_end = min(len(audio_words), audio_cursor + 5)

        for j in range(audio_cursor, search_end):
            sim = word_similarity(cw.text, audio_words[j].text)
            if sim > best_similarity:
                best_similarity = sim
                best_index = j

        if best_index is None or best_similarity < threshold:
            continue

        aw = audio_words[best_index]
        result.append((cw, aw, best_similarity))
        audio_cursor = best_index + 1

    return result


def median_absolute_deviation(values: list[float]) -> float:
    if not values:
        return float("inf")

    median = statistics.median(values)
    return statistics.median(abs(x - median) for x in values)


def weighted_median(values: list[float], weights: list[float]) -> float:
    """Return the first value whose cumulative weight reaches half the total."""
    if len(values) != len(weights) or not values:
        raise ValueError("values and weights must have the same non-zero length")
    if any(weight <= 0 or not math.isfinite(weight) for weight in weights):
        raise ValueError("weights must be finite and positive")

    ordered = sorted(zip(values, weights))
    half_weight = sum(weights) / 2
    cumulative_weight = 0.0
    for index, (value, weight) in enumerate(ordered):
        cumulative_weight += weight
        if cumulative_weight > half_weight:
            return value
        if (
            math.isclose(cumulative_weight, half_weight)
            and index + 1 < len(ordered)
        ):
            return (value + ordered[index + 1][0]) / 2
    raise RuntimeError("Unable to calculate weighted median.")


def gap_match_context_weights(matches: list[Match]) -> list[float]:
    """Weight matches by local timing agreement with their lyric neighbors."""
    weights = [0.0] * len(matches)
    ordered_indices = sorted(
        range(len(matches)),
        key=lambda index: matches[index].chart.seconds,
    )

    for ordered_position, match_index in enumerate(ordered_indices):
        match = matches[match_index]
        left_neighbor = (
            matches[ordered_indices[ordered_position - 1]]
            if ordered_position > 0
            else None
        )
        right_neighbor = (
            matches[ordered_indices[ordered_position + 1]]
            if ordered_position + 1 < len(ordered_indices)
            else None
        )
        if (
            left_neighbor is not None
            and match.chart.seconds - left_neighbor.chart.seconds
            > GAP_MATCH_CONTEXT_RADIUS_SECONDS
        ):
            left_neighbor = None
        if (
            right_neighbor is not None
            and right_neighbor.chart.seconds - match.chart.seconds
            > GAP_MATCH_CONTEXT_RADIUS_SECONDS
        ):
            right_neighbor = None
        nearby_neighbors = [
            neighbor
            for neighbor in (left_neighbor, right_neighbor)
            if neighbor is not None
        ]
        supporting_left = (
            left_neighbor is not None
            and abs(match.gap_ms - left_neighbor.gap_ms)
            <= GAP_MATCH_CONTEXT_TOLERANCE_MS
        )
        supporting_right = (
            right_neighbor is not None
            and abs(match.gap_ms - right_neighbor.gap_ms)
            <= GAP_MATCH_CONTEXT_TOLERANCE_MS
        )

        if supporting_left and supporting_right:
            context_weight = 1.0
        elif supporting_left or supporting_right:
            context_weight = 0.8
        elif nearby_neighbors:
            context_weight = 0.2
        else:
            context_weight = 0.35

        similarity_weight = 0.9 + 0.1 * max(
            0.0,
            min(1.0, match.similarity),
        )
        weights[match_index] = context_weight * similarity_weight

    return weights


def count_supported_informative_phrases(
    matches: list[Match],
    weights: list[float],
    minimum_weight: float,
) -> int:
    """Count lyric phrase groups with at least two supported informative words."""
    ordered = sorted(
        zip(matches, weights),
        key=lambda item: item[0].chart.seconds,
    )
    phrase_counts: list[int] = []
    previous_match: Optional[Match] = None

    for match, weight in ordered:
        if (
            previous_match is None
            or match.chart.seconds - previous_match.chart.seconds > 0.65
        ):
            phrase_counts.append(0)
        if (
            weight >= minimum_weight
            and (
                len(normalize_word(match.chart.text)) >= 3
                or len(normalize_word(match.audio.text)) >= 3
            )
        ):
            phrase_counts[-1] += 1
        previous_match = match

    return sum(count >= 2 for count in phrase_counts)


def detect_gap_drift_warning(matches: list[Match]) -> str:
    """Flag a sustained opening offset trend that may confound the GAP estimate."""
    if len(matches) < MIN_GAP_DRIFT_MATCHES:
        return ""

    chart_times = [match.chart.seconds for match in matches]
    chart_span = max(chart_times) - min(chart_times)
    if chart_span < MIN_GAP_DRIFT_SPAN_SECONDS:
        return ""

    slopes = [
        (second.gap_ms - first.gap_ms)
        / (second.chart.seconds - first.chart.seconds)
        for index, first in enumerate(matches)
        for second in matches[index + 1:]
        if second.chart.seconds != first.chart.seconds
    ]
    if not slopes:
        return ""

    slope_ms_per_second = statistics.median(slopes)
    trend_ms = slope_ms_per_second * chart_span
    intercept_ms = statistics.median(
        match.gap_ms - slope_ms_per_second * match.chart.seconds
        for match in matches
    )
    residual_mad_ms = median_absolute_deviation([
        match.gap_ms - (
            intercept_ms + slope_ms_per_second * match.chart.seconds
        )
        for match in matches
    ])
    if (
        abs(trend_ms) < MIN_GAP_DRIFT_MS
        or residual_mad_ms > MAX_GAP_DRIFT_RESIDUAL_MAD_MS
    ):
        return ""

    return (
        f"Opening match offsets show a sustained {trend_ms:+.0f} ms trend "
        f"over {chart_span:.1f} seconds; a BPM mismatch may be affecting "
        "the GAP estimate. Review before accepting or writing."
    )


def evaluate_matches(
    matches: list[Match],
    old_gap_ms: float,
    threshold: float,
    max_correction_ms: float,
) -> tuple[Optional[float], Optional[float], str, str]:
    """
    Estimate the new GAP and decide whether the evidence is usable.

    The criteria are intentionally more permissive than the previous script.
    """
    if not matches:
        return None, None, "Low", "No ordered lyric matches were found."

    match_weights = gap_match_context_weights(matches)
    gap_values = [match.gap_ms for match in matches]
    suggested_gap = weighted_median(gap_values, match_weights)
    correction = suggested_gap - old_gap_ms
    spread = weighted_median(
        [abs(gap - suggested_gap) for gap in gap_values],
        match_weights,
    )
    supported_matches = [
        match
        for match, weight in zip(matches, match_weights)
        if weight >= 0.7
    ]

    strong_matches = [
        m for m in supported_matches
        if m.similarity >= max(threshold + 0.12, 0.67)
    ]

    distinct_phrases = 0
    ordered_matches = sorted(matches, key=lambda match: match.chart.seconds)
    if len(ordered_matches) >= 2:
        # A phrase boundary is approximated by a gap in chart timing.
        for previous, current in zip(ordered_matches, ordered_matches[1:]):
            if current.chart.seconds - previous.chart.seconds > 0.65:
                distinct_phrases += 1

    matched_span = (
        ordered_matches[-1].audio.start - ordered_matches[0].audio.start
        if len(matches) >= 2
        else 0.0
    )

    informative_matches = [
        match
        for match, weight in zip(matches, match_weights)
        if (
            weight >= 0.7
            and (
                len(normalize_word(match.chart.text)) >= 3
                or len(normalize_word(match.audio.text)) >= 3
            )
        )
    ]
    supported_phrases = count_supported_informative_phrases(
        matches,
        match_weights,
        minimum_weight=0.7,
    )

    if len(matches) < 3:
        return (
            suggested_gap,
            correction,
            "Low",
            f"Only {len(matches)} ordered matches; at least 3 are needed.",
        )

    if len(informative_matches) < 2 or supported_phrases < 1:
        return (
            suggested_gap,
            correction,
            "Low",
            "Too few context-supported informative word matches; matches "
            "must agree with nearby lyric timings.",
        )

    if spread > 1500:
        return (
            suggested_gap,
            correction,
            "Low",
            f"Match timing is inconsistent (median absolute deviation "
            f"{spread:.0f} ms).",
        )

    if abs(correction) > max_correction_ms:
        return (
            suggested_gap,
            correction,
            "Low",
            f"Proposed correction {correction:+.0f} ms exceeds the allowed "
            f"limit of {max_correction_ms:.0f} ms.",
        )

    # Lower confidence is expected for fuzzy singing recognition. The
    # threshold is intentionally not a hard veto once sequence/timing agree.
    if (
        len(informative_matches) >= 5
        and supported_phrases >= 2
        and spread <= 600
        and len(strong_matches) >= 2
    ):
        confidence = "High"
    elif (
        len(informative_matches) >= 3
        and supported_phrases >= 2
        and spread <= 1000
    ):
        confidence = "Medium"
    else:
        confidence = "Low"

    reasons = [
        f"{len(matches)} ordered matches",
        f"{len(supported_matches)} context-supported matches",
        f"{len(informative_matches)} context-supported informative matches",
        f"{len(strong_matches)} stronger supported matches",
        f"{supported_phrases} supported phrase groups",
        f"weighted timing MAD {spread:.0f} ms",
    ]
    if supported_phrases < 2:
        reasons.append(
            "matches are concentrated in fewer than 2 supported phrase groups"
        )
    if distinct_phrases:
        reasons.append(f"{distinct_phrases} possible phrase boundaries")
    if matched_span:
        reasons.append(f"{matched_span:.2f}s audio span")

    return suggested_gap, correction, confidence, "; ".join(reasons)


def summarize_bpm_sample(label: str, matches: list[Match]) -> Optional[BpmSample]:
    """Summarize a sample using its median timing offset and match spread."""
    if len(matches) < 3:
        return None
    offsets = [match.gap_ms for match in matches]
    return BpmSample(
        label=label,
        chart_time_seconds=statistics.median(m.chart.seconds for m in matches),
        offset_ms=statistics.median(offsets),
        match_count=len(matches),
        spread_ms=median_absolute_deviation(offsets),
    )


def select_bpm_sample_windows(
    chart_words: list[ChartWord],
    corrected_gap_ms: float,
    audio_duration: float,
    opening_seconds: float,
    sample_window_seconds: float = 45.0,
    max_target_distance: float = 10.0,
) -> tuple[list[tuple[str, float, float]], list[str]]:
    """Select non-overlapping audio windows centered on lyric clusters."""
    if len(chart_words) < 3:
        return [], ["Skipped later samples: fewer than 3 chart lyric words."]

    chart_start = chart_words[0].seconds
    chart_end = chart_words[-1].seconds
    chart_span = chart_end - chart_start
    if chart_span < 45.0:
        return [], [
            f"Skipped later samples: lyric timeline spans only "
            f"{chart_span:.1f}s; at least 45s is required."
        ]

    clusters: list[list[ChartWord]] = []
    for word in chart_words:
        if (
            not clusters
            or word.seconds - clusters[-1][-1].seconds > 0.65
        ):
            clusters.append([])
        clusters[-1].append(word)

    gap_seconds = corrected_gap_ms / 1000.0
    used_windows: list[tuple[float, float]] = []
    windows: list[tuple[str, float, float]] = []
    notes: list[str] = []

    for label, fraction in (("50% (middle)", 0.50), ("90% (end)", 0.90)):
        target_audio_time = chart_start + chart_span * fraction + gap_seconds
        candidates = []

        for cluster in clusters:
            anchor = min(
                cluster,
                key=lambda word: abs(word.seconds + gap_seconds - target_audio_time),
            )
            audio_center = anchor.seconds + gap_seconds
            distance = abs(audio_center - target_audio_time)
            audio_start = audio_center - sample_window_seconds / 2.0
            audio_end = audio_start + sample_window_seconds

            if audio_start < opening_seconds or audio_end > audio_duration:
                continue
            if any(
                min(audio_end, used_end) > max(audio_start, used_start)
                for used_start, used_end in used_windows
            ):
                continue

            candidates.append(
                (distance, audio_start, anchor.text, anchor.seconds)
            )

        candidates.sort(key=lambda candidate: (candidate[0], candidate[1]))
        if not candidates:
            notes.append(
                f"{label}: skipped; no lyric cluster has a full "
                f"{sample_window_seconds:.0f}s non-overlapping window "
                "inside the available audio."
            )
            continue

        distance, audio_start, lyric, chart_time = next(
            (candidate for candidate in candidates
             if candidate[0] <= max_target_distance),
            candidates[0],
        )
        used_windows.append((audio_start, audio_start + sample_window_seconds))
        windows.append((label, audio_start, sample_window_seconds))

        if distance > max_target_distance:
            notes.append(
                f"{label}: using lyric cluster near '{lyric}' "
                f"at chart {chart_time:.1f}s, "
                f"{distance:.1f}s from the target (over "
                f"{max_target_distance:.0f}s)."
            )
        else:
            notes.append(
                f"{label}: snapped to lyric cluster near '{lyric}' "
                f"at chart {chart_time:.1f}s, "
                f"{distance:.1f}s from the target."
            )

    return windows, notes


def separate_vocals(
    audio_path: Path,
    device: str,
    output_directory: Path,
) -> Path:
    """Run Demucs and return its isolated vocals.wav stem."""
    subprocess.run(
        [
            sys.executable,
            "-m",
            "demucs.separate",
            "--two-stems=vocals",
            "-d",
            device,
            "--out",
            str(output_directory),
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    vocal_stems = list(output_directory.rglob("vocals.wav"))
    if len(vocal_stems) != 1:
        raise RuntimeError(
            "Demucs completed but expected exactly one vocals.wav stem; "
            f"found {len(vocal_stems)}."
        )
    return vocal_stems[0]


def create_audio_snippet(
    audio_path: Path,
    duration_seconds: float,
    output_directory: Path,
) -> Path:
    """Create a stream-copied audio snippet for opening-window analysis."""
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_name",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(audio_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    codec_name = probe.stdout.strip().casefold()
    if not codec_name:
        raise RuntimeError(
            f"ffprobe did not identify an audio codec in {audio_path}."
        )
    snippet_suffix = ".mp2" if codec_name == "mp2" else audio_path.suffix
    snippet_path = output_directory / (
        f"{audio_path.stem}_{duration_seconds:g}s{snippet_suffix}"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-i",
            str(audio_path),
            "-t",
            f"{duration_seconds:g}",
            "-map",
            "0:a:0",
            "-c:a",
            "copy",
            str(snippet_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return snippet_path


def analyze_chart(
    chart_path: Path,
    args: argparse.Namespace,
) -> Analysis:
    gap_ms, bpm, chart_words = parse_chart(chart_path)
    audio_path = find_audio_for_chart(chart_path)
    if (
        audio_path is None
        or bpm is None
        or bpm <= 0
        or gap_ms is None
        or not chart_words
    ):
        return _analyze_chart_with_audio(chart_path, args)

    opening_windows = select_gap_opening_windows(
        chart_words,
        args.audio_opening_seconds,
        args.chart_opening_seconds,
        gap_ms,
    )
    audio_opening_seconds, _ = opening_windows

    with tempfile.TemporaryDirectory(prefix="karaoke-demucs-") as temporary:
        try:
            demucs_input_path = audio_path
            if not args.bpm:
                demucs_input_path = create_audio_snippet(
                    audio_path,
                    audio_opening_seconds + 1.0,
                    Path(temporary),
                )
            print(f"    Demucs processing: {chart_path.name}", flush=True)
            vocals_path = separate_vocals(
                demucs_input_path,
                args.device,
                Path(temporary),
            )
        except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
            reason = (
                "Audio preparation or Demucs vocal separation failed: "
                f"{type(exc).__name__}: {exc}"
            )
            if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
                reason += f"\nCommand stderr: {exc.stderr.strip()}"
            if args.debug:
                reason += "\n" + traceback.format_exc()
            return Analysis(
                chart_path=chart_path,
                audio_path=audio_path,
                old_gap_ms=gap_ms,
                bpm=bpm,
                status="Analysis failed",
                reason=reason,
                chart_words=chart_words,
                unknown_note_types=detect_unknown_note_types(chart_path),
            )
        return _analyze_chart_with_audio(
            chart_path,
            args,
            transcription_audio_path=vocals_path,
            opening_windows=opening_windows,
        )


def _analyze_chart_with_audio(
    chart_path: Path,
    args: argparse.Namespace,
    transcription_audio_path: Optional[Path] = None,
    opening_windows: Optional[tuple[float, float]] = None,
) -> Analysis:
    gap_ms, bpm, chart_words = parse_chart(chart_path)
    unknown_note_types = detect_unknown_note_types(chart_path)
    audio_path = find_audio_for_chart(chart_path)

    analysis = Analysis(
        chart_path=chart_path,
        audio_path=audio_path,
        old_gap_ms=gap_ms,
        bpm=bpm,
        status="Analysis failed",
        audio_opening_seconds=args.audio_opening_seconds,
        chart_opening_seconds=args.chart_opening_seconds,
        chart_words=chart_words,
        unknown_note_types=unknown_note_types,
    )

    if bpm is None or bpm <= 0:
        analysis.reason = "Missing or invalid #BPM."
        return analysis

    if gap_ms is None:
        analysis.reason = "Missing #GAP."
        return analysis

    if not chart_words:
        analysis.reason = "No lyric words with valid note timings were found."
        return analysis

    if audio_path is None:
        analysis.reason = audio_lookup_failure_reason(chart_path)
        return analysis

    transcription_audio_path = transcription_audio_path or audio_path

    if opening_windows is None:
        opening_windows = select_gap_opening_windows(
            chart_words,
            args.audio_opening_seconds,
            args.chart_opening_seconds,
            gap_ms,
        )
    audio_opening_seconds, chart_opening_seconds = opening_windows
    analysis.audio_opening_seconds = audio_opening_seconds
    analysis.chart_opening_seconds = chart_opening_seconds

    # Restrict matching to the opening lyrics. Do not search the entire song
    # for a convenient but unrelated word.
    opening_chart_words = [
        word for word in chart_words
        if (
            word.seconds <= chart_opening_seconds
            and 0
            <= word.seconds + gap_ms / 1000.0
            <= audio_opening_seconds
        )
    ]

    if not opening_chart_words:
        analysis.reason = "No chart lyric words in the opening window."
        return analysis

    try:
        print(f"    WhisperX processing: {chart_path.name}", flush=True)
        (
            opening_audio_words,
            duration,
            transcription,
            detected_language,
            _,
            opening_transcriptions,
            transcription_model,
        ) = transcribe_audio_windows(
            audio_path=transcription_audio_path,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
            batch_size=args.batch_size,
            opening_seconds=audio_opening_seconds,
            extra_windows=[],
        )
    except Exception as exc:
        analysis.reason = (
            f"{type(exc).__name__}: {exc}\n"
            "Run again with --debug for the traceback."
        )
        if args.debug:
            analysis.reason += "\n" + traceback.format_exc()
        return analysis

    analysis.audio_duration = duration
    analysis.transcription = transcription
    analysis.status = "Analysis complete"

    # Preserve the existing GAP estimate: align the chart's opening lyrics
    # against the opening vocal-stem window.
    opening_chart_words = [
        word for word in chart_words
        if (
            word.seconds <= chart_opening_seconds
            and 0
            <= word.seconds + gap_ms / 1000.0
            <= audio_opening_seconds
        )
    ]
    matches = local_alignment(
        opening_chart_words,
        prepare_audio_words_for_matching(
            opening_audio_words,
            chart_words,
            detected_language,
        ),
        threshold=args.match_threshold,
    )
    analysis.matches = matches

    suggested, correction, confidence, reason = evaluate_matches(
        matches,
        old_gap_ms=gap_ms,
        threshold=args.match_threshold,
        max_correction_ms=args.max_correction_ms,
    )

    analysis.suggested_gap_ms = suggested
    analysis.correction_ms = correction
    analysis.confidence = confidence
    analysis.reason = reason
    analysis.gap_drift_warning = detect_gap_drift_warning(matches)
    if not opening_audio_words:
        analysis.reason = (
            f"No aligned opening audio word timestamps were returned "
            f"(language: {detected_language}). {reason}"
        )

    if not args.bpm:
        analysis.bpm_reason = "BPM analysis disabled; pass --bpm to enable."
        return analysis

    gap_for_sampling = suggested if suggested is not None else gap_ms
    if confidence == "Low":
        analysis.bpm_sample_notes.append(
            "GAP confidence is Low; BPM samples will still be attempted. "
            + (
                "Using the diagnostic estimated GAP for sample placement."
                if suggested is not None
                else "Using the chart's current #GAP for sample placement."
            )
        )
    elif confidence == "Not assessed":
        analysis.bpm_sample_notes.append(
            "GAP confidence was not assessed; using the chart's current #GAP "
            "for sample placement."
        )

    if not opening_audio_words:
        analysis.bpm_sample_notes.append(
            f"No aligned opening audio word timestamps were returned "
            f"(language: {detected_language}); no Beginning BPM sample is available."
        )

    # Reuse the opening GAP matches as the first BPM sample; do not transcribe
    # the opening region again just to create a separate BPM sample.
    if chart_words:
        beginning_matches = matches
        beginning_sample = summarize_bpm_sample("Beginning", beginning_matches)
        bpm_samples: list[BpmSample] = []
        if beginning_sample:
            bpm_samples.append(beginning_sample)
        analysis.bpm_sample_diagnostics.append(
            BpmSampleDiagnostic(
                label="Beginning",
                audio_start=0.0,
                audio_end=min(duration, audio_opening_seconds),
                chart_words=opening_chart_words,
                transcription=opening_transcriptions.get("opening", ""),
                aligned_words=opening_audio_words,
                matches=beginning_matches,
                note=(
                    ""
                    if beginning_sample
                    else "Fewer than 3 matched words; excluded from BPM estimation."
                ),
            )
        )

        extra_windows, selection_notes = select_bpm_sample_windows(
            chart_words=chart_words,
            corrected_gap_ms=gap_for_sampling,
            audio_duration=duration,
            opening_seconds=audio_opening_seconds,
        )
        analysis.bpm_sample_notes.extend(selection_notes)

        transcribed_windows: dict[str, list[AudioWord]] = {}
        sample_transcriptions: dict[str, str] = {}
        if extra_windows:
            try:
                print(
                    f"    WhisperX processing BPM samples: {chart_path.name}",
                    flush=True,
                )
                (
                    _,
                    _,
                    _,
                    _,
                    transcribed_windows,
                    sample_transcriptions,
                    _,
                ) = transcribe_audio_windows(
                    audio_path=transcription_audio_path,
                    model_name=args.model,
                    language=detected_language,
                    device=args.device,
                    compute_type=args.compute_type,
                    batch_size=args.batch_size,
                    opening_seconds=0,
                    extra_windows=extra_windows,
                    model=transcription_model,
                )
            except Exception as exc:
                analysis.bpm_sample_notes.append(
                    f"Later BPM sample transcription failed: "
                    f"{type(exc).__name__}: {exc}"
                )
                if args.debug:
                    analysis.bpm_sample_notes.append(traceback.format_exc())

        for label, requested_start, requested_duration in extra_windows:
            sample_audio = transcribed_windows.get(label, [])
            chart_window_start = requested_start - gap_for_sampling / 1000.0
            chart_window_end = chart_window_start + requested_duration
            chart_sample_words = [
                word for word in chart_words
                if chart_window_start <= word.seconds <= chart_window_end
            ]
            sample_matches = local_alignment(
                chart_sample_words,
                prepare_audio_words_for_matching(
                    sample_audio,
                    chart_words,
                    detected_language,
                ),
                threshold=args.match_threshold,
            )
            sample = summarize_bpm_sample(label, sample_matches)
            sample_note = ""
            if sample:
                bpm_samples.append(sample)
            else:
                if not sample_audio:
                    sample_note = "No aligned audio word timestamps were returned."
                else:
                    sample_note = "Fewer than 3 ordered lyric matches."
                analysis.bpm_sample_notes.append(f"{label}: {sample_note}")

            analysis.bpm_sample_diagnostics.append(
                BpmSampleDiagnostic(
                    label=label,
                    audio_start=requested_start,
                    audio_end=requested_start + requested_duration,
                    chart_words=chart_sample_words,
                    transcription=sample_transcriptions.get(label, ""),
                    aligned_words=sample_audio,
                    matches=sample_matches,
                    note=sample_note,
                )
            )

        analysis.bpm_samples = bpm_samples
        (
            analysis.suggested_bpm,
            analysis.bpm_correction,
            analysis.bpm_confidence,
            analysis.bpm_reason,
        ) = estimate_bpm_from_samples(bpm_samples, bpm)
        if (
            confidence in {"Low", "Not assessed"}
            and analysis.suggested_bpm is not None
        ):
            analysis.bpm_write_block_reason = (
                f"GAP confidence is {confidence}"
            )
            analysis.bpm_reason += (
                f" GAP confidence is {confidence}; BPM result is diagnostic only."
            )

    return analysis


def _analyze_chart_worker(
    connection,
    chart_path: Path,
    args: argparse.Namespace,
    capture_output: bool,
) -> None:
    output = io.StringIO()
    stdout_context = redirect_stdout(output) if capture_output else nullcontext()
    stderr_context = redirect_stderr(output) if capture_output else nullcontext()
    try:
        with stdout_context, stderr_context:
            try:
                result = analyze_chart(chart_path, args)
            except Exception as exc:
                result = Analysis(
                    chart_path=chart_path,
                    audio_path=find_audio_for_chart(chart_path),
                    old_gap_ms=None,
                    bpm=None,
                    status="Analysis failed",
                    reason=f"{type(exc).__name__}: {exc}",
                )
                if args.debug:
                    result.reason += "\n" + traceback.format_exc()
        connection.send(ChartAnalysisOutput(result, output.getvalue()))
    finally:
        connection.close()


def analyze_chart_in_child_process(
    chart_path: Path,
    args: argparse.Namespace,
) -> Analysis:
    return _run_chart_in_child_process(
        chart_path,
        args,
        capture_output=False,
    ).analysis


def analyze_chart_in_child_process_capturing_output(
    chart_path: Path,
    args: argparse.Namespace,
) -> ChartAnalysisOutput:
    return _run_chart_in_child_process(
        chart_path,
        args,
        capture_output=True,
    )


def _run_chart_in_child_process(
    chart_path: Path,
    args: argparse.Namespace,
    capture_output: bool,
) -> ChartAnalysisOutput:
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_analyze_chart_worker,
        args=(send_connection, chart_path, args, capture_output),
    )
    try:
        with _ACTIVE_CHILD_PROCESSES_LOCK:
            if _CHART_PROCESS_TERMINATION_REQUESTED.is_set():
                raise InterruptedError(
                    "Chart analysis cancelled before its worker process started."
                )
            process.start()
            _ACTIVE_CHILD_PROCESSES.add(process)
    except BaseException:
        receive_connection.close()
        send_connection.close()
        raise

    send_connection.close()
    try:
        try:
            result = receive_connection.recv()
        except EOFError:
            result = None
    except BaseException:
        if process.is_alive():
            process.terminate()
        process.join()
        raise
    finally:
        receive_connection.close()
        with _ACTIVE_CHILD_PROCESSES_LOCK:
            _ACTIVE_CHILD_PROCESSES.discard(process)
    process.join()

    if result is None:
        return ChartAnalysisOutput(
            Analysis(
                chart_path=chart_path,
                audio_path=find_audio_for_chart(chart_path),
                old_gap_ms=None,
                bpm=None,
                status="Analysis failed",
                reason=(
                    "Per-chart analysis process exited without returning a result "
                    f"(exit code {process.exitcode})."
                ),
            )
        )
    return result


def terminate_active_chart_processes() -> None:
    with _ACTIVE_CHILD_PROCESSES_LOCK:
        _CHART_PROCESS_TERMINATION_REQUESTED.set()
        processes = list(_ACTIVE_CHILD_PROCESSES)
        for process in processes:
            if process.is_alive():
                process.terminate()


def is_cuda_out_of_memory(reason: str) -> bool:
    """Return whether a failure reason explicitly reports CUDA OOM."""
    normalized_reason = reason.casefold()
    return (
        "cuda out of memory" in normalized_reason
        or "cuda error: out of memory" in normalized_reason
        or (
            "outofmemoryerror" in normalized_reason
            and "cuda" in normalized_reason
        )
    )


def analyze_chart_with_cpu_fallback(
    chart_path: Path,
    args: argparse.Namespace,
) -> Analysis:
    """Retry a CUDA out-of-memory failure once on CPU for this chart."""
    result = analyze_chart_in_child_process(chart_path, args)
    if (
        args.device != "cuda"
        or result.status != "Analysis failed"
        or not is_cuda_out_of_memory(result.reason)
    ):
        return result

    print(
        f"    CUDA out of memory; retrying on CPU: {chart_path.name}",
        flush=True,
    )
    cpu_args = argparse.Namespace(**vars(args))
    cpu_args.device = "cpu"
    cpu_args.compute_type = "int8"
    try:
        cpu_result = analyze_chart_in_child_process(chart_path, cpu_args)
    except Exception as exc:
        raise RuntimeError(
            "CUDA out-of-memory attempt failed.\n"
            f"GPU error: {result.reason}\n"
            "CPU retry could not complete: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if cpu_result.status == "Analysis failed":
        cpu_result.reason = (
            "GPU attempt failed with CUDA out of memory.\n"
            f"GPU error: {result.reason}\n"
            "CPU retry also failed.\n"
            f"CPU error: {cpu_result.reason}"
        )
    else:
        cpu_result.reason = (
            f"{cpu_result.reason}\n"
            "GPU attempt ran out of memory; analysis succeeded on CPU retry."
        ).strip()
    return cpu_result


def _analyze_chart_on_device(
    chart_path: Path,
    args: argparse.Namespace,
    device: str,
) -> ChartAnalysisOutput:
    worker_args = argparse.Namespace(**vars(args))
    worker_args.device = device
    if not args.compute_type_explicit:
        worker_args.compute_type = "int8" if device == "cpu" else "float16"

    try:
        return analyze_chart_in_child_process_capturing_output(
            chart_path,
            worker_args,
        )
    except Exception as exc:
        result = Analysis(
            chart_path=chart_path,
            audio_path=find_audio_for_chart(chart_path),
            old_gap_ms=None,
            bpm=None,
            status="Analysis failed",
            reason=f"{type(exc).__name__}: {exc}",
        )
        if args.debug:
            result.reason += "\n" + traceback.format_exc()
        return ChartAnalysisOutput(result)


def analyze_charts_on_both_devices(
    chart_paths: list[Path],
    args: argparse.Namespace,
) -> Iterator[tuple[Path, ChartAnalysisOutput]]:
    with _ACTIVE_CHILD_PROCESSES_LOCK:
        _CHART_PROCESS_TERMINATION_REQUESTED.clear()

    remaining_charts = deque(chart_paths)
    cpu_retries: deque[tuple[Path, str]] = deque()
    executors = {
        "cpu": ThreadPoolExecutor(max_workers=1, thread_name_prefix="chart-cpu"),
        "cuda": ThreadPoolExecutor(max_workers=1, thread_name_prefix="chart-cuda"),
    }
    active_jobs: dict[Future[ChartAnalysisOutput], tuple[Path, str, str]] = {}
    active_devices: set[str] = set()

    def submit(chart_path: Path, device: str, gpu_error: str = "") -> None:
        worker_args = argparse.Namespace(**vars(args))
        if gpu_error:
            worker_args.compute_type = "int8"
            worker_args.compute_type_explicit = True
        future = executors[device].submit(
            _analyze_chart_on_device,
            chart_path,
            worker_args,
            device,
        )
        active_jobs[future] = (chart_path, device, gpu_error)
        active_devices.add(device)
        label = "CPU retry" if gpu_error else f"{device.upper()} worker"
        print(f"[{label}] Analyzing: {chart_path}", flush=True)

    try:
        while remaining_charts or cpu_retries or active_jobs:
            if "cuda" not in active_devices and remaining_charts:
                submit(remaining_charts.popleft(), "cuda")

            if "cpu" not in active_devices:
                if cpu_retries:
                    retry_path, gpu_error = cpu_retries.popleft()
                    submit(retry_path, "cpu", gpu_error)
                elif remaining_charts:
                    submit(remaining_charts.popleft(), "cpu")

            if not active_jobs:
                continue

            completed, _ = wait(active_jobs, return_when=FIRST_COMPLETED)
            completed_outputs: list[tuple[Path, ChartAnalysisOutput]] = []
            for future in completed:
                chart_path, device, gpu_error = active_jobs.pop(future)
                active_devices.remove(device)
                output = future.result()
                if (
                    device == "cuda"
                    and not gpu_error
                    and output.analysis.status == "Analysis failed"
                    and is_cuda_out_of_memory(output.analysis.reason)
                ):
                    if output.logs:
                        print(output.logs, end="", flush=True)
                    print(
                        f"CUDA out of memory; queueing CPU retry: {chart_path.name}",
                        flush=True,
                    )
                    cpu_retries.appendleft((chart_path, output.analysis.reason))
                else:
                    if gpu_error:
                        cpu_result = output.analysis
                        if cpu_result.status == "Analysis failed":
                            cpu_result.reason = (
                                "GPU attempt failed with CUDA out of memory.\n"
                                f"GPU error: {gpu_error}\n"
                                "CPU retry also failed.\n"
                                f"CPU error: {cpu_result.reason}"
                            )
                        else:
                            cpu_result.reason = (
                                f"{cpu_result.reason}\n"
                                "GPU attempt ran out of memory; "
                                "analysis succeeded on CPU retry."
                            ).strip()
                    completed_outputs.append((chart_path, output))

            for chart_path, output in completed_outputs:
                yield chart_path, output
    except BaseException:
        for future in active_jobs:
            future.cancel()
        terminate_active_chart_processes()
        raise
    finally:
        for executor in executors.values():
            executor.shutdown(wait=True, cancel_futures=True)


def archive_existing_backup(backup_path: Path) -> Optional[Path]:
    """Move an existing latest backup into a timestamped history directory."""
    if not backup_path.exists():
        return None

    archive_directory = backup_path.parent / "backups"
    archive_directory.mkdir(exist_ok=True)
    if archive_directory.is_symlink() or not archive_directory.is_dir():
        raise OSError(
            f"Backup archive path is not a regular directory: {archive_directory}"
        )

    timestamp = datetime.now().strftime("%Y-%m-%d-T%H-%M-%S")
    sequence = 0
    while True:
        archive_path = archive_directory / (
            f"{timestamp}_{sequence:03d}_{backup_path.name}"
        )
        try:
            archive_file = archive_path.open("xb")
        except FileExistsError:
            sequence += 1
            continue
        try:
            with archive_file, backup_path.open("rb") as source:
                shutil.copyfileobj(source, archive_file)
                archive_file.flush()
                os.fsync(archive_file.fileno())
            shutil.copystat(backup_path, archive_path)
            if archive_path.read_bytes() != backup_path.read_bytes():
                raise OSError(
                    f"Archived backup does not match the previous backup: "
                    f"{archive_path}"
                )
            backup_path.unlink()
        except OSError as exc:
            try:
                archive_path.unlink(missing_ok=True)
            except OSError as cleanup_error:
                raise OSError(
                    f"{exc}; could not remove incomplete archive {archive_path}: "
                    f"{cleanup_error}"
                ) from exc
            raise
        return archive_path


def backup_details(
    backup_path: Path,
    archived_backup_path: Optional[Path],
) -> str:
    if archived_backup_path is None:
        return f"backup: {backup_path}"
    return (
        f"previous backup archived to {archived_backup_path}; "
        f"latest backup: {backup_path}"
    )


def write_chart_updates(
    result: Analysis,
    write_gap: bool,
    write_bpm: bool,
    preserved_backup: Optional[bytes] = None,
    allow_small_updates: bool = False,
) -> tuple[str, str]:
    """Back up a chart once and atomically write eligible GAP/BPM suggestions."""
    gap_status = ""
    bpm_status = ""
    if write_gap:
        if result.confidence not in {"High", "Medium"}:
            gap_status = (
                f"Skipped: GAP confidence is {result.confidence}; "
                "only Medium or High confidence is writable."
            )
        elif result.suggested_gap_ms is None or not math.isfinite(result.suggested_gap_ms):
            gap_status = "Skipped: no finite GAP suggestion is available."
        elif result.old_gap_ms is None or not math.isfinite(result.old_gap_ms):
            gap_status = "Skipped: no finite current #GAP value is available."

    if write_bpm:
        if result.bpm_write_block_reason:
            bpm_status = (
                f"Skipped: {result.bpm_write_block_reason}; "
                "BPM result is diagnostic only."
            )
        elif result.bpm_confidence not in {"High", "Medium"}:
            bpm_status = (
                f"Skipped: BPM confidence is {result.bpm_confidence}; "
                "only Medium or High confidence is writable."
            )
        elif (
            result.suggested_bpm is None
            or not math.isfinite(result.suggested_bpm)
            or result.suggested_bpm <= 0
        ):
            bpm_status = "Skipped: no finite positive BPM suggestion is available."
        elif result.bpm is None or not math.isfinite(result.bpm) or result.bpm <= 0:
            bpm_status = "Skipped: no finite positive current #BPM value is available."

    if not (
        (write_gap and not gap_status)
        or (write_bpm and not bpm_status)
    ):
        return gap_status, bpm_status

    chart_path = result.chart_path
    if chart_path.is_symlink():
        refusal = "Skipped: chart is a symbolic link; refusing to replace its path."
        return (
            refusal if write_gap and not gap_status else gap_status,
            refusal if write_bpm and not bpm_status else bpm_status,
        )

    try:
        original = chart_path.read_bytes()
    except OSError as exc:
        failure = f"Failed to read chart before update: {type(exc).__name__}: {exc}"
        return (
            failure if write_gap and not gap_status else gap_status,
            failure if write_bpm and not bpm_status else bpm_status,
        )

    lines = original.splitlines(keepends=True)
    updates: list[tuple[str, int, bytes, str, float, float]] = []
    small_update_notes: dict[str, str] = {}
    requested_fields = []
    if write_gap and not gap_status:
        requested_fields.append(
            ("#GAP", GAP_BYTES_RE, result.old_gap_ms, result.suggested_gap_ms, 0)
        )
    if write_bpm and not bpm_status:
        requested_fields.append(
            ("#BPM", BPM_BYTES_RE, result.bpm, result.suggested_bpm, 3)
        )

    for field_name, pattern, analyzed_value, suggested_value, decimals in requested_fields:
        matches = [
            (index, pattern.match(line))
            for index, line in enumerate(lines)
            if pattern.match(line)
        ]
        if len(matches) != 1:
            status = (
                f"Skipped: expected exactly one writable {field_name} line, "
                f"found {len(matches)}."
            )
            if field_name == "#GAP":
                gap_status = status
            else:
                bpm_status = status
            continue

        line_index, match = matches[0]
        if match is None:
            status = f"Skipped: could not safely parse the existing {field_name} line."
            if field_name == "#GAP":
                gap_status = status
            else:
                bpm_status = status
            continue

        current_value = float(match.group(3).decode("ascii").replace(",", "."))
        if current_value != analyzed_value:
            if field_name == "#GAP":
                status = (
                    "Skipped: chart #GAP changed after analysis "
                    f"(analyzed {format_ms(analyzed_value)}, "
                    f"now {format_ms(current_value)})."
                )
            else:
                status = (
                    "Skipped: chart #BPM changed after analysis "
                    f"(analyzed {analyzed_value}, now {current_value})."
                )
            if field_name == "#GAP":
                gap_status = status
            else:
                bpm_status = status
            continue

        new_value_text = f"{suggested_value:.{decimals}f}"
        new_value = float(new_value_text)
        if field_name == "#GAP":
            change = abs(new_value - current_value)
            if change == 0:
                gap_status = f"No update needed: {field_name} is already {new_value_text}."
                continue
            if not allow_small_updates and change < MIN_GAP_WRITE_CHANGE_MS:
                gap_status = (
                    f"Skipped: proposed #GAP change is {change:.1f} ms, below "
                    f"the {MIN_GAP_WRITE_CHANGE_MS:.0f} ms write threshold."
                )
                continue
            if allow_small_updates and change < MIN_GAP_WRITE_CHANGE_MS:
                small_update_notes[field_name] = (
                    "below the normal 100 ms threshold; written because "
                    "--allow-small-updates was supplied"
                )
        elif new_value == current_value:
            bpm_status = f"No update needed: {field_name} is already {new_value_text}."
            continue
        elif field_name == "#BPM":
            change_percent = abs(new_value - current_value) / current_value * 100
            if not allow_small_updates and (
                change_percent < MIN_BPM_WRITE_CHANGE_PERCENT
                and not math.isclose(
                    change_percent,
                    MIN_BPM_WRITE_CHANGE_PERCENT,
                    rel_tol=1e-12,
                )
            ):
                bpm_status = (
                    f"Skipped: proposed #BPM change is {change_percent:.3f}%, "
                    f"below the {MIN_BPM_WRITE_CHANGE_PERCENT:.2f}% write "
                    "threshold."
                )
                continue
            if (
                allow_small_updates
                and change_percent < MIN_BPM_WRITE_CHANGE_PERCENT
                and not math.isclose(
                    change_percent,
                    MIN_BPM_WRITE_CHANGE_PERCENT,
                    rel_tol=1e-12,
                )
            ):
                small_update_notes[field_name] = (
                    "below the normal 0.50% threshold; written because "
                    "--allow-small-updates was supplied"
                )

        updates.append(
            (field_name, line_index, match.group(2), new_value_text, current_value, new_value)
        )

    if not updates:
        return gap_status, bpm_status

    for field_name, line_index, prefix, new_value_text, _, _ in updates:
        match = (GAP_BYTES_RE if field_name == "#GAP" else BPM_BYTES_RE).match(
            lines[line_index]
        )
        if match is None:
            status = f"Skipped: could not safely parse the existing {field_name} line."
            if field_name == "#GAP":
                gap_status = status
            else:
                bpm_status = status
            continue
        lines[line_index] = (
            (match.group(1) or b"")
            + prefix
            + new_value_text.encode("ascii")
            + match.group(4)
        )
    updated = b"".join(lines)
    backup_path = chart_path.with_suffix(".bak")
    if backup_path.is_symlink():
        failure = "Failed to update chart: refusing to overwrite a symbolic-link backup."
        return (
            failure if any(field == "#GAP" for field, *_ in updates) else gap_status,
            failure if any(field == "#BPM" for field, *_ in updates) else bpm_status,
        )

    temporary_path: Optional[Path] = None
    backup_temporary_path: Optional[Path] = None
    archived_backup_path: Optional[Path] = None
    update_error: Optional[OSError] = None
    cleanup_error: Optional[OSError] = None
    try:
        if preserved_backup is None:
            expected_backup = original
            descriptor, backup_temporary_name = tempfile.mkstemp(
                prefix=f".{backup_path.name}.",
                suffix=".tmp",
                dir=chart_path.parent,
            )
            backup_temporary_path = Path(backup_temporary_name)
            os.close(descriptor)
            shutil.copy2(chart_path, backup_temporary_path)
            if backup_temporary_path.read_bytes() != expected_backup:
                failure = (
                    "Skipped: chart contents changed while preparing the backup; "
                    f"the refreshed backup is {backup_path}."
                )
                return (
                    failure if any(field == "#GAP" for field, *_ in updates) else gap_status,
                    failure if any(field == "#BPM" for field, *_ in updates) else bpm_status,
                )
            archived_backup_path = archive_existing_backup(backup_path)
            os.replace(backup_temporary_path, backup_path)
            backup_temporary_path = None
        else:
            expected_backup = preserved_backup
        if backup_path.read_bytes() != expected_backup:
            failure = (
                "Skipped: chart contents changed while preparing the backup; "
                f"the refreshed backup is {backup_path}."
            )
            return (
                failure if any(field == "#GAP" for field, *_ in updates) else gap_status,
                failure if any(field == "#BPM" for field, *_ in updates) else bpm_status,
            )
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{chart_path.name}.",
            suffix=".tmp",
            dir=chart_path.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(descriptor, "wb") as temporary_file:
            temporary_file.write(updated)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.chmod(temporary_path, stat.S_IMODE(chart_path.stat().st_mode))
        os.replace(temporary_path, chart_path)
    except OSError as exc:
        update_error = exc
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_error = exc
        if backup_temporary_path is not None:
            try:
                backup_temporary_path.unlink(missing_ok=True)
            except OSError as exc:
                cleanup_error = exc

    if update_error is not None:
        failure = (
            f"Failed to update chart: {type(update_error).__name__}: "
            f"{update_error}"
        )
        if archived_backup_path is not None:
            failure += f" Previous backup archived to {archived_backup_path}."
        if backup_path.exists():
            failure += f" Backup: {backup_path}"
        for field_name, *_ in updates:
            if field_name == "#GAP":
                gap_status = failure
            else:
                bpm_status = failure
    else:
        for field_name, _, _, new_value_text, current_value, _ in updates:
            small_update_note = small_update_notes.get(field_name)
            if small_update_note:
                small_update_note = f"; {small_update_note}"
            else:
                small_update_note = ""
            if field_name == "#GAP":
                status = (
                    f"Updated #GAP from {format_ms(current_value)} to "
                    f"{format_ms(float(new_value_text))}{small_update_note}; "
                    f"{backup_details(backup_path, archived_backup_path)}."
                )
            else:
                status = (
                    f"Updated #BPM from {current_value:g} to "
                    f"{new_value_text}{small_update_note}; "
                    f"{backup_details(backup_path, archived_backup_path)}."
                )
            if field_name == "#GAP":
                gap_status = status
            else:
                bpm_status = status

    if cleanup_error is not None:
        cleanup_message = (
            f" Temporary-file cleanup failed: {type(cleanup_error).__name__}: "
            f"{cleanup_error}"
        )
        if any(field == "#GAP" for field, *_ in updates):
            gap_status += cleanup_message
        if any(field == "#BPM" for field, *_ in updates):
            bpm_status += cleanup_message

    return gap_status, bpm_status


def write_gap_update(
    result: Analysis,
    preserved_backup: Optional[bytes] = None,
    allow_small_updates: bool = False,
) -> str:
    """Back up a chart and write an eligible GAP suggestion."""
    return write_chart_updates(
        result,
        write_gap=True,
        write_bpm=False,
        preserved_backup=preserved_backup,
        allow_small_updates=allow_small_updates,
    )[0]


def write_bpm_update(result: Analysis, allow_small_updates: bool = False) -> str:
    """Back up a chart and write an eligible BPM suggestion."""
    return write_chart_updates(
        result,
        write_gap=False,
        write_bpm=True,
        allow_small_updates=allow_small_updates,
    )[1]


def reanalyze_after_bpm_update(
    first_pass: Analysis,
    args: argparse.Namespace,
) -> Analysis:
    """Recalculate GAP and verify BPM after a successful BPM chart update."""
    chart_path = first_pass.chart_path
    backup_path = chart_path.with_suffix(".bak")
    try:
        preserved_backup = backup_path.read_bytes()
    except OSError as exc:
        preserved_backup = None
        backup_error = (
            f"Could not preserve the original chart backup for the GAP update: "
            f"{type(exc).__name__}: {exc}"
        )
    else:
        backup_error = ""

    reanalysis_args = argparse.Namespace(**vars(args))
    if reanalysis_args.device == "both":
        reanalysis_args.device = "cuda"

    try:
        result = analyze_chart_with_cpu_fallback(chart_path, reanalysis_args)
    except Exception as exc:
        result = Analysis(
            chart_path=chart_path,
            audio_path=first_pass.audio_path,
            old_gap_ms=None,
            bpm=None,
            status="Analysis failed",
            reason=(
                "Reanalysis after the #BPM update failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )
        if args.debug:
            result.reason += "\n" + traceback.format_exc()

    result.bpm_write_status = first_pass.bpm_write_status
    result.bpm_sample_notes.append(
        "Second analysis pass ran after the #BPM update to recalculate GAP "
        "and verify BPM; no second BPM update was attempted."
    )
    if result.status == "Analysis failed":
        result.gap_write_status = (
            "Skipped: #BPM was updated, but the second analysis pass failed; "
            "the first-pass GAP was not written."
        )
    elif backup_error:
        result.gap_write_status = (
            "Skipped: #BPM was updated and GAP was recalculated, but the "
            f"original backup could not be verified. {backup_error}"
        )
    elif preserved_backup is None:
        result.gap_write_status = (
            "Skipped: #BPM was updated, but the original chart backup is "
            "unavailable; GAP was not written."
        )
    else:
        result.gap_write_status = write_gap_update(
            result,
            preserved_backup=preserved_backup,
            allow_small_updates=getattr(args, "allow_small_updates", False),
        )
    return result


def write_first_pass_chart_updates(result: Analysis, args: argparse.Namespace) -> bool:
    """Write eligible first-pass values and report whether BPM changed."""
    if args.write_bpm:
        result.bpm_write_status = write_bpm_update(
            result,
            allow_small_updates=getattr(args, "allow_small_updates", False),
        )
        if result.bpm_write_status.startswith("Updated #BPM "):
            return True
    if args.write_gap:
        result.gap_write_status = write_gap_update(
            result,
            allow_small_updates=getattr(args, "allow_small_updates", False),
        )
    return False


def format_ms(value: Optional[float]) -> str:
    if value is None:
        return "Not assessed"
    return f"{value:.0f} ms"


def format_bpm_change(
    current_bpm: Optional[float],
    correction: Optional[float],
    diagnostic_only: bool = False,
) -> Optional[str]:
    if (
        current_bpm is None
        or correction is None
        or not math.isfinite(current_bpm)
        or current_bpm <= 0
        or not math.isfinite(correction)
    ):
        return None

    change_percent = correction / current_bpm * 100
    direction = (
        "increase chart tempo"
        if change_percent > 0
        else "decrease chart tempo"
        if change_percent < 0
        else "no chart-tempo change"
    )
    label = "BPM change (diagnostic only)" if diagnostic_only else "BPM change"
    return f"{label}: {change_percent:+.2f}% ({direction})"


def format_summary_report(
    results: list[Analysis],
    args: argparse.Namespace,
    include_header: bool = True,
) -> str:
    lines = []

    if include_header:
        lines.extend([
            f"Version: {VERSION}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Command line: {getattr(args, 'command_line', 'Not captured')}",
            "",
            f"Match threshold: {args.match_threshold:.2f}",
            f"WhisperX batch size: {args.batch_size}",
            f"Initial audio opening window: {args.audio_opening_seconds:.1f} seconds",
            f"Initial chart opening window: {args.chart_opening_seconds:.1f} seconds",
            (
                "Writing #BPM also enables #GAP writing; a successful BPM "
                "update triggers a second analysis pass before GAP is written."
                if args.write_bpm
                else ""
            ),
            (
                "--allow-small-updates was supplied: the normal GAP and BPM "
                "size thresholds are bypassed; write options, confidence, and "
                "all other safeguards still apply."
                if getattr(args, "allow_small_updates", False)
                else ""
            ),
            "",
        ])

    for result in results:
        audio_window = (
            result.audio_opening_seconds
            if result.audio_opening_seconds is not None
            else args.audio_opening_seconds
        )
        chart_window = (
            result.chart_opening_seconds
            if result.chart_opening_seconds is not None
            else args.chart_opening_seconds
        )
        lines.extend([
            "-" * 78,
            f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Chart: {result.chart_path}",
            f"Audio: {result.audio_path or 'Not found'}",
            f"Status: {result.status}",
            f"Audio opening window used: {audio_window:.1f} seconds",
            f"Chart opening window used: {chart_window:.1f} seconds",
            f"Current #GAP: {format_ms(result.old_gap_ms)}",
            f"Current #BPM: {result.bpm:.1f}" if result.bpm is not None else "Current #BPM: Unknown",
        ])

        if (
            result.suggested_gap_ms is not None
            and result.correction_ms is not None
            and result.confidence in {"High", "Medium"}
        ):
            lines.extend([
                f"Suggested #GAP: {result.suggested_gap_ms:.0f}",
                f"Correction: {result.correction_ms:+.0f} ms",
            ])
        else:
            lines.extend([
                "Suggested #GAP: Withheld",
                (
                    f"Correction: {result.correction_ms:+.0f} ms (diagnostic only)"
                    if result.correction_ms is not None
                    else "Correction: Not assessed"
                ),
            ])

        lines.extend([
            f"Confidence: {result.confidence}",
            f"Reason: {(result.reason or 'No additional details').splitlines()[0]}",
        ])
        if result.gap_drift_warning:
            lines.append(f"Warning: {result.gap_drift_warning}")
        if args.bpm:
            if (
                result.suggested_bpm is not None
                and result.bpm_correction is not None
                and result.bpm_confidence in {"High", "Medium"}
                and not result.bpm_write_block_reason
            ):
                lines.extend([
                    f"Suggested #BPM: {result.suggested_bpm:.3f}",
                    f"BPM correction: {result.bpm_correction:+.3f}",
                    f"BPM confidence: {result.bpm_confidence}",
                ])
                bpm_change = format_bpm_change(
                    result.bpm,
                    result.bpm_correction,
                )
                if bpm_change:
                    lines.append(bpm_change)
            else:
                lines.extend([
                    "Suggested #BPM: Withheld",
                    (
                        f"BPM correction: {result.bpm_correction:+.3f} "
                        "(diagnostic only)"
                        if result.bpm_correction is not None
                        else "BPM correction: Not assessed"
                    ),
                    f"BPM confidence: {result.bpm_confidence}",
                ])
                if result.bpm_write_block_reason:
                    lines.append(
                        "BPM update: Withheld because "
                        f"{result.bpm_write_block_reason}."
                    )
                bpm_change = format_bpm_change(
                    result.bpm,
                    result.bpm_correction,
                    diagnostic_only=True,
                )
                if bpm_change:
                    lines.append(bpm_change)
        if args.write_gap:
            lines.append(f"#GAP write: {result.gap_write_status}")
        if args.write_bpm:
            lines.append(f"#BPM write: {result.bpm_write_status}")
        if result.bpm_sample_notes:
            lines.extend(["BPM sampling notes:"])
            lines.extend(f"  - {note}" for note in result.bpm_sample_notes)
        lines.append("")

    return "\n".join(lines)


def format_report(
    results: list[Analysis],
    args: argparse.Namespace,
    include_header: bool = True,
) -> str:
    if args.report_level == "summary":
        return format_summary_report(results, args, include_header)

    lines = [
    ]

    if include_header:
        lines.extend([
            "UltraStar GAP Analysis Report",
            "=" * 78,
            f"Version: {VERSION}",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Command line: {getattr(args, 'command_line', 'Not captured')}",
            "",
            (
                "Safety: source audio and chart files were read only."
                if not (args.write_gap or args.write_bpm)
                else (
                    "Safety: source audio is read only; eligible chart values "
                    "are updated only at Medium or High confidence, after a "
                    ".bak backup."
                )
            ),
            f"Match threshold: {args.match_threshold:.2f}",
            f"WhisperX batch size: {args.batch_size}",
            f"Initial audio opening window: {args.audio_opening_seconds:.1f} seconds",
            f"Initial chart opening window: {args.chart_opening_seconds:.1f} seconds",
            (
                "Writing #BPM also enables #GAP writing; a successful BPM "
                "update triggers a second analysis pass before GAP is written."
                if args.write_bpm
                else ""
            ),
            (
                "--allow-small-updates was supplied: the normal GAP and BPM "
                "size thresholds are bypassed; write options, confidence, and "
                "all other safeguards still apply."
                if getattr(args, "allow_small_updates", False)
                else ""
            ),
            "BPM verification can sample later sections without transcribing the full song.",
            "",
        ])

    for result in results:
        audio_window = (
            result.audio_opening_seconds
            if result.audio_opening_seconds is not None
            else args.audio_opening_seconds
        )
        chart_window = (
            result.chart_opening_seconds
            if result.chart_opening_seconds is not None
            else args.chart_opening_seconds
        )
        lines.extend([
            "-" * 78,
            f"Processed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Chart: {result.chart_path}",
            f"Audio: {result.audio_path or 'Not found'}",
            f"Status: {result.status}",
            f"Audio opening window used: {audio_window:.1f} seconds",
            f"Chart opening window used: {chart_window:.1f} seconds",
            f"Current #GAP: {format_ms(result.old_gap_ms)}",
            f"Current #BPM: {result.bpm if result.bpm is not None else 'Unknown'}",
        ])

        # Keep the recommendation near the current settings, before diagnostic details.
        if (
            result.suggested_gap_ms is not None
            and result.correction_ms is not None
            and result.confidence in {"High", "Medium"}
        ):
            lines.extend([
                f"Suggested #GAP: {result.suggested_gap_ms:.0f}",
                f"Correction: {result.correction_ms:+.0f} ms",
            ])
        else:
            lines.extend([
                "Suggested #GAP: WITHHELD because the evidence is insufficient.",
                f"Estimated GAP (diagnostic only): "
                f"{format_ms(result.suggested_gap_ms)}",
                f"Estimated correction (diagnostic only): "
                f"{format_ms(result.correction_ms)}",
            ])

        lines.extend([
            f"Confidence: {result.confidence}",
            f"Reason: {result.reason or 'No additional details'}",
        ])
        if result.gap_drift_warning:
            lines.append(f"Warning: {result.gap_drift_warning}")
        if args.write_gap:
            lines.append(f"#GAP write: {result.gap_write_status}")
        if args.write_bpm:
            lines.append(f"#BPM write: {result.bpm_write_status}")

        if not args.bpm:
            lines.extend(["BPM verification: Disabled (pass --bpm to enable)."])
        elif (
            result.suggested_bpm is not None
            and result.bpm_confidence in {"High", "Medium"}
            and not result.bpm_write_block_reason
        ):
            lines.extend([
                f"Suggested #BPM: {result.suggested_bpm:.3f}",
                f"BPM correction: {result.bpm_correction:+.3f}",
                f"BPM confidence: {result.bpm_confidence}",
                f"BPM assessment: {result.bpm_reason}",
            ])
            bpm_change = format_bpm_change(
                result.bpm,
                result.bpm_correction,
            )
            if bpm_change:
                lines.append(bpm_change)
        elif result.suggested_bpm is not None:
            lines.extend([
                f"Estimated #BPM (diagnostic only): {result.suggested_bpm:.3f}",
                f"BPM correction (diagnostic only): {result.bpm_correction:+.3f}",
                f"BPM confidence: {result.bpm_confidence}",
                f"BPM assessment: {result.bpm_reason}",
            ])
            if result.bpm_write_block_reason:
                lines.append(
                    "BPM update: Withheld because "
                    f"{result.bpm_write_block_reason}."
                )
            bpm_change = format_bpm_change(
                result.bpm,
                result.bpm_correction,
                diagnostic_only=True,
            )
            if bpm_change:
                lines.append(bpm_change)
        elif args.bpm:
            lines.extend([
                "Suggested #BPM: WITHHELD because the evidence is insufficient.",
                (
                    f"BPM correction (diagnostic only): "
                    f"{result.bpm_correction:+.3f}"
                    if result.bpm_correction is not None
                    else "BPM correction: Not assessed"
                ),
                f"BPM confidence: {result.bpm_confidence}",
                f"BPM assessment: {result.bpm_reason or 'Not assessed'}",
            ])
            bpm_change = format_bpm_change(
                result.bpm,
                result.bpm_correction,
                diagnostic_only=True,
            )
            if bpm_change:
                lines.append(bpm_change)

        if result.transcription:
            lines.extend([
                "",
                "Whisper transcription (opening):",
                result.transcription,
            ])

        if result.chart_words:
            chart_opening_words = [
                word.text for word in result.chart_words
                if word.seconds <= chart_window
            ]
            if chart_opening_words:
                lines.extend([
                    "Chart transcription (opening):",
                    " ".join(chart_opening_words),
                ])

        if result.matches:
            lines.extend([
                "",
                "Ordered lyric matches:",
                "Chart word | Audio word | Similarity | Chart time | "
                "Audio time | GAP estimate",
            ])

            for match in result.matches:
                lines.append(
                    f"{match.chart.text:12} | "
                    f"{match.audio.text:16} | "
                    f"{match.similarity:5.2f} | "
                    f"{match.chart.seconds:8.3f}s | "
                    f"{match.audio.start:8.3f}s | "
                    f"{match.gap_ms:8.0f} ms"
                )

        if not args.bpm:
            lines.extend(["", "BPM verification: Disabled (pass --bpm to enable)."])
        else:
            lines.extend(["", "BPM verification:"])
            if result.bpm_samples:
                lines.append("Sample | Chart time | Median offset | Matches | Offset MAD")
                for sample in result.bpm_samples:
                    lines.append(
                        f"{sample.label:10} | {sample.chart_time_seconds:8.2f}s | "
                        f"{sample.offset_ms:+8.0f} ms | {sample.match_count:7d} | "
                        f"{sample.spread_ms:8.0f} ms"
                    )
            else:
                lines.append("No reliable BPM timing samples were available.")

            if result.bpm_sample_notes:
                lines.append("BPM sampling notes:")
                lines.extend(f"  - {note}" for note in result.bpm_sample_notes)

            later_sample_diagnostics = [
                diagnostic
                for diagnostic in result.bpm_sample_diagnostics
                if diagnostic.label != "Beginning"
            ]
            if later_sample_diagnostics:
                lines.extend([
                    "Beginning BPM sample reuses the opening GAP transcription "
                    "and matches shown below.",
                    "Later BPM sample transcription and matching details:",
                ])
                for diagnostic in later_sample_diagnostics:
                    lines.extend([
                        "",
                        (
                            f"{diagnostic.label}: audio "
                            f"{diagnostic.audio_start:.2f}-{diagnostic.audio_end:.2f}s"
                        ),
                        "Chart words: "
                        + (
                            " ".join(word.text for word in diagnostic.chart_words)
                            or "(none in corresponding chart-time window)"
                        ),
                        "Whisper transcription: "
                        + (diagnostic.transcription or "(empty transcription)"),
                        "Aligned Whisper words: "
                        + (
                            " ".join(
                                f"{word.text} [{word.start:.2f}-{word.end:.2f}s]"
                                for word in diagnostic.aligned_words
                            )
                            or "(none)"
                        ),
                    ])
                    if diagnostic.matches:
                        lines.extend([
                            "Matched chart/Whisper words:",
                            "Chart word | Whisper word | Similarity | "
                            "Chart time | Audio time | Offset",
                        ])
                        for match in diagnostic.matches:
                            lines.append(
                                f"{match.chart.text:12} | "
                                f"{match.audio.text:13} | "
                                f"{match.similarity:.2f} | "
                                f"{match.chart.seconds:8.3f}s | "
                                f"{match.audio.start:8.3f}s | "
                                f"{match.gap_ms:+8.0f}ms"
                            )
                    else:
                        lines.append("Matched chart/Whisper words: (none)")
                    if diagnostic.note:
                        lines.append(f"Sample result: {diagnostic.note}")

        if result.unknown_note_types:
            grouped: dict[str, list[int]] = {}
            for note_type, line_number in result.unknown_note_types:
                grouped.setdefault(note_type, []).append(line_number)
            lines.append("Warning: Unknown note type(s) detected; these lines were skipped:")
            for note_type, line_numbers in sorted(grouped.items()):
                locations = ", ".join(str(n) for n in line_numbers)
                lines.append(f"  {note_type!r} at chart line(s): {locations}")

        if (
            result.suggested_gap_ms is not None
            and result.correction_ms is not None
            and result.confidence in {"High", "Medium"}
        ):
            lines.extend([
                "Decision: provisional suggestion; review the matches before "
                "changing the chart.",
            ])

        lines.append("")

    return "\n".join(lines)


def format_review_report_entry(result: Analysis) -> str:
    eligible_gap = (
        result.audio_path is not None
        and result.suggested_gap_ms is not None
        and result.correction_ms is not None
        and math.isfinite(result.correction_ms)
        and abs(result.correction_ms) >= 1000
        and result.confidence in {"High", "Medium"}
    )
    if not eligible_gap and not result.gap_drift_warning:
        return ""

    lines = [f"Audio: {result.audio_path or 'Not found'}"]
    if eligible_gap:
        lines.extend([
            f"GAP Correction: {result.correction_ms:+.0f} ms",
            f"Confidence: {result.confidence}",
        ])
    if result.gap_drift_warning:
        lines.append(f"Warning: {result.gap_drift_warning}")
    return "\n".join(lines) + "\n\n"


def resolve_source_directories(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_dir():
        return [source]
    if not source.is_file():
        raise ValueError(
            f"--source must be an existing directory or a text file: {source}"
        )

    try:
        lines = source.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise ValueError(f"Could not read source list {source}: {exc}") from exc

    directories: list[Path] = []
    for line_number, line in enumerate(lines, start=1):
        entry = line.strip()
        if not entry:
            continue

        entry_path = Path(entry).expanduser()
        directory = (
            entry_path if entry_path.is_absolute() else source.parent / entry_path
        ).resolve()
        if not directory.is_dir():
            raise ValueError(
                f"Source list {source}, line {line_number}: "
                f"not an existing directory: {entry}"
            )
        directories.append(directory)

    if not directories:
        raise ValueError(f"Source list contains no directory paths: {source}")

    unique_directories = sorted(set(directories), key=lambda directory: directory.parts)
    non_overlapping_directories: list[Path] = []
    non_overlapping_set: set[Path] = set()
    for directory in unique_directories:
        parent = directory.parent
        while parent != directory and parent not in non_overlapping_set:
            if parent == parent.parent:
                parent = directory
                break
            parent = parent.parent
        if parent == directory:
            non_overlapping_directories.append(directory)
            non_overlapping_set.add(directory)
    return non_overlapping_directories


def find_chart_paths(
    source_directories: list[Path],
    excluded_paths: set[Path],
) -> list[Path]:
    charts_by_resolved_path: dict[Path, Path] = {}
    for directory in source_directories:
        for chart_path in directory.rglob("*.txt"):
            if not chart_path.is_file():
                continue
            resolved_chart_path = chart_path.resolve()
            if resolved_chart_path in excluded_paths:
                continue
            charts_by_resolved_path.setdefault(resolved_chart_path, chart_path)
    return sorted(charts_by_resolved_path.values())


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate UltraStar #GAP values and verify #BPM using sampled "
            "chart lyrics and WhisperX word timestamps. Source files are read only "
            "unless a chart-writing option is enabled."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help=(
            "Directory containing charts, or a UTF-8 text file with one song "
            "directory per line (relative paths are relative to the list file)."
        ),
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Path to the text report to create.",
    )
    parser.add_argument(
        "--report-level",
        choices=("all", "summary"),
        default="all",
        help="Report detail level: all (default) or summary.",
    )
    parser.add_argument(
        "--language",
        default=None,
        help="WhisperX language code; omit to detect the language automatically.",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="WhisperX model name (default: medium).",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=("cpu", "cuda", "both"),
        help=(
            "Inference device: cpu, cuda, or both (one CPU and one CUDA "
            "analysis at a time; default: cpu)."
        ),
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="WhisperX compute type; defaults to int8 on CPU and float16 on CUDA.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        choices=(4, 8, 16, 32, 64),
        default=4,
        help="WhisperX transcription batch size (choices: 4, 8, 16, 32, 64; default: 4).",
    )
    parser.add_argument(
        "--match-threshold",
        type=float,
        default=0.55,
        help=(
            "Minimum fuzzy word similarity from 0 to 1. Lower is more "
            "permissive; default: 0.55."
        ),
    )
    parser.add_argument(
        "--audio-opening-seconds",
        type=float,
        default=60.0,
        help=(
            "Minimum audio opening window to transcribe; sparse chart lyrics "
            "may extend it to 90 seconds (default: 60)."
        ),
    )
    parser.add_argument(
        "--chart-opening-seconds",
        type=float,
        default=60.0,
        help=(
            "Minimum chart opening window to consider; sparse lyrics may "
            "extend it to 90 seconds (default: 60)."
        ),
    )
    parser.add_argument(
        "--max-correction-ms",
        type=float,
        default=15000.0,
        help="Withhold suggestions exceeding this correction (default: 15000).",
    )
    parser.add_argument(
        "--bpm",
        action="store_true",
        help="Enable BPM analysis (disabled by default).",
    )
    parser.add_argument(
        "--write-gap",
        action="store_true",
        help=(
            "Update chart #GAP values when confidence is Medium or High; "
            "refresh a sibling .bak backup first. Implied by --write-bpm."
        ),
    )
    parser.add_argument(
        "--write-bpm",
        action="store_true",
        help=(
            "Update chart #BPM values when confidence is Medium or High; "
            "requires --bpm, also enables #GAP writing, and reanalyzes "
            "after a successful BPM update."
        ),
    )
    parser.add_argument(
        "--allow-small-updates",
        action="store_true",
        help=(
            "Allow GAP and BPM writes below their normal size thresholds; "
            "all other write safeguards still apply."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include full exception tracebacks in the report.",
    )

    args = parser.parse_args()
    args.command_line = shlex.join([sys.executable, *sys.argv])

    if not 0.0 <= args.match_threshold <= 1.0:
        parser.error("--match-threshold must be between 0 and 1.")

    if args.audio_opening_seconds <= 0 or args.chart_opening_seconds <= 0:
        parser.error("Opening windows must be greater than zero.")

    if args.max_correction_ms < 0:
        parser.error("--max-correction-ms cannot be negative.")

    if args.write_bpm and not args.bpm:
        parser.error("--write-bpm requires --bpm.")
    if args.write_bpm:
        args.write_gap = True

    args.compute_type_explicit = args.compute_type is not None
    if args.compute_type is None:
        args.compute_type = "int8" if args.device == "cpu" else "float16"

    source = Path(args.source).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    review_report_path = report_path.with_name(
        f"{report_path.stem}_review{report_path.suffix}"
    )

    try:
        source_directories = resolve_source_directories(source)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    # Existing reports may be overwritten, including reports inside the source
    # tree. Still refuse to overwrite an actual chart or audio file.
    report_paths = {report_path, review_report_path}
    for output_path in report_paths:
        if output_path.exists() and output_path.suffix.casefold() in AUDIO_SUFFIXES:
            print(
                f"Error: report path points to an audio file: {output_path}; "
                "refusing to overwrite it.",
                file=sys.stderr,
            )
            return 2
        if output_path.exists() and is_existing_ultrastar_chart(output_path):
            print(
                f"Error: report path points to an existing UltraStar chart: "
                f"{output_path}; refusing to overwrite it.",
                file=sys.stderr,
            )
            return 2

    excluded_paths = set(report_paths)
    if source.is_file():
        excluded_paths.add(source)
    chart_paths = find_chart_paths(
        source_directories,
        excluded_paths,
    )

    if not chart_paths:
        print(
            "No .txt chart files found under source path(s): "
            + ", ".join(str(directory) for directory in source_directories),
            file=sys.stderr,
        )
        return 1

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as report_file:
            with review_report_path.open("w", encoding="utf-8") as review_report_file:
                report_file.write(format_report([], args))
                review_report_file.write(
                    "Songs with recommended GAP corrections of at least 1,000 ms "
                    "or a possible opening timing-drift warning\n\n"
                )
                song_durations: list[float] = []

                def append_result(result: Analysis) -> None:
                    report_file.write(
                        format_report([result], args, include_header=False)
                    )
                    review_report_file.write(format_review_report_entry(result))
                    report_file.flush()
                    review_report_file.flush()

                if args.device == "both":
                    both_started_at = time.monotonic()
                    pending_bpm_reanalysis: list[Analysis] = []
                    for index, (_, output) in enumerate(
                        analyze_charts_on_both_devices(chart_paths, args),
                        start=1,
                    ):
                        if output.logs:
                            print(output.logs, end="", flush=True)
                            if not output.logs.endswith("\n"):
                                print(flush=True)
                        if write_first_pass_chart_updates(output.analysis, args):
                            pending_bpm_reanalysis.append(output.analysis)
                        else:
                            append_result(output.analysis)
                        estimated_end = estimate_parallel_completion_time(
                            time.monotonic() - both_started_at,
                            index,
                            len(chart_paths),
                        )
                        end_text = (
                            format_estimated_end(estimated_end)
                            if estimated_end is not None
                            else "calculating"
                        )
                        print(
                            f"[{index}/{len(chart_paths)} completed; "
                            f"Estimated End {end_text}] "
                            f"{output.analysis.chart_path}: "
                            f"{output.analysis.status}: "
                            f"{output.analysis.confidence}",
                            flush=True,
                        )
                    for reanalysis_index, first_pass in enumerate(
                        pending_bpm_reanalysis,
                        start=1,
                    ):
                        print(
                            f"[BPM update reanalysis "
                            f"{reanalysis_index}/{len(pending_bpm_reanalysis)}] "
                            f"Analyzing: {first_pass.chart_path}",
                            flush=True,
                        )
                        result = reanalyze_after_bpm_update(first_pass, args)
                        append_result(result)
                        print(
                            f"    Second-pass {result.status}: "
                            f"{result.confidence}",
                            flush=True,
                        )
                else:
                    for index, chart_path in enumerate(chart_paths, start=1):
                        estimated_end = estimate_completion_time(
                            song_durations,
                            len(chart_paths) - index + 1,
                        )
                        end_text = (
                            format_estimated_end(estimated_end)
                            if estimated_end is not None
                            else "calculating"
                        )
                        print(
                            f"[{index}/{len(chart_paths)} Estimated End {end_text}] "
                            f"Analyzing: {chart_path}"
                        )
                        song_started_at = time.monotonic()
                        try:
                            result = analyze_chart_with_cpu_fallback(
                                chart_path,
                                args,
                            )
                        except Exception as exc:
                            result = Analysis(
                                chart_path=chart_path,
                                audio_path=find_audio_for_chart(chart_path),
                                old_gap_ms=None,
                                bpm=None,
                                status="Analysis failed",
                                reason=f"{type(exc).__name__}: {exc}",
                            )
                            if args.debug:
                                result.reason += "\n" + traceback.format_exc()
                        if write_first_pass_chart_updates(result, args):
                            result = reanalyze_after_bpm_update(result, args)
                        append_result(result)
                        song_durations.append(time.monotonic() - song_started_at)
                        print(f"    {result.status}: {result.confidence}")
                        del result
    except OSError as exc:
        print(f"Could not write report(s): {exc}", file=sys.stderr)
        return 2

    print()
    print(f"Report written to: {report_path}")
    print(f"Review report written to: {review_report_path}")
    if args.write_gap or args.write_bpm:
        print(
            "Audio files were not modified; eligible chart #GAP and/or #BPM "
            "values may have been updated."
        )
    else:
        print("Audio and chart files were not modified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())