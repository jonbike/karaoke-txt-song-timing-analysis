#!/usr/bin/env python3
"""
Estimate UltraStar Deluxe #GAP values using WhisperX word timestamps.

Safety:
- Reads audio and .txt chart files only.
- Does not modify charts, audio, or other source files.
- Writes a text report only unless --write-gap is enabled.

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
import difflib
from datetime import datetime, timedelta
from importlib.metadata import version
import math
import multiprocessing
import os
import re
import shutil
import stat
import statistics
import subprocess
import sys
import tempfile
import time
import traceback

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from korean_romanizer import Romanizer


VERSION = version("karaoke-txt-song-timing-analysis")


NOTE_RE = re.compile(
    r"^([:*FRG])\s*(-?\d+)\s+(\d+)\s+(-?\d+)\s(.*)$",
    re.IGNORECASE,
)

# Structural note-line pattern used to flag unsupported note-type markers.
ANY_NOTE_RE = re.compile(r"^([^\s:#-])\s+-?\d+\s+\d+\s+-?\d+\s+.*$")

NUMBER_RE = r"[+-]?\d+(?:[.,]\d+)?"
GAP_RE = re.compile(rf"^\s*#GAP:\s*({NUMBER_RE})", re.I)
MIN_GAP_WRITE_CHANGE_MS = 100.0
GAP_BYTES_RE = re.compile(
    rb"^(\xef\xbb\xbf)?([ \t]*#GAP:[ \t]*)"
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
    bpm_sample_notes: list[str] = field(default_factory=list)
    bpm_sample_diagnostics: list[BpmSampleDiagnostic] = field(default_factory=list)
    gap_write_status: str = ""


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
    supported = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".opus"}
    reference_tag, audio_reference = read_audio_reference(chart_path)

    if audio_reference:
        # Prefer #AUDIO; legacy #MP3 is selected only when #AUDIO is absent.
        reference = Path(audio_reference.replace("\\", "/")).expanduser()
        candidate = reference if reference.is_absolute() else chart_path.parent / reference
        resolved = resolve_case_insensitive(candidate)
        if resolved and resolved.suffix.casefold() in supported:
            return resolved
        return None

    for candidate in chart_path.parent.iterdir():
        if (
            candidate.is_file()
            and candidate.stem.casefold() == chart_path.stem.casefold()
            and candidate.suffix.casefold() in supported
        ):
            return candidate

    return None


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
    opening_seconds: float,
    extra_windows: list[tuple[str, float, float]],
) -> tuple[
    list[AudioWord],
    float,
    str,
    Optional[str],
    dict[str, list[AudioWord]],
    dict[str, str],
]:
    """Transcribe the opening plus selected later windows, loading models once.

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
            batch_size=4,
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
    if len(usable) < 2:
        return (
            None, None, "Low",
            f"Only {len(usable)} reliable timing samples; at least 2 are needed.",
        )

    span = max(s.chart_time_seconds for s in usable) - min(
        s.chart_time_seconds for s in usable
    )
    if span < 30.0:
        return (
            None, None, "Low",
            f"Reliable samples span only {span:.1f}s; at least 30s are needed.",
        )

    # Theil-Sen slope: median of all pairwise offset slopes, robust to one bad sample.
    slopes = []
    for i, first in enumerate(usable):
        for second in usable[i + 1:]:
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
        for sample in usable
    )
    residuals = [
        sample.offset_ms - (intercept + slope * sample.chart_time_seconds * 1000.0)
        for sample in usable
    ]
    residual_mad = median_absolute_deviation(residuals)
    sample_summary = (
        f"{len(usable)} reliable samples over {span:.1f}s; "
        f"timing-drift residual MAD {residual_mad:.0f} ms."
    )

    if relative_change > 0.15:
        confidence = "Low"
        reason = (
            f"Estimated BPM change {relative_change * 100:.1f}% exceeds the "
            f"15% safety limit; review sample matches. {sample_summary}"
        )
    elif len(usable) == 2:
        confidence = "Low"
        reason = (
            f"Only 2 reliable samples over {span:.1f}s; at least 3 are needed "
            f"for a stronger confidence assessment. {sample_summary}"
        )
    elif (
        len(usable) >= 4
        and all(sample.match_count >= 4 for sample in usable)
        and residual_mad <= 500
        and all(sample.spread_ms <= 700 for sample in usable)
    ):
        confidence = "High"
        reason = sample_summary
    elif residual_mad <= 1000:
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

    gap_values = [m.gap_ms for m in matches]
    suggested_gap = statistics.median(gap_values)
    correction = suggested_gap - old_gap_ms
    spread = median_absolute_deviation(gap_values)

    strong_matches = [
        m for m in matches
        if m.similarity >= max(threshold + 0.12, 0.67)
    ]

    distinct_phrases = 0
    if len(matches) >= 2:
        # A phrase boundary is approximated by a gap in chart timing.
        for previous, current in zip(matches, matches[1:]):
            if current.chart.seconds - previous.chart.seconds > 0.65:
                distinct_phrases += 1

    matched_span = (
        matches[-1].audio.start - matches[0].audio.start
        if len(matches) >= 2
        else 0.0
    )

    # Short common words are useful to anchor the sequence, but don't allow
    # them alone to justify a GAP change.
    informative_matches = [
        m for m in matches
        if len(normalize_word(m.chart.text)) >= 3
        or len(normalize_word(m.audio.text)) >= 3
    ]

    if len(matches) < 3:
        return (
            suggested_gap,
            correction,
            "Low",
            f"Only {len(matches)} ordered matches; at least 3 are needed.",
        )

    if len(informative_matches) < 2:
        return (
            suggested_gap,
            correction,
            "Low",
            "Too few informative word matches; short common words alone "
            "are not enough.",
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
    if len(matches) >= 5 and spread <= 600 and len(strong_matches) >= 2:
        confidence = "High"
    elif len(matches) >= 3 and spread <= 1000:
        confidence = "Medium"
    else:
        confidence = "Low"

    reasons = [
        f"{len(matches)} ordered matches",
        f"{len(strong_matches)} stronger matches",
        f"timing MAD {spread:.0f} ms",
    ]
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
    snippet_path = output_directory / (
        f"{audio_path.stem}_{duration_seconds:g}s{audio_path.suffix}"
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
            if args.bpm == "off":
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
        reference_tag, audio_reference = read_audio_reference(chart_path)
        if audio_reference:
            analysis.reason = (
                f"No audio file referenced by {reference_tag} "
                f"({audio_reference!r}) was found."
            )
        else:
            analysis.reason = (
                "No #AUDIO/#MP3 reference or same-basename audio file was found."
            )
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
        ) = transcribe_audio_windows(
            audio_path=transcription_audio_path,
            model_name=args.model,
            language=args.language,
            device=args.device,
            compute_type=args.compute_type,
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
    if not opening_audio_words:
        analysis.reason = (
            f"No aligned opening audio word timestamps were returned "
            f"(language: {detected_language}). {reason}"
        )

    if args.bpm == "off":
        analysis.bpm_reason = "BPM analysis disabled by --bpm off."
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

    # Beginning sample comes from the existing opening match, limited to the
    # first 24 chart seconds so it represents one region rather than the whole
    # opening alignment.
    if chart_words:
        beginning_limit = chart_words[0].seconds + 24.0
        beginning_matches = [m for m in matches if m.chart.seconds <= beginning_limit]
        beginning_sample = summarize_bpm_sample("Beginning", beginning_matches)
        bpm_samples: list[BpmSample] = []
        if beginning_sample:
            bpm_samples.append(beginning_sample)
        beginning_chart_words = [
            word for word in chart_words
            if word.seconds <= beginning_limit
        ]
        analysis.bpm_sample_diagnostics.append(
            BpmSampleDiagnostic(
                label="Beginning",
                audio_start=0.0,
                audio_end=min(duration, audio_opening_seconds),
                chart_words=beginning_chart_words,
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
                ) = transcribe_audio_windows(
                    audio_path=transcription_audio_path,
                    model_name=args.model,
                    language=detected_language,
                    device=args.device,
                    compute_type=args.compute_type,
                    opening_seconds=0,
                    extra_windows=extra_windows,
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
            analysis.bpm_confidence = "Low"
            analysis.bpm_reason += (
                f" GAP confidence is {confidence}; BPM result is diagnostic only."
            )

    return analysis


def _analyze_chart_worker(connection, chart_path: Path, args: argparse.Namespace) -> None:
    try:
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
        connection.send(result)
    finally:
        connection.close()


def analyze_chart_in_child_process(
    chart_path: Path,
    args: argparse.Namespace,
) -> Analysis:
    context = multiprocessing.get_context("spawn")
    receive_connection, send_connection = context.Pipe(duplex=False)
    process = context.Process(
        target=_analyze_chart_worker,
        args=(send_connection, chart_path, args),
    )
    try:
        process.start()
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
    process.join()

    if result is None:
        return Analysis(
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
    return result


def write_gap_update(result: Analysis) -> str:
    """Back up a chart and atomically write a confident GAP suggestion."""
    if result.confidence not in {"High", "Medium"}:
        return (
            f"Skipped: GAP confidence is {result.confidence}; "
            "only Medium or High confidence is writable."
        )
    if result.suggested_gap_ms is None or not math.isfinite(result.suggested_gap_ms):
        return "Skipped: no finite GAP suggestion is available."

    chart_path = result.chart_path
    if chart_path.is_symlink():
        return "Skipped: chart is a symbolic link; refusing to replace its path."
    try:
        original = chart_path.read_bytes()
        current_gap, _, _ = parse_chart(chart_path)
    except (OSError, UnicodeError, ValueError) as exc:
        return f"Failed to read chart before update: {type(exc).__name__}: {exc}"

    if current_gap != result.old_gap_ms:
        return (
            "Skipped: chart #GAP changed after analysis "
            f"(analyzed {format_ms(result.old_gap_ms)}, now {format_ms(current_gap)})."
        )

    lines = original.splitlines(keepends=True)
    gap_line_indexes = [
        index for index, line in enumerate(lines)
        if GAP_BYTES_RE.match(line)
    ]
    if len(gap_line_indexes) != 1:
        return (
            "Skipped: expected exactly one writable #GAP line, "
            f"found {len(gap_line_indexes)}."
        )

    new_gap_text = f"{result.suggested_gap_ms:.0f}"
    new_gap_value = float(new_gap_text)
    gap_change = abs(new_gap_value - current_gap)
    if gap_change == 0:
        return f"No update needed: #GAP is already {new_gap_text}."
    if gap_change < MIN_GAP_WRITE_CHANGE_MS:
        return (
            f"Skipped: proposed #GAP change is {gap_change:.1f} ms, below "
            f"the {MIN_GAP_WRITE_CHANGE_MS:.0f} ms write threshold."
        )

    line_index = gap_line_indexes[0]
    match = GAP_BYTES_RE.match(lines[line_index])
    if match is None:
        return "Skipped: could not safely parse the existing #GAP line."
    updated_line = (
        (match.group(1) or b"")
        + match.group(2)
        + new_gap_text.encode("ascii")
        + match.group(4)
    )
    lines[line_index] = updated_line
    updated = b"".join(lines)
    backup_path = chart_path.with_suffix(".bak")
    if backup_path.is_symlink():
        return "Failed to update chart: refusing to overwrite a symbolic-link backup."
    temporary_path: Optional[Path] = None
    update_error: Optional[OSError] = None
    cleanup_error: Optional[OSError] = None

    try:
        shutil.copy2(chart_path, backup_path)
        if backup_path.read_bytes() != original:
            return (
                "Skipped: chart contents changed while preparing the backup; "
                f"the refreshed backup is {backup_path}."
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

    if update_error is not None:
        status = (
            f"Failed to update chart: {type(update_error).__name__}: "
            f"{update_error}"
        )
        if backup_path.exists():
            status += f" Backup: {backup_path}"
    else:
        status = (
            f"Updated #GAP from {format_ms(current_gap)} to {new_gap_text}; "
            f"backup: {backup_path}."
        )
    if cleanup_error is not None:
        status += (
            f" Temporary-file cleanup failed: {type(cleanup_error).__name__}: "
            f"{cleanup_error}"
        )

    return status


def format_ms(value: Optional[float]) -> str:
    if value is None:
        return "Not assessed"
    return f"{value:.0f} ms"


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
            "",
            f"Match threshold: {args.match_threshold:.2f}",
            f"Initial audio opening window: {args.audio_opening_seconds:.1f} seconds",
            f"Initial chart opening window: {args.chart_opening_seconds:.1f} seconds",
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
        if args.write_gap:
            lines.append(f"#GAP write: {result.gap_write_status}")
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
            "",
            (
                "Safety: source audio and chart files were read only."
                if not args.write_gap
                else (
                    "Safety: source audio is read only; charts are updated only "
                    "when GAP confidence is Medium or High, after a .bak backup."
                )
            ),
            f"Match threshold: {args.match_threshold:.2f}",
            f"Initial audio opening window: {args.audio_opening_seconds:.1f} seconds",
            f"Initial chart opening window: {args.chart_opening_seconds:.1f} seconds",
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
        if args.write_gap:
            lines.append(f"#GAP write: {result.gap_write_status}")

        if args.bpm == "off":
            lines.extend(["BPM verification: Disabled by --bpm off."])
        elif (
            result.suggested_bpm is not None
            and result.bpm_confidence in {"High", "Medium"}
        ):
            lines.extend([
                f"Suggested #BPM: {result.suggested_bpm:.3f}",
                f"BPM correction: {result.bpm_correction:+.3f}",
                f"BPM confidence: {result.bpm_confidence}",
                f"BPM assessment: {result.bpm_reason}",
            ])
        elif result.suggested_bpm is not None:
            lines.extend([
                f"Estimated #BPM (diagnostic only): {result.suggested_bpm:.3f}",
                f"BPM correction (diagnostic only): {result.bpm_correction:+.3f}",
                f"BPM confidence: {result.bpm_confidence}",
                f"BPM assessment: {result.bpm_reason}",
            ])
        elif args.bpm != "off":
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

        if args.bpm == "off":
            lines.extend(["", "BPM verification: Disabled by --bpm off."])
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
    if (
        result.audio_path is None
        or result.suggested_gap_ms is None
        or result.correction_ms is None
        or not math.isfinite(result.correction_ms)
        or abs(result.correction_ms) < 1000
        or result.confidence not in {"High", "Medium"}
    ):
        return ""

    return "\n".join([
        f"Audio: {result.audio_path}",
        f"GAP Correction: {result.correction_ms:+.0f} ms",
        f"Confidence: {result.confidence}",
    ]) + "\n\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate UltraStar #GAP values and verify #BPM using sampled "
            "chart lyrics and WhisperX word timestamps. Source files are never modified."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Directory containing UltraStar .txt charts and matching audio.",
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
        choices=("cpu", "cuda"),
        help="Inference device (default: cpu).",
    )
    parser.add_argument(
        "--compute-type",
        default=None,
        help="WhisperX compute type; defaults to int8 on CPU and float16 on CUDA.",
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
        choices=("on", "off"),
        default="off",
        help="Enable or disable BPM analysis (default: off).",
    )
    parser.add_argument(
        "--write-gap",
        action="store_true",
        help=(
            "Update chart #GAP values when confidence is Medium or High; "
            "refresh a sibling .bak backup first."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Include full exception tracebacks in the report.",
    )

    args = parser.parse_args()

    if not 0.0 <= args.match_threshold <= 1.0:
        parser.error("--match-threshold must be between 0 and 1.")

    if args.audio_opening_seconds <= 0 or args.chart_opening_seconds <= 0:
        parser.error("Opening windows must be greater than zero.")

    if args.max_correction_ms < 0:
        parser.error("--max-correction-ms cannot be negative.")

    if args.compute_type is None:
        args.compute_type = "int8" if args.device == "cpu" else "float16"

    source = Path(args.source).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()
    review_report_path = report_path.with_name(
        f"{report_path.stem}_review{report_path.suffix}"
    )

    if not source.is_dir():
        print(f"Error: source directory does not exist: {source}", file=sys.stderr)
        return 2

    # Existing reports may be overwritten, including reports inside the source
    # tree. Still refuse to overwrite an actual chart or audio file.
    audio_suffixes = {".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".opus"}
    report_paths = {report_path, review_report_path}
    for output_path in report_paths:
        if output_path.exists() and output_path.suffix.casefold() in audio_suffixes:
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

    chart_paths = sorted(
        p for p in source.rglob("*.txt")
        if p.is_file() and p.resolve() not in report_paths
    )

    if not chart_paths:
        print(f"No .txt chart files found under: {source}", file=sys.stderr)
        return 1

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as report_file:
            with review_report_path.open("w", encoding="utf-8") as review_report_file:
                report_file.write(format_report([], args))
                review_report_file.write(
                    "Songs with recommended GAP corrections of at least 1,000 ms\n\n"
                )
                song_durations: list[float] = []
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
                        result = analyze_chart_in_child_process(chart_path, args)
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

                    if args.write_gap:
                        result.gap_write_status = write_gap_update(result)
                    report_file.write(
                        format_report([result], args, include_header=False)
                    )
                    review_report_file.write(format_review_report_entry(result))
                    report_file.flush()
                    review_report_file.flush()
                    song_durations.append(time.monotonic() - song_started_at)
                    print(f"    {result.status}: {result.confidence}")
                    del result
    except OSError as exc:
        print(f"Could not write report(s): {exc}", file=sys.stderr)
        return 2

    print()
    print(f"Report written to: {report_path}")
    print(f"Review report written to: {review_report_path}")
    if args.write_gap:
        print("Audio files were not modified; eligible chart #GAP values may have been updated.")
    else:
        print("Audio and chart files were not modified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())