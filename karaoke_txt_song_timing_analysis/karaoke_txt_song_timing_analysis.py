#!/usr/bin/env python3
"""
Estimate UltraStar Deluxe #GAP values using WhisperX word timestamps.

Safety:
- Reads audio and .txt chart files only.
- Does not modify charts, audio, or other source files.
- Writes a text report only.

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
import math
import re
import statistics
import sys
import traceback

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


NOTE_RE = re.compile(
    r"^([:*FRG])\s*(-?\d+)\s+(\d+)\s+(-?\d+)\s+(.*)$",
    re.IGNORECASE,
)

GAP_RE = re.compile(r"^\s*#GAP:\s*(-?\d+(?:\.\d+)?)", re.I)
BPM_RE = re.compile(r"^\s*#BPM:\s*(-?\d+(?:\.\d+)?)", re.I)

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?", re.I)


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
    chart_words: Optional[list[ChartWord]] = None


def normalize_word(text: str) -> str:
    """Normalize a word for forgiving text comparisons."""
    return "".join(TOKEN_RE.findall(text.lower()))


def tokens(text: str) -> list[str]:
    return [normalize_word(x) for x in TOKEN_RE.findall(text) if normalize_word(x)]


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

    def flush_pending_word() -> None:
        nonlocal pending_text, pending_beat
        word = normalize_word(pending_text)
        if word and pending_beat is not None:
            raw_words.append((word, pending_beat))
        pending_text = ""
        pending_beat = None

    with path.open("r", encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            # Remove only the line ending. Do not strip lyric trailing spaces:
            # they carry the word-boundary information.
            line = line.rstrip("\r\n")

            gap_match = GAP_RE.match(line)
            if gap_match:
                gap_ms = float(gap_match.group(1))
                continue

            bpm_match = BPM_RE.match(line)
            if bpm_match:
                bpm = float(bpm_match.group(1))
                continue

            # A phrase break ends any unfinished word.
            if re.match(r"^\s*-\s*-?\d+\s*$", line):
                flush_pending_word()
                continue

            note_match = NOTE_RE.match(line)
            if not note_match:
                continue

            beat = int(note_match.group(2))
            lyric = note_match.group(5)

            # A tilde is a note-extension marker, not a sung word. It also
            # safely closes any pending lyric word before the extension.
            if lyric.strip() == "~":
                flush_pending_word()
                continue

            # Detect the boundary before removing whitespace. Notes such as
            # "an", "ti", "christ " become the complete word "antichrist".
            ends_word = bool(lyric) and lyric[-1].isspace()
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


def find_audio_for_chart(chart_path: Path) -> Optional[Path]:
    """Find audio with the same basename, case-insensitively."""
    supported = {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".aac", ".opus"}

    for candidate in chart_path.parent.iterdir():
        if (
            candidate.is_file()
            and candidate.stem.casefold() == chart_path.stem.casefold()
            and candidate.suffix.casefold() in supported
        ):
            return candidate

    return None


def transcribe_opening(
    audio_path: Path,
    model_name: str,
    language: Optional[str],
    device: str,
    compute_type: str,
    opening_seconds: float,
) -> tuple[list[AudioWord], float, str, str]:
    """
    Transcribe and align the opening section with WhisperX.

    WhisperX versions differ slightly in return types, so this function
    validates the returned data and gives a useful error when incompatible.
    """
    try:
        import whisperx
    except ImportError as exc:
        raise RuntimeError(
            "WhisperX is not installed in this Python environment."
        ) from exc

    audio = whisperx.load_audio(str(audio_path))
    sample_rate = 16000
    full_duration = len(audio) / sample_rate

    opening_samples = int(min(full_duration, opening_seconds) * sample_rate)
    audio_opening = audio[:opening_samples]

    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
    )

    result = model.transcribe(
        audio_opening,
        batch_size=4,
        language=language,
    )

    if not isinstance(result, dict):
        raise TypeError(
            "WhisperX transcription returned "
            f"{type(result).__name__}, expected a dictionary."
        )

    segments = result.get("segments", [])
    detected_language = result.get("language") or language or "en"
    transcription_text = str(result.get("text", "") or "")

    if not transcription_text:
        transcription_text = " ".join(
            str(seg.get("text", ""))
            for seg in segments
            if isinstance(seg, dict)
        ).strip()

    if not segments:
        return [], full_duration, transcription_text, detected_language

    align_model, metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )

    aligned_result = whisperx.align(
        segments,
        align_model,
        metadata,
        audio_opening,
        device,
        return_char_alignments=False,
    )

    # Some versions return (aligned_result, metadata); others return only
    # the aligned result.
    if isinstance(aligned_result, tuple):
        aligned_result = aligned_result[0]

    if not isinstance(aligned_result, dict):
        raise TypeError(
            "WhisperX alignment returned "
            f"{type(aligned_result).__name__}, expected a dictionary."
        )

    audio_words: list[AudioWord] = []

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
                start = float(start)
                end = float(end)
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

    return audio_words, full_duration, transcription_text, detected_language


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


def analyze_chart(
    chart_path: Path,
    args: argparse.Namespace,
) -> Analysis:
    gap_ms, bpm, chart_words = parse_chart(chart_path)
    audio_path = find_audio_for_chart(chart_path)

    analysis = Analysis(
        chart_path=chart_path,
        audio_path=audio_path,
        old_gap_ms=gap_ms,
        bpm=bpm,
        status="Analysis failed",
        chart_words=chart_words,
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
        analysis.reason = "No audio file with the same basename was found."
        return analysis

    # Restrict matching to the opening lyrics. Do not search the entire song
    # for a convenient but unrelated word.
    opening_chart_words = [
        word for word in chart_words
        if word.seconds <= args.chart_opening_seconds
    ]

    if not opening_chart_words:
        analysis.reason = "No chart lyric words in the opening window."
        return analysis

    try:
        audio_words, duration, transcription, detected_language = (
            transcribe_opening(
                audio_path=audio_path,
                model_name=args.model,
                language=args.language,
                device=args.device,
                compute_type=args.compute_type,
                opening_seconds=args.audio_opening_seconds,
            )
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

    if not audio_words:
        analysis.reason = (
            f"No aligned audio word timestamps were returned. "
            f"Detected language: {detected_language}."
        )
        return analysis

    matches = local_alignment(
        opening_chart_words,
        audio_words,
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

    return analysis


def format_ms(value: Optional[float]) -> str:
    if value is None:
        return "Not assessed"
    return f"{value:.0f} ms"


def format_report(results: list[Analysis], args: argparse.Namespace) -> str:
    lines = [
        "UltraStar GAP Analysis Report",
        "=" * 78,
        "",
        "Safety: source audio and chart files were read only.",
        f"Match threshold: {args.match_threshold:.2f}",
        f"Audio opening window: {args.audio_opening_seconds:.1f} seconds",
        f"Chart opening window: {args.chart_opening_seconds:.1f} seconds",
        "",
    ]

    for result in results:
        lines.extend([
            "-" * 78,
            f"Chart: {result.chart_path}",
            f"Audio: {result.audio_path or 'Not found'}",
            f"Status: {result.status}",
            f"Current #GAP: {format_ms(result.old_gap_ms)}",
            f"#BPM: {result.bpm if result.bpm is not None else 'Unknown'}",
            f"Confidence: {result.confidence}",
            f"Reason: {result.reason or 'No additional details'}",
        ])

        if result.transcription:
            lines.extend([
                "",
                "Whisper transcription (opening):",
                result.transcription,
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

        # Never claim that a correction is safe just because a number exists.
        if (
            result.suggested_gap_ms is not None
            and result.correction_ms is not None
            and result.confidence in {"High", "Medium"}
        ):
            lines.extend([
                "",
                f"Suggested #GAP: {result.suggested_gap_ms:.0f}",
                f"Correction: {result.correction_ms:+.0f} ms",
                "Decision: provisional suggestion; review the matches before "
                "changing the chart.",
            ])
        else:
            lines.extend([
                "",
                "Suggested #GAP: WITHHELD because the evidence is insufficient.",
                f"Estimated GAP (diagnostic only): "
                f"{format_ms(result.suggested_gap_ms)}",
                f"Estimated correction (diagnostic only): "
                f"{format_ms(result.correction_ms)}",
            ])

        lines.append("")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate UltraStar #GAP values by matching opening chart lyrics "
            "to WhisperX word timestamps. Source files are never modified."
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
        "--language",
        default="en",
        help="WhisperX language code (default: en).",
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
        default=45.0,
        help="How many seconds of audio to transcribe (default: 45).",
    )
    parser.add_argument(
        "--chart-opening-seconds",
        type=float,
        default=45.0,
        help="How far into chart timing to consider opening lyrics (default: 45).",
    )
    parser.add_argument(
        "--max-correction-ms",
        type=float,
        default=15000.0,
        help="Withhold suggestions exceeding this correction (default: 15000).",
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

    if not source.is_dir():
        print(f"Error: source directory does not exist: {source}", file=sys.stderr)
        return 2

    # Avoid accidentally overwriting a chart or audio file with the report.
    if report_path.suffix.lower() in {
        ".txt", ".m4a", ".mp3", ".wav", ".flac", ".ogg", ".aac", ".opus"
    } and report_path.exists():
        try:
            if report_path.resolve().parent == source:
                print(
                    "Error: report path points to an existing file inside the "
                    "source directory. Choose a separate report filename.",
                    file=sys.stderr,
                )
                return 2
        except OSError:
            pass

    chart_paths = sorted(
        p for p in source.rglob("*.txt")
        if p.is_file() and p.resolve() != report_path
    )

    if not chart_paths:
        print(f"No .txt chart files found under: {source}", file=sys.stderr)
        return 1

    results: list[Analysis] = []

    for index, chart_path in enumerate(chart_paths, start=1):
        print(f"[{index}/{len(chart_paths)}] Analyzing: {chart_path}")
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

        results.append(result)
        print(f"    {result.status}: {result.confidence}")

    report = format_report(results, args)

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        print(f"Could not write report: {exc}", file=sys.stderr)
        return 2

    print()
    print(f"Report written to: {report_path}")
    print("Audio and chart files were not modified.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())