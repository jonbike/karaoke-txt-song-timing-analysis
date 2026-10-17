
#!/usr/bin/env python3
"""Read-only UltraStar GAP analysis using opening lyric alignment."""

from __future__ import annotations

import argparse
import difflib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus"
}

# UltraStar note format: : beat duration pitch lyric
NOTE_RE = re.compile(r"^[:*]\s*(-?\d+)\s+(\d+)\s+(-?\d+)\s+(.*)$")


@dataclass
class ChartWord:
    text: str
    beat: float
    seconds: float


@dataclass
class AudioWord:
    text: str
    start: float
    end: float


@dataclass
class Match:
    chart: ChartWord
    audio: AudioWord
    similarity: float
    offset: float


def normalize(text: str) -> str:
    """Normalize punctuation and UltraStar syllable markers for comparison."""
    text = text.lower().replace("’", "'").replace("~", "")
    return "".join(c for c in text if c.isalnum())


def similarity(a: str, b: str) -> float:
    """Fuzzy spelling comparison; short common words are deliberately strict."""
    a, b = normalize(a), normalize(b)

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0

    score = difflib.SequenceMatcher(None, a, b).ratio()

    # Very short words like "I", "a", and "to" are poor fuzzy matches.
    if min(len(a), len(b)) <= 2:
        return 0.0

    # Help with ASR variants where one word is a plausible longer form.
    if a in b or b in a:
        score = max(score, min(len(a), len(b)) / max(len(a), len(b)))

    return score


def parse_chart(path: Path):
    """Read chart timing and reconstruct words from lyric-note boundaries.

    A trailing space in a note's lyric field marks a word boundary in
    standard UltraStar files. The original lyric field must not be stripped
    before checking that trailing space.
    """
    gap_ms = None
    bpm = None
    relative = False
    warnings = []
    notes = []

    try:
        lines = path.read_text(
            encoding="utf-8-sig", errors="replace"
        ).splitlines()
    except OSError as exc:
        return [], None, None, False, [f"Could not read chart: {exc}"]

    for line in lines:
        if line.startswith("#GAP:"):
            try:
                gap_ms = float(line.split(":", 1)[1].strip())
            except ValueError:
                warnings.append("Invalid #GAP value.")

        elif line.startswith("#BPM:"):
            try:
                bpm = float(line.split(":", 1)[1].strip())
            except ValueError:
                warnings.append("Invalid #BPM value.")

        elif line.upper().startswith("#RELATIVE:"):
            value = line.split(":", 1)[1].strip().upper()
            relative = value not in {"NO", "0", "FALSE", ""}

        else:
            match = NOTE_RE.match(line)
            if match:
                beat = int(match.group(1))
                raw_lyric = match.group(4)
                boundary = raw_lyric.endswith(" ")
                lyric = raw_lyric.strip()

                if lyric:
                    notes.append((beat, lyric, boundary))

    if gap_ms is None:
        warnings.append("Missing #GAP.")
    if bpm is None or bpm <= 0:
        warnings.append("Missing or invalid #BPM.")
    if relative:
        warnings.append("#RELATIVE chart: skipped.")

    words = []

    if gap_ms is not None and bpm is not None and bpm > 0 and not relative:
        seconds_per_beat = 60.0 / bpm
        parts = []
        first_beat = None

        def flush():
            nonlocal parts, first_beat
            if parts and first_beat is not None:
                word = "".join(parts).replace("~", "")
                if normalize(word):
                    words.append(
                        ChartWord(
                            text=word,
                            beat=first_beat,
                            seconds=gap_ms / 1000.0
                            + first_beat * seconds_per_beat,
                        )
                    )
            parts = []
            first_beat = None

        for beat, lyric, boundary in notes:
            if first_beat is None:
                first_beat = float(beat)

            # Tilde is a syllable continuation marker, not a word separator.
            parts.append(lyric.replace("~", ""))

            if boundary:
                flush()

        flush()

    return words, gap_ms, bpm, relative, warnings


def find_audio(chart_path: Path, source: Path):
    candidates = [
        p for p in chart_path.parent.iterdir()
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    ]

    exact = [
        p for p in candidates
        if p.stem.lower() == chart_path.stem.lower()
    ]
    if exact:
        return exact[0]

    if candidates:
        candidates.sort(key=lambda p: p.name.lower())
        return candidates[0]

    for p in source.rglob("*"):
        if (
            p.is_file()
            and p.suffix.lower() in AUDIO_EXTENSIONS
            and p.stem.lower() == chart_path.stem.lower()
        ):
            return p

    return None



def transcribe_opening(whisperx, audio_path, model_name, language,
                       device, compute_type, opening_seconds):
    """Transcribe and align only the beginning of the audio."""
    audio = whisperx.load_audio(str(audio_path))
    sample_rate = 16000
    duration = len(audio) / sample_rate
    audio = audio[:int(min(duration, opening_seconds) * sample_rate)]

    model = whisperx.load_model(
        model_name,
        device,
        compute_type=compute_type,
        language=language,
    )

    result = model.transcribe(
        audio,
        batch_size=4,
        language=language,
    )

    # Some wrappers return a plain string rather than a result dictionary.
    if isinstance(result, str):
        raise TypeError(
            "model.transcribe() returned a string instead of a dictionary: "
            + repr(result[:200])
        )
    if not isinstance(result, dict):
        raise TypeError(
            "Unexpected transcription result type: "
            + type(result).__name__
        )

    detected_language = result.get("language", language)
    segments = result.get("segments", [])

    if not segments:
        return [], duration, result.get("text", ""), detected_language

    align_model, metadata = whisperx.load_align_model(
        language_code=detected_language,
        device=device,
    )

    aligned_result = whisperx.align(
        segments,
        align_model,
        metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    # WhisperX normally returns (aligned_result, audio_metadata), but
    # wrappers or versions may return the result dictionary directly.
    if isinstance(aligned_result, tuple):
        aligned = aligned_result[0]
    else:
        aligned = aligned_result

    if not isinstance(aligned, dict):
        raise TypeError(
            "whisperx.align() returned unexpected type: "
            + type(aligned).__name__
        )

    words = []
    for segment in aligned.get("segments", []):
        if not isinstance(segment, dict):
            continue

        for item in segment.get("words", []):
            if not isinstance(item, dict):
                continue

            text = str(item.get("word", "")).strip()
            start = item.get("start")
            end = item.get("end")

            if text and start is not None and end is not None:
                words.append(
                    AudioWord(text, float(start), float(end))
                )

    transcription_text = result.get("text", "")
    if not transcription_text:
        transcription_text = " ".join(
            str(segment.get("text", ""))
            for segment in segments
            if isinstance(segment, dict)
        ).strip()

    return words, duration, transcription_text, detected_language


def local_alignment(chart_words, audio_words):
    """Find the strongest local, order-preserving sequence of word matches.

    Unlike whole-song matching, this cannot jump to a later audio section:
    the supplied audio words come only from the opening crop.
    """
    chart = chart_words[:100]
    audio = audio_words
    n, m = len(chart), len(audio)

    if not n or not m:
        return []

    gap_penalty = -0.65
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[0] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            sim = similarity(chart[i - 1].text, audio[j - 1].text)

            # Accept imperfect longer words, but avoid fuzzy matches on
            # tiny words that occur everywhere.
            if sim >= 0.50:
                diagonal = dp[i - 1][j - 1] + (2.0 * sim - 0.45)
            else:
                diagonal = dp[i - 1][j - 1] - 0.9

            up = dp[i - 1][j] + gap_penalty
            left = dp[i][j - 1] + gap_penalty
            best = max(0.0, diagonal, up, left)

            dp[i][j] = best
            if best == 0:
                trace[i][j] = 0
            elif best == diagonal:
                trace[i][j] = 1
            elif best == up:
                trace[i][j] = 2
            else:
                trace[i][j] = 3

    # Begin traceback at the strongest local-alignment endpoint.
    best_score = 0.0
    end_i = end_j = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if dp[i][j] > best_score:
                best_score = dp[i][j]
                end_i, end_j = i, j

    matches = []
    i, j = end_i, end_j

    while i > 0 and j > 0 and dp[i][j] > 0:
        direction = trace[i][j]

        if direction == 1:
            sim = similarity(chart[i - 1].text, audio[j - 1].text)

            if sim >= 0.50:
                cw = chart[i - 1]
                aw = audio[j - 1]
                matches.append(
                    Match(cw, aw, sim, aw.start - cw.seconds)
                )

            i -= 1
            j -= 1

        elif direction == 2:
            i -= 1
        elif direction == 3:
            j -= 1
        else:
            break

    matches.reverse()
    return matches


def median(values):
    values = sorted(values)
    if not values:
        return 0.0
    middle = len(values) // 2
    if len(values) % 2:
        return values[middle]
    return (values[middle - 1] + values[middle]) / 2.0


def evaluate(matches, max_correction):
    """Calculate a robust offset and decide whether it is safe to suggest."""
    if not matches:
        return None, [], 0.0, 0.0, 0.0, "No usable ordered matches."

    offsets = [m.offset for m in matches]
    center = median(offsets)
    mad = median([abs(value - center) for value in offsets])
    threshold = max(1.25, 3.0 * mad)

    inliers = [
        match for match in matches
        if abs(match.offset - center) <= threshold
    ]

    if not inliers:
        return center, [], mad, 0.0, 0.0, "No consistent timing matches."

    final_offset = median([m.offset for m in inliers])
    final_mad = median([
        abs(m.offset - final_offset) for m in inliers
    ])
    span = (
        inliers[-1].audio.start - inliers[0].audio.start
        if len(inliers) > 1 else 0.0
    )
    ratio = len(inliers) / len(matches)

    reasons = []
    if len(inliers) < 5:
        reasons.append("fewer than 5 consistent matches")
    if span < 3.0:
        reasons.append("matches cover less than 3 seconds")
    if ratio < 0.60:
        reasons.append("low inlier ratio")
    if final_mad > 1.25:
        reasons.append("inconsistent timing offsets")
    if abs(final_offset) > max_correction:
        reasons.append("correction exceeds plausibility limit")

    if reasons:
        confidence = "Low confidence: " + "; ".join(reasons)
    elif len(inliers) >= 10 and span >= 8.0 and final_mad <= 0.55:
        confidence = "High confidence (heuristic)"
    else:
        confidence = "Medium confidence (heuristic)"

    return final_offset, inliers, final_mad, ratio, span, confidence


def build_report(source, language, model, device, results,
                 opening_seconds, max_correction):
    lines = [
        "ULTRASTAR LYRIC-MATCH GAP ANALYSIS",
        "==================================",
        f"Source: {source}",
        f"Language: {language}",
        f"Model: {model}",
        f"Device: {device}",
        f"Charts found: {len(results)}",
        "",
        "IMPORTANT:",
        "- Original audio and chart files are never modified.",
        "- Only the requested report file is written.",
        "- Corrections are advisory and never auto-applied.",
        f"- Only the first {opening_seconds:.1f} seconds are transcribed.",
        "- Matching is local and order-preserving.",
        "- A single matching word can never produce a correction.",
        "- Confidence labels are heuristics, not probabilities.",
        "- #RELATIVE charts are skipped.",
        "- This script estimates GAP only, not BPM corrections.",
        "- Positive correction means increase #GAP; negative means decrease it.",
        f"- Maximum permitted correction magnitude: {max_correction:.1f} s",
        "",
    ]

    for r in results:
        lines.extend([
            "-" * 72,
            f"Chart: {r['chart']}",
            f"Audio: {r.get('audio') or 'NOT FOUND'}",
        ])

        for warning in r.get("warnings", []):
            lines.append(f"Warning: {warning}")

        lines.append(
            "Chart opening words: "
            + " ".join(w.text for w in r.get("chart_words", [])[:35])
        )

        if "audio_text" in r:
            lines.append("Recognized opening transcription: " + r["audio_text"])
            lines.append(
                f"Recognized aligned word count: {len(r.get('audio_words', []))}"
            )

        if r.get("gap_ms") is not None:
            lines.append(f"Current #GAP: {r['gap_ms']:.1f} ms")
        if r.get("bpm") is not None:
            lines.append(f"#BPM: {r['bpm']:.3f}")
        if r.get("status"):
            lines.append("Status: " + r["status"])

        matches = r.get("matches", [])
        lines.append(f"Ordered lyric matches: {len(matches)}")

        if r.get("offset") is not None:
            lines.append(f"Median offset: {r['offset']:+.3f} s")
            lines.append(f"MAD: {r.get('mad', 0.0):.3f} s")
            lines.append(
                f"Consistent matches: {len(r.get('inliers', []))} / {len(matches)}"
            )
            lines.append(f"Inlier audio span: {r.get('span', 0.0):.1f} s")
            lines.append(f"Inlier ratio: {r.get('ratio', 0.0):.1%}")

        if matches:
            lines.append("Matched words (chart -> audio):")
            for m in matches[:35]:
                lines.append(
                    f"  {m.chart.text} -> {m.audio.text}: "
                    f"chart {m.chart.seconds:.3f}s, "
                    f"audio {m.audio.start:.3f}s, "
                    f"offset {m.offset:+.3f}s, "
                    f"similarity {m.similarity:.2f}"
                )

        if r.get("correction_ms") is not None:
            lines.append(f"Provisional GAP correction: {r['correction_ms']:+.0f} ms")
            lines.append(f"Suggested #GAP: {r['suggested_gap_ms']:.0f} ms")
        else:
            lines.append(
                "Suggested #GAP: WITHHELD because confidence is not sufficient"
            )

        lines.append("Confidence: " + r.get("confidence", "Not assessed"))
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Read-only UltraStar GAP analysis"
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--model", default="medium")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--compute-type", default=None)
    parser.add_argument("--opening-seconds", type=float, default=45.0)
    parser.add_argument("--max-correction", type=float, default=15.0)
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    report_path = Path(args.report).expanduser().resolve()

    if not source.is_dir():
        print(f"Source directory does not exist: {source}", file=sys.stderr)
        return 2
    if args.opening_seconds <= 0 or args.max_correction <= 0:
        print("Opening duration and maximum correction must be positive.", file=sys.stderr)
        return 2

    charts = []
    for p in sorted(source.rglob("*.txt")):
        if p.resolve() == report_path:
            continue
        try:
            first_lines = p.read_text(
                encoding="utf-8-sig", errors="replace"
            ).splitlines()[:300]
        except OSError:
            continue

        if any(
            line.startswith(("#TITLE:", "#ARTIST:", "#BPM:", "#GAP:"))
            or NOTE_RE.match(line)
            for line in first_lines
        ):
            charts.append(p)

    compute_type = args.compute_type or (
        "int8" if args.device == "cpu" else "float16"
    )

    results = []
    whisperx = None

    for chart_path in charts:
        chart_words, gap_ms, bpm, relative, warnings = parse_chart(chart_path)

        item = {
            "chart": chart_path,
            "audio": None,
            "chart_words": chart_words,
            "gap_ms": gap_ms,
            "bpm": bpm,
            "warnings": warnings,
            "matches": [],
            "inliers": [],
            "confidence": "Not assessed",
        }

        if relative:
            item["status"] = "Skipped: #RELATIVE chart."
            results.append(item)
            continue

        if not chart_words:
            item["status"] = "Skipped: no parsed lyric words or invalid timing metadata."
            results.append(item)
            continue

        if gap_ms is None or bpm is None or bpm <= 0:
            item["status"] = "Skipped: missing #GAP or valid #BPM."
            results.append(item)
            continue

        audio_path = find_audio(chart_path, source)
        item["audio"] = audio_path

        if audio_path is None:
            item["status"] = "Skipped: no matching audio file found."
            results.append(item)
            continue

        try:
            if whisperx is None:
                import whisperx as whisperx_module
                whisperx = whisperx_module

            audio_words, duration, audio_text, detected_language = (
                transcribe_opening(
                    whisperx,
                    audio_path,
                    args.model,
                    args.language,
                    args.device,
                    compute_type,
                    args.opening_seconds,
                )
            )

            item["audio_words"] = audio_words
            item["audio_text"] = audio_text
            item["status"] = (
                f"Opening-only transcription "
                f"({min(duration, args.opening_seconds):.1f} s); "
                f"language={detected_language}"
            )

            # Do not compare lyrics far beyond the audio opening.
            latest_beat = (
                args.opening_seconds + args.max_correction
            ) * bpm / 60.0
            chart_prefix = [
                w for w in chart_words if w.beat <= latest_beat
            ][:100]

            matches = local_alignment(chart_prefix, audio_words)
            item["matches"] = matches

            offset, inliers, mad, ratio, span, confidence = evaluate(
                matches, args.max_correction
            )

            item.update({
                "offset": offset,
                "inliers": inliers,
                "mad": mad,
                "ratio": ratio,
                "span": span,
                "confidence": confidence,
            })

            if offset is not None and inliers and not confidence.startswith("Low confidence"):
                item["correction_ms"] = offset * 1000.0
                item["suggested_gap_ms"] = gap_ms + offset * 1000.0
            else:
                item["correction_ms"] = None
                item["suggested_gap_ms"] = None

        except Exception as exc:
            item["status"] = f"Analysis failed: {type(exc).__name__}: {exc}"
            item["confidence"] = "Not assessed"

        results.append(item)

    report = build_report(
        source,
        args.language,
        args.model,
        args.device,
        results,
        args.opening_seconds,
        args.max_correction,
    )

    try:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(report, encoding="utf-8")
    except OSError as exc:
        print(f"Could not write report: {exc}", file=sys.stderr)
        return 2

    print(report)
    print(f"\nReport written to: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())