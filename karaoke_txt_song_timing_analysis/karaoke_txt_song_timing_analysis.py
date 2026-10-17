
#!/usr/bin/env python3
"""
UltraStar Deluxe GAP analysis using ordered lyric alignment.

Read-only with respect to UltraStar charts and audio:
    - Never modifies original chart or audio files.
    - Writes only the requested report.
    - Never automatically applies a GAP correction.

Strategy:
    1. Transcribe the opening audio segment first.
    2. Search overlapping audio windows for the chart's opening lyrics.
    3. Estimate the offset using robust statistics.
    4. If the opening is insufficient, transcribe the full song and retry.
    5. Reject unreliable estimates and only suggest a GAP when confidence
       is moderate.

Recommended for CPU-only systems:
    poetry run python ultrastar_gap.py --language en --model medium

For faster processing:
    poetry run python ultrastar_gap.py --language en --model small
"""

import argparse
import math
import re
import statistics
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

LANGUAGES = ("en", "de", "it", "ko")

AUDIO_EXTENSIONS = {
    ".mp3", ".ogg", ".opus", ".wav", ".flac",
    ".m4a", ".aac", ".wma",
}

SAMPLE_RATE = 16000

META_RE = re.compile(r"^#([A-Z0-9_]+)\s*:\s*(.*)$", re.I)
NOTE_RE = re.compile(r"^[:FRGB*]\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(.*)$")
REST_RE = re.compile(r"^-\s*(-?\d+)")


def clean_word(text):
    """Normalize a lyric/transcript token for comparison."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = text.replace("\u2019", "'")
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text, flags=re.UNICODE)
    return text


def parse_chart(path):
    """Read GAP, BPM, audio filename, relative mode and ordered lyric words."""
    gap = bpm = None
    audio_name = None
    relative = False
    words = []
    pending = ""
    pending_beat = None

    def finish_pending():
        nonlocal pending, pending_beat

        token = clean_word(pending)
        if token and pending_beat is not None:
            words.append({
                "word": token,
                "beat": pending_beat,
            })

        pending = ""
        pending_beat = None

    for raw_line in path.read_text(
        encoding="utf-8-sig",
        errors="replace",
    ).splitlines():

        line = raw_line.strip()

        meta = META_RE.match(line)
        if meta:
            key = meta.group(1).upper()
            value = meta.group(2).strip()

            try:
                if key == "GAP":
                    gap = float(value.replace(",", "."))
                elif key == "BPM":
                    bpm = float(value.replace(",", "."))
            except ValueError:
                pass

            if key in ("AUDIO", "MP3"):
                audio_name = value
            elif key == "RELATIVE":
                relative = value.casefold() in ("yes", "true", "1")

            continue

        if REST_RE.match(line):
            finish_pending()
            continue

        note = NOTE_RE.match(line)
        if not note:
            continue

        beat = float(note.group(1))
        lyric = note.group(4)

        if pending_beat is None:
            pending_beat = beat

        # Preserve whitespace before deciding whether a word has ended.
        for char in lyric:
            if char.isspace():
                finish_pending()
            else:
                pending += char

    finish_pending()

    return gap, bpm, audio_name, words, relative


def find_audio(chart, audio_name):
    """Locate the chart's audio without modifying any files."""
    folder = chart.parent

    if audio_name:
        named = Path(audio_name)
        candidates = [
            named if named.is_absolute() else folder / named,
            folder / named.name,
        ]

        for candidate in candidates:
            if (
                candidate.is_file()
                and candidate.suffix.lower() in AUDIO_EXTENSIONS
            ):
                return candidate

    try:
        for candidate in folder.iterdir():
            if (
                candidate.is_file()
                and candidate.suffix.lower() in AUDIO_EXTENSIONS
                and candidate.stem.casefold() == chart.stem.casefold()
            ):
                return candidate
    except OSError:
        pass

    return None


def transcribe_words(
    whisperx,
    audio,
    model,
    align_model,
    align_metadata,
    device,
    start_seconds=0.0,
    end_seconds=None,
):
    """
    Transcribe and align a portion of a preloaded 16 kHz audio array.

    Returned timestamps are absolute seconds in the original audio, even
    when only a cropped segment was processed.
    """
    duration = len(audio) / SAMPLE_RATE

    start_seconds = max(0.0, min(start_seconds, duration))

    if end_seconds is None:
        end_seconds = duration

    end_seconds = max(start_seconds, min(end_seconds, duration))

    start_sample = int(start_seconds * SAMPLE_RATE)
    end_sample = int(end_seconds * SAMPLE_RATE)

    audio_crop = audio[start_sample:end_sample]

    if len(audio_crop) == 0:
        return [], "Empty audio segment"

    result = model.transcribe(
        audio_crop,
        batch_size=4,
    )

    segments = result.get("segments", [])

    if not segments:
        return [], "No speech recognized"

    try:
        aligned = whisperx.align(
            segments,
            align_model,
            align_metadata,
            audio_crop,
            device,
            return_char_alignments=False,
        )
        segments = aligned.get("segments", [])
    except Exception as exc:
        return [], f"Word alignment failed ({type(exc).__name__})"

    words = []

    for segment in segments:
        for item in segment.get("words", []):
            token = clean_word(str(item.get("word", "")))
            start = item.get("start")
            end = item.get("end")

            if not token or start is None or end is None:
                continue

            try:
                start = float(start) + start_seconds
                end = float(end) + start_seconds
            except (TypeError, ValueError):
                continue

            if (
                math.isfinite(start)
                and math.isfinite(end)
                and end >= start
            ):
                words.append((token, start, end))

    words.sort(key=lambda item: (item[1], item[2]))

    return words, "Word timestamps aligned"


def word_similarity(a, b):
    """Conservative similarity for transcript and chart words."""
    if a == b:
        return 1.0

    if not a or not b:
        return 0.0

    # Short/common words are too ambiguous for fuzzy matching.
    if min(len(a), len(b)) <= 2:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def ordered_alignment(
    chart_words,
    audio_words,
    fuzzy_threshold=0.76,
):
    """
    Globally align two short word sequences while preserving order.

    A mismatch is more expensive than deleting both words, so unrelated
    words do not unnecessarily block subsequent valid matches.
    """
    n = len(chart_words)
    m = len(audio_words)

    if not n or not m:
        return []

    gap_cost = -0.65
    mismatch_score = -1.8

    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    trace = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i * gap_cost
        trace[i][0] = "up"

    for j in range(1, m + 1):
        dp[0][j] = j * gap_cost
        trace[0][j] = "left"

    for i in range(1, n + 1):
        cw = chart_words[i - 1]["word"]

        for j in range(1, m + 1):
            aw = audio_words[j - 1][0]
            similarity = word_similarity(cw, aw)

            if cw == aw:
                match_score = 3.0
            elif similarity >= fuzzy_threshold:
                match_score = 3.0 * similarity - 1.0
            else:
                match_score = mismatch_score

            choices = (
                (dp[i - 1][j - 1] + match_score, "diag"),
                (dp[i - 1][j] + gap_cost, "up"),
                (dp[i][j - 1] + gap_cost, "left"),
            )

            dp[i][j], trace[i][j] = max(
                choices,
                key=lambda choice: choice[0],
            )

    pairs = []
    i, j = n, m

    while i > 0 or j > 0:
        step = trace[i][j]

        if step == "diag":
            cw = chart_words[i - 1]["word"]
            aw, start, end = audio_words[j - 1]
            similarity = word_similarity(cw, aw)

            if cw == aw or similarity >= fuzzy_threshold:
                pairs.append({
                    "chart_index": i - 1,
                    "audio_index": j - 1,
                    "chart_word": cw,
                    "audio_word": aw,
                    "audio_start": start,
                    "audio_end": end,
                    "similarity": (
                        1.0 if cw == aw else similarity
                    ),
                })

            i -= 1
            j -= 1

        elif step == "up":
            i -= 1

        elif step == "left":
            j -= 1

        else:
            break

    pairs.reverse()
    return pairs


def median_mad(values):
    """Return median and median absolute deviation."""
    median = statistics.median(values)
    mad = statistics.median(
        abs(value - median) for value in values
    )
    return median, mad


def analyse_window(
    chart_words,
    audio_words,
    bpm,
    gap_ms,
    fuzzy_threshold,
):
    """
    Estimate the timing correction for one audio window.

    Positive correction means increase #GAP.
    Negative correction means decrease #GAP.
    """
    pairs = ordered_alignment(
        chart_words,
        audio_words,
        fuzzy_threshold=fuzzy_threshold,
    )

    if not pairs:
        return None

    beat_seconds = 60.0 / bpm
    observations = []

    for pair in pairs:
        chart_time = (
            gap_ms / 1000.0
            + chart_words[pair["chart_index"]]["beat"] * beat_seconds
        )

        offset = pair["audio_start"] - chart_time

        if not math.isfinite(offset):
            continue

        pair["chart_time"] = chart_time
        pair["offset"] = offset
        observations.append(pair)

    if not observations:
        return None

    offsets = [pair["offset"] for pair in observations]
    median_offset, mad = median_mad(offsets)

    # Robust outlier rejection. The minimum accommodates small timing
    # differences; the cap prevents extremely broad acceptance.
    threshold = max(0.35, 3.5 * 1.4826 * mad)
    threshold = min(threshold, 1.5)

    inliers = [
        pair
        for pair in observations
        if abs(pair["offset"] - median_offset) <= threshold
    ]

    span = 0.0

    if len(inliers) >= 2:
        span = (
            max(pair["audio_start"] for pair in inliers)
            - min(pair["audio_start"] for pair in inliers)
        )

    ratio = len(inliers) / max(1, len(observations))

    return {
        "pairs": pairs,
        "observations": observations,
        "inliers": inliers,
        "median_offset": median_offset,
        "mad": mad,
        "threshold": threshold,
        "span": span,
        "ratio": ratio,
    }


def find_best_window(
    chart_words,
    audio_words,
    bpm,
    gap_ms,
    window_seconds,
    step_seconds,
    max_correction_seconds,
    fuzzy_threshold,
):
    """
    Search overlapping audio windows for the chart's opening lyrics.

    Only the chart's opening words are considered. Later chart verses cannot
    take over the alignment simply because they contain common words.
    """
    if not chart_words or not audio_words:
        return None

    latest_time = max(word[1] for word in audio_words)

    starts = [0.0]
    start = step_seconds

    while start <= latest_time:
        starts.append(start)
        start += step_seconds

    candidates = []

    for window_start in starts:
        window_end = window_start + window_seconds

        window_words = [
            word for word in audio_words
            if window_start <= word[1] < window_end
        ]

        if len(window_words) < 4:
            continue

        stats = analyse_window(
            chart_words,
            window_words,
            bpm,
            gap_ms,
            fuzzy_threshold,
        )

        if stats is None:
            continue

        stats["window_start"] = window_start
        stats["window_end"] = window_end
        stats["window_words"] = window_words

        # The correction must be physically plausible. This protects against
        # matching a repeated phrase minutes away from the real opening.
        plausible = (
            abs(stats["median_offset"]) <= max_correction_seconds
        )

        stats["plausible"] = plausible

        # Prefer reliable matches, broad timing coverage, and low dispersion.
        # Plausible corrections outrank implausible ones.
        rank = (
            int(plausible),
            len(stats["inliers"]),
            stats["span"],
            stats["ratio"],
            -stats["mad"],
            -abs(stats["median_offset"]),
        )

        candidates.append((rank, stats))

    if not candidates:
        return None

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    return candidates[0][1]


def confidence_label(stats, max_correction_seconds):
    """Conservative heuristic; not a statistical probability."""
    if stats is None:
        return "Low confidence: no usable alignment"

    offset = stats["median_offset"]
    inliers = stats["inliers"]
    span = stats["span"]
    mad = stats["mad"]
    ratio = stats["ratio"]

    if abs(offset) > max_correction_seconds:
        return (
            "Low confidence: correction exceeds configured "
            "plausibility limit"
        )

    if len(inliers) < 5:
        return "Low confidence: fewer than 5 consistent matches"

    if span < 5:
        return "Low confidence: matches cover less than 5 seconds"

    if ratio < 0.35:
        return "Low confidence: too many timing outliers"

    if mad > 0.8:
        return "Low confidence: offsets are highly inconsistent"

    if (
        len(inliers) >= 8
        and span >= 8
        and mad <= 0.25
        and ratio >= 0.60
    ):
        return (
            "Moderate confidence: consistent ordered matches; "
            "validate manually"
        )

    return (
        "Low-to-moderate confidence: review matches and "
        "validate in-game"
    )


def format_stats(stats, chart_words):
    """Format the selected alignment for the report."""
    lines = []

    if stats is None:
        lines.append("Matched words: none reliable enough to estimate timing")
        return lines

    inliers = stats["inliers"]

    lines.extend([
        f"Search window: {stats['window_start']:.1f}-"
        f"{stats['window_end']:.1f} s",
        f"Ordered lyric matches: {len(stats['pairs'])}",
        f"Consistent timing matches: {len(inliers)} / "
        f"{len(stats['observations'])}",
        f"Median offset: {stats['median_offset']:+.3f} s",
        f"Median absolute deviation (MAD): {stats['mad']:.3f} s",
        f"Inlier audio span: {stats['span']:.1f} s",
        f"Inlier ratio: {stats['ratio']:.1%}",
        "Matched words (chart -> audio):",
    ])

    display = inliers

    if len(display) > 30:
        indices = sorted({
            round(i * (len(display) - 1) / 29)
            for i in range(30)
        })
        display = [display[i] for i in indices]

    for pair in display:
        lines.append(
            f"  {pair['chart_word']} -> {pair['audio_word']}: "
            f"chart {pair['chart_time']:.3f}s, "
            f"audio {pair['audio_start']:.3f}s, "
            f"offset {pair['offset']:+.3f}s"
        )

    return lines


def main():
    parser = argparse.ArgumentParser(
        description="Read-only UltraStar GAP analysis using lyric alignment."
    )

    parser.add_argument(
        "--source",
        default="~/Music",
        help="Root directory containing UltraStar charts (default: ~/Music)",
    )
    parser.add_argument(
        "--language",
        choices=LANGUAGES,
        required=True,
        help="Language of the vocals",
    )
    parser.add_argument(
        "--report",
        default="~/Music/ultrastar_gap_report.txt",
        help="Output report path",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="WhisperX device (default: cpu)",
    )
    parser.add_argument(
        "--model",
        default="medium",
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--opening-seconds",
        type=float,
        default=45.0,
        help="Duration of initial transcription in seconds (default: 45)",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=30.0,
        help="Duration of each alignment window (default: 30)",
    )
    parser.add_argument(
        "--window-step",
        type=float,
        default=15.0,
        help="Step between overlapping windows (default: 15)",
    )
    parser.add_argument(
        "--chart-words",
        type=int,
        default=100,
        help="Maximum opening chart words to align (default: 100)",
    )
    parser.add_argument(
        "--max-correction-seconds",
        type=float,
        default=15.0,
        help="Reject corrections larger than this absolute value (default: 15)",
    )
    parser.add_argument(
        "--fuzzy-threshold",
        type=float,
        default=0.76,
        help="Minimum fuzzy word similarity (default: 0.76)",
    )

    args = parser.parse_args()

    if (
        args.opening_seconds <= 0
        or args.window_seconds <= 0
        or args.window_step <= 0
        or args.chart_words < 5
        or args.max_correction_seconds <= 0
        or not 0.5 <= args.fuzzy_threshold <= 1.0
    ):
        parser.error("Invalid timing, word-count, or similarity setting")

    source = Path(args.source).expanduser().resolve()
    report = Path(args.report).expanduser().resolve()

    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    try:
        import whisperx
    except ImportError:
        raise SystemExit(
            "WhisperX is missing. Activate your Poetry environment and run:\n"
            "poetry run pip install whisperx"
        )

    charts = []

    for path in sorted(source.rglob("*.txt")):
        if path.resolve() == report:
            continue

        try:
            data = parse_chart(path)
        except OSError:
            continue

        gap, bpm, audio_name, chart_words, relative = data

        if gap is not None or bpm is not None or chart_words:
            charts.append((path, data))

    if not charts:
        raise SystemExit(f"No likely UltraStar charts found in {source}")

    compute_type = "int8" if args.device == "cpu" else "float16"

    print(f"Loading WhisperX model: {args.model} ({args.device})")

    model = whisperx.load_model(
        args.model,
        args.device,
        language=args.language,
        compute_type=compute_type,
    )

    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=args.language,
            device=args.device,
        )
    except Exception as exc:
        raise SystemExit(
            f"Could not load word-alignment model: {type(exc).__name__}: {exc}"
        )

    output = [
        "ULTRASTAR LYRIC-MATCH GAP ANALYSIS",
        "==================================",
        f"Source: {source}",
        f"Language: {args.language}",
        f"Model: {args.model}",
        f"Device: {args.device}",
        f"Charts found: {len(charts)}",
        "",
        "IMPORTANT:",
        "- Original audio and chart files are never modified.",
        "- Only the requested report file is written.",
        "- GAP corrections are advisory and never auto-applied.",
        "- The opening is transcribed first; full-song transcription is a fallback.",
        "- Overlapping windows search for the opening chart lyrics.",
        "- Order-preserving matching and robust statistics reject outliers.",
        "- Confidence labels are heuristics, not probabilities.",
        "- Distorted vocals, backing vocals, long intros and BPM errors can mislead.",
        "- #RELATIVE charts are skipped.",
        "- This script estimates GAP only, not BPM corrections.",
        "- Positive correction means increase #GAP; negative means decrease it.",
        f"- Maximum permitted correction magnitude: "
        f"{args.max_correction_seconds:.1f} s",
        "",
    ]

    for chart, data in charts:
        gap, bpm, audio_name, chart_words, relative = data
        audio_path = find_audio(chart, audio_name)

        output.extend([
            "-" * 72,
            f"Chart: {chart}",
            f"Audio: {audio_path or 'NOT FOUND'}",
            "Chart opening words: " + " ".join(
                word["word"] for word in chart_words[:28]
            ),
            f"Current #GAP: {gap if gap is not None else 'MISSING'} ms",
            f"#BPM: {bpm if bpm is not None else 'MISSING'}",
        ])

        if relative:
            output.extend(["Status: #RELATIVE chart; skipped", ""])
            continue

        if (
            gap is None
            or bpm is None
            or bpm <= 0
            or not chart_words
        ):
            output.extend([
                "Status: missing/invalid GAP, BPM, or chart lyrics; skipped",
                "",
            ])
            continue

        if audio_path is None:
            output.extend(["Status: audio file not found; skipped", ""])
            continue

        try:
            audio = whisperx.load_audio(str(audio_path))
        except Exception as exc:
            output.extend([
                f"Status: audio loading failed ({type(exc).__name__}: {exc})",
                "",
            ])
            continue

        duration = len(audio) / SAMPLE_RATE

        if duration <= 0:
            output.extend(["Status: empty audio; skipped", ""])
            continue

        chart_prefix = chart_words[:args.chart_words]

        # Stage 1: transcribe only the opening portion.
        opening_end = min(args.opening_seconds, duration)

        print(f"Analysing opening: {chart.stem}")

        try:
            opening_words, opening_status = transcribe_words(
                whisperx,
                audio,
                model,
                align_model,
                align_metadata,
                args.device,
                start_seconds=0.0,
                end_seconds=opening_end,
            )
        except Exception as exc:
            output.extend([
                f"Status: opening transcription failed "
                f"({type(exc).__name__}: {exc})",
                "",
            ])
            continue

        stats = find_best_window(
            chart_prefix,
            opening_words,
            bpm,
            gap,
            args.window_seconds,
            args.window_step,
            args.max_correction_seconds,
            args.fuzzy_threshold,
        )

        # Require enough timing evidence before avoiding the fallback.
        opening_usable = (
            stats is not None
            and stats["plausible"]
            and len(stats["inliers"]) >= 5
            and stats["span"] >= 5.0
        )

        transcript_status = (
            f"Opening-only transcription ({opening_end:.1f} s); "
            f"{opening_status}"
        )
        audio_words = opening_words

        # Stage 2: if the opening did not provide enough evidence, process
        # the whole song and search its overlapping windows.
        if not opening_usable and duration > opening_end:
            print(f"Opening inconclusive; analysing full song: {chart.stem}")

            try:
                full_words, full_status = transcribe_words(
                    whisperx,
                    audio,
                    model,
                    align_model,
                    align_metadata,
                    args.device,
                    start_seconds=0.0,
                    end_seconds=duration,
                )
            except Exception as exc:
                output.extend([
                    f"Status: full-song transcription failed "
                    f"({type(exc).__name__}: {exc})",
                    "",
                ])
                continue

            if full_words:
                audio_words = full_words
                transcript_status = (
                    f"Full-song fallback ({duration:.1f} s); {full_status}"
                )

                stats = find_best_window(
                    chart_prefix,
                    full_words,
                    bpm,
                    gap,
                    args.window_seconds,
                    args.window_step,
                    args.max_correction_seconds,
                    args.fuzzy_threshold,
                )

        output.append(
            "Recognized opening words: "
            + " ".join(word[0] for word in audio_words[:28])
        )
        output.append(f"Recognized aligned word count: {len(audio_words)}")
        output.append(f"Transcription: {transcript_status}")

        if stats is None:
            output.extend([
                "Matched words: none reliable enough to estimate timing",
                "Provisional GAP correction: unavailable",
                "Suggested #GAP: unavailable",
                "Confidence: Low confidence: no usable alignment",
                "",
            ])
            continue

        output.extend(format_stats(stats, chart_prefix))

        correction_ms = stats["median_offset"] * 1000.0
        confidence = confidence_label(
            stats,
            args.max_correction_seconds,
        )

        output.append(f"Provisional GAP correction: {correction_ms:+.0f} ms")

        if confidence.startswith("Moderate confidence"):
            suggested_gap = gap + correction_ms
            output.append(
                f"Suggested #GAP: {suggested_gap:.0f} ms "
                "(estimate only; validate manually, never auto-applied)"
            )
        else:
            output.append(
                "Suggested #GAP: WITHHELD because confidence "
                "is not sufficient"
            )

        output.append(f"Confidence: {confidence}")
        output.append("")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(output) + "\n", encoding="utf-8")

    print(f"Report saved to: {report}")
    print("Original audio and chart files were not modified.")


if __name__ == "__main__":
    main()