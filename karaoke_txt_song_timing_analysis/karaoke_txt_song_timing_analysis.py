#!/usr/bin/env python3
"""Read-only UltraStar GAP analysis using ordered lyric alignment.

This script never edits song audio or UltraStar chart files. It writes only
the requested report file. GAP estimates are advisory and require listening
or in-game validation before use.
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
    ".mp3", ".ogg", ".opus", ".wav", ".flac", ".m4a", ".aac", ".wma"
}
META_RE = re.compile(r"^#([A-Z0-9_]+)\s*:\s*(.*)$", re.I)
NOTE_RE = re.compile(r"^[:FRGB*]\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(.*)$")
REST_RE = re.compile(r"^-\s*(-?\d+)")


def clean_word(text):
    """Normalize a lyric/transcript token for comparison."""
    text = unicodedata.normalize("NFKC", text).casefold()
    # Keep letters/numbers and apostrophes inside words.
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text, flags=re.UNICODE)
    text = text.replace("’", "'")
    return text


def split_lyric_words(text):
    return [w for w in (clean_word(x) for x in text.split()) if w]


def parse_chart(path):
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
            words.append({"word": token, "beat": pending_beat})
        pending = ""
        pending_beat = None

    for raw_line in path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines():
        meta = META_RE.match(raw_line.strip())
        if meta:
            key, value = meta.group(1).upper(), meta.group(2).strip()
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

        rest = REST_RE.match(raw_line.strip())
        if rest:
            finish_pending()
            continue

        note = NOTE_RE.match(raw_line)
        if not note:
            continue

        beat = float(note.group(1))
        lyric = note.group(4)
        if pending_beat is None:
            pending_beat = beat

        # UltraStar marks word boundaries with whitespace at the end of a
        # lyric syllable. Preserve that trailing whitespace before stripping.
        for char in lyric:
            if char.isspace():
                finish_pending()
            else:
                pending += char

    finish_pending()
    return gap, bpm, audio_name, words, relative


def find_audio(chart, audio_name):
    folder = chart.parent
    if audio_name:
        named = Path(audio_name)
        candidates = [
            named if named.is_absolute() else folder / named,
            folder / named.name,
        ]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() in AUDIO_EXTENSIONS:
                return candidate
    try:
        for candidate in folder.iterdir():
            if (candidate.is_file()
                    and candidate.suffix.lower() in AUDIO_EXTENSIONS
                    and candidate.stem.casefold() == chart.stem.casefold()):
                return candidate
    except OSError:
        pass
    return None


def transcribe_words(audio_path, language, whisperx, model, device):
    """Return ordered (normalized word, start, end) tuples."""
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=4)
    segments = result.get("segments", [])
    if not segments:
        return [], "No speech recognized"

    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        aligned = whisperx.align(
            segments, align_model, metadata, audio, device,
            return_char_alignments=False,
        )
        segments = aligned.get("segments", [])
    except Exception as exc:
        return [], f"Word alignment failed ({type(exc).__name__})"

    words = []
    for segment in segments:
        for item in segment.get("words", []):
            start = item.get("start")
            end = item.get("end")
            token = clean_word(str(item.get("word", "")))
            if token and start is not None and end is not None:
                try:
                    start, end = float(start), float(end)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(start) and math.isfinite(end) and end >= start:
                    words.append((token, start, end))
    words.sort(key=lambda x: (x[1], x[2]))
    return words, "Word timestamps aligned"


def word_similarity(a, b):
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    # Do not let short/common tokens match arbitrary longer words.
    if min(len(a), len(b)) <= 2:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def ordered_alignment(chart_words, audio_words):
    """Global monotonic DP alignment with gaps and conservative fuzzy matches.

    A match can only connect words in their original order. Exact matches are
    preferred; fuzzy matches are permitted for longer words to handle modest
    ASR errors such as 'antichrist' / 'anticoaster'. Repeated short words do
    not get fuzzy-matched.
    """
    n, m = len(chart_words), len(audio_words)
    if not n or not m:
        return []

    # Similarity scores are intentionally conservative. Gap costs allow ASR
    # insertions/deletions without shifting all subsequent word matches.
    gap_cost = -0.72
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
            sim = word_similarity(cw, aw)
            if cw == aw:
                match_score = 2.8
            elif sim >= 0.78:
                match_score = 2.8 * sim - 0.9
            else:
                match_score = -1.25
            choices = (
                (dp[i - 1][j - 1] + match_score, "diag"),
                (dp[i - 1][j] + gap_cost, "up"),
                (dp[i][j - 1] + gap_cost, "left"),
            )
            best = max(choices, key=lambda x: x[0])
            dp[i][j], trace[i][j] = best

    pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        step = trace[i][j]
        if step == "diag":
            cw = chart_words[i - 1]["word"]
            aw, start, end = audio_words[j - 1]
            sim = word_similarity(cw, aw)
            if cw == aw or sim >= 0.78:
                pairs.append({
                    "chart_index": i - 1,
                    "audio_index": j - 1,
                    "chart_word": cw,
                    "audio_word": aw,
                    "audio_start": start,
                    "audio_end": end,
                    "similarity": 1.0 if cw == aw else sim,
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
    median = statistics.median(values)
    mad = statistics.median(abs(v - median) for v in values)
    return median, mad


def estimate_gap(chart_words, audio_words, bpm, gap_ms):
    """Estimate constant offset only from monotonic, locally consistent matches."""
    pairs = ordered_alignment(chart_words, audio_words)
    if not pairs:
        return None, pairs, [], "Low confidence: no ordered lyric matches"

    beat_seconds = 60.0 / bpm
    observations = []
    for pair in pairs:
        chart_time = gap_ms / 1000.0 + chart_words[pair["chart_index"]]["beat"] * beat_seconds
        offset = pair["audio_start"] - chart_time
        if math.isfinite(offset):
            pair["chart_time"] = chart_time
            pair["offset"] = offset
            observations.append(pair)

    if not observations:
        return None, pairs, [], "Low confidence: no valid timing observations"

    offsets = [p["offset"] for p in observations]
    med, mad = median_mad(offsets)
    # Keep matches near the median offset. Floor avoids rejecting every match
    # when the MAD is near zero; cap prevents broad, obviously wrong alignments.
    threshold = max(0.45, 3.5 * 1.4826 * mad)
    threshold = min(threshold, 2.5)
    inliers = [p for p in observations if abs(p["offset"] - med) <= threshold]

    # Reject a result that is dominated by one repeated phrase or a small
    # number of matches. Report all observations for transparency.
    span = 0.0
    if len(inliers) >= 2:
        span = max(p["audio_start"] for p in inliers) - min(
            p["audio_start"] for p in inliers
        )

    if abs(med) > 5.0:
        confidence = "Low confidence: median offset exceeds 5 seconds; likely bad alignment"
    elif len(inliers) < 5:
        confidence = "Low confidence: fewer than 5 consistent matches"
    elif span < 5:
        confidence = "Low confidence: matches cover less than 5 seconds"
    elif len(inliers) / max(1, len(observations)) < 0.35:
        confidence = "Low confidence: too many timing outliers"
    elif mad > 0.8:
        confidence = "Low confidence: offsets are highly inconsistent"
    elif len(inliers) >= 12 and span >= 20 and mad <= 0.25:
        confidence = "Moderate confidence: many consistent ordered matches; validate manually"
    else:
        confidence = "Low-to-moderate confidence: validate matches and listen in-game"

    # Positive correction means increase GAP to delay the chart. If audio is
    # earlier than chart (negative offset), reduce GAP.
    correction_ms = med * 1000.0
    stats = {
        "median_offset": med,
        "mad": mad,
        "threshold": threshold,
        "inliers": inliers,
        "span": span,
        "observations": observations,
    }
    return correction_ms, pairs, stats, confidence


def main():
    parser = argparse.ArgumentParser(
        description="Read-only UltraStar lyric-match GAP analysis."
    )
    parser.add_argument("--source", default="~/Music")
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument("--report", default="~/Music/ultrastar_gap_report.txt")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--model", default="medium")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    report = Path(args.report).expanduser().resolve()
    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    try:
        import whisperx
    except ImportError:
        raise SystemExit(
            "WhisperX missing. Activate your Poetry environment and install it:\n"
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

    model = whisperx.load_model(
        args.model, args.device, language=args.language, compute_type="int8"
    )

    output = [
        "ULTRASTAR LYRIC-MATCH GAP ANALYSIS",
        "==================================",
        f"Source: {source}",
        f"Language: {args.language}",
        f"Charts found: {len(charts)}",
        "",
        "IMPORTANT:",
        "- Read-only: original audio and chart files are never modified.",
        "- Only the report file is written.",
        "- Previous report's -135731 ms estimate is INVALID; do not apply it.",
        "- Matches are monotonic and order-preserving, but still require review.",
        "- Confidence labels are heuristic, not probabilities.",
        "- Instrumental intros, backing vocals and transcription errors can mislead.",
        "- #RELATIVE charts are skipped.",
        "- This version estimates GAP only; it does not estimate BPM correction.",
        "- Positive correction means increasing #GAP to delay the chart.",
        "",
    ]

    for chart, data in charts:
        gap, bpm, audio_name, chart_words, relative = data
        audio_path = find_audio(chart, audio_name)
        output.extend(["-" * 72, f"Chart: {chart}", f"Audio: {audio_path or 'NOT FOUND'}"])
        output.append("Chart opening words: " + " ".join(w["word"] for w in chart_words[:28]))
        output.append(f"Current #GAP: {gap if gap is not None else 'MISSING'} ms")
        output.append(f"#BPM: {bpm if bpm is not None else 'MISSING'}")

        if relative:
            output.extend(["Status: #RELATIVE chart; skipped", ""])
            continue
        if gap is None or bpm is None or bpm <= 0 or not chart_words:
            output.extend(["Status: missing/invalid GAP, BPM, or chart lyrics; skipped", ""])
            continue
        if audio_path is None:
            output.extend(["Status: audio file not found; skipped", ""])
            continue

        try:
            audio_words, transcript_status = transcribe_words(
                audio_path, args.language, whisperx, model, args.device
            )
        except Exception as exc:
            output.extend([f"Status: transcription failed ({type(exc).__name__}: {exc})", ""])
            continue

        output.append("Recognized opening words: " + " ".join(w[0] for w in audio_words[:28]))
        output.append(f"Recognized aligned word count: {len(audio_words)}")
        output.append(f"Transcription: {transcript_status}")

        # Limit dynamic-programming memory/time on very long songs while retaining
        # broad coverage. Sample neither sequence; use a reasonable cap and warn.
        chart_for_match = chart_words
        audio_for_match = audio_words
        if len(chart_for_match) > 1600 or len(audio_for_match) > 2400:
            output.append("WARNING: long sequence capped for alignment; matches may be incomplete")
            chart_for_match = chart_for_match[:1600]
            audio_for_match = audio_for_match[:2400]

        correction, pairs, stats, confidence = estimate_gap(
            chart_for_match, audio_for_match, bpm, gap
        )
        output.append(f"Ordered lyric matches: {len(pairs)}")
        if stats:
            inliers = stats["inliers"]
            output.append(f"Consistent timing matches: {len(inliers)} / {len(stats['observations'])}")
            output.append(f"Median offset: {stats['median_offset']:+.3f} s")
            output.append(f"Median absolute deviation (MAD): {stats['mad']:.3f} s")
            output.append(f"Inlier audio span: {stats['span']:.1f} s")
            output.append("Matched words (chart -> audio):")
            # Show a useful sample spread across the matched song, not just the opening.
            display = inliers
            if len(display) > 30:
                indices = sorted({round(i * (len(display) - 1) / 29) for i in range(30)})
                display = [display[i] for i in indices]
            for p in display:
                output.append(
                    f"  {p['chart_word']} -> {p['audio_word']}: "
                    f"chart {p['chart_time']:.3f}s, audio {p['audio_start']:.3f}s, "
                    f"offset {p['offset']:+.3f}s"
                )
        else:
            output.append("Matched words: none reliable enough to estimate timing")

        if correction is not None:
            output.append(f"Provisional GAP correction: {correction:+.0f} ms")
            if confidence.startswith("Moderate confidence"):
                suggested = gap + correction
                output.append(
                    f"Suggested #GAP: {suggested:.0f} ms "
                    "(estimate only; validate manually, never auto-applied)"
                )
            else:
                output.append(
                    "Suggested #GAP: WITHHELD because confidence is not sufficient"
                )
        else:
            output.append("Provisional GAP correction: unavailable")
            output.append("Suggested #GAP: unavailable")

        output.append(f"Confidence: {confidence}")
        output.append("")

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(output) + "\n", encoding="utf-8")
    print(f"Report saved to: {report}")
    print("Original audio and chart files were not modified.")


if __name__ == "__main__":
    main()