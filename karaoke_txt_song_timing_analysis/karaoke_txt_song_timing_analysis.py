
#!/usr/bin/env python3
"""Read-only UltraStar GAP analysis using lyric-to-audio word matching."""

import argparse
import re
import statistics
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

LANGUAGES = ("en", "de", "it", "ko")
AUDIO_EXTENSIONS = {
    ".mp3", ".ogg", ".opus", ".wav", ".flac", ".m4a", ".aac", ".wma"
}

META_RE = re.compile(r"^#([A-Z0-9_]+)\s*:\s*(.*)$", re.I)

# Preserve trailing lyric whitespace: in UltraStar, it marks word boundaries.
NOTE_RE = re.compile(
    r"^[:FRGB-]\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(.*)$"
)

WORD_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize_word(text):
    """Normalize a word for matching, ignoring punctuation and case."""
    return "".join(WORD_RE.findall(text.casefold()))


def parse_chart(path):
    gap = bpm = first_beat = None
    audio_name = None
    relative = False
    first_lyric = ""

    # Each entry is (beat, complete_word).
    lyric_words = []
    pending_word = ""
    pending_beat = None

    def finish_word():
        nonlocal pending_word, pending_beat

        word = normalize_word(pending_word)
        if word and pending_beat is not None:
            lyric_words.append((pending_beat, word))

        pending_word = ""
        pending_beat = None

    lines = path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines()

    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped:
            continue

        meta = META_RE.match(stripped)
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

        # Match the unstripped line so trailing spaces remain available.
        note = NOTE_RE.match(raw_line)
        if not note:
            continue

        beat = float(note.group(1))
        lyric = note.group(4)

        if first_beat is None:
            first_beat = beat
            first_lyric = lyric.strip()[:100]

        # Assemble syllables into words. A space ends a word; a pipe
        # marks a phrase boundary. Tilde denotes a held note, not a word.
        for char in lyric:
            if char == "~":
                continue
            if char == "|" or char.isspace():
                finish_word()
            else:
                if pending_beat is None:
                    pending_beat = beat
                pending_word += char

    finish_word()

    return (
        gap, bpm, audio_name, first_beat, first_lyric,
        relative, lyric_words,
    )


def find_audio(chart, audio_name):
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


def recognize_words(
    audio_path, language, whisperx, model, device,
    align_model, align_metadata,
):
    """Return recognized words with aligned audio start times."""
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=4)

    segments = result.get("segments", [])
    if not segments:
        return [], "No speech recognized"

    if align_model is None:
        return [], "Word alignment model unavailable"

    aligned = whisperx.align(
        segments,
        align_model,
        align_metadata,
        audio,
        device,
        return_char_alignments=False,
    )

    words = []
    for segment in aligned.get("segments", []):
        for word in segment.get("words", []):
            if word.get("start") is None:
                continue

            normalized = normalize_word(str(word.get("word", "")))
            if not normalized:
                continue

            words.append({
                "word": normalized,
                "start": float(word["start"]),
            })

    words.sort(key=lambda item: item["start"])

    if not words:
        return [], "No aligned word timestamps"

    return words, "Word alignment completed"


def match_opening_lyrics(chart_words, audio_words):
    """
    Match an opening window of chart lyrics to recognized words.

    Multiword matching blocks are preferred. A unique, distinctive
    single-word match is permitted but receives lower confidence.
    """
    # Limit matching to the opening section. This reduces the chance
    # of matching a repeated chorus or unrelated later lyric by mistake.
    chart_window = chart_words[:100]
    audio_window = audio_words[:160]

    chart_tokens = [word for _, word in chart_window]
    audio_tokens = [word["word"] for word in audio_window]

    if not chart_tokens or not audio_tokens:
        return [], "No words available for matching"

    matcher = SequenceMatcher(
        None, chart_tokens, audio_tokens, autojunk=False
    )

    chart_counts = Counter(chart_tokens)
    audio_counts = Counter(audio_tokens)
    matches = []

    for block in matcher.get_matching_blocks():
        if block.size == 0:
            continue

        # Prefer a sequence of at least two matching words.
        # Single-word matches are accepted only if distinctive and unique.
        if block.size == 1:
            chart_word = chart_tokens[block.a]
            if (
                len(chart_word) < 5
                or chart_counts[chart_word] != 1
                or audio_counts[chart_word] != 1
            ):
                continue

        for offset in range(block.size):
            ci = block.a + offset
            ai = block.b + offset

            beat, lyric_word = chart_window[ci]
            spoken = audio_window[ai]

            matches.append({
                "lyric": lyric_word,
                "audio": spoken["word"],
                "beat": beat,
                "audio_time": spoken["start"],
                "chart_index": ci,
                "audio_index": ai,
                "block_size": block.size,
            })

    matches.sort(key=lambda item: item["chart_index"])
    return matches, "Opening lyric match completed"


def estimate_gap(chart_words, audio_words, gap, bpm):
    """Estimate the GAP correction from matched word-start times."""
    matches, match_note = match_opening_lyrics(
        chart_words, audio_words
    )

    if not matches:
        return None, None, matches, "No reliable lyric sequence match"

    offsets = []
    for match in matches:
        expected_time = (
            gap / 1000.0
            + match["beat"] * 60.0 / bpm
        )
        match["expected_time"] = expected_time
        match["offset"] = match["audio_time"] - expected_time
        offsets.append(match["offset"])

    # Robustly reject timing outliers before taking the median.
    center = statistics.median(offsets)
    deviations = [abs(value - center) for value in offsets]
    mad = statistics.median(deviations)

    # At least 0.35 seconds of tolerance; widen it if the matches
    # naturally have more variation.
    tolerance = max(0.35, 3.0 * mad)
    inliers = [
        match for match in matches
        if abs(match["offset"] - center) <= tolerance
    ]

    if not inliers:
        return None, None, matches, "Matched timings were inconsistent"

    # Prefer a sequence rather than isolated word matches.
    longest_block = max(
        (match["block_size"] for match in inliers),
        default=0,
    )

    if len(inliers) < 2 or longest_block < 2:
        return (
            None, None, matches,
            "Low confidence: fewer than two words in a matching sequence",
        )

    correction_seconds = statistics.median(
        match["offset"] for match in inliers
    )

    spread = statistics.median(
        abs(match["offset"] - correction_seconds)
        for match in inliers
    )

    # Confidence is descriptive, not a calibrated probability.
    if len(inliers) >= 5 and longest_block >= 3 and spread <= 0.20:
        confidence = "High"
    elif len(inliers) >= 3 and longest_block >= 2 and spread <= 0.40:
        confidence = "Medium"
    else:
        confidence = "Low"

    details = (
        f"{confidence}; {len(inliers)} timing matches; "
        f"median deviation {spread:.3f} s"
    )

    if len(inliers) < len(matches):
        details += (
            f"; rejected {len(matches) - len(inliers)} timing outlier(s)"
        )

    return correction_seconds, inliers, matches, details


def main():
    parser = argparse.ArgumentParser(
        description="Estimate UltraStar GAP corrections without editing songs."
    )
    parser.add_argument("--source", default="~/Music")
    parser.add_argument(
        "--language", choices=LANGUAGES, required=True
    )
    parser.add_argument(
        "--report", default="~/Music/ultrastar_gap_report.txt"
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"), default="cpu"
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    report = Path(args.report).expanduser().resolve()

    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    try:
        import whisperx
    except ImportError:
        raise SystemExit(
            "WhisperX missing. Activate your Poetry environment and install it."
        )

    charts = []
    for path in sorted(source.rglob("*.txt")):
        if path.resolve() == report:
            continue
        try:
            data = parse_chart(path)
        except OSError:
            continue

        gap, bpm, audio_name, beat, lyric, relative, lyric_words = data
        if gap is not None or bpm is not None or beat is not None:
            charts.append((path, data))

    if not charts:
        raise SystemExit(f"No likely UltraStar charts found in {source}")

    print(f"Found {len(charts)} charts.")
    print("Loading WhisperX model...")

    model = whisperx.load_model(
        "small",
        args.device,
        language=args.language,
        compute_type="int8",
    )

    try:
        align_model, align_metadata = whisperx.load_align_model(
            language_code=args.language,
            device=args.device,
        )
    except Exception as exc:
        print(f"Warning: alignment model unavailable: {type(exc).__name__}")
        align_model, align_metadata = None, None

    output = [
        "ULTRASTAR LYRIC-MATCH GAP ANALYSIS",
        "==================================",
        f"Source: {source}",
        f"Language: {args.language}",
        f"Charts found: {len(charts)}",
        "",
        "IMPORTANT:",
        "- No original song or chart files are modified.",
        "- GAP estimates depend on correct lyric matches.",
        "- Confidence labels are heuristic, not probabilities.",
        "- Instrumental intros, backing vocals and transcription errors can mislead.",
        "- #RELATIVE charts are skipped.",
        "- BPM drift and final-word timing are not analyzed.",
        "- Positive correction means increasing #GAP to delay the chart.",
        "",
    ]

    for number, (chart, data) in enumerate(charts, 1):
        (
            gap, bpm, audio_name, beat, lyric,
            relative, chart_words,
        ) = data

        audio_path = find_audio(chart, audio_name)
        chart_time = None
        correction = None
        suggested = None
        matches = []
        all_matches = []
        recognized_preview = ""
        note = ""

        print(f"[{number}/{len(charts)}] {chart.name}")

        if relative:
            note = "#RELATIVE chart; skipped"
        elif gap is None or bpm is None or bpm <= 0 or beat is None:
            note = "Missing or invalid GAP/BPM/first-note data"
        elif not chart_words:
            note = "No lyric words could be parsed from chart"
        elif audio_path is None:
            note = "Audio file not found"
        else:
            chart_time = gap / 1000.0 + beat * 60.0 / bpm

            try:
                audio_words, align_note = recognize_words(
                    audio_path,
                    args.language,
                    whisperx,
                    model,
                    args.device,
                    align_model,
                    align_metadata,
                )

                recognized_preview = " ".join(
                    word["word"] for word in audio_words[:20]
                )

                if not audio_words:
                    note = align_note
                else:
                    (
                        correction_seconds,
                        matches,
                        all_matches,
                        note,
                    ) = estimate_gap(
                        chart_words, audio_words, gap, bpm
                    )

                    if correction_seconds is not None:
                        correction = correction_seconds * 1000.0
                        suggested = gap + correction

            except Exception as exc:
                note = f"Recognition failed: {type(exc).__name__}: {exc}"

        output.extend([
            "-" * 72,
            f"Chart: {chart}",
            f"Audio: {audio_path or 'NOT FOUND'}",
            f"First chart lyric: {lyric or '(not parsed)'}",
            f"Chart opening words: {' '.join(w for _, w in chart_words[:20])}",
            f"Recognized opening words: {recognized_preview or '(none)'}",
            f"Current #GAP: {gap if gap is not None else 'MISSING'} ms",
            f"#BPM: {bpm if bpm is not None else 'MISSING'}",
            f"Chart first-note time: {chart_time:.3f} s"
                if chart_time is not None
                else "Chart first-note time: unavailable",
            f"Matched word count: {len(matches)}",
        ])

        if matches:
            output.append("Matched words (lyric -> audio):")
            for match in matches[:20]:
                output.append(
                    f"  {match['lyric']} -> {match['audio']}: "
                    f"chart {match['expected_time']:.3f}s, "
                    f"audio {match['audio_time']:.3f}s, "
                    f"offset {match['offset']:+.3f}s"
                )

        output.extend([
            f"Estimated GAP correction: {correction:+.0f} ms"
                if correction is not None
                else "Estimated GAP correction: unavailable",
            f"Suggested #GAP: {suggested:.0f} ms"
                if suggested is not None
                else "Suggested #GAP: unavailable",
            f"Confidence: {note}",
            "",
        ])

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(output), encoding="utf-8")
    print(f"\nReport saved to: {report}")
    print("Original song files were not modified.")


if __name__ == "__main__":
    main()