#!/usr/bin/env python3
"""Read-only, first-pass UltraStar GAP analysis."""

import argparse
import re
from pathlib import Path

LANGUAGES = ("en", "de", "it", "ko")
AUDIO_EXTENSIONS = {
    ".mp3", ".ogg", ".opus", ".wav", ".flac", ".m4a", ".aac", ".wma"
}
META_RE = re.compile(r"^#([A-Z0-9_]+)\s*:\s*(.*)$", re.I)
NOTE_RE = re.compile(r"^[:FRGB-]\s*(-?\d+)\s+(-?\d+)\s+(-?\d+)\s*(.*)$")


def parse_chart(path):
    gap = bpm = first_beat = None
    audio_name = None
    first_lyric = ""
    relative = False

    for line in path.read_text(
        encoding="utf-8-sig", errors="replace"
    ).splitlines():
        line = line.strip()
        meta = META_RE.match(line)

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
                relative = value.lower() in ("yes", "true", "1")
            continue

        note = NOTE_RE.match(line)
        if note and first_beat is None:
            first_beat = float(note.group(1))
            first_lyric = note.group(4).replace("|", " ").strip()[:100]

    return gap, bpm, audio_name, first_beat, first_lyric, relative


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

    for candidate in folder.iterdir():
        if (candidate.is_file()
                and candidate.suffix.lower() in AUDIO_EXTENSIONS
                and candidate.stem.casefold() == chart.stem.casefold()):
            return candidate

    return None


def first_word_time(audio_path, language, whisperx, model, device):
    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=4)
    segments = result.get("segments", [])

    if not segments:
        return None, "", "No speech recognized"

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
        first = segments[0]
        return (
            first.get("start"),
            str(first.get("text", "")).strip(),
            f"Low confidence: word alignment unavailable ({type(exc).__name__})",
        )

    words = [
        word
        for segment in segments
        for word in segment.get("words", [])
        if word.get("start") is not None
        and str(word.get("word", "")).strip()
    ]

    if not words:
        return None, "", "No aligned word timestamps"

    words.sort(key=lambda word: word["start"])
    preview = " ".join(
        str(word.get("word", "")).strip() for word in words[:8]
    )
    return (
        float(words[0]["start"]),
        preview,
        "Low confidence: singing recognition and onset need listening review",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Estimate GAP corrections; never edits song files."
    )
    parser.add_argument("--source", default="~/Music")
    parser.add_argument("--language", choices=LANGUAGES, required=True)
    parser.add_argument(
        "--report", default="~/Music/ultrastar_gap_report.txt"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    report = Path(args.report).expanduser().resolve()

    if not source.is_dir():
        raise SystemExit(f"Source directory not found: {source}")

    try:
        import whisperx
    except ImportError:
        raise SystemExit(
            "WhisperX missing. Activate your virtual environment and run:\n"
            "pip install whisperx"
        )

    charts = []
    for path in sorted(source.rglob("*.txt")):
        if path.resolve() == report:
            continue
        try:
            data = parse_chart(path)
        except OSError:
            continue
        gap, bpm, audio_name, beat, lyric, relative = data
        if gap is not None or bpm is not None or beat is not None:
            charts.append((path, data))

    if not charts:
        raise SystemExit(f"No likely UltraStar charts found in {source}")

    model = whisperx.load_model(
        "small", args.device, language=args.language, compute_type="int8"
    )

    output = [
        "ULTRASTAR GAP-ONLY ANALYSIS",
        "===========================",
        f"Source: {source}",
        f"Language: {args.language}",
        f"Charts found: {len(charts)}",
        "",
        "IMPORTANT:",
        "- No song files are modified.",
        "- Suggested corrections are estimates, not guaranteed fixes.",
        "- Instrumental intros and backing vocals can mislead speech recognition.",
        "- #RELATIVE charts and charts with missing timing data are skipped.",
        "- BPM drift and the final word are NOT analyzed in this stage.",
        "- Positive correction means increasing #GAP to delay the chart.",
        "",
    ]

    for number, (chart, data) in enumerate(charts, 1):
        gap, bpm, audio_name, beat, lyric, relative = data
        audio_path = find_audio(chart, audio_name)
        chart_time = None
        onset = None
        words = ""
        note = ""

        if relative:
            note = "#RELATIVE chart; skipped"
        elif gap is None or bpm is None or bpm <= 0 or beat is None:
            note = "Missing or invalid GAP/BPM/first-note data"
        else:
            # Approximate UltraStar chart timing; verify suggestions in-game.
            chart_time = gap / 1000.0 + beat * 60.0 / bpm
            if audio_path is None:
                note = "Audio file not found"
            else:
                try:
                    onset, words, note = first_word_time(
                        audio_path, args.language, whisperx, model, args.device
                    )
                except Exception as exc:
                    note = f"Recognition failed: {type(exc).__name__}: {exc}"

        correction = None
        suggested = None
        if onset is not None and chart_time is not None:
            correction = (onset - chart_time) * 1000
            suggested = gap + correction

        output.extend([
            "-" * 72,
            f"Chart: {chart}",
            f"Audio: {audio_path or 'NOT FOUND'}",
            f"First chart lyric: {lyric or '(not parsed)'}",
            f"Recognized opening words: {words or '(none)'}",
            f"Current #GAP: {gap if gap is not None else 'MISSING'} ms",
            f"#BPM: {bpm if bpm is not None else 'MISSING'}",
            f"Chart first-note time: {chart_time:.3f} s"
                if chart_time is not None else "Chart first-note time: unavailable",
            f"Recognized first-word time: {onset:.3f} s"
                if onset is not None else "Recognized first-word time: unavailable",
            f"Estimated GAP correction: {correction:+.0f} ms"
                if correction is not None else "Estimated GAP correction: unavailable",
            f"Suggested #GAP: {suggested:.0f} ms"
                if suggested is not None else "Suggested #GAP: unavailable",
            f"Confidence: {note}",
            "",
        ])

    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(output), encoding="utf-8")
    print(f"\nReport saved to: {report}")
    print("Original song files were not modified.")


if __name__ == "__main__":
    main()