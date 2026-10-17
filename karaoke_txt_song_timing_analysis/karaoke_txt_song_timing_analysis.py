#!/usr/bin/env python3
"""Estimate UltraStar lyric GAP corrections; never modify song files."""

import argparse
import re
import sys
from pathlib import Path

LANGUAGES = ("en", "de", "it", "ko")

HEADER_RE = re.compile(r"^\s*#([A-Z0-9_]+)\s*:\s*(.*?)\s*$", re.I)
NOTE_RE = re.compile(r"^\s*[:*FRG]\s+(-?\d+)\s+\d+\s+-?\d+\s+(.+?)\s*$")


def parse_number(value):
    return float(value.strip().replace(",", "."))


def read_song(path):
    # UltraStar files are normally UTF-8; tolerate older legacy files.
    raw = path.read_bytes()
    text = None
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue

    headers = {}
    first_note_beat = None
    first_note_text = ""

    for line in text.splitlines():
        match = HEADER_RE.match(line)
        if match:
            headers[match.group(1).upper()] = match.group(2)
            continue

        note = NOTE_RE.match(line)
        if note:
            first_note_beat = int(note.group(1))
            first_note_text = note.group(2)
            break

    if first_note_beat is None:
        raise ValueError("no lyric notes found")

    if "BPM" not in headers:
        raise ValueError("missing #BPM")

    bpm = parse_number(headers["BPM"])
    if bpm <= 0:
        raise ValueError("invalid #BPM")

    gap = parse_number(headers.get("GAP", "0"))
    relative = headers.get("RELATIVE", "no").strip().lower() == "yes"

    audio_name = headers.get("AUDIO") or headers.get("MP3")
    if not audio_name:
        raise ValueError("missing #AUDIO or #MP3")

    audio_path = (path.parent / audio_name).resolve()
    if not audio_path.is_file():
        raise ValueError(f"audio file not found: {audio_name}")

    # First-note time: GAP plus beat position, in milliseconds.
    # BPM in UltraStar is normally four times musical quarter-note BPM.
    first_lyric_ms = gap + first_note_beat * 60000.0 / (bpm * 4.0)

    return {
        "title": headers.get("TITLE", path.stem),
        "artist": headers.get("ARTIST", "Unknown"),
        "audio": audio_path,
        "gap": gap,
        "bpm": bpm,
        "beat": first_note_beat,
        "lyric_ms": first_lyric_ms,
        "lyric": first_note_text,
        "relative": relative,
    }


def detect_first_word(audio_path, language, model, device, batch_size):
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size)

    segments = result.get("segments", [])
    if not segments:
        raise ValueError("speech recognition returned no words")

    # Prefer word-level alignment when the language model is available.
    try:
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=device
        )
        aligned = whisperx.align(
            segments, align_model, metadata, audio, device,
            return_char_alignments=False,
        )

        words = []
        for segment in aligned.get("segments", []):
            for word in segment.get("words", []):
                if word.get("start") is not None and word.get("word", "").strip():
                    words.append(word)

        if words:
            first = min(words, key=lambda w: w["start"])
            return float(first["start"]), first["word"].strip(), "word alignment"

    except Exception as exc:
        alignment_note = f"word alignment unavailable: {exc}"
    else:
        alignment_note = "word alignment returned no timed words"

    # Fallback: segment-level start time, less precise than word alignment.
    valid = [
        s for s in segments
        if s.get("start") is not None and s.get("text", "").strip()
    ]
    if not valid:
        raise ValueError("no timed transcription segments")

    first = min(valid, key=lambda s: s["start"])
    first_text = first["text"].strip().split()[0]
    return (
        float(first["start"]),
        first_text,
        f"segment estimate; {alignment_note}",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Check UltraStar first-lyric timing against recognized vocals."
    )
    parser.add_argument("--folder", required=True, type=Path,
                        help="Song folder; searched recursively")
    parser.add_argument("--language", required=True, choices=LANGUAGES,
                        help="en, de, it, or ko")
    parser.add_argument("--output", type=Path,
                        help="Report path (default: FOLDER/sync_report.txt)")
    parser.add_argument("--model", default="small",
                        help="Whisper model size (default: small; e.g. medium)")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--min-correction-ms", type=float, default=80,
                        help="Smaller differences are reported as near-synced")
    args = parser.parse_args()

    folder = args.folder.expanduser().resolve()
    if not folder.is_dir():
        parser.error(f"not a directory: {folder}")

    report_path = (
        args.output.expanduser().resolve()
        if args.output else folder / "sync_report.txt"
    )

    songs = sorted(folder.rglob("*.txt"))
    songs = [
        p for p in songs
        if p.resolve() != report_path
        and not p.name.endswith(".bak.txt")
        and not p.name.endswith(".txt.bak")
    ]
    if not songs:
        parser.error("no .txt song files found")

    try:
        import whisperx
        import torch
    except ImportError:
        sys.exit("Missing dependencies. Activate your venv and run: pip install whisperx")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        sys.exit("CUDA requested but unavailable. Try --device cpu.")

    compute_type = "int8" if device == "cpu" else "float16"
    print(f"Loading WhisperX model '{args.model}' for {args.language}...")
    model = whisperx.load_model(
        args.model,
        device,
        language=args.language,
        compute_type=compute_type,
    )

    lines = [
        "UltraStar synchronization report",
        "=" * 32,
        f"Folder: {folder}",
        f"Language: {args.language}",
        f"Songs scanned: {len(songs)}",
        "Mode: report only; no song files will be changed",
        "",
    ]

    for index, path in enumerate(songs, 1):
        print(f"[{index}/{len(songs)}] {path.name}")
        lines.extend([f"Song file: {path}", "-" * 60])

        try:
            song = read_song(path)
            detected_s, word, method = detect_first_word(
                song["audio"], args.language, model, device, args.batch_size
            )

            detected_ms = detected_s * 1000.0
            difference_ms = detected_ms - song["lyric_ms"]
            proposed_gap = song["gap"] + difference_ms

            lines.extend([
                f"Title: {song['title']}",
                f"Artist: {song['artist']}",
                f"Audio: {song['audio'].name}",
                f"First lyric text: {song['lyric']}",
                f"First recognized word: {word}",
                f"Current GAP: {song['gap']:.0f} ms",
                f"First lyric timestamp: {song['lyric_ms']:.0f} ms",
                f"Detected word start: {detected_ms:.0f} ms",
                f"Suggested correction: {difference_ms:+.0f} ms",
                f"Suggested GAP: {proposed_gap:.0f} ms",
                f"Detection method: {method}",
            ])

            if abs(difference_ms) < args.min_correction_ms:
                status = "NEAR SYNC (within threshold)"
            elif proposed_gap < 0:
                status = "REVIEW: suggested GAP is negative"
            elif song["relative"]:
                status = "REVIEW: #RELATIVE is enabled"
            else:
                status = "REVIEW RECOMMENDED"

            lines.extend([f"Status: {status}", ""])

        except Exception as exc:
            lines.extend([
                f"Status: ERROR - {type(exc).__name__}: {exc}",
                "",
            ])
            print(f"  Error: {exc}", file=sys.stderr)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nReport written to: {report_path}")


if __name__ == "__main__":
    main()
