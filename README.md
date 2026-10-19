# Karaoke TXT Song Timing Analysis

Analyze UltraStar Deluxe karaoke charts against their audio to estimate
whether the chart's `#GAP` value should be adjusted and optionally verify its
`#BPM`. The tool uses WhisperX transcription and word timestamps to compare
chart lyrics with the recorded vocals, then writes its findings to a text
report. GAP analysis uses the opening lyrics; optional BPM analysis samples
selected sections of the song and is disabled by default.

By default, analysis does not change chart or audio files. Optional `#GAP`
updates are available with `--write-gap`; see [Writing suggested GAP values](#writing-suggested-gap-values)
before enabling it.

## Requirements

- Python 3.13
- [Poetry](https://python-poetry.org/)
- Audio files supported by WhisperX, such as MP3, M4A, WAV, FLAC, OGG, AAC,
  or Opus

WhisperX and its Python dependencies are installed through Poetry. The selected
WhisperX model may need to be downloaded the first time it is used. CPU
inference is supported; CUDA can be selected on a machine with a compatible
CUDA/PyTorch setup.

## Installation

Clone the repository, change into its directory, and install the project and
its dependencies:

```sh
poetry install
```

## Preparing your files

Pass `--source` a directory containing UltraStar `.txt` chart files and their
audio. The directory is searched recursively. Each chart should include valid
`#BPM` and `#GAP` values and lyric note lines with usable timings.

Audio is found from the chart's `#AUDIO` entry (or legacy `#MP3` entry), or by
looking for a supported audio file with the same base name in the chart's
directory. If a chart references an audio file, that reference must resolve to
a supported audio file.

## Usage

Run the analysis from the repository root. For example:

```sh
poetry run python karaoke_txt_song_timing_analysis/karaoke_txt_song_timing_analysis.py --source "~/Music/Testing" --report "~/Music/gap_report.txt" --language de --model small --device cpu --match-threshold 0.55 --write-gap
```

This scans `~/Music/Testing` and its subdirectories, analyzes charts using the
German WhisperX model `small` on the CPU, and writes the report to
`~/Music/gap_report.txt`. The quoted home-directory paths are expanded by the
program. In this example, `--write-gap` also enables eligible chart updates;
omit it to produce a report without changing chart files.

The report is created or overwritten at the path given by `--report`. It
includes each chart's current and suggested `#GAP`, match confidence, analysis
details, and any reason a suggestion was withheld or a chart could not be
analyzed. The command also prints progress and the report location to the
terminal.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--source` | Required | Directory containing charts and matching audio. |
| `--report` | Required | Text report path; an existing report at this path is overwritten. |
| `--language` | `en` | WhisperX language code, such as `de` for German. |
| `--model` | `medium` | WhisperX model name. |
| `--device` | `cpu` | Inference device: `cpu` or `cuda`. |
| `--compute-type` | Device-dependent | WhisperX compute type; defaults to `int8` on CPU and `float16` on CUDA. |
| `--match-threshold` | `0.55` | Minimum fuzzy word similarity from 0 to 1. Lower values allow less-exact matches. |
| `--audio-opening-seconds` | `45` | Length of the opening audio window to transcribe. |
| `--chart-opening-seconds` | `45` | Maximum chart time considered for opening lyrics. |
| `--max-correction-ms` | `15000` | Maximum correction allowed for a GAP suggestion. |
| `--bpm` | `off` | Set to `on` to enable BPM analysis using sampled song sections. |
| `--write-gap` | Off | Update charts when a GAP suggestion is eligible. |
| `--debug` | Off | Include full exception tracebacks in the report. |

## Writing suggested GAP values

Without `--write-gap`, chart and audio files are read only. With
`--write-gap`, a chart is updated only when the GAP estimate has Medium or High
confidence, is finite, and differs from the current value by at least 100 ms.
Before changing a chart, the tool refreshes a sibling backup named
`<chart>.txt.bak` (for example, `song.txt.bak`). Audio files are never
modified. Review the report and keep backups before enabling chart updates.

## Limitations

- GAP analysis compares the chart's opening lyrics with the opening audio
  window; it does not transcribe the entire song for this estimate.
- Charts without a valid BPM, GAP, lyric timings, or matching audio cannot
  produce a GAP suggestion. The report records the reason.
- BPM verification is disabled by default. Enable it with `--bpm on`; it
  samples selected sections rather than transcribing the entire song.
