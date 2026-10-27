# Karaoke TXT Song Timing Analysis

Analyze UltraStar Deluxe karaoke charts against their audio to estimate
whether the chart's `#GAP` value should be adjusted and optionally verify its
`#BPM`. The tool uses WhisperX transcription and word timestamps to compare
chart lyrics with the recorded vocals, then writes its findings to a text
report. GAP analysis uses the opening lyrics; optional BPM analysis samples
selected sections of the song and is disabled by default. Demucs separates the
vocals before transcription so both GAP and optional BPM analysis use the
isolated vocal stem.

By default, analysis does not change chart or audio files. Optional `#GAP` and
`#BPM` updates are available with `--write-gap` and `--write-bpm`; see
[Writing suggested chart values](#writing-suggested-chart-values) before enabling
them.

## Requirements

- Python 3.13
- [Poetry](https://python-poetry.org/)
- Audio files supported by WhisperX, such as MP3, M4A, WAV, FLAC, OGG, AAC,
  Opus, or MP4 containers with an audio stream

Poetry installs the project's Python dependencies, including
`korean-romanizer` for Korean lyric matching and Demucs for music source
separation. WhisperX and its transcription dependencies are declared for Linux
and Windows. Windows dependency resolution and runtime behavior have not yet
been tested, so Windows support is intended but unverified. Demucs 4.1.0's
PyTorch dependency constraints allow it on Linux and Windows; macOS is not
supported by this dependency setup.

The selected WhisperX model may need to be downloaded the first time it is
used. CPU inference is supported; CUDA can be selected on a machine with a
compatible CUDA/PyTorch setup. Demucs runs on the same device selected for
analysis (`cpu` by default, or `cuda` when requested). It writes its
`vocals.wav` stem to an isolated temporary directory; that directory is
removed after the song has been analyzed. Demucs failures are reported and do
not silently fall back to transcribing the original mix.
The WhisperX transcription model is loaded once per song and reused for its
opening and optional later BPM windows.
With the default `--bpm off`, `ffmpeg` is also used to make a temporary,
stream-copied clip one second longer than the selected opening window before
Demucs runs. The script probes the first audio stream to choose a matching
snippet extension; for example, MP2 audio in a file named `.mp3` is written
into a temporary `.mp2` snippet instead of being incorrectly muxed as MP3.
With `--bpm on`, Demucs receives the full audio so later BPM samples remain
available. If a song fails with a CUDA out-of-memory error
while CUDA is selected, that song is retried once using CPU and `int8`; the
next song still uses the originally selected device.

When WhisperX detects Korean and chart lyrics contain no Hangul, aligned
Whisper words are romanized using Revised Romanization for lyric matching.
Korean-script charts continue to be matched against the original Hangul
transcription. The transcription shown in the report remains WhisperX's
original output.

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

To process only selected song directories, `--source` can instead point to a
UTF-8 text file containing one directory path per line. Blank lines are
ignored. Paths can be absolute or relative to the list file's directory. Each
listed directory is searched recursively; overlapping entries are handled
without processing the same chart more than once. If any listed path does not
exist or is not a directory, the command stops and identifies its line number.
For example, `unfinished-folders.txt` could contain:

```text
/music/Karaoke/Artist A - Song One
../Karaoke/Artist B - Song Two
```

Then pass the list file as `--source`:

```sh
poetry run python karaoke_txt_song_timing_analysis/karaoke_txt_song_timing_analysis.py --source "~/Music/unfinished-folders.txt" --report "~/Music/retry_report.txt" --device both
```

To generate such a list from a previous run, compare the original source tree
with its main report. The helper includes charts missing from the report and
charts whose status is not `Analysis complete` (including worker failures):

```sh
poetry run python karaoke_txt_song_timing_analysis/generate_retry_list.py \
  --source "~/Music/Testing" \
  --report "~/Music/gap_report.txt" \
  --output "~/Music/unfinished-folders.txt"
```

The helper reports how many song directories it selected. The output must be
outside the original source tree, and an existing output file is not replaced
unless `--overwrite` is supplied. Inspect the generated list, then pass it to
the analysis program as `--source` to retry those directories.

Audio is found from the chart's `#AUDIO` entry (or legacy `#MP3` entry), or by
looking for a supported audio file with the same base name in the chart's
directory. If a chart references an audio file, that reference must resolve to
a supported audio file. MP4 containers are supported when they contain an
audio stream. If a referenced file exists but has an unsupported extension, the
report identifies that file type separately from a missing file.

## Usage

Run the analysis from the repository root. For example:

```sh
poetry run python karaoke_txt_song_timing_analysis/karaoke_txt_song_timing_analysis.py --source "~/Music/Testing" --report "~/Music/gap_report.txt" --model small --device cpu --match-threshold 0.55 --write-gap
```

This scans `~/Music/Testing` and its subdirectories, analyzes charts using the
WhisperX model `small` on the CPU with automatic language detection, and writes
the report to `~/Music/gap_report.txt`. The quoted home-directory paths are
expanded by the program. In this example, `--write-gap` also enables eligible
chart updates; omit it to produce a report without changing chart files.

To specify a language instead of using automatic detection, pass a WhisperX
language code, for example `--language de` for German.

The report is created or overwritten at the path given by `--report`. It
includes each chart's current and suggested `#GAP`, match confidence, analysis
details, and any reason a suggestion was withheld or a chart could not be
analyzed. A companion report with `_review` appended to the filename is also
created or overwritten (for example, `gap_report.txt` produces
`gap_report_review.txt`). It lists eligible Medium/High GAP suggestions whose
absolute correction is at least 1,000 ms, and also includes every song that
triggers the opening timing-drift warning described below. Both reports are
updated after each song is processed. The command also prints progress and
both report locations to the terminal.

## Resuming after an interruption

Charts are processed in sorted path order. After each chart finishes, its
result is appended to both reports and flushed to disk. If a run stops, the
main report therefore records charts that completed; a chart that was still
being analyzed when the run stopped may not have an entry. Check its `Chart:`
paths to identify completed charts, and include the interrupted chart in any
retry.

There is no automatic resume or skip option. Starting the command again with
the same `--report` path creates new reports and starts processing from the
beginning; save the existing main and review reports elsewhere first if you
need to keep them. To retry only unfinished charts, make a source directory
containing those charts and their audio, preserving any relative paths used by
chart audio references, or generate a directory list with the helper described
in [Preparing your files](#preparing-your-files) and pass that list to
`--source`. Alternatively, rerun the original source if reprocessing completed
charts is acceptable.

In a Linux terminal, **Ctrl+Z suspends the process; it does not stop it**. The
same run can continue from where it paused: use `jobs -l` to find its job
number, then run `fg %1` (replace `1` with the job number shown). Press Ctrl+C
only if you intend to stop that resumed run. A suspended process remains alive
and may continue holding GPU memory and temporary files.

Demucs working files are created in temporary directories named
`karaoke-demucs-*` under the system temporary directory (commonly `/tmp`).
Python normally removes them when analysis of a song exits, including when
Ctrl+C interrupts it. A suspended run keeps its temporary directory; a forced
kill, crash, or power loss may leave one behind. After confirming no analysis
process is still running, you can inspect matching directories and remove
only confirmed leftovers from this tool.

Use `--report-level summary` to write only the key settings and per-chart
results. The default `--report-level all` keeps the detailed report. When
`--write-gap` is enabled, either report level includes the result of each
chart's `#GAP` write attempt.

## Options

| Option | Default | Description |
| --- | --- | --- |
| `--source` | Required | Directory containing charts and matching audio. |
| `--report` | Required | Text report path; an existing report at this path is overwritten. |
| `--report-level` | `all` | Report detail level: `all` for full diagnostics or `summary` for key results. |
| `--language` | Automatic detection | WhisperX language code, such as `de` for German. |
| `--model` | `medium` | WhisperX model name. |
| `--device` | `cpu` | Inference device: `cpu`, `cuda`, or `both` (one CPU and one CUDA song at a time). |
| `--compute-type` | Device-dependent | WhisperX compute type; defaults to `int8` on CPU and `float16` on CUDA. In `both` mode, those defaults apply to their respective workers; an explicit value is used on both. |
| `--batch-size` | `4` | WhisperX transcription batch size. Choose `4`, `8`, `16`, `32`, or `64`; larger batches use more memory. |
| `--match-threshold` | `0.55` | Minimum fuzzy word similarity from 0 to 1. Lower values allow less-exact matches. |
| `--audio-opening-seconds` | `60` | Minimum opening audio window; may extend to 90 seconds when chart lyrics are sparse or the current `#GAP` puts their start late in the window. |
| `--chart-opening-seconds` | `60` | Minimum chart opening window; may extend to 90 seconds when chart lyrics are sparse or the current `#GAP` puts their start late in the window. |
| `--max-correction-ms` | `15000` | Maximum correction allowed for a GAP suggestion. |
| `--bpm` | `off` | Set to `on` to enable BPM analysis using sampled song sections. |
| `--write-gap` | Off | Update charts when a GAP suggestion is eligible. |
| `--write-bpm` | Off | Update charts when a BPM suggestion is eligible; requires `--bpm on`. |
| `--debug` | Off | Include full exception tracebacks in the report. |

### Running on CPU and CUDA together

Use `--device both` to analyze up to two songs at once: one on CPU and one on
CUDA. This mode is opt-in; `--device cpu` and `--device cuda` continue to
process one song at a time. In `both` mode, reports are written by the parent
process as complete song results finish, so entries appear in completion order
rather than sorted path order. Worker log output is grouped by song and shown
when that song's analysis finishes. Progress includes an estimated finish time
based on the average completion rate observed so far; it is approximate and
should become more useful as more charts complete. Progress counts chart files,
not source-list lines or song directories. A directory with multiple duet
charts contributes multiple jobs to the total.

If a CUDA analysis runs out of memory, its song waits for the CPU worker slot
and is retried there with `int8`. It does not start an additional CPU analysis
while that slot is occupied. `--device both` may or may not be faster: CPU,
memory, or disk contention can offset the benefit of overlapping work.
Benchmark it with representative songs on your machine before relying on a
speedup.

## Writing suggested chart values

Without `--write-gap` or `--write-bpm`, chart and audio files are read only.
`--write-gap` updates a chart only when the GAP estimate has Medium or High
confidence, is finite, and differs from the current value by at least 100 ms.
`--write-bpm` requires `--bpm on` and updates only finite, positive BPM
suggestions with Medium or High confidence. BPM values are written to three
decimal places; there is no additional minimum-change threshold. If both
options are enabled, eligible values are written together from one chart
snapshot. Before changing a chart, the tool refreshes a sibling backup named
`<chart-stem>.bak` (for example, `song.bak` for `song.txt`). Audio files are never
modified. Review the report and keep backups before enabling chart updates.

When a BPM correction is available, reports also show its signed percentage
relative to the current BPM: `(suggested BPM - current BPM) / current BPM ×
100`. A positive percentage means an increase to the chart tempo; a negative
percentage means a decrease. This describes the suggested chart `#BPM` change,
not a change to the audio playback speed. Diagnostic-only estimates are
labeled accordingly.

## How GAP estimates handle outliers

The GAP estimate is the median of the timing offsets from accepted ordered
lyric matches. Each accepted match contributes equally: similarity is used to
decide whether a match meets `--match-threshold`, not to give its timing more
or less weight. Matches are not discarded just because their offsets are far
from the median. The median limits the influence of a small minority of
outliers, while the median absolute deviation (MAD) describes how spread out
the offsets are and contributes to the confidence rating. A high MAD can
therefore make a result Low confidence and prevent it from being recommended
or written, even though the median still provides a diagnostic estimate.

The report also warns when at least 10 opening matches spanning at least 20
chart seconds show a Theil-Sen offset trend of at least 1,000 ms across that
span, with trend-fit residual MAD no greater than 1,000 ms. This can indicate
BPM drift is affecting the GAP estimate. The warning is diagnostic only: it
does not change the GAP estimate, confidence, or write eligibility. The
companion `_review` report includes every song with this warning, even if its
GAP correction would not otherwise qualify for that report. This warning is
covered by a regression test using a real opening-match example with a
localized outlier.

## Limitations

- GAP analysis starts with the opening audio and chart windows, mapping chart
  word times into audio time using the current `#GAP`. If fewer than three
  informative chart words (at least three characters each) fall in both
  initial windows, or the first lyric is expected in the latter half of the
  audio window, both windows extend to 90 seconds. Explicitly configured
  windows longer than 90 seconds are preserved. When BPM verification is off,
  Demucs is run on a temporary audio clip one second longer than the selected
  audio window; with `--bpm on`, it uses the full audio to allow later samples.
  It does not transcribe the entire song for the GAP estimate.
- Charts without a valid BPM, GAP, lyric timings, or matching audio cannot
  produce a GAP suggestion. The report records the reason.
- BPM verification is disabled by default. Enable it with `--bpm on`; it
  samples selected sections rather than transcribing the entire song. Its
  beginning sample reuses the opening GAP matches, while middle/end samples use
  separately selected lyric sections. With `--bpm off`, opening matches are
  still checked for a possible timing-drift warning, but no BPM estimate or
  suggested `#BPM` is produced.
