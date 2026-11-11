# Changelog

Notable user-facing changes are listed here, grouped by project version.

## 0.3.5

- Automatically extend omitted opening-window options to cover late opening
  lyrics; explicitly supplied audio and chart window values remain fixed.
- List omitted command-line options and their defaults in report headers, and
  remove redundant safety and initial-window lines.

## 0.3.4

- Retry CUDA/GPU analysis failures once on CPU, including failures during later
  BPM sample transcription.
- Include every failed chart analysis, with its path and reason, in the
  companion review report.
- Document all command-line options in the README.

## 0.3.3

- Lower the minimum BPM write threshold from 0.50% to 0.35%.
- Report original and current GAP/BPM values together, and calculate BPM
  correction and percentage change against the original BPM after reanalysis.

## 0.3.2

- Add `--allow-small-updates` to permit below-threshold GAP and BPM writes
  without bypassing confidence or other safeguards.
- Include the full command line in reports and explain when the option permits
  a small update.
- Clarify that estimates may be imperfect and should be verified when possible.

## 0.3.1

- Preserve replaced chart `.bak` files in a `backups/` directory using local
  `YYYY-MM-DD-THH-MM-SS` timestamps and collision-safe sequence numbers, while
  keeping the latest `.bak` beside the chart.

## 0.3.0

- Make BPM analysis opt-in with the boolean `--bpm` flag. `--write-bpm`
  requires `--bpm`.
- Support Python 3.12 and 3.13, and require Poetry 2.5.1 or later.
- Add the GPL-3.0-only license and document typical workflows, dependencies,
  and how to run tests.

## 0.2.18

- Report BPM confidence independently from GAP confidence, while clearly
  marking BPM results that cannot be applied because GAP confidence is too low.

## 0.2.17

- After writing a BPM correction, reanalyze the chart to verify BPM and
  calculate the GAP using the updated tempo.
- Preserve the original chart backup across both writes; never write a stale
  first-pass GAP alongside a BPM change.

## 0.2.16

- Use context-weighted lyric matches to make GAP estimates more robust to
  outliers.
- Strengthen BPM confidence assessment and chart-write safeguards.

## 0.2.15

- Handle MP2 audio correctly when creating temporary opening clips, including
  MP2 audio found in files with an `.mp3` extension.
