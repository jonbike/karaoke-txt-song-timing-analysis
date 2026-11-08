# Changelog

Notable user-facing changes are listed here, grouped by project version.

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
