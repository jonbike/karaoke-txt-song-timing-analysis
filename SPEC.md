# Karaoke TXT Song Timing Analysis Specification

## Status and maintenance

This is the living specification for the tool. It describes intended behavior,
including user-visible guarantees and safeguards. The [README](./README.md)
remains the operational guide for installation and command examples.

When behavior changes intentionally, update this specification and the
implementation/tests together. If this document and the implementation
disagree, treat the discrepancy as a specification or implementation defect;
do not silently assume either one is correct.

## 1. Purpose

The tool analyzes UltraStar Deluxe `.txt` karaoke charts against their
associated audio to:

- Estimate whether the chart's `#GAP` should change by matching chart lyrics
  against recognized and aligned vocal words.
- Optionally estimate a tempo correction for `#BPM` from lyric-timing samples
  taken at the beginning, middle, and end of the chart.
- Produce diagnostic and summary reports.
- Optionally write eligible `#GAP` and/or `#BPM` suggestions to charts, only
  when explicitly requested.

It is an analysis and assisted-correction tool. Suggestions are not guaranteed
to be correct; reports must make confidence, withheld results, and write
outcomes clear.

## 2. Scope and non-goals

### Runtime environment

- The project supports Python 3.12 and 3.13 and is installed with Poetry.
- WhisperX is configured for Linux and Windows; Windows behavior is intended
  but not yet verified. The current dependency setup does not support macOS.
- CPU is supported. CUDA requires a compatible CUDA/PyTorch environment.

### In scope

- Recursively finding UltraStar `.txt` charts beneath a source directory.
- Resolving audio from `#AUDIO`, legacy `#MP3`, or a supported same-basename
  audio file.
- Parsing chart timing and lyric words.
- Separating vocals, transcribing with WhisperX, aligning words, and comparing
  chart and audio timings.
- Estimating and reporting GAP and optional BPM results.
- Safely updating chart tags under explicit write switches.

### Out of scope

- Modifying audio files.
- Transcribing the entire song solely to estimate GAP.
- Automatically changing chart values without the corresponding write option.
- Guaranteeing that ASR, word alignment, or any inferred correction is correct.
- Supporting macOS with the current dependency setup.

## 3. Inputs and chart interpretation

1. The required `--source` path may be a directory or a UTF-8 text file
   containing one song-directory path per line. Search each directory
   recursively for `.txt` charts, excluding report output paths. Ignore blank
   lines; resolve relative list entries from the list file's parent directory.
   Validate all entries before processing and report an invalid entry with its
   line number. Reject an empty list. Deduplicate repeated and overlapping
   directories and chart paths.
2. A chart should provide valid positive `#BPM`, a `#GAP`, and lyric note lines
   with usable timing. A chart without the required information cannot produce
   a valid timing estimate; report the reason.
3. Resolve audio using the chart's `#AUDIO` reference, legacy `#MP3`
   reference, or a supported audio file with the same basename in the chart
   directory. If a reference is present but does not resolve to a supported
   audio file, report that failure rather than silently selecting unrelated
   audio.
4. Support audio formats accepted by the installed WhisperX stack, including
   MP3, M4A, WAV, FLAC, OGG, AAC, Opus, and MP4 containers with an audio
   stream. When a referenced file exists but its extension is unsupported,
   report that separately from a missing referenced file.
5. Interpret UltraStar note text as potentially syllabic. Reconstruct lyric
   words using chart word boundaries; a word's chart time is the beat of its
   first fragment converted using `#BPM`. Respect relative timing when
   `#RELATIVE: YES` is present. Phrase breaks and note extensions must not be
   treated as sung words.
6. If WhisperX detects Korean and chart words contain no Hangul, romanize
   aligned audio words using Revised Romanization for matching. For charts
   containing Hangul, match against the original Korean text. Reports retain
   WhisperX's original transcription.

## 4. GAP analysis

1. GAP analysis is always performed when chart, audio, and timing inputs are
   usable; `--bpm` only controls the optional BPM verification.
2. Start from the configured opening audio and chart windows (both default to
   60 seconds). Extend either window to at least 90 seconds if fewer than
   three informative chart words (normalized length at least three) map into
   both initial windows using the chart's current `#GAP`, or if the first
   expected lyric falls in the latter half of the audio opening window.
   Explicitly configured windows longer than 90 seconds remain unchanged.
3. Separate vocals with Demucs before WhisperX transcription. Never silently
   fall back to transcribing the original mix after vocal-separation failure.
   With BPM disabled, a temporary stream-copied audio clip one second longer
   than the selected opening window may be used as Demucs input. Probe its
   first audio stream and choose `.mp2` for MP2 streams so the temporary file
   extension matches the copied codec; preserve existing suffix behavior for
   other codecs. A missing or failed codec probe must be reported rather than
   guessed. With BPM enabled, preserve full audio for later samples.
4. Match chart lyrics and aligned audio words in order using fuzzy word
   similarity. The per-match timing offset is:

   `audio word start - chart word time`

   The GAP estimate is the context-weighted median of accepted ordered-match
   offsets. Weight each match using the nearest preceding and following
   matches within 8 chart seconds: give it full context weight when both
   neighbors agree within 600 ms, reduced weight when only one agrees, very
   low weight when nearby matches disagree, and low weight when it has no
   nearby context. Similarity may adjust the context weight modestly, but
   accepted matches must not be discarded solely because their offsets
   disagree. Use weighted MAD for offset spread. Report raw and
   context-supported match counts, supported phrase groups, weighted timing
   spread, confidence, and reason a result is withheld.
   Also issue a diagnostic warning when at least 10 opening matches spanning
   at least 20 chart seconds show a Theil-Sen offset trend of at least
   1000 ms across the span, with trend-fit residual MAD no greater than
   1000 ms. The warning indicates that BPM drift may be affecting the GAP
   estimate; it must not change the estimate, confidence, or write eligibility.
   Include all warned songs in the companion review report, even when they
   would not otherwise meet its GAP-correction inclusion criteria.
5. The default match threshold is `0.55`. A proposed absolute correction above
   `--max-correction-ms` (default `15000`) is Low confidence and is withheld
   from normal GAP recommendations.
6. GAP confidence rules:
   - Low if there are fewer than three ordered matches, fewer than two
     context-supported informative matches, no supported phrase group,
     weighted timing MAD above 1500 ms, or the absolute correction exceeds
     the configured maximum.
   - High if there are at least five context-supported informative matches
     across at least two supported phrase groups, weighted MAD no greater than
     600 ms, and at least two strong supported matches. A strong match has
     similarity at least `max(match threshold + 0.12, 0.67)`.
   - Otherwise Medium if there are at least three context-supported
     informative matches across at least two supported phrase groups and
     weighted MAD no greater than 1000 ms; otherwise Low.
   Lower-confidence estimates may be shown for diagnosis, but must not be
   presented as writable suggestions.

## 5. BPM verification

1. BPM verification is disabled by default. The `--bpm` flag enables it.
2. Use no more than three BPM samples:
   - **Beginning:** reuse the opening GAP matches; do not transcribe that
     opening again solely for BPM.
   - **Middle and end:** select lyric clusters near the 50% and 90% points of
     the chart lyric timeline, respectively. Use non-overlapping 45-second
     windows within the available audio and snap to nearby lyric clusters.
3. Transcribe and align the selected windows, then summarize each sample by
   median chart time, median offset, match count, and offset MAD.
4. Estimate BPM from the slope of offset versus chart time using a robust
   Theil-Sen estimator. Do not interpret a constant GAP offset as tempo drift;
   BPM is inferred from change in offset over chart time.
5. A sample is currently considered usable for estimation when it has at least
   three matches and offset MAD no greater than 1200 ms. Require at least two
   usable samples spanning at least 30 seconds. When at least two usable samples
   each have at least 40 matches and offset MAD no greater than 200 ms, estimate
   BPM using only those strong samples; weaker samples must not pull the
   estimate away from them. Report the samples actually used and identify any
   weaker samples excluded from the estimate.
6. A proposed relative BPM change above 15% is Low confidence. Two ordinary
   usable samples cannot receive stronger than Low confidence. Two strong
   samples as defined above, spanning at least 30 seconds, can receive High
   confidence. With three or more samples, High confidence requires at least
   four matches in every sample, residual MAD no greater than 500 ms, and sample
   offset MAD no greater than 700 ms. When the estimate uses three or more
   strong samples, the maximum absolute residual must also be no greater than
   500 ms so one conflicting strong section cannot be hidden by the median.
   Otherwise Medium requires residual MAD no greater than 1000 ms; when using
   strong samples, maximum absolute residual must also be no greater than
   1000 ms. Larger residuals are Low confidence.
7. Report BPM confidence from the BPM estimator independently of GAP
   confidence. Low or unassessed GAP confidence still blocks BPM updates and
   marks an otherwise usable BPM estimate diagnostic-only; report the
   GAP-based write block separately without changing the BPM confidence.
   Diagnostic BPM values must not be written to the chart.
8. BPM sampling notes and per-sample diagnostics must make skipped windows and
   unreliable samples understandable. A zero residual MAD from only two
   ordinary samples must not be described as proof of a reliable trend; the
   strong-sample match-count and offset-MAD criteria are required for two
   sections to establish High confidence.
9. When a BPM correction is available, reports include its signed percentage
   relative to the current BPM: `(suggested BPM - current BPM) / current BPM ×
   100`. Positive means an increase to the chart tempo; negative means a
   decrease. This is a chart `#BPM` change, not audio playback-speed
   modification. Label diagnostic-only percentages as diagnostic-only.

## 6. Transcription, model lifetime, and recovery

1. Use the selected WhisperX model (`medium` by default), device (`cpu` by
   default; also accept `both`), compute type (`int8` on CPU and `float16` on
   CUDA unless overridden), and transcription batch size.
   In `both` mode, run at most one CPU analysis and one CUDA analysis at once.
   Use each device's default compute type unless the user explicitly supplies
   `--compute-type`, in which case use that value on both workers.
2. `--batch-size` accepts only `4`, `8`, `16`, `32`, or `64`; default to `4`.
   Apply the selected batch size to opening and later-window transcription.
3. Load the WhisperX transcription model once per song and reuse it for that
   song's opening and optional BPM windows. Do not reload it between those
   passes.
4. Keep per-song process isolation. The model and temporary resources are
   released when that song's worker process exits; do not implicitly cache the
   model across songs.
5. If CUDA reports an out-of-memory failure, retry that song once on CPU with
   `int8`. The retry must not change the configured device for subsequent
   songs. In `both` mode, queue the retry for the single CPU slot; never start
   a second CPU analysis while that slot is occupied. Do not retry unrelated
   failures.
6. Clean temporary vocal stems and snippets after analysis, including normal
   exception and Ctrl+C interruption paths. A suspended process (for example,
   after Ctrl+Z in a Linux terminal) remains alive and retains its temporary
   resources until resumed or terminated. Forced termination, a crash, or
   power loss may prevent cleanup. Report failures explicitly; `--debug` may
   add tracebacks.

## 7. Reports and command-line behavior

1. Require `--source` and `--report`. Expand `~` in supplied paths.
2. `--report-level all` is the default and includes detailed diagnostics;
   `summary` includes key settings and per-chart results.
3. Create or overwrite the requested report and a companion review report
   whose name adds `_review` before the extension. Process discovered chart
   paths in sorted order for `cpu` and `cuda`; in `both` mode, write results
   in completion order. In every mode, update and flush both reports after
   each completed chart, so completed results are retained if later work
   fails or the run is interrupted. A chart interrupted before its report
   entry is written may need to be retried.
4. The review report includes eligible Medium/High GAP suggestions whose
   absolute correction is at least 1000 ms, and every chart that triggers the
   opening timing-drift warning, regardless of GAP suggestion eligibility.
5. Include current and suggested values, confidence, reasons, BPM sample
   details when enabled (including the signed percentage BPM change), and the
   result of each requested chart write.
   Summary reports must retain write outcomes and key BPM information when
   applicable.
6. Refuse output paths that would overwrite an existing audio file or
   UltraStar chart. If no eligible charts are found, report that and return a
   non-success exit status.
7. Validate CLI constraints before processing: match threshold in `[0, 1]`,
   opening windows greater than zero, non-negative maximum GAP correction,
   supported report level/device/batch-size choices, and `--write-bpm` only
   with `--bpm`.
8. There is no automatic resume or skip option. A new run overwrites its
   reports and starts from the beginning; users who want to retry only
   unfinished charts must preserve the old reports and use the retry-list
   helper to provide a source containing only their directories.
9. In `both` mode, keep report writing in the parent process and write each
   complete per-chart entry as its result becomes available. Capture worker
   logs and display them grouped by song so concurrent output cannot
   interleave. Show an estimated finish time in progress output based on the
   elapsed wall time and completed-chart throughput; no estimate is available
   before the first completion. Progress totals count discovered chart files,
   not source-list entries or directories.
10. The retry-list helper compares a prior main report with the original
    source tree and writes one directory per chart that is missing from the
    report or whose status is not `Analysis complete`. Deduplicate directories
    that contain multiple charts and do not overwrite an existing output list
    unless `--overwrite` is supplied. The generated list must be accepted as
    `--source`.

## 8. Chart-writing safeguards

1. Without `--write-gap` or `--write-bpm`, chart and audio files remain
   unchanged.
2. `--write-gap` may update only a finite suggestion with Medium or High GAP
   confidence whose rounded change is at least 100 ms. Do not update if the
   chart's GAP has changed since analysis.
3. `--write-bpm` requires `--bpm`. It may update only a finite positive
   BPM suggestion with Medium or High BPM confidence when its rounded change
   is at least 0.5% of the current BPM. Round to three decimal places before
   evaluating the percentage and writing. `--write-bpm` implies `--write-gap`.
   If the BPM is updated, write BPM alone first, then reanalyze the chart with
   the updated BPM. Use the second pass to verify BPM and calculate the GAP;
   never write the first-pass GAP after changing BPM. Write GAP only from the
   second pass, preserving the original pre-BPM backup across both writes. Do
   not automatically write a second BPM suggestion. If reanalysis fails, retain
   the BPM update and clearly report that GAP was not recalculated or written.
   If no BPM update occurs, the first-pass GAP may be written normally. Do not
   update if the chart's BPM has changed since analysis.
4. If BPM is not updated and both GAP and BPM writes are enabled, combine any
   eligible first-pass changes into one chart update and one backup. When BPM
   is updated, follow the two-pass sequence in rule 3 and preserve the original
   backup for the later GAP write.
5. Before replacing a chart, refresh the sibling `<chart-stem>.bak` backup and
   verify it matches the bytes read for the update. If that backup already
   exists, preserve it in a `backups/` subdirectory using a local-time,
   filename-sortable `YYYY-MM-DD-THH-MM-SS` timestamp followed by a
   collision-safe sequence number and the original backup filename. Never
   overwrite an archived backup.
   Refuse to replace chart or backup symlinks.
6. Write through a temporary file in the chart directory and atomically
   replace the chart. Preserve the chart's file mode and line endings. Never
   modify audio.
7. Report each write as updated, no-op, skipped with a reason, or failed;
   do not claim a successful write if backup or replacement failed.

## 9. Acceptance criteria for changes

A change conforms to this specification when:

- Existing behavior remains unchanged unless the change intentionally updates
  this document and its related user documentation.
- GAP and BPM behavior is tested against their measured thresholds and
  decision boundaries, including withheld and diagnostic-only cases.
- Opening timing-drift warning behavior is tested at its thresholds, with an
  isolated-outlier negative case and a regression case based on real match
  data; warned charts are verified to appear in the companion review report
  without changing GAP confidence or write eligibility.
- WhisperX calls use the configured model, device, compute type, language, and
  batch size; one model instance is reused for all transcription windows
  within a song.
- Chart writes are opt-in, validate confidence and stale values, preserve an
  original backup, and leave audio untouched.
- Reports explain estimates, confidence, samples, skips, and write outcomes
  without overstating what the evidence establishes.
- Failure paths clean temporary resources and surface errors explicitly.
- The relevant test suite passes, and CLI/documentation changes stay aligned.
