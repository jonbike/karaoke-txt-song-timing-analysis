#!/usr/bin/env python3
"""Generate a source-list file for charts missing or failed in an analysis report."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def unprocessed_directories(source: Path, report: Path) -> list[Path]:
    source = source.expanduser().resolve()
    report = report.expanduser().resolve()
    if not source.is_dir():
        raise ValueError(f"Source is not an existing directory: {source}")
    if not report.is_file():
        raise ValueError(f"Report is not an existing file: {report}")

    chart_statuses: dict[Path, str | None] = {}
    current_chart: Path | None = None
    for line in report.read_text(encoding="utf-8").splitlines():
        if line.startswith("Chart: "):
            chart_text = line.removeprefix("Chart: ").strip()
            current_chart = Path(chart_text).expanduser().resolve()
            chart_statuses.setdefault(current_chart, None)
        elif line.startswith("Status: ") and current_chart is not None:
            chart_statuses[current_chart] = line.removeprefix("Status: ").strip()

    retry_directories: set[Path] = set()
    for chart in source.rglob("*.txt"):
        if not chart.is_file():
            continue
        resolved_chart = chart.resolve()
        status = chart_statuses.get(resolved_chart)
        if status != "Analysis complete":
            retry_directories.add(resolved_chart.parent)

    return sorted(retry_directories)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "List song directories whose charts are missing from the report "
            "or have a status other than Analysis complete."
        )
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Original source directory searched by the analysis run.",
    )
    parser.add_argument(
        "--report",
        required=True,
        help="Main analysis report to compare against.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path for the generated directory list.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    report = Path(args.report).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if output == report:
        parser.error("--output must not be the same file as --report.")
    if output == source:
        parser.error("--output must not be the source directory.")
    if source.is_dir() and output.is_relative_to(source):
        parser.error(
            "--output must be outside the source directory so it is not "
            "discovered as a chart on a later run."
        )
    if output.exists() and not args.overwrite:
        parser.error(f"Output already exists; choose another path or use --overwrite: {output}")

    try:
        directories = unprocessed_directories(source, report)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            "".join(f"{directory}\n" for directory in directories),
            encoding="utf-8",
        )
    except (OSError, UnicodeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print(f"Unfinished song directories: {len(directories)}")
    print(f"Directory list written to: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
