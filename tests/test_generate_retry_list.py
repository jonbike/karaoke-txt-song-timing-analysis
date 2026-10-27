import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from karaoke_txt_song_timing_analysis import generate_retry_list


class GenerateRetryListTests(unittest.TestCase):
    def test_selects_missing_and_failed_charts_by_unique_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "songs"
            successful_directory = source / "artist" / "successful"
            failed_directory = source / "artist" / "failed"
            missing_directory = source / "artist" / "missing"
            for directory in (
                successful_directory,
                failed_directory,
                missing_directory,
            ):
                directory.mkdir(parents=True)

            successful_chart = successful_directory / "successful.txt"
            failed_chart = failed_directory / "failed.txt"
            missing_chart = missing_directory / "missing.txt"
            for chart in (successful_chart, failed_chart, missing_chart):
                chart.write_text("chart", encoding="utf-8")

            report = root / "report.txt"
            report.write_text(
                "\n".join([
                    f"Chart: {successful_chart}",
                    "Status: Analysis complete",
                    f"Chart: {failed_chart}",
                    "Status: Analysis failed",
                    "Reason: Per-chart analysis process exited without returning a result (exit code 1).",
                ]),
                encoding="utf-8",
            )

            directories = generate_retry_list.unprocessed_directories(source, report)

            self.assertEqual(
                directories,
                sorted([failed_directory.resolve(), missing_directory.resolve()]),
            )

    def test_cli_writes_directory_list_and_requires_overwrite_for_existing_file(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "songs"
            song_directory = source / "artist - song"
            song_directory.mkdir(parents=True)
            (song_directory / "artist - song.txt").write_text(
                "chart",
                encoding="utf-8",
            )
            report = root / "report.txt"
            report.write_text("", encoding="utf-8")
            output = root / "retry-list.txt"

            with (
                patch(
                    "sys.argv",
                    [
                        "generate_retry_list.py",
                        "--source",
                        str(source),
                        "--report",
                        str(report),
                        "--output",
                        str(output),
                    ],
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(generate_retry_list.main(), 0)

            self.assertEqual(
                output.read_text(encoding="utf-8"),
                f"{song_directory.resolve()}\n",
            )

            with (
                patch(
                    "sys.argv",
                    [
                        "generate_retry_list.py",
                        "--source",
                        str(source),
                        "--report",
                        str(report),
                        "--output",
                        str(output),
                    ],
                ),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit) as raised,
            ):
                generate_retry_list.main()
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
