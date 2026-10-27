import contextlib
import io
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from karaoke_txt_song_timing_analysis import karaoke_txt_song_timing_analysis as analysis


class SourceListTests(unittest.TestCase):
    def test_resolves_relative_paths_from_list_file_and_removes_overlaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_list_directory = root / "lists"
            source_list_directory.mkdir()
            songs = root / "songs"
            nested_songs = songs / "nested"
            nested_songs.mkdir(parents=True)
            source_list = source_list_directory / "unfinished.txt"
            source_list.write_text(
                "\n../songs/nested\n../songs\n../songs\n",
                encoding="utf-8",
            )

            directories = analysis.resolve_source_directories(source_list)

            self.assertEqual(directories, [songs.resolve()])

    def test_reports_invalid_entry_with_source_list_line_number(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_list = root / "unfinished.txt"
            source_list.write_text("\nmissing-song\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, r"line 2: not an existing directory"):
                analysis.resolve_source_directories(source_list)

    def test_rejects_empty_source_list(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            source_list = Path(temporary_directory) / "unfinished.txt"
            source_list.write_text("\n  \n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "contains no directory paths"):
                analysis.resolve_source_directories(source_list)

    def test_handles_thousands_of_unrelated_directories_without_pairwise_checks(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_list = root / "unfinished.txt"
            source_list.write_text(
                "".join(f"/songs/Artist {index:04d} - Song\n" for index in range(3000)),
                encoding="utf-8",
            )

            def is_directory(path: Path) -> bool:
                return path != source_list

            with (
                patch.object(Path, "is_dir", autospec=True, side_effect=is_directory),
                patch.object(
                    Path,
                    "is_relative_to",
                    side_effect=AssertionError("pairwise ancestor checks are too slow"),
                ),
            ):
                directories = analysis.resolve_source_directories(source_list)

            self.assertEqual(len(directories), 3000)

    def test_main_processes_charts_from_source_list_without_duplicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            list_directory = root / "lists"
            list_directory.mkdir()
            songs = root / "songs"
            nested_song = songs / "nested"
            nested_song.mkdir(parents=True)
            first_chart = songs / "first" / "first.txt"
            first_chart.parent.mkdir()
            first_chart.write_text("chart", encoding="utf-8")
            nested_chart = nested_song / "nested.txt"
            nested_chart.write_text("chart", encoding="utf-8")
            source_list = list_directory / "unfinished.txt"
            source_list.write_text(
                "../songs/nested\n../songs\n",
                encoding="utf-8",
            )
            report_path = root / "report.txt"
            processed: list[Path] = []

            def fake_analysis(chart_path: Path, args: Namespace) -> analysis.Analysis:
                processed.append(chart_path)
                return analysis.Analysis(
                    chart_path=chart_path,
                    audio_path=None,
                    old_gap_ms=None,
                    bpm=None,
                    status="Analysis complete",
                )

            with (
                patch(
                    "sys.argv",
                    [
                        "karaoke_txt_song_timing_analysis.py",
                        "--source",
                        str(source_list),
                        "--report",
                        str(report_path),
                    ],
                ),
                patch.object(
                    analysis,
                    "analyze_chart_with_cpu_fallback",
                    side_effect=fake_analysis,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(analysis.main(), 0)

            self.assertEqual(processed, [first_chart, nested_chart])
            report = report_path.read_text(encoding="utf-8")
            self.assertEqual(report.count("Chart:"), 2)
            self.assertIn(str(first_chart), report)
            self.assertIn(str(nested_chart), report)


if __name__ == "__main__":
    unittest.main()
