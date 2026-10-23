import contextlib
import io
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from karaoke_txt_song_timing_analysis import karaoke_txt_song_timing_analysis as analysis


class ReviewReportTests(unittest.TestCase):
    def test_completion_estimate_uses_recent_completed_songs(self) -> None:
        now = datetime(2026, 10, 23, 8, 0)

        estimate = analysis.estimate_completion_time(
            list(range(1, 12)),
            remaining_songs=2,
            now=now,
        )

        self.assertEqual(estimate, datetime(2026, 10, 23, 8, 0, 13))

    def test_completion_estimate_is_unavailable_before_first_song_finishes(self) -> None:
        self.assertIsNone(analysis.estimate_completion_time([], remaining_songs=3))

    def test_estimated_end_format_includes_day_offset_when_needed(self) -> None:
        now = datetime(2026, 10, 23, 8, 0)

        self.assertEqual(
            analysis.format_estimated_end(datetime(2026, 10, 23, 23, 56), now),
            "23:56",
        )
        self.assertEqual(
            analysis.format_estimated_end(datetime(2026, 10, 24, 0, 5), now),
            "+1 day 00:05",
        )
        self.assertEqual(
            analysis.format_estimated_end(datetime(2026, 10, 25, 3, 0), now),
            "+2 days 03:00",
        )

    def test_entry_includes_eligible_corrections_of_at_least_one_second(self) -> None:
        result = analysis.Analysis(
            chart_path=Path("/charts/song.txt"),
            audio_path=Path("/audio/song.mp3"),
            old_gap_ms=100,
            bpm=120,
            status="Analyzed",
            suggested_gap_ms=1100,
            correction_ms=1000,
            confidence="High",
        )

        self.assertEqual(
            analysis.format_review_report_entry(result),
            "Audio: /audio/song.mp3\n"
            "GAP Correction: +1000 ms\n"
            "Confidence: High\n\n",
        )

    def test_entry_excludes_small_and_ineligible_corrections(self) -> None:
        base = {
            "chart_path": Path("/charts/song.txt"),
            "audio_path": Path("/audio/song.mp3"),
            "old_gap_ms": 100,
            "bpm": 120,
            "status": "Analyzed",
            "suggested_gap_ms": 1300,
            "confidence": "Medium",
        }
        results = [
            analysis.Analysis(**base, correction_ms=999),
            analysis.Analysis(
                **{**base, "confidence": "Low"},
                correction_ms=1000,
            ),
            analysis.Analysis(
                **{**base, "suggested_gap_ms": None},
                correction_ms=1000,
            ),
            analysis.Analysis(
                **{**base, "audio_path": None},
                correction_ms=1000,
            ),
        ]

        self.assertTrue(
            all(not analysis.format_review_report_entry(result) for result in results)
        )

    def test_main_writes_filtered_review_report_next_to_normal_report(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "songs"
            source.mkdir()
            charts = [source / f"song_{index}.txt" for index in range(3)]
            for chart in charts:
                chart.write_text("#TITLE:Test\n", encoding="utf-8")
            report_path = root / "gap_report_english.txt"
            results = [
                analysis.Analysis(
                    chart_path=charts[0],
                    audio_path=Path("/audio/song_0.mp3"),
                    old_gap_ms=100,
                    bpm=120,
                    status="Analyzed",
                    suggested_gap_ms=1300,
                    correction_ms=1200,
                    confidence="High",
                ),
                analysis.Analysis(
                    chart_path=charts[1],
                    audio_path=Path("/audio/song_1.mp3"),
                    old_gap_ms=100,
                    bpm=120,
                    status="Analyzed",
                    suggested_gap_ms=900,
                    correction_ms=800,
                    confidence="Medium",
                ),
                analysis.Analysis(
                    chart_path=charts[2],
                    audio_path=Path("/audio/song_2.mp3"),
                    old_gap_ms=100,
                    bpm=120,
                    status="Analyzed",
                    suggested_gap_ms=None,
                    correction_ms=2000,
                    confidence="Low",
                ),
            ]

            output = io.StringIO()
            with (
                patch(
                    "sys.argv",
                    [
                        "karaoke_txt_song_timing_analysis.py",
                        "--source",
                        str(source),
                        "--report",
                        str(report_path),
                    ],
                ),
                patch.object(
                    analysis,
                    "analyze_chart_in_child_process",
                    side_effect=results,
                ),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(analysis.main(), 0)

            console_output = output.getvalue()
            self.assertIn(
                "[1/3 Estimated End calculating] Analyzing:",
                console_output,
            )
            self.assertRegex(
                console_output,
                r"\[2/3 Estimated End \d{2}:\d{2}\] Analyzing:",
            )

            review_path = root / "gap_report_english_review.txt"
            review = review_path.read_text(encoding="utf-8")
            self.assertIn("Audio: /audio/song_0.mp3", review)
            self.assertIn("GAP Correction: +1200 ms", review)
            self.assertIn("Confidence: High", review)
            self.assertNotIn("song_1.mp3", review)
            self.assertNotIn("song_2.mp3", review)
            self.assertTrue(report_path.exists())


if __name__ == "__main__":
    unittest.main()
