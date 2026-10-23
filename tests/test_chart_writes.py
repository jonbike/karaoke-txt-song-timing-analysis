import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from karaoke_txt_song_timing_analysis import karaoke_txt_song_timing_analysis as analysis


class ChartWriteTests(unittest.TestCase):
    def test_write_bpm_updates_chart_and_keeps_original_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\r\n#GAP: 100\r\n#BPM: 120\r\n"
            chart_path.write_bytes(original)
            result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analyzed",
                suggested_bpm=120.1234,
                bpm_confidence="High",
            )

            status = analysis.write_bpm_update(result)

            self.assertIn("Updated #BPM from 120 to 120.123", status)
            self.assertEqual(chart_path.read_bytes(), b"#TITLE:Song\r\n#GAP: 100\r\n#BPM: 120.123\r\n")
            self.assertEqual(chart_path.with_suffix(".bak").read_bytes(), original)

    def test_combined_updates_share_one_backup_of_original_chart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\n#GAP: 100\n#BPM: 120\n"
            chart_path.write_bytes(original)
            result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analyzed",
                suggested_gap_ms=250,
                confidence="High",
                suggested_bpm=121.25,
                bpm_confidence="Medium",
            )

            gap_status, bpm_status = analysis.write_chart_updates(
                result,
                write_gap=True,
                write_bpm=True,
            )

            self.assertIn("Updated #GAP", gap_status)
            self.assertIn("Updated #BPM", bpm_status)
            self.assertEqual(
                chart_path.read_bytes(),
                b"#TITLE:Song\n#GAP: 250\n#BPM: 121.250\n",
            )
            self.assertEqual(chart_path.with_suffix(".bak").read_bytes(), original)

    def test_bpm_write_skips_low_confidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\n#BPM: 120\n"
            chart_path.write_bytes(original)
            result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=None,
                bpm=120,
                status="Analyzed",
                suggested_bpm=121,
                bpm_confidence="Low",
            )

            status = analysis.write_bpm_update(result)

            self.assertIn("only Medium or High confidence is writable", status)
            self.assertEqual(chart_path.read_bytes(), original)
            self.assertFalse(chart_path.with_suffix(".bak").exists())

    def test_bpm_write_uses_three_decimal_precision_without_minimum_threshold(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\n#BPM: 120\n"
            chart_path.write_bytes(original)
            result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=None,
                bpm=120,
                status="Analyzed",
                suggested_bpm=120.001,
                bpm_confidence="Medium",
            )

            status = analysis.write_bpm_update(result)

            self.assertIn("Updated #BPM from 120 to 120.001", status)
            self.assertEqual(
                chart_path.read_bytes(),
                b"#TITLE:Song\n#BPM: 120.001\n",
            )

    def test_summary_report_includes_bpm_suggestion_and_write_status(self) -> None:
        result = analysis.Analysis(
            chart_path=Path("/charts/song.txt"),
            audio_path=Path("/audio/song.mp3"),
            old_gap_ms=100,
            bpm=120,
            status="Analyzed",
            suggested_gap_ms=100,
            correction_ms=0,
            confidence="High",
            suggested_bpm=120.125,
            bpm_correction=0.125,
            bpm_confidence="High",
            bpm_write_status="Updated #BPM from 120 to 120.125.",
        )
        args = analysis.argparse.Namespace(
            report_level="summary",
            match_threshold=0.55,
            batch_size=4,
            audio_opening_seconds=60,
            chart_opening_seconds=60,
            bpm="on",
            write_gap=False,
            write_bpm=True,
        )

        report = analysis.format_summary_report([result], args)

        self.assertIn("WhisperX batch size: 4", report)
        self.assertIn("Suggested #BPM: 120.125", report)
        self.assertIn("#BPM write: Updated #BPM from 120 to 120.125.", report)

    def test_write_bpm_requires_bpm_analysis(self) -> None:
        stderr = io.StringIO()
        with (
            patch(
                "sys.argv",
                [
                    "karaoke_txt_song_timing_analysis.py",
                    "--source",
                    "/unused",
                    "--report",
                    "/unused/report.txt",
                    "--write-bpm",
                ],
            ),
            contextlib.redirect_stderr(stderr),
            self.assertRaises(SystemExit) as raised,
        ):
            analysis.main()

        self.assertEqual(raised.exception.code, 2)
        self.assertIn("--write-bpm requires --bpm on", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
