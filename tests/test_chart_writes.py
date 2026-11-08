import contextlib
from datetime import datetime
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from karaoke_txt_song_timing_analysis import karaoke_txt_song_timing_analysis as analysis


class ChartWriteTests(unittest.TestCase):
    def test_existing_backup_is_archived_before_latest_backup_is_refreshed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\n#GAP: 100\n"
            previous_backup = b"#TITLE:Older version\n#GAP: 80\n"
            chart_path.write_bytes(original)
            chart_path.with_suffix(".bak").write_bytes(previous_backup)
            result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analyzed",
                suggested_gap_ms=250,
                confidence="High",
            )

            with patch.object(analysis, "datetime") as mocked_datetime:
                mocked_datetime.now.return_value = datetime(
                    2026,
                    11,
                    8,
                    11,
                    33,
                    56,
                )
                status = analysis.write_gap_update(result)

            archived_backup = (
                chart_path.parent
                / "backups"
                / "2026-11-08-T11-33-56_000_song.bak"
            )
            self.assertIn(f"previous backup archived to {archived_backup}", status)
            self.assertEqual(archived_backup.read_bytes(), previous_backup)
            self.assertEqual(chart_path.with_suffix(".bak").read_bytes(), original)
            self.assertEqual(chart_path.read_bytes(), b"#TITLE:Song\n#GAP: 250\n")

    def test_archive_name_collision_uses_next_sequence_without_overwriting(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\n#GAP: 100\n"
            previous_backup = b"#TITLE:Older version\n#GAP: 80\n"
            collision_content = b"existing archive"
            chart_path.write_bytes(original)
            chart_path.with_suffix(".bak").write_bytes(previous_backup)
            archive_directory = chart_path.parent / "backups"
            archive_directory.mkdir()
            first_archive = archive_directory / "2026-11-08-T11-33-56_000_song.bak"
            first_archive.write_bytes(collision_content)
            result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analyzed",
                suggested_gap_ms=250,
                confidence="High",
            )

            with patch.object(analysis, "datetime") as mocked_datetime:
                mocked_datetime.now.return_value = datetime(
                    2026,
                    11,
                    8,
                    11,
                    33,
                    56,
                )
                analysis.write_gap_update(result)

            second_archive = archive_directory / "2026-11-08-T11-33-56_001_song.bak"
            self.assertEqual(first_archive.read_bytes(), collision_content)
            self.assertEqual(second_archive.read_bytes(), previous_backup)

    def test_successive_updates_preserve_each_previous_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\n#GAP: 100\n"
            older_backup = b"#TITLE:Older version\n#GAP: 80\n"
            chart_path.write_bytes(original)
            chart_path.with_suffix(".bak").write_bytes(older_backup)
            first_result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analyzed",
                suggested_gap_ms=250,
                confidence="High",
            )
            second_result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=250,
                bpm=120,
                status="Analyzed",
                suggested_gap_ms=400,
                confidence="High",
            )

            with patch.object(analysis, "datetime") as mocked_datetime:
                mocked_datetime.now.side_effect = [
                    datetime(2026, 11, 8, 11, 33, 56),
                    datetime(2026, 11, 8, 11, 33, 57),
                ]
                first_status = analysis.write_gap_update(first_result)
                second_status = analysis.write_gap_update(second_result)

            archive_directory = chart_path.parent / "backups"
            first_archive = archive_directory / "2026-11-08-T11-33-56_000_song.bak"
            second_archive = archive_directory / "2026-11-08-T11-33-57_000_song.bak"
            self.assertIn(f"previous backup archived to {first_archive}", first_status)
            self.assertIn(f"previous backup archived to {second_archive}", second_status)
            self.assertEqual(first_archive.read_bytes(), older_backup)
            self.assertEqual(second_archive.read_bytes(), original)
            self.assertEqual(
                chart_path.with_suffix(".bak").read_bytes(),
                b"#TITLE:Song\n#GAP: 250\n",
            )
            self.assertEqual(chart_path.read_bytes(), b"#TITLE:Song\n#GAP: 400\n")

    def test_archive_failure_preserves_chart_and_latest_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\n#GAP: 100\n"
            previous_backup = b"#TITLE:Older version\n#GAP: 80\n"
            chart_path.write_bytes(original)
            chart_path.with_suffix(".bak").write_bytes(previous_backup)
            (chart_path.parent / "backups").write_text(
                "not a directory",
                encoding="utf-8",
            )
            result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analyzed",
                suggested_gap_ms=250,
                confidence="High",
            )

            status = analysis.write_gap_update(result)

            self.assertIn("Failed to update chart", status)
            self.assertEqual(chart_path.read_bytes(), original)
            self.assertEqual(chart_path.with_suffix(".bak").read_bytes(), previous_backup)

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
                suggested_bpm=120.75,
                bpm_confidence="High",
            )

            status = analysis.write_bpm_update(result)

            self.assertIn("Updated #BPM from 120 to 120.750", status)
            self.assertEqual(chart_path.read_bytes(), b"#TITLE:Song\r\n#GAP: 100\r\n#BPM: 120.750\r\n")
            self.assertEqual(chart_path.with_suffix(".bak").read_bytes(), original)

    def test_combined_updates_share_one_backup_of_original_chart(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\n#GAP: 100\n#BPM: 120\n"
            previous_backup = b"#TITLE:Previous\n#GAP: 50\n#BPM: 100\n"
            chart_path.write_bytes(original)
            chart_path.with_suffix(".bak").write_bytes(previous_backup)
            result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analyzed",
                suggested_gap_ms=250,
                confidence="High",
                gap_drift_warning=(
                    "Opening match offsets show a sustained -2,750 ms trend."
                ),
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
            archived_backups = list((chart_path.parent / "backups").glob("*.bak"))
            self.assertEqual(len(archived_backups), 1)
            self.assertEqual(archived_backups[0].read_bytes(), previous_backup)

    def test_bpm_write_reanalyzes_before_writing_gap_and_preserves_backup(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\n#GAP: 100\n#BPM: 120\n"
            previous_backup = b"#TITLE:Previous\n#GAP: 50\n#BPM: 100\n"
            chart_path.write_bytes(original)
            chart_path.with_suffix(".bak").write_bytes(previous_backup)
            first_pass = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analysis complete",
                suggested_bpm=120.6,
                bpm_confidence="High",
                suggested_gap_ms=500,
                correction_ms=400,
                confidence="High",
            )
            second_pass = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120.6,
                status="Analysis complete",
                suggested_bpm=120.6,
                bpm_confidence="High",
                suggested_gap_ms=250,
                correction_ms=150,
                confidence="High",
            )
            args = analysis.argparse.Namespace(
                device="both",
                debug=False,
            )

            def reanalyze(
                path: Path,
                passed_args: analysis.argparse.Namespace,
            ) -> analysis.Analysis:
                self.assertEqual(path, chart_path)
                self.assertEqual(passed_args.device, "cuda")
                self.assertEqual(
                    chart_path.read_bytes(),
                    b"#TITLE:Song\n#GAP: 100\n#BPM: 120.600\n",
                )
                return second_pass

            with patch.object(
                analysis,
                "analyze_chart_with_cpu_fallback",
                side_effect=reanalyze,
            ):
                analysis.write_first_pass_chart_updates(
                    first_pass,
                    analysis.argparse.Namespace(write_bpm=True, write_gap=True),
                )
                result = analysis.reanalyze_after_bpm_update(first_pass, args)

            self.assertEqual(result, second_pass)
            self.assertIn("Updated #BPM from 120 to 120.600", result.bpm_write_status)
            self.assertIn("Updated #GAP from 100 ms to 250 ms", result.gap_write_status)
            self.assertEqual(
                chart_path.read_bytes(),
                b"#TITLE:Song\n#GAP: 250\n#BPM: 120.600\n",
            )
            self.assertEqual(chart_path.with_suffix(".bak").read_bytes(), original)
            archived_backups = list((chart_path.parent / "backups").glob("*.bak"))
            self.assertEqual(len(archived_backups), 1)
            self.assertEqual(archived_backups[0].read_bytes(), previous_backup)
            self.assertTrue(
                any("Second analysis pass ran" in note for note in result.bpm_sample_notes)
            )

    def test_failed_bpm_reanalysis_never_writes_first_pass_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            original = b"#TITLE:Song\n#GAP: 100\n#BPM: 120\n"
            chart_path.write_bytes(original)
            first_pass = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analysis complete",
                suggested_bpm=120.6,
                bpm_confidence="High",
            )
            args = analysis.argparse.Namespace(device="cpu", debug=False)

            first_pass.bpm_write_status = analysis.write_bpm_update(first_pass)
            with patch.object(
                analysis,
                "analyze_chart_with_cpu_fallback",
                return_value=analysis.Analysis(
                    chart_path=chart_path,
                    audio_path=None,
                    old_gap_ms=None,
                    bpm=None,
                    status="Analysis failed",
                    reason="test failure",
                ),
            ):
                result = analysis.reanalyze_after_bpm_update(first_pass, args)

            self.assertIn("second analysis pass failed", result.gap_write_status)
            self.assertIn("first-pass GAP was not written", result.gap_write_status)
            self.assertEqual(
                chart_path.read_bytes(),
                b"#TITLE:Song\n#GAP: 100\n#BPM: 120.600\n",
            )

    def test_skipped_bpm_update_writes_gap_from_first_pass(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            chart_path.write_bytes(b"#TITLE:Song\n#GAP: 100\n#BPM: 120\n")
            result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analysis complete",
                suggested_bpm=120.3,
                bpm_confidence="High",
                suggested_gap_ms=250,
                confidence="High",
            )

            bpm_was_updated = analysis.write_first_pass_chart_updates(
                result,
                analysis.argparse.Namespace(write_bpm=True, write_gap=True),
            )

            self.assertFalse(bpm_was_updated)
            self.assertIn("below the 0.50% write threshold", result.bpm_write_status)
            self.assertIn("Updated #GAP", result.gap_write_status)
            self.assertEqual(
                chart_path.read_bytes(),
                b"#TITLE:Song\n#GAP: 250\n#BPM: 120\n",
            )

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

    def test_bpm_write_skips_change_below_percentage_threshold(self) -> None:
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
                suggested_bpm=120.599,
                bpm_confidence="Medium",
            )

            status = analysis.write_bpm_update(result)

            self.assertIn("proposed #BPM change is 0.499%", status)
            self.assertIn("below the 0.50% write threshold", status)
            self.assertEqual(chart_path.read_bytes(), original)
            self.assertFalse(chart_path.with_suffix(".bak").exists())

    def test_bpm_write_accepts_change_at_percentage_threshold(self) -> None:
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
                suggested_bpm=120.6,
                bpm_confidence="Medium",
            )

            status = analysis.write_bpm_update(result)

            self.assertIn("Updated #BPM from 120 to 120.600", status)
            self.assertEqual(
                chart_path.read_bytes(),
                b"#TITLE:Song\n#BPM: 120.600\n",
            )

    def test_summary_report_includes_bpm_suggestion_and_write_status(self) -> None:
        result = analysis.Analysis(
            chart_path=Path("/charts/song.txt"),
            audio_path=Path("/audio/song.mp3"),
            old_gap_ms=100,
            bpm=312,
            status="Analyzed",
            suggested_gap_ms=100,
            correction_ms=0,
            confidence="High",
            suggested_bpm=333.152,
            bpm_correction=21.152,
            bpm_confidence="High",
            bpm_write_status="Updated #BPM from 312 to 333.152.",
        )
        args = analysis.argparse.Namespace(
            report_level="summary",
            match_threshold=0.55,
            batch_size=4,
            audio_opening_seconds=60,
            chart_opening_seconds=60,
            bpm=True,
            write_gap=False,
            write_bpm=True,
        )

        report = analysis.format_summary_report([result], args)

        self.assertIn("WhisperX batch size: 4", report)
        self.assertIn("Suggested #BPM: 333.152", report)
        self.assertIn("BPM change: +6.78% (increase chart tempo)", report)
        self.assertIn("#BPM write: Updated #BPM from 312 to 333.152.", report)

        detailed_args = analysis.argparse.Namespace(**vars(args))
        detailed_args.report_level = "all"
        detailed_report = analysis.format_report([result], detailed_args)
        self.assertIn("BPM change: +6.78% (increase chart tempo)", detailed_report)

        diagnostic_result = analysis.Analysis(
            chart_path=result.chart_path,
            audio_path=result.audio_path,
            old_gap_ms=result.old_gap_ms,
            bpm=312,
            status="Analyzed",
            suggested_bpm=290,
            bpm_correction=-22,
            bpm_confidence="Low",
        )
        diagnostic_report = analysis.format_summary_report(
            [diagnostic_result],
            args,
        )
        self.assertIn(
            "BPM change (diagnostic only): -7.05% (decrease chart tempo)",
            diagnostic_report,
        )

        gap_blocked_result = analysis.Analysis(
            chart_path=result.chart_path,
            audio_path=result.audio_path,
            old_gap_ms=result.old_gap_ms,
            bpm=312,
            status="Analyzed",
            suggested_bpm=333.152,
            bpm_correction=21.152,
            bpm_confidence="High",
            bpm_write_block_reason="GAP confidence is Low",
        )
        gap_blocked_report = analysis.format_summary_report(
            [gap_blocked_result],
            args,
        )
        self.assertIn("BPM confidence: High", gap_blocked_report)
        self.assertIn(
            "BPM update: Withheld because GAP confidence is Low.",
            gap_blocked_report,
        )

    def test_bpm_change_format_uses_signed_percent_and_chart_direction(self) -> None:
        self.assertEqual(
            analysis.format_bpm_change(100, 20),
            "BPM change: +20.00% (increase chart tempo)",
        )
        self.assertEqual(
            analysis.format_bpm_change(100, -20, diagnostic_only=True),
            "BPM change (diagnostic only): -20.00% (decrease chart tempo)",
        )
        self.assertIsNone(analysis.format_bpm_change(None, 20))
        self.assertIsNone(analysis.format_bpm_change(0, 20))

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
        self.assertIn("--write-bpm requires --bpm", stderr.getvalue())

    def test_write_bpm_implies_gap_writing_in_cli(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source = root / "songs"
            source.mkdir()
            chart_path = source / "song.txt"
            chart_path.write_text(
                "#TITLE:Song\n#GAP: 100\n#BPM: 120\n",
                encoding="utf-8",
            )
            report_path = root / "report.txt"
            result = analysis.Analysis(
                chart_path=chart_path,
                audio_path=None,
                old_gap_ms=100,
                bpm=120,
                status="Analysis complete",
                suggested_gap_ms=300,
                confidence="Low",
                suggested_bpm=121,
                bpm_correction=1,
                bpm_confidence="High",
                bpm_reason="Strong, consistent BPM samples.",
                bpm_write_block_reason="GAP confidence is Low",
            )

            with (
                patch(
                    "sys.argv",
                    [
                        "karaoke_txt_song_timing_analysis.py",
                        "--source",
                        str(source),
                        "--report",
                        str(report_path),
                        "--bpm",
                        "--write-bpm",
                    ],
                ),
                patch.object(
                    analysis,
                    "analyze_chart_with_cpu_fallback",
                    return_value=result,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(analysis.main(), 0)

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("Writing #BPM also enables #GAP writing", report)
            self.assertIn("#GAP write: Skipped: GAP confidence is Low", report)
            self.assertIn("Estimated #BPM (diagnostic only): 121.000", report)
            self.assertIn("BPM confidence: High", report)
            self.assertIn(
                "BPM update: Withheld because GAP confidence is Low.",
                report,
            )
            self.assertIn(
                "#BPM write: Skipped: GAP confidence is Low; "
                "BPM result is diagnostic only.",
                report,
            )
            self.assertEqual(
                chart_path.read_text(encoding="utf-8"),
                "#TITLE:Song\n#GAP: 100\n#BPM: 120\n",
            )


if __name__ == "__main__":
    unittest.main()
