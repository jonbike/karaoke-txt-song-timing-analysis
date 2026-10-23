import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from karaoke_txt_song_timing_analysis import karaoke_txt_song_timing_analysis as analysis


class SeparateVocalsTests(unittest.TestCase):
    def test_runs_demucs_with_selected_device_and_returns_vocal_stem(self) -> None:
        for device in ("cpu", "cuda"):
            with self.subTest(device=device), tempfile.TemporaryDirectory() as temp:
                output_directory = Path(temp)
                command_seen: list[str] = []

                def run_demucs(command: list[str], **kwargs: object) -> None:
                    command_seen.extend(command)
                    self.assertTrue(kwargs["check"])
                    self.assertTrue(kwargs["capture_output"])
                    output_path = Path(command[command.index("--out") + 1])
                    vocal_path = (
                        output_path / "htdemucs" / "track" / "vocals.wav"
                    )
                    vocal_path.parent.mkdir(parents=True)
                    vocal_path.touch()

                with patch.object(
                    analysis.subprocess,
                    "run",
                    side_effect=run_demucs,
                ) as run:
                    vocals_path = analysis.separate_vocals(
                        Path("/audio/song.m4a"),
                        device,
                        output_directory,
                    )

                self.assertEqual(vocals_path.name, "vocals.wav")
                self.assertIn("--two-stems=vocals", command_seen)
                self.assertEqual(command_seen[command_seen.index("-d") + 1], device)
                run.assert_called_once()

    def test_creates_audio_snippet_with_stream_copy_and_requested_duration(self) -> None:
        audio_path = Path("/audio/song.mp3")
        with tempfile.TemporaryDirectory() as temp:
            output_directory = Path(temp)
            command_seen: list[str] = []

            def create_snippet(command: list[str], **kwargs: object) -> None:
                command_seen.extend(command)
                self.assertTrue(kwargs["check"])
                self.assertTrue(kwargs["capture_output"])
                Path(command[-1]).touch()

            with patch.object(
                analysis.subprocess,
                "run",
                side_effect=create_snippet,
            ) as run:
                snippet_path = analysis.create_audio_snippet(
                    audio_path,
                    61,
                    output_directory,
                )

        self.assertEqual(snippet_path.name, "song_61s.mp3")
        self.assertEqual(command_seen[command_seen.index("-t") + 1], "61")
        self.assertEqual(command_seen[command_seen.index("-map") + 1], "0:a:0")
        self.assertEqual(command_seen[command_seen.index("-c:a") + 1], "copy")
        run.assert_called_once()

    def test_fails_if_demucs_does_not_create_vocals_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(analysis.subprocess, "run"),
                self.assertRaisesRegex(RuntimeError, "expected exactly one vocals.wav"),
            ):
                analysis.separate_vocals(
                    Path("/audio/song.m4a"),
                    "cpu",
                    Path(temp),
                )


class AnalyzeChartWithDemucsTests(unittest.TestCase):
    def test_uses_temporary_vocal_stem_and_cleans_it_after_analysis(self) -> None:
        audio_path = Path("/audio/song.m4a")
        chart_path = Path("/charts/song.txt")
        chart_words = [analysis.ChartWord("hello", 0, 0, 0)]
        args = Namespace(
            device="cpu",
            debug=False,
            bpm="on",
            audio_opening_seconds=60,
            chart_opening_seconds=60,
        )
        temporary_stems: list[Path] = []

        def separate(
            source: Path,
            device: str,
            output_directory: Path,
        ) -> Path:
            self.assertEqual(source, audio_path)
            self.assertEqual(device, "cpu")
            vocal_path = output_directory / "htdemucs" / "song" / "vocals.wav"
            vocal_path.parent.mkdir(parents=True)
            vocal_path.touch()
            temporary_stems.append(vocal_path)
            return vocal_path

        def analyze(
            path: Path,
            passed_args: Namespace,
            transcription_audio_path: Path,
            opening_windows: tuple[float, float],
        ) -> analysis.Analysis:
            self.assertEqual(path, chart_path)
            self.assertIs(passed_args, args)
            self.assertTrue(transcription_audio_path.is_file())
            self.assertEqual(transcription_audio_path.name, "vocals.wav")
            self.assertEqual(opening_windows, (90, 90))
            return analysis.Analysis(
                chart_path=path,
                audio_path=audio_path,
                old_gap_ms=100,
                bpm=120,
                status="Analysis complete",
            )

        with (
            patch.object(
                analysis,
                "parse_chart",
                return_value=(100, 120, chart_words),
            ),
            patch.object(analysis, "find_audio_for_chart", return_value=audio_path),
            patch.object(analysis, "separate_vocals", side_effect=separate),
            patch.object(
                analysis,
                "_analyze_chart_with_audio",
                side_effect=analyze,
            ),
        ):
            result = analysis.analyze_chart(chart_path, args)

        self.assertEqual(result.status, "Analysis complete")
        self.assertEqual(len(temporary_stems), 1)
        self.assertFalse(temporary_stems[0].exists())

    def test_reports_demucs_failure_without_falling_back_to_original_audio(self) -> None:
        audio_path = Path("/audio/song.m4a")
        chart_path = Path("/charts/song.txt")
        args = Namespace(
            device="cuda",
            debug=False,
            bpm="on",
            audio_opening_seconds=60,
            chart_opening_seconds=60,
        )

        with (
            patch.object(
                analysis,
                "parse_chart",
                return_value=(
                    100,
                    120,
                    [analysis.ChartWord("hello", 0, 0, 0)],
                ),
            ),
            patch.object(analysis, "find_audio_for_chart", return_value=audio_path),
            patch.object(
                analysis,
                "separate_vocals",
                side_effect=subprocess.CalledProcessError(
                    1,
                    ["demucs"],
                    stderr="model failed",
                ),
            ),
            patch.object(analysis, "_analyze_chart_with_audio") as analyze,
        ):
            result = analysis.analyze_chart(chart_path, args)

        self.assertEqual(result.status, "Analysis failed")
        self.assertEqual(result.audio_path, audio_path)
        self.assertIn("Demucs vocal separation failed", result.reason)
        self.assertIn("model failed", result.reason)
        analyze.assert_not_called()

    def test_uses_buffered_opening_snippet_when_bpm_analysis_is_off(self) -> None:
        audio_path = Path("/audio/song.mp3")
        chart_path = Path("/charts/song.txt")
        chart_words = [
            analysis.ChartWord(text, index, index * 10, index)
            for index, text in enumerate(("hello", "world", "again"))
        ]
        args = Namespace(
            device="cpu",
            debug=False,
            bpm="off",
            audio_opening_seconds=60,
            chart_opening_seconds=60,
        )
        demucs_inputs: list[Path] = []

        def create_snippet(
            source: Path,
            duration: float,
            output_directory: Path,
        ) -> Path:
            self.assertEqual(source, audio_path)
            self.assertEqual(duration, 61)
            snippet = output_directory / "song_61s.mp3"
            snippet.touch()
            return snippet

        def separate(
            source: Path,
            device: str,
            output_directory: Path,
        ) -> Path:
            demucs_inputs.append(source)
            vocal_path = output_directory / "htdemucs" / "song" / "vocals.wav"
            vocal_path.parent.mkdir(parents=True)
            vocal_path.touch()
            return vocal_path

        with (
            patch.object(
                analysis,
                "parse_chart",
                return_value=(100, 120, chart_words),
            ),
            patch.object(analysis, "find_audio_for_chart", return_value=audio_path),
            patch.object(analysis, "create_audio_snippet", side_effect=create_snippet),
            patch.object(analysis, "separate_vocals", side_effect=separate),
            patch.object(
                analysis,
                "_analyze_chart_with_audio",
                return_value=analysis.Analysis(
                    chart_path=chart_path,
                    audio_path=audio_path,
                    old_gap_ms=100,
                    bpm=120,
                    status="Analysis complete",
                ),
            ) as analyze,
        ):
            result = analysis.analyze_chart(chart_path, args)

        self.assertEqual(result.status, "Analysis complete")
        self.assertEqual(len(demucs_inputs), 1)
        self.assertEqual(demucs_inputs[0].name, "song_61s.mp3")
        self.assertFalse(demucs_inputs[0].parent.exists())
        self.assertEqual(analyze.call_args.kwargs["opening_windows"], (60, 60))

    def test_prints_whisperx_progress_before_transcription(self) -> None:
        chart_path = Path("/charts/song.txt")
        audio_path = Path("/audio/song.mp3")
        chart_words = [
            analysis.ChartWord(text, index, index * 10, index)
            for index, text in enumerate(("hello", "world", "again"))
        ]
        args = Namespace(
            audio_opening_seconds=60,
            chart_opening_seconds=60,
            device="cpu",
            compute_type="int8",
            model="medium",
            language="en",
            bpm="off",
            debug=False,
            match_threshold=0.55,
            max_correction_ms=15000,
        )

        with (
            patch.object(
                analysis,
                "parse_chart",
                return_value=(100, 120, chart_words),
            ),
            patch.object(analysis, "find_audio_for_chart", return_value=audio_path),
            patch.object(
                analysis,
                "transcribe_audio_windows",
                return_value=([], 120.0, "", "en", {}, {"opening": ""}),
            ),
            patch("builtins.print") as print_output,
        ):
            analysis._analyze_chart_with_audio(chart_path, args)

        print_output.assert_any_call(
            "    WhisperX processing: song.txt",
            flush=True,
        )

    def test_cleans_up_vocal_stem_when_analysis_raises(self) -> None:
        audio_path = Path("/audio/song.m4a")
        chart_path = Path("/charts/song.txt")
        args = Namespace(
            device="cpu",
            debug=False,
            bpm="on",
            audio_opening_seconds=60,
            chart_opening_seconds=60,
        )
        temporary_stems: list[Path] = []

        def separate(
            source: Path,
            device: str,
            output_directory: Path,
        ) -> Path:
            self.assertEqual(source, audio_path)
            self.assertEqual(device, "cpu")
            vocal_path = output_directory / "htdemucs" / "song" / "vocals.wav"
            vocal_path.parent.mkdir(parents=True)
            vocal_path.touch()
            temporary_stems.append(vocal_path)
            return vocal_path

        with (
            patch.object(
                analysis,
                "parse_chart",
                return_value=(
                    100,
                    120,
                    [analysis.ChartWord("hello", 0, 0, 0)],
                ),
            ),
            patch.object(analysis, "find_audio_for_chart", return_value=audio_path),
            patch.object(analysis, "separate_vocals", side_effect=separate),
            patch.object(
                analysis,
                "_analyze_chart_with_audio",
                side_effect=ValueError("transcription failed"),
            ),
            self.assertRaisesRegex(ValueError, "transcription failed"),
        ):
            analysis.analyze_chart(chart_path, args)

        self.assertEqual(len(temporary_stems), 1)
        self.assertFalse(temporary_stems[0].exists())


if __name__ == "__main__":
    unittest.main()
