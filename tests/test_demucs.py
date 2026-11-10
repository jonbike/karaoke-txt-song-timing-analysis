import subprocess
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
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
            commands_seen: list[list[str]] = []

            def create_snippet(command: list[str], **kwargs: object) -> None:
                commands_seen.append(command)
                self.assertTrue(kwargs["check"])
                self.assertTrue(kwargs["capture_output"])
                if command[0] == "ffmpeg":
                    Path(command[-1]).touch()

            with patch.object(
                analysis.subprocess,
                "run",
                side_effect=[
                    SimpleNamespace(stdout="mp3\n"),
                    create_snippet,
                ],
            ) as run:
                snippet_path = analysis.create_audio_snippet(
                    audio_path,
                    61,
                    output_directory,
                )

        self.assertEqual(snippet_path.name, "song_61s.mp3")
        ffprobe_command, ffmpeg_command = run.call_args_list[0].args[0], run.call_args_list[1].args[0]
        self.assertEqual(ffprobe_command[0], "ffprobe")
        self.assertEqual(ffmpeg_command[ffmpeg_command.index("-t") + 1], "61")
        self.assertEqual(ffmpeg_command[ffmpeg_command.index("-map") + 1], "0:a:0")
        self.assertEqual(ffmpeg_command[ffmpeg_command.index("-c:a") + 1], "copy")
        self.assertEqual(run.call_count, 2)

    def test_mp2_stream_uses_mp2_snippet_extension(self) -> None:
        audio_path = Path("/audio/song.mp3")
        with tempfile.TemporaryDirectory() as temp:
            output_directory = Path(temp)

            def create_snippet(command: list[str], **kwargs: object) -> None:
                Path(command[-1]).touch()

            with patch.object(
                analysis.subprocess,
                "run",
                side_effect=[
                    SimpleNamespace(stdout="mp2\n"),
                    create_snippet,
                ],
            ):
                snippet_path = analysis.create_audio_snippet(
                    audio_path,
                    61,
                    output_directory,
                )

        self.assertEqual(snippet_path.name, "song_61s.mp2")

    def test_fails_if_audio_codec_cannot_be_probed(self) -> None:
        with (
            patch.object(
                analysis.subprocess,
                "run",
                return_value=SimpleNamespace(stdout=" \n"),
            ) as run,
            self.assertRaisesRegex(RuntimeError, "did not identify an audio codec"),
        ):
            analysis.create_audio_snippet(
                Path("/audio/song.mp3"),
                61,
                Path("/tmp"),
            )
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


class AudioLookupTests(unittest.TestCase):
    def test_supports_mp4_referenced_by_legacy_mp3_tag(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            song_directory = Path(temporary_directory)
            chart_path = song_directory / "ABC - Poison Arrow.txt"
            audio_path = song_directory / "ABC - Poison Arrow.mp4"
            chart_path.write_text(
                "#MP3: ABC - Poison Arrow.mp4\n",
                encoding="utf-8",
            )
            audio_path.touch()

            self.assertEqual(analysis.find_audio_for_chart(chart_path), audio_path)

    def test_supports_same_basename_mp4_audio(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            song_directory = Path(temporary_directory)
            chart_path = song_directory / "song.txt"
            audio_path = song_directory / "song.mp4"
            chart_path.write_text("#TITLE:song\n", encoding="utf-8")
            audio_path.touch()

            self.assertEqual(analysis.find_audio_for_chart(chart_path), audio_path)

    def test_distinguishes_unsupported_reference_from_missing_reference(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            song_directory = Path(temporary_directory)
            chart_path = song_directory / "song.txt"
            chart_path.write_text(
                "#MP3: song.mkv\n",
                encoding="utf-8",
            )
            (song_directory / "song.mkv").touch()

            unsupported_reason = analysis.audio_lookup_failure_reason(chart_path)
            self.assertIn("unsupported file type '.mkv'", unsupported_reason)
            self.assertNotIn("was found", unsupported_reason)

            chart_path.write_text(
                "#MP3: missing.mp4\n",
                encoding="utf-8",
            )
            missing_reason = analysis.audio_lookup_failure_reason(chart_path)
            self.assertIn("No audio file referenced by #MP3", missing_reason)
            self.assertIn("was found", missing_reason)


class AnalyzeChartWithDemucsTests(unittest.TestCase):
    def test_uses_temporary_vocal_stem_and_cleans_it_after_analysis(self) -> None:
        audio_path = Path("/audio/song.m4a")
        chart_path = Path("/charts/song.txt")
        chart_words = [analysis.ChartWord("hello", 0, 0, 0)]
        args = Namespace(
            device="cpu",
            debug=False,
            bpm=True,
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
            bpm=True,
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
            bpm=False,
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
            batch_size=4,
            model="medium",
            language="en",
            bpm=False,
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
                return_value=([], 120.0, "", "en", {}, {"opening": ""}, object()),
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
            bpm=True,
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


class CpuFallbackTests(unittest.TestCase):
    def setUp(self) -> None:
        self.chart_path = Path("/charts/song.txt")
        self.args = Namespace(
            device="cuda",
            compute_type="float16",
            model="medium",
            debug=False,
        )

    def make_result(self, status: str, reason: str) -> analysis.Analysis:
        return analysis.Analysis(
            chart_path=self.chart_path,
            audio_path=Path("/audio/song.mp3"),
            old_gap_ms=100,
            bpm=120,
            status=status,
            reason=reason,
        )

    def test_retries_cuda_oom_on_cpu_without_changing_device_for_next_song(self) -> None:
        next_chart = Path("/charts/next_song.txt")
        gpu_failure = self.make_result(
            "Analysis failed",
            "OutOfMemoryError: CUDA out of memory. Tried to allocate 160 MiB.",
        )
        cpu_success = self.make_result("Analysis complete", "GAP estimated.")
        next_song_success = analysis.Analysis(
            chart_path=next_chart,
            audio_path=Path("/audio/next_song.mp3"),
            old_gap_ms=100,
            bpm=120,
            status="Analysis complete",
        )
        call_devices: list[tuple[Path, str, str]] = []

        def analyze(path: Path, args: Namespace) -> analysis.Analysis:
            call_devices.append((path, args.device, args.compute_type))
            if path == self.chart_path and args.device == "cuda":
                return gpu_failure
            if path == self.chart_path:
                return cpu_success
            return next_song_success

        with patch.object(
            analysis,
            "analyze_chart_in_child_process",
            side_effect=analyze,
        ):
            retried_result = analysis.analyze_chart_with_cpu_fallback(
                self.chart_path,
                self.args,
            )
            next_result = analysis.analyze_chart_with_cpu_fallback(
                next_chart,
                self.args,
            )

        self.assertEqual(retried_result.status, "Analysis complete")
        self.assertIn("succeeded on CPU retry", retried_result.reason)
        self.assertEqual(next_result.status, "Analysis complete")
        self.assertEqual(
            call_devices,
            [
                (self.chart_path, "cuda", "float16"),
                (self.chart_path, "cpu", "int8"),
                (next_chart, "cuda", "float16"),
            ],
        )
        self.assertEqual(self.args.device, "cuda")
        self.assertEqual(self.args.compute_type, "float16")

    def test_does_not_retry_non_cuda_failures(self) -> None:
        failure = self.make_result("Analysis failed", "WhisperX model not found.")

        with patch.object(
            analysis,
            "analyze_chart_in_child_process",
            return_value=failure,
        ) as analyze:
            result = analysis.analyze_chart_with_cpu_fallback(
                self.chart_path,
                self.args,
            )

        self.assertIs(result, failure)
        analyze.assert_called_once_with(self.chart_path, self.args)

    def test_retries_non_memory_cuda_errors_on_cpu(self) -> None:
        cuda_failure = self.make_result(
            "Analysis failed",
            "RuntimeError: CUDA error: device-side assert triggered.",
        )
        cpu_success = self.make_result("Analysis complete", "GAP estimated.")

        with patch.object(
            analysis,
            "analyze_chart_in_child_process",
            side_effect=(cuda_failure, cpu_success),
        ) as analyze:
            result = analysis.analyze_chart_with_cpu_fallback(
                self.chart_path,
                self.args,
            )

        self.assertIs(result, cpu_success)
        self.assertIn("CUDA/GPU attempt failed", result.reason)
        self.assertIn("succeeded on CPU retry", result.reason)
        self.assertEqual(analyze.call_count, 2)
        self.assertEqual(analyze.call_args_list[1].args[1].device, "cpu")

    def test_retries_whisper_batch_size_memory_error_on_cpu(self) -> None:
        gpu_failure = self.make_result(
            "Analysis failed",
            "MemoryError: batch_size ( 32) is probably too large. "
            "Try with a smaller value until memory error disappears.",
        )
        cpu_success = self.make_result("Analysis complete", "GAP estimated.")

        with patch.object(
            analysis,
            "analyze_chart_in_child_process",
            side_effect=(gpu_failure, cpu_success),
        ) as analyze:
            result = analysis.analyze_chart_with_cpu_fallback(
                self.chart_path,
                self.args,
            )

        self.assertIs(result, cpu_success)
        self.assertIn("succeeded on CPU retry", result.reason)
        self.assertEqual(analyze.call_count, 2)
        self.assertEqual(analyze.call_args_list[1].args[1].device, "cpu")
        self.assertEqual(analyze.call_args_list[1].args[1].compute_type, "int8")

    def test_retries_cuda_oom_reported_by_bpm_sample_on_cpu(self) -> None:
        gpu_result = self.make_result("Analysis complete", "GAP estimated.")
        gpu_result.bpm_sample_notes.append(
            "Later BPM sample transcription failed: "
            "OutOfMemoryError: CUDA out of memory."
        )
        cpu_result = self.make_result("Analysis complete", "GAP estimated.")

        with patch.object(
            analysis,
            "analyze_chart_in_child_process",
            side_effect=(gpu_result, cpu_result),
        ) as analyze:
            result = analysis.analyze_chart_with_cpu_fallback(
                self.chart_path,
                self.args,
            )

        self.assertIs(result, cpu_result)
        self.assertIn("succeeded on CPU retry", result.reason)
        self.assertEqual(analyze.call_count, 2)
        self.assertEqual(analyze.call_args_list[1].args[1].device, "cpu")
        self.assertEqual(analyze.call_args_list[1].args[1].compute_type, "int8")

    def test_reports_cuda_and_cpu_errors_when_cpu_retry_fails(self) -> None:
        gpu_failure = self.make_result(
            "Analysis failed",
            "CUDA out of memory.",
        )
        cpu_failure = self.make_result(
            "Analysis failed",
            "CPU transcription failed.",
        )
        with patch.object(
            analysis,
            "analyze_chart_in_child_process",
            side_effect=(gpu_failure, cpu_failure),
        ):
            result = analysis.analyze_chart_with_cpu_fallback(
                self.chart_path,
                self.args,
            )

        self.assertIn("CUDA/GPU error: CUDA out of memory.", result.reason)
        self.assertIn("CPU error: CPU transcription failed.", result.reason)


if __name__ == "__main__":
    unittest.main()
