import tempfile
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path

from karaoke_txt_song_timing_analysis import karaoke_txt_song_timing_analysis as analysis


class BpmEstimationTests(unittest.TestCase):
    def test_beginning_bpm_sample_reuses_all_opening_gap_matches(self) -> None:
        chart_words = [
            analysis.ChartWord(text=f"word{index}", beat=0, seconds=40 + index, index=index)
            for index in range(3)
        ]
        audio_words = [
            analysis.AudioWord(text=f"word{index}", start=10 + index, end=11 + index, index=index)
            for index in range(3)
        ]
        matches = [
            analysis.Match(
                chart=chart_word,
                audio=audio_word,
                similarity=1.0,
                gap_ms=(audio_word.start - chart_word.seconds) * 1000,
            )
            for chart_word, audio_word in zip(chart_words, audio_words)
        ]
        args = analysis.argparse.Namespace(
            audio_opening_seconds=60,
            chart_opening_seconds=60,
            model="small",
            language=None,
            device="cpu",
            compute_type="int8",
            batch_size=4,
            match_threshold=0.55,
            max_correction_ms=15000,
            bpm="on",
            debug=False,
        )

        with tempfile.TemporaryDirectory() as temporary_directory:
            chart_path = Path(temporary_directory) / "song.txt"
            chart_path.write_text("#TITLE:Test\n", encoding="utf-8")
            with (
                patch.object(
                    analysis,
                    "parse_chart",
                    return_value=(2000, 120, chart_words),
                ),
                patch.object(analysis, "detect_unknown_note_types", return_value=[]),
                patch.object(analysis, "find_audio_for_chart", return_value=chart_path),
                patch.object(
                    analysis,
                    "transcribe_audio_windows",
                    return_value=(
                        audio_words,
                        180,
                        "opening",
                        "en",
                        {},
                        {"opening": "opening"},
                        object(),
                    ),
                ),
                patch.object(analysis, "prepare_audio_words_for_matching", return_value=[]),
                patch.object(analysis, "local_alignment", return_value=matches),
                patch.object(analysis, "select_bpm_sample_windows", return_value=([], [])),
            ):
                result = analysis._analyze_chart_with_audio(
                    chart_path,
                    args,
                    opening_windows=(60, 60),
                )

        self.assertEqual(len(result.bpm_samples), 1)
        self.assertEqual(result.bpm_samples[0].label, "Beginning")
        self.assertEqual(result.bpm_samples[0].match_count, 3)
        self.assertEqual(result.bpm_samples[0].chart_time_seconds, 41)
        self.assertEqual(result.bpm_sample_diagnostics[0].matches, matches)

    def test_transcription_passes_configured_batch_size_to_whisperx(self) -> None:
        whisperx = MagicMock()
        whisperx.load_audio.return_value = [0.0] * (60 * 16000)
        model = whisperx.load_model.return_value
        model.transcribe.return_value = {
            "language": "en",
            "segments": [],
            "text": "hello",
        }

        with patch.dict("sys.modules", {"whisperx": whisperx}):
            opening_result = analysis.transcribe_audio_windows(
                audio_path=Path("/audio/song.wav"),
                model_name="medium",
                language="en",
                device="cuda",
                compute_type="float16",
                batch_size=16,
                opening_seconds=1,
                extra_windows=[],
            )
            analysis.transcribe_audio_windows(
                audio_path=Path("/audio/song.wav"),
                model_name="medium",
                language="en",
                device="cuda",
                compute_type="float16",
                batch_size=16,
                opening_seconds=0,
                extra_windows=[("Middle", 10, 5)],
                model=opening_result[6],
            )

        whisperx.load_model.assert_called_once()
        self.assertEqual(model.transcribe.call_count, 2)
        self.assertTrue(
            all(
                call.kwargs["batch_size"] == 16
                for call in model.transcribe.call_args_list
            )
        )

    def test_three_qualified_samples_can_receive_high_confidence(self) -> None:
        samples = [
            analysis.BpmSample(
                label="Beginning",
                chart_time_seconds=11.60,
                offset_ms=2134,
                match_count=19,
                spread_ms=32,
            ),
            analysis.BpmSample(
                label="Middle",
                chart_time_seconds=89.80,
                offset_ms=2146,
                match_count=41,
                spread_ms=39,
            ),
            analysis.BpmSample(
                label="End",
                chart_time_seconds=151.04,
                offset_ms=2080,
                match_count=36,
                spread_ms=278,
            ),
        ]

        _, _, confidence, _ = analysis.estimate_bpm_from_samples(samples, 168.04)

        self.assertEqual(confidence, "High")

    def test_opening_gap_sample_contributes_to_bpm_slope(self) -> None:
        samples = [
            analysis.BpmSample(
                label="Beginning",
                chart_time_seconds=54.441,
                offset_ms=-20801,
                match_count=11,
                spread_ms=1035,
            ),
            analysis.BpmSample(
                label="50% (middle)",
                chart_time_seconds=96.62,
                offset_ms=2128,
                match_count=19,
                spread_ms=40,
            ),
            analysis.BpmSample(
                label="90% (end)",
                chart_time_seconds=184.70,
                offset_ms=2057,
                match_count=10,
                spread_ms=340,
            ),
        ]

        suggested, correction, confidence, reason = (
            analysis.estimate_bpm_from_samples(samples, 340.0)
        )

        self.assertAlmostEqual(suggested, 289.243, places=3)
        self.assertAlmostEqual(correction, -50.757, places=3)
        self.assertEqual(confidence, "Medium")
        self.assertIn("3 reliable samples", reason)

    def test_three_samples_still_need_four_matches_each_for_high_confidence(self) -> None:
        samples = [
            analysis.BpmSample(
                label="Beginning",
                chart_time_seconds=0,
                offset_ms=0,
                match_count=3,
                spread_ms=30,
            ),
            analysis.BpmSample(
                label="Middle",
                chart_time_seconds=70,
                offset_ms=10,
                match_count=4,
                spread_ms=30,
            ),
            analysis.BpmSample(
                label="End",
                chart_time_seconds=140,
                offset_ms=20,
                match_count=4,
                spread_ms=30,
            ),
        ]

        _, _, confidence, _ = analysis.estimate_bpm_from_samples(samples, 168.04)

        self.assertEqual(confidence, "Medium")


if __name__ == "__main__":
    unittest.main()
