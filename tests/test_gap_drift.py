import unittest
from pathlib import Path

from karaoke_txt_song_timing_analysis import karaoke_txt_song_timing_analysis as analysis


class GapDriftWarningTests(unittest.TestCase):
    @staticmethod
    def make_matches(offsets_ms: list[float]) -> list[analysis.Match]:
        return GapDriftWarningTests.make_matches_at_times(
            [(index * 4.0, offset_ms) for index, offset_ms in enumerate(offsets_ms)]
        )

    @staticmethod
    def make_matches_at_times(
        observations: list[tuple[float, float]],
    ) -> list[analysis.Match]:
        matches = []
        for index, (chart_time, offset_ms) in enumerate(observations):
            chart_word = analysis.ChartWord(
                text=f"word{index}",
                beat=0,
                seconds=chart_time,
                index=index,
            )
            audio_word = analysis.AudioWord(
                text=f"word{index}",
                start=chart_time + offset_ms / 1000,
                end=chart_time + offset_ms / 1000 + 0.2,
                index=index,
            )
            matches.append(
                analysis.Match(
                    chart=chart_word,
                    audio=audio_word,
                    similarity=1.0,
                    gap_ms=offset_ms,
                )
            )
        return matches

    def test_warns_for_the_reported_opening_offset_trend(self) -> None:
        observations = [
            (0.000, 13555), (0.481, 13334), (0.962, 13174),
            (1.298, 13098), (2.163, 12893), (3.510, 12907),
            (3.990, 12847), (4.615, 12883), (5.096, 12822),
            (5.385, 12713), (7.019, 12680), (7.260, 12679),
            (7.644, 12676), (8.125, 12475), (8.365, 12495),
            (8.654, 12506), (9.375, 12405), (9.663, 12518),
            (10.288, 14895), (13.317, 12366), (13.846, 12297),
            (14.519, 12065), (14.712, 12052), (15.048, 12116),
            (15.385, 11919), (16.923, 11822), (17.260, 11805),
            (17.596, 11869), (18.413, 11773), (18.654, 11732),
            (20.192, 11695), (20.529, 11678), (21.346, 11582),
            (21.635, 11493), (21.971, 11437), (22.885, 11684),
            (23.413, 11456), (26.010, 11200), (26.442, 11129),
            (26.731, 11380), (27.019, 11332), (28.269, 11283),
            (28.702, 11150), (29.327, 11066), (30.000, 11073),
            (30.337, 11016), (32.260, 11163), (32.885, 10918),
            (33.221, 10802), (33.462, 10822), (34.808, 10996),
            (35.240, 10844), (35.625, 10820), (35.913, 10872),
            (36.827, 10678), (39.327, 10559), (42.644, 10424),
            (43.269, 10379), (43.606, 10862), (46.394, 10155),
        ]

        warning = analysis.detect_gap_drift_warning(
            self.make_matches_at_times(observations)
        )

        self.assertIn("sustained", warning)
        self.assertIn("BPM mismatch may be affecting the GAP estimate", warning)

    def test_warns_for_sustained_opening_offset_trend(self) -> None:
        offsets = [13_500 - index * 250 for index in range(12)]

        warning = analysis.detect_gap_drift_warning(self.make_matches(offsets))

        self.assertIn("sustained -2750 ms trend over 44.0 seconds", warning)
        self.assertIn("BPM mismatch may be affecting the GAP estimate", warning)

    def test_does_not_warn_for_a_single_offset_outlier(self) -> None:
        offsets = [12_000] * 12
        offsets[6] += 2_000

        self.assertEqual(
            analysis.detect_gap_drift_warning(self.make_matches(offsets)),
            "",
        )

    def test_warning_is_in_diagnostics_and_review_even_without_gap_eligibility(self) -> None:
        warning = "Opening match offsets show a sustained -2,750 ms trend."
        result = analysis.Analysis(
            chart_path=Path("/charts/song.txt"),
            audio_path=Path("/audio/song.mp3"),
            old_gap_ms=12_000,
            bpm=120,
            status="Analyzed",
            confidence="Low",
            gap_drift_warning=warning,
        )
        args = analysis.argparse.Namespace(
            report_level="summary",
            match_threshold=0.55,
            batch_size=4,
            audio_opening_seconds=60.0,
            chart_opening_seconds=60.0,
            bpm=False,
            write_gap=False,
            write_bpm=False,
        )

        report = analysis.format_report([result], args)
        review_entry = analysis.format_review_report_entry(result)

        self.assertIn(f"Warning: {warning}", report)
        self.assertIn("Audio: /audio/song.mp3", review_entry)
        self.assertIn(f"Warning: {warning}", review_entry)
        self.assertNotIn("GAP Correction:", review_entry)


if __name__ == "__main__":
    unittest.main()
