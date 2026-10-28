import unittest

from karaoke_txt_song_timing_analysis import karaoke_txt_song_timing_analysis as analysis


class GapMatchWeightingTests(unittest.TestCase):
    @staticmethod
    def make_matches(
        chart_times: list[float],
        offsets_ms: list[float],
    ) -> list[analysis.Match]:
        return [
            analysis.Match(
                chart=analysis.ChartWord(
                    text=f"lyric{index}",
                    beat=0,
                    seconds=chart_time,
                    index=index,
                ),
                audio=analysis.AudioWord(
                    text=f"lyric{index}",
                    start=chart_time + offset_ms / 1000,
                    end=chart_time + offset_ms / 1000 + 0.3,
                    index=index,
                ),
                similarity=0.95,
                gap_ms=offset_ms,
            )
            for index, (chart_time, offset_ms) in enumerate(
                zip(chart_times, offsets_ms, strict=True)
            )
        ]

    def test_isolated_offset_outlier_is_downweighted(self) -> None:
        matches = self.make_matches(
            [0.0, 0.2, 0.4, 2.0, 2.2, 2.4],
            [2500, 2500, 2500, 4000, 2500, 2500],
        )

        weights = analysis.gap_match_context_weights(matches)
        suggested, correction, confidence, reason = analysis.evaluate_matches(
            matches,
            old_gap_ms=2000,
            threshold=0.55,
            max_correction_ms=15000,
        )

        self.assertLess(weights[3], weights[2])
        self.assertEqual(suggested, 2500)
        self.assertEqual(correction, 500)
        self.assertEqual(confidence, "High")
        self.assertIn("5 context-supported informative matches", reason)
        self.assertIn("2 supported phrase groups", reason)
        self.assertIn("weighted timing MAD 0 ms", reason)

    def test_many_matches_in_one_phrase_do_not_give_high_confidence(self) -> None:
        matches = self.make_matches(
            [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
            [2500, 2500, 2500, 2500, 2500, 2500],
        )

        _, _, confidence, reason = analysis.evaluate_matches(
            matches,
            old_gap_ms=2000,
            threshold=0.55,
            max_correction_ms=15000,
        )

        self.assertEqual(confidence, "Low")
        self.assertIn("fewer than 2 supported phrase groups", reason)

    def test_matches_without_local_timing_support_are_not_confident(self) -> None:
        matches = self.make_matches(
            [0.0, 10.0, 20.0, 30.0],
            [2500, 2500, 2500, 2500],
        )

        _, _, confidence, reason = analysis.evaluate_matches(
            matches,
            old_gap_ms=2000,
            threshold=0.55,
            max_correction_ms=15000,
        )

        self.assertEqual(confidence, "Low")
        self.assertIn("context-supported informative", reason)


if __name__ == "__main__":
    unittest.main()
