import unittest

from karaoke_txt_song_timing_analysis import karaoke_txt_song_timing_analysis as analysis


class BpmEstimationTests(unittest.TestCase):
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
