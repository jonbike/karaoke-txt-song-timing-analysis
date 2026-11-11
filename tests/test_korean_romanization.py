import unittest

from karaoke_txt_song_timing_analysis.karaoke_txt_song_timing_analysis import (
    AudioWord,
    ChartWord,
    local_alignment,
    prepare_audio_words_for_matching,
    select_gap_opening_windows,
)


class PrepareAudioWordsForMatchingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.audio_words = [
            AudioWord(text="안녕하세요", start=1.25, end=2.5, index=3),
        ]

    def test_romanizes_korean_for_latin_chart_lyrics(self) -> None:
        chart_words = [
            ChartWord(text="annyeonghaseyo", beat=0, seconds=1, index=0),
        ]

        result = prepare_audio_words_for_matching(
            self.audio_words,
            chart_words,
            "ko",
        )

        self.assertEqual([word.text for word in result], ["annyeonghaseyo"])
        self.assertEqual(
            (result[0].start, result[0].end, result[0].index),
            (1.25, 2.5, 3),
        )
        self.assertEqual(self.audio_words[0].text, "안녕하세요")
        matches = local_alignment(chart_words, result, threshold=0.55)
        self.assertEqual(len(matches), 1)
        self.assertEqual(matches[0].similarity, 1.0)

    def test_keeps_hangul_for_korean_script_chart_lyrics(self) -> None:
        chart_words = [
            ChartWord(text="안녕하세요", beat=0, seconds=1, index=0),
        ]

        result = prepare_audio_words_for_matching(
            self.audio_words,
            chart_words,
            "ko",
        )

        self.assertIs(result, self.audio_words)
        self.assertEqual(result[0].text, "안녕하세요")

    def test_does_not_romanize_non_korean_transcriptions(self) -> None:
        result = prepare_audio_words_for_matching(
            self.audio_words,
            [],
            "en",
        )

        self.assertIs(result, self.audio_words)

    def test_accepts_region_tagged_korean_language_code(self) -> None:
        result = prepare_audio_words_for_matching(
            self.audio_words,
            [
                ChartWord(
                    text="annyeonghaseyo",
                    beat=0,
                    seconds=1,
                    index=0,
                ),
            ],
            "ko-KR",
        )

        self.assertEqual(result[0].text, "annyeonghaseyo")


class SelectGapOpeningWindowsTests(unittest.TestCase):
    def test_keeps_60_second_windows_with_three_informative_words(self) -> None:
        chart_words = [
            ChartWord(text=text, beat=0, seconds=index * 10, index=index)
            for index, text in enumerate(("hello", "world", "again"))
        ]

        self.assertEqual(
            select_gap_opening_windows(chart_words, 60, 60),
            (60, 60),
        )

    def test_extends_when_existing_gap_puts_chart_start_late_in_audio(self) -> None:
        chart_words = [
            ChartWord(text=text, beat=index, seconds=index, index=index)
            for index, text in enumerate(("hello", "world", "again"))
        ]

        self.assertEqual(
            select_gap_opening_windows(chart_words, 60, 60, 56380),
            (90, 90),
        )

    def test_keeps_window_when_existing_gap_does_not_put_chart_start_late(self) -> None:
        chart_words = [
            ChartWord(text=text, beat=index, seconds=index, index=index)
            for index, text in enumerate(("hello", "world", "again"))
        ]

        self.assertEqual(
            select_gap_opening_windows(chart_words, 60, 60, 20000),
            (60, 60),
        )

    def test_extends_audio_window_past_90_seconds_for_long_existing_gap(self) -> None:
        chart_words = [
            ChartWord(text=text, beat=index, seconds=index, index=index)
            for index, text in enumerate(("hello", "world", "again"))
        ]

        self.assertEqual(
            select_gap_opening_windows(chart_words, 60, 60, 110000),
            (142, 90),
        )

    def test_extends_both_windows_to_late_chart_lyrics(self) -> None:
        chart_words = [
            ChartWord(text=text, beat=index, seconds=110 + index, index=index)
            for index, text in enumerate(("hello", "world", "again"))
        ]

        self.assertEqual(
            select_gap_opening_windows(chart_words, 60, 60),
            (142, 112),
        )

    def test_extends_both_windows_when_fewer_than_three_informative_words(self) -> None:
        chart_words = [
            ChartWord(text="hello", beat=0, seconds=10, index=0),
            ChartWord(text="world", beat=0, seconds=20, index=1),
        ]

        self.assertEqual(
            select_gap_opening_windows(chart_words, 60, 60),
            (90, 90),
        )

    def test_extends_when_chart_has_no_words_in_first_minute(self) -> None:
        chart_words = [
            ChartWord(text="hello", beat=0, seconds=70, index=0),
            ChartWord(text="world", beat=0, seconds=75, index=1),
            ChartWord(text="again", beat=0, seconds=80, index=2),
        ]

        self.assertEqual(
            select_gap_opening_windows(chart_words, 60, 60),
            (110, 90),
        )

    def test_short_words_do_not_count_as_informative(self) -> None:
        chart_words = [
            ChartWord(text=text, beat=0, seconds=index, index=index)
            for index, text in enumerate(("a", "to", "the"))
        ]

        self.assertEqual(
            select_gap_opening_windows(chart_words, 60, 60),
            (90, 90),
        )

    def test_does_not_shrink_explicit_windows_above_90_seconds(self) -> None:
        self.assertEqual(select_gap_opening_windows([], 120, 120), (120, 120))

    def test_explicit_audio_and_chart_windows_disable_auto_extension(self) -> None:
        chart_words = [
            ChartWord(text=text, beat=index, seconds=110 + index, index=index)
            for index, text in enumerate(("hello", "world", "again"))
        ]

        self.assertEqual(
            select_gap_opening_windows(
                chart_words,
                60,
                60,
                auto_extend_audio=False,
                auto_extend_chart=False,
            ),
            (60, 60),
        )

    def test_explicit_audio_window_can_be_fixed_while_chart_window_extends(self) -> None:
        chart_words = [
            ChartWord(text=text, beat=index, seconds=110 + index, index=index)
            for index, text in enumerate(("hello", "world", "again"))
        ]

        self.assertEqual(
            select_gap_opening_windows(
                chart_words,
                60,
                60,
                auto_extend_audio=False,
            ),
            (60, 112),
        )


if __name__ == "__main__":
    unittest.main()
