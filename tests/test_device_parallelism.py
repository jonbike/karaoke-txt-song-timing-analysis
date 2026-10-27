import multiprocessing
import sys
import threading
import time
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

from karaoke_txt_song_timing_analysis import karaoke_txt_song_timing_analysis as analysis


class BothDeviceAnalysisTests(unittest.TestCase):
    def test_estimates_finish_from_elapsed_time_and_completed_throughput(self) -> None:
        now = analysis.datetime(2026, 10, 27, 9, 0)

        estimated_end = analysis.estimate_parallel_completion_time(
            elapsed_seconds=120,
            completed_songs=4,
            total_songs=10,
            now=now,
        )

        self.assertEqual(estimated_end, now + analysis.timedelta(seconds=180))
        self.assertIsNone(
            analysis.estimate_parallel_completion_time(120, 0, 10, now=now)
        )

    def make_analysis(
        self,
        chart_path: Path,
        status: str = "Analysis complete",
        reason: str = "",
    ) -> analysis.Analysis:
        return analysis.Analysis(
            chart_path=chart_path,
            audio_path=chart_path.with_suffix(".mp3"),
            old_gap_ms=0,
            bpm=120,
            status=status,
            reason=reason,
        )

    def test_runs_one_worker_per_device_and_queues_oom_retry(self) -> None:
        charts = [
            Path("/charts/first.txt"),
            Path("/charts/second.txt"),
            Path("/charts/third.txt"),
        ]
        args = Namespace(
            compute_type_explicit=True,
            compute_type="float32",
            debug=False,
        )
        cpu_started = threading.Event()
        lock = threading.Lock()
        active = {"cpu": 0, "cuda": 0}
        max_active = {"cpu": 0, "cuda": 0}
        started_at: dict[tuple[Path, str], float] = {}
        finished_at: dict[tuple[Path, str], float] = {}
        compute_types: dict[tuple[Path, str], str] = {}

        def fake_analyze(
            chart_path: Path,
            worker_args: Namespace,
            device: str,
        ) -> analysis.ChartAnalysisOutput:
            with lock:
                active[device] += 1
                max_active[device] = max(max_active[device], active[device])
                started_at[(chart_path, device)] = time.monotonic()
                compute_types[(chart_path, device)] = worker_args.compute_type

            if chart_path == charts[0] and device == "cuda":
                self.assertTrue(cpu_started.wait(timeout=2))
                time.sleep(0.02)
                result = self.make_analysis(
                    chart_path,
                    status="Analysis failed",
                    reason="CUDA out of memory.",
                )
            elif chart_path == charts[1] and device == "cpu":
                cpu_started.set()
                time.sleep(0.12)
                result = self.make_analysis(chart_path)
            elif chart_path == charts[0] and device == "cpu":
                result = self.make_analysis(chart_path)
            else:
                time.sleep(0.03)
                result = self.make_analysis(chart_path)

            with lock:
                finished_at[(chart_path, device)] = time.monotonic()
                active[device] -= 1
            return analysis.ChartAnalysisOutput(result, f"log:{device}:{chart_path.name}\n")

        with patch.object(
            analysis,
            "_analyze_chart_on_device",
            side_effect=fake_analyze,
        ):
            outputs = list(analysis.analyze_charts_on_both_devices(charts, args))

        self.assertEqual(max_active, {"cpu": 1, "cuda": 1})
        self.assertEqual(
            {chart_path for chart_path, _ in outputs},
            set(charts),
        )
        self.assertEqual(outputs[0][0], charts[2])
        self.assertEqual(compute_types[(charts[1], "cpu")], "float32")
        self.assertEqual(compute_types[(charts[0], "cpu")], "int8")
        self.assertGreaterEqual(
            started_at[(charts[0], "cpu")],
            finished_at[(charts[1], "cpu")],
        )
        self.assertTrue(
            any(
                output.analysis.chart_path == charts[0]
                and "succeeded on CPU retry" in output.analysis.reason
                for _, output in outputs
            )
        )
        self.assertTrue(all(output.logs for _, output in outputs))

    def test_device_worker_uses_device_defaults_and_explicit_compute_type(self) -> None:
        chart_path = Path("/charts/song.txt")
        result = self.make_analysis(chart_path)

        for explicit, requested_type, expected_cpu_type in (
            (False, "float16", "int8"),
            (True, "float32", "float32"),
        ):
            with self.subTest(explicit=explicit), patch.object(
                analysis,
                "analyze_chart_in_child_process_capturing_output",
                return_value=analysis.ChartAnalysisOutput(result),
            ) as worker:
                args = Namespace(
                    compute_type_explicit=explicit,
                    compute_type=requested_type,
                    debug=False,
                )
                analysis._analyze_chart_on_device(chart_path, args, "cpu")

            worker_args = worker.call_args.args[1]
            self.assertEqual(worker_args.device, "cpu")
            self.assertEqual(worker_args.compute_type, expected_cpu_type)

    def test_worker_captures_logs_for_parent_process(self) -> None:
        chart_path = Path("/charts/song.txt")
        receive_connection, send_connection = multiprocessing.Pipe(duplex=False)

        def fake_analyze(path: Path, args: Namespace) -> analysis.Analysis:
            print("worker stdout")
            print("worker stderr", file=sys.stderr)
            return self.make_analysis(path)

        with patch.object(analysis, "analyze_chart", side_effect=fake_analyze):
            analysis._analyze_chart_worker(
                send_connection,
                chart_path,
                Namespace(debug=False),
                capture_output=True,
            )

        output = receive_connection.recv()
        receive_connection.close()
        self.assertIn("worker stdout", output.logs)
        self.assertIn("worker stderr", output.logs)


if __name__ == "__main__":
    unittest.main()
