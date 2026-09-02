import importlib.util
import pathlib
import unittest


SCRIPT = pathlib.Path(__file__).with_name("compare_benchmarks.py")
SPEC = importlib.util.spec_from_file_location("compare_benchmarks", SCRIPT)
compare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(compare)


class CompareBenchmarksTest(unittest.TestCase):
    def test_warm_metrics_use_median(self):
        iterations = [
            {"time_millis": 100, "cache_cpu_time": 100},
            {"time_millis": 10, "cache_cpu_time": 1},
            {"time_millis": 11, "cache_cpu_time": 2},
            {"time_millis": 500, "cache_cpu_time": 100},
        ]

        self.assertEqual(
            compare.get_warm_metrics(iterations),
            {"time_millis": 11, "cache_cpu_time": 2},
        )

    def test_highlight_respects_configured_threshold(self):
        self.assertEqual(
            compare.format_change_percentage(114, 100, 15, "slower_only"),
            "+14.0%",
        )
        self.assertEqual(
            compare.format_change_percentage(114, 100, 10, "slower_only"),
            "**+14.0%**",
        )


if __name__ == "__main__":
    unittest.main()
