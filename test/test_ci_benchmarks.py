import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ci_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("ci_benchmarks", SCRIPT_PATH)
ci_benchmarks = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ci_benchmarks
SPEC.loader.exec_module(ci_benchmarks)


ENCODE_OUTPUT = """\
     Running benches/encode_benchmark.rs (target/release/deps/encode_benchmark)
Running CI benchmark subset
Symbol size: 1280 bytes (without pre-built plan)
symbol count = 10, encoded 1 MB in 0.004secs, throughput: 3979.5Mbit/s
symbol count = 100, encoded 1.95 MB in 0.008secs, throughput: 1953.1Mbit/s

Symbol size: 1280 bytes (with pre-built plan)
symbol count = 10, encoded 1.99 MB in 0.004secs, throughput: 3979.5Mbit/s
"""


DECODE_OUTPUT = """\
     Running benches/decode_benchmark.rs (target/release/deps/decode_benchmark)
Running CI benchmark subset
Symbol size: 1280 bytes
symbol count = 10, decoded 1 MB in 0.007secs using 0.0% overhead, throughput: 2274.0Mbit/s
symbol count = 100, decoded 1.95 MB in 0.010secs using 5.0% overhead, throughput: 1562.5Mbit/s
"""


class CiBenchmarkTests(unittest.TestCase):
    def test_parse_custom_throughput_accepts_integer_and_decimal_mb(self):
        metrics = ci_benchmarks.parse_custom_throughput(
            "\n".join([ENCODE_OUTPUT, DECODE_OUTPUT])
        )

        self.assertEqual(
            set(metrics),
            {
                "encode_benchmark/encoded/without pre-built plan/symbols=10",
                "encode_benchmark/encoded/without pre-built plan/symbols=100",
                "encode_benchmark/encoded/with pre-built plan/symbols=10",
                "decode_benchmark/decoded/1280 bytes/symbols=10/overhead=0.0%",
                "decode_benchmark/decoded/1280 bytes/symbols=100/overhead=5.0%",
            },
        )
        self.assertEqual(
            metrics[
                "encode_benchmark/encoded/without pre-built plan/symbols=10"
            ].mbits_per_second,
            3979.5,
        )
        self.assertEqual(
            metrics[
                "decode_benchmark/decoded/1280 bytes/symbols=100/overhead=5.0%"
            ].mbits_per_second,
            1562.5,
        )

    def test_run_benchmarks_aggregates_all_quick_benchmark_commands(self):
        calls = []

        def fake_run_command(command, env=None):
            calls.append((command, env))
            if command == ["git", "checkout", "--detach", "HEAD"]:
                return ci_benchmarks.CommandResult(0, "checked out")
            if "codec_benchmark" in command:
                return ci_benchmarks.CommandResult(0, "codec output")
            if "encode_benchmark" in command:
                return ci_benchmarks.CommandResult(0, ENCODE_OUTPUT)
            if "decode_benchmark" in command:
                return ci_benchmarks.CommandResult(0, DECODE_OUTPUT)
            raise AssertionError(f"unexpected command: {command}")

        with tempfile.TemporaryDirectory() as tmpdir:
            target_dir = Path(tmpdir) / "target"
            with patch.object(ci_benchmarks, "run_command", side_effect=fake_run_command):
                with patch.object(ci_benchmarks, "resolve_ref", return_value="abc123"):
                    run = ci_benchmarks.run_benchmarks("PR", "HEAD", target_dir)

        self.assertTrue(run.success)
        self.assertIn("codec output", run.output)
        self.assertIn(ENCODE_OUTPUT, run.output)
        self.assertIn(DECODE_OUTPUT, run.output)
        self.assertEqual([call[0] for call in calls[1:]], ci_benchmarks.QUICK_BENCH_COMMANDS)
        self.assertEqual(len(run.throughput), 5)
        for _, env in calls[1:]:
            self.assertEqual(env["CARGO_TARGET_DIR"], str(target_dir))


if __name__ == "__main__":
    unittest.main()
