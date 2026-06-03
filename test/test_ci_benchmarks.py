import importlib.util
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
BENCHES_DIR = REPO_ROOT / "benches"
ENCODE_DEFAULT_SYMBOL_COUNTS = (
    10,
    100,
    250,
    500,
    1000,
    2000,
    5000,
    10000,
    20000,
    50000,
)
DECODE_DEFAULT_SYMBOL_COUNTS = (10, 100, 250, 500, 1000)
CI_SYMBOL_COUNTS = (10, 100, 250, 500, 1000, 2000, 5000, 10000, 20000)
CI_DECODE_SYMBOL_COUNTS = (10, 100, 250, 500, 1000)
SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "ci_benchmarks.py"
SPEC = importlib.util.spec_from_file_location("ci_benchmarks", SCRIPT_PATH)
ci_benchmarks = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ci_benchmarks
SPEC.loader.exec_module(ci_benchmarks)


def format_encode_lines(mbits_per_second):
    return "\n".join(
        (
            f"symbol count = {count}, encoded 1.95 MB in 0.008123456secs, "
            f"throughput: {mbits_per_second:.3f}Mbit/s"
        )
        for count in CI_SYMBOL_COUNTS
    )


def format_decode_lines(overhead, mbits_per_second):
    return "\n".join(
        (
            f"symbol count = {count}, decoded 1.95 MB in 0.010123456secs "
            f"using {overhead:.1f}% overhead, throughput: {mbits_per_second:.3f}Mbit/s"
        )
        for count in CI_DECODE_SYMBOL_COUNTS
    )


def read_bench_source(bench_name):
    return (BENCHES_DIR / bench_name).read_text(encoding="utf-8")


def parse_usize_array(source, const_name):
    match = re.search(
        rf"(?:pub )?const {const_name}:\s*\[usize;\s*\d+\]\s*=\s*\[(?P<counts>[^\]]+)\];",
        source,
    )
    if not match:
        raise AssertionError(f"{const_name} constant was not found")
    return tuple(int(count.strip()) for count in match.group("counts").split(","))


def parse_usize_expr(source, const_name):
    match = re.search(rf"const {const_name}:\s*usize\s*=\s*(?P<expr>[^;]+);", source)
    if not match:
        raise AssertionError(f"{const_name} constant was not found")
    return " ".join(match.group("expr").split())


def parse_ci_symbol_counts(const_name):
    source = (BENCHES_DIR / "ci_symbol_counts" / "mod.rs").read_text(
        encoding="utf-8"
    )
    return parse_usize_array(source, const_name)


ENCODE_OUTPUT = """\
     Running benches/encode_benchmark.rs (target/release/deps/encode_benchmark)
Running CI benchmark subset
Symbol size: 1280 bytes (without pre-built plan)
{without_plan}

Symbol size: 1280 bytes (with pre-built plan)
{with_plan}
""".format(
    without_plan=format_encode_lines(3979.543),
    with_plan=format_encode_lines(1953.123),
)


DECODE_OUTPUT = """\
     Running benches/decode_benchmark.rs (target/release/deps/decode_benchmark)
Running CI benchmark subset
Symbol size: 1280 bytes
{without_overhead}

{with_overhead}
""".format(
    without_overhead=format_decode_lines(0.0, 2274.321),
    with_overhead=format_decode_lines(5.0, 1562.543),
)


class CiBenchmarkTests(unittest.TestCase):
    def test_shared_ci_symbol_counts_match_required_sequence(self):
        counts = parse_ci_symbol_counts("CI_SYMBOL_COUNTS")
        decode_counts = parse_ci_symbol_counts("CI_DECODE_SYMBOL_COUNTS")

        self.assertEqual(counts, CI_SYMBOL_COUNTS)
        self.assertEqual(decode_counts, CI_DECODE_SYMBOL_COUNTS)
        self.assertEqual(counts, tuple(sorted(counts)))
        self.assertEqual(decode_counts, tuple(sorted(decode_counts)))
        self.assertEqual(counts[:-1], (10, 100, 250, 500, 1000, 2000, 5000, 10000))
        self.assertEqual(counts[-1], 20000)
        self.assertEqual(decode_counts, counts[:5])
        self.assertTrue(set(counts).issubset(ENCODE_DEFAULT_SYMBOL_COUNTS))
        self.assertEqual(decode_counts, DECODE_DEFAULT_SYMBOL_COUNTS)
        self.assertLess(len(counts), len(ENCODE_DEFAULT_SYMBOL_COUNTS))
        self.assertLess(len(decode_counts), len(counts))

    def test_ci_benchmark_sources_use_shared_symbol_counts(self):
        encode_source = read_bench_source("encode_benchmark.rs")
        decode_source = read_bench_source("decode_benchmark.rs")

        self.assertIn("mod ci_symbol_counts;", encode_source)
        self.assertIn("use ci_symbol_counts::CI_SYMBOL_COUNTS;", encode_source)
        self.assertNotIn("const CI_SYMBOL_COUNTS", encode_source)
        self.assertNotIn("CI_OVERHEAD_SYMBOL_COUNTS", encode_source)

        self.assertIn("mod ci_symbol_counts;", decode_source)
        self.assertIn(
            "use ci_symbol_counts::CI_DECODE_SYMBOL_COUNTS;",
            decode_source,
        )
        self.assertNotIn("const CI_DECODE_SYMBOL_COUNTS", decode_source)
        self.assertNotIn("CI_OVERHEAD_SYMBOL_COUNTS", decode_source)

    def test_encode_benchmark_restores_original_workload_and_counts(self):
        source = read_bench_source("encode_benchmark.rs")

        self.assertEqual(
            parse_usize_expr(source, "TARGET_TOTAL_BYTES"),
            "128 * 1024 * 1024",
        )
        self.assertEqual(
            parse_usize_array(source, "SYMBOL_COUNTS"),
            ENCODE_DEFAULT_SYMBOL_COUNTS,
        )
        self.assertEqual(
            parse_usize_expr(source, "CI_TARGET_TOTAL_BYTES"),
            "8 * 1024 * 1024",
        )

    def test_decode_benchmark_keeps_default_repair_only_counts_bounded(self):
        source = read_bench_source("decode_benchmark.rs")

        self.assertEqual(
            parse_usize_expr(source, "TARGET_TOTAL_BYTES"),
            "128 * 1024 * 1024",
        )
        self.assertEqual(
            parse_usize_array(source, "SYMBOL_COUNTS"),
            DECODE_DEFAULT_SYMBOL_COUNTS,
        )
        self.assertEqual(
            parse_usize_expr(source, "CI_TARGET_TOTAL_BYTES"),
            "8 * 1024 * 1024",
        )

    def test_ci_targets_remain_bounded(self):
        for bench_name in ("encode_benchmark.rs", "decode_benchmark.rs"):
            source = read_bench_source(bench_name)

            self.assertEqual(
                parse_usize_expr(source, "CI_TARGET_TOTAL_BYTES"),
                "8 * 1024 * 1024",
            )

    def test_ci_mode_only_selects_bounded_runtime_subset(self):
        for bench_name, ci_const in (
            ("encode_benchmark.rs", "CI_SYMBOL_COUNTS"),
            ("decode_benchmark.rs", "CI_DECODE_SYMBOL_COUNTS"),
        ):
            source = read_bench_source(bench_name)

            self.assertRegex(
                source,
                rf"(?s)if ci_mode_enabled\(\)\s*\{{.*CI_TARGET_TOTAL_BYTES.*{ci_const}\.as_slice\(\)",
            )
            self.assertRegex(
                source,
                r"(?s)\}\s*else\s*\{.*TARGET_TOTAL_BYTES.*SYMBOL_COUNTS\.as_slice\(\)",
            )

    def test_decode_benchmark_uses_repair_only_chunks_for_all_rows(self):
        source = read_bench_source("decode_benchmark.rs")

        self.assertIn(
            "let elements_and_overhead = (symbol_count as f64 * (1.0 + overhead)) as u32;",
            source,
        )
        self.assertIn(
            "let mut packets = encoder.repair_packets(0, iterations as u32 * elements_and_overhead);",
            source,
        )
        self.assertIn(
            "let start = packets.len() - elements_and_overhead as usize;",
            source,
        )
        self.assertIn("decoder.decode(packets.drain(start..))", source)
        self.assertNotRegex(source, r"fn\s+repair_packets\s*\(")
        for forbidden in (
            "EncodingPacket",
            "source_packets",
            "MAX_REPAIR_ONLY_SYMBOLS",
            "MAX_REPAIR_MIX_SYMBOLS",
            "duplicate_count",
            "received_per_iteration",
        ):
            self.assertNotIn(forbidden, source)

    def test_encode_benchmark_preserves_original_modes(self):
        source = read_bench_source("encode_benchmark.rs")

        self.assertIn("pre_plan: bool", source)
        self.assertIn("Some(SourceBlockEncodingPlan::generate(symbol_count as u16))", source)
        self.assertIn("SourceBlockEncoder::with_encoding_plan", source)
        self.assertIn("SourceBlockEncoder::new(1, &config, &data)", source)
        self.assertIn("let packets = encoder.repair_packets(0, 1);", source)

    def test_parse_custom_throughput_accepts_integer_and_decimal_mb(self):
        metrics = ci_benchmarks.parse_custom_throughput(
            "\n".join([ENCODE_OUTPUT, DECODE_OUTPUT])
        )

        expected_names = {
            f"encode_benchmark/encoded/without pre-built plan/symbols={count}"
            for count in CI_SYMBOL_COUNTS
        }
        expected_names.update(
            f"encode_benchmark/encoded/with pre-built plan/symbols={count}"
            for count in CI_SYMBOL_COUNTS
        )
        expected_names.update(
            f"decode_benchmark/decoded/1280 bytes/symbols={count}/overhead={overhead}%"
            for count in CI_DECODE_SYMBOL_COUNTS
            for overhead in ("0.0", "5.0")
        )
        self.assertEqual(set(metrics), expected_names)
        self.assertIn(
            "encode_benchmark/encoded/without pre-built plan/symbols=20000",
            metrics,
        )
        self.assertNotIn(
            "decode_benchmark/decoded/1280 bytes/symbols=20000/overhead=5.0%",
            metrics,
        )
        self.assertEqual(
            metrics[
                "encode_benchmark/encoded/without pre-built plan/symbols=10"
            ].mbits_per_second,
            3979.543,
        )
        self.assertEqual(
            metrics[
                "decode_benchmark/decoded/1280 bytes/symbols=100/overhead=5.0%"
            ].mbits_per_second,
            1562.543,
        )

    def test_parse_custom_throughput_accepts_original_integer_mb_lines(self):
        output = """\
     Running benches/encode_benchmark.rs (target/release/deps/encode_benchmark)
Symbol size: 1280 bytes (without pre-built plan)
symbol count = 10, encoded 127 MB in 0.123secs, throughput: 12.3Mbit/s
     Running benches/decode_benchmark.rs (target/release/deps/decode_benchmark)
Symbol size: 1280 bytes
symbol count = 10, decoded 127 MB in 0.456secs using 0.0% overhead, throughput: 45.6Mbit/s
"""

        metrics = ci_benchmarks.parse_custom_throughput(output)

        self.assertEqual(
            metrics[
                "encode_benchmark/encoded/without pre-built plan/symbols=10"
            ].mbits_per_second,
            12.3,
        )
        self.assertEqual(
            metrics[
                "decode_benchmark/decoded/1280 bytes/symbols=10/overhead=0.0%"
            ].mbits_per_second,
            45.6,
        )

    def test_custom_throughput_groups_share_ci_symbol_counts(self):
        metrics = ci_benchmarks.parse_custom_throughput(
            "\n".join([ENCODE_OUTPUT, DECODE_OUTPUT])
        )
        counts_by_group = {}
        for name in metrics:
            group_name, symbol_count = ci_benchmarks.benchmark_group_and_symbol_count(
                name
            )
            counts_by_group.setdefault(group_name, set()).add(symbol_count)

        self.assertEqual(
            counts_by_group,
            {
                "encode_benchmark/encoded/without pre-built plan": set(CI_SYMBOL_COUNTS),
                "encode_benchmark/encoded/with pre-built plan": set(CI_SYMBOL_COUNTS),
                "decode_benchmark/decoded/1280 bytes/overhead=0.0%": set(
                    CI_DECODE_SYMBOL_COUNTS
                ),
                "decode_benchmark/decoded/1280 bytes/overhead=5.0%": set(
                    CI_DECODE_SYMBOL_COUNTS
                ),
            },
        )

    def test_render_custom_table_includes_all_ci_symbol_rows(self):
        metrics = ci_benchmarks.parse_custom_throughput(
            "\n".join([ENCODE_OUTPUT, DECODE_OUTPUT])
        )

        table = ci_benchmarks.render_custom_table(metrics, metrics, "master", "PR")

        rows = [row for row in table.splitlines() if row.startswith("| `")]
        self.assertEqual(len(rows), 28)
        self.assertNotIn("additional throughput rows omitted", table)
        self.assertNotIn(
            "decode_benchmark/decoded/1280 bytes/symbols=20000/overhead=5.0%",
            table,
        )
        self.assertIn(
            "encode_benchmark/encoded/without pre-built plan/symbols=20000",
            table,
        )

    def test_format_mbits_preserves_custom_benchmark_precision(self):
        self.assertEqual(ci_benchmarks.format_mbits(3979.54321), "3979.543 Mbit/s")

    def test_render_custom_table_groups_rows_before_sorting_by_symbol_count(self):
        metrics = {
            name: ci_benchmarks.ThroughputMetric(name, 1000.0)
            for name in [
                "encode_benchmark/encoded/without pre-built plan/symbols=100",
                "encode_benchmark/encoded/without pre-built plan/symbols=10",
                "encode_benchmark/encoded/with pre-built plan/symbols=100",
                "encode_benchmark/encoded/with pre-built plan/symbols=10",
                "decode_benchmark/decoded/1280 bytes/symbols=50/overhead=0.0%",
            ]
        }

        table = ci_benchmarks.render_custom_table(metrics, metrics, "master", "PR")

        rows = table.splitlines()[2:]
        rendered_names = [row.split("|")[1].strip(" `") for row in rows]
        self.assertEqual(
            rendered_names,
            [
                "encode_benchmark/encoded/without pre-built plan/symbols=10",
                "encode_benchmark/encoded/without pre-built plan/symbols=100",
                "encode_benchmark/encoded/with pre-built plan/symbols=10",
                "encode_benchmark/encoded/with pre-built plan/symbols=100",
                "decode_benchmark/decoded/1280 bytes/symbols=50/overhead=0.0%",
            ],
        )

    def test_render_criterion_table_groups_rows_before_sorting_numeric_ids(self):
        metrics = {
            name: ci_benchmarks.CriterionMetric(name, 1000.0, None, None, None)
            for name in [
                "roundtrip/source_only/100",
                "roundtrip/source_only/10",
                "roundtrip/repair_only/50",
            ]
        }

        table = ci_benchmarks.render_criterion_table(metrics, metrics, "master", "PR")

        rows = table.splitlines()[2:]
        rendered_names = [row.split("|")[1].strip(" `") for row in rows]
        self.assertEqual(
            rendered_names,
            [
                "roundtrip/source_only/10",
                "roundtrip/source_only/100",
                "roundtrip/repair_only/50",
            ],
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
        self.assertEqual(len(run.throughput), 28)
        for _, env in calls[1:]:
            self.assertEqual(env["CARGO_TARGET_DIR"], str(target_dir))


if __name__ == "__main__":
    unittest.main()
