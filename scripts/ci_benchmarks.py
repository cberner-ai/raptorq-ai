#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


COMMENT_MARKER = "<!-- raptorq-benchmark-ci -->"
MAX_TABLE_ROWS = 80
MAX_FAILURE_LOG_LINES = 80
QUICK_BENCH_COMMANDS = [
    [
        "cargo",
        "bench",
        "--features",
        "benchmarking",
        "--bench",
        "codec_benchmark",
        "--",
        "--quick",
    ],
    [
        "cargo",
        "bench",
        "--features",
        "benchmarking",
        "--bench",
        "encode_benchmark",
        "--",
        "--ci",
    ],
    [
        "cargo",
        "bench",
        "--features",
        "benchmarking",
        "--bench",
        "decode_benchmark",
        "--",
        "--ci",
    ],
]


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str


@dataclass(frozen=True)
class CriterionMetric:
    name: str
    mean_ns: float
    lower_ns: Optional[float]
    upper_ns: Optional[float]
    throughput_bytes: Optional[float]


@dataclass(frozen=True)
class ThroughputMetric:
    name: str
    mbits_per_second: float


@dataclass(frozen=True)
class BenchmarkRun:
    label: str
    ref: str
    sha: str
    success: bool
    output: str
    target_dir: Path
    elapsed_seconds: float
    criterion: dict[str, CriterionMetric]
    throughput: dict[str, ThroughputMetric]


def run_command(cmd: list[str], env: Optional[dict[str, str]] = None) -> CommandResult:
    print(f"$ {shlex.join(cmd)}", flush=True)
    process = subprocess.Popen(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    lines: list[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        print(line, end="", flush=True)
        lines.append(line.rstrip("\n"))

    return CommandResult(process.wait(), "\n".join(lines))


def git_output(args: list[str]) -> str:
    return subprocess.check_output(
        ["git", *args],
        stderr=subprocess.DEVNULL,
        text=True,
    ).strip()


def resolve_ref(ref: str, short: bool = False) -> str:
    try:
        args = ["rev-parse"]
        if short:
            args.append("--short=12")
        args.append(ref)
        return git_output(args)
    except subprocess.CalledProcessError:
        return "unknown"


def parse_criterion_metrics(target_dir: Path) -> dict[str, CriterionMetric]:
    criterion_dir = target_dir / "criterion"
    if not criterion_dir.exists():
        return {}

    metrics: dict[str, CriterionMetric] = {}
    for estimates_path in sorted(criterion_dir.rglob("new/estimates.json")):
        relative_parts = estimates_path.relative_to(criterion_dir).parts
        if len(relative_parts) < 3 or relative_parts[-2:] != ("new", "estimates.json"):
            continue

        name = "/".join(relative_parts[:-2])
        try:
            estimates = json.loads(estimates_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue

        mean = estimates.get("mean", {})
        mean_ns = mean.get("point_estimate")
        if not isinstance(mean_ns, (int, float)):
            continue

        interval = mean.get("confidence_interval", {})
        lower_ns = interval.get("lower_bound")
        upper_ns = interval.get("upper_bound")
        if not isinstance(lower_ns, (int, float)):
            lower_ns = None
        if not isinstance(upper_ns, (int, float)):
            upper_ns = None

        throughput_bytes = None
        benchmark_path = estimates_path.parent / "benchmark.json"
        if benchmark_path.exists():
            try:
                benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
                throughput = benchmark.get("throughput")
                if isinstance(throughput, dict) and isinstance(
                    throughput.get("Bytes"), (int, float)
                ):
                    throughput_bytes = float(throughput["Bytes"])
            except (OSError, json.JSONDecodeError):
                pass

        metrics[name] = CriterionMetric(
            name=name,
            mean_ns=float(mean_ns),
            lower_ns=float(lower_ns) if lower_ns is not None else None,
            upper_ns=float(upper_ns) if upper_ns is not None else None,
            throughput_bytes=throughput_bytes,
        )

    return metrics


def parse_custom_throughput(output: str) -> dict[str, ThroughputMetric]:
    running_re = re.compile(r"Running\s+benches/([^/\s]+)\.rs")
    section_re = re.compile(r"Symbol size:\s*(?P<size>\d+) bytes(?: \((?P<mode>[^)]+)\))?")
    throughput_re = re.compile(
        r"symbol count = (?P<count>\d+), (?P<kind>encoded|decoded) "
        r"(?P<mb>[0-9]+(?:\.[0-9]+)?) MB in (?P<secs>[0-9.]+)secs"
        r"(?: using (?P<overhead>[0-9.]+)% overhead)?, "
        r"throughput: (?P<mbits>[0-9.]+)Mbit/s"
    )

    current_bench: Optional[str] = None
    current_mode: Optional[str] = None
    metrics: dict[str, ThroughputMetric] = {}

    for line in output.splitlines():
        running_match = running_re.search(line)
        if running_match:
            current_bench = running_match.group(1)
            current_mode = None
            continue

        section_match = section_re.search(line)
        if section_match and current_bench:
            current_mode = section_match.group("mode") or f"{section_match.group('size')} bytes"
            continue

        throughput_match = throughput_re.search(line)
        if not throughput_match or not current_bench:
            continue

        mode = current_mode or "default"
        name = (
            f"{current_bench}/{throughput_match.group('kind')}/{mode}/"
            f"symbols={throughput_match.group('count')}"
        )
        overhead = throughput_match.group("overhead")
        if overhead is not None:
            name = f"{name}/overhead={overhead}%"

        metrics[name] = ThroughputMetric(
            name=name,
            mbits_per_second=float(throughput_match.group("mbits")),
        )

    return metrics


def run_benchmarks(label: str, ref: str, target_dir: Path) -> BenchmarkRun:
    started = time.monotonic()
    sha = resolve_ref(ref)
    checkout_result = run_command(["git", "checkout", "--detach", ref])
    checkout_output = checkout_result.output

    if checkout_result.returncode != 0:
        elapsed = time.monotonic() - started
        return BenchmarkRun(
            label=label,
            ref=ref,
            sha=sha,
            success=False,
            output=checkout_output,
            target_dir=target_dir,
            elapsed_seconds=elapsed,
            criterion={},
            throughput={},
        )

    sha = resolve_ref("HEAD")
    target_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["CARGO_TARGET_DIR"] = str(target_dir)
    outputs = [checkout_output]
    success = True
    for command in QUICK_BENCH_COMMANDS:
        bench_result = run_command(command, env=env)
        outputs.append(bench_result.output)
        if bench_result.returncode != 0:
            success = False
            break

    output = "\n".join(part for part in outputs if part)
    elapsed = time.monotonic() - started

    return BenchmarkRun(
        label=label,
        ref=ref,
        sha=sha,
        success=success,
        output=output,
        target_dir=target_dir,
        elapsed_seconds=elapsed,
        criterion=parse_criterion_metrics(target_dir),
        throughput=parse_custom_throughput(output),
    )


def format_duration(ns: Optional[float]) -> str:
    if ns is None:
        return "n/a"
    if ns >= 1_000_000_000:
        return f"{ns / 1_000_000_000:.2f} s"
    if ns >= 1_000_000:
        return f"{ns / 1_000_000:.2f} ms"
    if ns >= 1_000:
        return f"{ns / 1_000:.2f} us"
    return f"{ns:.2f} ns"


def format_elapsed(seconds: float) -> str:
    if seconds >= 60:
        minutes = int(seconds // 60)
        remainder = int(seconds % 60)
        return f"{minutes}m {remainder}s"
    return f"{seconds:.1f}s"


def format_mbits(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} Mbit/s"


def format_criterion_throughput(metric: Optional[CriterionMetric]) -> str:
    if metric is None or metric.throughput_bytes is None or metric.mean_ns <= 0:
        return "n/a"
    mib_per_second = metric.throughput_bytes / (metric.mean_ns / 1_000_000_000) / (1024 * 1024)
    return f"{mib_per_second:.1f} MiB/s"


def format_delta(
    head_value: Optional[float],
    base_value: Optional[float],
    lower_is_better: bool,
) -> str:
    if head_value is None or base_value is None or base_value == 0:
        return "n/a"

    delta = (head_value - base_value) / base_value * 100
    if abs(delta) < 0.05:
        return "0.0%"

    sign = "+" if delta > 0 else ""
    if lower_is_better:
        direction = "faster" if delta < 0 else "slower"
    else:
        direction = "higher" if delta > 0 else "lower"
    return f"{sign}{delta:.1f}% {direction}"


def escape_cell(value: str) -> str:
    return value.replace("|", "\\|")


def benchmark_group_and_symbol_count(name: str) -> tuple[str, Optional[int]]:
    symbol_match = re.search(r"(?:^|/)symbols=(\d+)(?=/|$)", name)
    if symbol_match:
        group_name = f"{name[: symbol_match.start()]}{name[symbol_match.end() :]}"
        group_name = group_name.removeprefix("/")
        return (group_name, int(symbol_match.group(1)))

    id_match = re.search(r"(?:^|/)(\d+)$", name)
    if id_match:
        group_name = name[: id_match.start()]
        return (group_name, int(id_match.group(1)))

    return (name, None)


def benchmark_within_group_sort_key(name: str) -> tuple[int, int, str]:
    _, symbol_count = benchmark_group_and_symbol_count(name)
    if symbol_count is None:
        return (1, 0, name)
    return (0, symbol_count, name)


def sorted_benchmark_names(
    base_metrics: dict[str, object],
    head_metrics: dict[str, object],
) -> list[str]:
    seen_names: set[str] = set()
    group_order: list[str] = []
    names_by_group: dict[str, list[str]] = {}

    for metrics in (base_metrics, head_metrics):
        for name in metrics:
            if name in seen_names:
                continue
            seen_names.add(name)

            group_name, _ = benchmark_group_and_symbol_count(name)
            if group_name not in names_by_group:
                group_order.append(group_name)
                names_by_group[group_name] = []
            names_by_group[group_name].append(name)

    return [
        name
        for group_name in group_order
        for name in sorted(
            names_by_group[group_name],
            key=benchmark_within_group_sort_key,
        )
    ]


def render_criterion_table(
    base_metrics: dict[str, CriterionMetric],
    head_metrics: dict[str, CriterionMetric],
    base_label: str,
    head_label: str,
) -> str:
    names = sorted_benchmark_names(base_metrics, head_metrics)
    if not names:
        return "No Criterion result files were found."

    rows = [
        f"| Benchmark | {base_label} mean | {head_label} mean | Change | {head_label} throughput |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name in names[:MAX_TABLE_ROWS]:
        base = base_metrics.get(name)
        head = head_metrics.get(name)
        rows.append(
            "| "
            f"`{escape_cell(name)}` | "
            f"{format_duration(base.mean_ns if base else None)} | "
            f"{format_duration(head.mean_ns if head else None)} | "
            f"{format_delta(head.mean_ns if head else None, base.mean_ns if base else None, True)} | "
            f"{format_criterion_throughput(head)} |"
        )

    if len(names) > MAX_TABLE_ROWS:
        rows.append("")
        rows.append(f"_{len(names) - MAX_TABLE_ROWS} additional Criterion rows omitted._")

    return "\n".join(rows)


def render_custom_table(
    base_metrics: dict[str, ThroughputMetric],
    head_metrics: dict[str, ThroughputMetric],
    base_label: str,
    head_label: str,
) -> str:
    names = sorted_benchmark_names(base_metrics, head_metrics)
    if not names:
        return "No custom throughput lines were found."

    rows = [
        f"| Benchmark | {base_label} throughput | {head_label} throughput | Change |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name in names[:MAX_TABLE_ROWS]:
        base = base_metrics.get(name)
        head = head_metrics.get(name)
        rows.append(
            "| "
            f"`{escape_cell(name)}` | "
            f"{format_mbits(base.mbits_per_second if base else None)} | "
            f"{format_mbits(head.mbits_per_second if head else None)} | "
            f"{format_delta(head.mbits_per_second if head else None, base.mbits_per_second if base else None, False)} |"
        )

    if len(names) > MAX_TABLE_ROWS:
        rows.append("")
        rows.append(f"_{len(names) - MAX_TABLE_ROWS} additional throughput rows omitted._")

    return "\n".join(rows)


def tail_log(output: str) -> str:
    lines = output.strip().splitlines()
    if not lines:
        return "(no output captured)"
    return "\n".join(lines[-MAX_FAILURE_LOG_LINES:])


def render_failure_details(run: BenchmarkRun) -> str:
    return "\n".join(
        [
            f"<details><summary>{run.label} benchmark log tail</summary>",
            "",
            "```text",
            tail_log(run.output),
            "```",
            "",
            "</details>",
        ]
    )


def run_status(run: BenchmarkRun) -> str:
    return "passed" if run.success else "failed"


def render_comment(base_run: BenchmarkRun, head_run: BenchmarkRun) -> str:
    run_url = ""
    if all(
        os.environ.get(name)
        for name in ("GITHUB_SERVER_URL", "GITHUB_REPOSITORY", "GITHUB_RUN_ID")
    ):
        run_url = (
            f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}"
            f"/actions/runs/{os.environ['GITHUB_RUN_ID']}"
        )

    lines = [
        COMMENT_MARKER,
        "## Performance benchmarks",
        "",
        (
            f"Compared `{head_run.label}` `{resolve_ref(head_run.sha, short=True)}` "
            f"against `{base_run.label}` `{resolve_ref(base_run.sha, short=True)}`."
        ),
        "",
        f"| Ref | SHA | Status | Runtime | Criterion results | Custom throughput results |",
        "| --- | --- | --- | ---: | ---: | ---: |",
        (
            f"| `{base_run.label}` | `{resolve_ref(base_run.sha, short=True)}` | "
            f"{run_status(base_run)} | {format_elapsed(base_run.elapsed_seconds)} | "
            f"{len(base_run.criterion)} | {len(base_run.throughput)} |"
        ),
        (
            f"| `{head_run.label}` | `{resolve_ref(head_run.sha, short=True)}` | "
            f"{run_status(head_run)} | {format_elapsed(head_run.elapsed_seconds)} | "
            f"{len(head_run.criterion)} | {len(head_run.throughput)} |"
        ),
        "",
    ]

    if run_url:
        lines.extend([f"Full CI run: {run_url}", ""])

    if not base_run.success:
        lines.extend(
            [
                f"`{base_run.label}` benchmarks failed, so comparison values are unavailable where base data could not be parsed.",
                "",
            ]
        )
    if not head_run.success:
        lines.extend(
            [
                f"`{head_run.label}` benchmarks failed. Fix the benchmark build or run before using these results.",
                "",
            ]
        )

    if head_run.success or head_run.criterion:
        lines.extend(
            [
                "### Criterion",
                "",
                render_criterion_table(
                    base_run.criterion,
                    head_run.criterion,
                    base_run.label,
                    head_run.label,
                ),
                "",
            ]
        )

    if head_run.success or head_run.throughput:
        lines.extend(
            [
                "### Custom throughput",
                "",
                render_custom_table(
                    base_run.throughput,
                    head_run.throughput,
                    base_run.label,
                    head_run.label,
                ),
                "",
            ]
        )

    if not base_run.success:
        lines.extend([render_failure_details(base_run), ""])
    if not head_run.success:
        lines.extend([render_failure_details(head_run), ""])

    return "\n".join(lines).strip() + "\n"


def maybe_fetch_master(base_ref: str) -> None:
    if base_ref != "origin/master":
        return
    fetch_result = run_command(
        ["git", "fetch", "--no-tags", "origin", "master:refs/remotes/origin/master"]
    )
    if fetch_result.returncode != 0:
        print("Unable to refresh origin/master; using any existing local ref.", file=sys.stderr)


def write_comment_outputs(comment: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    comment_path = output_dir / "comment.md"
    comment_path.write_text(comment, encoding="utf-8")

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        Path(summary_path).write_text(comment.replace(f"{COMMENT_MARKER}\n", ""), encoding="utf-8")


def main() -> int:
    benchmark_dir = Path(os.environ.get("BENCHMARK_OUTPUT_DIR", "target/benchmark-ci")).resolve()
    base_ref = os.environ.get("BENCH_BASE_REF", "origin/master")
    head_ref = os.environ.get("BENCH_HEAD_REF", "HEAD")
    base_label = os.environ.get("BENCH_BASE_LABEL", "master")
    head_label = os.environ.get("BENCH_HEAD_LABEL", "PR")

    maybe_fetch_master(base_ref)

    base_run = run_benchmarks(base_label, base_ref, benchmark_dir / "base-target")
    head_run = run_benchmarks(head_label, head_ref, benchmark_dir / "head-target")

    comment = render_comment(base_run, head_run)
    write_comment_outputs(comment, benchmark_dir)

    return 0 if head_run.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
