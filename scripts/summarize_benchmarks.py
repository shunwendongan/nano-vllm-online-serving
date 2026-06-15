import argparse
import json
from pathlib import Path


TABLE_COLUMNS = [
    "experiment",
    "run_id",
    "backend",
    "policy",
    "success",
    "errors",
    "req_s",
    "tok_s",
    "ttft_p95_s",
    "latency_p95_s",
    "cache_hit",
    "preemptions",
    "evictions",
    "slo_pass",
]


def _round(value, digits=4):
    if isinstance(value, (int, float)):
        return round(float(value), digits)
    return value


def _safe_get(report: dict, *keys, default=""):
    for key in keys:
        value = report.get(key)
        if value not in (None, ""):
            return value
    return default


def find_report_files(root: str | Path):
    root = Path(root)
    if not root.exists():
        return []
    return sorted(root.rglob("*_bench.json"))


def load_report(path: str | Path, root: str | Path):
    path = Path(path)
    root = Path(root)
    report = json.loads(path.read_text(encoding="utf-8"))
    relative = path.relative_to(root)
    parts = relative.parts
    experiment = parts[0] if len(parts) >= 1 else ""
    run_id = parts[1] if len(parts) >= 2 else ""
    report_file = parts[-1]
    metrics = report.get("server_metrics") or {}
    return {
        "experiment": experiment,
        "run_id": run_id,
        "report_file": report_file,
        "path": str(path),
        "model": report.get("model_name") or metrics.get("model_name") or "",
        "backend": _safe_get(report, "backend", default=metrics.get("attention_backend", "")),
        "policy": _safe_get(report, "scheduler_policy", default=metrics.get("scheduler_policy", "")),
        "requests": report.get("requests", 0),
        "success": report.get("success", 0),
        "errors": report.get("errors", 0),
        "req_s": _round(report.get("request_per_s", 0.0)),
        "tok_s": _round(report.get("completion_tok_per_s", 0.0)),
        "ttft_p95_s": _round(report.get("ttft_p95_s", 0.0)),
        "latency_p95_s": _round(report.get("latency_p95_s", 0.0)),
        "cache_hit": _round(report.get("server_prefix_cache_hit_rate", 0.0)),
        "preemptions": report.get("server_preemptions", 0),
        "evictions": report.get("server_evictions", 0),
        "slo_pass": report.get("slo_pass", ""),
        "bottleneck_analysis": report.get("bottleneck_analysis") or [],
    }


def load_reports(root: str | Path):
    return [load_report(path, root) for path in find_report_files(root)]


def _format_cell(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value).replace("|", "\\|")


def _best_row(rows, key, reverse=False):
    candidates = [
        row for row in rows
        if isinstance(row.get(key), (int, float)) and row.get("success", 0)
    ]
    if not candidates:
        return None
    return sorted(candidates, key=lambda row: row[key], reverse=reverse)[0]


def aggregate_findings(rows):
    findings = []
    failed = [row for row in rows if row.get("errors", 0)]
    if failed:
        findings.append({
            "severity": "high",
            "area": "request_failures",
            "finding": f"{len(failed)} run(s) reported request errors.",
            "recommendation": "Inspect validation_output.txt and server_log_tail before interpreting throughput.",
        })

    kv_pressure = [row for row in rows if row.get("preemptions", 0) or row.get("evictions", 0)]
    if kv_pressure:
        findings.append({
            "severity": "medium",
            "area": "kv_pressure",
            "finding": f"{len(kv_pressure)} run(s) reported preemptions or evictions.",
            "recommendation": (
                "Lower concurrency/max_tokens, raise GPU memory utilization cautiously, "
                "increase kvcache watermark for decode-heavy traffic, or move long-context tests to A100."
            ),
        })

    cache_misses = [
        row for row in rows
        if row.get("success", 0)
        and row.get("cache_hit", 0.0) == 0
        and "hf_auto" not in str(row.get("backend", ""))
    ]
    if cache_misses:
        findings.append({
            "severity": "medium",
            "area": "prefix_cache",
            "finding": "Native runs did not observe prefix-cache reuse.",
            "recommendation": (
                "Reuse stable prompt prefixes and cache namespaces, add explicit cache breakpoints, "
                "then inspect /cache/inspect miss reasons."
            ),
        })

    high_ttft = [
        row for row in rows
        if row.get("success", 0)
        and row.get("latency_p95_s", 0.0)
        and row.get("ttft_p95_s", 0.0) >= row.get("latency_p95_s", 0.0) * 0.6
    ]
    if high_ttft:
        findings.append({
            "severity": "medium",
            "area": "prefill_ttft",
            "finding": f"{len(high_ttft)} run(s) are dominated by first-token latency.",
            "recommendation": (
                "Warm the model, reduce prompt length, tune chunked prefill, and compare prefill_first "
                "versus decode_first under the same concurrency."
            ),
        })

    best_throughput = _best_row(rows, "tok_s", reverse=True)
    best_latency = _best_row(rows, "latency_p95_s", reverse=False)
    if best_throughput:
        findings.append({
            "severity": "info",
            "area": "best_throughput",
            "finding": (
                f"Highest completion throughput: {best_throughput['tok_s']} tok/s "
                f"({best_throughput['experiment']} / {best_throughput['run_id']})."
            ),
            "recommendation": "Use this run as the throughput baseline when comparing scheduler policies.",
        })
    if best_latency:
        findings.append({
            "severity": "info",
            "area": "best_latency",
            "finding": (
                f"Lowest p95 latency: {best_latency['latency_p95_s']} s "
                f"({best_latency['experiment']} / {best_latency['run_id']})."
            ),
            "recommendation": "Use this run as the latency baseline when tuning high-concurrency serving.",
        })
    return findings


def build_summary(rows, root: str | Path):
    root = Path(root)
    return {
        "root": str(root),
        "report_count": len(rows),
        "rows": rows,
        "findings": aggregate_findings(rows),
    }


def markdown_summary(summary: dict):
    lines = [
        "# CloudStudio Benchmark Summary",
        "",
        f"Report root: `{summary['root']}`",
        f"Report count: `{summary['report_count']}`",
        "",
    ]
    if not summary["rows"]:
        lines.append("No benchmark reports found yet.")
        return "\n".join(lines) + "\n"

    lines.extend([
        "## Runs",
        "",
        "| " + " | ".join(TABLE_COLUMNS) + " |",
        "| " + " | ".join(["---"] * len(TABLE_COLUMNS)) + " |",
    ])
    for row in summary["rows"]:
        lines.append("| " + " | ".join(_format_cell(row.get(column, "")) for column in TABLE_COLUMNS) + " |")

    lines.extend(["", "## Findings", ""])
    for finding in summary["findings"]:
        lines.append(
            f"- **{finding['severity']} / {finding['area']}**: {finding['finding']} "
            f"Recommendation: {finding['recommendation']}"
        )
    return "\n".join(lines) + "\n"


def write_summary(summary: dict, output_json: str | Path, output_markdown: str | Path):
    output_json = Path(output_json)
    output_markdown = Path(output_markdown)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output_markdown.write_text(markdown_summary(summary), encoding="utf-8")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Summarize nano-vLLM benchmark reports")
    parser.add_argument("--root", default="reports/cloudstudio", help="Report root to scan")
    parser.add_argument("--output-json", default="", help="Summary JSON path")
    parser.add_argument("--output-markdown", default="", help="Summary Markdown path")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    root = Path(args.root)
    summary = build_summary(load_reports(root), root)
    output_json = args.output_json or root / "summary.json"
    output_markdown = args.output_markdown or root / "summary.md"
    write_summary(summary, output_json, output_markdown)
    print(markdown_summary(summary), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
