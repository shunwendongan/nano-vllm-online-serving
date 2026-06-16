import argparse
import json
import os
import statistics
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed


def post_json(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = response.read()
    latency = time.perf_counter() - start
    return latency, json.loads(body)


def post_stream(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    first_token_at = None
    token_count = 0
    usage = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data or data == "[DONE]":
                continue
            payload = json.loads(data)
            if payload.get("token_id") is not None:
                token_count += 1
                if first_token_at is None:
                    first_token_at = time.perf_counter()
            else:
                chunk_text = ""
                choices = payload.get("choices") or []
                if choices:
                    choice = choices[0]
                    chunk_text = choice.get("text") or choice.get("delta", {}).get("content", "")
                if chunk_text:
                    token_count += 1
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
            if payload.get("usage"):
                usage = payload["usage"]
            if payload.get("finished"):
                break
    latency = time.perf_counter() - start
    ttft = (first_token_at - start) if first_token_at is not None else latency
    return latency, ttft, token_count, usage


def completion_tokens_from_response(response: dict, fallback: int = 0):
    usage = response.get("usage") or {}
    if "completion_tokens" in usage:
        return usage["completion_tokens"]
    if "token_ids" in response:
        return len(response["token_ids"])
    return fallback


def get_json(url: str, timeout: float):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.loads(response.read())
    except Exception:
            return None


def ensure_parent_dir(path: str):
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)


def percentile(values: list[float], p: float):
    if not values:
        return 0.0
    values = sorted(values)
    index = min(len(values) - 1, int(round((len(values) - 1) * p)))
    return values[index]


def build_payload(args):
    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
    }
    if args.cache_namespace:
        payload["cache_control"] = {
            "type": "ephemeral",
            "ttl": args.cache_ttl,
            "namespace": args.cache_namespace,
        }
        if args.cache_breakpoints:
            payload["cache_control"]["cache_breakpoints"] = [
                int(token_count)
                for token_count in args.cache_breakpoints.split(",")
                if token_count.strip()
            ]
        elif args.cacheable_prefix_tokens:
            payload["cache_control"]["cacheable_prefix_tokens"] = args.cacheable_prefix_tokens
    if args.request_namespace:
        payload["request_namespace"] = args.request_namespace
    return payload


def analyze_bottlenecks(report: dict):
    metrics = report.get("server_metrics") or {}
    backend = report.get("backend") or metrics.get("attention_backend") or metrics.get("model_backend", "")
    model_backend = metrics.get("model_backend") or ("hf_auto" if backend == "hf_auto" else "")
    latency_p95 = float(report.get("latency_p95_s") or 0.0)
    ttft_p95 = float(report.get("ttft_p95_s") or 0.0)
    completion_tok_per_s = float(report.get("completion_tok_per_s") or 0.0)
    errors = int(report.get("errors") or 0)
    success = int(report.get("success") or 0)
    analysis = []

    if model_backend == "hf_auto" or backend == "hf_auto":
        analysis.append({
            "severity": "info",
            "area": "backend_boundary",
            "finding": "hf_auto is a Transformers compatibility path, not native nano-vLLM continuous batching.",
            "evidence": f"backend={backend}, model_backend={model_backend or 'unknown'}",
            "recommendation": (
                "Report gpt-oss hf_auto smoke results separately. Use Qwen3 native + flash-attn for "
                "nano-vLLM scheduler, paged KV, prefix cache, and CUDA extension performance claims."
            ),
        })

    if errors:
        analysis.append({
            "severity": "high",
            "area": "request_failures",
            "finding": "Some benchmark requests failed.",
            "evidence": f"errors={errors}, error_status_counts={report.get('error_status_counts') or {}}",
            "recommendation": (
                "Check GPU memory, server log tail, model download/auth failures, HTTP timeout, and Colab "
                "runtime instability before interpreting throughput."
            ),
        })

    if success and ttft_p95 and latency_p95 and ttft_p95 >= latency_p95 * 0.6:
        analysis.append({
            "severity": "medium",
            "area": "ttft",
            "finding": "First-token latency dominates end-to-end latency.",
            "evidence": f"ttft_p95_s={ttft_p95}, latency_p95_s={latency_p95}",
            "recommendation": (
                "Run a warmup request, shorten or stabilize the prompt, inspect prefill/model placement, and "
                "check whether Transformers offloaded any weights to CPU or disk."
            ),
        })
    elif success and latency_p95 and ttft_p95 and latency_p95 >= max(ttft_p95 * 3, ttft_p95 + 1.0):
        analysis.append({
            "severity": "medium",
            "area": "decode_throughput",
            "finding": "Decode latency dominates after first token.",
            "evidence": (
                f"latency_p95_s={latency_p95}, ttft_p95_s={ttft_p95}, "
                f"completion_tok_per_s={completion_tok_per_s}"
            ),
            "recommendation": (
                "Reduce max_tokens or concurrency to find the knee point. For hf_auto, treat slow decode as "
                "Transformers backend overhead; native MoE/MXFP4 support is required before attributing this "
                "to nano-vLLM scheduler performance."
            ),
        })

    if success and completion_tok_per_s <= 0:
        analysis.append({
            "severity": "medium",
            "area": "token_accounting",
            "finding": "Successful requests produced no measured completion throughput.",
            "evidence": f"success={success}, completion_tokens={report.get('completion_tokens')}",
            "recommendation": "Verify streaming chunks, usage.completion_tokens, endpoint choice, and max_tokens.",
        })

    if model_backend != "hf_auto":
        prefix_hit_rate = float(report.get("server_prefix_cache_hit_rate") or 0.0)
        cache_reads = int(report.get("cache_read_input_tokens") or 0)
        cache_creates = int(report.get("cache_creation_input_tokens") or 0)
        if cache_creates and prefix_hit_rate <= 0 and cache_reads <= 0:
            analysis.append({
                "severity": "medium",
                "area": "prefix_cache",
                "finding": "Prefix cache was written but no warm-cache reuse was observed.",
                "evidence": (
                    f"cache_creation_input_tokens={cache_creates}, cache_read_input_tokens={cache_reads}, "
                    f"server_prefix_cache_hit_rate={prefix_hit_rate}"
                ),
                "recommendation": (
                    "Reuse the same cache namespace and stable prompt prefix, add explicit cache breakpoints, "
                    "and inspect /cache/inspect miss reasons."
                ),
            })

    preemptions = int(report.get("server_preemptions") or 0)
    evictions = int(report.get("server_evictions") or 0)
    if preemptions or evictions:
        analysis.append({
            "severity": "medium",
            "area": "kv_pressure",
            "finding": "KV block pressure was observed during the run.",
            "evidence": f"preemptions={preemptions}, evictions={evictions}",
            "recommendation": (
                "Lower concurrency or max_tokens, increase GPU memory utilization cautiously, tune "
                "kvcache_watermark_blocks, and compare scheduler policies."
            ),
        })

    if not analysis:
        analysis.append({
            "severity": "info",
            "area": "summary",
            "finding": "No obvious bottleneck was detected from the available benchmark fields.",
            "evidence": "Run native Qwen3 policy/cache experiments for deeper scheduler and paged-KV conclusions.",
            "recommendation": "Increase requests/concurrency gradually and compare cold vs warm prefix-cache runs.",
        })
    return analysis


def run(args):
    endpoint = args.endpoint
    if args.stream and endpoint == "/generate":
        endpoint = "/generate_stream"
    base_url = args.url.rstrip("/")
    url = base_url + endpoint
    payload = build_payload(args)
    latencies = []
    ttfts = []
    completion_tokens = 0
    prompt_tokens = 0
    input_tokens = 0
    cache_read_input_tokens = 0
    cache_creation_input_tokens = 0
    errors = 0
    error_status_counts = {}
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        if args.stream:
            futures = [pool.submit(post_stream, url, payload, args.timeout) for _ in range(args.requests)]
        else:
            futures = [pool.submit(post_json, url, payload, args.timeout) for _ in range(args.requests)]
        for future in as_completed(futures):
            try:
                if args.stream:
                    latency, ttft, token_count, usage = future.result()
                    latencies.append(latency)
                    ttfts.append(ttft)
                    completion_tokens += usage.get("completion_tokens", token_count)
                else:
                    latency, response = future.result()
                    latencies.append(latency)
                    usage = response.get("usage") or {}
                    completion_tokens += completion_tokens_from_response(response)
                prompt_tokens += usage.get("prompt_tokens", 0)
                input_tokens += usage.get("input_tokens", 0)
                cache_read_input_tokens += usage.get("cache_read_input_tokens", 0)
                cache_creation_input_tokens += usage.get("cache_creation_input_tokens", 0)
            except urllib.error.HTTPError as exc:
                errors += 1
                key = str(exc.code)
                error_status_counts[key] = error_status_counts.get(key, 0) + 1
                if args.verbose:
                    body = exc.read().decode("utf-8", errors="replace")
                    print(f"request failed: HTTP {exc.code}: {body}")
            except Exception as exc:
                errors += 1
                key = type(exc).__name__
                error_status_counts[key] = error_status_counts.get(key, 0) + 1
                if args.verbose:
                    print(f"request failed: {exc}")
    elapsed = time.perf_counter() - start
    success = len(latencies)
    metrics = get_json(base_url + "/metrics", args.timeout) if args.fetch_metrics else None
    report = {
        "url": url,
        "model_name": args.model_name,
        "backend": args.backend,
        "scheduler_policy": args.scheduler_policy,
        "stream": args.stream,
        "requests": args.requests,
        "success": success,
        "errors": errors,
        "error_rate": round(errors / args.requests, 6) if args.requests else 0.0,
        "error_status_counts": error_status_counts,
        "concurrency": args.concurrency,
        "elapsed_s": round(elapsed, 4),
        "request_per_s": round(success / elapsed, 4) if elapsed else 0.0,
        "completion_tokens": completion_tokens,
        "completion_tok_per_s": round(completion_tokens / elapsed, 4) if elapsed else 0.0,
        "prompt_tokens": prompt_tokens,
        "input_tokens": input_tokens,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": cache_creation_input_tokens,
        "latency_avg_s": round(statistics.mean(latencies), 4) if latencies else 0.0,
        "latency_p50_s": round(percentile(latencies, 0.50), 4),
        "latency_p95_s": round(percentile(latencies, 0.95), 4),
        "latency_p99_s": round(percentile(latencies, 0.99), 4),
        "latency_max_s": round(max(latencies), 4) if latencies else 0.0,
        "ttft_avg_s": round(statistics.mean(ttfts), 4) if ttfts else 0.0,
        "ttft_p50_s": round(percentile(ttfts, 0.50), 4),
        "ttft_p95_s": round(percentile(ttfts, 0.95), 4),
        "ttft_p99_s": round(percentile(ttfts, 0.99), 4),
        "server_metrics": metrics,
    }
    if metrics:
        report["backend"] = args.backend or metrics.get("attention_backend") or metrics.get("model_backend", "")
        report["scheduler_policy"] = args.scheduler_policy or metrics.get("scheduler_policy", "")
        report["server_prefix_cache_hit_rate"] = metrics.get("prefix_cache_hit_rate", 0.0)
        report["server_preemptions"] = metrics.get("preemptions", 0)
        report["server_evictions"] = metrics.get("evictions", 0)
        report["server_recent_ttft_p95_s"] = metrics.get("recent_ttft_p95_s", 0.0)
        report["server_recent_latency_p95_s"] = metrics.get("recent_latency_p95_s", 0.0)
        report["server_admission_slo_rejections"] = metrics.get("admission_slo_rejections", "")
        report["server_admission_overload_reason"] = metrics.get("admission_overload_reason", "")
        report["server_stream_flush_tokens"] = metrics.get("stream_flush_tokens", "")
        report["server_stream_flush_interval_s"] = metrics.get("stream_flush_interval_s", "")
    slo_failures = []
    if args.fail_on_errors and errors:
        slo_failures.append(f"errors={errors}")
    if args.slo_latency_p95_s is not None and report["latency_p95_s"] > args.slo_latency_p95_s:
        slo_failures.append(f"latency_p95_s={report['latency_p95_s']} > {args.slo_latency_p95_s}")
    if args.slo_ttft_p95_s is not None and report["ttft_p95_s"] > args.slo_ttft_p95_s:
        slo_failures.append(f"ttft_p95_s={report['ttft_p95_s']} > {args.slo_ttft_p95_s}")
    if args.min_completion_tok_per_s is not None and report["completion_tok_per_s"] < args.min_completion_tok_per_s:
        slo_failures.append(
            f"completion_tok_per_s={report['completion_tok_per_s']} < {args.min_completion_tok_per_s}"
        )
    report["slo_pass"] = not slo_failures
    report["slo_failures"] = slo_failures
    report["bottleneck_analysis"] = analyze_bottlenecks(report)
    report_json = json.dumps(report, indent=2)
    print(report_json)
    if args.report_json_path:
        ensure_parent_dir(args.report_json_path)
        with open(args.report_json_path, "w", encoding="utf-8") as f:
            f.write(report_json + "\n")
    if args.report_markdown_path:
        ensure_parent_dir(args.report_markdown_path)
        with open(args.report_markdown_path, "w", encoding="utf-8") as f:
            f.write(markdown_summary(report))
    return 1 if slo_failures else 0


def markdown_summary(report: dict):
    rows = [
        ("model", report.get("model_name") or ""),
        ("backend", report.get("backend") or ""),
        ("scheduler_policy", report.get("scheduler_policy") or ""),
        ("requests", report.get("requests")),
        ("success", report.get("success")),
        ("errors", report.get("errors")),
        ("concurrency", report.get("concurrency")),
        ("request_per_s", report.get("request_per_s")),
        ("completion_tok_per_s", report.get("completion_tok_per_s")),
        ("latency_p50_s", report.get("latency_p50_s")),
        ("latency_p95_s", report.get("latency_p95_s")),
        ("latency_p99_s", report.get("latency_p99_s")),
        ("ttft_p50_s", report.get("ttft_p50_s")),
        ("ttft_p95_s", report.get("ttft_p95_s")),
        ("ttft_p99_s", report.get("ttft_p99_s")),
        ("cache_read_input_tokens", report.get("cache_read_input_tokens")),
        ("cache_creation_input_tokens", report.get("cache_creation_input_tokens")),
        ("server_prefix_cache_hit_rate", report.get("server_prefix_cache_hit_rate")),
        ("server_preemptions", report.get("server_preemptions")),
        ("server_evictions", report.get("server_evictions")),
        ("server_admission_slo_rejections", report.get("server_admission_slo_rejections", "")),
        ("server_admission_overload_reason", report.get("server_admission_overload_reason", "")),
        ("server_stream_flush_tokens", report.get("server_stream_flush_tokens", "")),
        ("server_stream_flush_interval_s", report.get("server_stream_flush_interval_s", "")),
        ("slo_pass", report.get("slo_pass")),
    ]
    lines = [
        "# nano-vLLM Online Benchmark Summary",
        "",
        "| Metric | Value |",
        "| --- | --- |",
    ]
    for key, value in rows:
        lines.append(f"| {key} | {value} |")
    analysis = report.get("bottleneck_analysis") or analyze_bottlenecks(report)
    lines.extend(["", "## Bottleneck Analysis", ""])
    for item in analysis:
        lines.append(
            f"- **{item.get('severity', 'info')} / {item.get('area', 'unknown')}**: "
            f"{item.get('finding', '')} Evidence: {item.get('evidence', '')} "
            f"Recommendation: {item.get('recommendation', '')}"
        )
    failures = report.get("slo_failures") or []
    if failures:
        lines.extend(["", "## SLO Failures", ""])
        lines.extend(f"- {failure}" for failure in failures)
    return "\n".join(lines) + "\n"


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark a running nano-vLLM HTTP server")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--endpoint", default="/generate")
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--prompt", default="Explain KV cache optimization in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--cache-namespace", default="")
    parser.add_argument("--cache-ttl", default="5m")
    parser.add_argument("--cacheable-prefix-tokens", type=int, default=0)
    parser.add_argument("--cache-breakpoints", default="")
    parser.add_argument("--request-namespace", default="")
    parser.add_argument("--model-name", default="")
    parser.add_argument("--backend", default="")
    parser.add_argument("--scheduler-policy", default="")
    parser.add_argument("--report-json-path", default="")
    parser.add_argument("--report-markdown-path", default="")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--fetch-metrics", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--fail-on-errors", action="store_true")
    parser.add_argument("--slo-latency-p95-s", type=float, default=None)
    parser.add_argument("--slo-ttft-p95-s", type=float, default=None)
    parser.add_argument("--min-completion-tok-per-s", type=float, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
