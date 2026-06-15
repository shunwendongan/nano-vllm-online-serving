import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request


def request_json(method: str, url: str, payload: dict | None = None, timeout: float = 30.0):
    data = None
    headers = {}
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def request_text(method: str, url: str, timeout: float = 30.0):
    request = urllib.request.Request(url, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read().decode("utf-8")


def stream_events(url: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events = []
    first_token_latency_s = None
    started_at = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw_line in response:
            line = raw_line.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            payload_text = line[5:].strip()
            if not payload_text or payload_text == "[DONE]":
                continue
            event = json.loads(payload_text)
            if event.get("text") and first_token_latency_s is None:
                first_token_latency_s = time.perf_counter() - started_at
            events.append(event)
            if event.get("finished"):
                break
    return events, first_token_latency_s


def run_command(command: list[str], cwd: str, timeout: float | None = None):
    completed = subprocess.run(
        command,
        cwd=cwd,
        timeout=timeout,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "output": completed.stdout,
    }


def wait_ready(base_url: str, timeout_s: float, process: subprocess.Popen | None = None):
    deadline = time.time() + timeout_s
    last_error = None
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited before ready with code {process.returncode}")
        try:
            status = request_json("GET", f"{base_url}/readyz", timeout=5.0)
            if status.get("ready"):
                return status
        except Exception as exc:
            last_error = str(exc)
        time.sleep(1.0)
    raise RuntimeError(f"server did not become ready within {timeout_s}s; last_error={last_error}")


def build_runtime_check_command(args):
    return [
        args.python,
        "-m",
        "nanovllm.check_runtime",
        "--model",
        args.model,
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--cuda-device-offset",
        str(args.cuda_device_offset),
        "--distributed-backend",
        args.distributed_backend,
        "--model-backend",
        args.model_backend,
        "--attention-backend",
        args.attention_backend,
    ]


def build_server_command(args):
    command = [
        args.python,
        "-m",
        "nanovllm.serve",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--max-prefill-chunk-tokens",
        str(args.max_prefill_chunk_tokens),
        "--min-prefill-chunk-tokens",
        str(args.min_prefill_chunk_tokens),
        "--scheduler-fairness",
        args.scheduler_fairness,
        "--kvcache-watermark-blocks",
        str(args.kvcache_watermark_blocks),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--distributed-backend",
        args.distributed_backend,
        "--distributed-init-method",
        args.distributed_init_method,
        "--cuda-device-offset",
        str(args.cuda_device_offset),
        "--ipc-shm-name",
        args.ipc_shm_name,
        "--prefix-cache-min-tokens",
        str(args.prefix_cache_min_tokens),
        "--max-cached-blocks",
        str(args.max_cached_blocks),
        "--max-cached-blocks-per-namespace",
        str(args.max_cached_blocks_per_namespace),
        "--kv-cache-dtype",
        args.kv_cache_dtype,
        "--kv-compression",
        args.kv_compression,
        "--op-backend",
        args.op_backend,
        "--attention-backend",
        args.attention_backend,
        "--model-backend",
        args.model_backend,
        "--max-pending-requests",
        str(args.max_pending_requests),
        "--max-active-requests",
        str(args.max_active_requests),
        "--max-pending-requests-per-namespace",
        str(args.max_pending_requests_per_namespace),
        "--max-active-requests-per-namespace",
        str(args.max_active_requests_per_namespace),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--queue-timeout-s",
        str(args.queue_timeout_s),
        "--max-pending-prompt-tokens",
        str(args.max_pending_prompt_tokens),
        "--max-active-tokens",
        str(args.max_active_tokens),
        "--max-active-tokens-per-namespace",
        str(args.max_active_tokens_per_namespace),
        "--metrics-window-size",
        str(args.metrics_window_size),
        "--request-log-path",
        args.request_log_path,
    ]
    if args.enforce_eager:
        command.append("--enforce-eager")
    if args.disable_prefix_cache:
        command.append("--disable-prefix-cache")
    return command


def build_benchmark_command(args, base_url: str):
    report_json_path, report_markdown_path = benchmark_report_paths(args)
    return [
        args.python,
        "bench_online.py",
        "--url",
        base_url,
        "--stream",
        "--requests",
        str(args.benchmark_requests),
        "--concurrency",
        str(args.benchmark_concurrency),
        "--prompt",
        args.prompt,
        "--max-tokens",
        str(args.max_tokens),
        "--fetch-metrics",
        "--fail-on-errors",
        "--model-name",
        os.path.basename(os.path.normpath(args.model)) or args.model,
        "--backend",
        args.model_backend if args.model_backend == "hf_auto" else args.attention_backend,
        "--scheduler-policy",
        "hf_auto" if args.model_backend == "hf_auto" else args.scheduler_fairness,
        "--report-json-path",
        report_json_path,
        "--report-markdown-path",
        report_markdown_path,
        "--cache-namespace",
        args.cache_namespace,
        *(
            ["--request-namespace", args.request_namespace]
            if args.request_namespace
            else []
        ),
        "--timeout",
        str(args.http_timeout_s),
        *(
            ["--slo-latency-p95-s", str(args.slo_latency_p95_s)]
            if args.slo_latency_p95_s is not None
            else []
        ),
        *(
            ["--slo-ttft-p95-s", str(args.slo_ttft_p95_s)]
            if args.slo_ttft_p95_s is not None
            else []
        ),
        *(
            ["--min-completion-tok-per-s", str(args.min_completion_tok_per_s)]
            if args.min_completion_tok_per_s is not None
            else []
        ),
    ]


def _safe_name(value: str):
    result = "".join(ch.lower() if ch.isalnum() else "_" for ch in value)
    result = "_".join(part for part in result.split("_") if part)
    return result or "model"


def default_benchmark_report_prefix(args):
    model_hint = args.model.lower()
    if args.model_backend == "hf_auto" and "gpt-oss" in model_hint:
        return "gpt_oss_hf_auto_bench"
    model_name = os.path.basename(os.path.normpath(args.model)) or args.model
    backend = args.model_backend if args.model_backend == "hf_auto" else args.attention_backend
    return f"{_safe_name(model_name)}_{_safe_name(backend)}_bench"


def benchmark_report_paths(args):
    prefix = default_benchmark_report_prefix(args)
    json_path = args.benchmark_report_json_path or os.path.join(args.report_dir, f"{prefix}.json")
    markdown_path = args.benchmark_report_markdown_path or os.path.join(args.report_dir, f"{prefix}.md")
    return json_path, markdown_path


def build_cache_probe_payload(args):
    stable_prefix = " ".join([args.cache_probe_token] * args.cache_probe_repetitions)
    prompt = (
        f"{stable_prefix}\n\n"
        "Use the cached context above to answer in one short sentence."
    )
    return {
        "prompt": prompt,
        "max_tokens": args.cache_probe_max_tokens,
        "temperature": 0.0,
        "request_namespace": args.request_namespace or args.cache_namespace,
        "cache_control": {
            "type": "ephemeral",
            "ttl": "5m",
            "namespace": args.cache_namespace,
        },
    }


def validate_cache_prewarm(args, base_url: str):
    if args.model_backend == "hf_auto":
        return {
            "name": "cache_prewarm",
            "skipped": True,
            "reason": "hf_auto does not use native paged prefix cache",
        }
    if args.disable_prefix_cache or args.skip_cache_probe:
        return {"name": "cache_prewarm", "skipped": True}
    payload = build_cache_probe_payload(args)
    payload["max_tokens"] = 0
    response = request_json("POST", f"{base_url}/cache/prewarm", payload, timeout=args.http_timeout_s)
    usage = response.get("usage") or {}
    if response.get("finish_reason") != "cache_warmed":
        raise RuntimeError(f"/cache/prewarm did not report cache_warmed: {response}")
    if usage.get("prompt_tokens", 0) <= 0:
        raise RuntimeError(f"/cache/prewarm returned invalid usage: {usage}")
    return {
        "name": "cache_prewarm",
        "skipped": False,
        "finish_reason": response.get("finish_reason"),
        "cache_read_input_tokens": usage.get("cache_read_input_tokens", 0),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens", 0),
        "cache": response.get("cache"),
    }


def validate_prefix_cache_probe(args, base_url: str):
    if args.model_backend == "hf_auto":
        return {
            "name": "prefix_cache_probe",
            "skipped": True,
            "reason": "hf_auto does not use native paged prefix cache",
        }
    if args.disable_prefix_cache or args.skip_cache_probe:
        return {"name": "prefix_cache_probe", "skipped": True}
    payload = build_cache_probe_payload(args)
    before = request_json("GET", f"{base_url}/metrics", timeout=args.http_timeout_s)
    cold = request_json("POST", f"{base_url}/generate", payload, timeout=args.http_timeout_s)
    warm = request_json("POST", f"{base_url}/generate", payload, timeout=args.http_timeout_s)
    after = request_json("GET", f"{base_url}/metrics", timeout=args.http_timeout_s)
    warm_usage = warm.get("usage") or {}
    cache_read_input_tokens = warm_usage.get("cache_read_input_tokens", 0)
    hit_delta = after.get("prefix_cache_hits", 0) - before.get("prefix_cache_hits", 0)
    if not cold.get("token_ids"):
        raise RuntimeError("prefix cache cold probe returned no token_ids")
    if not warm.get("token_ids"):
        raise RuntimeError("prefix cache warm probe returned no token_ids")
    if cache_read_input_tokens <= 0 and hit_delta <= 0:
        raise RuntimeError(
            "prefix cache probe did not observe a warm cache hit; "
            f"cache_read_input_tokens={cache_read_input_tokens}, prefix_cache_hit_delta={hit_delta}"
        )
    return {
        "name": "prefix_cache_probe",
        "skipped": False,
        "cache_read_input_tokens": cache_read_input_tokens,
        "cache_creation_input_tokens": warm_usage.get("cache_creation_input_tokens", 0),
        "prefix_cache_hit_delta": hit_delta,
        "cached_blocks": after.get("cached_blocks"),
        "prompt_chars": len(payload["prompt"]),
    }


def start_server(args, cwd: str, log_file):
    command = build_server_command(args)
    return subprocess.Popen(
        command,
        cwd=cwd,
        text=True,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )


def terminate_server(process: subprocess.Popen):
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=20)


def run_validation(args):
    cwd = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_url = f"http://{args.host}:{args.port}"
    results = {"base_url": base_url, "checks": []}

    if not args.skip_runtime_check:
        runtime = run_command(build_runtime_check_command(args), cwd=cwd, timeout=args.command_timeout_s)
        results["checks"].append({"name": "runtime", **runtime})
        if runtime["returncode"] != 0:
            print(json.dumps(results, indent=2, ensure_ascii=False))
            return 1

    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False, suffix=".log") as server_log:
        server_log_path = server_log.name
        process = start_server(args, cwd, server_log)
        returncode = 0
        try:
            ready = wait_ready(base_url, args.server_ready_timeout_s, process=process)
            results["checks"].append({"name": "readyz", "status": ready})

            models = request_json("GET", f"{base_url}/v1/models", timeout=args.http_timeout_s)
            results["checks"].append({"name": "v1_models", "response": models})

            cache_prewarm = validate_cache_prewarm(args, base_url)
            results["checks"].append(cache_prewarm)

            payload = {
                "prompt": args.prompt,
                "max_tokens": args.max_tokens,
                "temperature": 0.0,
                "request_namespace": args.request_namespace or args.cache_namespace,
                "cache_control": {
                    "type": "ephemeral",
                    "ttl": "5m",
                    "namespace": args.cache_namespace,
                },
            }
            generate = request_json("POST", f"{base_url}/generate", payload, timeout=args.http_timeout_s)
            results["checks"].append({"name": "generate", "response": generate})
            if not generate.get("token_ids"):
                raise RuntimeError("/generate returned no token_ids")

            stream, ttft_s = stream_events(f"{base_url}/generate_stream", payload, timeout=args.http_timeout_s)
            results["checks"].append({
                "name": "generate_stream",
                "events": len(stream),
                "ttft_s": ttft_s,
                "final": stream[-1] if stream else None,
            })
            if not stream or not stream[-1].get("finished"):
                raise RuntimeError("/generate_stream did not finish")

            metrics = request_json("GET", f"{base_url}/metrics", timeout=args.http_timeout_s)
            results["checks"].append({"name": "metrics", "response": metrics})
            required_metrics = [
                "active_estimated_tokens",
                "pending_estimated_tokens",
                "active_requests_by_namespace",
                "pending_requests_by_namespace",
                "active_estimated_tokens_by_namespace",
                "pending_estimated_tokens_by_namespace",
                "prefix_cache_hits_by_namespace",
                "cache_read_input_tokens_by_namespace",
                "recent_ttft_p95_s",
                "recent_latency_p95_s",
                "recent_decode_tok_s",
                "prefix_cache_hit_rate",
            ]
            missing_metrics = [key for key in required_metrics if key not in metrics]
            if missing_metrics:
                raise RuntimeError(f"/metrics missing required keys: {missing_metrics}")
            if metrics.get("request_log_errors"):
                raise RuntimeError(f"request logging failed: {metrics.get('last_request_log_error')}")

            prometheus_metrics = request_text("GET", f"{base_url}/metrics/prometheus", timeout=args.http_timeout_s)
            results["checks"].append({
                "name": "metrics_prometheus",
                "sample": prometheus_metrics[:1000],
            })
            required_prometheus_metrics = [
                "nanovllm_active_requests",
                "nanovllm_prefix_cache_hit_rate",
                "nanovllm_recent_ttft_p95_s",
                "nanovllm_recent_decode_tok_s",
            ]
            missing_prometheus_metrics = [
                name for name in required_prometheus_metrics if name not in prometheus_metrics
            ]
            if missing_prometheus_metrics:
                raise RuntimeError(
                    f"/metrics/prometheus missing required metrics: {missing_prometheus_metrics}"
                )

            cache_stats = request_json("GET", f"{base_url}/cache/stats", timeout=args.http_timeout_s)
            results["checks"].append({"name": "cache_stats", "response": cache_stats})
            required_cache_stats = [
                "cached_blocks_by_namespace",
                "prefix_cache_hits_by_namespace",
                "cache_read_input_tokens_by_namespace",
                "cache_creation_input_tokens_by_namespace",
                "prefix_cache_hit_rate",
                "evictions",
                "global_quota_evictions",
                "duplicate_cache_blocks_skipped",
            ]
            missing_cache_stats = [key for key in required_cache_stats if key not in cache_stats]
            if missing_cache_stats:
                raise RuntimeError(f"/cache/stats missing required keys: {missing_cache_stats}")

            cache_inspect = request_json("GET", f"{base_url}/cache/inspect", timeout=args.http_timeout_s)
            results["checks"].append({"name": "cache_inspect", "response": cache_inspect})
            required_cache_inspect = [
                "cached_blocks",
                "prefix_cache_hit_rate",
                "prefix_cache_miss_reasons",
            ]
            missing_cache_inspect = [key for key in required_cache_inspect if key not in cache_inspect]
            if missing_cache_inspect:
                raise RuntimeError(f"/cache/inspect missing required keys: {missing_cache_inspect}")

            cache_probe = validate_prefix_cache_probe(args, base_url)
            results["checks"].append(cache_probe)

            if not args.skip_benchmark:
                benchmark = run_command(build_benchmark_command(args, base_url), cwd=cwd, timeout=args.command_timeout_s)
                results["checks"].append({"name": "benchmark", **benchmark})
                if benchmark["returncode"] != 0:
                    raise RuntimeError("benchmark failed")
        except Exception as exc:
            results["error"] = str(exc)
            returncode = 1
        finally:
            terminate_server(process)
            with open(server_log_path, encoding="utf-8", errors="replace") as f:
                results["server_log_tail"] = f.read()[-8000:]
            try:
                os.remove(server_log_path)
            except OSError:
                pass

    print(json.dumps(results, indent=2, ensure_ascii=False))
    return returncode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Validate nano-vLLM online serving on a CUDA GPU host")
    parser.add_argument("--model", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--distributed-backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--distributed-init-method", default="tcp://127.0.0.1:2333")
    parser.add_argument("--cuda-device-offset", type=int, default=0)
    parser.add_argument("--ipc-shm-name", default="nanovllm")
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-prefill-chunk-tokens", type=int, default=2048)
    parser.add_argument("--min-prefill-chunk-tokens", type=int, default=1)
    parser.add_argument(
        "--scheduler-fairness",
        default="alternate",
        choices=["alternate", "fcfs", "prefill_first", "decode_first", "cache_aware_lpm"],
    )
    parser.add_argument("--kvcache-watermark-blocks", type=int, default=0)
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--prefix-cache-min-tokens", type=int, default=0)
    parser.add_argument("--max-cached-blocks", type=int, default=0)
    parser.add_argument("--max-cached-blocks-per-namespace", type=int, default=0)
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "fp8_e4m3", "fp8_e5m2"])
    parser.add_argument(
        "--kv-compression",
        default="none",
        choices=[
            "none",
            "kivi_exp",
            "snapkv_exp",
            "h2o_exp",
            "streamingllm_exp",
            "turboquant_exp",
            "paged_eviction_exp",
        ],
    )
    parser.add_argument("--op-backend", default="torch", choices=["torch", "triton", "cuda_ext"])
    parser.add_argument("--attention-backend", default="flash_attn", choices=["flash_attn", "cuda_ext"])
    parser.add_argument("--model-backend", default="native", choices=["native", "hf_auto"])
    parser.add_argument("--max-pending-requests", type=int, default=1024)
    parser.add_argument("--max-active-requests", type=int, default=512)
    parser.add_argument("--max-pending-requests-per-namespace", type=int, default=0)
    parser.add_argument("--max-active-requests-per-namespace", type=int, default=0)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--queue-timeout-s", type=float, default=10.0)
    parser.add_argument("--max-pending-prompt-tokens", type=int, default=65536)
    parser.add_argument("--max-active-tokens", type=int, default=65536)
    parser.add_argument("--max-active-tokens-per-namespace", type=int, default=65536)
    parser.add_argument("--metrics-window-size", type=int, default=1024)
    parser.add_argument("--request-log-path", default="online_requests.jsonl")
    parser.add_argument("--prompt", default="Explain continuous batching and paged KV cache in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--cache-namespace", default="validation")
    parser.add_argument("--request-namespace", default="")
    parser.add_argument("--benchmark-requests", type=int, default=16)
    parser.add_argument("--benchmark-concurrency", type=int, default=4)
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--benchmark-report-json-path", default="")
    parser.add_argument("--benchmark-report-markdown-path", default="")
    parser.add_argument("--cache-probe-token", default="cache_probe_token")
    parser.add_argument("--cache-probe-repetitions", type=int, default=384)
    parser.add_argument("--cache-probe-max-tokens", type=int, default=8)
    parser.add_argument("--skip-cache-probe", action="store_true")
    parser.add_argument("--slo-latency-p95-s", type=float, default=None)
    parser.add_argument("--slo-ttft-p95-s", type=float, default=None)
    parser.add_argument("--min-completion-tok-per-s", type=float, default=None)
    parser.add_argument("--server-ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--http-timeout-s", type=float, default=120.0)
    parser.add_argument("--command-timeout-s", type=float, default=600.0)
    parser.add_argument("--skip-runtime-check", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--no-enforce-eager", dest="enforce_eager", action="store_false")
    parser.set_defaults(enforce_eager=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    try:
        raise SystemExit(run_validation(parse_args()))
    except (RuntimeError, urllib.error.URLError, urllib.error.HTTPError) as exc:
        print(json.dumps({"error": str(exc)}, indent=2, ensure_ascii=False))
        raise SystemExit(1)
