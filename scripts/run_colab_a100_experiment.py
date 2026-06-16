import argparse
import csv
import json
import os
import shlex
import subprocess
import time
import zipfile
from datetime import datetime
from pathlib import Path


REPO_URL = "https://github.com/shunwendongan/nano-vllm-online-serving.git"
COMMIT = "63ddd4fd16dc0935096dcf2e3720ef87fd79e1a3"
PATCH_RELATIVE_PATH = Path("artifacts/colab_a100_20260616/nano_vllm_colab_a100_20260616.patch")
METADATA_RELATIVE_PATH = Path("artifacts/colab_a100_20260616/nano_vllm_colab_a100_20260616_metadata.json")
DRIVE_ROOT = Path("reports/colab_a100_20260616")
RUNS_ROOT = DRIVE_ROOT / "runs"
WORK_ROOT = Path("/content/nano-vllm-colab-a100-work")
REPO_DIR = Path.cwd()
PATCH_PATH = REPO_DIR / PATCH_RELATIVE_PATH
METADATA_PATH = REPO_DIR / METADATA_RELATIVE_PATH
STATUS_PATH = DRIVE_ROOT / "branch_status.json"
FINAL_REPORT_PATH = DRIVE_ROOT / "colab_a100_benchmark_report_2026-06-16.md"
ZIP_PATH = DRIVE_ROOT / "colab_a100_results_2026-06-16.zip"


def now():
    return datetime.now().isoformat(timespec="seconds")


def run_cmd(command, cwd=None, timeout=None, check=True, log_path=None):
    printable = command if isinstance(command, str) else " ".join(shlex.quote(str(part)) for part in command)
    print(f"\n$ {printable}", flush=True)
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        shell=isinstance(command, str),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=timeout,
    )
    output = completed.stdout or ""
    if output:
        print(output, end="" if output.endswith("\n") else "\n", flush=True)
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"\n$ {printable}\n")
            handle.write(output)
            if output and not output.endswith("\n"):
                handle.write("\n")
            handle.write(f"[exit_code={completed.returncode}]\n")
    if check and completed.returncode != 0:
        raise RuntimeError(f"command failed with exit code {completed.returncode}: {printable}")
    return completed


def git_root(path: Path):
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    if completed.returncode == 0 and completed.stdout.strip():
        return Path(completed.stdout.strip())
    return None


def in_colab():
    try:
        import google.colab  # noqa: F401
        return True
    except Exception:
        return False


def configure_runtime(args):
    global REPO_URL, COMMIT, DRIVE_ROOT, RUNS_ROOT, WORK_ROOT, REPO_DIR
    global PATCH_PATH, METADATA_PATH, STATUS_PATH, FINAL_REPORT_PATH, ZIP_PATH

    REPO_URL = args.repo_url or os.environ.get("NANOVLLM_A100_REPO_URL", REPO_URL)
    COMMIT = args.git_ref or os.environ.get("NANOVLLM_A100_GIT_REF", COMMIT)
    output_root = (
        args.output_root
        or os.environ.get("NANOVLLM_A100_OUTPUT_ROOT")
        or ("/content/drive/MyDrive/nano-vllm-colab-a100/2026-06-16" if in_colab() else "reports/colab_a100_20260616")
    )
    DRIVE_ROOT = Path(output_root)
    RUNS_ROOT = DRIVE_ROOT / "runs"
    WORK_ROOT = Path(args.work_root or os.environ.get("NANOVLLM_A100_WORK_ROOT", str(WORK_ROOT)))

    script_root = git_root(Path(__file__).resolve().parent)
    cwd_root = git_root(Path.cwd())
    if args.clone_repo:
        REPO_DIR = WORK_ROOT / "nano-vllm-online-serving"
    else:
        REPO_DIR = Path(args.repo_dir).resolve() if args.repo_dir else (cwd_root or script_root or WORK_ROOT / "nano-vllm-online-serving")

    PATCH_PATH = Path(args.patch_path).resolve() if args.patch_path else REPO_DIR / PATCH_RELATIVE_PATH
    METADATA_PATH = Path(args.metadata_path).resolve() if args.metadata_path else REPO_DIR / METADATA_RELATIVE_PATH
    STATUS_PATH = DRIVE_ROOT / "branch_status.json"
    FINAL_REPORT_PATH = DRIVE_ROOT / "colab_a100_benchmark_report_2026-06-16.md"
    ZIP_PATH = DRIVE_ROOT / "colab_a100_results_2026-06-16.zip"


def prepare_output_root():
    if str(DRIVE_ROOT).startswith("/content/drive") and in_colab():
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)


def detect_gpu():
    completed = run_cmd(
        [
            "bash",
            "-lc",
            "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true",
        ],
        check=False,
    )
    gpu_text = completed.stdout.strip()
    a100_detected = "A100" in gpu_text
    (DRIVE_ROOT / "gpu_detect.txt").write_text(gpu_text + "\n", encoding="utf-8")
    print(f"GPU detected: {gpu_text or 'none'}")
    return gpu_text, a100_detected


def summary_patch_applied():
    bench_path = REPO_DIR / "bench_online.py"
    return bench_path.exists() and "server_admission_slo_rejections" in bench_path.read_text(encoding="utf-8")


def prepare_repo(clone_repo=False):
    if clone_repo or not (REPO_DIR / ".git").exists():
        WORK_ROOT.mkdir(parents=True, exist_ok=True)
        if REPO_DIR.exists():
            run_cmd(["git", "-C", str(REPO_DIR), "fetch", "origin"], check=False)
        else:
            run_cmd(["git", "clone", REPO_URL, str(REPO_DIR)])
        run_cmd(["git", "-C", str(REPO_DIR), "checkout", COMMIT])
        run_cmd(["git", "-C", str(REPO_DIR), "reset", "--hard", COMMIT])
        run_cmd(["git", "-C", str(REPO_DIR), "clean", "-fd", "-e", ".cache/", "-e", "reports/", "-e", "colab_generated_configs/"])

    if summary_patch_applied():
        print("Benchmark summary patch is already present; skipping patch apply.")
    elif PATCH_PATH.exists():
        run_cmd(["git", "-C", str(REPO_DIR), "apply", "--check", str(PATCH_PATH)])
        run_cmd(["git", "-C", str(REPO_DIR), "apply", str(PATCH_PATH)])
    else:
        raise FileNotFoundError(
            f"Patch is not applied and patch file is missing: {PATCH_PATH}. "
            "Run this script from the PR/source checkout, or pass --patch-path."
        )
    run_cmd(["git", "-C", str(REPO_DIR), "status", "--short"], log_path=DRIVE_ROOT / "repo_status_after_patch.txt")


def config_path(preferred, fallback):
    preferred_path = REPO_DIR / preferred
    if preferred_path.exists():
        return preferred_path
    return REPO_DIR / fallback


def write_generated_config(name, base_path, overrides):
    config_dir = REPO_DIR / "colab_generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    dest = config_dir / f"{name}.env"
    base_text = Path(base_path).read_text(encoding="utf-8")
    override_text = "\n".join(f"{key}={value}" for key, value in overrides.items())
    dest.write_text(base_text.rstrip() + "\n\n# Colab A100 generated overrides\n" + override_text + "\n", encoding="utf-8")
    return dest


def run_complete(run_dir):
    run_dir = Path(run_dir)
    return (
        (run_dir / "validation_output.txt").exists()
        and (run_dir / "gpu_smi.csv").exists()
        and any(run_dir.rglob("*_bench.json"))
    )


def start_gpu_sampler(gpu_csv):
    gpu_csv = Path(gpu_csv)
    gpu_csv.parent.mkdir(parents=True, exist_ok=True)
    gpu_csv.write_text("timestamp,name,utilization_gpu_pct,memory_used_mib,power_w,clocks_sm_mhz\n", encoding="utf-8")
    command = (
        "while true; do "
        "nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,power.draw,clocks.sm "
        "--format=csv,noheader,nounits >> "
        f"{shlex.quote(str(gpu_csv))}; "
        "sleep 1; "
        "done"
    )
    return subprocess.Popen(["bash", "-lc", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def stop_gpu_sampler(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()


def load_statuses():
    if STATUS_PATH.exists():
        return json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    return []


def save_statuses(statuses):
    STATUS_PATH.write_text(json.dumps(statuses, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def upsert_status(statuses, item):
    statuses[:] = [existing for existing in statuses if existing.get("name") != item.get("name")]
    statuses.append(item)
    save_statuses(statuses)


def run_branch(branch, statuses):
    experiment_name = branch["experiment_name"]
    run_id = branch["run_id"]
    run_dir = RUNS_ROOT / experiment_name / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    if run_complete(run_dir):
        upsert_status(statuses, {
            "name": branch["name"],
            "status": "skipped_complete",
            "run_dir": str(run_dir),
            "updated_at": now(),
        })
        print(f"Skipping completed branch: {branch['name']}")
        return

    config = write_generated_config(
        branch["name"],
        branch["base_config"],
        {
            "EXPERIMENT_NAME": experiment_name,
            "OUTPUT_ROOT": str(RUNS_ROOT),
            "RUN_ID": run_id,
            **branch["overrides"],
        },
    )
    upsert_status(statuses, {
        "name": branch["name"],
        "status": "running",
        "run_dir": str(run_dir),
        "config": str(config),
        "started_at": now(),
    })
    run_cmd("fuser -k 8000/tcp || true", cwd=REPO_DIR, check=False, log_path=run_dir / "branch.log")
    sampler = start_gpu_sampler(run_dir / "gpu_smi.csv")
    started = time.time()
    completed = None
    try:
        completed = run_cmd(
            ["python", "scripts/run_colab_config.py", "--config", str(config), "--run-id", run_id],
            cwd=REPO_DIR,
            timeout=int(branch.get("timeout_s", 7200)),
            check=False,
            log_path=run_dir / "branch.log",
        )
    finally:
        stop_gpu_sampler(sampler)
    elapsed_s = round(time.time() - started, 2)
    bench_reports = sorted(str(path) for path in run_dir.rglob("*_bench.json"))
    status = "success" if completed and completed.returncode == 0 and bench_reports else "failed"
    guardrail = ""
    if bench_reports:
        try:
            report = json.loads(Path(bench_reports[-1]).read_text(encoding="utf-8"))
            if float(report.get("error_rate") or 0.0) > 0.05:
                status = "capacity_boundary"
                guardrail = "error_rate_gt_5pct"
        except Exception as exc:
            guardrail = f"bench_report_parse_error={exc}"
    log_text = (run_dir / "branch.log").read_text(encoding="utf-8", errors="replace") if (run_dir / "branch.log").exists() else ""
    if "CUDA out of memory" in log_text or "OutOfMemoryError" in log_text:
        status = "capacity_boundary"
        guardrail = "oom"
    if "server exited before ready" in log_text or "fatal" in log_text.lower():
        status = "capacity_boundary" if status != "success" else status
        guardrail = guardrail or "server_fatal_or_not_ready"
    upsert_status(statuses, {
        "name": branch["name"],
        "status": status,
        "returncode": completed.returncode if completed else None,
        "guardrail": guardrail,
        "run_dir": str(run_dir),
        "bench_reports": bench_reports,
        "elapsed_s": elapsed_s,
        "updated_at": now(),
    })


def build_branches(a100_detected):
    baseline = config_path(
        "configs/cloudstudio/qwen3_native_a100_enterprise_serving.env",
        "configs/cloudstudio/qwen3_native_flash_attn_baseline.env",
    )
    high = config_path(
        "configs/cloudstudio/qwen3_native_a100_high_concurrency.env",
        "configs/cloudstudio/qwen3_native_flash_attn_baseline.env",
    )
    long_context = config_path(
        "configs/cloudstudio/qwen3_native_a100_long_context.env",
        "configs/cloudstudio/qwen3_native_flash_attn_baseline.env",
    )
    common = {
        "INSTALL_PROFILE": "native",
        "INSTALL_FLASH_ATTN": "auto",
        "SETUP_PYTHON": "python",
        "RUNNER_PYTHON": "python",
        "TORCH_INDEX_URL": "https://download.pytorch.org/whl/cu121",
        "HF_MODEL_ID": "Qwen/Qwen3-0.6B",
        "HF_LOCAL_DIR": ".cache/models/Qwen3-0.6B",
        "MODEL": ".cache/models/Qwen3-0.6B",
        "MODEL_BACKEND": "native",
        "ATTENTION_BACKEND": "flash_attn",
        "OP_BACKEND": "torch",
        "HOST": "127.0.0.1",
        "PORT": "8000",
        "HTTP_TIMEOUT_S": "420",
        "COMMAND_TIMEOUT_S": "3600",
        "SERVER_READY_TIMEOUT_S": "600",
        "QUEUE_TIMEOUT_S": "60",
        "REQUEST_TIMEOUT_S": "420",
    }
    branches = [{
        "name": "smoke_baseline",
        "experiment_name": "smoke_baseline",
        "run_id": "smoke-baseline",
        "base_config": baseline,
        "timeout_s": 3600,
        "overrides": {
            **common,
            "SCHEDULER_FAIRNESS": "alternate",
            "BENCHMARK_REQUESTS": "32",
            "BENCHMARK_CONCURRENCY": "4",
            "MAX_TOKENS": "64",
        },
    }]
    if not a100_detected:
        return branches
    for policy in ["alternate", "prefill_first", "decode_first"]:
        for concurrency in [32, 64]:
            branches.append({
                "name": f"scheduler_{policy}_c{concurrency}",
                "experiment_name": f"scheduler_{policy}",
                "run_id": f"c{concurrency}_mt128",
                "base_config": high,
                "timeout_s": 5400,
                "overrides": {
                    **common,
                    "SCHEDULER_FAIRNESS": policy,
                    "BENCHMARK_REQUESTS": "256",
                    "BENCHMARK_CONCURRENCY": str(concurrency),
                    "MAX_TOKENS": "128",
                    "MAX_NUM_SEQS": "512",
                    "MAX_NUM_BATCHED_TOKENS": "32768",
                },
            })
    for concurrency in [1, 2, 4, 8, 16, 32, 64]:
        for max_tokens in [32, 128, 256]:
            branches.append({
                "name": f"sweep_c{concurrency}_mt{max_tokens}",
                "experiment_name": f"sweep_mt{max_tokens}",
                "run_id": f"c{concurrency}",
                "base_config": high,
                "timeout_s": 5400,
                "overrides": {
                    **common,
                    "SCHEDULER_FAIRNESS": "alternate",
                    "BENCHMARK_REQUESTS": "128",
                    "BENCHMARK_CONCURRENCY": str(concurrency),
                    "MAX_TOKENS": str(max_tokens),
                    "MAX_NUM_SEQS": "512",
                    "MAX_NUM_BATCHED_TOKENS": "32768",
                },
            })
    branches.append({
        "name": "long_context_cache",
        "experiment_name": "long_context_cache",
        "run_id": "c16_mt128",
        "base_config": long_context,
        "timeout_s": 7200,
        "overrides": {
            **common,
            "SCHEDULER_FAIRNESS": "cache_aware_lpm",
            "BENCHMARK_REQUESTS": "128",
            "BENCHMARK_CONCURRENCY": "16",
            "MAX_TOKENS": "128",
            "MAX_MODEL_LEN": "16384",
            "CACHE_PROBE_REPETITIONS": "4096",
            "CACHE_PROBE_MAX_TOKENS": "16",
        },
    })
    return branches


def run_setup_and_tests():
    setup_config = config_path(
        "configs/cloudstudio/qwen3_native_a100_enterprise_serving.env",
        "configs/cloudstudio/qwen3_native_flash_attn_baseline.env",
    )
    run_cmd(
        ["bash", "scripts/setup_colab_gpu.sh", str(setup_config)],
        cwd=REPO_DIR,
        timeout=3600,
        log_path=DRIVE_ROOT / "setup_colab_gpu.log",
    )
    test_targets = (
        "tests.test_async_engine tests.test_api_server tests.test_bench_online "
        "tests.test_summarize_benchmarks tests.test_run_colab_config tests.test_validate_online_gpu"
    )
    pytest_targets = (
        "tests/test_async_engine.py tests/test_api_server.py tests/test_bench_online.py "
        "tests/test_summarize_benchmarks.py tests/test_run_colab_config.py tests/test_validate_online_gpu.py"
    )
    run_cmd(
        f"python -m pytest {pytest_targets} || python -m unittest {test_targets}",
        cwd=REPO_DIR,
        timeout=1800,
        log_path=DRIVE_ROOT / "touched_tests.log",
    )


def gpu_summary(run_dir):
    csv_path = Path(run_dir) / "gpu_smi.csv"
    if not csv_path.exists():
        return {}
    util = []
    mem = []
    power = []
    with csv_path.open(encoding="utf-8", errors="replace") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                util.append(float(row.get("utilization_gpu_pct") or 0))
                mem.append(float(row.get("memory_used_mib") or 0))
                power.append(float(row.get("power_w") or 0))
            except ValueError:
                continue
    if not util and not mem and not power:
        return {}
    return {
        "gpu_util_avg": round(sum(util) / len(util), 2) if util else "",
        "gpu_util_max": round(max(util), 2) if util else "",
        "mem_used_max_mib": round(max(mem), 2) if mem else "",
        "power_avg_w": round(sum(power) / len(power), 2) if power else "",
    }


def collect_benchmark_rows():
    rows = []
    for path in sorted(RUNS_ROOT.rglob("*_bench.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        metrics = report.get("server_metrics") or {}
        run_dir = path.parent
        row = {
            "run": str(path.relative_to(RUNS_ROOT)),
            "requests": report.get("requests", ""),
            "concurrency": report.get("concurrency", ""),
            "max_tokens": report.get("max_tokens", ""),
            "success": report.get("success", ""),
            "errors": report.get("errors", ""),
            "req_s": report.get("request_per_s", ""),
            "tok_s": report.get("completion_tok_per_s", ""),
            "ttft_p50_s": report.get("ttft_p50_s", ""),
            "ttft_p95_s": report.get("ttft_p95_s", ""),
            "ttft_p99_s": report.get("ttft_p99_s", ""),
            "latency_p50_s": report.get("latency_p50_s", ""),
            "latency_p95_s": report.get("latency_p95_s", ""),
            "latency_p99_s": report.get("latency_p99_s", ""),
            "server_recent_ttft_p95_s": report.get("server_recent_ttft_p95_s", metrics.get("recent_ttft_p95_s", "")),
            "server_recent_latency_p95_s": report.get(
                "server_recent_latency_p95_s",
                metrics.get("recent_latency_p95_s", ""),
            ),
            "server_queue_wait_p95_s": metrics.get("recent_queue_wait_p95_s", ""),
            "prefix_hit": report.get("server_prefix_cache_hit_rate", ""),
            "cache_read_tokens": report.get("cache_read_input_tokens", ""),
            "preemptions": report.get("server_preemptions", ""),
            "evictions": report.get("server_evictions", ""),
            "admission_rejections": report.get("server_admission_slo_rejections", ""),
            "stream_flush_tokens": report.get("server_stream_flush_tokens", ""),
        }
        row.update(gpu_summary(run_dir))
        rows.append(row)
    return rows


def markdown_table(rows, columns):
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")).replace("|", "\\|") for col in columns) + " |")
    return "\n".join(lines)


def generate_final_report(gpu_text, a100_detected):
    summary_json = DRIVE_ROOT / "summary.json"
    summary_md = DRIVE_ROOT / "summary.md"
    run_cmd(
        [
            "python",
            "scripts/summarize_benchmarks.py",
            "--root",
            str(RUNS_ROOT),
            "--output-json",
            str(summary_json),
            "--output-markdown",
            str(summary_md),
        ],
        cwd=REPO_DIR,
        check=False,
        log_path=DRIVE_ROOT / "summary_generation.log",
    )
    statuses = load_statuses()
    rows = collect_benchmark_rows()
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}
    sections = [
        "# Colab A100 Benchmark Report 2026-06-16",
        "",
        f"- Generated at: `{now()}`",
        f"- Repository: `{REPO_URL}`",
        f"- Git ref: `{COMMIT}`",
        f"- Patch SHA256: `{metadata.get('patch_sha256', '')}`",
        f"- GPU detected: `{gpu_text or 'none'}`",
        f"- A100 detected: `{a100_detected}`",
        "",
        "## Scope Notes",
        "",
        "- This run does not implement new SSE flush or admission-control behavior.",
        "- `server_admission_*` and `server_stream_flush_*` fields are passive report fields; blank values mean the current server did not emit those metrics.",
        "- If A100 was not detected, only smoke data should be interpreted.",
        "",
        "## Branch Status",
        "",
    ]
    if statuses:
        sections.append(markdown_table(statuses, ["name", "status", "guardrail", "returncode", "elapsed_s", "run_dir"]))
    else:
        sections.append("No branch status was recorded.")
    sections.extend(["", "## Benchmark Rows", ""])
    if rows:
        sections.append(markdown_table(rows, [
            "run",
            "requests",
            "concurrency",
            "success",
            "errors",
            "req_s",
            "tok_s",
            "ttft_p95_s",
            "latency_p95_s",
            "server_recent_ttft_p95_s",
            "server_recent_latency_p95_s",
            "server_queue_wait_p95_s",
            "prefix_hit",
            "cache_read_tokens",
            "preemptions",
            "evictions",
            "gpu_util_avg",
            "gpu_util_max",
            "mem_used_max_mib",
            "power_avg_w",
        ]))
    else:
        sections.append("No benchmark JSON files were found.")
    sections.extend([
        "",
        "## Interpretation Guide",
        "",
        "- Decode bottleneck: latency p95 grows much faster than TTFT p95 while completion tok/s flattens.",
        "- Queue/admission bottleneck: queue wait p95 rises before GPU utilization saturates.",
        "- KV pressure: preemptions, evictions, or memory pressure rise in the same run.",
        "- Prefix-cache benefit: cache read tokens and prefix hit rate rise while TTFT remains low.",
        "",
        "## Artifact Index",
        "",
        f"- Raw runs: `{RUNS_ROOT}`",
        f"- Aggregate summary JSON: `{summary_json}`",
        f"- Aggregate summary Markdown: `{summary_md}`",
        f"- Status JSON: `{STATUS_PATH}`",
        f"- Results zip: `{ZIP_PATH}`",
    ])
    FINAL_REPORT_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")


def zip_results():
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in DRIVE_ROOT.rglob("*"):
            if path == ZIP_PATH or path.is_dir():
                continue
            archive.write(path, path.relative_to(DRIVE_ROOT))


def main():
    args = parse_args()
    configure_runtime(args)
    print("Starting Colab A100 runner", flush=True)
    prepare_output_root()
    gpu_text, a100_detected = detect_gpu()
    prepare_repo(clone_repo=args.clone_repo)
    if not args.skip_setup_and_tests:
        run_setup_and_tests()
    branches = build_branches(a100_detected)
    statuses = load_statuses()
    for branch in branches:
        run_branch(branch, statuses)
    generate_final_report(gpu_text, a100_detected)
    zip_results()
    print(f"Final report: {FINAL_REPORT_PATH}")
    print(f"Results zip: {ZIP_PATH}")


def parse_args():
    parser = argparse.ArgumentParser(description="Run resumable nano-vLLM A100 benchmark experiments.")
    parser.add_argument("--output-root", default="", help="Output directory. Defaults to Drive in Colab, reports/ otherwise.")
    parser.add_argument("--repo-dir", default="", help="Existing repository checkout. Defaults to cwd or this script's repo.")
    parser.add_argument("--clone-repo", action="store_true", help="Clone --repo-url into --work-root before running.")
    parser.add_argument("--repo-url", default="", help="Git repository URL used with --clone-repo.")
    parser.add_argument("--git-ref", default="", help="Git commit or branch used with --clone-repo.")
    parser.add_argument("--work-root", default="", help="Scratch root for clone mode.")
    parser.add_argument("--patch-path", default="", help="Patch file to apply if the checkout is not already patched.")
    parser.add_argument("--metadata-path", default="", help="Patch metadata JSON path for the final report.")
    parser.add_argument("--skip-setup-and-tests", action="store_true", help="Skip dependency setup and touched tests.")
    return parser.parse_args()


if __name__ == "__main__":
    main()
