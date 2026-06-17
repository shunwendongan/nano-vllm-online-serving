import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


VALUE_ARGS = {
    "MODEL": "--model",
    "PYTHON": "--python",
    "HOST": "--host",
    "PORT": "--port",
    "TENSOR_PARALLEL_SIZE": "--tensor-parallel-size",
    "GPU_MEMORY_UTILIZATION": "--gpu-memory-utilization",
    "DISTRIBUTED_BACKEND": "--distributed-backend",
    "DISTRIBUTED_INIT_METHOD": "--distributed-init-method",
    "CUDA_DEVICE_OFFSET": "--cuda-device-offset",
    "IPC_SHM_NAME": "--ipc-shm-name",
    "MAX_NUM_BATCHED_TOKENS": "--max-num-batched-tokens",
    "MAX_NUM_SEQS": "--max-num-seqs",
    "MAX_MODEL_LEN": "--max-model-len",
    "MAX_PREFILL_CHUNK_TOKENS": "--max-prefill-chunk-tokens",
    "MIN_PREFILL_CHUNK_TOKENS": "--min-prefill-chunk-tokens",
    "SCHEDULER_FAIRNESS": "--scheduler-fairness",
    "KVCACHE_WATERMARK_BLOCKS": "--kvcache-watermark-blocks",
    "PREFIX_CACHE_MIN_TOKENS": "--prefix-cache-min-tokens",
    "MAX_CACHED_BLOCKS": "--max-cached-blocks",
    "MAX_CACHED_BLOCKS_PER_NAMESPACE": "--max-cached-blocks-per-namespace",
    "KV_CACHE_DTYPE": "--kv-cache-dtype",
    "KV_COMPRESSION": "--kv-compression",
    "OP_BACKEND": "--op-backend",
    "ATTENTION_BACKEND": "--attention-backend",
    "MODEL_BACKEND": "--model-backend",
    "MAX_PENDING_REQUESTS": "--max-pending-requests",
    "MAX_ACTIVE_REQUESTS": "--max-active-requests",
    "MAX_PENDING_REQUESTS_PER_NAMESPACE": "--max-pending-requests-per-namespace",
    "MAX_ACTIVE_REQUESTS_PER_NAMESPACE": "--max-active-requests-per-namespace",
    "REQUEST_TIMEOUT_S": "--request-timeout-s",
    "QUEUE_TIMEOUT_S": "--queue-timeout-s",
    "MAX_PENDING_PROMPT_TOKENS": "--max-pending-prompt-tokens",
    "MAX_ACTIVE_TOKENS": "--max-active-tokens",
    "MAX_ACTIVE_TOKENS_PER_NAMESPACE": "--max-active-tokens-per-namespace",
    "METRICS_WINDOW_SIZE": "--metrics-window-size",
    "STREAM_INTERVAL": "--stream-interval",
    "PROMPT": "--prompt",
    "MAX_TOKENS": "--max-tokens",
    "CACHE_NAMESPACE": "--cache-namespace",
    "REQUEST_NAMESPACE": "--request-namespace",
    "BENCHMARK_REQUESTS": "--benchmark-requests",
    "BENCHMARK_CONCURRENCY": "--benchmark-concurrency",
    "BENCHMARK_REPORT_JSON_PATH": "--benchmark-report-json-path",
    "BENCHMARK_REPORT_MARKDOWN_PATH": "--benchmark-report-markdown-path",
    "CACHE_PROBE_TOKEN": "--cache-probe-token",
    "CACHE_PROBE_REPETITIONS": "--cache-probe-repetitions",
    "CACHE_PROBE_MAX_TOKENS": "--cache-probe-max-tokens",
    "SLO_LATENCY_P95_S": "--slo-latency-p95-s",
    "SLO_TTFT_P95_S": "--slo-ttft-p95-s",
    "MIN_COMPLETION_TOK_PER_S": "--min-completion-tok-per-s",
    "SERVER_READY_TIMEOUT_S": "--server-ready-timeout-s",
    "HTTP_TIMEOUT_S": "--http-timeout-s",
    "COMMAND_TIMEOUT_S": "--command-timeout-s",
}

TRUE_FLAGS = {
    "DISABLE_PREFIX_CACHE": "--disable-prefix-cache",
    "SKIP_CACHE_PROBE": "--skip-cache-probe",
    "SKIP_RUNTIME_CHECK": "--skip-runtime-check",
    "SKIP_BENCHMARK": "--skip-benchmark",
}

CONTROL_KEYS = {
    "EXPERIMENT_NAME",
    "OUTPUT_ROOT",
    "RUN_DIR",
    "RUN_ID",
    "RUNNER_PYTHON",
}

SETUP_ONLY_KEYS = {
    "HF_MODEL_ID",
    "HF_LOCAL_DIR",
    "INSTALL_PROFILE",
    "INSTALL_TORCH",
    "INSTALL_FLASH_ATTN",
    "SETUP_PYTHON",
    "MODEL_CACHE_DIR",
    "TORCH_INDEX_URL",
    "FLASH_ATTN_PACKAGE",
    "PIP_EXTRA_PACKAGES",
    "MAX_JOBS",
    "SKIP_MODEL_DOWNLOAD",
    "ALLOW_RUNTIME_CHECK_FAILURE",
}

FALSE_VALUES = {"", "0", "false", "no", "off"}
TRUE_VALUES = {"1", "true", "yes", "on"}


def _strip_inline_comment(value: str):
    quote = None
    for index, char in enumerate(value):
        if char in ("'", '"'):
            quote = None if quote == char else char
        if char == "#" and quote is None and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.strip()


def parse_env_line(line: str):
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    if line.startswith("export "):
        line = line[len("export "):].strip()
    if "=" not in line:
        raise ValueError(f"invalid config line without '=': {line}")
    key, value = line.split("=", 1)
    key = key.strip()
    value = _strip_inline_comment(value)
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1]
    value = os.path.expandvars(value)
    if not key:
        raise ValueError("empty config key")
    return key, value


def load_env_config(path: str | Path):
    config = {}
    path = Path(path)
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            parsed = parse_env_line(line)
        except ValueError as exc:
            raise ValueError(f"{path}:{lineno}: {exc}") from exc
        if parsed is None:
            continue
        key, value = parsed
        config[key] = value
    return config


def apply_env_overrides(config: dict[str, str], environ: dict[str, str] | None = None):
    environ = os.environ if environ is None else environ
    override_keys = set(VALUE_ARGS) | set(TRUE_FLAGS) | {"ENFORCE_EAGER"}
    addable_keys = {"STREAM_INTERVAL"}
    overrides = {}
    for key in sorted(override_keys):
        if key not in environ:
            continue
        if key not in config and key not in addable_keys:
            continue
        old_value = config.get(key)
        new_value = environ[key]
        if old_value == new_value:
            continue
        config[key] = new_value
        overrides[key] = {"old": old_value, "new": new_value}
    return overrides


def parse_bool(value: str, key: str):
    normalized = str(value).strip().lower()
    if normalized in TRUE_VALUES:
        return True
    if normalized in FALSE_VALUES:
        return False
    raise ValueError(f"{key} must be a boolean value, got {value!r}")


def validate_config_keys(config: dict[str, str], allow_unknown: bool = False):
    known = set(VALUE_ARGS) | set(TRUE_FLAGS) | CONTROL_KEYS | SETUP_ONLY_KEYS | {"ENFORCE_EAGER"}
    unknown = sorted(set(config) - known)
    if unknown and not allow_unknown:
        raise ValueError(f"unknown config keys: {', '.join(unknown)}")
    if not config.get("MODEL"):
        raise ValueError("MODEL is required in the Colab config")
    return unknown


def resolve_run_dir(config: dict[str, str], config_path: str | Path, run_id: str | None = None):
    if config.get("RUN_DIR"):
        return Path(config["RUN_DIR"])
    experiment_name = config.get("EXPERIMENT_NAME") or Path(config_path).stem
    output_root = config.get("OUTPUT_ROOT", "reports/colab")
    resolved_run_id = run_id or config.get("RUN_ID") or time.strftime("%Y%m%d-%H%M%S")
    return Path(output_root) / experiment_name / resolved_run_id


def build_validate_command(
    config: dict[str, str],
    run_dir: str | Path,
    validate_python: str | None = None,
):
    command = [
        validate_python or config.get("RUNNER_PYTHON") or sys.executable,
        "scripts/validate_online_gpu.py",
    ]
    for key, flag in VALUE_ARGS.items():
        value = config.get(key)
        if value is None or value == "":
            continue
        command.extend([flag, value])
    for key, flag in TRUE_FLAGS.items():
        if parse_bool(config.get(key, "false"), key):
            command.append(flag)
    if not parse_bool(config.get("ENFORCE_EAGER", "true"), "ENFORCE_EAGER"):
        command.append("--no-enforce-eager")

    run_dir = str(Path(run_dir))
    command.extend(["--report-dir", run_dir])
    if not config.get("REQUEST_LOG_PATH"):
        command.extend(["--request-log-path", str(Path(run_dir) / "online_requests.jsonl")])
    else:
        command.extend(["--request-log-path", config["REQUEST_LOG_PATH"]])
    return command


def write_run_metadata(
    config: dict[str, str],
    config_path: str | Path,
    run_dir: str | Path,
    command: list[str],
    dry_run: bool,
    env_overrides: dict[str, dict[str, str | None]] | None = None,
):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config_path": str(config_path),
        "dry_run": dry_run,
        "command": command,
        "command_shell": shlex.join(command),
        "config": config,
        "env_overrides": env_overrides or {},
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    (run_dir / "resolved_config.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (run_dir / "command.txt").write_text(shlex.join(command) + "\n", encoding="utf-8")
    return metadata


def run_from_config(args):
    config = load_env_config(args.config)
    env_overrides = apply_env_overrides(config)
    unknown = validate_config_keys(config, allow_unknown=args.allow_unknown)
    run_dir = resolve_run_dir(config, args.config, run_id=args.run_id)
    command = build_validate_command(config, run_dir, validate_python=args.validate_python)
    metadata = write_run_metadata(config, args.config, run_dir, command, args.dry_run, env_overrides=env_overrides)
    if unknown:
        metadata["ignored_unknown_keys"] = unknown
    print(json.dumps({
        "run_dir": str(run_dir),
        "dry_run": args.dry_run,
        "command": command,
        "command_shell": shlex.join(command),
        "env_overrides": env_overrides,
    }, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0
    completed = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parents[1],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    output = completed.stdout or ""
    print(output, end="" if output.endswith("\n") else "\n")
    Path(run_dir, "validation_output.txt").write_text(output, encoding="utf-8")
    return completed.returncode


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run Colab/GPU validation from a configs/colab/*.env file",
    )
    parser.add_argument("--config", required=True, help="Path to a Colab .env config")
    parser.add_argument("--run-id", default="", help="Optional stable run id for output paths")
    parser.add_argument("--dry-run", action="store_true", help="Print and record the command without running it")
    parser.add_argument("--allow-unknown", action="store_true", help="Ignore unknown config keys")
    parser.add_argument("--validate-python", default="", help="Python executable used to run validate_online_gpu.py")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    return run_from_config(args)


if __name__ == "__main__":
    raise SystemExit(main())
