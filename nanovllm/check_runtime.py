import argparse
import importlib
import os
import sys
from dataclasses import dataclass


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def _version_of(module):
    return getattr(module, "__version__", "unknown")


def check_python():
    version = ".".join(map(str, sys.version_info[:3]))
    if sys.version_info < (3, 10):
        return CheckResult("python", "fail", f"Python {version}; requires >=3.10,<3.13")
    if sys.version_info >= (3, 13):
        return CheckResult("python", "warn", f"Python {version}; project metadata requires <3.13")
    return CheckResult("python", "ok", f"Python {version}")


def check_model_path(model, model_backend: str = "native"):
    if not model:
        return CheckResult("model_path", "fail", "--model is required")
    if not os.path.isdir(model):
        if model_backend == "hf_auto":
            return CheckResult("model_path", "ok", f"HuggingFace model id or remote path: {model}")
        return CheckResult("model_path", "fail", f"not a directory: {model}")
    return CheckResult("model_path", "ok", model)


def check_backend_args(model_backend, attention_backend):
    results = []
    if model_backend not in ("native", "hf_auto"):
        results.append(CheckResult("model_backend", "fail", f"unsupported: {model_backend}"))
    else:
        results.append(CheckResult("model_backend", "ok", model_backend))
    if attention_backend not in ("flash_attn", "cuda_ext"):
        results.append(CheckResult("attention_backend", "fail", f"unsupported: {attention_backend}"))
    else:
        results.append(CheckResult("attention_backend", "ok", attention_backend))
    return results


def check_gpt_oss_compat(model):
    try:
        from nanovllm.models.gpt_oss_compat import inspect_gpt_oss_config
    except Exception as exc:
        return CheckResult("gpt_oss_compat", "fail", str(exc))
    report = inspect_gpt_oss_config(model).to_dict()
    if report["is_gpt_oss"]:
        status = "warn" if not report["native_supported"] else "ok"
        return CheckResult("gpt_oss_compat", status, str(report))
    return CheckResult("gpt_oss_compat", "ok", "not a gpt-oss config")


def check_transformers(model):
    try:
        transformers = importlib.import_module("transformers")
    except ImportError as exc:
        return CheckResult("transformers", "fail", str(exc))
    try:
        config = transformers.AutoConfig.from_pretrained(model)
    except Exception as exc:
        return CheckResult("hf_config", "fail", f"transformers {_version_of(transformers)}; {exc}")
    model_type = getattr(config, "model_type", "unknown")
    return CheckResult("hf_config", "ok", f"transformers {_version_of(transformers)}; model_type={model_type}")


def check_torch(tensor_parallel_size, cuda_device_offset, distributed_backend):
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        return [CheckResult("torch", "fail", str(exc))]

    results = [CheckResult("torch", "ok", f"torch {_version_of(torch)}")]
    if not torch.cuda.is_available():
        results.append(CheckResult("cuda", "fail", "torch.cuda.is_available() is false"))
        return results

    device_count = torch.cuda.device_count()
    required = cuda_device_offset + tensor_parallel_size
    if device_count < required:
        results.append(
            CheckResult(
                "cuda",
                "fail",
                f"device_count={device_count}, need at least {required} for offset={cuda_device_offset}, tp={tensor_parallel_size}",
            )
        )
    else:
        names = [torch.cuda.get_device_name(i) for i in range(cuda_device_offset, required)]
        results.append(CheckResult("cuda", "ok", f"device_count={device_count}; selected={names}"))

    dist = getattr(torch, "distributed", None)
    if dist is None or not dist.is_available():
        results.append(CheckResult("torch.distributed", "fail", "torch.distributed is unavailable"))
    elif distributed_backend == "nccl" and not dist.is_nccl_available():
        results.append(CheckResult("torch.distributed", "fail", "NCCL backend is unavailable"))
    elif distributed_backend == "gloo" and not dist.is_gloo_available():
        results.append(CheckResult("torch.distributed", "fail", "Gloo backend is unavailable"))
    else:
        results.append(CheckResult("torch.distributed", "ok", f"{distributed_backend} backend is available"))
    return results


def check_flash_attn():
    try:
        module = importlib.import_module("flash_attn")
    except ImportError as exc:
        return CheckResult("flash_attn", "fail", str(exc))
    missing = [
        name
        for name in ("flash_attn_varlen_func", "flash_attn_with_kvcache")
        if not hasattr(module, name)
    ]
    if missing:
        return CheckResult("flash_attn", "fail", f"missing symbols: {', '.join(missing)}")
    return CheckResult("flash_attn", "ok", f"flash_attn {_version_of(module)}")


def check_triton():
    try:
        module = importlib.import_module("triton")
    except ImportError as exc:
        return CheckResult("triton", "fail", str(exc))
    return CheckResult("triton", "ok", f"triton {_version_of(module)}")


def run_checks(args):
    results = [
        check_python(),
        check_model_path(args.model, args.model_backend),
    ]
    results.extend(check_backend_args(args.model_backend, args.attention_backend))
    results.append(check_gpt_oss_compat(args.model))
    if os.path.isdir(args.model) or args.model_backend == "hf_auto":
        results.append(check_transformers(args.model))
    results.extend(check_torch(args.tensor_parallel_size, args.cuda_device_offset, args.distributed_backend))
    if args.model_backend == "hf_auto":
        results.append(CheckResult("flash_attn", "skip", "not required for model_backend=hf_auto"))
    else:
        results.append(check_flash_attn())
    results.append(check_triton())
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Check whether this runtime can serve nano-vLLM")
    parser.add_argument("--model", required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--cuda-device-offset", type=int, default=0)
    parser.add_argument("--distributed-backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--model-backend", default="native", choices=["native", "hf_auto"])
    parser.add_argument("--attention-backend", default="flash_attn", choices=["flash_attn", "cuda_ext"])
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = run_checks(args)
    for result in results:
        print(f"[{result.status.upper()}] {result.name}: {result.detail}")
    return 1 if any(result.status == "fail" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
