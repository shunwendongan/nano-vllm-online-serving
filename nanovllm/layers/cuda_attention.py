from __future__ import annotations

import importlib
import os
import shutil


_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None


def _source_paths():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return [
        os.path.join(root, "kernels", "cuda_ext", "bindings.cpp"),
        os.path.join(root, "kernels", "cuda_ext", "attention.cu"),
    ]


def _cuda_cflags():
    flags = ["-O3"]
    host_compiler = (
        os.environ.get("NANOVLLM_CUDA_HOST_COMPILER")
        or shutil.which("g++-12")
        or shutil.which("g++-11")
    )
    if host_compiler:
        flags.append(f"-ccbin={host_compiler}")
    else:
        flags.append("-allow-unsupported-compiler")
    return flags


def _is_float32_dtype(dtype):
    return str(dtype) in ("torch.float32", "float32")


def _float32_contiguous(tensor, name: str, *, allow_cast: bool):
    dtype = getattr(tensor, "dtype", None)
    if hasattr(tensor, "contiguous"):
        tensor = tensor.contiguous()
    if dtype is None or _is_float32_dtype(dtype):
        return tensor, None
    if not allow_cast:
        raise RuntimeError(
            f"cuda_ext paged decode requires {name} to be float32; "
            "set kv_cache_dtype='float32' when attention_backend='cuda_ext'."
        )
    if not hasattr(tensor, "float"):
        raise RuntimeError(f"cuda_ext could not cast {name} to float32")
    tensor = tensor.float()
    if hasattr(tensor, "contiguous"):
        tensor = tensor.contiguous()
    return tensor, dtype


def _restore_dtype(tensor, dtype):
    if dtype is None or not hasattr(tensor, "to"):
        return tensor
    return tensor.to(dtype=dtype)


def _load_extension():
    global _EXTENSION, _EXTENSION_ERROR
    if _EXTENSION is not None:
        return _EXTENSION
    if _EXTENSION_ERROR is not None:
        raise RuntimeError(_extension_error_message()) from _EXTENSION_ERROR
    try:
        _EXTENSION = importlib.import_module("nanovllm_cuda_ext")
        return _EXTENSION
    except Exception as import_exc:
        try:
            import torch
            from torch.utils.cpp_extension import load
        except Exception as torch_exc:
            _EXTENSION_ERROR = torch_exc
            raise RuntimeError(_extension_error_message()) from torch_exc
        if not torch.cuda.is_available():
            _EXTENSION_ERROR = import_exc
            raise RuntimeError(_extension_error_message()) from import_exc
        try:
            _EXTENSION = load(
                name="nanovllm_cuda_ext",
                sources=_source_paths(),
                extra_cuda_cflags=_cuda_cflags(),
                verbose=False,
            )
            return _EXTENSION
        except Exception as build_exc:
            _EXTENSION_ERROR = build_exc
            raise RuntimeError(_extension_error_message()) from build_exc


def _extension_error_message():
    return (
        "attention_backend='cuda_ext' requires a compiled CUDA extension. "
        "Use attention_backend='flash_attn' on local non-CUDA machines, or build "
        "nanovllm/kernels/cuda_ext on a CUDA host before enabling cuda_ext."
    )


def is_cuda_ext_available() -> bool:
    try:
        _load_extension()
        return True
    except RuntimeError:
        return False


def dense_mha_attention(q, k, v, causal: bool = True):
    extension = _load_extension()
    q, output_dtype = _float32_contiguous(q, "q", allow_cast=True)
    k, _ = _float32_contiguous(k, "k", allow_cast=True)
    v, _ = _float32_contiguous(v, "v", allow_cast=True)
    return _restore_dtype(extension.dense_mha_forward(q, k, v, bool(causal)), output_dtype)


def gqa_mqa_attention(q, k, v, causal: bool = True):
    extension = _load_extension()
    q, output_dtype = _float32_contiguous(q, "q", allow_cast=True)
    k, _ = _float32_contiguous(k, "k", allow_cast=True)
    v, _ = _float32_contiguous(v, "v", allow_cast=True)
    return _restore_dtype(extension.gqa_mqa_forward(q, k, v, bool(causal)), output_dtype)


def streaming_gqa_attention(q, k, v, causal: bool = True):
    extension = _load_extension()
    q, output_dtype = _float32_contiguous(q, "q", allow_cast=True)
    k, _ = _float32_contiguous(k, "k", allow_cast=True)
    v, _ = _float32_contiguous(v, "v", allow_cast=True)
    return _restore_dtype(extension.streaming_gqa_forward(q, k, v, bool(causal)), output_dtype)


def paged_decode_attention(q, k_cache, v_cache, context_lens, block_tables, scale: float):
    extension = _load_extension()
    q, output_dtype = _float32_contiguous(q, "q", allow_cast=True)
    k_cache, _ = _float32_contiguous(k_cache, "k_cache", allow_cast=False)
    v_cache, _ = _float32_contiguous(v_cache, "v_cache", allow_cast=False)
    return _restore_dtype(extension.paged_decode_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        float(scale),
    ), output_dtype)
