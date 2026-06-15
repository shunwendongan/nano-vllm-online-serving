from __future__ import annotations

import importlib
import os


_EXTENSION = None
_EXTENSION_ERROR: Exception | None = None


def _source_paths():
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    return [
        os.path.join(root, "kernels", "cuda_ext", "bindings.cpp"),
        os.path.join(root, "kernels", "cuda_ext", "attention.cu"),
    ]


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
                extra_cuda_cflags=["-O3"],
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
    return _load_extension().dense_mha_forward(q, k, v, bool(causal))


def gqa_mqa_attention(q, k, v, causal: bool = True):
    return _load_extension().gqa_mqa_forward(q, k, v, bool(causal))


def streaming_gqa_attention(q, k, v, causal: bool = True):
    return _load_extension().streaming_gqa_forward(q, k, v, bool(causal))


def paged_decode_attention(q, k_cache, v_cache, context_lens, block_tables, scale: float):
    return _load_extension().paged_decode_attention(
        q,
        k_cache,
        v_cache,
        context_lens,
        block_tables,
        float(scale),
    )
