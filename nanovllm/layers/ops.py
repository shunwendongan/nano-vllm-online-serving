import torch
import torch.nn.functional as F
import triton
import triton.language as tl


_OPS_BACKEND = "torch"


def set_ops_backend(backend: str):
    global _OPS_BACKEND
    if backend not in ("torch", "triton", "cuda_ext"):
        raise ValueError(f"unsupported op backend: {backend}")
    _OPS_BACKEND = backend


def get_ops_backend():
    return _OPS_BACKEND


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    return x.to(orig_dtype).mul_(weight)


def add_rms_norm(x: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, eps: float):
    orig_dtype = x.dtype
    x = x.to(torch.float32).add_(residual.to(torch.float32))
    residual = x.to(orig_dtype)
    var = x.pow(2).mean(dim=-1, keepdim=True)
    x.mul_(torch.rsqrt(var + eps))
    return x.to(orig_dtype).mul_(weight), residual


def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    x, y = x.chunk(2, -1)
    return F.silu(x) * y


@triton.jit
def _store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id(0)
    offsets = tl.arange(0, D)
    key = tl.load(key_ptr + idx * key_stride + offsets)
    value = tl.load(value_ptr + idx * value_stride + offsets)
    slot = tl.load(slot_mapping_ptr + idx)
    cache_offsets = slot * D + offsets
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def _store_kvcache_torch(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    k_cache.view(-1, d)[slot_mapping] = key.reshape(n, d).to(k_cache.dtype)
    v_cache.view(-1, d)[slot_mapping] = value.reshape(n, d).to(v_cache.dtype)


def _store_kvcache_triton(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    _store_kvcache_kernel[(n,)](
        key,
        key.stride(0),
        value,
        value.stride(0),
        k_cache,
        v_cache,
        slot_mapping,
        d,
    )


def store_kvcache(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
):
    n, num_heads, head_dim = key.shape
    d = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == d and v_cache.stride(1) == d
    assert slot_mapping.numel() == n
    if _OPS_BACKEND == "torch":
        _store_kvcache_torch(key, value, k_cache, v_cache, slot_mapping)
    elif _OPS_BACKEND == "triton":
        _store_kvcache_triton(key, value, k_cache, v_cache, slot_mapping)
    else:
        raise NotImplementedError("cuda_ext backend is reserved until a compiled CUDA extension is added")
