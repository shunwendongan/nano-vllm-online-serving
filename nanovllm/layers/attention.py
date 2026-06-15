import torch
from torch import nn

from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from nanovllm.layers.ops import store_kvcache
from nanovllm.utils.context import get_context


_ATTENTION_BACKEND = "flash_attn"


def set_attention_backend(backend: str):
    global _ATTENTION_BACKEND
    if backend not in ("flash_attn", "cuda_ext"):
        raise ValueError(f"unsupported attention backend: {backend}")
    _ATTENTION_BACKEND = backend


def get_attention_backend():
    return _ATTENTION_BACKEND


class Attention(nn.Module):
    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            if context.block_tables is not None:
                k, v = k_cache, v_cache
            o = flash_attn_varlen_func(
                q,
                k,
                v,
                max_seqlen_q=context.max_seqlen_q,
                cu_seqlens_q=context.cu_seqlens_q,
                max_seqlen_k=context.max_seqlen_k,
                cu_seqlens_k=context.cu_seqlens_k,
                softmax_scale=self.scale,
                causal=True,
                block_table=context.block_tables,
            )
        elif _ATTENTION_BACKEND == "cuda_ext":
            from nanovllm.layers.cuda_attention import paged_decode_attention

            o = paged_decode_attention(
                q,
                k_cache,
                v_cache,
                context.context_lens,
                context.block_tables,
                self.scale,
            )
        else:
            o = flash_attn_with_kvcache(
                q.unsqueeze(1),
                k_cache,
                v_cache,
                cache_seqlens=context.context_lens,
                block_table=context.block_tables,
                softmax_scale=self.scale,
                causal=True,
            )
        return o.view(-1, self.num_heads * self.head_dim)
