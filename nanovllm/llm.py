from nanovllm.engine.llm_engine import LLMEngine


class LLM(LLMEngine):
    def __init__(
        self,
        model: str,
        enable_prefix_cache: bool = True,
        max_prefill_chunk_tokens: int = 2048,
        kv_cache_dtype: str = "auto",
        op_backend: str = "torch",
        attention_backend: str = "flash_attn",
        model_backend: str = "native",
        **kwargs,
    ):
        super().__init__(
            model,
            enable_prefix_cache=enable_prefix_cache,
            max_prefill_chunk_tokens=max_prefill_chunk_tokens,
            kv_cache_dtype=kv_cache_dtype,
            op_backend=op_backend,
            attention_backend=attention_backend,
            model_backend=model_backend,
            **kwargs,
        )
