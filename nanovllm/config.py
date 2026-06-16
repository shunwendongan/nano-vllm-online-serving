import os
from dataclasses import dataclass
from transformers import AutoConfig

@dataclass
class Config:
    model: str  # 模型目录或名称
    max_num_batched_tokens: int = 16384  # 单批次最大token数
    max_num_seqs: int = 512  # 单批次最大序列数
    max_model_len: int = 4096  # 单序列最大长度
    gpu_memory_utilization: float = 0.9  # GPU显存利用率上限
    tensor_parallel_size: int = 1  # 张量并行进程数
    enforce_eager: bool = False  # 是否强制eager模式
    hf_config: AutoConfig | None = None  # transformers的模型配置
    eos: int = -1  # 终止token id
    kvcache_block_size: int = 256  # KV缓存块大小
    num_kvcache_blocks: int = -1  # KV缓存块数量
    kvcache_watermark_blocks: int = 0  # 有decode压力时为decode保留的KV block数量
    enable_prefix_cache: bool = True  # 是否启用前缀KV缓存复用
    prefix_cache_eviction_policy: str = "lru"  # prefix cache淘汰策略
    prefix_cache_min_tokens: int = 0  # 少于该长度的前缀不进入prefix cache，避免短prompt污染缓存
    max_cached_blocks: int = 0  # 全局最多保留的空闲cached block数，0表示不限
    max_cached_blocks_per_namespace: int = 0  # 单个cache namespace最多保留的空闲cached block数，0表示不限
    max_prefill_chunk_tokens: int = 2048  # 在线调度中单轮prefill最多处理的token数
    min_prefill_chunk_tokens: int = 1  # decode压力下动态分块的最小prefill token预算
    scheduler_fairness: str = "alternate"  # prefill/decode公平性策略
    kv_cache_dtype: str = "auto"  # auto/float32/fp8_e4m3/fp8_e5m2，默认保持模型dtype
    kv_compression: str = "none"  # KIVI/SnapKV/TurboQuant等实验入口，默认不启用
    op_backend: str = "torch"  # torch/triton/cuda_ext，默认使用稳定PyTorch算子
    attention_backend: str = "flash_attn"  # flash_attn/cuda_ext，默认使用稳定flash-attn路径
    model_backend: str = "native"  # native/hf_auto，native使用nano-vLLM自研模型执行路径
    distributed_backend: str = "nccl"  # torch.distributed后端，默认GPU服务使用NCCL
    distributed_init_method: str = "tcp://127.0.0.1:2333"  # 进程组初始化地址
    cuda_device_offset: int = 0  # 多实例部署时从第几张CUDA卡开始绑定rank
    ipc_shm_name: str = "nanovllm"  # tensor parallel worker之间的共享内存名

    def __post_init__(self):
        assert os.path.isdir(self.model)  # 检查模型目录是否存在
        assert self.kvcache_block_size % 256 == 0  # KV缓存块大小必须为256的倍数
        assert self.kvcache_watermark_blocks >= 0
        assert 1 <= self.tensor_parallel_size <= 8  # 并行进程数必须在1~8之间
        assert self.max_prefill_chunk_tokens > 0
        assert self.min_prefill_chunk_tokens > 0
        assert self.prefix_cache_eviction_policy in ("lru",)
        assert self.prefix_cache_min_tokens >= 0
        assert self.max_cached_blocks >= 0
        assert self.max_cached_blocks_per_namespace >= 0
        assert self.scheduler_fairness in ("alternate", "fcfs", "prefill_first", "decode_first", "cache_aware_lpm")
        assert self.kv_cache_dtype in ("auto", "float32", "fp8_e4m3", "fp8_e5m2")
        assert self.kv_compression in (
            "none",
            "kivi_exp",
            "snapkv_exp",
            "h2o_exp",
            "streamingllm_exp",
            "turboquant_exp",
            "paged_eviction_exp",
        )
        assert self.op_backend in ("torch", "triton", "cuda_ext")
        assert self.attention_backend in ("flash_attn", "cuda_ext")
        assert self.model_backend in ("native", "hf_auto")
        assert self.distributed_backend in ("nccl", "gloo")
        assert self.distributed_init_method
        assert self.cuda_device_offset >= 0
        assert self.ipc_shm_name
        if self.kv_compression != "none":
            raise NotImplementedError(
                f"kv_compression={self.kv_compression!r} is reserved for future experiments; "
                "the current stable path implements paged KV cache, prefix reuse, and TTL/namespace controls."
            )
        self.hf_config = AutoConfig.from_pretrained(self.model)  # 加载transformers模型配置
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)  # 限制最大长度不超过模型支持
        assert self.max_num_batched_tokens > 0
        self.max_prefill_chunk_tokens = min(self.max_prefill_chunk_tokens, self.max_num_batched_tokens)
        self.min_prefill_chunk_tokens = min(self.min_prefill_chunk_tokens, self.max_prefill_chunk_tokens)
