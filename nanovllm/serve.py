import argparse

from nanovllm.entrypoints.openai.api_server import create_app


def parse_args():
    parser = argparse.ArgumentParser(description="Serve nano-vLLM over HTTP")
    parser.add_argument("--model", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
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
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--distributed-backend", default="nccl", choices=["nccl", "gloo"])
    parser.add_argument("--distributed-init-method", default="tcp://127.0.0.1:2333")
    parser.add_argument("--cuda-device-offset", type=int, default=0)
    parser.add_argument("--ipc-shm-name", default="nanovllm")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--disable-prefix-cache", action="store_true")
    parser.add_argument("--prefix-cache-min-tokens", type=int, default=0)
    parser.add_argument("--max-cached-blocks", type=int, default=0)
    parser.add_argument("--max-cached-blocks-per-namespace", type=int, default=0)
    parser.add_argument("--kv-cache-dtype", default="auto", choices=["auto", "float32", "fp8_e4m3", "fp8_e5m2"])
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
    parser.add_argument("--max-active-requests", type=int, default=None)
    parser.add_argument("--max-pending-requests-per-namespace", type=int, default=0)
    parser.add_argument("--max-active-requests-per-namespace", type=int, default=0)
    parser.add_argument("--max-active-tokens", type=int, default=None)
    parser.add_argument("--max-active-tokens-per-namespace", type=int, default=None)
    parser.add_argument("--output-queue-size", type=int, default=1024)
    parser.add_argument("--request-timeout-s", type=float, default=None)
    parser.add_argument("--queue-timeout-s", type=float, default=None)
    parser.add_argument("--request-log-path", default=None)
    parser.add_argument("--max-pending-prompt-tokens", type=int, default=None)
    parser.add_argument("--metrics-window-size", type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required for nanovllm.serve") from exc
    app = create_app(
        model=args.model,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_prefill_chunk_tokens=args.max_prefill_chunk_tokens,
        min_prefill_chunk_tokens=args.min_prefill_chunk_tokens,
        scheduler_fairness=args.scheduler_fairness,
        kvcache_watermark_blocks=args.kvcache_watermark_blocks,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        distributed_backend=args.distributed_backend,
        distributed_init_method=args.distributed_init_method,
        cuda_device_offset=args.cuda_device_offset,
        ipc_shm_name=args.ipc_shm_name,
        enforce_eager=args.enforce_eager,
        enable_prefix_cache=not args.disable_prefix_cache,
        prefix_cache_min_tokens=args.prefix_cache_min_tokens,
        max_cached_blocks=args.max_cached_blocks,
        max_cached_blocks_per_namespace=args.max_cached_blocks_per_namespace,
        kv_cache_dtype=args.kv_cache_dtype,
        kv_compression=args.kv_compression,
        op_backend=args.op_backend,
        attention_backend=args.attention_backend,
        model_backend=args.model_backend,
        max_pending_requests=args.max_pending_requests,
        max_active_requests=args.max_active_requests,
        max_pending_requests_per_namespace=args.max_pending_requests_per_namespace,
        max_active_requests_per_namespace=args.max_active_requests_per_namespace,
        max_active_tokens=args.max_active_tokens,
        max_active_tokens_per_namespace=args.max_active_tokens_per_namespace,
        output_queue_size=args.output_queue_size,
        request_timeout_s=args.request_timeout_s,
        queue_timeout_s=args.queue_timeout_s,
        request_log_path=args.request_log_path,
        max_pending_prompt_tokens=args.max_pending_prompt_tokens,
        metrics_window_size=args.metrics_window_size,
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
