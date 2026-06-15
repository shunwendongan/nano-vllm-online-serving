def num_blocks_for_tokens(num_tokens: int, block_size: int):
    return (num_tokens + block_size - 1) // block_size


def validate_request_limits(
    prompt_len: int,
    max_tokens: int,
    max_model_len: int,
    block_size: int,
    num_kvcache_blocks: int,
):
    if prompt_len <= 0:
        raise ValueError("prompt must contain at least one token")
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    if prompt_len > max_model_len:
        raise ValueError(f"prompt length {prompt_len} exceeds max_model_len={max_model_len}")
    if prompt_len + max_tokens > max_model_len:
        raise ValueError(
            f"prompt length {prompt_len} + max_tokens {max_tokens} exceeds max_model_len={max_model_len}"
        )
    if num_kvcache_blocks > 0:
        needed_blocks = num_blocks_for_tokens(prompt_len + max_tokens, block_size)
        if needed_blocks > num_kvcache_blocks:
            raise ValueError(
                f"request needs {needed_blocks} KV blocks but only {num_kvcache_blocks} are available"
            )
