import math

from nanovllm.sampling_params import SamplingParams


MAX_CACHE_BREAKPOINTS = 4


def _coerce_int(value, name: str):
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an integer") from exc


def _coerce_float(value, name: str):
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _coerce_bool(value, name: str):
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("1", "true", "yes", "on"):
            return True
        if lowered in ("0", "false", "no", "off"):
            return False
    raise ValueError(f"{name} must be a boolean")


def _coerce_non_negative_float(value, name: str):
    result = _coerce_float(value, name)
    if result < 0:
        raise ValueError(f"{name} must be non-negative")
    return result


def sampling_params_from_payload(payload: dict) -> SamplingParams:
    nested = payload.get("sampling_params") or {}
    temperature = _coerce_float(payload.get("temperature", nested.get("temperature", 1.0)), "temperature")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    max_tokens = _coerce_int(payload.get("max_tokens", nested.get("max_tokens", 64)), "max_tokens")
    if max_tokens < 0:
        raise ValueError("max_tokens must be non-negative")
    ignore_eos = payload.get("ignore_eos", nested.get("ignore_eos", False))
    return SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        ignore_eos=_coerce_bool(ignore_eos, "ignore_eos"),
    )


def prompt_from_payload(payload: dict):
    if "prompt_token_ids" in payload:
        return payload["prompt_token_ids"]
    return payload.get("prompt", "")


def request_options_from_payload(payload: dict) -> dict:
    options = {}
    if "request_timeout_s" in payload:
        options["request_timeout_s"] = _coerce_non_negative_float(
            payload["request_timeout_s"],
            "request_timeout_s",
        )
    if "queue_timeout_s" in payload:
        options["queue_timeout_s"] = _coerce_non_negative_float(payload["queue_timeout_s"], "queue_timeout_s")
    if "priority" in payload:
        options["priority"] = _coerce_int(payload["priority"], "priority")
    if payload.get("trace_id") is not None:
        options["trace_id"] = str(payload["trace_id"])
    request_namespace = (
        payload.get("request_namespace")
        or payload.get("tenant_id")
        or payload.get("tenant")
        or payload.get("namespace")
    )
    if request_namespace is not None:
        options["request_namespace"] = str(request_namespace)
    return options


def _ttl_from_cache_control(cache_control: dict | None):
    if not cache_control:
        return 300
    ttl = cache_control.get("ttl") or cache_control.get("ttl_seconds")
    if ttl in (None, "5m", "ephemeral"):
        return 300
    if ttl == "1h":
        return 3600
    return _coerce_non_negative_float(ttl, "cache ttl")


def _token_count_from_prompt(tokenizer, prompt):
    if isinstance(prompt, list):
        return len(prompt)
    if isinstance(prompt, str) and tokenizer is not None:
        return len(tokenizer.encode(prompt))
    return None


def _coerce_breakpoint_tokens(value):
    if value is None:
        return []
    if isinstance(value, (int, float, str)):
        return [int(value)]
    tokens = []
    for item in value:
        if isinstance(item, dict):
            item = (
                item.get("token_count")
                or item.get("tokens")
                or item.get("cacheable_prefix_tokens")
            )
        if item is not None:
            tokens.append(int(item))
    return tokens


def _normalize_breakpoint_tokens(tokens):
    normalized = sorted({int(token_count) for token_count in tokens if int(token_count) > 0})
    if len(normalized) > MAX_CACHE_BREAKPOINTS:
        normalized = normalized[-MAX_CACHE_BREAKPOINTS:]
    return normalized


def _set_breakpoint_options(options: dict, tokens):
    breakpoints = _normalize_breakpoint_tokens(tokens)
    options["cache_breakpoint_tokens"] = breakpoints or None
    options["cacheable_prefix_tokens"] = breakpoints[-1] if breakpoints else None
    return options


def _append_breakpoint(options: dict, token_count: int | None):
    if token_count is None:
        return options
    tokens = list(options.get("cache_breakpoint_tokens") or [])
    tokens.append(token_count)
    return _set_breakpoint_options(options, tokens)


def cache_options_from_payload(payload: dict, tokenizer=None, prompt=None) -> dict:
    cache_control = payload.get("cache_control") or {}
    cache_enabled = payload.get("cache_enabled")
    if cache_enabled is None:
        disable_cache = payload.get("disable_cache", False)
        if "disable_cache" in payload:
            disable_cache = _coerce_bool(disable_cache, "disable_cache")
        cache_enabled = not disable_cache
    else:
        cache_enabled = _coerce_bool(cache_enabled, "cache_enabled")
    if cache_control.get("type") in ("no-store", "none", "disabled"):
        cache_enabled = False
    no_store = cache_control.get("no_store", False)
    disable_cache = cache_control.get("disable_cache", False)
    if "no_store" in cache_control:
        no_store = _coerce_bool(no_store, "cache_control.no_store")
    if "disable_cache" in cache_control:
        disable_cache = _coerce_bool(disable_cache, "cache_control.disable_cache")
    if no_store or disable_cache:
        cache_enabled = False
    breakpoint_tokens = []
    explicit_breakpoint_tokens = []
    explicit_breakpoint_tokens.extend(_coerce_breakpoint_tokens(payload.get("cache_breakpoints")))
    explicit_breakpoint_tokens.extend(_coerce_breakpoint_tokens(cache_control.get("cache_breakpoints")))
    explicit_breakpoint_tokens.extend(_coerce_breakpoint_tokens(cache_control.get("breakpoints")))
    breakpoint_tokens.extend(explicit_breakpoint_tokens)
    cacheable_prefix_tokens = payload.get("cacheable_prefix_tokens")
    if cacheable_prefix_tokens is None:
        cacheable_prefix_tokens = cache_control.get("cacheable_prefix_tokens")
    if isinstance(cacheable_prefix_tokens, (list, tuple)):
        breakpoint_tokens.extend(_coerce_breakpoint_tokens(cacheable_prefix_tokens))
    elif cacheable_prefix_tokens is not None:
        breakpoint_tokens.append(int(cacheable_prefix_tokens))
    elif cache_control and not explicit_breakpoint_tokens:
        prompt_token_count = _token_count_from_prompt(tokenizer, prompt)
        if prompt_token_count is not None:
            breakpoint_tokens.append(prompt_token_count)
    ttl_seconds = payload.get("cache_ttl_seconds")
    if ttl_seconds is None:
        ttl_seconds = _ttl_from_cache_control(cache_control)
    namespace = payload.get("cache_namespace") or cache_control.get("namespace")
    options = {
        "cacheable_prefix_tokens": None,
        "cache_breakpoint_tokens": None,
        "cache_ttl_seconds": ttl_seconds,
        "cache_namespace": namespace,
        "cache_enabled": cache_enabled,
    }
    return _set_breakpoint_options(options, breakpoint_tokens)


def chat_prompt_from_messages(tokenizer, messages: list[dict], add_generation_prompt: bool = True) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    parts = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content", "")
        parts.append(f"{role}: {content}")
    if add_generation_prompt:
        parts.append("assistant:")
    return "\n".join(parts)


def chat_prompt_and_cache_options(tokenizer, messages: list[dict], payload: dict):
    prompt = chat_prompt_from_messages(tokenizer, messages)
    options = cache_options_from_payload(payload, tokenizer=tokenizer, prompt=prompt)
    breakpoint_tokens = list(options.get("cache_breakpoint_tokens") or [])
    for index, message in enumerate(messages):
        if message.get("cache_control"):
            cacheable_prompt = chat_prompt_from_messages(
                tokenizer,
                messages[:index + 1],
                add_generation_prompt=False,
            )
            token_count = _token_count_from_prompt(tokenizer, cacheable_prompt)
            if token_count is not None:
                breakpoint_tokens.append(token_count)
        content = message.get("content")
        if isinstance(content, list):
            for block_index, block in enumerate(content):
                if isinstance(block, dict) and block.get("cache_control"):
                    clipped = []
                    for msg in messages[:index]:
                        clipped.append(msg)
                    clipped_content = content[:block_index + 1]
                    clipped.append({**message, "content": clipped_content})
                    cacheable_prompt = chat_prompt_from_messages(
                        tokenizer,
                        clipped,
                        add_generation_prompt=False,
                    )
                    token_count = _token_count_from_prompt(tokenizer, cacheable_prompt)
                    if token_count is not None:
                        breakpoint_tokens.append(token_count)
    _set_breakpoint_options(options, breakpoint_tokens)
    return prompt, options
