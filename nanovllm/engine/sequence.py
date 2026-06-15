from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams = SamplingParams(),
        request_id: str | None = None,
        cacheable_prefix_tokens: int | None = None,
        cache_breakpoint_tokens: list[int] | None = None,
        cache_ttl_seconds: float | None = 300,
        cache_namespace: str | None = None,
        cache_enabled: bool = True,
    ):
        assert token_ids, "prompt must contain at least one token"
        self.seq_id = next(Sequence.counter)
        self.request_id = request_id or str(self.seq_id)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.prefill_target_tokens = self.num_prompt_tokens
        self.num_cached_tokens = 0
        self.num_computed_tokens = 0
        self.block_table: list[int] = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.cache_breakpoint_tokens = self._normalize_cache_breakpoints(
            cacheable_prefix_tokens,
            cache_breakpoint_tokens,
        )
        self.cacheable_prefix_tokens = (
            self.cache_breakpoint_tokens[-1]
            if self.cache_breakpoint_tokens
            else cacheable_prefix_tokens
        )
        self.cache_ttl_seconds = cache_ttl_seconds
        self.cache_namespace = cache_namespace or ""
        self.cache_enabled = cache_enabled
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0
        self.scheduled_prefill_start = 0
        self.scheduled_prefill_end = 0
        self.finish_reason: str | None = None

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def is_prompt_ready(self):
        return self.num_computed_tokens >= self.prefill_target_tokens

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return self.num_blocks_for_tokens(self.num_tokens)

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    @classmethod
    def num_blocks_for_tokens(cls, num_tokens: int):
        return (num_tokens + cls.block_size - 1) // cls.block_size

    @staticmethod
    def _normalize_cache_breakpoints(
        cacheable_prefix_tokens: int | None,
        cache_breakpoint_tokens: list[int] | None,
    ):
        breakpoints = []
        if cache_breakpoint_tokens:
            breakpoints.extend(int(token_count) for token_count in cache_breakpoint_tokens)
        if cacheable_prefix_tokens is not None:
            breakpoints.append(int(cacheable_prefix_tokens))
        return sorted({token_count for token_count in breakpoints if token_count > 0})

    def max_cacheable_tokens(self, up_to_token: int | None = None):
        if self.cache_breakpoint_tokens:
            if up_to_token is None:
                return self.cache_breakpoint_tokens[-1]
            eligible = [
                token_count
                for token_count in self.cache_breakpoint_tokens
                if token_count <= up_to_token
            ]
            return eligible[-1] if eligible else 0
        if self.cacheable_prefix_tokens is None:
            return up_to_token
        return self.cacheable_prefix_tokens

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def block_for_tokens(self, i: int, num_tokens: int):
        assert 0 <= i < self.num_blocks_for_tokens(num_tokens)
        return self.token_ids[i*self.block_size: min((i+1)*self.block_size, num_tokens)]

    def set_prefill_chunk(self, start: int, end: int):
        assert 0 <= start < end <= self.prefill_target_tokens
        self.scheduled_prefill_start = start
        self.scheduled_prefill_end = end

    def clear_prefill_chunk(self):
        self.scheduled_prefill_start = 0
        self.scheduled_prefill_end = 0

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def finish(self, reason: str):
        self.status = SequenceStatus.FINISHED
        self.finish_reason = reason

    def reset_cache_usage(self):
        self.cache_read_input_tokens = 0
        self.cache_creation_input_tokens = 0

    def cache_usage(self):
        input_tokens = max(
            0,
            self.num_prompt_tokens - self.cache_read_input_tokens - self.cache_creation_input_tokens,
        )
        return {
            "prompt_tokens": self.num_prompt_tokens,
            "input_tokens": input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
        }

    def __getstate__(self):
        return {
            "seq_id": self.seq_id,
            "request_id": self.request_id,
            "token_ids": self.token_ids,
            "num_tokens": self.num_tokens,
            "num_prompt_tokens": self.num_prompt_tokens,
            "prefill_target_tokens": self.prefill_target_tokens,
            "num_cached_tokens": self.num_cached_tokens,
            "num_computed_tokens": self.num_computed_tokens,
            "block_table": self.block_table,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "ignore_eos": self.ignore_eos,
            "cacheable_prefix_tokens": self.cacheable_prefix_tokens,
            "cache_breakpoint_tokens": self.cache_breakpoint_tokens,
            "cache_ttl_seconds": self.cache_ttl_seconds,
            "cache_namespace": self.cache_namespace,
            "cache_enabled": self.cache_enabled,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "scheduled_prefill_start": self.scheduled_prefill_start,
            "scheduled_prefill_end": self.scheduled_prefill_end,
        }

    def __setstate__(self, state):
        self.seq_id = state["seq_id"]
        self.request_id = state["request_id"]
        self.token_ids = state["token_ids"]
        self.last_token = self.token_ids[-1]
        self.num_tokens = state["num_tokens"]
        self.num_prompt_tokens = state["num_prompt_tokens"]
        self.prefill_target_tokens = state["prefill_target_tokens"]
        self.num_cached_tokens = state["num_cached_tokens"]
        self.num_computed_tokens = state["num_computed_tokens"]
        self.block_table = state["block_table"]
        self.temperature = state["temperature"]
        self.max_tokens = state["max_tokens"]
        self.ignore_eos = state["ignore_eos"]
        self.cacheable_prefix_tokens = state["cacheable_prefix_tokens"]
        self.cache_breakpoint_tokens = state.get("cache_breakpoint_tokens") or self._normalize_cache_breakpoints(
            self.cacheable_prefix_tokens,
            None,
        )
        self.cache_ttl_seconds = state["cache_ttl_seconds"]
        self.cache_namespace = state["cache_namespace"]
        self.cache_enabled = state.get("cache_enabled", True)
        self.cache_read_input_tokens = state.get("cache_read_input_tokens", 0)
        self.cache_creation_input_tokens = state.get("cache_creation_input_tokens", 0)
        self.scheduled_prefill_start = state["scheduled_prefill_start"]
        self.scheduled_prefill_end = state["scheduled_prefill_end"]
        self.status = SequenceStatus.RUNNING
        self.finish_reason = None
