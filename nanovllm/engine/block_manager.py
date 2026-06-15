from collections import deque
from dataclasses import dataclass
import hashlib
import struct
import time

try:
    import xxhash
except ImportError:
    xxhash = None

from nanovllm.engine.sequence import Sequence


@dataclass
class Block:
    block_id: int
    ref_count: int = 0
    hash: int = -1
    token_ids: list[int] | None = None
    namespace: str = ""
    last_accessed: int = 0
    expires_at: float | None = None

    @property
    def is_cached(self):
        return self.ref_count == 0 and self.hash != -1

    @property
    def is_free(self):
        return self.ref_count == 0 and self.hash == -1

    def reset_live(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = None

    def update_hash(self, hash_value: int, token_ids: list[int], clock: int, namespace: str = ""):
        self.hash = hash_value
        self.token_ids = list(token_ids)
        self.namespace = namespace
        self.last_accessed = clock

    def clear(self):
        self.ref_count = 0
        self.hash = -1
        self.token_ids = None
        self.namespace = ""
        self.expires_at = None


class BlockManager:
    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        enable_prefix_cache: bool = True,
        prefix_cache_min_tokens: int = 0,
        max_cached_blocks: int = 0,
        max_cached_blocks_per_namespace: int = 0,
    ):
        assert num_blocks > 0
        self.block_size = block_size
        self.enable_prefix_cache = enable_prefix_cache
        self.prefix_cache_min_tokens = prefix_cache_min_tokens
        self.max_cached_blocks = max_cached_blocks
        self.max_cached_blocks_per_namespace = max_cached_blocks_per_namespace
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = {}
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.cached_block_ids: set[int] = set()
        self.used_block_ids: set[int] = set()
        self.clock = 0
        self.prefix_cache_hits = 0
        self.prefix_cache_misses = 0
        self.prefix_cache_hits_by_namespace: dict[str, int] = {}
        self.prefix_cache_misses_by_namespace: dict[str, int] = {}
        self.cache_read_input_tokens_by_namespace: dict[str, int] = {}
        self.cache_creation_input_tokens_by_namespace: dict[str, int] = {}
        self.evictions = 0
        self.global_quota_evictions = 0
        self.namespace_quota_evictions = 0
        self.expired_purges = 0
        self.duplicate_cache_blocks_skipped = 0
        self.prefix_cache_miss_reasons: dict[str, int] = {
            "cache_disabled": 0,
            "namespace_mismatch": 0,
            "ttl_expired": 0,
            "prefix_shorter_than_min": 0,
            "no_full_block_at_breakpoint": 0,
            "hash_miss": 0,
            "token_guard_mismatch": 0,
        }

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        data = struct.pack(f"<{len(token_ids)}q", *token_ids)
        if xxhash is not None:
            h = xxhash.xxh64()
            if prefix != -1:
                h.update(prefix.to_bytes(8, "little"))
            h.update(data)
            return h.intdigest()
        h = hashlib.blake2b(digest_size=8)
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(data)
        return int.from_bytes(h.digest(), "little")

    @classmethod
    def compute_namespace_hash(cls, namespace: str):
        if not namespace:
            return -1
        data = namespace.encode("utf-8")
        if xxhash is not None:
            h = xxhash.xxh64()
            h.update(data)
            return h.intdigest()
        h = hashlib.blake2b(digest_size=8)
        h.update(data)
        return int.from_bytes(h.digest(), "little")

    @property
    def num_blocks(self):
        return len(self.blocks)

    @property
    def num_free_blocks(self):
        return len(self.free_block_ids)

    @property
    def num_cached_blocks(self):
        return len(self.cached_block_ids)

    @property
    def num_used_blocks(self):
        return len(self.used_block_ids)

    @property
    def cache_hit_rate(self):
        total = self.prefix_cache_hits + self.prefix_cache_misses
        return self.prefix_cache_hits / total if total else 0.0

    def cached_blocks_by_namespace(self):
        result: dict[str, int] = {}
        for block_id in self.cached_block_ids:
            namespace = self.blocks[block_id].namespace
            result[namespace] = result.get(namespace, 0) + 1
        return result

    def stats(self):
        self._purge_expired_cached_blocks()
        return {
            "num_blocks": self.num_blocks,
            "free_blocks": self.num_free_blocks,
            "cached_blocks": self.num_cached_blocks,
            "used_blocks": self.num_used_blocks,
            "cached_blocks_by_namespace": self.cached_blocks_by_namespace(),
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_misses": self.prefix_cache_misses,
            "prefix_cache_hits_by_namespace": dict(self.prefix_cache_hits_by_namespace),
            "prefix_cache_misses_by_namespace": dict(self.prefix_cache_misses_by_namespace),
            "cache_read_input_tokens_by_namespace": dict(self.cache_read_input_tokens_by_namespace),
            "cache_creation_input_tokens_by_namespace": dict(self.cache_creation_input_tokens_by_namespace),
            "prefix_cache_hit_rate": self.cache_hit_rate,
            "evictions": self.evictions,
            "global_quota_evictions": self.global_quota_evictions,
            "namespace_quota_evictions": self.namespace_quota_evictions,
            "max_cached_blocks": self.max_cached_blocks,
            "max_cached_blocks_per_namespace": self.max_cached_blocks_per_namespace,
            "expired_purges": self.expired_purges,
            "duplicate_cache_blocks_skipped": self.duplicate_cache_blocks_skipped,
            "prefix_cache_miss_reasons": dict(self.prefix_cache_miss_reasons),
        }

    def inspect(self):
        return {
            "block_size": self.block_size,
            "num_blocks": self.num_blocks,
            "free_blocks": self.num_free_blocks,
            "used_blocks": self.num_used_blocks,
            "cached_blocks": self.num_cached_blocks,
            "cached_blocks_by_namespace": self.cached_blocks_by_namespace(),
            "prefix_cache_hits": self.prefix_cache_hits,
            "prefix_cache_misses": self.prefix_cache_misses,
            "prefix_cache_hit_rate": self.cache_hit_rate,
            "prefix_cache_miss_reasons": dict(self.prefix_cache_miss_reasons),
            "evictions": self.evictions,
            "expired_purges": self.expired_purges,
            "max_cached_blocks": self.max_cached_blocks,
            "max_cached_blocks_per_namespace": self.max_cached_blocks_per_namespace,
        }

    def _tick(self):
        self.clock += 1
        return self.clock

    @staticmethod
    def _add_namespace_counter(counter: dict[str, int], namespace: str, value: int):
        counter[namespace] = counter.get(namespace, 0) + value

    def _prompt_tokens_in_block(self, seq: Sequence, block_index: int):
        block_start = block_index * self.block_size
        block_end = block_start + self.block_size
        return max(0, min(block_end, seq.num_prompt_tokens) - block_start)

    def _initial_prefix_hash(self, seq: Sequence):
        return self.compute_namespace_hash(seq.cache_namespace)

    def _purge_expired_cached_blocks(self):
        now = time.time()
        purged = 0
        for block_id in list(self.cached_block_ids):
            block = self.blocks[block_id]
            if block.expires_at is not None and block.expires_at <= now:
                self._purge_cached_block(block_id)
                purged += 1
        self.expired_purges += purged
        if purged:
            self._add_miss_reason("ttl_expired", purged)
        return purged

    def _purge_cached_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count == 0
        self.cached_block_ids.remove(block_id)
        self._remove_hash_mapping(block)
        block.clear()
        self.free_block_ids.append(block_id)

    def purge_cached_blocks(self, namespace: str | None = None):
        purged = 0
        for block_id in list(self.cached_block_ids):
            block = self.blocks[block_id]
            if namespace is None or block.namespace == namespace:
                self._purge_cached_block(block_id)
                purged += 1
        return purged

    def purge_expired_cached_blocks(self, namespace: str | None = None):
        now = time.time()
        purged = 0
        for block_id in list(self.cached_block_ids):
            block = self.blocks[block_id]
            namespace_matches = namespace is None or block.namespace == namespace
            if namespace_matches and block.expires_at is not None and block.expires_at <= now:
                self._purge_cached_block(block_id)
                purged += 1
        self.expired_purges += purged
        return purged

    def _remove_hash_mapping(self, block: Block):
        if block.hash != -1 and self.hash_to_block_id.get(block.hash) == block.block_id:
            del self.hash_to_block_id[block.hash]

    def _evict_cached_block(self) -> int:
        assert self.cached_block_ids
        block_id = min(self.cached_block_ids, key=lambda i: self.blocks[i].last_accessed)
        block = self.blocks[block_id]
        self.cached_block_ids.remove(block_id)
        self._remove_hash_mapping(block)
        block.clear()
        self.free_block_ids.append(block_id)
        self.evictions += 1
        return self.free_block_ids.popleft()

    def _enforce_namespace_quota(self, namespace: str):
        if self.max_cached_blocks_per_namespace <= 0:
            return 0
        block_ids = [
            block_id
            for block_id in self.cached_block_ids
            if self.blocks[block_id].namespace == namespace
        ]
        evicted = 0
        while len(block_ids) > self.max_cached_blocks_per_namespace:
            victim = min(block_ids, key=lambda i: self.blocks[i].last_accessed)
            self._purge_cached_block(victim)
            block_ids.remove(victim)
            self.evictions += 1
            self.namespace_quota_evictions += 1
            evicted += 1
        return evicted

    def _enforce_global_quota(self):
        if self.max_cached_blocks <= 0:
            return 0
        evicted = 0
        while len(self.cached_block_ids) > self.max_cached_blocks:
            victim = min(self.cached_block_ids, key=lambda i: self.blocks[i].last_accessed)
            self._purge_cached_block(victim)
            self.evictions += 1
            self.global_quota_evictions += 1
            evicted += 1
        return evicted

    def _take_free_block_id(self) -> int:
        self._purge_expired_cached_blocks()
        if self.free_block_ids:
            return self.free_block_ids.popleft()
        return self._evict_cached_block()

    def _allocate_new_block(self) -> Block:
        block_id = self._take_free_block_id()
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset_live()
        block.last_accessed = self._tick()
        self.used_block_ids.add(block_id)
        self.cached_block_ids.discard(block_id)
        return block

    def _refresh_expiry(self, block: Block, ttl_seconds: float | None):
        if ttl_seconds is None:
            block.expires_at = None
        else:
            block.expires_at = time.time() + ttl_seconds

    def _attach_cached_block(self, block_id: int, seq: Sequence):
        block = self.blocks[block_id]
        if block.ref_count == 0:
            self.cached_block_ids.discard(block_id)
            self.used_block_ids.add(block_id)
        block.ref_count += 1
        block.last_accessed = self._tick()
        self._refresh_expiry(block, seq.cache_ttl_seconds)

    def _release_block(self, block_id: int):
        block = self.blocks[block_id]
        assert block.ref_count > 0
        block.ref_count -= 1
        if block.ref_count:
            return
        self.used_block_ids.discard(block_id)
        block.last_accessed = self._tick()
        if self.enable_prefix_cache and block.hash != -1:
            self.cached_block_ids.add(block_id)
            self._enforce_namespace_quota(block.namespace)
            self._enforce_global_quota()
        else:
            self._remove_hash_mapping(block)
            block.clear()
            self.free_block_ids.append(block_id)

    def available_blocks(self):
        self._purge_expired_cached_blocks()
        return len(self.free_block_ids) + len(self.cached_block_ids)

    def _add_miss_reason(self, reason: str, value: int = 1):
        self.prefix_cache_miss_reasons[reason] = self.prefix_cache_miss_reasons.get(reason, 0) + value

    def _has_cached_tokens_in_other_namespace(self, token_ids: list[int], namespace: str):
        return any(
            self.blocks[block_id].token_ids == token_ids
            and self.blocks[block_id].namespace != namespace
            for block_id in self.cached_block_ids
        )

    def estimate_cached_prefix_tokens(self, seq: Sequence):
        if not self.enable_prefix_cache or not seq.cache_enabled:
            return 0
        cacheable_tokens = seq.max_cacheable_tokens(seq.prefill_target_tokens)
        max_cache_tokens = max(0, min(cacheable_tokens, seq.prefill_target_tokens - 1))
        if max_cache_tokens < self.prefix_cache_min_tokens:
            return 0
        max_cache_blocks = max_cache_tokens // self.block_size
        prefix_hash = self._initial_prefix_hash(seq)
        matched_blocks = 0
        for i in range(max_cache_blocks):
            token_ids = seq.block_for_tokens(i, seq.prefill_target_tokens)
            if len(token_ids) != self.block_size:
                break
            prefix_hash = self.compute_hash(token_ids, prefix_hash)
            block_id = self.hash_to_block_id.get(prefix_hash, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                break
            matched_blocks += 1
        return matched_blocks * self.block_size

    def can_allocate(self, seq: Sequence, target_num_tokens: int | None = None, reserve_blocks: int = 0) -> bool:
        target_num_tokens = target_num_tokens if target_num_tokens is not None else len(seq)
        needed_blocks = Sequence.num_blocks_for_tokens(target_num_tokens)
        missing = max(0, needed_blocks - len(seq.block_table))
        usable_blocks = max(0, self.available_blocks() - reserve_blocks)
        return missing <= usable_blocks

    def match_cached_prefix(self, seq: Sequence):
        if seq.block_table:
            return
        if not self.enable_prefix_cache or not seq.cache_enabled:
            self._add_miss_reason("cache_disabled")
            return
        seq.reset_cache_usage()
        self._purge_expired_cached_blocks()
        cacheable_tokens = seq.max_cacheable_tokens(seq.prefill_target_tokens)
        max_cache_tokens = max(0, min(cacheable_tokens, seq.prefill_target_tokens - 1))
        if max_cache_tokens < self.prefix_cache_min_tokens:
            self._add_miss_reason("prefix_shorter_than_min")
            return
        max_cache_blocks = max_cache_tokens // self.block_size
        if max_cache_tokens > 0 and max_cache_blocks == 0:
            self._add_miss_reason("no_full_block_at_breakpoint")
            return
        prefix_hash = self._initial_prefix_hash(seq)
        for i in range(max_cache_blocks):
            token_ids = seq.block_for_tokens(i, seq.prefill_target_tokens)
            if len(token_ids) != self.block_size:
                break
            prefix_hash = self.compute_hash(token_ids, prefix_hash)
            block_id = self.hash_to_block_id.get(prefix_hash, -1)
            if block_id == -1:
                self.prefix_cache_misses += 1
                self._add_namespace_counter(self.prefix_cache_misses_by_namespace, seq.cache_namespace, 1)
                if self._has_cached_tokens_in_other_namespace(token_ids, seq.cache_namespace):
                    self._add_miss_reason("namespace_mismatch")
                else:
                    self._add_miss_reason("hash_miss")
                break
            if self.blocks[block_id].token_ids != token_ids:
                self.prefix_cache_misses += 1
                self._add_namespace_counter(self.prefix_cache_misses_by_namespace, seq.cache_namespace, 1)
                self._add_miss_reason("token_guard_mismatch")
                break
            self._attach_cached_block(block_id, seq)
            seq.block_table.append(block_id)
            seq.num_cached_tokens += self.block_size
            seq.num_computed_tokens += self.block_size
            input_tokens = self._prompt_tokens_in_block(seq, i)
            seq.cache_read_input_tokens += input_tokens
            self.prefix_cache_hits += 1
            self._add_namespace_counter(self.prefix_cache_hits_by_namespace, seq.cache_namespace, 1)
            if input_tokens:
                self._add_namespace_counter(
                    self.cache_read_input_tokens_by_namespace,
                    seq.cache_namespace,
                    input_tokens,
                )

    def ensure_blocks(self, seq: Sequence, target_num_tokens: int):
        assert target_num_tokens > 0
        needed_blocks = Sequence.num_blocks_for_tokens(target_num_tokens)
        while len(seq.block_table) < needed_blocks:
            block = self._allocate_new_block()
            seq.block_table.append(block.block_id)

    def prepare_prefill(self, seq: Sequence, target_num_tokens: int):
        self.match_cached_prefix(seq)
        assert target_num_tokens > seq.num_computed_tokens
        self.ensure_blocks(seq, target_num_tokens)

    def can_append(self, seq: Sequence):
        return self.can_allocate(seq, len(seq))

    def may_append(self, seq: Sequence):
        self.ensure_blocks(seq, len(seq))

    def commit_computed_tokens(self, seq: Sequence, up_to_token: int):
        if not self.enable_prefix_cache or not seq.cache_enabled:
            return
        cacheable_tokens = seq.max_cacheable_tokens(up_to_token)
        if cacheable_tokens < self.prefix_cache_min_tokens:
            return
        full_blocks = min(up_to_token, cacheable_tokens) // self.block_size
        prefix_hash = self._initial_prefix_hash(seq)
        for i in range(full_blocks):
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            token_ids = seq.block_for_tokens(i, up_to_token)
            if block.hash != -1:
                prefix_hash = block.hash
                continue
            if len(token_ids) != self.block_size:
                continue
            h = self.compute_hash(token_ids, prefix_hash)
            existing_block_id = self.hash_to_block_id.get(h, -1)
            if (
                existing_block_id != -1
                and existing_block_id != block_id
                and self.blocks[existing_block_id].token_ids == token_ids
            ):
                self.duplicate_cache_blocks_skipped += 1
                prefix_hash = h
                continue
            block.update_hash(h, token_ids, self._tick(), namespace=seq.cache_namespace)
            self._refresh_expiry(block, seq.cache_ttl_seconds)
            self.hash_to_block_id[h] = block_id
            prefix_hash = h
            input_tokens = self._prompt_tokens_in_block(seq, i)
            seq.cache_creation_input_tokens += input_tokens
            if input_tokens:
                self._add_namespace_counter(
                    self.cache_creation_input_tokens_by_namespace,
                    seq.cache_namespace,
                    input_tokens,
                )

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            self._release_block(block_id)
        seq.num_cached_tokens = 0
        seq.num_computed_tokens = 0
        seq.reset_cache_usage()
        seq.block_table.clear()
