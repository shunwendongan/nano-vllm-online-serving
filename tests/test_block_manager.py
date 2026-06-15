import unittest
from unittest.mock import patch

from nanovllm.engine.block_manager import BlockManager
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def make_seq(tokens, request_id="r", **cache_options):
    return Sequence(tokens, SamplingParams(max_tokens=4), request_id=request_id, **cache_options)


class BlockManagerTest(unittest.TestCase):
    def test_full_blocks_are_retained_as_cached_prefix_after_deallocate(self):
        Sequence.block_size = 4
        manager = BlockManager(num_blocks=4, block_size=4, enable_prefix_cache=True)
        seq = make_seq([1, 2, 3, 4, 5], "a")

        manager.prepare_prefill(seq, 5)
        manager.commit_computed_tokens(seq, 5)
        manager.deallocate(seq)

        self.assertEqual(manager.num_cached_blocks, 1)
        self.assertEqual(manager.num_free_blocks, 3)

        other = make_seq([1, 2, 3, 4, 9], "b")
        manager.match_cached_prefix(other)

        self.assertEqual(other.num_cached_tokens, 4)
        self.assertEqual(other.num_computed_tokens, 4)
        self.assertEqual(len(other.block_table), 1)
        self.assertEqual(manager.prefix_cache_hits, 1)

    def test_duplicate_concurrent_prefix_blocks_keep_one_canonical_cache_entry(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=4, block_size=2, enable_prefix_cache=True)
        first = make_seq([1, 2, 3], "a", cache_namespace="tenant-a")
        second = make_seq([1, 2, 4], "b", cache_namespace="tenant-a")

        manager.prepare_prefill(first, 3)
        manager.prepare_prefill(second, 3)
        manager.commit_computed_tokens(first, 3)
        manager.commit_computed_tokens(second, 3)
        manager.deallocate(first)
        manager.deallocate(second)

        self.assertEqual(manager.num_cached_blocks, 1)
        self.assertEqual(manager.num_free_blocks, 3)
        self.assertEqual(manager.duplicate_cache_blocks_skipped, 1)
        self.assertEqual(manager.stats()["duplicate_cache_blocks_skipped"], 1)

        reused = make_seq([1, 2, 9], "c", cache_namespace="tenant-a")
        manager.match_cached_prefix(reused)

        self.assertEqual(reused.num_cached_tokens, 2)
        self.assertEqual(manager.prefix_cache_hits, 1)

    def test_cached_blocks_are_evicted_lru_when_free_blocks_are_exhausted(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=2, block_size=2, enable_prefix_cache=True)

        first = make_seq([1, 2], "a")
        manager.prepare_prefill(first, 2)
        manager.commit_computed_tokens(first, 2)
        manager.deallocate(first)

        second = make_seq([4, 5], "b")
        manager.prepare_prefill(second, 2)
        manager.commit_computed_tokens(second, 2)
        manager.deallocate(second)

        self.assertEqual(manager.num_cached_blocks, 2)

        third = make_seq([7, 8, 9, 10], "c")
        manager.prepare_prefill(third, 4)

        self.assertEqual(len(third.block_table), 2)
        self.assertEqual(manager.evictions, 2)
        self.assertEqual(manager.num_used_blocks, 2)

    def test_cache_namespace_isolates_prefix_hits(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=4, block_size=2, enable_prefix_cache=True)
        seq = make_seq([1, 2, 3], "a", cache_namespace="tenant-a")

        manager.prepare_prefill(seq, 3)
        manager.commit_computed_tokens(seq, 3)
        manager.deallocate(seq)

        same_tokens_other_namespace = make_seq([1, 2, 9], "b", cache_namespace="tenant-b")
        manager.match_cached_prefix(same_tokens_other_namespace)

        self.assertEqual(same_tokens_other_namespace.num_cached_tokens, 0)
        self.assertEqual(manager.prefix_cache_hits, 0)
        self.assertEqual(manager.stats()["prefix_cache_miss_reasons"]["namespace_mismatch"], 1)

    def test_prefix_cache_usage_metrics_are_tracked_by_namespace(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=6, block_size=2, enable_prefix_cache=True)
        tenant_a = make_seq([1, 2, 3], "a", cache_namespace="tenant-a")
        tenant_b = make_seq([4, 5, 6], "b", cache_namespace="tenant-b")

        for request in (tenant_a, tenant_b):
            manager.prepare_prefill(request, 3)
            manager.commit_computed_tokens(request, 3)
            manager.deallocate(request)

        tenant_a_hit = make_seq([1, 2, 9], "a-hit", cache_namespace="tenant-a")
        tenant_a_miss = make_seq([8, 9, 10], "a-miss", cache_namespace="tenant-a")
        tenant_b_hit = make_seq([4, 5, 9], "b-hit", cache_namespace="tenant-b")

        manager.match_cached_prefix(tenant_a_hit)
        manager.match_cached_prefix(tenant_a_miss)
        manager.match_cached_prefix(tenant_b_hit)
        stats = manager.stats()

        self.assertEqual(stats["cache_creation_input_tokens_by_namespace"], {"tenant-a": 2, "tenant-b": 2})
        self.assertEqual(stats["cache_read_input_tokens_by_namespace"], {"tenant-a": 2, "tenant-b": 2})
        self.assertEqual(stats["prefix_cache_hits_by_namespace"], {"tenant-a": 1, "tenant-b": 1})
        self.assertEqual(stats["prefix_cache_misses_by_namespace"], {"tenant-a": 2, "tenant-b": 1})

    def test_cache_creation_usage_counts_only_prompt_tokens_in_mixed_blocks(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=4, block_size=2, enable_prefix_cache=True)
        request = make_seq([1, 2, 3], "mixed", cache_namespace="tenant-a")

        manager.prepare_prefill(request, 3)
        manager.commit_computed_tokens(request, 3)
        request.append_token(9)
        manager.commit_computed_tokens(request, len(request))

        self.assertEqual(request.cache_usage(), {
            "prompt_tokens": 3,
            "input_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 3,
        })
        self.assertEqual(
            manager.stats()["cache_creation_input_tokens_by_namespace"],
            {"tenant-a": 3},
        )

    def test_cache_read_usage_counts_only_prompt_tokens_in_mixed_blocks(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=4, block_size=2, enable_prefix_cache=True)
        cached = make_seq([1, 2, 3], "cached", cache_namespace="tenant-a")

        manager.prepare_prefill(cached, 3)
        manager.commit_computed_tokens(cached, 3)
        cached.append_token(9)
        manager.commit_computed_tokens(cached, len(cached))
        manager.deallocate(cached)

        resumed = make_seq([1, 2, 3], "resumed", cache_namespace="tenant-a")
        resumed.append_token(9)
        resumed.append_token(10)
        resumed.prefill_target_tokens = len(resumed)
        manager.match_cached_prefix(resumed)

        self.assertEqual(resumed.num_cached_tokens, 4)
        self.assertEqual(resumed.cache_usage(), {
            "prompt_tokens": 3,
            "input_tokens": 0,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 0,
        })
        self.assertEqual(
            manager.stats()["cache_read_input_tokens_by_namespace"],
            {"tenant-a": 3},
        )

    def test_cacheable_prefix_tokens_limits_reuse_boundary(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=4, block_size=2, enable_prefix_cache=True)
        seq = make_seq([1, 2, 3, 4, 5], "a", cacheable_prefix_tokens=2)

        manager.prepare_prefill(seq, 5)
        manager.commit_computed_tokens(seq, 5)
        manager.deallocate(seq)

        self.assertEqual(manager.num_cached_blocks, 1)

        other = make_seq([1, 2, 3, 4, 9], "b", cacheable_prefix_tokens=2)
        manager.match_cached_prefix(other)

        self.assertEqual(other.num_cached_tokens, 2)

    def test_cache_breakpoint_tokens_limit_reuse_to_largest_breakpoint(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=6, block_size=2, enable_prefix_cache=True)
        seq = make_seq([1, 2, 3, 4, 5, 6, 7], "a", cache_breakpoint_tokens=[2, 4])

        manager.prepare_prefill(seq, 7)
        manager.commit_computed_tokens(seq, 7)
        manager.deallocate(seq)

        self.assertEqual(manager.num_cached_blocks, 2)

        cached_four = make_seq([1, 2, 3, 4, 9], "b", cache_breakpoint_tokens=[4])
        manager.match_cached_prefix(cached_four)
        cached_six = make_seq([1, 2, 3, 4, 5, 6, 9], "c", cache_breakpoint_tokens=[6])
        manager.match_cached_prefix(cached_six)

        self.assertEqual(cached_four.num_cached_tokens, 4)
        self.assertEqual(cached_six.num_cached_tokens, 4)

    def test_expired_cached_blocks_are_not_reused(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=2, block_size=2, enable_prefix_cache=True)
        seq = make_seq([1, 2, 3], "a", cache_ttl_seconds=1)

        with patch("nanovllm.engine.block_manager.time.time", return_value=10):
            manager.prepare_prefill(seq, 3)
            manager.commit_computed_tokens(seq, 3)
            manager.deallocate(seq)

        with patch("nanovllm.engine.block_manager.time.time", return_value=12):
            other = make_seq([1, 2, 9], "b", cache_ttl_seconds=1)
            manager.match_cached_prefix(other)

        self.assertEqual(other.num_cached_tokens, 0)
        self.assertEqual(manager.num_cached_blocks, 0)
        self.assertEqual(manager.expired_purges, 1)

    def test_stats_purges_expired_cached_blocks(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=2, block_size=2, enable_prefix_cache=True)
        seq = make_seq([1, 2, 3], "a", cache_ttl_seconds=1)

        with patch("nanovllm.engine.block_manager.time.time", return_value=10):
            manager.prepare_prefill(seq, 3)
            manager.commit_computed_tokens(seq, 3)
            manager.deallocate(seq)

        with patch("nanovllm.engine.block_manager.time.time", return_value=12):
            stats = manager.stats()

        self.assertEqual(stats["cached_blocks"], 0)
        self.assertEqual(stats["free_blocks"], 2)
        self.assertEqual(stats["expired_purges"], 1)

    def test_purge_expired_cached_blocks_by_namespace(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=4, block_size=2, enable_prefix_cache=True)
        tenant_a = make_seq([1, 2, 3], "a", cache_namespace="tenant-a", cache_ttl_seconds=1)
        tenant_b = make_seq([4, 5, 6], "b", cache_namespace="tenant-b", cache_ttl_seconds=100)

        with patch("nanovllm.engine.block_manager.time.time", return_value=10):
            for seq in (tenant_a, tenant_b):
                manager.prepare_prefill(seq, 3)
                manager.commit_computed_tokens(seq, 3)
                manager.deallocate(seq)

        with patch("nanovllm.engine.block_manager.time.time", return_value=12):
            purged = manager.purge_expired_cached_blocks(namespace="tenant-a")

        self.assertEqual(purged, 1)
        self.assertEqual(manager.cached_blocks_by_namespace(), {"tenant-b": 1})
        self.assertEqual(manager.expired_purges, 1)

    def test_purge_cached_blocks_by_namespace(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=4, block_size=2, enable_prefix_cache=True)
        tenant_a = make_seq([1, 2, 3], "a", cache_namespace="tenant-a")
        tenant_b = make_seq([4, 5, 6], "b", cache_namespace="tenant-b")

        for seq in (tenant_a, tenant_b):
            manager.prepare_prefill(seq, 3)
            manager.commit_computed_tokens(seq, 3)
            manager.deallocate(seq)

        self.assertEqual(manager.cached_blocks_by_namespace(), {"tenant-a": 1, "tenant-b": 1})

        purged = manager.purge_cached_blocks(namespace="tenant-a")

        self.assertEqual(purged, 1)
        self.assertEqual(manager.cached_blocks_by_namespace(), {"tenant-b": 1})
        same_tenant = make_seq([1, 2, 9], "same", cache_namespace="tenant-a")
        other_tenant = make_seq([4, 5, 9], "other", cache_namespace="tenant-b")
        manager.match_cached_prefix(same_tenant)
        manager.match_cached_prefix(other_tenant)
        self.assertEqual(same_tenant.num_cached_tokens, 0)
        self.assertEqual(other_tenant.num_cached_tokens, 2)

    def test_purge_all_cached_blocks(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=4, block_size=2, enable_prefix_cache=True)
        seq = make_seq([1, 2, 3], "a")

        manager.prepare_prefill(seq, 3)
        manager.commit_computed_tokens(seq, 3)
        manager.deallocate(seq)

        self.assertEqual(manager.purge_cached_blocks(), 1)
        self.assertEqual(manager.num_cached_blocks, 0)
        self.assertEqual(manager.num_free_blocks, 4)

    def test_disabled_cache_neither_reads_nor_writes_prefix_blocks(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=4, block_size=2, enable_prefix_cache=True)
        cached = make_seq([1, 2, 3], "cached")
        manager.prepare_prefill(cached, 3)
        manager.commit_computed_tokens(cached, 3)
        manager.deallocate(cached)

        no_store = make_seq([1, 2, 9], "no-store", cache_enabled=False)
        manager.match_cached_prefix(no_store)
        manager.prepare_prefill(no_store, 3)
        manager.commit_computed_tokens(no_store, 3)
        manager.deallocate(no_store)

        self.assertEqual(no_store.num_cached_tokens, 0)
        self.assertEqual(manager.prefix_cache_hits, 0)
        self.assertEqual(manager.num_cached_blocks, 1)

    def test_namespace_cache_quota_keeps_earliest_reusable_prefix_blocks(self):
        Sequence.block_size = 2
        manager = BlockManager(
            num_blocks=4,
            block_size=2,
            enable_prefix_cache=True,
            max_cached_blocks_per_namespace=1,
        )
        seq = make_seq([1, 2, 3, 4, 5], "a", cache_namespace="tenant-a")

        manager.prepare_prefill(seq, 5)
        manager.commit_computed_tokens(seq, 5)
        manager.deallocate(seq)

        self.assertEqual(manager.cached_blocks_by_namespace(), {"tenant-a": 1})
        self.assertEqual(manager.namespace_quota_evictions, 1)
        self.assertEqual(manager.evictions, 1)

        same_prefix = make_seq([1, 2, 9], "b", cache_namespace="tenant-a")
        manager.match_cached_prefix(same_prefix)

        self.assertEqual(same_prefix.num_cached_tokens, 2)

    def test_namespace_cache_quota_is_per_namespace(self):
        Sequence.block_size = 2
        manager = BlockManager(
            num_blocks=4,
            block_size=2,
            enable_prefix_cache=True,
            max_cached_blocks_per_namespace=1,
        )

        tenant_a = make_seq([1, 2, 3], "a", cache_namespace="tenant-a")
        tenant_b = make_seq([4, 5, 6], "b", cache_namespace="tenant-b")
        for seq in (tenant_a, tenant_b):
            manager.prepare_prefill(seq, 3)
            manager.commit_computed_tokens(seq, 3)
            manager.deallocate(seq)

        self.assertEqual(manager.cached_blocks_by_namespace(), {"tenant-a": 1, "tenant-b": 1})
        self.assertEqual(manager.namespace_quota_evictions, 0)

    def test_global_cache_quota_limits_total_cached_blocks_across_namespaces(self):
        Sequence.block_size = 2
        manager = BlockManager(
            num_blocks=6,
            block_size=2,
            enable_prefix_cache=True,
            max_cached_blocks=2,
        )

        tenant_a = make_seq([1, 2, 3, 4, 5], "a", cache_namespace="tenant-a")
        tenant_b = make_seq([6, 7, 8, 9, 10], "b", cache_namespace="tenant-b")
        for seq in (tenant_a, tenant_b):
            manager.prepare_prefill(seq, 5)
            manager.commit_computed_tokens(seq, 5)
            manager.deallocate(seq)

        self.assertEqual(manager.num_cached_blocks, 2)
        self.assertEqual(manager.num_free_blocks, 4)
        self.assertEqual(manager.global_quota_evictions, 2)
        self.assertEqual(manager.evictions, 2)
        self.assertEqual(manager.stats()["max_cached_blocks"], 2)

        evicted_prefix = make_seq([1, 2, 99], "evicted", cache_namespace="tenant-a")
        retained_prefix = make_seq([6, 7, 99], "retained", cache_namespace="tenant-b")
        manager.match_cached_prefix(evicted_prefix)
        manager.match_cached_prefix(retained_prefix)

        self.assertEqual(evicted_prefix.num_cached_tokens, 0)
        self.assertEqual(retained_prefix.num_cached_tokens, 2)

    def test_prefix_cache_miss_reason_diagnostics_and_inspect_are_aggregate_only(self):
        Sequence.block_size = 2
        manager = BlockManager(num_blocks=4, block_size=2, enable_prefix_cache=True)

        disabled = make_seq([1, 2, 3], "disabled", cache_enabled=False)
        manager.match_cached_prefix(disabled)

        miss = make_seq([9, 9, 8], "miss")
        manager.match_cached_prefix(miss)

        prefix_hash = manager.compute_namespace_hash("")
        forged_hash = manager.compute_hash([7, 7], prefix_hash)
        cached = make_seq([1, 2, 3], "cached")
        manager.prepare_prefill(cached, 3)
        manager.commit_computed_tokens(cached, 3)
        manager.deallocate(cached)
        manager.hash_to_block_id[forged_hash] = next(iter(manager.cached_block_ids))
        guarded = make_seq([7, 7, 8], "guarded")
        manager.match_cached_prefix(guarded)

        inspect = manager.inspect()
        reasons = inspect["prefix_cache_miss_reasons"]
        self.assertEqual(reasons["cache_disabled"], 1)
        self.assertEqual(reasons["hash_miss"], 2)
        self.assertEqual(reasons["token_guard_mismatch"], 1)
        self.assertEqual(inspect["cached_blocks"], 1)
        self.assertNotIn("token_ids", inspect)

    def test_prefix_cache_boundary_and_ttl_miss_reasons(self):
        Sequence.block_size = 4
        manager = BlockManager(num_blocks=2, block_size=4, enable_prefix_cache=True)
        partial = make_seq([1, 2, 3, 4], "partial", cacheable_prefix_tokens=3)
        manager.match_cached_prefix(partial)
        self.assertEqual(manager.inspect()["prefix_cache_miss_reasons"]["no_full_block_at_breakpoint"], 1)

        short_manager = BlockManager(
            num_blocks=2,
            block_size=4,
            enable_prefix_cache=True,
            prefix_cache_min_tokens=8,
        )
        short = make_seq([1, 2, 3, 4], "short")
        short_manager.match_cached_prefix(short)
        self.assertEqual(
            short_manager.inspect()["prefix_cache_miss_reasons"]["prefix_shorter_than_min"],
            1,
        )

        Sequence.block_size = 2
        ttl_manager = BlockManager(num_blocks=2, block_size=2, enable_prefix_cache=True)
        ttl_seq = make_seq([5, 6, 7], "ttl", cache_ttl_seconds=1)
        with patch("nanovllm.engine.block_manager.time.time", return_value=10):
            ttl_manager.prepare_prefill(ttl_seq, 3)
            ttl_manager.commit_computed_tokens(ttl_seq, 3)
            ttl_manager.deallocate(ttl_seq)
        with patch("nanovllm.engine.block_manager.time.time", return_value=12):
            ttl_manager.stats()
        self.assertEqual(ttl_manager.inspect()["prefix_cache_miss_reasons"]["ttl_expired"], 1)


if __name__ == "__main__":
    unittest.main()
