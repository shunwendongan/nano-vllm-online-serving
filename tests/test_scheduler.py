import unittest
from types import SimpleNamespace

from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


def config(**overrides):
    defaults = dict(
        max_num_seqs=8,
        max_num_batched_tokens=4,
        max_prefill_chunk_tokens=4,
        scheduler_fairness="alternate",
        eos=-1,
        num_kvcache_blocks=16,
        kvcache_block_size=4,
        enable_prefix_cache=True,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def seq(tokens, request_id):
    return Sequence(tokens, SamplingParams(max_tokens=2, ignore_eos=True), request_id=request_id)


class SchedulerTest(unittest.TestCase):
    def test_prefill_is_chunked_by_budget(self):
        scheduler = Scheduler(config(max_num_batched_tokens=3, max_prefill_chunk_tokens=3))
        request = seq([1, 2, 3, 4, 5, 6, 7], "long")
        scheduler.add(request)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [request])
        self.assertEqual(request.scheduled_prefill_start, 0)
        self.assertEqual(request.scheduled_prefill_end, 3)

        scheduler.postprocess(scheduled, [10], is_prefill=True)

        self.assertEqual(request.num_computed_tokens, 3)
        self.assertFalse(request.is_prompt_ready)
        self.assertIn(request, scheduler.running)

    def test_decode_pressure_shrinks_prefill_chunk_budget(self):
        scheduler = Scheduler(config(
            max_num_batched_tokens=8,
            max_prefill_chunk_tokens=8,
            min_prefill_chunk_tokens=2,
            kvcache_block_size=2,
            num_kvcache_blocks=16,
        ))
        first = Sequence([1, 2], SamplingParams(max_tokens=4, ignore_eos=True), request_id="decode")
        scheduler.add(first)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [11], is_prefill=is_prefill)

        long_prompt = seq([3, 4, 5, 6, 7, 8, 9, 10], "long")
        scheduler.add(long_prompt)
        scheduled, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        scheduler.postprocess(scheduled, [12], is_prefill=is_prefill)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [long_prompt])
        self.assertEqual(long_prompt.scheduled_prefill_start, 0)
        self.assertEqual(long_prompt.scheduled_prefill_end, 4)

    def test_kvcache_watermark_delays_prefill_under_decode_pressure(self):
        scheduler = Scheduler(config(
            max_num_batched_tokens=4,
            max_prefill_chunk_tokens=4,
            kvcache_block_size=2,
            num_kvcache_blocks=4,
            kvcache_watermark_blocks=1,
        ))
        first = Sequence([1, 2], SamplingParams(max_tokens=4, ignore_eos=True), request_id="decode")
        scheduler.add(first)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [11], is_prefill=is_prefill)

        waiting = seq([3, 4, 5, 6], "waiting")
        scheduler.add(waiting)

        scheduled, is_prefill = scheduler.schedule()
        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [first])
        scheduler.postprocess(scheduled, [12], is_prefill=is_prefill)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [first])
        self.assertIn(waiting, scheduler.waiting)
        self.assertEqual(scheduler.stats()["prefill_watermark_delays"], 1)

    def test_new_prefill_can_enter_between_decode_iterations(self):
        scheduler = Scheduler(config(max_num_batched_tokens=4, max_prefill_chunk_tokens=4))
        first = seq([1, 2], "a")
        scheduler.add(first)

        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [11], is_prefill=is_prefill)
        self.assertTrue(first.is_prompt_ready)

        second = seq([3, 4], "b")
        scheduler.add(second)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertIn(first, scheduled)

        scheduler.postprocess(scheduled, [12], is_prefill=is_prefill)
        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertIn(second, scheduled)

    def test_finished_sequence_releases_live_blocks_as_cache(self):
        scheduler = Scheduler(config(kvcache_block_size=2, num_kvcache_blocks=4))
        request = Sequence([1, 2, 3], SamplingParams(max_tokens=1, ignore_eos=True), request_id="done")
        scheduler.add(request)

        scheduled, is_prefill = scheduler.schedule()
        events = scheduler.postprocess(scheduled, [9], is_prefill=is_prefill)

        self.assertTrue(request.is_finished)
        self.assertEqual(events[0]["usage"], {
            "prompt_tokens": 3,
            "input_tokens": 1,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 2,
        })
        stats = scheduler.stats()
        self.assertEqual(stats["running"], 0)
        self.assertGreaterEqual(stats["cached_blocks"], 1)

    def test_prefix_cache_hit_reports_cache_read_usage(self):
        scheduler = Scheduler(config(kvcache_block_size=2, num_kvcache_blocks=4))
        first = Sequence([1, 2, 3], SamplingParams(max_tokens=1, ignore_eos=True), request_id="first")
        scheduler.add(first)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [9], is_prefill=is_prefill)

        second = Sequence([1, 2, 4], SamplingParams(max_tokens=1, ignore_eos=True), request_id="second")
        scheduler.add(second)
        scheduled, is_prefill = scheduler.schedule()
        events = scheduler.postprocess(scheduled, [10], is_prefill=is_prefill)

        self.assertEqual(events[0]["usage"], {
            "prompt_tokens": 3,
            "input_tokens": 1,
            "cache_read_input_tokens": 2,
            "cache_creation_input_tokens": 0,
        })

    def test_prefix_cache_min_tokens_skips_short_prompts(self):
        scheduler = Scheduler(config(
            kvcache_block_size=2,
            num_kvcache_blocks=4,
            prefix_cache_min_tokens=4,
        ))
        request = Sequence([1, 2, 3], SamplingParams(max_tokens=1, ignore_eos=True), request_id="short")
        scheduler.add(request)

        scheduled, is_prefill = scheduler.schedule()
        events = scheduler.postprocess(scheduled, [9], is_prefill=is_prefill)

        self.assertEqual(events[0]["usage"], {
            "prompt_tokens": 3,
            "input_tokens": 3,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        })
        self.assertEqual(scheduler.stats()["cached_blocks"], 0)

    def test_zero_max_tokens_prefills_prompt_and_warms_cache_without_decode(self):
        scheduler = Scheduler(config(kvcache_block_size=2, num_kvcache_blocks=4))
        request = Sequence([1, 2, 3], SamplingParams(max_tokens=0, ignore_eos=True), request_id="prewarm")
        scheduler.add(request)

        scheduled, is_prefill = scheduler.schedule()
        events = scheduler.postprocess(scheduled, [99], is_prefill=is_prefill)

        self.assertTrue(is_prefill)
        self.assertEqual(events[0]["token_id"], None)
        self.assertTrue(events[0]["finished"])
        self.assertEqual(events[0]["finish_reason"], "cache_warmed")
        self.assertEqual(events[0]["usage"], {
            "prompt_tokens": 3,
            "input_tokens": 1,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 2,
        })
        self.assertEqual(request.completion_token_ids, [])
        stats = scheduler.stats()
        self.assertEqual(stats["running"], 0)
        self.assertEqual(stats["waiting"], 0)
        self.assertEqual(stats["cached_blocks"], 1)

    def test_decode_preempts_victim_when_kv_blocks_are_exhausted(self):
        scheduler = Scheduler(config(
            max_num_seqs=2,
            max_num_batched_tokens=4,
            max_prefill_chunk_tokens=4,
            kvcache_block_size=2,
            num_kvcache_blocks=2,
        ))
        first = Sequence([1, 2], SamplingParams(max_tokens=2, ignore_eos=True), request_id="first")
        second = Sequence([3, 4], SamplingParams(max_tokens=2, ignore_eos=True), request_id="second")
        scheduler.add(first)
        scheduler.add(second)

        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [10, 20], is_prefill=is_prefill)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [first])
        self.assertIn(second, scheduler.waiting)
        self.assertEqual(second.prefill_target_tokens, len(second))
        self.assertEqual(second.num_computed_tokens, 0)
        self.assertEqual(scheduler.stats()["preemptions"], 1)

    def test_decode_preemption_does_not_remove_already_scheduled_sequence(self):
        scheduler = Scheduler(config(
            max_num_seqs=2,
            max_num_batched_tokens=4,
            max_prefill_chunk_tokens=4,
            kvcache_block_size=2,
            num_kvcache_blocks=3,
        ))
        first = Sequence([1, 2], SamplingParams(max_tokens=2, ignore_eos=True), request_id="first")
        second = Sequence([3, 4], SamplingParams(max_tokens=2, ignore_eos=True), request_id="second")
        scheduler.add(first)
        scheduler.add(second)

        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [10, 20], is_prefill=is_prefill)

        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [first])
        self.assertNotIn(second, scheduled)
        self.assertIn(second, scheduler.waiting)
        self.assertEqual(second.prefill_target_tokens, len(second))
        self.assertEqual(second.num_prompt_tokens, 2)

        events = scheduler.postprocess(scheduled, [11], is_prefill=is_prefill)
        self.assertTrue(events[0]["finished"])

        scheduled, is_prefill = scheduler.schedule()
        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [second])
        events = scheduler.postprocess(scheduled, [21], is_prefill=is_prefill)

        self.assertTrue(events[0]["finished"])
        self.assertEqual(events[0]["usage"]["prompt_tokens"], 2)
        self.assertLessEqual(
            events[0]["usage"]["cache_read_input_tokens"]
            + events[0]["usage"]["cache_creation_input_tokens"],
            events[0]["usage"]["prompt_tokens"],
        )

    def test_scheduler_reports_policy_and_step_decision_counters(self):
        scheduler = Scheduler(config(scheduler_fairness="decode_first"))
        request = seq([1, 2], "policy")
        scheduler.add(request)

        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [10], is_prefill=is_prefill)
        scheduled, is_prefill = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [request])
        stats = scheduler.stats()
        self.assertEqual(stats["scheduler_policy"], "decode_first")
        self.assertEqual(stats["scheduler_prefill_steps"], 1)
        self.assertEqual(stats["scheduler_decode_steps"], 1)
        self.assertEqual(stats["scheduler_policy_decisions"], {"prefill": 1, "decode": 1})

    def test_cache_aware_lpm_admits_longest_cached_prefix_first(self):
        scheduler = Scheduler(config(
            scheduler_fairness="cache_aware_lpm",
            kvcache_block_size=2,
            num_kvcache_blocks=8,
        ))
        seed = Sequence([1, 2, 3], SamplingParams(max_tokens=0, ignore_eos=True), request_id="seed")
        scheduler.add(seed)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [99], is_prefill=is_prefill)

        miss = seq([9, 9, 8], "miss")
        hit = seq([1, 2, 8], "hit")
        scheduler.add(miss)
        scheduler.add(hit)

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual([request.request_id for request in scheduled][:2], ["hit", "miss"])
        self.assertEqual(hit.num_cached_tokens, 2)

    def test_cache_aware_lpm_starvation_guard_can_override_prefix_affinity(self):
        scheduler = Scheduler(config(
            scheduler_fairness="cache_aware_lpm",
            kvcache_block_size=2,
            num_kvcache_blocks=8,
        ))
        seed = Sequence([1, 2, 3], SamplingParams(max_tokens=0, ignore_eos=True), request_id="seed")
        scheduler.add(seed)
        scheduled, is_prefill = scheduler.schedule()
        scheduler.postprocess(scheduled, [99], is_prefill=is_prefill)

        miss = seq([9, 9, 8], "miss")
        hit = seq([1, 2, 8], "hit")
        scheduler.add(miss)
        scheduler.add(hit)
        scheduler.waiting_ages["miss"] = scheduler.cache_aware_starvation_limit

        scheduled, is_prefill = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled[0].request_id, "miss")


if __name__ == "__main__":
    unittest.main()
