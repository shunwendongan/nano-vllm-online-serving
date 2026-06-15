import unittest

from nanovllm.engine.hf_auto_engine import HFAutoAsyncEngine


class HFAutoEngineTest(unittest.IsolatedAsyncioTestCase):
    async def test_metrics_cover_serving_validation_keys_without_loading_model(self):
        engine = HFAutoAsyncEngine("openai/gpt-oss-20b")

        metrics = engine.metrics()
        inspect = await engine.cache_inspect()

        for key in (
            "model_backend",
            "attention_backend",
            "scheduler_policy",
            "active_estimated_tokens",
            "pending_estimated_tokens",
            "active_requests_by_namespace",
            "pending_requests_by_namespace",
            "active_estimated_tokens_by_namespace",
            "pending_estimated_tokens_by_namespace",
            "prefix_cache_hits_by_namespace",
            "cache_read_input_tokens_by_namespace",
            "recent_ttft_p95_s",
            "recent_latency_p95_s",
            "recent_decode_tok_s",
            "prefix_cache_hit_rate",
        ):
            self.assertIn(key, metrics)
        self.assertEqual(metrics["model_backend"], "hf_auto")
        self.assertEqual(metrics["attention_backend"], "transformers")
        self.assertFalse(inspect["prefix_cache_supported"])


if __name__ == "__main__":
    unittest.main()
