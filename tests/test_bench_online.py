import argparse
import unittest

from bench_online import analyze_bottlenecks, build_payload, completion_tokens_from_response, markdown_summary, percentile


class BenchOnlineTest(unittest.TestCase):
    def test_completion_tokens_prefers_usage(self):
        response = {
            "token_ids": [1, 2],
            "usage": {"completion_tokens": 5},
        }

        self.assertEqual(completion_tokens_from_response(response), 5)

    def test_completion_tokens_falls_back_to_generate_token_ids(self):
        response = {"token_ids": [1, 2, 3]}

        self.assertEqual(completion_tokens_from_response(response), 3)

    def test_percentile_handles_empty_and_tail_latency(self):
        self.assertEqual(percentile([], 0.99), 0.0)
        self.assertEqual(percentile([1.0, 2.0, 100.0], 0.50), 2.0)
        self.assertEqual(percentile([1.0, 2.0, 100.0], 0.99), 100.0)

    def test_build_payload_separates_cache_and_request_namespace(self):
        payload = build_payload(argparse.Namespace(
            prompt="hello",
            max_tokens=4,
            temperature=0.0,
            cache_namespace="cache-a",
            cache_ttl="1h",
            cacheable_prefix_tokens=0,
            cache_breakpoints="",
            request_namespace="tenant-a",
        ))

        self.assertEqual(payload["cache_control"]["namespace"], "cache-a")
        self.assertEqual(payload["request_namespace"], "tenant-a")

    def test_build_payload_forwards_cache_breakpoints(self):
        payload = build_payload(argparse.Namespace(
            prompt="hello",
            max_tokens=4,
            temperature=0.0,
            cache_namespace="cache-a",
            cache_ttl="5m",
            cacheable_prefix_tokens=0,
            cache_breakpoints="128,256,512",
            request_namespace="",
        ))

        self.assertEqual(payload["cache_control"]["cache_breakpoints"], [128, 256, 512])

    def test_build_payload_forwards_cacheable_prefix_tokens(self):
        payload = build_payload(argparse.Namespace(
            prompt="hello",
            max_tokens=4,
            temperature=0.0,
            cache_namespace="cache-a",
            cache_ttl="5m",
            cacheable_prefix_tokens=256,
            cache_breakpoints="",
            request_namespace="",
        ))

        self.assertEqual(payload["cache_control"]["cacheable_prefix_tokens"], 256)

    def test_markdown_summary_includes_resume_metrics(self):
        markdown = markdown_summary({
            "model_name": "Qwen3-0.6B",
            "backend": "cuda_ext",
            "scheduler_policy": "cache_aware_lpm",
            "requests": 32,
            "success": 32,
            "errors": 0,
            "concurrency": 4,
            "completion_tok_per_s": 128.5,
            "latency_p95_s": 0.8,
            "ttft_p95_s": 0.12,
            "cache_read_input_tokens": 1024,
            "cache_creation_input_tokens": 256,
            "server_prefix_cache_hit_rate": 0.75,
            "server_preemptions": 1,
            "server_evictions": 2,
            "slo_pass": True,
        })

        self.assertIn("| backend | cuda_ext |", markdown)
        self.assertIn("| scheduler_policy | cache_aware_lpm |", markdown)
        self.assertIn("| ttft_p95_s | 0.12 |", markdown)
        self.assertIn("| server_prefix_cache_hit_rate | 0.75 |", markdown)
        self.assertIn("## Bottleneck Analysis", markdown)

    def test_bottleneck_analysis_marks_hf_auto_boundary_and_ttft(self):
        analysis = analyze_bottlenecks({
            "backend": "hf_auto",
            "success": 32,
            "errors": 0,
            "latency_p95_s": 2.0,
            "ttft_p95_s": 1.4,
            "completion_tok_per_s": 10.0,
            "server_metrics": {"model_backend": "hf_auto"},
        })

        areas = [item["area"] for item in analysis]
        self.assertIn("backend_boundary", areas)
        self.assertIn("ttft", areas)
        self.assertTrue(any("Transformers compatibility path" in item["finding"] for item in analysis))

    def test_bottleneck_analysis_detects_errors_and_decode_dominance(self):
        analysis = analyze_bottlenecks({
            "backend": "flash_attn",
            "success": 8,
            "errors": 2,
            "error_status_counts": {"500": 2},
            "latency_p95_s": 8.0,
            "ttft_p95_s": 1.0,
            "completion_tok_per_s": 4.0,
            "server_preemptions": 1,
            "server_evictions": 0,
        })

        areas = [item["area"] for item in analysis]
        self.assertIn("request_failures", areas)
        self.assertIn("decode_throughput", areas)
        self.assertIn("kv_pressure", areas)


if __name__ == "__main__":
    unittest.main()
