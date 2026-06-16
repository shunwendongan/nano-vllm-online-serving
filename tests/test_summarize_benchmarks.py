import json
import tempfile
import unittest
from pathlib import Path

from scripts.summarize_benchmarks import build_summary, load_reports, markdown_summary, write_summary


class SummarizeBenchmarksTest(unittest.TestCase):
    def test_loads_report_rows_from_cloudstudio_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "reports" / "cloudstudio"
            report_dir = root / "qwen3_native_decode_first" / "run-a"
            report_dir.mkdir(parents=True)
            (report_dir / "qwen3_flash_attn_bench.json").write_text(json.dumps({
                "backend": "flash_attn",
                "scheduler_policy": "decode_first",
                "requests": 32,
                "success": 31,
                "errors": 1,
                "request_per_s": 4.2,
                "completion_tok_per_s": 128.5,
                "ttft_p95_s": 0.8,
                "latency_p95_s": 2.0,
                "server_prefix_cache_hit_rate": 0.5,
                "server_preemptions": 2,
                "server_evictions": 0,
                "server_admission_slo_rejections": 4,
                "server_admission_overload_reason": "latency_slo",
                "server_stream_flush_tokens": 8,
                "server_stream_flush_interval_s": 0.02,
                "slo_pass": False,
            }), encoding="utf-8")

            rows = load_reports(root)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["experiment"], "qwen3_native_decode_first")
        self.assertEqual(rows[0]["run_id"], "run-a")
        self.assertEqual(rows[0]["backend"], "flash_attn")
        self.assertEqual(rows[0]["policy"], "decode_first")
        self.assertEqual(rows[0]["tok_s"], 128.5)
        self.assertEqual(rows[0]["admission_slo_rejections"], 4)
        self.assertEqual(rows[0]["admission_overload_reason"], "latency_slo")
        self.assertEqual(rows[0]["stream_flush_tokens"], 8)
        self.assertEqual(rows[0]["stream_flush_interval_s"], 0.02)

    def test_markdown_summary_includes_findings(self):
        summary = build_summary([
            {
                "experiment": "exp",
                "run_id": "run",
                "backend": "flash_attn",
                "policy": "alternate",
                "success": 8,
                "errors": 0,
                "req_s": 1.0,
                "tok_s": 10.0,
                "ttft_p95_s": 1.0,
                "latency_p95_s": 1.5,
                "cache_hit": 0.0,
                "preemptions": 0,
                "evictions": 1,
                "admission_slo_rejections": 0,
                "admission_overload_reason": "",
                "stream_flush_tokens": 1,
                "stream_flush_interval_s": 0,
                "slo_pass": True,
            }
        ], "reports/cloudstudio")

        markdown = markdown_summary(summary)

        self.assertIn("| exp | run | flash_attn | alternate |", markdown)
        self.assertIn("stream_flush_tokens", markdown)
        self.assertIn("kv_pressure", markdown)
        self.assertIn("prefill_ttft", markdown)

    def test_summary_does_not_treat_disabled_prefix_cache_as_miss(self):
        summary = build_summary([
            {
                "experiment": "cuda_ext",
                "run_id": "run",
                "backend": "cuda_ext",
                "policy": "alternate",
                "success": 8,
                "errors": 0,
                "req_s": 1.0,
                "tok_s": 10.0,
                "ttft_p95_s": 0.1,
                "latency_p95_s": 1.5,
                "cache_hit": 0.0,
                "preemptions": 0,
                "evictions": 0,
                "slo_pass": True,
                "prefix_cache_disabled": True,
            }
        ], "reports/cloudstudio")

        self.assertNotIn("prefix_cache", {finding["area"] for finding in summary["findings"]})

    def test_write_summary_creates_json_and_markdown(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_summary([], Path(tmp))
            output_json = Path(tmp) / "summary.json"
            output_markdown = Path(tmp) / "summary.md"

            write_summary(summary, output_json, output_markdown)

            self.assertTrue(output_json.exists())
            self.assertTrue(output_markdown.exists())
            self.assertIn("No benchmark reports", output_markdown.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
