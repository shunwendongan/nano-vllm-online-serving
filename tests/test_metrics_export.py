import unittest

from nanovllm.entrypoints.openai.api_server import _prometheus_metrics


class MetricsExportTest(unittest.TestCase):
    def test_prometheus_metrics_exports_numeric_values_and_namespace_labels(self):
        text = _prometheus_metrics({
            "active_requests": 2,
            "draining": False,
            "prefix_cache_hit_rate": 0.5,
            "active_requests_by_namespace": {
                "tenant-a": 1,
                "tenant/b": 2,
            },
            "last_engine_error": None,
        })

        self.assertIn("# TYPE nanovllm_active_requests gauge", text)
        self.assertIn("nanovllm_active_requests 2", text)
        self.assertIn("nanovllm_draining 0", text)
        self.assertIn("nanovllm_prefix_cache_hit_rate 0.5", text)
        self.assertIn('nanovllm_active_requests_by_namespace{namespace="tenant-a"} 1', text)
        self.assertIn('nanovllm_active_requests_by_namespace{namespace="tenant/b"} 2', text)
        self.assertNotIn("last_engine_error", text)

    def test_prometheus_metrics_sanitizes_names_and_escapes_labels(self):
        text = _prometheus_metrics({
            "1bad-key": 3,
            "active_requests_by_namespace": {
                'tenant"quoted': 1,
            },
        }, prefix="nano-vllm")

        self.assertIn("# TYPE nano_vllm_1bad_key gauge", text)
        self.assertIn("nano_vllm_1bad_key 3", text)
        self.assertIn('namespace="tenant\\"quoted"', text)


if __name__ == "__main__":
    unittest.main()
