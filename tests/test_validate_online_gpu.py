import unittest
import os
from unittest.mock import patch

from scripts.validate_online_gpu import (
    build_benchmark_command,
    build_cache_probe_payload,
    build_runtime_check_command,
    build_server_command,
    benchmark_report_paths,
    parse_args,
    validate_cache_prewarm,
    validate_prefix_cache_probe,
    wait_ready,
)


class ValidateOnlineGpuTest(unittest.TestCase):
    def test_server_command_forwards_deployment_and_resource_flags(self):
        args = parse_args([
            "--model", "model-dir",
            "--python", "python",
            "--host", "0.0.0.0",
            "--port", "9000",
            "--tensor-parallel-size", "2",
            "--gpu-memory-utilization", "0.8",
            "--distributed-backend", "gloo",
            "--distributed-init-method", "tcp://127.0.0.1:2444",
            "--cuda-device-offset", "1",
            "--ipc-shm-name", "nv-test",
            "--disable-prefix-cache",
            "--prefix-cache-min-tokens", "128",
            "--max-cached-blocks", "64",
            "--max-cached-blocks-per-namespace", "16",
            "--kv-cache-dtype", "auto",
            "--kv-compression", "h2o_exp",
            "--op-backend", "torch",
            "--attention-backend", "cuda_ext",
            "--model-backend", "hf_auto",
            "--scheduler-fairness", "cache_aware_lpm",
            "--max-pending-requests", "33",
            "--max-active-requests", "22",
            "--max-pending-requests-per-namespace", "11",
            "--max-active-requests-per-namespace", "7",
            "--max-pending-prompt-tokens", "4096",
            "--max-active-tokens", "8192",
            "--max-active-tokens-per-namespace", "2048",
            "--metrics-window-size", "64",
            "--no-enforce-eager",
        ])

        command = build_server_command(args)

        self.assertIn("nanovllm.serve", command)
        self.assertNotIn("--enforce-eager", command)
        for flag, value in {
            "--model": "model-dir",
            "--host": "0.0.0.0",
            "--port": "9000",
            "--tensor-parallel-size": "2",
            "--gpu-memory-utilization": "0.8",
            "--distributed-backend": "gloo",
            "--distributed-init-method": "tcp://127.0.0.1:2444",
            "--cuda-device-offset": "1",
            "--ipc-shm-name": "nv-test",
            "--prefix-cache-min-tokens": "128",
            "--max-cached-blocks": "64",
            "--max-cached-blocks-per-namespace": "16",
            "--kv-compression": "h2o_exp",
            "--attention-backend": "cuda_ext",
            "--model-backend": "hf_auto",
            "--scheduler-fairness": "cache_aware_lpm",
            "--max-pending-requests": "33",
            "--max-active-requests": "22",
            "--max-pending-requests-per-namespace": "11",
            "--max-active-requests-per-namespace": "7",
            "--max-pending-prompt-tokens": "4096",
            "--max-active-tokens": "8192",
            "--max-active-tokens-per-namespace": "2048",
            "--metrics-window-size": "64",
        }.items():
            self.assertEqual(command[command.index(flag) + 1], value)
        self.assertIn("--disable-prefix-cache", command)

    def test_runtime_check_command_forwards_cuda_selection(self):
        args = parse_args([
            "--model", "model-dir",
            "--python", "python",
            "--tensor-parallel-size", "4",
            "--cuda-device-offset", "2",
            "--distributed-backend", "gloo",
            "--model-backend", "hf_auto",
            "--attention-backend", "cuda_ext",
        ])

        command = build_runtime_check_command(args)

        self.assertEqual(command[command.index("--tensor-parallel-size") + 1], "4")
        self.assertEqual(command[command.index("--cuda-device-offset") + 1], "2")
        self.assertEqual(command[command.index("--distributed-backend") + 1], "gloo")
        self.assertEqual(command[command.index("--model-backend") + 1], "hf_auto")
        self.assertEqual(command[command.index("--attention-backend") + 1], "cuda_ext")

    def test_benchmark_command_includes_slo_gates(self):
        args = parse_args([
            "--model", "model-dir",
            "--python", "python",
            "--benchmark-requests", "9",
            "--benchmark-concurrency", "3",
            "--prompt", "benchmark prompt",
            "--max-tokens", "17",
            "--attention-backend", "cuda_ext",
            "--scheduler-fairness", "decode_first",
            "--cache-namespace", "tenant-a",
            "--request-namespace", "resource-a",
            "--slo-latency-p95-s", "8",
            "--slo-ttft-p95-s", "1",
            "--min-completion-tok-per-s", "12",
        ])

        command = build_benchmark_command(args, "http://127.0.0.1:8000")

        self.assertIn("--fail-on-errors", command)
        self.assertEqual(command[command.index("--requests") + 1], "9")
        self.assertEqual(command[command.index("--concurrency") + 1], "3")
        self.assertEqual(command[command.index("--prompt") + 1], "benchmark prompt")
        self.assertEqual(command[command.index("--max-tokens") + 1], "17")
        self.assertEqual(command[command.index("--backend") + 1], "cuda_ext")
        self.assertEqual(command[command.index("--scheduler-policy") + 1], "decode_first")
        self.assertEqual(command[command.index("--model-name") + 1], "model-dir")
        self.assertEqual(command[command.index("--report-json-path") + 1], os.path.join("reports", "model_dir_cuda_ext_bench.json"))
        self.assertEqual(command[command.index("--report-markdown-path") + 1], os.path.join("reports", "model_dir_cuda_ext_bench.md"))
        self.assertEqual(command[command.index("--cache-namespace") + 1], "tenant-a")
        self.assertEqual(command[command.index("--request-namespace") + 1], "resource-a")
        self.assertEqual(command[command.index("--slo-latency-p95-s") + 1], "8.0")
        self.assertEqual(command[command.index("--slo-ttft-p95-s") + 1], "1.0")
        self.assertEqual(command[command.index("--min-completion-tok-per-s") + 1], "12.0")

    def test_cache_probe_payload_uses_long_ephemeral_prompt(self):
        args = parse_args([
            "--model", "model-dir",
            "--cache-namespace", "cache-a",
            "--request-namespace", "tenant-a",
            "--cache-probe-token", "stable",
            "--cache-probe-repetitions", "3",
            "--cache-probe-max-tokens", "5",
        ])

        payload = build_cache_probe_payload(args)

        self.assertIn("stable stable stable", payload["prompt"])
        self.assertEqual(payload["max_tokens"], 5)
        self.assertEqual(payload["request_namespace"], "tenant-a")
        self.assertEqual(payload["cache_control"]["namespace"], "cache-a")
        self.assertEqual(payload["cache_control"]["type"], "ephemeral")

    def test_cache_probe_skips_when_prefix_cache_disabled(self):
        args = parse_args([
            "--model", "model-dir",
            "--disable-prefix-cache",
        ])

        result = validate_prefix_cache_probe(args, "http://unused")

        self.assertEqual(result, {"name": "prefix_cache_probe", "skipped": True})

    def test_cache_prewarm_uses_zero_max_tokens_endpoint(self):
        args = parse_args([
            "--model", "model-dir",
            "--cache-namespace", "cache-a",
            "--request-namespace", "tenant-a",
        ])
        calls = []

        def fake_request_json(method, url, payload=None, timeout=0):
            calls.append((method, url, payload, timeout))
            return {
                "finish_reason": "cache_warmed",
                "usage": {
                    "prompt_tokens": 16,
                    "input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "cache_creation_input_tokens": 16,
                },
                "cache": {"namespace": "cache-a"},
            }

        with patch("scripts.validate_online_gpu.request_json", side_effect=fake_request_json):
            result = validate_cache_prewarm(args, "http://127.0.0.1:8000")

        self.assertEqual(calls[0][0], "POST")
        self.assertEqual(calls[0][1], "http://127.0.0.1:8000/cache/prewarm")
        self.assertEqual(calls[0][2]["max_tokens"], 0)
        self.assertEqual(calls[0][2]["request_namespace"], "tenant-a")
        self.assertEqual(result["finish_reason"], "cache_warmed")
        self.assertEqual(result["cache_creation_input_tokens"], 16)

    def test_hf_auto_benchmark_uses_gpt_oss_report_paths_and_scheduler_label(self):
        args = parse_args([
            "--model", "openai/gpt-oss-20b",
            "--python", "python",
            "--model-backend", "hf_auto",
            "--benchmark-requests", "32",
            "--benchmark-concurrency", "4",
            "--max-tokens", "64",
        ])

        command = build_benchmark_command(args, "http://127.0.0.1:8000")
        json_path, markdown_path = benchmark_report_paths(args)

        self.assertEqual(command[command.index("--backend") + 1], "hf_auto")
        self.assertEqual(command[command.index("--scheduler-policy") + 1], "hf_auto")
        self.assertEqual(json_path, os.path.join("reports", "gpt_oss_hf_auto_bench.json"))
        self.assertEqual(markdown_path, os.path.join("reports", "gpt_oss_hf_auto_bench.md"))
        self.assertEqual(command[command.index("--report-json-path") + 1], json_path)
        self.assertEqual(command[command.index("--report-markdown-path") + 1], markdown_path)

    def test_hf_auto_skips_native_prefix_cache_probes(self):
        args = parse_args([
            "--model", "openai/gpt-oss-20b",
            "--model-backend", "hf_auto",
        ])

        prewarm = validate_cache_prewarm(args, "http://unused")
        probe = validate_prefix_cache_probe(args, "http://unused")

        self.assertTrue(prewarm["skipped"])
        self.assertTrue(probe["skipped"])
        self.assertIn("hf_auto", prewarm["reason"])

    def test_wait_ready_fails_fast_when_server_process_exits(self):
        class ExitedProcess:
            returncode = 123

            def poll(self):
                return self.returncode

        with self.assertRaisesRegex(RuntimeError, "server exited before ready"):
            wait_ready("http://unused", timeout_s=30, process=ExitedProcess())


if __name__ == "__main__":
    unittest.main()
