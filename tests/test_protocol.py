import unittest

from nanovllm.entrypoints.protocol import (
    cache_options_from_payload,
    chat_prompt_and_cache_options,
    request_options_from_payload,
    sampling_params_from_payload,
)
from nanovllm.entrypoints.openai.api_server import _openai_model_list, _openai_usage


class FakeTokenizer:
    def encode(self, text):
        return text.split()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        parts = [f"{m['role']}: {m['content']}" for m in messages]
        if add_generation_prompt:
            parts.append("assistant:")
        return "\n".join(parts)


class ProtocolTest(unittest.TestCase):
    def test_top_level_cache_control_maps_to_cache_options(self):
        payload = {
            "prompt": "a b c d",
            "cache_control": {
                "type": "ephemeral",
                "ttl": "1h",
                "namespace": "tenant-a",
            },
        }

        options = cache_options_from_payload(payload, tokenizer=FakeTokenizer(), prompt=payload["prompt"])

        self.assertEqual(options["cacheable_prefix_tokens"], 4)
        self.assertEqual(options["cache_breakpoint_tokens"], [4])
        self.assertEqual(options["cache_ttl_seconds"], 3600)
        self.assertEqual(options["cache_namespace"], "tenant-a")
        self.assertTrue(options["cache_enabled"])

    def test_no_store_cache_control_disables_prefix_cache(self):
        payload = {
            "prompt": "a b c d",
            "cache_control": {
                "type": "no-store",
                "namespace": "tenant-a",
            },
        }

        options = cache_options_from_payload(payload, tokenizer=FakeTokenizer(), prompt=payload["prompt"])

        self.assertFalse(options["cache_enabled"])
        self.assertEqual(options["cache_namespace"], "tenant-a")

    def test_top_level_disable_cache_disables_prefix_cache(self):
        payload = {
            "prompt": "a b c d",
            "disable_cache": True,
        }

        options = cache_options_from_payload(payload, tokenizer=FakeTokenizer(), prompt=payload["prompt"])

        self.assertFalse(options["cache_enabled"])

    def test_request_timeout_options_map_from_payload(self):
        options = request_options_from_payload({
            "request_timeout_s": 30,
            "queue_timeout_s": 2,
            "priority": "4",
            "tenant_id": "tenant-a",
            "trace_id": "trace-123",
        })

        self.assertEqual(options, {
            "request_timeout_s": 30,
            "queue_timeout_s": 2,
            "priority": 4,
            "request_namespace": "tenant-a",
            "trace_id": "trace-123",
        })

    def test_request_namespace_takes_precedence_over_tenant_aliases(self):
        options = request_options_from_payload({
            "request_namespace": "request-ns",
            "tenant_id": "tenant-a",
            "tenant": "tenant-b",
        })

        self.assertEqual(options["request_namespace"], "request-ns")

    def test_sampling_params_are_coerced_and_validated(self):
        sampling_params = sampling_params_from_payload({
            "temperature": "0.25",
            "max_tokens": "8",
            "ignore_eos": "true",
        })

        self.assertEqual(sampling_params.temperature, 0.25)
        self.assertEqual(sampling_params.max_tokens, 8)
        self.assertTrue(sampling_params.ignore_eos)
        self.assertEqual(sampling_params_from_payload({"max_tokens": 0}).max_tokens, 0)

        with self.assertRaisesRegex(ValueError, "max_tokens"):
            sampling_params_from_payload({"max_tokens": -1})
        with self.assertRaisesRegex(ValueError, "temperature"):
            sampling_params_from_payload({"temperature": -1})
        with self.assertRaisesRegex(ValueError, "ignore_eos"):
            sampling_params_from_payload({"ignore_eos": "maybe"})

    def test_request_options_reject_bad_resource_controls(self):
        with self.assertRaisesRegex(ValueError, "request_timeout_s"):
            request_options_from_payload({"request_timeout_s": -1})
        with self.assertRaisesRegex(ValueError, "queue_timeout_s"):
            request_options_from_payload({"queue_timeout_s": "nan"})
        with self.assertRaisesRegex(ValueError, "priority"):
            request_options_from_payload({"priority": "high"})

    def test_cache_options_reject_bad_ttl_and_boolean_controls(self):
        with self.assertRaisesRegex(ValueError, "cache ttl"):
            cache_options_from_payload({"cache_control": {"ttl": -1}})
        with self.assertRaisesRegex(ValueError, "disable_cache"):
            cache_options_from_payload({"disable_cache": "maybe"})
        with self.assertRaisesRegex(ValueError, "cache_enabled"):
            cache_options_from_payload({"cache_enabled": "maybe"})

    def test_chat_message_cache_control_sets_breakpoint(self):
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "system", "content": "stable system prompt", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "fresh question"},
        ]

        prompt, options = chat_prompt_and_cache_options(tokenizer, messages, {})

        self.assertIn("fresh question", prompt)
        self.assertEqual(options["cacheable_prefix_tokens"], 4)
        self.assertEqual(options["cache_breakpoint_tokens"], [4])
        self.assertEqual(options["cache_ttl_seconds"], 300)

    def test_payload_cache_breakpoints_are_normalized_to_latest_four(self):
        payload = {
            "prompt": "a b c d e f",
            "cache_breakpoints": [1, 2, 2, 3, 4, 5],
            "cache_control": {"ttl": "5m", "namespace": "tenant-a"},
        }

        options = cache_options_from_payload(payload, tokenizer=FakeTokenizer(), prompt=payload["prompt"])

        self.assertEqual(options["cache_breakpoint_tokens"], [2, 3, 4, 5])
        self.assertEqual(options["cacheable_prefix_tokens"], 5)
        self.assertEqual(options["cache_namespace"], "tenant-a")

    def test_chat_multiple_cache_controls_create_multiple_breakpoints(self):
        tokenizer = FakeTokenizer()
        messages = [
            {"role": "system", "content": "stable system prompt", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "fixed examples", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "fresh question"},
        ]

        prompt, options = chat_prompt_and_cache_options(
            tokenizer,
            messages,
            {"cache_control": {"type": "ephemeral"}},
        )

        self.assertIn("fresh question", prompt)
        self.assertEqual(options["cache_breakpoint_tokens"], [4, 7, 11])
        self.assertEqual(options["cacheable_prefix_tokens"], 11)

    def test_openai_usage_includes_prompt_cache_fields(self):
        usage = _openai_usage({
            "prompt_tokens": 8,
            "input_tokens": 2,
            "cache_read_input_tokens": 4,
            "cache_creation_input_tokens": 2,
        }, completion_tokens=3)

        self.assertEqual(usage["prompt_tokens"], 8)
        self.assertEqual(usage["completion_tokens"], 3)
        self.assertEqual(usage["total_tokens"], 11)
        self.assertEqual(usage["input_tokens"], 2)
        self.assertEqual(usage["cache_read_input_tokens"], 4)
        self.assertEqual(usage["cache_creation_input_tokens"], 2)

    def test_openai_model_list_shape(self):
        response = _openai_model_list("Qwen3-0.6B", created=123)

        self.assertEqual(response["object"], "list")
        self.assertEqual(response["data"][0]["id"], "Qwen3-0.6B")
        self.assertEqual(response["data"][0]["object"], "model")
        self.assertEqual(response["data"][0]["created"], 123)
        self.assertEqual(response["data"][0]["owned_by"], "nano-vllm")


if __name__ == "__main__":
    unittest.main()
