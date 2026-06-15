import tempfile
import unittest
import json
import os

from nanovllm.models.gpt_oss_compat import (
    GptOssNativeNotSupportedError,
    inspect_gpt_oss_config,
    is_gpt_oss_config,
    raise_native_not_supported,
)


class GptOssCompatTest(unittest.TestCase):
    def test_detects_gpt_oss_architecture_and_native_gaps(self):
        report = inspect_gpt_oss_config({
            "_name_or_path": "openai/gpt-oss-20b",
            "model_type": "gpt_oss",
            "architectures": ["GptOssForCausalLM"],
            "num_hidden_layers": 24,
            "num_attention_heads": 64,
            "num_key_value_heads": 8,
            "num_experts": 32,
            "num_experts_per_tok": 4,
            "quantization_config": {"quant_method": "mxfp4"},
        })

        self.assertTrue(report.is_gpt_oss)
        self.assertFalse(report.native_supported)
        self.assertTrue(report.uses_moe)
        self.assertEqual(report.num_key_value_heads, 8)
        self.assertEqual(report.quantization_hint, "mxfp4")
        self.assertTrue(any("MoE" in reason for reason in report.unsupported_reasons))
        self.assertTrue(any("MXFP4" in reason for reason in report.unsupported_reasons))

    def test_detects_gpt_oss_from_config_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "config.json"), "w", encoding="utf-8") as f:
                json.dump({"model_type": "gpt-oss", "architectures": ["GptOssForCausalLM"]}, f)

            self.assertTrue(is_gpt_oss_config(tmp))

    def test_native_error_points_to_hf_auto_backend(self):
        with self.assertRaises(GptOssNativeNotSupportedError) as raised:
            raise_native_not_supported({"model_type": "gpt_oss"})

        self.assertIn("--model-backend hf_auto", str(raised.exception))

    def test_non_gpt_oss_config_remains_native_supported(self):
        report = inspect_gpt_oss_config({"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"]})

        self.assertFalse(report.is_gpt_oss)
        self.assertTrue(report.native_supported)


if __name__ == "__main__":
    unittest.main()
