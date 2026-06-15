import unittest

from nanovllm.models.rope_scaling import normalize_default_rope_scaling


class RopeScalingTest(unittest.TestCase):
    def test_default_rope_scaling_dict_is_native_none(self):
        self.assertIsNone(normalize_default_rope_scaling({"rope_type": "default"}, model_name="Qwen3"))
        self.assertIsNone(normalize_default_rope_scaling({"type": "default"}, model_name="Qwen3"))

    def test_non_default_rope_scaling_fails_explicitly(self):
        with self.assertRaisesRegex(NotImplementedError, "unsupported Qwen3 rope_scaling"):
            normalize_default_rope_scaling({"rope_type": "yarn"}, model_name="Qwen3")


if __name__ == "__main__":
    unittest.main()
