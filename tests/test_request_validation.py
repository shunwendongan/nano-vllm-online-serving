import unittest

from nanovllm.engine.request_validation import validate_request_limits


class RequestValidationTest(unittest.TestCase):
    def test_rejects_empty_prompt(self):
        with self.assertRaisesRegex(ValueError, "prompt must contain"):
            validate_request_limits(0, 1, 16, 4, 8)

    def test_rejects_negative_max_tokens(self):
        with self.assertRaisesRegex(ValueError, "max_tokens"):
            validate_request_limits(1, -1, 16, 4, 8)

    def test_rejects_model_length_overflow(self):
        with self.assertRaisesRegex(ValueError, "exceeds max_model_len"):
            validate_request_limits(12, 5, 16, 4, 8)

    def test_rejects_kv_block_capacity_overflow(self):
        with self.assertRaisesRegex(ValueError, "KV blocks"):
            validate_request_limits(8, 8, 32, 4, 3)

    def test_allows_request_within_model_and_kv_limits(self):
        validate_request_limits(8, 8, 32, 4, 4)

    def test_allows_zero_max_tokens_for_cache_prewarm(self):
        validate_request_limits(8, 0, 32, 4, 2)


if __name__ == "__main__":
    unittest.main()
