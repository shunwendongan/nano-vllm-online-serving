import unittest

from nanovllm.layers import cuda_attention


class CudaAttentionTest(unittest.TestCase):
    def test_cuda_ext_reports_availability_as_boolean(self):
        self.assertIsInstance(cuda_attention.is_cuda_ext_available(), bool)

    def test_cuda_ext_fails_fast_without_compiled_extension(self):
        if cuda_attention.is_cuda_ext_available():
            self.skipTest("compiled CUDA extension is available on this host")

        with self.assertRaisesRegex(RuntimeError, "attention_backend='cuda_ext' requires"):
            cuda_attention.paged_decode_attention(None, None, None, None, None, 1.0)


if __name__ == "__main__":
    unittest.main()
