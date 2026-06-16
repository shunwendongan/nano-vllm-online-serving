import unittest
from unittest import mock

from nanovllm.layers import cuda_attention


class FakeTensor:
    def __init__(self, dtype):
        self.dtype = dtype
        self.contiguous_calls = 0
        self.float_calls = 0
        self.to_calls = []

    def contiguous(self):
        self.contiguous_calls += 1
        return self

    def float(self):
        self.float_calls += 1
        return FakeTensor("torch.float32")

    def to(self, *, dtype):
        self.to_calls.append(dtype)
        return FakeTensor(dtype)


class CudaAttentionTest(unittest.TestCase):
    def test_cuda_ext_reports_availability_as_boolean(self):
        self.assertIsInstance(cuda_attention.is_cuda_ext_available(), bool)

    def test_cuda_ext_fails_fast_without_compiled_extension(self):
        if cuda_attention.is_cuda_ext_available():
            self.skipTest("compiled CUDA extension is available on this host")

        with self.assertRaisesRegex(RuntimeError, "attention_backend='cuda_ext' requires"):
            cuda_attention.paged_decode_attention(None, None, None, None, None, 1.0)

    def test_cuda_cflags_prefers_explicit_host_compiler(self):
        with mock.patch.dict(
            "os.environ",
            {"NANOVLLM_CUDA_HOST_COMPILER": "/usr/bin/g++-12"},
        ), mock.patch("shutil.which", return_value="/usr/bin/g++-11"):
            self.assertEqual(cuda_attention._cuda_cflags(), ["-O3", "-ccbin=/usr/bin/g++-12"])

    def test_cuda_cflags_prefers_supported_discovered_compiler(self):
        def fake_which(name):
            return {"g++-12": "/usr/bin/g++-12", "g++-11": "/usr/bin/g++-11"}.get(name)

        with mock.patch.dict("os.environ", {}, clear=True), mock.patch("shutil.which", fake_which):
            self.assertEqual(cuda_attention._cuda_cflags(), ["-O3", "-ccbin=/usr/bin/g++-12"])

    def test_cuda_cflags_falls_back_to_unsupported_compiler_flag(self):
        with mock.patch.dict("os.environ", {}, clear=True), mock.patch("shutil.which", return_value=None):
            self.assertEqual(cuda_attention._cuda_cflags(), ["-O3", "-allow-unsupported-compiler"])

    def test_float32_bridge_casts_query_and_restores_output_dtype(self):
        query = FakeTensor("torch.bfloat16")
        converted, original_dtype = cuda_attention._float32_contiguous(query, "q", allow_cast=True)
        output = FakeTensor("torch.float32")

        restored = cuda_attention._restore_dtype(output, original_dtype)

        self.assertEqual(converted.dtype, "torch.float32")
        self.assertEqual(original_dtype, "torch.bfloat16")
        self.assertEqual(query.contiguous_calls, 1)
        self.assertEqual(query.float_calls, 1)
        self.assertEqual(output.to_calls, ["torch.bfloat16"])
        self.assertEqual(restored.dtype, "torch.bfloat16")

    def test_float32_bridge_rejects_non_float32_kv_cache_without_copying(self):
        cache = FakeTensor("torch.float16")

        with self.assertRaisesRegex(RuntimeError, "kv_cache_dtype='float32'"):
            cuda_attention._float32_contiguous(cache, "k_cache", allow_cast=False)

        self.assertEqual(cache.float_calls, 0)


if __name__ == "__main__":
    unittest.main()
