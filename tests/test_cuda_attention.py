import unittest
from unittest import mock

from nanovllm.layers import cuda_attention


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


if __name__ == "__main__":
    unittest.main()
