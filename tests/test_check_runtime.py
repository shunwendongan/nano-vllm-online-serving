import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from nanovllm import check_runtime


class CheckRuntimeTest(unittest.TestCase):
    def test_model_path_check_reports_missing_and_existing_directory(self):
        missing = check_runtime.check_model_path("__definitely_missing_model__")
        self.assertEqual(missing.status, "fail")

        remote_hf = check_runtime.check_model_path("openai/gpt-oss-20b", model_backend="hf_auto")
        self.assertEqual(remote_hf.status, "ok")

        with tempfile.TemporaryDirectory() as tmp:
            existing = check_runtime.check_model_path(tmp)

        self.assertEqual(existing.status, "ok")

    def test_flash_attn_check_validates_required_symbols(self):
        module = SimpleNamespace(__version__="test")
        with patch.object(check_runtime.importlib, "import_module", return_value=module):
            result = check_runtime.check_flash_attn()

        self.assertEqual(result.status, "fail")
        self.assertIn("missing symbols", result.detail)

        module.flash_attn_varlen_func = object()
        module.flash_attn_with_kvcache = object()
        with patch.object(check_runtime.importlib, "import_module", return_value=module):
            result = check_runtime.check_flash_attn()

        self.assertEqual(result.status, "ok")

    def test_torch_check_reports_device_count_against_tensor_parallel_size(self):
        cuda = SimpleNamespace(
            is_available=lambda: True,
            device_count=lambda: 1,
            get_device_name=lambda index: f"cuda:{index}",
        )
        distributed = SimpleNamespace(
            is_available=lambda: True,
            is_nccl_available=lambda: True,
            is_gloo_available=lambda: True,
        )
        torch = SimpleNamespace(__version__="test", cuda=cuda, distributed=distributed)
        with patch.object(check_runtime.importlib, "import_module", return_value=torch):
            results = check_runtime.check_torch(
                tensor_parallel_size=2,
                cuda_device_offset=0,
                distributed_backend="nccl",
            )

        self.assertEqual(results[0].status, "ok")
        self.assertEqual(results[1].name, "cuda")
        self.assertEqual(results[1].status, "fail")

    def test_hf_auto_runtime_checks_skip_flash_attn_requirement(self):
        with patch.object(check_runtime, "check_python", return_value=check_runtime.CheckResult("python", "ok", "")), \
             patch.object(check_runtime, "check_model_path", return_value=check_runtime.CheckResult("model_path", "ok", "")), \
             patch.object(check_runtime, "check_gpt_oss_compat", return_value=check_runtime.CheckResult("gpt_oss_compat", "warn", "")), \
             patch.object(check_runtime, "check_transformers", return_value=check_runtime.CheckResult("hf_config", "ok", "")), \
             patch.object(check_runtime, "check_torch", return_value=[check_runtime.CheckResult("torch", "ok", "")]), \
             patch.object(check_runtime, "check_triton", return_value=check_runtime.CheckResult("triton", "ok", "")):
            results = check_runtime.run_checks(check_runtime.parse_args([
                "--model", "openai/gpt-oss-20b",
                "--model-backend", "hf_auto",
            ]))

        flash_attn = next(result for result in results if result.name == "flash_attn")
        self.assertEqual(flash_attn.status, "skip")

    def test_backend_arg_check_accepts_current_stage_switches(self):
        results = check_runtime.check_backend_args("hf_auto", "cuda_ext")

        self.assertEqual([result.status for result in results], ["ok", "ok"])

    def test_gpt_oss_runtime_check_warns_for_native_gap(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/config.json", "w", encoding="utf-8") as f:
                f.write('{"model_type": "gpt_oss", "architectures": ["GptOssForCausalLM"]}')

            result = check_runtime.check_gpt_oss_compat(tmp)

        self.assertEqual(result.name, "gpt_oss_compat")
        self.assertEqual(result.status, "warn")
        self.assertIn("native_supported", result.detail)


if __name__ == "__main__":
    unittest.main()
