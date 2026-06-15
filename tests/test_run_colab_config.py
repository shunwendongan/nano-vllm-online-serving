import tempfile
import unittest
from pathlib import Path

from scripts.run_colab_config import (
    build_validate_command,
    load_env_config,
    parse_env_line,
    resolve_run_dir,
    validate_config_keys,
)


class RunColabConfigTest(unittest.TestCase):
    def test_parse_env_line_handles_quotes_comments_and_export(self):
        self.assertEqual(parse_env_line("export MODEL='openai/gpt-oss-20b'"), ("MODEL", "openai/gpt-oss-20b"))
        self.assertEqual(parse_env_line('PROMPT="hello # not comment" # comment'), ("PROMPT", "hello # not comment"))
        self.assertIsNone(parse_env_line("# comment"))

    def test_load_env_config_and_build_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_path = Path(tmp) / "run.env"
            config_path.write_text(
                "\n".join([
                    "EXPERIMENT_NAME=test_run",
                    "MODEL=openai/gpt-oss-20b",
                    "MODEL_BACKEND=hf_auto",
                    "ATTENTION_BACKEND=flash_attn",
                    "BENCHMARK_REQUESTS=2",
                    "BENCHMARK_CONCURRENCY=1",
                    "MAX_TOKENS=4",
                    "SKIP_CACHE_PROBE=true",
                    "ENFORCE_EAGER=false",
                    "HF_MODEL_ID=ignored-by-runner",
                ]),
                encoding="utf-8",
            )

            config = load_env_config(config_path)
            validate_config_keys(config)
            run_dir = resolve_run_dir(config, config_path, run_id="fixed")
            command = build_validate_command(config, run_dir, validate_python="python")

        self.assertEqual(run_dir, Path("reports/colab") / "test_run" / "fixed")
        self.assertEqual(command[:2], ["python", "scripts/validate_online_gpu.py"])
        self.assertEqual(command[command.index("--model") + 1], "openai/gpt-oss-20b")
        self.assertEqual(command[command.index("--model-backend") + 1], "hf_auto")
        self.assertEqual(command[command.index("--benchmark-requests") + 1], "2")
        self.assertIn("--skip-cache-probe", command)
        self.assertIn("--no-enforce-eager", command)
        self.assertEqual(command[command.index("--report-dir") + 1], str(run_dir))
        self.assertEqual(command[command.index("--request-log-path") + 1], str(run_dir / "online_requests.jsonl"))

    def test_validate_config_keys_rejects_typos(self):
        with self.assertRaisesRegex(ValueError, "unknown config keys"):
            validate_config_keys({"MODEL": "model", "MODELL": "typo"})

    def test_model_is_required(self):
        with self.assertRaisesRegex(ValueError, "MODEL is required"):
            validate_config_keys({"MODEL_BACKEND": "hf_auto"})


if __name__ == "__main__":
    unittest.main()
