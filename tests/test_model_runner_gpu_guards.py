import unittest
import importlib.util
from unittest.mock import patch

missing_optional = [
    package
    for package in ("torch", "flash_attn")
    if importlib.util.find_spec(package) is None
]
if missing_optional:
    raise unittest.SkipTest(
        f"{', '.join(missing_optional)} required for ModelRunner/Sampler GPU guard tests"
    )

import torch

from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler


class ModelRunnerGpuGuardTest(unittest.TestCase):
    def test_prefill_last_indices_fall_back_to_sequence_length_without_scheduler_chunk(self):
        runner = object.__new__(ModelRunner)
        runner.rank = 0
        runner.sampler = Sampler()

        logits = torch.tensor([
            [0.0, 10.0],
            [10.0, 0.0],
            [0.0, 10.0],
            [10.0, 0.0],
            [0.0, 10.0],
        ])
        seqs = [
            Sequence([1, 2], request_id="a"),
            Sequence([3, 4, 5], request_id="b"),
        ]

        with patch.object(runner, "prepare_prefill", return_value=(torch.tensor([1, 2, 3, 4, 5]), None)), \
             patch.object(runner, "prepare_sample", return_value=torch.zeros(2)), \
             patch.object(runner, "run_model", return_value=logits):
            token_ids = runner.run(seqs, True)

        self.assertEqual(token_ids, [0, 1])

    def test_prefill_accepts_logits_already_reduced_by_lm_head(self):
        runner = object.__new__(ModelRunner)
        runner.rank = 0
        runner.sampler = Sampler()

        logits = torch.tensor([
            [0.0, 10.0],
            [10.0, 0.0],
        ])
        seqs = [
            Sequence([1, 2], request_id="a"),
            Sequence([3, 4, 5], request_id="b"),
        ]

        with patch.object(runner, "prepare_prefill", return_value=(torch.tensor([1, 2, 3, 4, 5]), None)), \
             patch.object(runner, "prepare_sample", return_value=torch.zeros(2)), \
             patch.object(runner, "run_model", return_value=logits):
            token_ids = runner.run(seqs, True)

        self.assertEqual(token_ids, [1, 0])

    def test_prefill_rejects_unexpected_logit_row_count(self):
        runner = object.__new__(ModelRunner)
        runner.rank = 0
        runner.sampler = Sampler()
        seqs = [
            Sequence([1, 2], request_id="a"),
            Sequence([3, 4, 5], request_id="b"),
        ]

        with patch.object(runner, "prepare_prefill", return_value=(torch.tensor([1, 2, 3, 4, 5]), None)), \
             patch.object(runner, "prepare_sample", return_value=torch.zeros(2)), \
             patch.object(runner, "run_model", return_value=torch.zeros(3, 2)):
            with self.assertRaisesRegex(RuntimeError, "prefill logits row count"):
                runner.run(seqs, True)

    def test_warmup_sets_prefill_chunk_before_calling_run(self):
        runner = object.__new__(ModelRunner)
        runner.config = type("Config", (), {
            "max_num_batched_tokens": 8,
            "max_model_len": 4,
            "max_num_seqs": 2,
        })()

        seen_chunks = []

        def fake_run(seqs, is_prefill):
            self.assertTrue(is_prefill)
            seen_chunks.extend((seq.scheduled_prefill_start, seq.scheduled_prefill_end) for seq in seqs)
            return [0 for _ in seqs]

        with patch("nanovllm.engine.model_runner.torch.cuda.empty_cache"), \
             patch("nanovllm.engine.model_runner.torch.cuda.reset_peak_memory_stats"), \
             patch.object(runner, "run", side_effect=fake_run):
            runner.warmup_model()

        self.assertEqual(seen_chunks, [(0, 4), (0, 4)])

    def test_warmup_uses_short_sequence_when_token_budget_is_below_model_len(self):
        runner = object.__new__(ModelRunner)
        runner.config = type("Config", (), {
            "max_num_batched_tokens": 2,
            "max_model_len": 4,
            "max_num_seqs": 8,
        })()

        seen_chunks = []

        def fake_run(seqs, is_prefill):
            self.assertTrue(is_prefill)
            seen_chunks.extend((len(seq), seq.scheduled_prefill_start, seq.scheduled_prefill_end) for seq in seqs)
            return [0 for _ in seqs]

        with patch("nanovllm.engine.model_runner.torch.cuda.empty_cache"), \
             patch("nanovllm.engine.model_runner.torch.cuda.reset_peak_memory_stats"), \
             patch.object(runner, "run", side_effect=fake_run):
            runner.warmup_model()

        self.assertEqual(seen_chunks, [(2, 0, 2)])

    def test_sampler_temperature_zero_does_not_divide_or_sample(self):
        logits = torch.tensor([[1.0, 3.0], [5.0, 4.0]])
        tokens = Sampler()(logits, torch.tensor([0.0, 0.0]))

        self.assertEqual(tokens.tolist(), [1, 0])


if __name__ == "__main__":
    unittest.main()
