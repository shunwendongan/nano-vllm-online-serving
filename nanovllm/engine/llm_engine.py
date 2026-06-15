import atexit
from dataclasses import fields
from time import perf_counter

import torch.multiprocessing as mp
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.request_validation import validate_request_limits
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class LLMEngine:
    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self.ps = []
        self.events = []
        self._closed = False
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        if hasattr(self, "model_runner"):
            self.model_runner.call("exit")
            del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self,
        prompt: str | list[int],
        sampling_params: SamplingParams,
        request_id: str | None = None,
        cacheable_prefix_tokens: int | None = None,
        cache_breakpoint_tokens: list[int] | None = None,
        cache_ttl_seconds: float | None = 300,
        cache_namespace: str | None = None,
        cache_enabled: bool = True,
    ):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        validate_request_limits(
            prompt_len=len(prompt),
            max_tokens=sampling_params.max_tokens,
            max_model_len=self.config.max_model_len,
            block_size=self.config.kvcache_block_size,
            num_kvcache_blocks=self.config.num_kvcache_blocks,
        )
        seq = Sequence(
            prompt,
            sampling_params,
            request_id=request_id,
            cacheable_prefix_tokens=cacheable_prefix_tokens,
            cache_breakpoint_tokens=cache_breakpoint_tokens,
            cache_ttl_seconds=cache_ttl_seconds,
            cache_namespace=cache_namespace,
            cache_enabled=cache_enabled,
        )
        self.scheduler.add(seq)
        return seq.request_id

    def abort_request(self, request_id: str):
        return self.scheduler.abort(request_id)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if is_prefill:
            num_tokens = sum(seq.scheduled_prefill_end - seq.scheduled_prefill_start for seq in seqs)
        else:
            num_tokens = -len(seqs)
        token_ids = self.model_runner.call("run", seqs, is_prefill)
        scheduler_events = self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = []
        for event in scheduler_events:
            seq = event["seq"]
            token_id = event["token_id"]
            outputs.append({
                "request_id": seq.request_id,
                "seq_id": seq.seq_id,
                "token_id": token_id,
                "text": "" if token_id is None else self.tokenizer.decode([token_id]),
                "completion_token_ids": list(seq.completion_token_ids),
                "finished": event["finished"],
                "finish_reason": event["finish_reason"],
                "usage": event.get("usage"),
            })
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def metrics(self):
        return {
            "model_backend": self.config.model_backend,
            "attention_backend": self.config.attention_backend,
            "op_backend": self.config.op_backend,
            **self.scheduler.stats(),
        }

    def purge_prefix_cache(self, namespace: str | None = None, expired_only: bool = False):
        return self.scheduler.purge_prefix_cache(namespace=namespace, expired_only=expired_only)

    def cache_inspect(self):
        return self.scheduler.cache_inspect()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict]:
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        request_ids = []
        for prompt, sp in zip(prompts, sampling_params):
            request_ids.append(self.add_request(prompt, sp))
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.is_finished():
            t = perf_counter()
            step_outputs, num_tokens = self.step()
            elapsed = max(perf_counter() - t, 1e-9)
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / elapsed
                else:
                    decode_throughput = -num_tokens / elapsed
                pbar.set_postfix({
                    "Prefill": f"{int(prefill_throughput)}tok/s",
                    "Decode": f"{int(decode_throughput)}tok/s",
                })
            for output in step_outputs:
                if output["finished"]:
                    outputs[output["request_id"]] = output["completion_token_ids"]
                    if use_tqdm:
                        pbar.update(1)
        if use_tqdm:
            pbar.close()
        return [
            {
                "text": self.tokenizer.decode(outputs[request_id]),
                "token_ids": outputs[request_id],
            }
            for request_id in request_ids
        ]
