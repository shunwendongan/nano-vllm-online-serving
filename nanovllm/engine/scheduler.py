from collections import deque
from typing import TYPE_CHECKING

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager

if TYPE_CHECKING:
    from nanovllm.config import Config


class Scheduler:
    def __init__(self, config: "Config"):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.max_prefill_chunk_tokens = config.max_prefill_chunk_tokens
        self.min_prefill_chunk_tokens = getattr(config, "min_prefill_chunk_tokens", 1)
        self.kvcache_watermark_blocks = getattr(config, "kvcache_watermark_blocks", 0)
        self.scheduler_fairness = config.scheduler_fairness
        self.eos = config.eos
        Sequence.block_size = config.kvcache_block_size
        self.block_manager = BlockManager(
            config.num_kvcache_blocks,
            config.kvcache_block_size,
            enable_prefix_cache=config.enable_prefix_cache,
            prefix_cache_min_tokens=getattr(config, "prefix_cache_min_tokens", 0),
            max_cached_blocks=getattr(config, "max_cached_blocks", 0),
            max_cached_blocks_per_namespace=getattr(config, "max_cached_blocks_per_namespace", 0),
        )
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
        self._last_step_was_prefill = False
        self.preemptions = 0
        self.prefill_watermark_delays = 0
        self.prefill_steps = 0
        self.decode_steps = 0
        self.policy_decisions = {"prefill": 0, "decode": 0}
        self.waiting_ages: dict[str, int] = {}
        self.cache_aware_starvation_limit = 8

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)
        self.waiting_ages[seq.request_id] = 0

    def abort(self, request_id: str):
        for queue in (self.waiting, self.running):
            for seq in list(queue):
                if seq.request_id == request_id:
                    queue.remove(seq)
                    self.waiting_ages.pop(seq.request_id, None)
                    if seq.block_table:
                        self.block_manager.deallocate(seq)
                    seq.finish("cancelled")
                    return True
        return False

    def stats(self):
        return {
            "waiting": len(self.waiting),
            "running": len(self.running),
            "scheduler_policy": self.scheduler_fairness,
            "scheduler_prefill_steps": self.prefill_steps,
            "scheduler_decode_steps": self.decode_steps,
            "scheduler_policy_decisions": dict(self.policy_decisions),
            "preemptions": self.preemptions,
            "prefill_watermark_delays": self.prefill_watermark_delays,
            **self.block_manager.stats(),
        }

    def purge_prefix_cache(self, namespace: str | None = None, expired_only: bool = False):
        if expired_only:
            return self.block_manager.purge_expired_cached_blocks(namespace=namespace)
        return self.block_manager.purge_cached_blocks(namespace=namespace)

    def cache_inspect(self):
        return self.block_manager.inspect()

    def _has_prefill_work(self):
        return bool(self.waiting) or any(not seq.is_prompt_ready for seq in self.running)

    def _has_decode_work(self):
        return any(seq.is_prompt_ready for seq in self.running)

    def _prefer_prefill(self):
        if self.scheduler_fairness == "fcfs":
            if any(not seq.is_prompt_ready for seq in self.running):
                return True
            if self._has_decode_work():
                return False
            return self._has_prefill_work()
        if self.scheduler_fairness == "prefill_first":
            return self._has_prefill_work()
        if self.scheduler_fairness == "decode_first":
            return not self._has_decode_work() and self._has_prefill_work()
        if not self._has_decode_work():
            return self._has_prefill_work()
        if not self._has_prefill_work():
            return False
        return not self._last_step_was_prefill

    def _age_waiting(self):
        for seq in self.waiting:
            self.waiting_ages[seq.request_id] = self.waiting_ages.get(seq.request_id, 0) + 1

    def _reorder_waiting_for_cache_affinity(self):
        if self.scheduler_fairness != "cache_aware_lpm" or len(self.waiting) <= 1:
            return
        ranked = []
        for index, seq in enumerate(self.waiting):
            age = self.waiting_ages.get(seq.request_id, 0)
            cached_tokens = self.block_manager.estimate_cached_prefix_tokens(seq)
            starved = age >= self.cache_aware_starvation_limit
            ranked.append((starved, cached_tokens, age, -index, seq))
        ranked.sort(reverse=True, key=lambda item: item[:4])
        self.waiting = deque(item[-1] for item in ranked)

    def _prefill_chunk_budget(self):
        budget = min(self.max_prefill_chunk_tokens, self.max_num_batched_tokens)
        decode_pressure = sum(1 for seq in self.running if seq.is_prompt_ready)
        if decode_pressure <= 0:
            return budget
        pressure_divisor = min(decode_pressure + 1, 8)
        return max(self.min_prefill_chunk_tokens, budget // pressure_divisor)

    def _prefill_reserve_blocks(self):
        return self.kvcache_watermark_blocks if self._has_decode_work() else 0

    def schedule(self) -> tuple[list[Sequence], bool]:
        self._age_waiting()
        if self._prefer_prefill():
            seqs = self._schedule_prefill()
            if seqs:
                self._last_step_was_prefill = True
                self.prefill_steps += 1
                self.policy_decisions["prefill"] += 1
                return seqs, True
            seqs = self._schedule_decode()
            if seqs:
                self._last_step_was_prefill = False
                self.decode_steps += 1
                self.policy_decisions["decode"] += 1
                return seqs, False
        else:
            seqs = self._schedule_decode()
            if seqs:
                self._last_step_was_prefill = False
                self.decode_steps += 1
                self.policy_decisions["decode"] += 1
                return seqs, False
            seqs = self._schedule_prefill()
            if seqs:
                self._last_step_was_prefill = True
                self.prefill_steps += 1
                self.policy_decisions["prefill"] += 1
                return seqs, True
        raise RuntimeError("no schedulable sequence; KV cache may be too small for one request")

    def _admit_waiting(self):
        self._reorder_waiting_for_cache_affinity()
        admitted = []
        reserve_blocks = self._prefill_reserve_blocks()
        while self.waiting and len(self.running) < self.max_num_seqs:
            seq = self.waiting[0]
            self.block_manager.match_cached_prefix(seq)
            target = min(
                seq.prefill_target_tokens,
                max(seq.num_computed_tokens + 1, min(seq.prefill_target_tokens, self.max_prefill_chunk_tokens)),
            )
            if not self.block_manager.can_allocate(seq, target, reserve_blocks=reserve_blocks):
                if reserve_blocks and self.block_manager.can_allocate(seq, target):
                    self.prefill_watermark_delays += 1
                break
            self.waiting.popleft()
            self.waiting_ages.pop(seq.request_id, None)
            seq.status = SequenceStatus.RUNNING
            self.running.append(seq)
            admitted.append(seq)
        return admitted

    def _schedule_prefill(self):
        self._admit_waiting()
        scheduled_seqs = []
        num_batched_tokens = 0
        prefill_chunk_budget = self._prefill_chunk_budget()
        reserve_blocks = self._prefill_reserve_blocks()
        candidates = [seq for seq in self.running if not seq.is_prompt_ready]
        for seq in candidates:
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            if num_batched_tokens >= self.max_num_batched_tokens:
                break
            remaining_budget = min(
                prefill_chunk_budget,
                self.max_num_batched_tokens - num_batched_tokens,
            )
            if remaining_budget <= 0:
                break
            target = min(seq.prefill_target_tokens, seq.num_computed_tokens + remaining_budget)
            if target <= seq.num_computed_tokens:
                continue
            if not self.block_manager.can_allocate(seq, target, reserve_blocks=reserve_blocks):
                continue
            self.block_manager.prepare_prefill(seq, target)
            seq.set_prefill_chunk(seq.num_computed_tokens, target)
            num_batched_tokens += target - seq.num_computed_tokens
            scheduled_seqs.append(seq)
        return scheduled_seqs

    def _schedule_decode(self):
        scheduled_seqs = []
        for seq in list(self.running):
            if len(scheduled_seqs) >= self.max_num_seqs:
                break
            if not seq.is_prompt_ready:
                continue
            while not self.block_manager.can_append(seq):
                victim = self._select_preemption_victim(exclude=seq, scheduled=scheduled_seqs)
                if victim is None:
                    self.preempt(seq)
                    break
                self.preempt(victim)
            if seq.status != SequenceStatus.RUNNING or not seq.is_prompt_ready:
                continue
            self.block_manager.may_append(seq)
            scheduled_seqs.append(seq)
        for seq in scheduled_seqs:
            if seq in self.running:
                self.running.remove(seq)
                self.running.append(seq)
        return scheduled_seqs

    def _select_preemption_victim(self, exclude: Sequence, scheduled: list[Sequence] | None = None):
        scheduled = scheduled or []
        for seq in reversed(self.running):
            if seq is not exclude and seq not in scheduled:
                return seq
        return None

    def preempt(self, seq: Sequence):
        if seq in self.running:
            self.running.remove(seq)
        seq.status = SequenceStatus.WAITING
        seq.prefill_target_tokens = len(seq)
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)
        self.waiting_ages[seq.request_id] = 0
        self.preemptions += 1

    def postprocess(self, seqs: list[Sequence], token_ids: list[int], is_prefill: bool):
        events = []
        for seq, token_id in zip(seqs, token_ids):
            if is_prefill:
                seq.num_computed_tokens = seq.scheduled_prefill_end
                self.block_manager.commit_computed_tokens(seq, seq.num_computed_tokens)
                seq.clear_prefill_chunk()
                if not seq.is_prompt_ready:
                    continue
                if seq.max_tokens == 0:
                    seq.finish("cache_warmed")
                    event = {
                        "seq": seq,
                        "token_id": None,
                        "finished": True,
                        "finish_reason": seq.finish_reason,
                        "usage": seq.cache_usage(),
                    }
                    self.block_manager.deallocate(seq)
                    if seq in self.running:
                        self.running.remove(seq)
                    events.append(event)
                    continue
            else:
                seq.num_computed_tokens = max(seq.num_computed_tokens, len(seq))
                self.block_manager.commit_computed_tokens(seq, len(seq))

            seq.append_token(token_id)
            event = {
                "seq": seq,
                "token_id": token_id,
                "finished": False,
                "finish_reason": None,
            }
            if not seq.ignore_eos and token_id == self.eos:
                seq.finish("stop")
            elif seq.num_completion_tokens >= seq.max_tokens:
                seq.finish("length")
            if seq.is_finished:
                event["usage"] = seq.cache_usage()
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)
                event["finished"] = True
                event["finish_reason"] = seq.finish_reason
            events.append(event)
        return events
