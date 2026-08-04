from __future__ import annotations

import json
import random
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from app.agent.registry import ToolRegistry
from app.agent_rl.actions import AgentAction
from app.agent_rl.rewards import (
    RewardBreakdown,
    RewardConfig,
    answer_is_correct,
    evidence_coverage,
)
from app.agent_rl.tasks import AgentRLTask
from app.agent_rl.trajectory import EnvTransition, ToolEvent

FinalizeFn = Callable[[AgentRLTask, Sequence[ToolEvent]], str]


class PersonalRAGEnv:
    """Deterministic, serializable RL environment around the existing tools.

    The environment trains only the controller. ``finalize_fn`` is injected and
    frozen, so the policy selects tools and decides when to stop while answer
    synthesis stays outside the trainable policy.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        finalize_fn: FinalizeFn,
        *,
        task: Optional[AgentRLTask] = None,
        allowed_tools: Sequence[str] = ("retrieve_docs", "calculator"),
        max_steps: int = 6,
        reward_config: Optional[RewardConfig] = None,
        seed: int = 0,
        env_version: str = "personal-rag-env-v0",
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        missing = [name for name in allowed_tools if name not in registry.names()]
        if missing:
            raise ValueError(f"allowed tools are not registered: {', '.join(missing)}")

        self.registry = registry
        self.finalize_fn = finalize_fn
        self.allowed_tools = tuple(allowed_tools)
        self.max_steps = max_steps
        self.reward_config = reward_config or RewardConfig()
        self.env_version = env_version
        self._initial_task = task
        self._initial_seed = seed
        self._rng = random.Random(seed)

        self.task: Optional[AgentRLTask] = None
        self.events: List[ToolEvent] = []
        self.transitions: List[EnvTransition] = []
        self.evidence_ids: set[str] = set()
        self._seen_calls: set[str] = set()
        self.step_count = 0
        self.terminated = False
        self.answer = ""
        self.stop_reason: Optional[str] = None
        self.finalizer_error: Optional[str] = None
        self._last_message = ""

    def reset(self, task: Optional[AgentRLTask] = None, *, seed: Optional[int] = None) -> Dict[str, Any]:
        selected = task or self._initial_task
        if selected is None:
            raise ValueError("reset requires a task")
        self.task = selected
        self.events = []
        self.transitions = []
        self.evidence_ids = set()
        self._seen_calls = set()
        self.step_count = 0
        self.terminated = False
        self.answer = ""
        self.stop_reason = None
        self.finalizer_error = None
        self._last_message = "episode reset"
        self._rng = random.Random(self._initial_seed if seed is None else seed)
        return self._observation()

    def step(self, raw_action: Any) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self.task is None:
            raise RuntimeError("call reset before step")
        if self.terminated:
            raise RuntimeError("episode is already terminated")

        before = self._observation()
        breakdown = RewardBreakdown()
        if isinstance(raw_action, AgentAction):
            action_dict: Dict[str, Any] = raw_action.to_dict()
        elif isinstance(raw_action, Mapping):
            action_dict = dict(raw_action)
        else:
            action_dict = {"invalid_raw_action": repr(raw_action)}
        self.step_count += 1

        try:
            action = AgentAction.from_raw(raw_action)
            action_dict = action.to_dict()
            breakdown.valid_action_format = self.reward_config.valid_action_format
        except ValueError as exc:
            self._last_message = f"invalid action: {exc}"
            breakdown.invalid_action = self.reward_config.invalid_action
            self._terminate_if_budget(breakdown)
            return self._finish_step(before, action_dict, breakdown)

        if action.final_answer:
            self._finish_episode(breakdown)
            return self._finish_step(before, action_dict, breakdown)

        assert action.tool is not None
        if action.tool not in self.allowed_tools:
            self._last_message = (
                f"invalid action: tool '{action.tool}' is not allowed; "
                f"available: {', '.join(self.allowed_tools)}"
            )
            breakdown.invalid_action = self.reward_config.invalid_action
            self._terminate_if_budget(breakdown)
            return self._finish_step(before, action_dict, breakdown)

        try:
            call_key = json.dumps(action.to_dict(), ensure_ascii=False, sort_keys=True)
        except TypeError:
            self._last_message = "invalid action: args must be JSON serializable"
            breakdown.invalid_action = self.reward_config.invalid_action
            self._terminate_if_budget(breakdown)
            return self._finish_step(before, action_dict, breakdown)

        if call_key in self._seen_calls:
            self._last_message = "duplicate call: choose different args or final_answer"
            breakdown.duplicate_call = self.reward_config.duplicate_call
            self.events.append(ToolEvent(
                step_index=self.step_count - 1,
                tool=action.tool,
                args=dict(action.args),
                observation=self._last_message,
                ok=False,
                error="duplicate call",
            ))
            self._terminate_if_budget(breakdown)
            return self._finish_step(before, action_dict, breakdown)

        self._seen_calls.add(call_key)
        result = self.registry.dispatch(action.tool, action.args)
        breakdown.tool_call_cost = self.reward_config.tool_call_cost
        if not result.ok:
            breakdown.invalid_action = self.reward_config.invalid_action

        event = ToolEvent(
            step_index=self.step_count - 1,
            tool=action.tool,
            args=dict(action.args),
            observation=result.content if result.ok else f"error: {result.error}",
            ok=result.ok,
            data=dict(result.data),
            error=result.error,
        )
        self.events.append(event)
        self._accumulate_evidence(event)
        self._last_message = event.observation
        self._terminate_if_budget(breakdown)
        return self._finish_step(before, action_dict, breakdown)

    def _finish_episode(self, breakdown: RewardBreakdown) -> None:
        assert self.task is not None
        finalizer_error = self._run_finalizer()

        coverage = evidence_coverage(self.task.gold_evidence_ids, self.evidence_ids)
        if self.task.gold_evidence_ids:
            breakdown.evidence_coverage = self.reward_config.evidence_coverage * coverage
        evidence_complete = not self.task.gold_evidence_ids or coverage >= 1.0
        if answer_is_correct(self.task, self.answer) and evidence_complete:
            breakdown.task_success = self.reward_config.task_success
        if self.task.gold_evidence_ids and coverage < 1.0:
            breakdown.premature_final_answer = self.reward_config.premature_final_answer

        self.terminated = True
        self.finalizer_error = finalizer_error
        self.stop_reason = "final_answer" if finalizer_error is None else "finalizer_error"
        self._last_message = "final_answer" if finalizer_error is None else f"finalizer error: {finalizer_error}"

    def _terminate_if_budget(self, breakdown: RewardBreakdown) -> None:
        if self.step_count < self.max_steps:
            return
        # Always synthesize an endpoint answer from the evidence collected so
        # controller stop failures and answer-generation quality remain
        # separately measurable. Budget exhaustion is still penalized and kept
        # as the stop reason; it is never converted into a successful stop.
        finalizer_error = self._run_finalizer()
        self.terminated = True
        self.stop_reason = "budget"
        breakdown.budget_exhausted = self.reward_config.budget_exhausted
        suffix = (
            f"; finalizer error: {finalizer_error}"
            if finalizer_error is not None
            else ""
        )
        self._last_message = f"{self._last_message}; step budget exhausted{suffix}"

    def _run_finalizer(self) -> Optional[str]:
        assert self.task is not None
        try:
            self.answer = str(self.finalize_fn(self.task, tuple(self.events)))
            self.finalizer_error = None
            return None
        except Exception as exc:  # observable worker failure, not a rollout crash
            self.answer = ""
            self.finalizer_error = str(exc)
            return self.finalizer_error

    def _finish_step(
        self,
        before: Dict[str, Any],
        action: Dict[str, Any],
        breakdown: RewardBreakdown,
    ) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        after = self._observation()
        reward = breakdown.total
        self.transitions.append(EnvTransition(
            step_index=self.step_count - 1,
            observation_before=before,
            action=action,
            observation_after=after,
            reward=reward,
            reward_breakdown=breakdown.to_dict(),
            terminated=self.terminated,
        ))
        info: Dict[str, Any] = {
            "task_id": self.task.task_id if self.task else None,
            "env_version": self.env_version,
            "reward_breakdown": breakdown.to_dict(),
            "stop_reason": self.stop_reason,
            "evidence_ids": sorted(self.evidence_ids),
            "finalizer_error": self.finalizer_error,
        }
        if self.terminated:
            info["answer"] = self.answer
        return after, reward, self.terminated, info

    def _observation(self) -> Dict[str, Any]:
        history = [self._event_view(event) for event in self.events]
        return {
            "env_version": self.env_version,
            "task_id": self.task.task_id if self.task else None,
            "question": self.task.question if self.task else "",
            "available_tools": self.registry.specs(list(self.allowed_tools)),
            "history": history,
            "evidence_ids": sorted(self.evidence_ids),
            "remaining_steps": max(self.max_steps - self.step_count, 0),
            "last_message": self._last_message,
            "terminated": self.terminated,
            "answer": self.answer if self.terminated else "",
        }

    def _event_view(self, event: ToolEvent) -> Dict[str, Any]:
        data: Dict[str, Any] = {}
        if event.tool == "calculator" and "result" in event.data:
            data["result"] = event.data["result"]
        elif event.tool == "retrieve_docs":
            data["evidence_ids"] = sorted(self._evidence_from_event(event))
        return {
            "step_index": event.step_index,
            "tool": event.tool,
            "args": dict(event.args),
            "observation": event.observation,
            "ok": event.ok,
            "data": data,
            "error": event.error,
        }

    def _accumulate_evidence(self, event: ToolEvent) -> None:
        self.evidence_ids.update(self._evidence_from_event(event))

    @staticmethod
    def _evidence_from_event(event: ToolEvent) -> set[str]:
        found: set[str] = set()
        for chunk in event.data.get("chunks", []) if isinstance(event.data, dict) else []:
            if not isinstance(chunk, dict):
                continue
            meta = chunk.get("metadata", {}) if isinstance(chunk.get("metadata", {}), dict) else {}
            explicit = chunk.get("evidence_id") or meta.get("evidence_id")
            if explicit:
                found.add(str(explicit))
                continue
            source = meta.get("source", chunk.get("source"))
            chunk_id = meta.get("chunk_id", chunk.get("chunk_id"))
            if source is not None and chunk_id is not None:
                found.add(f"{source}#{chunk_id}")
        return found
