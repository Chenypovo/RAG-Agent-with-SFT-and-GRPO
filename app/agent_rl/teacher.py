from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from app.agent.llm import CompleteFn
from app.agent_rl.actions import AgentAction
from app.agent_rl.env import PersonalRAGEnv
from app.agent_rl.policies import PolicyDecision, PromptOnlyPolicy, _SYSTEM_PROMPT
from app.agent_rl.rollouts import PolicyRollout, run_policy_episode
from app.agent_rl.scripted_agent import ScriptedAgentConfig, ScriptedTwoHopPolicy
from app.agent_rl.tasks import AgentRLTask


SCRIPTED_TEACHER_POLICY_VERSION = "scripted-two-hop-teacher-v1"
VERIFIER_GUIDED_TEACHER_POLICY_VERSION = "verifier-guided-llm-teacher-v1"


class ScriptedTeacherPolicy:
    """Record observation-only scripted actions in the prompt-policy SFT format.

    The wrapped controller receives only the environment observation.  Reference
    answers and evidence labels remain inside the task/verifier and are never
    available to ``decide``.
    """

    def __init__(
        self,
        config: Optional[ScriptedAgentConfig] = None,
        *,
        policy_version: str = SCRIPTED_TEACHER_POLICY_VERSION,
        max_observation_chars: int = 24_000,
    ) -> None:
        if not policy_version.strip():
            raise ValueError("policy_version must not be empty")
        self.scripted_policy = ScriptedTwoHopPolicy(config)
        self.policy_version = policy_version
        # Reuse PromptOnlyPolicy's renderer directly so teacher and learned
        # controller examples cannot silently drift to different prompts.
        self._prompt_renderer = PromptOnlyPolicy(
            lambda _system, _user: "",
            max_observation_chars=max_observation_chars,
        )

    def decide(self, observation: Mapping[str, Any]) -> PolicyDecision:
        scripted_action = self.scripted_policy.act(observation)
        action = AgentAction.from_raw(scripted_action).to_dict()
        raw_output = json.dumps(
            action,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        return PolicyDecision(
            raw_output=raw_output,
            action=action,
            parse_error=None,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=self._prompt_renderer._render_observation(observation),
            policy_version=self.policy_version,
        )

    def act(self, observation: Mapping[str, Any]) -> Any:
        return self.decide(observation).environment_action


class VerifierGuidedTeacherPolicy(PromptOnlyPolicy):
    """A frozen LLM teacher whose trajectories are selected after rollout.

    The policy sees the same observation-only prompt as the learned controller.
    Gold answers and supporting-fact labels are used only by the endpoint verifier
    in :func:`select_teacher_rollout`; they never enter ``decide``.
    """

    def __init__(
        self,
        complete_fn: CompleteFn,
        *,
        policy_version: str = VERIFIER_GUIDED_TEACHER_POLICY_VERSION,
        max_observation_chars: int = 24_000,
    ) -> None:
        super().__init__(
            complete_fn,
            policy_version=policy_version,
            max_observation_chars=max_observation_chars,
        )


@dataclass(frozen=True)
class TeacherCandidateSelection:
    """Best-of-N teacher selection with compact, serializable audit evidence."""

    task_id: str
    selected_candidate_index: int
    candidates: Tuple[PolicyRollout, ...]

    @property
    def selected(self) -> PolicyRollout:
        return self.candidates[self.selected_candidate_index]

    def to_audit_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "selected_candidate_index": self.selected_candidate_index,
            "candidates": [
                _candidate_audit(index, rollout)
                for index, rollout in enumerate(self.candidates)
            ],
        }


def generate_teacher_candidates(
    env: PersonalRAGEnv,
    task: AgentRLTask,
    policy: VerifierGuidedTeacherPolicy,
    *,
    candidates_per_task: int = 4,
    seed: int = 42,
    early_stop_on_complete_evidence: bool = True,
    existing_candidates: Sequence[PolicyRollout] = (),
    on_candidate: Optional[Callable[[int, PolicyRollout, float], None]] = None,
) -> TeacherCandidateSelection:
    """Sample complete adaptive trajectories and select one with the verifier.

    Candidate sampling is intentionally trajectory-level: the teacher chooses
    whether to stop or continue after every visible observation.  There is no
    fixed retrieval-hop target.  The environment seed is recorded separately
    for every candidate; stochasticity comes from the frozen teacher backend.
    """

    if candidates_per_task < 1:
        raise ValueError("candidates_per_task must be positive")
    candidates = list(existing_candidates)
    if len(candidates) > candidates_per_task or any(row.task_id != task.task_id for row in candidates):
        raise ValueError("saved candidates do not match task or candidate budget")
    if early_stop_on_complete_evidence and any(
        _rollout_is_clean(row) and row.verification.complete_sentence_evidence >= 1.0 for row in candidates
    ):
        return select_teacher_rollout(candidates)
    for index in range(len(candidates), candidates_per_task):
        started = time.perf_counter()
        candidate = run_policy_episode(env, task, policy, seed=seed + index)
        candidates.append(candidate)
        if on_candidate is not None:
            on_candidate(index, candidate, time.perf_counter() - started)
        if (
            early_stop_on_complete_evidence
            and _rollout_is_clean(candidate)
            and candidate.verification.complete_sentence_evidence >= 1.0
        ):
            break
    return select_teacher_rollout(candidates)


def select_teacher_rollout(
    candidates: Sequence[PolicyRollout],
) -> TeacherCandidateSelection:
    """Select the best clean trajectory without leaking verifier labels to it.

    Controller SFT prioritizes complete sentence evidence, then endpoint answer
    quality.  Tool-call cost breaks quality ties, so easy one-hop questions stay
    short while hard questions may use three or more retrievals.
    """

    rows = tuple(candidates)
    if not rows:
        raise ValueError("teacher selection requires at least one candidate")
    task_ids = {row.task_id for row in rows}
    if len(task_ids) != 1:
        raise ValueError("teacher candidates must belong to the same task")
    selected_index = max(
        range(len(rows)),
        key=lambda index: (_teacher_quality_key(rows[index]), -index),
    )
    return TeacherCandidateSelection(
        task_id=rows[0].task_id,
        selected_candidate_index=selected_index,
        candidates=rows,
    )


def _teacher_quality_key(rollout: PolicyRollout) -> Tuple[float, ...]:
    verification = rollout.verification
    tool_calls = sum(
        bool(decision.action and decision.action.get("tool"))
        for decision in rollout.decisions
    )
    return (
        float(_rollout_is_clean(rollout)),
        verification.complete_sentence_evidence,
        verification.joint_success,
        verification.answer_em,
        verification.sentence_recall,
        verification.document_recall,
        verification.answer_f1,
        float(rollout.stop_reason == "final_answer"),
        -float(tool_calls),
        -float(len(rollout.decisions)),
    )


def _rollout_is_clean(rollout: PolicyRollout) -> bool:
    if rollout.finalizer_error or rollout.stop_reason != "final_answer":
        return False
    if any(decision.parse_error for decision in rollout.decisions):
        return False
    return all(
        float(transition.get("reward_breakdown", {}).get(field, 0.0)) >= 0.0
        for transition in rollout.transitions
        for field in ("invalid_action", "duplicate_call")
    )


def _candidate_audit(index: int, rollout: PolicyRollout) -> Dict[str, Any]:
    action_counts: Dict[str, int] = {}
    for decision in rollout.decisions:
        action = decision.action or {}
        name = "invalid_json_action" if decision.parse_error else str(action.get("tool") or "final_answer")
        action_counts[name] = action_counts.get(name, 0) + 1
    return {
        "candidate_index": index,
        "seed": rollout.seed,
        "clean": _rollout_is_clean(rollout),
        "stop_reason": rollout.stop_reason,
        "n_decisions": len(rollout.decisions),
        "action_counts": action_counts,
        "verification": rollout.verification.to_dict(),
    }
