from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Protocol, Sequence

from app.agent_rl.env import PersonalRAGEnv
from app.agent_rl.policies import PolicyDecision
from app.agent_rl.tasks import AgentRLTask
from app.agent_rl.verifiers import VerificationResult, verify_task


class DecisionPolicy(Protocol):
    def decide(self, observation: Dict[str, Any]) -> PolicyDecision:
        ...


@dataclass(frozen=True)
class PolicyRollout:
    task_id: str
    seed: int
    env_version: str
    policy_version: str
    decisions: Sequence[PolicyDecision]
    transitions: Sequence[Dict[str, Any]]
    evidence_ids: Sequence[str]
    total_reward: float
    stop_reason: str
    answer: str
    finalizer_error: str
    verification: VerificationResult

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "seed": self.seed,
            "env_version": self.env_version,
            "policy_version": self.policy_version,
            "decisions": [decision.to_dict() for decision in self.decisions],
            "transitions": list(self.transitions),
            "evidence_ids": list(self.evidence_ids),
            "total_reward": self.total_reward,
            "stop_reason": self.stop_reason,
            "answer": self.answer,
            "finalizer_error": self.finalizer_error,
            "verification": self.verification.to_dict(),
        }


def run_policy_episode(
    env: PersonalRAGEnv,
    task: AgentRLTask,
    policy: DecisionPolicy,
    *,
    seed: int = 0,
) -> PolicyRollout:
    observation = env.reset(task, seed=seed)
    total_reward = 0.0
    decisions: List[PolicyDecision] = []
    info: Dict[str, Any] = {}

    while not observation["terminated"]:
        decision = policy.decide(observation)
        decisions.append(decision)
        observation, reward, _, info = env.step(decision.environment_action)
        total_reward += reward

    answer = str(info.get("answer", ""))
    evidence_ids = tuple(str(value) for value in info.get("evidence_ids", []))
    return PolicyRollout(
        task_id=task.task_id,
        seed=seed,
        env_version=env.env_version,
        policy_version=decisions[0].policy_version if decisions else "unknown",
        decisions=tuple(decisions),
        transitions=tuple(transition.to_dict() for transition in env.transitions),
        evidence_ids=evidence_ids,
        total_reward=total_reward,
        stop_reason=str(info.get("stop_reason", "")),
        answer=answer,
        finalizer_error=str(info.get("finalizer_error") or ""),
        verification=verify_task(task, predicted_answer=answer, evidence_ids=evidence_ids),
    )
