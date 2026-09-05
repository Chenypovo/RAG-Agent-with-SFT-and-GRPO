from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from app.agent.llm import CompleteFn
from app.agent_rl.actions import AgentAction


PROMPT_ONLY_POLICY_VERSION = "prompt-only-controller-v1"

_SYSTEM_PROMPT = """You are the controller of a retrieval agent.

Choose exactly one next action from the tools and state shown by the environment.
Return ONLY one JSON object in exactly one of these forms:
{"tool": "<tool name>", "args": {"argument": "value"}}
{"final_answer": true}

Rules:
- Use only tools listed in available_tools.
- Read tool observations before choosing the next action.
- For multi-hop questions, use different standalone retrieval queries.
- Do not repeat an identical tool call.
- Before stopping, check that the collected evidence answers every part of the question
  and resolves any bridge entity needed by a multi-hop question.
- If evidence is missing, contradictory, or a retrieval failed, use a new standalone
  query while steps remain; do not stop because a fixed number of calls was reached.
- After a duplicate-call error, never repeat that call; choose a different query or final_answer.
- Choose final_answer only when the collected evidence is sufficient.
- Do not include thoughts, plans, Markdown, or extra keys.
"""


@dataclass(frozen=True)
class PolicyDecision:
    """One policy output with enough provenance for evaluation and SFT filtering."""

    raw_output: str
    action: Optional[Dict[str, Any]]
    parse_error: Optional[str]
    system_prompt: str
    user_prompt: str
    policy_version: str

    @property
    def environment_action(self) -> Any:
        return self.action if self.action is not None else self.raw_output

    def to_dict(self) -> Dict[str, Any]:
        return {
            "raw_output": self.raw_output,
            "action": dict(self.action) if self.action is not None else None,
            "parse_error": self.parse_error,
            "system_prompt": self.system_prompt,
            "user_prompt": self.user_prompt,
            "policy_version": self.policy_version,
        }


class PromptOnlyPolicy:
    """Strict JSON controller backed by an injected, frozen completion function."""

    def __init__(
        self,
        complete_fn: CompleteFn,
        *,
        policy_version: str = PROMPT_ONLY_POLICY_VERSION,
        max_observation_chars: int = 24_000,
    ) -> None:
        if max_observation_chars < 256:
            raise ValueError("max_observation_chars must be at least 256")
        self.complete_fn = complete_fn
        self.policy_version = policy_version
        self.max_observation_chars = max_observation_chars

    def decide(self, observation: Mapping[str, Any]) -> PolicyDecision:
        user_prompt = self._render_observation(observation)
        raw_output = str(self.complete_fn(_SYSTEM_PROMPT, user_prompt) or "").strip()
        action: Optional[Dict[str, Any]] = None
        parse_error: Optional[str] = None
        try:
            parsed = json.loads(raw_output)
            action = AgentAction.from_raw(parsed).to_dict()
        except (json.JSONDecodeError, ValueError, TypeError) as exc:
            parse_error = str(exc)
        return PolicyDecision(
            raw_output=raw_output,
            action=action,
            parse_error=parse_error,
            system_prompt=_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            policy_version=self.policy_version,
        )

    def act(self, observation: Mapping[str, Any]) -> Any:
        return self.decide(observation).environment_action

    def _render_observation(self, observation: Mapping[str, Any]) -> str:
        state = self._compact_state(observation)
        payload = json.dumps(state, ensure_ascii=False, sort_keys=True)
        if len(payload) > self.max_observation_chars:
            question = str(state.get("question", ""))
            state = {
                "observation_truncated": True,
                "question": question[: max(self.max_observation_chars // 3, 64)],
                "remaining_steps": state.get("remaining_steps"),
                "available_tools": state.get("available_tools", []),
                "previous_tool_calls": state.get("previous_tool_calls", []),
                "last_message": str(state.get("last_message", ""))[:128],
            }
            payload = json.dumps(state, ensure_ascii=False, sort_keys=True)
        if len(payload) > self.max_observation_chars:
            payload = json.dumps({
                "observation_truncated": True,
                "question": str(state.get("question", ""))[:64],
                "remaining_steps": state.get("remaining_steps"),
            }, ensure_ascii=False, sort_keys=True)
        return (
            "Environment observation:\n"
            + payload
            + "\n\nChoose the next single action. Return ONLY JSON."
        )

    @staticmethod
    def _compact_state(observation: Mapping[str, Any]) -> Dict[str, Any]:
        """Keep decision-relevant state without repeating full retrieval payloads."""
        history = observation.get("history", [])
        events = history if isinstance(history, list) else []
        calls = []
        evidence_observations = []
        for raw_event in events:
            if not isinstance(raw_event, Mapping):
                continue
            event = dict(raw_event)
            calls.append({
                "step_index": event.get("step_index"),
                "tool": event.get("tool"),
                "args": event.get("args", {}),
                "ok": event.get("ok"),
                "error": event.get("error"),
            })
            if event.get("ok") and event.get("observation"):
                # max_steps=5 permits four useful calls plus a stop action. Keep
                # all four compact observations so a teacher/student can decide
                # adaptively instead of forgetting the first hop after step two.
                evidence_observations.append(str(event["observation"])[:5_000])

        last_message = str(observation.get("last_message", ""))
        state: Dict[str, Any] = {
            "question": observation.get("question", ""),
            "remaining_steps": observation.get("remaining_steps"),
            "available_tools": observation.get("available_tools", []),
            "previous_tool_calls": calls,
            "retrieved_evidence": evidence_observations[-4:],
            "last_message": last_message[:512],
        }
        if "duplicate call" in last_message.lower():
            state["required_next_action"] = (
                "Do not repeat any previous_tool_calls. If retrieved_evidence already "
                "answers every part of the question, choose final_answer now; otherwise "
                "use a different query."
            )
        elif calls and evidence_observations:
            state["decision_hint"] = (
                "Choose final_answer only if retrieved_evidence answers every part of the "
                "question; otherwise continue with a different standalone query."
            )
        return state
