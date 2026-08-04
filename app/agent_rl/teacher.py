from __future__ import annotations

import json
from typing import Any, Mapping, Optional

from app.agent_rl.actions import AgentAction
from app.agent_rl.policies import PolicyDecision, PromptOnlyPolicy, _SYSTEM_PROMPT
from app.agent_rl.scripted_agent import ScriptedAgentConfig, ScriptedTwoHopPolicy


SCRIPTED_TEACHER_POLICY_VERSION = "scripted-two-hop-teacher-v1"


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
