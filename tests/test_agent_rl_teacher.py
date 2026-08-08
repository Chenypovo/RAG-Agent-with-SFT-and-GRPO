import json

from app.agent_rl.adapters import build_minimal_registry
from app.agent_rl.env import PersonalRAGEnv
from app.agent_rl.policies import PromptOnlyPolicy
from app.agent_rl.rollouts import run_policy_episode
from app.agent_rl.scripted_agent import ScriptedAgentConfig
from app.agent_rl.tasks import AgentRLTask
from app.agent_rl.teacher import ScriptedTeacherPolicy


def test_teacher_decision_uses_exact_prompt_only_rendering_and_strict_json():
    observation = {
        "question": "Who founded Alpha?",
        "available_tools": [{"name": "retrieve_docs"}],
        "history": [],
        "remaining_steps": 3,
        "last_message": "episode reset",
    }
    captured = {}

    def complete(system_prompt, user_prompt):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        return '{"final_answer":true}'

    PromptOnlyPolicy(complete).decide(observation)
    decision = ScriptedTeacherPolicy(
        ScriptedAgentConfig(retrieval_hops=2, first_hop_k=7, follow_up_k=1)
    ).decide(observation)

    assert decision.system_prompt == captured["system_prompt"]
    assert decision.user_prompt == captured["user_prompt"]
    assert json.loads(decision.raw_output) == decision.action
    assert decision.raw_output == (
        '{"args":{"k":7,"query":"Who founded Alpha?"},"tool":"retrieve_docs"}'
    )
    assert decision.parse_error is None


def test_teacher_rollout_records_decisions_and_never_prompts_with_gold():
    chunks = {
        "Who founded Alpha?": [{
            "metadata": {
                "source": "docs/alpha",
                "chunk_id": 0,
                "text": "Alpha was founded by Beta Person.",
            }
        }],
        "Beta Person": [{
            "metadata": {
                "source": "docs/beta",
                "chunk_id": 0,
                "text": "Beta Person founded Alpha.",
            }
        }],
    }
    env = PersonalRAGEnv(
        registry=build_minimal_registry(lambda query: chunks.get(query, [])),
        finalize_fn=lambda task, events: "SECRET_GOLD_VALUE",
        allowed_tools=("retrieve_docs",),
        max_steps=3,
        env_version="teacher-test-v1",
    )
    task = AgentRLTask(
        task_id="teacher_1",
        question="Who founded Alpha?",
        gold_answers=("SECRET_GOLD_VALUE",),
        gold_evidence_ids=("docs/alpha#0", "docs/beta#0"),
    )

    rollout = run_policy_episode(
        env,
        task,
        ScriptedTeacherPolicy(ScriptedAgentConfig(
            retrieval_hops=2,
            first_hop_k=1,
            follow_up_k=1,
        )),
        seed=9,
    )

    assert len(rollout.decisions) == len(rollout.transitions) == 3
    assert [decision.action.get("tool", "final_answer") for decision in rollout.decisions] == [
        "retrieve_docs",
        "retrieve_docs",
        "final_answer",
    ]
    assert all(decision.parse_error is None for decision in rollout.decisions)
    assert all(json.loads(decision.raw_output) == decision.action for decision in rollout.decisions)
    assert "SECRET_GOLD_VALUE" not in "".join(
        decision.system_prompt + decision.user_prompt for decision in rollout.decisions
    )
    assert rollout.verification.joint_success == 1.0
    assert rollout.seed == 9
