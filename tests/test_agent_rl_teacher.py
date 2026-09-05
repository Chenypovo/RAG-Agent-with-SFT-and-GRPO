import json

from app.agent_rl.adapters import build_minimal_registry
from app.agent_rl.env import PersonalRAGEnv
from app.agent_rl.policies import PromptOnlyPolicy
from app.agent_rl.rollouts import run_policy_episode
from app.agent_rl.scripted_agent import ScriptedAgentConfig
from app.agent_rl.sft_data import SFTBuildConfig, build_sft_examples
from app.agent_rl.tasks import AgentRLTask
from app.agent_rl.teacher import (
    ScriptedTeacherPolicy,
    VerifierGuidedTeacherPolicy,
    generate_teacher_candidates,
    select_teacher_rollout,
)


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


def test_verifier_guided_teacher_selects_adaptive_three_decision_candidate():
    chunks = {
        "alpha clue": [{
            "metadata": {
                "source": "docs/alpha",
                "chunk_id": 0,
                "text": "Alpha points to the Beta bridge.",
            }
        }],
        "beta bridge": [{
            "metadata": {
                "source": "docs/beta",
                "chunk_id": 0,
                "text": "Beta contains the second supporting fact.",
            }
        }],
    }
    outputs = iter([
        # Candidate 0 stops with only half of the required evidence.
        '{"tool":"retrieve_docs","args":{"query":"alpha clue"}}',
        '{"final_answer":true}',
        # Candidate 1 observes the first result, continues, then stops.
        '{"tool":"retrieve_docs","args":{"query":"alpha clue"}}',
        '{"tool":"retrieve_docs","args":{"query":"beta bridge"}}',
        '{"final_answer":true}',
    ])
    seen_prompts = []

    def complete(system_prompt, user_prompt):
        seen_prompts.append(system_prompt + user_prompt)
        return next(outputs)

    env = PersonalRAGEnv(
        registry=build_minimal_registry(lambda query: chunks.get(query, [])),
        finalize_fn=lambda task, events: "SECRET_GOLD_VALUE",
        allowed_tools=("retrieve_docs",),
        max_steps=5,
        env_version="llm-teacher-test-v1",
    )
    task = AgentRLTask(
        task_id="adaptive-teacher",
        question="Which two facts answer the bridge question?",
        gold_answers=("SECRET_GOLD_VALUE",),
        gold_evidence_ids=("docs/alpha#0", "docs/beta#0"),
    )

    selection = generate_teacher_candidates(
        env,
        task,
        VerifierGuidedTeacherPolicy(complete),
        candidates_per_task=2,
        seed=20,
    )

    assert [len(row.decisions) for row in selection.candidates] == [2, 3]
    assert selection.selected_candidate_index == 1
    assert selection.selected.verification.complete_sentence_evidence == 1.0
    assert [
        decision.action.get("tool", "final_answer")
        for decision in selection.selected.decisions
    ] == ["retrieve_docs", "retrieve_docs", "final_answer"]
    assert "SECRET_GOLD_VALUE" not in "".join(seen_prompts)
    audit = selection.to_audit_dict()
    assert audit["selected_candidate_index"] == 1
    assert audit["candidates"][1]["action_counts"] == {
        "retrieve_docs": 2,
        "final_answer": 1,
    }
    examples, report = build_sft_examples(
        [selection.selected.to_dict()],
        config=SFTBuildConfig(success_metric="CompleteSentenceEvidence"),
    )
    assert [example["step"] for example in examples] == [0, 1, 2]
    assert [example["parsed_action"].get("tool", "final_answer") for example in examples] == [
        "retrieve_docs",
        "retrieve_docs",
        "final_answer",
    ]
    assert report["augmented_examples"] == 0


def test_verifier_guided_teacher_breaks_quality_tie_with_fewer_tool_calls():
    chunks = [{
        "metadata": {
            "source": "docs/answer",
            "chunk_id": 0,
            "text": "The complete answer is here.",
        }
    }]
    task = AgentRLTask(
        task_id="cost-tie-break",
        question="Where is the answer?",
        gold_answers=("answer",),
        gold_evidence_ids=("docs/answer#0",),
    )
    env = PersonalRAGEnv(
        registry=build_minimal_registry(lambda query: chunks),
        finalize_fn=lambda task, events: "answer",
        allowed_tools=("retrieve_docs",),
        max_steps=4,
    )
    long_outputs = iter([
        '{"tool":"retrieve_docs","args":{"query":"first"}}',
        '{"tool":"retrieve_docs","args":{"query":"second"}}',
        '{"final_answer":true}',
    ])
    short_outputs = iter([
        '{"tool":"retrieve_docs","args":{"query":"direct"}}',
        '{"final_answer":true}',
    ])
    long_rollout = run_policy_episode(
        env,
        task,
        VerifierGuidedTeacherPolicy(lambda _system, _user: next(long_outputs)),
        seed=1,
    )
    short_rollout = run_policy_episode(
        env,
        task,
        VerifierGuidedTeacherPolicy(lambda _system, _user: next(short_outputs)),
        seed=2,
    )

    selection = select_teacher_rollout([long_rollout, short_rollout])

    assert selection.selected_candidate_index == 1
    assert len(selection.selected.decisions) == 2


def test_verifier_guided_teacher_stops_sampling_after_clean_complete_candidate():
    outputs = iter([
        '{"tool":"retrieve_docs","args":{"query":"direct"}}',
        '{"final_answer":true}',
    ])
    env = PersonalRAGEnv(
        registry=build_minimal_registry(lambda query: [{
            "metadata": {
                "source": "docs/answer",
                "chunk_id": 0,
                "text": "Complete evidence.",
            }
        }]),
        finalize_fn=lambda task, events: "answer",
        allowed_tools=("retrieve_docs",),
        max_steps=5,
    )
    task = AgentRLTask(
        task_id="early-stop",
        question="Where is the answer?",
        gold_answers=("answer",),
        gold_evidence_ids=("docs/answer#0",),
    )

    selection = generate_teacher_candidates(
        env,
        task,
        VerifierGuidedTeacherPolicy(lambda _system, _user: next(outputs)),
        candidates_per_task=4,
    )

    assert len(selection.candidates) == 1
    assert selection.selected.verification.complete_sentence_evidence == 1.0
