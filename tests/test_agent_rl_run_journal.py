import json
import random

import pytest

from app.agent_rl.adapters import build_minimal_registry
from app.agent_rl.behaviour_audit import audit_behaviour
from app.agent_rl.env import PersonalRAGEnv
from app.agent_rl.policies import PromptOnlyPolicy
from app.agent_rl.rollouts import run_policy_episode
from app.agent_rl.run_journal import CandidateJournal, capture_rng, restore_rng
from app.agent_rl.tasks import AgentRLTask
from app.agent_rl.teacher import VerifierGuidedTeacherPolicy, generate_teacher_candidates


def _task_env():
    task = AgentRLTask(task_id="journal", question="Question", gold_answers=("Answer",), gold_evidence_ids=("doc#0",))
    def retrieve(query):
        if query == "fail":
            raise RuntimeError("controlled fixture failure")
        return [{"metadata": {"source": "doc", "chunk_id": 0, "text": "Answer"}}] if query == "good" else []
    env = PersonalRAGEnv(registry=build_minimal_registry(retrieve), finalize_fn=lambda task, events: "Answer", allowed_tools=("retrieve_docs",), max_steps=5)
    return task, env


def test_journal_restores_exact_rng_and_completed_candidate_without_resampling(tmp_path):
    import numpy as np
    import torch
    task, env = _task_env()
    policy = VerifierGuidedTeacherPolicy(lambda s, u: '{"final_answer":true}')
    journal = CandidateJournal(tmp_path / "journal", {"seed": 42})
    generate_teacher_candidates(env, task, policy, candidates_per_task=1,
        on_candidate=lambda i, r, t: journal.append(0, i, r, t))
    expected = (random.random(), np.random.random(), torch.rand(3))
    resumed = CandidateJournal(tmp_path / "journal", {"seed": 42}, resume=True)
    resumed.restore()
    actual = (random.random(), np.random.random(), torch.rand(3))
    assert expected[:2] == actual[:2]
    assert torch.equal(expected[2], actual[2])
    candidate = resumed.candidates_for(0, task.task_id)
    never_sample = VerifierGuidedTeacherPolicy(lambda s, u: pytest.fail("resampled completed candidate"))
    selection = generate_teacher_candidates(env, task, never_sample, candidates_per_task=1, existing_candidates=candidate)
    assert selection.selected.to_dict() == candidate[0].to_dict()
    with pytest.raises(ValueError, match="mismatch"):
        CandidateJournal(tmp_path / "journal", {"seed": 43}, resume=True)
    path = next((tmp_path / "journal").glob("candidate-*.json"))
    record = json.loads(path.read_text()); record["payload"]["runtime_seconds"] = -1
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="hash mismatch"):
        CandidateJournal(tmp_path / "journal", {"seed": 42}, resume=True)


def test_recovery_counts_each_failure_once_and_keeps_denominators():
    task, env = _task_env()
    outputs = iter([
        '{"tool":"retrieve_docs","args":{"query":"miss"}}',
        '{"tool":"retrieve_docs","args":{"query":"fail"}}',
        '{"tool":"retrieve_docs","args":{"query":"fail"}}',
        '{"tool":"retrieve_docs","args":{"query":"good"}}',
        '{"final_answer":true}',
    ])
    rollout = run_policy_episode(env, task, PromptOnlyPolicy(lambda s, u: next(outputs)))
    audit = audit_behaviour([rollout], gold_by_task={task.task_id: task.gold_evidence_ids})
    for kind in ("empty", "failed", "duplicate"):
        assert audit["recovery"][kind]["exposures"] == 1
        assert audit["recovery"][kind]["new_evidence_rate"] == 1
        assert audit["recovery"][kind]["joint_success_rate"] == 1
    assert audit["retrieval_call_buckets"]["4+"] == 1
    assert audit["duplicate_call_rate_per_decision"] == 0.2
    assert audit["duplicate_call_rate_per_retrieval_attempt"] == 0.25
    assert audit["recovery"]["contradictory"]["exposures"] is None


def test_invalid_json_is_not_counted_as_a_stop_or_successful_recovery():
    task, env = _task_env()
    outputs = iter(["not json", '{"final_answer":true}'])
    rollout = run_policy_episode(env, task, PromptOnlyPolicy(lambda s, u: next(outputs)))
    audit = audit_behaviour([rollout])
    assert audit["action_counts"] == {"invalid_json_action": 1, "final_answer": 1}
    assert audit["json_invalid_rate"] == 0.5
    assert audit["premature_stop_rate"] == 1
    assert audit["recovery"]["empty"]["new_evidence_rate"] is None


def test_tool_named_final_answer_is_reported_as_an_unsupported_action():
    task, env = _task_env()
    outputs = iter(['{"tool":"final_answer","args":{}}', '{"final_answer":true}'])
    rollout = run_policy_episode(env, task, PromptOnlyPolicy(lambda s, u: next(outputs)))
    audit = audit_behaviour([rollout])
    assert audit["action_counts"] == {"unsupported_tool:final_answer": 1, "final_answer": 1}
    assert audit["unsupported_tool_action_rate"] == 0.5
    assert audit["mean_retrieval_calls"] == 0
    assert audit["json_invalid_rate"] == 0
