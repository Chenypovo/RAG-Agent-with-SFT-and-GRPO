import json
from pathlib import Path

import pytest

from app.agent_rl.adapters import build_minimal_registry
from app.agent_rl.env import PersonalRAGEnv
from app.agent_rl.tasks import AgentRLTask, load_tasks


CORPUS = [
    {"metadata": {"source": "facts.md", "chunk_id": 1, "text": "The recorded value is 40."}},
    {"metadata": {"source": "other.md", "chunk_id": 2, "text": "An unrelated distractor."}},
]


def retrieve(query):
    return [CORPUS[0]] if "value" in query.lower() else []


def finalize(task, events):
    for event in reversed(events):
        if event.tool == "calculator" and "result" in event.data:
            return str(event.data["result"])
    if any("recorded value" in event.observation for event in events):
        return "40"
    return "unknown"


def make_env(*, max_steps=6):
    return PersonalRAGEnv(
        registry=build_minimal_registry(retrieve),
        finalize_fn=finalize,
        max_steps=max_steps,
        seed=42,
    )


def calc_task():
    return AgentRLTask(
        task_id="calc_001",
        question="Retrieve the value and add two.",
        task_type="calc",
        expected_value=42,
        gold_evidence_ids=("facts.md#1",),
    )


def test_reset_returns_json_serializable_contract():
    obs = make_env().reset(calc_task())
    json.dumps(obs)
    assert obs["task_id"] == "calc_001"
    assert [tool["name"] for tool in obs["available_tools"]] == ["retrieve_docs", "calculator"]
    assert obs["history"] == [] and obs["remaining_steps"] == 6


def test_retrieve_calculate_and_finish_successfully():
    env = make_env()
    env.reset(calc_task())

    obs, reward, done, info = env.step({"tool": "retrieve_docs", "args": {"query": "value"}})
    assert not done and info["evidence_ids"] == ["facts.md#1"]
    assert reward == pytest.approx(0.03)  # valid format minus tool cost

    obs, reward, done, _ = env.step({"tool": "calculator", "args": {"expression": "40 + 2"}})
    assert not done and obs["history"][-1]["data"]["result"] == 42
    assert reward == pytest.approx(0.03)

    obs, reward, done, info = env.step({"final_answer": True})
    assert done and obs["answer"] == "42" and info["stop_reason"] == "final_answer"
    assert info["reward_breakdown"]["task_success"] == 1.0
    assert info["reward_breakdown"]["evidence_coverage"] == 0.3
    assert reward == pytest.approx(1.35)
    assert len(env.transitions) == 3


def test_duplicate_call_is_observed_and_penalized():
    env = make_env()
    env.reset(calc_task())
    action = {"tool": "retrieve_docs", "args": {"query": "value"}}
    env.step(action)
    obs, reward, done, info = env.step(action)

    assert not done
    assert obs["history"][-1]["error"] == "duplicate call"
    assert info["reward_breakdown"]["duplicate_call"] == -0.1
    assert reward == pytest.approx(-0.05)


def test_unknown_tool_at_budget_boundary_terminates_cleanly():
    env = make_env(max_steps=1)
    env.reset(calc_task())
    obs, reward, done, info = env.step({"tool": "web_search", "args": {"query": "x"}})

    assert done and info["stop_reason"] == "budget"
    assert "not allowed" in obs["last_message"]
    assert info["reward_breakdown"]["invalid_action"] == -0.1
    assert info["reward_breakdown"]["budget_exhausted"] == -0.3
    assert reward == pytest.approx(-0.35)


def test_non_object_action_becomes_observable_error():
    env = make_env()
    env.reset(calc_task())
    obs, reward, done, info = env.step("not-json")

    assert not done
    assert "JSON object" in obs["last_message"]
    assert info["reward_breakdown"]["invalid_action"] == -0.1
    assert reward == pytest.approx(-0.1)


def test_premature_final_answer_gets_partial_coverage_and_penalty():
    task = AgentRLTask(
        task_id="mh_001",
        question="Use two pieces of evidence.",
        gold_answers=("40",),
        gold_evidence_ids=("facts.md#1", "missing.md#9"),
    )
    env = make_env()
    env.reset(task)
    env.step({"tool": "retrieve_docs", "args": {"query": "value"}})
    _, reward, done, info = env.step({"final_answer": True})

    assert done
    assert info["reward_breakdown"]["evidence_coverage"] == pytest.approx(0.15)
    assert info["reward_breakdown"]["premature_final_answer"] == -0.3
    assert reward == pytest.approx(0.9)  # success + valid + partial coverage - premature stop


def test_reset_isolates_episode_state():
    env = make_env()
    env.reset(calc_task())
    env.step({"tool": "retrieve_docs", "args": {"query": "value"}})
    obs = env.reset(calc_task(), seed=7)
    assert obs["history"] == [] and obs["evidence_ids"] == []
    assert env.transitions == []


def test_smoke_tasks_load_with_training_schema():
    path = Path(__file__).parents[1] / "data" / "agent_rl" / "smoke_tasks.jsonl"
    tasks = load_tasks(path)
    assert [task.task_type for task in tasks] == ["multihop", "calc"]
    assert tasks[1].expected_value == 80
