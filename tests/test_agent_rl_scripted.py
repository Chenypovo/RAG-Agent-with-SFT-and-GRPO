import pytest

from app.agent_rl.adapters import build_minimal_registry
from app.agent_rl.env import PersonalRAGEnv
from app.agent_rl.rewards import RewardConfig
from app.agent_rl.scripted_agent import (
    ScriptedAgentConfig,
    ScriptedTwoHopPolicy,
    build_follow_up_query,
    run_scripted_episode,
)
from app.agent_rl.tasks import AgentRLTask
from app.agent_rl.verifiers import (
    answer_exact_match,
    answer_f1,
    normalize_hotpot_answer,
    verify_task,
)


def test_hotpot_answer_metrics_normalize_articles_and_punctuation():
    assert normalize_hotpot_answer("The Arthur's Magazine!") == "arthurs magazine"
    assert answer_exact_match("the Delhi", ("Delhi",)) == 1.0
    assert answer_f1("President Nixon", ("President Richard Nixon",)) == pytest.approx(0.8)


def test_verifier_reports_answer_and_evidence_separately():
    task = AgentRLTask(
        task_id="verify_1",
        question="Where?",
        gold_answers=("Delhi",),
        gold_evidence_ids=("docs/a#0", "docs/b#1"),
        metadata={"gold_document_ids": ["docs/a", "docs/b"]},
    )
    result = verify_task(
        task,
        predicted_answer="the Delhi",
        evidence_ids=("docs/a#0", "docs/b#9"),
    )
    assert result.answer_em == 1.0
    assert result.sentence_recall == 0.5
    assert result.document_recall == 1.0
    assert result.joint_success == 0.0


def test_follow_up_query_extracts_visible_bridge_entity():
    event = {
        "observation": (
            "retrieved chunks:\n"
            "[hotpotqa/example#0] Badr Hari He competed in the Global Fighting Championship.\n"
            "[hotpotqa/other#0] Other page Nothing useful."
        )
    }
    assert build_follow_up_query("Who was the kick boxer?", event) == "Global Fighting Championship"


def test_scripted_policy_runs_two_retrievals_then_stops():
    chunks = {
        "Who founded Alpha?": [{
            "metadata": {
                "source": "docs/alpha",
                "chunk_id": 0,
                "text": "Alpha Alpha was founded by Beta Person.",
            }
        }],
        "Beta Person": [{
            "metadata": {
                "source": "docs/beta",
                "chunk_id": 0,
                "text": "Beta Person Beta Person founded Alpha.",
            }
        }],
    }
    registry = build_minimal_registry(lambda query: chunks.get(query, []))
    env = PersonalRAGEnv(
        registry=registry,
        finalize_fn=lambda task, events: "Beta Person",
        allowed_tools=("retrieve_docs",),
        max_steps=3,
        reward_config=RewardConfig(premature_final_answer=0.0),
    )
    task = AgentRLTask(
        task_id="bridge_1",
        question="Who founded Alpha?",
        gold_answers=("Beta Person",),
        gold_evidence_ids=("docs/alpha#0", "docs/beta#0"),
    )
    rollout = run_scripted_episode(
        env,
        task,
        ScriptedTwoHopPolicy(ScriptedAgentConfig(
            retrieval_hops=2,
            first_hop_k=1,
            follow_up_k=1,
        )),
    )
    assert [action.get("tool", "final_answer") for action in rollout.actions] == [
        "retrieve_docs", "retrieve_docs", "final_answer",
    ]
    assert rollout.verification.complete_sentence_evidence == 1.0
    assert rollout.verification.joint_success == 1.0
