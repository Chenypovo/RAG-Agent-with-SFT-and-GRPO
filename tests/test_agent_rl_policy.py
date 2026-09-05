import hashlib
import json
import sys
import types

import pytest

from app.agent_rl.adapters import build_minimal_registry
from app.agent_rl.env import PersonalRAGEnv
from app.agent_rl.evaluation import aggregate_policy_rollouts
from app.agent_rl.finalizers import FrozenAnswerFinalizer
from app.agent_rl.hf_backend import TransformersChatBackend
from app.agent_rl.policies import PromptOnlyPolicy
from app.agent_rl.rewards import answer_is_correct
from app.agent_rl.rollouts import run_policy_episode
from app.agent_rl.tasks import AgentRLTask
from app.agent_rl.trajectory import ToolEvent


def sequence_complete(outputs):
    remaining = list(outputs)

    def complete(system_prompt, user_prompt):
        assert system_prompt and user_prompt
        return remaining.pop(0)

    return complete


def test_transformers_backend_shares_model_and_disables_thinking():
    torch = pytest.importorskip("torch")

    class FakeTokenizer:
        pad_token_id = 0

        def __init__(self):
            self.template_options = {}

        def apply_chat_template(self, messages, **options):
            self.template_options = options
            return "rendered prompt"

        def __call__(self, prompt, return_tensors):
            return {"input_ids": torch.tensor([[1, 2]])}

        def decode(self, tokens, skip_special_tokens):
            return '{"final_answer": true}'

    class FakeEmbeddings:
        weight = torch.zeros(1)

    class FakeModel:
        def eval(self):
            return self

        def get_input_embeddings(self):
            return FakeEmbeddings()

        def generate(self, **options):
            assert options["do_sample"] is False
            assert options["max_new_tokens"] == 16
            return torch.tensor([[1, 2, 3]])

    tokenizer = FakeTokenizer()
    backend = TransformersChatBackend(
        "fake/model",
        tokenizer=tokenizer,
        model=FakeModel(),
        enable_thinking=False,
    )

    output = backend.make_complete_fn(max_new_tokens=16)("system", "user")

    assert output == '{"final_answer": true}'
    assert tokenizer.template_options["enable_thinking"] is False


def test_transformers_backend_records_sampling_options():
    torch = pytest.importorskip("torch")
    captured = {}

    class FakeTokenizer:
        pad_token_id = 0

        def apply_chat_template(self, messages, **options):
            return "prompt"

        def __call__(self, prompt, return_tensors):
            return {"input_ids": torch.tensor([[1]])}

        def decode(self, tokens, skip_special_tokens):
            return '{"final_answer": true}'

    class FakeEmbeddings:
        weight = torch.zeros(1)

    class FakeModel:
        def eval(self):
            return self

        def get_input_embeddings(self):
            return FakeEmbeddings()

        def generate(self, **options):
            captured.update(options)
            return torch.tensor([[1, 2]])

    backend = TransformersChatBackend(
        "fake/model", tokenizer=FakeTokenizer(), model=FakeModel()
    )
    backend.make_complete_fn(
        max_new_tokens=12, temperature=0.7, top_p=0.8, top_k=20
    )("system", "user")

    assert captured["do_sample"] is True
    assert captured["temperature"] == pytest.approx(0.7)
    assert captured["top_p"] == pytest.approx(0.8)
    assert captured["top_k"] == 20


def test_transformers_backend_loads_local_adapter_and_records_hashes(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    config_bytes = b'{"peft_type":"LORA"}\n'
    weights_bytes = b"fake adapter weights"
    (tmp_path / "adapter_config.json").write_bytes(config_bytes)
    (tmp_path / "adapter_model.safetensors").write_bytes(weights_bytes)
    calls = {}

    class FakeBaseModel:
        config = types.SimpleNamespace(_commit_hash="a" * 40)

        def eval(self):
            return self

    class FakeAdapterModel:
        config = types.SimpleNamespace(_commit_hash="b" * 40)

        def eval(self):
            calls["eval"] = True
            return self

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path, *, is_trainable):
            calls.update({
                "model": model,
                "adapter_path": adapter_path,
                "is_trainable": is_trainable,
            })
            return FakeAdapterModel()

    monkeypatch.setitem(sys.modules, "peft", types.SimpleNamespace(PeftModel=FakePeftModel))
    tokenizer = types.SimpleNamespace(init_kwargs={})
    base_model = FakeBaseModel()
    backend = TransformersChatBackend(
        "fake/model",
        tokenizer=tokenizer,
        model=base_model,
        adapter_path=str(tmp_path),
    )

    assert calls["model"] is base_model
    assert calls["adapter_path"] == str(tmp_path)
    assert calls["is_trainable"] is False
    assert calls["eval"] is True
    assert backend.resolved_commit == "a" * 40
    assert backend.adapter_provenance == {
        "path": str(tmp_path.resolve()),
        "adapter_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "weight_files": [{
            "path": "adapter_model.safetensors",
            "sha256": hashlib.sha256(weights_bytes).hexdigest(),
            "size_bytes": len(weights_bytes),
        }],
    }


def test_prompt_policy_accepts_only_strict_executable_action():
    policy = PromptOnlyPolicy(
        sequence_complete([
            '{"tool":"retrieve_docs","args":{"query":"Alpha founder"}}',
            "```json\n{\"final_answer\": true}\n```",
        ])
    )
    observation = {"question": "Who founded Alpha?", "available_tools": [], "history": []}

    valid = policy.decide(observation)
    invalid = policy.decide(observation)

    assert valid.action == {"tool": "retrieve_docs", "args": {"query": "Alpha founder"}}
    assert valid.parse_error is None
    assert invalid.action is None
    assert invalid.parse_error


def test_prompt_policy_compaction_keeps_valid_json():
    captured = {}

    def complete(system_prompt, user_prompt):
        captured["user"] = user_prompt
        return '{"final_answer": true}'

    policy = PromptOnlyPolicy(complete, max_observation_chars=256)
    policy.decide({
        "question": "Q" * 600,
        "history": [{"observation": "X" * 600}, {"observation": "Y" * 600}],
        "remaining_steps": 2,
    })
    payload = captured["user"].split("Environment observation:\n", 1)[1].split(
        "\n\nChoose the next single action", 1
    )[0]

    assert json.loads(payload)["observation_truncated"] is True


def test_prompt_policy_compacts_history_and_marks_duplicate_recovery():
    captured = {}

    def complete(system_prompt, user_prompt):
        captured["user"] = user_prompt
        return '{"final_answer": true}'

    PromptOnlyPolicy(complete).decide({
        "question": "Who founded Alpha?",
        "available_tools": [{"name": "retrieve_docs"}],
        "remaining_steps": 2,
        "last_message": "duplicate call: choose different args or final_answer",
        "history": [
            {
                "step_index": 0,
                "tool": "retrieve_docs",
                "args": {"query": "Alpha founder"},
                "ok": True,
                "error": None,
                "observation": "Alpha was founded by Beta Person.",
            },
            {
                "step_index": 1,
                "tool": "retrieve_docs",
                "args": {"query": "Alpha founder"},
                "ok": False,
                "error": "duplicate call",
                "observation": "duplicate call",
            },
        ],
    })
    payload = captured["user"].split("Environment observation:\n", 1)[1].split(
        "\n\nChoose the next single action", 1
    )[0]
    state = json.loads(payload)

    assert state["retrieved_evidence"] == ["Alpha was founded by Beta Person."]
    assert state["previous_tool_calls"][0]["args"] == {"query": "Alpha founder"}
    assert "Do not repeat" in state["required_next_action"]
    assert "evidence_ids" not in state


def test_prompt_policy_retains_four_retrieval_observations_for_adaptive_teacher():
    captured = {}

    def complete(system_prompt, user_prompt):
        captured["user"] = user_prompt
        return '{"final_answer": true}'

    PromptOnlyPolicy(complete).decide({
        "question": "Resolve a multi-hop chain.",
        "available_tools": [{"name": "retrieve_docs"}],
        "remaining_steps": 1,
        "last_message": "evidence four",
        "history": [
            {
                "step_index": index,
                "tool": "retrieve_docs",
                "args": {"query": f"query {index}"},
                "ok": True,
                "error": None,
                "observation": f"evidence {index}",
            }
            for index in range(1, 5)
        ],
    })
    payload = captured["user"].split("Environment observation:\n", 1)[1].split(
        "\n\nChoose the next single action", 1
    )[0]
    state = json.loads(payload)

    assert state["retrieved_evidence"] == [
        "evidence 1",
        "evidence 2",
        "evidence 3",
        "evidence 4",
    ]
    assert len(state["previous_tool_calls"]) == 4


def test_finalizer_compaction_keeps_valid_json():
    captured = {}

    def complete(system_prompt, user_prompt):
        captured["user"] = user_prompt
        return "answer"

    event = ToolEvent(
        step_index=0,
        tool="retrieve_docs",
        args={"query": "q"},
        observation="evidence " * 200,
        ok=True,
    )
    task = AgentRLTask(task_id="compact", question="Q?", gold_answers=("answer",))
    FrozenAnswerFinalizer(complete, max_event_chars=256)(task, [event])
    payload = captured["user"].split("Tool observations:\n", 1)[1].split(
        "\n\nReturn only", 1
    )[0]

    assert json.loads(payload)[0]["observation_truncated"] is True


def test_frozen_finalizer_never_receives_gold_answer():
    captured = {}

    def complete(system_prompt, user_prompt):
        captured["system"] = system_prompt
        captured["user"] = user_prompt
        return "Beta Person"

    task = AgentRLTask(
        task_id="hidden_gold",
        question="Who founded Alpha?",
        gold_answers=("SECRET_GOLD_VALUE",),
    )
    event = ToolEvent(
        step_index=0,
        tool="retrieve_docs",
        args={"query": "Alpha"},
        observation="[docs/alpha#0] Alpha was founded by Beta Person.",
        ok=True,
    )
    answer = FrozenAnswerFinalizer(complete)(task, [event])

    assert answer == "Beta Person"
    assert "SECRET_GOLD_VALUE" not in captured["system"] + captured["user"]
    assert "Alpha was founded by Beta Person" in captured["user"]


def test_environment_answer_reward_matches_hotpot_article_normalization():
    task = AgentRLTask(task_id="normalized", question="Where?", gold_answers=("Delhi",))

    assert answer_is_correct(task, "the Delhi")


def test_prompt_policy_rollout_records_provenance_and_endpoint_metrics():
    chunks = [{
        "metadata": {
            "source": "docs/alpha",
            "chunk_id": 0,
            "text": "Alpha was founded by Beta Person.",
        }
    }]
    registry = build_minimal_registry(lambda query: chunks if "Alpha" in query else [])
    policy = PromptOnlyPolicy(sequence_complete([
        json.dumps({"tool": "retrieve_docs", "args": {"query": "Alpha founder"}}),
        json.dumps({"final_answer": True}),
    ]))
    env = PersonalRAGEnv(
        registry=registry,
        finalize_fn=FrozenAnswerFinalizer(lambda system, user: "Beta Person"),
        allowed_tools=("retrieve_docs",),
        max_steps=3,
        env_version="test-env-v1",
    )
    task = AgentRLTask(
        task_id="alpha_1",
        question="Who founded Alpha?",
        gold_answers=("Beta Person",),
        gold_evidence_ids=("docs/alpha#0",),
    )

    rollout = run_policy_episode(env, task, policy, seed=7)
    report = aggregate_policy_rollouts([rollout])

    assert rollout.seed == 7 and rollout.env_version == "test-env-v1"
    assert rollout.verification.joint_success == 1.0
    assert rollout.transitions[0]["observation_before"]["question"] == task.question
    assert rollout.decisions[0].raw_output.startswith("{")
    assert report["metrics"]["InvalidActionRate"] == 0.0
    assert report["metrics"]["MeanToolCalls"] == 1.0
    assert report["metrics"]["CallsPerJointSuccess"] == 1.0
    assert report["metrics"]["FinalizerErrorRate"] == 0.0


def test_invalid_model_output_is_counted_in_controller_metrics():
    policy = PromptOnlyPolicy(sequence_complete([
        "not JSON",
        json.dumps({"final_answer": True}),
    ]))
    env = PersonalRAGEnv(
        registry=build_minimal_registry(lambda query: []),
        finalize_fn=lambda task, events: "unknown",
        allowed_tools=("retrieve_docs",),
        max_steps=2,
    )
    task = AgentRLTask(
        task_id="invalid_1",
        question="What is unknown?",
        gold_answers=("unknown",),
        gold_evidence_ids=("docs/missing#0",),
    )

    rollout = run_policy_episode(env, task, policy)
    report = aggregate_policy_rollouts([rollout])

    assert rollout.decisions[0].parse_error
    assert report["metrics"]["InvalidActionRate"] == pytest.approx(0.5)
    assert report["n_joint_successes"] == 0
    assert report["metrics"]["CallsPerJointSuccess"] is None
