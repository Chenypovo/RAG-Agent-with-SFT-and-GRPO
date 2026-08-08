import importlib.util
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_agent_grpo.py"
SPEC = importlib.util.spec_from_file_location("train_agent_grpo", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train_agent_grpo = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = train_agent_grpo
SPEC.loader.exec_module(train_agent_grpo)


def test_repository_grpo_config_is_valid_and_uses_frozen_sft_reference():
    config_path = PROJECT_ROOT / "configs" / "agent_rl" / "qwen3_1.7b_grpo.json"
    config = train_agent_grpo.load_config(config_path)

    assert config["rollout"]["group_size"] == 4
    assert config["rollout"] == {
        "group_size": 4,
        "tasks_per_update": 2,
        "max_steps": 5,
        "max_new_tokens": 128,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": 0,
    }
    assert config["model"]["init_adapter"].startswith("results/agent-sft-2k/")
    assert config["data"]["data_dir"].endswith("hotpotqa_train_2k_v3")
    assert config["training"]["clip_epsilon"] == 0.2
    assert config["training"]["beta"] == 0.04
    assert config["reward"]["evidence_coverage"] > abs(
        config["reward"]["tool_call_cost"]
    )
    assert config["reward"]["premature_final_answer"] < 0


def test_code_and_local_model_provenance_hash_exact_artifacts(tmp_path):
    config_path = tmp_path / "config.json"
    config_path.write_text('{"test": true}\n')
    model_path = tmp_path / "model"
    model_path.mkdir()
    expected_files = {
        "config.json": b'{"model_type": "qwen3"}\n',
        "model.safetensors.index.json": b'{"weight_map": {}}\n',
        "model-00001-of-00002.safetensors": b"shard-one",
        "model-00002-of-00002.safetensors": b"shard-two",
    }
    for name, content in expected_files.items():
        (model_path / name).write_bytes(content)
    (model_path / "tokenizer.json").write_bytes(b"not-requested")

    code = train_agent_grpo._code_provenance(config_path)
    model = train_agent_grpo._base_model_provenance(
        str(model_path),
        revision=None,
    )

    assert code["config"]["sha256"] == hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    assert code["train_script"]["sha256"] == hashlib.sha256(
        SCRIPT_PATH.read_bytes()
    ).hexdigest()
    assert Path(code["grpo_module"]["path"]).name == "grpo.py"
    assert model["source"] == "local_directory"
    assert {
        Path(row["path"]).name: row["sha256"]
        for row in model["local_artifact_hashes"]
    } == {
        name: hashlib.sha256(content).hexdigest()
        for name, content in expected_files.items()
    }


def test_runtime_versions_have_exact_required_keys():
    versions = train_agent_grpo.dependency_versions()

    assert set(versions) == {
        "python",
        "torch",
        "transformers",
        "peft",
        "bitsandbytes",
        "accelerate",
    }
    assert isinstance(versions["python"], str) and versions["python"]


def _temporary_config(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    task = {
        "task_id": "task-1",
        "question": "Who founded Alpha?",
        "gold_answers": ["Beta"],
        "gold_evidence_ids": ["docs/alpha#0"],
    }
    (data_dir / "tasks_train.jsonl").write_text(json.dumps(task) + "\n")
    (data_dir / "bm25.json").write_text("{}\n")
    (data_dir / "manifest.json").write_text("{}\n")
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}\n")
    (adapter / "adapter_model.safetensors").write_bytes(b"test-adapter")
    output = tmp_path / "must-not-exist"
    config = train_agent_grpo.load_config(
        PROJECT_ROOT / "configs" / "agent_rl" / "qwen3_1.7b_grpo.json"
    )
    config["data"]["data_dir"] = str(data_dir)
    config["data"]["max_tasks"] = 1
    config["model"]["init_adapter"] = str(adapter)
    config["output"]["dir"] = str(output)
    config["training"]["max_updates"] = 2
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    return config, config_path, output


def test_dry_run_validates_plan_without_loading_model_or_creating_output(tmp_path):
    config, config_path, output = _temporary_config(tmp_path)

    report = train_agent_grpo.make_dry_run_report(config, config_path=config_path)

    assert report["valid"] is True
    assert report["data"]["available_tasks"] == 1
    assert report["plan"]["planned_episodes"] == 16
    assert report["plan"]["optimization_epochs_per_rollout_batch"] == 2
    assert report["plan"]["reference_policy"] == "frozen copy of the initial SFT adapter"
    assert report["plan"]["loss_scope"] == "generated action tokens only"
    assert report["effects"] == {
        "model_loaded": False,
        "gpu_initialized": False,
        "output_created": False,
    }
    assert not output.exists()


def test_cli_dry_run_applies_small_run_overrides(tmp_path):
    _, config_path, output = _temporary_config(tmp_path)
    report_path = tmp_path / "dry-run.json"

    result = train_agent_grpo.main([
        "--config",
        str(config_path),
        "--max-updates",
        "1",
        "--max-tasks",
        "1",
        "--group-size",
        "2",
        "--tasks-per-update",
        "1",
        "--max-steps",
        "2",
        "--optimization-epochs",
        "1",
        "--dry-run",
        "--report",
        str(report_path),
    ])

    assert result == 0
    report = json.loads(report_path.read_text())
    assert report["plan"]["updates"] == 1
    assert report["plan"]["planned_episodes"] == 2
    assert report["plan"]["maximum_decisions"] == 4
    assert report["plan"]["optimization_epochs"] == 1
    assert not output.exists()


def test_config_rejects_single_rollout_group():
    config = train_agent_grpo.load_config(
        PROJECT_ROOT / "configs" / "agent_rl" / "qwen3_1.7b_grpo.json"
    )
    config["rollout"]["group_size"] = 1

    with pytest.raises(ValueError, match="at least 2"):
        train_agent_grpo.validate_config(config)


def test_config_rejects_sampling_distribution_that_does_not_match_logprobs():
    config = train_agent_grpo.load_config(
        PROJECT_ROOT / "configs" / "agent_rl" / "qwen3_1.7b_grpo.json"
    )
    config["rollout"]["temperature"] = 0.8

    with pytest.raises(ValueError, match="on-policy log probabilities"):
        train_agent_grpo.validate_config(config)


def test_config_rejects_positive_reward_for_extra_valid_tool_call():
    config = train_agent_grpo.load_config(
        PROJECT_ROOT / "configs" / "agent_rl" / "qwen3_1.7b_grpo.json"
    )
    config["reward"]["valid_action_format"] = 0.03
    config["reward"]["tool_call_cost"] = -0.02

    with pytest.raises(ValueError, match="must not earn positive"):
        train_agent_grpo.validate_config(config)


def test_task_selection_is_deterministic_and_wraps():
    tasks = [
        train_agent_grpo.AgentRLTask(
            task_id=f"task-{index}",
            question="question",
            gold_answers=("answer",),
        )
        for index in range(3)
    ]

    selected = train_agent_grpo._select_tasks(tasks, update=3, tasks_per_update=2)

    assert [task.task_id for task in selected] == ["task-1", "task-2"]


class _CharacterTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        rendered = "".join(message["content"] for message in messages)
        return rendered + ("<assistant>" if add_generation_prompt else "")

    @staticmethod
    def __call__(text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def test_prompt_encoding_preserves_prefix_and_latest_observation_when_truncated():
    policy = train_agent_grpo.TrainableGRPOPolicy(
        model=object(),
        tokenizer=_CharacterTokenizer(),
        max_new_tokens=10,
        temperature=0.8,
        top_p=0.9,
        top_k=20,
        max_seq_length=40,
        preserve_prompt_prefix_tokens=8,
        compute_dtype="bfloat16",
    )
    original = "system" + "observation-" * 20 + "<assistant>"

    ids, truncated = policy._encode_prompt("system", "observation-" * 20)

    assert truncated is True
    assert len(ids) == 30
    assert ids[:8] == [ord(value) for value in original[:8]]
    assert ids[-22:] == [ord(value) for value in original[-22:]]


def test_raw_sampler_uses_cached_forward_softmax_and_stops_at_eos():
    torch = pytest.importorskip("torch")

    class FakeModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(8, 2)
            self.embedding.weight.requires_grad_(False)
            self.calls = []

        def get_input_embeddings(self):
            return self.embedding

        def forward(
            self,
            *,
            input_ids,
            attention_mask,
            past_key_values=None,
            use_cache=True,
        ):
            self.calls.append({
                "input_length": input_ids.shape[1],
                "attention_length": attention_mask.shape[1],
                "past": past_key_values,
                "use_cache": use_cache,
            })
            logits = torch.full((1, input_ids.shape[1], 8), -100.0)
            logits[:, -1, 2 if past_key_values is None else 1] = 100.0
            return SimpleNamespace(
                logits=logits,
                past_key_values=("cache", len(self.calls)),
            )

    class FakeTokenizer(_CharacterTokenizer):
        eos_token_id = 1

    model = FakeModel()
    policy = train_agent_grpo.TrainableGRPOPolicy(
        model=model,
        tokenizer=FakeTokenizer(),
        max_new_tokens=4,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        max_seq_length=32,
        preserve_prompt_prefix_tokens=4,
        compute_dtype="bfloat16",
    )
    input_ids = torch.tensor([[3, 4, 5]])

    sampled = policy._sample_raw_policy_tokens(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
    )

    assert sampled == [2, 1]
    assert model.calls == [
        {"input_length": 3, "attention_length": 3, "past": None, "use_cache": True},
        {
            "input_length": 1,
            "attention_length": 4,
            "past": ("cache", 1),
            "use_cache": True,
        },
    ]


def test_adapter_guards_freeze_reference_and_reject_mismatched_start():
    torch = pytest.importorskip("torch")

    class AdapterSlot(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([value]))

    class FakeAdapterModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapters = torch.nn.ModuleDict({
                "policy": AdapterSlot(1.0),
                "reference": AdapterSlot(1.0),
            })
            self.active_adapter = ""

        def set_adapter(self, adapter):
            self.active_adapter = adapter

    model = FakeAdapterModel()
    train_agent_grpo._activate_adapter(model, "policy")
    train_agent_grpo._assert_adapter_training_state(model, active_adapter="policy")
    train_agent_grpo._assert_policy_matches_reference(model)
    assert model.adapters.policy.weight.requires_grad is True
    assert model.adapters.reference.weight.requires_grad is False

    model.adapters.reference.weight.data.add_(1.0)
    with pytest.raises(RuntimeError, match="identical weights"):
        train_agent_grpo._assert_policy_matches_reference(model)


def test_dropout_and_optimizer_finite_guards():
    torch = pytest.importorskip("torch")
    model = torch.nn.Sequential(torch.nn.Dropout(0.25), torch.nn.AlphaDropout(0.1))

    assert train_agent_grpo._disable_all_dropout(model) == 2
    train_agent_grpo._assert_dropout_disabled(model)

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.grad = torch.tensor([1.0])
    optimizer.step()
    train_agent_grpo._assert_optimizer_finite(optimizer)
    optimizer.state[parameter]["exp_avg"].fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="optimizer state"):
        train_agent_grpo._assert_optimizer_finite(optimizer)


def test_prepare_output_dir_cleans_stale_run_only_with_explicit_overwrite(tmp_path):
    output = tmp_path / "grpo-output"
    output.mkdir()
    (output / "stale.json").write_text("stale\n")

    with pytest.raises(RuntimeError, match="not empty"):
        train_agent_grpo._prepare_output_dir(output, overwrite=False)

    train_agent_grpo._prepare_output_dir(output, overwrite=True)
    assert output.is_dir()
    assert list(output.iterdir()) == []


def test_save_policy_adapter_selects_only_policy_and_returns_reloadable_path(tmp_path):
    torch = pytest.importorskip("torch")

    class AdapterSlot(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0]))

    class FakeSaveModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.adapters = torch.nn.ModuleDict({
                "policy": AdapterSlot(),
                "reference": AdapterSlot(),
            })
            self.saved_adapters = None

        def set_adapter(self, _adapter):
            return None

        def save_pretrained(self, output_dir, *, selected_adapters):
            self.saved_adapters = list(selected_adapters)
            adapter_dir = Path(output_dir) / "policy"
            adapter_dir.mkdir()
            (adapter_dir / "adapter_config.json").write_text("{}\n")
            (adapter_dir / "adapter_model.safetensors").write_bytes(b"policy")

    class FakeTokenizer:
        @staticmethod
        def save_pretrained(output_dir):
            (Path(output_dir) / "tokenizer_config.json").write_text("{}\n")

    model = FakeSaveModel()
    output_root = tmp_path / "adapter"

    adapter_path = train_agent_grpo._save_policy_adapter(
        model,
        FakeTokenizer(),
        output_root,
    )

    assert model.saved_adapters == ["policy"]
    assert adapter_path == output_root / "policy"
    assert (adapter_path / "adapter_model.safetensors").is_file()
    assert (adapter_path / "tokenizer_config.json").is_file()
    assert not (output_root / "reference").exists()
