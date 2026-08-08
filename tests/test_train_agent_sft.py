import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "train_agent_sft.py"
SPEC = importlib.util.spec_from_file_location("train_agent_sft", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
train_agent_sft = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(train_agent_sft)


def _messages_row(task_id="task-1"):
    return {
        "schema_version": "agent-sft-example-v1",
        "task_id": task_id,
        "messages": [
            {"role": "system", "content": "Return exactly one JSON action."},
            {"role": "user", "content": "Question and current evidence"},
            {
                "role": "assistant",
                "content": '{"tool":"retrieve_docs","args":{"query":"evidence"}}',
            },
        ],
    }


def test_repository_qlora_config_is_valid():
    config_path = PROJECT_ROOT / "configs" / "agent_rl" / "qwen3_1.7b_qlora.json"
    config = train_agent_sft.load_config(config_path)

    assert config["model"]["name_or_path"] == "Qwen/Qwen3-1.7B"
    assert config["quantization"] == {
        "bits": 4,
        "quant_type": "nf4",
        "double_quant": True,
    }
    assert config["data"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "q_proj" in config["lora"]["target_modules"]


def test_load_training_rows_accepts_both_builder_output_formats(tmp_path):
    path = tmp_path / "sft.jsonl"
    messages_row = _messages_row()
    prompt_response_row = {
        "task_id": "task-2",
        "prompt": messages_row["messages"][:-1],
        "response": messages_row["messages"][-1]["content"],
    }
    path.write_text(
        json.dumps(messages_row) + "\n" + json.dumps(prompt_response_row) + "\n",
        encoding="utf-8",
    )

    rows, stats = train_agent_sft.load_training_rows(path)

    assert len(rows) == 2
    assert rows[1][-1]["role"] == "assistant"
    assert stats["examples"] == 2
    assert stats["unique_task_ids"] == 2


def test_load_training_rows_rejects_label_after_assistant(tmp_path):
    path = tmp_path / "bad.jsonl"
    row = _messages_row()
    row["messages"].append({"role": "user", "content": "leaked label"})
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="final message must be assistant"):
        train_agent_sft.load_training_rows(path)


def test_dry_run_validates_without_downloading_or_creating_output(tmp_path):
    input_path = tmp_path / "sft.jsonl"
    input_path.write_text(
        json.dumps(_messages_row()) + "\n" + json.dumps(_messages_row("task-2")) + "\n",
        encoding="utf-8",
    )
    output_dir = tmp_path / "must-not-be-created"
    report_path = tmp_path / "dry-run.json"
    config_path = PROJECT_ROOT / "configs" / "agent_rl" / "qwen3_1.7b_qlora.json"

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--config",
            str(config_path),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
            "--dry-run",
            "--report",
            str(report_path),
        ],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    report = json.loads(completed.stdout)
    assert report == json.loads(report_path.read_text(encoding="utf-8"))
    assert report["valid"] is True
    assert report["data"]["examples"] == 2
    assert report["plan"]["estimated_optimizer_updates"] == 2
    assert report["effects"] == {
        "model_downloaded": False,
        "gpu_initialized": False,
        "output_created": False,
    }
    assert set(report["dependencies"]) == set(train_agent_sft.REQUIRED_TRAINING_PACKAGES)
    assert not output_dir.exists()


class _CharacterTokenizer:
    """Small deterministic stand-in for chat-template masking tests."""

    @staticmethod
    def apply_chat_template(messages, *, tokenize, add_generation_prompt, **kwargs):
        assert tokenize is False
        assert kwargs == {"enable_thinking": False}
        rendered = "".join(
            f"<{message['role']}>{message['content']}</{message['role']}>"
            for message in messages
        )
        if add_generation_prompt:
            rendered += "<assistant>"
        return rendered

    @staticmethod
    def __call__(text, *, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(character) for character in text]}


def test_tokenization_masks_prompt_and_preserves_response_labels():
    messages = _messages_row()["messages"]

    tokenized, stats = train_agent_sft._tokenize_rows(
        _CharacterTokenizer(),
        [messages],
        max_seq_length=2048,
        preserve_prompt_prefix_tokens=64,
        chat_template_kwargs={"enable_thinking": False},
    )

    row = tokenized[0]
    first_label = next(index for index, value in enumerate(row["labels"]) if value != -100)
    assert row["labels"][:first_label] == [-100] * first_label
    assert row["labels"][first_label:] == row["input_ids"][first_label:]
    assert stats == {"examples": 1, "truncated_prompts": 0}


def test_tokenization_truncates_only_prompt_and_keeps_trainable_response():
    messages = _messages_row()["messages"]
    messages[1]["content"] = "long observation " * 100

    tokenized, stats = train_agent_sft._tokenize_rows(
        _CharacterTokenizer(),
        [messages],
        max_seq_length=256,
        preserve_prompt_prefix_tokens=32,
        chat_template_kwargs={"enable_thinking": False},
    )

    row = tokenized[0]
    assert len(row["input_ids"]) == 256
    assert any(value != -100 for value in row["labels"])
    assert stats == {"examples": 1, "truncated_prompts": 1}
