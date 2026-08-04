import json
import subprocess
import sys
from pathlib import Path

import hashlib

import pytest

from app.agent_rl.sft_data import SFTBuildConfig, build_sft_examples


def decision(step, *, parse_error=None):
    action = (
        {"final_answer": True}
        if step == 1
        else {"tool": "retrieve_docs", "args": {"query": "Alpha founder"}}
    )
    return {
        "system_prompt": "Choose one action.",
        "user_prompt": f"Environment observation for step {step}",
        "raw_output": json.dumps(action),
        "action": action,
        "parse_error": parse_error,
        "policy_version": "teacher-v2",
    }


def rollout(task_id, score, *, decisions=None):
    selected_decisions = decisions if decisions is not None else [decision(0), decision(1)]
    tool_event = {
        "step_index": 0,
        "tool": "retrieve_docs",
        "args": {"query": "Alpha founder"},
        "observation": "Alpha was founded by Beta Person.",
        "ok": True,
        "data": {"evidence_ids": ["docs/alpha#0"]},
        "error": None,
    }
    transitions = []
    history = []
    for step, selected_decision in enumerate(selected_decisions):
        if selected_decision.get("action", {}).get("tool"):
            history = [*history, {**tool_event, "step_index": step}]
        transitions.append({
            "step_index": step,
            "action": selected_decision.get("action"),
            "observation_after": {"history": list(history)},
            "reward_breakdown": {
                "invalid_action": 0.0,
                "duplicate_call": 0.0,
            },
        })
    return {
        "task_id": task_id,
        "policy_version": "teacher-v2",
        "decisions": selected_decisions,
        "transitions": transitions,
        "verification": {
            "AnswerEM": score,
            "JointEM": score,
            "JointSuccess": score,
        },
        "verifier_provenance": {
            "name": "hotpotqa-verifier",
            "version": "v1",
            "gold_answers": ["MUST_NOT_LEAK"],
        },
        "gold_answers": ["MUST_NOT_LEAK"],
    }


def test_builder_keeps_only_clean_successful_episodes_and_splits_decisions():
    valid = rollout("kept", 1.0)
    parse_failed = rollout(
        "parse-failed",
        1.0,
        decisions=[decision(0, parse_error="invalid JSON"), decision(1)],
    )
    unsuccessful = rollout("wrong-answer", 0.0)

    examples, report = build_sft_examples([valid, parse_failed, unsuccessful])

    assert [row["step"] for row in examples] == [0, 1]
    assert all(row["task_id"] == "kept" for row in examples)
    assert examples[0]["policy_version"] == "teacher-v2"
    assert examples[0]["messages"][-1] == {
        "role": "assistant",
        "content": valid["decisions"][0]["raw_output"],
    }
    provenance = examples[0]["verifier_provenance"]
    assert provenance["metric"] == "JointSuccess"
    assert provenance["score"] == 1.0
    assert provenance["source_provenance"] == {
        "name": "hotpotqa-verifier",
        "version": "v1",
    }
    assert "MUST_NOT_LEAK" not in json.dumps(examples)
    assert report["episodes_seen"] == 3
    assert report["episodes_kept"] == 1
    assert report["examples_written"] == 2
    assert report["filter_counts"] == {
        "below_success_threshold": 1,
        "parse_error": 1,
    }


def test_builder_supports_configurable_metric_threshold_and_prompt_response():
    rows = [rollout("high", 0.75), rollout("low", 0.49)]
    examples, report = build_sft_examples(
        rows,
        config=SFTBuildConfig(
            success_metric="JointEM",
            min_success=0.5,
            output_format="prompt_response",
        ),
    )

    assert len(examples) == 2
    assert all("messages" not in example for example in examples)
    assert examples[0]["prompt"] == [
        {"role": "system", "content": "Choose one action."},
        {"role": "user", "content": "Environment observation for step 0"},
    ]
    assert examples[0]["response"] == rows[0]["decisions"][0]["raw_output"]
    assert report["config"] == {
        "success_metric": "JointEM",
        "min_success": 0.5,
        "output_format": "prompt_response",
        "require_clean_transitions": True,
    }
    assert report["filter_counts"] == {"below_success_threshold": 1}


def test_cli_writes_jsonl_and_statistics_report(tmp_path):
    input_path = tmp_path / "rollouts.jsonl"
    output_path = tmp_path / "sft.jsonl"
    report_path = tmp_path / "report.json"
    input_path.write_text(
        "\n".join(json.dumps(row) for row in [rollout("ok", 1.0), rollout("bad", 0.0)])
        + "\n",
        encoding="utf-8",
    )
    project_root = Path(__file__).resolve().parents[1]

    completed = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "build_agent_sft_data.py"),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--report",
            str(report_path),
        ],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )

    output_rows = [json.loads(line) for line in output_path.read_text().splitlines()]
    report = json.loads(report_path.read_text())
    assert len(output_rows) == 2
    assert report["examples_written"] == 2
    assert report["episodes_filtered"] == 1
    assert report["artifacts"]["input"]["sha256"] == hashlib.sha256(
        input_path.read_bytes()
    ).hexdigest()
    assert report["artifacts"]["output"]["sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()
    assert len(report["artifacts"]["report"]["sha256"]) == 64
    assert json.loads(completed.stdout)["episodes_kept"] == 1


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda row: row["transitions"][0]["reward_breakdown"].update(
                invalid_action=-0.1
            ),
            "invalid_action_transition",
        ),
        (
            lambda row: row["transitions"][0]["reward_breakdown"].update(
                duplicate_call=-0.1
            ),
            "duplicate_call_transition",
        ),
        (
            lambda row: row["transitions"][0]["observation_after"]["history"][0].update(
                ok=False,
                error="backend failed",
            ),
            "unsuccessful_tool_event",
        ),
    ],
)
def test_builder_rejects_successful_episode_with_unsafe_transition(mutation, reason):
    row = rollout("unsafe", 1.0)
    mutation(row)

    examples, report = build_sft_examples([row])

    assert examples == []
    assert report["filter_counts"] == {reason: 1}


def test_final_answer_transition_does_not_require_tool_event():
    row = rollout("valid-final", 1.0)
    assert len(row["transitions"][1]["observation_after"]["history"]) == 1

    examples, report = build_sft_examples([row])

    assert len(examples) == 2
    assert report["episodes_kept"] == 1


def test_cli_refuses_overwrite_unless_explicitly_enabled(tmp_path):
    input_path = tmp_path / "rollouts.jsonl"
    output_path = tmp_path / "sft.jsonl"
    report_path = tmp_path / "report.json"
    input_path.write_text(json.dumps(rollout("ok", 1.0)) + "\n", encoding="utf-8")
    output_path.write_text("do not replace\n", encoding="utf-8")
    project_root = Path(__file__).resolve().parents[1]
    command = [
        sys.executable,
        str(project_root / "scripts" / "build_agent_sft_data.py"),
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--report",
        str(report_path),
    ]

    refused = subprocess.run(command, cwd=project_root, capture_output=True, text=True)
    assert refused.returncode != 0
    assert output_path.read_text() == "do not replace\n"
    assert not report_path.exists()

    subprocess.run(
        [*command, "--overwrite"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert output_path.read_text() != "do not replace\n"
    assert report_path.exists()
