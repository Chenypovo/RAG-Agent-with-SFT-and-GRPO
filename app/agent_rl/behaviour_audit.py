"""Post-episode behaviour measurements; never feed these labels to a policy."""
from __future__ import annotations

from collections import Counter


def audit_behaviour(rollouts, *, gold_by_task=None):
    rows = [r.to_dict() if hasattr(r, "to_dict") else r for r in rollouts]
    retrieval_counts, decision_counts, actions = Counter(), Counter(), Counter()
    totals = Counter()
    recovery = {name: Counter() for name in ("empty", "failed", "duplicate")}
    two_call = Counter()
    for row in rows:
        decisions = row["decisions"]
        transitions = row["transitions"]
        retrievals = 0
        two_call_checked = False
        gold = set((gold_by_task or {}).get(row["task_id"], ()))
        for decision, transition in zip(decisions, transitions):
            action = decision.get("action") or {}
            if decision.get("parse_error"):
                actions["invalid_json_action"] += 1
                totals["json_invalid"] += 1
            elif action.get("tool") not in (None, "retrieve_docs"):
                actions["unsupported_tool:" + str(action["tool"])] += 1
                totals["unsupported_tool"] += 1
            else:
                actions[action.get("tool") or "final_answer"] += 1
            before = transition.get("observation_before", {})
            if retrievals == 2 and not two_call_checked:
                two_call_checked = True
                if gold and not gold.issubset(before.get("evidence_ids", [])):
                    two_call["opportunities_with_incomplete_evidence"] += 1
                    if action.get("final_answer"):
                        two_call["stop_with_incomplete_evidence"] += 1
                    elif action.get("tool") == "retrieve_docs":
                        two_call["continue_with_incomplete_evidence"] += 1
                        two_call["continued_then_complete_evidence"] += row["verification"].get("CompleteSentenceEvidence", 0) >= 1
                        two_call["continued_then_joint_success"] += row["verification"].get("JointSuccess", 0) >= 1
                    else:
                        two_call["invalid_next_action"] += 1
            retrievals += int(action.get("tool") == "retrieve_docs")
            breakdown = transition.get("reward_breakdown", {})
            totals["duplicate"] += int(float(breakdown.get("duplicate_call", 0)) < 0)
            totals["premature_stop"] += int(float(breakdown.get("premature_final_answer", 0)) < 0)
        retrieval_counts[str(retrievals)] += 1
        decision_counts[str(len(decisions))] += 1
        totals["decisions"] += len(decisions)
        totals["retrieval_calls"] += retrievals
        totals["budget"] += row.get("stop_reason") == "budget"
        totals["explicit_stop"] += row.get("stop_reason") == "final_answer"
        # The final observation contains each tool event once; avoid counting
        # repeated copies of history across transitions as repeated failures.
        events = transitions[-1].get("observation_after", {}).get("history", []) if transitions else []
        seen = set()
        for index, event in enumerate(events):
            if event.get("tool") != "retrieve_docs":
                continue
            evidence = set(event.get("data", {}).get("evidence_ids", []))
            error = str(event.get("error") or "").lower()
            kind = "duplicate" if "duplicate call" in error else (
                "failed" if not event.get("ok") else ("empty" if not evidence else None)
            )
            if kind:
                counts = recovery[kind]
                counts["exposures"] += 1
                counts["has_later_action"] += any(
                    int(t.get("step_index", -1)) > int(event.get("step_index", -1)) for t in transitions
                )
                later = [e for e in events[index + 1:] if e.get("tool") == "retrieve_docs" and e.get("ok")]
                obtained_new_evidence = any(set(e.get("data", {}).get("evidence_ids", [])) - seen for e in later)
                counts["recovered_new_evidence"] += obtained_new_evidence
                counts["recovered_joint_success"] += obtained_new_evidence and row["verification"].get("JointSuccess", 0) >= 1
            seen.update(evidence)
    n = len(rows)
    buckets = {k: 0 for k in ("0", "1", "2", "3", "4+")}
    for calls, count in retrieval_counts.items():
        buckets[calls if int(calls) < 4 else "4+"] += count
    recovery_report = {}
    for kind, counts in recovery.items():
        exposures = counts["exposures"]
        recovery_report[kind] = {
            **{k: counts[k] for k in ("exposures", "has_later_action", "recovered_new_evidence", "recovered_joint_success")},
            "new_evidence_rate": counts["recovered_new_evidence"] / exposures if exposures else None,
            "joint_success_rate": counts["recovered_joint_success"] / exposures if exposures else None,
        }
    recovery_report["contradictory"] = {
        "exposures": None, "new_evidence_rate": None, "joint_success_rate": None,
        "status": "unresolved: no adjudicated contradiction labels in this protocol",
    }
    return {
        "n_episodes": n, "n_decisions": totals["decisions"],
        "action_counts": dict(actions),
        "retrieval_calls_per_episode": dict(sorted(retrieval_counts.items())),
        "decisions_per_episode": dict(sorted(decision_counts.items())),
        "retrieval_call_buckets": buckets,
        "retrieval_call_proportions": {k: v / n if n else None for k, v in buckets.items()},
        "mean_retrieval_calls": totals["retrieval_calls"] / n if n else None,
        "json_invalid_rate": totals["json_invalid"] / totals["decisions"] if totals["decisions"] else None,
        "unsupported_tool_action_count": totals["unsupported_tool"],
        "unsupported_tool_action_rate": totals["unsupported_tool"] / totals["decisions"] if totals["decisions"] else None,
        "premature_stop_rate": totals["premature_stop"] / n if n else None,
        "duplicate_call_rate_per_decision": totals["duplicate"] / totals["decisions"] if totals["decisions"] else None,
        "duplicate_call_rate_per_retrieval_attempt": totals["duplicate"] / totals["retrieval_calls"] if totals["retrieval_calls"] else None,
        "budget_exhaustion_rate": totals["budget"] / n if n else None,
        "explicit_stop_rate": totals["explicit_stop"] / n if n else None,
        "recovery": recovery_report,
        "decision_after_two_calls_with_incomplete_evidence": dict(two_call) if gold_by_task is not None else None,
        "definitions": {
            "json_invalid_rate": "strict JSON action parsing/schema errors divided by all model decisions; retrieval execution failures are counted separately",
            "retrieval_calls": "retrieve_docs action attempts, including duplicates; historical MeanToolCalls counts all tool-name attempts and also includes unsupported tool names",
            "premature_stop": "explicit final_answer with incomplete gold sentence evidence, divided by all episodes",
            "recovery": "a subsequent successful retrieval yields a previously unseen sentence ID; also report endpoint JointSuccess",
            "zero_exposures": "null rate; absence of an opportunity is not evidence of recovery",
        },
    }
