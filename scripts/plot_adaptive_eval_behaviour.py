"""Export descriptive matched-benchmark behaviour figures on AutoDL."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    results = Path(args.run_root) / "results"
    output = results / "figures"
    targets = [output / ("matched_eval_stopping." + suffix) for suffix in ("png", "svg", "csv")]
    if any(path.exists() for path in targets):
        raise FileExistsError("matched-evaluation figure already exists")
    stopping = json.loads((results / "stopping_decision_audit.json").read_text())
    names = ("prompt", "e1_sft", "e2_grpo")
    labels = ("Prompt-only", "E1 SFT", "E2 SFT + GRPO")
    colours = ("#717985", "#208278", "#356d9d")
    reports = {name: json.loads((results / "evaluation" / (name + ".json")).read_text()) for name in names}
    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.4), gridspec_kw={"height_ratios": [1, 1.15]})
    rows = []
    x = np.arange(6)
    for index, (name, label, colour) in enumerate(zip(names, labels, colours)):
        behaviour = reports[name]["behaviour"]
        assert behaviour["n_episodes"] == 1000
        counts = [behaviour["retrieval_calls_per_episode"].get(str(call), 0) for call in range(6)]
        percentages = [100 * count / 1000 for count in counts]
        positions = x + (index - 1) * 0.25
        axes[0].bar(positions, percentages, width=0.23, label=label, color=colour)
        for call, position, count, percentage in zip(range(6), positions, counts, percentages):
            if count:
                axes[0].text(position, percentage + 1.5, f"{percentage:.1f}", ha="center", fontsize=8)
            rows.append({"panel": "retrieval_distribution", "policy": name, "retrieval_attempts": call,
                "evidence_state": "all", "numerator": count, "denominator": 1000, "percentage": percentage})
    axes[0].set_xticks(x, [str(call) for call in range(6)])
    axes[0].set_xlabel("retrieve_docs attempts per task")
    axes[0].set_title("A. Retrieval-call distribution (same 1,000 benchmark/dev tasks)", loc="left", fontsize=11)
    axes[0].set_ylabel("Tasks (%)")
    axes[0].set_ylim(0, 115)
    axes[0].set_yticks([0, 25, 50, 75, 100])
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    for index, (state, label, colour) in enumerate((
        ("incomplete", "Gold sentence evidence incomplete", "#b66a4a"),
        ("complete", "Gold sentence evidence complete", "#356d9d"),
    )):
        for policy_index, name in enumerate(names):
            match = [row for row in stopping["policies"][name]["conditional_decisions"]
                if row["prior_retrieval_attempts"] == 2 and row["evidence_state"] == state]
            count = match[0] if match else {"stop": 0, "opportunities": 0}
            n = count["opportunities"]
            percentage = 100 * count["stop"] / n if n else 0
            position = policy_index + (index - 0.5) * 0.36
            axes[1].bar(position, percentage, width=0.33, color=colour, label=label if policy_index == 0 else None)
            annotation = f"{percentage:.1f}%\n(n={n})" if n else "No exposure"
            axes[1].text(position, percentage + 2, annotation, ha="center", fontsize=9)
            rows.append({"panel": "stop_after_two", "policy": name, "retrieval_attempts": 2,
                "evidence_state": state, "numerator": count["stop"], "denominator": n,
                "percentage": percentage if n else None})
    axes[1].set_xticks(np.arange(3), labels)
    axes[1].set_ylabel("Next decision is stop (%)")
    axes[1].set_ylim(0, 125)
    axes[1].set_yticks([0, 25, 50, 75, 100])
    axes[1].set_title("B. Stopping immediately after two retrieval attempts", loc="left", fontsize=11)
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper left")
    for ax in axes:
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle("Observed stopping behaviour after adaptive-teacher training", fontsize=14, x=0.085, ha="left", y=0.99)
    fig.text(0.085, 0.02,
        "One seed; reused benchmark/dev. Gold labels are applied only after episodes.\n"
        "Panel B conditions on observed evidence: task groups differ across policies; this is not a causal test.",
        fontsize=9, color="#424954", va="bottom")
    fig.tight_layout(rect=(0, 0.085, 1, 0.965), h_pad=2)
    output.mkdir(exist_ok=True)
    fig.savefig(targets[0], dpi=200, facecolor="white")
    fig.savefig(targets[1], facecolor="white")
    plt.close(fig)
    with targets[2].open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"outputs": list(map(str, targets))}))


if __name__ == "__main__":
    main()
