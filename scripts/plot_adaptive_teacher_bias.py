"""Export descriptive training-trajectory distributions on the experiment host."""
from __future__ import annotations

import argparse
from collections import Counter
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

    root = Path(args.run_root)
    output = root / "results/figures"
    output.mkdir(exist_ok=True)
    targets = [output / ("teacher_training_bias." + extension) for extension in ("png", "svg", "csv")]
    if any(path.exists() for path in targets):
        raise FileExistsError("training-bias figure already exists")
    old = Counter()
    with (root / "audit/e0/teacher_train1617.jsonl").open() as handle:
        for line in handle:
            row = json.loads(line)
            old[sum((d.get("action") or {}).get("tool") == "retrieve_docs" for d in row["decisions"])] += 1
    assert old == {2: 1617}
    audit = json.loads((root / "results/e1_teacher/integrity_audit.json").read_text())
    cohorts = [("E0 scripted teacher", dict(old), 1617, "#717985")]
    for label, key, colour in [
        ("Frozen 7B: all candidates", "all_candidate_behaviour_recomputed", "#657fa7"),
        ("Frozen 7B: selected candidates", "selected_behaviour_recomputed", "#356d9d"),
        ("E1 SFT: retained trajectories", "sft_kept_behaviour", "#208278"),
    ]:
        data = audit[key]
        cohorts.append((label, {int(k): v for k, v in data["retrieval_calls_per_episode"].items()}, data["n_episodes"], colour))
    rows = []
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 8.4), sharex=True, sharey=True)
    for ax, (label, counts, n, colour) in zip(axes, cohorts):
        assert sum(counts.values()) == n
        values = [100 * counts.get(call, 0) / n for call in range(1, 6)]
        ax.bar(range(1, 6), values, color=colour, width=0.66)
        ax.set_title(f"{label}  |  n = {n:,}", loc="left", fontsize=11, pad=6)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 50, 100], ["0%", "50%", "100%"])
        ax.grid(axis="y", alpha=0.16)
        ax.set_axisbelow(True)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for call, value in zip(range(1, 6), values):
            ax.text(call, value + 2, f"{value:.1f}%", ha="center", va="bottom", fontsize=9)
            rows.append({"cohort": label, "n_trajectories": n, "retrieval_attempts": call,
                "count": counts.get(call, 0), "percentage": value})
    axes[-1].set_xticks(range(1, 6))
    axes[-1].set_xlabel("retrieve_docs attempts per trajectory", fontsize=11)
    fig.suptitle("Retrieval-call distributions in training trajectories", fontsize=14, x=0.10, ha="left", y=0.99)
    fig.text(0.10, 0.024,
        "Descriptive training-set counts; no benchmark-performance claim.\n"
        "The two-call peak is replaced by a concentration near the action budget; no one-call trajectory appears.",
        fontsize=9, color="#424954", va="bottom")
    fig.tight_layout(rect=(0, 0.075, 1, 0.96), h_pad=1.35)
    fig.savefig(targets[0], dpi=200, facecolor="white")
    fig.savefig(targets[1], facecolor="white")
    plt.close(fig)
    with targets[2].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"outputs": list(map(str, targets)), "cohorts": len(cohorts)}))


if __name__ == "__main__":
    main()
