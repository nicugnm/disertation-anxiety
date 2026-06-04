"""Figure for Experiment 9 (DANN transfer). Reads experiments/exp_dann_transfer.json,
writes docs/figures/dann_transfer.png — anxiety AUROC & F1 (in_dist vs cross_heldout)
and the neutral false-positive rate, per model."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = Path("experiments/exp_dann_transfer.json")
OUT = Path("docs/figures/dann_transfer.png")
ORDER = ["multitask", "dann_subreddit", "dann_group"]
LABELS = {"multitask": "multitask\n(no DANN)", "dann_subreddit": "DANN\n(subreddit)", "dann_group": "DANN\n(group)"}


def main() -> None:
    rows = json.loads(SRC.read_text())
    anx = {(r["model"], r["split"]): r for r in rows if r["target"] == "anxiety"}
    x = np.arange(len(ORDER))
    w = 0.38
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))

    for ax, metric, title in [(axes[0], "auroc", "Anxiety AUROC"), (axes[1], "f1", "Anxiety F1")]:
        ind = [anx[(m, "in_dist")][metric] for m in ORDER]
        crs = [anx[(m, "cross_heldout")][metric] for m in ORDER]
        ax.bar(x - w / 2, ind, w, label="in-distribution", color="#4C72B0")
        ax.bar(x + w / 2, crs, w, label="cross (held-out anxiety subs)", color="#DD8452")
        for i, (a, b) in enumerate(zip(ind, crs)):
            ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
            ax.text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
        ax.set_title(title); ax.set_ylim(0, 1.08); ax.set_xticks(x)
        ax.set_xticklabels([LABELS[m] for m in ORDER]); ax.legend(fontsize=8)

    fp = [anx[(m, "neutral")]["pred_pos_rate"] for m in ORDER]
    bars = axes[2].bar(x, fp, color=["#4C72B0", "#55A868", "#C44E52"])
    for i, v in enumerate(fp):
        axes[2].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=9)
    axes[2].set_title("False-positive rate on neutral subs\n(lower is better)")
    axes[2].set_ylim(0, max(fp) * 1.25); axes[2].set_xticks(x)
    axes[2].set_xticklabels([LABELS[m] for m in ORDER])

    fig.suptitle("Experiment 9 — Domain-adversarial (DANN) vs plain multi-task transfer", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=130)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
