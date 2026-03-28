"""Generate autoresearch progress plot for the DQN optimization phase (iterations 0-22)."""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# --- Data: DQN phase only (iterations 0-22) ---
iterations = list(range(23))
metrics = [
    207.6,  # 0: baseline
    158.4,  # 1: lr 5e-4
    468.6,  # 2: tau 0.005
    474.2,  # 3: lr 3e-4 + anneal 100k
    412.2,  # 4: wider net 256 + 20 grad
    500.0,  # 5: 2x buffer + 2x warmup  *** KEEP ***
    193.8,  # 6: lr 3e-4 on best
    27.2,   # 7: fast anneal + 200k frames
    161.8,  # 8: tau 0.005 + lr 2e-4
    241.8,  # 9: 20 grad + batch 128
    500.0,  # 10: eps_end 0.1
    183.2,  # 11: anneal 100k + lr 5e-4
    346.4,  # 12: clip 1.0 + 20 grad
    500.0,  # 13: 300k + tau 0.005 (kept then reverted)
    245.2,  # 14: confirmation failed
    270.0,  # 15: anneal 100k
    373.6,  # 16: gamma 0.999
    473.8,  # 17: fpb 500 + 5 grad
    276.4,  # 18: eval every 5/10ep
    237.4,  # 19: clip 10 + lr 1e-3
    166.2,  # 20: batch 64 + 30 grad
    39.6,   # 21: warmup 5k + anneal 20k + eps 0.01
    500.0,  # 22: 3x64 network  *** KEEP ***
]

status = [
    "keep",     # 0
    "discard",  # 1
    "discard",  # 2
    "discard",  # 3
    "discard",  # 4
    "keep",     # 5
    "discard",  # 6
    "discard",  # 7
    "discard",  # 8
    "discard",  # 9
    "discard",  # 10
    "discard",  # 11
    "discard",  # 12
    "reverted", # 13
    "discard",  # 14
    "discard",  # 15
    "discard",  # 16
    "discard",  # 17
    "discard",  # 18
    "discard",  # 19
    "discard",  # 20
    "discard",  # 21
    "keep",     # 22
]

# Short descriptions for kept iterations
annotations = {
    0:  "baseline DQN\n(2×128, buf=50k)",
    5:  "2× buffer (100k)\n+ 2× warmup (10k)",
    22: "3×64 deeper\nnarrower network",
}

# Descriptions for notable discards
discard_annotations = {
    1:  "lr↑ 5e-4",
    7:  "200k frames\n+ fast anneal",
    13: "300k + τ=0.005\n(reverted: failed\nconfirmation)",
    21: "fast anneal\n+ low ε",
}

# --- Compute running best ---
running_best = []
best = 0
for i, (m, s) in enumerate(zip(metrics, status)):
    if s == "keep":
        best = max(best, m)
    running_best.append(best)

# --- Plot ---
with plt.xkcd(scale=1, length=100, randomness=2):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Discarded experiments: faded dots
    disc_x = [i for i, s in enumerate(status) if s == "discard"]
    disc_y = [metrics[i] for i in disc_x]
    ax.scatter(disc_x, disc_y, color="#cccccc", s=40, zorder=2, label="Discarded")

    # Reverted experiment: special marker
    rev_x = [i for i, s in enumerate(status) if s == "reverted"]
    rev_y = [metrics[i] for i in rev_x]
    ax.scatter(rev_x, rev_y, color="#ffaa44", s=60, marker="x", zorder=3,
               linewidths=2, label="Kept then reverted")

    # Running best staircase
    stair_x, stair_y = [0], [running_best[0]]
    for i in range(1, len(running_best)):
        if running_best[i] != running_best[i - 1]:
            stair_x.extend([i, i])
            stair_y.extend([running_best[i - 1], running_best[i]])
        else:
            stair_x.append(i)
            stair_y.append(running_best[i])
    ax.plot(stair_x, stair_y, color="#2ecc71", linewidth=2, alpha=0.7,
            zorder=1, label="Running best")

    # Kept experiments: bold green circles
    keep_x = [i for i, s in enumerate(status) if s == "keep"]
    keep_y = [metrics[i] for i in keep_x]
    ax.scatter(keep_x, keep_y, color="#2ecc71", s=100, zorder=4,
               edgecolors="white", linewidths=1.5, label="Kept")

    # Annotations for kept experiments
    for i, txt in annotations.items():
        y_offset = 25 if metrics[i] < 450 else -45
        x_offset = 15
        ax.annotate(
            txt, (i, metrics[i]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=13, fontweight="bold", color="#1a7a42",
            rotation=25,
            ha="left",
            arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.5),
            path_effects=[pe.withStroke(linewidth=3, foreground="white")],
        )

    # Annotations for notable discards
    for i, txt in discard_annotations.items():
        y_offset = -35 if metrics[i] > 200 else 25
        ax.annotate(
            txt, (i, metrics[i]),
            xytext=(10, y_offset),
            textcoords="offset points",
            fontsize=11, color="#999999",
            rotation=20,
            ha="left",
            arrowprops=dict(arrowstyle="->", color="#cccccc", lw=1),
            path_effects=[pe.withStroke(linewidth=2, foreground="white")],
        )

    # Formatting
    ax.set_title("Autoresearch Progress: 23 Experiments, 3 Kept Improvements (DQN Phase)",
                 fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Experiment #", fontsize=16)
    ax.set_ylabel("Average Reward (higher is better)", fontsize=16)
    ax.set_xlim(-1, 23)
    ax.set_ylim(-20, 560)
    ax.set_xticks(range(0, 23, 2))
    ax.axhline(y=500, color="#2ecc71", linestyle="--", alpha=0.3, linewidth=1)
    ax.text(22.5, 505, "perfect score (500)", fontsize=12, color="#2ecc71",
            ha="right", alpha=0.6)
    ax.axhline(y=400, color="#e67e22", linestyle="--", alpha=0.2, linewidth=1)
    ax.text(22.5, 405, "target (≥400)", fontsize=12, color="#e67e22",
            ha="right", alpha=0.5)
    ax.legend(loc="lower right", fontsize=13, framealpha=0.9)

    fig.tight_layout()
    fig.savefig("progress.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("Saved progress.png")
