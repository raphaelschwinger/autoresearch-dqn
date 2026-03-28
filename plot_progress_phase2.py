"""Generate autoresearch progress plot for the REINFORCE optimization phase (iterations 23-39)."""
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np

# --- Data: Phase 2 (iterations 23-39) ---
# Metric: frames to convergence (≥490 reward). None = did not converge.
experiments = [
    # (iter, frames, reward, status, label)
    (23, 9868,   497.8, "candidate", "REINFORCE\n9.8k frames"),
    (24, None,   8.8,   "discard",   "PPO TorchRL\n(failed)"),
    (25, 200000, 273.0, "discard",   None),
    (26, 300000, 236.0, "discard",   None),
    (27, 20807,  500.0, "keep",      "REINFORCE\nconfirmed"),  # avg of 27824 and 13790
    (28, None,   9.6,   "discard",   "actor-critic\n(9.6 on confirm)"),
    (29, None,   500.0, "discard",   None),  # no frame data
    (30, None,   500.0, "discard",   None),  # no frame data
    (31, None,   22.4,  "discard",   "lr 2e-2"),
    (32, 10122,  495.8, "keep",      "eval every 5 ep\n10.1k frames"),
    (33, 12570,  500.0, "discard",   None),
    (34, 26000,  427.6, "discard",   "grad clip\n26k frames"),
    (35, 34840,  432.0, "discard",   "lr 5e-3\n34.8k frames"),
    (36, 19190,  500.0, "discard",   None),
    (37, None,   342.6, "discard",   "greedy eval\n(never converges)"),
    (38, None,   114.8, "discard",   "entropy bonus\n(diverges)"),
    (39, 37389,  484.0, "verify",    "final verify\n37.4k frames"),
]

DNF_Y = 350000  # y-position for "did not converge" experiments

# --- Compute running best (lower is better) ---
running_best = []
best = float("inf")
for it, frames, reward, status, _ in experiments:
    if status == "keep" and frames is not None:
        best = min(best, frames)
    running_best.append(best if best < float("inf") else None)

# --- Plot ---
with plt.xkcd(scale=1, length=100, randomness=2):
    fig, ax = plt.subplots(figsize=(14, 7))

    # Separate by category
    for it, frames, reward, status, label in experiments:
        y = frames if frames is not None else DNF_Y
        if status == "discard":
            ax.scatter(it, y, color="#cccccc", s=40, zorder=2)
        elif status == "candidate":
            ax.scatter(it, y, color="#f39c12", s=80, zorder=3,
                       edgecolors="white", linewidths=1.5)
        elif status == "keep":
            ax.scatter(it, y, color="#2ecc71", s=100, zorder=4,
                       edgecolors="white", linewidths=1.5)
        elif status == "verify":
            ax.scatter(it, y, color="#3498db", s=60, zorder=3,
                       marker="D", edgecolors="white", linewidths=1)

    # Running best staircase (only after first keep)
    keep_iters = [(it, frames) for it, frames, _, status, _ in experiments
                  if status == "keep" and frames is not None]
    if keep_iters:
        stair_x = []
        stair_y = []
        cur_best = float("inf")
        for it, frames, _, status, _ in experiments:
            if status == "keep" and frames is not None:
                if cur_best == float("inf"):
                    stair_x.append(it)
                    stair_y.append(frames)
                    cur_best = frames
                elif frames < cur_best:
                    stair_x.extend([it, it])
                    stair_y.extend([cur_best, frames])
                    cur_best = frames
                else:
                    stair_x.append(it)
                    stair_y.append(cur_best)
            elif cur_best < float("inf"):
                stair_x.append(it)
                stair_y.append(cur_best)
        ax.plot(stair_x, stair_y, color="#2ecc71", linewidth=2, alpha=0.7, zorder=1)

    # Annotations
    for it, frames, reward, status, label in experiments:
        if label is None:
            continue
        y = frames if frames is not None else DNF_Y
        if status == "keep":
            y_off = -40
            ax.annotate(
                label, (it, y),
                xytext=(15, y_off), textcoords="offset points",
                fontsize=13, fontweight="bold", color="#1a7a42",
                rotation=25, ha="left",
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=1.5),
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
        elif status == "candidate":
            ax.annotate(
                label, (it, y),
                xytext=(15, 25), textcoords="offset points",
                fontsize=13, fontweight="bold", color="#e67e22",
                rotation=25, ha="left",
                arrowprops=dict(arrowstyle="->", color="#f39c12", lw=1.5),
                path_effects=[pe.withStroke(linewidth=3, foreground="white")],
            )
        elif status == "verify":
            ax.annotate(
                label, (it, y),
                xytext=(10, 25), textcoords="offset points",
                fontsize=11, color="#3498db",
                rotation=20, ha="left",
                arrowprops=dict(arrowstyle="->", color="#3498db", lw=1),
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )
        else:  # discard with label
            y_off = 25 if y > 100000 else -35
            ax.annotate(
                label, (it, y),
                xytext=(10, y_off), textcoords="offset points",
                fontsize=11, color="#999999",
                rotation=20, ha="left",
                arrowprops=dict(arrowstyle="->", color="#cccccc", lw=1),
                path_effects=[pe.withStroke(linewidth=2, foreground="white")],
            )

    # "Did not converge" zone
    ax.axhline(y=DNF_Y, color="#e74c3c", linestyle=":", alpha=0.3, linewidth=1)
    ax.text(39.5, DNF_Y * 1.05, "did not converge", fontsize=12, color="#e74c3c",
            ha="right", alpha=0.6)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#f39c12',
               markersize=9, markeredgecolor='white', label='Candidate'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ecc71',
               markersize=10, markeredgecolor='white', label='Kept'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc',
               markersize=7, label='Discarded'),
        Line2D([0], [0], marker='D', color='w', markerfacecolor='#3498db',
               markersize=7, markeredgecolor='white', label='Final verify'),
        Line2D([0], [0], color='#2ecc71', linewidth=2, alpha=0.7, label='Running best'),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=13, framealpha=0.9)

    # Formatting
    ax.set_title("Autoresearch Progress: REINFORCE Optimization Phase (Iterations 23–39)",
                 fontsize=18, fontweight="bold", pad=15)
    ax.set_xlabel("Experiment #", fontsize=16)
    ax.set_ylabel("Frames to Convergence (lower is better)", fontsize=16)
    ax.set_xlim(22, 40)
    ax.set_yscale("log")
    ax.set_ylim(5000, 600000)
    ax.set_xticks(range(23, 40))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(
        lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))

    fig.tight_layout()
    fig.savefig("progress_phase2.png", dpi=150, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    print("Saved progress_phase2.png")
