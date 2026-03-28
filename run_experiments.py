"""Parallel experiment runner for autoresearch DQN.

Generates N variant DQN scripts with different hyperparameter changes,
runs them all in parallel, and reports which one achieved the best METRIC.
"""
import subprocess
import tempfile
import os
import sys
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_base_script():
    return Path("dqn.py").read_text()


def make_variant(base: str, replacements: dict[str, tuple[str, str]], description: str) -> str:
    """Apply text replacements to base script. Returns modified script."""
    script = base
    for old, new in replacements.values():
        script = script.replace(old, new)
    return script


def run_variant(args: tuple[str, str, str]) -> tuple[str, float, str]:
    """Run a variant script and extract METRIC. Returns (name, metric, output)."""
    name, script_content, tmpdir = args
    script_path = os.path.join(tmpdir, f"dqn_{name}.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        output = result.stdout + result.stderr

        # Extract METRIC line
        match = re.search(r"METRIC:\s+([\d.]+)", output)
        if match:
            metric = float(match.group(1))
        else:
            metric = 0.0

        # Extract last few reward lines for context
        reward_lines = [l for l in output.split("\n") if "reward=" in l]
        summary = "\n".join(reward_lines[-5:]) if reward_lines else "(no reward output)"

        return name, metric, f"{summary}\nMETRIC: {metric}"
    except subprocess.TimeoutExpired:
        return name, 0.0, "TIMEOUT (>5min)"
    except Exception as e:
        return name, 0.0, f"ERROR: {e}"


# Define experiments: each is a dict of {label: (old_text, new_text)}
# Each experiment changes ONE thing from baseline
EXPERIMENTS = {
    "tau_0.005": {
        "tau": ("tau=0.001", "tau=0.005"),
    },
    "lr_3e-4_anneal_100k": {
        "lr": ("lr=1e-4", "lr=3e-4"),
        "anneal": ("annealing_num_steps=50_000", "annealing_num_steps=100_000"),
    },
    "bigger_buffer_warmup": {
        "buffer": ("max_size=50_000", "max_size=100_000"),
        "warmup": ("if len(buffer) < 5000:", "if len(buffer) < 10000:"),
    },
    "wider_net_more_grad_steps": {
        "net": (
            "nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2))",
            "nn.Sequential(nn.Linear(4, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 2))",
        ),
        "grad": ("for _ in range(10):", "for _ in range(20):"),
    },
}

DESCRIPTIONS = {
    "tau_0.005": "increase soft update tau from 0.001 to 0.005",
    "lr_3e-4_anneal_100k": "increase lr to 3e-4 + slower epsilon annealing (100k steps)",
    "bigger_buffer_warmup": "double replay buffer size (100k) + double warmup (10k)",
    "wider_net_more_grad_steps": "wider network (256 hidden) + 20 grad steps per batch",
}


if __name__ == "__main__":
    base = read_base_script()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Prepare all variants
        tasks = []
        for name, replacements in EXPERIMENTS.items():
            variant = make_variant(base, replacements, DESCRIPTIONS[name])
            tasks.append((name, variant, tmpdir))

        print(f"Running {len(tasks)} experiments in parallel...")
        for name in EXPERIMENTS:
            print(f"  - {name}: {DESCRIPTIONS[name]}")
        print()

        # Run all in parallel
        results = {}
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(run_variant, task): task[0] for task in tasks}
            for future in as_completed(futures):
                name, metric, output = future.result()
                results[name] = (metric, output)
                print(f"=== {name} completed: METRIC={metric:.1f} ===")
                print(f"  {DESCRIPTIONS[name]}")
                print()

        # Rank results
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY (sorted by metric, higher is better)")
        print("=" * 60)
        ranked = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
        for rank, (name, (metric, output)) in enumerate(ranked, 1):
            print(f"  {rank}. {name}: METRIC={metric:.1f} — {DESCRIPTIONS[name]}")

        winner_name = ranked[0][0]
        winner_metric = ranked[0][1][0]
        print(f"\nWINNER: {winner_name} (METRIC={winner_metric:.1f})")
        print(f"Description: {DESCRIPTIONS[winner_name]}")
