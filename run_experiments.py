"""Parallel experiment runner for autoresearch.

Runs full standalone scripts in parallel, reports METRIC and FRAMES for each.
"""
import subprocess
import tempfile
import os
import sys
import re
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed


def run_script(args: tuple[str, str, str]) -> tuple[str, float, str, float]:
    """Run a script and extract METRIC. Returns (name, metric, output, elapsed_seconds)."""
    name, script_content, tmpdir = args
    script_path = os.path.join(tmpdir, f"{name}.py")
    with open(script_path, "w") as f:
        f.write(script_content)

    t0 = time.time()
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "PYTHONUNBUFFERED": "1"}
        )
        elapsed = time.time() - t0
        output = result.stdout + result.stderr

        match = re.search(r"METRIC:\s+([\d.]+)", output)
        metric = float(match.group(1)) if match else 0.0

        reward_lines = [l for l in output.split("\n") if "reward=" in l or "METRIC" in l or "FRAMES" in l]
        summary = "\n".join(reward_lines[-8:]) if reward_lines else "(no output)"

        return name, metric, f"{summary}\n[{elapsed:.1f}s]", elapsed
    except subprocess.TimeoutExpired:
        return name, 0.0, "TIMEOUT (>5min)", 300.0
    except Exception as e:
        return name, 0.0, f"ERROR: {e}", 0.0


FULL_SCRIPTS = {}
DESCRIPTIONS = {}
BASE = Path("dqn.py").read_text()

# --- Variant 1: Higher LR (2e-2) ---
FULL_SCRIPTS["lr_2e-2"] = BASE.replace("lr=1e-2", "lr=2e-2")
DESCRIPTIONS["lr_2e-2"] = "REINFORCE with lr=2e-2 — faster gradient updates"

# --- Variant 2: Smaller network (64 hidden) ---
FULL_SCRIPTS["small_net"] = BASE.replace(
    "nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))",
    "nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 2))"
)
DESCRIPTIONS["small_net"] = "REINFORCE with tiny network (32 hidden) — faster per-step, less capacity"

# --- Variant 3: Actor-critic (learned baseline) ---
FULL_SCRIPTS["actor_critic"] = '''
import torch
import torch.nn as nn
from torch.distributions import Categorical
import gymnasium as gym

torch.manual_seed(0)

policy = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))
value_fn = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 1))
optimizer = torch.optim.Adam(list(policy.parameters()) + list(value_fn.parameters()), lr=1e-2)

total_frames = 0
for episode in range(500):
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=episode)
    log_probs, values, rewards_ep = [], [], []

    for t in range(500):
        obs_t = torch.FloatTensor(obs)
        logits = policy(obs_t)
        v = value_fn(obs_t)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_probs.append(dist.log_prob(action))
        values.append(v.squeeze())
        obs, reward, terminated, truncated, _ = env.step(action.item())
        rewards_ep.append(reward)
        total_frames += 1
        if terminated or truncated:
            break
    env.close()

    returns = []
    G = 0
    for r in reversed(rewards_ep):
        G = r + 0.99 * G
        returns.insert(0, G)
    returns = torch.FloatTensor(returns)
    values = torch.stack(values)
    advantages = returns - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = -sum(lp * a for lp, a in zip(log_probs, advantages))
    value_loss = nn.functional.mse_loss(values, returns)
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 10 == 0:
        eval_rewards = []
        for _ in range(5):
            e = gym.make("CartPole-v1")
            o, _ = e.reset()
            ep_r = 0
            for _ in range(500):
                with torch.no_grad():
                    a = Categorical(logits=policy(torch.FloatTensor(o))).sample().item()
                o, r, term, trunc, _ = e.step(a)
                ep_r += r
                if term or trunc:
                    break
            e.close()
            eval_rewards.append(ep_r)
        avg = sum(eval_rewards) / 5
        print(f"Episode {episode} ({total_frames} frames): reward={avg:.1f}")
        if avg >= 490:
            print(f"Goal achieved at {total_frames} frames!")
            break

final_rewards = []
for _ in range(5):
    e = gym.make("CartPole-v1")
    o, _ = e.reset()
    ep_r = 0
    for _ in range(500):
        with torch.no_grad():
            a = Categorical(logits=policy(torch.FloatTensor(o))).sample().item()
        o, r, term, trunc, _ = e.step(a)
        ep_r += r
        if term or trunc:
            break
    e.close()
    final_rewards.append(ep_r)
print(f"METRIC: {sum(final_rewards)/5:.1f}")
print(f"FRAMES: {total_frames}")
'''
DESCRIPTIONS["actor_critic"] = "Actor-critic with learned value baseline — lower variance gradients"

# --- Variant 4: REINFORCE with eval every 5 episodes ---
FULL_SCRIPTS["eval_every5"] = BASE.replace(
    "if episode % 10 == 0:", "if episode % 5 == 0:"
)
DESCRIPTIONS["eval_every5"] = "REINFORCE with eval every 5 episodes — catch convergence earlier"


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmpdir:
        tasks = [(name, script, tmpdir) for name, script in FULL_SCRIPTS.items()]

        print(f"Running {len(tasks)} experiments in parallel...")
        for name in FULL_SCRIPTS:
            print(f"  - {name}: {DESCRIPTIONS[name]}")
        print()

        results = {}
        with ProcessPoolExecutor(max_workers=len(tasks)) as executor:
            futures = {executor.submit(run_script, task): task[0] for task in tasks}
            for future in as_completed(futures):
                name, metric, output, elapsed = future.result()
                results[name] = (metric, output, elapsed)
                print(f"=== {name} completed: METRIC={metric:.1f} ({elapsed:.1f}s) ===")
                print(f"  {DESCRIPTIONS[name]}")
                print()

        print("\n" + "=" * 70)
        print("RESULTS (sorted by metric, then by speed)")
        print("=" * 70)
        ranked = sorted(results.items(), key=lambda x: (x[1][0], -x[1][2]), reverse=True)
        for rank, (name, (metric, output, elapsed)) in enumerate(ranked, 1):
            frames_match = re.search(r"FRAMES:\s+(\d+)", output)
            frames_str = f", {frames_match.group(1)} frames" if frames_match else ""
            print(f"  {rank}. {name}: METRIC={metric:.1f} ({elapsed:.1f}s{frames_str}) — {DESCRIPTIONS[name]}")

        winner = ranked[0]
        print(f"\nWINNER: {winner[0]} (METRIC={winner[1][0]:.1f}, {winner[1][2]:.1f}s)")
