---
title: "From DQN to REINFORCE in 39 Iterations: Letting an AI Agent Optimize Its Own RL Algorithm"
authors:
  - name: Raphael Schwinger
date: 2026-03-28
subject: Reinforcement Learning, Autonomous Experimentation
keywords: autoresearch, reinforcement learning, DQN, REINFORCE, CartPole, autonomous agents
license: CC-BY-4.0
---

# From DQN to REINFORCE in 39 Iterations

_Letting an AI agent optimize its own RL algorithm on CartPole-v1 using the autoresearch loop._

## The Idea

What happens when you give an AI coding agent a reinforcement learning script and tell it to make it better --- autonomously, with no human in the loop?

That is the premise behind [autoresearch](https://github.com/karpathy/autoresearch), a pattern introduced by Andrej Karpathy in March 2026 {cite}`karpathy2026autoresearch`. The core loop is deceptively simple: **modify the code, run verification, keep if improved, discard if not, repeat**. Karpathy demonstrated this on LLM training, running 700 experiments in two days and discovering 20 optimizations that transferred to larger models.

We applied this pattern --- via the [Claude Code autoresearch skill](https://github.com/uditgoenka/autoresearch) by Udit Goenka {cite}`goenka2026autoresearch` --- to a different domain: optimizing a Deep Q-Network (DQN) {cite}`mnih2013dqn` agent on the classic CartPole-v1 benchmark. The agent started with a baseline scoring 207.6 average reward and, after 39 iterations, arrived at a REINFORCE {cite}`williams1992reinforce` implementation achieving a perfect score of 500 in under 10 seconds --- a 43$\times$ reduction in required training frames.

## The Setup

The rules were straightforward:

- **Goal:** achieve a perfect score of 500 on CartPole-v1, consistently $\geq 400$
- **Constraint:** training + evaluation must complete in under 5 minutes
- **Metric:** average reward over 5 evaluation episodes (higher is better)
- **Scope:** a single Python file (`algorithm.py`)

The autoresearch loop works as follows:

```
LOOP:
  1. Review current state + git history + results log
  2. Pick ONE focused change
  3. Make the change, git commit
  4. Run verification (python algorithm.py)
  5. If metric improved → keep. If worse → git revert.
  6. Log the result to autoresearch-results.tsv
  7. Repeat.
```

Every experiment is committed _before_ verification, so `git revert` provides clean rollback. Failed experiments stay visible in the git history --- they are the agent's memory of what not to try again.

## Phase 1: Optimizing DQN (Iterations 0--22)

The initial implementation used TorchRL {cite}`bou2024torchrl` with a standard DQN setup: two-layer network (128 hidden units), $\varepsilon$-greedy exploration, replay buffer of 50,000 transitions, soft target updates ($\tau = 0.001$), and Adam optimizer at $\text{lr} = 10^{-4}$.

**Baseline result: METRIC = 207.6.** The agent would briefly spike to $\sim$400 reward around step 241k, then catastrophically collapse back to $\sim$9.

The first 22 iterations explored the DQN hyperparameter space:

| Iteration | Change | METRIC | Status |
|:---------:|--------|-------:|--------|
| 0 | Baseline DQN | 207.6 | baseline |
| 1 | lr $10^{-4} \to 5 \times 10^{-4}$ | 158.4 | discard |
| 5 | Buffer $50\text{k} \to 100\text{k}$, warmup $5\text{k} \to 10\text{k}$ | **500.0** | **keep** |
| 13 | Total frames $500\text{k} \to 300\text{k}$, $\tau \to 0.005$ | 500.0 $\to$ 245.2 | discard (failed confirmation) |
| 22 | Network $2 \times 128 \to 3 \times 64$ | **500.0** | **keep** |

The breakthrough came at iteration 5: **doubling the replay buffer** (to 100k) **and the warmup period** (to 10k frames) stabilized learning dramatically. The larger buffer provides more diverse training data, and the longer warmup ensures the buffer contains enough varied transitions before gradient updates begin. This single change took the agent from an unstable 207.6 to a confirmed perfect 500.

A deeper, narrower network ($3 \times 64$ instead of $2 \times 128$) was kept at iteration 22 as well --- it produced a more stable representation that jumped directly from the "stuck at 9" phase to 500 without the intermediate oscillation.

But the best DQN configuration still required $\sim$421,000 frames and about 3 minutes of wall-clock time. Could we do better?

## Phase 2: The Algorithm Switch (Iterations 23--39)

At iteration 23, we expanded the search space beyond DQN to include other RL algorithms. Four candidates were tested in parallel:

| Algorithm | METRIC | Frames | Time |
|-----------|-------:|-------:|-----:|
| REINFORCE | **497.8** | **9,868** | **4.8s** |
| PPO (TorchRL) | 8.8 | 100,352 | 46.8s |
| DQN (aggressive) | 273.0 | 200,000 | 65.2s |
| Double DQN | 236.0 | 300,000 | 91.9s |

REINFORCE --- the simplest policy gradient algorithm, dating back to Williams (1992) {cite}`williams1992reinforce` --- crushed DQN by achieving near-perfect reward in **43$\times$ fewer frames**. The entire implementation is under 80 lines of pure PyTorch:

```python
policy = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 2))
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-2)

# For each episode: collect trajectory, compute returns,
# update policy with REINFORCE gradient
loss = -sum(log_prob * G for log_prob, G in zip(log_probs, returns))
```

No replay buffer. No target network. No $\varepsilon$-greedy exploration. No TorchRL dependency. Just a two-layer network, Adam at $\text{lr} = 10^{-2}$, and the vanilla policy gradient theorem.

The remaining iterations (28--39) attempted to improve the REINFORCE implementation further: actor-critic baselines, entropy bonuses, gradient clipping, learning rate tuning, and deterministic evaluation. **None of them improved on the base REINFORCE.** Several actively hurt performance:

- **Actor-critic** (iteration 28): scored 500 in a parallel test but collapsed to 9.6 on confirmation --- a cautionary tale about trusting unconfirmed parallel results
- **Entropy bonus** (iteration 38): prevented convergence entirely (METRIC = 114.8) by encouraging too much exploration
- **Gradient clipping** (iteration 34): smoothed the training curve but slowed convergence from 10k to 26k frames

The only improvement that stuck was evaluating every 5 episodes instead of 10 (iteration 32), which caught convergence earlier without affecting training dynamics.

## What We Learned

### 1. Simpler algorithms win on simple problems

CartPole has a 4-dimensional observation space and 2 actions. DQN's machinery --- replay buffers, target networks, $\varepsilon$-greedy schedules --- is designed for complex, high-dimensional environments like Atari {cite}`mnih2015humanlevel`. For CartPole, this machinery is pure overhead. REINFORCE's direct policy gradient is sufficient and dramatically more efficient.

This is not a novel insight (any RL textbook will tell you this), but the autoresearch loop _rediscovered_ it empirically in 39 iterations without being told which algorithm to prefer.

### 2. Parallel experiments need confirmation runs

We ran batches of 4 experiments in parallel using Python's `ProcessPoolExecutor`. This was excellent for screening --- it identified REINFORCE as a candidate in one batch --- but **parallel results cannot be trusted as final**. Process-level non-determinism (even with fixed seeds) means that a metric achieved in a forked subprocess may not reproduce when running the same script directly. Three experiments that scored 500 in parallel failed their confirmation runs.

The protocol that worked: use parallel batches for exploration, then always verify winners with a sequential confirmation run before committing.

### 3. RL metrics are inherently noisy

Across our 39 iterations, the same configuration could produce METRIC values ranging from 484 to 500 on different runs. REINFORCE is especially volatile: the agent might solve CartPole at episode 80 (10k frames) on one run and episode 235 (37k frames) on the next.

This noise makes the autoresearch keep/discard decision harder than in deterministic domains like test coverage or bundle size. We found that seed pinning helps with the training side (`torch.manual_seed(0)`, `env.reset(seed=episode)`), but evaluation remains stochastic. The best strategy is to accept the variance and focus on whether changes produce _consistently_ good results across multiple runs, rather than chasing a single peak metric.

### 4. Git as memory is powerful

The autoresearch loop's use of git as memory proved invaluable. Every experiment --- successful or not --- lives in the commit history. The agent can read `git log` to see what was tried, inspect `git diff` on kept commits to understand _why_ something worked, and avoid repeating failed approaches. Over 39 iterations, this produced a rich experimental record that a human researcher could review in minutes.

## Final Numbers

| Configuration | Commit | METRIC | Frames | Wall Time |
|:--------------|:------:|-------:|-------:|----------:|
| DQN Baseline | `2cb5664` | 207.6 | ~421,000 | ~3 min |
| DQN Optimized (buffer + warmup + 3$\times$64 net) | `6c821f9` | 500.0 | ~421,000 | ~3 min |
| REINFORCE (eval every 5 episodes) | `a066fb7` | 500.0 | ~10,000 | ~5 sec |

**39 iterations. 4 keeps. 34 discards. 0 crashes. One algorithm switch that changed everything.**

The full experiment log, git history, and code are available in the [autoresearch-dqn repository](https://github.com/rschwinger/autoresearch-dqn).

## References

```{bibliography}
:style: unsrt
```

---

```{code-block} bibtex
:filename: references.bib

@misc{karpathy2026autoresearch,
  author       = {Karpathy, Andrej},
  title        = {autoresearch: {AI} agents running research on single-{GPU} nanochat training automatically},
  year         = {2026},
  howpublished = {\url{https://github.com/karpathy/autoresearch}},
  note         = {Released March 7, 2026. 21,000+ GitHub stars within days of release.}
}

@misc{goenka2026autoresearch,
  author       = {Goenka, Udit},
  title        = {Claude Autoresearch Skill --- Autonomous goal-directed iteration for {Claude Code}},
  year         = {2026},
  howpublished = {\url{https://github.com/uditgoenka/autoresearch}},
  note         = {Claude Code skill implementing the autoresearch loop pattern.}
}

@article{mnih2013dqn,
  author  = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Graves, Alex and Antonoglou, Ioannis and Wierstra, Daan and Riedmiller, Martin},
  title   = {Playing {Atari} with Deep Reinforcement Learning},
  journal = {arXiv preprint arXiv:1312.5602},
  year    = {2013}
}

@article{mnih2015humanlevel,
  author  = {Mnih, Volodymyr and Kavukcuoglu, Koray and Silver, David and Rusu, Andrei A. and Veness, Joel and Bellemare, Marc G. and Graves, Alex and Riedmiller, Martin and Fidjeland, Andreas K. and Ostrovski, Georg and others},
  title   = {Human-level control through deep reinforcement learning},
  journal = {Nature},
  volume  = {518},
  number  = {7540},
  pages   = {529--533},
  year    = {2015},
  doi     = {10.1038/nature14236}
}

@article{williams1992reinforce,
  author  = {Williams, Ronald J.},
  title   = {Simple statistical gradient-following algorithms for connectionist reinforcement learning},
  journal = {Machine Learning},
  volume  = {8},
  number  = {3--4},
  pages   = {229--256},
  year    = {1992},
  doi     = {10.1007/BF00992696}
}

@article{bou2024torchrl,
  author  = {Bou, Albert and Bettini, Matteo and Dittert, Sebastian and Kumar, Vikash and Sodhani, Shagun and Yang, Xiaomeng and De Fabritiis, Gianni and Moens, Vincent},
  title   = {{TorchRL}: A data-driven decision-making library for {PyTorch}},
  journal = {arXiv preprint arXiv:2306.00577},
  year    = {2024}
}
```
