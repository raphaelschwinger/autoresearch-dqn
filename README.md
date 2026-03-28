# Autoresearch DQN on Cartpole

In this repository we let an [autoresearch agent](https://github.com/karpathy/autoresearch) lose at Cartpole.

The goal is to implement a [DQN](https://arxiv.org/abs/1312.5602) agent that can learn to play Cartpole. The agent should iterate over the codebase to find find the cleanest and most performant implementation of the algorithm.

## Rules
- The agent should be able to learn to play Cartpole achieving a perfect score of 500.
- The agent should reach the perfect score in the least number of iterations possible while consistently achieving a score of at least 400.
- A trainig + eval run should not take longer than 5min.
- The code should be as clean and readable as possible, meaning as few imports and dependencies as possible.
- The code should import and use the following libraries:
```python
import torch
import torch.nn as nn
from torchrl.envs import GymEnv
from torchrl.modules import QValueActor, EGreedyModule
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict.nn import TensorDictModule, TensorDictSequential
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd autoresearch-dqn

# Install dependencies (Python 3.10+)
pip install torch torchrl tensordict gymnasium
```

## Running the autoresearch loop

The project uses the [autoresearch pattern](https://github.com/uditgoenka/autoresearch) to autonomously iterate on the DQN implementation.

### Single experiment (sequential loop)

```bash
# Run the current DQN implementation
python dqn.py
```

The script prints training progress and ends with `METRIC: <avg_reward>` — the average reward over 5 evaluation episodes.

### Parallel experiments

Run 4 hyperparameter variants simultaneously and compare results:

```bash
python run_experiments.py
```

This generates 4 variant scripts with different single changes, runs them all in parallel, and reports which configuration achieved the best metric.

### Autoresearch loop with Claude Code

To run the full autonomous iteration loop:

```bash
# Install the autoresearch skill (https://github.com/uditgoenka/autoresearch)
# Then invoke:
/autoresearch
Goal: DQN agent consistently achieves >=400 reward on CartPole-v1, ideally 500
Scope: dqn.py
Metric: Average eval reward (higher is better)
Verify: python dqn.py 2>&1 | grep 'METRIC:' | tail -1 | awk '{print $2}'
```

The loop follows: **Modify -> Verify -> Keep/Discard -> Repeat** until the goal is achieved. Results are logged to `autoresearch-results.tsv`.

## Initial implementation

```python
import torch
import torch.nn as nn
from torchrl.envs import GymEnv
from torchrl.modules import QValueActor, EGreedyModule
from torchrl.objectives import DQNLoss, SoftUpdate
from torchrl.collectors import SyncDataCollector
from torchrl.data import ReplayBuffer, LazyTensorStorage
from tensordict.nn import TensorDictModule, TensorDictSequential

torch.manual_seed(0)

env = GymEnv("CartPole-v1", categorical_action_encoding=True)

qnet = TensorDictModule(
    nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 2)),
    in_keys=["observation"],
    out_keys=["action_value"],
)
policy = QValueActor(qnet, in_keys=["observation"], spec=env.action_spec)

exploration_policy = EGreedyModule(spec=env.action_spec, eps_init=1.0, eps_end=0.05, annealing_num_steps=50_000)
exploration_module = TensorDictSequential(policy, exploration_policy)

collector = SyncDataCollector(
    create_env_fn=lambda: GymEnv("CartPole-v1", categorical_action_encoding=True),
    policy=exploration_module,
    frames_per_batch=1000,
    total_frames=500_000,
)

buffer = ReplayBuffer(storage=LazyTensorStorage(max_size=50_000))
loss_fn = DQNLoss(policy, action_space=env.action_spec)
loss_fn.make_value_estimator(gamma=0.99)
updater = SoftUpdate(loss_fn, tau=0.001)
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)

total_frames = 0
for i, batch in enumerate(collector):
    buffer.extend(batch)
    total_frames += batch.numel()

    if len(buffer) < 5000:
        continue

    for _ in range(10):
        loss = loss_fn(buffer.sample(256))
        optimizer.zero_grad()
        loss["loss"].backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        updater.step()

    exploration_policy.step(frames=batch.numel())

    if i % 10 == 0:
        rewards = []
        with torch.no_grad():
            for _ in range(5):
                rewards.append(env.rollout(max_steps=500, policy=policy)["next", "reward"].sum().item())
        print(f"Step {total_frames}: loss={loss['loss'].item():.3f}, reward={sum(rewards)/5:.1f}, eps={exploration_policy.eps:.3f}")

collector.shutdown()
env.close()
```