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
    values_t = torch.stack(values)
    advantages = returns - values_t.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    policy_loss = -sum(lp * a for lp, a in zip(log_probs, advantages))
    value_loss = nn.functional.mse_loss(values_t, returns)
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
