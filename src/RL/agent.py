import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions, hidden_dim=128):
        super().__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(obs_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        features = self.feature_layer(x)

        value = self.value_stream(features)                  # (B, 1)
        advantage = self.advantage_stream(features)          # (B, A)

        q_values = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_values

class HospitalAgent:
    def __init__(
        self,
        obs_size=24,
        n_actions=10,
        lr=1e-4,
        gamma=0.99,
        batch_size=64,
        buffer_size=10_000,
        grad_clip=10.0
    ):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.grad_clip = grad_clip

        self.q = QNetwork(obs_size, n_actions)
        self.q_target = QNetwork(obs_size, n_actions)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

        self.memory = deque(maxlen=buffer_size)

        self.eps = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.05

    def select_action(self, obs, action_mask):
        valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
        if not valid_actions:
            return 0

        if random.random() < self.eps:
            return random.choice(valid_actions)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            qvals = self.q(obs_t).squeeze(0)

        mask_t = torch.tensor(action_mask, dtype=torch.bool)
        qvals[~mask_t] = -1e9

        return int(torch.argmax(qvals).item())

    def store(self, obs, action, reward, next_obs, action_mask, next_action_mask):
        self.memory.append(
            (obs, action, reward, next_obs, action_mask, next_action_mask)
        )
    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        obs, act, rew, next_obs, action_mask, next_action_mask = zip(*batch)

        obs = torch.tensor(np.array(obs), dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.long)
        rew = torch.tensor(rew, dtype=torch.float32)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32)
        next_action_mask = torch.tensor(next_action_mask, dtype=torch.bool)

        # Q(s, a)
        q_vals = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            neg_inf = torch.full((self.batch_size, self.n_actions), -1e9)

            # 1. online-сеть выбирает действие
            next_q_online = self.q(next_obs)
            next_q_online = torch.where(next_action_mask, next_q_online, neg_inf)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            # 2. target-сеть оценивает выбранное действие
            next_q_target = self.q_target(next_obs)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)

            target = rew + self.gamma * next_q

        loss = self.loss_fn(q_vals, target)

        self.optim.zero_grad()
        loss.backward()

        # gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)

        self.optim.step()

        # epsilon decay
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def update_target(self):
        self.q_target.load_state_dict(self.q.state_dict())