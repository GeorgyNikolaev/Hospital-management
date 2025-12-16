import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class QNetwork(nn.Module):
    def __init__(self, obs_size, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions)
        )

    def forward(self, x):
        return self.net(x)


class HospitalAgent:
    """DQN агент"""
    def __init__(self, obs_size=24, n_actions=10, lr=1e-3, gamma=0.99):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.gamma = gamma

        self.q = QNetwork(obs_size, n_actions)
        self.q_target = QNetwork(obs_size, n_actions)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optim = optim.Adam(self.q.parameters(), lr=lr)

        self.memory = deque(maxlen=5000)
        self.batch_size = 64

        self.eps = 1.0  # exploration
        self.eps_decay = 0.995
        self.eps_min = 0.05

    def select_action(self, obs, action_mask):
        """
        obs: np.array или list
        action_mask: list/iterable длины n_actions, 1 допустимо, 0 запрещено
        """
        # безопасная обработка маски: если нет ни одного разрешённого — возвращаем 0
        valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
        if len(valid_actions) == 0:
            return 0

        if random.random() < self.eps:
            return random.choice(valid_actions)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # (1, obs_size)
        with torch.no_grad():
            qvals = self.q(obs_t).squeeze(0)  # (n_actions,)

        # маскируем запрещённые действия - делаем их очень маленькими
        mask_t = torch.tensor(action_mask, dtype=torch.bool)
        qvals[~mask_t] = -1e9

        return int(torch.argmax(qvals).item())

    def store(self, obs, action, reward, next_obs, action_mask, next_action_mask):
        # сохраняем маски тоже
        self.memory.append((obs, action, reward, next_obs, action_mask, next_action_mask))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        obs, act, rew, next_obs, action_mask, next_action_mask = zip(*batch)

        obs = torch.tensor(np.array(obs), dtype=torch.float32)              # (B, obs_size)
        act = torch.tensor(act, dtype=torch.long)                  # (B,)
        rew = torch.tensor(rew, dtype=torch.float32)              # (B,)
        next_obs = torch.tensor(np.array(next_obs), dtype=torch.float32)    # (B, obs_size)
        # action_mask = torch.tensor(action_mask, dtype=torch.bool) # (B, n_actions)
        next_action_mask = torch.tensor(next_action_mask, dtype=torch.bool) # (B, n_actions)

        q_vals = self.q(obs).gather(1, act.unsqueeze(1)).squeeze(1)  # (B,)

        with torch.no_grad():
            next_q_all = self.q_target(next_obs)  # (B, n_actions)
            # применяем маску по-строчно: там где mask==0 ставим -inf
            neg_inf = torch.full_like(next_q_all, -1e9)
            # where mask true -> take value, else -1e9
            next_q_masked = torch.where(next_action_mask, next_q_all, neg_inf)
            next_max = next_q_masked.max(dim=1)[0]  # (B,)
            target = rew + self.gamma * next_max

        loss = nn.MSELoss()(q_vals, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # обновление eps
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def update_target(self):
        self.q_target.load_state_dict(self.q.state_dict())
