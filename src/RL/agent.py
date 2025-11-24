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
    """
    DQN агент, работающий в реальном времени:
    - принимает текущее состояние
    - выдает действие
    - обучается на опыте
    """
    def __init__(self, obs_size=6, n_actions=3, lr=1e-3, gamma=0.99):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.gamma = gamma

        self.q = QNetwork(obs_size, n_actions)
        self.q_target = QNetwork(obs_size, n_actions)
        self.q_target.load_state_dict(self.q.state_dict())

        self.optim = optim.Adam(self.q.parameters(), lr=lr)

        self.memory = deque(maxlen=50000)
        self.batch_size = 64

        self.eps = 1.0  # exploration
        self.eps_decay = 0.995
        self.eps_min = 0.05

    def select_action(self, obs):
        """
        eps-greedy выбор действия
        """
        if random.random() < self.eps:
            return random.randint(0, self.n_actions - 1)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            qvals = self.q(obs_t)
        return int(torch.argmax(qvals).item())

    def store(self, obs, action, reward, next_obs):
        self.memory.append((obs, action, reward, next_obs))

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)

        obs, act, rew, next_obs = zip(*batch)

        obs = torch.tensor(obs, dtype=torch.float32)
        act = torch.tensor(act, dtype=torch.long)
        rew = torch.tensor(rew, dtype=torch.float32)
        next_obs = torch.tensor(next_obs, dtype=torch.float32)

        q_vals = self.q(obs).gather(1, act.unsqueeze(1)).squeeze()

        with torch.no_grad():
            next_q = self.q_target(next_obs).max(1)[0]
            target = rew + self.gamma * next_q

        loss = nn.MSELoss()(q_vals, target)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # обновление eps
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

    def update_target(self):
        self.q_target.load_state_dict(self.q.state_dict())
