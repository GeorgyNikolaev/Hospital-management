import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


def init_orthogonal(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight, gain=1.0)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

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
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, n_actions)
        )
        self.apply(init_orthogonal)  # 🔑 Инициализация весов

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class HospitalAgent:
    def __init__(
        self, obs_size=24, n_actions=10, lr=1e-4, gamma=0.99,
        batch_size=64, buffer_size=1_000, grad_clip=10.0,
        n_step=3, device="cpu"
    ):
        self.obs_size = obs_size
        self.n_actions = n_actions
        self.gamma = gamma
        self.n_step = n_step
        self.gamma_n = gamma ** n_step
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.device = torch.device(device)

        self.q = QNetwork(obs_size, n_actions).to(self.device)
        self.q_target = QNetwork(obs_size, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=buffer_size)

        self.eps = 1.0
        self.eps_decay = 0.998  # Чуть медленнее для эпизодов 150-200 шагов
        self.eps_min = 0.05
        self.total_steps = 0

        # Буфер для накопления n-step переходов
        self.n_step_buffer = []

    def store(self, obs, action, reward, next_obs, done, action_mask, next_action_mask):
        self.n_step_buffer.append((obs, action, reward, next_obs, done, action_mask, next_action_mask))

        # Агрегируем, когда набрали n шагов или эпизод завершился
        if len(self.n_step_buffer) >= self.n_step or done:
            agg_reward = 0.0
            discount = 1.0
            final_next_obs = next_obs
            final_done = done
            final_next_mask = next_action_mask

            # Идём по буферу, считаем дисконтированную награду
            for i, (s, a, r, s_, d, am, nam) in enumerate(self.n_step_buffer):
                agg_reward += discount * r
                discount *= self.gamma
                if d:
                    final_next_obs = s_
                    final_done = True
                    final_next_mask = nam
                    break
                if i == len(self.n_step_buffer) - 1:
                    final_next_obs = s_
                    final_next_mask = nam

            first = self.n_step_buffer[0]
            self.memory.append((
                first[0], first[1], agg_reward,
                final_next_obs, final_done,
                first[5], final_next_mask
            ))
            self.n_step_buffer.clear()

    def select_action(self, obs, action_mask):
        self.total_steps += 1
        self.eps = max(self.eps_min, self.eps * self.eps_decay)

        valid = [i for i, m in enumerate(action_mask) if m]
        if not valid: return 0
        if random.random() < self.eps:
            return random.choice(valid)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        with torch.no_grad():
            q = self.q(obs_t).squeeze(0)
            q[~mask_t] = -1e9
        return int(torch.argmax(q).item())

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        obs, act, rew, next_obs, done, action_mask, next_action_mask = zip(*batch)

        # 🔑 Жёсткое приведение к (batch_size,)
        obs_t = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        act_t = torch.tensor(act, dtype=torch.long).unsqueeze(1).to(self.device)
        rew_t = torch.tensor(rew, dtype=torch.float32).view(-1).to(self.device)  # (B,)
        next_obs_t = torch.tensor(np.array(next_obs), dtype=torch.float32).to(self.device)
        done_t = torch.tensor(done, dtype=torch.float32).view(-1).to(self.device)  # (B,)
        next_mask_t = torch.tensor(next_action_mask, dtype=torch.bool).to(self.device)

        # Q(s, a)
        q_vals = self.q(obs_t).gather(1, act_t).squeeze(1)

        with torch.no_grad():
            neg_inf = torch.full((self.batch_size, self.n_actions), -1e9, device=self.device)

            # Double DQN: online выбирает, target оценивает
            next_q_online = self.q(next_obs_t)
            next_q_online = torch.where(next_mask_t, next_q_online, neg_inf)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_target = self.q_target(next_obs_t)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)

            # 🔑 Target строго 1D, нет broadcasting
            target = rew_t + self.gamma_n * (1.0 - done_t) * next_q

        loss = self.loss_fn(q_vals, target)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optim.step()

        # 🔑 Soft target update (заменяет day % 20)
        tau = 0.005
        for t_param, s_param in zip(self.q_target.parameters(), self.q.parameters()):
            t_param.data.copy_(tau * s_param.data + (1.0 - tau) * t_param.data)