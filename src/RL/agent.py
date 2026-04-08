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
    def __init__(self, obs_size, n_actions, hidden_dim=128):  # 🔑 Увеличил до 256 для стабильности
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
        self.apply(init_orthogonal)

    def forward(self, x):
        features = self.feature_layer(x)
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class HospitalAgent:
    def __init__(
            self, obs_size=24, n_actions=10, lr=5e-4, gamma=0.99,
            batch_size=64, buffer_size=20_000, grad_clip=5.0,
            n_step=2, device="cpu", reward_scale=1  # 🔑 Критически важно
    ):
        self.n_actions = n_actions
        self.n_step = n_step
        self.gamma = gamma
        self.gamma_n = gamma ** n_step
        self.batch_size = batch_size
        self.grad_clip = grad_clip
        self.device = torch.device(device)
        self.reward_scale = reward_scale  # Делитель для наград

        self.q = QNetwork(obs_size, n_actions).to(self.device)
        self.q_target = QNetwork(obs_size, n_actions).to(self.device)
        self.q_target.load_state_dict(self.q.state_dict())
        self.q_target.eval()

        self.optim = optim.Adam(self.q.parameters(), lr=lr, weight_decay=1e-4)
        self.loss_fn = nn.SmoothL1Loss()
        self.memory = deque(maxlen=buffer_size)

        # 🔑 Линейный decay вместо экспоненциального
        self.eps = 1.0
        self.eps_min = 0.05
        self.eps_decay_steps = 80_000  # За сколько шагов достичь eps_min
        self.warmup_steps = 2_000  # Не обучаем первые N шагов
        self.total_steps = 0
        self.n_step_buffer = deque()

    def store(self, obs, action, reward, next_obs, done, action_masks, next_action_mask):
        reward /= self.reward_scale
        self.n_step_buffer.append((obs, action, reward, next_obs, done, next_action_mask))

        # Если буфер ещё не накопил n шагов и эпизод не завершён — ждём
        if len(self.n_step_buffer) < self.n_step and not done:
            return

        # Формируем переход из первого элемента буфера
        agg_reward = 0.0
        discount = 1.0
        final_next_obs = None
        final_done = False
        final_next_mask = None

        for i, (s, a, r, s_, d, nam) in enumerate(self.n_step_buffer):
            agg_reward += discount * r
            discount *= self.gamma
            if d:
                final_next_obs = s_
                final_done = True
                final_next_mask = nam
                break
            final_next_obs = s_
            final_next_mask = nam
            final_done = d

        first = self.n_step_buffer[0]
        self.memory.append((
            first[0], first[1], agg_reward,
            final_next_obs, final_done, final_next_mask
        ))

        # Сдвигаем окно (удаляем первый элемент)
        self.n_step_buffer.popleft()  # используйте collections.deque

    def select_action(self, obs, action_mask):
        self.total_steps += 1

        # 🔑 Линейное затухание exploration
        self.eps = max(self.eps_min, 1.0 - self.total_steps / self.eps_decay_steps)

        valid = [i for i, m in enumerate(action_mask) if m]
        if not valid:
            return 0  # fallback
        if random.random() < self.eps:
            return random.choice(valid)

        obs_t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
        mask_t = torch.tensor(action_mask, dtype=torch.bool).to(self.device)
        with torch.no_grad():
            q = self.q(obs_t).squeeze(0)
            q[~mask_t] = -1e9
        return int(torch.argmax(q).item())

    def train_step(self):
        # 🔑 Warmup + минимальный размер буфера
        if self.total_steps < self.warmup_steps or len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        obs, act, rew, next_obs, done, next_action_mask = zip(*batch)

        obs_t = torch.tensor(np.array(obs), dtype=torch.float32).to(self.device)
        act_t = torch.tensor(act, dtype=torch.long).unsqueeze(1).to(self.device)
        rew_t = torch.tensor(rew, dtype=torch.float32).view(-1).to(self.device)
        next_obs_t = torch.tensor(np.array(next_obs), dtype=torch.float32).to(self.device)
        done_t = torch.tensor(done, dtype=torch.float32).view(-1).to(self.device)
        next_mask_t = torch.tensor(next_action_mask, dtype=torch.bool).to(self.device)

        q_vals = self.q(obs_t).gather(1, act_t).squeeze(1)

        with torch.no_grad():
            neg_inf = torch.full((self.batch_size, self.n_actions), -1e9, device=self.device)

            next_q_online = self.q(next_obs_t)
            next_q_online = torch.where(next_mask_t, next_q_online, neg_inf)
            next_actions = next_q_online.argmax(dim=1, keepdim=True)

            next_q_target = self.q_target(next_obs_t)
            next_q = next_q_target.gather(1, next_actions).squeeze(1)

            target = rew_t + self.gamma_n * (1.0 - done_t) * next_q

        loss = self.loss_fn(q_vals, target)
        self.optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), self.grad_clip)
        self.optim.step()

        # Soft target update
        tau = 0.005
        for t, s in zip(self.q_target.parameters(), self.q.parameters()):
            t.data.copy_(tau * s.data + (1.0 - tau) * t.data)