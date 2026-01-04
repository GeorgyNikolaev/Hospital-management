import os
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
from dataclasses import replace
import matplotlib.pyplot as plt
import copy

from src.RL.agent import HospitalAgent
from src.RL.env import HospitalEnv
from src.RL.run import run_with_rl


def train_epochs(
    hospitals_cfg,
    init_params,
    days,
    num_epochs,
    agents=None,
    save_dir="checkpoints",
    logs_csv="training_log.csv",
    seed_base=1234
):
    os.makedirs(save_dir, exist_ok=True)

    # если агенты None — создаём новые
    if agents is None:
        agents = [HospitalAgent() for _ in hospitals_cfg]

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))
    records = []

    best_value = None

    for epoch in range(1, num_epochs + 1):
        rng = np.random.RandomState(seed_base + epoch)  # детерминированное варьирование
        hospitals_cfg_copy = [copy.deepcopy(h) for h in hospitals_cfg]
        # создаём новые envs для эпизода
        envs = [HospitalEnv(i) for i in range(len(hospitals_cfg_copy))]

        init_params_copy = replace(init_params)

        # запускаем один эпизод (эпидемию)
        logs, des, agents, episode_rewards = run_with_rl(
            hospitals_cfg=hospitals_cfg_copy,
            init_params=init_params_copy,
            days=days,
            rng=rng,
            agents=agents,
            envs=envs
        )

        # summary metrics для эпохи
        total_deaths = sum(logs.get("deaths", [0]))
        total_rejected = sum(logs.get("rejected", [0]))
        total_infected = sum(logs.get("infection", [0]))
        mean_agent_reward = float(np.mean(episode_rewards))
        sum_agent_rewards = float(np.sum(episode_rewards))

        # логируем в TensorBoard
        writer.add_scalar("epoch/mean_agent_reward", mean_agent_reward, epoch)
        writer.add_scalar("epoch/sum_agent_rewards", sum_agent_rewards, epoch)
        writer.add_scalar("epoch/total_deaths", total_deaths, epoch)
        writer.add_scalar("epoch/total_rejected", total_rejected, epoch)
        writer.add_scalar("epoch/total_infected", total_infected, epoch)

        # записываем summary в records
        rec = {
            "epoch": epoch,
            "mean_agent_reward": mean_agent_reward,
            "sum_agent_rewards": sum_agent_rewards,
            "total_deaths": total_deaths,
            "total_rejected": total_rejected,
            "total_infected": total_infected
        }
        records.append(rec)

        # сохраняем master CSV (накопительный)
        df_records = pd.DataFrame(records)
        df_records.to_csv(os.path.join(save_dir, logs_csv), index=False)

        for i, agent in enumerate(agents):
            value = episode_rewards[i]

            if best_value is None or value > best_value:
                best_value = value
                path = os.path.join(save_dir, f"agent_best_.pt")
                ckpt = {
                    "epoch": epoch,
                    "model_state": agent.q.state_dict(),
                    "optimizer_state": agent.optim.state_dict(),
                    "eps": agent.eps,
                    "metric_value": value,
                    "metric_name": "reward"
                }
                torch.save(ckpt, path)
                for j in range(len(agents)):
                    agents[j].q.load_state_dict(agent.q.state_dict())
                    agents[j].q_target.load_state_dict(agent.q.state_dict())
                    agents[j].optim.load_state_dict(agent.optim.state_dict())

        print(f"[Epoch {epoch:3}] deaths={total_deaths:7.1f}, rejected={total_rejected:7.1f}, mean_reward={mean_agent_reward:7.3f}")

    writer.close()
    return agents, pd.DataFrame(records)


def load_agent_checkpoint(agent: HospitalAgent, ckpt_path: str, load_optimizer=True):
    data = torch.load(ckpt_path, map_location="cpu")
    agent.q.load_state_dict(data["model_state"])
    agent.q_target.load_state_dict(data["model_state"])
    if load_optimizer and "optimizer_state" in data:
        try:
            agent.optim.load_state_dict(data["optimizer_state"])
        except Exception:
            # если оптимизатор несовместим — пропускаем
            pass
    if "eps" in data:
        agent.eps = data["eps"]
    return agent
