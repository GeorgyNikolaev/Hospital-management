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
    save_every=5,
    best_metric="deaths",   # "deaths" минимизировать; или "reward" максимизировать
    seed_base=1234
):
    os.makedirs(save_dir, exist_ok=True)

    # если агенты None — создаём новые
    if agents is None:
        agents = [HospitalAgent() for _ in hospitals_cfg]

    writer = SummaryWriter(log_dir=os.path.join(save_dir, "tb"))
    records = []

    # best tracking
    best_value = None
    best_path = None


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

        if epoch % 50 == 0:
            data_array = np.array(logs["actions"])

            # Создаем график
            plt.figure(figsize=(10, 6))

            # X координаты (номера подмассивов)
            x = range(len(logs["actions"]))

            # Для каждого элемента в подмассиве строим отдельный график
            for i in range(data_array.shape[1]):  # data_array.shape[1] = 3
                y = [subarray[i] for subarray in logs["actions"]]
                plt.plot(x, y, 'o-', label=f'Элемент {i + 1}', markersize=8)

            plt.xlabel('Номер подмассива')
            plt.ylabel('Значение')
            plt.title('Графики значений по позициям в подмассивах')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()

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

        # сохраняем per-day time-series logs в отдельный CSV (опционально)
        # преобразуем logs (dict of lists) -> df и сохраним в per-epoch csv

        # df_epoch = pd.DataFrame(logs)
        # df_epoch.to_csv(os.path.join(save_dir, f"epoch_{epoch}_timeseries.csv"), index=False)

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

        # # чекпоинт моделей периодически
        # if epoch % save_every == 0:
        #     for i, agent in enumerate(agents):
        #         ckpt = {
        #             "epoch": epoch,
        #             "model_state": agent.q.state_dict(),
        #             "optimizer_state": agent.optim.state_dict(),
        #             "eps": agent.eps
        #         }
        #         torch.save(ckpt, os.path.join(save_dir, f"agent{i}_epoch{epoch}.pt"))

        # сохранить лучшую модель по best_metric
        if best_metric == "deaths":
            metric_value = total_deaths  # минимизировать
            is_better = (best_value is None) or (metric_value < best_value)
        elif best_metric == "reward":
            metric_value = sum_agent_rewards  # максимизировать
            is_better = (best_value is None) or (metric_value > best_value)
        else:
            metric_value = sum_agent_rewards
            is_better = (best_value is None) or (metric_value > best_value)

        if is_better:
            best_value = metric_value
            # удаляем старый best, если был
            if best_path is not None and os.path.exists(best_path):
                try:
                    os.remove(best_path)
                except Exception:
                    pass
            # сохраняем новые best
            for i, agent in enumerate(agents):
                path = os.path.join(save_dir, f"agent{i}_best.pt")
                ckpt = {
                    "epoch": epoch,
                    "model_state": agent.q.state_dict(),
                    "optimizer_state": agent.optim.state_dict(),
                    "eps": agent.eps,
                    "metric_value": metric_value,
                    "metric_name": best_metric
                }
                torch.save(ckpt, path)
            best_path = os.path.join(save_dir, f"agent0_best.pt")  # just marker

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
