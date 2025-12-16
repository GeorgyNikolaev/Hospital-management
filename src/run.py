import copy
import json
from dataclasses import replace

import numpy as np

from typing import List

import pandas as pd
from matplotlib import pyplot as plt

from src.RL.agent import HospitalAgent
from src.RL.env import HospitalEnv
from src.RL.run import run_with_rl
from src.RL.train import load_agent_checkpoint
from src.core.config import settings
from src.core.models import SEIRHCDParams, Hospital
from src.utils.plots import plot_SD_results, plot_SD_DES_results, save_SD_results, save_SD_DES_results, plot_RL_results
from src.sd import run_sd
from src.des import run_des


def run_two_way(
    init_params: SEIRHCDParams,
    hospitals_cfg: List[Hospital],
    days: int,
):
    rng = np.random.RandomState(settings.RANDOM_SEED)

    # Моделирование SD модели
    # sd_logs = run_sd.run(init_params)
    # Графики
    # plot_SD_results(sd_logs)
    # save_SD_results(sd_logs)

    # Моделирование SD <-> DES
    # des_logs, des = run_des.run(hospitals_cfg, init_params, days, rng)
    # Сохранение данных
    # plot_SD_DES_results(des_logs)
    # save_SD_DES_results(des_logs, des)

    agents = [HospitalAgent() for _ in hospitals_cfg]
    for i in range(len(agents)):
        agents[i] = load_agent_checkpoint(agents[i], f"checkpoints/hospital_rl/agent_best.pt")

    envs = [HospitalEnv(i) for i in range(len(hospitals_cfg))]

    rl_logs, des, agents, _ = run_with_rl(hospitals_cfg, init_params, days, rng, agents, envs, False)
    rl_logs = pd.DataFrame(rl_logs)
    plot_SD_DES_results(rl_logs)
    # print(json.dumps(logs, indent=2, ensure_ascii=False))
    plot_RL_results(rl_logs)



    data_array = np.array(rl_logs["actions"])
    data_array = np.array(data_array.tolist())

    # Создаем график
    # Словарь соответствия числовых значений и текстовых меток
    action_labels = {
        0: 'Ничего не делать',
        1: 'Купить 1 койку',
        2: 'Купить 5 коек',
        3: 'Купить 1 аппарат ИВЛ',
        4: 'Купить 5 аппаратов ИВЛ',
        5: 'Законсервировать 1 койку',
        6: 'Законсервировать 5 коек',
        7: 'Законсервировать 1 аппарат ИВЛ',
        8: 'Законсервировать 5 аппаратов ИВЛ',
        9: 'Срочно выделить бюджет'
    }

    plt.figure(figsize=(10, 6))
    x = range(len(rl_logs["actions"]))

    for i in range(data_array.shape[1]):
        y = [subarray[i] for subarray in rl_logs["actions"]]
        plt.plot(x, y, 'o-', label=f'Больница {i + 1}', markersize=8)

    plt.xlabel('День')
    plt.title('График принятия решений больниц')

    # Устанавливаем метки на оси Y
    plt.yticks(ticks=list(action_labels.keys()), labels=list(action_labels.values()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # Чтобы метки не обрезались
    plt.show()
