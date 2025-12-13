import copy
import json
from dataclasses import replace

import numpy as np

from typing import List

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
    sd_logs = run_sd.run(init_params)
    # Графики
    plot_SD_results(sd_logs)
    save_SD_results(sd_logs)

    # Моделирование SD <-> DES
    des_logs, des = run_des.run(hospitals_cfg, init_params, days, rng)
    # Сохранение данных
    plot_SD_DES_results(des_logs)
    save_SD_DES_results(des_logs, des)

    agents = [HospitalAgent() for _ in hospitals_cfg]
    for i in range(len(agents)):
        agents[i] = load_agent_checkpoint(agents[i], f"checkpoints/hospital_rl/agent{i}_best.pt")

    envs = [HospitalEnv(i) for i in range(len(hospitals_cfg))]

    logs, des, agents, _ = run_with_rl(hospitals_cfg, init_params, days, rng, agents, envs, False)
    # print(json.dumps(logs, indent=2, ensure_ascii=False))
    plot_RL_results(logs)



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
