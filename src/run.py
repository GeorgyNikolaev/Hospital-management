"""Запуск симуляции"""
import copy
import numpy as np

from typing import List
from dataclasses import replace

from sd.seir_model import simulate_seir_hcd
from src.RL.agent import HospitalAgent
from src.RL.env import HospitalEnv
from src.RL.run import run_with_rl
from src.RL.train import load_agent_checkpoint
from src.core.config import settings
from src.core.models import SEIRHCDParams, Hospital
from src.utils import plots
from src.des import run_des
from src.TTM.run import run_ttm


def run_two_way(
    init_params: SEIRHCDParams,
    hospitals_cfg: List[Hospital],
    days: int,
):
    """Запуск симуляции"""
    rng = np.random.RandomState(settings.RANDOM_SEED)

    # Моделирование SD модели
    # init_params_sd = copy.deepcopy(init_params)
    #
    # sd_logs = simulate_seir_hcd(params=init_params_sd, days=settings.DAYS)
    # plots.display_SD_results(sd_logs, init_params_sd)
    # plots.plot_SD_results(sd_logs)
    # plots.save_SD_results(sd_logs)

    # Моделирование SD <-> DES
    hospitals_cfg_des = [copy.deepcopy(h) for h in hospitals_cfg]
    init_params_des = replace(init_params)

    des_logs, des = run_des.run(hospitals_cfg_des, init_params_des, days, rng)
    plots.display_results(des_logs)
    # plots.plot_SD_DES_results(des_logs)
    # plots.save_SD_DES_results(des_logs, des)

    # Моделирование с TTM управлением больниц
    hospitals_cfg_ttm = [copy.deepcopy(h) for h in hospitals_cfg]
    init_params_ttm = replace(init_params)

    ttm_logs, des = run_ttm(hospitals_cfg_ttm, init_params_ttm, days, rng)
    plots.display_results(ttm_logs)
    # plots.plot_SD_DES_results(ttm_logs)
    # plots.plot_RL_results(ttm_logs)
    # plots.plot_RL_actions(ttm_logs)

    # Использование RL агента
    hospitals_cfg_rl = [copy.deepcopy(h) for h in hospitals_cfg]
    init_params_rl = replace(init_params)
    agents = [HospitalAgent() for _ in hospitals_cfg_rl]
    envs = [HospitalEnv(i) for i in range(len(hospitals_cfg_rl))]
    for i in range(len(agents)):
        agents[i] = load_agent_checkpoint(agents[i], f"checkpoints/hospital_rl/agent_best_30_01_2026.pt")

    rl_logs, des, agents, _ = run_with_rl(hospitals_cfg_rl, init_params_rl, days, rng, agents, envs, False)
    plots.display_results(rl_logs)
    # plots.plot_SD_DES_results(rl_logs)
    plots.plot_RL_results(rl_logs)
    # plots.plot_DES_vs_RL(des_logs, rl_logs)
    plots.plot_RL_actions(rl_logs)

