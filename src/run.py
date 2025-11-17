import numpy as np

from typing import List

from src.core.config import settings
from src.core.models import SEIRHCDParams, HospitalConfig
from src.utils.plots import plot_SD_results, plot_SD_DES_results, save_SD_results, save_SD_DES_results
from src.sd import run_sd
from src.des import run_des


def run_two_way(
    init_params: SEIRHCDParams,
    hospitals_cfg: List[HospitalConfig],
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
