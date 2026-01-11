"""Точка входа"""
import os
import json

from typing import List, Optional
from src.RL.train import train_epochs
from src.core.config import settings
from src.core.models import Hospital
from src.run import run_two_way
from src.utils.utils import make_params_consistent

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)
IS_TRAIN = False

def load_hospital_config(path: Optional[str]) -> List[Hospital]:
    """Загрузка данныз о госпиталях"""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Hospital config JSON not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        cfgs = []
        for item in data:
            cfgs.append(Hospital(
                id=item.get("id"),
                name=item.get("name"),
                beds=int(item.get("beds")),
                reserve_beds=int(item.get("reserve_beds")),
                icu=int(item.get("icu_beds")),
                reserve_icu=int(item.get("reserve_icu")),
                quality=float(item.get("quality", 1.0)),
                rng_seed=settings.RANDOM_SEED,
                costs=item.get("costs"),
                budget=item.get("budget")
            ))
        return cfgs
    return []


def main():
    """Главный файл"""
    print("Loading data...")
    hosp_config = "data/hospitals.json"
    hospitals = load_hospital_config(hosp_config)

    params = make_params_consistent(
        population=settings.POPULATION,
        sigma=settings.SIGMA,
        gamma=settings.GAMMA,
        R0=settings.R0,
        p_hosp=settings.P_HOSP,
        p_icu=settings.P_INC,
        p_death=settings.P_DEATH,
        hosp_duration=settings.HOSP_DURATION,
        icu_duration=settings.INC_DURATION,
        initial_exposed=settings.INITIAL_EXPOSED,
        initial_infectious=settings.INITIAL_INFECTIOUS
    )

    if IS_TRAIN:
        agents, df_summary = train_epochs(
            hospitals_cfg=hospitals,
            init_params=params,
            days=110,
            num_epochs=500,
            save_dir="checkpoints/hospital_rl",
            seed_base=42
        )
    else:
        run_two_way(init_params=params, hospitals_cfg=hospitals, days=settings.DAYS)


if __name__ == "__main__":
    main()
