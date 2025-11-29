""" """
import os
import json
import numpy as np
import pandas as pd

from typing import List, Optional

from src.core.config import settings
from src.core.models import Hospital
from src.run import run_two_way
from src.utils.utils import make_params_consistent

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_local_data(path: str, location: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"], dayfirst=False)
    if "location" in df.columns:
        df = df[df["location"] == location].copy()
    # If still empty, try exact match with variants (case-insensitive)
    if df.empty and "location" in pd.read_csv(path, nrows=5).columns:
        raise ValueError(f"No rows for region '{location}' in {path}. Check 'location' values.")
    df = df.sort_values("date").reset_index(drop=True)
    return df


def derive_observed_new_hosp(df: pd.DataFrame) -> np.ndarray:
    """Убираем нуль из данных"""
    n = len(df)
    if "total_cases" in df.columns:
        if df["total_cases"].notnull().sum() > 0:
            hp = df["total_cases"].fillna(0).values

            # Убираем нули в начале
            while len(hp) > 0 and hp[0] == 0:
                hp = hp[1:]

            if len(hp) == 0:
                return np.array([])

            # Находим только точки, где значение меняется
            diffs = []
            prev_value = 0
            for i in range(len(hp)):
                if hp[i] != prev_value:
                    diffs.append(hp[i] - prev_value)
                    prev_value = hp[i]

            return np.array(diffs)
    # fallback: none available
    return np.zeros(n)


def load_hospital_config(path: Optional[str]) -> List[Hospital]:
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

    # initial SEIR params
    region = "Australia"
    data_path = "D:/MEPHI/Simulation_modeling/data/owid-covid-data.csv"

    # real_data = load_real_data(data_path, region)
    # days = len(real_data)

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

    run_two_way(init_params=params, hospitals_cfg=hospitals, days=settings.DAYS)


if __name__ == "__main__":
    main()
