from typing import List

import pandas as pd

from src.core.config import settings
from src.core.models import SEIRHCDParams, Hospital, Patient
from src.des.des_model import DES
from src.sd.seir_model import simulate_seir_hcd
from src.utils.utils import sample_arrivals


def run(hospitals_cfg: List[Hospital], init_params: SEIRHCDParams, days: int, rng):
    """Симуляция"""
    des = DES(hospitals_cfg, rng_seed=settings.RANDOM_SEED)
    params = init_params

    # calibrate if requested and data provided
    # if calibrate and (observed_new_hosp is not None) and observed_new_hosp.sum() > 0:
    #     use_days = min(len(observed_new_hosp), max(30, days // 3))
    #     params = calibrate_seir(params, observed_new_hosp[:use_days], days=use_days)
    #     print(f"Calibrated: beta={params.beta:.4f}, hosp_rate={params.hosp_rate:.4f}")

    logs = {"day": [],
            "infection": [],
            "hosp_expected": [],
            "icu_expected": [],
            "deaths_expected": [],
            "population": []
            }

    pid = 0
    beta_modifier = 1.0

    def beta_time_fn(ti, beta):
        return beta * beta_modifier

    seir_df = None
    for day in range(days):
        # print(f"day: {day}")
        seir_params_today = SEIRHCDParams(
            population=params.population,
            beta=params.beta,
            sigma=params.sigma,
            gamma=params.gamma,
            gamma_h=params.gamma_h,
            gamma_c=params.gamma_c,
            mu_c=params.mu_c,
            alpha_h=params.alpha_h,
            alpha_c=params.alpha_c,
            p_hosp=params.p_hosp,
            p_icu=params.p_icu,
            p_death=params.p_death,
            initial_exposed=params.initial_exposed,
            initial_infectious=params.initial_infectious
        )

        seir_df = simulate_seir_hcd(params=seir_params_today, days=1, start_day=day+1, beta_time_fn=beta_time_fn, data=seir_df)

        expected_hosp = seir_df["new_hospitalizations"].iloc[-1]
        expected_icu = seir_df["new_icu"].iloc[-1]

        # Создаем запись пациентов на день
        events = sample_arrivals(expected_hosp, expected_icu, rng)
        for ev in events:
            pid += 1
            los = rng.gamma(shape=2.0, scale=5.0) if ev["severity"] == "icu" else rng.gamma(shape=2.0, scale=3.0)
            p = Patient(id=pid, absolute_time=day + ev["time_frac"], severity=ev["severity"], los=max(1.0, los))
            p = des.attempt_admit(p, max_wait=0.5)

        metric_day = {}
        for h in des.hospitals:
            hospital_metrics = h.daily_metrics(day=day)
            for key, value in hospital_metrics.items():
                if key == "day":
                    metric_day[key] = value
                elif key in metric_day:
                    metric_day[key] += value  # суммируем
                else:
                    metric_day[key] = value  # создаем новую запись

        seir_df.loc[seir_df.index[-1], "H"] += metric_day["admitted_hosp"]
        seir_df.loc[seir_df.index[-1], "C"] += metric_day["admitted_icu"]
        seir_df.loc[seir_df.index[-1], "D"] += metric_day["deaths"]

        # 1) Смертность сокращает численность населения (вычитается из N)
        if metric_day["deaths"] > 0:
            params.population = max(0, params.population - metric_day["deaths"])

        # 2) Высокий уровень отторжения => увеличить бета-модификатор (поведенческую реакцию)
        overload = metric_day["rejected"] / max(1.0, max(1.0, expected_hosp + expected_icu))
        if overload < 0.1:
            beta_modifier *= max(0.6, 1.0 - 0.25 * min(1.0, overload))
        else:
            beta_modifier += (1.0 - beta_modifier) * 0.02

        # Логирование
        logs["infection"].append(seir_df["new_infected"].iloc[-1])
        logs["hosp_expected"].append(expected_hosp)
        logs["icu_expected"].append(expected_icu)
        logs["deaths_expected"].append(seir_df["new_deaths"].iloc[-1])
        logs["population"].append(params.population)

        for k, v in metric_day.items():
            if k in logs:
                logs[k].append(v)
            else:
                logs[k] = [v]

    logs = pd.DataFrame(logs)
    return logs, des