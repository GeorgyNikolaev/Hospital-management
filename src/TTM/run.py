"""Threshold-Triggered Management - управление больницами через пороговые значения"""
import pandas as pd

from typing import List
from TTM.agent import Agent
from src.core.config import settings
from src.core.models import Hospital, SEIRHCDParams, Patient
from src.des.des_model import DES
from src.sd.seir_model import simulate_seir_hcd
from src.utils.utils import sample_arrivals, change_beta_modifier


def run_ttm(
        hospitals_cfg: List[Hospital],
        init_params: SEIRHCDParams,
        days: int,
        rng,
):
    """Функция для запуска симуляции"""
    des = DES(hospitals_cfg, rng_seed=settings.RANDOM_SEED)
    params = init_params

    agent = Agent()

    logs = {"day": [],
            "infection": [],
            "hosp_expected": [],
            "icu_expected": [],
            "deaths_expected": [],
            "population": [],
            "actions": [],
            "beds": [],
            "icu": []
            }

    pid = 0
    beta_modifier = 1.0

    def beta_time_fn(ti, beta):
        """Корректировка beta параметра"""
        return beta * beta_modifier

    seir_df = None

    # === INITIAL STATE COLLECTION ===
    initial_metrics = []
    for h in des.hospitals:
        h.beds = 5
        h.icu = 2

        metrics = h.daily_metrics()
        initial_metrics.append(metrics)

    # init environments
    obs_list = initial_metrics

    expected_hosp = 0
    expected_icu = 0

    for day in range(days):
        # print(f"day: {day}")
        actions = []
        action_masks = []

        if day == 0:
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
            seir_df = simulate_seir_hcd(params=seir_params_today, days=7, start_day=day+1, beta_time_fn=beta_time_fn, data=seir_df)

            expected_hosp = seir_df["new_hospitalizations"].iloc[-7]
            expected_icu = seir_df["new_icu"].iloc[-7]

            expected_hosp_3 = seir_df["new_hospitalizations"].iloc[-5]
            expected_icu_3 = seir_df["new_icu"].iloc[-5]
            expected_hosp_7 = seir_df["new_hospitalizations"].iloc[-1]
            expected_icu_7 = seir_df["new_icu"].iloc[-1]

            for hid in range(len(des.hospitals)):
                des.hospitals[hid].save_daily_metrics(day=day)
                obs_list[hid]["expected_hosp_1_day"] = expected_hosp
                obs_list[hid]["expected_icu_1_day"] = expected_icu
                obs_list[hid]["expected_hosp_3_day"] = expected_hosp_3
                obs_list[hid]["expected_icu_3_day"] = expected_icu_3
                obs_list[hid]["expected_hosp_7_day"] = expected_hosp_7
                obs_list[hid]["expected_icu_7_day"] = expected_icu_7

        for hid in range(len(des.hospitals)):
            des.hospitals[hid].save_daily_metrics(day=day)

            # вычисляем маску действий для больницы (текущий бюджет и резервы учтены внутри Hospital)
            mask = des.hospitals[hid].get_action_mask()
            action_masks.append(mask)

            action = agent.select_action(obs_list[hid], mask)  # передаём маску
            actions.append(action)

            # применяем действие к больнице
            des.hospitals[hid].apply_action(action)

        logs["actions"].append(actions)

        # Создаем запись пациентов на день
        events = sample_arrivals(expected_hosp, expected_icu, rng)
        for ev in events:
            pid += 1
            los = rng.gamma(shape=2.0, scale=5.0) if ev["severity"] == "icu" else rng.gamma(shape=2.0, scale=3.0)
            p = Patient(id=pid, absolute_time=day + ev["time_frac"], severity=ev["severity"], los=max(1.0, los))
            des.attempt_admit(p, max_wait=0.5)

        metric_day = {}
        for h in des.hospitals:
            hospital_metrics = h.daily_metrics(day=day)
            for key, value in hospital_metrics.items():
                if key == "day":
                    metric_day[key] = day
                elif key in metric_day:
                    metric_day[key] += value  # суммируем
                else:
                    metric_day[key] = value  # создаем новую запись

        # 1) Смертность сокращает численность населения (вычитается из N)
        if metric_day["deaths"] > 0:
            params.population = max(1000, params.population - metric_day["deaths"])

        # 2) Высокий уровень отторжения => увеличить бета-модификатор (поведенческую реакцию)
        overload = metric_day["rejected"] / max(1.0, len(events))
        beta_modifier = change_beta_modifier(beta_modifier, overload)

        # Логирование
        logs["infection"].append(seir_df["new_infected"].iloc[-7])
        logs["hosp_expected"].append(expected_hosp)
        logs["icu_expected"].append(expected_icu)
        logs["deaths_expected"].append(seir_df["new_deaths"].iloc[-7])
        logs["population"].append(params.population)

        for k, v in metric_day.items():
            logs.setdefault(k, []).append(v)

        new_obs_list = []

        seir_params_tomorrow = SEIRHCDParams(
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
        seir_df = simulate_seir_hcd(params=seir_params_tomorrow, days=7, start_day=day + 2, beta_time_fn=beta_time_fn,
                                    data=seir_df)

        expected_hosp = seir_df["new_hospitalizations"].iloc[-7]
        expected_icu = seir_df["new_icu"].iloc[-7]

        expected_hosp_3 = seir_df["new_hospitalizations"].iloc[-5]
        expected_icu_3 = seir_df["new_icu"].iloc[-5]
        expected_hosp_7 = seir_df["new_hospitalizations"].iloc[-1]
        expected_icu_7 = seir_df["new_icu"].iloc[-1]


        for hid, h in enumerate(des.hospitals):
            metrics = h.daily_metrics(day=day)
            metrics["expected_hosp_1_day"] = expected_hosp / 3
            metrics["expected_icu_1_day"] = expected_icu / 3
            metrics["expected_hosp_3_day"] = expected_hosp_3 / 3
            metrics["expected_icu_3_day"] = expected_icu_3 / 3
            metrics["expected_hosp_7_day"] = expected_hosp_7 / 3
            metrics["expected_icu_7_day"] = expected_icu_7 / 3

            new_obs_list.append(metrics)

        obs_list = new_obs_list

    return pd.DataFrame(logs), des
