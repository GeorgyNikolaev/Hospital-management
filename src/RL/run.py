"""Обучение"""
import json
from typing import List

import numpy as np
import pandas as pd

from src.core.config import settings
from src.core.models import Hospital, SEIRHCDParams, Patient
from src.des.des_model import DES
from src.sd.seir_model import simulate_seir_hcd
from src.utils.utils import sample_arrivals


def run_with_rl(
        hospitals_cfg: List[Hospital],
        init_params: SEIRHCDParams,
        days: int,
        rng,
        agents,          # список объектов HospitalAgent, один на больницу
        envs,            # список HospitalEnv
        is_train: bool=True
):
    des = DES(hospitals_cfg, rng_seed=settings.RANDOM_SEED)
    params = init_params

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
    obs_list = [envs[i].reset(initial_metrics[i]) for i in range(len(envs))]

    # добавим аккумулятор reward по агентам
    n_agents = len(agents)
    episode_rewards = [0.0 for _ in range(n_agents)]

    expected_hosp = 0
    expected_icu = 0

    expected_hosp_5 = 0
    expected_icu_5 = 0
    expected_hosp_15 = 0
    expected_icu_15 = 0

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
            seir_df = simulate_seir_hcd(params=seir_params_today, days=15, start_day=day+1, beta_time_fn=beta_time_fn, data=seir_df)

            expected_hosp = seir_df["new_hospitalizations"].iloc[-15]
            expected_icu = seir_df["new_icu"].iloc[-15]

            expected_hosp_5 = seir_df["new_hospitalizations"].iloc[-5]
            expected_icu_5 = seir_df["new_icu"].iloc[-5]
            expected_hosp_15 = seir_df["new_hospitalizations"].iloc[-1]
            expected_icu_15 = seir_df["new_icu"].iloc[-1]

            for hid, agent in enumerate(agents):
                des.hospitals[hid].save_daily_metrics(day=day)
                obs_list[hid] = np.append(obs_list[hid], expected_hosp)
                obs_list[hid] = np.append(obs_list[hid], expected_icu)
                obs_list[hid] = np.append(obs_list[hid], expected_hosp_5)
                obs_list[hid] = np.append(obs_list[hid], expected_icu_5)
                obs_list[hid] = np.append(obs_list[hid], expected_hosp_15)
                obs_list[hid] = np.append(obs_list[hid], expected_icu_15)

        for hid, agent in enumerate(agents):
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
            p = des.attempt_admit(p, max_wait=0.5)

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
        # print(json.dumps(metric_day, indent=2, ensure_ascii=False))
        # print(metric_day["expenses"])

        # 1) Смертность сокращает численность населения (вычитается из N)
        if metric_day["deaths"] > 0:
            params.population = max(1000, params.population - metric_day["deaths"])

        # 2) Высокий уровень отторжения => увеличить бета-модификатор (поведенческую реакцию)
        overload = metric_day["rejected"] / max(1.0, len(events))
        # --- параметры регулирования ---
        OVERLOAD_LOW = 0.05  # система чувствует себя комфортно
        OVERLOAD_HIGH = 0.20  # начинается перегрузка

        BETA_MIN = 0.4
        BETA_MAX = 1.6

        UP_RATE = 0.03  # скорость ослабления ограничений
        DOWN_RATE = 0.08  # скорость ужесточения (всегда быстрее!)

        # --- регулирование ---
        if overload < OVERLOAD_LOW:
            # система справляется → ослабляем ограничения → beta ↑
            beta_modifier += UP_RATE * (1.0 - overload / OVERLOAD_LOW)

        elif overload > OVERLOAD_HIGH:
            # перегрузка → ужесточаем ограничения → beta ↓
            beta_modifier -= DOWN_RATE * (overload - OVERLOAD_HIGH) / (1.0 - OVERLOAD_HIGH)

        else:
            # нейтральная зона → мягко возвращаем к 1.0
            beta_modifier += 0.02 * (1.0 - beta_modifier)

        # --- жёсткие границы ---
        beta_modifier = max(BETA_MIN, min(BETA_MAX, beta_modifier))

        # print(metric_day["expenses"])

        # Логирование
        logs["infection"].append(seir_df["new_infected"].iloc[-15])
        logs["hosp_expected"].append(expected_hosp)
        logs["icu_expected"].append(expected_icu)
        logs["deaths_expected"].append(seir_df["new_deaths"].iloc[-15])
        logs["population"].append(params.population)

        for k, v in metric_day.items():
            if k in logs:
                logs[k].append(v)
            else:
                logs[k] = [v]


        new_obs_list = []
        rewards = []

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
        seir_df = simulate_seir_hcd(params=seir_params_tomorrow, days=15, start_day=day + 2, beta_time_fn=beta_time_fn,
                                    data=seir_df)

        expected_hosp = seir_df["new_hospitalizations"].iloc[-15]
        expected_icu = seir_df["new_icu"].iloc[-15]

        expected_hosp_5 = seir_df["new_hospitalizations"].iloc[-5]
        expected_icu_5 = seir_df["new_icu"].iloc[-5]
        expected_hosp_15 = seir_df["new_hospitalizations"].iloc[-1]
        expected_icu_15 = seir_df["new_icu"].iloc[-1]


        for hid, h in enumerate(des.hospitals):
            metrics = h.daily_metrics(day=day)
            metrics["expected_hosp_1_day"] = expected_hosp / 3
            metrics["expected_icu_1_day"] = expected_icu / 3
            metrics["expected_hosp_5_day"] = expected_hosp_5 / 3
            metrics["expected_icu_5_day"] = expected_icu_5 / 3
            metrics["expected_hosp_15_day"] = expected_hosp_15 / 3
            metrics["expected_icu_15_day"] = expected_icu_15 / 3

            next_obs, reward = envs[hid].step(metrics, actions[hid])

            # вычисляем маску действий для next state (важно: budget/резервы уже обновлены после apply_action)
            next_mask = des.hospitals[hid].get_action_mask()

            new_obs_list.append(next_obs)
            rewards.append(reward)

            # сохраняем опыт с маской состояния и маской next
            agents[hid].store(obs_list[hid], actions[hid], reward, next_obs, action_masks[hid], next_mask)

            # аккумулируем reward для эпизода
            episode_rewards[hid] += float(reward)

        if is_train:
            for agent in agents:
                agent.train_step()

            if day % 20 == 0:
                for agent in agents:
                    agent.update_target()

        # обновляем состояния
        obs_list = new_obs_list

    return pd.DataFrame(logs), des, agents, episode_rewards
