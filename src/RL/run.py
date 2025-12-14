"""Обучение"""
import json
from typing import List

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
        metrics = h.daily_metrics()
        initial_metrics.append(metrics)

    # init environments
    obs_list = [envs[i].reset(initial_metrics[i]) for i in range(len(envs))]

    # добавим аккумулятор reward по агентам
    n_agents = len(agents)
    episode_rewards = [0.0 for _ in range(n_agents)]

    for day in range(days):
        # print(f"day: {day}")
        actions = []
        action_masks = []

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
                    metric_day[key] = day
                elif key in metric_day:
                    metric_day[key] += value  # суммируем
                else:
                    metric_day[key] = value  # создаем новую запись
        # print(json.dumps(metric_day, indent=2, ensure_ascii=False))

        # 1) Смертность сокращает численность населения (вычитается из N)
        if metric_day["deaths"] > 0:
            params.population = max(1000, params.population - metric_day["deaths"])

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


        new_obs_list = []
        rewards = []

        for hid, h in enumerate(des.hospitals):
            metrics = h.daily_metrics(day=day)
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

    return logs, des, agents, episode_rewards
