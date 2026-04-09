import math

import numpy as np


class HospitalEnv:
    """
    Среда для одной больницы.
    Данные подаются каждый день извне.
    """
    def __init__(self, hospital_id, obs_size=26):
        """
        obs_size — размер вектора состояния.
        """
        self.hid = hospital_id
        self.obs_size = obs_size
        self.last_obs = None
        self.day = 0

    def build_obs(self, metrics_: dict):
        metrics = metrics_.copy()

        # 🔑 Добавляем тренд (1 день vs 7 дней). Критически важно для понимания фазы эпидемии
        hosp_1 = metrics.get("expected_hosp_1_day", 0)
        hosp_7 = metrics.get("expected_hosp_7_day", 1e-5)
        icu_1 = metrics.get("expected_icu_1_day", 0)
        icu_7 = metrics.get("expected_icu_7_day", 1e-5)

        # Тренд: >0 растёт, <0 падает, нормализуем в [-1, 1]
        # print(hosp_1, hosp_7, icu_1, icu_7)
        metrics["hosp_trend"] = np.clip((hosp_7 - hosp_1) / max(hosp_7, 1.0), -1.0, 1.0)
        metrics["icu_trend"] = np.clip((icu_7 - icu_1) / max(icu_7, 1.0), -1.0, 1.0)

        scales = {
            "day": 100, "infection": 100,
            "expected_hosp_1_day": 15, "expected_hosp_3_day": 15, "expected_hosp_7_day": 15,
            "expected_icu_1_day": 5, "expected_icu_3_day": 5, "expected_icu_7_day": 5,
            "deaths_expected": 10,
            "beds": 40, "icu": 20,
            "admitted": 30, "admitted_hosp": 20, "admitted_icu": 10,
            "rejected": 30, "rejected_hosp": 20, "rejected_icu": 10,
            "deaths": 30, "deaths_hosp": 20, "deaths_icu": 10,
            "reserve_beds": 20, "occupied_beds": 20,
            "reserve_icu": 10, "occupied_icu": 10,
            "budget": 100_000_000, "expenses": 1_000_000,
            "hosp_trend": 1.0, "icu_trend": 1.0  # Новые признаки
        }
        x = [metrics[k] / scales.get(k, 1) for k in metrics.keys()]
        x = np.array(x, dtype=np.float32) + float(1e-6)
        # print(x.tolist())
        return x

    def reset(self, metrics):
        """
        Начало эпизода. Metrics — метрики на день 0.
        """
        self.day = 0
        self.last_obs = self.build_obs(metrics)
        return self.last_obs

    def step(self, metrics, action):
        """
        metrics — фактические метрики больницы после применения действия агента.
        action — решение агента

        Возвращает:
        next_obs, reward
        """
        next_obs = self.build_obs(metrics)
        reward = self.compute_reward(metrics, action)

        self.last_obs = next_obs
        self.day += 1

        return next_obs, reward

    def compute_reward(self, metrics, action):
        """
        Функция награды с нормализованными относительными значениями.
        Все компоненты приведены к диапазону [0, 1] для корректного сравнения.
        """
        reward = 0.0
        EPS = 1e-5  # Защита от деления на ноль

        # 1. Критические штрафы (нормализованы на общий поток пациентов)
        total_patients = max(metrics['admitted'] + metrics['rejected'], 1)

        death_ratio = metrics['deaths'] / total_patients
        reject_ratio = metrics['rejected'] / total_patients

        reward -= death_ratio * 6
        reward -= reject_ratio * 3

        # 2. Подсчет загруженности
        target_occupancy = 0.85
        bed_occupancy = (metrics['occupied_beds'] + EPS) / (metrics['beds'] + EPS)
        icu_occupancy = (metrics['occupied_icu'] + EPS) / (metrics['icu'] + EPS)

        def calc_resource_reward(a: float, b: float, x: float):
            return a * math.exp(b * (x - target_occupancy) ** 2) - (a - 2 + 0.3)

        bed_reward = calc_resource_reward(3, -20, bed_occupancy) if bed_occupancy < target_occupancy else calc_resource_reward(3, -30, bed_occupancy)
        icu_reward = calc_resource_reward(3, -20, icu_occupancy) if icu_occupancy < target_occupancy else calc_resource_reward(3, -30, icu_occupancy)
        reward += (bed_reward + icu_reward) * 1
        # print(metrics['day'], death_ratio, reject_ratio, bed_reward, icu_reward)

        # 4. Бюджетная эффективность (нормализована на критический порог)
        CRITICAL_BUDGET = 5_000_000
        if metrics['budget'] < CRITICAL_BUDGET:
            budget_ratio = (metrics['budget'] - CRITICAL_BUDGET) / CRITICAL_BUDGET
            reward += budget_ratio * 0.5  # Штраф за критический бюджет (сильнее)


        # print("death_ratio", death_ratio, " reject_ratio", reject_ratio, " bed_reward", bed_reward, " icu_reward", icu_reward)

        trend_h = metrics.get("hosp_trend", 0)
        trend_i = metrics.get("icu_trend", 0)

        if action in [1, 2] and trend_h < -0.1 or action in [5, 6] and trend_h > 0.1 or \
           action in [3, 4] and trend_i < -0.1 or action in [7, 8] and trend_i > 0.1:  # Эпидемия явно идёт на спад
            reward += 0.5  # Сильное поощрение за "умное" сокращение мощностей

        # 5. Корректировка за действия (шкала [-0.2, +0.2])
        if action == 9:  # Срочное выделение бюджета
            if metrics['budget'] < CRITICAL_BUDGET:
                reward -= 0.15  # Обоснованное действие

            else:
                reward -= 5  # Неоправданный расход

        return reward

