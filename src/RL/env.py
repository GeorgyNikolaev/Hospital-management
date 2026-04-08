import math

import numpy as np


class HospitalEnv:
    """
    Среда для одной больницы.
    Данные подаются каждый день извне.
    """
    def __init__(self, hospital_id, obs_size=24):
        """
        obs_size — размер вектора состояния.
        """
        self.hid = hospital_id
        self.obs_size = obs_size
        self.last_obs = None
        self.day = 0

    # def build_obs(self, metrics_: dict):
    #     """
    #     Привязка к твоим метрикам: используем поля, которые реально заполняет Hospital.
    #     Возвращает нормализованный вектор [beds, occupied_beds, icu, occupied_icu, admitted, rejected]
    #     """
    #     metrics = metrics_.copy()
    #     metrics["budget"] /= 50_000_000
    #     metrics["expenses"] /= 1_000_000
    #     x = np.array(list(metrics.values()), dtype=np.float32)
    #
    #     # простая нормализация по максимуму в векторе (защита от деления на 0)
    #     x += 1
    #     denom = max(1.0, float(np.max(x)))
    #     x /= denom
    #     return x

    def build_obs(self, metrics_: dict):
        """
        Привязка к твоим метрикам: используем поля, которые реально заполняет Hospital.
        Возвращает нормализованный вектор [beds, occupied_beds, icu, occupied_icu, admitted, rejected]
        """
        metrics = metrics_.copy()
        # Жёсткие масштабирующие множители (подберите под диапазоны вашей симуляции)
        scales = {
            "day": 174,
            "infection": 5118.033885908577,
            "expected_hosp_1_day": 205,
            "expected_hosp_3_day": 205,
            "expected_hosp_7_day": 205,
            "expected_icu_1_day": 37,
            "expected_icu_3_day": 37,
            "expected_icu_7_day": 37,
            "deaths_expected": 22,
            "beds": 400,
            "icu": 130,
            "admitted": 69,
            "admitted_hosp": 57,
            "admitted_icu": 20,
            "rejected": 627,
            "rejected_hosp": 557,
            "rejected_icu": 118,
            "deaths": 122,
            "deaths_hosp": 108,
            "deaths_icu": 27,
            "reserve_beds": 45,
            "occupied_beds": 288,
            "reserve_icu": 131,
            "occupied_icu": 121,
            "budget": 300_000_000,
            "expenses": 38_000_000,
        }

        x = []
        for key in metrics:
            val = scales.get(key)
            if not val:
                raise Exception(f"PENIS {key}")
            x.append(val / scales[key])

        return np.array(x, dtype=np.float32)

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
        target_occupancy = 0.70
        bed_occupancy = (metrics['occupied_beds'] + EPS) / (metrics['beds'] + EPS)
        icu_occupancy = (metrics['occupied_icu'] + EPS) / (metrics['icu'] + EPS)

        def calc_resource_reward(a: float, b: float, x: float):
            return a * math.exp(b * (x - target_occupancy) ** 2) - (a - 1 + 0.3)

        bed_reward = calc_resource_reward(3, -5, bed_occupancy) if bed_occupancy < target_occupancy else calc_resource_reward(1, -100, bed_occupancy)
        icu_reward = calc_resource_reward(3, -5, icu_occupancy) if icu_occupancy < target_occupancy else calc_resource_reward(1, -100, icu_occupancy)
        # print(bed_reward, bed_occupancy, icu_reward, icu_occupancy)
        reward += (bed_reward + icu_reward) * 0.5

        # 4. Бюджетная эффективность (нормализована на критический порог)
        CRITICAL_BUDGET = 5_000_000
        if metrics['budget'] < CRITICAL_BUDGET:
            budget_ratio = (metrics['budget'] - CRITICAL_BUDGET) / CRITICAL_BUDGET
            reward += budget_ratio * 0.5  # Штраф за критический бюджет (сильнее)

        # print("death_ratio", death_ratio, " reject_ratio", reject_ratio, " bed_reward", bed_reward, " icu_reward", icu_reward)

        # 5. Корректировка за действия (шкала [-0.2, +0.2])
        if action == 9:  # Срочное выделение бюджета
            if metrics['budget'] < CRITICAL_BUDGET:
                reward -= 0.2  # Обоснованное действие
            else:
                reward -= 10  # Неоправданный расход

        return reward


