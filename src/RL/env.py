import math

import numpy as np
from mpmath.math2 import EPS


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

    def build_obs(self, metrics_: dict):
        """
        Привязка к твоим метрикам: используем поля, которые реально заполняет Hospital.
        Возвращает нормализованный вектор [beds, occupied_beds, icu, occupied_icu, admitted, rejected]
        """
        metrics = metrics_.copy()
        metrics["budget"] /= 50_000_000
        metrics["expenses"] /= 1_000_000
        x = np.array(list(metrics.values()), dtype=np.float32)

        # простая нормализация по максимуму в векторе (защита от деления на 0)
        x += 1
        denom = max(1.0, float(np.max(x)))
        x /= denom
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

        reward -= death_ratio * 3
        reward -= reject_ratio * 2

        # 2. Подсчет загруженности
        target_occupancy = 0.85
        bed_occupancy = metrics['occupied_beds'] / (metrics['beds'] + EPS)
        icu_occupancy = metrics['occupied_icu'] / (metrics['icu'] + EPS)

        bed_reward = math.exp(-10 * (bed_occupancy - target_occupancy) ** 2)
        icu_reward = math.exp(-10 * (icu_occupancy - target_occupancy) ** 2)
        reward += (bed_reward + icu_reward) * 0.5

        # 4. Бюджетная эффективность (нормализована на критический порог)
        CRITICAL_BUDGET = 5_000_000
        budget_ratio = (metrics['budget'] - CRITICAL_BUDGET) / CRITICAL_BUDGET
        reward -= budget_ratio * 0.01  # Штраф за критический бюджет (сильнее)

        # 5. Корректировка за действия (шкала [-0.2, +0.2])
        if action == 9:  # Срочное выделение бюджета
            if metrics['budget'] < CRITICAL_BUDGET:
                reward -= 0.15  # Обоснованное действие
            else:
                reward -= 1  # Неоправданный расход

        return reward


