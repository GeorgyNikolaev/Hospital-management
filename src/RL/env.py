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

    def build_obs(self, metrics_: dict):
        """
        Привязка к твоим метрикам: используем поля, которые реально заполняет Hospital.
        Возвращает нормализованный вектор [beds, occupied_beds, icu, occupied_icu, admitted, rejected]
        """
        metrics = metrics_.copy()
        # Жёсткие масштабирующие множители (подберите под диапазоны вашей симуляции)
        scales = {
            "day": 100,
            "infection": 100,
            "expected_hosp_1_day": 15,
            "expected_hosp_3_day": 15,
            "expected_hosp_7_day": 15,
            "expected_icu_1_day": 5,
            "expected_icu_3_day": 5,
            "expected_icu_7_day": 5,
            "deaths_expected": 10,
            "beds": 40,
            "icu": 20,
            "admitted": 30,
            "admitted_hosp": 20,
            "admitted_icu": 10,
            "rejected": 30,
            "rejected_hosp": 20,
            "rejected_icu": 10,
            "deaths": 30,
            "deaths_hosp": 20,
            "deaths_icu": 10,
            "reserve_beds": 20,
            "occupied_beds": 20,
            "reserve_icu": 10,
            "occupied_icu": 10,
            "budget": 100_000_000,
            "expenses": 1_000_000,
        }

        x = []
        for key, val in metrics.items():
            sc = scales.get(key, 1)
            x.append(val / sc)

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

    # def compute_reward(self, metrics, action):
    #     """
    #     Функция награды с нормализованными относительными значениями.
    #     Все компоненты приведены к диапазону [0, 1] для корректного сравнения.
    #     """
    #     reward = 0.0
    #     EPS = 1e-5  # Защита от деления на ноль
    #
    #     # 1. Критические штрафы (нормализованы на общий поток пациентов)
    #     total_patients = max(metrics['admitted'] + metrics['rejected'], 1)
    #
    #     death_ratio = metrics['deaths'] / total_patients
    #     reject_ratio = metrics['rejected'] / total_patients
    #
    #     reward -= death_ratio * 20
    #     reward -= reject_ratio * 20
    #
    #     # 2. Подсчет загруженности
    #     target_occupancy = 0.70
    #     bed_occupancy = (metrics['occupied_beds'] + EPS) / (metrics['beds'] + EPS)
    #     icu_occupancy = (metrics['occupied_icu'] + EPS) / (metrics['icu'] + EPS)
    #
    #     def calc_resource_reward(a: float, b: float, x: float):
    #         return a * math.exp(b * (x - target_occupancy) ** 2) - (a - 2 + 0.3)
    #
    #     bed_reward = calc_resource_reward(3, -20, bed_occupancy) if bed_occupancy < target_occupancy else calc_resource_reward(3, -30, bed_occupancy)
    #     icu_reward = calc_resource_reward(3, -20, icu_occupancy) if icu_occupancy < target_occupancy else calc_resource_reward(3, -30, icu_occupancy)
    #     # print(bed_reward, bed_occupancy, icu_reward, icu_occupancy)
    #     # print(metrics['day'], death_ratio, reject_ratio, bed_reward, icu_reward)
    #     reward += (bed_reward + icu_reward) * 3
    #
    #     # 4. Бюджетная эффективность (нормализована на критический порог)
    #     CRITICAL_BUDGET = 5_000_000
    #     if metrics['budget'] < CRITICAL_BUDGET:
    #         budget_ratio = (metrics['budget'] - CRITICAL_BUDGET) / CRITICAL_BUDGET
    #         reward += budget_ratio * 0.5  # Штраф за критический бюджет (сильнее)
    #
    #
    #     # print("death_ratio", death_ratio, " reject_ratio", reject_ratio, " bed_reward", bed_reward, " icu_reward", icu_reward)
    #
    #     # 5. Корректировка за действия (шкала [-0.2, +0.2])
    #     if action == 9:  # Срочное выделение бюджета
    #         if metrics['budget'] < CRITICAL_BUDGET:
    #             reward -= 0.2  # Обоснованное действие
    #
    #         else:
    #             reward -= 10  # Неоправданный расход
    #
    #     # print(reward)
    #
    #     return reward

    def compute_reward(self, metrics, action):
        reward = 0.0

        # 1. Прямые штрафы за плохие исходы
        reward -= metrics['deaths'] * 2.0
        reward -= metrics['rejected'] * 1.5

        # 2. Непрерывный штраф за расходы (покупка больше не "бесплатная")
        reward -= metrics['expenses'] * 0.0002

        # 3. Управление загруженностью (целевой диапазон 0.65 - 0.85)
        bed_occ = metrics['occupied_beds'] / max(metrics['beds'], 1.0)
        icu_occ = metrics['occupied_icu'] / max(metrics['icu'], 1.0)

        def occ_score(occ):
            if 0.65 <= occ <= 0.85:
                return 1.0  # Идеально
            elif occ < 0.65:
                return -(0.65 - occ) * 4.0  # Штраф за простой ресурсов
            else:
                return -(occ - 0.85) * 6.0  # Штраф за перегрузку

        reward += occ_score(bed_occ) + occ_score(icu_occ)

        # 4. Поощрение консервации/освобождения при простое
        if action in [5, 6, 7, 8]:  # Действия консервации/освобождения
            if bed_occ < 0.6 or icu_occ < 0.6:
                reward += 0.6  # +награда за "умное" освобождение

        # 5. Штраф за панические запросы бюджета
        if action == 9:
            reward -= 0.5 if metrics['budget'] > 20_000_000 else 0.0

        # Клиппинг награды для стабильности DQN
        return float(np.clip(reward, -10.0, 5.0))


