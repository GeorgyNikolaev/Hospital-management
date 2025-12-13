import numpy as np


class HospitalEnv:
    """
    Среда для одной больницы.
    Данные подаются каждый день извне.
    """
    def __init__(self, hospital_id, obs_size=6):
        """
        obs_size — размер вектора состояния.
        """
        self.hid = hospital_id
        self.obs_size = obs_size
        self.last_obs = None
        self.day = 0

    def build_obs(self, metrics: dict):
        """
        Привязка к твоим метрикам: используем поля, которые реально заполняет Hospital.
        Возвращает нормализованный вектор [beds, occupied_beds, icu, occupied_icu, admitted, rejected]
        """
        x = np.array(list(metrics.values()), dtype=np.float32)

        # простая нормализация по максимуму в векторе (защита от деления на 0)
        denom = max(1.0, float(np.max(x)))
        return x / (denom + 1e-9)

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
        Сбалансированная функция награды для среды управления больницей.
        Масштабы нормированы, штрафы сглажены, добавлен ICU rejected.
        """

        reward = 0.0

        # Масштабы
        W_REJECT = -8.0  # отказ = плохо, но не фатально
        W_REJECT_ICU = -12.0  # отказ в ICU — хуже
        W_DEATH = -15.0  # смерть — главный штраф
        W_UTIL = 5.0  # хорошее использование ресурсов
        W_ACTION = 0.2  # штраф за дорогое действие
        W_EXCESS = -4.0  # избыточные ресурсы

        admitted = metrics.get("admitted", 0)
        admitted_icu = metrics.get("admitted_icu", 0)

        # ----------------------------- #
        # 1. Штраф за отказы
        # ----------------------------- #
        if admitted > 0:
            reject_rate = metrics["rejected"] / admitted
            reward += reject_rate * W_REJECT

        if admitted_icu > 0:
            reject_rate_icu = metrics["rejected_icu"] / admitted_icu
            reward += reject_rate_icu * W_REJECT_ICU

        # ----------------------------- #
        # 2. Штраф за смерти
        # ----------------------------- #
        if admitted > 0:
            death_rate = metrics["deaths"] / admitted
            reward += death_rate * W_DEATH

        if admitted_icu > 0:
            death_rate_icu = metrics["deaths_icu"] / admitted_icu
            reward += death_rate_icu * W_DEATH * 1.3

        # ----------------------------- #
        # 3. Награда/штраф за использование ресурсов
        # ----------------------------- #
        def util_reward(util, ideal_low, ideal_high, weight):
            """
            Плавная функция:
            - в целевом диапазоне → +weight
            - если недозагрузка — близко к 0
            - если перегруз — штраф пропорционально перегрузу
            """
            if ideal_low <= util <= ideal_high:
                return weight
            elif util < ideal_low:
                return weight * (util / ideal_low)  # 0 → половина → целевое
            else:  # util > ideal_high
                overload = util - ideal_high
                return -weight * (1 + 4 * overload)  # штраф растёт быстро

        # обычные койки
        if metrics["beds"] > 0:
            util_beds = metrics["occupied_beds"] / metrics["beds"]
            reward += util_reward(util_beds, 0.7, 0.9, W_UTIL)

        # ICU
        if metrics["icu"] > 0:
            util_icu = metrics["occupied_icu"] / metrics["icu"]
            reward += util_reward(util_icu, 0.6, 0.85, W_UTIL * 1.5)

        # ----------------------------- #
        # 4. Стоимость действий
        # ----------------------------- #
        action_costs = {
            0: 0,
            1: -10,
            2: -45,
            3: -50,
            4: -225,
            5: 5,
            6: 15,
            7: 8,
            8: 25,
            9: -50,
        }

        reward += action_costs.get(action, 0) * W_ACTION

        # ----------------------------- #
        # 5. Штраф за избыточные ресурсы
        # ----------------------------- #
        if metrics["beds"] > 0:
            if metrics["occupied_beds"] / metrics["beds"] < 0.3:
                reward += W_EXCESS

        if metrics["icu"] > 0:
            if metrics["occupied_icu"] / metrics["icu"] < 0.2:
                reward += W_EXCESS * 1.5

        return float(reward)


