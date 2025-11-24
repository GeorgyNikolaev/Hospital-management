import numpy as np


class HospitalEnv:
    """
    Среда для одной больницы.
    Данные подаются каждый день извне.
    """
    def __init__(self, hospital_id, obs_size=6):
        """
        obs_size — размер вектора состояния.
        hospital_id — просто идентификатор (удобно для логов).
        """
        self.hid = hospital_id
        self.obs_size = obs_size
        self.last_obs = None
        self.day = 0

    def build_obs(self, metrics):
        """
        metrics: словарь с метриками больницы, например:
        {
            "beds_total": ...,
            "beds_occupied": ...,
            "icu_total": ...,
            "icu_occupied": ...,
            "admitted": ...,
            "rejected": ...
        }
        Возвращает вектор состояния.
        """
        x = np.array([
            metrics["beds_total"],
            metrics["beds_occupied"],
            metrics["icu_total"],
            metrics["icu_occupied"],
            metrics["admitted"],
            metrics["rejected"],
        ], dtype=np.float32)

        # нормализация простейшая
        return x / (np.max([1.0, np.max(x)]) + 1e-9)

    def reset(self, metrics):
        """
        Начало эпизода. metrics — метрики на день 0.
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
        Простая схема награды:
        - штраф за отказ (rejected)
        - штраф за перегрузку
        - штраф за высокую занятость ICU
        - небольшая награда за экономию коек, если действие было "сократить"
        """
        rejected = metrics["rejected"]
        occ = metrics["beds_occupied"] / max(1, metrics["beds_total"])
        icu_occ = metrics["icu_occupied"] / max(1, metrics["icu_total"])

        reward = -2.0 * rejected - 3.0 * max(0, occ - 0.9) - 4.0 * max(0, icu_occ - 0.8)

        # если действие = 1 (например, увеличить ресурсы) → штраф (дорого)
        if action == 1:
            reward -= 0.5
        # действие = 0 (ничего не делать) → ок
        # действие = 2 (уменьшить ресурсы) → маленький бонус
        if action == 2:
            reward += 0.2

        return reward
