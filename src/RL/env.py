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
        reward -= (death_ratio * 0.7) + (reject_ratio * 0.3)  # 70% веса на смерти, 30% на отказы

        # 2. Штрафы за нехватку ресурсов (нормализованы на прогноз спроса)
        hosp_shortage_weights = {1: 0.6, 5: 0.3, 15: 0.1}  # Веса для разных горизонтов
        icu_shortage_weights = {1: 0.7, 5: 0.2, 15: 0.1}  # ИВЛ важнее в краткосрочной перспективе

        for days in [1, 5, 15]:
            # Прогнозируемый спрос
            hosp_demand = max(metrics[f'expected_hosp_{days}_day'], 1)
            icu_demand = max(metrics[f'expected_icu_{days}_day'], 1)

            # Текущие ресурсы
            available_beds = metrics['beds'] + metrics['reserve_beds']
            available_icu = metrics['icu'] + metrics['reserve_icu']

            # Относительный дефицит (0 = нет дефицита, 1 = полный дефицит)
            hosp_shortage_ratio = max(0, (hosp_demand - available_beds) / hosp_demand)
            icu_shortage_ratio = max(0, (icu_demand - available_icu) / icu_demand)

            # Применяем веса горизонтов
            reward -= hosp_shortage_ratio * hosp_shortage_weights[days] * 0.4  # 40% от общего веса
            reward -= icu_shortage_ratio * icu_shortage_weights[days] * 0.6  # 60% от общего веса (ИВЛ критичнее)

        # 3. Штрафы за избыток ресурсов (нормализованы на прогноз)
        max_hosp_demand = max(1, metrics['expected_hosp_15_day'])
        max_icu_demand = max(1, metrics['expected_icu_15_day'])

        # Избыток >30% от прогноза
        beds_ratio = (metrics['beds'] + metrics['reserve_beds']) / max_hosp_demand
        icu_ratio = (metrics['icu'] + metrics['reserve_icu']) / max_icu_demand

        if beds_ratio > 1.3:
            excess_beds_ratio = min((beds_ratio - 1.3) / 0.7, 1.0)  # Нормализуем до [0,1]
            reward -= excess_beds_ratio * 0.05  # Мягкий штраф

        if icu_ratio > 1.3:
            excess_icu_ratio = min((icu_ratio - 1.3) / 0.7, 1.0)
            reward -= excess_icu_ratio * 0.1  # Штраф сильнее для ИВЛ

        # 4. Бюджетная эффективность (нормализована на критический порог)
        CRITICAL_BUDGET = 2000000
        expense_ratio = metrics['expenses'] / (CRITICAL_BUDGET + EPS)
        budget_ratio = max(0, (CRITICAL_BUDGET - metrics['budget']) / CRITICAL_BUDGET)

        reward -= expense_ratio * 0.03  # Штраф за расходы
        reward -= budget_ratio * 0.07  # Штраф за критический бюджет (сильнее)

        # 5. Корректировка за действия (шкала [-0.2, +0.2])
        if action == 9:  # Срочное выделение бюджета
            if metrics['budget'] < CRITICAL_BUDGET * 1.5:
                reward += 0.15  # Обоснованное действие
            else:
                reward -= 0.1  # Неоправданный расход

        if action in [5, 6, 7, 8]:  # Консервация ресурсов
            current_bed_deficit = max(0, metrics['expected_hosp_1_day'] - (metrics['beds'] + metrics['reserve_beds']))
            current_icu_deficit = max(0, metrics['expected_icu_1_day'] - (metrics['icu'] + metrics['reserve_icu']))
            if current_bed_deficit > 0 or current_icu_deficit > 0:
                reward -= 0.2  # Критическая ошибка

        if action in [1, 2, 3, 4]:  # Покупка ресурсов
            future_bed_deficit = max(0, metrics['expected_hosp_5_day'] - (metrics['beds'] + metrics['reserve_beds']))
            future_icu_deficit = max(0, metrics['expected_icu_5_day'] - (metrics['icu'] + metrics['reserve_icu']))
            if future_bed_deficit > 0 or future_icu_deficit > 0:
                reward -= 0.1  # Недостаточная закупка

        return reward


