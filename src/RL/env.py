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

    def build_obs(self, metrics: dict):
        """
        Привязка к твоим метрикам: используем поля, которые реально заполняет Hospital.
        Возвращает нормализованный вектор [beds, occupied_beds, icu, occupied_icu, admitted, rejected]
        """
        metrics["budget"] /= 50_000_000
        metrics["expenses"] /= 10_000_000
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
        Функция награды для RL-агента управления ресурсами больницы.
        Цель: минимизировать смерти и отказы, эффективно распределяя ресурсы с учётом прогнозов.
        """
        reward = 0.0

        # 1. Критические штрафы (приоритет)
        DEATH_PENALTY = 1000  # Высокий штраф за смерть
        REJECT_PENALTY = 200  # Штраф за отказ в приёме
        reward -= metrics['deaths'] * DEATH_PENALTY
        reward -= metrics['rejected'] * REJECT_PENALTY

        # 2. Штрафы за нехватку ресурсов относительно прогнозов
        BED_SHORTAGE_PENALTY = 80  # Больничные койки
        ICU_SHORTAGE_PENALTY = 150  # Аппараты ИВЛ (выше из-за критичности)

        # Проверяем нехватку по горизонтам 1/5/15 дней с экспоненциальным весом близости
        for days in [1, 5, 15]:
            weight = 1.0 / (0.5 * days)  # Чем ближе срок - тем выше вес

            # Прогноз по обычным койкам
            hosp_shortage = max(0, metrics[f'expected_hosp_{days}_day'] - (metrics['beds'] + metrics['reserve_beds']))
            reward -= hosp_shortage * BED_SHORTAGE_PENALTY * weight

            # Прогноз по ИВЛ
            icu_shortage = max(0, metrics[f'expected_icu_{days}_day'] - (metrics['icu'] + metrics['reserve_icu']))
            reward -= icu_shortage * ICU_SHORTAGE_PENALTY * weight

        # 3. Штрафы за неэффективное использование ресурсов
        OVERSTOCK_PENALTY = 5  # Избыток коек
        ICU_OVERSTOCK_PENALTY = 15  # Избыток ИВЛ (дороже в содержании)

        # Рассчитываем максимальную прогнозируемую потребность за 15 дней
        max_hosp_demand = max(metrics['expected_hosp_1_day'], metrics['expected_hosp_5_day'],
                              metrics['expected_hosp_15_day'])
        max_icu_demand = max(metrics['expected_icu_1_day'], metrics['expected_icu_5_day'],
                             metrics['expected_icu_15_day'])

        # Штраф за избыток более чем на 30% от максимального прогноза
        if (metrics['beds'] + metrics['reserve_beds']) > max_hosp_demand * 1.3:
            excess_beds = (metrics['beds'] + metrics['reserve_beds']) - max_hosp_demand * 1.3
            reward -= excess_beds * OVERSTOCK_PENALTY

        if (metrics['icu'] + metrics['reserve_icu']) > max_icu_demand * 1.3:
            excess_icu = (metrics['icu'] + metrics['reserve_icu']) - max_icu_demand * 1.3
            reward -= excess_icu * ICU_OVERSTOCK_PENALTY

        # 4. Учёт бюджетной эффективности
        BUDGET_PENALTY = 0.005  # Штраф за каждый потраченный доллар
        CRITICAL_BUDGET_THRESHOLD = 2000000  # Критический уровень бюджета

        # Базовый штраф за расходы
        reward -= metrics['expenses'] * BUDGET_PENALTY

        # Дополнительный штраф при критически низком бюджете
        if metrics['budget'] < CRITICAL_BUDGET_THRESHOLD:
            reward -= (CRITICAL_BUDGET_THRESHOLD - metrics['budget']) * 0.02

        # 5. Корректировка за действия
        # 5.1. Срочное выделение бюджета (действие 9)
        if action == 9:
            if metrics['budget'] < CRITICAL_BUDGET_THRESHOLD * 1.5:
                reward += 300  # Бонус за спасение ситуации
            else:
                reward -= 100  # Штраф за неоправданное использование

        # 5.2. Консервация ресурсов (действия 5-8) при существующем дефиците
        if action in [5, 6, 7, 8]:  # Консервация коек/ИВЛ
            current_bed_deficit = max(0, metrics['expected_hosp_1_day'] - (metrics['beds'] + metrics['reserve_beds']))
            current_icu_deficit = max(0, metrics['expected_icu_1_day'] - (metrics['icu'] + metrics['reserve_icu']))

            if current_bed_deficit > 0 or current_icu_deficit > 0:
                reward -= 250  # Штраф за усугубление дефицита

        # 5.3. Покупка ресурсов (действия 1-4) без покрытия прогнозов
        if action in [1, 2, 3, 4]:
            # Проверяем, покрыли ли мы прогноз на 5 дней после покупки
            future_bed_deficit = max(0, metrics['expected_hosp_5_day'] - (metrics['beds'] + metrics['reserve_beds']))
            future_icu_deficit = max(0, metrics['expected_icu_5_day'] - (metrics['icu'] + metrics['reserve_icu']))

            if future_bed_deficit > 0 or future_icu_deficit > 0:
                reward -= 150  # Штраф за недостаточную закупку

        return reward


