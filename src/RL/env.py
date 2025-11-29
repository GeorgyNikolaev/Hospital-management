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
        Расчет награды для агента управления больницей
        Стратегия: баланс между качеством медицинской помощи и финансовой эффективностью
        """
        reward = 0

        # Базовые веса для разных компонент награды
        WEIGHT_ADMITTED = 2.0
        WEIGHT_REJECTED = -5.0
        WEIGHT_DEATHS = -10.0
        WEIGHT_RESOURCE_UTILIZATION = 1.0
        WEIGHT_BUDGET = 0.5
        WEIGHT_ACTION_COST = -0.1

        # 1. Награда за принятых пациентов (положительная)
        reward += metrics["admitted"] * WEIGHT_ADMITTED

        # 2. Штраф за отказов в госпитализации (сильно отрицательная)
        reward += metrics["rejected"] * WEIGHT_REJECTED

        # 3. Штраф за смерти (очень сильно отрицательная)
        reward += metrics["deaths"] * WEIGHT_DEATHS
        # Дополнительный штраф за смерти в ICU
        reward += metrics["deaths_icu"] * WEIGHT_DEATHS * 1.5

        # 4. Награда за эффективное использование ресурсов
        if metrics["beds"] > 0:
            bed_utilization = metrics["occupied_beds"] / metrics["beds"]
            # Идеальная загрузка 70-90% - максимальная награда
            if 0.7 <= bed_utilization <= 0.9:
                reward += WEIGHT_RESOURCE_UTILIZATION * 2
            elif bed_utilization > 0.9:
                # Перегрузка - штраф
                reward += WEIGHT_RESOURCE_UTILIZATION * (1 - bed_utilization)
            else:
                # Недогрузка - небольшой штраф
                reward += WEIGHT_RESOURCE_UTILIZATION * (bed_utilization - 0.5)

        if metrics["icu"] > 0:
            icu_utilization = metrics["occupied_icu"] / metrics["icu"]
            if 0.6 <= icu_utilization <= 0.85:
                reward += WEIGHT_RESOURCE_UTILIZATION * 3  # ICU важнее обычных коек
            elif icu_utilization > 0.85:
                reward += WEIGHT_RESOURCE_UTILIZATION * (1 - icu_utilization) * 2
            else:
                reward += WEIGHT_RESOURCE_UTILIZATION * (icu_utilization - 0.4)

        # 5. Учет бюджета (положительная за экономию, отрицательная за перерасход)
        budget_balance = metrics.get("budget", 0) - metrics.get("expenses", 0)
        reward += budget_balance * WEIGHT_BUDGET

        # 6. Штраф за дорогостоящие действия (стимулируем разумное использование)
        action_costs = {
            0: 0,  # Ничего не делать - бесплатно
            1: 10,  # Купить 1 койку
            2: 45,  # Купить 5 коек (скидка за опт)
            3: 50,  # Купить 1 ИВЛ
            4: 225,  # Купить 5 ИВЛ (скидка за опт)
            5: 2,  # Законсервировать 1 койку
            6: 8,  # Законсервировать 5 коек
            7: 5,  # Законсервировать 1 ИВЛ
            8: 20,  # Законсервировать 5 ИВЛ
            9: 1,  # Срочный бюджет (административные расходы)
        }

        reward += action_costs.get(action, 0) * WEIGHT_ACTION_COST

        # 7. Бонус за предотвращение кризисных ситуаций
        crisis_penalty = 0
        if metrics["rejected"] > metrics["admitted"] * 0.1:  # Более 10% отказов
            crisis_penalty -= 20
        if metrics["occupied_icu"] >= metrics["icu"]:  # Переполнение ICU
            crisis_penalty -= 30

        reward += crisis_penalty

        # 8. Штраф за избыточные ресурсы (стимулируем эффективное планирование)
        excess_resources_penalty = 0
        if metrics["beds"] > 0 and metrics["occupied_beds"] / metrics["beds"] < 0.3:
            excess_resources_penalty -= 5
        if metrics["icu"] > 0 and metrics["occupied_icu"] / metrics["icu"] < 0.2:
            excess_resources_penalty -= 8

        reward += excess_resources_penalty

        return float(reward)

