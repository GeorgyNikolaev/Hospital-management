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

    def build_obs(self, metrics):
        """
        Привязка к твоим метрикам: используем поля, которые реально заполняет Hospital.
        Возвращает нормализованный вектор [beds, occupied_beds, icu, occupied_icu, admitted, rejected]
        """
        x = np.array([
            metrics.get("beds", 0),  # total beds
            metrics.get("occupied_beds", 0),  # occupied beds
            metrics.get("icu", 0),  # total icu
            metrics.get("occupied_icu", 0),  # occupied icu
            metrics.get("admitted", 0),
            metrics.get("rejected", 0),
        ], dtype=np.float32)

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
        Возвращает scalar reward (float).
        Используемые ключи metrics (обязательные):
          - "beds_total", "beds_occupied", "icu_total", "icu_occupied", "admitted", "rejected"
        Дополнительные (опционально, если есть в вашей симуляции):
          - "maintenance_cost"            : суммарные ежедневные расходы на поддержание (float)
          - "bed_maint_cost", "icu_maint_cost" : поддержание на одну единицу (float)
          - "budget_spent"                : потрачено средств на покупки в этот день (float)
          - "budget"                      : оставшийся бюджет (float)  (необязательно)
        Параметры-коэффициенты вынесены в локальные переменные — легко настраивать.
        """

        # --- извлекаем базовые метрики (ожидаются всегда) ---
        beds_total = float(metrics.get("beds_total", 0.0))
        beds_occupied = float(metrics.get("beds_occupied", 0.0))
        icu_total = float(metrics.get("icu_total", 0.0))
        icu_occupied = float(metrics.get("icu_occupied", 0.0))
        admitted = float(metrics.get("admitted", 0.0))
        rejected = float(metrics.get("rejected", 0.0))

        # --- дополнительные метрики (опционально) ---
        maintenance_cost = metrics.get("maintenance_cost", None)
        bed_maint_cost = float(metrics.get("bed_maint_cost", 1.0))
        icu_maint_cost = float(metrics.get("icu_maint_cost", 5.0))
        budget_spent = float(metrics.get("budget_spent", 0.0))
        # budget = metrics.get("budget", None)  # для информации, если нужно

        # --- коэффициенты (настраиваемые) ---
        REJECT_PENALTY = 50.0  # тяжёлый штраф за один отказ
        ADMIT_REWARD = 1.0  # положительная мотивация за госпитализацию
        ICU_IMPORTANCE = 2.0  # ICU важнее, поэтому интенсивнее штрафуем нагрузку
        UTIL_THRESHOLD = 0.8  # порог комфортной загрузки
        UTIL_PENALTY_SCALE = 20.0  # масштаб штрафа за превышение порога (квадратично)
        MAINT_PENALTY_SCALE = 0.1  # масштаб штрафа за maintenance_cost / расчетной стоимости
        PURCHASE_PENALTY_SCALE = 0.5  # масштаб штрафа за деньги, потраченные на покупку в этот день
        RESERVE_BONUS = 2.0  # бонус за консервацию (reserve) — стимулирует экономию
        BUY_PENALTY = 1.0  # небольшой штраф за покупку (чтобы не покупать лишнее)
        MAX_ABS_REWARD = 1e6  # защита от аномалий

        # --- базовый reward ---
        reward = 0.0

        # 1) поощрение за госпитализацию и сильный штраф за отказ
        reward += ADMIT_REWARD * admitted
        reward -= REJECT_PENALTY * rejected

        # 2) загрузка (нормализуем; если нет коек, считаем загрузку = 1.0)
        bed_util = beds_occupied / max(1.0, beds_total)
        icu_util = icu_occupied / max(1.0, icu_total)

        # штраф за превышение порога — квадратичная зависимость для сильной реакции на большие перегрузки
        bed_excess = max(0.0, bed_util - UTIL_THRESHOLD)
        icu_excess = max(0.0, icu_util - UTIL_THRESHOLD)

        reward -= UTIL_PENALTY_SCALE * (bed_excess ** 2)
        reward -= UTIL_PENALTY_SCALE * ICU_IMPORTANCE * (icu_excess ** 2)

        # 3) штраф за поддержание (maintenance): используем явное значение, если есть,
        #    иначе аппроксимируем как per-unit * количество
        if maintenance_cost is not None:
            reward -= MAINT_PENALTY_SCALE * maintenance_cost
        else:
            approx_maintenance = bed_maint_cost * beds_total + icu_maint_cost * icu_total
            reward -= MAINT_PENALTY_SCALE * approx_maintenance

        # 4) штраф за покупки, если присутствует (чтобы агент думал о бюджете)
        if budget_spent is not None and budget_spent > 0.0:
            reward -= PURCHASE_PENALTY_SCALE * budget_spent

        # 5) бонусы / штрафы, зависящие от действия:
        #    actions: 1-4 = покупка, 5-8 = консервация (reserve), 0 = ничего, 9 = экстренный бюджет
        if action in (5, 6, 7, 8):
            # консервация уменьшает будущие maintenance_cost — небольшой стимулирующий бонус
            reward += RESERVE_BONUS
        elif action in (1, 2, 3, 4):
            # покупка увеличивает ресурсы, но тратит бюджет — небольшой отрицательный сигнал,
            # чтобы не покупать опрометчиво (реальная дисциплина достигается через budget_spent)
            reward -= BUY_PENALTY
        elif action == 9:
            # экстренное выделение бюджета — нейтрально/немного отрицательно
            reward -= BUY_PENALTY * 2.0

        # 6) дополнительная адаптивная штрафная составляющая:
        #    если нет коек вообще (beds_total==0) и есть пациенты — очень большой штраф через rejected уже есть,
        #    но добавим мягкий штраф за нулевой резерв мощности
        if beds_total <= 0 and (admitted + rejected) > 0:
            reward -= 20.0
        if icu_total <= 0 and icu_occupied > 0:
            reward -= 40.0

        # 7) финальная защита и нормировка: предотвращаем NaN/inf и экстремы
        if not np.isfinite(reward):
            reward = -1e3
        # при желании можно масштабировать/нормировать reward по дню или по населению — пока возвращаем raw float
        reward = float(max(-MAX_ABS_REWARD, min(MAX_ABS_REWARD, reward)))

        return reward

