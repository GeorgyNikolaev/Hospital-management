"""Класс агента"""


levels = {
    "growth" :
        {
            "low_1_level": 0.3,
            "low_2_level": 0.4,
            "top_1_level": 0.9,
            "top_2_level": 0.95
        },
    "fall" :
        {
            "low_1_level": 0.001,
            "low_2_level": 0.01,
            "top_1_level": 0.05,
            "top_2_level": 0.1
        }
}

class Agent:
    """"Агент для TTM системы управления больницей"""
    profile_type: int # Тип профиля

    def __init__(self, profile_type: int=0):
        self.profile_type = profile_type

    @staticmethod
    def select_action(obs:dict, mask) -> int:
        """Функция для выбора действия"""
        expected_1_day = obs["expected_hosp_1_day"] + obs["expected_icu_1_day"]
        expected_5_day = obs["expected_hosp_5_day"] + obs["expected_icu_5_day"]
        expected_15_day = obs["expected_hosp_15_day"] + obs["expected_icu_15_day"]

        if expected_1_day < expected_5_day < expected_15_day:
           levels_ = levels["growth"]
        elif expected_1_day > expected_5_day > expected_15_day:
            levels_ = levels["fall"]
        else:
            return 0

        free_beds = obs["beds"] - obs["occupied_beds"]
        free_icu = obs["icu"] - obs["occupied_icu"]

        beds_level = free_beds / obs["beds"] if obs["beds"] != 0 else 1
        icu_level = free_icu / obs["icu"] if obs["icu"] != 0 else 1

        print(beds_level, icu_level)

        if beds_level <= levels_["low_1_level"]:
            return Agent.try_buy_beds(action=2, mask=mask)
        elif levels_["low_1_level"] < beds_level <= levels_["low_2_level"]:
            return Agent.try_buy_beds(action=1, mask=mask)
        elif levels_["top_1_level"] < beds_level <= levels_["top_2_level"]:
            return 5
        elif beds_level > levels_["top_2_level"]:
            return 6

        if icu_level <= levels_["low_1_level"]:
            return Agent.try_buy_beds(action=4, mask=mask)
        elif levels_["low_1_level"] < icu_level <= levels_["low_2_level"]:
            return Agent.try_buy_beds(action=3, mask=mask)
        elif levels_["top_1_level"] < icu_level <= levels_["top_2_level"]:
            return 7
        elif icu_level > levels_["top_2_level"]:
            return 8

        return 0

    @staticmethod
    def try_buy_beds(action: int, mask: list[int]) -> int:
        """Попытка купить койки или ИВЛ"""
        if mask[action] == 0:
            return 9
        return action