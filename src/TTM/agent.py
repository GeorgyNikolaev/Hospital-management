"""Класс агента"""

class Agent:
    """"Агент для TTM системы управления больницей"""
    profile_type: int # Тип профиля

    def __init__(self, profile_type: int=0):
        self.profile_type = profile_type

    def select_action(self, obs, mask) -> int:
        """Функция для выбора действия"""
        return 0