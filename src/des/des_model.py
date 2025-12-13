"""DES модель"""
import math

import numpy as np

from typing import List
from src.core.models import Hospital, Patient
from src.core.config import settings


class DES:
    """DES model"""
    def __init__(self, hospitals: List[Hospital], rng_seed: int = 0):
        self.hospitals = hospitals
        self.rng = np.random.RandomState(rng_seed)

    def assign_preferential(self, now, severity, use_random=False):
        """Оценивает больницы и возвращает список лучших"""
        if use_random:
            # Возвращаем случайный порядок больниц
            hospitals = self.hospitals.copy()
            np.random.shuffle(hospitals)
            return hospitals

        scores = []

        for h in self.hospitals:
            # Текущая занятость
            occ = h.current_occ(now)

            # Свободные места с учетом типа пациента
            if severity == "icu":
                available_beds = max(0, h.icu - occ["icu_in_use"])
            else:
                available_beds = max(0, h.beds - occ["beds_in_use"])

            # Вариант A: Произведение качества и доступности (ваш вариант, но с нормализацией)
            score = h.quality * available_beds

            # Вариант B: Логарифмическая (чтобы избежать слишком больших чисел)
            # score = h.quality * np.log1p(available_beds)

            # Вариант C: С учетом расстояния (если есть координаты)
            # distance = self.calculate_distance(patient.location, h.location)
            # score = h.quality * available_beds / (1 + distance)

            scores.append([h, score])

        # Сортируем по убыванию score
        ranked_hospitals = [x[0] for x in sorted(scores, key=lambda x: x[1], reverse=True)]

        return ranked_hospitals

    def attempt_admit(self, patient: Patient, max_wait=settings.MAX_WAIT):
        """Добавляем пациента в больницу"""
        now = patient.absolute_time
        hospitals_order = self.assign_preferential(now, patient.severity)

        for h in hospitals_order:
            admitted = h.try_admit_to_hospital(patient, now, max_wait)
            if admitted:
                return patient  # Добавили пользователя в больницу

        patient.admitted = False
        patient.discharged_time = now
        base_death = settings.P_DEATH_HOSP_UNTREATED if patient.severity == "ward" else settings.P_DEATH_HOSP_UNTREATED

        fallback = hospitals_order[0]
        avg_quality = fallback.quality
        patient.died = (self.rng.rand() < min(0.99, base_death * (1 / avg_quality)))

        # self.metrics[fallback.name]["rejected"] += 1
        # self.metrics[fallback.name]["patients"].append(asdict(patient))
        day = math.floor(patient.absolute_time)
        metric = self.hospitals[0].metrics[day]
        if patient.died:
            metric["deaths"] += 1
            if patient.severity == "icu" :
                metric["deaths_icu"] += 1
            else:
                metric["deaths_hosp"] += 1
        return patient

