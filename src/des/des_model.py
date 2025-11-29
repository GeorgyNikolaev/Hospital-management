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

    def assign_preferential(self, now, severity):
        """Оценивает больницы и возвращает список лучших"""
        scores = []

        for h in self.hospitals:
            occ = h.current_occ(now)
            cap_left = max(0, h.beds - occ["beds_in_use"]) + max(0, h.icu - occ["icu_in_use"])
            scores.append([h, max(1e-6, cap_left) * h.quality])

        return [x[0] for x in sorted(scores, key=lambda x: x[1], reverse=True)]

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

