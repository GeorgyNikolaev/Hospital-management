"""DES модель"""
import heapq
import numpy as np
import pandas as pd

from typing import List
from dataclasses import asdict
from src.core.models import HospitalConfig, Patient
from src.core.config import settings


class DES:
    """DES model"""
    def __init__(self, hospitals: List[HospitalConfig], rng_seed: int = 0):
        self.hospitals = hospitals
        self.rng = np.random.RandomState(rng_seed)
        self.beds_heap = {h.name: [] for h in hospitals}
        self.icu_heap = {h.name: [] for h in hospitals}
        self.metrics = {h.name: {"admitted": 0, "rejected": 0, "deaths": 0, "patients": []} for h in hospitals}

    def _purge(self, now):
        """Очистка коек"""
        for h in self.hospitals:
            while self.beds_heap[h.name] and self.beds_heap[h.name][0] <= now:
                heapq.heappop(self.beds_heap[h.name])
            while self.icu_heap[h.name] and self.icu_heap[h.name][0] <= now:
                heapq.heappop(self.icu_heap[h.name])

    def current_occ(self, now: float):
        """Возвращает словарь с количеством занятых коек и ИВЛ для каждой больницы"""
        self._purge(now)

        occ = {}
        for h in self.hospitals:
            occ[h.name] = {
                "beds_in_use": len(self.beds_heap[h.name]),
                "icu_in_use": len(self.icu_heap[h.name])
            }
        return occ

    def assign_preferential(self, now, severity):
        """Оценивает больницы и возвращает список лучших"""
        occ = self.current_occ(now)
        scores = []

        for h in self.hospitals:
            cap_left = max(0, h.beds - occ[h.name]["beds_in_use"]) + max(0, h.icu_beds - occ[h.name]["icu_in_use"])
            scores.append([h, max(1e-6, cap_left) * h.quality])

        return [x[0] for x in sorted(scores, key=lambda x: x[1], reverse=True)]
        # arr = np.array(scores)
        # if arr.sum() == 0:
        #     return None
        # probs = arr / arr.sum()
        # idx = int(self.rng.choice(len(self.hospitals), p=probs))
        # return self.hospitals[idx]

    def attempt_admit(self, patient: Patient, max_wait=settings.MAX_WAIT):
        """Добавляем пациента в больницу"""
        now = patient.absolute_time
        hospitals_order = self.assign_preferential(now, patient.severity)
        tried = []

        for h in hospitals_order:
            admitted = self._try_admit_to_hospital(h, patient, now, max_wait)
            tried.append(h.name)
            if admitted:
                return patient  # Добавили пользователя в больницу

        patient.admitted = False
        patient.discharged_time = now
        base_death = settings.P_DEATH_HOSP_UNTREATED if patient.severity == "ward" else settings.P_DEATH_HOSP_UNTREATED

        fallback = hospitals_order[0]
        avg_quality = fallback.quality
        patient.died = (self.rng.rand() < min(0.99, base_death * (1 / avg_quality)))

        self.metrics[fallback.name]["rejected"] += 1
        self.metrics[fallback.name]["patients"].append(asdict(patient))
        if patient.died:
            self.metrics[fallback.name]["deaths"] += 1
        return patient

    def _try_admit_to_hospital(self, h: HospitalConfig, patient: Patient, now: float, max_wait: float):
        occ = self.current_occ(now)[h.name]
        needs_icu = patient.severity == "icu"
        avail_bed = (occ["beds_in_use"] < h.beds)
        avail_icu = (occ["icu_in_use"] < h.icu_beds)
        admit_time = now
        if needs_icu:
            if avail_icu:
                patient.admitted = True
            else:
                if self.icu_heap[h.name]:
                    earliest = self.icu_heap[h.name][occ["icu_in_use"] - h.icu_beds]
                    if earliest <= now + max_wait:
                        admit_time = earliest
                        patient.admitted = True
        else:
            if avail_bed:
                patient.admitted = True
            else:
                if self.beds_heap[h.name]:
                    earliest = self.beds_heap[h.name][occ["beds_in_use"] - h.beds]
                    if earliest <= now + max_wait:
                        admit_time = earliest
                        patient.admitted = True

        if not patient.admitted:
            return False

        patient.assigned_hospital = h.name
        patient.waited = max(0.0, admit_time - now)
        release = admit_time + patient.los * (1 / h.quality)

        if needs_icu:
            heapq.heappush(self.icu_heap[h.name], release)
        else:
            heapq.heappush(self.beds_heap[h.name], release)

        patient.discharged_time = release

        base_death = settings.P_DEATH_HOSP_TREATED if patient.severity == "ward" else settings.P_DEATH_INC_TREATED
        patient.died = (self.rng.rand() < min(0.99, base_death * (1 / h.quality)))
        self.metrics[h.name]["admitted"] += 1

        if patient.died:
            self.metrics[h.name]["deaths"] += 1
        self.metrics[h.name]["patients"].append(asdict(patient))
        return True

    def daily_metrics(self):
        rows = []
        for h in self.hospitals:
            m = self.metrics[h.name]
            rows.append({"hospital": h.name, "admitted": m["admitted"], "rejected": m["rejected"], "deaths": m["deaths"], "events": len(m["patients"])})
        return pd.DataFrame(rows)