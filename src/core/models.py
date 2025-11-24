"""Модели"""
import heapq
import math
from dataclasses import dataclass, asdict, field
from typing import Optional

import numpy as np

from src.core.config import settings


@dataclass
class SEIRHCDParams:
    """Настройка SEIR-H-C-D модели"""
    population: int

    # SEIR core
    beta: float
    sigma: float  # E -> I rate
    gamma: float  # I -> R rate

    # hospital / ICU / death rates (they will be computed from probabilities if not provided)
    gamma_h: float  # H -> R rate (1 / hosp_duration)
    gamma_c: float  # C -> R rate (1 / icu_duration)
    mu_c: float     # C -> D rate

    # transition I -> H and H -> C as rates
    alpha_h: float  # I -> H
    alpha_c: float  # H -> C

    # probabilities / delays (kept for record)
    p_hosp: float
    p_icu: float
    p_death: float

    # initial conditions
    initial_exposed: int = 0
    initial_infectious: int = 0


@dataclass
class Patient:
    """Конфиг пациента"""
    id: int
    absolute_time: float
    severity: str
    los: float
    assigned_hospital: Optional[str] = None
    admitted: bool = False
    discharged_time: Optional[float] = None
    died: bool = False
    waited: float = 0.0


@dataclass
class Hospital:
    """Конфиг госпиталя"""
    id: int
    name: str
    beds: int
    icu_beds: int
    quality: float  # <1 better, >1 worse
    rng_seed: int = 42
    beds_heap: list = field(default_factory=list, init=False, repr=False)
    icu_heap: list = field(default_factory=list, init=False, repr=False)
    metrics: dict = field(default_factory=dict, init=False, repr=False)
    patients: list = field(default_factory=list, init=False, repr=False)
    rng: np.random.RandomState = field(init=False, repr=False)

    def __post_init__(self):
        self.rng_seed = 42  # или любое другое значение по умолчанию
        self.rng = np.random.RandomState(self.rng_seed)

    def _purge(self, now):
        """Очистка коек"""
        while self.beds_heap and self.beds_heap[0] <= now:
            heapq.heappop(self.beds_heap)
        while self.icu_heap and self.icu_heap[0] <= now:
            heapq.heappop(self.icu_heap)

    def current_occ(self, now: float):
        """Возвращает словарь с количеством занятых коек и ИВЛ для каждой больницы"""
        self._purge(now)

        occ = {
            "beds_in_use": len(self.beds_heap),
            "icu_in_use": len(self.icu_heap)
        }
        return occ

    def try_admit_to_hospital(self, patient: Patient, now: float, max_wait: float):
        """Попытка оформить пациента"""
        occ = self.current_occ(now)
        needs_icu = patient.severity == "icu"
        avail_bed = (occ["beds_in_use"] < self.beds)
        avail_icu = (occ["icu_in_use"] < self.icu_beds)
        admit_time = now

        day = math.floor(admit_time)
        if day in self.metrics:
            metric = self.metrics[day]
        else:
            metric = {"day": day,
                      "admitted": 0,
                      "admitted_hosp": 0,
                      "admitted_icu": 0,
                      "rejected": 0,
                      "rejected_hosp": 0,
                      "rejected_icu": 0,
                      "deaths": 0,
                      "deaths_hosp": 0,
                      "deaths_icu": 0}
            self.metrics[day] = metric


        if needs_icu:
            if avail_icu:
                patient.admitted = True
            else:
                if self.icu_heap:
                    earliest = self.icu_heap[occ["icu_in_use"] - self.icu_beds]
                    if earliest <= now + max_wait:
                        admit_time = earliest
                        patient.admitted = True
        else:
            if avail_bed:
                patient.admitted = True
            else:
                if self.beds_heap:
                    earliest = self.beds_heap[occ["beds_in_use"] - self.beds]
                    if earliest <= now + max_wait:
                        admit_time = earliest
                        patient.admitted = True

        if not patient.admitted:
            metric["rejected"] += 1
            if needs_icu:
                metric["rejected_icu"] += 1
            else:
                metric["rejected_hosp"] += 1
            return False

        patient.assigned_hospital = self.name
        patient.waited = max(0.0, admit_time - now)
        release = admit_time + patient.los * (1 / self.quality)

        metric["admitted"] += 1
        if needs_icu:
            metric["admitted_icu"] += 1
            heapq.heappush(self.icu_heap, release)
        else:
            metric["admitted_hosp"] += 1
            heapq.heappush(self.beds_heap, release)

        patient.discharged_time = release

        base_death = settings.P_DEATH_HOSP_TREATED if patient.severity == "ward" else settings.P_DEATH_INC_TREATED
        patient.died = (self.rng.rand() < min(0.99, base_death * (1 / self.quality)))

        if patient.died:
            metric["deaths"] += 1
            if needs_icu:
                metric["deaths_icu"] += 1
            else:
                metric["deaths_hosp"] += 1
        self.patients.append(asdict(patient))
        return True

    def daily_metrics(self, day):
        """Дневная метрика"""
        return self.metrics.get(day, {"day": day,
                                      "admitted": 0,
                                      "admitted_hosp": 0,
                                      "admitted_icu": 0,
                                      "rejected": 0,
                                      "rejected_hosp": 0,
                                      "rejected_icu": 0,
                                      "deaths": 0,
                                      "deaths_hosp": 0,
                                      "deaths_icu": 0})

    def apply_action(self, action):
        if action == 0:
            return
        elif action == 1:
            self.beds = int(self.beds_total * 1.10)
            self.icu = int(self.icu_total * 1.10)
        elif action == 2:
            self.beds_total = int(self.beds_total * 0.90)
            self.icu_total = int(self.icu_total * 0.90)
