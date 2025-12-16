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


class Hospital:
    """Конфиг госпиталя"""
    beds_heap: list = field(default_factory=list, init=False, repr=False)
    icu_heap: list = field(default_factory=list, init=False, repr=False)
    metrics: dict = field(default_factory=dict, init=False, repr=False)
    patients: list = field(default_factory=list, init=False, repr=False)
    rng: np.random.RandomState = field(init=False, repr=False)

    def __init__(self,
                 id: int,
                 name: str,
                 beds: int,
                 reserve_beds: int,
                 icu: int,
                 reserve_icu: int,
                 quality: float,
                 costs: dict,
                 budget: int,
                 rng_seed: int = 42,):
        self.rng = np.random.RandomState(rng_seed)

        self.name = name
        self.id = id
        self.beds = beds
        self.reserve_beds = reserve_beds
        self.icu = icu
        self.reserve_icu = reserve_icu
        self.quality = quality
        self.costs = costs
        self.budget = budget

        self.beds_heap: list = []
        self.icu_heap: list = []
        self.metrics: dict = {}
        self.patients: list = []

    @property
    def _default_metric_day(self) -> dict[str, int]:
        # print(self.costs["bed_day"] * self.beds + self.costs["icu_day"] * self.icu)
        return {
            "day": 0,
            "admitted": 0,
            "admitted_hosp": 0,
            "admitted_icu": 0,
            "rejected": 0,
            "rejected_hosp": 0,
            "rejected_icu": 0,
            "deaths": 0,
            "deaths_hosp": 0,
            "deaths_icu": 0,
            "beds": self.beds,
            "reserve_beds": self.reserve_beds,
            "occupied_beds": 0,
            # "occupied_beds": len(self.beds_heap),
            "icu": self.icu,
            "reserve_icu": self.reserve_icu,
            "occupied_icu": 0,
            # "occupied_icu": len(self.icu_heap),
            "budget": self.budget,
            "expenses": self.costs["beds_day"] * self.beds + self.costs["icu_day"] * self.icu
        }

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
        avail_icu = (occ["icu_in_use"] < self.icu)
        admit_time = now

        day = math.floor(admit_time)
        metric = self.save_daily_metrics(day=day)

        if needs_icu:
            if avail_icu:
                patient.admitted = True
            else:
                if self.icu_heap:
                    try:
                        earliest = self.icu_heap[occ["icu_in_use"]-1]
                        if earliest <= now + max_wait:
                            admit_time = earliest
                            patient.admitted = True
                    except IndexError as e:
                        print(len(self.icu_heap), occ["icu_in_use"])
        else:
            if avail_bed:
                patient.admitted = True
            else:
                if self.beds_heap:
                    earliest = self.beds_heap[occ["beds_in_use"]-1]
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

        # if needs_icu:
        #     metric["occupied_icu"] += 1
        # else:
        #     metric["occupied_beds"] += 1

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

    def daily_metrics(self, day: int = 0):
        """Дневная метрика"""
        metric = self.metrics.get(day)
        if not metric:
            metric = self._default_metric_day
            metric["day"] = day
            metric["budget"] -= metric["expenses"]
            self.metrics[day] = metric

        occ = self.current_occ(day)
        metric["occupied_beds"] = occ["beds_in_use"]
        metric["occupied_icu"] = occ["icu_in_use"]

        metric["beds"] = self.beds
        metric["icu"] = self.icu

        return metric

    def save_daily_metrics(self, day: int = 0):
        """Сохранение дневной метрики"""
        metric = self.metrics.get(day)
        if not metric:
            metric = self._default_metric_day
            metric["day"] = day
            metric["budget"] -= metric["expenses"]
            self.metrics[day] = metric
        metric["beds"] = self.beds
        metric["icu"] = self.icu

        return metric


    def get_action_mask(self):
        mask = [1] * 10
        ocp_b = len(self.beds_heap)
        ocp_i = len(self.icu_heap)
        if self.costs["beds_purchase"] > self.budget and self.reserve_beds < 1: mask[1] = 0
        if self.costs["beds_purchase"] * 5 > self.budget and self.reserve_beds < 5: mask[2] = 0
        if self.costs["icu_purchase"] > self.budget and self.reserve_icu < 1: mask[3] = 0
        if self.costs["icu_purchase"] * 5 > self.budget and self.reserve_icu < 5: mask[4] = 0
        if self.beds-ocp_b < 1: mask[5] = 0
        if self.beds-ocp_b < 5: mask[6] = 0
        if self.icu-ocp_i < 1: mask[7] = 0
        if self.icu-ocp_i < 5: mask[8] = 0
        return mask

    def apply_action(self, action):
        """Принятие действия к больнице"""
        if action == 0:  # Ничего не делать
            return
        elif action == 1: # Купить 1 койку (освободить)
            self._add_beds("beds", 1)
        elif action == 2: # Купить 5 коек (освободить)
            self._add_beds("beds", 5)
        elif action == 3: # Купить 1 аппарат ИВЛ (освободить)
            self._add_beds("icu", 1)
        elif action == 4: # Купить 5 аппаратов ИВЛ (освободить)
            self._add_beds("icu", 5)
        elif action == 5: # Законсервировать 1 койку
            self._add_beds("beds", 1, is_reserved=True)
        elif action == 6: # Законсервировать 5 коек
            self._add_beds("beds", 5, is_reserved=True)
        elif action == 7: # Законсервировать 1 аппарат ИВЛ
            self._add_beds("icu", 1, is_reserved=True)
        elif action == 8: # Законсервировать 5 аппаратов ИВЛ
            self._add_beds("icu", 5, is_reserved=True)
        elif action == 9: # Срочно выделить бюджет на закупку
            return

    def _add_beds(self, bed_type: str, n: int, is_reserved: bool=False):
        r = getattr(self, "reserve_" + bed_type, 0)
        b = getattr(self, bed_type, 0)

        if is_reserved:
            ocp = len(getattr(self, bed_type + "_heap", 0))
            if (b - ocp) >= n:
                setattr(self, "reserve_" + bed_type, r + n)
                setattr(self, bed_type, b - n)
        else:
            if r >= n:
                setattr(self, bed_type, b + n)
                setattr(self, "reserve_" + bed_type, r - n)
            elif self.costs[bed_type + "_purchase"] * n < self.budget:
                self.budget -= self.costs[bed_type + "_purchase"] * n
                setattr(self, bed_type, b + n)

        


