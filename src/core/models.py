"""Модели"""
from dataclasses import dataclass
from typing import Optional


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
class HospitalConfig:
    """Конфиг госпиталя"""
    name: str
    beds: int
    icu_beds: int
    ventilators: int
    quality: float  # <1 better, >1 worse


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