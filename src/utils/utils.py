"""Вспомогательные функции"""
import time
import numpy as np

from src.core.models import SEIRHCDParams


def sample_arrivals(expected_hosp: float, expected_icu: float, rng: np.random.RandomState):
    """Функция распределяет пациентов в течение дня используя нормально распределение"""
    ward_lambda = max(0.0, expected_hosp)
    icu_lambda = max(0.0, expected_icu)
    n_ward = rng.poisson(lam=ward_lambda)
    n_icu = rng.poisson(lam=icu_lambda)
    # n_ward = round(expected_hosp+0.5)
    # n_icu = round(expected_icu+0.5)
    events = []
    for _ in range(n_ward):
        events.append({"time_frac": rng.random(), "severity": "ward"})
    for _ in range(n_icu):
        events.append({"time_frac": rng.random(), "severity": "icu"})
    events.sort(key=lambda x: x["time_frac"])
    return events

def now_str():
    return time.strftime("%Y%m%d_%H%M%S")

def make_params_consistent(
    population: int,
    sigma: float,
    gamma: float,
    R0: float = None,
    beta: float = None,
    p_hosp: float = 0.5,
    p_icu: float = 0.05,
    p_death: float = 0.047,
    hosp_duration: float = 14.0,
    icu_duration: float = 17.0,
    initial_exposed: int = 0,
    initial_infectious: int = 0,
):
    """
    Construct SEIRHCDParams with internal consistency:
      - If beta is None, use R0 * gamma
      - Compute alpha_h so that P(hospitalized | infected) = p_hosp using competing exponentials:
            p_hosp = alpha_h / (alpha_h + gamma)  => alpha_h = p_hosp * gamma / (1 - p_hosp)
      - Compute alpha_c so that P(ICU | hospitalized) = p_icu_among_hosp using:
            p_icu_among_hosp = alpha_c / (alpha_c + gamma_h) => alpha_c = p_icu_among_hosp * gamma_h / (1 - p_icu_among_hosp)
      - Compute mu_c so that P(death | ICU) = p_death_among_icu using:
            p_death_among_icu = mu_c / (mu_c + gamma_c) => mu_c = p_death_among_icu * gamma_c / (1 - p_death_among_icu)

    Notes:
      - This enforces probabilistic consistency; mean delays are governed by rates (exponentials).
      - If you require deterministic delays, implement delay queues or Erlang chains (not done here).
    """
    if beta is None:
        if R0 is None:
            raise ValueError("Provide either beta or R0.")
        beta = R0 * gamma

    gamma_h = 1.0 / hosp_duration
    gamma_c = 1.0 / icu_duration

    # compute alpha_h from p_hosp (probability that an infectious individual will be hospitalized before recovery)
    if not (0 <= p_hosp < 1):
        raise ValueError("p_hosp must be in [0, 1).")
    alpha_h = (p_hosp * gamma) / (1.0 - p_hosp) if p_hosp > 0 else 0.0

    # among hospitalized, fraction to ICU:
    if p_hosp == 0:
        p_icu_among_hosp = 0.0
    else:
        p_icu_among_hosp = p_icu / p_hosp if p_hosp > 0 else 0.0
    # clamp to <1
    if p_icu_among_hosp >= 1.0:
        raise ValueError("p_icu must be < p_hosp (or check probabilities).")
    alpha_c = (p_icu_among_hosp * gamma_h) / (1.0 - p_icu_among_hosp) if p_icu_among_hosp > 0 else 0.0

    # death among ICU
    p_death_among_icu = p_death / p_icu if p_icu > 0 else 0.0
    if p_death_among_icu >= 1.0:
        raise ValueError("p_death must be < p_icu (or check probabilities).")
    mu_c = (p_death_among_icu * gamma_c) / (1.0 - p_death_among_icu) if p_death_among_icu > 0 else 0.0

    return SEIRHCDParams(
        population=population,
        beta=beta,
        sigma=sigma,
        gamma=gamma,
        gamma_h=gamma_h,
        gamma_c=gamma_c,
        mu_c=mu_c,
        alpha_h=alpha_h,
        alpha_c=alpha_c,
        p_hosp=p_hosp,
        p_icu=p_icu,
        p_death=p_death,
        initial_exposed=initial_exposed,
        initial_infectious=initial_infectious
    )

