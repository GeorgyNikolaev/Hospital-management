import numpy as np

from scipy.optimize import minimize
from src.core.models import SEIRHCDParams
from src.sd.seir_model import simulate_seir_hcd


def calibrate_seir(params: SEIRHCDParams, observed: np.ndarray, days: int):
    x0 = np.array([params.beta, params.p_hosp])
    bounds = [(0.01, 3.0), (1e-4, 0.5)]

    def loss(x):
        p = SEIRHCDParams(population=params.population, beta=float(x[0]), sigma=params.sigma, gamma=params.gamma,
                       hosp_rate=float(x[1]), icu_rate=params.icu_rate, hosp_delay=params.hosp_delay, icu_delay=params.icu_delay,
                       initial_exposed=params.initial_exposed, initial_infectious=params.initial_infectious)
        sim = simulate_seir_hcd(p, days=days)
        pred = sim["expected_hospitalizations"].values[:len(observed)]
        obs = observed[:len(pred)]
        return float(np.mean(np.abs(pred - obs)))

    res = minimize(loss, x0, bounds=bounds, method="L-BFGS-B", options={"maxiter": 200})
    calibrated = res.x
    return SEIRHCDParams(population=params.population, beta=float(calibrated[0]), sigma=params.sigma, gamma=params.gamma,
                      hosp_rate=float(calibrated[1]), icu_rate=params.icu_rate, hosp_delay=params.hosp_delay, icu_delay=params.icu_delay,
                      initial_exposed=params.initial_exposed, initial_infectious=params.initial_infectious)
