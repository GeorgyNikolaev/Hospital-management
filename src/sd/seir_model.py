import json

import numpy as np
import pandas as pd

from src.core.models import SEIRHCDParams


def simulate_seir_hcd(params: SEIRHCDParams, days: int, start_day: int = 1, dt: float = 1.0, beta_time_fn=None, data: pd.DataFrame = None) -> pd.DataFrame:
    """Симуляция SEIR-H-C-D модели"""
    n_steps = int(np.floor(days / dt)) + 1
    t = np.linspace(0.0, start_day + days - 1, n_steps + start_day - 1)

    # state arrays
    if data is None:
        S, E, I, H, C, R, D = [np.zeros(n_steps) for _ in range(7)]
        # initial conditions
        S[0] = params.population - params.initial_exposed - params.initial_infectious
        E[0] = params.initial_exposed
        I[0] = params.initial_infectious
    else:
        zeros_row = pd.DataFrame([[0] * len(data.columns)], columns=data.columns)
        data = pd.concat([data, zeros_row], ignore_index=True)
        S = data.S
        E = data.E
        I = data.I
        R = data.R
        H = data.H
        C = data.C
        D = data.D

    flow_new_exposed = np.zeros(n_steps+start_day-1)
    flow_E_to_I = np.zeros(n_steps+start_day-1)
    flow_I_to_H = np.zeros(n_steps+start_day-1)
    flow_I_to_R = np.zeros(n_steps+start_day-1)
    flow_H_to_C = np.zeros(n_steps+start_day-1)
    flow_H_to_R = np.zeros(n_steps+start_day-1)
    flow_C_to_R = np.zeros(n_steps+start_day-1)
    flow_C_to_D = np.zeros(n_steps+start_day-1)


    def deriv(ti, S_, E_, I_, H_, C_, R_, D_):
        beta = params.beta if beta_time_fn is None else beta_time_fn(ti, params.beta)

        new_inf_flow = beta * S_ * I_ / params.population        # S -> E
        E_to_I_flow = params.sigma * E_                          # E -> I
        I_to_H_flow = params.alpha_h * I_                        # I -> H
        I_to_R_flow = params.gamma * I_                          # I -> R
        H_to_C_flow = params.alpha_c * H_                        # H -> C
        H_to_R_flow = params.gamma_h * H_                        # H -> R
        C_to_R_flow = params.gamma_c * C_                        # C -> R
        C_to_D_flow = params.mu_c * C_                           # C -> D

        dS = -new_inf_flow
        dE = new_inf_flow - E_to_I_flow
        dI = E_to_I_flow - I_to_H_flow - I_to_R_flow
        dH = I_to_H_flow - H_to_C_flow - H_to_R_flow
        dC = H_to_C_flow - C_to_R_flow - C_to_D_flow
        dR = I_to_R_flow + H_to_R_flow + C_to_R_flow
        dD = C_to_D_flow

        flows = {
            "new_inf_flow": new_inf_flow,
            "E_to_I_flow": E_to_I_flow,
            "I_to_H_flow": I_to_H_flow,
            "I_to_R_flow": I_to_R_flow,
            "H_to_C_flow": H_to_C_flow,
            "H_to_R_flow": H_to_R_flow,
            "C_to_R_flow": C_to_R_flow,
            "C_to_D_flow": C_to_D_flow
        }

        return (dS, dE, dI, dH, dC, dR, dD), flows

    for i in range(start_day, start_day + n_steps - 1):
        ti = t[i-1]
        state = (S[i-1], E[i-1], I[i-1], H[i-1], C[i-1], R[i-1], D[i-1])

        k1, flows1 = deriv(ti, *state)
        mid_state1 = tuple(x + dt/2 * k for x, k in zip(state, k1))

        k2, flows2 = deriv(ti + dt/2, *mid_state1)
        mid_state2 = tuple(x + dt/2 * k for x, k in zip(state, k2))

        k3, flows3 = deriv(ti + dt/2, *mid_state2)
        end_state = tuple(x + dt * k for x, k in zip(state, k3))

        k4, flows4 = deriv(ti + dt, *end_state)

        updates = [x + dt*(k1_j + 2*k2_j + 2*k3_j + k4_j)/6
                   for x, k1_j, k2_j, k3_j, k4_j in zip(state, k1, k2, k3, k4)]

        S[i], E[i], I[i], H[i], C[i], R[i], D[i] = updates

        flow_new_exposed[i] = flows1["new_inf_flow"]
        flow_E_to_I[i] = flows1["E_to_I_flow"]
        flow_I_to_H[i] = flows1["I_to_H_flow"]
        flow_I_to_R[i] = flows1["I_to_R_flow"]
        flow_H_to_C[i] = flows1["H_to_C_flow"]
        flow_H_to_R[i] = flows1["H_to_R_flow"]
        flow_C_to_R[i] = flows1["C_to_R_flow"]
        flow_C_to_D[i] = flows1["C_to_D_flow"]

    new_infected = flow_new_exposed * dt
    new_E_to_I = flow_E_to_I * dt
    new_hospitalizations = flow_I_to_H * dt
    new_icu = flow_H_to_C * dt
    new_deaths = flow_C_to_D * dt
    new_recoveries = (flow_I_to_R + flow_H_to_R + flow_C_to_R) * dt

    new_data = {
        "t": t,
        "S": S, "E": E, "I": I, "H": H, "C": C, "R": R, "D": D,
        "new_infected": concat(data["new_infected"] if not data is None else None, new_infected, start_day),
        "E_to_I": concat(data["E_to_I"] if not data is None else None, new_E_to_I, start_day),
        "new_hospitalizations": concat(data["new_hospitalizations"] if not data is None else None, new_hospitalizations, start_day),
        "new_icu": concat(data["new_icu"] if not data is None else None, new_icu, start_day),
        "new_deaths": concat(data["new_deaths"] if not data is None else None, new_deaths, start_day),
        "new_recoveries": concat(data["new_recoveries"] if not data is None else None, new_recoveries, start_day),
        "rate_new_infected": concat(data["rate_new_infected"] if not data is None else None, flow_new_exposed, start_day),
        "rate_I_to_H": concat(data["rate_I_to_H"] if not data is None else None, flow_I_to_H, start_day),
        "rate_I_to_R": concat(data["rate_I_to_R"] if not data is None else None, flow_I_to_R, start_day),
        "rate_H_to_C": concat(data["rate_H_to_C"] if not data is None else None, flow_H_to_C, start_day),
        "rate_C_to_D": concat(data["rate_C_to_D"] if not data is None else None, flow_C_to_D, start_day),
    }

    # Перед созданием DataFrame преобразовать все Series в numpy arrays
    for key in new_data:
        if hasattr(new_data[key], 'values'):
            new_data[key] = new_data[key].values

    df = pd.DataFrame(new_data)

    return df

def concat(data_1, data_2, index: int):
    return pd.concat([
            data_1[:index],
            pd.Series(data_2[index:])
        ]) if data_1 is not None else data_2