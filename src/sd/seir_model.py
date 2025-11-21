import json

import numpy as np
import pandas as pd

from src.core.models import SEIRHCDParams


def simulate_seir_hcd(params: SEIRHCDParams, days: int, start_day: int = 1, dt: float = 1.0, beta_time_fn=None, data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Simulate SEIR-H-C-D with RK4. All flows are rates * state.
    Returns pandas DataFrame with compartments and instantaneous flows (per day).
    """
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

    # flow recording arrays (instantaneous rates; units: individuals per day)
    flow_new_exposed = np.zeros(n_steps+start_day-1)   # new exposures (infections)
    flow_E_to_I = np.zeros(n_steps+start_day-1)
    flow_I_to_H = np.zeros(n_steps+start_day-1)
    flow_I_to_R = np.zeros(n_steps+start_day-1)
    flow_H_to_C = np.zeros(n_steps+start_day-1)
    flow_H_to_R = np.zeros(n_steps+start_day-1)
    flow_C_to_R = np.zeros(n_steps+start_day-1)
    flow_C_to_D = np.zeros(n_steps+start_day-1)


    def deriv(ti, S_, E_, I_, H_, C_, R_, D_):
        beta = params.beta if beta_time_fn is None else beta_time_fn(ti, params.beta)

        # instantaneous flows (rates * counts)
        new_inf_flow = beta * S_ * I_ / params.population        # S -> E
        E_to_I_flow = params.sigma * E_                          # E -> I
        I_to_H_flow = params.alpha_h * I_                        # I -> H
        I_to_R_flow = params.gamma * I_                          # I -> R (recovery from I)
        H_to_C_flow = params.alpha_c * H_                        # H -> C
        H_to_R_flow = params.gamma_h * H_                        # H -> R (discharge from hospital)
        C_to_R_flow = params.gamma_c * C_                        # C -> R
        C_to_D_flow = params.mu_c * C_                           # C -> D

        dS = -new_inf_flow
        dE = new_inf_flow - E_to_I_flow
        dI = E_to_I_flow - I_to_H_flow - I_to_R_flow
        dH = I_to_H_flow - H_to_C_flow - H_to_R_flow
        dC = H_to_C_flow - C_to_R_flow - C_to_D_flow
        dR = I_to_R_flow + H_to_R_flow + C_to_R_flow
        dD = C_to_D_flow

        # return derivatives and the instantaneous flows for recording
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

    # RK4 integration loop
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

        # record flows at this time step using flows evaluated at beginning (k1) scaled to individuals/day
        # (alternatively could record average of flows1.flows4; we'll record flows evaluated at t_{i-1})
        flow_new_exposed[i] = flows1["new_inf_flow"]
        flow_E_to_I[i] = flows1["E_to_I_flow"]
        flow_I_to_H[i] = flows1["I_to_H_flow"]
        flow_I_to_R[i] = flows1["I_to_R_flow"]
        flow_H_to_C[i] = flows1["H_to_C_flow"]
        flow_H_to_R[i] = flows1["H_to_R_flow"]
        flow_C_to_R[i] = flows1["C_to_R_flow"]
        flow_C_to_D[i] = flows1["C_to_D_flow"]

    # Convert instantaneous rates to counts per dt (here dt in days)
    new_infected = flow_new_exposed * dt
    new_E_to_I = flow_E_to_I * dt
    new_hospitalizations = flow_I_to_H * dt
    new_icu = flow_H_to_C * dt
    new_deaths = flow_C_to_D * dt
    new_recoveries = (flow_I_to_R + flow_H_to_R + flow_C_to_R) * dt

    new_data = {
        "t": t,
        "S": S, "E": E, "I": I, "H": H, "C": C, "R": R, "D": D,
        "new_infected": pd.concat([
            data["new_infected"][:start_day],
            pd.Series(new_infected[start_day:])
        ]) if data is not None else new_infected,
        "E_to_I": pd.concat([
            data["E_to_I"][:start_day],
            pd.Series(new_E_to_I[start_day:])
        ]) if data is not None else new_E_to_I,
        "new_hospitalizations": pd.concat([
            data["new_hospitalizations"][:start_day],
            pd.Series(new_hospitalizations[start_day:])
        ]) if data is not None else new_hospitalizations,
        "new_icu": pd.concat([
            data["new_icu"][:start_day],
            pd.Series(new_icu[start_day:])
        ]) if data is not None else new_icu,
        "new_deaths": pd.concat([
            data["new_deaths"][:start_day],
            pd.Series(new_deaths[start_day:])
        ]) if data is not None else new_deaths,
        "new_recoveries": pd.concat([
            data["new_recoveries"][:start_day],
            pd.Series(new_recoveries[start_day:])
        ]) if data is not None else new_recoveries,
        "rate_new_infected": pd.concat([
            data["rate_new_infected"][:start_day],
            pd.Series(flow_new_exposed[start_day:])
        ]) if data is not None else flow_new_exposed,
        "rate_I_to_H": pd.concat([
            data["rate_I_to_H"][:start_day],
            pd.Series(flow_I_to_H[start_day:])
        ]) if data is not None else flow_I_to_H,
        "rate_I_to_R": pd.concat([
            data["rate_I_to_R"][:start_day],
            pd.Series(flow_I_to_R[start_day:])
        ]) if data is not None else flow_I_to_R,
        "rate_H_to_C": pd.concat([
            data["rate_I_to_R"][:start_day],
            pd.Series(flow_H_to_C[start_day:])
        ]) if data is not None else flow_H_to_C,
        "flow_C_to_D": pd.concat([
            data["flow_C_to_D"][:start_day],
            pd.Series(flow_C_to_D[start_day:])
        ]) if data is not None else flow_C_to_D,
    }

    # Перед созданием DataFrame преобразовать все Series в numpy arrays
    for key in new_data:
        if hasattr(new_data[key], 'values'):
            new_data[key] = new_data[key].values  # Из Series в numpy array

    df = pd.DataFrame(new_data)

    return df
