from src.RL.env import HospitalEnv

epidemic_data = [
    {'hosp_real': 50, 'inc_real': 10, 'admitted': 60, 'infection': 200, 'rejected': 0, 'deaths_real': 2},
    {'hosp_real': 70, 'inc_real': 15, 'admitted': 80, 'infection': 250, 'rejected': 0, 'deaths_real': 3},
    {'hosp_real': 40, 'inc_real': 8, 'admitted': 45, 'infection': 150, 'rejected': 0, 'deaths_real': 1},
    {'hosp_real': 40, 'inc_real': 8, 'admitted': 45, 'infection': 150, 'rejected': 0, 'deaths_real': 1},
    {'hosp_real': 40, 'inc_real': 8, 'admitted': 45, 'infection': 150, 'rejected': 0, 'deaths_real': 1},
]

hospital_profiles = {
    "HOSP_A": {"budget": 5_000_000.0, "beds": 40, "vents": 5, "costs": {
        "bed_purchase": 100_000.0, "vent_purchase": 200_000.0,
        "staff_per_day": 10_000.0, "patient_per_day": 300.0,
        "bed_rent_per_day": 50.0, "vent_rent_per_day": 200.0}},
    "HOSP_B": {"budget": 2_000_000.0, "beds": 30, "vents": 3, "costs": {
        "bed_purchase": 90_000.0, "vent_purchase": 180_000.0,
        "staff_per_day": 8_000.0, "patient_per_day": 250.0,
        "bed_rent_per_day": 40.0, "vent_rent_per_day": 180.0}},
}

env = HospitalEnv(epidemic_data, hospital_profiles)
obs, info = env.reset(seed=42)
print("obs.shape =", obs.shape)
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
print("reward:", reward, "truncated:", truncated)
env.render()
