import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Optional

from botocore.utils import switch_host_with_param

from src.core.models import SEIRHCDParams, HospitalConfig, Patient
from src.utils.utils import sample_arrivals, now_str
from src.des.des_model import DES
from src.sd.seir_model import simulate_seir_hcd


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_two_way(
    init_params: SEIRHCDParams,
    hospitals_cfg: List[HospitalConfig],
    days: int,
    observed_new_hosp: Optional[np.ndarray] = None,
    calibrate: bool = True
):
    rng = np.random.RandomState(12345)
    des = DES(hospitals_cfg, rng_seed=12345)
    params = init_params

    # calibrate if requested and data provided
    # if calibrate and (observed_new_hosp is not None) and observed_new_hosp.sum() > 0:
    #     use_days = min(len(observed_new_hosp), max(30, days // 3))
    #     params = calibrate_seir(params, observed_new_hosp[:use_days], days=use_days)
    #     print(f"Calibrated: beta={params.beta:.4f}, hosp_rate={params.hosp_rate:.4f}")

    # logs
    logs = {"day": [], "today_hosp": [], "today_inc": [], "today_deaths": [], "today_admitted": [], "today_rejected": [], "sd_expected": [], "final_expected": [], "admitted": [], "rejected": [], "deaths": [], "population": []}
    pid = 0

    beta_modifier = 1.0

    def beta_time_fn(ti, beta):
        return beta * beta_modifier

    for day in range(days):
        print(f"day: {day}")
        seir_params_today = SEIRHCDParams(
            population=params.population,
            beta=params.beta,
            sigma=params.sigma,
            gamma=params.gamma,
            gamma_h=params.gamma_h,
            gamma_c=params.gamma_c,
            mu_c=params.mu_c,
            alpha_h=params.alpha_h,
            alpha_c=params.alpha_c,
            p_hosp=params.p_hosp,
            p_icu=params.p_icu,
            p_death=params.p_death,
            initial_exposed=params.initial_exposed,
            initial_infectious=params.initial_infectious
        )

        # seir_df = simulate_seir_hcd(seir_params_today, days=1)
        # mask = (seir_df["t"].astype(int) == day)
        # expected_today = float(seir_df.loc[mask, "new_hospitalizations"].values[0]) if mask.any() else 0.0
        # expected_icu_today = float(seir_df.loc[mask, "new_icu"].values[0]) if mask.any() else 0.0

        seir_df = simulate_seir_hcd(seir_params_today, days=day, beta_time_fn=beta_time_fn)

        expected_today = seir_df["new_hospitalizations"].iloc[-1]
        expected_icu_today = seir_df["new_icu"].iloc[-1]

        # Создаем запись пациентов на день
        events = sample_arrivals(expected_today, expected_icu_today, rng)
        today_hosp = 0
        today_inc = 0
        today_deaths = 0
        today_admitted = 0
        today_rejected = 0
        for ev in events:
            pid += 1
            los = rng.gamma(shape=2.0, scale=5.0) if ev["severity"] == "icu" else rng.gamma(shape=2.0, scale=3.0)
            p = Patient(id=pid, absolute_time=day + ev["time_frac"], severity=ev["severity"], los=max(1.0, los))
            p = des.attempt_admit(p, max_wait=0.5)

            if p.severity == "ward":
                today_hosp += 1
            elif p.severity == "icu":
                today_inc += 1
            if p.died:
                today_deaths += 1
            if p.admitted:
                today_admitted += 1
            else:
                today_rejected += 1



        # Получаем метрики
        daily = des.daily_metrics()
        admitted = int(daily["admitted"].sum())
        rejected = int(daily["rejected"].sum())
        deaths = int(daily["deaths"].sum())

        # 1) Смертность сокращает численность населения (вычитается из N)
        # if deaths > 0:
        #     params.population = max(0, params.population - deaths)

        # 2) high rejection => reduce beta_modifier (behavioural response)
        # overload = rejected / max(1.0, max(1.0, expected_today))
        # if overload > 0.1:
        #     beta_modifier *= max(0.6, 1.0 - 0.25 * min(1.0, overload))
        # else:
        #     beta_modifier += (1.0 - beta_modifier) * 0.02
        #
        # # 3) При тяжёлых стойких отторжениях увеличивается частота госпитализаций (ухудшается клиническое течение)
        # if rejected > 0:
        #     params.p_hosp = min(0.5, params.p_hosp * (1.0 + 0.005 * rejected))

        # Логирование
        logs["day"].append(day)
        logs["today_hosp"].append(today_hosp)
        logs["today_inc"].append(today_inc)
        logs["today_deaths"].append(today_deaths)
        logs["today_admitted"].append(today_admitted)
        logs["today_rejected"].append(today_rejected)
        logs["sd_expected"].append(expected_today)
        logs["final_expected"].append(expected_today)
        logs["admitted"].append(admitted)
        logs["rejected"].append(rejected)
        logs["deaths"].append(deaths)
        logs["population"].append(params.population)



    # postprocess
    log_df = pd.DataFrame(logs)

    plt.figure(figsize=(8, 5))
    # plt.plot(log_df["day"], log_df["today_hosp"], label="hospitalizations")
    # plt.plot(log_df["day"], log_df["today_inc"], label="icu hospitalizations")
    plt.plot(log_df["day"], log_df["today_deaths"], label="deaths")
    plt.plot(log_df["day"], log_df["today_admitted"], label="admitted")
    plt.plot(log_df["day"], log_df["today_rejected"], label="rejected")
    plt.legend()
    plt.show()

    # Создаем фигуру с несколькими субплогами
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Анализ моделирования эпидемии', fontsize=16, fontweight='bold')

    # 1. График ожидаемых случаев vs фактические
    axes[0, 0].plot(log_df['day'], log_df['sd_expected'], label='Ожидаемые случаи сегодня', linewidth=2)
    axes[0, 0].plot(log_df['day'], log_df['final_expected'], label='Финальные ожидаемые', linewidth=2, linestyle='--')
    axes[0, 0].set_title('Ожидаемые случаи заболевания')
    axes[0, 0].set_xlabel('День')
    axes[0, 0].set_ylabel('Количество случаев')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # 2. График госпитализаций и отказов
    axes[0, 1].plot(log_df['day'], log_df['admitted'], label='Госпитализировано', linewidth=2)
    axes[0, 1].plot(log_df['day'], log_df['rejected'], label='Отказано', linewidth=2)
    axes[0, 1].set_title('Госпитализации и отказы')
    axes[0, 1].set_xlabel('День')
    axes[0, 1].set_ylabel('Количество людей')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # 3. График смертности
    axes[0, 2].plot(log_df['day'], log_df['deaths'], label='Смерти', color='red', linewidth=2)
    axes[0, 2].set_title('Динамика смертности')
    axes[0, 2].set_xlabel('День')
    axes[0, 2].set_ylabel('Количество смертей')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)

    # 4. Накопительные показатели
    cumulative_admitted = log_df['admitted']
    cumulative_rejected = log_df['rejected']
    cumulative_deaths = log_df['deaths']

    axes[1, 0].plot(log_df['day'], cumulative_admitted, label='Всего госпитализировано', linewidth=2)
    axes[1, 0].plot(log_df['day'], cumulative_rejected, label='Всего отказано', linewidth=2)
    axes[1, 0].plot(log_df['day'], cumulative_deaths, label='Всего смертей', linewidth=2)
    axes[1, 0].set_title('Накопительные показатели')
    axes[1, 0].set_xlabel('День')
    axes[1, 0].set_ylabel('Количество людей')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # 5. Доля отказов от госпитализаций
    rejection_rate = log_df['rejected'] / (log_df['admitted'] + log_df['rejected'] + 1e-8) * 100
    axes[1, 1].plot(log_df['day'], rejection_rate, label='Доля отказов (%)', color='orange', linewidth=2)
    axes[1, 1].set_title('Доля отказов в госпитализации')
    axes[1, 1].set_xlabel('День')
    axes[1, 1].set_ylabel('Процент отказов')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # 6. Соотношение показателей к населению
    population = log_df['population'].iloc[0]
    axes[1, 2].plot(log_df['day'], log_df['admitted'] / population * 100, label='Госпитализировано (% населения)',
                    alpha=0.7)
    axes[1, 2].plot(log_df['day'], log_df['deaths'] / population * 100, label='Смерти (% населения)', alpha=0.7)
    axes[1, 2].set_title('Показатели относительно населения')
    axes[1, 2].set_xlabel('День')
    axes[1, 2].set_ylabel('Процент от населения')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    patients = []

    for hname in des.metrics:
        patients.extend(des.metrics[hname]["patients"])

    patients_df = pd.DataFrame(patients)
    ts = now_str()
    log_path = os.path.join(RESULTS_DIR, f"log_daily.csv")
    patients_path = os.path.join(RESULTS_DIR, f"patients.csv")
    log_df.to_csv(log_path, index=False)
    patients_df.to_csv(patients_path, index=False)

    # plot overview
    plt.figure(figsize=(10,4))
    plt.plot(log_df["day"], log_df["sd_expected"], label="SD expected hosp")
    plt.plot(log_df["day"], log_df["final_expected"], label="final expected")
    plt.plot(log_df["day"], log_df["admitted"], label="admitted")
    plt.plot(log_df["day"], log_df["rejected"], label="rejected")
    plt.plot(log_df["day"], log_df["deaths"], label="deaths")
    plt.xlabel("day"); plt.ylabel("counts"); plt.legend(); plt.grid(True); plt.title("Two-way SD<->DES dynamics")
    out_png = os.path.join(RESULTS_DIR, f"overview.png")
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()
    return {"log_csv": log_path, "patients_csv": patients_path, "overview_png": out_png}