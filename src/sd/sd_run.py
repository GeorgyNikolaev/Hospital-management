import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.core.config import settings
from src.core.models import SEIRHCDParams
from src.sd.seir_model import simulate_seir_hcd
from src.utils.utils import make_params_consistent


def load_real_data(csv_path, region: str):
    df = pd.read_csv(csv_path, parse_dates=["date"])

    df_region = df[df["location"] == region].copy()
    df_region.sort_values("date", inplace=True)

    required_columns = ["date", "new_deaths", "hosp_patients", "icu_patients"]
    missing = [c for c in required_columns if c not in df_region.columns]
    if missing:
        raise ValueError(f"В данных отсутствуют столбцы: {missing}")

    df_region.reset_index(drop=True, inplace=True)
    return df_region


def evaluate_model(real_df: pd.DataFrame, model_df: pd.DataFrame):
    merged = real_df.merge(model_df, left_index=True, right_index=True, suffixes=('_real', '_model'))

    metrics = {}
    for col_real, col_model in [
        ("new_deaths_real", "new_deaths_model"),
        ("hosp_patients", "H"),
        ("icu_patients", "C")
    ]:
        mse = np.mean((merged[col_real] - merged[col_model])**2)
        metrics[col_model] = mse

    return metrics, merged


def plot_results(merged: pd.DataFrame):
    fig, axes = plt.subplots(3, 1, figsize=(12, 16))

    metrics_map = [
        ("hosp_patients", "H", "Госпитализированные"),
        ("icu_patients", "C", "Пациенты в ИТ"),
        ("new_deaths_real", "new_deaths_model", "Смерти в день")
    ]

    for ax, (real_col, model_col, title) in zip(axes, metrics_map):
        ax.plot(merged.index, merged[real_col], label="Фактические", linewidth=2)
        ax.plot(merged.index, merged[model_col], linestyle="--", label="Модель")
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()


import matplotlib.pyplot as plt
import seaborn as sns


def plot_seirhcd_results(results_df, title="SEIR-HCD Model Simulation"):
    """
    Построение комплексного графика с состояниями и потоками
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # График 1: Основные состояния SEIHCRD
    ax1.plot(results_df['t'], results_df['S'], label='Susceptible (S)', linewidth=2)
    ax1.plot(results_df['t'], results_df['E'], label='Exposed (E)', linewidth=2)
    ax1.plot(results_df['t'], results_df['I'], label='Infectious (I)', linewidth=2)
    ax1.plot(results_df['t'], results_df['H'], label='Hospitalized (H)', linewidth=2)
    ax1.plot(results_df['t'], results_df['C'], label='Critical (C)', linewidth=2)
    ax1.plot(results_df['t'], results_df['R'], label='Recovered (R)', linewidth=2)
    ax1.plot(results_df['t'], results_df['D'], label='Deceased (D)', linewidth=2)

    ax1.set_title(f'{title} - States', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of People', fontsize=12)
    ax1.set_xlabel('Time (days)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Потоки (новые случаи)
    ax2.plot(results_df['t'], results_df['new_infected'],
             label='New Infections', linewidth=2, color='red')
    ax2.plot(results_df['t'], results_df['new_hospitalizations'],
             label='New Hospitalizations', linewidth=2, color='orange')
    ax2.plot(results_df['t'], results_df['new_icu'],
             label='New ICU Cases', linewidth=2, color='purple')
    ax2.plot(results_df['t'], results_df['new_deaths'],
             label='New Deaths', linewidth=2, color='black')

    ax2.set_title('Daily Flows', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of People per Day', fontsize=12)
    ax2.set_xlabel('Time (days)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# Использование
# results = simulate_seir_hcd(params, days=180)
# plot_seirhcd_results(results)


if __name__ == "__main__":
    region = "Australia"
    data_path = "D:/MEPHI/Simulation_modeling/data/owid-covid-data.csv"

    # real_data = load_real_data(data_path, region)
    # days = len(real_data)

    params = make_params_consistent(
        population=settings.POPULATION,
        sigma=settings.SIGMA,
        gamma=settings.GAMMA,
        R0=settings.R0,
        p_hosp=settings.P_HOSP,
        p_icu=settings.P_INC,
        p_death=settings.P_DEATH,
        hosp_duration=settings.HOSP_DURATION,
        icu_duration=settings.INC_DURATION,
        initial_exposed=settings.INITIAL_EXPOSED,
        initial_infectious=settings.INITIAL_INFECTIOUS
    )

    results_df = simulate_seir_hcd(params=params, days=settings.DAYS)

    plot_seirhcd_results(results_df)

    # Основные показатели
    total_population = params.population
    final_susceptible = results_df['S'].iloc[-1]
    final_recovered = results_df['R'].iloc[-1]
    final_deaths = results_df['D'].iloc[-1]

    # Пиковые значения
    peak_infectious_idx = results_df['I'].idxmax()
    peak_infectious_time = results_df['t'][peak_infectious_idx]
    peak_hosp_idx = results_df['H'].idxmax()
    peak_hosp_time = results_df['t'][peak_hosp_idx]

    print(f"\n📈 ПИКОВЫЕ НАГРУЗКИ:")
    print(f"   Пик заразных: {results_df['I'].max():,.0f} чел. (день {peak_infectious_time:.0f})")
    print(f"   Пик госпитализаций: {results_df['H'].max():,.0f} чел. (день {peak_hosp_time:.0f})")
    print(f"   Пик в реанимации: {results_df['C'].max():,.0f} чел.")
    print(f"   Макс. новых заражений в день: {results_df['new_infected'].max():,.0f} чел.")

    # Суммарные потоки
    total_infected = results_df['new_infected'].sum()
    total_hospitalizations = results_df['new_hospitalizations'].sum()
    total_icu = results_df['new_icu'].sum()
    total_deaths_flow = results_df['new_deaths'].sum()

    print(f"\n📊 СУММАРНЫЕ ПОТОКИ:")
    print(f"   Всего заражений: {total_infected:,.0f} чел.")
    print(f"   Всего госпитализаций: {total_hospitalizations:,.0f} чел.")
    print(f"   Всего в реанимации: {total_icu:,.0f} чел.")
    print(f"   Всего смертей: {total_deaths_flow:,.0f} чел.")

    # Проценты от зараженных
    print(f"\n📋 СТРУКТУРА ЗАБОЛЕВАНИЯ:")
    print(f"   Госпитализировано: {total_hospitalizations / total_infected * 100:.1f}% от зараженных")
    print(f"   В реанимации: {total_icu / total_infected * 100:.1f}% от зараженных")
    print(f"   Умерло: {total_deaths_flow / total_infected * 100:.2f}% от зараженных")

    # Временные параметры
    print(f"\n⏰ ВРЕМЕННЫЕ ХАРАКТЕРИСТИКИ:")
    print(f"   Длительность эпидемии: {results_df['t'].iloc[-1]:.0f} дней")
    print(f"   Время до пика: {peak_infectious_time:.0f} дней")
    print(f"   Задержка пика госпитализаций: {peak_hosp_time - peak_infectious_time:.1f} дней")

    # R0 и эффективность
    print(f"\n🔬 ЭПИДЕМИОЛОГИЧЕСКИЕ ПАРАМЕТРЫ:")
    print(f"   Базовое R₀: {params.beta / params.gamma:.2f}")
    print(f"   Длительность инкубации: {1 / params.sigma:.1f} дней")
    print(f"   Длительность заразности: {1 / params.gamma:.1f} дней")

    # metrics, df_merged = evaluate_model(real_data, df_model)
    # print("MSE метрики для проверки точности:")
    # print(metrics)
    #
    # plot_results(df_merged)
