import os

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

from src.des.des_model import DES
from src.utils.utils import now_str
from src.core.models import SEIRHCDParams

RESULTS_DIR = "results/"
os.makedirs(RESULTS_DIR, exist_ok=True)

def plot_SD_results(results_df, save_path: str):
    """Построение комплексного графика с состояниями и потоками"""
    os.makedirs(RESULTS_DIR + save_path, exist_ok=True)
    title = "SEIR-HCD Model Simulation"
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # График 1: Основные состояния SEIHCRD
    ax1.plot(results_df['day'], results_df['S'], label='Susceptible (S)', linewidth=2)
    ax1.plot(results_df['day'], results_df['E'], label='Exposed (E)', linewidth=2)
    ax1.plot(results_df['day'], results_df['I'], label='Infectious (I)', linewidth=2)
    ax1.plot(results_df['day'], results_df['H'], label='Hospitalized (H)', linewidth=2)
    ax1.plot(results_df['day'], results_df['C'], label='Critical (C)', linewidth=2)
    ax1.plot(results_df['day'], results_df['R'], label='Recovered (R)', linewidth=2)
    ax1.plot(results_df['day'], results_df['D'], label='Deceased (D)', linewidth=2)

    ax1.set_title(f'{title} - States', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of People', fontsize=12)
    ax1.set_xlabel('Time (days)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # График 2: Потоки (новые случаи)
    ax2.plot(results_df['day'], results_df['new_infected'],
             label='New Infections', linewidth=2, color='red')
    ax2.plot(results_df['day'], results_df['new_hospitalizations'],
             label='New Hospitalizations', linewidth=2, color='orange')
    ax2.plot(results_df['day'], results_df['new_icu'],
             label='New ICU Cases', linewidth=2, color='purple')
    ax2.plot(results_df['day'], results_df['new_deaths'],
             label='New Deaths', linewidth=2, color='black')

    ax2.set_title('Daily Flows', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of People per Day', fontsize=12)
    ax2.set_xlabel('Time (days)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    out_png = os.path.join(RESULTS_DIR + save_path, f"1.png")
    plt.savefig(out_png, dpi=150)
    plt.show()
    plt.close()

def plot_SD_DES_results(log_df: pd.DataFrame, save_path: str):
    """Выводим графики для SD <-> DES модели"""
    os.makedirs(RESULTS_DIR + save_path, exist_ok=True)

    # Создаем фигуру с несколькими субплогами
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle('Анализ моделирования эпидемии', fontsize=16, fontweight='bold')

    axes[0, 0].plot(log_df["day"], log_df["admitted_hosp"], label="Принятые госпитализированные")
    axes[0, 0].plot(log_df["day"], log_df["admitted_icu"], label="Принятые в ICU")
    axes[0, 0].plot(log_df["day"], log_df["rejected_hosp"], label="Отказы в госпитализации")
    axes[0, 0].plot(log_df["day"], log_df["rejected_icu"], label="Отказы в ICU")
    axes[0, 0].plot(log_df["day"], log_df["deaths_hosp"], label="Умершие госпитализированные")
    axes[0, 0].plot(log_df["day"], log_df["deaths_icu"], label="Увершие в ICU")
    axes[0, 0].legend()

    axes[0, 1].plot(log_df["day"], log_df["admitted"], label="Принятые")
    axes[0, 1].plot(log_df["day"], log_df["rejected"], label="Отказы")
    axes[0, 1].plot(log_df["day"], log_df["deaths"], label="Умершие")
    axes[0, 1].legend()

    # 3. График смертности
    axes[1, 0].plot(log_df['day'], log_df['deaths'], label='Смерти реальный', linewidth=2)
    axes[1, 0].plot(log_df['day'], log_df['deaths_expected'], label='Смерти ожидаемые', linewidth=2)
    axes[1, 0].set_title('Динамика смертности')
    axes[1, 0].set_xlabel('День')
    axes[1, 0].set_ylabel('Количество смертей')
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

    plt.tight_layout()

    out_png = os.path.join(RESULTS_DIR + save_path, f"2.png")
    plt.savefig(out_png)
    plt.show()
    plt.close()

def plot_RL_results(metrics_dict, save_path: str):
    """
        Построение графиков из словаря метрик
        """
    os.makedirs(RESULTS_DIR + save_path, exist_ok=True)

    days = metrics_dict['day']

    # Создаем сетку графиков
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('Анализ работы системы здравоохранения', fontsize=16, fontweight='bold')

    # 1. График инфекций и ожидаемых показателей
    # ax1 = axes[0, 0]
    # ax1.plot(days, metrics_dict['infection'], 'r-', label='Инфекции', linewidth=2)
    # ax1.plot(days, metrics_dict['hosp_expected'], 'g--', label='Ожидаемые госпитализации', alpha=0.7)
    # ax1.plot(days, metrics_dict['icu_expected'], 'b--', label='Ожидаемые ICU', alpha=0.7)
    # ax1.plot(days, metrics_dict['deaths_expected'], 'k--', label='Ожидаемые смерти', alpha=0.7)
    # ax1.set_title('Эпидемиологическая ситуация')
    # ax1.set_xlabel('Дни')
    # ax1.set_ylabel('Количество')
    # ax1.legend()
    # ax1.grid(True, alpha=0.3)

    # 2. График госпитализаций и отказов
    ax2 = axes[0, 0]
    ax2.plot(days, metrics_dict['admitted'], 'g-', label='Принято всего', linewidth=2)
    ax2.plot(days, metrics_dict['admitted_hosp'], 'b-', label='Принято в стационар', alpha=0.8)
    ax2.plot(days, metrics_dict['admitted_icu'], 'r-', label='Принято в ICU', alpha=0.8)
    ax2.plot(days, metrics_dict['rejected'], 'k-', label='Отказов всего', linewidth=2)
    ax2.plot(days, metrics_dict['rejected_hosp'], 'k--', label='Отказов стационар', alpha=0.7)
    ax2.plot(days, metrics_dict['rejected_icu'], 'k:', label='Отказов ICU', alpha=0.7)
    ax2.set_title('Госпитализации и отказы')
    ax2.set_xlabel('Дни')
    ax2.set_ylabel('Количество')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. График смертности
    ax3 = axes[0, 1]
    ax3.plot(days, metrics_dict['deaths'], 'k-', label='Смерти всего', linewidth=2)
    ax3.plot(days, metrics_dict['deaths_hosp'], 'r-', label='Смерти в стационаре', alpha=0.8)
    ax3.plot(days, metrics_dict['deaths_icu'], 'b-', label='Смерти в ICU', alpha=0.8)
    ax3.plot(days, metrics_dict['deaths_expected'], 'g--', label='Ожидаемые смерти', alpha=0.6)
    ax3.set_title('Смертность')
    ax3.set_xlabel('Дни')
    ax3.set_ylabel('Количество')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. График использования коек
    ax4 = axes[1, 0]
    # Расчет доступных коек (общие - законсервированные)
    available_beds = np.array(metrics_dict['beds'])
    available_icu = np.array(metrics_dict['icu'])

    ax4.plot(days, available_beds, 'b-', label='Доступные койки', linewidth=2)
    ax4.plot(days, metrics_dict['occupied_beds'], 'r-', label='Занятые койки', linewidth=2)
    ax4.plot(days, available_icu, 'c-', label='Доступные ICU', linewidth=2)
    ax4.plot(days, metrics_dict['occupied_icu'], 'm-', label='Занятые ICU', linewidth=2)
    ax4.fill_between(days, metrics_dict['occupied_beds'], available_beds, alpha=0.3, color='blue',
                     label='Свободные койки')
    ax4.fill_between(days, metrics_dict['occupied_icu'], available_icu, alpha=0.3, color='cyan', label='Свободные ICU')
    ax4.set_title('Использование коечного фонда')
    ax4.set_xlabel('Дни')
    ax4.set_ylabel('Количество коек')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # 5. График финансовых показателей
    ax5 = axes[2, 0]
    ax5.plot(days, metrics_dict['budget'], 'g-', label='Бюджет', linewidth=2)
    # ax5.fill_between(days, metrics_dict['expenses'], metrics_dict['budget'],
    #                  where=np.array(metrics_dict['budget']) >= np.array(metrics_dict['expenses']),
    #                  alpha=0.3, color='green', label='Профицит')
    # ax5.fill_between(days, metrics_dict['expenses'], metrics_dict['budget'],
    #                  where=np.array(metrics_dict['budget']) < np.array(metrics_dict['expenses']),
    #                  alpha=0.3, color='red', label='Дефицит')
    ax5.set_title('Бюджет')
    ax5.set_xlabel('Дни')
    ax5.set_ylabel('Денежные единицы')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 5. График финансовых показателей
    ax5 = axes[2, 1]
    ax5.plot(days, metrics_dict['expenses'], 'r-', label='Расходы', linewidth=2)
    # ax5.fill_between(days, metrics_dict['expenses'], metrics_dict['budget'],
    #                  where=np.array(metrics_dict['budget']) >= np.array(metrics_dict['expenses']),
    #                  alpha=0.3, color='green', label='Профицит')
    # ax5.fill_between(days, metrics_dict['expenses'], metrics_dict['budget'],
    #                  where=np.array(metrics_dict['budget']) < np.array(metrics_dict['expenses']),
    #                  alpha=0.3, color='red', label='Дефицит')
    ax5.set_title('Расходы')
    ax5.set_xlabel('Дни')
    ax5.set_ylabel('Денежные единицы')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 6. График эффективности системы
    ax6 = axes[1, 1]
    # Расчет ключевых показателей эффективности
    rejection_rate = []
    mortality_rate = []
    bed_utilization = []

    for i in range(len(days)):
        total_patients = metrics_dict['admitted'][i] + metrics_dict['rejected'][i]
        rejection_rate.append(metrics_dict['rejected'][i] / total_patients if total_patients > 0 else 0)
        mortality_rate.append(
            metrics_dict['deaths'][i] / metrics_dict['admitted'][i] if metrics_dict['admitted'][i] > 0 else 0)
        bed_utilization.append(
            metrics_dict['occupied_beds'][i] / metrics_dict['beds'][i] if metrics_dict['beds'][i] > 0 else 0)

    ax6.plot(days, rejection_rate, 'r-', label='Уровень отказов', linewidth=2)
    ax6.plot(days, mortality_rate, 'k-', label='Уровень смертности', linewidth=2)
    ax6.plot(days, bed_utilization, 'b-', label='Загрузка коек', linewidth=2)
    ax6.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='Целевой уровень отказов (10%)')
    ax6.axhline(y=0.8, color='b', linestyle='--', alpha=0.5, label='Оптимальная загрузка (80%)')
    ax6.set_title('Ключевые показатели эффективности')
    ax6.set_xlabel('Дни')
    ax6.set_ylabel('Доля')
    ax6.set_ylim(0, 1)
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()

    # if save_path:
    #     plt.savefig(save_path, dpi=300, bbox_inches='tight')

    out_png = os.path.join(RESULTS_DIR + save_path, f"3.png")
    plt.savefig(out_png)
    plt.show()
    plt.close()

def plot_DES_vs_TTM_vs_RL(des_log, ttm_logs, rl_log, save_path: str = ""):
    os.makedirs(RESULTS_DIR + save_path, exist_ok=True)

    days = des_log['day']

    from statsmodels.nonparametric.smoothers_lowess import lowess

    def smooth_lowess(x, y, frac=0.3):
        """
        LOWESS сглаживание
        frac - доля данных для локальной регрессии (0.1-0.5)
        Чем меньше frac, тем более извилистая кривая
        Чем больше frac, тем более гладкая
        """
        smoothed = lowess(y, x, frac=frac, return_sorted=False)
        return smoothed

    # Красивые цвета (современная палитра)
    COLORS = {
        'des': '#FFB3B3',  # бледно-розовый (сырые данные)
        'ttm': '#bd7ebe',  # бледно-голубой (сырые данные)
        'rl': '#ffb55a',  # бледно-лавандовый (сырые данные)
        'trend_des': '#E63946',  # насыщенный красный (тренд)
        'trend_ttm': '#9b19f5',  # темно-синий (тренд)
        'trend_rl': '#ffa300'  # фиолетовый (тренд)
    }

    # Создаем сетку графиков
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Static vs TTM vs RL', fontsize=16, fontweight='bold')

    ax1 = axes[0, 0]

    ax1.plot(days, ttm_logs['admitted'], color=COLORS['ttm'], alpha=0.4, linewidth=1.5, label='TTM (raw)')
    ax1.plot(days, rl_log['admitted'], color=COLORS['rl'], alpha=0.4, linewidth=1.5, label='RL (raw)')

    # Использование
    ttm_smooth = smooth_lowess(days, ttm_logs['admitted'], frac=0.2)
    rl_smooth = smooth_lowess(days, rl_log['admitted'], frac=0.2)

    # Сглаженные тренды (яркие)
    ax1.plot(days, ttm_smooth, color=COLORS['trend_ttm'], linewidth=3, linestyle='-', label='TTM (trend)')
    ax1.plot(days, rl_smooth, color=COLORS['trend_rl'], linewidth=3, linestyle='-', label='RL (trend)')

    # ax1.plot(days, des_log['admitted'], 'r-', label='Static', linewidth=2)
    # ax1.plot(days, ttm_logs['admitted'], 'b-', label='TTM', linewidth=2)
    # ax1.plot(days, rl_log['admitted'], 'g-', label='RL', linewidth=2)
    ax1.set_title('Принято', fontsize=12)
    ax1.set_xlabel('Дни', fontsize=10)
    ax1.set_ylabel('Люди', fontsize=10)
    ax1.legend(loc='best', framealpha=0.9)
    ax1.grid(True, alpha=0.3, linestyle='--')

    ax2 = axes[0, 1]

    ax2.plot(days, ttm_logs['rejected'], color=COLORS['ttm'], alpha=0.4, linewidth=1.5, label='TTM (raw)')
    ax2.plot(days, rl_log['rejected'], color=COLORS['rl'], alpha=0.4, linewidth=1.5, label='RL (raw)')

    # Использование
    ttm_smooth = smooth_lowess(days, ttm_logs['rejected'], frac=0.2)
    rl_smooth = smooth_lowess(days, rl_log['rejected'], frac=0.2)

    # Сглаженные тренды (яркие)
    ax2.plot(days, ttm_smooth, color=COLORS['trend_ttm'], linewidth=3, linestyle='-', label='TTM (trend)')
    ax2.plot(days, rl_smooth, color=COLORS['trend_rl'], linewidth=3, linestyle='-', label='RL (trend)')

    # ax2.plot(days, des_log['rejected'], 'r-', label='Static', linewidth=2)
    # ax2.plot(days, ttm_logs['rejected'], 'b-', label='TTM', linewidth=2)
    # ax2.plot(days, rl_log['rejected'], 'g-', label='RL', linewidth=2)
    ax2.set_title('Отказы', fontsize=12)
    ax2.set_xlabel('Дни', fontsize=10)
    ax2.set_ylabel('Люди', fontsize=10)
    ax2.legend(loc='best', framealpha=0.9)
    ax2.grid(True, alpha=0.3, linestyle='--')

    ax3 = axes[1, 0]
    ax3.plot(days, ttm_logs['deaths'], color=COLORS['ttm'], alpha=0.4, linewidth=1.5, label='TTM (raw)')
    ax3.plot(days, rl_log['deaths'], color=COLORS['rl'], alpha=0.4, linewidth=1.5, label='RL (raw)')

    # Использование
    ttm_smooth = smooth_lowess(days, ttm_logs['deaths'], frac=0.2)
    rl_smooth = smooth_lowess(days, rl_log['deaths'], frac=0.2)

    # Сглаженные тренды (яркие)
    ax3.plot(days, ttm_smooth, color=COLORS['trend_ttm'], linewidth=3, linestyle='-', label='TTM (trend)')
    ax3.plot(days, rl_smooth, color=COLORS['trend_rl'], linewidth=3, linestyle='-', label='RL (trend)')

    ax3.set_title('Смерти', fontsize=12)
    ax3.set_xlabel('Дни', fontsize=10)
    ax3.set_ylabel('Люди', fontsize=10)
    ax3.legend(loc='best', framealpha=0.9)
    ax3.grid(True, alpha=0.3, linestyle='--')

    ax4 = axes[1, 1]
    # Расчет доступных коек (общие - законсервированные)

    # ax4.plot(days, des_log["beds"], 'r-', label='Доступные койки Static', linewidth=1)
    ax4.plot(days, ttm_logs["beds"], 'b-', label='Доступные койки TTM', linewidth=1)
    ax4.plot(days, rl_log["beds"], 'g-', label='Доступные койки RL', linewidth=1)

    # ax4.plot(days, des_log["occupied_beds"], 'c-', label='Занятые койки Static', linewidth=1)
    ax4.plot(days, ttm_logs["occupied_beds"], 'm-', label='Занятые койки TTM', linewidth=1)
    ax4.plot(days, rl_log["occupied_beds"], 'k-', label='Занятые койки RL', linewidth=1)

    # ax4.fill_between(days, des_log["occupied_beds"], des_log["beds"], alpha=0.3, color='cyan',
    #                  label='Свободные койки DES')
    ax4.fill_between(days, ttm_logs["occupied_beds"], ttm_logs["beds"], alpha=0.3, color='yellow',
                     label='Свободные койки TTM')
    ax4.fill_between(days, rl_log["occupied_beds"], rl_log["beds"], alpha=0.3, color='blue',
                     label='Свободные койки RL')

    ax4.set_title('Использование коечного фонда')
    ax4.set_xlabel('Дни')
    ax4.set_ylabel('Количество коек')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = os.path.join(RESULTS_DIR + save_path, f"des_ttm_rl.png")
    plt.savefig(out_png)
    plt.show()
    plt.close()

def plot_RL_actions(rl_logs, save_path: str):
    os.makedirs(RESULTS_DIR + save_path, exist_ok=True)

    data_array = np.array(rl_logs["actions"])
    data_array = np.array(data_array.tolist())

    # Создаем график
    # Словарь соответствия числовых значений и текстовых меток
    action_labels = {
        0: 'Ничего не делать',
        1: 'Купить 5 койку',
        2: 'Купить 10 коек',
        3: 'Купить 1 аппарат ИВЛ',
        4: 'Купить 5 аппаратов ИВЛ',
        5: 'Законсервировать 5 коек',
        6: 'Законсервировать 10 коек',
        7: 'Законсервировать 1 аппарат ИВЛ',
        8: 'Законсервировать 5 аппаратов ИВЛ',
        9: 'Срочно выделить бюджет'
    }

    plt.figure(figsize=(10, 6))
    x = range(len(rl_logs["actions"]))

    # for i in range(data_array.shape[1]):
    for i in range(1):
        y = [subarray[i] for subarray in rl_logs["actions"]]
        plt.plot(x, y, 'o-', label=f'Больница {i + 1}', markersize=8)

    plt.xlabel('День')
    plt.title('График принятия решений больниц')

    # Устанавливаем метки на оси Y
    plt.yticks(ticks=list(action_labels.keys()), labels=list(action_labels.values()))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()  # Чтобы метки не обрезались
    out_png = os.path.join(RESULTS_DIR + save_path, f"actions.png")
    plt.savefig(out_png)
    plt.show()
    plt.close()

def save_SD_results(results_df):
    """Сохраняет результаты SD модели"""

def save_SD_DES_results(log_df: pd.DataFrame, des: DES):
    """Сохраняет результаты SD <-> DES модели"""
    patients = []

    for h in des.hospitals:
        patients.extend(h.patients)

    patients_df = pd.DataFrame(patients)
    ts = now_str()
    # log_path = os.path.join(RESULTS_DES_DIR, f"log_daily.csv")
    # patients_path = os.path.join(RESULTS_DES_DIR, f"patients.csv")
    # log_df.to_csv(log_path, index=False)
    # patients_df.to_csv(patients_path, index=False)

    # plot overview
    plt.figure(figsize=(10,4))
    plt.plot(log_df["day"], log_df["infection"], label="infection")
    plt.plot(log_df["day"], log_df["admitted"], label="admitted")
    plt.plot(log_df["day"], log_df["rejected"], label="rejected")
    plt.plot(log_df["day"], log_df["deaths"], label="deaths_real")
    plt.xlabel("day"); plt.ylabel("counts"); plt.legend(); plt.grid(True); plt.title("Two-way SD<->DES dynamics")
    # out_png = os.path.join(RESULTS_DES_DIR, f"overview.png")
    # plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def display_results(results_df):
    """
    Выводит агрегированные метрики сценария управления эпидемией.
    Функция устойчива к отсутствию отдельных столбцов.
    """

    def has(col):
        return col in results_df.columns

    def ssum(col):
        return results_df[col].sum() if has(col) else None

    def smax(col):
        return results_df[col].max() if has(col) else None

    def smean(col):
        return results_df[col].mean() if has(col) else None

    def idxmax(col):
        return results_df[col].idxmax() if has(col) else None

    def fmt(x):
        return f"{x:,.0f}" if x is not None else "N/A"

    # ================= ПИКОВЫЕ НАГРУЗКИ =================
    print("\n📈 ПИКОВЫЕ НАГРУЗКИ:")

    hosp_idx = idxmax('hosp_expected')
    if hosp_idx is not None:
        print(
            f"   Пик госпитализаций: "
            f"{fmt(results_df.loc[hosp_idx, 'hosp_expected'])} чел. "
            f"(день {results_df.loc[hosp_idx, 'day']:.0f})"
        )

    icu_idx = idxmax('icu_expected')
    if icu_idx is not None:
        print(
            f"   Пик ICU: "
            f"{fmt(results_df.loc[icu_idx, 'icu_expected'])} чел. "
            f"(день {results_df.loc[icu_idx, 'day']:.0f})"
        )

    # ================= СУММАРНЫЕ ИСХОДЫ =================
    print("\n📊 СУММАРНЫЕ ИСХОДЫ:")

    total_admitted = ssum('admitted')
    total_deaths = ssum('deaths')

    print(f"   Всего принято: {fmt(total_admitted)}")
    print(f"   Всего смертей: {fmt(total_deaths)}")

    # ================= ОТКАЗЫ =================
    print("\n🚨 ОТКАЗЫ В ПОМОЩИ:")

    rejected = ssum('rejected')
    rejected_hosp = ssum('rejected_hosp')
    rejected_icu = ssum('rejected_icu')

    print(f"   Всего отказов: {fmt(rejected)}")
    print(f"   └─ по койкам: {fmt(rejected_hosp)}")
    print(f"   └─ по ICU: {fmt(rejected_icu)}")

    if has('admitted') and has('rejected'):
        total_requests = results_df['admitted'].sum() + results_df['rejected'].sum()
        if total_requests > 0:
            print(f"   Доля отказов: {results_df['rejected'].sum() / total_requests:.2%}")

    # ================= ЗАГРУЗКА РЕСУРСОВ =================
    print("\n🏥 ЗАГРУЗКА РЕСУРСОВ:")

    if has('occupied_beds') and has('beds'):
        bed_util = results_df['occupied_beds'] / results_df['beds']
        print(f"   Койки — пик: {bed_util.max():.1%}")
        print(f"   Койки — средняя: {bed_util.mean():.1%}")

    if has('occupied_icu') and has('icu'):
        icu_util = results_df['occupied_icu'] / results_df['icu']
        print(f"   ICU — пик: {icu_util.max():.1%}")
        print(f"   ICU — средняя: {icu_util.mean():.1%}")

    # ================= СМЕРТНОСТЬ =================
    print("\n⚰️ СМЕРТНОСТЬ:")

    print(f"   Смерти в стационаре: {fmt(ssum('deaths_hosp'))}")
    print(f"   Смерти в ICU: {fmt(ssum('deaths_icu'))}")

    if has('hosp_expected') and has('icu_expected') and has('deaths'):
        total_inf = results_df['hosp_expected'].sum() + results_df['icu_expected'].sum()
        if total_inf > 0:
            print(f"   Летальность: {results_df['deaths'].sum() / total_inf:.2%}")

    # ================= ЭКОНОМИКА =================
    print("\n💰 ЭКОНОМИКА:")

    print(f"   Общие расходы: {fmt(ssum('expenses'))}")
    if has('budget'):
        print(f"   Финальный бюджет: {fmt(results_df['budget'].iloc[-1])}")

    if has('actions'):
        budget_requests = results_df['actions'].apply(lambda a: sum(x == 9 for x in a)).sum()
        print(f"   Запросы на выделение бюджета: {budget_requests}")

    if has('expenses') and has('deaths') and results_df['deaths'].sum() > 0:
        print(
            f"   Стоимость одной жизни: "
            f"{results_df['expenses'].sum() / (results_df['infection'].sum() - results_df['deaths'].sum()):,.0f}"
        )

        print(
            f"   Стоимость одной смерти: "
            f"{results_df['expenses'].sum() / results_df['deaths'].sum() :,.0f}"
        )

    print("\n" + "=" * 60)