from matplotlib import pyplot as plt


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