# Код на Python для расчёта всех формул из задания.
# Комментарии на русском языке.
# Выполняет пошаговые вычисления и выводит результаты с пояснениями.
import math

import numpy as np
from matplotlib import pyplot as plt

# Входные данные (из вашего примера)
W_MW = 1400.0  # электрическая мощность, МВт
eta_percent = 34.0  # КПД, %
KIUM = 0.88  # коэффициент использования мощности (КИУМ)
x_percent = 4.8  # обогащение топлива, %
n = 2.5  # кратность перегрузок
price_C_per_MWh = 80.0  # отпускная цена электроэнергии, $/МВт·ч
hours_per_year = 365 * 24  # часы в году

# Стоимости и параметры топлива/фабрикации/разделения (в соответствии с данными)
China_tax = 75   # $/(МВт*час) стоимость электро энергии в Китае
C_F = 60.0       # $/кгU, стоимость природного урана (концентрата)
C_UF6 = 20.0     # $/кгU, стоимость конвертации в UF6
C_R = 70.0      # $/EPP, стоимость работы разделения за 1 EPP (ERP в тексте — ЕРР)
C_D = 10.0       # $/кгU, стоимость отвала (бедного урана)
C_fab = 360.0    # $/кгU, стоимость фабрикации ТВС
COYAT = 800.0   # $/кгU, СОЯТ (?? — здесь использую значение 1000 $/кгU как в тексте)
Y_OM_per_MWh = 18.0  # $/МВт·ч, O&M удельные затраты
K_billion = 4.2   # млрд $ (капитальные затраты)
K = K_billion * 1e9  # $
H = 0.25  # налог на прибыль
T_c = 7.0   # построечный период, лет
T_e = 60.0  # эксплуатационный срок, лет
s = 0.0668    # ставка дисконтирования (8%)

# Дополнительные данные для расчёта потребности в природном уране
c_percent = 0.711  # концентрация природного урана, %
y_percent = 0.2    # глубина отвала (отвального продукта), %

# Расчёты
# 1) Годовое производство электроэнергии E
E_MWh = W_MW * hours_per_year * KIUM

# 2) Годовая выручка V
V_USD = E_MWh * price_C_per_MWh

# 3) Глубина выгорания B (МВт·сут/кгU)
B_MWday_per_kg = 14.8 * x_percent * n / (n + 1.0)  # формула из текста

# 4) Тепловая мощность Q (МВт)
eta = eta_percent / 100.0
Q_MW = W_MW / eta

# 5) Ежегодная потребность в топливе P (кг/год)
# Формула: P = 365 * (Q * KIUM) / B  ; B в МВт·сут/кг -> Q [МВт] * 365 [сут] дает МВт·сут/год
P_kg_per_year = 365.0 * (Q_MW * KIUM) / B_MWday_per_kg

# 6) Потребность в природном уране F (кг/год) при заданных концентрациях
x = x_percent / 100.0
y = y_percent / 100.0
c = c_percent / 100.0
F_kg_per_year = P_kg_per_year * (x - y) / (c - y)

# 7) Масса отвала D (бедного урана)
D_kg_per_year = F_kg_per_year - P_kg_per_year

# 8) Функция Ф(z) = (1 - 2z) * ln(1 - z) / z
def Phi(z):
    if z <= 0 or z >= 1:
        return float('nan')
    return (1.0 - 2.0 * z) * math.log((1.0 - z) / z)

Phi_x = Phi(x)
Phi_y = Phi(y)
Phi_c = Phi(c)

# 9) Ежегодная потребность в работе разделения R (EPP/год)
R_EPP_per_year = P_kg_per_year * Phi_x + D_kg_per_year * Phi_y - F_kg_per_year * Phi_c

# 10) Стоимость обогащённого урана C_X по формуле:
# C_X = F/P * (C_F + C_UF6) + R/P * C_R + D/P * C_D
C_X_per_kgU = (F_kg_per_year / P_kg_per_year) * (C_F + C_UF6) + (R_EPP_per_year / P_kg_per_year) * C_R + (D_kg_per_year / P_kg_per_year) * C_D

# 11) Стоимость ТВС на 1 кг топлива C_TVS (включая фабрикацию)
C_TVS_per_kgU = C_X_per_kgU + C_fab

# 12) Топливная составляющая Y_T (годовая): YT = P * (C_TVS + COYAT)
Y_T_USD_per_year = P_kg_per_year * (C_TVS_per_kgU + COYAT)

# 13) Годовые O&M затраты Y_O&M = Y_OM_per_MWh * E
Y_OM_USD_per_year = Y_OM_per_MWh * E_MWh

# 14) Годовые эксплуатационные затраты Y = Y_O&M + Y_T
Y_total_USD_per_year = Y_OM_USD_per_year + Y_T_USD_per_year

# 15) IRR0 "идеального" проекта: ((1-H)*(V-Y))/K
IRR0 = ((1.0 - H) * (V_USD - Y_total_USD_per_year)) / K

# 16) коэффициенты φ_C и φ_E
# Функции приведения
def phi_C_func(s):
    return ((1.0 + s) ** T_c - 1.0) / (s * T_c)

def phi_E_func(s):
    return 1.0 - (1.0 + s) ** (-T_e)

phi_C = phi_C_func(s)
phi_E = phi_E_func(s)

# 17) эффективная ставка s_eff = s * φ_C / φ_E
s_eff = s * phi_C / phi_E

# 18) LCOE = (s_eff * K + (1-H) * Y) / ((1-H) * E)
def LCOE_per_MWh_def(s):
    phi_C = phi_C_func(s)
    phi_E = phi_E_func(s)
    s_eff = s * phi_C / phi_E
    LCOE = (s_eff * K + (1.0 - H) * Y_total_USD_per_year) / ((1.0 - H) * E_MWh)
    return LCOE

LCOE_per_MWh = LCOE_per_MWh_def(s)

# 19) Капитальная составляющая и топливная составляющая LCOE
capital_component_per_MWh = (s_eff * K) / ((1.0 - H) * E_MWh)
fuel_component_per_MWh = Y_T_USD_per_year / ((1.0 - H) * E_MWh)

# 20) Доля составляющих LCOE
capital_share = capital_component_per_MWh / LCOE_per_MWh
fuel_share = fuel_component_per_MWh / LCOE_per_MWh

# 21) Период окупаемости Θ (формула из текста)
# Θ = (-ln(1 - (s*φ_C)/IRR0 )) / ln(1 + s)
# проверим аргумент логарифма
denom_expr = 1.0 - (s * phi_C) / IRR0 if IRR0 != 0 else float('nan')
if denom_expr <= 0 or denom_expr >= 1:
    Theta_years = float('nan')
else:
    Theta_years = -math.log(denom_expr) / math.log(1.0 + s)

# Функция NPV(s) по формуле в задании (используем I_C = K * phi_C)
def NPV_of_s(s, price_C=price_C_per_MWh):
    phi_C = phi_C_func(s)
    phi_E = phi_E_func(s)
    V = E_MWh * price_C
    NPV = -K * phi_C + (1.0 - H) * (V - Y_total_USD_per_year) * phi_E / s
    return NPV

# 22) NPV расчет как в тексте:
NPV_USD = NPV_of_s(s)

# Вывод результатов (округлённо и с пояснениями)
def fmt(x, digits=3):
    return f"{x:,.{digits}f}"

print("Результаты расчётов:")
print(f"1) Годовое производство электроэнергии E = {fmt(E_MWh, 0)} МВт·ч/год")
print(f"2) Годовая выручка V = {fmt(V_USD,0)} $/год")
print(f"3) Глубина выгорания B = {fmt(B_MWday_per_kg,3)} МВт·сут/кгU")
print(f"4) Тепловая мощность Q = {fmt(Q_MW,3)} МВт")
print(f"5) Ежегодная потребность в топливе P = {fmt(P_kg_per_year,3)} кг/год")
print(f"6) Потребность в природном уране F = {fmt(F_kg_per_year,3)} кг/год")
print(f"7) Масса отвала D = {fmt(D_kg_per_year,3)} кг/год")
print(f"8) Функции Phi(x), Phi(y), Phi(c) = {fmt(Phi_x,3)}, {fmt(Phi_y,3)}, {fmt(Phi_c,3)}")
print(f"9) Ежегодная потребность в работе разделения R = {fmt(R_EPP_per_year,3)} ЕРР/год")
print(f"10) C_X (стоимость обогащённого урана) = {fmt(C_X_per_kgU,2)} $/кгU")
print(f"11) C_TVS (стоимость ТВС на 1 кг топлива) = {fmt(C_TVS_per_kgU,2)} $/кгU")
print(f"12) Топливная составляющая Y_T = {fmt(Y_T_USD_per_year,0)} $/год")
print(f"13) Y_O&M = {fmt(Y_OM_USD_per_year,0)} $/год")
print(f"14) Годовые эксплуатационные затраты Y = {fmt(Y_total_USD_per_year,0)} $/год")
print(f"15) IRR0 (идеальный) = {fmt(IRR0*100,4)} %/год")
print(f"16) phi_C = {fmt(phi_C,5)}, phi_E = {fmt(phi_E,5)}")
print(f"17) s_eff = {fmt(s_eff*100,4)} %/год")
print(f"18) LCOE = {fmt(LCOE_per_MWh,2)} $/МВт·ч")
print(f"    - Капитальная составляющая = {fmt(capital_component_per_MWh,2)} $/МВт·ч")
print(f"    - Топливная составляющая = {fmt(fuel_component_per_MWh,2)} $/МВт·ч")
print(f"    - Доля капитальной составляющей = {fmt(capital_share*100,2)} %")
print(f"    - Доля топливной составляющей = {fmt(fuel_share*100,2)} %")
print(f"19) Период окупаемости Θ = {fmt(Theta_years,3)} лет (если вычислим)")
print(f"20) NPV = {fmt(NPV_USD/1e9,6)} млрд $")

# Для удобства также возвращаем все значения в словаре (если нужно использовать в дальнейшем)
results = {
    'E_MWh': E_MWh,
    'V_USD': V_USD,
    'B_MWday_per_kg': B_MWday_per_kg,
    'Q_MW': Q_MW,
    'P_kg_per_year': P_kg_per_year,
    'F_kg_per_year': F_kg_per_year,
    'D_kg_per_year': D_kg_per_year,
    'Phi_x': Phi_x,
    'Phi_y': Phi_y,
    'Phi_c': Phi_c,
    'R_EPP_per_year': R_EPP_per_year,
    'C_X_per_kgU': C_X_per_kgU,
    'C_TVS_per_kgU': C_TVS_per_kgU,
    'Y_T_USD_per_year': Y_T_USD_per_year,
    'Y_OM_USD_per_year': Y_OM_USD_per_year,
    'Y_total_USD_per_year': Y_total_USD_per_year,
    'IRR0': IRR0,
    'phi_C': phi_C,
    'phi_E': phi_E,
    's_eff': s_eff,
    'LCOE_per_MWh': LCOE_per_MWh,
    'capital_component_per_MWh': capital_component_per_MWh,
    'fuel_component_per_MWh': fuel_component_per_MWh,
    'capital_share': capital_share,
    'fuel_share': fuel_share,
    'Theta_years': Theta_years,
    'NPV_USD': NPV_USD
}

results

# Интервал ставок дисконтирования для анализа
s_values = np.linspace(0.001, 0.20, 400)  # от 0.1% до 20%
NPV_values = np.array([NPV_of_s(s) for s in s_values])

# Более простой подход - отмечаем ближайшие точки
zero_crossings = np.where(np.diff(np.sign(NPV_values)))[0]

plt.figure(figsize=(8,5))
plt.plot(s_values*100, NPV_values)
plt.xlabel("Ставка дисконтирования s, %")
plt.ylabel("NPV, $")
plt.title("NPV как функция ставки дисконтирования s")
plt.grid(True)
plt.axhline(0, linestyle='--', color='red')

sign_changes = np.where(np.sign(NPV_values[:-1]) != np.sign(NPV_values[1:]))[0]
IRR_project = None
if len(sign_changes) > 0:
    idx = sign_changes[0]
    # линейная интерполяция для приближённого корня
    s1, s2 = s_values[idx], s_values[idx+1]
    npv1, npv2 = NPV_values[idx], NPV_values[idx+1]
    IRR_project = s1 - npv1 * (s2 - s1) / (npv2 - npv1)
    IRR_project_percent = IRR_project * 100.0
else:
    IRR_project_percent = None

plt.plot(IRR_project_percent, 0, 'ro', markersize=8)
if IRR_project_percent is not None:
    plt.annotate(f'IRR ≈ {IRR_project_percent:.2f}%',
                xy=(IRR_project_percent, 2),
                xytext=(IRR_project_percent + 1, 0.25*10**10),  # смещение текста от точки
                arrowprops=dict(arrowstyle='->', color='black'),
                fontsize=12,
                color='black')
# plt.show()

print(IRR_project_percent)

# 2) Рентабельность проекта NPV/I_C vs s
def profitability_ratio(s, price_C=price_C_per_MWh):
    phi_C = phi_C_func(s)
    I_C = K * phi_C
    NPV = NPV_of_s(s, price_C)
    return NPV / I_C

profit_ratios = np.array([profitability_ratio(s) for s in s_values])

plt.figure(figsize=(8,5))
plt.plot(s_values*100, profit_ratios)
plt.xlabel("Ставка дисконтирования s, %")
plt.ylabel("Рентабельность NPV / I_C (безразмерная)")
plt.title("Рентабельность проекта (NPV/I_C) как функция ставки s")
plt.grid(True)
# отметим уровни 1 и 2
plt.axhline(1.0, linestyle='--', color='red')
plt.axhline(2.0, linestyle='--', color='red')

# Найдём s при которых NPV/I_C = 1 и =2 (если в интервале)
def find_s_for_profit(target):
    idx = np.where(np.sign(profit_ratios - target)[:-1] != np.sign(profit_ratios - target)[1:])[0]
    if len(idx) == 0:
        return None
    i = idx[0]
    s1, s2 = s_values[i], s_values[i+1]
    p1, p2 = profit_ratios[i], profit_ratios[i+1]
    s_target = s1 + (target - p1) * (s2 - s1) / (p2 - p1)
    return s_target

s_for_1 = find_s_for_profit(1.0)
s_for_2 = find_s_for_profit(2.0)

plt.plot(s_for_1*100, 1, 'ro', markersize=5)
plt.annotate(f's ≈ {s_for_1*100:.2f}%',
                xy=(s_for_1*100, 1),
                xytext=(s_for_1*100 + 0.5, 1.2),  # смещение текста от точки
                fontsize=12,
                color='black')
plt.plot(s_for_2*100, 2, 'ro', markersize=5)
plt.annotate(f's ≈ {s_for_2*100:.2f}%',
                xy=(s_for_2*100, 2),
                xytext=(s_for_2*100 + 0.5, 2.2),  # смещение текста от точки
                fontsize=12,
                color='black')
# plt.show()


if IRR_project_percent is not None:
    s_choice = min(0.8 * IRR_project, s)  # пример выбора: 80% от проектной IRR или базовая
else:
    s_choice = s

s_choice_percent = s_choice * 100.0

# Построим NPV(C) на интервале цен от 20 до 200 $/MWh
C_values = np.linspace(20.0, 200.0, 300)
NPV_C_values = np.array([NPV_of_s(s_choice, price_C=c) for c in C_values])

plt.figure(figsize=(8,5))
plt.plot(C_values, NPV_C_values)
plt.xlabel("Отпускная цена электроэнергии C, $/МВт·ч")
plt.ylabel("NPV, $")
plt.title(f"NPV(C) при выбранной ставке дисконтирования s = {s_choice_percent:.2f} %")
plt.grid(True)

LCOE_choice = LCOE_per_MWh_def(s_choice)

plt.axhline(0.0, linestyle='--', color='red')
plt.plot(LCOE_choice, 0, 'ro', markersize=5)
plt.annotate(f'C ≈ {LCOE_choice:.2f}',
                xy=(LCOE_choice, 0),
                xytext=(LCOE_choice - 35, 0.05 * 10**10),  # смещение текста от точки
                fontsize=12,
                arrowprops=dict(arrowstyle='->', color='black'),
                color='black')
plt.plot(China_tax, 0.1 * 10**10, 'ro', markersize=5)
plt.annotate(f'C ≈ {China_tax:.2f}',
                xy=(China_tax, 0.1 * 10**10),
                xytext=(China_tax - 25, 0.22 * 10**10),  # смещение текста от точки
                fontsize=12,
                arrowprops=dict(arrowstyle='->', color='black'),
                color='black')
# plt.show()

phi_C_choice = phi_C_func(s_choice)
phi_E_choice = phi_E_func(s_choice)
LCOE_choice = LCOE_per_MWh_def(s_choice)

t_max = int(T_c + T_e)
t_vals = np.linspace(0, t_max, t_max+1)
NPV_t = np.zeros_like(t_vals, dtype=float)

for i, t in enumerate(t_vals):
    if t <= T_c:
        phi_C_t = ((1.0 + s_choice) ** t - 1.0) / (s_choice * T_c) if t > 0 else 0.0
        NPV_t[i] = -K * phi_C_t
    else:
        phi_C_Tc = phi_C_choice  # фиксированная phi_C при t > T_c
        phi_E_t = 1.0 - (1.0 + s_choice) ** (-(t - T_c))
        NPV_t[i] = -K * phi_C_Tc + (1.0 - H) * (E_MWh * price_C_per_MWh - Y_total_USD_per_year) * (phi_E_t / s_choice)

plt.figure(figsize=(8,5))
plt.plot(t_vals, NPV_t/1e6)  # NPV в млн $ для удобства
plt.xlabel("Время t, лет (от начала проекта)")
plt.ylabel("NPV(t), млн $")
plt.title(f"NPV как функция времени (s = {s_choice_percent:.2f} % )")
plt.grid(True)
plt.axhline(0.0, linestyle='--', color='red')
plt.plot(28.5, 0, 'ro', markersize=5)
plt.annotate(f'29 лет',
                xy=(28.5, 0),
                xytext=(22, 100),  # смещение текста от точки
                fontsize=12,
                color='black')
plt.show()

LCOE_values = np.array([LCOE_per_MWh_def(s) for s in s_values])

# 1) График NPV(s)
plt.figure(figsize=(8,5))
plt.plot(s_values*100, LCOE_values)  # по оси X представим в процентах
plt.xlabel("Ставка дисконтирования s, %")
plt.ylabel("LCOE, $")
plt.title("LCOE как функция ставки дисконтирования s")
plt.grid(True)
plt.plot(7.6, China_tax, 'ro', markersize=5)
plt.annotate(f'C ≈ {China_tax:.2f}',
                xy=(7.6, China_tax),
                xytext=(4, China_tax+10),  # смещение текста от точки
                fontsize=12,
                arrowprops=dict(arrowstyle='->', color='black'),
                color='black')
plt.plot(6.6, LCOE_choice, 'ro', markersize=5)
plt.annotate(f'C ≈ {LCOE_choice:.2f}',
                xy=(6.6, LCOE_choice),
                xytext=(3, LCOE_choice+2),  # смещение текста от точки
                fontsize=12,
                arrowprops=dict(arrowstyle='->', color='black'),
                color='black')
plt.show()

om_component_per_MWh = Y_OM_USD_per_year / ((1.0 - H) * E_MWh)

components = np.array([capital_component_per_MWh, fuel_component_per_MWh, om_component_per_MWh])
labels = ["Капитальная", "Топливная", "Операционная"]

plt.figure(figsize=(6,6))
plt.pie(components, labels=labels, autopct='%1.1f%%')
plt.title("Структура стоимости электроэнергии АЭС ($/MWh)")
plt.show()