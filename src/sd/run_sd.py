import pandas as pd

from src.core.config import settings
from src.core.models import SEIRHCDParams
from src.sd.seir_model import simulate_seir_hcd
from src.utils.plots import display_SD_results


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

def run(params: SEIRHCDParams):
    """Симуляция"""
    results_df = simulate_seir_hcd(params=params, days=settings.DAYS)

    display_SD_results(results_df, params)

    return results_df
