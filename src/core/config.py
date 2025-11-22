import json
from email.policy import default

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Настройки проекта"""
    # Общие настройки
    DAYS: int = Field(default=150, alias="DAYS")
    POPULATION: int = Field(default=5_000, alias="POPULATION")
    RANDOM_SEED: int = Field(default=42, alias="RANDOM_SEED")

    # Статистика смертности от коронавируса
    P_DEATH_HOSP_TREATED: float = Field(default=0.10, alias="P_DEATH_HOSP_TREATED")
    P_DEATH_HOSP_UNTREATED: float = Field(default=0.40, alias="P_DEATH_HOSP_UNTREATED")
    P_DEATH_INC_TREATED: float = Field(default=0.60, alias="P_DEATH_INC_TREATED")
    P_DEATH_INC_UNTREATED: float = Field(default=0.90, alias="P_DEATH_INC_UNTREATED")

    # Конфиг пациента
    MAX_WAIT: float = Field(default=0.5, alias="MAX_WAIT")

    # Настройки SEIR-H-C-D модели
    SIGMA: float = Field(default=1.0 / 5.2, alias="SIGMA")
    GAMMA: float = Field(default=1.0 / 10.0, alias="GAMMA")
    R0: float = Field(default=4.5, alias="R0")
    P_HOSP: float = Field(default=0.5, alias="P_HOSP")  # Доля инфицированных, требующих госпитализации
    P_INC: float = Field(default=0.05, alias="P_INC")  # Доля инфицированных, требующих ICU
    P_DEATH: float = Field(default=0.047, alias="P_DEATH")  # Доля инфицированных, умирающих
    HOSP_DURATION: float = Field(default=14.0, alias="HOSP_DURATION")  # Средняя длительность в госпитале
    INC_DURATION: float = Field(default=17.0, alias="INC_DURATION")  # Средняя длительность в ICU
    INITIAL_EXPOSED: int = Field(default=200, alias="INITIAL_EXPOSED")
    INITIAL_INFECTIOUS: int = Field(default=150, alias="INITIAL_INFECTIOUS")


with open("data/epidemics.json", encoding="utf-8") as f:
    data = json.load(f)

settings = Settings()
for k, v in data[0]["data"].items():
    try:
        settings.__setattr__(str(k).upper(), v)
    except Exception:
        print(k)

print(settings)