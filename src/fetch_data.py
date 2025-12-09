import requests
import pandas as pd
from pathlib import Path

DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

BASE_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

# Примерные координаты Бишкека
LAT = 42.8746
LON = 74.5698


def fetch_from_api_and_save(past_days: int = 7, forecast_days: int = 1) -> Path:
    """
    Тянет реальные данные качества воздуха из Open-Meteo и сохраняет в CSV.

    Берём:
      - pm2_5 (основа для AQI)
      - pm10 (на всякий случай)
      - us_aqi (готовый AQI от модели Open-Meteo)

    past_days: сколько дней назад захватывать (до 92)
    forecast_days: сколько дней вперёд (до 7)
    """

    params = {
        "latitude": LAT,
        "longitude": LON,
        "hourly": ",".join(["pm2_5", "pm10", "us_aqi"]),
        "timezone": "auto",
        "past_days": past_days,
        "forecast_days": forecast_days,
    }

    resp = requests.get(BASE_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    hourly = data.get("hourly", {})
    if not hourly:
        raise RuntimeError("Open-Meteo вернул пустой hourly блок")

    df = pd.DataFrame(hourly)

    # time -> datetime
    df["datetime"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["time"])

    # Подгоним названия колонок под наш проект
    rename_map = {
        "pm2_5": "pm25",
        "us_aqi": "aqi_external",
    }
    df = df.rename(columns=rename_map)

    DATA_RAW.mkdir(parents=True, exist_ok=True)
    file_path = DATA_RAW / "bishkek_air_opemeteo.csv"
    df.to_csv(file_path, index=False)

    return file_path


def load_raw_data() -> pd.DataFrame:
    """
    Загружает последний сырой файл.
    Если его нет — тянет с API.
    """
    file_path = DATA_RAW / "bishkek_air_opemeteo.csv"
    if not file_path.exists():
        file_path = fetch_from_api_and_save()
    return pd.read_csv(file_path, parse_dates=["datetime"])
