import pandas as pd
from pathlib import Path
from .aqi_utils import pm25_to_aqi

DATA_PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month
    return df


def add_aqi_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["aqi"] = df["pm25"].apply(pm25_to_aqi)
    return df


def make_supervised(df: pd.DataFrame, target_col: str = "aqi", n_hours_ahead: int = 1) -> pd.DataFrame:
    """
    Создаём supervised-датасет:
    признаки = текущее время, погода и т.д.
    таргет = AQI через n_hours_ahead часов.
    """
    df = df.copy().sort_values("datetime")

    df["target"] = df[target_col].shift(-n_hours_ahead)

    # удалить последние строки, где таргет NaN
    df = df.dropna(subset=["target"])

    return df


def preprocess_for_training(raw_df: pd.DataFrame, n_hours_ahead: int = 1) -> pd.DataFrame:
    df = raw_df.copy()

    df = add_aqi_column(df)
    df = add_time_features(df)

    df = make_supervised(df, target_col="aqi", n_hours_ahead=n_hours_ahead)

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    out_path = DATA_PROCESSED / f"training_data_{n_hours_ahead}h.csv"
    df.to_csv(out_path, index=False)

    return df
