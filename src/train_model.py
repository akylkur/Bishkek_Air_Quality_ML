from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .fetch_data import load_raw_data
from .preprocess import preprocess_for_training

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def train_aqi_model(n_hours_ahead: int = 1):
    raw_df = load_raw_data()
    df = preprocess_for_training(raw_df, n_hours_ahead=n_hours_ahead)

    feature_cols = ["pm25", "temperature", "humidity", "wind_speed", "hour", "dayofweek", "month"]
    feature_cols = [c for c in feature_cols if c in df.columns]

    X = df[feature_cols]
    y = df["target"]

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)
    mae = mean_absolute_error(y_val, y_pred)
    mse = mean_squared_error(y_val, y_pred)
    rmse = mse ** 0.5

    print(f"MAE: {mae:.2f}, RMSE: {rmse:.2f}")


    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / f"aqi_model_{n_hours_ahead}h.joblib"
    joblib.dump({"model": model, "features": feature_cols}, model_path)

    print(f"Модель сохранена в {model_path}")


if __name__ == "__main__":
    # Обучаем модели для горизонтов 1..24 часов
    for h in range(1, 25):
        print("=" * 50)
        print(f"Обучаем модель для горизонта {h} ч вперёд")
        train_aqi_model(n_hours_ahead=h)