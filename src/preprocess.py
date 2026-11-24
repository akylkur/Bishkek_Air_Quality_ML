import pandas as pd


def features(frame):
    data = frame[["us_aqi", "pm2_5", "pm10"]].copy()
    for lag in (1, 3, 24):
        data[f"aqi_lag_{lag}"] = frame.us_aqi.shift(lag)
    data["hour"] = frame.index.hour
    data["month"] = frame.index.month
    return data


def targets(frame):
    return pd.DataFrame({h: frame.us_aqi.shift(-h) for h in range(1, 25)})
