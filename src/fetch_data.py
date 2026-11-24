from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / "data/bishkek_2025.csv"
ZONE = "Asia/Bishkek"


def fetch(**period):
    params = dict(latitude=42.8746, longitude=74.5698, hourly="us_aqi,pm2_5,pm10", timezone=ZONE)
    response = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality",
                            params={**params, **period}, timeout=20)
    response.raise_for_status()
    return hourly_frame(response.json())


def hourly_frame(payload):
    frame = pd.DataFrame(payload["hourly"]).set_index("time")
    frame.index = pd.to_datetime(frame.index).tz_localize(ZONE)
    return frame.sort_index().asfreq("h")


def archive():
    frame = pd.read_csv(ARCHIVE, index_col="time", parse_dates=True)
    frame.index = frame.index.tz_convert(ZONE)
    return frame.asfreq("h")


if __name__ == "__main__":
    fetch(start_date="2025-01-01", end_date="2025-12-31").to_csv(ARCHIVE)
