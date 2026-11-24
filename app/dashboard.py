import sys
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.aqi_utils import aqi_category
from src.fetch_data import ZONE, archive, fetch
from src.preprocess import features

st.set_page_config(page_title="Bishkek | Air quality research", layout="wide")


@st.cache_data(ttl=3600)
def recent_hours():
    return fetch(past_days=7, forecast_days=1)


@st.cache_resource
def forest():
    return joblib.load(ROOT / "models/us_aqi_forest.joblib")


st.markdown("""
<style>
.block-container {max-width: 78rem; padding-top: 2.6rem;}
[data-testid="stHeadingWithActionElements"] h1 {
    font-family: Georgia, serif; font-size: clamp(2.5rem, 6vw, 4.6rem);
    font-weight: 500; letter-spacing: -.055em; line-height: 1.07;
}
[data-testid="stMetric"] {border-top: 3px solid #196864; padding: .8rem 0;}
[data-testid="stMetricValue"] {font-variant-numeric: tabular-nums;}
[data-baseweb="tab-list"] {border-bottom: 1px solid #cdd6d2; gap: 1.4rem;}
</style>
""", unsafe_allow_html=True)
st.title("Bishkek air quality")
st.caption("42.87° N, 74.57° E   ·   Hourly estimates from CAMS via Open-Meteo   ·   Bishkek local time (UTC+6)")
research, current, records = st.tabs(["Forecast evaluation", "Latest forecast", "Data explorer"])
scores = pd.read_csv(ROOT / "reports/metrics.csv", dtype={"Month": str})
with research:
    heading, selector = st.columns([3, 1], vertical_alignment="center")
    heading.header("Forecast accuracy")
    period = selector.selectbox("Test month", list(reversed(scores.Month.unique())))
    fold = scores[scores.Month == period]
    metric = st.radio("Error measure", ["MAE", "RMSE"], horizontal=True)
    comparison = fold.pivot(index="Hours ahead", columns="Model", values=metric)
    pooled = comparison.mean() if metric == "MAE" else (comparison.pow(2).mean()) ** 0.5
    change = 100 * (1 - pooled["Random forest"] / pooled["Persistence"])
    a, b, c = st.columns(3)
    a.metric(f"Random forest · {metric}", f'{pooled["Random forest"]:.2f}')
    b.metric(f"Persistence · {metric}", f'{pooled["Persistence"]:.2f}')
    c.metric("Forest error reduction", f"{change:.1f}%")
    plot, notes = st.columns([2, 1], gap="large")
    with plot:
        st.subheader("Error by lead time")
        st.line_chart(comparison, color=["#987448", "#146963"], x_label="Hours ahead",
                      y_label=f"{metric} (AQI points)", height=305)
    with notes:
        st.subheader("Reading the results")
        definition = ("Mean absolute error averages the size of each miss."
                      if metric == "MAE" else
                      "Root mean squared error gives larger misses more weight.")
        st.write(f"{definition} Lower values mean better forecasts.")
        st.write("The random forest averages 80 trees to predict the next 24 hourly AQI values. "
                 "Persistence repeats the latest AQI at every lead time.")
        st.caption(f"{fold.Origins.iloc[0]:,} hourly starting times · 24 horizons per start")
    with st.expander("Inspect all 24 horizons"):
        st.dataframe(comparison.round(2), width="stretch")
    st.download_button("Download evaluation results", scores.to_csv(index=False), "forecast_metrics.csv")
with current:
    st.header("Latest 24-hour forecast")
    st.caption("Loads the latest hourly estimates from Open-Meteo. Results are cached for one hour.")
    if st.toggle("Load recent data"):
        try:
            now = pd.Timestamp.now(tz=ZONE)
            recent = recent_hours().loc[:now]
            rows = features(recent).dropna()
            if rows.empty:
                st.warning("There is insufficient recent data to build the forecast.")
            else:
                row = rows.tail(1)
                origin = row.index[0]
                st.metric("US AQI at forecast origin", f"{row.us_aqi.iloc[0]:.0f}", help=aqi_category(row.us_aqi.iloc[0]))
                st.caption(f"Origin: {origin:%Y-%m-%d %H:%M %Z} · {aqi_category(row.us_aqi.iloc[0])}")
                if now - origin >= pd.Timedelta(hours=2):
                    st.warning("The forecast origin is more than two hours old.")
                result = pd.DataFrame({"Random forest": forest().predict(row)[0], "Persistence": row.us_aqi.iloc[0]},
                                      index=pd.date_range(origin + pd.Timedelta(hours=1), periods=24, freq="h"))
                result.index.name = "Bishkek local time"
                st.line_chart(result, y_label="US AQI")
                st.dataframe(result.round(1), use_container_width=True)
        except (requests.RequestException, ValueError, KeyError) as error:
            st.warning(f"Recent data unavailable. {error}")
with records:
    st.header("Hourly archive")
    history = archive()
    month = st.selectbox("Archive month", sorted(history.index.strftime("%Y-%m").unique()))
    sample = history.loc[month].rename(columns={"us_aqi": "US AQI", "pm2_5": "PM2.5 (µg/m³)", "pm10": "PM10 (µg/m³)"})
    st.line_chart(sample["US AQI"], y_label="US AQI")
    st.dataframe(sample, use_container_width=True)
    st.download_button("Download hourly data", sample.to_csv(), f"bishkek-{month}.csv")
