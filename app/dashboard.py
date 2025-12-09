import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from src.fetch_data import load_raw_data
from src.preprocess import add_aqi_column, add_time_features
from src.aqi_utils import aqi_category

MODELS_DIR = ROOT / "models"

MONTHS_RU = {
    1: "января",
    2: "февраля",
    3: "марта",
    4: "апреля",
    5: "мая",
    6: "июня",
    7: "июля",
    8: "августа",
    9: "сентября",
    10: "октября",
    11: "ноября",
    12: "декабря",
}


def aqi_color_hex(aqi: float) -> str:
    aqi = float(aqi)
    if aqi <= 50:
        return "#4CAF50"  # good
    elif aqi <= 100:
        return "#FFC107"  # moderate
    elif aqi <= 150:
        return "#FF9800"  # unhealthy for sensitive
    elif aqi <= 200:
        return "#F44336"  # unhealthy
    elif aqi <= 300:
        return "#9C27B0"  # very unhealthy
    else:
        return "#795548"  # hazardous


def load_model(h: int):
    path = MODELS_DIR / f"aqi_model_{h}h.joblib"
    data = joblib.load(path)
    return data["model"], data["features"]


def format_dt_ru(dt: pd.Timestamp) -> str:
    return f"{dt.day} {MONTHS_RU[dt.month]} {dt.strftime('%H:%M')}"


def main():
    st.set_page_config(
        page_title="Bishkek Air Quality",
        layout="wide",
    )

    # ---------- Глобальные стили ----------
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] {
            background: linear-gradient(to bottom, #F0F8FF, #FFFFFF);
        }
        [data-testid="stHeader"] {background: transparent;}

        html, body, [data-testid="stMarkdownContainer"] {
            color: #111827;
        }

        .stTabs [role="tab"] {
            color: #111827;
            font-weight: 500;
        }
        .stTabs [role="tab"][aria-selected="true"] {
            border-bottom: 2px solid #EF4444;
            font-weight: 600;
        }

        .aq-card {
            border-radius: 18px;
            background: #FFFFFF;
            border: 1px solid #E5E7EB;
            padding: 1.5rem;
            box-shadow: 0 18px 35px rgba(15, 23, 42, 0.06);
            color: #111827;
        }

        .aq-pill {
            border-radius: 999px;
            padding: 0.15rem 0.65rem;
            font-size: 0.7rem;
            background:#EFF6FF;
            color:#1D4ED8;
            text-transform: uppercase;
            letter-spacing: .08em;
        }

        .aq-section-title{
            font-size:1.2rem;
            font-weight:600;
            margin-bottom:0.6rem;
            color:#111827;
        }

        .aq-subtle {
            color:#6B7280;
            font-size:0.85rem;
        }

        .aq-tag {
            font-size:0.75rem;
            text-transform:uppercase;
            letter-spacing:.08em;
            color:#6B7280;
        }

        .aq-chip {
            border-radius: 999px;
            padding: 0.25rem 0.7rem;
            font-size: 0.7rem;
            border: 1px solid #D1D5DB;
            color:#374151;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # ---------- ДАННЫЕ ----------
    raw_df = load_raw_data()
    df = add_aqi_column(add_time_features(raw_df))
    df = df.sort_values("datetime").reset_index(drop=True)

    latest = df.iloc[-1]
    latest_aqi = float(latest["aqi"])
    latest_time = latest["datetime"]
    latest_cat = aqi_category(int(latest_aqi))
    latest_color = aqi_color_hex(latest_aqi)

    # мульти-прогноз 1–24 ч
    multi_preds: dict[int, float] = {}
    for h in range(1, 25):
        try:
            model, feats = load_model(h)
            X = pd.DataFrame([{col: latest[col] for col in feats}])
            pred = float(model.predict(X)[0])
            multi_preds[h] = pred
        except Exception:
            continue

    # ---------- ВЕРХНИЙ БЛОК ----------
    st.markdown(
        """
        <div style="display:flex; align-items:flex-start; justify-content:space-between; margin-bottom:1.5rem;">
          <div>
            <div class="aq-pill" style="display:inline-flex; align-items:center; gap:0.35rem;">
              <span>☁️</span> <span>Air Quality Overview</span>
            </div>
            <h1 style="font-size:3.2rem; line-height:1.05; margin-top:0.7rem; color:#111827;">
              Bishkek<br/>Air Quality
            </h1>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_over_left, col_over_right = st.columns([1, 2])

    with col_over_left:
        st.markdown(
            f"""
            <div class="aq-card" style="text-align:center; height:100%; display:flex; flex-direction:column; justify-content:space-between;">
              <div style="font-size:0.8rem; text-transform:uppercase; color:#6B7280; letter-spacing:.08em; margin-bottom:0.5rem;">
                Текущий индекс AQI
              </div>
              <div>
                <div style="font-size:4.5rem; font-weight:600; color:{latest_color}; margin-bottom:0.25rem;">
                  {latest_aqi:.0f}
                </div>
                <div style="font-size:1.2rem; font-weight:500; margin-bottom:0.5rem;">
                  {latest_cat}
                </div>
              </div>
              <div class="aq-subtle">
                Обновлено: {format_dt_ru(latest_time)}<br/>
                По данным Open-Meteo.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_over_right:
        pct = min(latest_aqi / 300, 1.0) * 100
        st.markdown(
            f"""
            <div class="aq-card" style="height:100%; display:flex; flex-direction:column; justify-content:space-between;">
              <div>
                <div class="aq-section-title">Что означает этот уровень воздуха?</div>
                <div style="margin-bottom:0.5rem; font-weight:500;">
                  Уровень сейчас: {latest_cat}
                </div>
                <div style="width:100%; background:#E5E7EB; border-radius:999px; height:10px; overflow:hidden; margin-bottom:0.6rem;">
                  <div style="width:{pct:.1f}%; background:{latest_color}; height:10px;"></div>
                </div>
                <p class="aq-subtle">
                  Для большинства людей воздух приемлем. Чувствительным группам (дети, люди с астмой)
                  лучше избегать длительных тренировок на улице.
                </p>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- Блок "Что это значит для вас" (по центру) ----------
    spacer_l, col_means, spacer_r = st.columns([1, 2, 1])

    with col_means:
        st.markdown(
            """
            <div class="aq-card" style="text-align:center;">
              <div class="aq-section-title" style="margin-bottom:1.2rem;">
                Что это значит для вас
              </div>
              <div style="display:flex; gap:1.5rem;">
                <div style="flex:1; text-align:center;">
                  <div style="font-size:2rem; margin-bottom:0.5rem;">🚶‍♂️</div>
                  <div style="font-weight:600; margin-bottom:0.2rem;">Обычные люди</div>
                  <div class="aq-subtle">
                    Можно спокойно гулять и заниматься обычной активностью на улице,
                    если не чувствуете дискомфорта.
                  </div>
                </div>
                <div style="flex:1; text-align:center;">
                  <div style="font-size:2rem; margin-bottom:0.5rem;">🏃‍♀️</div>
                  <div style="font-weight:600; margin-bottom:0.2rem;">Чувствительные группы</div>
                  <div class="aq-subtle">
                    Если есть проблемы с лёгкими или сердцем, астма или другие заболевания —
                    лучше сократить тяжёлые нагрузки и длительные тренировки на улице.
                  </div>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---------- Основные загрязнители (под ним, на всю ширину) ----------
    pm25 = float(latest["pm25"]) if "pm25" in latest.index else None
    pm10 = float(latest["pm10"]) if "pm10" in latest.index else None
    no2 = float(latest["no2"]) if "no2" in latest.index else None

    def pollutant_row(label, value):
        if value is None:
            text = "—"
            width = 0
            color = "#9CA3AF"
        else:
            text = f"{value:.1f} µg/m³"
            width = min(value / 150, 1.0) * 100
            color = "#F9A825" if label == "PM2.5" else "#4CAF50"

        return (
            f"<div style='display:flex; align-items:center; gap:0.5rem; "
            f"margin-bottom:0.4rem;'>"
            f"<span style='width:3rem; font-weight:500;'>{label}</span>"
            f"<span style='width:5rem; font-size:0.85rem; color:#6B7280;'>{text}</span>"
            f"<div style='flex:1; background:#E5E7EB; border-radius:999px; height:6px;'>"
            f"<div style='width:{width:.1f}%; background:{color}; height:6px; "
            f"border-radius:999px;'></div>"
            f"</div></div>"
        )

    pollutants_html = (
        "<div class='aq-card' style='margin-top:1.5rem;'>"
        "<div class='aq-section-title'>Основные загрязнители</div>"
        + pollutant_row("PM2.5", pm25)
        + pollutant_row("PM10", pm10)
        + pollutant_row("NO₂", no2)
        + "</div>"
    )

    st.markdown(pollutants_html, unsafe_allow_html=True)

    st.markdown("---")

    # ---------- ТАБЫ ----------
    tab_overview, tab_forecast, tab_history = st.tabs(
        ["📊 Обзор по часам", "🔮 Прогноз", "📅 История"]
    )

    # ====== TAB 1: ОБЗОР ПО ЧАСАМ ======
    with tab_overview:
        st.subheader("Как меняется воздух в течение суток")

        df["date"] = df["datetime"].dt.date
        last_week = df[df["datetime"] > df["datetime"].max() - pd.Timedelta(days=7)]
        hw = last_week.copy()
        hw["hour"] = hw["datetime"].dt.hour
        pivot = hw.pivot_table(
            values="aqi", index="date", columns="hour", aggfunc="mean"
        )

        if not pivot.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            sns.heatmap(
                pivot,
                ax=ax,
                cmap="YlOrRd",
                cbar_kws={"label": "AQI"},
            )
            ax.set_xlabel("Час")
            ax.set_ylabel("Дата")
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("Недостаточно данных для тепловой карты.")

        st.markdown("### Health recommendations & AQI guide")

        guide_cols = st.columns(4)
        ranges = [
            ("Good (0–50)", "Можно спокойно гулять.", "#4CAF50"),
            (
                "Moderate (51–100)",
                "Обычно нормально, но следите за самочувствием.",
                "#FFC107",
            ),
            (
                "Unhealthy for sensitive (101–150)",
                "Чувствительным лучше сократить время на улице.",
                "#FF9800",
            ),
            (
                "Unhealthy (151+)",
                "По возможности оставайтесь в помещении.",
                "#F44336",
            ),
        ]

        for col, (title, text, color) in zip(guide_cols, ranges):
            with col:
                st.markdown(
                    f"""
                    <div class="aq-card" style="border-top:4px solid {color}; padding-top:1rem;">
                      <div style="font-weight:600; margin-bottom:0.25rem;">{title}</div>
                      <div class="aq-subtle">{text}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # ====== TAB 2: ПРОГНОЗ ======
    with tab_forecast:
        st.subheader("Hourly air quality forecast — Bishkek")

        col_top_l, col_top_r = st.columns([1.5, 1])

        with col_top_l:
            st.caption("Data source: твоя ML-модель по историческим данным")

            if multi_preds:
                horizons = sorted(multi_preds.keys())
                vals = [multi_preds[h] for h in horizons]

                fig, ax = plt.subplots(figsize=(9, 4))

                ax.axhspan(151, 500, color="#F44336", alpha=0.15)
                ax.axhspan(101, 150, color="#FF9800", alpha=0.15)
                ax.axhspan(51, 100, color="#FFC107", alpha=0.15)
                ax.axhspan(0, 50, color="#4CAF50", alpha=0.15)

                ax.plot(horizons, vals, marker="o", color="#6B4F2A", linewidth=2)
                ax.set_xlabel("Часы вперёд")
                ax.set_ylabel("AQI")
                ax.set_ylim(0, max(160, max(vals) + 10))
                ax.grid(True, linestyle="--", alpha=0.3)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Нет обученных моделей для прогноза 1–24 часа.")

        with col_top_r:
            st.markdown("#### Выбери горизонт прогноза")
            h_sel = st.slider("Через сколько часов", 1, 24, 3)
            if h_sel in multi_preds:
                v = multi_preds[h_sel]
                col = aqi_color_hex(v)
                cat = aqi_category(int(v))
                st.markdown(
                    f"""
                    <div class="aq-card">
                      <div class="aq-tag">AQI через {h_sel} ч</div>
                      <div style="font-size:2.5rem; font-weight:600; color:{col}; margin:0.3rem 0;">
                        {v:.0f}
                      </div>
                      <div style="font-weight:500; margin-bottom:0.3rem;">{cat}</div>
                      <div class="aq-subtle">
                        Рекомендация: ориентируйся на шкалу ниже — если цвет становится оранжевым или
                        красным, лучше ограничить длительные прогулки.
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.error("Для этого горизонта нет модели.")

        st.markdown("### Почасовой прогноз (таблица на 24 часа вперёд)")

        rows = []
        for hh in range(1, 25):
            if hh in multi_preds:
                val = multi_preds[hh]
                rows.append(
                    {
                        "Через (ч)": hh,
                        "AQI": int(val),
                        "Категория": aqi_category(int(val)),
                    }
                )
        if rows:
            df_hourly = pd.DataFrame(rows)
            st.dataframe(df_hourly, hide_index=True)
        else:
            st.info("Пока нет данных для почасового прогноза.")

    # ====== TAB 3: ИСТОРИЯ ======
    with tab_history:
        st.subheader("Historical air quality trends")

        df["date"] = df["datetime"].dt.date
        last_30 = df[df["datetime"] > df["datetime"].max() - pd.Timedelta(days=30)]
        daily = last_30.groupby("date")["aqi"].mean().reset_index()
        daily["aqi_round"] = daily["aqi"].round().astype(int)

        col_hist_chart, col_hist_side = st.columns([2, 1])

        with col_hist_chart:
            if not daily.empty:
                fig, ax = plt.subplots(figsize=(9, 4))

                ax.axhspan(151, 500, color="#F44336", alpha=0.15)
                ax.axhspan(101, 150, color="#FF9800", alpha=0.15)
                ax.axhspan(51, 100, color="#FFC107", alpha=0.15)
                ax.axhspan(0, 50, color="#4CAF50", alpha=0.15)

                ax.plot(daily["date"], daily["aqi"], marker="o", color="white")
                ax.set_facecolor("#111827")
                fig.patch.set_facecolor("#111827")
                ax.tick_params(colors="#E5E7EB")
                ax.yaxis.label.set_color("#E5E7EB")
                ax.xaxis.label.set_color("#E5E7EB")
                ax.set_xlabel("Дата")
                ax.set_ylabel("Средний AQI")
                plt.xticks(rotation=45)
                plt.tight_layout()
                st.pyplot(fig)
            else:
                st.info("Недостаточно данных для месячного тренда.")

        with col_hist_side:
            if not daily.empty:
                monthly_avg = daily["aqi"].mean()
                best_row = daily.loc[daily["aqi"].idxmin()]
                worst_row = daily.loc[daily["aqi"].idxmax()]
                best_date = pd.to_datetime(best_row["date"])
                worst_date = pd.to_datetime(worst_row["date"])

                st.markdown(
                    f"""
                    <div class="aq-card" style="background:#111827; color:#E5E7EB; margin-bottom:0.7rem;">
                      <div class="aq-subtle" style="margin-bottom:0.2rem;">Monthly average AQI</div>
                      <div style="font-size:2rem; font-weight:600;">{monthly_avg:.0f}</div>
                      <div class="aq-subtle">{aqi_category(int(monthly_avg))}</div>
                    </div>
                    <div class="aq-card" style="background:#022C22; color:#D1FAE5; margin-bottom:0.7rem;">
                      <div class="aq-subtle" style="margin-bottom:0.2rem;">Best day</div>
                      <div style="font-size:1.4rem; font-weight:600;">
                        {best_date.strftime('%d %b')}
                      </div>
                      <div class="aq-subtle">AQI {best_row['aqi_round']} — {aqi_category(int(best_row['aqi_round']))}</div>
                    </div>
                    <div class="aq-card" style="background:#3F0F12; color:#FECACA;">
                      <div class="aq-subtle" style="margin-bottom:0.2rem;">Worst day</div>
                      <div style="font-size:1.4rem; font-weight:600;">
                        {worst_date.strftime('%d %b')}
                      </div>
                      <div class="aq-subtle">AQI {worst_row['aqi_round']} — {aqi_category(int(worst_row['aqi_round']))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.info("Пока нет статистики для лучших/худших дней.")

        st.caption(
            "Вся история построена на тех же данных, что и модель. "
            "Это статистика, поэтому при реальных пожарах/тумане качество может отличаться."
        )


if __name__ == "__main__":
    main()