import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from .fetch_data import ROOT, archive
from .preprocess import features, targets

MODEL = ROOT / "models/us_aqi_forest.joblib"
REPORTS = ROOT / "reports"


def fit(x, y):
    forest = RandomForestRegressor(n_estimators=80, max_depth=12, min_samples_leaf=5,
                                   random_state=42, n_jobs=2)
    return forest.fit(x, y)


def main():
    frame = archive()
    x, y = features(frame), targets(frame)
    valid = x.notna().all(axis=1) & y.notna().all(axis=1)
    x, y = x[valid], y[valid]
    records = []
    hour = pd.Timedelta(hours=1)
    for month in (7, 10, 12):
        start = pd.Timestamp(year=2025, month=month, day=1, tz=frame.index.tz)
        end = start + pd.offsets.MonthBegin()
        train = x.index + 24 * hour < start
        test = (x.index >= start) & (x.index + 24 * hour < end)
        forest = fit(x[train], y[train])
        for name, pred in [("Random forest", forest.predict(x[test])),
                           ("Persistence", x.loc[test, ["us_aqi"]].to_numpy())]:
            errors = y[test].to_numpy() - pred
            for i in range(24):
                e = errors[:, i]
                records.append([start.strftime("%Y-%m"), i + 1, name, len(e),
                                np.abs(e).mean(), np.sqrt(np.mean(e**2))])
    scores = pd.DataFrame(records, columns=["Month", "Hours ahead", "Model", "Origins", "MAE", "RMSE"])
    scores.to_csv(REPORTS / "metrics.csv", index=False)
    final = scores[scores.Month == "2025-12"]
    ax = final.pivot(index="Hours ahead", columns="Model", values="MAE").plot(
        color=["#92724c", "#166566"], linewidth=2, figsize=(8, 3.3))
    ax.set(title="December holdout · forecast error by lead time", ylabel="MAE (US AQI points)")
    ax.figure.tight_layout()
    ax.figure.savefig(REPORTS / "holdout.png", dpi=150)
    joblib.dump(fit(x, y), MODEL, compress=3)


if __name__ == "__main__":
    main()
