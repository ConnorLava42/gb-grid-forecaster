import matplotlib.pyplot as plt
import pandas as pd

from gridforecast.config import FIG_DIR, REPORT_DIR, UK
from gridforecast.evaluation import metrics

fig, ax = plt.subplots(figsize=(10, 4.5))
for name in ("naive", "lgbm", "mlp", "lgbm_mlp"):
    p = pd.read_parquet(REPORT_DIR / "backtests" / f"{name}.parquet")
    month = p["target_time"].dt.tz_convert(UK).dt.strftime("%Y-%m")  # sorts correctly
    by_month = metrics.score_by(p, month)["mae"]
    labels = pd.to_datetime(by_month.index).strftime("%b %y")  # "Nov 25", "Dec 25", ...
    ax.plot(labels, by_month.values, marker="o", label=name)
ax.set(ylabel="MAE (MW)", title="Forecast error by month, test year")
ax.legend(frameon=False)
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "monthly_mae.png", dpi=150, bbox_inches="tight")
