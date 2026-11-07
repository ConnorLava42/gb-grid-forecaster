"""What drives demand (SHAP) and does football move it (event study)."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from gridforecast.config import FIG_DIR


def shap_plots(
    model, X: pd.DataFrame, out_dir: Path = FIG_DIR, n: int = 5_000, seed: int = 42
) -> list[Path]:
    """Four SHAP charts for the median model of a fitted LGBMQuantile."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = X.sample(min(n, len(X)), random_state=seed)
    explainer = shap.TreeExplainer(model.models_["q50"])
    values = explainer(sample)
    paths = []

    def save(name):
        path = out_dir / f"{name}.png"
        plt.gcf().savefig(path, dpi=150, bbox_inches="tight")
        plt.close("all")
        paths.append(path)

    shap.plots.beeswarm(values, max_display=15, show=False)
    plt.title("What drives the demand forecast")
    save("shap_beeswarm")

    shap.plots.scatter(values[:, "temperature_2m"], color=values[:, "hh"], show=False)
    plt.title("Effect of temperature (colour = half-hour of the day)")
    save("shap_temperature")

    shap.plots.scatter(values[:, "shortwave_radiation"], color=values[:, "hh"], show=False)
    plt.title("Effect of solar radiation (colour = half-hour of the day)")
    save("shap_radiation")

    # Explain the single highest forecast in the sample, usually a cold evening peak
    peak = int(np.argmax(model.models_["q50"].predict(sample)))
    shap.plots.waterfall(values[peak], max_display=12, show=False)
    plt.title("One forecast, explained feature by feature")
    save("shap_waterfall_peak")
    return paths


def tv_pickup_event_study(
    preds: pd.DataFrame, matches: pd.DataFrame, window=(-4, 8), n_boot: int = 2000, seed: int = 42
) -> pd.DataFrame:
    """Average forecast error around England kick-offs, against control days.

    `preds` should come from a model WITHOUT the football features, so any
    football effect shows up in its errors. Offsets are in half-hours from
    the half-hour containing kick-off. Controls are the same times 7 and 14
    days earlier. Returns mean residual and a bootstrap 95% interval.
    """
    rng = np.random.default_rng(seed)
    d1 = preds[preds["lead_day"] == 1].set_index("target_time")
    resid = (d1["indo_mw"] - d1["q50"]).dropna()
    offsets = np.arange(window[0], window[1] + 1)

    def profile(anchor):
        times = anchor + pd.to_timedelta(offsets * 30, unit="min")
        return resid.reindex(times).to_numpy()

    match_rows, control_rows = [], []
    for k in matches["kickoff_utc"]:
        anchor = k.floor("30min")
        match_rows.append(profile(anchor))
        for days in (7, 14):
            control_rows.append(profile(anchor - pd.Timedelta(days=days)))

    out = {}
    for label, rows in (("match", match_rows), ("control", control_rows)):
        arr = np.array(rows)
        arr = arr[~np.isnan(arr).all(axis=1)]  # drop matches outside the backtest
        if len(arr) == 0:
            continue
        boots = np.array(
            [np.nanmean(arr[rng.integers(0, len(arr), len(arr))], axis=0) for _ in range(n_boot)]
        )
        out[f"{label}_mean"] = np.nanmean(arr, axis=0)
        out[f"{label}_lo"] = np.nanpercentile(boots, 2.5, axis=0)
        out[f"{label}_hi"] = np.nanpercentile(boots, 97.5, axis=0)
        out[f"{label}_n"] = np.full(len(offsets), len(arr))
    return pd.DataFrame(out, index=pd.Index(offsets * 30, name="minutes_from_kickoff"))


def plot_event_study(study: pd.DataFrame, path: Path = FIG_DIR / "tv_pickup.png") -> Path:
    """Match-night errors vs control nights, with 95% bands."""
    fig, ax = plt.subplots(figsize=(9, 5))
    x = study.index
    for label, colour in (("control", "grey"), ("match", "crimson")):
        if f"{label}_mean" not in study:
            continue
        n = int(study[f"{label}_n"].iloc[0])
        ax.plot(x, study[f"{label}_mean"], color=colour, lw=2, label=f"{label} (n={n})")
        ax.fill_between(x, study[f"{label}_lo"], study[f"{label}_hi"], color=colour, alpha=0.2)
    for minute, text in ((0, "kick-off"), (45, "half-time"), (112, "full time")):
        ax.axvline(minute, color="black", lw=0.8, ls="--")
        ax.text(
            minute,
            0.98,
            text,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", edgecolor="none", pad=1),
        )
    ax.axhline(0, color="black", lw=0.8)
    ax.set(
        xlabel="Minutes from kick-off (half-hour periods)",
        ylabel="Actual minus forecast (MW)",
        title="Forecast error around England matches, model without football features",
    )
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
