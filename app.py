from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="DengueForecast", page_icon="🦟", layout="wide")

BASE = Path(__file__).parent
CITY = {"sj": "San Juan", "iq": "Iquitos"}
W = ["reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k",
     "reanalysis_min_air_temp_k", "station_min_temp_c", "station_avg_temp_c",
     "reanalysis_relative_humidity_percent", "precipitation_amt_mm",
     "station_precip_mm", "ndvi_se", "ndvi_sw"]
BASELINE = "week_median_smooth"
LABEL = {
    "const_median": "Constant median",
    "week_median_smooth": "Seasonal baseline (smoothed week median)",
    "RF_log|season_only": "RF log, season only",
    "RF_log|no_ndvi": "RF log, no NDVI",
    "RF_log|full": "RF log, full weather",
    "GBR_mae|no_ndvi": "GBR (MAE loss), no NDVI",
    "GBR_mae|full": "GBR (MAE loss), full weather",
    "blend (RF_log full + week median)": "Blend (RF log full + baseline)",
}
BACKTEST_MODELS = ["week_median_smooth", "const_median", "RF_log|season_only",
                   "RF_log|no_ndvi", "RF_log|full"]
N_SPLITS, TEST_SIZE = 6, 52


# ---------------- data and results ----------------
@st.cache_data
def load_results():
    p = BASE / "dengue_cv6.csv"
    if not p.exists():
        return None
    r = pd.read_csv(p, index_col=0)
    r.columns = [f"Fold {i + 1}" for i in range(r.shape[1])]
    r["city"] = [i.split("|", 1)[0] for i in r.index]
    r["model"] = [i.split("|", 1)[1] for i in r.index]
    return r


@st.cache_data
def load_raw():
    fp, lp = BASE / "dengue_features_train.csv", BASE / "dengue_labels_train.csv"
    if not (fp.exists() and lp.exists()):
        return None
    df = pd.read_csv(fp).merge(pd.read_csv(lp), on=["city", "year", "weekofyear"])
    df["week_start_date"] = pd.to_datetime(df["week_start_date"])
    return df.sort_values(["city", "week_start_date"]).reset_index(drop=True)


def summarize(r, city):
    d = r[r.city == city].set_index("model")
    F = d[[c for c in d.columns if c.startswith("Fold")]]
    ref = F.loc[BASELINE]
    rows = []
    for m in F.index:
        v, diff = F.loc[m], F.loc[m] - ref
        se_d = diff.std(ddof=1) / np.sqrt(len(diff))
        if m == BASELINE:
            verdict = "reference"
        elif abs(diff.mean()) < 2 * se_d:
            verdict = "within noise"
        else:
            verdict = "better" if diff.mean() < 0 else "worse"
        rows.append({"model": m, "Model": LABEL.get(m, m), "Mean MAE": v.mean(),
                     "SE": v.std(ddof=1) / np.sqrt(len(v)),
                     "Diff vs baseline": diff.mean(), "Diff SE": se_d,
                     "Folds better": f"{int((diff < 0).sum())}/{len(diff)}",
                     "Verdict": verdict})
    return pd.DataFrame(rows).sort_values("Mean MAE").reset_index(drop=True)


# ---------------- pipeline (same as the analysis script) ----------------
def circ_smooth(s, w=5):
    vals = s.reindex(range(1, 54)).interpolate(limit_direction="both").values
    pad = w // 2
    ext = np.r_[vals[-pad:], vals, vals[:pad]]
    return pd.Series(np.convolve(ext, np.ones(w) / w, mode="valid"), index=range(1, 54))


def baseline_preds(train, test):
    med = train["total_cases"].median()
    wk = train.groupby("weekofyear")["total_cases"].median()
    return {"const_median": np.full(len(test), med),
            "week_median_smooth": test["weekofyear"].map(circ_smooth(wk)).values}


def build_features(g):
    out = pd.DataFrame(index=g.index)
    for col in W:
        s = g[col]
        out[col + "_r4"] = s.rolling(4, min_periods=1).mean()
        out[col + "_r4_l4"] = s.shift(4).rolling(4, min_periods=1).mean()
        out[col + "_r4_l8"] = s.shift(8).rolling(4, min_periods=1).mean()
    ang = 2 * np.pi * g["weekofyear"] / 52.18
    out["wk_sin"], out["wk_cos"] = np.sin(ang), np.cos(ang)
    return out


def city_prep(df, city):
    g = df[df.city == city].reset_index(drop=True).copy()
    g[W] = g[W].ffill()                       # previous known week only
    F = build_features(g)
    keep = np.arange(len(g)) >= 12
    return g[keep].reset_index(drop=True), F[keep].reset_index(drop=True)


@st.cache_data
def fold_windows(city):
    g, _ = city_prep(load_raw(), city)
    out = []
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE).split(g)):
        a, b = g.iloc[te]["week_start_date"].min(), g.iloc[te]["week_start_date"].max()
        out.append(f"Fold {k + 1}: {a:%b %Y} to {b:%b %Y}")
    return out


@st.cache_data(show_spinner="Fitting model...")
def backtest(city, fold, model):
    g, F = city_prep(load_raw(), city)
    tr, te = list(TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_SIZE).split(g))[fold]
    train, test = g.iloc[tr], g.iloc[te]
    ytr, yte = train["total_cases"].values, test["total_cases"].values
    base = baseline_preds(train, test)
    if model in base:
        p = base[model]
    else:
        sets = {"season_only": ["wk_sin", "wk_cos"],
                "no_ndvi": [c for c in F.columns if "ndvi" not in c],
                "full": list(F.columns)}
        cols = sets[model.split("|")[1]]
        med = F.iloc[tr][cols].median()
        m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                  random_state=42, n_jobs=-1)
        m.fit(F.iloc[tr][cols].fillna(med), np.log1p(ytr))
        p = np.clip(np.expm1(m.predict(F.iloc[te][cols].fillna(med))), 0, None)
    return {"date": test["week_start_date"].values, "actual": yte, "pred": p,
            "mae": float(np.abs(yte - p).mean())}


# ---------------- page ----------------
st.title("🦟 DengueForecast")
st.caption("Can weather forecast weekly dengue cases? DrivenData DengAI data: San Juan and Iquitos")
st.info("**Finding:** weather-based models did not reliably beat a plain seasonal average. "
        "The gains are within fold-to-fold noise, and no model predicted outbreak peaks.")

raw, results = load_raw(), load_results()
tab1, tab2, tab3, tab4 = st.tabs(["Cases explorer", "Backtest", "Model comparison", "About"])

# ----- Tab 1 -----
with tab1:
    if raw is None:
        st.warning("Raw competition data not found. Place `dengue_features_train.csv` and "
                   "`dengue_labels_train.csv` next to `app.py` for the interactive view.")
        for f in ["cases_by_city.png", "seasonal_profile.png"]:
            if (BASE / f).exists():
                st.image(str(BASE / f))
    else:
        city = st.radio("City", list(CITY), format_func=CITY.get, horizontal=True, key="c1")
        g = raw[raw.city == city]
        a, b, c, d = st.columns(4)
        a.metric("Weeks", f"{len(g):,}")
        b.metric("Mean cases/week", f"{g.total_cases.mean():.1f}")
        c.metric("Median", f"{g.total_cases.median():.0f}")
        d.metric("Peak week", f"{g.total_cases.max():.0f}")
        fig = px.line(g, x="week_start_date", y="total_cases", title=f"Weekly dengue cases: {CITY[city]}")
        fig.update_xaxes(rangeslider_visible=True, title="")
        st.plotly_chart(fig, width="stretch")
        seas = raw.groupby(["city", "weekofyear"])["total_cases"].median().reset_index()
        seas["City"] = seas["city"].map(CITY)
        st.plotly_chart(px.line(seas, x="weekofyear", y="total_cases", color="City",
                                title="Median cases by week of year"), width="stretch")
        st.caption("San Juan peaks between weeks 36 and 46. Iquitos is flat with small bumps at the "
                   "start and end of the year, so the cities need separate models.")

# ----- Tab 2 -----
with tab2:
    if raw is None:
        st.warning("Backtesting refits models, so it needs the raw competition data next to `app.py`.")
        if (BASE / "forecast_vs_actual.png").exists():
            st.image(str(BASE / "forecast_vs_actual.png"), caption="Last 52 weeks held out (static figure)")
    else:
        c1, c2 = st.columns(2)
        city = c1.radio("City", list(CITY), format_func=CITY.get, horizontal=True, key="c2")
        wins = fold_windows(city)
        fold = wins.index(c2.selectbox("Held-out window (always after the training data)", wins,
                                       index=len(wins) - 1))
        chosen = st.multiselect("Models to overlay", BACKTEST_MODELS,
                                default=["week_median_smooth", "RF_log|full"],
                                format_func=lambda m: LABEL[m])
        if chosen:
            fig, rows = go.Figure(), []
            first = True
            for m in chosen:
                r = backtest(city, fold, m)
                if first:
                    fig.add_trace(go.Scatter(x=r["date"], y=r["actual"], name="Actual",
                                             line=dict(color="black", width=3)))
                    first = False
                fig.add_trace(go.Scatter(x=r["date"], y=r["pred"], name=LABEL[m]))
                rows.append({"Model": LABEL[m], "MAE on this window": round(r["mae"], 2)})
            fig.update_layout(title=f"{CITY[city]}: forecast versus actual", yaxis_title="Weekly cases")
            st.plotly_chart(fig, width="stretch")
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
            st.caption("Models are trained only on weeks before the window. Forecasts are smooth and "
                       "stay near the seasonal average, so outbreak peaks are missed.")

# ----- Tab 3 -----
with tab3:
    if results is None:
        st.warning("`dengue_cv6.csv` not found next to `app.py`.")
    else:
        city = st.radio("City", list(CITY), format_func=CITY.get, horizontal=True, key="c3")
        s = summarize(results, city)
        fig = px.bar(s.iloc[::-1], x="Mean MAE", y="Model", error_x="SE", orientation="h",
                     title=f"Mean MAE over 6 rolling folds of 52 weeks: {CITY[city]} (bars show ±1 SE)")
        fig.update_layout(yaxis_title="")
        st.plotly_chart(fig, width="stretch")

        show = s[["Model", "Mean MAE", "Diff vs baseline", "Diff SE", "Folds better", "Verdict"]].copy()
        st.dataframe(show.round(2), hide_index=True, width="stretch")
        st.caption("Difference is measured against the smoothed seasonal baseline, fold by fold. "
                   "Negative means the model is better. 'Within noise' means the mean difference is "
                   "smaller than 2 standard errors.")

        folds = [c for c in results.columns if c.startswith("Fold")]
        pick = st.multiselect("Per-fold MAE", s["model"].tolist(),
                              default=[BASELINE, "RF_log|full"], format_func=lambda m: LABEL.get(m, m))
        if pick:
            d = results[(results.city == city) & (results.model.isin(pick))]
            long = d.melt(id_vars=["model"], value_vars=folds, var_name="Fold", value_name="MAE")
            long["Model"] = long["model"].map(lambda m: LABEL.get(m, m))
            st.plotly_chart(px.line(long, x="Fold", y="MAE", color="Model", markers=True),
                            width="stretch")

# ----- Tab 4 -----
with tab4:
    st.markdown("""
**Question.** Do weather and vegetation data predict weekly dengue cases better than a seasonal average?

**Design.** Weather-only forecasting, because the competition test set is 3 to 5 years ahead and recent
case counts would not be available. Features are 4-week rolling means of 10 weather variables, the same
means lagged by 4 and 8 weeks, and week of year. Missing values are forward-filled within each city.

**Validation.** Expanding-window, rolling-origin. The test window is always after the training data.
Six folds of 52 weeks. Baselines: constant median and a smoothed week-of-year median.

**Limitations**
- Two cities and a benchmark dataset, not an operational system.
- Six evaluation windows, and outbreak years dominate the error.
- MAE only, no prediction intervals, and no competition leaderboard score.
- Reported cases are not true infections.
- Not for public-health decisions.

**Data.** DrivenData DengAI competition data is not redistributed here.
""")
