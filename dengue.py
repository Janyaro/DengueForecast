import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X = pd.read_csv("dengue_features_train.csv")
y = pd.read_csv("dengue_labels_train.csv")
df = X.merge(y, on=["city", "year", "weekofyear"])
df["week_start_date"] = pd.to_datetime(df["week_start_date"])
df = df.sort_values(["city", "week_start_date"]).reset_index(drop=True)

print(df.shape)
print(df.groupby("city").agg(
    weeks=("total_cases", "size"),
    start=("week_start_date", "min"), end=("week_start_date", "max"),
    mean=("total_cases", "mean"), median=("total_cases", "median"),
    max=("total_cases", "max")))

# hafton ke beech gap
for c, g in df.groupby("city"):
    print(c, "gap days:", g["week_start_date"].diff().dt.days.value_counts().to_dict())

# missing values
miss = df.isna().sum()
print(miss[miss > 0].sort_values(ascending=False))

# cases ka graph
fig, axes = plt.subplots(2, 1, figsize=(10, 6))
for ax, (c, g) in zip(axes, df.groupby("city")):
    ax.plot(g["week_start_date"], g["total_cases"])
    ax.set_title(f"Weekly dengue cases: {c}")
plt.tight_layout()
plt.savefig("cases_by_city.png", dpi=130)
plt.close()

# mausam ke hisaab se cases (saal ke 4 hisse)
print(df.groupby(["city", pd.cut(df["weekofyear"], [0, 13, 26, 39, 53])],
                 observed=True)["total_cases"].mean().round(1))

# total_cases se sab se zyada taalluq
num = df.select_dtypes("number").columns.drop(["year", "weekofyear"])
for c, g in df.groupby("city"):
    corr = g[num].corr()["total_cases"].drop("total_cases")
    print(f"\n{c}: top correlations with total_cases")
    print(corr.reindex(corr.abs().sort_values(ascending=False).index).head(6).round(3))

from sklearn.model_selection import TimeSeriesSplit

def circ_smooth(s, w=5):
    vals = s.reindex(range(1, 54)).interpolate(limit_direction="both").values
    pad = w // 2
    ext = np.r_[vals[-pad:], vals, vals[:pad]]
    return pd.Series(np.convolve(ext, np.ones(w) / w, mode="valid"), index=range(1, 54))

def baseline_preds(train, test):
    med = train["total_cases"].median()
    wk = train.groupby("weekofyear")["total_cases"].median()
    return {
        "const_median": np.full(len(test), med),
        "week_median": test["weekofyear"].map(wk).fillna(med).values,
        "week_median_smooth": test["weekofyear"].map(circ_smooth(wk)).values,
    }

rows, info = [], []
for c, g in df.groupby("city"):
    g = g.reset_index(drop=True)
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=3, test_size=104).split(g)):
        train, test = g.iloc[tr], g.iloc[te]
        info.append({"city": c, "fold": k, "train_weeks": len(train),
                     "test_start": test["week_start_date"].min().date(),
                     "test_end": test["week_start_date"].max().date(),
                     "test_mean": round(test["total_cases"].mean(), 1),
                     "test_max": int(test["total_cases"].max())})
        for name, p in baseline_preds(train, test).items():
            rows.append({"city": c, "fold": k, "model": name,
                         "mae": np.abs(test["total_cases"].values - p).mean()})

print(pd.DataFrame(info))
res = pd.DataFrame(rows)
print(res.pivot_table(index=["city", "model"], columns="fold", values="mae").round(2))
print(res.groupby(["city", "model"])["mae"].mean().round(2))

# saal ke hafte ka profile
fig, ax = plt.subplots(figsize=(8, 4))
for c, g in df.groupby("city"):
    ax.plot(g.groupby("weekofyear")["total_cases"].median(), label=c)
ax.set_xlabel("Week of year"); ax.set_ylabel("Median weekly cases"); ax.legend()
plt.tight_layout(); plt.savefig("seasonal_profile.png", dpi=130); plt.close()


from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

W = ["reanalysis_specific_humidity_g_per_kg", "reanalysis_dew_point_temp_k",
     "reanalysis_min_air_temp_k", "station_min_temp_c", "station_avg_temp_c",
     "reanalysis_relative_humidity_percent", "precipitation_amt_mm",
     "station_precip_mm", "ndvi_se", "ndvi_sw"]

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

def make_models():
    return {
        "RF": (RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                     random_state=42, n_jobs=-1), False),
        "RF_log": (RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                                         random_state=42, n_jobs=-1), True),
        "GBR_mae": (GradientBoostingRegressor(loss="absolute_error", n_estimators=200, max_depth=3,
                                              learning_rate=0.05, subsample=0.8, random_state=42), False),
        "Ridge_log": (make_pipeline(StandardScaler(), Ridge(alpha=100.0)), True),
    }

city_data = {}
for c, g in df.groupby("city"):
    g = g.reset_index(drop=True)
    g[W] = g[W].ffill()                      # sirf pichli value se
    F = build_features(g)
    keep = np.arange(len(g)) >= 12           # shuru ke 12 hafte (lag ke liye)
    city_data[c] = (g[keep].reset_index(drop=True), F[keep].reset_index(drop=True))

rows = []
for c, (g, F) in city_data.items():
    for k, (tr, te) in enumerate(TimeSeriesSplit(n_splits=3, test_size=104).split(g)):
        train, test = g.iloc[tr], g.iloc[te]
        ytr, yte = train["total_cases"].values, test["total_cases"].values
        med = F.iloc[tr].median()
        Xtr, Xte = F.iloc[tr].fillna(med), F.iloc[te].fillna(med)

        for name, p in baseline_preds(train, test).items():
            rows.append({"city": c, "fold": k, "model": name, "mae": np.abs(yte - p).mean()})
        for name, (m, use_log) in make_models().items():
            m.fit(Xtr, np.log1p(ytr) if use_log else ytr)
            p = m.predict(Xte)
            p = np.clip(np.expm1(p) if use_log else p, 0, None)
            rows.append({"city": c, "fold": k, "model": name, "mae": np.abs(yte - p).mean()})

res3 = pd.DataFrame(rows)
res3.to_csv("dengue_results.csv", index=False)
print(res3.pivot_table(index=["city", "model"], columns="fold", values="mae").round(2))

avg = res3.groupby(["city", "model"])["mae"].mean().unstack(0).round(2)
base = avg.loc["const_median"]
avg["iq_skill_%"] = ((1 - avg["iq"] / base["iq"]) * 100).round(1)
avg["sj_skill_%"] = ((1 - avg["sj"] / base["sj"]) * 100).round(1)
print(avg)

# RF ki top features (poore data par, sirf samajhne ke liye)
for c, (g, F) in city_data.items():
    m = RandomForestRegressor(n_estimators=300, min_samples_leaf=5, max_features=0.5,
                              random_state=42, n_jobs=-1)
    m.fit(F.fillna(F.median()), g["total_cases"].values)
    s = pd.Series(m.feature_importances_, index=F.columns).sort_values(ascending=False)
    print(f"\n{c}: top features\n", s.head(8).round(3))