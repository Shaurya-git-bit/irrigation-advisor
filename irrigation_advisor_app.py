# app.py
# Single-file Streamlit app for "AI Irrigation Advisor for Punjab wheat farmers".
# Run with: streamlit run app.py
#
# requirements.txt (recommended pins):
#   streamlit==1.40.0
#   xgboost==2.0.3
#   pandas==2.2.2
#   numpy==1.26.4
#   shap==0.46.0
#   matplotlib==3.8.4
#
# Notes:
# - Only uses: streamlit, xgboost, pandas, numpy, shap, matplotlib, json, pathlib.
# - All file paths are relative to app.py using Path(__file__).parent.
# - SHAP is fed a pandas DataFrame with the exact FEATURES column order so the
#   waterfall plot shows real feature names (which are then mapped to friendly
#   labels). Without the DataFrame, SHAP labels features as "Feature 12", etc.
# - st.image uses width="stretch" (supported in recent Streamlit); if you pin an
#   older Streamlit, replace width="stretch" with use_container_width=True.

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor

# ---------------------------------------------------------------------------
# Paths (relative to app.py only, never absolute)
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
PLOTS_DIR = BASE / "plots"

# ---------------------------------------------------------------------------
# Page config + light green/white styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Punjab Irrigation Advisor", layout="wide")

st.markdown(
    """
    <style>
    .stApp { background-color: #f6fbf6; }
    .banner { padding: 1.2rem 1.5rem; border-radius: 12px; font-size: 1.35rem;
              font-weight: 600; color: white; margin-bottom: 1rem; }
    .banner-wait { background-color: #2e7d32; }
    .banner-irrigate { background-color: #c62828; }
    .metric-tile { background: white; border: 1px solid #d7e8d7; border-radius: 10px;
                   padding: 0.9rem; text-align: center; }
    .metric-label { color: #4a6b4a; font-size: 0.85rem; }
    .metric-value { color: #1b5e20; font-size: 1.5rem; font-weight: 700; }
    .note { color: #557055; font-size: 0.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Friendly feature-name dictionary (fallback = raw name)
# ---------------------------------------------------------------------------
FRIENDLY = {
    "rain_lag1": "Rain yesterday",
    "rain_mm": "Rain today",
    "sm_0_7": "Soil moisture today",
    "sm_0_7_lag1": "Soil moisture yesterday",
    "sm_trend1": "1-day moisture change",
    "sm_trend3": "3-day moisture change",
    "sm_trend7": "7-day moisture change",
    "temp_min": "Night temperature",
    "temp_range": "Day-night temperature gap",
    "doy_cos": "Time of year",
    "days_since_rain": "Days since last rain",
    "das_wheat": "Crop age",
    "sm_std7": "Weekly moisture swings",
    "solar_mj_m2": "Sunlight",
    "sm_deep_trend3": "Deeper soil trend",
}

def friendly(name):
    return FRIENDLY.get(name, name)

# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource
def load_model():
    m = XGBRegressor()
    m.load_model(str(MODEL_DIR / "xgb_h3.json"))
    return m

@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

@st.cache_data
def load_features():
    with open(MODEL_DIR / "features.json") as f:
        return json.load(f)

@st.cache_data
def load_thresholds():
    with open(MODEL_DIR / "thresholds.json") as f:
        return json.load(f)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_DIR / "app_data.csv", parse_dates=["date"])

@st.cache_data
def load_curves():
    return pd.read_csv(RESULTS_DIR / "fix1_curves.csv")

@st.cache_data
def load_horizons():
    return pd.read_csv(RESULTS_DIR / "horizon_results.csv")

# ---------------------------------------------------------------------------
# Friendly file check
# ---------------------------------------------------------------------------
def require(paths):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        st.error(
            "Some required files are missing. Please make sure these exist next to app.py:\n\n"
            + "\n".join(f"- {m}" for m in missing)
        )
        st.stop()

REQUIRED = [
    MODEL_DIR / "xgb_h3.json",
    MODEL_DIR / "features.json",
    MODEL_DIR / "thresholds.json",
    DATA_DIR / "app_data.csv",
    RESULTS_DIR / "fix1_curves.csv",
    RESULTS_DIR / "horizon_results.csv",
    PLOTS_DIR / "plot1_safety_vs_savings.png",
    PLOTS_DIR / "plot2_season_timeline.png",
    PLOTS_DIR / "plot3_pred_vs_actual.png",
    PLOTS_DIR / "plot4_horizons.png",
    PLOTS_DIR / "shap_waterfall.png",
]
require(REQUIRED)

# ---------------------------------------------------------------------------
# Load everything once
# ---------------------------------------------------------------------------
FEATURES = load_features()
THRESHOLDS = load_thresholds()
MARGIN_DEFAULT = float(THRESHOLDS.get("margin", 0.015))
DRY_THRESHOLD = {k: float(v) for k, v in THRESHOLDS["dry_threshold"].items()}

model = load_model()
df = load_data()
curves = load_curves()
horizons = load_horizons()

# ---------------------------------------------------------------------------
# Sidebar menu
# ---------------------------------------------------------------------------
st.sidebar.title("Punjab Irrigation Advisor")
st.sidebar.caption("AI advice for wheat farmers")
page = st.sidebar.radio("Go to", ["Advisor", "Results", "About"], index=0)

# ---------------------------------------------------------------------------
# Cross-page helpers
# ---------------------------------------------------------------------------
SAFETY_LEVELS = {
    "Save more water": 0.005,
    "Balanced": 0.015,
    "Extra safe": 0.03,
    "Maximum safety": 0.045,
}

def pct(x):
    return f"{x * 100:.1f}%"

def banner(text, kind):
    st.markdown(f'<div class="banner banner-{kind}">{text}</div>', unsafe_allow_html=True)

def metric_tile(label, value):
    st.markdown(
        f'<div class="metric-tile"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )

def tradeoff_sentence(margin):
    row = curves[(curves["rule"] == "model") & np.isclose(curves["margin"], margin)]
    if row.empty:
        return "Trade-off information is unavailable for this safety level."
    r = row.iloc[0]
    return (
        f"Skips about {r['skipped_%']:.0f}% of scheduled irrigations; "
        f"misses about {r['needed_missed_%']:.0f}% of the ones crops really needed."
    )

# ===========================================================================
# PAGE 1: ADVISOR
# ===========================================================================
if page == "Advisor":
    st.title("Should I irrigate today?")
    st.caption("Advice for the next 3 days, wheat only, based on current soil moisture.")

    # Wheat rows only, crop age <= 150 days
    wheat = df[(df["crop"] == "wheat") & (df["days_after_sowing"] <= 150)].copy()
    wheat = wheat.dropna(subset=["sm_0_7", "days_after_sowing"])

    if wheat.empty:
        st.warning("No wheat records were found in the data file.")
        st.stop()

    districts = sorted(wheat["district"].unique().tolist())

    col_left, col_right = st.columns([1, 1])

    with col_left:
        district = st.selectbox("District", districts)

    # Date picker limited to wheat dates available for that district
    d_dates = wheat[wheat["district"] == district]["date"].dt.date
    min_date, max_date = d_dates.min(), d_dates.max()

    with col_right:
        chosen_date = st.date_input(
            "Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

    # Safety level selector
    level = st.radio(
        "Safety level",
        list(SAFETY_LEVELS.keys()),
        index=1,
        horizontal=True,
    )
    margin = SAFETY_LEVELS[level]
    st.markdown(
        f'<span class="note">{tradeoff_sentence(margin)}</span>',
        unsafe_allow_html=True,
    )

    # Find the exact row for chosen district and date
    row_df = wheat[
        (wheat["district"] == district)
        & (wheat["date"].dt.date == chosen_date)
    ]

    # If exact date missing, fall back to nearest available date for that district
    if row_df.empty:
        dist_rows = wheat[wheat["district"] == district].reset_index(drop=True)
        nearest_idx = (dist_rows["date"].dt.date - chosen_date).abs().argsort().iloc[0]
        nearest = dist_rows.iloc[nearest_idx]
        st.info(
            f"No record for {chosen_date}; using nearest available date "
            f"{nearest['date'].date()}."
        )
        row_df = dist_rows.iloc[[nearest_idx]]

    row = row_df.iloc[0]

    # Feature vector in the exact order the model needs
    try:
        feat_dict = {f: float(row[f]) for f in FEATURES}
    except KeyError as e:
        st.error(f"Missing feature column in data: {e}")
        st.stop()

    # DataFrame with named columns so SHAP shows real feature names
    X_one = pd.DataFrame([feat_dict], columns=FEATURES)

    # Model predicts the CHANGE in soil moisture over 3 days
    delta = float(model.predict(X_one)[0])
    sm_today = float(row["sm_0_7"])
    predicted = sm_today + delta

    # District dry threshold
    if district not in DRY_THRESHOLD:
        st.error(f"No dry threshold found for district: {district}")
        st.stop()
    thr = DRY_THRESHOLD[district]
    decision_bound = thr + margin

    # Exact decision logic
    if predicted >= decision_bound:
        banner("WAIT: soil should stay wet for the next 3 days", "wait")
    else:
        banner("IRRIGATE: soil is expected to get too dry within 3 days", "irrigate")

    # Metric tiles
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_tile("Soil moisture today", pct(sm_today))
    with c2:
        metric_tile("Expected in 3 days", pct(predicted))
    with c3:
        metric_tile("Dry-soil limit for this district", pct(thr))

    # Why this answer? SHAP waterfall
    st.subheader("Why this answer?")
    try:
        explainer = load_explainer(model)
        # Pass the DataFrame so SHAP uses FEATURES as feature names
        sv = explainer(X_one)

        fig = plt.figure(figsize=(8, 5))
        shap.plots.waterfall(sv[0], max_display=8, show=False)

        # Relabel y-axis ticks from raw feature names to friendly names
        ax = plt.gca()
        labels = [t.get_text() for t in ax.get_yticklabels()]
        new_labels = []
        for lab in labels:
            if not lab:
                new_labels.append(lab)
                continue
            # SHAP labels look like "rain_lag1 = 0.0" or "rain_lag1"
            key = lab.split(" ")[0]
            if key in FRIENDLY:
                new_labels.append(lab.replace(key, FRIENDLY[key], 1))
            else:
                new_labels.append(lab)
        ax.set_yticklabels(new_labels)

        plt.tight_layout()
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)

        st.markdown(
            '<span class="note">Bars show what pushes soil moisture up (red) '
            "or down (blue) over the next 3 days.</span>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.info("Explanation is unavailable for this record.")
        st.caption(f"Detail: {e}")

    # Water saved this season
    st.subheader("Water saved this season")

    # Wheat season Nov-Apr containing the chosen date
    if chosen_date.month >= 11:
        season_start = pd.Timestamp(chosen_date.year, 11, 1)
        season_end = pd.Timestamp(chosen_date.year + 1, 4, 30)
    else:
        season_start = pd.Timestamp(chosen_date.year - 1, 11, 1)
        season_end = pd.Timestamp(chosen_date.year, 4, 30)

    season = wheat[
        (wheat["district"] == district)
        & (wheat["date"] >= season_start)
        & (wheat["date"] <= season_end)
    ].copy()

    if season.empty:
        st.markdown(
            '<span class="note">No wheat season data for this date and district.</span>',
            unsafe_allow_html=True,
        )
    else:
        # Fixed calendar rule: irrigate when DAS >= 21 and (DAS - 21) % 20 == 0
        das = season["days_after_sowing"]
        season["calendar"] = (das >= 21) & (((das - 21) % 20) == 0)

        # Advisor WAIT decisions across the season
        try:
            X_all = season[FEATURES].astype(float).copy()
            deltas = model.predict(X_all)
        except KeyError as e:
            st.error(f"Missing feature column in data: {e}")
            deltas = np.zeros(len(season))

        season["predicted"] = season["sm_0_7"].to_numpy() + deltas
        season["thr"] = season["district"].map(DRY_THRESHOLD).astype(float)
        season["wait"] = season["predicted"] >= (season["thr"] + margin)

        sched = season[season["calendar"]]
        total_sched = len(sched)
        skip = int(sched["wait"].sum())

        if total_sched > 0:
            st.markdown(
                f"**{skip} of {total_sched} scheduled irrigations could be skipped** "
                f"in the {season_start.strftime('%b %Y')}–{season_end.strftime('%b %Y')} "
                "wheat season."
            )
        else:
            st.markdown(
                '<span class="note">No calendar irrigations fall in this season.</span>',
                unsafe_allow_html=True,
            )
        st.caption("Calendar rule: irrigate every 20 days from day 21 after sowing.")

# ===========================================================================
# PAGE 2: RESULTS
# ===========================================================================
elif page == "Results":
    st.title("Results")
    st.caption(
        "Measured on unseen test years where noted. Nothing here is fitted to the test years."
    )

    # Plot 1
    st.subheader("Safety vs water saved")
    st.image(str(PLOTS_DIR / "plot1_safety_vs_savings.png"), width="stretch")
    st.markdown(
        "At the same crop safety, the AI skips more unnecessary irrigations than "
        "checking today's soil, and the advantage grows as you ask for more safety. "
        "(Tested on 2024-25, unseen years.)"
    )

    # Plot 2
    st.subheader("One wheat season: calendar vs advisor (Muktsar)")
    st.image(str(PLOTS_DIR / "plot2_season_timeline.png"), width="stretch")
    st.markdown(
        "Showcase district Muktsar, one wheat season: fixed calendar irrigations "
        "compared with the advisor's WAIT decisions."
    )

    # Plot 3
    st.subheader("Prediction vs actual")
    st.image(str(PLOTS_DIR / "plot3_pred_vs_actual.png"), width="stretch")
    st.markdown(
        "Predicted vs actual soil moisture 3 days ahead; the model follows the trend "
        "and reacts to sudden rain a day or two late because it does not see future rain."
    )

    # Plot 4
    st.subheader("Error reduction by horizon")
    st.image(str(PLOTS_DIR / "plot4_horizons.png"), width="stretch")
    try:
        h = {int(r["horizon_days"]): float(r["improvement_%"]) for _, r in horizons.iterrows()}
        parts = [f"{h[k]:.1f}% at {k} day" for k in (1, 3, 5) if k in h]
        if parts:
            st.markdown(
                "Error reduction vs the 'soil stays the same' baseline: " + ", ".join(parts) + "."
            )
        else:
            st.markdown("Error reduction vs the 'soil stays the same' baseline.")
    except Exception:
        st.markdown("Error reduction vs the 'soil stays the same' baseline.")

    st.markdown(
        '<span class="note">The safety margin was chosen on 2023 and scored on 2024-25.</span>',
        unsafe_allow_html=True,
    )

# ===========================================================================
# PAGE 3: ABOUT
# ===========================================================================
elif page == "About":
    st.title("About this project")

    # Problem in 4 short lines
    st.subheader("The problem")
    st.markdown(
        "- Punjab extracts **156%** of its groundwater recharge (national average 61%, CGWB 2025).\n"
        "- **111 of 153** assessment units are over-exploited.\n"
        "- About **95%** of water use is irrigation, and there are about **1.45 million** tubewells.\n"
        "- Many farmers irrigate on a fixed calendar even before rain or when soil is still wet."
    )

    # How it works in 4 plain steps
    st.subheader("How it works")
    st.markdown(
        "1. Weather and soil records for each Punjab district are turned into simple features "
        "(recent rain, moisture changes, temperature, crop age).\n"
        "2. A model learns how soil moisture in the top 7 cm changes over the next 3 days.\n"
        "3. For wheat, the advisor compares the expected moisture with a district dry limit "
        "plus a safety margin.\n"
        "4. If soil should stay wet, the app says WAIT; otherwise it says IRRIGATE."
    )

    # Limitations and roadmap
    st.subheader("Limitations and roadmap")
    st.markdown(
        "- This demo uses recorded 2023–2025 data, not a live feed.\n"
        "- It does not yet use a live rain forecast, so it can react to sudden rain a day or two late.\n"
        "- Next step: add a live rain forecast and live weather feed, then extend beyond wheat."
    )

    st.markdown("---")
    st.markdown("**Built for Neural Storm 2026, problem GP-02**")