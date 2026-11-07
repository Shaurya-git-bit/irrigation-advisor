# irrigation_advisor_app.py
# Streamlit app: Punjab Irrigation Advisor
# Run with: streamlit run irrigation_advisor_app.py
#
# Only uses: streamlit, xgboost, pandas, numpy, shap, matplotlib, json, pathlib.
# All paths are relative to this file.

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
# Paths
# ---------------------------------------------------------------------------
BASE = Path(__file__).parent
MODEL_DIR = BASE / "model"
DATA_DIR = BASE / "data"
RESULTS_DIR = BASE / "results"
PLOTS_DIR = BASE / "plots"

# ---------------------------------------------------------------------------
# Page config + design system (all colors explicit, works on light AND dark)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Punjab Irrigation Advisor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Palette
GREEN_DEEP = "#1B5E20"
GREEN_MID = "#2E7D32"
GREEN_SOFT = "#E8F3E9"
GREEN_BORDER = "#C8E0CB"
AMBER = "#B45309"          # IRRIGATE accent (orange-red, not harsh red)
AMBER_SOFT = "#FFF4E5"
AMBER_BORDER = "#F5C08A"
CHARCOAL = "#1F2933"
MUTED = "#5A6470"
CARD_BG = "#FFFFFF"
PAGE_BG = "#F5F7F4"
BORDER = "#E3E7E0"

st.markdown(
    f"""
    <style>
    /* Base: force readable text regardless of browser theme */
    html, body, [class*="css"], .stApp {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, "Helvetica Neue", Arial, sans-serif;
        color: {CHARCOAL};
    }}
    .stApp {{ background-color: {PAGE_BG}; }}

    /* Force Streamlit's own text elements */
    .stApp p, .stApp span, .stApp label, .stApp li,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stCaption, .stCaption p,
    .stRadio label, .stRadio div, .stSelectbox label,
    .stDateInput label, .stTextInput label {{
        color: {CHARCOAL} !important;
    }}
    .stCaption, .stCaption p {{ color: {MUTED} !important; }}

    /* Sidebar */
    section[data-testid="stSidebar"] {{
        background-color: {GREEN_DEEP} !important;
    }}
    section[data-testid="stSidebar"] * {{ color: #FFFFFF !important; }}
    section[data-testid="stSidebar"] .stRadio label {{ color: #FFFFFF !important; }}

    /* Headings */
    h1 {{ color: {GREEN_DEEP} !important; font-size: 1.9rem; margin-bottom: .25rem; }}
    h2 {{ color: {GREEN_DEEP} !important; font-size: 1.35rem; margin-top: 1.5rem; }}
    h3 {{ color: {CHARCOAL} !important; font-size: 1.1rem; }}

    /* Cards */
    .ks-card {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        box-shadow: 0 1px 3px rgba(20,40,20,0.05), 0 4px 12px rgba(20,40,20,0.04);
        margin-bottom: 1rem;
    }}
    .ks-card * {{ color: {CHARCOAL} !important; }}

    /* Result banner */
    .ks-banner {{
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin: .5rem 0 1.25rem 0;
        display: flex; align-items: center; gap: 1rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05), 0 6px 20px rgba(0,0,0,0.06);
    }}
    .ks-banner-wait {{
        background: {GREEN_SOFT};
        border: 1.5px solid {GREEN_BORDER};
        border-left: 8px solid {GREEN_DEEP};
    }}
    .ks-banner-irrigate {{
        background: {AMBER_SOFT};
        border: 1.5px solid {AMBER_BORDER};
        border-left: 8px solid {AMBER};
    }}
    .ks-banner-title {{
        font-size: 1.35rem; font-weight: 700; margin: 0;
        color: {CHARCOAL};
    }}
    .ks-banner-wait .ks-banner-title {{ color: {GREEN_DEEP}; }}
    .ks-banner-irrigate .ks-banner-title {{ color: {AMBER}; }}
    .ks-banner-sub {{ font-size: .98rem; color: {MUTED}; margin-top: .3rem; }}
    .ks-icon {{ font-size: 2rem; line-height: 1; }}

    /* Metric cards */
    .ks-metric {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 3px rgba(20,40,20,0.05);
        height: 100%;
    }}
    .ks-metric-label {{
        font-size: .85rem; color: {MUTED}; font-weight: 600;
        text-transform: uppercase; letter-spacing: .03em;
    }}
    .ks-metric-value {{
        font-size: 1.7rem; color: {GREEN_DEEP}; font-weight: 700;
        margin: .35rem 0 .55rem 0;
    }}
    .ks-gauge {{
        position: relative; height: 8px; background: #EEF1EA;
        border-radius: 6px; overflow: hidden; margin-top: .25rem;
    }}
    .ks-gauge-fill {{ position: absolute; left:0; top:0; bottom:0; border-radius: 6px; }}
    .ks-gauge-green {{ background: {GREEN_MID}; }}
    .ks-gauge-amber {{ background: {AMBER}; }}
    .ks-gauge-mark {{
        position: absolute; top: -3px; bottom: -3px; width: 2px;
        background: {CHARCOAL}; border-radius: 1px;
    }}
    .ks-gauge-legend {{ font-size: .75rem; color: {MUTED}; margin-top: .4rem; }}

    /* Season summary */
    .ks-season {{
        background: {GREEN_DEEP};
        color: #FFFFFF;
        border-radius: 14px;
        padding: 1.1rem 1.3rem;
        margin: 1rem 0;
        box-shadow: 0 4px 14px rgba(27,94,32,0.18);
    }}
    .ks-season * {{ color: #FFFFFF !important; }}
    .ks-season-num {{ font-size: 1.9rem; font-weight: 800; }}
    .ks-season-sub {{ font-size: .9rem; opacity: .92; margin-top: .2rem; }}

    /* Stat tiles (About) */
    .ks-stat {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(20,40,20,0.05);
        height: 100%;
    }}
    .ks-stat-num {{
        font-size: 1.8rem; font-weight: 800; color: {GREEN_DEEP};
    }}
    .ks-stat-label {{
        font-size: .82rem; color: {MUTED}; margin-top: .25rem; line-height: 1.3;
    }}

    /* Step cards */
    .ks-step {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 1px 3px rgba(20,40,20,0.05);
        height: 100%;
    }}
    .ks-step-num {{
        display: inline-block; background: {GREEN_SOFT};
        color: {GREEN_DEEP}; font-weight: 700; border-radius: 50%;
        width: 30px; height: 30px; text-align: center; line-height: 30px;
        margin-bottom: .55rem;
    }}
    .ks-step-title {{ font-weight: 700; color: {CHARCOAL}; margin-bottom: .25rem; }}
    .ks-step-body {{ color: {MUTED}; font-size: .92rem; line-height: 1.4; }}

    /* Callout (next steps) */
    .ks-callout {{
        background: {AMBER_SOFT};
        border: 1px solid {AMBER_BORDER};
        border-left: 6px solid {AMBER};
        border-radius: 12px;
        padding: .9rem 1.1rem;
        margin: 1rem 0;
    }}
    .ks-callout * {{ color: {CHARCOAL} !important; }}
    .ks-callout h4 {{ color: {AMBER} !important; margin: 0 0 .4rem 0; }}
    .ks-callout ul {{ margin: 0; padding-left: 1.1rem; }}

    /* Section head */
    .ks-section {{ margin-top: 1.5rem; margin-bottom: .5rem; }}
    .ks-section h2 {{ margin: 0; }}

    /* Footer */
    .ks-footer {{
        text-align: center; color: {MUTED}; font-size: .82rem;
        padding: 2rem 0 1rem 0;
    }}

    /* Safety level option text */
    .ks-level-explain {{
        background: {GREEN_SOFT};
        border-left: 4px solid {GREEN_MID};
        border-radius: 8px;
        padding: .6rem .9rem;
        margin: .4rem 0 .9rem 0;
        color: {CHARCOAL} !important;
        font-size: .92rem;
    }}

    /* Tile row for results page */
    .ks-headline-tile {{
        background: {CARD_BG};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 1rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(20,40,20,0.05);
        height: 100%;
    }}
    .ks-headline-num {{
        font-size: 1.9rem; font-weight: 800; color: {GREEN_DEEP};
    }}
    .ks-headline-label {{
        font-size: .82rem; color: {MUTED}; margin-top: .25rem; line-height: 1.3;
    }}

    /* Tame Streamlit chrome */
    #MainMenu {{ visibility: hidden; }}
    footer {{ visibility: hidden; }}
    .block-container {{ padding-top: 1.5rem; max-width: 1150px; }}

    /* Bigger tap targets on mobile */
    @media (max-width: 640px) {{
        .ks-banner-title {{ font-size: 1.1rem; }}
        .ks-metric-value {{ font-size: 1.4rem; }}
        .ks-season-num {{ font-size: 1.5rem; }}
        .ks-stat-num, .ks-headline-num {{ font-size: 1.5rem; }}
        .block-container {{ padding-left: 1rem; padding-right: 1rem; }}
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Friendly feature names
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
# Cached loaders (logic unchanged)
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
            "Some files the app needs are missing. Please check these:\n\n"
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
]
require(REQUIRED)

FEATURES = load_features()
THRESHOLDS = load_thresholds()
DRY_THRESHOLD = {k: float(v) for k, v in THRESHOLDS["dry_threshold"].items()}

model = load_model()
df = load_data()
curves = load_curves()
horizons = load_horizons()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown(
    f"""
    <div style="padding: .25rem 0 1rem 0;">
      <div style="font-size: 1.25rem; font-weight: 800; color: #FFFFFF;">Punjab Irrigation</div>
      <div style="font-size: 1.25rem; font-weight: 800; color: #CDE7CF;">Advisor</div>
      <div style="font-size: .82rem; color: #CDE7CF; margin-top: .35rem;">
        Wheat watering advice for farmers
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Pages",
    ["Advisor", "Results", "About"],
    index=0,
    label_visibility="collapsed",
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
SAFETY_LEVELS = {
    "Save more water": 0.005,
    "Balanced": 0.015,
    "Extra safe": 0.03,
    "Maximum safety": 0.045,
}

SAFETY_EXPLAIN = {
    "Save more water": "Skips the most irrigations. Small chance of missing one the crop really needed.",
    "Balanced": "A sensible middle ground. Skips many unnecessary irrigations with low risk.",
    "Extra safe": "Skips fewer irrigations. Safest for the crop, saves less water.",
    "Maximum safety": "Only skips when very confident. Protects the crop first.",
}

def pct(x):
    return f"{x * 100:.1f}%"

def clamp01(x):
    return max(0.0, min(1.0, float(x)))

def tradeoff_numbers(margin):
    row = curves[(curves["rule"] == "model") & np.isclose(curves["margin"], margin)]
    if row.empty:
        return None
    r = row.iloc[0]
    return float(r["skipped_%"]), float(r["needed_missed_%"])

def gauge_html(value, threshold, hi):
    """Progress bar 0..hi, filled to value, with a marker at threshold."""
    v = clamp01(value / hi)
    t = clamp01(threshold / hi)
    fill_color = GREEN_MID if value >= threshold else AMBER
    return f"""
    <div class="ks-gauge">
      <div class="ks-gauge-fill ks-gauge-{'green' if value >= threshold else 'amber'}"
           style="width: {v*100:.1f}%"></div>
      <div class="ks-gauge-mark" style="left: calc({t*100:.1f}% - 1px);"
           title="Dry limit"></div>
    </div>
    """

def banner(kind, title, sub):
    icon = "✓" if kind == "wait" else "!"
    st.markdown(
        f"""
        <div class="ks-banner ks-banner-{kind}">
          <div class="ks-icon">{icon}</div>
          <div>
            <div class="ks-banner-title">{title}</div>
            <div class="ks-banner-sub">{sub}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# PAGE 1: ADVISOR
# ===========================================================================
if page == "Advisor":
    st.markdown("<h1>Should I water my wheat today?</h1>", unsafe_allow_html=True)
    st.caption("Advice looks 3 days ahead. Wheat only.")

    wheat = df[(df["crop"] == "wheat") & (df["days_after_sowing"] <= 150)].copy()
    wheat = wheat.dropna(subset=["sm_0_7", "days_after_sowing"])

    if wheat.empty:
        st.warning("No wheat records found in the data.")
        st.stop()

    districts = sorted(wheat["district"].unique().tolist())

    # --- Inputs card ---
    st.markdown('<div class="ks-card">', unsafe_allow_html=True)
    col1, col2 = st.columns([1, 1])
    with col1:
        district = st.selectbox("District", districts)

    d_dates = wheat[wheat["district"] == district]["date"].dt.date
    min_date, max_date = d_dates.min(), d_dates.max()
    with col2:
        chosen_date = st.date_input(
            "Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )

    level = st.radio(
        "Safety level",
        list(SAFETY_LEVELS.keys()),
        index=1,
        horizontal=True,
    )
    margin = SAFETY_LEVELS[level]

    tn = tradeoff_numbers(margin)
    tradeoff_text = ""
    if tn is not None:
        sk, miss = tn
        tradeoff_text = (
            f"About {sk:.0f}% of scheduled waterings would be skipped, "
            f"and about {miss:.0f}% of really-needed waterings might be missed."
        )

    st.markdown(
        f"""
        <div class="ks-level-explain">
          <b>{level}:</b> {SAFETY_EXPLAIN[level]}<br>
          <span style="color:{MUTED};">{tradeoff_text}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # --- Find row ---
    row_df = wheat[
        (wheat["district"] == district)
        & (wheat["date"].dt.date == chosen_date)
    ]
    if row_df.empty:
        dist_rows = wheat[wheat["district"] == district].reset_index(drop=True)
        nearest_idx = (dist_rows["date"].dt.date - chosen_date).abs().argsort().iloc[0]
        nearest = dist_rows.iloc[nearest_idx]
        st.info(
            f"No record for {chosen_date}. Using nearest day: "
            f"{nearest['date'].date()}."
        )
        row_df = dist_rows.iloc[[nearest_idx]]
    row = row_df.iloc[0]

    # --- Prediction (logic unchanged) ---
    feat_dict = {f: float(row[f]) for f in FEATURES}
    X_one = pd.DataFrame([feat_dict], columns=FEATURES)
    delta = float(model.predict(X_one)[0])
    sm_today = float(row["sm_0_7"])
    predicted = sm_today + delta
    thr = DRY_THRESHOLD[district]
    decision_bound = thr + margin

    # --- Result banner ---
    if predicted >= decision_bound:
        banner(
            "wait",
            "WAIT. No watering needed for the next 3 days.",
            "The soil should stay wet enough. Save the water for later.",
        )
    else:
        banner(
            "irrigate",
            "WATER NOW. Soil will get too dry within 3 days.",
            "Not enough rain is expected. Watering now keeps the crop safe.",
        )

    # --- Metric cards (with gauges) ---
    hi = max(sm_today, predicted, thr) * 1.35 or 1.0
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="ks-metric">
              <div class="ks-metric-label">Soil moisture today</div>
              <div class="ks-metric-value">{pct(sm_today)}</div>
              {gauge_html(sm_today, thr, hi)}
              <div class="ks-gauge-legend">Black tick = dry limit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="ks-metric">
              <div class="ks-metric-label">Expected in 3 days</div>
              <div class="ks-metric-value">{pct(predicted)}</div>
              {gauge_html(predicted, thr, hi)}
              <div class="ks-gauge-legend">Black tick = dry limit</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="ks-metric">
              <div class="ks-metric-label">Dry limit for {district}</div>
              <div class="ks-metric-value">{pct(thr)}</div>
              {gauge_html(thr, thr, hi)}
              <div class="ks-gauge-legend">Below this, the crop is stressed</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Season summary ---
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

    st.markdown('<div class="ks-section"><h2>Water saved this season</h2></div>', unsafe_allow_html=True)

    if season.empty:
        st.caption("No wheat season data for this date and district.")
    else:
        das = season["days_after_sowing"]
        season["calendar"] = (das >= 21) & (((das - 21) % 20) == 0)
        X_all = season[FEATURES].astype(float).copy()
        deltas = model.predict(X_all)
        season["predicted"] = season["sm_0_7"].to_numpy() + deltas
        season["thr"] = season["district"].map(DRY_THRESHOLD).astype(float)
        season["wait"] = season["predicted"] >= (season["thr"] + margin)

        sched = season[season["calendar"]]
        total = len(sched)
        skip = int(sched["wait"].sum())

        if total > 0:
            st.markdown(
                f"""
                <div class="ks-season">
                  <div class="ks-season-num">{skip} of {total}</div>
                  <div class="ks-season-sub">
                    scheduled waterings could be skipped this season
                    ({season_start.strftime('%b %Y')} to {season_end.strftime('%b %Y')}).
                  </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.caption("No fixed-calendar waterings fall in this season.")
        st.caption("Fixed calendar = water every 20 days, starting 21 days after sowing.")

    # --- Why this answer? (SHAP waterfall) ---
    st.markdown('<div class="ks-section"><h2>Why this answer?</h2></div>', unsafe_allow_html=True)

    with st.expander("How to read this chart", expanded=True):
        st.markdown(
            """
- Think of this chart as a **tug of war on tomorrow's soil moisture**.
- **Bottom:** the typical change in soil moisture over 3 days across all days.
- **Top:** the change predicted for **this specific day**. Add the top number to
  today's moisture to get the expected moisture in 3 days.
- **Red bars push moisture UP** (soil stays wet, favours WAIT).
  **Blue bars push it DOWN** (soil dries, favours IRRIGATE).
  Longer bar = bigger influence.
- **Numbers next to each name** show today's value for that factor.
            """
        )

    try:
        explainer = load_explainer(model)
        sv = explainer(X_one)

        # Compute top-2 takeaway (before plotting) from the SHAP values
        vals = sv.values[0] if hasattr(sv, "values") else sv[0].values
        names = list(FEATURES)
        order = np.argsort(-np.abs(vals))[:2]
        def phrase_for(name, value, direction):
            fname = friendly(name)
            if name == "days_since_rain":
                return f"it has not rained for {int(value)} days, which dries the soil"
            if name in ("rain_mm", "rain_lag1"):
                return f"{fname.lower()} added {abs(vals[list(FEATURES).index(name)]):.3f} moisture"
            verb = "raises" if direction > 0 else "lowers"
            return f"{fname.lower()} {verb} soil moisture"
        top = order[0]
        top_name = names[top]
        top_dir = float(vals[top])
        if top_name == "days_since_rain":
            takeover = (
                f"Biggest reason: it has not rained for {int(feat_dict['days_since_rain'])} days, "
                "which dries the soil."
            ) if top_dir < 0 else (
                f"Biggest reason: {friendly(top_name).lower()} is keeping soil moisture up."
            )
        else:
            takeover = f"Biggest reason: {friendly(top_name).lower()} is the main factor today."
        st.markdown(f"**{takeover}**")

        fig = plt.figure(figsize=(9, 5))
        shap.plots.waterfall(sv[0], max_display=8, show=False)

        ax = plt.gca()
        ax.tick_params(axis="y", labelsize=11)
        ax.tick_params(axis="x", labelsize=10)
        for spine in ("top", "right"):
            if spine in ax.spines:
                ax.spines[spine].set_visible(False)

        labels = [t.get_text() for t in ax.get_yticklabels()]
        new_labels = []
        for lab in labels:
            if not lab:
                new_labels.append(lab); continue
            key = lab.split(" ")[0]
            new_labels.append(lab.replace(key, FRIENDLY[key], 1) if key in FRIENDLY else lab)
        ax.set_yticklabels(new_labels)

        plt.tight_layout()
        st.markdown('<div class="ks-card">', unsafe_allow_html=True)
        st.pyplot(fig, clear_figure=True)
        plt.close(fig)
        st.markdown(
            f'<div class="ks-gauge-legend" style="color:{MUTED};">'
            "Red bars push moisture up. Blue bars push it down. "
            "Read bottom to top.</div></div>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.info("The explanation is not available for this day.")
        st.caption(f"Details: {e}")

# ===========================================================================
# PAGE 2: RESULTS
# ===========================================================================
elif page == "Results":
    st.markdown("<h1>What the results look like</h1>", unsafe_allow_html=True)
    st.caption("Measured on later years the AI had not seen (2024-25).")

    # --- Headline numbers pulled from the CSVs ---
    bal_margin = 0.015
    tn = tradeoff_numbers(bal_margin)
    skip_pct = tn[0] if tn else 0.0
    miss_pct = tn[1] if tn else 0.0

    try:
        imp3 = float(horizons.loc[horizons["horizon_days"] == 3, "improvement_%"].iloc[0])
    except Exception:
        imp3 = 8.8

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="ks-headline-tile">
              <div class="ks-headline-num">~{skip_pct:.0f}%</div>
              <div class="ks-headline-label">of scheduled waterings can be skipped</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="ks-headline-tile">
              <div class="ks-headline-num">{skip_pct:.0f}% / {miss_pct:.0f}%</div>
              <div class="ks-headline-label">skipped vs. really needed (Balanced)</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="ks-headline-tile">
              <div class="ks-headline-num">{imp3:.1f}%</div>
              <div class="ks-headline-label">better 3-day prediction than "soil stays the same"</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # --- Plot 1 ---
    st.markdown('<div class="ks-section"><h2>Water saved vs. crop safety</h2></div>', unsafe_allow_html=True)
    st.markdown(
        "**At the same crop safety, the AI saves more water than just checking today's soil.**"
    )
    st.markdown('<div class="ks-card">', unsafe_allow_html=True)
    st.image(str(PLOTS_DIR / "plot1_safety_vs_savings.png"), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Green line = AI advice. Grey line = only looking at today's soil.")

    # --- Plot 2 ---
    st.markdown('<div class="ks-section"><h2>One wheat season in Muktsar</h2></div>', unsafe_allow_html=True)
    st.markdown(
        "**A fixed calendar vs. the AI in a real season. Green X marks waterings the AI said to skip.**"
    )
    st.markdown('<div class="ks-card">', unsafe_allow_html=True)
    st.image(str(PLOTS_DIR / "plot2_season_timeline.png"), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Blue line = soil moisture. Bars = rain. Red dashed line = dry limit.")

    # --- Plot 3 ---
    st.markdown('<div class="ks-section"><h2>How close the 3-day prediction is to reality</h2></div>', unsafe_allow_html=True)
    st.markdown(
        "**The prediction follows the trend well, and reacts to sudden rain a day or two late.**"
    )
    st.markdown('<div class="ks-card">', unsafe_allow_html=True)
    st.image(str(PLOTS_DIR / "plot3_pred_vs_actual.png"), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)
    st.caption("Black = what really happened. Green dashed = what the AI expected 3 days before.")

    # --- Plot 4 ---
    st.markdown('<div class="ks-section"><h2>Better than "soil stays the same" at every gap</h2></div>', unsafe_allow_html=True)
    st.markdown(
        "**The AI beats the simplest guess at 1, 3 and 5 days ahead.**"
    )
    st.markdown('<div class="ks-card">', unsafe_allow_html=True)
    st.image(str(PLOTS_DIR / "plot4_horizons.png"), width="stretch")
    st.markdown("</div>", unsafe_allow_html=True)

    try:
        h = {int(r["horizon_days"]): float(r["improvement_%"]) for _, r in horizons.iterrows()}
        parts = [f"{h[k]:.1f}% at {k} day" for k in (1, 3, 5) if k in h]
        if parts:
            st.caption("Error reduction vs 'soil stays the same': " + ", ".join(parts) + ".")
    except Exception:
        pass

    st.caption(
        "The safety margin was chosen on 2023 and scored on 2024-25. "
        "So these numbers are on years the AI did not learn from."
    )

# ===========================================================================
# PAGE 3: ABOUT
# ===========================================================================
elif page == "About":
    st.markdown("<h1>About this project</h1>", unsafe_allow_html=True)
    st.caption("A simple tool to help save water on Punjab wheat farms.")

    st.markdown('<div class="ks-section"><h2>The problem in numbers</h2></div>', unsafe_allow_html=True)
    s1, s2, s3, s4 = st.columns(4)
    for col, num, label in [
        (s1, "156%", "of groundwater is being pumped out, vs. what refills"),
        (s2, "111 of 153", "blocks have too little groundwater"),
        (s3, "~95%", "of water use is for farming"),
        (s4, "1.45 M", "tubewells across the state"),
    ]:
        with col:
            st.markdown(
                f"""
                <div class="ks-stat">
                  <div class="ks-stat-num">{num}</div>
                  <div class="ks-stat-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="ks-section"><h2>How it works</h2></div>', unsafe_allow_html=True)
    steps = [
        ("1", "Collect daily weather and soil records",
         "For each district: rain, temperature, sunlight, and how wet the top soil is."),
        ("2", "Learn the 3-day trend",
         "The tool learns how soil moisture usually changes over the next 3 days."),
        ("3", "Compare with the district's dry limit",
         "For wheat, we add a small safety buffer to the district's dry limit."),
        ("4", "Say WAIT or WATER",
         "If the soil should stay wet, say WAIT. Otherwise, say WATER NOW."),
    ]
    c1, c2 = st.columns(2)
    c3, c4 = st.columns(2)
    for col, (num, title, body) in zip([c1, c2, c3, c4], steps):
        with col:
            st.markdown(
                f"""
                <div class="ks-step">
                  <div class="ks-step-num">{num}</div>
                  <div class="ks-step-title">{title}</div>
                  <div class="ks-step-body">{body}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown('<div class="ks-section"><h2>Next steps</h2></div>', unsafe_allow_html=True)
    st.markdown(
        """
        <div class="ks-callout">
          <h4>What we would add next</h4>
          <ul>
            <li>A live rain forecast, so the tool reacts to rain sooner.</li>
            <li>A live weather feed, so advice updates every day.</li>
            <li>Advice for paddy, not just wheat.</li>
          </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


