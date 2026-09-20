"""Streamlit dashboard for the EPL transfer-value predictor.

Loads the pipeline trained by ``train.py`` (cached, so inference is instant),
turns the sidebar controls into a live valuation, and exposes the regression
diagnostics behind two tabs.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from src.data_loader import AGE_RANGE, ASSISTS_RANGE, GOALS_RANGE, MINUTES_RANGE, PEAK_AGE
from src.model import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    get_coefficients,
    get_intercept,
    load_metadata,
    load_model,
    load_test_predictions,
    predict_value,
)
from src.preprocessor import MIN_NINETIES, REFERENCE_POSITION
from src.visualize import plot_actual_vs_predicted, plot_coefficients, plot_residuals

PROCESSED_DATASET_PATH = Path("data/processed/players_processed.csv")

POSITION_LABELS = {
    "FW": "FW - Forward",
    "MF": "MF - Midfielder",
    "DF": "DF - Defender",
    "GK": "GK - Goalkeeper",
}
#: Display-only conversion for the metric card. Indicative, not a live FX quote.
EUR_TO_GBP = 0.85
CURRENCY_SYMBOL = {"EUR": "€", "GBP": "£"}


# --------------------------------------------------------------------------- #
# Cached loaders - the app never retrains, it only reads artefacts from disk.
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def get_model(path: str = str(DEFAULT_MODEL_PATH)):
    return load_model(path)


@st.cache_data(show_spinner=False)
def get_metadata(path: str = str(DEFAULT_METADATA_PATH)) -> dict:
    return load_metadata(path)


@st.cache_data(show_spinner=False)
def get_predictions(path: str = str(DEFAULT_PREDICTIONS_PATH)) -> pd.DataFrame | None:
    return load_test_predictions(path)


@st.cache_data(show_spinner=False)
def get_training_data(path: str = str(PROCESSED_DATASET_PATH)) -> pd.DataFrame | None:
    file = Path(path)
    return pd.read_csv(file) if file.exists() else None


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="EPL Transfer Value Predictor",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.2rem; max-width: 1280px; }
      h1 { letter-spacing: -0.02em; }
      div[data-testid="stMetricValue"] { font-size: 2.6rem; color: #2f7d6f; }
      div[data-testid="stSidebarUserContent"] h2 { margin-top: 0; }
      .caption-rule { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("EPL Transfer Value Predictor")
st.markdown(
    '<p class="caption-rule">Linear regression on Premier League performance data. '
    "Move the sliders to value a player.</p>",
    unsafe_allow_html=True,
)

# Everything below needs a trained pipeline; fail loudly but helpfully.
try:
    pipeline = get_model()
except FileNotFoundError:
    st.error("No trained model found at `models/model.pkl`.")
    st.markdown("Train one first - it takes a couple of seconds and needs no API key:")
    st.code("python train.py --mock", language="bash")
    st.stop()

metadata = get_metadata()
coefficients = get_coefficients(pipeline)
test_predictions = get_predictions()
training_data = get_training_data()


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Player profile")

    position = st.selectbox(
        "Position",
        options=list(POSITION_LABELS),
        format_func=lambda code: POSITION_LABELS[code],
        index=0,
        help="Encoded as a one-hot feature; GK is the model's reference category.",
    )
    age = st.slider("Age", AGE_RANGE[0], AGE_RANGE[1], 25, help=f"Values peak around {PEAK_AGE:g}.")

    st.subheader("Season output")
    goals = st.slider("Goals", GOALS_RANGE[0], GOALS_RANGE[1], 12)
    assists = st.slider("Assists", ASSISTS_RANGE[0], ASSISTS_RANGE[1], 6)
    minutes = st.slider(
        "Minutes played",
        MINUTES_RANGE[0],
        MINUTES_RANGE[1],
        2200,
        step=10,
        help="A full 38-match season is 3,420 minutes.",
    )

    st.divider()
    currency = st.radio("Display currency", ("EUR", "GBP"), horizontal=True)

    st.divider()
    source = metadata.get("data_source", "unknown")
    target_source = metadata.get("target_source", "unknown")
    trained_at = metadata.get("trained_at", "unknown")
    st.caption(
        f"**Data source:** {source}  \n"
        f"**Target values:** {target_source}  \n"
        f"**Trained:** {trained_at}"
    )
    if target_source == "synthetic":
        st.caption(
            ":warning: Trained on a synthetic valuation baseline. "
            "Join a real market-value CSV for meaningful figures."
        )


# --------------------------------------------------------------------------- #
# Prediction
# --------------------------------------------------------------------------- #
raw_prediction = predict_value(
    pipeline,
    age=age,
    goals=goals,
    assists=assists,
    minutes=minutes,
    position=position,
    clip_negative=False,
)
# OLS is unbounded, so extreme profiles can extrapolate below zero.
value_eur_m = max(raw_prediction, 0.0)
was_clipped = raw_prediction < 0.0
displayed = value_eur_m * (EUR_TO_GBP if currency == "GBP" else 1.0)
symbol = CURRENCY_SYMBOL[currency]

# Compare against the squad the model learned from, for context.
delta_text = None
if training_data is not None and "market_value_eur_m" in training_data.columns:
    median_value = float(training_data["market_value_eur_m"].median())
    difference = value_eur_m - median_value
    delta_text = f"{difference:+.1f}M vs. squad median ({median_value:.1f}M)"

left, right = st.columns([1.15, 1], gap="large")

with left:
    st.metric(
        label="Predicted transfer value",
        value=f"{symbol}{displayed:,.1f}M",
        delta=delta_text,
        border=True,
    )
    if was_clipped:
        st.caption(
            f":warning: The model extrapolated to {raw_prediction:.1f}M for this profile and the "
            "figure was floored at zero. Player profiles this far outside the training "
            "distribution are where a linear model breaks down."
        )
    if currency == "GBP":
        st.caption(f"Converted from euros at an indicative {EUR_TO_GBP:.2f} EUR/GBP rate.")

with right:
    nineties = max(minutes / 90.0, MIN_NINETIES)
    a, b, c = st.columns(3)
    a.metric("Goals / 90", f"{goals / nineties:.2f}", border=True)
    b.metric("Assists / 90", f"{assists / nineties:.2f}", border=True)
    c.metric("Season played", f"{minutes / MINUTES_RANGE[1]:.0%}", border=True)

st.divider()


# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
performance_tab, impact_tab = st.tabs(["Model Performance", "Feature Impact"])

with performance_tab:
    st.subheader("How well does the model fit?")

    test_metrics = metadata.get("test_metrics", {})
    if test_metrics:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("R² (held-out)", f"{test_metrics.get('r2', float('nan')):.3f}", border=True)
        m2.metric("MAE", f"€{test_metrics.get('mae', float('nan')):.2f}M", border=True)
        m3.metric("RMSE", f"€{test_metrics.get('rmse', float('nan')):.2f}M", border=True)
        m4.metric("Test players", f"{test_metrics.get('n_samples', 0)}", border=True)

    if test_predictions is None or test_predictions.empty:
        st.info("No saved test predictions yet. Re-run `python train.py --mock` to generate them.")
    else:
        scatter_col, residual_col = st.columns(2, gap="large")
        with scatter_col:
            st.pyplot(
                plot_actual_vs_predicted(
                    test_predictions["actual"], test_predictions["predicted"]
                )
            )
            st.caption(
                "Each dot is a held-out player. The dashed line is a perfect prediction - "
                "dots above it are over-valued by the model, dots below under-valued."
            )
        with residual_col:
            st.pyplot(
                plot_residuals(test_predictions["actual"], test_predictions["predicted"])
            )
            st.caption(
                "Residuals should scatter randomly around zero. A funnel shape means the "
                "model is less reliable for expensive players."
            )

    with st.expander("How these numbers are produced"):
        cv = metadata.get("cross_validation", {})
        st.markdown(
            f"""
            - **Model:** `{metadata.get('model', 'sklearn.linear_model.LinearRegression')}`
            - **Features:** {', '.join(metadata.get('features', []))}
            - **Rows:** {metadata.get('n_rows', 'n/a')} players, held-out split scored above
            - **Cross-validated R²:** {cv.get('cv_r2_mean', float('nan')):.3f}
              (+/- {cv.get('cv_r2_std', float('nan')):.3f} over {cv.get('cv_folds', 0)} folds)

            MAE and RMSE are in the same units as the target (EUR millions), so an MAE of
            2.5 means the model is off by about 2.5M for a typical player. RMSE punishes
            large misses harder, so the gap between the two shows how much the errors are
            driven by a handful of outliers.
            """
        )

with impact_tab:
    st.subheader("Which stats drive the valuation?")
    st.markdown(
        "Numeric features are standardised before fitting, so each bar is the change in "
        "predicted value for a **one-standard-deviation** increase in that feature - "
        f"directly comparable across features. Position bars read against {REFERENCE_POSITION}."
    )

    chart_col, table_col = st.columns([1.4, 1], gap="large")
    with chart_col:
        st.pyplot(plot_coefficients(coefficients))
    with table_col:
        st.dataframe(
            coefficients[["label", "coefficient"]].rename(
                columns={"label": "Feature", "coefficient": "Coefficient"}
            ),
            hide_index=True,
            height=380,
            column_config={
                "Feature": st.column_config.TextColumn(width="medium"),
                "Coefficient": st.column_config.NumberColumn(
                    "€M per SD", format="%+.2f", width="small"
                ),
            },
        )
        st.metric("Intercept", f"€{get_intercept(pipeline):.2f}M", border=True)

    st.caption(
        "These are *conditional* effects: each coefficient holds the others fixed. "
        "Correlated features (goals and goals-per-90, for instance) share the credit "
        "between them, so read the group rather than a single bar."
    )

st.divider()
st.caption(
    "Predictions are indicative only - a linear model on public match statistics cannot "
    "capture contract length, injuries, sell-on clauses or club negotiating position."
)
