"""EPL Player Scouting & Transfer Valuation Dashboard.

Pick a real Premier League player, see the stats the model was given, and
compare the market's price against what the model thinks they are worth.

Run with::

    streamlit run app.py
"""

from __future__ import annotations

import html
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_loader import (
    AGE_RANGE,
    ASSISTS_RANGE,
    GOALS_RANGE,
    MINUTES_RANGE,
    PEAK_AGE,
    TARGET_EUR,
)
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
    scout_table,
)
from src.preprocessor import MIN_NINETIES, REFERENCE_POSITION, TARGET
from src.reference_data import PROVENANCE, SEASON, SNAPSHOT_NOTICE, SQUAD_AS_OF, STATS_SEASON
from src.visualize import plot_actual_vs_predicted, plot_coefficients, plot_residuals

PROCESSED_DATASET_PATH = Path("data/processed/players_processed.csv")

POSITION_LABELS = {
    "GK": "Goalkeeper",
    "DF": "Defender",
    "MF": "Midfielder",
    "FW": "Forward",
}
ALL_CLUBS = "All clubs"

#: Display-only conversion for the value cards. Indicative, not a live FX quote.
EUR_TO_GBP = 0.85
CURRENCY_SYMBOL = {"EUR": "€", "GBP": "£"}

BARGAIN_COLOR = "#2f7d6f"
PREMIUM_COLOR = "#c2543d"


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
def get_dataset(path: str = str(PROCESSED_DATASET_PATH)) -> pd.DataFrame | None:
    file = Path(path)
    return pd.read_csv(file) if file.exists() else None


def money(value_eur_m: float, currency: str) -> str:
    """Format a EUR-millions figure in the currently selected currency."""
    rate = EUR_TO_GBP if currency == "GBP" else 1.0
    return f"{CURRENCY_SYMBOL[currency]}{value_eur_m * rate:,.1f}M"


# --------------------------------------------------------------------------- #
# Page setup
# --------------------------------------------------------------------------- #
st.set_page_config(
    page_title="EPL Scouting & Valuation",
    page_icon=":soccer:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
      .block-container { padding-top: 2.0rem; max-width: 1320px; }
      h1 { letter-spacing: -0.02em; margin-bottom: 0.2rem; }
      div[data-testid="stMetricValue"] { font-size: 2.1rem; }
      .player-banner {
        border: 1px solid #e3e8e6; border-left: 5px solid #2f7d6f;
        border-radius: 10px; padding: 1.0rem 1.3rem; background: #fbfdfc;
        margin-bottom: 1.1rem;
      }
      .player-banner .name { font-size: 1.85rem; font-weight: 700; line-height: 1.15; }
      .player-banner .meta { color: #55606b; font-size: 1.0rem; margin-top: 0.15rem; }
      .player-banner .stats { margin-top: 0.85rem; display: flex; flex-wrap: wrap; gap: 0.5rem; }
      .player-banner .chip {
        background: #eef4f2; border-radius: 999px; padding: 0.3rem 0.85rem;
        font-size: 0.9rem; color: #1f2933; white-space: nowrap;
      }
      .player-banner .chip b { color: #2f7d6f; }
      .caption-rule { color: #6b7280; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("EPL Player Scouting & Transfer Valuation")
st.markdown(
    '<p class="caption-rule">Select a Premier League player to compare their market '
    "price against a linear model trained on performance data.</p>",
    unsafe_allow_html=True,
)

# Everything below needs a trained pipeline and a dataset.
try:
    pipeline = get_model()
except FileNotFoundError:
    st.error("No trained model found at `models/model.pkl`.")
    st.markdown("Train one first - it takes a couple of seconds and needs no API key:")
    st.code("python train.py", language="bash")
    st.stop()

dataset = get_dataset()
if dataset is None or dataset.empty:
    st.error("No processed dataset found at `data/processed/players_processed.csv`.")
    st.code("python train.py", language="bash")
    st.stop()

metadata = get_metadata()
coefficients = get_coefficients(pipeline)
test_predictions = get_predictions()

# One vectorised pass over the league: actual vs predicted vs delta.
league = scout_table(pipeline, dataset).sort_values("player_name").reset_index(drop=True)

if metadata.get("target_source") == "reference":
    st.warning(SNAPSHOT_NOTICE, icon=":material/history:")


# --------------------------------------------------------------------------- #
# Sidebar: club -> player selection
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Find a player")

    clubs = [ALL_CLUBS] + sorted(league["team"].dropna().unique().tolist())
    club = st.selectbox("Filter by club", clubs, index=0)

    squad = league if club == ALL_CLUBS else league[league["team"] == club]
    squad = squad.sort_values("player_name")

    player_name = st.selectbox(
        "Select player",
        squad["player_name"].tolist(),
        index=0,
        help="Type to search. Narrow the list first with the club filter.",
    )

    st.divider()
    currency = st.radio("Display currency", ("EUR", "GBP"), horizontal=True)

    st.divider()
    st.caption(
        f"**Squads:** {SEASON}, checked {SQUAD_AS_OF}  \n"
        f"**Statistics:** {STATS_SEASON} season  \n"
        f"**Players:** {len(league)} across {league['team'].nunique()} clubs  \n"
        f"**Data source:** {metadata.get('data_source', 'unknown')}  \n"
        f"**Trained:** {metadata.get('trained_at', 'unknown')}"
    )
    if metadata.get("target_source") == "reference":
        st.caption(
            f":warning: {PROVENANCE}. For current squads run `train.py --mode api`; "
            "for real valuations pass `--market-values`."
        )

player = league[league["player_name"] == player_name].iloc[0]


# --------------------------------------------------------------------------- #
# Player summary banner
# --------------------------------------------------------------------------- #
minutes = float(player["minutes_played"])
goals = int(player["goals"])
assists = int(player["assists"])
nineties = max(minutes / 90.0, MIN_NINETIES)
ga_per_90 = (goals + assists) / nineties

st.markdown(
    f"""
    <div class="player-banner">
      <div class="name">{html.escape(str(player['player_name']))}</div>
      <div class="meta">{html.escape(str(player['team']))} &nbsp;&middot;&nbsp;
        {POSITION_LABELS.get(player['position'], player['position'])} &nbsp;&middot;&nbsp;
        {int(player['age'])} years old
        &nbsp;&middot;&nbsp; <span style="color:#8a949e">{html.escape(STATS_SEASON)} stats</span></div>
      <div class="stats">
        <span class="chip">Minutes <b>{minutes:,.0f}</b></span>
        <span class="chip">Goals <b>{goals}</b></span>
        <span class="chip">Assists <b>{assists}</b></span>
        <span class="chip">G+A per 90 <b>{ga_per_90:.2f}</b></span>
        <span class="chip">Goals per 90 <b>{goals / nineties:.2f}</b></span>
        <span class="chip">Assists per 90 <b>{assists / nineties:.2f}</b></span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Valuation comparison
# --------------------------------------------------------------------------- #
actual_m = float(player[TARGET])
predicted_m = float(player["predicted_value_eur_m"])
delta_m = float(player["delta_eur_m"])  # actual - predicted
delta_pct = float(player["delta_pct"])
overvalued = delta_m > 0
verdict = "Overvalued" if overvalued else "Bargain / Undervalued"

col_actual, col_predicted, col_delta = st.columns(3, gap="medium")
col_actual.metric("Actual market value", money(actual_m, currency), border=True)
col_predicted.metric("Model predicted value", money(predicted_m, currency), border=True)
with col_delta:
    st.metric(
        "Valuation delta",
        f"{'+' if overvalued else '-'}{money(abs(delta_m), currency)}",
        delta=f"{delta_pct:+.0f}% · {verdict}",
        # Positive delta means the market pays more than the output justifies,
        # which is bad news for a buyer - so invert Streamlit's usual colouring.
        delta_color="inverse",
        border=True,
    )

st.caption(
    "Delta is **actual minus predicted**. A positive delta means the market prices the "
    "player above what their minutes, output and age justify; a negative delta means the "
    "model sees more value than the asking price."
)


# --------------------------------------------------------------------------- #
# What-if simulator
# --------------------------------------------------------------------------- #
with st.expander("Simulate performance change", expanded=False):
    st.markdown(
        f"Sliders start at {player['player_name']}'s {STATS_SEASON} numbers. "
        "Move them to see how the valuation responds - *what if he scored five more?*"
    )

    # Keying on the player name gives each player a fresh set of sliders, so
    # switching players re-seeds them instead of carrying the last one's values.
    key = f"sim_{player_name}"
    sim_left, sim_right = st.columns(2, gap="large")
    with sim_left:
        sim_goals = st.slider("Goals", *GOALS_RANGE, goals, key=f"{key}_goals")
        sim_assists = st.slider("Assists", *ASSISTS_RANGE, assists, key=f"{key}_assists")
    with sim_right:
        sim_minutes = st.slider(
            "Minutes played", *MINUTES_RANGE, int(minutes), step=10, key=f"{key}_minutes"
        )
        sim_age = st.slider(
            "Age", *AGE_RANGE, int(player["age"]), key=f"{key}_age",
            help=f"Values peak around {PEAK_AGE:g}.",
        )

    sim_value = predict_value(
        pipeline,
        age=sim_age,
        goals=sim_goals,
        assists=sim_assists,
        minutes_played=sim_minutes,
        position=player["position"],
    )
    shift = sim_value - predicted_m
    changed = (
        sim_goals != goals
        or sim_assists != assists
        or sim_minutes != int(minutes)
        or sim_age != int(player["age"])
    )

    sim_a, sim_b, sim_c = st.columns(3, gap="medium")
    sim_a.metric("Baseline prediction", money(predicted_m, currency), border=True)
    sim_b.metric(
        "Simulated prediction",
        money(sim_value, currency),
        delta=f"{'+' if shift >= 0 else '-'}{money(abs(shift), currency)}" if changed else None,
        border=True,
    )
    sim_c.metric(
        "vs. market price",
        f"{'+' if sim_value - actual_m >= 0 else '-'}{money(abs(sim_value - actual_m), currency)}",
        border=True,
    )
    if not changed:
        st.caption("Move a slider to simulate a different season.")


st.divider()

# --------------------------------------------------------------------------- #
# Tabs
# --------------------------------------------------------------------------- #
scout_tab, performance_tab, impact_tab = st.tabs(
    ["League Scout Table", "Model Performance", "Feature Impact"]
)

with scout_tab:
    st.subheader("Where does the model disagree with the market?")

    filter_col, budget_col, view_col = st.columns([1.2, 1.2, 1], gap="large")
    with filter_col:
        positions = st.multiselect(
            "Positions",
            options=list(POSITION_LABELS),
            default=list(POSITION_LABELS),
            help="GK goalkeeper, DF defender, MF midfielder, FW forward.",
        )
    with budget_col:
        budget_ceiling = float(np.ceil(league[TARGET].max() / 10.0) * 10.0)
        max_budget = st.slider(
            f"Maximum budget ({CURRENCY_SYMBOL['EUR']}M, actual market value)",
            0.0,
            budget_ceiling,
            budget_ceiling,
            step=5.0,
        )
    with view_col:
        view = st.radio(
            "Show",
            ("Top 10 bargains", "Top 10 overvalued", "Full league"),
            index=0,
        )

    filtered = league[league["position"].isin(positions) & (league[TARGET] <= max_budget)]

    if view == "Top 10 bargains":
        # Most negative delta = model sees the most value the market is not charging for.
        shown = filtered.nsmallest(10, "delta_eur_m")
        st.caption("Players the model values **above** their market price - scouting targets.")
    elif view == "Top 10 overvalued":
        shown = filtered.nlargest(10, "delta_eur_m")
        st.caption("Players the market prices **above** what their output justifies.")
    else:
        shown = filtered.sort_values("delta_eur_m")
        st.caption(f"All {len(shown)} players matching the filters, biggest bargains first.")

    if shown.empty:
        st.info("No players match those filters. Widen the budget or add a position.")
    else:
        display = shown[
            [
                "player_name", "team", "position", "age", "minutes_played",
                "goals", "assists", TARGET, "predicted_value_eur_m",
                "delta_eur_m", "delta_pct", "verdict",
            ]
        ].rename(
            columns={
                "player_name": "Player", "team": "Club", "position": "Pos",
                "age": "Age", "minutes_played": "Mins", "goals": "G", "assists": "A",
                TARGET: "Market", "predicted_value_eur_m": "Model",
                "delta_eur_m": "Delta", "delta_pct": "Delta %", "verdict": "Verdict",
            }
        )
        st.dataframe(
            display,
            hide_index=True,
            height=min(60 + 35 * len(display), 620),
            column_config={
                "Player": st.column_config.TextColumn(width="medium"),
                "Club": st.column_config.TextColumn(width="small"),
                "Mins": st.column_config.NumberColumn(format="%d"),
                "Market": st.column_config.NumberColumn(
                    "Market €M", format="%.1f", help="Actual recorded market value"
                ),
                "Model": st.column_config.NumberColumn(
                    "Model €M", format="%.1f", help="Model prediction"
                ),
                "Delta": st.column_config.NumberColumn(
                    "Delta €M", format="%+.1f", help="Actual minus predicted"
                ),
                "Delta %": st.column_config.NumberColumn(format="%+.0f%%"),
            },
        )

    st.caption(
        ":warning: Every player here was in the model's training data, so these are "
        "in-sample fits. Treat the ranking as a shortlist of *questions worth asking*, "
        "not as valuations - the held-out accuracy in **Model Performance** is the honest "
        "measure of what the model knows."
    )

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
        st.info("No saved test predictions yet. Re-run `python train.py` to generate them.")
    else:
        scatter_col, residual_col = st.columns(2, gap="large")
        with scatter_col:
            st.pyplot(
                plot_actual_vs_predicted(test_predictions["actual"], test_predictions["predicted"])
            )
            st.caption(
                "Each dot is a held-out player. The dashed line is a perfect prediction - "
                "dots below it are players the model prices under the market."
            )
        with residual_col:
            st.pyplot(plot_residuals(test_predictions["actual"], test_predictions["predicted"]))
            st.caption(
                "Residuals should scatter randomly around zero. A funnel shape means the "
                "model is less reliable for expensive players."
            )

    with st.expander("How these numbers are produced"):
        cv = metadata.get("cross_validation", {})
        st.markdown(
            f"""
            - **Model:** `{metadata.get('model', 'sklearn.linear_model.LinearRegression')}`
              fitted on `log1p(value)`, so effects are multiplicative and predictions stay positive
            - **Features:** {', '.join(metadata.get('features', []))}
            - **Rows:** {metadata.get('n_rows', 'n/a')} players, held-out split scored above
            - **Cross-validated R²:** {cv.get('cv_r2_mean', float('nan')):.3f}
              (+/- {cv.get('cv_r2_std', float('nan')):.3f} over {cv.get('cv_folds', 0)} folds)

            MAE and RMSE are in the same units as the target (EUR millions). An R² around
            0.5 is what four performance statistics can honestly explain about transfer
            values - contract length, injuries, reputation and selling-club leverage drive
            the rest, and none of them are in the feature set.
            """
        )

with impact_tab:
    st.subheader("Which stats drive the valuation?")
    st.markdown(
        "The model fits `log1p(value)`, so each bar is the **percentage change** in "
        "predicted value from a one-standard-deviation increase in that feature. "
        f"Position bars read against a {REFERENCE_POSITION}."
    )

    chart_col, table_col = st.columns([1.4, 1], gap="large")
    with chart_col:
        st.pyplot(plot_coefficients(coefficients))
    with table_col:
        st.dataframe(
            coefficients[["label", "pct_per_sd"]].rename(
                columns={"label": "Feature", "pct_per_sd": "Effect"}
            ),
            hide_index=True,
            height=340,
            column_config={
                "Feature": st.column_config.TextColumn(width="medium"),
                "Effect": st.column_config.NumberColumn("% per SD", format="%+.1f%%"),
            },
        )
        st.metric(
            f"Baseline (average {REFERENCE_POSITION})",
            f"€{get_intercept(pipeline):.1f}M",
            border=True,
        )

    st.caption(
        "These are *conditional* effects - each holds the others fixed. The position bars "
        "are the clearest example of why that matters: holding goals-per-90 constant, the "
        "model reports a discount for forwards, because \"a forward scoring at a "
        "goalkeeper's rate\" is a counterfactual that never occurs in the data. Read "
        "position and output together, not bar by bar."
    )

st.divider()
st.caption(
    "Predictions are indicative only - a linear model on public match statistics cannot "
    "capture contract length, injuries, sell-on clauses or club negotiating position. "
    f"{PROVENANCE}."
)
