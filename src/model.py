"""Model construction, training, evaluation and persistence.

A single :class:`~sklearn.pipeline.Pipeline` holds both the preprocessing and
the ``LinearRegression`` estimator, so the artefact saved to ``models/model.pkl``
is self-contained: the dashboard loads it and calls ``predict`` on raw player
attributes without repeating any transformation logic.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from .preprocessor import (
    FEATURE_COLUMNS,
    TARGET,
    build_preprocessor,
    humanise_feature_names,
    make_player_frame,
)

LOGGER = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/model.pkl")
DEFAULT_METADATA_PATH = Path("models/metadata.json")
DEFAULT_PREDICTIONS_PATH = Path("data/processed/test_predictions.csv")

#: Step names inside the pipeline, referenced when digging out coefficients.
PREPROCESSOR_STEP = "preprocessor"
REGRESSOR_STEP = "regressor"


# --------------------------------------------------------------------------- #
# Build & train
# --------------------------------------------------------------------------- #
def build_pipeline() -> Pipeline:
    """Preprocessing + ordinary least squares on a log-transformed target.

    Market values are strongly right-skewed (EUR 1.5M to EUR 180M here, skew
    1.9), which a plain OLS fit handles badly: it chases the handful of
    superstars and can extrapolate to negative fees for fringe players.
    Regressing on ``log1p(value)`` makes the model *multiplicative* - each
    feature scales value by a factor rather than adding a fixed number of
    millions - which is how the transfer market actually behaves. It also
    keeps predictions positive and improves cross-validated R^2.

    ``TransformedTargetRegressor`` applies ``expm1`` on the way out, so
    ``predict`` still returns EUR millions and callers see no difference.
    """
    return Pipeline(
        steps=[
            (PREPROCESSOR_STEP, build_preprocessor()),
            (
                REGRESSOR_STEP,
                TransformedTargetRegressor(
                    regressor=LinearRegression(),
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


def _linear_model(pipeline: Pipeline) -> LinearRegression:
    """Reach the underlying OLS estimator through the target transformer."""
    regressor = pipeline.named_steps[REGRESSOR_STEP]
    return getattr(regressor, "regressor_", regressor)


def train_model(X_train: pd.DataFrame, y_train: pd.Series) -> Pipeline:
    """Fit the full pipeline on the training split."""
    pipeline = build_pipeline()
    pipeline.fit(X_train[list(FEATURE_COLUMNS)], y_train)
    LOGGER.info("Trained LinearRegression on %d players", len(X_train))
    return pipeline


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def evaluate(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Score a fitted pipeline.

    Returns R^2, MAE and RMSE (the last two in EUR millions, the same units as
    the target, so they can be read as "average error in millions"), plus MAPE
    for a scale-free sanity check.
    """
    predictions = pipeline.predict(X[list(FEATURE_COLUMNS)])
    y_true = np.asarray(y, dtype=float)

    mae = float(mean_absolute_error(y_true, predictions))
    rmse = float(np.sqrt(mean_squared_error(y_true, predictions)))
    # Guard against division by zero even though cleaning drops non-positive targets.
    denominator = np.where(np.abs(y_true) < 1e-9, np.nan, y_true)
    mape = float(np.nanmean(np.abs((y_true - predictions) / denominator)) * 100.0)

    return {
        "r2": float(r2_score(y_true, predictions)),
        "mae": mae,
        "rmse": rmse,
        "mape": mape,
        "n_samples": int(len(y_true)),
    }


def cross_validate_r2(
    X: pd.DataFrame, y: pd.Series, n_splits: int = 5, random_state: int = 42
) -> dict[str, float]:
    """K-fold R^2 on the full dataset - a stability check on the single split."""
    n_splits = max(2, min(n_splits, len(X) // 2))
    scores = cross_val_score(
        build_pipeline(),
        X[list(FEATURE_COLUMNS)],
        y,
        cv=KFold(n_splits=n_splits, shuffle=True, random_state=random_state),
        scoring="r2",
    )
    return {"cv_r2_mean": float(scores.mean()), "cv_r2_std": float(scores.std()), "cv_folds": n_splits}


def get_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Extract standardised regression coefficients, largest effect first.

    Numeric features are standardised, so magnitudes are directly comparable.
    Because the model fits ``log1p(value)``, a coefficient is a *multiplicative*
    effect: ``exp(coef)`` is the factor a one-standard-deviation increase
    applies to the prediction. ``pct_per_sd`` expresses that as a percentage,
    which is what the dashboard plots.
    """
    preprocessor = pipeline.named_steps[PREPROCESSOR_STEP]
    linear = _linear_model(pipeline)

    raw_names = list(preprocessor.get_feature_names_out())
    coefficients = np.asarray(linear.coef_, dtype=float).ravel()

    frame = pd.DataFrame(
        {
            "feature": raw_names,
            "label": humanise_feature_names(raw_names),
            "coefficient": coefficients,
        }
    )
    frame["multiplier_per_sd"] = np.exp(frame["coefficient"])
    frame["pct_per_sd"] = (frame["multiplier_per_sd"] - 1.0) * 100.0
    frame["abs_coefficient"] = frame["coefficient"].abs()
    return frame.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def get_intercept(pipeline: Pipeline) -> float:
    """Baseline prediction in EUR millions: an average player at the reference position."""
    return float(np.expm1(_linear_model(pipeline).intercept_))


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict_value(
    pipeline: Pipeline,
    age: float,
    goals: float,
    assists: float,
    minutes_played: float,
    position: str,
    clip_negative: bool = True,
) -> float:
    """Predict one player's transfer value, in EUR millions.

    An unconstrained linear model can extrapolate below zero for profiles at
    the edge of the training distribution (an ageing defender with no output,
    say), which is meaningless as a transfer fee. ``clip_negative`` floors the
    result at zero; pass ``False`` to see the raw fit, which is how the
    dashboard detects that a prediction was clipped and says so.
    """
    features = make_player_frame(age, goals, assists, minutes_played, position)
    prediction = float(pipeline.predict(features)[0])
    return max(prediction, 0.0) if clip_negative else prediction


def predict_frame(pipeline: Pipeline, frame: pd.DataFrame, clip_negative: bool = True) -> np.ndarray:
    """Predict values (EUR millions) for a whole player frame at once.

    The scout table needs a prediction for every player in the league; doing
    that as one vectorised call rather than a row-by-row loop keeps the
    dashboard responsive.
    """
    features = frame[list(FEATURE_COLUMNS)]
    predictions = np.asarray(pipeline.predict(features), dtype=float)
    return np.maximum(predictions, 0.0) if clip_negative else predictions


def scout_table(pipeline: Pipeline, dataset: pd.DataFrame) -> pd.DataFrame:
    """Build the league-wide valuation comparison.

    Returns one row per player with their actual value, the model's prediction
    and the gap between them::

        delta = actual market value - model prediction

    A **positive** delta means the market prices the player above what their
    output justifies (overvalued). A **negative** delta means the model sees
    more value than the market is charging - a scout's bargain.
    """
    table = dataset.copy()
    table["predicted_value_eur_m"] = predict_frame(pipeline, table)
    table["delta_eur_m"] = table[TARGET] - table["predicted_value_eur_m"]
    # Percentage keeps a EUR 2M gap on a EUR 4M player from being dwarfed by a
    # EUR 10M gap on a EUR 150M player.
    table["delta_pct"] = 100.0 * table["delta_eur_m"] / table[TARGET].replace(0, np.nan)
    table["verdict"] = np.where(table["delta_eur_m"] > 0, "Overvalued", "Undervalued")
    return table


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def save_model(pipeline: Pipeline, path: str | Path = DEFAULT_MODEL_PATH) -> Path:
    """Serialise the fitted pipeline with joblib."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, destination)
    LOGGER.info("Saved model to %s", destination)
    return destination


def load_model(path: str | Path = DEFAULT_MODEL_PATH) -> Pipeline:
    """Load a pipeline saved by :func:`save_model`."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"No model at {source}. Train one first: `python train.py --mode mock`."
        )
    return joblib.load(source)


def save_metadata(metadata: Mapping[str, Any], path: str | Path = DEFAULT_METADATA_PATH) -> Path:
    """Write run metrics and provenance next to the model as JSON."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(dict(metadata), indent=2, default=str), encoding="utf-8")
    LOGGER.info("Saved metadata to %s", destination)
    return destination


def load_metadata(path: str | Path = DEFAULT_METADATA_PATH) -> dict[str, Any]:
    """Read the metadata JSON, or return an empty dict when absent."""
    source = Path(path)
    if not source.exists():
        return {}
    try:
        return json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        LOGGER.warning("Could not parse %s", source)
        return {}


def build_metadata(
    train_metrics: Mapping[str, float],
    test_metrics: Mapping[str, float],
    cv_metrics: Mapping[str, float] | None = None,
    data_source: str = "unknown",
    target_source: str = "synthetic",
    n_rows: int = 0,
) -> dict[str, Any]:
    """Assemble the provenance record the dashboard displays."""
    metadata: dict[str, Any] = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": "sklearn.linear_model.LinearRegression",
        "target": TARGET,
        "target_units": "EUR millions",
        "features": list(FEATURE_COLUMNS),
        "data_source": data_source,
        "target_source": target_source,
        "n_rows": int(n_rows),
        "train_metrics": dict(train_metrics),
        "test_metrics": dict(test_metrics),
    }
    if cv_metrics:
        metadata["cross_validation"] = dict(cv_metrics)
    return metadata


def save_test_predictions(
    y_true: pd.Series,
    y_pred: np.ndarray,
    path: str | Path = DEFAULT_PREDICTIONS_PATH,
) -> Path:
    """Persist held-out actual/predicted pairs for the dashboard's scatter plot."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"actual": np.asarray(y_true, dtype=float), "predicted": np.asarray(y_pred, dtype=float)}).to_csv(
        destination, index=False
    )
    LOGGER.info("Saved %d test predictions to %s", len(y_true), destination)
    return destination


def load_test_predictions(path: str | Path = DEFAULT_PREDICTIONS_PATH) -> pd.DataFrame | None:
    """Load the saved predictions, or ``None`` when the file is missing."""
    source = Path(path)
    if not source.exists():
        return None
    return pd.read_csv(source)


def format_metrics(metrics: Mapping[str, float]) -> str:
    """One-line, human-readable metric summary for console output."""
    return (
        f"R2 = {metrics['r2']:6.3f} | "
        f"MAE = EUR {metrics['mae']:5.2f}M | "
        f"RMSE = EUR {metrics['rmse']:5.2f}M | "
        f"MAPE = {metrics['mape']:5.1f}%"
    )
