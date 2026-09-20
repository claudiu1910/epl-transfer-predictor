"""Cleaning, feature engineering and the sklearn preprocessing pipeline.

The module owns the feature contract shared by training (`train.py`) and
inference (`app.py`): if a column is added here, both sides pick it up without
further edits.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data_loader import (
    AGE_RANGE,
    ASSISTS_RANGE,
    GOALS_RANGE,
    MAX_SEASON_MINUTES,
    MINUTES_RANGE,
    PEAK_AGE,
    POSITIONS,
    normalise_position,
)

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Feature contract
# --------------------------------------------------------------------------- #
TARGET = "market_value_eur_m"

#: Straight from the data source (and from the dashboard sliders).
RAW_NUMERIC_FEATURES: tuple[str, ...] = ("age", "goals", "assists", "minutes")

#: Derived in :func:`engineer_features`.  All of them are computable from the
#: raw four, so the dashboard never needs inputs the user cannot provide.
ENGINEERED_FEATURES: tuple[str, ...] = (
    "goals_per_90",
    "assists_per_90",
    "years_from_peak_sq",
)

NUMERIC_FEATURES: tuple[str, ...] = RAW_NUMERIC_FEATURES + ENGINEERED_FEATURES
CATEGORICAL_FEATURES: tuple[str, ...] = ("position",)
FEATURE_COLUMNS: tuple[str, ...] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

#: Reference category for the one-hot encoder; coefficients read as a premium
#: (or discount) relative to this position.
REFERENCE_POSITION = "GK"
_ENCODER_CATEGORIES = [REFERENCE_POSITION] + [p for p in POSITIONS if p != REFERENCE_POSITION]

#: Minimum 90s used as a per-90 denominator, so a 12-minute cameo with one goal
#: does not turn into 7.5 goals per 90.
MIN_NINETIES = 1.0

_COLUMN_RANGES = {
    "age": AGE_RANGE,
    "goals": GOALS_RANGE,
    "assists": ASSISTS_RANGE,
    "minutes": MINUTES_RANGE,
}


# --------------------------------------------------------------------------- #
# Cleaning
# --------------------------------------------------------------------------- #
def clean_players(frame: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Coerce types, fill gaps and clip the raw player frame into shape.

    Args:
        frame: Raw output of :mod:`src.data_loader`.
        require_target: Drop rows without a usable market value.  Set to
            ``False`` when cleaning a single row for inference.

    Returns:
        A cleaned copy - the input is never mutated.
    """
    data = frame.copy()

    if "name" in data.columns:
        data["name"] = data["name"].astype(str).str.strip()
        before = len(data)
        data = data[data["name"].str.len() > 0].drop_duplicates(subset=["name"], keep="first")
        if before != len(data):
            LOGGER.info("Dropped %d blank/duplicate player rows", before - len(data))

    # Positions -> GK / DF / MF / FW.
    if "position" not in data.columns:
        data["position"] = "MF"
    data["position"] = data["position"].map(normalise_position).astype("string")

    # Numerics: coerce, then fill.  Age falls back to the squad median because
    # a zero would be nonsense; counting stats legitimately default to zero.
    for column in RAW_NUMERIC_FEATURES:
        if column not in data.columns:
            data[column] = np.nan
        data[column] = pd.to_numeric(data[column], errors="coerce")

    if data["age"].notna().any():
        data["age"] = data["age"].fillna(data["age"].median())
    else:
        data["age"] = data["age"].fillna(float(np.mean(AGE_RANGE)))

    for column in ("goals", "assists", "minutes"):
        missing = int(data[column].isna().sum())
        if missing:
            LOGGER.info("Filling %d missing '%s' values with 0", missing, column)
        data[column] = data[column].fillna(0.0)

    # Clip to the domains the dashboard exposes, so training and inference
    # never see different ranges.
    for column, (low, high) in _COLUMN_RANGES.items():
        data[column] = data[column].clip(lower=low, upper=high)

    if "matches" in data.columns:
        data["matches"] = pd.to_numeric(data["matches"], errors="coerce").fillna(0).clip(lower=0)

    if require_target:
        if TARGET not in data.columns:
            raise KeyError(f"Expected a '{TARGET}' column; got {list(data.columns)}")
        data[TARGET] = pd.to_numeric(data[TARGET], errors="coerce")
        before = len(data)
        data = data[data[TARGET].notna() & (data[TARGET] > 0)]
        if before != len(data):
            LOGGER.info("Dropped %d rows without a positive market value", before - len(data))

    return data.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Feature engineering
# --------------------------------------------------------------------------- #
def engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add the derived columns listed in :data:`ENGINEERED_FEATURES`.

    * ``goals_per_90`` / ``assists_per_90`` - rate stats, which separate a
      prolific substitute from a goalless ever-present.
    * ``years_from_peak_sq`` - squared distance from the mid-twenties peak,
      letting a *linear* model bend the age/value relationship into the arc the
      transfer market actually pays for.
    """
    data = frame.copy()
    nineties = np.maximum(data["minutes"].to_numpy(dtype=float) / 90.0, MIN_NINETIES)

    data["goals_per_90"] = data["goals"].to_numpy(dtype=float) / nineties
    data["assists_per_90"] = data["assists"].to_numpy(dtype=float) / nineties
    data["years_from_peak_sq"] = (data["age"].to_numpy(dtype=float) - PEAK_AGE) ** 2
    return data


def prepare_dataset(frame: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Convenience wrapper: :func:`clean_players` then :func:`engineer_features`."""
    return engineer_features(clean_players(frame, require_target=require_target))


def make_player_frame(
    age: float,
    goals: float,
    assists: float,
    minutes: float,
    position: str,
) -> pd.DataFrame:
    """Build a single-row, fully engineered feature frame for inference.

    Used by the dashboard so a live prediction goes through exactly the same
    transformations as the training data.
    """
    row = pd.DataFrame(
        [
            {
                "name": "input",
                "age": age,
                "goals": goals,
                "assists": assists,
                "minutes": minutes,
                "position": normalise_position(position),
            }
        ]
    )
    return prepare_dataset(row, require_target=False)[list(FEATURE_COLUMNS)]


# --------------------------------------------------------------------------- #
# sklearn preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor() -> ColumnTransformer:
    """Standardise numerics and one-hot encode position.

    Scaling matters here beyond convergence: with every numeric feature on a
    unit scale the regression coefficients become directly comparable, which is
    what the dashboard's "Feature Impact" chart reads off.
    """
    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), list(NUMERIC_FEATURES)),
            (
                "categorical",
                OneHotEncoder(
                    categories=[_ENCODER_CATEGORIES],
                    drop="first",  # REFERENCE_POSITION becomes the baseline
                    sparse_output=False,
                ),
                list(CATEGORICAL_FEATURES),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_dataset(
    frame: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify_by_position: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split into train/test feature frames and target series.

    Stratifies on position when every position has at least two players, so a
    small squad sample cannot leave e.g. goalkeepers out of the training set.
    """
    features = frame[list(FEATURE_COLUMNS)]
    target = frame[TARGET]

    stratify = None
    if stratify_by_position:
        counts = frame["position"].value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = frame["position"]
        else:
            LOGGER.info("Skipping stratification - not enough players per position")

    return train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )


def humanise_feature_names(names: Sequence[str]) -> list[str]:
    """Turn encoder output into labels fit for a chart axis."""
    pretty = {
        "age": "Age",
        "goals": "Goals",
        "assists": "Assists",
        "minutes": "Minutes played",
        "goals_per_90": "Goals per 90",
        "assists_per_90": "Assists per 90",
        "years_from_peak_sq": f"Age distance from {PEAK_AGE:g} (squared)",
    }
    labels = []
    for name in names:
        if name in pretty:
            labels.append(pretty[name])
        elif name.startswith("position_"):
            labels.append(f"Position: {name.split('_', 1)[1]} (vs {REFERENCE_POSITION})")
        else:
            labels.append(name.replace("_", " ").capitalize())
    return labels
