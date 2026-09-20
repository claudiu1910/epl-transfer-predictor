#!/usr/bin/env python3
"""Train the transfer-value model and save the pipeline for the scouting dashboard.

Examples
--------
Default: the bundled 2024/25 dataset of real Premier League players::

    python train.py

Live statistics from football-data.org, valuations from the bundled dataset::

    export FOOTBALL_DATA_API_KEY=...
    python train.py --mode api

Your own market-value export, joined on player name::

    python train.py --market-values data/raw/my_values.csv

Synthetic players, for testing the pipeline without real names::

    python train.py --mock
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from src.data_loader import (
    API_KEY_ENV_VAR,
    REFERENCE_DATASET_PATH,
    TARGET_EUR,
    MissingAPIKeyError,
    load_dataset,
)
from src.model import (
    DEFAULT_METADATA_PATH,
    DEFAULT_MODEL_PATH,
    DEFAULT_PREDICTIONS_PATH,
    build_metadata,
    cross_validate_r2,
    evaluate,
    format_metrics,
    get_coefficients,
    get_intercept,
    scout_table,
    save_metadata,
    save_model,
    save_test_predictions,
    train_model,
)
from src.preprocessor import (
    FEATURE_COLUMNS,
    IDENTITY_COLUMNS,
    TARGET,
    prepare_dataset,
    split_dataset,
)
from src.visualize import REPORTS_DIR, save_diagnostic_figures

LOGGER = logging.getLogger("train")

RAW_DATASET_PATH = Path("data/raw/players_raw.csv")
PROCESSED_DATASET_PATH = Path("data/processed/players_processed.csv")
SCOUT_TABLE_PATH = Path("data/processed/scout_table.csv")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and persist the EPL transfer-value regression pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    source = parser.add_argument_group("data source")
    source.add_argument(
        "--mode",
        choices=("auto", "reference", "api", "mock"),
        default="auto",
        help="'reference' uses the bundled dataset of real Premier League players, "
        "'api' pulls live stats from football-data.org, 'mock' generates synthetic "
        "players. 'auto' resolves to 'reference'.",
    )
    source.add_argument(
        "--mock",
        action="store_true",
        help="Shorthand for --mode mock; synthetic players instead of real ones.",
    )
    source.add_argument(
        "--reference-path",
        type=Path,
        default=REFERENCE_DATASET_PATH,
        help="CSV holding the curated dataset; generated on first use.",
    )
    source.add_argument(
        "--rebuild-reference",
        action="store_true",
        help="Overwrite the reference CSV from src/reference_data.py before training.",
    )
    source.add_argument("--n-mock", type=int, default=320, help="Players to generate in mock mode.")
    source.add_argument(
        "--scorer-limit", type=int, default=100, help="Rows requested from the scorers endpoint."
    )
    source.add_argument(
        "--include-non-scorers",
        action="store_true",
        help="Keep squad members with no scoring record (their stats are zeroed).",
    )

    target = parser.add_argument_group("target variable")
    target.add_argument(
        "--market-values",
        type=Path,
        default=None,
        help="CSV of real market values to join on player name. Omitted -> synthetic baseline.",
    )
    target.add_argument(
        "--on-missing",
        choices=("drop", "synthesise"),
        default="drop",
        help="What to do with players the market-value CSV does not cover.",
    )

    training = parser.add_argument_group("training")
    training.add_argument("--test-size", type=float, default=0.2, help="Held-out fraction.")
    training.add_argument("--random-state", type=int, default=42, help="Seed for splits and mock data.")
    training.add_argument("--cv-folds", type=int, default=5, help="K-fold count for the R^2 stability check.")
    training.add_argument("--no-cv", action="store_true", help="Skip cross-validation.")

    output = parser.add_argument_group("output")
    output.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    output.add_argument("--metadata-path", type=Path, default=DEFAULT_METADATA_PATH)
    output.add_argument("--predictions-path", type=Path, default=DEFAULT_PREDICTIONS_PATH)
    output.add_argument("--reports-dir", type=Path, default=REPORTS_DIR)
    output.add_argument("--no-figures", action="store_true", help="Skip writing diagnostic PNGs.")
    output.add_argument("-v", "--verbose", action="store_true", help="Debug-level logging.")

    args = parser.parse_args(argv)
    if args.mock:
        args.mode = "mock"
    return args


def _banner(text: str) -> None:
    print(f"\n{text}\n{'-' * len(text)}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)-8s %(name)s: %(message)s",
    )

    # -- 1. Load ------------------------------------------------------------ #
    _banner("1. Loading data")
    if args.rebuild_reference:
        from src.data_loader import build_reference_dataset

        build_reference_dataset(args.reference_path)
        print(f"Rebuilt reference dataset -> {args.reference_path}")

    try:
        raw = load_dataset(
            mode=args.mode,
            market_value_csv=args.market_values,
            reference_path=args.reference_path,
            n_mock=args.n_mock,
            seed=args.random_state,
            scorer_limit=args.scorer_limit,
            include_non_scorers=args.include_non_scorers,
            on_missing=args.on_missing,
        )
    except MissingAPIKeyError as exc:
        print(f"\nERROR: {exc}\n", file=sys.stderr)
        print(
            f"Set {API_KEY_ENV_VAR}, or use the bundled dataset with --mode reference.",
            file=sys.stderr,
        )
        return 2

    data_source = raw.attrs.get("source", args.mode)
    if args.market_values:
        target_source = "csv"
    elif data_source == "mock":
        target_source = "synthetic"
    else:
        target_source = "reference"
    print(f"Loaded {len(raw)} players (source: {data_source}, target: {target_source})")

    if raw.empty:
        print("ERROR: no players returned - nothing to train on.", file=sys.stderr)
        return 1

    RAW_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(RAW_DATASET_PATH, index=False)
    print(f"Raw snapshot -> {RAW_DATASET_PATH}")

    # -- 2. Clean & engineer ------------------------------------------------- #
    _banner("2. Preprocessing")
    dataset = prepare_dataset(raw)
    print(f"{len(dataset)} players survive cleaning")
    print(f"Features: {', '.join(FEATURE_COLUMNS)}")
    print("\nPosition mix:")
    print(dataset["position"].value_counts().to_string())
    print("\nClubs represented:", dataset["team"].nunique())
    print(f"\nTarget ({TARGET}, EUR millions):")
    print(dataset[TARGET].describe().round(2).to_string())

    if len(dataset) < 20:
        print(
            f"\nWARNING: only {len(dataset)} rows - metrics will be extremely noisy.",
            file=sys.stderr,
        )

    PROCESSED_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(PROCESSED_DATASET_PATH, index=False)
    print(f"\nProcessed dataset -> {PROCESSED_DATASET_PATH}")

    # -- 3. Split & train ----------------------------------------------------- #
    _banner("3. Training")
    X_train, X_test, y_train, y_test = split_dataset(
        dataset, test_size=args.test_size, random_state=args.random_state
    )
    print(f"Train: {len(X_train)} players | Test: {len(X_test)} players")

    pipeline = train_model(X_train, y_train)

    # -- 4. Evaluate ---------------------------------------------------------- #
    _banner("4. Evaluation")
    train_metrics = evaluate(pipeline, X_train, y_train)
    test_metrics = evaluate(pipeline, X_test, y_test)
    print(f"Train  {format_metrics(train_metrics)}")
    print(f"Test   {format_metrics(test_metrics)}")

    cv_metrics = None
    if not args.no_cv and len(dataset) >= 10:
        cv_metrics = cross_validate_r2(
            dataset[list(FEATURE_COLUMNS)],
            dataset[TARGET],
            n_splits=args.cv_folds,
            random_state=args.random_state,
        )
        print(
            f"{cv_metrics['cv_folds']}-fold CV R2 = "
            f"{cv_metrics['cv_r2_mean']:.3f} (+/- {cv_metrics['cv_r2_std']:.3f})"
        )

    coefficients = get_coefficients(pipeline)
    print(f"\nBaseline (average midfielder): EUR {get_intercept(pipeline):.2f}M")
    print("\nStandardised effects (change in value per std. dev. of each feature):")
    print(
        coefficients[["label", "pct_per_sd"]]
        .rename(columns={"label": "feature", "pct_per_sd": "pct_change"})
        .to_string(index=False, float_format=lambda v: f"{v:+.1f}%")
    )

    # -- 5. Persist ------------------------------------------------------------ #
    _banner("5. Saving artefacts")
    save_model(pipeline, args.model_path)
    print(f"Model     -> {args.model_path}")

    y_pred_test = pipeline.predict(X_test[list(FEATURE_COLUMNS)])
    save_test_predictions(y_test, y_pred_test, args.predictions_path)
    print(f"Test set  -> {args.predictions_path}")

    # League-wide valuation comparison, ready for the dashboard's scout tab.
    scouting = scout_table(pipeline, dataset)
    scouting_columns = [
        *IDENTITY_COLUMNS,
        "age",
        "minutes_played",
        "goals",
        "assists",
        TARGET,
        TARGET_EUR,
        "predicted_value_eur_m",
        "delta_eur_m",
        "delta_pct",
        "verdict",
    ]
    SCOUT_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    scouting[scouting_columns].to_csv(SCOUT_TABLE_PATH, index=False)
    print(f"Scout tbl -> {SCOUT_TABLE_PATH}")

    bargains = scouting.nsmallest(3, "delta_eur_m")
    premiums = scouting.nlargest(3, "delta_eur_m")
    print("\nBiggest model-vs-market gaps:")
    for label, rows in (("bargain (undervalued)", bargains), ("premium (overvalued)", premiums)):
        for _, row in rows.iterrows():
            print(
                f"  {label:22} {row['player_name']:<24} "
                f"market EUR {row[TARGET]:6.1f}M | model EUR {row['predicted_value_eur_m']:6.1f}M "
                f"({row['delta_eur_m']:+.1f}M)"
            )

    metadata = build_metadata(
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        cv_metrics=cv_metrics,
        data_source=str(data_source),
        target_source=target_source,
        n_rows=len(dataset),
    )
    metadata["coefficients"] = coefficients[["feature", "label", "coefficient"]].to_dict("records")
    metadata["intercept"] = get_intercept(pipeline)
    save_metadata(metadata, args.metadata_path)
    print(f"Metadata  -> {args.metadata_path}")

    if not args.no_figures:
        written = save_diagnostic_figures(
            y_test, y_pred_test, coefficients, values=dataset[TARGET], output_dir=args.reports_dir
        )
        print(f"Figures   -> {', '.join(str(p) for p in written)}")

    _banner("Done")
    print("Launch the scouting dashboard with:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
