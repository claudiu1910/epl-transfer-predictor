"""EPL transfer-value predictor.

A small, modular pipeline: fetch Premier League player statistics, engineer
features, fit a linear regression on transfer values, and serve predictions
through a Streamlit dashboard.

Submodules
----------
``data_loader``   football-data.org ingestion, mock generator, target attachment
``preprocessor``  cleaning, feature engineering, sklearn ColumnTransformer
``model``         pipeline construction, metrics, persistence
``visualize``     matplotlib/seaborn regression diagnostics
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["data_loader", "preprocessor", "model", "visualize", "__version__"]
