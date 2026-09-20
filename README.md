# EPL Transfer Value Predictor

Predict English Premier League player transfer values from performance statistics,
end to end: API ingestion → cleaning and feature engineering → linear regression →
an interactive Streamlit dashboard.

The whole pipeline runs offline with no API key, so you can clone, train and open
the dashboard in under a minute.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py --mock      # train on generated data, no API key needed
streamlit run app.py        # open the dashboard
```

`train.py --mock` writes `models/model.pkl`, `models/metadata.json`,
`data/processed/test_predictions.csv` and a set of diagnostic PNGs in `reports/`.
The dashboard loads those artefacts, so predictions are instant — it never retrains.

## Using live Premier League data

Get a free key at [football-data.org](https://www.football-data.org/client/register),
then:

```bash
export FOOTBALL_DATA_API_KEY=your_key_here
python train.py --mode api
```

`--mode auto` (the default) uses the API when the key is set and falls back to the
mock generator when it is not.

### Rate limiting

The free tier allows **10 requests per minute**. `RateLimiter` in
[`src/data_loader.py`](src/data_loader.py) enforces this with a sliding window,
blocking before a call rather than reacting to an HTTP 429. If the server throttles
anyway, the client honours `Retry-After`, and it retries transient 5xx errors with
exponential backoff. A full pull costs **2 requests** (squads + scorers), so you
will not come close to the limit.

---

## The target variable

football-data.org tracks **match events, not transfer fees**, so the target has to
come from somewhere else. Two supported modes:

**1. Join your own market values (recommended for real results)**

```bash
python train.py --mode api --market-values data/raw/market_values.csv
```

The CSV needs a name column (`name` / `player` / `player_name`) and a value column
(`market_value_eur_m` / `market_value` / `value`). Values are read as EUR millions;
raw euro amounts such as `45000000` are detected and scaled automatically. Names are
matched case- and accent-insensitively. See
[`data/raw/market_values.example.csv`](data/raw/market_values.example.csv).

Players the CSV does not cover are dropped by default, which keeps the training set
entirely real. Pass `--on-missing synthesise` to fill the gaps instead; those rows
are flagged `synthetic` in the `value_source` column.

**2. Synthetic valuation baseline (default, for offline runs)**

`synthesise_market_values()` derives a plausible figure from a *multiplicative*
model — a bell-shaped age curve peaking at 25.5, a position premium
(FW 1.45 / MF 1.22 / DF 0.96 / GK 0.72), playing time, goal and assist output, and
lognormal noise. The multiplicative structure keeps the task genuinely non-trivial
for a linear model.

> **This is a documented stand-in, not a market quote.** The dashboard labels any
> model trained this way. Treat the numbers as a demonstration of the pipeline,
> not as transfer advice.

### A note on minutes played

The free tier does not expose per-player minutes, so `minutes` is estimated as
`playedMatches × 78`. The constant is `MINUTES_PER_APPEARANCE_ESTIMATE` in
`src/data_loader.py`. It is an assumption, not a measurement — swap in a real
minutes source if you have one.

---

## Project layout

```
epl-transfer-predictor/
├── data/
│   ├── raw/                     # API snapshots, your market-value CSV
│   └── processed/               # cleaned dataset + held-out predictions
├── models/                      # model.pkl + metadata.json (git-ignored)
├── reports/                     # diagnostic PNGs from training
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # API client, rate limiting, mock data, target
│   ├── preprocessor.py          # cleaning, feature engineering, ColumnTransformer
│   ├── model.py                 # pipeline, metrics, persistence
│   └── visualize.py             # matplotlib/seaborn diagnostics
├── app.py                       # Streamlit dashboard
├── train.py                     # trains and saves the pipeline
├── requirements.txt
└── README.md
```

## Features and model

| Feature | Source | Why |
| --- | --- | --- |
| `age` | API / slider | Value tracks a career arc |
| `goals`, `assists` | API / sliders | Direct attacking output |
| `minutes` | API / slider | Proxy for trust and availability |
| `goals_per_90`, `assists_per_90` | derived | Separates an efficient substitute from a goalless ever-present |
| `years_from_peak_sq` | derived | `(age − 25.5)²`, letting a *linear* model bend the age curve |
| `position` | API / dropdown | One-hot encoded, GK as the reference category |

Every derived feature is computable from the four slider inputs, so the dashboard
never needs data a user cannot supply.

The pipeline is a single `sklearn.pipeline.Pipeline`:
`StandardScaler` on numerics + `OneHotEncoder` on position → `LinearRegression`.
Because it is saved whole, the dashboard calls `predict()` on raw player attributes
with no duplicated transformation logic.

Scaling is not just for conditioning — with all numerics on a unit scale, the
coefficients become directly comparable, which is exactly what the **Feature Impact**
tab reads off.

## Metrics

`train.py` reports R², MAE and RMSE on the held-out split, plus a 5-fold
cross-validated R² as a stability check. MAE and RMSE are in EUR millions, the same
units as the target, so an MAE of 3.9 means the model is off by about €3.9M for a
typical player.

A representative `--mock` run (320 players, 80/20 split):

```
Train  R2 =  0.838 | MAE = EUR  4.01M | RMSE = EUR  6.24M
Test   R2 =  0.862 | MAE = EUR  3.86M | RMSE = EUR  7.31M
5-fold CV R2 = 0.809 (+/- 0.115)
```

## Dashboard

```bash
streamlit run app.py
```

* **Sidebar** — position dropdown (FW / MF / DF / GK) and sliders for age (17–38),
  goals (0–40), assists (0–25) and minutes (0–3420), plus a €/£ display toggle.
* **Metric card** — predicted value, with the gap to the squad median for context.
* **Model Performance** — R²/MAE/RMSE, actual-vs-predicted scatter with the y = x
  line, and a residual plot.
* **Feature Impact** — standardised regression coefficients, largest effect first.

## CLI reference

```
python train.py --help

--mode {auto,api,mock}   data source (default: auto)
--mock                   shorthand for --mode mock
--n-mock N               players to generate in mock mode (default: 320)
--market-values PATH     CSV of real market values to join on player name
--on-missing {drop,synthesise}
                         what to do with players the CSV misses (default: drop)
--scorer-limit N         rows requested from the scorers endpoint (default: 100)
--include-non-scorers    keep squad members with no scoring record
--test-size F            held-out fraction (default: 0.2)
--random-state N         seed for splits and mock data (default: 42)
--cv-folds N / --no-cv   cross-validation control
--no-figures             skip writing diagnostic PNGs
-v, --verbose            debug logging
```

Exit codes: `0` success, `1` no usable rows, `2` missing or rejected API key.

## Limitations

Worth being blunt about, since the dashboard produces a confident-looking number:

* **Linear regression cannot capture the transfer market.** Real fees are driven by
  contract length, injury history, release clauses, agent fees, selling-club leverage
  and buyer desperation. None of that is in the feature set.
* **Predictions can extrapolate below zero.** OLS is unbounded, so an ageing player
  with no output can produce a negative fit. The dashboard floors the figure at zero
  and tells you when it had to.
* **Coefficients are conditional effects.** Correlated features (`goals` and
  `goals_per_90`) split the credit between them, so read the group rather than a
  single bar. A negative `goals_per_90` alongside a large positive `goals` is that
  effect, not a claim that scoring efficiently reduces value.
* **The free tier only covers scorers.** By default the dataset is the ~100 players
  on the scorers list, which skews attacking. `--include-non-scorers` adds the rest
  of the squads with zeroed output, which is truthful but adds a lot of zero-value
  rows.
* **Synthetic targets measure the pipeline, not the market.** A strong R² on mock
  data means the code works, nothing more.

## Licence

MIT
