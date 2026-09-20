# EPL Player Scouting & Transfer Valuation Dashboard

Pick a real Premier League player, see the statistics the model was given, and
compare the market's price against what a linear regression thinks they are
worth. Players the model prices *above* the market are scouting targets;
players it prices *below* are where the market is paying for something the
stats cannot see.

Runs offline with no API key, but **the live API is the accurate path** — see
[The dataset](#the-dataset) for why.

---

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python train.py            # trains on 279 real players across all 20 clubs
streamlit run app.py       # open the dashboard
```

`train.py` uses the live API when `FOOTBALL_DATA_API_KEY` is set and falls back
to the bundled snapshot (with a warning) when it is not.

`train.py` writes `models/model.pkl`, `models/metadata.json`, the league-wide
`data/processed/scout_table.csv` and diagnostic PNGs in `reports/`. The
dashboard reads those artefacts, so it never retrains.

## What the dashboard does

**Find a player** — filter by club (or search all 20), then pick a player. The
banner shows their club, position, age and real recorded output: minutes,
goals, assists, and G+A per 90.

**Valuation comparison** — three cards side by side:

| | |
| --- | --- |
| **Actual market value** | what the dataset records |
| **Model predicted value** | what the regression infers from age, minutes, output and position |
| **Valuation delta** | `actual − predicted`, labelled *Overvalued* (red) or *Bargain / Undervalued* (green) |

A **positive** delta means the market prices the player above what their output
justifies. A **negative** delta means the model sees more value than the asking
price.

**Simulate performance change** — an expander under the player card, with
sliders pre-seeded to that player's real numbers. Push Saka's goals from 6 to
11 and the prediction moves from €94.6M to €166.8M in real time.

**League Scout Table** — the full dataset with three views (Top 10 bargains,
Top 10 overvalued, full league), filterable by position and by a maximum
transfer budget.

**Model Performance** — held-out R², MAE and RMSE, an actual-vs-predicted
scatter with the y = x line, and a residual plot.

**Feature Impact** — standardised regression coefficients as percentage effects.

---

## The dataset

The bundled snapshot covers **279 players across all 20 clubs** of the 2026/27
Premier League season, with the columns the pipeline expects:

```
player_name, team, position, age, minutes_played, goals, assists, actual_market_value_eur
```

It lives at [`data/raw/epl_players_2026_27.csv`](data/raw/epl_players_2026_27.csv),
generated from the table in [`src/reference_data.py`](src/reference_data.py) and
regenerated automatically if you delete it.

> ### What is verified and what is not
>
> **Verified by research (September 2026):**
>
> * The 20 clubs in the 2026/27 season. Coventry City, Hull City and Ipswich
>   Town came up; Burnley, West Ham United and Wolverhampton Wanderers went down.
> * Squad membership after the summer 2026 window — Enzo Fernández at
>   Manchester City, Bernardo Silva and Rodri gone, Bruno Guimarães at Arsenal,
>   Morgan Rogers at Chelsea, Carlos Baleba at Manchester United.
>
> **Approximate, and not to be relied on:**
>
> * Ages (correct to within about a year).
> * Appearance statistics, which are prior-season (2025/26) figures.
> * Market values, which are rough September 2026 numbers.
>
> A market value is also **not a transfer fee**. Enzo Fernández is carried here
> at €130M; he moved for £125m, and the two figures measure different things —
> a fee reflects contract length, buyer competition and timing.
>
> **Any hand-maintained squad list goes stale at the next transfer window.**
> That is not a caveat to work around, it is the reason `--mode api` exists.
> The snapshot is the offline fallback, not the source of truth.

### Two stat vintages, on purpose

Squads are current; statistics are from the **last completed season (2025/26)**,
because the current campaign is only a handful of matchweeks old and nobody has
a meaningful sample yet. That split is standard for scouting data, and the
dashboard labels both vintages.

Players at the promoted clubs posted those numbers in the **Championship**, so
the dataset carries a `stats_competition` column and the model uses it as a
feature. Without it every promoted-club player reads as a screaming bargain:
before the fix, the entire top three of the bargains table was Coventry and
Hull players. Adding it lifted cross-validated R² from 0.48 to 0.67.

### Using your own valuations

```bash
python train.py --market-values path/to/real_values.csv
```

The CSV needs a name column (`player_name` / `name` / `player`) and a value
column (`actual_market_value_eur` / `market_value_eur_m` / `value`). Raw euro
amounts such as `45000000` are detected and scaled to millions automatically,
and names are matched case- and accent-insensitively. See
[`data/raw/market_values.example.csv`](data/raw/market_values.example.csv).

Players the CSV does not cover are dropped by default; pass
`--on-missing synthesise` to fill the gaps instead (those rows are flagged in
`value_source`).

### Using live data (recommended)

`/competitions/PL/teams` returns **today's** squads, so club, position and age
stay correct through every transfer window without anyone hand-editing a CSV.
Get a free key at [football-data.org](https://www.football-data.org/client/register):

```bash
export FOOTBALL_DATA_API_KEY=your_key_here
python train.py --mode api
```

`--mode auto` (the default) picks this automatically whenever the key is set.
Valuations still come from `--market-values` or the bundled CSV, because no free
API publishes transfer fees.

**Rate limiting.** The free tier allows 10 requests per minute. `RateLimiter`
in [`src/data_loader.py`](src/data_loader.py) enforces this with a sliding
window, blocking *before* a call rather than reacting to an HTTP 429. If the
server throttles anyway the client honours `Retry-After`, and it retries
transient 5xx errors with exponential backoff. A full pull costs 2 requests.

**Two caveats on live data.** The free tier does not expose per-player minutes,
so `minutes_played` is estimated as `playedMatches × 78`
(`MINUTES_PER_APPEARANCE_ESTIMATE`). And the scorers endpoint only returns
players who have scored, so a live pull covers roughly 100 attacking players
rather than the full league; `--include-non-scorers` adds the rest with zeroed
output.

---

## The model

A single `sklearn.pipeline.Pipeline`: `StandardScaler` on numerics +
`OneHotEncoder` on position → `LinearRegression`, wrapped in a
`TransformedTargetRegressor` that fits `log1p(value)`.

**Why the log target.** Market values are strongly right-skewed (€1.5M to
€180M here, skew 1.9). Plain OLS chases the handful of superstars and
extrapolates to *negative* fees for fringe players. Regressing on the log makes
the model multiplicative — each feature scales value by a factor rather than
adding a fixed number of millions, which is how the transfer market actually
behaves. Predictions stay positive automatically, and cross-validated R² rose
from 0.33 to 0.42.

**Features:**

| Feature | Why |
| --- | --- |
| `age` | Value tracks a career arc |
| `minutes_played` | Proxy for trust and availability, and it restores output *volume* |
| `goals_per_90`, `assists_per_90` | Rate stats: separates an efficient substitute from a goalless ever-present |
| `years_from_peak_sq` | `(age − 25.5)²`, letting a *linear* model bend the age curve |
| `position` | One-hot encoded, midfielder as the reference category |
| `stats_competition` | Premier League vs Championship, so promoted clubs' output is discounted rather than taken at face value |

Raw `goals` and `assists` are deliberately **excluded**: they correlate ~0.9
with their own per-90 rates, and feeding both makes the two split the credit so
neither coefficient can be read on its own. Volume is still represented,
because rate × minutes reconstructs it. Cross-validated R² is slightly *better*
without them (0.44 vs 0.42) and the coefficients become readable, which is the
whole point of the Feature Impact tab. The what-if simulator still takes goals
and assists as inputs — they flow through the per-90 rates.

## Metrics

A representative run on the bundled dataset (279 players, 80/20 split):

```
Train  R2 =  0.703 | MAE = EUR  8.26M | RMSE = EUR 11.52M
Test   R2 =  0.705 | MAE = EUR 13.88M | RMSE = EUR 19.54M
5-fold CV R2 = 0.670 (+/- 0.128)
```

An R² around 0.7 is about what these statistics can honestly explain about
transfer values. On synthetic data (`--mock`) the same pipeline
scores R² 0.89 — a useful reminder that a strong score on generated data
measures the code, not the market.

---

## Project layout

```
epl-transfer-predictor/
├── data/
│   ├── raw/
│   │   ├── epl_players_2026_27.csv   # the bundled dataset
│   │   └── market_values.example.csv # schema for your own valuations
│   └── processed/                    # cleaned dataset, predictions, scout table
├── models/                           # model.pkl + metadata.json (git-ignored)
├── reports/                          # diagnostic PNGs from training
├── src/
│   ├── reference_data.py             # the curated squad table
│   ├── data_loader.py                # API client, rate limiting, mock data, target
│   ├── preprocessor.py               # cleaning, feature engineering, ColumnTransformer
│   ├── model.py                      # pipeline, metrics, scout table, persistence
│   └── visualize.py                  # matplotlib/seaborn diagnostics
├── app.py                            # Streamlit dashboard
├── train.py                          # trains and saves the pipeline
├── requirements.txt
└── README.md
```

## CLI reference

```
python train.py --help

--mode {auto,reference,api,mock}   data source (default: auto -> reference)
--mock                             shorthand for --mode mock
--reference-path PATH              where the bundled CSV lives
--rebuild-reference                overwrite it from src/reference_data.py
--n-mock N                         players to generate in mock mode
--market-values PATH               your own valuations, joined on player name
--on-missing {drop,synthesise}     what to do with players the CSV misses
--scorer-limit N                   rows requested from the scorers endpoint
--include-non-scorers              keep squad members with no scoring record
--test-size F / --random-state N   split control
--cv-folds N / --no-cv             cross-validation control
--no-figures                       skip writing diagnostic PNGs
-v, --verbose                      debug logging
```

Exit codes: `0` success, `1` no usable rows, `2` missing or rejected API key.

## Limitations

Worth being blunt about, since the dashboard produces confident-looking numbers:

* **Everything in the scout table is an in-sample fit.** Every player was in
  the training data. Treat the ranking as a shortlist of questions worth
  asking, not as valuations. The held-out metrics are the honest measure.
* **The model cannot see what it was not given.** Rodri appears as the most
  "overvalued" player in the league purely because a season-ending injury left
  him with 450 minutes. Van Dijk looks overvalued because a centre-back's
  contribution barely registers in goals and assists. Both are the model being
  wrong in an instructive way, not the market being wrong.
* **Coefficients are conditional effects.** The position bars are the clearest
  example: holding goals-per-90 constant, the model reports a discount for
  forwards, because "a forward scoring at a goalkeeper's rate" is a
  counterfactual that never occurs in the data. Read position and output
  together, not bar by bar.
* **Real fees are driven by things absent from the feature set** — contract
  length, injury history, release clauses, agent fees, selling-club leverage
  and buyer desperation.
* **The bundled figures are approximate and hand-maintained.** Squads go stale
  at every transfer window; `--mode api` is the only path that tracks current
  rosters without someone editing a CSV. See the dataset note above.
* **Statistics and squads are from different dates** — 2025/26 output, 2026/27
  clubs. A player who moved in the summer carries his old club's numbers.

## Licence

MIT
