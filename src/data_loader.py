"""Data ingestion for the EPL transfer-value predictor.

Two interchangeable sources feed the same tidy DataFrame contract:

1. ``football-data.org`` v4 (live).  The free tier exposes squad lists
   (``/competitions/{code}/teams``) and scoring statistics
   (``/competitions/{code}/scorers``).  Both are pulled through a shared
   rate-limited client that respects the 10-requests-per-minute quota.
2. A deterministic mock generator, so the whole pipeline (training,
   evaluation, dashboard) can be exercised without an API key.

Either way the returned frame carries these columns::

    name, team, position, age, goals, assists, minutes, matches

``football-data.org`` tracks match events rather than transfer fees, so the
target variable is attached separately by :func:`attach_market_values`, which
supports joining a local CSV of real market values *or* deriving a realistic
synthetic valuation baseline.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
API_BASE_URL = "https://api.football-data.org/v4"
API_KEY_ENV_VAR = "FOOTBALL_DATA_API_KEY"
DEFAULT_COMPETITION = "PL"

# football-data.org free tier: 10 calls / minute.
FREE_TIER_MAX_CALLS = 10
FREE_TIER_WINDOW_SECONDS = 60.0

POSITIONS: tuple[str, ...] = ("GK", "DF", "MF", "FW")

#: Canonical column contract produced by every loader in this module.
PLAYER_COLUMNS: tuple[str, ...] = (
    "name",
    "team",
    "position",
    "age",
    "goals",
    "assists",
    "minutes",
    "matches",
)

# The API does not publish per-player minutes on the free tier, so minutes are
# estimated from appearances.  Kept explicit (and documented in the README) so
# the assumption is never mistaken for a measured value.
MINUTES_PER_APPEARANCE_ESTIMATE = 78.0
MAX_SEASON_MINUTES = 3420  # 38 matches x 90 minutes

# Slider domains used by the dashboard; loaders clip to the same ranges so the
# training distribution and the UI can never drift apart.
AGE_RANGE = (17, 38)
GOALS_RANGE = (0, 40)
ASSISTS_RANGE = (0, 25)
MINUTES_RANGE = (0, MAX_SEASON_MINUTES)

# Order matters: "Defensive Midfield" must resolve to MF, not DF.
_POSITION_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("GK", ("goalkeep", "keeper", "gk")),
    ("MF", ("midfield", "mf")),
    ("DF", ("defen", "back", "df")),
    ("FW", ("offence", "offense", "forward", "striker", "wing", "attack", "fw")),
)


class MissingAPIKeyError(RuntimeError):
    """Raised when a live pull is requested without ``FOOTBALL_DATA_API_KEY``."""


# --------------------------------------------------------------------------- #
# Rate limiting
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Sliding-window rate limiter.

    Blocks in :meth:`acquire` until another call fits inside the window, which
    keeps the free tier's 10/minute quota from ever returning HTTP 429.
    """

    def __init__(
        self,
        max_calls: int = FREE_TIER_MAX_CALLS,
        period: float = FREE_TIER_WINDOW_SECONDS,
    ) -> None:
        self.max_calls = max_calls
        self.period = period
        self._calls: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        """Reserve a slot, sleeping if the window is currently full."""
        while True:
            with self._lock:
                now = time.monotonic()
                # Drop timestamps that have aged out of the window.
                while self._calls and now - self._calls[0] >= self.period:
                    self._calls.popleft()

                if len(self._calls) < self.max_calls:
                    self._calls.append(now)
                    return

                sleep_for = self.period - (now - self._calls[0]) + 0.05

            LOGGER.info("Rate limit reached - waiting %.1fs before next call", sleep_for)
            time.sleep(sleep_for)


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class FootballDataClient:
    """Thin, polite wrapper around the football-data.org v4 REST API."""

    RETRYABLE_STATUS = frozenset({500, 502, 503, 504})

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = API_BASE_URL,
        timeout: float = 20.0,
        max_retries: int = 3,
        limiter: RateLimiter | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv(API_KEY_ENV_VAR)
        if not self.api_key:
            raise MissingAPIKeyError(
                f"No API key found. Set the {API_KEY_ENV_VAR} environment variable "
                "(free key: https://www.football-data.org/client/register) or run in "
                "mock mode with `python train.py --mode mock`."
            )

        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.limiter = limiter or RateLimiter()
        self.session = session or requests.Session()
        self.session.headers.update({"X-Auth-Token": self.api_key})

    # -- low level ---------------------------------------------------------- #
    def get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        """GET ``path`` and return the decoded JSON body.

        Waits for a free rate-limit slot before every attempt, honours a
        ``Retry-After`` header on HTTP 429, and retries transient 5xx errors
        with exponential backoff.
        """
        url = f"{self.base_url}/{path.lstrip('/')}"
        last_error: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            self.limiter.acquire()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:  # network hiccup
                last_error = exc
                backoff = 2.0**attempt
                LOGGER.warning("Request to %s failed (%s); retrying in %.0fs", url, exc, backoff)
                time.sleep(backoff)
                continue

            if response.status_code == 429:
                # Server-side throttle: trust Retry-After when present.
                retry_after = _coerce_float(response.headers.get("Retry-After"), default=60.0)
                LOGGER.warning("HTTP 429 from API; sleeping %.0fs before retry", retry_after)
                time.sleep(retry_after + 0.5)
                continue

            if response.status_code in (401, 403):
                raise MissingAPIKeyError(
                    f"API rejected the key with HTTP {response.status_code}. "
                    f"Check that {API_KEY_ENV_VAR} holds a valid football-data.org token."
                )

            if response.status_code in self.RETRYABLE_STATUS:
                backoff = 2.0**attempt
                LOGGER.warning(
                    "HTTP %s from API (attempt %d/%d); retrying in %.0fs",
                    response.status_code,
                    attempt,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)
                continue

            response.raise_for_status()
            return response.json()

        raise RuntimeError(f"Giving up on {url} after {self.max_retries} attempts") from last_error

    # -- endpoints ---------------------------------------------------------- #
    def competition_teams(self, competition: str = DEFAULT_COMPETITION) -> list[dict[str, Any]]:
        """Squad lists for every club in the competition (1 API call)."""
        return self.get(f"/competitions/{competition}/teams").get("teams", [])

    def competition_scorers(
        self, competition: str = DEFAULT_COMPETITION, limit: int = 100
    ) -> list[dict[str, Any]]:
        """Top scorers with goals / assists / appearances (1 API call)."""
        payload = self.get(f"/competitions/{competition}/scorers", params={"limit": limit})
        return payload.get("scorers", [])


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def normalise_position(raw: Any) -> str:
    """Map an API position label onto one of ``GK / DF / MF / FW``.

    Handles the v4 vocabulary ("Centre-Back", "Defensive Midfield",
    "Left Winger", "Offence", ...) as well as already-normalised codes.
    Unknown or missing labels fall back to ``MF``, the modal position.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return "MF"

    text = str(raw).strip().lower()
    if not text:
        return "MF"

    for code, keywords in _POSITION_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return code
    return "MF"


def age_from_date_of_birth(dob: Any, on: pd.Timestamp | None = None) -> float | None:
    """Convert an ISO ``YYYY-MM-DD`` birth date into an age in whole years."""
    if not dob:
        return None
    birth = pd.to_datetime(dob, errors="coerce")
    if pd.isna(birth):
        return None

    reference = on or pd.Timestamp.today().normalize()
    years = reference.year - birth.year
    # Subtract one if this year's birthday has not happened yet.
    if (reference.month, reference.day) < (birth.month, birth.day):
        years -= 1
    return float(years)


def _normalise_name(series: pd.Series) -> pd.Series:
    """Lower-cased, accent-stripped key used for CSV joins.

    Digits are deliberately kept: they are the only thing separating players
    who would otherwise share a key ("Danny Ward" appears twice in a typical
    Premier League season), and dropping them turns a one-to-one join into a
    row-multiplying cross product.
    """
    return (
        series.astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.lower()
        .str.replace(r"[^a-z0-9 ]", "", regex=True)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )


def _empty_players_frame() -> pd.DataFrame:
    return pd.DataFrame({column: pd.Series(dtype="object") for column in PLAYER_COLUMNS})


# --------------------------------------------------------------------------- #
# Live API loaders
# --------------------------------------------------------------------------- #
def squads_to_frame(teams: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Flatten ``/competitions/{code}/teams`` into one row per squad member."""
    rows: list[dict[str, Any]] = []
    for team in teams:
        team_name = team.get("shortName") or team.get("name") or "Unknown"
        for player in team.get("squad") or []:
            rows.append(
                {
                    "player_id": player.get("id"),
                    "name": player.get("name"),
                    "team": team_name,
                    "position": normalise_position(player.get("position")),
                    "age": age_from_date_of_birth(player.get("dateOfBirth")),
                    "nationality": player.get("nationality"),
                }
            )
    return pd.DataFrame(rows)


def scorers_to_frame(scorers: Iterable[dict[str, Any]]) -> pd.DataFrame:
    """Flatten ``/competitions/{code}/scorers`` into per-player statistics."""
    rows: list[dict[str, Any]] = []
    for entry in scorers:
        player = entry.get("player") or {}
        team = entry.get("team") or {}
        matches = entry.get("playedMatches") or 0
        rows.append(
            {
                "player_id": player.get("id"),
                "name": player.get("name"),
                "team": team.get("shortName") or team.get("name"),
                "position": normalise_position(player.get("position") or player.get("section")),
                "age": age_from_date_of_birth(player.get("dateOfBirth")),
                "goals": entry.get("goals") or 0,
                "assists": entry.get("assists") or 0,
                "matches": matches,
                # Free tier has no minutes field - estimate from appearances.
                "minutes": round(matches * MINUTES_PER_APPEARANCE_ESTIMATE),
            }
        )
    return pd.DataFrame(rows)


def fetch_epl_players(
    api_key: str | None = None,
    competition: str = DEFAULT_COMPETITION,
    scorer_limit: int = 100,
    include_non_scorers: bool = False,
    client: FootballDataClient | None = None,
) -> pd.DataFrame:
    """Pull live Premier League players and their season statistics.

    Costs 2 API calls (squads + scorers), comfortably inside the free tier.

    Args:
        api_key: Overrides ``FOOTBALL_DATA_API_KEY`` when given.
        competition: Competition code, ``PL`` for the Premier League.
        scorer_limit: How many rows to request from the scorers endpoint.
        include_non_scorers: Also keep squad members with no scoring record.
            Their goals/assists/minutes are zeroed, which is truthful but
            heavily inflates the number of zero-value rows, so it is off by
            default.
        client: Pre-built client, mainly for testing.

    Returns:
        DataFrame following the :data:`PLAYER_COLUMNS` contract.
    """
    client = client or FootballDataClient(api_key=api_key)

    LOGGER.info("Fetching %s squads...", competition)
    squads = squads_to_frame(client.competition_teams(competition))

    LOGGER.info("Fetching %s scorers...", competition)
    scorers = scorers_to_frame(client.competition_scorers(competition, limit=scorer_limit))

    if squads.empty and scorers.empty:
        return _empty_players_frame()

    # "right" keeps only players with a scoring record; "outer" adds the rest of
    # the squad *and* keeps scorers missing from the squad lists (loanees, players
    # who moved mid-season), which a left join would silently drop.
    how: Literal["outer", "right"] = "outer" if include_non_scorers else "right"
    merged = squads.merge(
        scorers,
        on="player_id",
        how=how,
        suffixes=("", "_scorer"),
    )

    # Squad payload is the better source for name/position/age; fall back to
    # whatever the scorers endpoint returned.
    for column in ("name", "team", "position", "age"):
        fallback = f"{column}_scorer"
        if fallback in merged.columns:
            merged[column] = merged[column].combine_first(merged[fallback])

    for column, default in (("goals", 0), ("assists", 0), ("matches", 0), ("minutes", 0)):
        if column not in merged.columns:
            merged[column] = default
        # Squad members with no scoring record genuinely have zero output.
        merged[column] = (
            pd.to_numeric(merged[column], errors="coerce").fillna(default).astype(int)
        )

    merged["position"] = merged["position"].map(normalise_position)
    merged = merged.dropna(subset=["name"])

    LOGGER.info("Fetched %d players from the API", len(merged))
    return merged[list(PLAYER_COLUMNS)].reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Mock / offline generator
# --------------------------------------------------------------------------- #
_MOCK_TEAMS = (
    "Arsenal", "Aston Villa", "Bournemouth", "Brentford", "Brighton",
    "Chelsea", "Crystal Palace", "Everton", "Fulham", "Ipswich",
    "Leicester", "Liverpool", "Man City", "Man United", "Newcastle",
    "Nottingham Forest", "Southampton", "Tottenham", "West Ham", "Wolves",
)
_MOCK_FIRST_NAMES = (
    "Alex", "Bruno", "Callum", "Diego", "Emile", "Finn", "Gabriel", "Harvey",
    "Ivan", "Jonas", "Kai", "Luka", "Mateo", "Noah", "Omar", "Pedro",
    "Quentin", "Rafael", "Samir", "Tomas", "Umar", "Viktor", "Wes", "Yuri",
)
_MOCK_LAST_NAMES = (
    "Adeyemi", "Bennett", "Costa", "Duarte", "Ellis", "Fabregas", "Gunnarsson",
    "Hoffmann", "Iversen", "Jankovic", "Kovacic", "Lindqvist", "Moretti",
    "Nowak", "Okafor", "Petrov", "Quinn", "Romano", "Silva", "Thorne",
    "Ustinov", "Vermeulen", "Wallace", "Zielinski",
)

#: Expected goals-per-90 by position, used to shape the mock distribution.
_MOCK_GOAL_RATE = {"GK": 0.002, "DF": 0.035, "MF": 0.110, "FW": 0.330}
_MOCK_ASSIST_RATE = {"GK": 0.004, "DF": 0.045, "MF": 0.130, "FW": 0.180}


def generate_mock_players(n_players: int = 320, seed: int | None = 42) -> pd.DataFrame:
    """Generate a realistic offline stand-in for a season of EPL player data.

    Statistics are drawn from position-aware distributions (forwards score,
    goalkeepers do not; minutes follow a skewed starter/squad-player split) so
    the downstream model sees the same kinds of correlations it would meet in
    the live feed.
    """
    rng = np.random.default_rng(seed)

    positions = rng.choice(POSITIONS, size=n_players, p=[0.10, 0.33, 0.34, 0.23])
    ages = np.clip(np.round(rng.normal(26.0, 4.2, n_players)), *AGE_RANGE)

    # Beta(2.2, 1.8) gives a realistic mix of rotation players and ever-presents.
    minutes = np.clip(np.round(rng.beta(2.2, 1.8, n_players) * MAX_SEASON_MINUTES), *MINUTES_RANGE)
    nineties = minutes / 90.0

    # Per-player quality multiplier: a few standouts, most around average.
    quality = rng.lognormal(mean=0.0, sigma=0.45, size=n_players)

    goal_rates = np.array([_MOCK_GOAL_RATE[p] for p in positions])
    assist_rates = np.array([_MOCK_ASSIST_RATE[p] for p in positions])

    goals = np.clip(rng.poisson(goal_rates * nineties * quality), *GOALS_RANGE)
    assists = np.clip(rng.poisson(assist_rates * nineties * quality), *ASSISTS_RANGE)
    matches = np.round(minutes / MINUTES_PER_APPEARANCE_ESTIMATE).astype(int)

    first = rng.choice(_MOCK_FIRST_NAMES, size=n_players)
    last = rng.choice(_MOCK_LAST_NAMES, size=n_players)
    names = [f"{f} {l}" for f, l in zip(first, last)]
    # Disambiguate the inevitable duplicates from a small name pool.
    names = pd.Series(names)
    duplicated = names.duplicated(keep=False)
    names[duplicated] = names[duplicated] + " " + (names.groupby(names).cumcount() + 1).astype(str)

    frame = pd.DataFrame(
        {
            "name": names.to_numpy(),
            "team": rng.choice(_MOCK_TEAMS, size=n_players),
            "position": positions,
            "age": ages.astype(int),
            "goals": goals.astype(int),
            "assists": assists.astype(int),
            "minutes": minutes.astype(int),
            "matches": matches,
        }
    )
    LOGGER.info("Generated %d mock players (seed=%s)", len(frame), seed)
    return frame[list(PLAYER_COLUMNS)]


# --------------------------------------------------------------------------- #
# Target variable: market values
# --------------------------------------------------------------------------- #
#: Baseline value in EUR millions for an average-aged, average-minutes player.
BASE_VALUE_EUR_M = 11.5
PEAK_AGE = 25.5
AGE_DECAY = 46.0  # width of the age-value curve
POSITION_VALUE_MULTIPLIER = {"FW": 1.45, "MF": 1.22, "DF": 0.96, "GK": 0.72}
MIN_VALUE_EUR_M = 0.4


def synthesise_market_values(
    frame: pd.DataFrame,
    seed: int | None = 7,
    noise_sigma: float = 0.17,
) -> pd.Series:
    """Derive a realistic synthetic transfer valuation, in EUR millions.

    The baseline is deliberately *multiplicative* - value compounds across a
    bell-shaped age curve, a position premium, playing time and output - which
    mirrors how the transfer market actually behaves and keeps the task
    non-trivial for a linear model.

    This is a documented stand-in, not a market quote: use
    ``--market-values path/to/values.csv`` to train on real fees.
    """
    rng = np.random.default_rng(seed)

    age = frame["age"].to_numpy(dtype=float)
    goals = frame["goals"].to_numpy(dtype=float)
    assists = frame["assists"].to_numpy(dtype=float)
    minutes = frame["minutes"].to_numpy(dtype=float)
    positions = frame["position"].astype(str).to_numpy()

    # Bell curve peaking in the mid-twenties, tapering towards both ends.
    age_factor = np.exp(-((age - PEAK_AGE) ** 2) / AGE_DECAY)
    # Regular starters are worth materially more than fringe players.
    minutes_share = np.clip(minutes / MAX_SEASON_MINUTES, 0.0, 1.0)
    availability_factor = 0.35 + 1.25 * minutes_share
    # Output premium: goals carry more weight than assists.
    output_factor = 1.0 + 0.155 * goals + 0.105 * assists
    position_factor = np.array([POSITION_VALUE_MULTIPLIER.get(p, 1.0) for p in positions])
    noise = rng.lognormal(mean=0.0, sigma=noise_sigma, size=len(frame))

    values = (
        BASE_VALUE_EUR_M
        * age_factor
        * availability_factor
        * output_factor
        * position_factor
        * noise
    )
    return pd.Series(np.round(np.maximum(values, MIN_VALUE_EUR_M), 2), index=frame.index)


def load_market_values_csv(csv_path: str | Path, value_column: str | None = None) -> pd.DataFrame:
    """Read a local market-value CSV into ``name`` / ``market_value_eur_m``.

    The file needs a player-name column (``name``, ``player`` or ``player_name``)
    and a value column.  Values are interpreted as EUR millions unless the
    column name mentions plain euros, in which case they are scaled down.
    """
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Market-value CSV not found: {path}")

    table = pd.read_csv(path)
    lower = {c.lower().strip(): c for c in table.columns}

    name_col = next((lower[c] for c in ("name", "player", "player_name", "full_name") if c in lower), None)
    if name_col is None:
        raise ValueError(f"{path} needs a player name column (name / player / player_name)")

    if value_column is None:
        value_column = next(
            (
                lower[c]
                for c in (
                    "market_value_eur_m",
                    "market_value_m",
                    "market_value",
                    "value_eur",
                    "value",
                    "transfer_value",
                )
                if c in lower
            ),
            None,
        )
    if value_column is None:
        raise ValueError(f"{path} needs a market value column (e.g. market_value_eur_m)")

    values = pd.to_numeric(
        table[value_column].astype(str).str.replace(r"[^0-9eE.\-+]", "", regex=True),
        errors="coerce",
    )
    # Raw euro amounts (e.g. 45000000) get converted to millions.
    if values.dropna().median() > 10_000:
        values = values / 1_000_000.0

    out = pd.DataFrame({"name": table[name_col].astype(str), "market_value_eur_m": values})
    return out.dropna(subset=["market_value_eur_m"]).drop_duplicates(subset=["name"])


def attach_market_values(
    frame: pd.DataFrame,
    market_value_csv: str | Path | None = None,
    on_missing: Literal["drop", "synthesise"] = "drop",
    seed: int | None = 7,
) -> pd.DataFrame:
    """Attach the ``market_value_eur_m`` target column.

    With ``market_value_csv`` the values are joined on a normalised player
    name; without it a synthetic baseline is generated.  ``on_missing``
    controls what happens to players the CSV does not cover - dropping them
    (default) keeps the training set purely real, while ``synthesise`` fills
    the gaps with the baseline and flags the rows in ``value_source``.
    """
    result = frame.copy()

    if market_value_csv is None:
        result["market_value_eur_m"] = synthesise_market_values(result, seed=seed)
        result["value_source"] = "synthetic"
        return result

    values = load_market_values_csv(market_value_csv)
    values["_join_key"] = _normalise_name(values["name"])
    result["_join_key"] = _normalise_name(result["name"])

    # Two CSV rows sharing a normalised key would multiply the player rows, so
    # collapse them first and let pandas assert the join stays many-to-one.
    duplicate_keys = int(values["_join_key"].duplicated().sum())
    if duplicate_keys:
        LOGGER.warning(
            "%d duplicate player names in %s - keeping the first of each",
            duplicate_keys,
            market_value_csv,
        )
        values = values.drop_duplicates(subset=["_join_key"], keep="first")

    result = result.merge(
        values[["_join_key", "market_value_eur_m"]],
        on="_join_key",
        how="left",
        validate="m:1",
    )
    matched = result["market_value_eur_m"].notna()
    LOGGER.info(
        "Matched %d/%d players against %s", int(matched.sum()), len(result), market_value_csv
    )
    result["value_source"] = np.where(matched, "csv", "missing")

    if on_missing == "synthesise":
        gaps = ~matched
        if gaps.any():
            filled = synthesise_market_values(result.loc[gaps], seed=seed)
            result.loc[gaps, "market_value_eur_m"] = filled
            result.loc[gaps, "value_source"] = "synthetic"
    else:
        result = result[matched]

    return result.drop(columns=["_join_key"]).reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Top-level entry point
# --------------------------------------------------------------------------- #
def load_dataset(
    mode: Literal["auto", "api", "mock"] = "auto",
    market_value_csv: str | Path | None = None,
    n_mock: int = 320,
    seed: int | None = 42,
    api_key: str | None = None,
    scorer_limit: int = 100,
    include_non_scorers: bool = False,
    on_missing: Literal["drop", "synthesise"] = "drop",
) -> pd.DataFrame:
    """Load features + target in one call.

    ``mode="auto"`` uses the live API when ``FOOTBALL_DATA_API_KEY`` is set and
    transparently falls back to the mock generator otherwise, so a fresh clone
    can train and launch the dashboard with zero configuration.
    """
    resolved = mode
    if mode == "auto":
        resolved = "api" if (api_key or os.getenv(API_KEY_ENV_VAR)) else "mock"
        LOGGER.info("mode=auto resolved to '%s'", resolved)

    if resolved == "api":
        try:
            players = fetch_epl_players(
                api_key=api_key,
                scorer_limit=scorer_limit,
                include_non_scorers=include_non_scorers,
            )
        except (MissingAPIKeyError, requests.RequestException, RuntimeError) as exc:
            if mode == "api":  # explicit request - do not silently degrade
                raise
            LOGGER.warning("API load failed (%s); falling back to mock data", exc)
            players = generate_mock_players(n_players=n_mock, seed=seed)
    else:
        players = generate_mock_players(n_players=n_mock, seed=seed)

    players.attrs["source"] = resolved
    dataset = attach_market_values(
        players, market_value_csv=market_value_csv, on_missing=on_missing, seed=seed
    )
    dataset.attrs["source"] = resolved
    return dataset
