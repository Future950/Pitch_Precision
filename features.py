# ============================================================
# fifa_predictor/utils/features.py
# Data loading + feature engineering
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

DATA_DIR = Path(__file__).parent / "data"

# Maps team names used in results.csv to those used in the FIFA ranking CSV.
TEAM_NAME_MAP = {
    "United States":  "USA",
    "Iran":           "IR Iran",
    "South Korea":    "Korea Republic",
    "Cape Verde":     "Cabo Verde",
    "Curacao":        "Curacao",
    "Czech Republic": "Czechia",
    "DR Congo":       "Congo DR",
    "Ivory Coast":    "Cote d'Ivoire",
}


# ─────────────────────────────────────────────
# 1. LOAD RAW DATA
# ─────────────────────────────────────────────

def load_results(path=None):
    path = path or DATA_DIR / "results.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_rankings(path=None):
    path = path or DATA_DIR / "fifa_rankings.csv"
    df = pd.read_csv(path, parse_dates=["rank_date"])
    return df


# ─────────────────────────────────────────────
# 2. SQUAD MARKET VALUES  (2026 WC teams, EUR)
# ─────────────────────────────────────────────

# Source: Transfermarkt via planetfootball.com (June 2026)
WC2026_SQUAD_VALUES = {
    "France":                    1_520_000_000,
    "England":                   1_360_000_000,
    "Spain":                     1_220_000_000,
    "Portugal":                  1_010_000_000,
    "Germany":                     947_000_000,
    "Brazil":                      928_200_000,
    "Argentina":                   807_500_000,
    "Netherlands":                 754_200_000,
    "Norway":                      589_900_000,
    "Belgium":                     547_500_000,
    "Ivory Coast":                 522_100_000,
    "Senegal":                     478_100_000,
    "Turkey":                      473_700_000,
    "Morocco":                     447_700_000,
    "Sweden":                      406_080_000,
    "Croatia":                     387_300_000,
    "United States":               385_600_000,
    "Ecuador":                     368_700_000,
    "Uruguay":                     359_300_000,
    "Switzerland":                 332_500_000,
    "Colombia":                    302_350_000,
    "Japan":                       270_850_000,
    "Algeria":                     256_900_000,
    "Austria":                     245_200_000,
    "Ghana":                       234_500_000,
    "Canada":                      198_650_000,
    "Mexico":                      191_850_000,
    "Czech Republic":              188_180_000,
    "Scotland":                    170_250_000,
    "Paraguay":                    153_650_000,
    "Bosnia and Herzegovina":      146_400_000,
    "DR Congo":                    143_900_000,
    "South Korea":                 139_050_000,
    "Egypt":                       116_480_000,
    "Uzbekistan":                   85_330_000,
    "Australia":                    77_450_000,
    "Tunisia":                      69_950_000,
    "Haiti":                        55_900_000,
    "Cape Verde":                   49_250_000,
    "South Africa":                 49_250_000,
    "Saudi Arabia":                 40_680_000,
    "Panama":                       34_550_000,
    "New Zealand":                  34_450_000,
    "Iran":                         32_050_000,
    "Curacao":                      25_780_000,
    "Iraq":                         21_200_000,
    "Jordan":                       20_300_000,
    "Qatar":                        19_930_000,
}

# Reference ranking points as of April 2026 (matches the squad value snapshot date)
_REF_RANK_PTS = {
    "France": 1877.32, "England": 1825.97, "Spain": 1876.40, "Portugal": 1763.83,
    "Germany": 1730.37, "Brazil": 1761.16, "Argentina": 1874.81, "Netherlands": 1757.87,
    "Norway": 1557.44, "Belgium": 1734.71, "Ivory Coast": 1568.62, "Senegal": 1688.99,
    "Turkey": 1579.47, "Morocco": 1755.87, "Sweden": 1533.19, "Croatia": 1717.07,
    "United States": 1673.13, "Ecuador": 1570.77, "Uruguay": 1673.07, "Switzerland": 1649.40,
    "Colombia": 1693.09, "Japan": 1660.43, "Algeria": 1571.03, "Austria": 1597.40,
    "Ghana": 1381.25, "Canada": 1551.50, "Mexico": 1681.03, "Czech Republic": 1484.82,
    "Scotland": 1518.77, "Paraguay": 1488.05, "Bosnia and Herzegovina": 1332.30,
    "DR Congo": 1460.00, "South Korea": 1612.55, "Egypt": 1570.67, "Uzbekistan": 1458.73,
    "Australia": 1605.60, "Tunisia": 1490.00, "Haiti": 1274.46, "Cape Verde": 1380.53,
    "South Africa": 1416.66, "Saudi Arabia": 1431.30, "Panama": 1539.16,
    "New Zealand": 1197.68, "Iran": 1605.12, "Curacao": 1272.71, "Iraq": 1433.07,
    "Jordan": 1374.13, "Qatar": 1459.45,
}


def get_squad_value_log(rankings_df, team, before_date) -> float:
    """
    Estimate log10(squad_value_eur) for a team at a historical date.

    For WC2026 teams: scale the known 2026 value by the team's ranking ratio at
    the requested date vs. its 2026 reference ranking.
    For other teams: estimate from ranking points using a fitted power-law.

    Returns log10(value) — roughly 7.3 (weakest) to 9.2 (France).
    """
    ref_value = WC2026_SQUAD_VALUES.get(team)
    if ref_value is not None:
        ref_pts = _REF_RANK_PTS.get(team, 1500.0)
        hist_pts = get_fifa_rank_points(rankings_df, team, before_date)
        ratio = (hist_pts / ref_pts) if ref_pts > 0 else 1.0
        est = max(ref_value * ratio, 1_000_000)
    else:
        # Power-law approximation for teams outside the WC2026 lookup
        rank_pts = get_fifa_rank_points(rankings_df, team, before_date)
        est = max(50_000_000 * ((rank_pts / 1500.0) ** 3), 1_000_000)
    return float(np.log10(est))


# ─────────────────────────────────────────────
# 3. ELO RATINGS
# ─────────────────────────────────────────────

def _k_factor(tournament) -> float:
    t = str(tournament)
    if "FIFA World Cup" in t and "Qualif" not in t:
        return 60.0
    if any(x in t for x in ["UEFA Euro", "Copa America", "Africa Cup", "Gold Cup", "Asian Cup"]):
        return 50.0
    if "Qualif" in t or "qualifier" in t.lower():
        return 40.0
    if "Friendly" in t:
        return 20.0
    return 32.0


def compute_elo_history(results_df: pd.DataFrame, initial_elo: float = 1500.0) -> dict:
    """
    One-pass ELO computation over all matches (sorted by date).
    Returns dict: team -> {'dates': [...], 'elos': [...]}
    Each entry stores the ELO BEFORE that match (for feature lookup).
    """
    elo = defaultdict(lambda: initial_elo)
    history: dict = defaultdict(lambda: {"dates": [], "elos": []})

    for _, match in results_df.sort_values("date").iterrows():
        home = match["home_team"]
        away = match["away_team"]
        date = match["date"]

        history[home]["dates"].append(date)
        history[home]["elos"].append(float(elo[home]))
        history[away]["dates"].append(date)
        history[away]["elos"].append(float(elo[away]))

        exp_home = 1.0 / (1.0 + 10.0 ** ((elo[away] - elo[home]) / 400.0))
        hs, as_ = match["home_score"], match["away_score"]
        s = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        k = _k_factor(match.get("tournament", ""))

        elo[home] += k * (s - exp_home)
        elo[away] += k * ((1.0 - s) - (1.0 - exp_home))

    return dict(history)


def get_elo_rating(elo_history: dict, team: str, before_date) -> float:
    """Binary-search lookup: ELO for team just before before_date."""
    if team not in elo_history:
        return 1500.0
    dates = elo_history[team]["dates"]
    elos  = elo_history[team]["elos"]
    lo, hi = 0, len(dates)
    while lo < hi:
        mid = (lo + hi) // 2
        if dates[mid] < before_date:
            lo = mid + 1
        else:
            hi = mid
    return elos[lo - 1] if lo > 0 else 1500.0


# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────

def get_team_form(results_df, team, before_date, n=10, decay=0.85):
    """Exponentially recency-weighted form score over last n matches (Win=1, Draw=0.5, Loss=0)."""
    mask = (
        ((results_df["home_team"] == team) | (results_df["away_team"] == team))
        & (results_df["date"] < before_date)
    )
    recent = results_df[mask].tail(n).copy()
    if recent.empty:
        return 0.5

    def score(row):
        if row["home_team"] == team:
            if row["home_score"] > row["away_score"]: return 1.0
            if row["home_score"] == row["away_score"]: return 0.5
            return 0.0
        else:
            if row["away_score"] > row["home_score"]: return 1.0
            if row["away_score"] == row["home_score"]: return 0.5
            return 0.0

    recent["result"] = recent.apply(score, axis=1)
    n_matches = len(recent)
    # most recent match -> weight decay^0 = 1.0, oldest -> decay^(n_matches-1)
    weights = decay ** np.arange(n_matches - 1, -1, -1)
    return float(np.average(recent["result"], weights=weights))


def get_goals_avg(results_df, team, before_date, n=10, cap=4, decay=0.85):
    """
    Recency-weighted average goals scored/conceded in last n matches.

    Per-match goal counts are winsorized at `cap` so a single blowout result
    (e.g. a 7-1 win) doesn't dominate the rolling average, and matches are
    weighted with exponential decay so recent form counts more than older form.
    """
    mask = (
        ((results_df["home_team"] == team) | (results_df["away_team"] == team))
        & (results_df["date"] < before_date)
    )
    recent = results_df[mask].tail(n).copy()
    if recent.empty:
        return 1.2, 1.2

    scored, conceded = [], []
    for _, row in recent.iterrows():
        if row["home_team"] == team:
            scored.append(row["home_score"])
            conceded.append(row["away_score"])
        else:
            scored.append(row["away_score"])
            conceded.append(row["home_score"])

    scored = np.minimum(np.asarray(scored, dtype=float), cap)
    conceded = np.minimum(np.asarray(conceded, dtype=float), cap)

    n_matches = len(recent)
    weights = decay ** np.arange(n_matches - 1, -1, -1)

    return float(np.average(scored, weights=weights)), float(np.average(conceded, weights=weights))


def get_draw_rate(results_df, team, before_date, n=20):
    """Fraction of last n matches that ended in a draw."""
    mask = (
        ((results_df["home_team"] == team) | (results_df["away_team"] == team))
        & (results_df["date"] < before_date)
    )
    recent = results_df[mask].tail(n)
    if recent.empty:
        return 0.25
    draws = int((recent["home_score"] == recent["away_score"]).sum())
    return float(draws / len(recent))


def get_fifa_rank_points(rankings_df, team, before_date):
    """Most recent FIFA ranking points for a team before a given date."""
    team = TEAM_NAME_MAP.get(team, team)
    mask = (rankings_df["country_full"] == team) & (rankings_df["rank_date"] < before_date)
    subset = rankings_df[mask]
    if subset.empty:
        return 1000.0
    return float(subset.sort_values("rank_date").iloc[-1]["total_points"])


def get_head_to_head(results_df, home_team, away_team, before_date, n=10):
    """Win rate for home_team in last n head-to-head meetings."""
    mask = (
        (
            ((results_df["home_team"] == home_team) & (results_df["away_team"] == away_team))
            | ((results_df["home_team"] == away_team) & (results_df["away_team"] == home_team))
        )
        & (results_df["date"] < before_date)
    )
    h2h = results_df[mask].tail(n)
    if h2h.empty:
        return 0.5
    wins = sum(
        1 for _, row in h2h.iterrows()
        if (row["home_team"] == home_team and row["home_score"] > row["away_score"])
        or (row["away_team"] == home_team and row["away_score"] > row["home_score"])
    )
    return float(wins / len(h2h))


def get_h2h_avg_goals(results_df, home_team, away_team, before_date, n=10):
    """Average total goals per match in last n H2H meetings."""
    mask = (
        (
            ((results_df["home_team"] == home_team) & (results_df["away_team"] == away_team))
            | ((results_df["home_team"] == away_team) & (results_df["away_team"] == home_team))
        )
        & (results_df["date"] < before_date)
    )
    h2h = results_df[mask].tail(n)
    if h2h.empty:
        return 2.5  # average total goals in competitive internationals
    return float((h2h["home_score"] + h2h["away_score"]).mean())


# ─────────────────────────────────────────────
# 4. BUILD FULL FEATURE MATRIX
# ─────────────────────────────────────────────

def build_feature_matrix(results_df, rankings_df=None, elo_history=None):
    """
    Build a row-per-match feature matrix.

    Parameters
    ----------
    results_df  : competitive matches to featurise (pre-filtered)
    rankings_df : FIFA ranking points table
    elo_history : pre-computed dict from compute_elo_history() — pass the
                  dict built from ALL results for best ELO calibration
    """
    if elo_history is None:
        print("  Computing ELO from results subset (for best results pass full results).")
        elo_history = compute_elo_history(results_df)

    rows = []
    n_total = len(results_df)

    for count, (_, match) in enumerate(results_df.iterrows()):
        date = match["date"]
        home = match["home_team"]
        away = match["away_team"]

        home_form = get_team_form(results_df, home, date)
        away_form = get_team_form(results_df, away, date)

        hgs10, hgc10 = get_goals_avg(results_df, home, date, n=10)
        ags10, agc10 = get_goals_avg(results_df, away, date, n=10)
        hgs5,  _     = get_goals_avg(results_df, home, date, n=5)
        ags5,  _     = get_goals_avg(results_df, away, date, n=5)

        home_draw = get_draw_rate(results_df, home, date)
        away_draw = get_draw_rate(results_df, away, date)

        home_pts = get_fifa_rank_points(rankings_df, home, date) if rankings_df is not None else 1000.0
        away_pts = get_fifa_rank_points(rankings_df, away, date) if rankings_df is not None else 1000.0

        home_elo = get_elo_rating(elo_history, home, date)
        away_elo = get_elo_rating(elo_history, away, date)

        home_sv = get_squad_value_log(rankings_df, home, date) if rankings_df is not None else 8.0
        away_sv = get_squad_value_log(rankings_df, away, date) if rankings_df is not None else 8.0

        h2h        = get_head_to_head(results_df, home, away, date)
        h2h_goals  = get_h2h_avg_goals(results_df, home, away, date)

        hs  = match["home_score"]
        as_ = match["away_score"]
        outcome = 0 if hs > as_ else (1 if hs == as_ else 2)

        rows.append({
            "date":                    date,
            "home_team":               home,
            "away_team":               away,
            "home_form":               home_form,
            "away_form":               away_form,
            "home_goals_scored_avg":   hgs10,
            "home_goals_conceded_avg": hgc10,
            "away_goals_scored_avg":   ags10,
            "away_goals_conceded_avg": agc10,
            "home_goals_scored_avg_5": hgs5,
            "away_goals_scored_avg_5": ags5,
            "home_draw_rate":          home_draw,
            "away_draw_rate":          away_draw,
            "home_rank_points":        home_pts,
            "away_rank_points":        away_pts,
            "rank_diff":               home_pts - away_pts,
            "home_elo":                home_elo,
            "away_elo":                away_elo,
            "elo_diff":                home_elo - away_elo,
            "home_squad_value_log":    home_sv,
            "away_squad_value_log":    away_sv,
            "squad_value_log_diff":    home_sv - away_sv,
            "home_attack_vs_away_def": hgs10 / max(agc10, 0.5),
            "away_attack_vs_home_def": ags10 / max(hgc10, 0.5),
            "h2h_home_winrate":        h2h,
            "h2h_avg_goals":           h2h_goals,
            "is_neutral":              int(match.get("neutral", False)),
            "outcome":                 outcome,
            "home_goals":              hs,
            "away_goals":              as_,
        })

        if count % 1000 == 0 and count > 0:
            print(f"  Processed {count:,} / {n_total:,} matches...")

    return pd.DataFrame(rows)


FEATURE_COLS = [
    "home_form",
    "away_form",
    "home_goals_scored_avg",
    "home_goals_conceded_avg",
    "away_goals_scored_avg",
    "away_goals_conceded_avg",
    "home_goals_scored_avg_5",
    "away_goals_scored_avg_5",
    "home_draw_rate",
    "away_draw_rate",
    "home_rank_points",
    "away_rank_points",
    "rank_diff",
    "home_elo",
    "away_elo",
    "elo_diff",
    "home_squad_value_log",
    "away_squad_value_log",
    "squad_value_log_diff",
    "home_attack_vs_away_def",
    "away_attack_vs_home_def",
    "h2h_home_winrate",
    "h2h_avg_goals",
    "is_neutral",
]
