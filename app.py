"""
Pitch Precision — FIFA World Cup 2026 Predictor Web App
Flask backend serving the Pitch Precision UI.
"""

import sys
import math
import json
from pathlib import Path
from datetime import datetime, timedelta

import joblib
import pandas as pd
import numpy as np
import requests
from flask import Flask, jsonify, request, render_template, abort

BASE_DIR = Path(__file__).parent
sys.path.insert(0, str(BASE_DIR))

from features import (
    load_results, load_rankings, compute_elo_history,
    get_team_form, get_goals_avg, get_draw_rate,
    get_fifa_rank_points, get_head_to_head, get_elo_rating,
    get_squad_value_log, WC2026_SQUAD_VALUES,
)
from outcome_model import predict_match_outcome
from goals_model import predict_match_goals

app = Flask(__name__)

DATA_DIR = BASE_DIR / "data"
OUT_DIR  = BASE_DIR / "outputs"
REFERENCE_DATE = pd.Timestamp("2026-06-11")
TODAY = datetime.now()

# ── Load data at startup ──────────────────────────────────────────────────────
print("Loading results and rankings data...")
results_df  = load_results(DATA_DIR / "results.csv")
rankings_df = load_rankings(DATA_DIR / "fifa_ranking.csv")
print("Computing ELO history (may take ~30s)...")
elo_history = compute_elo_history(results_df)

# ── Load trained models ───────────────────────────────────────────────────────
outcome_model    = joblib.load(OUT_DIR / "outcome_model.pkl")
home_goals_model = joblib.load(OUT_DIR / "goals_home_model.pkl")
away_goals_model = joblib.load(OUT_DIR / "goals_away_model.pkl")

tournament_df = pd.read_csv(OUT_DIR / "tournament_predictions.csv")
print("All models and data loaded.")

# ── WC 2026 groups ────────────────────────────────────────────────────────────
WC_2026_GROUPS = {
    "A": ["Mexico",        "South Africa",           "South Korea",  "Czech Republic"],
    "B": ["Canada",        "Bosnia and Herzegovina", "Switzerland",  "Qatar"],
    "C": ["United States", "Paraguay",               "Australia",    "Turkey"],
    "D": ["Brazil",        "Morocco",                "Scotland",     "Haiti"],
    "E": ["Germany",       "Ivory Coast",            "Ecuador",      "Curacao"],
    "F": ["Netherlands",   "Japan",                  "Sweden",       "Tunisia"],
    "G": ["Belgium",       "Egypt",                  "Iran",         "New Zealand"],
    "H": ["Spain",         "Saudi Arabia",           "Uruguay",      "Cape Verde"],
    "I": ["France",        "Senegal",                "Iraq",         "Norway"],
    "J": ["Argentina",     "Algeria",                "Austria",      "Jordan"],
    "K": ["Portugal",      "DR Congo",               "Uzbekistan",   "Colombia"],
    "L": ["England",       "Croatia",                "Ghana",        "Panama"],
}

ALL_TEAMS = [t for teams in WC_2026_GROUPS.values() for t in teams]

# Normalize ESPN team names → our internal names (ESPN uses different spellings)
ESPN_NAME_MAP = {
    "Czechia":              "Czech Republic",
    "Bosnia-Herzegovina":   "Bosnia and Herzegovina",
    "Bosnia & Herzegovina": "Bosnia and Herzegovina",
    "USA":                  "United States",
    "Korea Republic":       "South Korea",
    "IR Iran":              "Iran",
    "Côte d'Ivoire":        "Ivory Coast",
    "Cote d'Ivoire":        "Ivory Coast",
    "Ivory Coast":          "Ivory Coast",
    "Congo DR":             "DR Congo",
    "Congo, DR":            "DR Congo",
    "Cabo Verde":           "Cape Verde",
    "Türkiye":              "Turkey",
    "Curacao":              "Curacao",
    "Curaçao":              "Curacao",
    "New Zealand":          "New Zealand",
}

COUNTRY_CODES = {
    "Mexico": "mx",         "South Africa": "za",           "South Korea": "kr",
    "Czech Republic": "cz", "Czechia": "cz",                "Canada": "ca",
    "Bosnia and Herzegovina": "ba",
    "Switzerland": "ch",    "Qatar": "qa",                  "United States": "us",
    "Paraguay": "py",       "Australia": "au",              "Turkey": "tr",
    "Brazil": "br",         "Morocco": "ma",                "Scotland": "gb-sct",
    "Haiti": "ht",          "Germany": "de",                "Ivory Coast": "ci",
    "Ecuador": "ec",        "Curacao": "cw",                "Netherlands": "nl",
    "Japan": "jp",          "Sweden": "se",                 "Tunisia": "tn",
    "Belgium": "be",        "Egypt": "eg",                  "Iran": "ir",
    "New Zealand": "nz",    "Spain": "es",                  "Saudi Arabia": "sa",
    "Uruguay": "uy",        "Cape Verde": "cv",             "France": "fr",
    "Senegal": "sn",        "Iraq": "iq",                   "Norway": "no",
    "Argentina": "ar",      "Algeria": "dz",                "Austria": "at",
    "Jordan": "jo",         "Portugal": "pt",               "DR Congo": "cd",
    "Uzbekistan": "uz",     "Colombia": "co",               "England": "gb-eng",
    "Croatia": "hr",        "Ghana": "gh",                  "Panama": "pa",
}

# ── Pre-compute team stats once for fast API responses ────────────────────────
print("Pre-computing team feature stats...")
TEAM_STATS = {}
for group, teams in WC_2026_GROUPS.items():
    for team in teams:
        form             = get_team_form(results_df, team, REFERENCE_DATE)
        scored, conceded = get_goals_avg(results_df, team, REFERENCE_DATE, n=10)
        scored5, _       = get_goals_avg(results_df, team, REFERENCE_DATE, n=5)
        draw_rate        = get_draw_rate(results_df, team, REFERENCE_DATE)
        elo              = get_elo_rating(elo_history, team, REFERENCE_DATE)
        rank_pts         = get_fifa_rank_points(rankings_df, team, REFERENCE_DATE)
        sv_log           = get_squad_value_log(rankings_df, team, REFERENCE_DATE)
        TEAM_STATS[team] = {
            "group":           group,
            "form":            form,
            "goals_scored":    scored,
            "goals_conceded":  conceded,
            "goals_scored_5":  scored5,
            "draw_rate":       draw_rate,
            "elo":             elo,
            "rank_pts":        rank_pts,
            "squad_value_log": sv_log,
            "squad_value_eur": WC2026_SQUAD_VALUES.get(team, int(10 ** sv_log)),
            "country_code":    COUNTRY_CODES.get(team, "un"),
        }

# Normalize each stat across WC teams for radar chart (0–100)
_elos     = [v["elo"]             for v in TEAM_STATS.values()]
_pts      = [v["rank_pts"]        for v in TEAM_STATS.values()]
_scored   = [v["goals_scored"]    for v in TEAM_STATS.values()]
_conceded = [v["goals_conceded"]  for v in TEAM_STATS.values()]
_forms    = [v["form"]            for v in TEAM_STATS.values()]
_svs      = [v["squad_value_log"] for v in TEAM_STATS.values()]

def _safe(val, default=0.0):
    """Return val unless it is NaN/Inf/None, in which case return default."""
    try:
        v = float(val)
        return default if (math.isnan(v) or math.isinf(v)) else v
    except (TypeError, ValueError):
        return default


def _clean(obj):
    """Recursively replace NaN/Inf floats in dicts and lists with 0 so
    Flask's JSON serialiser never emits the invalid literal NaN."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, float):
        return _safe(obj, 0.0)
    return obj


def _norm(val, lo, hi):
    val = _safe(val, lo)
    return round((val - lo) / (hi - lo) * 100, 1) if hi > lo else 50.0

for team, stats in TEAM_STATS.items():
    stats["radar"] = {
        "attack":        _norm(stats["goals_scored"],    min(_scored),   max(_scored)),
        "form":          _norm(stats["form"],            min(_forms),    max(_forms)),
        "elo_strength":  _norm(stats["elo"],             min(_elos),     max(_elos)),
        "defence":       _norm(max(_conceded) - stats["goals_conceded"], 0, max(_conceded) - min(_conceded)),
        "squad_value":   _norm(stats["squad_value_log"], min(_svs),      max(_svs)),
    }

print("Team stats pre-computed.")


# ── Helper functions ──────────────────────────────────────────────────────────

def _fmt_squad_value(log_val):
    """Format a log10(EUR) squad value as a human-readable string."""
    v = 10 ** _safe(log_val, 7.0)
    if v >= 1e9:
        return f"€{v/1e9:.1f}B"
    return f"€{v/1e6:.0f}M"


def make_features(home, away, date=REFERENCE_DATE, neutral=1):
    hf          = get_team_form(results_df, home, date)
    af          = get_team_form(results_df, away, date)
    hgs, hgc    = get_goals_avg(results_df, home, date, n=10)
    ags, agc    = get_goals_avg(results_df, away, date, n=10)
    hgs5, _     = get_goals_avg(results_df, home, date, n=5)
    ags5, _     = get_goals_avg(results_df, away, date, n=5)
    hdr         = get_draw_rate(results_df, home, date)
    adr         = get_draw_rate(results_df, away, date)
    hp          = get_fifa_rank_points(rankings_df, home, date)
    ap          = get_fifa_rank_points(rankings_df, away, date)
    helo        = get_elo_rating(elo_history, home, date)
    aelo        = get_elo_rating(elo_history, away, date)
    hsv         = get_squad_value_log(rankings_df, home, date)
    asv         = get_squad_value_log(rankings_df, away, date)
    h2h         = get_head_to_head(results_df, home, away, date)
    return {
        "home_form": hf, "away_form": af,
        "home_goals_scored_avg": hgs, "home_goals_conceded_avg": hgc,
        "away_goals_scored_avg": ags, "away_goals_conceded_avg": agc,
        "home_goals_scored_avg_5": hgs5, "away_goals_scored_avg_5": ags5,
        "home_draw_rate": hdr, "away_draw_rate": adr,
        "home_rank_points": hp,  "away_rank_points": ap,
        "rank_diff": hp - ap,
        "home_elo": helo, "away_elo": aelo, "elo_diff": helo - aelo,
        "home_squad_value_log": hsv, "away_squad_value_log": asv,
        "squad_value_log_diff": hsv - asv,
        "h2h_home_winrate": h2h,
        "is_neutral": neutral,
    }


def _poisson_pmf(k, lam):
    if lam <= 0:
        return 0.0
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def most_likely_score(home_xg, away_xg, max_g=7):
    best, best_p = (0, 0), 0.0
    for h in range(max_g):
        for a in range(max_g):
            p = _poisson_pmf(h, home_xg) * _poisson_pmf(a, away_xg)
            if p > best_p:
                best_p, best = p, (h, a)
    return best


def get_h2h_stats(home, away, n=20):
    mask = (
        ((results_df.home_team == home) & (results_df.away_team == away)) |
        ((results_df.home_team == away) & (results_df.away_team == home))
    )
    h2h   = results_df[mask].tail(n)
    total = len(h2h)
    if total == 0:
        return {"total": 0, "home_wins": 0, "away_wins": 0, "draws": 0}
    hw = sum(
        1 for _, r in h2h.iterrows()
        if (r.home_team == home and r.home_score > r.away_score)
        or (r.away_team == home and r.away_score > r.home_score)
    )
    aw = sum(
        1 for _, r in h2h.iterrows()
        if (r.home_team == away and r.home_score > r.away_score)
        or (r.away_team == away and r.away_score > r.home_score)
    )
    return {"total": total, "home_wins": hw, "away_wins": aw, "draws": total - hw - aw}


def get_recent_form(team, n=5):
    mask   = (results_df.home_team == team) | (results_df.away_team == team)
    recent = results_df[mask].dropna(subset=["home_score", "away_score"]).tail(n)
    matches = []
    for _, r in recent.iterrows():
        if r.home_team == team:
            opp, gf, ga = r.away_team, int(r.home_score), int(r.away_score)
        else:
            opp, gf, ga = r.home_team, int(r.away_score), int(r.home_score)
        result = "W" if gf > ga else ("D" if gf == ga else "L")
        matches.append({
            "opponent":   opp,
            "result":     result,
            "score":      f"{gf}–{ga}",
            "date":       r.date.strftime("%b %d, %Y"),
            "tournament": str(r.get("tournament", "")),
            "opp_code":   COUNTRY_CODES.get(opp, "un"),
        })
    return matches


# ── Fixture generation ────────────────────────────────────────────────────────

def generate_group_fixtures():
    """
    Build a fixture list for the WC 2026 group stage.
    Matchday 1: Jun 11–16, Matchday 2: Jun 18–23, Matchday 3: Jun 25–28.
    """
    fixtures = []
    groups_list = list(WC_2026_GROUPS.items())  # preserve order A–L
    md_starts = {
        1: datetime(2026, 6, 11),
        2: datetime(2026, 6, 18),
        3: datetime(2026, 6, 25),
    }
    # Standard FIFA group stage pairings (1v2+3v4, 1v3+2v4, 1v4+2v3)
    pairings = [
        [(0, 1), (2, 3)],
        [(0, 2), (1, 3)],
        [(0, 3), (1, 2)],
    ]
    for g_idx, (group, teams) in enumerate(groups_list):
        day_offset = g_idx // 2   # spread 2 groups per calendar day
        for md_idx, pairs in enumerate(pairings):
            matchday  = md_idx + 1
            date      = md_starts[matchday] + timedelta(days=day_offset)
            date_str  = date.strftime("%Y-%m-%d")
            if date.date() < TODAY.date():
                status = "completed"
            elif date.date() == TODAY.date():
                status = "live"
            else:
                status = "upcoming"
            for h_i, a_i in pairs:
                home, away = teams[h_i], teams[a_i]
                fixtures.append({
                    "id":       f"{group}{matchday}{h_i}{a_i}",
                    "stage":    f"Group {group}",
                    "group":    group,
                    "matchday": matchday,
                    "home":     home,
                    "away":     away,
                    "date":     date_str,
                    "status":   status,
                    "home_code": COUNTRY_CODES.get(home, "un"),
                    "away_code": COUNTRY_CODES.get(away, "un"),
                })
    return sorted(fixtures, key=lambda x: (x["date"], x["group"]))


def fetch_espn_fixtures():
    """
    Attempt to get live fixture data from ESPN's public soccer API.
    Returns a list of fixture dicts, or None on failure.
    """
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/soccer/fifa.world/scoreboard"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        data   = resp.json()
        events = data.get("events", [])
        if not events:
            return None
        fixtures = []
        for ev in events:
            comp        = ev.get("competitions", [{}])[0]
            competitors = comp.get("competitors", [])
            if len(competitors) < 2:
                continue
            home_c = next((c for c in competitors if c.get("homeAway") == "home"), competitors[0])
            away_c = next((c for c in competitors if c.get("homeAway") == "away"), competitors[1])
            home   = ESPN_NAME_MAP.get(home_c.get("team", {}).get("displayName", ""), home_c.get("team", {}).get("displayName", ""))
            away   = ESPN_NAME_MAP.get(away_c.get("team", {}).get("displayName", ""), away_c.get("team", {}).get("displayName", ""))
            stype  = comp.get("status", {}).get("type", {}).get("name", "")
            if "FINAL" in stype:
                status = "completed"
            elif "IN_PROGRESS" in stype:
                status = "live"
            else:
                status = "upcoming"
            notes = comp.get("notes", [])
            stage = notes[0].get("headline", "FIFA World Cup") if notes else "FIFA World Cup"
            fixtures.append({
                "id":       ev.get("id", ""),
                "stage":    stage,
                "home":     home,
                "away":     away,
                "date":     ev.get("date", "")[:10],
                "status":   status,
                "home_code": COUNTRY_CODES.get(home, "un"),
                "away_code": COUNTRY_CODES.get(away, "un"),
            })
        return fixtures or None
    except Exception:
        return None


def fetch_espn_standings():
    """
    Fetch live group standings from ESPN's soccer standings API.
    Returns a dict keyed by group letter ("A"–"L"), each value a list of
    team dicts sorted by position, or None on failure.
    """
    # Note-colour → qualification status used for row highlighting
    NOTE_STATUS = {
        "#81D6AC": "qualified",   # Advance to Round of 32
        "#B5E7CE": "potential",   # Best 8 third-place
        "#FF7F84": "eliminated",  # Eliminated
    }
    try:
        url = "https://site.api.espn.com/apis/v2/sports/soccer/fifa.world/standings"
        resp = requests.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        data    = resp.json()
        groups  = data.get("children", [])
        if not groups:
            return None

        standings = {}
        for g in groups:
            raw_abbrev = g.get("abbreviation", g.get("name", "?"))
            letter = raw_abbrev.split()[-1]          # "Group A" → "A"
            entries = g.get("standings", {}).get("entries", [])
            teams = []
            for entry in entries:
                raw_name = entry.get("team", {}).get("displayName", "")
                name     = ESPN_NAME_MAP.get(raw_name, raw_name)
                stats    = {s["name"]: _safe(s.get("value", 0)) for s in entry.get("stats", [])}
                note_clr = entry.get("note", {}).get("color", "")
                status   = NOTE_STATUS.get(f"#{note_clr.lstrip('#')}", "unknown")
                teams.append({
                    "team":    name,
                    "country_code": COUNTRY_CODES.get(name, "un"),
                    "played":  int(stats.get("gamesPlayed", 0)),
                    "won":     int(stats.get("wins",        0)),
                    "drawn":   int(stats.get("ties",        0)),
                    "lost":    int(stats.get("losses",      0)),
                    "gf":      int(stats.get("pointsFor",   0)),
                    "ga":      int(stats.get("pointsAgainst", 0)),
                    "gd":      int(stats.get("pointDifferential", 0)),
                    "pts":     int(stats.get("points",      0)),
                    "status":  status,
                })
            if teams:
                standings[letter] = teams
        return standings or None
    except Exception as e:
        print(f"ESPN standings error: {e}")
        return None


def empty_standings():
    """Fallback: return groups with all-zero stats so the UI still renders."""
    result = {}
    for letter, teams in WC_2026_GROUPS.items():
        result[letter] = [
            {
                "team": t, "country_code": COUNTRY_CODES.get(t, "un"),
                "played": 0, "won": 0, "drawn": 0, "lost": 0,
                "gf": 0, "ga": 0, "gd": 0, "pts": 0, "status": "unknown",
            }
            for t in teams
        ]
    return result


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/match")
def match():
    return render_template("match.html", teams=sorted(ALL_TEAMS))


@app.route("/team")
def team():
    return render_template("team.html", teams=sorted(ALL_TEAMS), initial_team="")


@app.route("/team/<path:name>")
def team_detail(name):
    if name not in ALL_TEAMS:
        abort(404)
    return render_template("team.html", teams=sorted(ALL_TEAMS), initial_team=name)


# ── API endpoints ─────────────────────────────────────────────────────────────

@app.route("/api/fixtures")
def api_fixtures():
    espn = fetch_espn_fixtures()
    if espn:
        return jsonify({"source": "espn", "fixtures": espn})
    return jsonify({"source": "generated", "fixtures": generate_group_fixtures()})


@app.route("/api/teams")
def api_teams():
    return jsonify(sorted(ALL_TEAMS))


@app.route("/api/standings")
def api_standings():
    live = fetch_espn_standings()
    if live:
        return jsonify({"source": "espn", "standings": live})
    return jsonify({"source": "generated", "standings": empty_standings()})


@app.route("/api/tournament")
def api_tournament():
    records = tournament_df.to_dict("records")
    for r in records:
        team = r["team"]
        r["country_code"] = COUNTRY_CODES.get(team, "un")
        r["group"] = next(
            (g for g, ts in WC_2026_GROUPS.items() if team in ts), "?"
        )
    return jsonify(records)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    body = request.get_json(force=True)
    home    = body.get("home_team", "").strip()
    away    = body.get("away_team", "").strip()
    neutral = int(body.get("neutral", 1))
    date_s  = body.get("date", REFERENCE_DATE.strftime("%Y-%m-%d"))

    if not home or not away:
        return jsonify({"error": "home_team and away_team are required"}), 400
    if home == away:
        return jsonify({"error": "Teams must be different"}), 400

    date  = pd.Timestamp(date_s)
    feats = make_features(home, away, date, neutral)

    outcome = predict_match_outcome(outcome_model, feats)
    goals   = predict_match_goals(home_goals_model, away_goals_model, feats)

    h_xg = round(float(goals["home_xg"]), 2)
    a_xg = round(float(goals["away_xg"]), 2)
    score = most_likely_score(h_xg, a_xg)

    # Derive model factors
    form_diff = (feats["home_form"] - feats["away_form"]) * 100
    elo_diff  = feats["elo_diff"]
    h2h_rate  = feats["h2h_home_winrate"]
    rank_diff = feats["rank_diff"]
    sv_diff   = feats["squad_value_log_diff"]
    # log10 diff → % difference in actual value: 10^d - 1
    sv_pct    = round((_safe(10 ** sv_diff, 1.0) - 1.0) * 100, 1)

    factors = [
        {
            "name":        "Squad Form",
            "icon":        "trending_up",
            "impact":      round(form_diff, 1),
            "positive":    form_diff >= 0,
            "description": (
                f"{home}: {feats['home_form']*100:.0f}% form score  "
                f"vs  {away}: {feats['away_form']*100:.0f}%"
            ),
        },
        {
            "name":        "ELO Strength",
            "icon":        "hub",
            "impact":      round(elo_diff / 20, 1),
            "positive":    elo_diff >= 0,
            "description": (
                f"{home} ELO {feats['home_elo']:.0f}  "
                f"vs  {away} ELO {feats['away_elo']:.0f}"
            ),
        },
        {
            "name":        "Squad Value",
            "icon":        "diamond",
            "impact":      sv_pct,
            "positive":    sv_diff >= 0,
            "description": (
                f"{home}: {_fmt_squad_value(feats['home_squad_value_log'])}  "
                f"vs  {away}: {_fmt_squad_value(feats['away_squad_value_log'])}"
            ),
        },
        {
            "name":        "Head-to-Head",
            "icon":        "history_edu",
            "impact":      round((h2h_rate - 0.5) * 30, 1),
            "positive":    h2h_rate >= 0.5,
            "description": (
                f"{home} wins {h2h_rate*100:.0f}% of historical encounters"
            ),
        },
        {
            "name":        "Ranking Strength",
            "icon":        "leaderboard",
            "impact":      round(rank_diff / 50, 1),
            "positive":    rank_diff >= 0,
            "description": (
                f"FIFA pts: {home} {feats['home_rank_points']:.0f}  "
                f"vs  {away} {feats['away_rank_points']:.0f}"
            ),
        },
    ]

    # Stats for comparison table
    total_xg = h_xg + a_xg + 1e-6
    total_scored = feats["home_goals_scored_avg"] + feats["away_goals_scored_avg"] + 1e-6
    total_form   = feats["home_form"] + feats["away_form"] + 1e-6

    stats = [
        {
            "metric":    "Form Score",
            "home_val":  f"{feats['home_form']*100:.1f}%",
            "away_val":  f"{feats['away_form']*100:.1f}%",
            "home_pct":  feats["home_form"] / total_form,
            "away_pct":  feats["away_form"] / total_form,
        },
        {
            "metric":    "Goals Scored (avg / match)",
            "home_val":  f"{feats['home_goals_scored_avg']:.2f}",
            "away_val":  f"{feats['away_goals_scored_avg']:.2f}",
            "home_pct":  feats["home_goals_scored_avg"] / total_scored,
            "away_pct":  feats["away_goals_scored_avg"] / total_scored,
        },
        {
            "metric":    "Goals Conceded (avg / match)",
            "home_val":  f"{feats['home_goals_conceded_avg']:.2f}",
            "away_val":  f"{feats['away_goals_conceded_avg']:.2f}",
            "home_pct":  1 - min(feats["home_goals_conceded_avg"] / 4, 1),
            "away_pct":  1 - min(feats["away_goals_conceded_avg"] / 4, 1),
        },
        {
            "metric":    "Expected Goals (xG)",
            "home_val":  str(h_xg),
            "away_val":  str(a_xg),
            "home_pct":  h_xg / total_xg,
            "away_pct":  a_xg / total_xg,
        },
    ]
    # Normalize stats bar percentages so they sum to 1
    for s in stats[:2]:
        s["home_pct"] = round(s["home_pct"], 3)
        s["away_pct"] = round(s["away_pct"], 3)

    h2h_stats = get_h2h_stats(home, away)

    return jsonify(_clean({
        "home_team":     home,
        "away_team":     away,
        "home_code":     COUNTRY_CODES.get(home, "un"),
        "away_code":     COUNTRY_CODES.get(away, "un"),
        "home_win_prob": round(float(outcome["home_win"]) * 100, 1),
        "draw_prob":     round(float(outcome["draw"])     * 100, 1),
        "away_win_prob": round(float(outcome["away_win"]) * 100, 1),
        "home_xg":       h_xg,
        "away_xg":       a_xg,
        "predicted_score": f"{score[0]} – {score[1]}",
        "model_confidence": round(max(float(outcome["home_win"]), float(outcome["away_win"])) * 100),
        "h2h":           h2h_stats,
        "factors":       factors,
        "stats":         stats,
    }))


@app.route("/api/team/<path:name>")
def api_team(name):
    if name not in TEAM_STATS:
        return jsonify({"error": f'Team "{name}" not found'}), 404

    stats = TEAM_STATS[name]

    row  = tournament_df[tournament_df["team"] == name]
    traj = {}
    if not row.empty:
        r    = row.iloc[0]
        traj = {
            "win":        round(_safe(r.get("win_pct",        0)), 1),
            "final":      round(_safe(r.get("final_pct",      0)), 1),
            "semi":       round(_safe(r.get("semi_pct",       0)), 1),
            "qf":         round(_safe(r.get("qf_pct",         0)), 1),
            "r16":        round(_safe(r.get("r16_pct",        0)), 1),
            "r32":        round(_safe(r.get("r32_pct",        0)), 1),
            "group_exit": round(_safe(r.get("group_exit_pct", 0)), 1),
        }

    return jsonify(_clean({
        "team":              name,
        "country_code":      stats["country_code"],
        "group":             stats["group"],
        "form":              round(_safe(stats["form"]) * 100, 1),
        "goals_scored":      round(_safe(stats["goals_scored"]), 2),
        "goals_conceded":    round(_safe(stats["goals_conceded"]), 2),
        "elo":               round(_safe(stats["elo"], 1500)),
        "rank_points":       round(_safe(stats["rank_pts"], 1000)),
        "squad_value_eur":   stats.get("squad_value_eur", 0),
        "squad_value_fmt":   _fmt_squad_value(_safe(stats.get("squad_value_log", 8.0))),
        "radar":             stats["radar"],
        "trajectory":        traj,
        "recent_form":       get_recent_form(name, n=5),
    }))


if __name__ == "__main__":
    app.run(debug=True, port=5000, use_reloader=False)
