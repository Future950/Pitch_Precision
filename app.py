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
    get_fifa_rank_points, get_head_to_head, get_h2h_avg_goals,
    get_elo_rating, get_squad_value_log, WC2026_SQUAD_VALUES,
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

_elo_cache = OUT_DIR / "elo_history.pkl"
if _elo_cache.exists():
    print("Loading cached ELO history...")
    elo_history = joblib.load(_elo_cache)
else:
    print("Computing ELO history (first run, ~30s — will be cached)...")
    elo_history = compute_elo_history(results_df)
    joblib.dump(elo_history, _elo_cache)
    print("ELO history cached to disk.")

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
    h2h_goals   = get_h2h_avg_goals(results_df, home, away, date)
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
        "home_attack_vs_away_def": hgs / max(agc, 0.5),
        "away_attack_vs_home_def": ags / max(hgc, 0.5),
        "h2h_home_winrate": h2h,
        "h2h_avg_goals":    h2h_goals,
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
            st     = comp.get("status", {}).get("type", {})
            stype  = st.get("name", "")
            period = comp.get("status", {}).get("period", 0) or 0
            # completed: full-time, final, or completed flag
            if st.get("completed") or "FINAL" in stype or "FULL_TIME" in stype:
                status = "completed"
            # live: any in-play state (1st half, 2nd half, HT, ET, pens...)
            elif period > 0 or any(x in stype for x in (
                "IN_PROGRESS", "HALFTIME", "FIRST_HALF", "SECOND_HALF",
                "EXTRA_TIME", "OVERTIME", "PENALTY", "PAUSE"
            )):
                status = "live"
            else:
                status = "upcoming"

            home_score = home_c.get("score", "") if status in ("live", "completed") else ""
            away_score = away_c.get("score", "") if status in ("live", "completed") else ""
            clock      = comp.get("status", {}).get("displayClock", "") if status == "live" else ""

            notes = comp.get("notes", [])
            stage = notes[0].get("headline", "FIFA World Cup") if notes else "FIFA World Cup"
            fixtures.append({
                "id":         ev.get("id", ""),
                "stage":      stage,
                "home":       home,
                "away":       away,
                "home_score": str(home_score),
                "away_score": str(away_score),
                "clock":      clock,
                "date":       ev.get("date", "")[:10],
                "status":     status,
                "home_code":  COUNTRY_CODES.get(home, "un"),
                "away_code":  COUNTRY_CODES.get(away, "un"),
            })
        return fixtures or None
    except Exception:
        return None


# citizenship string → ISO flag code (ESPN uses full country names)
_CITIZENSHIP_CODES = {
    **COUNTRY_CODES,
    "United States": "us", "USA": "us",
    "South Korea": "kr", "Korea Republic": "kr",
    "Ivory Coast": "ci", "Cote d'Ivoire": "ci",
    "DR Congo": "cd", "Congo DR": "cd",
    "Bosnia and Herzegovina": "ba",
    "Cape Verde": "cv", "Cabo Verde": "cv",
    "Czech Republic": "cz", "Czechia": "cz",
    "Türkiye": "tr", "Turkey": "tr",
    "Curacao": "cw", "Curaçao": "cw",
    "Norway": "no", "Sweden": "se", "Austria": "at",
    "Colombia": "co", "Algeria": "dz", "Jordan": "jo",
    "Uzbekistan": "uz", "Ghana": "gh", "Panama": "pa",
    "Scotland": "gb-sct", "England": "gb-eng",
    "Australia": "au", "New Zealand": "nz",
}

_STAT_LABELS = {
    "goalsLeaders":   ("Goals",            "sports_score",   "text-primary"),
    "assistsLeaders": ("Assists",          "gesture",        "text-tertiary"),
    "shotsOnTarget":  ("Shots on Target",  "ads_click",      "text-tertiary"),
    "totalShots":     ("Total Shots",      "target",         "text-secondary"),
    "yellowCards":    ("Yellow Cards",     "square",         "text-yellow-500"),
    "redCards":       ("Red Cards",        "square",         "text-error"),
    "saves":          ("Saves (GK)",       "shield",         "text-primary"),
    "accuratePasses": ("Accurate Passes",  "multiple_stop",  "text-secondary"),
}

_player_cache: dict = {"data": None, "ts": 0.0}
_img_cache:    dict = {}   # player_name → best available image URL

def _resolve_player_image(name: str, espn_url: str) -> str:
    """
    Return the best available headshot URL for a player.
    Priority: ESPN CDN → TheSportsDB → ui-avatars initials.
    """
    if espn_url:
        return espn_url
    if name in _img_cache:
        return _img_cache[name]
    try:
        import urllib.parse as _up
        search = (f"https://www.thesportsdb.com/api/v1/json/3/searchplayers.php"
                  f"?p={_up.quote(name)}")
        r = requests.get(search, timeout=5, headers={"User-Agent": "Mozilla/5.0"})
        players = r.json().get("player") or []
        img = ""
        if players:
            img = players[0].get("strThumb") or players[0].get("strCutout") or ""
        if not img:
            img = (f"https://ui-avatars.com/api/?name={_up.quote(name)}"
                   f"&background=006d37&color=ffffff&size=128&bold=true&rounded=true")
        _img_cache[name] = img
        return img
    except Exception:
        return (f"https://ui-avatars.com/api/?name={name[:20].replace(' ', '+')}"
                f"&background=006d37&color=ffffff&size=128&bold=true&rounded=true")


def fetch_player_stats():
    """
    Fetch live WC 2026 player stats from ESPN's leaders API.
    Cached for 5 minutes. Returns dict: category_label -> list of player dicts.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    now = time.time()
    if _player_cache["data"] and now - _player_cache["ts"] < 300:
        return _player_cache["data"]

    try:
        url = ("https://sports.core.api.espn.com/v2/sports/soccer/leagues"
               "/fifa.world/seasons/2026/types/1/leaders")
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return None
        cats = resp.json().get("categories", [])

        def _fetch_athlete(ref):
            try:
                r = requests.get(ref, timeout=5)
                a = r.json()
                name    = a.get("displayName", "?")
                pos     = a.get("position", {}).get("abbreviation", "")
                country = a.get("citizenship", "")
                espn_hs = a.get("headshot", {}).get("href", "")
                headshot = _resolve_player_image(name, espn_hs)
                return {
                    "name":     name,
                    "headshot": headshot,
                    "country":  country,
                    "code":     _CITIZENSHIP_CODES.get(country, "un"),
                    "position": pos,
                }
            except Exception:
                return {"name": "?", "headshot": "", "country": "", "code": "un", "position": ""}

        result = {}
        for cat in cats:
            cat_key = cat.get("name", "")
            if cat_key not in _STAT_LABELS:
                continue
            label, icon, color = _STAT_LABELS[cat_key]
            leaders = cat.get("leaders", [])[:10]
            refs    = [l.get("athlete", {}).get("$ref", "") for l in leaders]
            values  = [l.get("value", 0) for l in leaders]
            with ThreadPoolExecutor(max_workers=6) as ex:
                athletes = list(ex.map(_fetch_athlete, refs))
            max_val = max((v for v in values), default=1) or 1
            entries = [
                {
                    "rank":    i + 1,
                    "player":  ath["name"],
                    "headshot":ath["headshot"],
                    "country": ath["country"],
                    "code":    ath["code"],
                    "position":ath["position"],
                    "value":   int(v) if float(v) == int(float(v)) else round(float(v), 1),
                    "pct":     round(_safe(v) / max_val * 100, 1),
                }
                for i, (v, ath) in enumerate(zip(values, athletes))
            ]
            result[label] = {"icon": icon, "color": color, "entries": entries}

        if result:
            _player_cache["data"] = result
            _player_cache["ts"]   = now
        return result or None
    except Exception as e:
        print(f"Player stats error: {e}")
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
            raw_entries = g.get("standings", {}).get("entries", [])
            # ESPN entries arrive in arbitrary order — sort by their own rank stat
            def _entry_rank(e):
                s = {x["name"]: x.get("value", 99) for x in e.get("stats", [])}
                return int(_safe(s.get("rank", 99), 99))
            entries = sorted(raw_entries, key=_entry_rank)

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


@app.route("/players")
def players():
    return render_template("players.html")


@app.route("/bracket")
def bracket():
    return render_template("bracket.html")


# ── Bracket cache ─────────────────────────────────────────────────────────────
_bracket_cache: dict = {"data": None, "ts": 0.0}

ROUND_ORDER = ["Round of 32", "Round of 16", "Quarter-Finals",
               "Semi-Finals", "Third Place", "Final"]

# Official 2026 WC R32 display order — source: FIFA official bracket (USA Today image)
# Pairs that feed the SAME R16 slot are always ADJACENT (pos N and N+1 where N is even).
# Left bracket: pos 0-7 (Groups A-F area), Right bracket: pos 8-15 (Groups G-L area)
R32_BRACKET_POS: dict = {
    # Left bracket ──────────────────────────────────────────────────────────────
    frozenset(["Germany",       "Paraguay"]):               0,  # ESPN#4  → R16 with FRA/SWE
    frozenset(["France",        "Sweden"]):                 1,  # ESPN#6  → R16 with GER/PAR
    frozenset(["South Africa",  "Canada"]):                 2,  # ESPN#1  → R16 with NED/MAR
    frozenset(["Netherlands",   "Morocco"]):                3,  # ESPN#3  → R16 with RSA/CAN ★
    frozenset(["United States", "Bosnia and Herzegovina"]): 4,  # ESPN#10 → R16 with BEL/SEN
    frozenset(["Belgium",       "Senegal"]):                5,  # ESPN#9  → R16 with USA/BIH
    frozenset(["Portugal",      "Croatia"]):                6,  # ESPN#12 → R16 with ESP/AUT
    frozenset(["Spain",         "Austria"]):                7,  # ESPN#11 → R16 with POR/CRO
    # Right bracket ─────────────────────────────────────────────────────────────
    frozenset(["Brazil",        "Japan"]):                  8,  # ESPN#2  → R16 with CIV/NOR
    frozenset(["Ivory Coast",   "Norway"]):                 9,  # ESPN#5  → R16 with BRA/JPN
    frozenset(["Mexico",        "Ecuador"]):                10, # ESPN#7  → R16 with ENG/COD
    frozenset(["England",       "DR Congo"]):               11, # ESPN#8  → R16 with MEX/ECU
    frozenset(["Switzerland",   "Algeria"]):                12, # ESPN#13 → R16 with ARG/CPV
    frozenset(["Argentina",     "Cape Verde"]):             13, # ESPN#15 → R16 with SUI/ALG
    frozenset(["Australia",     "Egypt"]):                  14, # ESPN#14 → R16 with COL/GHA
    frozenset(["Colombia",      "Ghana"]):                  15, # ESPN#16 → R16 with AUS/EGY
}

# ESPN internal R32 match number → bracket display position (0-15)
# All R16/QF/SF sorting flows through bracket positions via _team_bpos()
ESPN_R32_NUM_TO_BPOS: dict = {
    4: 0, 6: 1, 1: 2, 3: 3, 10: 4, 9: 5, 12: 6, 11: 7,
    2: 8, 5: 9, 7: 10, 8: 11, 13: 12, 15: 13, 14: 14, 16: 15,
}

# Legacy dicts kept for compatibility (not used by primary sort path)
R16_ESPN_PAIR_POS: dict = {}
QF_ESPN_PAIR_POS:  dict = {}


def _infer_round(home: str, away: str, date: str) -> str:
    """Infer knockout round from ESPN placeholder team names."""
    combined = home + away
    if "Semifinal" in combined:
        return "Third Place" if "Loser" in combined else "Final"
    if "Quarterfinal" in combined:
        return "Semi-Finals"
    if "Round of 16" in combined:
        return "Quarter-Finals"
    if "Round of 32" in combined:
        return "Round of 16"
    return "Round of 32"


def _clean_team(name: str) -> str:
    """Normalize ESPN team names and shorten TBD placeholders."""
    name = ESPN_NAME_MAP.get(name, name)
    for pattern, short in [
        ("Third Place Group ", "3rd · Grp "),
        ("Group ",            "Grp "),
        (" 2nd Place",        " (2nd)"),
        (" Winner",           " ✓"),
        (" Loser",            " (3rd)"),
        ("Round of 32 ",      "R32 #"),
        ("Round of 16 ",      "R16 #"),
        ("Quarterfinal ",     "QF"),
        ("Semifinal ",        "SF"),
    ]:
        name = name.replace(pattern, short)
    return name


def fetch_bracket():
    """
    Fetch full knockout bracket from ESPN (R32 → Final).
    Returns dict with 'rounds' list, each containing 'matches'.
    Cached 5 minutes.
    """
    import time as _time
    now = _time.time()
    if _bracket_cache["data"] and now - _bracket_cache["ts"] < 90:
        return _bracket_cache["data"]

    try:
        from datetime import datetime as _dt, timedelta as _td

        # ── Build substitution map from live standings ──────────────
        # Maps ESPN placeholder strings → real qualified team names.
        subst = {}
        try:
            stnd = fetch_espn_standings()
            if stnd:
                for letter, teams in stnd.items():
                    g = f"Group {letter}"
                    if teams:
                        subst[f"{g} Winner"]    = teams[0]["team"]
                    if len(teams) > 1:
                        subst[f"{g} 2nd Place"] = teams[1]["team"]
                    if len(teams) > 2:
                        subst[f"{g} 3rd Place"] = teams[2]["team"]
        except Exception as se:
            print(f"Standings sub error: {se}")

        def _resolve(raw: str):
            """Return (display_name, country_code, is_tbd)."""
            # Real team name already (no placeholder keywords)
            if not any(k in raw for k in ["Winner", "Place", "Loser", "Round of", "Quarterfinal", "Semifinal"]):
                name = ESPN_NAME_MAP.get(raw, raw)
                return name, COUNTRY_CODES.get(name, "un"), False
            # Known from standings substitution
            if raw in subst:
                name = subst[raw]
                return name, COUNTRY_CODES.get(name, "un"), False
            # Still unknown — clean and mark TBD
            return _clean_team(ESPN_NAME_MAP.get(raw, raw)), "un", True

        all_matches = []

        for i in range(25):           # Jun 28 – Jul 22 (R32 starts Jun 28)
            d = (_dt(2026, 6, 28) + _td(days=i)).strftime("%Y%m%d")
            try:
                r = requests.get(
                    "https://site.api.espn.com/apis/site/v2/sports/soccer"
                    f"/fifa.world/scoreboard?dates={d}",
                    timeout=5, headers={"User-Agent": "Mozilla/5.0"})
                for ev in r.json().get("events", []):
                    comp  = ev["competitions"][0]
                    comps = comp["competitors"]
                    h_c   = comps[0] if comps else {}
                    a_c   = comps[1] if len(comps) > 1 else {}
                    raw_h = h_c.get("team", {}).get("displayName", "TBD")
                    raw_a = a_c.get("team", {}).get("displayName", "TBD")
                    hs    = h_c.get("score", "")
                    as_   = a_c.get("score", "")
                    st    = comp["status"]["type"]["name"]
                    date  = ev.get("date", "")[:10]

                    home, home_code, h_tbd = _resolve(raw_h)
                    away, away_code, a_tbd = _resolve(raw_a)

                    winner = ""
                    if st == "STATUS_FINAL":
                        try:
                            winner = home if int(hs) > int(as_) else (
                                away if int(as_) > int(hs) else "")
                        except (ValueError, TypeError):
                            pass
                    if not winner and h_c.get("winner", False):
                        winner = home
                    elif not winner and a_c.get("winner", False):
                        winner = away

                    all_matches.append({
                        "id":         ev.get("id", ""),
                        "date":       date,
                        "round":      _infer_round(raw_h, raw_a, date),
                        "home":       home,
                        "home_code":  home_code,
                        "home_score": str(hs),
                        "away":       away,
                        "away_code":  away_code,
                        "away_score": str(as_),
                        "status":     st,
                        "winner":     winner,
                        "is_tbd":     h_tbd or a_tbd,
                    })
            except Exception:
                pass

        by_round = {r: [] for r in ROUND_ORDER}
        for m in all_matches:
            if m["round"] in by_round:
                by_round[m["round"]].append(m)

        # ── Sort rounds by official bracket position ─────────────────
        import re as _re

        def _r32_sort_key(m):
            key = frozenset([m["home"], m["away"]])
            return R32_BRACKET_POS.get(key, 99)

        def _team_bpos(name: str):
            """Return bracket position (0-15) for a team name or R32 placeholder."""
            # Real team name
            for k, p in R32_BRACKET_POS.items():
                if name in k:
                    return p
            # Placeholder "R32 #X ✓"
            nums = _re.findall(r'R32 #(\d+)', name)
            if nums:
                return ESPN_R32_NUM_TO_BPOS.get(int(nums[0]), 99)
            return 99

        def _r16_sort_key(m):
            p1 = _team_bpos(m["home"])
            p2 = _team_bpos(m["away"])
            if p1 < 99 or p2 < 99:
                return min(p1, p2) // 2   # pairs (0,1)→0  (2,3)→1  etc.
            return 99

        def _r16_bpos(name: str):
            """Bracket pos of the R32 team behind an R16 team name or R16 placeholder."""
            # Real team name already resolved from R16
            for k, p in R32_BRACKET_POS.items():
                if name in k:
                    return p
            # "R32 #X" placeholder inside an R16 entry
            nums = _re.findall(r'R32 #(\d+)', name)
            if nums:
                return ESPN_R32_NUM_TO_BPOS.get(int(nums[0]), 99)
            # R16 placeholders "R16 #X" — approximate by R16 index
            nums16 = _re.findall(r'R16 #(\d+)', name)
            if nums16:
                return (int(nums16[0]) - 1) * 2   # R16 #1 → 0, #2 → 2 …
            return 99

        def _qf_sort_key(m):
            p1 = _r16_bpos(m["home"])
            p2 = _r16_bpos(m["away"])
            return min(p1, p2) // 4 if min(p1, p2) < 99 else 99

        by_round["Round of 32"].sort(key=_r32_sort_key)
        by_round["Round of 16"].sort(key=_r16_sort_key)
        by_round["Quarter-Finals"].sort(key=_qf_sort_key)
        # SF and Final keep date order (only 2 and 1 match respectively)

        # ── Fill missing R32 slots from live standings ───────────────
        # ESPN sometimes publishes R32 matches gradually after the group
        # stage ends. Fill any gaps using the standings so all qualified
        # teams are visible. Unassigned teams are paired sequentially;
        # remaining best 3rd-place qualifiers are included too.
        try:
            stnd = fetch_espn_standings()
            if stnd and len(by_round["Round of 32"]) < 16:
                assigned = set()
                for m in by_round["Round of 32"]:
                    assigned.add(m["home"])
                    assigned.add(m["away"])

                # Collect unassigned top-2 teams (guaranteed R32 qualifiers)
                unassigned = []
                for letter in sorted(stnd.keys()):
                    for td in stnd[letter][:2]:
                        t = td["team"]
                        if t and t not in assigned:
                            unassigned.append(t)
                            assigned.add(t)

                # Add best unassigned 3rd-place teams to fill remaining slots
                # Total teams still needed = (16 - ESPN_count) * 2 - top2_unassigned
                total_teams_needed = max(0, (16 - len(by_round["Round of 32"])) * 2 - len(unassigned))
                third_candidates = []
                for letter in sorted(stnd.keys()):
                    if len(stnd[letter]) > 2:
                        t = stnd[letter][2]["team"]
                        pts = _safe(stnd[letter][2].get("pts", 0))
                        if t and t not in assigned:
                            third_candidates.append((pts, t))
                third_candidates.sort(reverse=True)
                for _, t in third_candidates[:total_teams_needed]:
                    if t not in assigned:
                        unassigned.append(t)
                        assigned.add(t)

                # Pair teams sequentially and append as bracket matches
                for i in range(0, len(unassigned) - 1, 2):
                    home, away = unassigned[i], unassigned[i + 1]
                    by_round["Round of 32"].append({
                        "id":         f"gen_{i}",
                        "date":       "TBD",
                        "round":      "Round of 32",
                        "home":       home,
                        "home_code":  COUNTRY_CODES.get(home, "un"),
                        "home_score": "",
                        "away":       away,
                        "away_code":  COUNTRY_CODES.get(away, "un"),
                        "away_score": "",
                        "status":     "upcoming",
                        "winner":     "",
                        "is_tbd":     True,
                    })
                # If an odd team is left, add vs TBD
                if len(unassigned) % 2 == 1:
                    home = unassigned[-1]
                    by_round["Round of 32"].append({
                        "id":         f"gen_odd",
                        "date":       "TBD",
                        "round":      "Round of 32",
                        "home":       home,
                        "home_code":  COUNTRY_CODES.get(home, "un"),
                        "home_score": "", "away": "TBD", "away_code": "un",
                        "away_score": "", "status": "upcoming",
                        "winner": "", "is_tbd": True,
                    })
        except Exception as ge:
            print(f"R32 fill error: {ge}")

        rounds = [{"name": r, "matches": by_round[r]}
                  for r in ROUND_ORDER if by_round[r]]
        result = {"rounds": rounds}
        _bracket_cache["data"]  = result
        _bracket_cache["ts"]    = now
        return result
    except Exception as e:
        print(f"Bracket fetch error: {e}")
        return None


@app.route("/api/bracket")
def api_bracket():
    data = fetch_bracket()
    if not data:
        return jsonify({"error": "Could not fetch bracket"}), 503
    return jsonify(data)


@app.route("/api/player-stats")
def api_player_stats():
    data = fetch_player_stats()
    if not data:
        return jsonify({"error": "Could not fetch player stats"}), 503
    return jsonify(data)


_group_fixtures_cache: dict = {"data": None, "ts": 0.0}

# Reverse lookup: team name → group letter
_TEAM_TO_GROUP = {
    t: letter
    for letter, teams in WC_2026_GROUPS.items()
    for t in teams
}


def fetch_all_group_fixtures():
    """
    Fetch all 72 group stage matches from ESPN (Jun 11–28) with live scores.
    Organised by group letter A–L, sorted by date, with matchday assigned.
    Cached 5 minutes.
    """
    import time as _t
    from datetime import datetime as _dt, timedelta as _td
    now = _t.time()
    if _group_fixtures_cache["data"] and now - _group_fixtures_cache["ts"] < 300:
        return _group_fixtures_cache["data"]

    all_matches = []
    for i in range(18):          # Jun 11 – Jun 28
        d = (_dt(2026, 6, 11) + _td(days=i)).strftime("%Y%m%d")
        try:
            r = requests.get(
                "https://site.api.espn.com/apis/site/v2/sports/soccer"
                f"/fifa.world/scoreboard?dates={d}",
                timeout=5, headers={"User-Agent": "Mozilla/5.0"})
            for ev in r.json().get("events", []):
                comp  = ev["competitions"][0]
                comps = comp["competitors"]
                h_c   = comps[0] if comps else {}
                a_c   = comps[1] if len(comps) > 1 else {}

                raw_h = h_c.get("team", {}).get("displayName", "")
                raw_a = a_c.get("team", {}).get("displayName", "")
                home  = ESPN_NAME_MAP.get(raw_h, raw_h)
                away  = ESPN_NAME_MAP.get(raw_a, raw_a)

                hs    = str(h_c.get("score", ""))
                as_   = str(a_c.get("score", ""))

                st     = comp["status"]["type"]
                sname  = st.get("name", "")
                period = comp["status"].get("period", 0) or 0
                if st.get("completed") or "FINAL" in sname or "FULL_TIME" in sname:
                    status = "completed"
                elif period > 0 or any(x in sname for x in (
                    "IN_PROGRESS", "HALFTIME", "FIRST_HALF", "SECOND_HALF",
                    "EXTRA_TIME", "OVERTIME", "PENALTY", "PAUSE"
                )):
                    status = "live"
                else:
                    status = "upcoming"

                # Both teams must be in the same group for a valid group match
                g_home = _TEAM_TO_GROUP.get(home)
                g_away = _TEAM_TO_GROUP.get(away)
                group  = g_home if g_home and g_home == g_away else None
                winner = ""
                if status == "completed" and hs.isdigit() and as_.isdigit():
                    winner = home if int(hs) > int(as_) else (
                        away if int(as_) > int(hs) else "draw")

                all_matches.append({
                    "date":       ev.get("date", "")[:10],
                    "group":      group or "?",
                    "home":       home,
                    "home_code":  COUNTRY_CODES.get(home, "un"),
                    "home_score": hs if status in ("completed", "live") else "",
                    "away":       away,
                    "away_code":  COUNTRY_CODES.get(away, "un"),
                    "away_score": as_ if status in ("completed", "live") else "",
                    "status":     status,
                    "winner":     winner,
                })
        except Exception as e:
            print(f"Group fixtures {d}: {e}")

    # Organise by group, sort by date, assign matchday
    by_group: dict = {g: [] for g in "ABCDEFGHIJKL"}
    for m in all_matches:
        if m["group"] in by_group:
            by_group[m["group"]].append(m)
    for matches in by_group.values():
        matches.sort(key=lambda x: x["date"])
        for i, m in enumerate(matches):
            m["matchday"] = i // 2 + 1

    result = {"groups": by_group}
    _group_fixtures_cache["data"] = result
    _group_fixtures_cache["ts"]   = now
    return result


@app.route("/api/group-fixtures")
def api_group_fixtures():
    data = fetch_all_group_fixtures()
    if not data:
        return jsonify({"error": "Could not fetch fixtures"}), 503
    return jsonify(data)


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
