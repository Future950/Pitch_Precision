# ============================================================
# fifa_predictor/main.py
# Full pipeline: train -> evaluate -> simulate -> report
# ============================================================

import pandas as pd
import numpy as np
from pathlib import Path

from features import (
    load_results, load_rankings, build_feature_matrix, compute_elo_history,
    get_team_form, get_goals_avg, get_draw_rate,
    get_fifa_rank_points, get_head_to_head, get_h2h_avg_goals,
    get_elo_rating, get_squad_value_log,
)
from outcome_model import train_outcome_model, predict_match_outcome
from goals_model import train_goals_model, predict_match_goals
from simulator import build_team_feature_store, run_tournament_simulation

DATA_DIR = Path("data")
OUT_DIR  = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)


# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────

print("=" * 55)
print("STEP 1 -- Loading data")
print("=" * 55)

results_df  = load_results(DATA_DIR / "results.csv")
rankings_df = load_rankings(DATA_DIR / "fifa_ranking.csv")

print(f"  Match results loaded   : {len(results_df):,} rows")
print(f"  FIFA rankings loaded   : {len(rankings_df):,} rows")
print(f"  Date range             : {results_df['date'].min().date()} to {results_df['date'].max().date()}")

print("\n  Pre-computing ELO ratings from full results history...")
elo_history = compute_elo_history(results_df)
print(f"  ELO history built for {len(elo_history):,} teams.")


# ─────────────────────────────────────────────
# STEP 2: BUILD FEATURE MATRIX
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 2 -- Building feature matrix")
print("=" * 55)

# Competitive matches from 1994 onwards (modern era)
competitive = results_df[
    results_df["tournament"].str.contains(
        "FIFA World Cup|UEFA|CONMEBOL|CAF|AFC|CONCACAF|Copa|Gold Cup|Nations League",
        case=False, na=False
    )
    & (results_df["date"] >= pd.Timestamp("1994-01-01"))
].copy()

print(f"  Competitive matches (1994+) : {len(competitive):,}")

feature_df = build_feature_matrix(competitive, rankings_df, elo_history=elo_history)
feature_df.to_csv(OUT_DIR / "feature_matrix.csv", index=False)
print(f"  Feature matrix saved -> outputs/feature_matrix.csv")
print(f"  Feature matrix shape : {feature_df.shape}")


# ─────────────────────────────────────────────
# STEP 3: TRAIN MODELS
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 3 -- Training models")
print("=" * 55)

print("\n--- Training Outcome Model (XGBoost Classifier) ---")
outcome_model, outcome_metrics = train_outcome_model(feature_df)

print("\n--- Training Goals Model (XGBoost Poisson Regressor) ---")
home_goals_model, away_goals_model, goals_metrics = train_goals_model(feature_df)


# ─────────────────────────────────────────────
# STEP 4: SINGLE MATCH DEMO
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 4 -- Single match prediction demo")
print("=" * 55)

REFERENCE_DATE = pd.Timestamp("2026-07-01")

def make_matchup_features(home, away, results_df, rankings_df, elo_history, date, neutral=1):
    hf      = get_team_form(results_df, home, date)
    af      = get_team_form(results_df, away, date)
    hgs, hgc = get_goals_avg(results_df, home, date, n=10)
    ags, agc = get_goals_avg(results_df, away, date, n=10)
    hgs5, _ = get_goals_avg(results_df, home, date, n=5)
    ags5, _ = get_goals_avg(results_df, away, date, n=5)
    hdr     = get_draw_rate(results_df, home, date)
    adr     = get_draw_rate(results_df, away, date)
    hp      = get_fifa_rank_points(rankings_df, home, date)
    ap      = get_fifa_rank_points(rankings_df, away, date)
    helo    = get_elo_rating(elo_history, home, date)
    aelo    = get_elo_rating(elo_history, away, date)
    hsv     = get_squad_value_log(rankings_df, home, date)
    asv     = get_squad_value_log(rankings_df, away, date)
    h2h      = get_head_to_head(results_df, home, away, date)
    h2hgoals = get_h2h_avg_goals(results_df, home, away, date)
    return {
        "home_form": hf, "away_form": af,
        "home_goals_scored_avg": hgs, "home_goals_conceded_avg": hgc,
        "away_goals_scored_avg": ags, "away_goals_conceded_avg": agc,
        "home_goals_scored_avg_5": hgs5, "away_goals_scored_avg_5": ags5,
        "home_draw_rate": hdr, "away_draw_rate": adr,
        "home_rank_points": hp, "away_rank_points": ap,
        "rank_diff": hp - ap,
        "home_elo": helo, "away_elo": aelo, "elo_diff": helo - aelo,
        "home_squad_value_log": hsv, "away_squad_value_log": asv,
        "squad_value_log_diff": hsv - asv,
        "home_attack_vs_away_def": hgs / max(agc, 0.5),
        "away_attack_vs_home_def": ags / max(hgc, 0.5),
        "h2h_home_winrate": h2h,
        "h2h_avg_goals": h2hgoals,
        "is_neutral": neutral,
    }

demo_match = ("Brazil", "Argentina")
features   = make_matchup_features(*demo_match, results_df, rankings_df, elo_history, REFERENCE_DATE)

outcome_probs = predict_match_outcome(outcome_model, features)
goals_pred    = predict_match_goals(home_goals_model, away_goals_model, features)

print(f"\n  {demo_match[0]} vs {demo_match[1]}")
print(f"  Outcome probabilities:")
print(f"    {demo_match[0]} win : {outcome_probs['home_win']*100:.1f}%")
print(f"    Draw              : {outcome_probs['draw']*100:.1f}%")
print(f"    {demo_match[1]} win : {outcome_probs['away_win']*100:.1f}%")
print(f"  Expected Goals:")
print(f"    {demo_match[0]}: {goals_pred['home_xg']:.2f}")
print(f"    {demo_match[1]}: {goals_pred['away_xg']:.2f}")


# ─────────────────────────────────────────────
# STEP 5: TOURNAMENT SIMULATION
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 5 -- Monte Carlo tournament simulation (2026 WC)")
print("=" * 55)

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

all_teams = [t for group in WC_2026_GROUPS.values() for t in group]

print("  Building team feature store...")
team_store = build_team_feature_store(
    results_df, rankings_df, all_teams, REFERENCE_DATE, elo_history=elo_history
)

simulation_results = run_tournament_simulation(
    groups=WC_2026_GROUPS,
    team_store=team_store,
    outcome_model=outcome_model,
    home_goals_model=home_goals_model,
    away_goals_model=away_goals_model,
    n_simulations=10_000,
)


# ─────────────────────────────────────────────
# STEP 6: PRINT AND SAVE RESULTS
# ─────────────────────────────────────────────

print("\n" + "=" * 55)
print("STEP 6 -- Results")
print("=" * 55)

print("\n TOURNAMENT WIN PROBABILITIES (Top 16)\n")
print(f"{'Team':<28} {'Win%':>5}  {'Final%':>6}  {'Semi%':>5}  {'QF%':>5}  {'R16%':>5}  {'R32%':>5}  {'Out%':>5}")
print("-" * 80)
for _, row in simulation_results.head(16).iterrows():
    print(
        f"{row['team']:<28} {row['win_pct']:>5.1f}%  "
        f"{row['final_pct']:>5.1f}%  {row['semi_pct']:>5.1f}%  "
        f"{row['qf_pct']:>5.1f}%  {row['r16_pct']:>5.1f}%  "
        f"{row['r32_pct']:>5.1f}%  {row['group_exit_pct']:>5.1f}%"
    )

out_path = OUT_DIR / "tournament_predictions.csv"
simulation_results.to_csv(out_path, index=False)
print(f"\n  Full results saved -> {out_path}")

metrics_path = OUT_DIR / "model_metrics.txt"
with open(metrics_path, "w") as f:
    f.write("=== Outcome Model ===\n")
    for k, v in outcome_metrics.items():
        f.write(f"  {k}: {v}\n")
    f.write("\n=== Goals Model ===\n")
    for k, v in goals_metrics.items():
        f.write(f"  {k}: {v}\n")

print(f"  Model metrics saved -> {metrics_path}")
print("\nDone.")
