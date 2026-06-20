"""
Retrain only the outcome model (class-balanced) on the existing feature matrix,
then re-run the tournament simulation.
"""
import pandas as pd
import joblib
from pathlib import Path

from features import load_results, load_rankings, compute_elo_history, get_elo_rating
from outcome_model import train_outcome_model
from goals_model import load_goals_models
from simulator import build_team_feature_store, run_tournament_simulation

DATA_DIR = Path("data")
OUT_DIR  = Path("outputs")

print("Loading feature matrix...")
feature_df = pd.read_csv(OUT_DIR / "feature_matrix.csv", parse_dates=["date"])
print(f"  Shape: {feature_df.shape}")

print("Retraining outcome model with balanced class weights...")
outcome_model, metrics = train_outcome_model(feature_df, save=True)

print("\n=== New Outcome Model Metrics ===")
for k, v in metrics.items():
    print(f"  {k}: {v}")

print("\nLoading goals models and running tournament simulation...")
results_df  = load_results(DATA_DIR / "results.csv")
rankings_df = load_rankings(DATA_DIR / "fifa_ranking.csv")

print("  Computing ELO history...")
elo_history = compute_elo_history(results_df)

home_goals_model, away_goals_model = load_goals_models()

REFERENCE_DATE = pd.Timestamp("2026-06-10")

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

results = run_tournament_simulation(
    groups=WC_2026_GROUPS,
    team_store=team_store,
    outcome_model=outcome_model,
    home_goals_model=home_goals_model,
    away_goals_model=away_goals_model,
    n_simulations=10_000,
)

results.to_csv(OUT_DIR / "tournament_predictions.csv", index=False)

print("\nTOURNAMENT WIN PROBABILITIES (Top 16)\n")
print(f"{'Team':<28} {'Win%':>5}  {'Final%':>6}  {'Semi%':>5}  {'QF%':>5}  {'R16%':>5}  {'R32%':>5}  {'Out%':>5}")
print("-" * 80)
for _, row in results.head(16).iterrows():
    print(
        f"{row['team']:<28} {row['win_pct']:>5.1f}%  "
        f"{row['final_pct']:>5.1f}%  {row['semi_pct']:>5.1f}%  "
        f"{row['qf_pct']:>5.1f}%  {row['r16_pct']:>5.1f}%  "
        f"{row['r32_pct']:>5.1f}%  {row['group_exit_pct']:>5.1f}%"
    )

print("\nDone.")
