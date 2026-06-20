"""
Predict a single match.

Usage:
    python predict.py "Brazil" "Argentina"
    python predict.py "England" "France" --neutral 0
    python predict.py "Germany" "Spain" --date 2026-07-01
"""
import argparse
import joblib
import pandas as pd
from pathlib import Path

from features import (
    load_results, load_rankings, compute_elo_history,
    get_team_form, get_goals_avg, get_draw_rate,
    get_fifa_rank_points, get_head_to_head, get_elo_rating, get_squad_value_log,
)
from outcome_model import predict_match_outcome
from goals_model import predict_match_goals

DATA_DIR = Path("data")
OUT_DIR  = Path("outputs")

parser = argparse.ArgumentParser()
parser.add_argument("home", help="Home team name (use quotes for multi-word names)")
parser.add_argument("away", help="Away team name")
parser.add_argument("--neutral", type=int, default=1, help="1=neutral venue (default), 0=home ground")
parser.add_argument("--date", default="2026-06-17", help="Reference date YYYY-MM-DD")
args = parser.parse_args()

results_df  = load_results(DATA_DIR / "results.csv")
rankings_df = load_rankings(DATA_DIR / "fifa_ranking.csv")

print("Computing ELO history...")
elo_history = compute_elo_history(results_df)

outcome_model    = joblib.load(OUT_DIR / "outcome_model.pkl")
home_goals_model = joblib.load(OUT_DIR / "goals_home_model.pkl")
away_goals_model = joblib.load(OUT_DIR / "goals_away_model.pkl")

date = pd.Timestamp(args.date)
home, away = args.home, args.away

features = {
    "home_form":               get_team_form(results_df, home, date),
    "away_form":               get_team_form(results_df, away, date),
    "home_goals_scored_avg":   get_goals_avg(results_df, home, date, n=10)[0],
    "home_goals_conceded_avg": get_goals_avg(results_df, home, date, n=10)[1],
    "away_goals_scored_avg":   get_goals_avg(results_df, away, date, n=10)[0],
    "away_goals_conceded_avg": get_goals_avg(results_df, away, date, n=10)[1],
    "home_goals_scored_avg_5": get_goals_avg(results_df, home, date, n=5)[0],
    "away_goals_scored_avg_5": get_goals_avg(results_df, away, date, n=5)[0],
    "home_draw_rate":          get_draw_rate(results_df, home, date),
    "away_draw_rate":          get_draw_rate(results_df, away, date),
    "home_rank_points":        get_fifa_rank_points(rankings_df, home, date),
    "away_rank_points":        get_fifa_rank_points(rankings_df, away, date),
    "rank_diff":               get_fifa_rank_points(rankings_df, home, date) - get_fifa_rank_points(rankings_df, away, date),
    "home_elo":                get_elo_rating(elo_history, home, date),
    "away_elo":                get_elo_rating(elo_history, away, date),
    "elo_diff":                get_elo_rating(elo_history, home, date) - get_elo_rating(elo_history, away, date),
    "home_squad_value_log":    get_squad_value_log(rankings_df, home, date),
    "away_squad_value_log":    get_squad_value_log(rankings_df, away, date),
    "squad_value_log_diff":    get_squad_value_log(rankings_df, home, date) - get_squad_value_log(rankings_df, away, date),
    "h2h_home_winrate":        get_head_to_head(results_df, home, away, date),
    "is_neutral":              args.neutral,
}

probs = predict_match_outcome(outcome_model, features)
goals = predict_match_goals(home_goals_model, away_goals_model, features)

venue = "neutral venue" if args.neutral else "home ground"
print(f"\n  {home} vs {away}  ({venue})")
print(f"  {'-' * 42}")
print(f"  {home:<22} win : {probs['home_win']*100:>5.1f}%")
print(f"  {'Draw':<22}     : {probs['draw']*100:>5.1f}%")
print(f"  {away:<22} win : {probs['away_win']*100:>5.1f}%")
print(f"\n  Expected goals : {home} {goals['home_xg']:.2f}  -  {goals['away_xg']:.2f} {away}")
print(f"  ELO            : {home} {features['home_elo']:.0f}  -  {features['away_elo']:.0f} {away}")
print(f"  FIFA pts       : {home} {features['home_rank_points']:.0f}  -  {features['away_rank_points']:.0f} {away}")
print(f"  H2H win rate   : {home} {features['h2h_home_winrate']*100:.0f}%")
print()
