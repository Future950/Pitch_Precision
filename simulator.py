# ============================================================
# fifa_predictor/models/simulator.py
# Monte Carlo tournament simulator
# ============================================================

import numpy as np
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Tuple

from goals_model import predict_match_goals, sample_scoreline
from outcome_model import predict_match_outcome
from features import FEATURE_COLS


def build_team_feature_store(results_df, rankings_df, teams, reference_date, elo_history=None):
    """
    Pre-compute feature vectors for each team.
    Returns dict: team_name -> feature_dict
    """
    from features import (
        get_team_form, get_goals_avg, get_draw_rate,
        get_fifa_rank_points, get_elo_rating, compute_elo_history,
        get_squad_value_log,
    )

    if elo_history is None:
        elo_history = compute_elo_history(results_df)

    store = {}
    for team in teams:
        form = get_team_form(results_df, team, reference_date)
        gs10, gc10 = get_goals_avg(results_df, team, reference_date, n=10)
        gs5,  _    = get_goals_avg(results_df, team, reference_date, n=5)
        dr   = get_draw_rate(results_df, team, reference_date)
        pts  = get_fifa_rank_points(rankings_df, team, reference_date)
        elo  = get_elo_rating(elo_history, team, reference_date)
        sv   = get_squad_value_log(rankings_df, team, reference_date)
        store[team] = {
            "form":                form,
            "goals_scored_avg":    gs10,
            "goals_conceded_avg":  gc10,
            "goals_scored_avg_5":  gs5,
            "draw_rate":           dr,
            "rank_points":         pts,
            "elo":                 elo,
            "squad_value_log":     sv,
        }
    return store


def build_match_features(home_team, away_team, team_store):
    """Assemble feature dict for a specific matchup from the team store."""
    h = team_store[home_team]
    a = team_store[away_team]
    return {
        "home_form":                 h["form"],
        "away_form":                 a["form"],
        "home_goals_scored_avg":     h["goals_scored_avg"],
        "home_goals_conceded_avg":   h["goals_conceded_avg"],
        "away_goals_scored_avg":     a["goals_scored_avg"],
        "away_goals_conceded_avg":   a["goals_conceded_avg"],
        "home_goals_scored_avg_5":   h["goals_scored_avg_5"],
        "away_goals_scored_avg_5":   a["goals_scored_avg_5"],
        "home_draw_rate":            h["draw_rate"],
        "away_draw_rate":            a["draw_rate"],
        "home_rank_points":          h["rank_points"],
        "away_rank_points":          a["rank_points"],
        "rank_diff":                 h["rank_points"] - a["rank_points"],
        "home_elo":                  h["elo"],
        "away_elo":                  a["elo"],
        "elo_diff":                  h["elo"] - a["elo"],
        "home_squad_value_log":      h["squad_value_log"],
        "away_squad_value_log":      a["squad_value_log"],
        "squad_value_log_diff":      h["squad_value_log"] - a["squad_value_log"],
        "h2h_home_winrate":          0.5,
        "is_neutral":                1,
    }


def simulate_match(home_team, away_team, team_store, outcome_model, hgm, agm, knockout=False):
    features = build_match_features(home_team, away_team, team_store)
    xg = predict_match_goals(hgm, agm, features)
    home_g, away_g = sample_scoreline(xg["home_xg"], xg["away_xg"])

    if home_g > away_g:
        winner = home_team
    elif away_g > home_g:
        winner = away_team
    else:
        winner = np.random.choice([home_team, away_team]) if knockout else "draw"

    return winner, home_g, away_g


def simulate_group_stage(groups, team_store, outcome_model, hgm, agm):
    """Simulate full group stage. Returns (qualified dict, third_place_info list)."""
    qualified = {}
    third_place_info = []

    for group_name, teams in groups.items():
        points = defaultdict(int)
        gd = defaultdict(int)

        for i in range(len(teams)):
            for j in range(i + 1, len(teams)):
                home, away = teams[i], teams[j]
                winner, hg, ag = simulate_match(home, away, team_store, outcome_model, hgm, agm)
                gd[home] += hg - ag
                gd[away] += ag - hg
                if winner == home:
                    points[home] += 3
                elif winner == away:
                    points[away] += 3
                else:
                    points[home] += 1
                    points[away] += 1

        standings = sorted(teams, key=lambda t: (points[t], gd[t]), reverse=True)
        qualified[group_name] = standings
        third_place_info.append((points[standings[2]], gd[standings[2]], standings[2]))

    return qualified, third_place_info


def simulate_knockout_stage(qualified, third_place_info, team_store, outcome_model, hgm, agm):
    """
    2026 format: 12 groups -> top 2 (24 teams) + best 8 third-place = 32-team bracket.
    Returns (stage_reached dict, champion str).
    """
    group_names = sorted(qualified.keys())
    top2 = []
    for g in group_names:
        top2.append((g, 1, qualified[g][0]))
        top2.append((g, 2, qualified[g][1]))

    third_sorted = sorted(third_place_info, key=lambda x: (x[0], x[1]), reverse=True)
    best_third = [t[2] for t in third_sorted[:8]]

    bracket = [t for _, _, t in top2] + best_third
    r32_matches = [(bracket[i], bracket[31 - i]) for i in range(16)]

    stage_reached = {}
    current_round = r32_matches
    round_names = ["Round of 32", "Round of 16", "Quarter-Finals", "Semi-Finals", "Final"]
    champion = None

    for round_name in round_names:
        next_round = []
        for home, away in current_round:
            # Mark both teams as having reached this round; loser keeps this stage
            stage_reached[home] = round_name
            stage_reached[away] = round_name
            winner, _, _ = simulate_match(home, away, team_store, outcome_model, hgm, agm, knockout=True)
            next_round.append(winner)

        if len(next_round) == 1:
            champion = next_round[0]
            stage_reached[champion] = "Winner"
            break

        current_round = [(next_round[i], next_round[i + 1]) for i in range(0, len(next_round), 2)]

    return stage_reached, champion


STAGE_ORDER = [
    "Group Stage",
    "Round of 32",
    "Round of 16",
    "Quarter-Finals",
    "Semi-Finals",
    "Final",
    "Winner",
]


def run_tournament_simulation(groups, team_store, outcome_model, home_goals_model, away_goals_model, n_simulations=10_000):
    """Run N full tournament simulations and aggregate results."""
    all_teams = [t for group in groups.values() for t in group]
    stage_counts = {team: defaultdict(int) for team in all_teams}
    win_counts = defaultdict(int)

    print(f"Running {n_simulations:,} tournament simulations...")

    for sim in range(n_simulations):
        if sim % 1000 == 0 and sim > 0:
            print(f"  Completed {sim:,} simulations...")

        qualified, third_place_info = simulate_group_stage(
            groups, team_store, outcome_model, home_goals_model, away_goals_model
        )

        third_sorted = sorted(third_place_info, key=lambda x: (x[0], x[1]), reverse=True)
        advancing_third = {t[2] for t in third_sorted[:8]}

        group_eliminated = []
        for group_standings in qualified.values():
            group_eliminated.append(group_standings[3])
            if group_standings[2] not in advancing_third:
                group_eliminated.append(group_standings[2])

        stage_reached, champion = simulate_knockout_stage(
            qualified, third_place_info, team_store, outcome_model, home_goals_model, away_goals_model
        )

        for team in group_eliminated:
            stage_counts[team]["Group Stage"] += 1

        for team, stage in stage_reached.items():
            stage_counts[team][stage] += 1

        win_counts[champion] += 1

    rows = []
    for team in all_teams:
        total = n_simulations
        rows.append({
            "team":           team,
            "win_pct":        round(win_counts[team] / total * 100, 2),
            "final_pct":      round(stage_counts[team].get("Final", 0) / total * 100, 2),
            "semi_pct":       round(stage_counts[team].get("Semi-Finals", 0) / total * 100, 2),
            "qf_pct":         round(stage_counts[team].get("Quarter-Finals", 0) / total * 100, 2),
            "r16_pct":        round(stage_counts[team].get("Round of 16", 0) / total * 100, 2),
            "r32_pct":        round(stage_counts[team].get("Round of 32", 0) / total * 100, 2),
            "group_exit_pct": round(stage_counts[team].get("Group Stage", 0) / total * 100, 2),
        })

    return pd.DataFrame(rows).sort_values("win_pct", ascending=False).reset_index(drop=True)
