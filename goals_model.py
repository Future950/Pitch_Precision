# ============================================================
# fifa_predictor/models/goals_model.py
# Poisson regression - predicts expected goals per team
# ============================================================

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb

from features import FEATURE_COLS

HOME_GOALS_MODEL_PATH = Path(__file__).parent / "outputs" / "goals_home_model.pkl"
AWAY_GOALS_MODEL_PATH = Path(__file__).parent / "outputs" / "goals_away_model.pkl"


def train_goals_model(feature_df: pd.DataFrame, save=True):
    """
    Train two XGBoost Poisson regressors for home and away expected goals.
    Returns home_model, away_model, metrics dict.
    """
    df = feature_df.dropna(subset=FEATURE_COLS + ["home_goals", "away_goals"]).copy()

    X = df[FEATURE_COLS].values
    y_home = df["home_goals"].values
    y_away = df["away_goals"].values

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_home_train, y_home_test = y_home[:split], y_home[split:]
    y_away_train, y_away_test = y_away[:split], y_away[split:]

    params = dict(
        n_estimators=1000,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.5,
        objective="count:poisson",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    home_model = xgb.XGBRegressor(**params)
    home_model.fit(X_train, y_home_train, eval_set=[(X_test, y_home_test)], verbose=100)

    away_model = xgb.XGBRegressor(**params)
    away_model.fit(X_train, y_away_train, eval_set=[(X_test, y_away_test)], verbose=100)

    home_preds = home_model.predict(X_test)
    away_preds = away_model.predict(X_test)

    metrics = {
        "home_goals_mae":  round(mean_absolute_error(y_home_test, home_preds), 4),
        "away_goals_mae":  round(mean_absolute_error(y_away_test, away_preds), 4),
        "home_goals_rmse": round(float(np.sqrt(mean_squared_error(y_home_test, home_preds))), 4),
        "away_goals_rmse": round(float(np.sqrt(mean_squared_error(y_away_test, away_preds))), 4),
    }

    print("\n===== Goals Model Evaluation =====")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    if save:
        HOME_GOALS_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(home_model, HOME_GOALS_MODEL_PATH)
        joblib.dump(away_model, AWAY_GOALS_MODEL_PATH)
        print(f"Models saved -> {HOME_GOALS_MODEL_PATH.parent}")

    return home_model, away_model, metrics


def load_goals_models():
    return joblib.load(HOME_GOALS_MODEL_PATH), joblib.load(AWAY_GOALS_MODEL_PATH)


def predict_match_goals(home_model, away_model, features: dict) -> dict:
    X = np.array([[features[col] for col in FEATURE_COLS]])
    home_xg = float(home_model.predict(X)[0])
    away_xg = float(away_model.predict(X)[0])
    home_xg = max(0.1, min(home_xg, 6.0))
    away_xg = max(0.1, min(away_xg, 6.0))
    return {"home_xg": round(home_xg, 3), "away_xg": round(away_xg, 3)}


def sample_scoreline(home_xg: float, away_xg: float) -> tuple:
    return int(np.random.poisson(home_xg)), int(np.random.poisson(away_xg))
