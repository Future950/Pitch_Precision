# ============================================================
# fifa_predictor/models/outcome_model.py
# XGBoost classifier - predicts Win / Draw / Loss
# ============================================================

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import accuracy_score, log_loss, classification_report
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb

from features import FEATURE_COLS

MODEL_PATH = Path(__file__).parent / "outputs" / "outcome_model.pkl"


def train_outcome_model(feature_df: pd.DataFrame, save=True):
    """
    Train an XGBoost multi-class classifier on match outcomes.

    Returns model, eval_metrics dict.
    """
    df = feature_df.dropna(subset=FEATURE_COLS + ["outcome"]).copy()

    X = df[FEATURE_COLS].values
    y = df["outcome"].values  # 0=home win, 1=draw, 2=away win

    split = int(len(df) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = xgb.XGBClassifier(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.7,
        min_child_weight=5,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.5,
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=50,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=100,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "log_loss": round(log_loss(y_test, y_prob), 4),
        "best_iteration": int(model.best_iteration),
    }

    print("\n===== Outcome Model Evaluation =====")
    print(f"Accuracy      : {metrics['accuracy']}")
    print(f"Log-Loss      : {metrics['log_loss']}")
    print(f"Best iteration: {metrics['best_iteration']}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"]))

    if save:
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        print(f"Model saved -> {MODEL_PATH}")

    return model, metrics


def load_outcome_model():
    return joblib.load(MODEL_PATH)


def predict_match_outcome(model, features: dict) -> dict:
    X = np.array([[features[col] for col in FEATURE_COLS]])
    probs = model.predict_proba(X)[0]
    return {
        "home_win": round(float(probs[0]), 4),
        "draw":     round(float(probs[1]), 4),
        "away_win": round(float(probs[2]), 4),
    }
