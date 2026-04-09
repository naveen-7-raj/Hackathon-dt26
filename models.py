"""
models.py
Trains two models:
  1. biased_model  – LogisticRegression on raw biased data (no mitigation)
  2. fair_model    – Same architecture but trained with fairness-aware sample weights
Returns scaler, models, test split, and sample weights used for the fair model.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42


def compute_fairness_weights(df_train, group_col="region", label_col="approved"):
    """
    Reweighting strategy: upweight samples from disadvantaged groups so that
    the weighted positive-rate is equal across groups.
    """
    weights = np.ones(len(df_train))
    groups = df_train[group_col].unique()
    overall_pos_rate = df_train[label_col].mean()

    for g in groups:
        mask = df_train[group_col] == g
        group_pos_rate = df_train.loc[mask, label_col].mean()
        if group_pos_rate > 0:
            adjustment = overall_pos_rate / group_pos_rate
            weights[mask.values] = adjustment

    # Also reweight by gender
    for g in df_train["gender"].unique():
        mask = df_train["gender"] == g
        group_pos_rate = df_train.loc[mask, label_col].mean()
        if group_pos_rate > 0:
            adjustment = overall_pos_rate / group_pos_rate
            weights[mask.values] *= adjustment

    # Normalize so mean weight ≈ 1
    weights = weights / weights.mean()
    return weights


def train_models(X: np.ndarray, y: np.ndarray, df_full, feature_names):
    """
    Parameters
    ----------
    X           : encoded feature matrix (full dataset)
    y           : labels (full dataset)
    df_full     : original DataFrame (for group columns)
    feature_names : list of feature column names

    Returns
    -------
    dict with keys: scaler, biased_model, fair_model,
                    X_train, X_test, y_train, y_test,
                    df_train, df_test, accuracy_biased, accuracy_fair
    """
    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(y)), test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )
    df_train = df_full.iloc[idx_tr].reset_index(drop=True)
    df_test  = df_full.iloc[idx_te].reset_index(drop=True)

    # Scale features
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # ── Biased model ──────────────────────────────────────────────────────────
    biased_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    biased_model.fit(X_tr_sc, y_tr)

    # ── Fair model (reweighted) ───────────────────────────────────────────────
    sample_weights = compute_fairness_weights(df_train, group_col="region", label_col="approved")
    fair_model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, C=1.0)
    fair_model.fit(X_tr_sc, y_tr, sample_weight=sample_weights)

    acc_biased = accuracy_score(y_te, biased_model.predict(X_te_sc))
    acc_fair   = accuracy_score(y_te, fair_model.predict(X_te_sc))

    return {
        "scaler": scaler,
        "biased_model": biased_model,
        "fair_model": fair_model,
        "X_train": X_tr_sc,
        "X_test": X_te_sc,
        "y_train": y_tr,
        "y_test": y_te,
        "df_train": df_train,
        "df_test": df_test,
        "accuracy_biased": round(acc_biased, 4),
        "accuracy_fair": round(acc_fair, 4),
        "feature_names": feature_names,
    }
