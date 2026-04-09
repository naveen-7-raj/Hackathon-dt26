"""
xai_explainer.py
SHAP-based explainability:
  - Global feature importance (mean absolute SHAP values)
  - Local explanation for a single prediction
"""

import numpy as np
import shap
from sklearn.linear_model import LogisticRegression


FEATURE_LABELS = {
    "income": "Income",
    "education": "Education Level",
    "age": "Age",
    "gender_enc": "Gender (Female)",
    "region_enc": "Region (Rural)",
}


def compute_global_shap(model: LogisticRegression, X_sample: np.ndarray, feature_names: list) -> dict:
    """
    Returns mean |SHAP| per feature across a sample of the training/test set.
    Uses KernelExplainer on a background of 200 samples for speed.
    """
    # For LogisticRegression use LinearExplainer (exact and fast)
    explainer = shap.LinearExplainer(model, X_sample, feature_perturbation="interventional")
    shap_values = explainer.shap_values(X_sample)       # shape (n, features)

    mean_abs = np.abs(shap_values).mean(axis=0)
    total = mean_abs.sum() if mean_abs.sum() > 0 else 1.0

    return {
        "feature_names": [FEATURE_LABELS.get(f, f) for f in feature_names],
        "raw_names": feature_names,
        "mean_abs_shap": mean_abs.tolist(),
        "importance_pct": (mean_abs / total * 100).round(2).tolist(),
    }


def compute_local_shap(model: LogisticRegression, X_background: np.ndarray,
                       x_instance: np.ndarray, feature_names: list) -> dict:
    """
    Returns SHAP values for a single prediction instance.
    x_instance should be a 1-D array (already scaled).
    """
    explainer = shap.LinearExplainer(model, X_background, feature_perturbation="interventional")
    sv = explainer.shap_values(x_instance.reshape(1, -1))[0]   # shape (features,)

    base_value = explainer.expected_value
    prediction_prob = float(model.predict_proba(x_instance.reshape(1, -1))[0, 1])

    contributions = []
    for i, (fname, sval) in enumerate(zip(feature_names, sv)):
        contributions.append({
            "feature": FEATURE_LABELS.get(fname, fname),
            "raw_name": fname,
            "shap_value": round(float(sval), 5),
            "direction": "positive" if sval > 0 else "negative",
        })

    # Sort by absolute value descending
    contributions.sort(key=lambda c: abs(c["shap_value"]), reverse=True)

    return {
        "base_value": round(float(base_value), 5),
        "prediction_prob": round(prediction_prob, 4),
        "approved": prediction_prob >= 0.5,
        "contributions": contributions,
    }
