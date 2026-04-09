"""
bias_metrics.py
Computes fairness metrics for a trained classifier on a test set:
  - Demographic Parity Difference (DPD)
  - Equal Opportunity Difference (EOD)
  - Disparate Impact Ratio (DIR)
  - Approval rates by group (gender, region)
"""

import numpy as np
import pandas as pd


def _group_metrics(y_true, y_pred, group_series):
    """Compute per-group positive rates and TPRs."""
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": group_series})
    results = {}
    for g, sub in df.groupby("group"):
        pos_rate = sub["y_pred"].mean()
        tp_mask = sub["y_true"] == 1
        tpr = sub.loc[tp_mask, "y_pred"].mean() if tp_mask.sum() > 0 else 0.0
        results[g] = {"positive_rate": pos_rate, "tpr": tpr, "n": len(sub)}
    return results


def compute_bias_metrics(y_true, y_pred, df_test):
    """
    Returns a dict with bias metrics for gender and region groups.
    """
    out = {}

    for attr in ["gender", "region"]:
        gm = _group_metrics(y_true, y_pred, df_test[attr])
        groups = list(gm.keys())

        # Approval rates per group
        out[f"{attr}_approval_rates"] = {
            g: round(float(gm[g]["positive_rate"]), 4) for g in groups
        }

        # Demographic Parity Difference  (max – min across groups)
        pos_rates = [gm[g]["positive_rate"] for g in groups]
        out[f"{attr}_dpd"] = round(float(max(pos_rates) - min(pos_rates)), 4)

        # Equal Opportunity Difference
        tprs = [gm[g]["tpr"] for g in groups]
        out[f"{attr}_eod"] = round(float(max(tprs) - min(tprs)), 4)

        # Disparate Impact Ratio  (min / max)
        if max(pos_rates) > 0:
            out[f"{attr}_dir"] = round(float(min(pos_rates) / max(pos_rates)), 4)
        else:
            out[f"{attr}_dir"] = None

    # Overall approval rate
    out["overall_approval_rate"] = round(float(y_pred.mean()), 4)

    return out


def summarize_dataset(df: pd.DataFrame) -> dict:
    """High-level stats about the dataset."""
    return {
        "n_samples": len(df),
        "approval_rate": round(df["approved"].mean(), 4),
        "gender_dist": df["gender"].value_counts().to_dict(),
        "region_dist": df["region"].value_counts().to_dict(),
        "education_dist": df["education"].value_counts().sort_index().to_dict(),
        "income_mean": round(df["income"].mean(), 2),
        "income_median": round(df["income"].median(), 2),
        "age_mean": round(df["age"].mean(), 2),
        "approval_by_gender": df.groupby("gender")["approved"].mean().round(4).to_dict(),
        "approval_by_region": df.groupby("region")["approved"].mean().round(4).to_dict(),
        "approval_by_education": df.groupby("education")["approved"].mean().round(4).to_dict(),
    }
