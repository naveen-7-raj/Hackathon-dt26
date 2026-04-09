"""
app.py
FairLens AI – Flask backend
Serves the frontend dashboard and REST API for bias analysis.
"""

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS

from data_generator import generate_dataset, encode_features
from models import train_models
from bias_metrics import compute_bias_metrics, summarize_dataset
from xai_explainer import compute_global_shap, compute_local_shap

app = Flask(__name__)
CORS(app)

# ── In-memory state ───────────────────────────────────────────────────────────
STATE: dict = {}
INITIALIZED = False


def initialize_system():
    global STATE, INITIALIZED
    print("[FairLens] Generating dataset …")
    df = generate_dataset(n_samples=100_000)
    X, y, feature_names = encode_features(df)

    print("[FairLens] Training models …")
    result = train_models(X, y, df, feature_names)

    # Shorter alias
    scaler      = result["scaler"]
    bm          = result["biased_model"]
    fm          = result["fair_model"]
    X_test      = result["X_test"]
    y_test      = result["y_test"]
    df_test     = result["df_test"]

    y_pred_biased = bm.predict(X_test)
    y_pred_fair   = fm.predict(X_test)

    print("[FairLens] Computing bias metrics …")
    metrics_before = compute_bias_metrics(y_test, y_pred_biased, df_test)
    metrics_after  = compute_bias_metrics(y_test, y_pred_fair,   df_test)

    print("[FairLens] Computing SHAP (sample of 2000) …")
    shap_sample_idx = np.random.default_rng(42).choice(len(X_test), size=min(2000, len(X_test)), replace=False)
    X_shap = X_test[shap_sample_idx]

    global_shap_biased = compute_global_shap(bm, X_shap, feature_names)
    global_shap_fair   = compute_global_shap(fm, X_shap, feature_names)

    STATE = {
        "df": df,
        "feature_names": feature_names,
        "scaler": scaler,
        "biased_model": bm,
        "fair_model": fm,
        "X_test": X_test,
        "y_test": y_test,
        "df_test": df_test,
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "accuracy_biased": result["accuracy_biased"],
        "accuracy_fair": result["accuracy_fair"],
        "global_shap_biased": global_shap_biased,
        "global_shap_fair": global_shap_fair,
        "X_shap_background": X_shap,
        "dataset_summary": summarize_dataset(df),
    }
    INITIALIZED = True
    print("[FairLens] Initialization complete.")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/initialize", methods=["POST"])
def api_initialize():
    initialize_system()
    return jsonify({"status": "ok", "message": "System initialized successfully."})


@app.route("/api/overview")
def api_overview():
    if not INITIALIZED:
        return jsonify({"error": "Not initialized"}), 400
    return jsonify(STATE["dataset_summary"])


@app.route("/api/bias")
def api_bias():
    if not INITIALIZED:
        return jsonify({"error": "Not initialized"}), 400
    return jsonify({
        "before": STATE["metrics_before"],
        "after":  STATE["metrics_after"],
        "accuracy_biased": STATE["accuracy_biased"],
        "accuracy_fair":   STATE["accuracy_fair"],
    })


@app.route("/api/shap/global")
def api_shap_global():
    if not INITIALIZED:
        return jsonify({"error": "Not initialized"}), 400
    return jsonify({
        "biased": STATE["global_shap_biased"],
        "fair":   STATE["global_shap_fair"],
    })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """
    Expects JSON body:
    { income, education, age, gender ("male"/"female"), region ("urban"/"rural") }
    Returns prediction + local SHAP from both biased and fair models.
    """
    if not INITIALIZED:
        return jsonify({"error": "Not initialized"}), 400

    body = request.get_json()
    x_raw = _parse_input(body)
    x_scaled = STATE["scaler"].transform(x_raw.reshape(1, -1))[0]

    bg = STATE["X_shap_background"]

    biased_local = compute_local_shap(STATE["biased_model"], bg, x_scaled, STATE["feature_names"])
    fair_local   = compute_local_shap(STATE["fair_model"],   bg, x_scaled, STATE["feature_names"])

    return jsonify({
        "biased": biased_local,
        "fair":   fair_local,
        "inputs": body,
    })


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    """Lightweight prediction for real-time what-if (no local SHAP for speed)."""
    if not INITIALIZED:
        return jsonify({"error": "Not initialized"}), 400

    body = request.get_json()
    x_raw = _parse_input(body)
    x_scaled = STATE["scaler"].transform(x_raw.reshape(1, -1))

    bm = STATE["biased_model"]
    fm = STATE["fair_model"]

    prob_biased = float(bm.predict_proba(x_scaled)[0, 1])
    prob_fair   = float(fm.predict_proba(x_scaled)[0, 1])

    return jsonify({
        "biased_prob": round(prob_biased, 4),
        "fair_prob":   round(prob_fair, 4),
        "biased_decision": "Approved" if prob_biased >= 0.5 else "Rejected",
        "fair_decision":   "Approved" if prob_fair   >= 0.5 else "Rejected",
        "bias_flag": prob_biased < 0.5 and prob_fair >= 0.5,  # fair model would approve
    })


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_input(body: dict) -> np.ndarray:
    """Convert raw form values to the encoded feature vector (unscaled)."""
    income    = float(body.get("income", 50000))
    education = float(body.get("education", 2))
    age       = float(body.get("age", 35))
    gender    = body.get("gender", "male")
    region    = body.get("region", "urban")

    gender_enc = 1.0 if gender == "female" else 0.0
    region_enc = 1.0 if region == "rural"  else 0.0

    return np.array([income, education, age, gender_enc, region_enc])


if __name__ == "__main__":
    print("=" * 60)
    print("  FairLens AI – Bias Detection & Fairness Platform")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(debug=False, port=5000)
