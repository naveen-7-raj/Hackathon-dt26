"""
app.py  -  FairLens AI
Auto-initializes in a background thread on startup.
Exposes /api/status so the browser can poll progress.
"""

import sys
import io
import threading
import traceback
import warnings

warnings.filterwarnings("ignore")

# Prevent Windows cp1252 console from killing the background thread
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

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

# --- Shared state ---------------------------------------------------------
STATE       = {}
INIT_STATUS = "idle"   # idle | running | done | error
INIT_STEP   = "Waiting to start..."
INIT_ERROR  = ""
_lock       = threading.Lock()


# --- Background initializer -----------------------------------------------
def _initialize():
    global STATE, INIT_STATUS, INIT_STEP, INIT_ERROR

    def setstep(msg):
        global INIT_STEP
        with _lock:
            INIT_STEP = msg
        print("[FairLens]", msg)

    try:
        with _lock:
            INIT_STATUS = "running"

        setstep("Generating 100,000 synthetic records...")
        df = generate_dataset(n_samples=100_000)
        X, y, feature_names = encode_features(df)

        setstep("Training biased and fair models...")
        result = train_models(X, y, df, feature_names)

        scaler  = result["scaler"]
        bm      = result["biased_model"]
        fm      = result["fair_model"]
        X_test  = result["X_test"]
        y_test  = result["y_test"]
        df_test = result["df_test"]

        y_pred_biased = bm.predict(X_test)
        y_pred_fair   = fm.predict(X_test)

        setstep("Computing bias metrics...")
        metrics_before = compute_bias_metrics(y_test, y_pred_biased, df_test)
        metrics_after  = compute_bias_metrics(y_test, y_pred_fair,   df_test)

        setstep("Computing SHAP explanations (300 samples)...")
        rng    = np.random.default_rng(42)
        n_shap = min(300, len(X_test))
        idx    = rng.choice(len(X_test), size=n_shap, replace=False)
        X_shap = X_test[idx]

        global_shap_biased = compute_global_shap(bm, X_shap, feature_names)
        global_shap_fair   = compute_global_shap(fm, X_shap, feature_names)

        setstep("Finalising dashboard...")

        with _lock:
            STATE = {
                "df":                 df,
                "feature_names":      feature_names,
                "scaler":             scaler,
                "biased_model":       bm,
                "fair_model":         fm,
                "X_test":             X_test,
                "y_test":             y_test,
                "df_test":            df_test,
                "metrics_before":     metrics_before,
                "metrics_after":      metrics_after,
                "accuracy_biased":    result["accuracy_biased"],
                "accuracy_fair":      result["accuracy_fair"],
                "global_shap_biased": global_shap_biased,
                "global_shap_fair":   global_shap_fair,
                "X_shap_background":  X_shap,
                "dataset_summary":    summarize_dataset(df),
            }
            INIT_STATUS = "done"
            INIT_STEP   = "Ready!"

        print("[FairLens] Initialization complete!")

    except Exception:
        err = traceback.format_exc()
        with _lock:
            INIT_STATUS = "error"
            INIT_ERROR  = err
            INIT_STEP   = "Error - see server console"
        print("[FairLens] INIT ERROR:\n", err)


def start_init():
    t = threading.Thread(target=_initialize, daemon=True)
    t.start()


# Auto-start when the module loads
start_init()


# --- Routes ---------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/status")
def api_status():
    with _lock:
        return jsonify({
            "status": INIT_STATUS,
            "step":   INIT_STEP,
            "error":  INIT_ERROR,
        })


@app.route("/api/initialize", methods=["POST"])
def api_initialize():
    global INIT_ERROR
    with _lock:
        s = INIT_STATUS
    if s == "done":
        return jsonify({"status": "ok", "message": "Already initialized."})
    if s == "running":
        return jsonify({"status": "running", "message": "Already in progress."})
    with _lock:
        INIT_ERROR = ""
    start_init()
    return jsonify({"status": "running", "message": "Started."})


@app.route("/api/overview")
def api_overview():
    with _lock:
        if INIT_STATUS != "done":
            return jsonify({"error": "Not ready"}), 400
        return jsonify(STATE["dataset_summary"])


@app.route("/api/bias")
def api_bias():
    with _lock:
        if INIT_STATUS != "done":
            return jsonify({"error": "Not ready"}), 400
        return jsonify({
            "before":          STATE["metrics_before"],
            "after":           STATE["metrics_after"],
            "accuracy_biased": STATE["accuracy_biased"],
            "accuracy_fair":   STATE["accuracy_fair"],
        })


@app.route("/api/shap/global")
def api_shap_global():
    with _lock:
        if INIT_STATUS != "done":
            return jsonify({"error": "Not ready"}), 400
        return jsonify({
            "biased": STATE["global_shap_biased"],
            "fair":   STATE["global_shap_fair"],
        })


@app.route("/api/predict", methods=["POST"])
def api_predict():
    with _lock:
        if INIT_STATUS != "done":
            return jsonify({"error": "Not ready"}), 400
        scaler = STATE["scaler"]
        bm     = STATE["biased_model"]
        fm     = STATE["fair_model"]
        bg     = STATE["X_shap_background"]
        fnames = STATE["feature_names"]

    body     = request.get_json()
    x_raw    = _parse_input(body)
    x_scaled = scaler.transform(x_raw.reshape(1, -1))[0]

    biased_local = compute_local_shap(bm, bg, x_scaled, fnames)
    fair_local   = compute_local_shap(fm, bg, x_scaled, fnames)

    return jsonify({"biased": biased_local, "fair": fair_local, "inputs": body})


@app.route("/api/whatif", methods=["POST"])
def api_whatif():
    with _lock:
        if INIT_STATUS != "done":
            return jsonify({"error": "Not ready"}), 400
        scaler = STATE["scaler"]
        bm     = STATE["biased_model"]
        fm     = STATE["fair_model"]

    body     = request.get_json()
    x_raw    = _parse_input(body)
    x_scaled = scaler.transform(x_raw.reshape(1, -1))

    pb = float(bm.predict_proba(x_scaled)[0, 1])
    pf = float(fm.predict_proba(x_scaled)[0, 1])

    return jsonify({
        "biased_prob":     round(pb, 4),
        "fair_prob":       round(pf, 4),
        "biased_decision": "Approved" if pb >= 0.5 else "Rejected",
        "fair_decision":   "Approved" if pf >= 0.5 else "Rejected",
        "bias_flag":       pb < 0.5 and pf >= 0.5,
    })


# --- Helper ---------------------------------------------------------------
def _parse_input(body: dict) -> np.ndarray:
    return np.array([
        float(body.get("income",    50000)),
        float(body.get("education", 2)),
        float(body.get("age",       35)),
        1.0 if body.get("gender", "male")  == "female" else 0.0,
        1.0 if body.get("region", "urban") == "rural"  else 0.0,
    ])


if __name__ == "__main__":
    print("=" * 55)
    print("  FairLens AI -  http://localhost:5000")
    print("  Initialization runs automatically in background")
    print("=" * 55)
    app.run(debug=False, port=5000, threaded=True)
