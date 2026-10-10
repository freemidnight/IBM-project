"""
E-Commerce Sales Analytics - Flask Backend API
Serves model predictions and dataset statistics.
"""

from flask import Flask, request, jsonify
import os
import json
import joblib
import numpy as np

app = Flask(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# ── Load artifacts at startup ──────────────────────────────
model    = joblib.load(os.path.join(MODELS_DIR, "best_model.pkl"))
le_dict  = joblib.load(os.path.join(MODELS_DIR, "label_encoders.pkl"))

with open(os.path.join(MODELS_DIR, "feature_meta.json")) as f:
    meta = json.load(f)

with open(os.path.join(MODELS_DIR, "dataset_stats.json")) as f:
    dataset_stats = json.load(f)

FEATURES = meta["features"]

CAT_DEFAULTS = {
    "sales_channel":    "Website",
    "payment_method":   "Credit Card",
    "shipping_method":  "Standard",
    "customer_segment": "Consumer",
    "region":           "North",
    "order_status":     "Completed",
    "return_status":    "No Return",
}


def encode_feature(col, val):
    """Encode a categorical feature using the stored LabelEncoder."""
    if col in le_dict:
        le = le_dict[col]
        try:
            return int(le.transform([str(val)])[0])
        except ValueError:
            # Unseen label → use most frequent class (index 0)
            return 0
    return val


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "service": "E-Commerce Sales Analytics API",
        "version": "1.0",
        "endpoints": ["/predict", "/model_info", "/dataset_stats", "/health"]
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": meta["best_model"]})


@app.route("/model_info", methods=["GET"])
def model_info():
    return jsonify({
        "best_model":  meta["best_model"],
        "target":      meta["target"],
        "features":    FEATURES,
        "rf_metrics":  meta["rf_metrics"],
        "gb_metrics":  meta["gb_metrics"],
    })


@app.route("/dataset_stats", methods=["GET"])
def dataset_statistics():
    return jsonify(dataset_stats)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict net_sales given order features.
    Accepts JSON body with any subset of FEATURES; missing ones use defaults.
    """
    body = request.get_json(force=True, silent=True) or {}

    row = []
    for feat in FEATURES:
        if feat in body:
            val = body[feat]
        elif feat in CAT_DEFAULTS:
            val = CAT_DEFAULTS[feat]
        else:
            val = 0

        val = encode_feature(feat, val)
        row.append(float(val))

    prediction = float(model.predict([row])[0])
    # Estimate profit (rough: ~43% margin based on dataset average)
    est_profit = round(prediction * 0.43, 2)

    return jsonify({
        "predicted_net_sales": round(prediction, 2),
        "estimated_profit":    est_profit,
        "model_used":          meta["best_model"],
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
