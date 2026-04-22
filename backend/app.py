"""
backend/app.py  –  Flask REST API for Prescription Error Detection
Endpoint : POST /predict
           GET  /health
"""

import os
import sys
import json
import logging

from flask import Flask, request, jsonify
from flask_cors import CORS

# ── make sure we can import from the project root ──────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from utils.predictor import PrescriptionPredictor

# ─────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)   # allow frontend (any origin) to call the API

MODEL_DIR = os.path.join(ROOT, "model")

# Load model once at startup
logger.info("Loading model from %s …", MODEL_DIR)
predictor = PrescriptionPredictor(MODEL_DIR)
logger.info("Model loaded ✓")


# ─────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Simple health-check."""
    return jsonify({"status": "ok", "model": "DistilBERT prescription classifier"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body  : { "text": "<prescription text>" }
    Returns:
    {
      "label"      : "safe" | "overdose" | "drug_interaction",
      "confidence" : 0.95,
      "probabilities": { "safe": 0.05, "overdose": 0.02, "drug_interaction": 0.93 },
      "explanation": "...",
      "input_text" : "..."
    }
    """
    data = request.get_json(silent=True)

    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field in JSON body"}), 400

    text = str(data["text"]).strip()
    if not text:
        return jsonify({"error": "Prescription text cannot be empty"}), 400

    if len(text) > 1000:
        return jsonify({"error": "Text too long (max 1000 characters)"}), 400

    try:
        result = predictor.predict(text)
        result["input_text"] = text
        logger.info("Prediction: %s  (conf %.3f)  for: %s",
                    result["label"], result["confidence"], text[:60])
        return jsonify(result)

    except Exception as e:
        logger.exception("Prediction failed")
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


@app.errorhandler(404)
def not_found(_):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(405)
def method_not_allowed(_):
    return jsonify({"error": "Method not allowed"}), 405


# ─────────────────────────────────────────────
#  Entry-point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info("Starting Flask server on http://localhost:%d", port)
    app.run(host="0.0.0.0", port=port, debug=False)
