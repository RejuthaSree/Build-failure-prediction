"""Flask REST API exposing the build-failure model.

Endpoints
---------
GET  /health           -> liveness probe plus the model path
POST /predict          -> score a single build record
POST /predict/batch     -> score a list of build records

The app uses the *factory* pattern (``create_app``) so tests can inject a model
from a temporary directory instead of relying on a globally loaded one.
"""

from __future__ import annotations

import os

import pandas as pd
from flask import Flask, jsonify, request

from . import config
from .predict import load_default_model, predict_batch, predict_build


def create_app(model_path=config.MODEL_PATH) -> Flask:
    """Build a Flask application bound to the model at ``model_path``."""
    app = Flask(__name__)
    model = load_default_model(model_path)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok", "model": str(model_path)})

    @app.post("/predict")
    def predict_one():
        payload = request.get_json(silent=True)
        if not isinstance(payload, dict):
            return jsonify({"error": "body must be a JSON object"}), 400
        try:
            return jsonify(predict_build(model, payload))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422

    @app.post("/predict/batch")
    def predict_many():
        payload = request.get_json(silent=True)
        if not isinstance(payload, list) or not payload:
            return jsonify({"error": "body must be a non-empty JSON array"}), 400
        try:
            scored = predict_batch(model, pd.DataFrame(payload))
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 422
        return jsonify(scored.to_dict(orient="records"))

    return app


if __name__ == "__main__":  # pragma: no cover - manual entry point
    port = int(os.environ.get("PORT", "5000"))
    create_app().run(host="0.0.0.0", port=port)
