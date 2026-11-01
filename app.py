
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request
from flasgger import Swagger

app = Flask(__name__)

app.config["SWAGGER"] = {
    "title": "Student Result Prediction API",
    "uiversion": 3
}

swagger_config = Swagger.DEFAULT_CONFIG.copy()

swagger_config.update({
    "specs": [
        {
            "endpoint": "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/docs/",
    "swagger_ui_bundle_js":
        "https://unpkg.com/swagger-ui-dist@3/swagger-ui-bundle.js",
    "swagger_ui_standalone_preset_js":
        "https://unpkg.com/swagger-ui-dist@3/swagger-ui-standalone-preset.js",
    "jquery_js":
        "https://cdnjs.cloudflare.com/ajax/libs/jquery/2.2.4/jquery.min.js",
    "swagger_ui_css":
        "https://unpkg.com/swagger-ui-dist@3/swagger-ui.css"
})

swagger = Swagger(app, config=swagger_config)

MODEL_PATH = Path(__file__).resolve().parent / "student_result_model.pkl"

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "student_result_model.pkl was not found. "
            "Run the training pipeline first."
        )

    return joblib.load(MODEL_PATH)


@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
        "version": "2.0",
        "message": "Updated deployment from GitHub Actions and GHCR"
    })


@app.get("/health")
def health_check():
    """
    Health Check
    ---
    tags:
      - System
    responses:
      200:
        description: API is healthy
    """
    return jsonify({
        "status": "healthy",
        "service": "student-result-prediction"
    }), 200


@app.post("/predict")
def predict():
    """
    Predict Student Result
    ---
    tags:
      - Prediction
    consumes:
      - application/json
    parameters:
      - in: body
        name: student
        required: true
        schema:
          type: object
          required:
            - attendance
            - internal_marks
            - assignment_marks
            - previous_score
          properties:
            attendance:
              type: number
              example: 90
            internal_marks:
              type: number
              example: 85
            assignment_marks:
              type: number
              example: 88
            previous_score:
              type: number
              example: 80
    responses:
      200:
        description: Student prediction generated successfully
      400:
        description: Invalid or missing input
      500:
        description: Model file error
    """
    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        return jsonify({
            "error": "Missing required fields",
            "missing_fields": missing_fields
        }), 400

    try:
        sample = pd.DataFrame([{
            feature: float(data[feature])
            for feature in FEATURES
        }])

        model = load_model()
        prediction_code = int(model.predict(sample)[0])

        prediction = "PASS" if prediction_code == 1 else "FAIL"

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except (TypeError, ValueError):
        return jsonify({
            "error": "All input features must be numeric"
        }), 400

    except FileNotFoundError as error:
        return jsonify({
            "error": str(error)
        }), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
