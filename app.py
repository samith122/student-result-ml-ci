
from pathlib import Path
import logging
import sys
import time
import uuid

import joblib
import pandas as pd
from flask import Flask, jsonify, request, g
from flasgger import Swagger


# --------------------------------------------------
# 1. CONFIGURE LOGGING
# --------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)

logger = logging.getLogger("student-result-api")


# --------------------------------------------------
# 2. FLASK APPLICATION
# --------------------------------------------------

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


# --------------------------------------------------
# 3. MODEL CONFIGURATION
# --------------------------------------------------

MODEL_PATH = Path(__file__).resolve().parent / "student_result_model.pkl"

FEATURES = [
    "attendance",
    "internal_marks",
    "assignment_marks",
    "previous_score"
]


def load_model():
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "student_result_model.pkl was not found. "
                "Run the training pipeline first."
            )

        model = joblib.load(MODEL_PATH)

        logger.info(
            "event=model_loaded model_file=%s",
            MODEL_PATH.name
        )

        return model

    except Exception:
        logger.exception(
            "event=model_load_failed model_file=%s",
            MODEL_PATH.name
        )
        raise


# --------------------------------------------------
# 4. REQUEST LOGGING MIDDLEWARE
# --------------------------------------------------

@app.before_request
def before_request_logging():
    g.request_id = request.headers.get(
        "X-Request-ID",
        str(uuid.uuid4())
    )

    g.start_time = time.perf_counter()


@app.after_request
def after_request_logging(response):
    duration_ms = (
        time.perf_counter() - g.start_time
    ) * 1000

    response.headers["X-Request-ID"] = g.request_id

    log_level = (
        logging.WARNING
        if response.status_code >= 400
        else logging.INFO
    )

    logger.log(
        log_level,
        "event=http_request request_id=%s method=%s path=%s "
        "status=%s duration_ms=%.2f",
        g.request_id,
        request.method,
        request.path,
        response.status_code,
        duration_ms
    )

    return response


# --------------------------------------------------
# 5. HOME ENDPOINT
# --------------------------------------------------

@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "service": "student-result-prediction",
        "version": "2.0",
        "message": "Updated deployment from GitHub Actions and GHCR"
    })


# --------------------------------------------------
# 6. HEALTH CHECK
# --------------------------------------------------

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


# --------------------------------------------------
# 7. STUDENT RESULT PREDICTION
# --------------------------------------------------

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
        description: Model file or prediction error
    """

    logger.info(
        "event=prediction_request request_id=%s feature_names=%s",
        g.request_id,
        FEATURES
    )

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        logger.warning(
            "event=invalid_json request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "JSON request body is required"
        }), 400

    missing_fields = [
        feature for feature in FEATURES
        if feature not in data
    ]

    if missing_fields:
        logger.warning(
            "event=missing_fields request_id=%s fields=%s",
            g.request_id,
            missing_fields
        )

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

        prediction = (
            "PASS" if prediction_code == 1 else "FAIL"
        )

        logger.info(
            "event=prediction_success request_id=%s prediction=%s",
            g.request_id,
            prediction
        )

        return jsonify({
            "prediction": prediction,
            "prediction_code": prediction_code
        }), 200

    except (TypeError, ValueError):
        logger.warning(
            "event=invalid_numeric_input request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "All input features must be numeric"
        }), 400

    except FileNotFoundError:
        logger.exception(
            "event=prediction_failed request_id=%s reason=model_not_found",
            g.request_id
        )

        return jsonify({
            "error": "Model file unavailable"
        }), 500

    except Exception:
        logger.exception(
            "event=prediction_failed request_id=%s",
            g.request_id
        )

        return jsonify({
            "error": "Prediction failed"
        }), 500


# --------------------------------------------------
# 8. APPLICATION STARTUP
# --------------------------------------------------

if __name__ == "__main__":
    logger.info("event=application_starting")

    app.run(
        host="0.0.0.0",
        port=5000
    )
