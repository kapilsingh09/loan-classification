import os
import logging
import time
from collections import defaultdict
from functools import wraps
from datetime import datetime

from flask import Flask, request, render_template, jsonify, abort
import pandas as pd
import pickle
import joblib
import numpy as np

# ─────────────────────────────────────────────
# App Setup
# ─────────────────────────────────────────────
app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-me-in-production")

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("predictions.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Load Models (fail fast on startup)
# ─────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
    scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
    X_train_columns = pickle.load(open(os.path.join(BASE_DIR, "X_train_columns.pkl"), "rb"))
    logger.info("✅ Models loaded successfully.")
except Exception as e:
    logger.critical(f"❌ Failed to load models: {e}")
    raise SystemExit(1)

# ─────────────────────────────────────────────
# Constants & Validation Rules
# ─────────────────────────────────────────────
EDUCATION_MAP = {
    "High School": 0,
    "Associate": 1,
    "Bachelor": 2,
    "Master": 3,
    "Doctorate": 4,
}

VALID_GENDERS         = {"male", "female"}
VALID_HOME_OWNERSHIP  = {"RENT", "OWN", "MORTGAGE", "OTHER"}
VALID_LOAN_INTENT     = {"PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"}
VALID_DEFAULTS        = {"No", "Yes"}

FIELD_LIMITS = {
    "person_age":                    (18,  100),
    "person_income":                 (1_000, 10_000_000),
    "person_emp_exp":                (0,   60),
    "loan_amnt":                     (500, 1_000_000),
    "loan_int_rate":                 (1,   40),
    "loan_percent_income":           (0.01, 1.0),
    "cb_person_cred_hist_length":    (0,   60),
    "credit_score":                  (300, 850),
}

REQUIRED_FIELDS = [
    "person_age", "person_gender", "person_education", "person_income",
    "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income",
    "cb_person_cred_hist_length", "credit_score", "previous_loan_defaults_on_file",
    "person_home_ownership", "loan_intent",
]

# ─────────────────────────────────────────────
# In-Memory Rate Limiter
# ─────────────────────────────────────────────
# Stores: { ip: [timestamp, timestamp, ...] }
_rate_store: dict = defaultdict(list)

RATE_LIMIT_REQUESTS = int(os.environ.get("RATE_LIMIT_REQUESTS", 10))   # max requests
RATE_LIMIT_WINDOW   = int(os.environ.get("RATE_LIMIT_WINDOW",   60))   # per N seconds

def is_rate_limited(ip: str) -> tuple[bool, int]:
    """Returns (limited: bool, retry_after_seconds: int)."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    # Prune old timestamps
    _rate_store[ip] = [t for t in _rate_store[ip] if t > window_start]
    if len(_rate_store[ip]) >= RATE_LIMIT_REQUESTS:
        oldest = _rate_store[ip][0]
        retry_after = int(RATE_LIMIT_WINDOW - (now - oldest)) + 1
        return True, retry_after
    _rate_store[ip].append(now)
    return False, 0

def rate_limit(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        ip = request.headers.get("X-Forwarded-For", request.remote_addr)
        ip = ip.split(",")[0].strip()          # handle proxy chains
        limited, retry_after = is_rate_limited(ip)
        if limited:
            logger.warning(f"Rate limit hit — IP: {ip}")
            if request.method == "POST" and request.is_json:
                return jsonify(error=f"Too many requests. Retry after {retry_after}s."), 429
            return render_template(
                "index.html",
                prediction=None,
                error=f"⚠️ Too many requests. Please wait {retry_after} seconds and try again.",
            ), 429
        return f(*args, **kwargs)
    return decorated

# ─────────────────────────────────────────────
# Input Validator
# ─────────────────────────────────────────────
def validate_input(data: dict) -> list[str]:
    """Returns a list of error strings (empty = valid)."""
    errors = []

    # 1. Required fields
    for field in REQUIRED_FIELDS:
        if field not in data or str(data[field]).strip() == "":
            errors.append(f"Missing required field: '{field}'.")

    if errors:          # no point continuing if fields are missing
        return errors

    # 2. Categorical checks
    if data["person_gender"] not in VALID_GENDERS:
        errors.append(f"Invalid gender: '{data['person_gender']}'.")

    if data["person_education"] not in EDUCATION_MAP:
        errors.append(f"Invalid education level: '{data['person_education']}'.")

    if data["person_home_ownership"] not in VALID_HOME_OWNERSHIP:
        errors.append(f"Invalid home ownership: '{data['person_home_ownership']}'.")

    if data["loan_intent"] not in VALID_LOAN_INTENT:
        errors.append(f"Invalid loan intent: '{data['loan_intent']}'.")

    if data["previous_loan_defaults_on_file"] not in VALID_DEFAULTS:
        errors.append(f"Invalid value for previous_loan_defaults_on_file.")

    # 3. Numeric range checks
    for field, (lo, hi) in FIELD_LIMITS.items():
        raw = data.get(field, "")
        try:
            val = float(raw)
        except (ValueError, TypeError):
            errors.append(f"'{field}' must be a number.")
            continue
        if not (lo <= val <= hi):
            errors.append(f"'{field}' must be between {lo} and {hi} (got {val}).")

    # 4. Business logic checks
    try:
        age = float(data["person_age"])
        exp = float(data["person_emp_exp"])
        if exp >= age:
            errors.append("Employment experience cannot be greater than or equal to age.")
    except (ValueError, TypeError):
        pass    # already caught above

    return errors

# ─────────────────────────────────────────────
# Security Headers (added to every response)
# ─────────────────────────────────────────────
@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"]  = "nosniff"
    response.headers["X-Frame-Options"]         = "DENY"
    response.headers["X-XSS-Protection"]        = "1; mode=block"
    response.headers["Referrer-Policy"]         = "strict-origin-when-cross-origin"
    response.headers["Cache-Control"]           = "no-store"
    return response

# ─────────────────────────────────────────────
# Prediction Helper
# ─────────────────────────────────────────────
def run_prediction(data: dict) -> tuple[str, float]:
    """
    Preprocesses form data, runs the model, and returns
    (result_text, probability_percent).
    """
    df = pd.DataFrame([data])

    df["person_gender"]                     = df["person_gender"].map({"female": 0, "male": 1})
    df["previous_loan_defaults_on_file"]    = df["previous_loan_defaults_on_file"].map({"No": 0, "Yes": 1})
    df["person_education"]                  = df["person_education"].map(EDUCATION_MAP)

    num_cols = [
        "person_age", "person_income", "person_emp_exp", "loan_amnt",
        "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score",
    ]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col])

    home_val   = df["person_home_ownership"].iloc[0]
    intent_val = df["loan_intent"].iloc[0]
    df = df.drop(columns=["person_home_ownership", "loan_intent"], errors="ignore")

    scaler_cols = list(scaler.feature_names_in_)
    scaler_row  = {}
    for col in scaler_cols:
        if col in df.columns:
            scaler_row[col] = float(df[col].iloc[0])
        elif col == f"person_home_ownership_{home_val}":
            scaler_row[col] = 1.0
        else:
            scaler_row[col] = 0.0

    df_scaler  = pd.DataFrame([scaler_row], columns=scaler_cols)
    df_scaled  = pd.DataFrame(scaler.transform(df_scaler), columns=scaler_cols)

    for col in X_train_columns:
        if col.startswith("loan_intent_"):
            df_scaled[col] = 1.0 if col == f"loan_intent_{intent_val}" else 0.0

    df_final = df_scaled[X_train_columns]

    pred = int(model.predict(df_final)[0])
    prob = float(model.predict_proba(df_final)[0][1])

    result_text = "Approved ✅" if pred == 0 else "Rejected ❌"
    return result_text, round(prob * 100, 2)

# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.route("/", methods=["GET", "POST"])
@rate_limit
def home():
    if request.method == "POST":
        data = request.form.to_dict()

        # ── Validate ──────────────────────────
        errors = validate_input(data)
        if errors:
            logger.warning(f"Validation failed: {errors}")
            return render_template(
                "index.html",
                prediction=None,
                error=" | ".join(errors),
            )

        # ── Predict ───────────────────────────
        try:
            result_text, prob = run_prediction(data)

            logger.info(
                f"Prediction made — result={result_text!r} prob={prob}% "
                f"ip={request.headers.get('X-Forwarded-For', request.remote_addr)} "
                f"ts={datetime.utcnow().isoformat()}"
            )

            return render_template("index.html", prediction=result_text, prob=prob)

        except Exception as e:
            logger.error(f"Prediction error: {e}", exc_info=True)
            return render_template(
                "index.html",
                prediction=None,
                error="An internal error occurred. Please try again later.",
            )

    return render_template("index.html", prediction=None)


# ─────────────────────────────────────────────
# Health Check (for load balancers / uptime monitors)
# ─────────────────────────────────────────────
@app.route("/health", methods=["GET"])
def health():
    return jsonify(status="ok", timestamp=datetime.utcnow().isoformat()), 200


# ─────────────────────────────────────────────
# Error Handlers
# ─────────────────────────────────────────────
@app.errorhandler(404)
def not_found(e):
    return render_template("index.html", prediction=None, error="Page not found."), 404

@app.errorhandler(405)
def method_not_allowed(e):
    return render_template("index.html", prediction=None, error="Method not allowed."), 405

@app.errorhandler(500)
def server_error(e):
    logger.error(f"500 error: {e}")
    return render_template("index.html", prediction=None, error="Internal server error."), 500


# ─────────────────────────────────────────────
# Entry Point  (dev only — use gunicorn in prod)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    # Never run debug=True in production!
    debug_mode = os.environ.get("FLASK_DEBUG", "false").lower() == "true"
    app.run(debug=debug_mode, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))