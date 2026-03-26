import os
from flask import Flask, request, render_template
import pandas as pd
import pickle
import joblib
import numpy as np

app = Flask(__name__)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "scaler.pkl"))
X_train_columns = pickle.load(open(os.path.join(BASE_DIR, "X_train_columns.pkl"), "rb"))

if model is None or scaler is None or X_train_columns is None:
    raise Exception("Model, scaler, or X_train_columns not found. Please ensure they are in the same directory.")

education_map = {
    'High School': 0,
    'Associate': 1,
    'Bachelor': 2,
    'Master': 3,
    'Doctorate': 4
}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        try:
            data = request.form.to_dict()

            df = pd.DataFrame([data])

            # Preprocessing
            df['person_gender'] = df['person_gender'].map({'female': 0, 'male': 1})
            df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'No': 0, 'Yes': 1})
            df['person_education'] = df['person_education'].map(education_map)

            # Convert numeric columns
            num_cols = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                        'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']
            for col in num_cols:
                df[col] = pd.to_numeric(df[col])

            # Manual one-hot encoding for home_ownership and loan_intent
            home_val = df['person_home_ownership'].iloc[0]
            intent_val = df['loan_intent'].iloc[0]
            df = df.drop(columns=['person_home_ownership', 'loan_intent'], errors='ignore')

            # Get scaler's feature names
            scaler_cols = list(scaler.feature_names_in_)

            # Build the scaler input (15 columns: numerics + defaults + home_ownership one-hot)
            scaler_row = {}
            for col in scaler_cols:
                if col in df.columns:
                    scaler_row[col] = float(df[col].iloc[0])
                elif col == f'person_home_ownership_{home_val}':
                    scaler_row[col] = 1.0
                else:
                    scaler_row[col] = 0.0

            df_scaler = pd.DataFrame([scaler_row], columns=scaler_cols)

            # Scale the 15 features
            df_scaled = pd.DataFrame(scaler.transform(df_scaler), columns=scaler_cols)

            # Now add loan_intent one-hot columns (unscaled, as they were not in the scaler)
            for col in X_train_columns:
                if col.startswith('loan_intent_'):
                    df_scaled[col] = 1.0 if col == f'loan_intent_{intent_val}' else 0.0

            # Ensure final column order matches training
            df_final = df_scaled[X_train_columns]

            # Predict
            pred = int(model.predict(df_final)[0])
            prob = float(model.predict_proba(df_final)[0][1])

            result_text = "Approved ✅" if pred == 0 else "Rejected ❌"
            return render_template("index.html", prediction=result_text, prob=round(prob * 100, 2))

        except Exception as e:
            return render_template("index.html", prediction=None, error=str(e))

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)