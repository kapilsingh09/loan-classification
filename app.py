import os
from flask import Flask, request, render_template
import pandas as pd
import pickle
import joblib

app = Flask(__name__)

# Load models
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
scaler = joblib.load(open(os.path.join(BASE_DIR, "scaler.pkl"), "rb"))
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
        data = request.form.to_dict()

        df = pd.DataFrame([data])

        # 🔹 Preprocessing
        df['person_gender'] = df['person_gender'].map({'female':0, 'male':1})
        df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'No':0, 'Yes':1})
        df['person_education'] = df['person_education'].map(education_map)

        # Convert numeric
        num_cols = ['person_age','person_income','person_emp_exp','loan_amnt',
                    'loan_int_rate','loan_percent_income','cb_person_cred_hist_length','credit_score']
        for col in num_cols:
            df[col] = pd.to_numeric(df[col])

        # One-hot
        df = pd.get_dummies(df, columns=['person_home_ownership', 'loan_intent'])

        # Align
        for col in X_train_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[X_train_columns]

        # Scale
        df = pd.DataFrame(scaler.transform(df), columns=X_train_columns)

        # Predict
        pred = model.predict(df)[0]
        prob = model.predict_proba(df)[0][1]

        return render_template("index.html", prediction=pred, prob=round(prob*100,2))

    return render_template("index.html", prediction=None)

if __name__ == "__main__":
    app.run(debug=True)