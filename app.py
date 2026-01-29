from flask import Flask, render_template, request
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

app = Flask(__name__)

# Render compatibility: Port configuration
PORT = int(os.environ.get("PORT", 5000))

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"

@app.route("/", methods=["GET", "POST"])
def home():
    results = None
    patient_result = None

    # CSV UPLOAD & TRAIN logic
    if request.method == "POST" and "csv" in request.files:
        file = request.files["csv"]
        if file.filename != '':
            path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(path)

            df = pd.read_csv(path)
            df.columns = df.columns.str.strip()

            le = LabelEncoder()
            # Safety check: Ensuring columns exist
            if "GENDER" in df.columns and "LUNG_CANCER" in df.columns:
                df["GENDER"] = le.fit_transform(df["GENDER"])
                df["LUNG_CANCER"] = le.fit_transform(df["LUNG_CANCER"])

                X = df.drop("LUNG_CANCER", axis=1)
                y = df["LUNG_CANCER"]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.3, random_state=42, stratify=y
                )

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                lr = LogisticRegression(max_iter=2000)
                lr.fit(X_train, y_train)
                lr_pred = lr.predict(X_test)

                rf = RandomForestClassifier(n_estimators=100, random_state=42)
                rf.fit(X_train, y_train)
                rf_pred = rf.predict(X_test)

                lr_acc = round(accuracy_score(y_test, lr_pred) * 100, 2)
                rf_acc = round(accuracy_score(y_test, rf_pred) * 100, 2)
                cmatrix = confusion_matrix(y_test, lr_pred).tolist()

                joblib.dump(lr, MODEL_PATH)
                joblib.dump(scaler, SCALER_PATH)

                results = {"lr_acc": lr_acc, "rf_acc": rf_acc, "cm": cmatrix}

    # PATIENT TEST logic
    if request.method == "POST" and "gender" in request.form:
        if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
            model = joblib.load(MODEL_PATH)
            scaler = joblib.load(SCALER_PATH)

            gender_val = int(request.form["gender"])
            age = int(request.form["age"])
            symptoms = [int(request.form[s]) for s in [
                "smoking", "yellow_fingers", "anxiety", "peer_pressure", 
                "chronic_disease", "fatigue", "allergy", "wheezing", 
                "alcohol", "coughing", "breath", "swallow", "chest_pain"
            ]]

            data = [gender_val, age] + symptoms
            features = scaler.transform([data])

            prob = model.predict_proba(features)[0][1]
            percentage = round(prob * 100, 2)
            gender = "Male" if gender_val == 1 else "Female"

            risk = "Low" if percentage < 30 else "Medium" if percentage < 60 else "High"
            
            score = sum(symptoms)
            stage = 1 if risk == "Low" or score <= 10 else 2 if score <= 18 else 3 if score <= 25 else 4

            patient_result = {
                "gender": gender, "percentage": percentage, 
                "risk": risk, "stage": stage
            }

    return render_template("index.html", results=results, patient_result=patient_result)

if __name__ == "__main__":
    # Render ke liye host '0.0.0.0' aur dynamic port zaroori hai
    app.run(host='0.0.0.0', port=PORT)
