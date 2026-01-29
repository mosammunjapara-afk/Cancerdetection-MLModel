import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load dataset
df = pd.read_csv("survey lung cancer.csv")
df.columns = df.columns.str.strip()

# Encode categorical columns
le = LabelEncoder()
df['GENDER'] = le.fit_transform(df['GENDER'])        # M/F → 1/0
df['LUNG_CANCER'] = le.fit_transform(df['LUNG_CANCER'])  # YES/NO → 1/0

# Split features & target
X = df.drop('LUNG_CANCER', axis=1)
y = df['LUNG_CANCER']

# 🔀 Train-Test Split (30% train, 70% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.7,
    random_state=42,
    stratify=y
)

# ⚖️ Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 🤖 Train model
model = LogisticRegression(max_iter=2000)
model.fit(X_train_scaled, y_train)

# 📊 Test accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("✅ Model Trained Successfully")
print("🎯 Accuracy:", round(accuracy * 100, 2), "%")
print("\n📄 Classification Report:\n", classification_report(y_test, y_pred))

# 💾 Save model & scaler
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("💾 model.pkl & scaler.pkl saved")
