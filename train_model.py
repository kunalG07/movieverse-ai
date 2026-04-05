import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------------
# LOAD TRAIN DATA ONLY
# -----------------------------
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv').squeeze()

print("✅ Training data loaded")

# -----------------------------
# ENCODE TARGET (ONLY TRAIN)
# -----------------------------
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)

# -----------------------------
# TRAIN MODEL (NO TEST DATA)
# -----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42
)

model.fit(X_train, y_train_encoded)

print("✅ Model trained using ONLY training data")

# -----------------------------
# SAVE MODEL & ENCODER
# -----------------------------
joblib.dump(model, 'model.pkl')
joblib.dump(le, 'encoder.pkl')

print("✅ Model & Encoder saved successfully!")