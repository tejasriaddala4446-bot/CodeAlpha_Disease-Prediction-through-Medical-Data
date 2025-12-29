# =========================================
# Diabetes Disease Prediction
# Clean & Warning-Free Code
# =========================================

import pandas as pd
import numpy as np
from io import StringIO
import warnings

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Ignore warnings (to keep output clean)
warnings.filterwarnings("ignore")

# -----------------------------
# 1. Embedded Dataset
# -----------------------------
data_csv = """Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
6,148,72,35,0,33.6,0.627,50,1
1,85,66,29,0,26.6,0.351,31,0
8,183,64,0,0,23.3,0.672,32,1
1,89,66,23,94,28.1,0.167,21,0
0,137,40,35,168,43.1,2.288,33,1
5,116,74,0,0,25.6,0.201,30,0
3,78,50,32,88,31.0,0.248,26,1
10,115,0,0,0,35.3,0.134,29,0
2,197,70,45,543,30.5,0.158,53,1
8,125,96,0,0,32.0,0.232,54,1
"""

data = pd.read_csv(StringIO(data_csv))
print("Dataset loaded successfully!\n")

# -----------------------------
# 2. Data Cleaning
# -----------------------------
columns_with_zero = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in columns_with_zero:
    data[col] = data[col].replace(0, data[col].mean())

# -----------------------------
# 3. Features & Target
# -----------------------------
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# -----------------------------
# 4. Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1
)

# -----------------------------
# 5. Train Model
# -----------------------------
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

print("Model trained successfully!\n")

# -----------------------------
# 6. Evaluation
# -----------------------------
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred, zero_division=0))

# -----------------------------
# 7. New Patient Prediction
# -----------------------------
new_patient = np.array([[2, 120, 70, 25, 100, 28.5, 0.5, 32]])
result = model.predict(new_patient)

print("\nNew Patient Result:")
if result[0] == 1:
    print("⚠️ Diabetic (Disease Detected)")
else:
    print("✅ Non-Diabetic (No Disease)")