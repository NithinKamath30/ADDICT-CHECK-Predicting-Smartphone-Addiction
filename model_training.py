import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib
import random
df = pd.read_csv("user_behavior_dataset.csv")
df['Addicted'] = (
    (df['Screen On Time (hours/day)'] > 5) &
    (df['App Usage Time (min/day)'] > 300) &
    (df['Number of Apps Installed'] > 50)
).astype(int)
for i in df.index:
    if random.random() < 0.01:
        df.at[i, 'Addicted'] = 1 - df.at[i, 'Addicted']
features = [
    'App Usage Time (min/day)',
    'Screen On Time (hours/day)',
    'Battery Drain (mAh/day)',
    'Number of Apps Installed',
    'Data Usage (MB/day)',
    'Age'
]
X = df[features]
y = df['Addicted']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
model = RandomForestClassifier(n_estimators=5, max_depth=4, random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
report = classification_report(y_test, y_pred)
print("Model Evaluation Metrics:")
print(f"Accuracy       : {accuracy:.4f}")
print(f"Precision      : {precision:.4f}")
print(f"Recall         : {recall:.4f}")
print(f"F1 Score       : {f1:.4f}")
print("\nClassification Report:")
print(report)
joblib.dump(model, 'model.pkl')
joblib.dump(scaler, 'scaler.pkl')
