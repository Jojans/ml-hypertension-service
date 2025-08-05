import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import pickle
import json

df = pd.read_csv("hypertension_dataset.csv")
df.dropna(inplace=True)

categorical_cols = ["BP_History", "Medication", "Family_History", "Exercise_Level", "Smoking_Status"]
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

X = df.drop("Has_Hypertension", axis=1)
y = df["Has_Hypertension"]
y = y.map({'No': 0, 'Yes': 1})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1_score": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba)
}

with open("model.pkl", "wb") as f:
    pickle.dump((clf, scaler, X.columns.tolist()), f)


with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Modelo entrenado y guardado como model/model.pkl")
print("Métricas de evaluación:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")