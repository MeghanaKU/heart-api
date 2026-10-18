import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

# Load dataset
df = pd.read_csv(r"C:\Users\Meghana\Downloads\MLOPS\data\raw\heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Start MLflow experiment
mlflow.set_experiment("heart-disease-experiment")

def evaluate_model(model, model_name):
    mlflow.start_run(run_name=model_name)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    auc = roc_auc_score(y_test, preds)

    mlflow.log_param("model_name", model_name)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("roc_auc", auc)

    mlflow.sklearn.log_model(
        model,
        model_name,
        skops_trusted_types=["sklearn.tree._tree.Tree"]
    )

    mlflow.end_run()

# Train Logistic Regression
evaluate_model(LogisticRegression(), "Logistic Regression")

# Train Random Forest
evaluate_model(RandomForestClassifier(), "Random Forest")
