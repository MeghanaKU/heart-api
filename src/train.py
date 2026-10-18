import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from data_loader import load_data
from preprocess import build_preprocess

def train():
    df = load_data()
    X = df.drop("target", axis=1)
    y = df["target"]

    pre = build_preprocess()

    models = {
        "logreg": LogisticRegression(max_iter=1000),
        "rf": RandomForestClassifier(n_estimators=200)
    }

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-exp")

    best_model = None
    best_score = -1

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            pipe = Pipeline([("pre", pre), ("model", model)])
            scores = cross_val_score(pipe, X, y, cv=5, scoring="roc_auc")
            mean_score = scores.mean()

            mlflow.log_param("model_name", name)
            mlflow.log_metric("roc_auc_mean", mean_score)

            if mean_score > best_score:
                best_score = mean_score
                best_model = pipe

    # ⭐ FIX: Fit the best model on full data
    best_model.fit(X, y)

    joblib.dump(best_model, "models/final_model.pkl")
    print("Model saved!")

if __name__ == "__main__":
    train()
