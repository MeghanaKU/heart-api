import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def test_data_loading():
    df = pd.read_csv("data/raw/heart.csv")
    assert df.shape[0] > 0
    assert "target" in df.columns

def test_train_test_split():
    df = pd.read_csv("data/raw/heart.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    assert len(X_train) > 0
    assert len(X_test) > 0

def test_model_training():
    df = pd.read_csv("data/raw/heart.csv")
    X = df.drop("target", axis=1)
    y = df["target"]

    model = LogisticRegression()
    model.fit(X, y)

    preds = model.predict(X)
    assert len(preds) == len(y)

