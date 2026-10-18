from src.preprocess import build_preprocess
import pandas as pd

def test_preprocess():
    df = pd.DataFrame({
        "age": [50, 60],
        "trestbps": [130, 140],
        "chol": [250, 260],
        "thalach": [150, 160],
        "oldpeak": [2.3, 1.4],
        "sex": [1, 0],
        "cp": [3, 2],
        "fbs": [0, 1],
        "restecg": [0, 1],
        "exang": [0, 1],
        "slope": [2, 1],
        "ca": [0, 2],
        "thal": [2, 3],
    })
    pre = build_preprocess()
    out = pre.fit_transform(df)
    assert out.shape[0] == 2
