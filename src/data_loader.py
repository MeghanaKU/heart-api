import pandas as pd

def load_data(path="data/raw/heart.csv"):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":
    print(load_data().head())
