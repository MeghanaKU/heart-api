from src.data_loader import load_data

def test_data_loader():
    df = load_data()
    assert df.shape[0] > 0
    assert "target" in df.columns
