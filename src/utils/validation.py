import pandas as pd

def assert_required_columns(df: pd.DataFrame, required_cols):
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
