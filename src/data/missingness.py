import pandas as pd
from typing import List

def add_missing_flags(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        # Treat string "None" as missing-like for categorical sensors
        is_none = out[col].astype(str).str.strip().str.lower().eq("none")
        out[f"is_missing_{col}"] = out[col].isna() | is_none
    return out

def session_missingness(df: pd.DataFrame, session_col: str) -> pd.DataFrame:
    return df.groupby(session_col).apply(lambda g: g.isna().mean())
