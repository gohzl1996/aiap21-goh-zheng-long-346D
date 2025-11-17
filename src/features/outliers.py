import pandas as pd
from typing import List

def add_outlier_flags_zscore(df: pd.DataFrame, cols: List[str], z=3.0) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            mu = out[c].mean()
            sd = out[c].std(ddof=1)
            out[f"is_outlier_{c}"] = ((out[c] - mu).abs() / (sd + 1e-9)) > z
    return out
