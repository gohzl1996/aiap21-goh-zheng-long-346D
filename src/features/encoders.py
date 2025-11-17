import pandas as pd
from typing import Dict, List, Optional

def ordinal_encode(df: pd.DataFrame, col: str, order: List[str]) -> pd.DataFrame:
    out = df.copy()
    mapping = {v: i for i, v in enumerate(order)}
    out[col] = out[col].astype(str).str.strip().str.lower().map(mapping)
    return out

def one_hot_encode(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    return pd.get_dummies(df, columns=cols, drop_first=False)

def encode_categoricals(
    df: pd.DataFrame,
    categorical_cols: List[str],
    ordinal_maps: Optional[Dict[str, List[str]]] = None
) -> pd.DataFrame:
    out = df.copy()
    ordinal_maps = ordinal_maps or {}
    one_hot_cols = []
    for c in categorical_cols:
        if c in ordinal_maps:
            out = ordinal_encode(out, c, ordinal_maps[c])
        else:
            one_hot_cols.append(c)
    if one_hot_cols:
        out = one_hot_encode(out, one_hot_cols)
    return out
