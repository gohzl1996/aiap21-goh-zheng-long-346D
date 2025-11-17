import pandas as pd

def report_duplicates(df: pd.DataFrame) -> int:
    return int(df.duplicated().sum())

def drop_full_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop_duplicates(keep="first").copy()