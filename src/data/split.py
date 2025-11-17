import pandas as pd
from sklearn.model_selection import train_test_split

def stratified_split(df: pd.DataFrame, target: str, test_size=0.2, random_state=42):
    y = df[target].astype(str).str.strip()
    X = df.drop(columns=[target])
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
