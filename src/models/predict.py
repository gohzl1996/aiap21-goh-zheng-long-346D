import pandas as pd

def predict(model, X: pd.DataFrame):
    return model.predict(X)

def predict_proba(model, X: pd.DataFrame):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)
    return None
