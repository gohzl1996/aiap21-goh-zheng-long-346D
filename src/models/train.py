import pandas as pd
from sklearn.utils.class_weight import compute_sample_weight

def compute_sample_weights(y):
    """
    Compute balanced sample weights for a target vector y.
    Works with both pandas Series and numpy arrays.
    """
    return compute_sample_weight(class_weight="balanced", y=y)

def fit_model(model, X_train: pd.DataFrame, y_train, model_name: str, sw=None):
    """
    Fit a model with optional sample weights.
    - LogisticRegression: uses class_weight internally, so no sample_weight needed.
    - XGBClassifier: supports sample_weight.
    - MLPClassifier: does NOT support sample_weight.
    """

    # Compute sample weights if requested
    if sw is None and model_name == "xgb":
        sw = compute_sample_weights(y_train)

    try:
        if sw is not None and model_name == "xgb":
            # Only XGB supports sample_weight
            return model.fit(X_train, y_train, clf__sample_weight=sw)
        else:
            # LogisticRegression and MLP just fit normally
            return model.fit(X_train, y_train)
    except TypeError:
        # Fallback in case sample_weight sneaks into unsupported models
        return model.fit(X_train, y_train)
