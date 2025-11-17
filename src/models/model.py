from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import RobustScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# -------------------------------
# Custom class-conditional imputer
# -------------------------------
class ClassConditionalImputer(BaseEstimator, TransformerMixin):
    """
    Impute missing values conditionally on the target class.
    For numeric columns: median per class.
    For categorical columns: most frequent per class.
    """

    def __init__(self, target_col, numeric_cols=None, categorical_cols=None):
        self.target_col = target_col
        self.numeric_cols = numeric_cols or []
        self.categorical_cols = categorical_cols or []
        self.fill_values_ = {}

    def fit(self, X, y=None):
        df = pd.DataFrame(X).copy()
        if y is not None:
            df[self.target_col] = y

        self.fill_values_ = {}
        for cls in df[self.target_col].unique():
            cls_df = df[df[self.target_col] == cls]
            self.fill_values_[cls] = {}

            for col in self.numeric_cols:
                self.fill_values_[cls][col] = cls_df[col].median()

            for col in self.categorical_cols:
                mode_val = cls_df[col].mode()
                self.fill_values_[cls][col] = mode_val.iloc[0] if not mode_val.empty else None

        return self

    def transform(self, X):
        df = pd.DataFrame(X).copy()
        for cls, fill_map in self.fill_values_.items():
            mask = df[self.target_col] == cls
            for col, val in fill_map.items():
                df.loc[mask, col] = df.loc[mask, col].fillna(val)
        return df

# -------------------------------
# Preprocessor builder
# -------------------------------
def build_preprocessor(numeric_cols, passthrough_cols, target_col=None):
    to_float = FunctionTransformer(lambda X: X.astype(np.float64), validate=False)

    num_pipe = Pipeline([
        ("to_float", to_float),
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", RobustScaler())
    ])

    pass_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    ct = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_cols),
            ("pass", pass_pipe, passthrough_cols)
        ],
        remainder="drop"
    )

    # If target_col is provided, wrap with class-conditional imputer
    if target_col is not None:
        return Pipeline([
            ("class_cond_imputer", ClassConditionalImputer(
                target_col=target_col,
                numeric_cols=numeric_cols,
                categorical_cols=passthrough_cols
            )),
            ("ct", ct)
        ])
    else:
        return ct

# -------------------------------
# Model builders
# -------------------------------
def build_logreg(ct):
    return Pipeline([
        ("prep", ct),
        ("clf", LogisticRegression(
            max_iter=2000, multi_class="multinomial", class_weight="balanced"
        ))
    ])

def build_mlp(ct):
    return Pipeline([
        ("prep", ct),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            batch_size=256,
            learning_rate_init=1e-3,
            max_iter=300,
            random_state=42
        ))
    ])

def build_xgb(ct, n_classes: int):
    return Pipeline([
        ("prep", ct),
        ("clf", XGBClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            num_class=n_classes,
            reg_lambda=1.0,
            random_state=42,
            tree_method="hist"
        ))
    ])
