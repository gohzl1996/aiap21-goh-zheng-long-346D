from src.logging_config import configure_logging
from src import config
from src.data.load_data import load_sqlite
from src.data.dedupe import report_duplicates, drop_full_duplicates
from src.data.missingness import add_missing_flags
from src.data.split import stratified_split
from src.features.build_features import normalize_target, interaction_features
from src.features.outliers import add_outlier_flags_zscore
from src.features.encoders import encode_categoricals
from src.models.model import build_preprocessor, build_logreg, build_mlp, build_xgb
from src.models.train import fit_model
from src.models.predict import predict
from src.models.evaluate import evaluate
from src.utils.io import save_json
from src.utils.metrics import macro_f1, per_class_f1
from sklearn.preprocessing import LabelEncoder


def run_pipeline(db_path=None, table_name=None):
    log = configure_logging()
    db_path = db_path or config.DB_PATH
    table_name = table_name or config.TABLE_NAME

    # Load
    df = load_sqlite(db_path, table_name)
    log.info(f"Loaded shape: {df.shape}")

    # Dedupe
    dup_count = report_duplicates(df)
    df = drop_full_duplicates(df)
    log.info(f"Dropped {dup_count} exact duplicates. New shape: {df.shape}")

    # Normalize target labels
    df = normalize_target(df, config.TARGET_COLUMN)

    # Add missingness flags (incl 'None' semantics)
    df = add_missing_flags(df, config.MISSINGNESS_FLAG_COLUMNS)

    # Interaction features (non-destructive)
    df = interaction_features(df)

    # Outlier flags (Z-score based)
    df = add_outlier_flags_zscore(df, config.OUTLIER_COLUMNS_Z, z=3.0)

    # Encode categoricals (ordinal where specified; otherwise one-hot)
    df = encode_categoricals(df, config.CATEGORICAL_COLUMNS, ordinal_maps=config.ORDINAL_MAPS)

    # Prepare feature sets
    derived_flags = [c for c in df.columns if c.startswith("is_missing_") or c.startswith("is_outlier_")]
    one_hot_cats = [c for c in df.columns if any(prefix in c for prefix in ["CO_GasSensor_", "HVAC Operation Mode_", "Ambient Light Level_"])]
    ordinal_cols = [c for c in config.ORDINAL_MAPS.keys() if c in df.columns]  # already numeric after ordinal encode

    passthrough_cols = derived_flags + one_hot_cats + ordinal_cols + [config.SESSION_COLUMN]
    numeric_to_scale = [c for c in config.NUMERIC_COLUMNS + ["CO2_ratio", "MOx_1x4"] if c in df.columns]

    # Select columns + target for split
    selected_cols = numeric_to_scale + passthrough_cols + [config.TARGET_COLUMN]
    df_sel = df[selected_cols].copy()

    # Build common preprocessor
    preproc = build_preprocessor(numeric_to_scale, passthrough_cols)

    # Stratified split
    X_train, X_test, y_train, y_test = stratified_split(df_sel, config.TARGET_COLUMN, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)

    # Build models
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)
    n_classes = len(le.classes_)
    models = {
        "logreg": build_logreg(preproc),
        "mlp": build_mlp(preproc),
        "xgb": build_xgb(preproc, n_classes)
    }

    comparison = {}
    for name, model in models.items():
        log.info(f"Training {name}...")
        fitted = fit_model(model, X_train, y_train, model_name=name)
        y_pred = predict(fitted, X_test)
        results = evaluate(y_test, y_pred)
        comparison[name] = {
            "macro_f1": macro_f1(results["report"]),
            "per_class_f1": per_class_f1(results["report"]),
            "confusion_matrix": results["confusion_matrix"].tolist()
        }

    save_json(comparison, "artifacts/model_comparison.json")
    log.info("Model comparison complete.")
    return comparison

if __name__ == "__main__":
    from src.cli import build_parser

    args = build_parser().parse_args()
    run_pipeline(db_path=args.db_path, table_name=args.table)
