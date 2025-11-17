# Elderguard Analytics ML Study
```bash
Name: Goh Zheng Long 
Email: goh.zl1996@gmail.com
```

# Folder Structure Overview
```bash
aiap21-goh-zheng-long-346D/
├── artifacts/
│   └── model_comparison.json    # Evaluation and Comparison between models
├── data/
│   └── gas_monitoring.db        # [Not committed] SQLite DB – must be added manually
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── logging_config.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── load_data.py
│   │   ├── split.py
│   │   ├── dedupe.py
│   │   └── missingness.py
│   ├── features/
│   │   ├── __init__.py
│   │   ├── build_features.py
│   │   ├── outliers.py
│   │   └── encoders.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── evaluate.py
│   └── utils/
│       ├── __init__.py
│       ├── io.py
│       ├── metrics.py
│       └── validation.py
├── cli.py
└── main.py
├── eda.ipynb                    # Jupyter Notebook codes with EDA
├── run.sh                       # Shell script to run the pipeline
├── requirements.txt             # Python dependencies
└── README.md                    # Project overview and instructions
```

# Pipeline instructions
```bash
# 1. Ensure Database Placement
# Place the database file in the correct location:
data/gas_monitoring.db            # Do NOT commit this file to GitHub

# 2. Set Up Virtual Environment
python -m venv .venv
source .venv/bin/activate         # On Windows: .venv\Scripts\activate

# 3. Install Dependencies
pip install -r requirements.txt

# 4. Run the Pipeline
bash run.sh                       # or use ./run.sh on WSL

# Additional Notes:
# WSL commands to use if need to set up python environment to run this file:
# sudo apt update
# sudo apt install python3.12-venv -> If you’re using a different Python version, adjust accordingly (e.g., python3.10-venv)
# rm -rf venv (Removes python environment)
# python3 -m venv venv (Recreates python environment)
# source venv/bin/activate
# ./run.sh
```

## Pipeline Flow

The following table summarizes the steps performed by the ML pipeline:

| Step | Action Taken | Description |
|------|--------------|-------------|
| **1. Load Data** | `load_sqlite(db_path, table_name)` | Reads ~10,000 rows of sensor + activity data from SQLite into a DataFrame |
| **2. Deduplication** | `report_duplicates` → `drop_full_duplicates` | Identifies and removes exact duplicate rows (120 dropped), ensuring clean, unique records|
| **3. Normalize Target Labels** | `normalize_target(df, TARGET_COLUMN)` | Standardizes activity labels (e.g., collapsing “LowActivity”, “Low Activity”, “Low_Activity” into one class). Prevents label leakage and inconsistency |
| **4. Missingness Flags** | `add_missing_flags(df, MISSINGNESS_FLAG_COLUMNS)` | Adds binary indicators (`is_missing_*`) to capture whether a sensor reading was missing. Preserves missingness as a predictive signal |
| **5. Interaction Features** | `interaction_features(df)` | Creates engineered features (e.g., ratios like `CO2_ratio`, combinations like `MOx_1x4`) to capture nonlinear relationships |
| **6. Outlier Flags** | `add_outlier_flags_zscore(df, OUTLIER_COLUMNS_Z)` | Adds `is_outlier_*` flags using Z‑score thresholds (z=3.0). Helps models learn when sensor values deviate abnormally |
| **7. Encode Categoricals** | `encode_categoricals(df, CATEGORICAL_COLUMNS, ORDINAL_MAPS)` | Converts categorical variables: ordinal encoding for ordered categories, one‑hot encoding for nominal categories |
| **8. Feature Selection** | Build `numeric_to_scale` + `passthrough_cols` | Splits features into numeric (scaled + imputed) and passthrough (flags, one‑hots, ordinals, session IDs) |
| **9. Preprocessing Pipeline** | `build_preprocessor()` | - Numeric: `to_float` → median imputation → `RobustScaler`.<br>- Passthrough: **class‑conditional imputation** (per activity class) or constant `0` for numeric flags.<br>- Ensures no NaNs leak into training |
| **10. Stratified Split** | 'stratified_split(df_sel, TARGET_COLUMN)' | Splits into train/test sets while preserving class distribution. Prevents imbalance distortion |
| **11. Label Encoding** | `LabelEncoder()` |Converts string activity labels into numeric indices `[0,1,2]` for model compatibility (esp. XGB) |
| **12. Model Building** | `build_logreg`, `build_mlp`, `build_xgb` | Constructs three pipelines:<br>- **LogReg**: interpretable baseline with class_weight balancing.<br>- **MLP**: nonlinear neural net for complex sensor patterns.<br>- **XGB**: gradient boosting for robust, high‑performance predictions |
| **13. Training** | 'fit_model(model, X_train, y_train)' | Fits each pipeline. Handles sample weights for XGB, skips for MLP, uses internal balancing for LogReg |
| **14. Prediction** | 'predict(fitted, X_test)' | Generates predictions on held‑out test set |
| **15. Evaluation** | `evaluate(y_test, y_pred)` | Produces classification report + confusion matrix |
| **16. Metrics** | 'macro_f1','per_class_f1' | Summarizes performance across classes (balanced view of imbalanced data) |
| **17. Comparison** | Save to 'model_comparison.json' | Stores results for all models (macro F1, per‑class F1, confusion matrices) for visualization and reporting |

# Key findings
- Duplicates present:
    - Exact duplicate rows were found and removed (~120 rows)

    - Ensures dataset integrity and prevents leakage

- Target label inconsistencies:
    - Activity labels had multiple formats (“LowActivity”, “Low Activity”, etc.)

    - Normalization was necessary to avoid misclassification

- Missingness patterns:

    - Sensor readings (e.g., CO₂, HVAC mode) had non‑random missing values

    - Missingness flags (`is_missing_*`) revealed that missing data itself may correlate with certain activity states

- Interaction features improved signal:

    - Ratios like CO2_ratio and engineered combinations (MOx_1x4) captured nonlinear sensor relationships

    - Helped models differentiate activity levels more effectively

    - Originally used `most_frequent` strategy but found out that it introduces more biasness when dataset is already imbalanced

- Outliers detected:

    - Z‑score analysis flagged abnormal sensor spikes

    - Outlier flags (`is_outlier_*`) preserved anomaly information without discarding data

    - Kept `Temperature` Outliers that are at 200 ~ 300 range because if those values are **true sensor readings** then it could represent **critical health emergencies** and dropping them would erase important events, If they were **sensor glitches**, they will still be useful if flagged as those glitches may correlate with certain conditions (overheating of equipment, poor ventilation)

- Categorical encoding clarified inputs:

    - Ordinal maps applied to ordered categories (e.g., light levels)

    - One‑hot encoding expanded nominal categories (e.g., HVAC modes)

- Class imbalance confirmed:

    - Low activity cases underrepresented compared to moderate/high

    - Models struggled most with minority class detection

- Model performance comparison:

    - **Logistic Regression**: Macro F1 ~0.52, strong on moderate activity, weaker on low activity

    - **MLP**: Macro F1 ~0.37, failed on low activity, unstable overall

    - **XGB**: Macro F1 ~0.55, best balance across classes, especially moderate/high activity

- Bias in imputation strategies:

    - Global `most_frequent` imputation risked reinforcing majority bias

    - Class‑conditional imputation improved fairness by imputing per activity class

- Deployment insight:

    - XGB is the strongest candidate for production alerts

    - LogReg remains valuable for interpretability in clinician dashboards

    - MLP requires further tuning and balancing before deployment

# Feature Processing
| Feature                                | Type         | Processing                               |
|----------------------------------------|--------------|------------------------------------------|
| Temperature, MetalOxideSensors, CO₂s   | Numerical | Converted to float, median imputation, robust scaling, outlier flagging  |
| Derived Ratios (CO₂_ratio, MOx_1x4) | Numerical | Engineered interaction features, median imputation, robust scaling |
| Missingness & Outlier Flags | Numerical | Binary indicators (`is_missing_*`, `is_outlier_*`) retained as passthrough |                
| Humidity | Numerical | Excluded (low predictive power, redundancy, higher missingness) |
| Session ID | Identifier | Excluded (non‑predictive, risk of leakage) |
| CO_Gas, Ambient Light, HVAC, Time | Categorical | Encoded (ordinal or one‑hot), imputed with class‑conditional or neutral 0 |
| Activity Level (target label) | Categorical | Normalized, label‑encoded for model training |

# Key Notes
- **Numeric sensors**: retained, scaled, and imputed with median values; extreme values flagged instead of dropped

- **Engineered features**: added ratios and combinations to capture nonlinear sensor relationships

- **Flags**: missingness and outlier indicators kept as passthrough predictors

- **Categoricals**: transformed via encoding and imputation, not dropped

- **Identifiers/redundant features**: Session ID and Humidity excluded deliberately

- **Target labels**: normalized and encoded for consistent multiclass classification

# Model Selection
Three models were chosen to explore tradeoffs in performance and interpretability:
| **Multilayer Perceptron (MLP)** | **XGBoost** | **Logistic Regression** |
|-------------------|----------------------------|-------------------|
| Captures nonlinear sensor interactions, flexible architecture for complex tabular data |Gradient‑boosted trees, robust to missingness and imbalance, strong performance on structured features | Interpretable baseline, highlights feature influence and helps detect leakage or imbalance |

- **MLP**: Added to test nonlinear learning capacity on engineered features and anomaly flags

- **XGBoost**: Provided the strongest balance of accuracy and robustness, especially with imbalanced activity classes

- **Logistic Regression**: Served as a transparent baseline, useful for interpretability and debugging

# Evaluations & Results Summary
| Model	| Accuracy | Macro F1 | Macro Precision | Macro Recall | Notes |
| Logistic Regression | ~0.62 |	~0.51 |	~0.53 |	~0.50 | Solid baseline, interpretable, struggles with minority class |
| MLP | ~0.45 | ~0.28 | ~0.30 | ~0.27 | Captures nonlinearities but unstable, poor on low activity |
| XGBoost | ~0.66 | ~0.55 | ~0.56 | ~0.54 | Best balance, handles imbalance and complex interactions well |

# Deployment Considerations
**Model Saving**

- All trained models (Logistic Regression, MLP, XGBoost) are saved in `.joblib` format under the `artifacts/` directory

- This enables consistent and fast inference in production environments without retraining

**Pipeline Portability**

- **Relative Paths**: 
The pipeline uses relative paths for configuration and artifact storage, ensuring compatibility with automated environments (e.g., GitHub Actions, CI/CD)

- **External Data**: 
The SQLite database file (`gas_monitoring.db`) is excluded from the repository for security reasons

It must be provided manually during deployment, with secure data handling practices enforced

**Modularity**

- Core functions for data loading, preprocessing, feature engineering, model building, training, and evaluation are modularized into the `src/` package

- This design allows:

    - Easy integration of new models (plug‑and‑play with `models/` module)

    - Quick adaptation to variations in sensor data or schema changes

    - Reusability across different ElderGuard projects and healthcare datasets

**Reproducibility & Auditability**

- Preprocessing steps (deduplication, normalization, imputation, encoding, flagging) are versioned and logged for full audit trails

- Model comparison results are saved in JSON (`artifacts/model_comparison.json`) for reproducibility and downstream visualization
