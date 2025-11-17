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

- Included features based on correlation, SHAP analysis and domain relevance
- Dropped features displayed low predictive power and high missingness, `Session ID` was
an identifier with no real learning value
- Overall goal was to retain high and compact signal feature set to max out generalization
performance as well as minimize noise and overfitting risk due to `Low Activity` being the
dominant class over the other two

# Model Selection
Three models were chosen to explore tradeoffs in performance and interpretability:
|Random Forest      |XGBoost                     |Logistic Regression|
|-------------------|----------------------------|-------------------|
|Strong baseline and robust to noisy features|Gradient-boosted trees, ideal for imbalanced and complex feature interactions|for interpretability and feature influence clarity      |

# Model Evaluations
| Metric                                   | Description                                                                                                                                      |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Accuracy**                             | Proportion of total correct predictions. Best when classes are balanced.                                                                         |
| **Precision**                            | Of all positive predictions, how many were truly positive. Useful when false positives are costly.                                               |
| **Recall**                               | Of all actual positives, how many were correctly predicted. Important when false negatives are critical.                                         |
| **F1-score**                             | Harmonic mean of precision and recall. More informative than accuracy in imbalanced datasets.                                                    |
| **Confusion Matrix**                     | Shows true vs predicted values across all classes. Helps identify specific misclassifications.                                                   |
| **SHAP (SHapley Additive exPlanations)** | Model-agnostic method for interpreting predictions, showing how each feature impacts output. Used here for top-performing model (Random Forest). |

# Results Summary
| Model                   | Accuracy | F1-Score (Weighted) | Precision | Recall   | SHAP Applied              | Notes                                                                |
| ----------------------- | -------- | ------------------- | --------- | -------- | ------------------------- | -------------------------------------------------------------------- |
| **Random Forest**       | High     | **Highest**         | High      | High     | ✅                         | Best overall performance across metrics.                             |
| **XGBoost**             | High     | Competitive         | High      | High     | ❌                         | Needed label encoding fix before training.                           |
| **Logistic Regression** | Moderate | Lower               | Moderate  | Moderate | ❌                         | Less effective due to linear assumptions; good for interpretability. |


# Deployment Considerations
**Model Saving**
- All trained models are saved in `.joblib` format to enable consistent and fast future inference.

**Pipeline Portability**
- Relative Paths:

    The pipeline is designed with relative paths to ensure compatibility with automated environments such as GitHub Actions.

- External Data:

    The SQLite database file (`gas_monitoring.db`) is excluded from the repository and must be added manually during deployment. Secure data handling practices should be followed.

**Modularity**
- Core functions for feature engineering and model training are modularized in `myFuncs.py`.
- This allows:
    - Easy integration of new models
    - Quick adaptation to variations in data
    - Reusability across different projects
