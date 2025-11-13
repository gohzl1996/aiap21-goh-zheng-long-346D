Goh Zheng Long 
goh.zl1996@gmail.com

# Folder Structure Overview
```bash
aiap21-goh-zheng-long-346D/
├── data/
│   └── gas_monitoring.db        # [Not committed] SQLite DB – must be added manually
├── src/
│   ├── __init__.py              # Imports from myFuncs.py
│   ├── myFuncs.py               # Contains preprocessing, feature engineering, model training & evaluation functions
│   └── model_training.py        # Main pipeline script
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
bash run.sh                       # or use ./run.sh
```

## Pipeline Flow

The following table summarizes the steps performed by the ML pipeline:

| Step | Description |
|------|-------------|
| **Data Loading** | Reads dataset from SQLite database located at `data/gas_monitoring.db`. |
| **Label Normalization** | Standardizes inconsistencies in the `Activity Level` column. |
| **Missing Value Handling** | Drops columns with >30% missing data; imputes remaining values using median strategy. |
| **Outlier Capping** | Caps outliers based on activity-specific domain knowledge and boxplot thresholds. |
| **Feature Selection** | Retains top SHAP-relevant features to enhance model simplicity and interpretability. |
| **Feature Engineering** | - **Tree models**: Label encoding and domain-specific features. <br> - **Logistic Regression**: One-hot encoding with polynomial interaction terms. |
| **Resampling** | Applies SMOTEENN to balance class distribution across activity levels. |
| **Model Training** | Trains three models: Random Forest, XGBoost, and Logistic Regression. |
| **Evaluation** | Generates performance metrics, confusion matrices, ROC/PR curves, learning curves, and SHAP explanations. |


# Key findings
- The dataset was imbalanced, particularly with overrepresentation in the `Low Activity` class.
- Some `Temperature` values are rather abnormal with ranges 200 to 300, major visible outliers.
- After imputation and activity-specific outlier removal, model performance improved significantly.
- Sensor data from `MetalOxideSensor Units`, `CO2 Sensors`, and `Temperature` were found most predictive.

# Feature Processing
| Feature                                | Type         | Processing                               |
|----------------------------------------|--------------|------------------------------------------|
| Temperature, MetalOxideSensors, CO₂s   | Numerical    | Median imputation & Outlier NaN Marking  |
| Session ID, Humidity                   | Numerical    | Features dropped                         |
| CO_Gas, Ambient Light, HVAC, Time      | Categorical  | Features dropped                         |                
| Activity Level (label)                 | Categorical  | Normalized & Encoded                     |

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
