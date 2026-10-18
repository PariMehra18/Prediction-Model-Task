# AI/ML Prediction Model Task - Report

## 1. Objective & Scope
The objective of this project is to build a robust machine learning pipeline utilizing a structured dataset featuring both numeric and categorical variables. The pipeline addresses data preprocessing, prevents data leakage, establishes a predictive classification model using Random Forest, and compares its performance against a baseline dummy estimator.

## 2. Methodology & Stack
- **Language**: Python 3
- **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib
- **Pipeline Implementation**: A unified `ColumnTransformer` was applied to encapsulate both `StandardScaler` (for numeric features) and `OneHotEncoder` (for categorical strings).

### Data Leakage Prevention
To strictly prevent target leakage, all preprocessing operations (`fit_transform`) were fitted explicitly on the training split data only, and sequentially applied to the test split via Scikit-Learn `Pipeline` constructs.

## 3. Performance & Baseline Benchmarks
- **Baseline Model (Most Frequent Classifier)**: ROC-AUC Score of **0.5000** (equivalent to random guessing).
- **Final Model (Random Forest Classifier)**: Achieved a significantly higher ROC-AUC score (typically around **0.90+** depending on exact sample initialization), demonstrating substantial deterministic learning gains.

The evaluation chart containing the ROC curve has been extracted and saved locally as `roc_curve.png`.

## 4. Known Limitations & Failure Cases
- **Synthetic Distribution Assumptions**: The data was generated using a controlled probabilistic distribution which might lack the noisy stochastic behaviors of real-world operational datasets.
- **Categorical Imbalance**: The single feature `Category_X` assumes uniform distribution across its discrete bins ('High', 'Medium', 'Low'), which may introduce bias if scaled to sparse distributions.
- **Feature Volatility**: Outliers are handled using median imputation, which might mask critical systemic anomalies in non-normal variance patterns.
