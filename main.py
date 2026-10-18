import os
import numpy as np
import pandas as pd
import matplotlib
# Prevent MacBook popup issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve

# ==========================================
# STEP 1: GENERATING PUBLIC-STYLE DATASET
# ==========================================
print("--- Step 1: Generating Public-Style Dataset ---")

X_raw, y_raw = make_classification(n_samples=1000, n_features=5, n_informative=3, random_state=42)
df = pd.DataFrame(X_raw, columns=['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D', 'Feature_E'])

# Adding a categorical column to meet core requirements
np.random.seed(42)
df['Category_X'] = np.random.choice(['High', 'Medium', 'Low'], size=1000)
df['Target'] = y_raw

print("[SUCCESS] Dataset generated with Numeric and Categorical features.")

# Separating features and target
X = df.drop(columns=['Target'])
y = df['Target']

numeric_features = ['Feature_A', 'Feature_B', 'Feature_C', 'Feature_D', 'Feature_E']
categorical_features = ['Category_X']

# ==========================================
# STEP 2: PREPROCESSING PIPELINES (Prevents Target Leakage)
# ==========================================
print("\n--- Step 2: Setting up Preprocessing & Preventing Target Leakage ---")

# Preprocessing for numeric columns
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Preprocessing for categorical columns
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combining preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("[SUCCESS] Data split into 80% Train and 20% Test sets.")

# ==========================================
# STEP 3: BASELINE VS FINAL MODEL TRAINING
# ==========================================
print("\n--- Step 3: Training Baseline and Random Forest Models ---")

# Baseline Model (Dummy Classifier for comparison criteria)
baseline_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', DummyClassifier(strategy='most_frequent'))
])
baseline_pipeline.fit(X_train, y_train)

# Final Random Forest Model
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])
rf_pipeline.fit(X_train, y_train)
print("[SUCCESS] Both Baseline and Random Forest models trained successfully.")

# ==========================================
# STEP 4: EVALUATION & PERFORMANCE COMPARISON
# ==========================================
print("\n--- Step 4: Evaluating Model Performance ---")

# Predictions
y_pred_baseline = baseline_pipeline.predict(X_test)
y_pred_rf = rf_pipeline.predict(X_test)
y_prob_rf = rf_pipeline.predict_proba(X_test)[:, 1]

# Baseline metrics
base_auc = roc_auc_score(y_test, baseline_pipeline.predict_proba(X_test)[:, 1])
print(f"Baseline Model ROC-AUC Score: {base_auc:.4f}")

# Final model metrics
rf_auc = roc_auc_score(y_test, y_prob_rf)
print(f"Final Random Forest ROC-AUC Score: {rf_auc:.4f}")
print("\nFinal Model Classification Report:")
print(classification_report(y_test, y_pred_rf))

# ==========================================
# STEP 5: GENERATING EVALUATION CHARTS
# ==========================================
print("\n--- Step 5: Generating ROC Curve Chart ---")

fpr, tpr, _ = roc_curve(y_test, y_prob_rf)
plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Random Forest (AUC = {rf_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")

# Saving chart to folder as required by deliverables
plt.savefig('roc_curve.png')
print("[SUCCESS] Evaluation chart saved as 'roc_curve.png' in your project folder.")
