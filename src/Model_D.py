# Model D Development (Logistic Regression 2)
# 1. Load and Preprocess
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Load Data
file_path = '../data/processed/dataset_final.csv'

df = pd.read_csv(file_path)

# 2. Cleaning (Must match Model A exactly)
cols_to_drop = [
    'transaction_id', 'timestamp', 'customer_id', 
    'transaction_description', 'merchant_name', 'customer_support_note',
    'risk_category' 
]
df_clean = df.drop(columns=cols_to_drop, errors='ignore')

# 3. Encoding
df_encoded = pd.get_dummies(df_clean, drop_first=True)

# 4. Split
X = df_encoded.drop('is_fraud', axis=1)
y = df_encoded['is_fraud']

# 5. Train-Test Split (CRITICAL: Use random_state=42 to match Model A)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 6. Scaling (REQUIRED for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("✅ Data loaded, cleaned, and SCALED for Logistic Regression.")
print(f"Training Shape: {X_train_scaled.shape}")
# 2. Train & Tune Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# Setup Grid Search
param_grid = {
    'C': [0.01, 0.1, 1, 10],            # Regularization strength
    'solver': ['liblinear', 'lbfgs']    # Optimization algorithms
}

# Run Grid Search
grid_search = GridSearchCV(
    LogisticRegression(random_state=42, max_iter=1000),
    param_grid,
    cv=3,
    scoring='roc_auc',
    n_jobs=-1
)

print("Training Logistic Regression...")
grid_search.fit(X_train_scaled, y_train)

best_lr = grid_search.best_estimator_
print(f"✅ Best Parameters: {grid_search.best_params_}")
# 3. Evaluate and Interpret
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 1. Predictions
y_pred = best_lr.predict(X_test_scaled)
y_prob = best_lr.predict_proba(X_test_scaled)[:, 1]

# 2. Calculate Required Metrics
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
# RMSE is technically for regression, but we can calculate it on probabilities (Brier Score)
rmse = np.sqrt(mean_squared_error(y_test, y_prob))

print("--- Model D (Logistic Regression) Final Evaluation ---")
print(f"✅ Accuracy:  {acc:.4f}")
print(f"✅ F1-Score:  {f1:.4f}")
print(f"✅ ROC-AUC:   {auc:.4f}")
print(f"✅ RMSE:      {rmse:.4f}")

# 3. Visual: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Logistic Regression)')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# 4. Feature Importance (Coefficients)
# We save this text to a variable so we can print it for the LLM later
coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': best_lr.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Coefficient', y='Feature', data=coef_df.head(10), palette='coolwarm')
plt.title('Top Factors (Logistic Regression)')
plt.xlabel('Coefficient (Positive = Higher Fraud Risk)')
plt.axvline(x=0, color='black', linestyle='--')
plt.show()
# --- AUTO-GENERATED PROMPT FOR GEN-AI ---
# Run this cell, then COPY the output below and PASTE it into ChatGPT/Gemini.

print("📋 COPY THIS PROMPT INTO CHATGPT/GEMINI:\n")
print(f"I have trained a Logistic Regression model to detect fraud. Here are the results:")
print(f"- Accuracy: {acc:.4f}")
print(f"- F1-Score: {f1:.4f}")
print(f"- ROC-AUC: {auc:.4f}")
print(f"- RMSE: {rmse:.4f}")
print("\nThe top 5 most important features (and their weights) are:")
for index, row in coef_df.head(5).iterrows():
    print(f"- {row['Feature']}: {row['Coefficient']:.4f}")

print("\nBased on these results, please generate a 'Business Insight Summary' (approx 150 words).")
print("Explain what the metrics mean for a bank manager and interpret why these specific features are important predictors.")