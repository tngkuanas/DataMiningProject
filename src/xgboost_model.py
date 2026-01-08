import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from utils import save_results

def main():
    """
    Trains and evaluates an XGBoost classifier on the final, pre-processed dataset.
    """
    # --- 1. Load Data ---
    print("XGBoost: Loading final pre-processed dataset...")
    input_path = Path("data/processed/final_ml_ready_dataset.csv")
    results_path = "results/model_comparison.csv"
    
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{input_path}'. Please run all data preparation scripts first.")
        return

    # --- 2. Define Features (X) and Target (y) ---
    target_column = 'is_fraud'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # --- 3. Split Data ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # --- 4. Define and Train Model ---
    print("XGBoost: Training model...")
    model_name = "XGBoost"
    # XGBoost handles class imbalance with scale_pos_weight
    scale_pos_weight = y_train.value_counts()[0] / y_train.value_counts()[1]
    hyperparams = {'random_state': 42, 'eval_metric': 'logloss', 'scale_pos_weight': scale_pos_weight, 'n_estimators': 100}
    model = xgb.XGBClassifier(**hyperparams)
    
    model.fit(X_train, y_train)

    # --- 5. Evaluate Model ---
    print("XGBoost: Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # --- 6. Save Results ---
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    description = f"A gradient boosting framework. Hyperparameters: {hyperparams}"
    
    save_results(results_path, model_name, description, metrics)

if __name__ == "__main__":
    main()
