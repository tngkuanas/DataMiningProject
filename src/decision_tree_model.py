import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from pathlib import Path
from utils import save_results

def main():
    """
    Trains and evaluates a Decision Tree classifier on the final, pre-processed dataset.
    """
    # --- 1. Load Data ---
    print("Decision Tree: Loading final pre-processed dataset...")
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
    print("Decision Tree: Training model...")
    model_name = "Decision Tree"
    hyperparams = {'random_state': 42, 'class_weight': 'balanced', 'max_depth': 10}
    model = DecisionTreeClassifier(**hyperparams)
    
    model.fit(X_train, y_train)

    # --- 5. Evaluate Model ---
    print("Decision Tree: Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] # For ROC AUC

    # --- 6. Save Results ---
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    description = f"A standard Decision Tree classifier. Hyperparameters: {hyperparams}"
    
    save_results(results_path, model_name, description, metrics)

if __name__ == "__main__":
    main()
