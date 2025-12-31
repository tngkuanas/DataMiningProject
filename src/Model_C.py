import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score

def main():
    """
    Loads the preprocessed dataset, trains a Logistic Regression model,
    and evaluates its performance.
    """
    # Define relative path
    input_path = Path("data/processed/dataset_final.csv")

    # --- Data Loading ---
    print(f"Loading preprocessed data from '{input_path}'...")
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found at '{input_path}'. "
            "Ensure 'scaling_encoding.py' and 'final_dataset_cleaning.py' have been run."
        )
    
    df = pd.read_csv(input_path)
    print("Data loaded. Shape:", df.shape)

    # --- Feature and Target Split ---
    X = df.drop('is_fraud', axis=1)
    y = df['is_fraud']

    # --- Train-Test Split ---
    # Stratify y to maintain the same proportion of fraud cases in train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set shape: {X_train.shape}")
    print(f"Testing set shape: {X_test.shape}")

    # --- Model Training ---
    print("\nTraining Logistic Regression model...")
    # Increased max_iter for convergence with scaled data
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    print("Model training complete.")

    # --- Evaluation ---
    print("\n--- Logistic Regression Model Performance ---")
    
    # Predictions
    y_pred = lr_model.predict(X_test)
    
    # Get probabilities for ROC-AUC
    # Check if the model has the 'predict_proba' method
    if hasattr(lr_model, "predict_proba"):
        y_prob = lr_model.predict_proba(X_test)[:, 1]
        print(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
