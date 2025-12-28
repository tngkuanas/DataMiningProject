import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def main():
    """
    Loads the final dataset, drops unnecessary columns, applies specific scaling
    and one-hot encoding based on predefined lists, and saves the processed DataFrame.
    """
    # Define relative paths
    input_path = Path("data/processed/synthetic_fraud_dataset_features.csv")
    output_path = Path("data/processed/dataset_final.csv")

    # --- Data Loading ---
    print(f"Loading data from '{input_path}'...")
    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file not found at '{input_path}'. "
            "Make sure you are running this script from the root of the project directory."
        )
    
    df = pd.read_csv(input_path)
    print("Original data shape:", df.shape)

    # --- Feature and Target Split ---
    # Drop the target variable first
    X = df.drop(columns="is_fraud")
    Y = df["is_fraud"]

    # --- Drop Unnecessary Columns for Modeling ---
    print("\nDropping identifier and raw text columns...")
    cols_to_drop = [
        'transaction_id', 'customer_id', 'timestamp', 
        'transaction_description', 'merchant_name', 'customer_support_note'
    ]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    print(f"Dropped: {cols_to_drop}")
    print("Shape after dropping columns:", X.shape)

    # --- Define Explicit Column Lists for Preprocessing ---
    # 1. Columns for One-Hot Encoding
    cat_cols_to_encode = [
        'transaction_type', 'merchant_category', 'device_type', 'payment_channel',
        'description_tone', 'semantic_merchant_category'
    ]

    # 2. Columns for Scaling
    num_cols_to_scale = [
        'transaction_amount', 'account_balance_before', 'account_balance_after',
        'avg_transaction_amount_7d', 'time_since_last_transaction_hr',
        'account_age_days', 'description_risk_score'
    ]
    
    # Filter lists to only include columns that actually exist in the DataFrame
    cat_cols = [col for col in cat_cols_to_encode if col in X.columns]
    num_cols = [col for col in num_cols_to_scale if col in X.columns]

    print(f"\nIdentified {len(cat_cols)} columns for One-Hot Encoding.")
    print(f"Identified {len(num_cols)} columns for Scaling.")

    # --- Preprocessing Pipeline ---
    print("\nApplying scaling and one-hot encoding...")
    
    # Safe OneHotEncoder (handles different sklearn versions)
    try:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    # Define the transformers. Use 'remainder=passthrough' to keep all other columns.
    transformers = [
        ('cat', encoder, cat_cols),
        ('num', StandardScaler(), num_cols)
    ]

    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    # Fit and transform the data
    X_processed = preprocessor.fit_transform(X)

    # --- Create Processed DataFrame ---
    # Get feature names from the encoder
    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    
    # Get names of columns that were passed through
    # This requires knowing which columns were NOT in cat_cols or num_cols
    passthrough_cols = [col for col in X.columns if col not in cat_cols and col not in num_cols]
    
    # Combine all feature names in the correct order
    feature_names = encoded_cat_names + num_cols + passthrough_cols
    
    dataset_final = pd.DataFrame(X_processed, columns=feature_names)

    # Append the target variable
    dataset_final['is_fraud'] = Y.reset_index(drop=True)
    print("Preprocessing complete. New data shape:", dataset_final.shape)

    # --- Save Output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_final.to_csv(output_path, index=False)
    print(f"\nSaved ML-ready preprocessed data to '{output_path}'")
    
    print("\nSample of preprocessed data:")
    print(dataset_final.head())

if __name__ == "__main__":
    main()