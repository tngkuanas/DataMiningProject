import numpy as np
import pandas as pd
from pathlib import Path
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def main():
    """
    Loads features, applies scaling and encoding, drops a specified column,
    and saves the final ML-ready dataset.
    """
    # --- Path Definitions ---
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

    # --- Augment Data with 200 Noisy Rows ---
    print("\nAugmenting data with 200 new noisy rows...")
    n_new_rows = 200
    if len(df) >= n_new_rows:
        new_rows_sample = df.sample(n=n_new_rows, random_state=42, replace=True).copy()
        
        numeric_cols = new_rows_sample.select_dtypes(include=np.number).columns.tolist()
        # Exclude the target variable from having noise added
        if 'is_fraud' in numeric_cols:
            numeric_cols.remove('is_fraud')

        for col in numeric_cols:
            noise = np.random.normal(0, 1, new_rows_sample[col].shape)
            new_rows_sample[col] += noise
        
        df = pd.concat([df, new_rows_sample], ignore_index=True)
        print(f"Added {n_new_rows} new noisy rows. New total rows: {len(df)}")
    else:
        print("Not enough rows to sample for data augmentation.")

    # --- Feature and Target Split ---
    X = df.drop(columns="is_fraud")
    Y = df["is_fraud"]

    # --- Column Dropping (as per user instruction) ---
    print("\nDropping identifier, raw text, and leaky columns...")
    cols_to_drop = [
        'transaction_id', 'customer_id', 'timestamp', 
        'transaction_description', 'merchant_name', 'customer_support_note',
        'user_risk_score', 'previous_fraud_count'
    ]
    X = X.drop(columns=cols_to_drop, errors='ignore')
    print(f"Dropped: {cols_to_drop}")
    print("Shape after dropping columns:", X.shape)



    # --- Define Explicit Column Lists for Preprocessing ---
    cat_cols_to_encode = [
        'transaction_type', 'merchant_category', 'device_type', 'payment_channel',
        'description_tone', 'semantic_merchant_category'
    ]
    num_cols_to_scale = [
        'transaction_amount', 'account_balance_before', 'account_balance_after',
        'avg_transaction_amount_7d', 'time_since_last_transaction_hr',
        'account_age_days', 'description_risk_score'
    ]
    
    cat_cols = [col for col in cat_cols_to_encode if col in X.columns]
    num_cols = [col for col in num_cols_to_scale if col in X.columns]

    print(f"\nIdentified {len(cat_cols)} columns for One-Hot Encoding.")
    print(f"Identified {len(num_cols)} columns for Scaling.")

    # --- Preprocessing Pipeline ---
    print("\nApplying scaling and one-hot encoding...")
    try:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)

    transformers = [
        ('cat', encoder, cat_cols),
        ('num', StandardScaler(), num_cols)
    ]
    preprocessor = ColumnTransformer(transformers=transformers, remainder='passthrough')
    
    X_processed = preprocessor.fit_transform(X)

    # --- Create Processed DataFrame ---
    encoded_cat_names = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_cols).tolist()
    passthrough_cols = [col for col in X.columns if col not in cat_cols and col not in num_cols]
    feature_names = encoded_cat_names + num_cols + passthrough_cols
    
    dataset_final = pd.DataFrame(X_processed, columns=feature_names)

    # Append the target variable
    dataset_final['is_fraud'] = Y.reset_index(drop=True)
    print("Preprocessing complete. Shape before final cleaning:", dataset_final.shape)

    # --- Final Cleaning Step (from final_dataset_cleaning.py) ---
    if 'location_match' in dataset_final.columns:
        print("\nDropping 'location_match' column...")
        dataset_final = dataset_final.drop(columns=['location_match'])
        print("Column 'location_match' dropped.")
    else:
        print("\nColumn 'location_match' not found in the dataset. No action needed.")
    
    print("New data shape:", dataset_final.shape)

    # --- Flip 60% of Non-Fraud Labels to Fraud ---
    print("\nFlipping 60% of non-fraud labels to fraud...")
    non_fraud_indices = dataset_final[dataset_final['is_fraud'] == 0].index
    
    # Number of non-fraud samples to flip
    num_to_flip = int(len(non_fraud_indices) * 0.15)
    
    # Randomly choose indices to flip
    indices_to_flip = np.random.choice(non_fraud_indices, size=num_to_flip, replace=False)
    
    # Flip the labels
    dataset_final.loc[indices_to_flip, 'is_fraud'] = 1
    print(f"Flipped {num_to_flip} non-fraud labels to fraud.")
    print(f"New 'is_fraud' value counts:\n{dataset_final['is_fraud'].value_counts()}")


    # --- Save Output ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_final.to_csv(output_path, index=False)
    print(f"\nSaved final ML-ready data to '{output_path}'")
    
    print("\nSample of final data:")
    print(dataset_final.head())

if __name__ == "__main__":
    main()