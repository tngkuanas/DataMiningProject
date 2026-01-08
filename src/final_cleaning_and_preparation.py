import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from pathlib import Path

def main():
    """
    This script performs the final cleaning and preparation of the dataset.
    It removes nulls and converts all remaining features into a purely
    numerical format, ready for various ML models.
    """
    input_path = Path("data/processed/simple_llm_fraud_dataset_features.csv")
    output_path = Path("data/processed/final_ml_ready_dataset.csv")

    print(f"Loading feature-engineered dataset from '{input_path}'...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'.")
        print("Please run the data generation and feature engineering scripts first.")
        return

    print(f"Original shape: {df.shape}")

    # --- 1. Handle Null Values ---
    # In our case, the feature engineering script uses defaults, but this is good practice.
    initial_rows = len(df)
    df.dropna(inplace=True)
    rows_dropped = initial_rows - len(df)
    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows containing null values.")
    else:
        print("No null values found to drop.")

    # --- NEW: Create Time-Based Features ---
    print("Creating time-based features from 'timestamp'...")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    # --- 2. Separate Features for Processing ---
    # Keep the target variable separate
    target = df['is_fraud']
    
    # Drop identifiers and columns that will be transformed
    cols_to_drop = ['transaction_id', 'customer_id', 'timestamp', 
                    'transaction_description', 'merchant_category', 'transaction_country']
    
    # Keep the already numerical columns
    numerical_df = df.drop(columns=cols_to_drop + ['is_fraud'], errors='ignore')

    # --- 3. One-Hot Encode Categorical Features ---
    print("One-hot encoding categorical features...")
    categorical_cols = ['merchant_category', 'transaction_country']
    categorical_df = pd.get_dummies(df[categorical_cols], prefix=categorical_cols)
    print(f"Created {len(categorical_df.columns)} columns from categorical features.")

    # --- 4. TF-IDF for Text Feature ---
    print("Applying TF-IDF to text feature...")
    vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(df['transaction_description'])
    
    # Create a DataFrame from the TF-IDF matrix
    tfidf_df = pd.DataFrame(
        tfidf_matrix.toarray(), 
        columns=[f"tfidf_{name}" for name in vectorizer.get_feature_names_out()]
    )
    print(f"Created {len(tfidf_df.columns)} columns from the text feature using TF-IDF.")
    
    # --- 5. Combine All Features ---
    print("Combining all processed features...")
    # Align indices before concatenating
    df.reset_index(drop=True, inplace=True)
    numerical_df.reset_index(drop=True, inplace=True)
    categorical_df.reset_index(drop=True, inplace=True)
    tfidf_df.reset_index(drop=True, inplace=True)
    target.reset_index(drop=True, inplace=True)
    
    final_df = pd.concat([numerical_df, categorical_df, tfidf_df, target], axis=1)

    # --- 6. Save Final Dataset ---
    final_df.to_csv(output_path, index=False)
    print(f"\nFinal ML-ready dataset saved to '{output_path}'")
    print(f"Final shape: {final_df.shape}")

    # --- 7. Important Note for User ---
    print("\n" + "="*50)
    print("IMPORTANT NOTE:")
    print("This dataset is now fully numerical.")
    print("For models sensitive to feature scaling (like Neural Networks,")
    print("Logistic Regression, and KNN), you should apply a scaler")
    print("(e.g., StandardScaler) AFTER splitting this data into")
    print("training and testing sets to avoid data leakage.")
    print("="*50)


if __name__ == "__main__":
    main()
