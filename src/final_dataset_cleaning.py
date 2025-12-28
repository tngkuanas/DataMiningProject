import pandas as pd
from pathlib import Path

def main():
    """
    Loads the final processed dataset, drops the 'location_match' column,
    and saves it back to the same path.
    """
    input_output_path = Path("data/processed/dataset_final.csv")

    print(f"Loading data from '{input_output_path}'...")
    if not input_output_path.exists():
        raise FileNotFoundError(
            f"Input file not found at '{input_output_path}'. "
            "Please ensure 'data/dataset_final.csv' exists."
        )
    
    df = pd.read_csv(input_output_path)
    print("Original data shape:", df.shape)

    # Check if 'location_match' column exists before dropping
    if 'location_match' in df.columns:
        print("Dropping 'location_match' column...")
        df = df.drop(columns=['location_match'])
        print("Column 'location_match' dropped.")
    else:
        print("Column 'location_match' not found in the dataset. No action needed.")

    print("New data shape:", df.shape)

    # Save the modified DataFrame back to the same path
    df.to_csv(input_output_path, index=False)
    print(f"\nModified dataset saved back to '{input_output_path}'")
    
    print("\nSample of modified data:")
    print(df.head())

if __name__ == "__main__":
    main()

