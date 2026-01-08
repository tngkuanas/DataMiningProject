import pandas as pd
from pathlib import Path

def save_results(output_path_str, model_name, description, metrics):
    """
    Saves model evaluation metrics to a single CSV file.

    If the file exists, it updates the row for the given model_name
    or appends a new row if the model_name is not found.

    Args:
        output_path_str (str): The path to the output CSV file.
        model_name (str): The name of the model being evaluated.
        description (str): A short description of the model and its hyperparameters.
        metrics (dict): A dictionary of performance metrics (e.g., accuracy, f1_score).
    """
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a new DataFrame for the current results
    new_results_df = pd.DataFrame([metrics])
    new_results_df['model'] = model_name
    new_results_df['description'] = description
    
    # Reorder columns to have model and description first
    cols = ['model', 'description'] + [col for col in new_results_df.columns if col not in ['model', 'description']]
    new_results_df = new_results_df[cols]

    if output_path.exists():
        # Load existing results
        results_df = pd.read_csv(output_path)
        
        # Check if the model already has an entry
        if model_name in results_df['model'].values:
            # Replace the existing entry
            results_df = results_df[results_df['model'] != model_name]
            results_df = pd.concat([results_df, new_results_df], ignore_index=True)
            print(f"Updated results for '{model_name}' in '{output_path}'.")
        else:
            # Append new results
            results_df = pd.concat([results_df, new_results_df], ignore_index=True)
            print(f"Appended new results for '{model_name}' to '{output_path}'.")
    else:
        # Create a new results file
        results_df = new_results_df
        print(f"Created new results file and saved results for '{model_name}' to '{output_path}'.")

    results_df.to_csv(output_path, index=False)
