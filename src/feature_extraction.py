import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
import json
from tqdm.auto import tqdm
import math

# --- Environment and API Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
genai.configure(api_key=api_key)

def call_gemini_batch(prompt, model_name="gemini-3-flash-preview", max_retries=3, initial_delay=5):
    """
    Calls the Gemini API with a batch prompt and retry logic.
    Expects a JSON response.
    """
    model = genai.GenerativeModel(model_name)
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(contents=prompt)
            cleaned_text = resp.text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_text)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error decoding JSON on attempt {attempt + 1}: {e}. Raw response: {resp.text}")
            if attempt == max_retries - 1: return None
        except Exception as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                delay = initial_delay * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {delay}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"An unexpected error occurred: {e}")
                return None
    print(f"Failed to get a valid response after {max_retries} attempts.")
    return None

def create_batch_prompt(items_list, task_description, output_format_description):
    """Creates a flexible prompt for various batch processing tasks."""
    input_json = json.dumps({i: item for i, item in enumerate(items_list)}, indent=2)
    prompt = f"""
    You will be given a JSON object with numbered keys, each containing a text entry.
    Your task is to {task_description} for each entry.

    Respond with a single JSON object where the keys correspond to the input keys.
    The value for each key should be {output_format_description}.
    Ensure your output is a valid JSON object and nothing else.

    Input:
    {input_json}

    Output:
    """
    return prompt

def batch_process_column(series, task_description, output_format_description, item_key, batch_size, post_process_func, default_value):
    """Processes a pandas Series in batches, with flexible post-processing."""
    results = {}
    num_batches = math.ceil(len(series) / batch_size)
    
    with tqdm(total=num_batches, desc=f"Processing {item_key}") as pbar:
        for i in range(0, len(series), batch_size):
            batch = series[i:i + batch_size]
            prompt = create_batch_prompt(batch.tolist(), task_description, output_format_description)
            batch_results = call_gemini_batch(prompt)

            original_indices = batch.index
            if batch_results:
                for j, result_val in batch_results.items():
                    original_idx = original_indices[int(j)]
                    results[original_idx] = post_process_func(result_val)
            else:
                print(f"Warning: Batch starting at index {i} failed. Using default value.")
                for idx in original_indices:
                    results[idx] = default_value
            
            pbar.update(1)
            time.sleep(1)
            
    return pd.Series(results)

def main():
    input_path = 'data/raw/synthetic_fraud_dataset.csv'
    output_path = 'data/processed/synthetic_fraud_dataset_features.csv'
    
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Dataset loaded. Shape: {df.shape}")

    # --- LLM-Based Risk Scoring ---
    print("\nGenerating LLM-based risk scores for transaction descriptions...")
    risk_task = "rate the risk of this transaction description from 1 (Safe) to 10 (Critical) based on urgency and anomaly"
    risk_output_desc = "an integer between 1 and 10"
    # stay under 20 requests for 3 features -> ~6 requests per feature
    batch_size = math.ceil(len(df) / 6) 
    
    def process_risk_score(x):
        try: return int(x)
        except (ValueError, TypeError): return 5 # Default to neutral
    
    risk_scores = batch_process_column(df['transaction_description'], risk_task, risk_output_desc, "Risk Scores", batch_size, process_risk_score, 5)
    df['description_risk_score'] = risk_scores.reindex(df.index).fillna(5)

    # --- Sentiment/Urgency Extraction ---
    print("\nExtracting tone from transaction descriptions...")
    tone_task = "classify the tone of this text"
    tone_categories = ['routine', 'urgent', 'panic']
    tone_output_desc = f"a single-word category: {', '.join(tone_categories)}"
    
    def process_tone(x):
        val = str(x).lower()
        return val if val in tone_categories else 'routine'

    tones = batch_process_column(df['transaction_description'], tone_task, tone_output_desc, "Tones", batch_size, process_tone, 'routine')
    df['description_tone'] = tones.reindex(df.index).fillna('routine')
    
    # --- Semantic Merchant Segmentation ---
    print("\nGenerating semantic categories for merchant names...")
    merchant_task = "categorize this merchant into a broad business category"
    merchant_categories = ['daily essentials', 'luxury goods', 'crypto/finance', 'services', 'other']
    merchant_output_desc = f"a single category: {', '.join(merchant_categories)}"
    
    unique_merchants = df['merchant_name'].unique()
    merchant_batch_size = math.ceil(len(unique_merchants) / 6)
    
    def process_merchant_cat(x):
        val = str(x).lower()
        return val if val in merchant_categories else 'other'

    merchant_results = batch_process_column(pd.Series(unique_merchants), merchant_task, merchant_output_desc, "Merchants", merchant_batch_size, process_merchant_cat, 'other')
    
    merchant_map = dict(zip(unique_merchants, merchant_results))
    df['semantic_merchant_category'] = df['merchant_name'].map(merchant_map).fillna('other')

    # --- Finalizing and Saving ---
    # Drop original leaky columns if they exist
    df = df.drop(columns=['risk_category', 'support_sentiment'], errors='ignore')
    
    print(f"\nSaving final dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Feature extraction complete!")
    print(f"Output dataset shape: {df.shape}")
    
    print("\nSample of final dataset with new features:")
    new_cols = ['description_risk_score', 'description_tone', 'semantic_merchant_category', 'is_fraud']
    print(df[new_cols].head())
    
    print("\nNew Feature Distributions:")
    for col in new_cols[:-1]:
        print(f"\n--- {col} ---")
        print(df[col].value_counts(normalize=True).sort_index())

if __name__ == "__main__":
    main()