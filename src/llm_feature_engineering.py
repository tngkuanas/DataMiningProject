import os
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv
import time
import json
from tqdm.auto import tqdm
import math
from pathlib import Path

# --- Environment and API Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
genai.configure(api_key=api_key)

# --- LLM and Rate Limit Configuration ---
# Using a powerful model for complex extraction.
# Note: User mentioned 'gemini-3-flash', which is not a recognized public model.
# Using gemini-1.5-pro as a capable and reliable alternative.
LLM_MODEL = "gemini-3-flash-preview"
TIME_BETWEEN_REQUESTS = 15 # To stay under 5 RPM limit
MAX_DAILY_REQUESTS = 18 # Hard stop to protect budget
request_count = 0

def call_gemini_batch(prompt, max_retries=1):
    """
    Calls the Gemini API with a batch prompt, retry logic, and a hard request limit.
    """
    global request_count
    if request_count >= MAX_DAILY_REQUESTS:
        raise RuntimeError(f"Daily request limit of {MAX_DAILY_REQUESTS} reached. Aborting.")

    model = genai.GenerativeModel(LLM_MODEL)
    for attempt in range(max_retries):
        request_count += 1
        print(f"Making API Call #{request_count}...")
        try:
            resp = model.generate_content(contents=prompt)
            cleaned_text = resp.text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_text)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error decoding JSON on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to parse JSON. Raw response: {resp.text}")
                return None
        except Exception as e:
            if ("quota" in str(e).lower() or "rate limit" in str(e).lower()) and attempt < max_retries - 1:
                delay = TIME_BETWEEN_REQUESTS * (attempt + 1)
                print(f"Rate limit hit. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"An unrecoverable error occurred, aborting batch. Error: {e}")
                return None
    print(f"Failed to get a valid response after {max_retries} attempts.")
    return None

def create_combined_prompt(batch_df):
    """Creates a single, combined prompt for both consistency score and entity flags."""
    input_dict = batch_df[['transaction_description', 'merchant_category']].to_dict(orient='index')
    input_json = json.dumps(input_dict, indent=2)
    entity_list = ['payment_instruments', 'gift_cards', 'crypto', 'subscriptions', 'refunds', 'P2P', 'digital_goods']
    
    entity_flags_format = {f"entity_has_{entity}": "boolean" for entity in entity_list}
    output_format_json = {
        "consistency_score": "integer (1-10)",
        "entity_flags": entity_flags_format
    }
    output_format_str = json.dumps(output_format_json, indent=4)

    return f"""
    You are a data analysis expert. For each numbered item below, which contains a 'transaction_description' and a 'merchant_category', perform two tasks:
    1.  Generate a 'consistency_score': An integer from 1 (mismatched) to 10 (perfectly aligned).
    2.  Extract 'entity_flags': Determine if the description contains any of these entities: {entity_list}.

    Respond with a single JSON object where the keys are the original index numbers from the input.
    The value for each key must be another JSON object with the following structure:
    {output_format_str}

    Input:
    {input_json}

    Output:
    """


def main():
    input_path = 'data/raw/simple_llm_fraud_dataset.csv'
    output_path = 'data/processed/simple_llm_fraud_dataset_features.csv'

    print(f"Loading dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{input_path}'. Please generate the dataset first.")
        return

    print(f"Dataset loaded. Shape: {df.shape}")

    # --- Programmatic Feature Engineering (No LLM) ---
    print("\nAdding new programmatic features...")
    df['timestamp'] = pd.to_datetime(df['timestamp']) # Ensure timestamp is datetime
    df['description_length'] = df['transaction_description'].apply(len)
    df['transaction_frequency'] = df.groupby('customer_id')['transaction_id'].transform('count')
    print("Programmatic features 'description_length' and 'transaction_frequency' added.")

    # For 2619 rows, a batch size of 300 results in 9 requests total.
    # This is well under the 18 RPD hard stop and respects TPM/RPM.
    batch_size = 300
    num_batches = math.ceil(len(df) / batch_size)

    # --- Combined Feature Generation ---
    print("\n--- Generating LLM features (Consistency Score & Entity Flags) ---")
    all_entities = ['payment_instruments', 'gift_cards', 'crypto', 'subscriptions', 'refunds', 'P2P', 'digital_goods']
    
    # Initialize all new columns
    df['consistency_score'] = 5 # Default neutral score
    for entity in all_entities:
        df[f"entity_has_{entity}"] = 0

    with tqdm(total=num_batches, desc="Combined Feature Generation") as pbar:
        for i in range(num_batches):
            batch_df = df.iloc[i*batch_size : (i+1)*batch_size]
            prompt = create_combined_prompt(batch_df)
            
            try:
                batch_results = call_gemini_batch(prompt)
            except RuntimeError as e:
                print(f"Aborting script: {e}")
                break

            if batch_results:
                for original_idx, result_data in batch_results.items():
                    original_idx = int(original_idx)
                    
                    # Process consistency score
                    score = result_data.get('consistency_score', 5)
                    df.loc[original_idx, 'consistency_score'] = int(score)

                    # Process entity flags
                    flags = result_data.get('entity_flags', {})
                    if isinstance(flags, dict):
                        for col_name, value in flags.items():
                            if col_name in df.columns and isinstance(value, bool):
                                df.loc[original_idx, col_name] = int(value)
            else:
                print(f"Warning: Batch starting at index {i*batch_size} failed and was skipped.")
            
            pbar.update(1)
            # No time.sleep here as it's handled in the retry logic if needed,
            # and the user wants it to be fast. The RPM is managed by API time.
            # Correction: The user's rate limit is 5 RPM. The API call is fast.
            # The sleep is required to not hit the RPM limit.
            time.sleep(TIME_BETWEEN_REQUESTS)

    print("LLM-based feature generation complete.")

    # --- Finalizing and Saving ---
    print(f"\nSaving final dataset with new features to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print("Feature engineering complete!")
    print(f"Output dataset shape: {df.shape}")

    new_cols = ['consistency_score', 'description_length', 'transaction_frequency'] + [f"entity_has_{e}" for e in all_entities]
    print("\nSample of new features:")
    print(df[new_cols + ['is_fraud']].head())

if __name__ == "__main__":
    main()
