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
    Expects a JSON array response.
    """
    model = genai.GenerativeModel(model_name)
    
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(contents=prompt)
            # Clean the response to extract the JSON part
            cleaned_text = resp.text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_text)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error decoding JSON response on attempt {attempt + 1}: {e}")
            print(f"Raw response: {resp.text}")
            # Optional: a more aggressive cleaning attempt could be added here
            if attempt == max_retries - 1:
                return None # Failed to parse
        except Exception as e:
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                delay = initial_delay * (2 ** attempt)
                print(f"Rate limit hit. Retrying in {delay} seconds (attempt {attempt + 1}/{max_retries})...")
                time.sleep(delay)
            else:
                print(f"An unexpected error occurred: {e}")
                return None # Return None for unexpected errors
    
    print(f"Failed to get a valid response after {max_retries} attempts.")
    return None

def create_batch_prompt(items_list, task_description, item_key, categories):
    """Creates a prompt for batch processing."""
    # Create a JSON object with numbered keys for clear mapping
    input_json = json.dumps({i: item for i, item in enumerate(items_list)}, indent=2)
    
    prompt = f"""
    You will be given a JSON object containing multiple text entries to classify.
    Your task is to {task_description}.

    The possible categories are: {categories}.

    Respond with a single JSON object where the keys correspond to the input keys, and the value is the single-word category.
    Ensure your output is a valid JSON object and nothing else.

    Input:
    {input_json}

    Output:
    """
    return prompt

def batch_process_column(series, task_description, item_key, categories, batch_size):
    """Processes a pandas Series in batches using the LLM."""
    results = {}
    
    # Use tqdm to show progress over batches
    num_batches = math.ceil(len(series) / batch_size)
    with tqdm(total=num_batches, desc=f"Processing {item_key}s") as pbar:
        for i in range(0, len(series), batch_size):
            batch = series[i:i + batch_size]
            
            # For sentiment, handle empty notes before sending to LLM
            if item_key == "note":
                non_empty_batch = batch.dropna().loc[batch.str.strip() != '']
                empty_indices = batch.index.difference(non_empty_batch.index)
                for idx in empty_indices:
                    results[idx] = "neutral"
            else:
                non_empty_batch = batch

            if not non_empty_batch.empty:
                prompt = create_batch_prompt(non_empty_batch.tolist(), task_description, item_key, categories)
                batch_results = call_gemini_batch(prompt)

                if batch_results:
                    # Map results back to original DataFrame index
                    original_indices = non_empty_batch.index
                    for j, result_val in batch_results.items():
                        original_idx = original_indices[int(j)]
                        results[original_idx] = str(result_val).lower()
                else:
                    print(f"Warning: Batch starting at index {i} failed to process.")
                    # Fill failed batch with 'unknown'
                    for idx in non_empty_batch.index:
                        results[idx] = 'unknown'

            pbar.update(1)
            time.sleep(1) # Add a small delay between batches to help with rate limiting

    return pd.Series(results)


def main():
    input_path = 'data/synthetic_fraud_dataset.csv'
    output_path = 'data/synthetic_fraud_dataset_final.csv'
    
    print(f"Loading dataset from {input_path}...")
    df = pd.read_csv(input_path)
    print(f"Dataset loaded. Shape: {df.shape}")

    # Determine batch size to stay within 20 requests (10 per column)
    num_rows = len(df)
    max_requests_per_column = 10
    batch_size = math.ceil(num_rows / max_requests_per_column)
    print(f"Processing in batches of {batch_size} to meet request limit.")

    # --- Process Risk Category ---
    risk_task = 'categorize each transaction description into one of three risk levels: "high", "medium", or "low"'
    risk_categories = ["high", "medium", "low"]
    risk_results = batch_process_column(df['transaction_description'], risk_task, "description", risk_categories, batch_size)
    df['risk_category'] = risk_results.reindex(df.index).fillna('unknown')

    # --- Process Sentiment ---
    sentiment_task = 'analyze the sentiment of each customer support note and categorize it as "positive", "negative", or "neutral"'
    sentiment_categories = ["positive", "negative", "neutral"]
    sentiment_results = batch_process_column(df['customer_support_note'], sentiment_task, "note", sentiment_categories, batch_size)
    df['support_sentiment'] = sentiment_results.reindex(df.index).fillna('neutral') # Default to neutral for failed/empty

    # Clean up responses to ensure they are within the expected categories
    df['risk_category'] = df['risk_category'].apply(lambda x: x if x in risk_categories else 'unknown')
    df['support_sentiment'] = df['support_sentiment'].apply(lambda x: x if x in sentiment_categories else 'unknown')

    print(f"\nSaving final dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Feature extraction complete!")
    print(f"Output dataset shape: {df.shape}")
    print("\nSample of final dataset with new features:")
    print(df[['transaction_description', 'risk_category', 'customer_support_note', 'support_sentiment', 'is_fraud']].head())
    print("\nRisk Category Distribution:")
    print(df['risk_category'].value_counts(normalize=True))
    print("\nSentiment Distribution:")
    print(df['support_sentiment'].value_counts(normalize=True))

if __name__ == "__main__":
    main()