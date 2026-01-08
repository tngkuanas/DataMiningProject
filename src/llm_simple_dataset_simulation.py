import os
import random
import time
import json
from datetime import datetime, timedelta

import pandas as pd
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv

# --- Environment and API Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env. Please ensure you have a .env file with your API key.")
genai.configure(api_key=api_key)

def call_gemini(prompt, max_retries=3, initial_delay=5):
    """
    Calls the Gemini API with a prompt, handles JSON parsing, and implements retry logic.
    """
    model = genai.GenerativeModel("gemini-2.5-flash")
    for attempt in range(max_retries):
        try:
            resp = model.generate_content(contents=prompt)
            # Basic cleaning of the response text
            cleaned_text = resp.text.strip().replace("```json", "").replace("```", "")
            return json.loads(cleaned_text)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error decoding JSON on attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                print(f"Failed to parse JSON after {max_retries} attempts. Raw response: {resp.text}")
                return None
        except Exception as e:
            # Handle potential API rate limits or other errors
            if "quota" in str(e).lower() or "rate limit" in str(e).lower():
                delay = initial_delay * (2 ** attempt)
                print(f"Rate limit likely hit. Retrying in {delay}s...")
                time.sleep(delay)
            else:
                print(f"An unexpected error occurred: {e}")
                return None
    return None

def simulate_customers(n_customers=200):
    """Creates a simple dictionary of simulated customer profiles."""
    customers = {}
    for i in range(1, n_customers + 1):
        cust_id = f"CUST{i:04d}"
        customers[cust_id] = {
            "home_country": random.choice(["US", "GB", "CA", "AU", "DE", "FR"])
        }
    return customers

def create_prompt(batch_size, fraud_count, legit_count, customer_home_country):
    """Creates the anti-leakage prompt for the LLM."""
    return f"""
    You are a sophisticated financial transaction simulator. Your task is to generate a JSON list containing exactly {batch_size} realistic transaction scenarios.

    The list must contain:
    - {fraud_count} FRAUDULENT transaction scenarios.
    - {legit_count} LEGITIMATE transaction scenarios.
    Shuffle them randomly.

    For each scenario, provide a JSON object with ONLY the following structure:
    - "is_fraud": boolean
    - "transaction_amount": float (rounded to 2 decimal places)
    - "merchant_category": string (e.g., "Electronics", "Groceries", "Travel", "Services", "Online Retail")
    - "transaction_country": string (ISO 2-letter country code, e.g., "US", "CN", "GB", "MX", "NG")
    - "transaction_description": string (a short, 5-10 word description of the purchase)

    ANTI-LEAKAGE INSTRUCTIONS (VERY IMPORTANT):
    Your goal is to create a realistic dataset, not a perfect one. Do NOT create deterministic rules. Use the following probabilistic guidance:

    1. For FRAUDULENT scenarios:
       - `transaction_amount`: Should be DRAWN FROM A DISTRIBUTION that is SKEWED HIGH (e.g., most values between 100.00 and 2000.00).
       - `transaction_country`: There should be a HIGH PROBABILITY (around 70-80%) that this is DIFFERENT from the customer's home country ('{customer_home_country}').
       - `transaction_description`: The text should be MORE LIKELY to contain words related to urgency, anonymity, or luxury goods (e.g., "urgent international wire", "crypto purchase", "luxury watch order").

    2. For LEGITIMATE scenarios:
       - `transaction_amount`: Should be DRAWN FROM A DISTRIBUTION that is SKEWED LOW (e.g., most values between 5.00 and 150.00).
       - `transaction_country`: There should be a VERY HIGH PROBABILITY (around 95-99%) that this is the SAME as the customer's home country ('{customer_home_country}').
       - `transaction_description`: The text should be mundane and typical for everyday life (e.g., "coffee shop purchase", "monthly streaming subscription", "gas station fill-up").

    3. CRITICAL - FEATURE OVERLAP:
       - Ensure there is some overlap. A small number of legitimate transactions can have high amounts or suspicious descriptions. A small number of fraudulent transactions can appear normal. This is essential for realism.

    The final output MUST be a single, valid JSON array of {batch_size} objects. Do not include any other text, comments, or markdown.
    """

def main():
    # --- Simulation Parameters ---
    N_RECORDS = 2000
    BATCH_SIZE = 100
    FRAUD_RATE = 0.08
    N_CUSTOMERS = 200

    print("1. Simulating base customer profiles...")
    customers = simulate_customers(n_customers=N_CUSTOMERS)
    customer_ids = list(customers.keys())

    # --- Main Simulation Loop ---
    final_rows = []
    num_batches = (N_RECORDS + BATCH_SIZE - 1) // BATCH_SIZE
    
    print(f"2. Generating {N_RECORDS} records in {num_batches} batches via LLM...")
    with tqdm(total=N_RECORDS) as pbar:
        for i in range(num_batches):
            if len(final_rows) >= N_RECORDS:
                break

            fraud_scenarios_in_batch = int(BATCH_SIZE * FRAUD_RATE)
            legit_scenarios_in_batch = BATCH_SIZE - fraud_scenarios_in_batch

            # For simplicity, we provide one customer context per batch
            batch_customer_id = random.choice(customer_ids)
            batch_customer_country = customers[batch_customer_id]["home_country"]

            prompt = create_prompt(
                BATCH_SIZE,
                fraud_scenarios_in_batch,
                legit_scenarios_in_batch,
                batch_customer_country
            )

            scenario_batch = call_gemini(prompt)

            if not scenario_batch or not isinstance(scenario_batch, list):
                print(f"Warning: Skipping batch {i+1} due to invalid LLM response.")
                continue

            # Process the batch and add programmatically generated fields
            for scenario in scenario_batch:
                if len(final_rows) >= N_RECORDS:
                    break
                
                # For more variety, assign a random customer to each transaction
                customer_id = random.choice(customer_ids)
                
                # Generate timestamp sequentially
                timestamp = datetime(2025, 1, 1) + timedelta(days=len(final_rows)//10, hours=len(final_rows) % 24, minutes=random.randint(0, 59))

                final_rows.append({
                    "transaction_id": f"TX{len(final_rows):06d}",
                    "customer_id": customer_id,
                    "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    "transaction_amount": scenario.get("transaction_amount", 0.0),
                    "merchant_category": scenario.get("merchant_category", "Unknown"),
                    "transaction_country": scenario.get("transaction_country", "N/A"),
                    "transaction_description": scenario.get("transaction_description", ""),
                    "is_fraud": 1 if scenario.get("is_fraud", False) else 0
                })
                pbar.update(1)
            
            # Small delay to respect potential rate limits
            time.sleep(15)

    # --- Create DataFrame and Save ---
    print("\n3. Creating final DataFrame...")
    
    # Define the exact schema
    columns = [
        "transaction_id", "customer_id", "timestamp", "transaction_amount",
        "merchant_category", "transaction_country", "transaction_description", "is_fraud"
    ]
    df = pd.DataFrame(final_rows, columns=columns)

    output_dir = "data/raw"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'simple_llm_fraud_dataset.csv')

    df.to_csv(output_path, index=False)

    print(f"\nDataset successfully generated at '{output_path}'")
    print("Dataset shape:", df.shape)
    if not df.empty:
        print("\nSample of 5 records:")
        print(df.head())
        print("\nFraud distribution:")
        print(df['is_fraud'].value_counts(normalize=True))

if __name__ == "__main__":
    main()
