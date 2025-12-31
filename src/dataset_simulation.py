import os
import random
import textwrap
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm
import google.generativeai as genai
from dotenv import load_dotenv
import json
import time

# --- Environment and API Setup ---
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
genai.configure(api_key=api_key)

def call_gemini(prompt, model="models/gemini-3-flash-preview"):
    """Calls the Gemini API with a short delay."""
    model = genai.GenerativeModel(model)
    resp = model.generate_content(contents=prompt)
    time.sleep(2)  # Short delay between batch calls
    return resp.text

# --- Customer Simulation (with Risky Profiles) ---
def simulate_customers(n_customers=300):
    """Creates a dictionary of simulated customer profiles with a subset of high-risk users."""
    customers = {}
    account_start_date = datetime(2024, 1, 1)
    for i in range(1, n_customers + 1):
        cust_id = f"CUST{i:04d}"
        if random.random() < 0.1:
            user_risk_score = random.uniform(0.7, 1.0)
            kyc_verified = random.choice([True, False, False])
        else:
            user_risk_score = random.uniform(0.0, 0.3)
            kyc_verified = random.choice([True, True, False])

        customers[cust_id] = {
            "account_creation_date": account_start_date + timedelta(days=random.randint(0, 365)),
            "user_risk_score": user_risk_score, "initial_risk_score": user_risk_score,
            "kyc_verified": kyc_verified,
            "usual_country": random.choice(["US", "GB", "CA", "AU", "DE"]),
            "devices": random.sample(["mobile", "desktop", "tablet"], k=random.randint(1, 2)),
            "account_balance": round(random.uniform(50, 5000), 2),
            "transactions": [], "previous_fraud_count": 0, "countries_used": set()
        }
    return customers

# --- Simulation Parameters ---
N_RECORDS = 1000
BATCH_SIZE = 50
N_CUSTOMERS = 300
SIMULATION_START_DATE = datetime(2025, 1, 1)
SIMULATION_END_DATE = SIMULATION_START_DATE + timedelta(days=90)
FRAUD_RATE = 0.08

# --- Main Simulation Logic ---
print("Simulating customer profiles...")
customers = simulate_customers(n_customers=N_CUSTOMERS)

print("Pre-generating transaction timestamps...")
transaction_timestamps = []
records_generated = 0
while records_generated < N_RECORDS:
    burst_chance = 0.3
    if random.random() < burst_chance and (N_RECORDS - records_generated) > 3:
        burst_size = random.randint(2, 4)
        burst_time_base = SIMULATION_START_DATE + timedelta(seconds=random.randint(0, int((SIMULATION_END_DATE - SIMULATION_START_DATE).total_seconds() * 0.9)))
        for j in range(burst_size):
            offset = timedelta(minutes=random.randint(1, 180)) if j > 0 else timedelta(seconds=0)
            transaction_timestamps.append(burst_time_base + offset)
        records_generated += burst_size
    else:
        random_seconds = random.randint(0, int((SIMULATION_END_DATE - SIMULATION_START_DATE).total_seconds()))
        date = SIMULATION_START_DATE + timedelta(seconds=random_seconds)
        transaction_timestamps.append(date)
        records_generated += 1
transaction_timestamps.sort()

# --- Batch Generation ---
final_rows = []
num_batches = (N_RECORDS + BATCH_SIZE - 1) // BATCH_SIZE
fraud_archetypes = ['account_takeover_high_value_purchase', 'credential_stuffing_small_tests', 'card_not_present_overseas_goods', 'internal_transfer_to_mule_account']
transaction_types = ["transfer", "online purchase", "withdrawal"]
merchant_categories = ["Electronics", "Food", "Clothing", "Utilities", "Travel", "Entertainment", "Services", "Crypto", "Gambling"]
payment_channels = ["bank", "e-wallet", "card"]

print(f"Generating {N_RECORDS} records in {num_batches} batches of {BATCH_SIZE}...")
for i in tqdm(range(num_batches), desc="Generating Batches"):
    fraud_scenarios_in_batch = int(BATCH_SIZE * FRAUD_RATE)
    legit_scenarios_in_batch = BATCH_SIZE - fraud_scenarios_in_batch

    prompt = f"""
    You are a sophisticated financial transaction simulator. Your task is to generate a JSON list containing exactly {BATCH_SIZE} realistic transaction scenarios.

    The list must contain:
    - {fraud_scenarios_in_batch} FRAUDULENT transaction scenarios.
    - {legit_scenarios_in_batch} LEGITIMATE transaction scenarios.
    Shuffle them randomly.

    For each scenario, provide a JSON object with the following structure:
    - `is_fraud_scenario`: Boolean (true for fraud, false for legitimate).
    - `fraud_archetype`: (String, one of {fraud_archetypes} if fraudulent, else null).
    - `merchant_name`: Plausible business name.
    - `merchant_category`: One of {merchant_categories}.
    - `transaction_description`: Short (5-15 words).
    - `transaction_type`: One of {transaction_types}.
    - `payment_channel`: One of {payment_channels}.
    - `device_type_suggestion`: Suggest 'mobile', 'desktop', or 'tablet'.
    - `amount_as_percent_of_balance`: A float between 0.001 and 1.0.
    - `transaction_hour_suggestion`: An integer hour (0-23).
    - `location_match_suggestion`: Boolean (true/false).
    - `customer_support_note`: Optional brief note.

    RULES:
    1. For FRAUDULENT scenarios: BIAS towards high-risk categories, use URGENT/CONFUSED language, suggest UNUSUAL hours/amounts, and set `location_match_suggestion` to `false`.
    2. For LEGITIMATE scenarios: Generate patterns of normal, everyday activity with neutral text and plausible parameters.
    3. The final output MUST be a single, valid JSON array of {BATCH_SIZE} objects. Do not include any other text, comments, or markdown.
    """

    ai_response_text = call_gemini(prompt)
    ai_response_text = ai_response_text.strip().replace("```json", "").replace("```", "")
    
    try:
        scenario_batch = json.loads(ai_response_text)
        if not isinstance(scenario_batch, list) or len(scenario_batch) != BATCH_SIZE:
            raise ValueError(f"Expected a list of {BATCH_SIZE} items, but got {len(scenario_batch)}")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"\nError processing batch {i+1}: {e}. Skipping this batch.")
        continue

    start_index = i * BATCH_SIZE
    for j, scenario in enumerate(scenario_batch):
        record_index = start_index + j
        if record_index >= len(transaction_timestamps): break

        tx_date = transaction_timestamps[record_index]
        cust_id = f"CUST{random.randint(1, N_CUSTOMERS):04d}"
        customer = customers[cust_id]

        is_fraud_scenario = scenario.get('is_fraud_scenario', False)
        suggested_amount_percent = max(0.001, min(1.0, scenario.get("amount_as_percent_of_balance", 0.05)))
        transaction_amount = round(customer['account_balance'] * suggested_amount_percent, 2)
        
        account_balance_before = customer['account_balance']
        if transaction_amount <= account_balance_before:
            account_balance_after = round(account_balance_before - transaction_amount, 2)
        else:
            account_balance_after = account_balance_before
            transaction_amount = 0.0

        transaction_hour = scenario.get("transaction_hour_suggestion", tx_date.hour)
        tx_datetime = tx_date.replace(hour=transaction_hour, minute=random.randint(0,59), second=random.randint(0,59))
        
        last_tx_date = customer['transactions'][-1]['date'] if customer['transactions'] else customer['account_creation_date']
        time_since_last_transaction_hr = round((tx_datetime - last_tx_date).total_seconds() / 3600, 2)
        
        tx_last_24h = [tx for tx in customer['transactions'] if (tx_datetime - tx['date']).total_seconds() < 86400]
        transaction_frequency_24h = len(tx_last_24h)
        tx_last_7d = [tx for tx in customer['transactions'] if (tx_datetime - tx['date']).total_seconds() < 86400 * 7]
        avg_transaction_amount_7d = round(np.mean([t['amount'] for t in tx_last_7d]), 2) if tx_last_7d else 0
        num_failed_transactions_7d = sum(1 for _ in tx_last_7d if random.random() < customer['user_risk_score'] * 0.03)

        location_match_bool = scenario.get("location_match_suggestion", True)
        if not location_match_bool:
            customer['countries_used'].add(random.choice(["CN", "RU", "NG", "IN", "BR", "MX", "ZA"]))

        if is_fraud_scenario and transaction_amount > 0:
            customer['user_risk_score'] = min(1.0, customer['user_risk_score'] * 1.2 + 0.1)

        final_rows.append([
            f"TX{record_index:06d}", tx_datetime.strftime("%Y-%m-%d %H:%M:%S"), cust_id,
            transaction_amount, account_balance_before, account_balance_after,
            transaction_frequency_24h, avg_transaction_amount_7d, num_failed_transactions_7d,
            time_since_last_transaction_hr, transaction_hour, 1 if tx_datetime.weekday() >= 5 else 0,
            (tx_datetime - customer['account_creation_date']).days, customer['user_risk_score'], len(customer['devices']), len(customer['countries_used']),
            customer['previous_fraud_count'], 1 if customer['kyc_verified'] else 0,
            scenario.get("transaction_type", "online purchase"),
            scenario.get("merchant_category", "Services"),
            scenario.get("device_type_suggestion", random.choice(customer['devices'])),
            1 if location_match_bool else 0,
            scenario.get("payment_channel", "card"),
            scenario.get("transaction_description", "Default description"),
            scenario.get("merchant_name", "Default Merchant"),
            scenario.get("customer_support_note", ""),
            1 if is_fraud_scenario else 0
        ])
        
        if transaction_amount > 0:
            customer['account_balance'] = account_balance_after
            if is_fraud_scenario:
                customer['previous_fraud_count'] += 1
            customer['transactions'].append({'date': tx_datetime, 'amount': transaction_amount})

# --- Create DataFrame and Save ---
print("Creating final DataFrame...")
columns = [
    "transaction_id", "timestamp", "customer_id",
    "transaction_amount", "account_balance_before", "account_balance_after",
    "transaction_frequency_24h", "avg_transaction_amount_7d", "num_failed_transactions_7d",
    "time_since_last_transaction_hr", "transaction_hour", "is_weekend",
    "account_age_days", "user_risk_score", "num_devices_used", "num_countries_used",
    "previous_fraud_count", "kyc_verified",
    "transaction_type", "merchant_category", "device_type", "location_match", "payment_channel",
    "transaction_description", "merchant_name", "customer_support_note", "is_fraud"
]

df = pd.DataFrame(final_rows, columns=columns)

output_dir = "data/raw"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'synthetic_fraud_dataset.csv')

df.to_csv(output_path, index=False)

print(f"\nDataset successfully generated at '{output_path}'")
print("Dataset shape:", df.shape)
if not df.empty:
    print("\nSample of 5 records:")
    print(df.head())
    print("\nFraud distribution:")
    print(df['is_fraud'].value_counts(normalize=True))
