import google.generativeai as genai
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY not found in .env")
genai.configure(api_key=api_key)

print("Listing available models:")
for model in genai.list_models():
    print(f"Name: {model.name}")
    print(f"  Description: {model.description}")
    print(f"  Supported Generation Methods: {model.supported_generation_methods}")
    print(f"  Input Token Limit: {model.input_token_limit}")
    print(f"  Output Token Limit: {model.output_token_limit}")
    print("-" * 30)
