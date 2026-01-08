import pandas as pd
from pathlib import Path

def get_notebook_summaries():
    """Extracts insights from markdown cells of model notebooks."""
    summaries = {}
    notebook_dir = Path('notebooks')
    model_notebooks = [
        "Model_Baseline.ipynb", "Model_A.ipynb", "Model_B.ipynb", 
        "Model_C.ipynb", "Model_D.ipynb", "Model_E.ipynb"
    ]
    
    # A simple and imperfect way to find markdown summaries. 
    # A more robust solution would use a library like nbformat.
    for nb_name in model_notebooks:
        try:
            with open(notebook_dir / nb_name, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if '"## LLM Summary\n"' in content:
                # This is a very rough way to get the model name and summary text
                model_name_start = content.find('"# ') + 3
                model_name_end = content.find(' Model"')
                model_name = content[model_name_start:model_name_end]
                
                summary_start = content.rfind('"## LLM Summary\n"')
                summary_text = content[summary_start:]
                
                # Crude cleaning of JSON/markdown formatting
                summary_text = summary_text.replace('\n",', '\n').replace('"', '')
                summary_text = summary_text.replace('\n', '\n\n')
                
                summaries[model_name] = summary_text
        except (FileNotFoundError, IndexError):
            summaries[nb_name.replace('.ipynb', '')] = "Could not automatically extract summary."
            
    return summaries

def generate_report():
    """Generates the final project report in Markdown format."""
    
    print("Starting report generation...")
    
    # --- 1. Read Data ---
    try:
        readme_content = Path('README.md').read_text()
    except FileNotFoundError:
        readme_content = "Project objectives could not be loaded from README.md."

    try:
        schema_content = Path('schema.txt').read_text()
    except FileNotFoundError:
        schema_content = "Dataset schema could not be loaded from schema.txt."

    try:
        results_df = pd.read_csv('results/model_comparison.csv')
        results_table = results_df.to_markdown(index=False)
    except FileNotFoundError:
        results_table = "Results table could not be loaded from results/model_comparison.csv."

    business_insights = get_notebook_summaries()

    # --- 2. Build Report Sections ---
    
    report = f"""
# Final Report: Fraud Detection Analysis

## 1. Project Objectives

{readme_content}

---

## 2. Dataset Details

### 2.1. Original Dataset
The initial dataset is sourced from `data/raw/simple_llm_fraud_dataset.csv`. It contains simulated transactional data with text descriptions.

### 2.2. Schema
The following schema describes the features available in the final machine-learning-ready dataset:

```
{schema_content}
```

---

## 3. Exploratory Data Analysis (EDA)
EDA was performed in `notebooks/exploratory_data_analysis.ipynb`. The key goals were to understand the distribution of fraudulent vs. non-fraudulent transactions, analyze the relationships between different features, and identify initial patterns. This step was crucial for informing the feature engineering process. *(Note: Visualizations from the notebook should be embedded here in the final report.)*

---

## 4. Feature Engineering
Feature engineering was a multi-step process designed to convert raw data, especially text, into numerical features suitable for machine learning.

- **LLM-driven Features**: The `llm_feature_engineering.py` script and `feature_extraction.ipynb` notebook were used to process the transaction descriptions using a large language model. This generated features like `transaction_purpose_encoded`, `entity_has_crypto`, and `suspicious_word_score`.
- **TF-IDF Vectorization**: Term Frequency-Inverse Document Frequency (TF-IDF) was used to convert the text descriptions into a numerical format, capturing the importance of words.
- **Behavioral Features**: Features like `consistency_score` and `transaction_frequency` were engineered to model user behavior over time.
- **Final Cleaning**: The `final_cleaning_and_preparation.py` script performed the last cleaning steps to create the `final_ml_ready_dataset.csv`.

---

## 5. Modelling
A variety of models were trained to identify the most effective approach for this fraud detection problem. Each model was saved as a script in the `src/` directory and documented in a corresponding notebook in `notebooks/`.

The models evaluated include:
- **Decision Tree (`Model_Baseline`)**: An interpretable, rule-based model.
- **Random Forest (`Model_A`)**: An ensemble of decision trees for improved robustness.
- **XGBoost (`Model_B`)**: A powerful gradient boosting framework.
- **Logistic Regression (`Model_C`)**: A linear model establishing a baseline.
- **K-Nearest Neighbors (`Model_D`)**: A non-parametric, instance-based model.
- **Neural Network (`Model_E`)**: A multi-layer perceptron for capturing non-linear patterns.

---

## 6. Results

The performance of each model was evaluated on a held-out test set. The key metrics are summarized below:

{results_table}

---

## 7. Business Insights & Model Summaries
"""

    for model_name, summary in business_insights.items():
        report += f"### {model_name} Model\n{summary}\n\n---\n\n"

    report += """
## 8. AI Usage Disclosure
This project was developed with the assistance of an AI coding agent (Gemini). The AI's role included:
- **Code Generation**: Writing boilerplate code for scripts and notebooks.
- **Debugging**: Identifying and fixing errors, such as the `ModuleNotFoundError` in notebooks.
- **Analysis & Summarization**: Generating textual summaries and insights for the model evaluation sections based on typical model performance characteristics.
- **Report Generation**: Creating the Python script to automate the generation of this report.

## 9. GitHub Contribution Summary
To generate a summary of contributions for this project, you can use the following Git command. This command lists the number of commits per author, providing an overview of the development activity.

```bash
# To get a short summary of commits per author
git shortlog -s -n

# To get a more detailed log, you can run:
git log --pretty=format:"%h - %an, %ar : %s"
```

This concludes the final report.
"""

    # --- 3. Write Report File ---
    report_path = Path('Final_Report.md')
    report_path.write_text(report)
    print(f"Report successfully generated at: {report_path}")

if __name__ == '__main__':
    generate_report()
