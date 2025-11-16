# utils.py
"""
Utilities functions.
"""

import os
import re
import time
import json
import pandas as pd
from pathlib import Path
from datetime import date, datetime
from sqlalchemy import create_engine, Date
from dotenv import load_dotenv
import openai


# --------------------------
# Load environment variables
# --------------------------
project_root = Path.cwd().parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "market")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

# --------------------------
# Database connection
# --------------------------
def get_db_engine():
    return create_engine(f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}")

# --------------------------
# Incremental stock sector classification
# --------------------------
def classify_new_stocks_to_sectors(batch_size: int = 100, allowed_sectors: list = None):
    """
    Incrementally classify stocks into sectors using OpenAI LLM.
    Only new stocks not yet in 'stock_sectors' table are classified and uploaded.
    """
    if allowed_sectors is None:
        allowed_sectors = [
            "Agriculture", "Automobile", "Carbon Products", "Cement", "Ceramics", "Chemicals",
            "Construction", "Consumer Products", "Defense", "Diversified", "Education", "Electricals",
            "Energy", "Entertainment", "Environmental Services", "Financial Services", "Food & Beverage",
            "Healthcare", "Hospitality", "Industrial Equipment", "Jewelry", "Logistics", "Manufacturing",
            "Metals", "Paper", "Plastics", "Real Estate", "Retail", "Rubber", "Shipping", "Technology",
            "Telecommunications", "Textiles", "Trading"
        ]

    engine = get_db_engine()
    stock_list = pd.read_sql("SELECT * FROM stock_list", engine)

    # Load existing sectors
    try:
        existing_sectors = pd.read_sql("SELECT tradingsymbol FROM stock_sectors", engine)
        existing_symbols = set(existing_sectors['tradingsymbol'].tolist())
    except Exception:
        existing_symbols = set()

    # Find new stocks
    new_stocks = stock_list[~stock_list['tradingsymbol'].isin(existing_symbols)]
    if new_stocks.empty:
        print("No new stocks to classify.")
        return pd.DataFrame()  # nothing to do

    df_stocks = new_stocks.copy()
    sectors_result = []

    sector_list_str = ", ".join(allowed_sectors)

    for i in range(0, len(df_stocks), batch_size):
        batch = df_stocks.iloc[i:i + batch_size]

        prompt = f"""
Assign exactly one sector to each of the following stocks.
Choose the sector ONLY from this allowed list (do not invent new sectors):

{sector_list_str}

Return the output strictly as a JSON array of objects with fields:
- tradingsymbol
- name
- sector

Stocks:
"""
        for _, row in batch.iterrows():
            name = row['name'] if row['name'] else row['tradingsymbol']
            prompt += f"{row['tradingsymbol']} - {name}\n"

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        llm_text = response['choices'][0]['message']['content']

        # Extract JSON array from the model reply
        match = re.search(r"\[\s*{.*}\s*\]", llm_text, re.DOTALL)
        if match:
            try:
                batch_sectors = json.loads(match.group())
                sectors_result.extend(batch_sectors)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON in batch {i}-{i + batch_size}: {e}")
                print("LLM response:", llm_text)
        else:
            print(f"No JSON found in batch {i}-{i + batch_size}")
        
        time.sleep(1)  # avoid rate limit

    # Convert result to DataFrame
    df_sectors = pd.DataFrame(sectors_result)
    df_sectors['Date_Update'] = datetime.now().date()
    cols = ['Date_Update'] + [c for c in df_sectors.columns if c != 'Date_Update']
    stocktypes = df_sectors[cols]

    # Append new sectors to database
    stocktypes.to_sql(
        'stock_sectors',
        engine,
        if_exists='append',  # append instead of replace
        index=False,
        dtype={'Date_Update': Date()}
    )

    print(f"Classified and uploaded {len(df_sectors)} new stocks.")
    return df_sectors


