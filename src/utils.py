# utils.py
"""
Utilities functions.
"""

# Standard library
import json
import os
import re
import time
from datetime import datetime, date, timedelta, time as dt_time
from pathlib import Path
import traceback
import requests
import boto3

# Third-party libraries
import numpy as np
from openai import OpenAI
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from tqdm import tqdm
# from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from kite_connect import ZerodhaConnector
from kiteconnect.exceptions import InputException
from sqlalchemy import (
    Table, Column, Integer, String, Boolean, DateTime, MetaData, create_engine, Date, select
)
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient

# from pyspark.sql import SparkSession
# from pyspark.sql.functions import current_timestamp, lit, col


# --------------------------
# Load environment variables
# --------------------------
project_root = Path.cwd().parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)

session = boto3.session.Session()
client = session.client(service_name='secretsmanager',
                        region_name="eu-north-1" )
secrets = client.get_secret_value(SecretId='StockZerodhaRelated')['SecretString']
secrets = eval(secrets)

DB_USER = None
DB_PASSWORD = None
DB_SERVER = "databricks-zerodha"
DB_HOST = "databricks-zerodha.postgres.database.azure.com"
DB_PORT = "5432"
DB_NAME = "market"
OPENAI_API_KEY = secrets['OPENAI-API-KEY']
API_KEY = secrets['KITE-API-KEY']
API_SECRET = secrets['KITE-API-SECRET']
ACCESS_TOKEN = os.getenv("KITE-ACCESS-TOKEN", None)
client_openai = OpenAI(api_key=OPENAI_API_KEY)
pushover_api_token = secrets['PUSHOVER-API-TOKEN']
pushover_userkey = secrets['PUSHOVER-USER-KEY']
BUCKET = "stockzerodha"

# --------------------------
# Database connection
# --------------------------
def get_db_engine():
    return create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        connect_args={"sslmode": "require"}
    )


# --------------------------
# Check if market is open
# --------------------------
def is_market_open():
    # India timezone
    ist = pytz.timezone("Asia/Kolkata")
    
    # Current time in IST
    now_ist = datetime.now(ist).time()
    
    # NSE market hours in IST
    return dt_time(9, 15) <= now_ist <= dt_time(15, 30)


# ---- Compute indicators function ----
def compute_indicators(df):
    """
    Compute technical indicators:
    MA20, MA50, RSI14, ATR14, MACD + Signal, OBV, Avg Daily Volume, Volatility, Price Change 1w/1m
    """
    if df.empty:
        # Return as-is if no data
        return df

    # Ensure 'date' is index
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)

    # ---- Moving Averages ----
    df['MA20'] = df['close'].rolling(window=20).mean()
    df['MA50'] = df['close'].rolling(window=50).mean()
    
    # ---- RSI14 ----
    delta = df['close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    RS = avg_gain / avg_loss
    df['RSI14'] = 100 - (100 / (1 + RS))
    
    # ---- ATR14 ----
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    TR = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR14'] = TR.rolling(14).mean()
    
    # ---- MACD + Signal ----
    EMA12 = df['close'].ewm(span=12, adjust=False).mean()
    EMA26 = df['close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = EMA12 - EMA26
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ---- OBV ----
    df['OBV'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    
    # ---- Avg Daily volume and Volatility in one month ----
    df['avg_daily_volume'] = df['volume'].rolling(20).mean()  # last 20 trading days
    df['volatility_1m'] = df['close'].pct_change().rolling(22).std()  # ~22 trading days

    # ---- Price change metrics ----
    df['price_change_1w'] = df['close'].pct_change(5)   # ~1 week
    df['price_change_1m'] = df['close'].pct_change(22)  # ~1 month

    return df

# Load latest instrument list once (daily)
df_instr = pd.read_csv("https://api.kite.trade/instruments")

def get_instrument_token(symbol, exchange="NSE"):    
    row = df_instr[(df_instr["tradingsymbol"] == symbol) & (df_instr["exchange"] == 'NSE')]
    if row.empty:
        raise ValueError(f"{symbol} not found on {exchange}")
    return int(row["instrument_token"].iloc[0])

# ---- Function to fetch OHLCV ----
def fetch_ohlcv_from_zerodha(symbol, from_date, to_date, interval, exchange, kite):
    token = get_instrument_token(symbol, exchange)
    data = kite.historical_data(
        instrument_token=token,
        from_date=from_date,
        to_date=to_date,
        interval=interval
    )

    if len(data) > 1:
        df = pd.DataFrame(data)[['date','open','high','low','close','volume']]
    else:
        df = pd.DataFrame(columns=['open','high','low','close','volume'])
    return df

def score_stock(df, latest_row):
    score = 0
    
    # 1. Liquidity
    score += min(latest_row['avg_daily_volume'] / 1_000_000, 1) * 10  # scale to 0–10
    
    # 2. Volatility
    vol = latest_row['volatility_1m']
    if 0.02 <= vol <= 0.08:  # example range for "moderate" volatility
        score += 10
    
    # 3. Momentum
    score += max(latest_row['price_change_1w'], 0) * 50  # scale appropriately
    score += max(latest_row['price_change_1m'], 0) * 50
    
    # 4. Trend: MA20 > MA50
    if latest_row['MA20'] > latest_row['MA50']:
        score += 10
    
    # 5. RSI
    if 40 <= latest_row['RSI14'] <= 60:
        score += 5
    
    # 6. MACD bullish
    if latest_row['MACD'] > latest_row['MACD_signal']:
        score += 5
    
    # 7. OBV rising
    if df['OBV'].iloc[-1] > df['OBV'].iloc[-2]:
        score += 5
    
    return score


def get_last_saved_date(symbol, engine):
    query = f"""
        SELECT MAX("date") AS max_date 
        FROM stock_timeseries 
        WHERE tradingsymbol = '{symbol}'
    """
    df = pd.read_sql(query, engine)
    return df['max_date'][0]


def get_recent_news_sentiment(stock_name, max_articles=50):
    
    # Initialize sentiment analyzer
    sia = SentimentIntensityAnalyzer()
    
    # Fetch news
    googlenews = GoogleNews(lang='en', region='IN')
    googlenews.search(f"{stock_name} Stock")
    
    news_items = googlenews.result()[:max_articles]  # top articles
    df = pd.DataFrame(news_items)

    # --- Filter for last 24 hours ---
    now = datetime.now()
    filtered_rows = []

    for _, row in df.iterrows():
        date_str = row['date']
        if 'hour' in date_str:
            hours_ago = int(date_str.split()[0])
            article_time = now - timedelta(hours=hours_ago)
        elif 'day' in date_str:
            days_ago = int(date_str.split()[0])
            article_time = now - timedelta(days=days_ago)
        else:
            # Skip unknown formats
            continue
        
        if now - article_time <= timedelta(days=1):
            filtered_rows.append(row)

    df = pd.DataFrame(filtered_rows)

    if df.empty:
        return 0, df
    # --- Calculate sentiment ---
    df['sentiment'] = df.apply(lambda row: sia.polarity_scores(row['title'] + " " + row.get('desc',''))['compound'], axis=1)

    # Average sentiment
    avg_sentiment = df['sentiment'].mean().round(2) if not df.empty else 0
    return avg_sentiment, df

def get_trigger_price_close_to_ltp(ltp, buy=True, min_diff=0.1):
    """
    Returns a valid trigger price for GTT.
    min_diff: minimum required difference from LTP.
    """
    if buy:
        return round(ltp + min_diff, 2)
    else:
        return round(ltp - min_diff, 2)
    

def buy_stock(tradingsymbol, quantity=1, amo=True, product="CNC"):
    """
    Places a buy order for a given trading symbol.
    - Tries an AMO or regular order first.
    - If AMO fails due to maintenance, automatically places a GTT order at the current market price.

    Parameters:
        tradingsymbol (str): Stock trading symbol
        quantity (int): Number of shares to buy
        amo (bool): Whether to place the order as AMO (after market)
        product (str): 'CNC' for delivery, 'MIS' for intraday

    Returns:
        dict: {
            'success': bool,
            'symbol': str,
            'order_id': str or None,
            'gtt_id': str or None,
            'message': str
        }
    """

    kite_client = ZerodhaConnector(API_KEY, ACCESS_TOKEN)
    kite = kite_client.kite

    try:
        variety = kite.VARIETY_AMO if amo else kite.VARIETY_REGULAR

        # Place regular or AMO order
        order_id = kite.place_order(
            variety=variety,
            exchange=kite.EXCHANGE_NSE,
            tradingsymbol=tradingsymbol,
            transaction_type=kite.TRANSACTION_TYPE_BUY,
            quantity=quantity,
            order_type=kite.ORDER_TYPE_MARKET,
            product=product
        )

        return {
            "success": True,
            "symbol": tradingsymbol,
            "order_id": order_id,
            "gtt_id": None,
            "message": f"Order placed successfully for {tradingsymbol} | ID: {order_id}"
        }

    except InputException as e:
        # Extract message
        if isinstance(e.args[0], dict):
            message = e.args[0].get('message', str(e))
        else:
            message = str(e)
        clean_message = message.split('[')[0].strip()

        # Check if the error is due to AMO maintenance and trigger fallback to GTT
        if "AMO orders cannot be placed" in clean_message:
            try:
                # Fetch current market price for trigger
                ltp = kite.ltp(f"NSE:{tradingsymbol}")[f"NSE:{tradingsymbol}"]["last_price"]

                gtt_id = kite.place_gtt(
                        trigger_type=kite.GTT_TYPE_SINGLE,
                        tradingsymbol=tradingsymbol,
                        exchange=kite.EXCHANGE_NSE,
                        trigger_values=[get_trigger_price_close_to_ltp(ltp, buy=True)],          # list of trigger prices
                        last_price=ltp,                # current market price
                        orders=[{
                            "transaction_type": "BUY",
                            "quantity": quantity,
                            "order_type": "MARKET",
                            "product": product,
                            "price":ltp
                        }]
                    )

                return {
                    "success": True,
                    "symbol": tradingsymbol,
                    "order_id": None,
                    "gtt_id": gtt_id,
                    "message": f"AMO failed due to maintenance. GTT order placed at current market price {ltp}"
                }
            except Exception as gtt_e:
                return {
                    "success": False,
                    "symbol": tradingsymbol,
                    "order_id": None,
                    "gtt_id": None,
                    "message": f"AMO failed, and GTT fallback also failed: {str(gtt_e)}"
                }

        # Any other InputException
        return {
            "success": False,
            "symbol": tradingsymbol,
            "order_id": None,
            "gtt_id": None,
            "message": clean_message
        }

    except Exception as e:
        return {
            "success": False,
            "symbol": tradingsymbol,
            "order_id": None,
            "gtt_id": None,
            "message": f"{type(e).__name__}: {str(e)}"
        }

# IST timezone
IST = pytz.timezone("Asia/Kolkata")
def now_ist():
    return datetime.now(IST)

def process_and_store_intended_orders(top_stocks, amo_flag=True):
    """
    Place AMO buy orders for top_stocks and store intended order info in DB.
    - Prevents duplicate orders for the same tradingsymbol on the same day.
    - Stores Zurich server time for requested_at.
    - Captures Kite response info or exception messages for logging.
    """

    # --- DB setup ---
    engine = get_db_engine()
    metadata = MetaData()

    # Define table
    stock_intended_orders = Table(
        "stock_intended_orders",
        metadata,
        Column("id", Integer, primary_key=True),
        Column("tradingsymbol", String, nullable=False),
        Column("quantity", Integer, nullable=False),
        Column("amo", Boolean, nullable=False),
        Column("product", String, nullable=False),
        Column("order_type", String, default="MARKET"),
        Column("requested_at", DateTime, nullable=False),   # Zurich server time
        Column("order_id", String),
        Column("status", String),
        Column("error_message", String),
        Column("kite_status", String),
        Column("kite_message", String),
        Column("filled_quantity", Integer),
        Column("average_price", String)
    )

    # Create table if not exists
    metadata.create_all(engine)

    # --- Loop through each stock ---
    for index, row in top_stocks.iterrows():
        tradingsymbol = row["tradingsymbol"]
        quantity = 1
        amo_flag = True
        product_type = "CNC"

        # --- Duplicate check for same day ---
        today = date.today()
        with engine.connect() as conn:
            query = select(stock_intended_orders).where(
                (stock_intended_orders.c.tradingsymbol == tradingsymbol) &
                (stock_intended_orders.c.requested_at >= datetime(today.year, today.month, today.day)) &
                (stock_intended_orders.c.requested_at < datetime(today.year, today.month, today.day, 23, 59, 59))
            )
            result = conn.execute(query).first()

        if result:
            print(f"⚠️ Duplicate detected, skipping {tradingsymbol} for today.")
            continue

        # --- Initialize defaults ---
        order_id = None
        status = "FAILED"
        error_message = None
        kite_status = None
        kite_message = None
        filled_quantity = None
        average_price = None

        # --- Place order and handle exceptions ---
        print(f"🟦 Attempting to buy: {tradingsymbol}")
        kite_response = buy_stock(
            tradingsymbol,
            quantity=quantity,
            amo=amo_flag,
            product=product_type
        )

        if isinstance(kite_response, dict):
            order_id = kite_response.get("order_id")
            kite_status = "SUCCESS" if kite_response.get("success") else "FAILED"
            kite_message = kite_response.get("message")
            filled_quantity = kite_response.get("filled_quantity")
            average_price = kite_response.get("average_price")
            status = kite_status
        else:
            # Unexpected return type
            status = "FAILED"
            error_message = "Order failed (unexpected response from buy_stock)"
            kite_status = "UNKNOWN"
            kite_message = str(kite_response)

        # --- Insert into DB ---
        requested_at = datetime.now()  # Zurich server time
        with engine.connect() as conn:
            insert_stmt = stock_intended_orders.insert().values(
                tradingsymbol=tradingsymbol,
                quantity=quantity,
                amo=amo_flag,
                product=product_type,
                order_type="MARKET",
                requested_at=requested_at,
                order_id=order_id,
                status=status,
                error_message=error_message,
                kite_status=kite_status,
                kite_message=kite_message,
                filled_quantity=filled_quantity,
                average_price=average_price
            )
            conn.execute(insert_stmt)
            conn.commit()

    print("✔ All intended orders processed and stored (duplicates skipped, exceptions captured).")


# --------------------------
# Updates stock list in the database
# --------------------------
def database_update():
    """
    Fetch latest stock list from Kite API and write to Bronze layer in S3 (Parquet)
    """

    # -----------------------------
    # 1. Fetch stock data from Kite API
    # -----------------------------
    url = "https://api.kite.trade/instruments"
    df = pd.read_csv(url)

    # Keep only tradeable NSE stocks
    df = df[df['exchange'] == 'NSE']
    df = df[~df['tradingsymbol'].str.match(r'^\d.*-.{2}$')]
    df = df[df['segment'] == 'NSE']

    # Add update timestamp
    df['date_update'] = datetime.now().date()

    # Select relevant columns
    cols_sel = [
        'date_update', 'instrument_token', 'exchange_token',
        'tradingsymbol', 'name', 'instrument_type',
        'segment', 'exchange'
    ]
    df = df[cols_sel]

    # -----------------------------
    # 2. Save locally as Parquet
    # -----------------------------
    local_file = "/tmp/stock_list.parquet"
    df.to_parquet(local_file, index=False)

    # -----------------------------
    # 3. Upload to S3 (Bronze layer)
    # -----------------------------
    s3 = boto3.client("s3")

    bucket_name = "stockzerodha"

    s3_key = f"bronze/stock_list.parquet"

    s3.upload_file(local_file, bucket_name, s3_key)

    print(f"✅ Uploaded to S3: s3://{bucket_name}/{s3_key}")


def update_stock_timeseries_db():
    """
    Fetch OHLCV from Zerodha and write raw data to S3 (Bronze layer),
    preserving first-time vs incremental update logic.
    """

    # -----------------------------
    # Load stock list from S3
    # -----------------------------
    stock_list_path = "bronze/stock_list.parquet"

    df_stock_list = pd.read_parquet("s3://" + BUCKET + '/' + stock_list_path, engine="pyarrow")

    kite_client = ZerodhaConnector(API_KEY, ACCESS_TOKEN)
    kite = kite_client.kite

    to_date = datetime.now().date()
    from_date_default = to_date - relativedelta(months=60)
    

    # Check if Bronze table exists
    try:
        df_existing = pd.read_parquet(
                            f"s3://{BUCKET}/bronze/stock_timeseries.parquet",
                            engine="pyarrow"
                        )
        first_time = False
    except Exception:
        print("⚠️ Bronze stock_timeseries table does not exist. Running full initial load.")
        first_time = True
        df_existing = None

    print(df_existing)
    all_data = df_existing if df_existing is not None else []

    for _, stock in tqdm(df_stock_list.iterrows(), total=len(df_stock_list), desc="Updating Stocks"):
        symbol = stock['tradingsymbol']

        # Determine from_date
        if first_time:
            from_date = from_date_default
        else:
            # Check last saved date for this symbol
            df_symbol = df_existing[df_existing["tradingsymbol"] == symbol][["date"]]
            if not df_symbol.empty:
                last_saved = df_symbol["date"].max()
                last_saved = last_saved.date() if isinstance(last_saved, datetime) else last_saved
                if last_saved >= to_date:
                    continue
                from_date = last_saved + timedelta(days=1)
            else:
                from_date = from_date_default

        df = fetch_ohlcv_from_zerodha(symbol, from_date, to_date, "day", "NSE", kite)

        if not df.empty:
            df['tradingsymbol'] = symbol
            df['ingestion_ts'] = datetime.now()
            all_data.append(df)

    # -----------------------------
    # Write to S3 (Bronze layer)
    # -----------------------------
    print(all_data)
    if all_data:
        df_combined = pd.concat(all_data, ignore_index=True)

        local_file = "/tmp/stock_timeseries.parquet"
        df_combined.to_parquet(local_file, index=False)
    
        s3 = boto3.client("s3")

        s3_key = f"bronze/stock_timeseries.parquet"

        s3.upload_file(local_file, BUCKET, s3_key)

        print(f"✅ Successfully updated Bronze stock_timeseries: s3://{BUCKET}/{s3_key}")

    else:
        print("ℹ️ No new data to update.")

def update_stock_scores_db():
    """
    Compute stock scores from Bronze tables and write to Silver Delta table
    """
    to_date = datetime.now().date()

    # -----------------------------
    # 1. Load stock list
    # -----------------------------
    stock_list_path = "bronze/stock_list.parquet"
    df_stock_list = pd.read_parquet(
        f"s3://{BUCKET}/{stock_list_path}",
        engine="pyarrow"
    )

    if df_stock_list.empty:
        print("⚠️ No stocks found in stock_list.")
        return

    # -----------------------------
    # 2. Load full time series ONCE
    # -----------------------------
    df_ts_all = pd.read_parquet(
        f"s3://{BUCKET}/bronze/stock_timeseries.parquet",
        engine="pyarrow"
    )

    if df_ts_all.empty:
        print("⚠️ No time series data found.")
        return

    results = []

    # -----------------------------
    # 3. Process each stock via groupby
    # -----------------------------
    for symbol, df_ts in tqdm(df_ts_all.groupby("tradingsymbol"), desc="Scoring Stocks"):
        
        df_ts = df_ts.sort_values("date")

        if len(df_ts) < 2:
            continue

        # Compute indicators
        df_indicators = compute_indicators(df_ts)

        if df_indicators.empty:
            continue

        # Take latest row
        latest_row = df_indicators.iloc[-1]

        # Compute score
        score = score_stock(df_indicators, latest_row)

        results.append({
            "date_update": to_date,
            "tradingsymbol": symbol,
            "score": round(score, 2),
            "latest_close": round(latest_row["close"], 2)
        })

    # -----------------------------
    # 4. Write to Silver
    # -----------------------------
    if results:
        df_scores = pd.DataFrame(results)

        output_path = f"s3://{BUCKET}/silver/stock_scores.parquet"
        df_scores.to_parquet(output_path, engine="pyarrow", index=False)

        print(f"✅ Saved {len(df_scores)} stock scores to: {output_path}")
    else:
        print("⚠️ No scores calculated.")

# --------------------------
# Incremental stock sector classification
# --------------------------
def classify_new_stocks_to_sectors(batch_size: int = 100, allowed_sectors: list = None):
    """
    Incrementally classify new stocks into sectors using OpenAI LLM
    and write results to Silver Delta table.
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
    
    # -----------------------------
    # 1. Load stock list from Bronze
    # -----------------------------
    stock_list_path = "bronze/stock_list.parquet"
    df_stock_list = pd.read_parquet("s3://" + BUCKET + '/' + stock_list_path, engine="pyarrow")
    
    if df_stock_list.empty:
        print("⚠️ No stocks found in stock_list.")
        return pd.DataFrame()

    # -----------------------------
    # 2. Load existing sectors from Silver
    # -----------------------------
    try:
        stock_list_path = "silver/stock_sectors.parquet"
        df_existing = pd.read_parquet(
                            f"s3://{BUCKET}/{stock_list_path}",
                            engine="pyarrow"
                        )
        existing_symbols = set(df_existing['tradingsymbol'].tolist())
    except Exception:
        existing_symbols = set()

    # -----------------------------
    # 3. Find new stocks
    # -----------------------------
    new_stocks = df_stock_list[~df_stock_list['tradingsymbol'].isin(existing_symbols)]
    if new_stocks.empty:
        print("No new stocks to classify.")
        return pd.DataFrame()

    df_stocks = new_stocks.copy()
    sectors_result = []
    sector_list_str = ", ".join(allowed_sectors)

    # -----------------------------
    # 4. Classify in batches using LLM
    # -----------------------------
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

        response = client_openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        llm_text = response.choices[0].message.content

        # Extract JSON array from LLM response
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

    # -----------------------------
    # 5. Convert result to DataFrame
    # -----------------------------
    if sectors_result:
        df_sectors = pd.DataFrame(sectors_result)
        df_sectors['date_update'] = datetime.now().date()
        cols = ['date_update'] + [c for c in df_sectors.columns if c != 'date_update']
        df_sectors = df_sectors[cols]

        # -----------------------------
        # 6. Write to Silver Delta
        # -----------------------------
        output_path = f"s3://{BUCKET}/silver/stock_sectors.parquet"
        df_sectors.to_parquet(output_path, engine="pyarrow", index=False)


        print(f"✅ Classified and uploaded {len(df_sectors)} new stocks to Silver: stock_sectors.parquet")
        return df_sectors
    else:
        print("⚠️ No sectors classified.")
        return pd.DataFrame()


def select_top_stocks(top_n=10, price_limit=200, correlation_threshold=0.6):
    """
    Select top stocks based on score, sector diversification, price limit, and correlation filter.
    Writes final selection to Gold Delta table.
    """

    # --------------------------
    # Load data from Bronze / Silver
    # --------------------------

    stock_list = pd.read_parquet("s3://" + BUCKET + '/' + "bronze/stock_list.parquet", engine="pyarrow")
    df_sectors = pd.read_parquet("s3://" + BUCKET + '/' + "silver/stock_sectors.parquet", engine="pyarrow")
    ranked_stocks = pd.read_parquet("s3://" + BUCKET + '/' + "silver/stock_scores.parquet", engine="pyarrow")

    # Take latest scores only
    latest_date = ranked_stocks['date_update'].max()
    ranked_stocks = ranked_stocks[ranked_stocks['date_update'] == latest_date]

    # Merge stock names and sector info
    ranked_stocks = ranked_stocks.merge(df_sectors[['tradingsymbol', 'sector']],
                                        on='tradingsymbol', how='left')
    ranked_stocks = ranked_stocks.merge(stock_list[['tradingsymbol', 'name']],
                                        on='tradingsymbol', how='left')

    # Filter by price
    ranked_stocks = ranked_stocks[ranked_stocks['latest_close'] <= price_limit]

    selected_stocks = []
    selected_sectors = set()

    # --------------------------
    # Select top N with correlation filter
    # --------------------------
    for _, row in ranked_stocks.sort_values('score', ascending=False).iterrows():
        stock = row['tradingsymbol']
        sector = row['sector']

        # Load historical closes from Bronze timeseries
        df_stock = pd.read_parquet("s3://" + BUCKET + '/' + "bronze/stock_timeseries.parquet", engine="pyarrow")
        df_stock = df_stock[df_stock['tradingsymbol'] == stock].sort_values("date")
        if df_stock.empty:
            continue

        stock_returns = df_stock['close'].pct_change().dropna()
        skip = False

        # Check correlation with already selected stocks
        for sel_stock in selected_stocks:
            sel_df = pd.read_parquet("s3://" + BUCKET + '/' + "bronze/stock_timeseries.parquet", engine="pyarrow")
            sel_df = sel_df[sel_df['tradingsymbol'] == sel_stock].sort_values("date")
            sel_returns = sel_df['close'].pct_change().dropna()
            combined = pd.concat([stock_returns, sel_returns], axis=1, join='inner')
            if combined.shape[0] == 0:
                continue
            corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
            if abs(corr) >= correlation_threshold:
                skip = True
                break

        if skip:
            continue

        selected_stocks.append(stock)
        selected_sectors.add(sector)

        if len(selected_stocks) >= top_n:
            break

    print("Selected stocks:", selected_stocks)

    # --------------------------
    # Prepare final DataFrame
    # --------------------------
    final_stocks = ranked_stocks[ranked_stocks['tradingsymbol'].isin(selected_stocks)]
    top_stocks = final_stocks[['tradingsymbol', 'score', 'sector', 'latest_close']].copy()
    top_stocks['score'] = top_stocks['score'].round(2)
    top_stocks['latest_close'] = top_stocks['latest_close'].round(2)
    top_stocks['Date_Update'] = datetime.now().date()

    # --------------------------
    # Write to Gold Delta table
    # --------------------------
    
    local_file = "/tmp/top_stocks.parquet"
    top_stocks.to_parquet(local_file, index=False)

    s3 = boto3.client("s3")

    s3_key = f"gold/top_stocks.parquet"

    s3.upload_file(local_file, BUCKET, s3_key)

    print(f"✅ Top {len(top_stocks)} stocks written to Gold: top_stocks.parquet")
    return top_stocks

# from pyspark.sql import SparkSession, Window
# from pyspark.sql import functions as F

def create_stock_features_ml(bronze_table="stock_catalog.bronze.stock_timeseries",
                             silver_table="stock_catalog.silver.stock_features_ml",
                             momentum_windows=[5,20,60],
                             volatility_windows=[5,20,60],
                             min_volume=1000,
                             target_horizon=5):
    """
    Compute ML features and 5-day forward return target from OHLCV bronze table
    and write to silver table for training/testing.
    """
    spark = SparkSession.builder.getOrCreate()

    # Load bronze data
    df = spark.table(bronze_table)
    df = df.withColumn("date", F.to_date("date"))

    # --- Stock age features ---
    w_stock = Window.partitionBy("tradingsymbol").orderBy("date")
    df = df.withColumn("first_date", F.min("date").over(w_stock))
    df = df.withColumn("days_since_listing", F.datediff("date", "first_date"))
    df = df.withColumn("is_new_stock", F.when(F.col("days_since_listing") < 60, 1).otherwise(0))

    # --- Momentum features ---
    for w in momentum_windows:
        df = df.withColumn(f"momentum_{w}d", 
                           (F.col("close") - F.lag("close", w).over(w_stock)) / F.lag("close", w).over(w_stock))

    # --- Volatility features ---
    for w in volatility_windows:
        df = df.withColumn(f"volatility_{w}d",
                           F.stddev("close").over(Window.partitionBy("tradingsymbol")
                                                  .orderBy("date")
                                                  .rowsBetween(-w+1, 0)))

    # --- Average volume ---
    df = df.withColumn("avg_volume_20d",
                       F.avg("volume").over(Window.partitionBy("tradingsymbol")
                                             .orderBy("date")
                                             .rowsBetween(-19, 0)))
    # Filter illiquid days
    df = df.filter(F.col("avg_volume_20d") >= min_volume)

    # --- Compute forward return target ---
    df = df.withColumn("close_future", F.lead("close", target_horizon).over(w_stock))
    df = df.withColumn("future_5d_return", (F.col("close_future") - F.col("close")) / F.col("close"))
    df = df.drop("close_future")

    # --- Cross-sectional percentile rank per day ---
    w_date = Window.partitionBy("date").orderBy(F.col("future_5d_return").desc())
    df = df.withColumn("future_5d_return_rank",
                       F.percent_rank().over(w_date))

    # --- Select relevant columns ---
    feature_cols = ["date", "tradingsymbol", "close", "volume",
                    "days_since_listing", "is_new_stock"] + \
                   [f"momentum_{w}d" for w in momentum_windows] + \
                   [f"volatility_{w}d" for w in volatility_windows] + \
                   ["avg_volume_20d", "future_5d_return", "future_5d_return_rank"]

    df_features = df.select(*feature_cols)

    # --- Write to silver table ---
    df_features.write.format("delta") \
        .mode("overwrite") \
        .saveAsTable(silver_table)

    print(f"✅ Silver ML table created: {silver_table}")



def send_pushover_notification(top_stocks, top_n=10):
    """
    Sends a Pushover push notification with the top N stocks.

    Parameters:
    - top_stocks: pandas DataFrame containing at least 'tradingsymbol' column
    - top_n: number of top stocks to include in message
    """

    # Format top N stocks as text
    top_symbols = top_stocks['tradingsymbol'].head(top_n).tolist()
    top10_str = ", ".join(top_symbols)

    message = f"📈 Top {top_n} Stocks Today:\n{top10_str}"
    print(pushover_api_token)

    # Send POST request to Pushover API
    response = requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": pushover_api_token,
            "user": pushover_userkey,
            "message": message,
            "title": "Daily Stock Signals"
        }
    )

    if response.status_code == 200:
        print(f"✅ Pushover notification sent! Top {top_n} stocks included.")
    else:
        print(f"❌ Failed to send Pushover notification. Status code: {response.status_code}")


def send_email_notification(top_stocks, top_n=10):
    """
    Sends an email notification with the top N stocks.

    Parameters:
    - top_stocks: pandas DataFrame containing at least 'tradingsymbol' column
    - top_n: number of top stocks to include in message
    """

    # Format top N stocks as text
    top_symbols = top_stocks['tradingsymbol'].head(top_n).tolist()
    top10_str = ", ".join(top_symbols)

    subject = "Daily Stock Signals"
    body = f"📈 Top {top_n} Stocks Today:\n{top10_str}"

    # Send email using AWS SES
    ses_client = boto3.client('ses', region_name='eu-north-1')
    response = ses_client.send_email(
        Source='anubhaviiser@gmail.com',
        Destination={'ToAddresses': ['anubhaviiser@gmail.com']},
        Message={
            'Subject': {'Data': subject},
            'Body': {'Text': {'Data': body}}
        }
    )

    if response['ResponseMetadata']['HTTPStatusCode'] == 200:
        print(f"✅ Email notification sent! Top {top_n} stocks included.")
    else:
        print(f"❌ Failed to send email notification. Response: {response}")