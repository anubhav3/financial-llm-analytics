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

# Third-party libraries
import numpy as np
from openai import OpenAI
import pandas as pd
import pytz
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from kite_connect import ZerodhaConnector
from kiteconnect.exceptions import InputException
from sqlalchemy import (
    Table, Column, Integer, String, Boolean, DateTime, MetaData, create_engine, Date, select
)



# --------------------------
# Load environment variables
# --------------------------
project_root = Path.cwd().parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path, override=True)

DB_USER = os.getenv("POSTGRES_USER")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD")
DB_HOST = os.getenv("POSTGRES_HOST", "localhost")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "market")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_KEY = os.getenv("KITE_API_KEY", None)
API_SECRET = os.getenv("KITE_API_SECRET", None)
ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", None)
client = OpenAI()

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

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )

        llm_text = response.choices[0].message.content

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


# --------------------------
# Updates stock list in the database
# --------------------------
def database_update():
    
    # --- Connect to PostgreSQL ---
    engine = get_db_engine()

    # --- Read your stocks from the URL ---
    url = "https://api.kite.trade/instruments"
    df = pd.read_csv(url)

    # Keeping only the tradeable stocks
    df = df[df['exchange'] == 'NSE']
    df = df[~df['tradingsymbol'].str.match(r'^\d.*-.{2}$')]
    df = df[df['segment'] == 'NSE']

    df['Date_Update'] = datetime.now().date()
    cols_sel = ['Date_Update', 'instrument_token', 'exchange_token', 'tradingsymbol', 'name', 'instrument_type', 'segment', 'exchange']

    df = df[cols_sel]

    # --- Upload to database ---
    df.to_sql(
        'stock_list',
        engine,
        if_exists = 'replace',
        index = False,
        dtype={'date_update': Date()}
    )

    print("✅ Successfully uploaded stocklist to PostgreSQL!")
    
    
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
    if 0.02 <= vol <= 0.08:  # example range for “moderate” volatility
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

def update_stock_timeseries_db():

    engine = get_db_engine()

    # Load stock list
    stock_list = pd.read_sql("SELECT * FROM stock_list", engine)

    # load_dotenv(dotenv_path=env_path, override=True)
    # API_KEY = os.getenv("KITE_API_KEY")
    # ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", None)
    # print(ACCESS_TOKEN)
    
    kite_client = ZerodhaConnector(API_KEY, ACCESS_TOKEN)
    kite = kite_client.kite

    to_date = datetime.now().date()
    from_date_default = to_date - relativedelta(months=3)

    # Check if this is the first time stock_timeseries is used
    try:
        existing_symbols = pd.read_sql(
            "SELECT DISTINCT tradingsymbol FROM stock_timeseries LIMIT 1", 
            engine
        )
        first_time = False
    except Exception as e:
        # Table doesn't exist → first time update
        print("⚠️ stock_timeseries does not exist. Running full initial load.")
        first_time = True

    # Loop over stocks
    for _, stock in tqdm(stock_list.iterrows(), total=len(stock_list), desc="Updating Stocks"):

        symbol = stock['tradingsymbol']

        # Case 1: First time update → load full 3 months
        if first_time:
            from_date = from_date_default

        # Case 2: Table exists → use last saved date
        else:
            last_saved = get_last_saved_date(symbol, engine)

            if last_saved is not None:
                last_saved = pd.to_datetime(last_saved).date()

                if last_saved >= to_date:
                    # Already up-to-date
                    continue

                from_date = last_saved + timedelta(days=1)
            else:
                # Symbol does not exist in table → get default 3 months
                from_date = from_date_default
                
        # Fetch data from Zerodha
        df = fetch_ohlcv_from_zerodha(symbol, from_date, to_date, "day", "NSE", kite)
        df = df.reset_index(drop=True)

        if not df.empty:
            df['tradingsymbol'] = symbol

            df.to_sql(
                'stock_timeseries',
                engine,
                if_exists='append',
                index=False,
                dtype={'date': Date()}
            )

def update_stock_scores_db():
    engine = get_db_engine()

    # --------------------------
    # Load stock list
    # --------------------------
    stock_list = pd.read_sql("SELECT * FROM stock_list", engine)
    if stock_list.empty:
        print("⚠️ No stocks found in stock_list.")
        return

    results = []
    to_date = datetime.now().date()

    # --------------------------
    # Loop over each stock
    # --------------------------
    for _, stock in tqdm(stock_list.iterrows(), total=len(stock_list), desc="Scoring Stocks"):
        symbol = stock['tradingsymbol']

        # Load full OHLCV history for this stock
        df = pd.read_sql(
            "SELECT * FROM stock_timeseries WHERE tradingsymbol = %s ORDER BY date",
            engine,
            params=(symbol,)
        )


        if df.empty or len(df) < 2:
            # Not enough data to compute indicators/score
            continue

        # Compute technical indicators
        df_indicators = compute_indicators(df)

        # Take the latest row for scoring
        latest_row = df_indicators.iloc[-1]

        # Compute score
        score = score_stock(df_indicators, latest_row)

        results.append({
            "Date_Update": to_date,
            "tradingsymbol": symbol,
            "score": score,
            "latest_close": latest_row["close"]
        })

    # --------------------------
    # Save scores to database
    # --------------------------
    if results:
        df_scores = pd.DataFrame(results)
        df_scores['score'] = df_scores['score'].round(2)
        df_scores['latest_close'] = df_scores['latest_close'].round(2)
        cols_sel = ['Date_Update'] + [c for c in df_scores.columns if c != 'Date_Update']
        df_scores = df_scores[cols_sel]

        df_scores.to_sql(
            "stock_scores",
            engine,
            if_exists="append",
            index=False,
            dtype={'Date_Update': Date()}
        )

        print(f"✅ Saved {len(df_scores)} stock scores.")
    else:
        print("⚠️ No scores calculated.")
        
        
        
def select_top_stocks(top_n=10, price_limit=200, correlation_threshold=0.6):
    """
    Select top stocks based on score, sector diversification, price limit, and correlation filter.
    
    Saves the final selection to the 'stock_short_buy' table.
    """
    engine = get_db_engine()

    # --------------------------
    # Load necessary data
    # --------------------------
    stock_list = pd.read_sql("SELECT * FROM stock_list", engine)
    df_sectors = pd.read_sql("SELECT * FROM stock_sectors", engine)
    
    ranked_stocks = pd.read_sql("""
        SELECT *
        FROM stock_scores
        WHERE "Date_Update" = (
            SELECT MAX("Date_Update") FROM stock_scores
        )
    """, engine)
    
    # Merge stock names and sector info
    ranked_stocks = ranked_stocks.merge(df_sectors[['tradingsymbol', 'sector']],
                                        on='tradingsymbol', how='left')
    ranked_stocks = ranked_stocks.merge(stock_list[['tradingsymbol', 'name']],
                                        on='tradingsymbol', how='left')
    
    # Filter by price limit
    ranked_stocks_sub = ranked_stocks[ranked_stocks['latest_close'] <= price_limit]

    selected_stocks = []
    selected_sectors = set()

    # --------------------------
    # Select stocks based on score, sector, and correlation
    # --------------------------
    for _, row in ranked_stocks_sub.sort_values('score', ascending=False).iterrows():
        stock = row['tradingsymbol']
        sector = row['sector']

        # Skip if sector already selected (optional: uncomment if strict sector diversification)
        # if sector in selected_sectors:
        #     continue

        # Load historical closes
        df_stock = pd.read_sql(f"""
            SELECT date, close
            FROM stock_indicators
            WHERE tradingsymbol = '{stock}'
            ORDER BY date
        """, engine, parse_dates=['date'])

        if df_stock.empty:
            continue

        stock_returns = df_stock['close'].pct_change().dropna()
        skip = False

        # Check correlation with already selected stocks
        for sel_stock in selected_stocks:
            sel_df = pd.read_sql(f"""
                SELECT date, close
                FROM stock_indicators
                WHERE tradingsymbol = '{sel_stock}'
                ORDER BY date
            """, engine, parse_dates=['date'])
            sel_returns = sel_df['close'].pct_change().dropna()

            combined = pd.concat([stock_returns, sel_returns], axis=1, join='inner')
            if combined.shape[0] == 0:
                continue

            corr = combined.iloc[:, 0].corr(combined.iloc[:, 1])
            if abs(corr) >= correlation_threshold:
                print(f"Skipping {stock}, correlation with {sel_stock} = {corr:.2f}")
                skip = True
                break

        if skip:
            continue

        # Add to selection
        selected_stocks.append(stock)
        selected_sectors.add(sector)  # optional

        if len(selected_stocks) >= top_n:
            break

    print("Selected stocks:", selected_stocks)

    # --------------------------
    # Prepare final DataFrame
    # --------------------------
    final_stocks = ranked_stocks_sub[ranked_stocks_sub['tradingsymbol'].isin(selected_stocks)]
    top_stocks = final_stocks[['tradingsymbol', 'score', 'sector', 'latest_close']].copy()
    
    # Round numeric columns
    top_stocks['score'] = top_stocks['score'].round(2)
    top_stocks['latest_close'] = top_stocks['latest_close'].round(2)

    # Add update date
    top_stocks['Date_Update'] = datetime.now().date()
    cols = ['Date_Update'] + [c for c in top_stocks.columns if c != 'Date_Update']
    top_stocks = top_stocks[cols]

    # --------------------------
    # Save to database
    # --------------------------
    top_stocks.to_sql(
        'stock_short_buy',
        engine,
        if_exists='append',
        index=False,
        dtype={'Date_Update': Date()}
    )

    return top_stocks


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
