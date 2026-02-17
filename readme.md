# 📈 NSE Stock Selection & Auto-Buy System

A data-driven stock selection and automated trading pipeline using:

- Zerodha Kite API  
- PostgreSQL  
- Technical indicators  
- Correlation filtering  
- OpenAI (sector classification)

---

## 🚀 Overview

This project:

1. Updates NSE stock universe  
2. Downloads historical OHLCV data  
3. Computes technical indicators  
4. Scores stocks  
5. Classifies sectors (LLM-based)  
6. Selects top N stocks under:
   - Price constraint
   - Correlation threshold
7. (Optional) Places AMO/GTT buy orders  

---

## 🧠 Selection Logic

Stocks are ranked using:

- Liquidity (avg daily volume)
- Momentum (1-week / 1-month returns)
- Trend (MA20 > MA50)
- RSI range
- MACD crossover
- OBV trend
- Volatility filter

Final selection ensures:

- Price ≤ `PRICE_LIMIT`
- Pairwise correlation < `CORRELATION_THRESHOLD`
- Top `N` ranked stocks

---

## 🗄 Database Tables

- `stock_list`
- `stock_timeseries`
- `stock_scores`
- `stock_sectors`
- `stock_short_buy`
- `stock_intended_orders`

---

## ⚙️ Configuration

Create a `.env` file:

```env
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=market

OPENAI_API_KEY=

KITE_API_KEY=
KITE_API_SECRET=
KITE_ACCESS_TOKEN=
