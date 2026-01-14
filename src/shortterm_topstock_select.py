### Does:
# Selects top N stocks as per given criteria

PRICE_LIMIT = 200
CORRELATION_THRESHOLD = 0.6
TOP_N = 10

# Login to Kite
import subprocess
import sys
subprocess.run([sys.executable, "src/auto_kite_login.py"], check = True)

## Update the database with latest stocks
from utils import database_update
database_update()

## Update stock timeseries in the database
from utils import update_stock_timeseries_db
update_stock_timeseries_db()

## Update stock scores in the database
from utils import update_stock_scores_db
update_stock_scores_db()

## Classify new stocks to sectors
from utils import classify_new_stocks_to_sectors
classify_new_stocks_to_sectors()

## Select top stocks based on given criteria
from utils import select_top_stocks
select_top_stocks(top_n = TOP_N, price_limit = PRICE_LIMIT, correlation_threshold = CORRELATION_THRESHOLD)