from kiteconnect import KiteConnect
import logging
import os
from dotenv import load_dotenv, set_key
from pathlib import Path
import pandas as pd

# Load .env from project root
project_root = Path(__file__).resolve().parent.parent
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

# Enable logging
logging.basicConfig(level=logging.INFO)

class ZerodhaConnector:
    def __init__(self, api_key: str, access_token: str = None):
        self.api_key = api_key
        self.access_token = access_token
        self.kite = KiteConnect(api_key=self.api_key)
        
        if access_token:
            self.kite.set_access_token(access_token)
            logging.info("Access token set successfully.")
        else:
            logging.warning("Access token not set yet. Please generate it first.")

    def generate_login_url(self):
        """Generate the login URL for user authorization."""
        return self.kite.login_url()

    def generate_access_token(self, request_token: str, api_secret: str):
        """Generate access token using request_token and update .env."""
        data = self.kite.generate_session(request_token, api_secret=api_secret)
        self.access_token = data["access_token"]
        self.kite.set_access_token(self.access_token)
        logging.info("Access token generated and set successfully.")

        # Update the .env file
        set_key(str(env_path), "KITE_ACCESS_TOKEN", self.access_token)
        logging.info(f".env file updated with new access token at {env_path}")

        return self.access_token

    def get_profile(self):
        return self.kite.profile()

    def get_holdings(self):
        """Fetch holdings and return as a pandas DataFrame."""
        holdings = self.kite.holdings()
        if holdings:
            df = pd.DataFrame(holdings)
            # Keep key columns
            columns = ["tradingsymbol", "quantity", "average_price", "last_price", "pnl"]
            df = df[columns]
            # Format numeric values
            pd.options.display.float_format = "{:.2f}".format
            return df
        else:
            logging.info("No holdings found.")
            return pd.DataFrame()

    def get_positions(self):
        return self.kite.positions()

    def place_order(self, tradingsymbol, quantity, transaction_type, order_type="MARKET", product="CNC"):
        try:
            order_id = self.kite.place_order(
                variety=self.kite.VARIETY_REGULAR,
                tradingsymbol=tradingsymbol,
                exchange=self.kite.EXCHANGE_NSE,
                transaction_type=transaction_type,
                quantity=quantity,
                order_type=order_type,
                product=product,
            )
            logging.info(f"Order placed successfully. ID: {order_id}")
            return order_id
        except Exception as e:
            logging.error(f"Order placement failed: {e}")
            return None


if __name__ == "__main__":
    # Load API key and access token from .env
    API_KEY = os.getenv("KITE_API_KEY", None)
    API_SECRET = os.getenv("KITE_API_SECRET", None)
    ACCESS_TOKEN = os.getenv("KITE_ACCESS_TOKEN", None)

    kite_client = ZerodhaConnector(API_KEY, ACCESS_TOKEN)

    if not ACCESS_TOKEN:
        print("Login URL:", kite_client.generate_login_url())
        request_token = input("Enter the request_token from the redirect URL: ").strip()
        kite_client.generate_access_token(request_token, API_SECRET)
        print("Access token updated in .env. You can now rerun the script.")
    else:
        print("Your Holdings:")
        df_holdings = kite_client.get_holdings()
        print(df_holdings)
