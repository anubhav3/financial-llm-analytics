from kiteconnect import KiteConnect
from kiteconnect.exceptions import TokenException
import logging
import os
from dotenv import load_dotenv, set_key
from pathlib import Path
import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from urllib.parse import urlparse, parse_qs
import time
import pyotp
import boto3

# Load .env from project root
project_root = Path(os.getcwd()).resolve()
env_path = project_root / ".env"
load_dotenv(dotenv_path=env_path)

session = boto3.session.Session()
client = session.client(service_name='secretsmanager',
                        region_name="eu-north-1" )
secrets = client.get_secret_value(SecretId='StockZerodhaRelated')['SecretString']
secrets = eval(secrets)

logging.basicConfig(level=logging.INFO)

class ZerodhaConnector:
    def __init__(self, api_key: str, access_token: str = None):
        self.api_key = api_key
        self.access_token = access_token
        self.kite = KiteConnect(api_key=self.api_key)

        if access_token:
            try:
                self.kite.set_access_token(access_token)
                # logging.info("Access token set successfully.")
            except TokenException:
                logging.warning("Access token invalid or expired. Will generate a new token.")
                self.access_token = None
        else:
            logging.warning("Access token not set yet. Will generate automatically.")

    def generate_access_token(self, api_secret: str):
        """Automatically log in using Selenium and generate access token with TOTP."""
        USER_ID = secrets["ZERODHA-USERID"]
        PASSWORD = secrets["ZERODHA-PASSWORD"]
        TOTP_SECRET = secrets["ZERODHA-SECRET-TOTP"]

        if not all([USER_ID, PASSWORD, TOTP_SECRET]):
            raise ValueError("ZERODHA-USERID, ZERODHA-PASSWORD, and ZERODHA-SECRET-TOTP must be in the vault.")
        print("Starting Selenium to log in to Kite...")
        options = webdriver.ChromeOptions()
        options.add_argument("--start-maximized")
        options.add_argument("--headless")
        options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument("--window-size=1920,1080")
        driver = webdriver.Chrome(options=options)
        wait = WebDriverWait(driver, 30)

        try:
            driver.get(self.kite.login_url())
            print("Login page loaded. Entering credentials...")

            # Enter user ID
            user_el = wait.until(EC.presence_of_element_located((By.ID, "userid")))
            user_el.send_keys(USER_ID)

            # Enter password
            pwd_el = driver.find_element(By.ID, "password")
            pwd_el.send_keys(PASSWORD)

            # Click login
            login_btn = driver.find_element(By.CSS_SELECTOR, "button[type='submit']")
            login_btn.click()

            # Wait for TOTP input page
            otp_el = WebDriverWait(driver, 30).until(
                EC.presence_of_element_located((By.XPATH, "//input[@label='External TOTP']"))
            )

            # Generate TOTP and send
            totp = pyotp.TOTP(TOTP_SECRET)
            current_otp = totp.now()
            otp_el.clear()
            otp_el.send_keys(current_otp)

            # Instead of clicking button, just wait for redirect with request_token
            logging.info("TOTP entered, waiting for redirect...")
            request_token = None
            timeout = time.time() + 180  # 3 minutes
            while time.time() < timeout:
                current_url = driver.current_url
                if "request_token=" in current_url:
                    parsed = urlparse(current_url)
                    request_token = parse_qs(parsed.query)["request_token"][0]
                    logging.info("Request token obtained.")
                    break
                time.sleep(1)

            if not request_token:
                raise Exception("Failed to obtain request_token. Please complete login manually.")

            # Generate access token
            data = self.kite.generate_session(request_token, api_secret=api_secret)
            self.access_token = data["access_token"]
            self.kite.set_access_token(self.access_token)
            set_key(str(env_path), "KITE-ACCESS-TOKEN", self.access_token)
            logging.info("Access token generated.")

            return self.access_token

        finally:
            driver.quit()

    def _call_api(self, func, *args, **kwargs):
        """Helper method to retry API call if token expired."""
        try:
            return func(*args, **kwargs)
        except TokenException:
            logging.warning("Access token expired. Regenerating token...")
            self.generate_access_token(secrets["KITE-API-SECRET"])
            return func(*args, **kwargs)

    def get_profile(self):
        return self._call_api(self.kite.profile)

    def get_holdings(self):
        holdings = self._call_api(self.kite.holdings)
        if holdings:
            df = pd.DataFrame(holdings)
            columns = ["tradingsymbol", "quantity", "average_price", "last_price", "pnl"]
            df = df[[c for c in columns if c in df.columns]]
            pd.options.display.float_format = "{:.2f}".format
            return df
        else:
            logging.info("No holdings found.")
            return pd.DataFrame()

    def get_positions(self):
        return self._call_api(self.kite.positions)

    def place_order(self, tradingsymbol, quantity, transaction_type, order_type="MARKET", product="CNC"):
        return self._call_api(
            self.kite.place_order,
            variety=self.kite.VARIETY_REGULAR,
            tradingsymbol=tradingsymbol,
            exchange=self.kite.EXCHANGE_NSE,
            transaction_type=transaction_type,
            quantity=quantity,
            order_type=order_type,
            product=product
        )


if __name__ == "__main__":
    API_KEY = secrets['KITE-API-KEY']
    API_SECRET = secrets["KITE-API-SECRET"]
    ACCESS_TOKEN = secrets["KITE-ACCESS-TOKEN"]

    kite_client = ZerodhaConnector(API_KEY, ACCESS_TOKEN)

    # print("Your Holdings:")
    df_holdings = kite_client.get_holdings()
    # print(df_holdings)
