import ccxt
from loguru import logger

logger.info(f"ccxt version: {ccxt.__version__}")

try:
    exchange = ccxt.coinbase()
    tickers = exchange.fetch_tickers()
    logger.info(f"Successfully fetched {len(tickers)} tickers from Coinbase.")
    print(tickers)
except Exception as e:
    logger.error(f"Failed to fetch tickers from Coinbase: {e}")