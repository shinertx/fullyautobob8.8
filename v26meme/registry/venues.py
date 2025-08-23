venue_symbol_map = {
    "coinbase": {
        "BTC_USD_SPOT": "BTC/USD", "ETH_USD_SPOT": "ETH/USD", "SOL_USD_SPOT": "SOL/USD",
        "AVAX_USD_SPOT": "AVAX/USD","LINK_USD_SPOT": "LINK/USD","ADA_USD_SPOT": "ADA/USD",
        "XRP_USD_SPOT": "XRP/USD","DOGE_USD_SPOT":"DOGE/USD","MATIC_USD_SPOT":"MATIC/USD",
        "DOT_USD_SPOT":"DOT/USD","USDC_USD_SPOT":"USDC/USD","USDT_USD_SPOT":"USDT/USD",
        "BTC_USDC_SPOT":"BTC/USDC","ETH_USDC_SPOT":"ETH/USDC","SOL_USDC_SPOT":"SOL/USDC"
    },
    "kraken": {
        # Kraken uses XBT for Bitcoin
        "BTC_USD_SPOT": "XBT/USD", "ETH_USD_SPOT": "ETH/USD", "SOL_USD_SPOT": "SOL/USD",
        "AVAX_USD_SPOT": "AVAX/USD","LINK_USD_SPOT": "LINK/USD","ADA_USD_SPOT": "ADA/USD",
        "XRP_USD_SPOT": "XRP/USD","DOGE_USD_SPOT":"DOGE/USD","MATIC_USD_SPOT":"MATIC/USD",
        "DOT_USD_SPOT":"DOT/USD","USDC_USD_SPOT":"USDC/USD","USDT_USD_SPOT":"USDT/USD",
        "BTC_USDC_SPOT":"XBT/USDC","ETH_USDC_SPOT":"ETH/USDC","SOL_USDC_SPOT":"SOL/USDC"
    },
}

# NOTE: Symbols like RAD_USD_SPOT, GAIA_USD_SPOT, BIT_USD_SPOT, ALKIMI_USD_SPOT, API3_USD_SPOT appeared in staged_backfill logs.
# If required, extend canonical list after verifying they exist on venues (and add to allowed quotes if needed).
