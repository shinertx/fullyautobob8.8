import ccxt.async_support as ccxt
import asyncio
import json

async def inspect_exchange(exchange_name):
    print(f"\n--- Inspecting {exchange_name.upper()} ---")
    exchange = getattr(ccxt, exchange_name)()
    
    try:
        await exchange.load_markets()
        
        preferred_symbols = ['BTC/USD', 'BTC/USDT', 'ETH/USD', 'ETH/USDT']
        symbol = next((s for s in preferred_symbols if s in exchange.markets), None)
        
        if not symbol:
            print("Could not find a preferred symbol. Using the first available market.")
            # Fallback to a market that is likely to exist
            symbol = next((m for m in exchange.markets if m.endswith('/USD') or m.endswith('/USDT')), list(exchange.markets.keys())[0])

        print(f"Using symbol: {symbol}")

        # 1. Fetch Tickers (plural) to match screener implementation
        print(f"\n1. Fetching tickers for {symbol}...")
        # Use fetch_tickers (plural) as it's used in the screener
        tickers = await exchange.fetch_tickers([symbol])
        print(f"Tickers data received. Structure for {symbol}:")
        if symbol in tickers:
            print(json.dumps(tickers[symbol], indent=2, default=str))
        else:
            print(f"{symbol} not found in tickers response.")
            print(f"Available tickers in response: {list(tickers.keys())}")

        # 2. Fetch Order Book
        print(f"\n2. Fetching order book for {symbol}...")
        order_book = await exchange.fetch_order_book(symbol, limit=5)
        print("Order book data received. Structure:")
        print("Keys:", list(order_book.keys()))
        print("Sample Bids (top 2):", order_book.get('bids', [])[:2])
        print("Sample Asks (top 2):", order_book.get('asks', [])[:2])
        print("Full structure (first level):")
        print(json.dumps({k: type(v).__name__ for k, v in order_book.items()}, indent=2))

    except Exception as e:
        print(f"An error occurred with {exchange_name}: {e}")
    finally:
        if exchange:
            await exchange.close()

async def main():
    exchanges_to_inspect = ['kraken', 'coinbase']
    for exchange_name in exchanges_to_inspect:
        await inspect_exchange(exchange_name)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nScript interrupted by user.")
