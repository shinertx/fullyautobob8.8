import ccxt.async_support as ccxt
import asyncio
import json

async def main():
    # Use a common exchange for this example
    exchange = ccxt.kraken()
    
    print("--- Inspecting CCXT data structures ---")

    # 1. Fetch Tickers
    try:
        print("\n1. Fetching tickers for BTC/USDT and ETH/USDT...")
        # The unified method is fetch_tickers
        tickers = await exchange.fetch_tickers(['BTC/USDT', 'ETH/USDT'])
        print("Tickers data received.")
        
        if 'BTC/USDT' in tickers:
            print("\nStructure for a single ticker (BTC/USDT):")
            btc_ticker = tickers['BTC/USDT']
            print(json.dumps(btc_ticker, indent=2))
            
            print("\nKey fields for our use case:")
            print(f"  - Symbol: {btc_ticker.get('symbol')}")
            print(f"  - Last Price: {btc_ticker.get('last')}")
            print(f"  - 24h Volume: {btc_ticker.get('quoteVolume')}") # Usually quoteVolume (e.g., in USDT) is used for filtering
            print(f"  - Ask Price: {btc_ticker.get('ask')}")
            print(f"  - Bid Price: {btc_ticker.get('bid')}")
            print(f"  - Spread (calculated): {btc_ticker.get('ask', 0) - btc_ticker.get('bid', 0)}")

        else:
            print("BTC/USDT ticker not found in response.")

    except Exception as e:
        print(f"Error fetching tickers: {e}")

    # 2. Fetch Order Book
    try:
        print("\n\n2. Fetching order book for BTC/USDT...")
        # The unified method is fetch_order_book
        order_book = await exchange.fetch_order_book('BTC/USDT', limit=5)
        print("Order book data received.")
        
        print("\nOrder book structure:")
        # The order book itself is large, so let's just print the keys and a sample
        print("Keys:", list(order_book.keys()))
        
        # Bids are sorted high to low, asks are sorted low to high
        bids = order_book.get('bids', [])
        asks = order_book.get('asks', [])
        
        print("\nSample Bids (Price, Amount):", bids[:2])
        print("Sample Asks (Price, Amount):", asks[:2])
        
        if bids and asks:
            print(f"\nTop of book:")
            print(f"  - Best Bid Price: {bids[0][0]}")
            print(f"  - Best Bid Amount: {bids[0][1]}")
            print(f"  - Best Ask Price: {asks[0][0]}")
            print(f"  - Best Ask Amount: {asks[0][1]}")

    except Exception as e:
        print(f"Error fetching order book: {e}")

    # Always close the exchange connection
    await exchange.close()

if __name__ == "__main__":
    # Ensure we run the async main function
    asyncio.run(main())
