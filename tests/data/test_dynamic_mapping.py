from v26meme.registry.canonical import venue_symbol_for, make_canonical

class MockExchange:
    def __init__(self, markets):
        self.id = "kraken"
        self.markets = markets
    def load_markets(self):  # compatibility no-op
        return self.markets

def test_dynamic_venue_symbol_for_kraken_btc():
    markets = {
        "XBT/USD": {"symbol": "XBT/USD", "base": "BTC", "quote": "USD", "spot": True},
        "ETH/USD": {"symbol": "ETH/USD", "base": "ETH", "quote": "USD", "spot": True},
    }
    ex = MockExchange(markets)
    canon = make_canonical("BTC", "USD", "SPOT")
    sym = venue_symbol_for(ex, canon)
    assert sym == "XBT/USD"  # Kraken-specific unified symbol
