from v26meme.data.universe_screener import UniverseScreener
import ccxt

class DummyBucket:
    def __init__(self, max_req: int, min_sleep: int):
        self.calls = 0
    def consume(self, cost: int = 1):
        self.calls += cost
        return True

class DummyExchange:
    def __init__(self):
        self.id = 'dummyex'
        self.rateLimit = 0
        self.markets = {
            'BTC/USD': {
                'symbol': 'BTC/USD', 'base': 'BTC', 'quote': 'USD', 'spot': True,
                'precision': {'price': 2}, 'limits': {'amount': {'min': 0.001}},
            }
        }
    def load_markets(self):
        return self.markets
    def fetch_tickers(self, params=None):
        # Provide bid/ask for spread calc
        return {'BTC/USD': {'last': 100.0, 'bid': 99.99, 'ask': 100.01, 'quoteVolume': 5_000_000}}
    def fetch_order_book(self, symbol: str, limit: int = 50):
        bids = [(99.99, 1000.0)]
        asks = [(100.01, 1000.0)]
        return {'bids': bids, 'asks': asks}

class DummyPolicy:
    allowed_quotes_by_venue = {'dummyex': ('USD',)}
    allowed_quotes_global = ('USD',)

class DummyResolver:
    policy = DummyPolicy()


def test_screener_consumes_tokenbucket(monkeypatch):
    monkeypatch.setattr(ccxt, 'dummyex', DummyExchange, raising=False)
    monkeypatch.setattr('v26meme.data.universe_screener.get_resolver', lambda: DummyResolver())

    created: dict = {}
    def bucket_factory(max_req, min_sleep):
        b = DummyBucket(max_req, min_sleep)
        created['bucket'] = b
        return b
    monkeypatch.setattr('v26meme.data.universe_screener.TokenBucket', bucket_factory)

    screener_cfg = {
        'min_24h_volume_usd': 10_000,
        'max_spread_bps': 50,
        'max_impact_bps': 50,
        'typical_order_usd': 1000,
        'min_price': 0,
        'max_markets': 5,
        'stablecoin_parity_warn_bps': 100,
        'derivatives_enabled': False,
        'quotas': {'dummyex': {'max_requests_per_min': 1000, 'min_sleep_ms': 0}},
    }

    us = UniverseScreener(['dummyex'], screener_cfg)
    uni, raw = us.get_active_universe()
    # At least one consume for fetch_tickers
    assert created['bucket'].calls >= 1
    if uni:
        ids = [inst['market_id'] for inst in uni]
        assert any(mid.startswith('BTC_USD_SPOT') for mid in ids)
