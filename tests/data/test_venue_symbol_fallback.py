from v26meme.registry.canonical import venue_symbol_for
from v26meme.registry.resolver import configure

def setup_module():
    configure({"allowed_quotes_global": ["USD","USDT","USDC","EUR"]})

class _FakeEx:
    id = "kraken"
    def __init__(self):
        self.markets = {
            "BTC/USD":    {"base":"BTC","quote":"USD","spot":True},
            "MATIC/USDT": {"base":"MATIC","quote":"USDT","spot":True},
        }

def test_direct_usd_hit():
    ex = _FakeEx()
    assert venue_symbol_for(ex, "BTC_USD_SPOT") == "BTC/USD"

def test_fallback_to_stable_usd():
    ex = _FakeEx()
    assert venue_symbol_for(ex, "MATIC_USD_SPOT") == "MATIC/USDT"
