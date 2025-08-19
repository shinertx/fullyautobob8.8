import time
from v26meme.registry.catalog import build_snapshot

class _FakeEx:
    id = "coinbase"
    def __init__(self):
        self.markets = {
            "ETH/USD": {"base":"ETH","quote":"USD","spot":True},
            "ETH/EUR": {"base":"ETH","quote":"EUR","spot":True},
            "BTC/USDT": {"base":"BTC","quote":"USDT","spot":True},
            "BTC-PERP": {"base":"BTC","quote":"USD","future":True},
        }

def test_build_snapshot_filters_and_keys():
    ex = _FakeEx()
    # Provide allowed_quotes consistent with resolver defaults for test clarity
    snap = build_snapshot(ex, ("USD","USDT","EUR"))
    # Should include only spot markets
    assert set(snap.keys()) == {"ETH_USD_SPOT","ETH_EUR_SPOT","BTC_USDT_SPOT"}
    for canon, meta in snap.items():
        assert meta["base"] and meta["quote"]
        assert meta.get("spot") is True
        # canonical symbol mapping should round-trip base/quote to uppercase
        assert canon.startswith(meta["base"]) and canon.split("_")[1] == meta["quote"]
        assert "symbol" in meta
        # timestamp not included in snapshot meta currently; ensure no unexpected keys removed
        # (future: if ts added per-item adapt test)
