from v26meme.data.harvester import _resolve_timeframe_for_exchange

class _FakeEx:
    id = "coinbase"
    def __init__(self):
        self.timeframes = {"1m":None,"5m":None,"15m":None,"1h":None,"6h":None,"1d":None}
    def load_markets(self):
        pass

CFG = {"registry": {"timeframe_aliases_by_venue": {"coinbase": {"4h":"6h"}}}}

def test_alias_4h_to_6h():
    ex = _FakeEx()
    assert _resolve_timeframe_for_exchange("coinbase", ex, "4h", CFG) == "6h"

def test_supported_pass_through():
    ex = _FakeEx()
    assert _resolve_timeframe_for_exchange("coinbase", ex, "1h", CFG) == "1h"

def test_unsupported_none_if_no_alias():
    ex = _FakeEx()
    assert _resolve_timeframe_for_exchange("coinbase", ex, "2h", CFG) is None
