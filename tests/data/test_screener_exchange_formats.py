import pytest
from unittest.mock import MagicMock
from v26meme.data.universe_screener import UniverseScreener

@pytest.fixture
def screener_config():
    """Provides a default screener configuration for tests."""
    return {
        "max_markets": 5,
        "min_24h_volume_usd": 1000,
        "min_price": 0.01,
        "max_spread_bps": 100,
        "max_impact_bps": 100,
        "typical_order_usd": 100,
        "derivatives_enabled": False
    }

def test_coinbase_volume_format_is_handled(screener_config):
    """
    Verify screener correctly handles Coinbase's lack of volume data.
    Since fetch_tickers for Coinbase returns null for volume fields,
    any instrument from Coinbase should be filtered out if min_24h_volume_usd > 0.
    """
    mock_coinbase = MagicMock()
    mock_coinbase.id = 'coinbase'
    mock_coinbase.markets = {'BTC/USD': {'spot': True, 'symbol': 'BTC/USD', 'base': 'BTC', 'quote': 'USD', 'precision': {}, 'limits': {}}}
    mock_coinbase.fetch_tickers.return_value = {
        'BTC/USD': {
            'symbol': 'BTC/USD',
            'last': 50000,
            'bid': 49999,
            'ask': 50001,
            'quoteVolume': None, # This is the key finding
            'baseVolume': None,
            'info': {} # No volume data here either
        }
    }
    # The impact calculation part still needs a valid order book
    mock_coinbase.fetch_order_book.return_value = {
        'bids': [[49999, 1]], 'asks': [[50001, 1]]
    }

    screener = UniverseScreener(['coinbase'], screener_config, {})
    screener.exchanges['coinbase'] = mock_coinbase
    
    # With min_24h_volume_usd=1000, BTC/USD should be filtered out.
    instruments, _ = screener.get_active_universe()
    assert len(instruments) == 0

    # Now, if we set min volume to 0, it should pass the volume filter
    # (but might fail other filters like impact, which is fine for this test)
    screener.cfg['min_24h_volume_usd'] = 0
    instruments, _ = screener.get_active_universe()
    assert len(instruments) == 1
    assert instruments[0]['display'] == 'BTC_USD_SPOT'
    assert instruments[0]['volume_24h_usd'] == 0.0


def test_kraken_orderbook_format(screener_config):
    """Verify screener correctly parses Kraken's [price, size, timestamp] order book format."""
    mock_kraken = MagicMock()
    mock_kraken.id = 'kraken'
    mock_kraken.markets = {'XBT/USD': {'spot': True, 'symbol': 'XBT/USD', 'base': 'XBT', 'quote': 'USD'}}
    mock_kraken.fetch_tickers.return_value = {
        'XBT/USD': {
            'symbol': 'XBT/USD',
            'last': 50000,
            'bid': 49999,
            'ask': 50001,
            'quoteVolume': 2000000,
            'info': {}
        }
    }
    # Simulate Kraken's 3-element order book entries
    mock_kraken.fetch_order_book.return_value = {
        'bids': [[49999, 1, 1655599007]], 
        'asks': [[50001, 1, 1655599008]]
    }

    screener = UniverseScreener(['kraken'], screener_config, {})
    screener.exchanges['kraken'] = mock_kraken
    instruments, _ = screener.get_active_universe()

    assert len(instruments) == 1
    assert instruments[0]['display'] == 'XBT_USD_SPOT'


def test_screener_returns_empty_if_no_markets_pass(screener_config):
    """Ensure it returns empty lists if no markets pass filters."""
    mock_exchange = MagicMock()
    mock_exchange.id = 'testex'
    mock_exchange.markets = {'FAIL/USD': {'spot': True, 'symbol': 'FAIL/USD', 'base': 'FAIL', 'quote': 'USD'}}
    # Fails volume check
    mock_exchange.fetch_tickers.return_value = {'FAIL/USD': {'quoteVolume': 10}} 

    screener = UniverseScreener(['testex'], screener_config, {})
    screener.exchanges['testex'] = mock_exchange
    instruments, tickers = screener.get_active_universe()

    assert instruments == []
