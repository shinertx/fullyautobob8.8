from v26meme.registry.resolver import SymbolResolver, ResolverPolicy

def test_resolver_policy_typing_and_configure():
    r = SymbolResolver()
    assert isinstance(r.policy.allowed_quotes_global, tuple)
    assert all(isinstance(q, str) for q in r.policy.allowed_quotes_global)
    # configure with custom quotes and aliases
    r.configure({
        'allowed_quotes_global': ['USD','EUR'],
        'allowed_quotes_by_venue': {'kraken': ['USD','USDT','EUR']},
        'base_aliases': {'BTC':['XBT'],'ETH':['WETH']},
        'cache_ttl_s': 120
    })
    assert 'EUR' in r.policy.allowed_quotes_global
    aqbv = r.policy.allowed_quotes_by_venue or {}
    assert tuple(['USD','USDT','EUR']) == aqbv.get('kraken')
