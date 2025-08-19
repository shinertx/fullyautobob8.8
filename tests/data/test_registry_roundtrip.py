from v26meme.registry.canonical import make_canonical, parse_canonical

def test_roundtrip():
    c = make_canonical("btc","usd","spot")
    b,q,k = parse_canonical(c)
    assert (b,q,k)==("BTC","USD","SPOT")
