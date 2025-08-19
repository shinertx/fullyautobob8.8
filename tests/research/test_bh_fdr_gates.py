from v26meme.research.validation import benjamini_hochberg
def test_bh():
    p = [0.001,0.04,0.2,0.5,0.05]
    keep, thr = benjamini_hochberg(p, 0.1)
    assert any(keep) and 0<=thr<=0.1
