from v26meme.research.validation import benjamini_hochberg

def test_bh_gate_behavior():
    # Controlled p-values: first two significant under alpha=0.1 after ordering
    pvals = [0.02, 0.40, 0.001, 0.15, 0.08]
    keep, thresh = benjamini_hochberg(pvals, 0.1)
    assert any(keep)
    assert 0 <= thresh <= 0.1
    # All p-values inflated -> none kept
    p_bad = [0.5,0.6,0.7]
    keep2, thresh2 = benjamini_hochberg(p_bad, 0.1)
    assert not any(keep2)
    assert thresh2 == 0.0
