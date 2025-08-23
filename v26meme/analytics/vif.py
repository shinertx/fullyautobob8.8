import numpy as np
import pandas as pd
from typing import List

# PIT: Pure statistical transformation on already realized return series; no forward leakage.

def compute_vif(candidate: pd.Series, existing: List[pd.Series]) -> float:
    """Compute variance inflation factor (VIF) of a candidate return series relative to existing returns.

    VIF = 1 / (1 - R^2) where R^2 from OLS regression y ~ X (no intercept, demeaned) with y=candidate.
    Falls back to 1.0 when insufficient data or empty existing list.

    Parameters
    ----------
    candidate : pd.Series
        Candidate strategy returns (historical, realized, aligned to closed bars).
    existing : list[pd.Series]
        List of existing promoted alpha return series.

    Returns
    -------
    float
        Variance inflation factor (>=1). High values indicate multicollinearity / redundancy.
    """
    if not isinstance(candidate, pd.Series) or not existing:
        return 1.0
    # Align lengths conservatively (tail alignment to avoid forward look)
    min_len = min([len(candidate)] + [len(e) for e in existing])
    if min_len < 10:  # need enough observations
        return 1.0
    y = candidate.tail(min_len).to_numpy(dtype=float, copy=True)
    X_cols = []
    for e in existing:
        arr = e.tail(min_len).to_numpy(dtype=float, copy=True)
        X_cols.append(arr)
    if not X_cols:
        return 1.0
    X = np.column_stack(X_cols)
    # Demean to remove intercept (avoids adding constant col; stable for VIF calc)
    y_d = y - y.mean()
    X_d = X - X.mean(axis=0)
    try:
        # Solve least squares X_d beta = y_d
        beta, *_ = np.linalg.lstsq(X_d, y_d, rcond=None)
        y_hat = X_d @ beta
        ss_tot = np.sum(y_d**2)
        if ss_tot <= 0:
            return 1.0
        ss_res = np.sum((y_d - y_hat)**2)
        r2 = 1.0 - (ss_res / ss_tot)
        r2 = min(max(r2, 0.0), 0.999999)
        return float(1.0 / (1.0 - r2)) if r2 < 0.999999 else float('inf')
    except Exception:
        return 1.0
