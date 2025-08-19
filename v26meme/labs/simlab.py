import pandas as pd

def _evaluate_formula(row, formula):
    if not isinstance(formula[0], list):
        feature, op, value = formula
        try:
            if op == '>': return row[feature] > value
            if op == '<': return row[feature] < value
        except KeyError:
            return False
    left, logical_op, right = formula
    if logical_op == 'AND':
        return _evaluate_formula(row, left) and _evaluate_formula(row, right)
    else:
        return _evaluate_formula(row, left) or _evaluate_formula(row, right)

class SimLab:
    def __init__(self, fees_bps, slippage_bps, slippage_table=None):
        self.fee = fees_bps/10000.0
        self.slippage = slippage_bps/10000.0
        self.slip_table = slippage_table or {}

    def _stats(self, returns: pd.Series) -> dict:
        if returns.empty: return {"n_trades": 0}
        equity = (1+returns).cumprod()
        dd = (equity - equity.cummax()) / equity.cummax()
        down_std = returns[returns < 0].std()
        return {
            "n_trades": int(returns.shape[0]),
            "win_rate": float((returns > 0).mean()),
            "avg_return": float(returns.mean()),
            "sortino": float(returns.mean()/down_std) if (down_std is not None and down_std>0) else 0.0,
            "sharpe": float(returns.mean()/returns.std()) if (returns.std() is not None and returns.std()>0) else 0.0,
            "mdd": float(dd.min()),
            "returns": [float(x) for x in returns.tolist()],
        }

    def _slip_for(self, display: str) -> float:
        bps = float(self.slip_table.get(display, 0.0))
        return max(self.slippage, bps/10000.0)

    def run_backtest(self, df: pd.DataFrame, formula: list) -> dict:
        if df.empty or len(df) < 60: return {}
        df = df.copy()
        # derive conservative slippage per-canonical if annotated
        display = df.attrs.get("display", None)
        slip = self._slip_for(display) if display else self.slippage
        signals = df.apply(lambda row: _evaluate_formula(row, formula), axis=1)
        edges = signals.diff().fillna(0)
        in_trade, entry_price = False, 0.0
        trades = []
        for i in range(len(df)):
            if edges.iloc[i] > 0 and not in_trade:
                in_trade, entry_price = True, df['close'].iloc[i] * (1 + slip)
            elif edges.iloc[i] < 0 and in_trade:
                in_trade = False
                exit_price = df['close'].iloc[i] * (1 - slip)
                trades.append(((exit_price-entry_price)/entry_price) - (2*self.fee))
        if in_trade:
            exit_price = df['close'].iloc[-1] * (1 - slip)
            trades.append(((exit_price-entry_price)/entry_price) - (2*self.fee))
        if not trades: return {"all": {"n_trades": 0}}
        ser = pd.Series(trades)
        return {"all": self._stats(ser)}
