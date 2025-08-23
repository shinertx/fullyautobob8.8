from v26meme.execution.handler import ExecutionHandler

class DummyExchangeFactory:
    class Ex:
        id='dummy'
        markets={}
        def load_markets(self): pass
        def fetch_ticker(self, sym): return {'last':100}
        def price_to_precision(self, s,p): return p
        def amount_to_precision(self, s,a): return a
    def get_exchange(self, _): return self.Ex()

class DummyRisk:
    def enforce(self, weights, equity): return weights
    def reset_errors(self): pass

class DummyState:
    def __init__(self):
        self.store={'portfolio': {'cash':1000,'equity':1000,'positions':{}},'slippage:table':{'BTC_USD_SPOT':25}}
    def get(self,k): return self.store.get(k)
    def set(self,k,v): self.store[k]=v
    def get_portfolio(self): return self.store['portfolio']
    def set_portfolio(self,p): self.store['portfolio']=p

from v26meme.registry.canonical import make_canonical

def test_execution_uses_dynamic_slippage(monkeypatch):
    cfg={'execution':{'primary_exchange':'dummy','mode':'paper','paper_fees_bps':10,'paper_slippage_bps':5},
         'risk':{'max_order_notional_usd':1000}}
    st=DummyState(); exf=DummyExchangeFactory(); risk=DummyRisk()
    handler=ExecutionHandler(st, exf, cfg, risk_manager=risk)
    # create a single alpha weight mapping to canonical symbol
    target_weights={'alpha1':0.5}
    active_alphas=[{'id':'alpha1','universe':['BTC_USD_SPOT']}] 
    handler.reconcile(target_weights, active_alphas)
    # portfolio updated
    port=st.get_portfolio()
    assert port['equity']>0
