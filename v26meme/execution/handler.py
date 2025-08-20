from loguru import logger
import numpy as np
from v26meme.registry.canonical import venue_symbol_for

from v26meme.execution.micro_live import MicroLive

class ExecutionHandler:
    def __init__(self, state, exchange_factory, cfg, risk_manager=None):
        self.state, self.cfg = state, cfg['execution']
        self.exchange = exchange_factory.get_exchange(self.cfg['primary_exchange'])
        self.risk = risk_manager
        try: self.exchange.load_markets()
        except Exception as e: logger.warning(f"load_markets failed: {e}")
        self.fee = self.cfg['paper_fees_bps'] / 10000.0
        self.slippage = self.cfg['paper_slippage_bps'] / 10000.0
        self.max_order = cfg['risk']['max_order_notional_usd']

        ml_cfg = cfg.get('micro_live', {}) or {}
        self.micro = MicroLive(self.exchange, ml_cfg.get('notional_usd', 3.0),
                               ml_cfg.get('cycle_budget_trades', 5), ml_cfg.get('enabled', False))

    def _ex_symbol(self, canonical):
        return venue_symbol_for(self.exchange, canonical)

    def _price_to_precision(self, ex_sym: str, price: float) -> float:
        try: return float(self.exchange.price_to_precision(ex_sym, price))
        except Exception: return float(price)

    def _amount_to_precision(self, ex_sym: str, amount: float) -> float:
        try: return float(self.exchange.amount_to_precision(ex_sym, amount))
        except Exception: return float(amount)

    def _get_price(self, ex_sym: str) -> float:
        try:
            t = self.exchange.fetch_ticker(ex_sym)
            p = t.get('last') or t.get('close')
            return float(p) if p else 0.0
        except Exception as e:
            logger.error(f"fetch_ticker failed for {ex_sym}: {e}")
            return 0.0

    def _min_cost_ok(self, ex_sym: str, usd_value: float) -> bool:
        try:
            m = self.exchange.market(ex_sym)
            min_cost = (m.get('limits', {}) or {}).get('cost', {}).get('min')
            if min_cost is not None and usd_value < float(min_cost): return False
        except Exception: pass
        return usd_value <= self.max_order

    def reconcile(self, target_weights: dict, active_alphas: list):
        logger.info("Reconciliation cycle initiated...")

        cap = self.state.get('risk:current_max_order')
        if cap is not None:
            try: self.max_order = float(cap)
            except Exception: pass

        portfolio = self.state.get_portfolio()
        cash = float(portfolio.get('cash', 0.0))
        positions = dict(portfolio.get('positions', {}))

        alpha_to_symbol = {a['id']: a['universe'][0] for a in active_alphas}
        symbol_target = {}
        for aid, w in (target_weights or {}).items():
            sym = alpha_to_symbol.get(aid)
            if not sym: continue
            symbol_target[sym] = symbol_target.get(sym, 0.0) + float(w)

        equity_mark = cash
        price_cache = {}

        # Early risk halt flatten
        if self.state.get('risk:halted'):
            logger.warning("RISK HALTED - flatten all positions now")
            for sym in list(positions.keys()):
                symbol_target[sym] = 0.0
            active_alphas = []
            target_weights = {}

        for sym in set(list(symbol_target.keys()) + list(positions.keys())):
            ex_sym = self._ex_symbol(sym)
            if not ex_sym:
                logger.error(f"No exchange mapping for {sym}")
                price_cache[sym] = 0.0; continue
            px = self._get_price(ex_sym)
            price_cache[sym] = px
            amt = float(positions.get(sym, {}).get('amount', 0.0))
            positions[sym] = {'amount': amt, 'usd_value': amt * px}
            equity_mark += amt * px

        if self.risk:
            symbol_target = self.risk.enforce(symbol_target, equity_mark)
            # If halted, flatten current positions to cash
            if self.state.get('risk:halted'):
                logger.warning("Risk halted — flattening all positions to cash.")
                # force all held symbols to zero target to close them
                for sym in list(positions.keys()):
                    symbol_target[sym] = 0.0

        if self.cfg['mode'] == 'paper':
            for sym, w in (symbol_target or {}).items():
                ex_sym = self._ex_symbol(sym)
                if not ex_sym: continue
                px = price_cache.get(sym, 0.0)
                if px <= 0.0: continue
                target_usd = equity_mark * w
                cur_amt = float(positions.get(sym, {}).get('amount', 0.0))
                cur_usd = cur_amt * px
                delta_usd = target_usd - cur_usd
                if abs(delta_usd) < 1e-6: continue
                if not self._min_cost_ok(ex_sym, abs(delta_usd)):
                    logger.info(f"{ex_sym} trade ${abs(delta_usd):.2f} violates min/max notional; skipping")
                    continue

                side = 'buy' if delta_usd > 0 else 'sell'
                fill_px = px * (1 + self.slippage) if side == 'buy' else px * (1 - self.slippage)
                # Dynamic slippage override if table entry exists (bps)
                try:
                    slip_table = self.state.get('slippage:table') or {}
                    bps_entry = slip_table.get(sym)
                    if isinstance(bps_entry, (int, float)) and bps_entry >= 0:
                        dyn_slip = float(bps_entry) / 10000.0
                        fill_px = px * (1 + dyn_slip) if side == 'buy' else px * (1 - dyn_slip)
                except Exception:
                    pass
                fill_px = self._price_to_precision(ex_sym, fill_px)
                qty = abs(delta_usd) / max(1e-12, fill_px)
                qty = self._amount_to_precision(ex_sym, qty)
                if qty <= 0: continue
                trade_cost = qty * fill_px
                fees = trade_cost * self.fee

                if side == 'buy':
                    if cash < (trade_cost + fees):
                        logger.info(f"Insufficient cash for {ex_sym} buy ${trade_cost+fees:.2f}; skipping"); continue
                    cash -= (trade_cost + fees); cur_amt += qty
                else:
                    qty = min(qty, cur_amt)
                    if qty <= 0: continue
                    proceeds = qty * fill_px
                    cash += (proceeds - fees); cur_amt -= qty

                positions[sym] = {'amount': cur_amt, 'usd_value': cur_amt * px}

            equity = cash + sum(v['usd_value'] for v in positions.values())
            portfolio['cash'] = cash; portfolio['positions'] = positions; portfolio['equity'] = equity
            self.state.set_portfolio(portfolio)
            logger.info(f"[PAPER] Equity ${equity:.2f}, Cash ${cash:.2f}, Pos {len(positions)}")
            if self.risk: self.risk.reset_errors()

            # micro-live probes and slippage calibration table update
            def _record(res):
                key = f"micro:exec:{res.get('display','?')}"
                hist = self.state.get(key) or []; hist.append(res); hist = hist[-400:]
                self.state.set(key, hist)
                # update slippage p90 table
                bps = [h.get("slip_bps") for h in hist if h.get("ok")==1 and isinstance(h.get("slip_bps"), (int,float))]
                if len(bps)>=20:
                    p90 = float(np.percentile(bps, 90))
                    table = self.state.get("slippage:table") or {}
                    table[res.get('display','?')] = p90
                    self.state.set("slippage:table", table)

            if self.micro.enabled:
                insts = [a.get('instrument') for a in active_alphas if a.get('instrument')]
                if insts: self.micro.sample(insts, _record)
            return

        if self.cfg['mode'] == 'live':
            logger.warning("[LIVE] Disabled by default."); return
