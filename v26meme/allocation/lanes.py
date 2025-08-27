from typing import Dict, List
from loguru import logger
from statistics import mean
import math, time

from v26meme.core.dsl import Alpha


class LaneAllocationManager:
    """
    Dynamically scales lane budgets (core vs moonshot) based on recent risk-adjusted lane performance.
    Simple, deterministic rule:
      - Start from base fractions (core ~95%, moonshot ~5%).
      - Compute average Sortino of alphas per lane (if available).
      - If moonshot Sortino > core Sortino by margin, linearly shift up to moonshot.max_fraction.
      - If moonshot underperforms, clamp to min_fraction.
    """
    def __init__(self, cfg: dict, state):
        self.cfg_root = cfg
        self.cfg = cfg.get('lanes', {})
        self.state = state

    def _lane_stats(self, active_alphas: List[Alpha]) -> Dict[str, float]:
        lanes = {}
        for a in active_alphas:
            s = a.sortino()
            if s is None: continue
            lanes.setdefault(a.lane, []).append(float(s))
        return {k: mean(v) for k, v in lanes.items() if v}

    def _smooth(self, key: str, value: float) -> float:
        alpha = float(self.cfg.get('smoothing_alpha', 0.30))
        hist_key = f"lane:perf:ewma:{key}"
        prev = self.state.get(hist_key)
        prev_f = float(prev) if prev is not None else value
        ewma = (1 - alpha) * prev_f + alpha * value
        self.state.set(hist_key, ewma)
        return ewma

    def _budgets_from_stats(self, stats: Dict[str, float]) -> Dict[str, float]:
        core_cfg = self.cfg.get('core',     {'base_fraction': 0.95, 'min_fraction': 0.85, 'max_fraction': 0.99})
        moon_cfg = self.cfg.get('moonshot', {'base_fraction': 0.05, 'min_fraction': 0.01, 'max_fraction': 0.20})
        core_s = stats.get('core', 0.0); moon_s = stats.get('moonshot', 0.0)
        core_s = self._smooth('core', core_s)
        moon_s = self._smooth('moonshot', moon_s)
        dead = float(self.cfg.get('dead_band', 0.10))
        moon_target = moon_cfg['base_fraction']
        diff = moon_s - core_s
        if diff > dead:
            delta = min(moon_cfg['max_fraction'] - moon_cfg['base_fraction'], diff * 0.05)
            moon_target = moon_cfg['base_fraction'] + max(0.0, delta)
        elif diff < -dead:
            delta = min(moon_cfg['base_fraction'] - moon_cfg['min_fraction'], (-diff) * 0.05)
            moon_target = moon_cfg['base_fraction'] - max(0.0, delta)
        # step cap
        prev = self.state.get('lane:moonshot:last_budget')
        if prev is not None:
            try:
                prev_f = float(prev)
                step_cap = float(self.cfg.get('max_step_per_cycle', 0.02))
                moon_target = max(prev_f - step_cap, min(prev_f + step_cap, moon_target))
            except Exception:
                pass
        moon_target = max(moon_cfg['min_fraction'], min(moon_cfg['max_fraction'], moon_target))
        core_target = 1.0 - moon_target
        core_target = max(core_cfg['min_fraction'], min(core_cfg['max_fraction'], core_target))
        moon_target = 1.0 - core_target
        self.state.set('lane:moonshot:last_budget', moon_target)
        return {'core': core_target, 'moonshot': moon_target}

    def _apply_probation(self, weights: Dict[str, float], active_alphas: List[Alpha]) -> Dict[str, float]:
        prob_cfg = self.cfg.get('probation', {})
        trades_min = int(prob_cfg.get('trades_min', 50))
        cap = float(prob_cfg.get('weight_cap', 0.03))
        changed = False
        out = {}
        for a in active_alphas:
            aid = a.id
            w = weights.get(aid, 0.0)
            n_tr = a.trades()
            if a.lane == 'moonshot' and n_tr < trades_min and w > cap:
                w = cap
                changed = True
            out[aid] = w
        if changed:
            logger.info(f"Applied probation cap to some moonshot weights (cap={cap})")
        return out

    def _maybe_retag(self, active_alphas: List[Alpha]) -> int:
        rcfg = (self.cfg.get('retag') or {})
        if not rcfg.get('enabled', True):
            return 0
        min_tr = int(rcfg.get('min_trades', 300))
        min_sh = float(rcfg.get('min_sharpe', 1.0))
        min_so = float(rcfg.get('min_sortino', 1.2))
        max_mdd = float(rcfg.get('max_mdd', 0.25))
        retagged = 0
        for a in active_alphas:
            if a.lane == 'moonshot':
                if (a.trades() >= min_tr and a.sharpe() >= min_sh and
                    a.sortino() >= min_so and a.mdd() <= max_mdd):
                    a.lane = 'core'
                    retagged += 1
        if retagged:
            logger.info(f"Retagged {retagged} moonshot→core alphas")
        return retagged

    def apply_lane_budgets(self, alpha_weights: Dict[str, float], active_alphas: List[Alpha]) -> Dict[str, float]:
        if not alpha_weights: return {}
        # Retag pass (A)
        self._maybe_retag(active_alphas)  # modifies in-place
        stats = self._lane_stats(active_alphas)
        budgets = self._budgets_from_stats(stats)
        lane_sum = {'core': 0.0, 'moonshot': 0.0}
        alpha_lane = {a.id: a.lane for a in active_alphas}
        for aid, w in alpha_weights.items():
            lane_sum[alpha_lane.get(aid, 'core')] += float(w)
        out = {}
        for aid, w in alpha_weights.items():
            lane = alpha_lane.get(aid, 'core'); b = budgets.get(lane, 1.0)
            s = lane_sum.get(lane, 1e-9)
            scale = min(1.0, b / max(s, 1e-12)) if s>0 else 1.0
            out[aid] = float(w) * scale
        tot = sum(out.values())
        out = {k: v/tot for k,v in out.items()} if tot>0 else {}
        out = self._apply_probation(out, active_alphas)
        # Post-lane floor normalization to avoid weights below portfolio floor sneaking in
        try:
            floor = float(self.cfg_root.get('portfolio', {}).get('min_allocation_weight', 0.0))
            adjusted = {k: (0.0 if v < floor else v) for k,v in out.items()}
            s2 = sum(adjusted.values())
            if s2 > 0:
                adjusted = {k: v/s2 for k,v in adjusted.items()}
            out = adjusted
        except Exception:
            pass
        logger.info(f"Lane budgets applied: {budgets}, gross={sum(out.values()):.2f}")
        return out
