from typing import Dict, List
from loguru import logger
from statistics import mean

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
        self.cfg = cfg.get('lanes', {})
        self.state = state

    def _lane_stats(self, active_alphas: List[Dict]) -> Dict[str, float]:
        lanes = {}
        for a in active_alphas:
            lane = a.get('lane', 'core')
            s = (a.get('performance', {}).get('all', {}) or {}).get('sortino', None)
            if s is None: continue
            lanes.setdefault(lane, []).append(float(s))
        return {k: mean(v) for k, v in lanes.items() if v}

    def _budgets_from_stats(self, stats: Dict[str, float]) -> Dict[str, float]:
        core_cfg = self.cfg.get('core',     {'base_fraction': 0.95, 'min_fraction': 0.85, 'max_fraction': 0.99})
        moon_cfg = self.cfg.get('moonshot', {'base_fraction': 0.05, 'min_fraction': 0.01, 'max_fraction': 0.20})
        core_s = stats.get('core', 0.0); moon_s = stats.get('moonshot', 0.0)

        moon = moon_cfg['base_fraction']
        if moon_s > core_s:
            # shift toward max by margin
            delta = min(moon_cfg['max_fraction'] - moon_cfg['base_fraction'], (moon_s - core_s) * 0.05)
            moon = moon_cfg['base_fraction'] + max(0.0, delta)
        else:
            # degrade toward min by margin
            delta = min(moon_cfg['base_fraction'] - moon_cfg['min_fraction'], (core_s - moon_s) * 0.05)
            moon = moon_cfg['base_fraction'] - max(0.0, delta)

        moon = max(moon_cfg['min_fraction'], min(moon_cfg['max_fraction'], moon))
        core = 1.0 - moon
        core = max(core_cfg['min_fraction'], min(core_cfg['max_fraction'], core))
        moon = 1.0 - core
        return {'core': core, 'moonshot': moon}

    def apply_lane_budgets(self, alpha_weights: Dict[str, float], active_alphas: List[Dict]) -> Dict[str, float]:
        if not alpha_weights: return {}
        stats = self._lane_stats(active_alphas)
        budgets = self._budgets_from_stats(stats)
        # Sum weights per lane
        lane_sum = {'core': 0.0, 'moonshot': 0.0}
        alpha_lane = {}
        for a in active_alphas:
            lid = a['id']; lane = a.get('lane','core'); alpha_lane[lid] = lane
        for aid, w in alpha_weights.items():
            lane_sum[alpha_lane.get(aid, 'core')] += float(w)
        # Scale down lanes that exceed budget; keep others unchanged
        out = {}
        for aid, w in alpha_weights.items():
            lane = alpha_lane.get(aid, 'core'); b = budgets.get(lane, 1.0)
            s = lane_sum.get(lane, 1e-9)
            scale = min(1.0, b / max(s, 1e-12)) if s>0 else 1.0
            out[aid] = float(w) * scale
        # renormalize
        tot = sum(out.values())
        out = {k: v/tot for k,v in out.items()} if tot>0 else {}
        logger.info(f"Lane budgets applied: {budgets}, gross={sum(out.values()):.2f}")
        return out
