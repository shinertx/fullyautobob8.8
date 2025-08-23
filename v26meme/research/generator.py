import random, json, hashlib
from typing import List, Dict, Optional, Tuple

class GeneticGenerator:
    def __init__(self, features: List[str], population_size: int, seed: int = 1337,
                 feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
                 quantile_low: float = 0.10, quantile_high: float = 0.90,
                 max_tree_depth: int = 3,
                 mutation_cfg: Optional[Dict[str, float]] = None,
                 elite_fraction: float = 0.10,
                 survivor_mix_fraction: float = 0.25):
        """Genetic boolean formula generator (adaptive threshold domain, configurable depth & mutation).

        PIT: All randomness seeded; relies only on supplied *feature_stats* (closed-bar derived).

        Parameters (adaptive knobs, no magic numbers):
          max_tree_depth: maximum recursive depth for newly created formulas.
          mutation_cfg: dict with rates (threshold_rate, feature_rate, operator_rate, logical_rate, jitter_min, jitter_max)
          elite_fraction: fraction of top unique structures carried forward unchanged.
          survivor_mix_fraction: fraction of new population reseeded from elite via single-mutation perturbations.
        """
        self.features = features
        self.operators = ['>', '<']
        self.logical_ops = ['AND', 'OR']
        self.population_size = population_size
        self.population: List[list] = []
        self.feature_stats = feature_stats or {}
        self.q_low = quantile_low
        self.q_high = quantile_high
        self.max_tree_depth = max_tree_depth
        mcfg = mutation_cfg or {}
        self.mutation_threshold_rate = float(mcfg.get('threshold_rate', 0.30))
        self.mutation_feature_rate = float(mcfg.get('feature_rate', 0.10))
        self.mutation_operator_rate = float(mcfg.get('operator_rate', 0.10))
        self.mutation_logical_rate = float(mcfg.get('logical_rate', 0.10))
        self.mutation_jitter_min = float(mcfg.get('jitter_min', -0.2))
        self.mutation_jitter_max = float(mcfg.get('jitter_max', 0.2))
        self.threshold_min = -3.0
        self.threshold_max = 3.0
        self.elite_fraction = min(0.90, max(0.0, elite_fraction))
        self.survivor_mix_fraction = min(0.90, max(0.0, survivor_mix_fraction))
        random.seed(seed)

    def set_feature_stats(self, feature_stats: Dict[str, Dict[str, float]]) -> None:
        """Inject / replace empirical feature stats (PIT-safe) for adaptive threshold sampling."""
        self.feature_stats = feature_stats or {}

    # ---------------- internal helpers -----------------
    def _sample_threshold(self, feat: str) -> float:
        st = self.feature_stats.get(feat)
        if st:
            ql = float(st.get('q_low', st.get('min', self.threshold_min)))
            qh = float(st.get('q_high', st.get('max', self.threshold_max)))
            if ql == qh:
                # expand slightly to avoid zero-width domain
                span = abs(ql) if ql != 0 else 1.0
                ql -= 0.05 * span
                qh += 0.05 * span
            return random.uniform(min(ql, qh), max(ql, qh))
        # fallback: feature raw min/max if present
        if st and 'min' in st and 'max' in st:
            return random.uniform(float(st['min']), float(st['max']))
        # ultimate fallback legacy static domain
        return random.uniform(-2.0, 2.0)

    def _create_random_condition(self):
        feat = random.choice(self.features)
        return [feat, random.choice(self.operators), self._sample_threshold(feat)]

    def _create_random_formula(self, max_depth=None):
        if max_depth is None:
            max_depth = self.max_tree_depth
        if max_depth <= 0 or random.random() > 0.5:
            return self._create_random_condition()
        return [self._create_random_formula(max_depth-1), random.choice(self.logical_ops), self._create_random_formula(max_depth-1)]

    def initialize_population(self): 
        self.population = [self._create_random_formula() for _ in range(self.population_size)]

    def _crossover(self, p1, p2):
        child = json.loads(json.dumps(p1))
        p2_copy = json.loads(json.dumps(p2))

        nodes1 = self._get_subtrees(child)
        nodes2 = self._get_subtrees(p2_copy)
        
        crossover_point_in_child = random.choice(nodes1)
        node_to_insert = random.choice(nodes2)

        if crossover_point_in_child is child:
            return node_to_insert

        q = [child]
        while q:
            parent = q.pop(0)
            if isinstance(parent[0], list):
                if parent[0] is crossover_point_in_child:
                    parent[0] = node_to_insert
                    return child
                q.append(parent[0])
            if len(parent) > 2 and isinstance(parent[2], list):
                if parent[2] is crossover_point_in_child:
                    parent[2] = node_to_insert
                    return child
                q.append(parent[2])
        return child

    def run_evolution_cycle(self, fitness_scores: dict):
        if not self.population:
            self.initialize_population()
        str_pop = {json.dumps(f, separators=(',',':')): f for f in self.population}
        # Elite selection
        sorted_pop = sorted(str_pop.keys(), key=lambda s: (-fitness_scores.get(hashlib.sha256(s.encode()).hexdigest(), 0.0), s))
        elite_n = max(1, int(len(sorted_pop)*self.elite_fraction))
        elites = sorted_pop[:elite_n]
        new_structs = elites[:]  # retain elites
        # Survivor mix (perturbed elites) pre-fill
        mix_target = max(0, int(self.population_size * self.survivor_mix_fraction))
        while len(new_structs) < elite_n + mix_target and len(new_structs) < self.population_size:
            base = json.loads(random.choice(elites))
            # single mutation perturbation
            nodes = [n for n in self._get_subtrees(base) if not isinstance(n[0], list)]
            if nodes:
                n = random.choice(nodes)
                # jitter threshold
                try:
                    feat = n[0]; cur_thr = float(n[2])
                    jitter = 1.0 + random.uniform(self.mutation_jitter_min, self.mutation_jitter_max)
                    n[2] = max(self.threshold_min, min(self.threshold_max, cur_thr * jitter))
                except Exception:
                    pass
            new_structs.append(json.dumps(base, separators=(',',':')))
        # Crossover + standard mutation
        while len(new_structs) < self.population_size:
            p1_str, p2_str = random.choices(elites if len(elites) >=2 else sorted_pop[:max(2,len(sorted_pop)//2)], k=2)
            p1 = json.loads(p1_str); p2 = json.loads(p2_str)
            child = self._crossover(p1, p2)
            # Mutations (config-driven rates)
            if random.random() < self.mutation_threshold_rate:
                nodes = [n for n in self._get_subtrees(child) if not isinstance(n[0], list)]
                if nodes:
                    n = random.choice(nodes)
                    try:
                        feat = n[0]; cur_thr = float(n[2])
                        if random.random() < 0.40:
                            n[2] = self._sample_threshold(feat)
                        else:
                            jitter = 1.0 + random.uniform(self.mutation_jitter_min, self.mutation_jitter_max)
                            n[2] = max(self.threshold_min, min(self.threshold_max, cur_thr * jitter))
                    except Exception:
                        pass
            if random.random() < self.mutation_feature_rate:
                nodes = [n for n in self._get_subtrees(child) if not isinstance(n[0], list)]
                if nodes:
                    random.choice(nodes)[0] = random.choice(self.features)
            if random.random() < self.mutation_operator_rate:
                nodes = [n for n in self._get_subtrees(child) if not isinstance(n[0], list)]
                if nodes:
                    random.choice(nodes)[1] = random.choice(self.operators)
            if random.random() < self.mutation_logical_rate:
                nodes = [n for n in self._get_subtrees(child) if isinstance(n[0], list)]
                if nodes:
                    random.choice(nodes)[1] = random.choice(self.logical_ops)
            new_structs.append(json.dumps(child, separators=(',',':')))
        # Deduplicate preserving order
        seen = set(); final = []
        for s in new_structs:
            if s not in seen:
                seen.add(s); final.append(json.loads(s))
            if len(final) >= self.population_size:
                break
        self.population = final

    def _get_subtrees(self, formula):
        nodes, q = [], [formula]
        while q:
            node = q.pop(0); nodes.append(node)
            if isinstance(node[0], list): q.append(node[0])
            if len(node) > 2 and isinstance(node[2], list): q.append(node[2])
        return nodes
