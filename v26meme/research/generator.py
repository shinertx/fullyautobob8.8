import random, json, hashlib
from typing import List, Dict, Optional, Tuple

class GeneticGenerator:
    def __init__(self, features: List[str], population_size: int, seed: int = 1337,
                 feature_stats: Optional[Dict[str, Dict[str, float]]] = None,
                 quantile_low: float = 0.10, quantile_high: float = 0.90,
                 max_formula_depth: int = 2, depth3_prob: float = 0.0):
        """Genetic boolean formula generator (adaptive threshold domain).

        PIT: All randomness seeded; uses only historical *feature_stats* passed in (empirical
        quantiles computed from already harvested, closed bars) – no forward leakage.

        Parameters
        ----------
        features : list[str]
            Candidate feature names.
        population_size : int
            Target population size (fixed per evolution cycle).
        seed : int
            RNG seed for deterministic reproducibility.
        feature_stats : dict | None
            Mapping feature → {"q_low","q_high","min","max"} (empirical). If provided, the
            random threshold sampling domain for that feature is [q_low, q_high]. Fallback to
            [min, max] then static [-2,2] if absent (maintains backward compatibility).
        quantile_low / quantile_high : float
            Adaptive quantile bounds (configurable knobs, no magic numbers) used when feature_stats
            present. Default 10%–90% band encourages mid-distribution variance (reduces constant predicates).
        max_formula_depth : int
            Maximum recursive boolean tree depth (>=2). Raising depth increases interaction space.
        depth3_prob : float
            Probability of sampling depth=max_formula_depth (allows controlled deeper exploration without exploding search space).
        """
        self.features = features
        self.operators = ['>', '<']
        self.logical_ops = ['AND', 'OR']
        self.population_size = population_size
        self.population: List[list] = []
        # Mutation jitter bounds (adaptive knobs; kept explicit & documented)
        self.mutation_jitter_min = -0.2
        self.mutation_jitter_max = 0.2
        # Absolute hard safety rails (fallback if stats missing)
        self.threshold_min = -3.0
        self.threshold_max = 3.0
        self.feature_stats = feature_stats or {}
        self.q_low = quantile_low
        self.q_high = quantile_high
        self.max_formula_depth = max(2, int(max_formula_depth))
        self.depth3_prob = max(0.0, min(1.0, depth3_prob))
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

    def _create_random_formula(self, max_depth=2):
        # Allow deeper trees based on configured probability (depth3_prob) and max_formula_depth
        if max_depth == 2 and self.max_formula_depth >= 3 and random.random() < self.depth3_prob:
            max_depth = self.max_formula_depth
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
        if not self.population: self.initialize_population()
        # Deduplicate by structural JSON form
        str_pop = {json.dumps(f, separators=(',',':')): f for f in self.population}
        # Rank by fitness (deterministic tie-break via string)
        sorted_pop = sorted(str_pop.keys(), key=lambda s: (
            -fitness_scores.get(hashlib.sha256(s.encode()).hexdigest(), 0.0), s))
        elite_n = max(1, int(len(sorted_pop)*0.1))
        new_pop = sorted_pop[:elite_n]
        # Crossover + mutation
        while len(new_pop) < self.population_size:
            p1_str, p2_str = random.choices(sorted_pop[:max(2, len(sorted_pop)//2)], k=2)
            p1 = json.loads(p1_str)
            p2 = json.loads(p2_str)
            child = self._crossover(p1, p2)

            # Mutation
            if random.random() < 0.30:  # threshold mutation
                nodes = [n for n in self._get_subtrees(child) if not isinstance(n[0], list)]
                if nodes:
                    n = random.choice(nodes)
                    try:
                        feat = n[0]
                        cur_thr = float(n[2])
                        jitter = 1.0 + random.uniform(self.mutation_jitter_min, self.mutation_jitter_max)
                        base_thr = cur_thr * jitter
                        if random.random() < 0.40:
                            base_thr = self._sample_threshold(feat)
                        n[2] = max(self.threshold_min, min(self.threshold_max, base_thr))
                    except Exception:
                        pass
            
            if random.random() < 0.10: # feature mutation
                nodes = [n for n in self._get_subtrees(child) if not isinstance(n[0], list)]
                if nodes:
                    n = random.choice(nodes)
                    n[0] = random.choice(self.features)

            if random.random() < 0.10: # operator mutation
                nodes = [n for n in self._get_subtrees(child) if not isinstance(n[0], list)]
                if nodes:
                    n = random.choice(nodes)
                    n[1] = random.choice(self.operators)

            if random.random() < 0.10: # logical op mutation
                nodes = [n for n in self._get_subtrees(child) if isinstance(n[0], list)]
                if nodes:
                    n = random.choice(nodes)
                    n[1] = random.choice(self.logical_ops)

            new_pop.append(json.dumps(child))
        self.population = [json.loads(s) for s in new_pop]

    def _get_subtrees(self, formula):
        nodes, q = [], [formula]
        while q:
            node = q.pop(0); nodes.append(node)
            if isinstance(node[0], list): q.append(node[0])
            if len(node) > 2 and isinstance(node[2], list): q.append(node[2])
        return nodes
