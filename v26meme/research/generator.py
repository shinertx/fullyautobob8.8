import random, json, hashlib
from typing import List

class GeneticGenerator:
    def __init__(self, features: List[str], population_size: int, seed: int = 1337):
        """Genetic boolean formula generator.

        PIT: All operations deterministic under fixed seed.
        Adaptive knobs: mutation_jitter bounds exposed (no magic constants).
        """
        self.features = features
        self.operators = ['>', '<']
        self.logical_ops = ['AND', 'OR']
        self.population_size = population_size
        self.population = []
        self.mutation_jitter_min = -0.2
        self.mutation_jitter_max = 0.2
        self.threshold_min = -3.0
        self.threshold_max = 3.0
        random.seed(seed)

    def _create_random_condition(self):
        return [random.choice(self.features), random.choice(self.operators), random.uniform(-2.0, 2.0)]

    def _create_random_formula(self, d=2):
        return [self._create_random_formula(d-1), random.choice(self.logical_ops), self._create_random_formula(d-1)] if d>0 else self._create_random_condition()

    def initialize_population(self): 
        self.population = [self._create_random_formula() for _ in range(self.population_size)]

    def run_evolution_cycle(self, fitness_scores: dict):
        if not self.population: self.initialize_population()
        str_pop = {json.dumps(f): f for f in self.population}
        sorted_pop = sorted(str_pop.keys(), key=lambda s: fitness_scores.get(hashlib.sha256(s.encode()).hexdigest(), 0), reverse=True)
        elite_n = max(1, int(len(sorted_pop)*0.1))
        new_pop = sorted_pop[:elite_n]
        while len(new_pop) < self.population_size:
            p1, p2 = random.choices(sorted_pop[:max(2, len(sorted_pop)//2)], k=2)
            import json as _json
            child = _json.loads(random.choice([p1, p2]))
            if random.random() < 0.25:
                nodes = [n for n in self._get_subtrees(child) if not isinstance(n[0], list)]
                if nodes: 
                    import math
                    n = random.choice(nodes)
                    thr = float(n[2]);
                    jitter = 1.0 + random.uniform(self.mutation_jitter_min, self.mutation_jitter_max)
                    n[2] = max(self.threshold_min, min(self.threshold_max, thr * jitter))
            new_pop.append(_json.dumps(child))
        self.population = [json.loads(s) for s in new_pop]

    def _get_subtrees(self, formula):
        nodes, q = [], [formula]
        while q:
            node = q.pop(0); nodes.append(node)
            if isinstance(node[0], list): q.append(node[0])
            if isinstance(node[2], list): q.append(node[2])
        return nodes
