import random, json, hashlib
from typing import List, Dict, Optional, Tuple, Any
from collections import deque

class GeneticGenerator:
    def __init__(self, features: List[str], population_size: int, seed: int = 42, 
                 feature_stats: Optional[Dict[str, Dict[str, float]]] = None, 
                 fitness_weights: Optional[Dict[str, float]] = None,
                 stagnation_config: Optional[Dict] = None):
        """
        Initializes the GeneticGenerator.

        Args:
            features: A list of feature names to be used in formulas.
            population_size: The number of formulas in the population.
            seed: A random seed for reproducibility.
            feature_stats: Pre-computed statistics for each feature (min, max, etc.).
            fitness_weights: A dictionary of component weights for composite fitness calculation.
            stagnation_config: Configuration for stagnation detection and handling.
        """
        self.features = features
        self.population_size = population_size
        self.population: List[List] = []
        self.operators = ['<', '>', '<=', '>=']
        self.logical_operators = ['AND', 'OR']
        self.max_tree_depth = 3  # Controls complexity: 1=single rule, 2=A&(B|C) etc.
        self.feature_stats = feature_stats or {}
        random.seed(seed)

        # --- Genetic Algorithm Hyperparameters ---
        self.elite_fraction = 0.10  # Keep top 10% of formulas
        self.survivor_mix_fraction = 0.20 # In new generation, 20% are survivors from previous
        self.tournament_size = 3 # For parent selection
        
        # Mutation rates
        self.mutation_threshold_rate = 0.40
        self.mutation_feature_rate = 0.20
        self.mutation_operator_rate = 0.10
        self.mutation_logical_rate = 0.05
        
        # Mutation jitter and bounds
        self.mutation_jitter_min = -0.1
        self.mutation_jitter_max = 0.1
        self.threshold_min = -2.0
        self.threshold_max = 2.0

        # --- Composite Fitness & Stagnation ---
        self.fitness_weights = fitness_weights or {'profit_signal': 1.0}
        self.complexity_penalty_weight = self.fitness_weights.get('complexity_penalty', 0.001) # Penalize complexity
        self.stagnation_config = stagnation_config or {}
        if self.stagnation_config.get('enabled'):
            self.fitness_history = deque(maxlen=self.stagnation_config.get('window', 8))
        else:
            self.fitness_history = None
        
        self._original_mutation_rates = {
            'threshold': self.mutation_threshold_rate,
            'feature': self.mutation_feature_rate,
            'operator': self.mutation_operator_rate,
            'logical': self.mutation_logical_rate,
        }

    def set_feature_stats(self, feature_stats: Dict[str, Dict[str, float]]) -> None:
        """Inject / replace empirical feature stats (PIT-safe) for adaptive threshold sampling."""
        self.feature_stats = feature_stats or {}

    def set_features(self, features: List[str]) -> None:
        """Update the list of active features for subsequent mutations/creations."""
        self.features = features

    # ---------------- internal helpers -----------------
    def _get_composite_fitness(self, formula_hash: str, fitness_scores: dict) -> float:
        """Calculates a weighted fitness score from multiple signal components."""
        score_components = fitness_scores.get(formula_hash, {})
        if not isinstance(score_components, dict):
            # Backwards compatibility: if score is a float, treat it as the only component.
            return float(score_components)

        total_score = 0.0
        # Ensure weights sum to 1 to normalize score, or handle as relative importance
        for component, weight in self.fitness_weights.items():
            total_score += score_components.get(component, 0.0) * weight
        return total_score

    def _sample_threshold(self, feat: str) -> float:
        st = self.feature_stats.get(feat)
        
        # Prefer sampling from the inter-quartile or inter-decile range
        if st and 'q25' in st and 'q75' in st:
            q25 = float(st['q25'])
            q75 = float(st['q75'])
            if q25 < q75:
                return random.uniform(q25, q75)

        if st and 'q10' in st and 'q90' in st:
            q10 = float(st['q10'])
            q90 = float(st['q90'])
            if q10 < q90:
                return random.uniform(q10, q90)

        # Fallback to full min/max range
        if st and 'min' in st and 'max' in st:
            min_val = float(st['min'])
            max_val = float(st['max'])
            if min_val < max_val:
                return random.uniform(min_val, max_val)

        # Ultimate fallback to legacy static domain
        return random.uniform(self.threshold_min, self.threshold_max)

    def _create_random_condition(self):
        feat = random.choice(self.features)
        return [feat, random.choice(self.operators), self._sample_threshold(feat)]

    def _create_random_formula(self, max_depth=None):
        if max_depth is None:
            max_depth = self.max_tree_depth
        if max_depth <= 0 or random.random() > 0.75:
            return self._create_random_condition()
        return [self._create_random_formula(max_depth-1), random.choice(self.logical_operators), self._create_random_formula(max_depth-1)]

    def initialize_population(self): 
        self.population = [self._create_random_formula() for _ in range(self.population_size)]

    def seed_from_proposals(self, proposals: List[List[Any]]) -> int:
        """
        Seeds the population with external proposals, replacing random individuals.
        This is used to inject ideas from sources like an LLM.
        Returns the number of unique proposals injected.
        """
        if not proposals:
            return 0

        # Ensure population is initialized
        if not self.population:
            self.initialize_population()

        # Deduplicate proposals against the current population
        current_pop_hashes = {hashlib.sha256(json.dumps(f, separators=(',',':')).encode()).hexdigest() for f in self.population}
        
        unique_proposals = []
        for p in proposals:
            try:
                h = hashlib.sha256(json.dumps(p, separators=(',',':')).encode()).hexdigest()
                if h not in current_pop_hashes:
                    unique_proposals.append(p)
                    current_pop_hashes.add(h)
            except (TypeError, ValueError):
                continue # Skip proposals that can't be serialized

        if not unique_proposals:
            return 0

        # Replace random individuals in the population with the new proposals
        num_to_replace = min(len(self.population), len(unique_proposals))
        if num_to_replace > 0:
            replace_indices = random.sample(range(len(self.population)), num_to_replace)
            for i, proposal in zip(replace_indices, unique_proposals):
                self.population[i] = proposal
        
        return len(unique_proposals)

    def _tournament_selection(self, pool: List[str], fitness_scores: dict) -> str:
        """Selects a parent from a pool using tournament selection."""
        if not pool:
            raise ValueError("Cannot perform tournament selection on an empty pool.")
        
        # If pool is smaller than tournament size, just pick one randomly.
        # random.sample requires population to be at least k.
        if len(pool) < self.tournament_size:
            return random.choice(pool)

        tournament_contenders = random.sample(pool, self.tournament_size)
        
        winner = max(
            tournament_contenders, 
            key=lambda s: self._get_composite_fitness(hashlib.sha256(s.encode()).hexdigest(), fitness_scores)
        )
        return winner

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
        
        # Elite selection using composite fitness with complexity penalty
        def get_sort_key(s: str) -> Tuple[float, str]:
            formula = str_pop[s]
            # Complexity is the number of nodes in the formula's abstract syntax tree
            complexity = len(self._get_subtrees(formula))
            
            # Raw fitness from backtesting results
            raw_fitness = self._get_composite_fitness(hashlib.sha256(s.encode()).hexdigest(), fitness_scores)
            
            # Apply penalty: fitness decreases as complexity increases
            penalized_fitness = raw_fitness - (complexity * self.complexity_penalty_weight)
            
            # Sort descending by penalized fitness
            return (-penalized_fitness, s)

        sorted_pop = sorted(
            str_pop.keys(), 
            key=get_sort_key
        )
        elite_n = max(1, int(len(sorted_pop) * self.elite_fraction))
        elites = sorted_pop[:elite_n]

        # Stagnation check
        elite_fitnesses = [self._get_composite_fitness(hashlib.sha256(s.encode()).hexdigest(), fitness_scores) for s in elites]
        best_fitness = elite_fitnesses[0] if elite_fitnesses else 0.0
        self._check_and_handle_stagnation(best_fitness)

        # If all elites have zero fitness, the search is stuck. Broaden the pool to escape.
        is_stuck = all(f == 0.0 for f in elite_fitnesses)
        
        new_structs = elites[:]  # Elites are always preserved

        # --- Survivor Mix: Perturb top formulas to fill a portion of the next generation ---
        mix_target = max(0, int(self.population_size * self.survivor_mix_fraction))
        
        # If stuck, draw survivors from the whole population, not just non-performing elites
        survivor_pool = sorted_pop if is_stuck else elites
        
        num_to_mix = elite_n + mix_target
        while len(new_structs) < num_to_mix and len(new_structs) < self.population_size:
            if not survivor_pool: break
            base_str = random.choice(survivor_pool)
            base = json.loads(base_str)
            
            # Apply a single mutation/perturbation to the survivor candidate
            nodes = [n for n in self._get_subtrees(base) if not isinstance(n[0], list)]
            if nodes:
                n = random.choice(nodes)
                try:
                    feat = n[0]
                    cur_thr = float(n[2])
                    jitter = 1.0 + random.uniform(self.mutation_jitter_min, self.mutation_jitter_max)
                    n[2] = max(self.threshold_min, min(self.threshold_max, cur_thr * jitter))
                except (ValueError, IndexError):
                    pass # Ignore malformed nodes
            new_structs.append(json.dumps(base, separators=(',',':')))

        # --- Crossover & Mutation: Fill the rest of the population ---
        # If stuck, breed from the whole population to increase diversity. Otherwise, focus on elites.
        parent_pool = sorted_pop if is_stuck else elites
        if len(parent_pool) < 2:
            parent_pool = sorted_pop if len(sorted_pop) >= 2 else []

        while len(new_structs) < self.population_size:
            if len(parent_pool) < 2:
                # Not enough parents to breed, inject a purely random formula
                new_structs.append(json.dumps(self._create_random_formula(), separators=(',',':')))
                continue

            p1_str = self._tournament_selection(parent_pool, fitness_scores)
            p2_str = self._tournament_selection(parent_pool, fitness_scores)
            p1 = json.loads(p1_str)
            p2 = json.loads(p2_str)
            child = self._crossover(p1, p2)
            
            # Apply standard mutations based on configured rates
            self._mutate(child)
            new_structs.append(json.dumps(child, separators=(',',':')))
        
        # --- Finalize Population: Deduplicate and ensure size is correct ---
        seen = set()
        final_population = []
        for s in new_structs:
            if s not in seen:
                seen.add(s)
                final_population.append(json.loads(s))
            if len(final_population) >= self.population_size:
                break
        
        # If deduplication resulted in a smaller population, top up with random formulas
        while len(final_population) < self.population_size:
            final_population.append(self._create_random_formula())

        self.population = final_population

    def _check_and_handle_stagnation(self, best_fitness: float):
        """Checks for fitness stagnation and bumps mutation rates if necessary."""
        if not self.stagnation_config.get('enabled') or self.fitness_history is None:
            return

        self.fitness_history.append(best_fitness)

        # Not enough history to make a decision, ensure rates are normal
        if self.fitness_history.maxlen is None or len(self.fitness_history) < self.fitness_history.maxlen:
            self._reset_mutation_rates()
            return

        # Improvement check: is current best better than the oldest score by a delta?
        min_delta = self.stagnation_config.get('min_pval_delta', 0.02)
        if best_fitness > self.fitness_history[0] + min_delta:
            self._reset_mutation_rates() # Improvement seen, reset rates
        else:
            self._bump_mutation_rates() # Stagnation detected, bump rates

    def _reset_mutation_rates(self):
        """Resets mutation rates to their original configured values."""
        self.mutation_threshold_rate = self._original_mutation_rates['threshold']
        self.mutation_feature_rate = self._original_mutation_rates['feature']
        self.mutation_operator_rate = self._original_mutation_rates['operator']
        self.mutation_logical_rate = self._original_mutation_rates['logical']

    def _bump_mutation_rates(self):
        """Increases mutation rates by the configured bump amount."""
        bump = self.stagnation_config.get('mutation_bump', 0.15)
        self.mutation_threshold_rate = min(1.0, self._original_mutation_rates['threshold'] + bump)
        self.mutation_feature_rate = min(1.0, self._original_mutation_rates['feature'] + bump)
        self.mutation_operator_rate = min(1.0, self._original_mutation_rates['operator'] + bump)
        self.mutation_logical_rate = min(1.0, self._original_mutation_rates['logical'] + bump)

    def _mutate(self, formula: list):
        """Applies various mutations to a formula tree."""
        if random.random() < self.mutation_threshold_rate:
            nodes = [n for n in self._get_subtrees(formula) if not isinstance(n[0], list)]
            if nodes:
                node = random.choice(nodes)
                try:
                    feat, _, cur_thr = node
                    if random.random() < 0.40: # 40% chance to resample threshold from stats
                        node[2] = self._sample_threshold(feat)
                    else: # 60% chance to apply jitter
                        jitter = 1.0 + random.uniform(self.mutation_jitter_min, self.mutation_jitter_max)
                        node[2] = max(self.threshold_min, min(self.threshold_max, float(cur_thr) * jitter))
                except (ValueError, IndexError):
                    pass

        if random.random() < self.mutation_feature_rate:
            nodes = [n for n in self._get_subtrees(formula) if not isinstance(n[0], list)]
            if nodes:
                random.choice(nodes)[0] = random.choice(self.features)

        if random.random() < self.mutation_operator_rate:
            nodes = [n for n in self._get_subtrees(formula) if not isinstance(n[0], list)]
            if nodes:
                random.choice(nodes)[1] = random.choice(self.operators)

        if random.random() < self.mutation_logical_rate:
            nodes = [n for n in self._get_subtrees(formula) if isinstance(n[0], list)]
            if nodes:
                random.choice(nodes)[1] = random.choice(self.logical_operators)

    def _get_subtrees(self, formula):
        nodes, q = [], [formula]
        while q:
            node = q.pop(0); nodes.append(node)
            if isinstance(node[0], list): q.append(node[0])
            if len(node) > 2 and isinstance(node[2], list): q.append(node[2])
        return nodes
