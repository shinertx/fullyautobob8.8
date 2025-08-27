# tests/allocation/test_allocation.py
import unittest
from v26meme.core.dsl import Alpha
from v26meme.allocation.optimizer import PortfolioOptimizer
from v26meme.allocation.lanes import LaneAllocationManager

class MockState:
    def __init__(self):
        self.data = {}
    def get(self, key):
        return self.data.get(key)
    def set(self, key, value):
        self.data[key] = value

class TestAllocation(unittest.TestCase):

    def setUp(self):
        self.config = {
            'portfolio': {
                'min_allocation_weight': 0.05,
                'max_alpha_concentration': 0.40,
                'lanes': {
                    'core': {'kelly_fraction': 0.5},
                    'moonshot': {'kelly_fraction': 0.2}
                }
            },
            'lanes': {
                'probation': {
                    'trades_min': 50,
                    'weight_cap': 0.03
                },
                'retag': {
                    'enabled': True,
                    'min_trades': 100,
                    'min_sharpe': 1.5,
                    'min_sortino': 2.0,
                    'max_mdd': 0.15
                }
            }
        }
        self.optimizer = PortfolioOptimizer(self.config)
        self.lane_manager = LaneAllocationManager(self.config, MockState())

    def _create_alpha(self, id, lane, trades, sharpe, sortino, mdd, returns):
        return Alpha(
            id=id,
            formula=[],
            universe=["BTC_USD_SPOT"],
            lane=lane,
            performance={
                'all': {
                    'n_trades': trades,
                    'sharpe': sharpe,
                    'sortino': sortino,
                    'mdd': mdd,
                    'returns': returns
                }
            }
        )

    def test_optimizer_floor_and_cap(self):
        """
        Tests that the optimizer correctly applies the allocation floor and concentration cap.
        """
        alphas = [
            self._create_alpha("A1", "core", 100, 2.0, 3.0, 0.1, [0.01, 0.02, -0.005]), # High variance
            self._create_alpha("A2", "core", 100, 1.5, 2.5, 0.1, [0.001, 0.002, -0.0005]), # Low variance
            self._create_alpha("A3", "core", 100, 1.8, 2.8, 0.1, [0.001, 0.002, -0.0005]), # Low variance
            self._create_alpha("A4", "core", 100, 1.2, 2.0, 0.1, [0.001, 0.002, -0.0005]), # Low variance
        ]
        # A1 will have low weight due to high variance and should be floored to 0
        # A2, A3, A4 will get the remaining weight, none should exceed the cap.
        weights = self.optimizer.get_weights(alphas, 'all')
        
        self.assertNotIn('A1', weights) # Should be floored
        self.assertTrue(all(w > 0 for w in weights.values()))
        self.assertTrue(all(w <= self.config['portfolio']['max_alpha_concentration'] for w in weights.values()))
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=5)

    def test_lane_manager_probation(self):
        """
        Tests that the lane manager correctly caps the weight of moonshot alphas on probation.
        """
        alphas = [
            self._create_alpha("M1", "moonshot", 20, 3.0, 4.0, 0.05, [0.02, 0.03]), # Low trades, should be capped
            self._create_alpha("M2", "moonshot", 100, 2.0, 3.0, 0.1, [0.01, 0.01]),
            self._create_alpha("C1", "core", 200, 1.5, 2.5, 0.1, [0.005, 0.005]),
        ]
        initial_weights = {"M1": 0.1, "M2": 0.1, "C1": 0.8}
        
        capped_weights = self.lane_manager._apply_probation(initial_weights, alphas)
        
        self.assertLessEqual(capped_weights['M1'], self.config['lanes']['probation']['weight_cap'])
        self.assertEqual(capped_weights['M2'], initial_weights['M2']) # Unaffected
        self.assertEqual(capped_weights['C1'], initial_weights['C1']) # Unaffected

    def test_lane_manager_retagging(self):
        """
        Tests that the lane manager correctly retags a successful moonshot alpha to core.
        """
        alphas = [
            self._create_alpha("M1_SUCCESS", "moonshot", 150, 2.0, 2.5, 0.1, [0.01]*150), # Should be retagged
            self._create_alpha("M2_FAIL", "moonshot", 50, 1.0, 1.0, 0.3, [0.01]*50), # Should not be retagged
        ]
        
        retagged_count = self.lane_manager._maybe_retag(alphas)
        
        self.assertEqual(retagged_count, 1)
        self.assertEqual(alphas[0].lane, 'core')
        self.assertEqual(alphas[1].lane, 'moonshot')


if __name__ == '__main__':
    unittest.main()
