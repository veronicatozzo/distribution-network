import unittest
import numpy as np

from utils import QuantileScaler
from statsmodels.distributions.empirical_distribution import ECDF

class TestQuantileScaler(unittest.TestCase):
    def setUp(self):
        self.a = np.array([
            [0.33681467, 0.18897692],
            [0.98723588, 0.45421366],
            [0.36506861, 0.98719281],
            [0.68344273, 0.66833268],
            [0.5702027 , 0.96007762]
        ])
        self.b = np.array([
            [0.09330265, 0.31026214],
            [0.31301988, 0.93153685],
            [0.25749002, 0.0702172 ]
        ])

        self.a_results = np.array([
            [0.2, 0.2],
            [1. , 0.4],
            [0.4, 1. ],
            [0.8, 0.6],
            [0.6, 0.8]
        ]) * 2 - 1
        self.b_results = np.array([
            [0. , 0.2],
            [0. , 0.6],
            [0. , 0. ]]) * 2 - 1

    def test_fit_transform(self):
        quantile_scaler = QuantileScaler()
        new_a = quantile_scaler.fit_transform(self.a)
        new_b = quantile_scaler.transform(self.b)
        np.testing.assert_array_almost_equal(new_a, self.a_results)
        np.testing.assert_array_almost_equal(new_b, self.b_results)