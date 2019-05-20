import unittest
import numpy as np
from neurenorm import generate_random_filtered_data, renormalization_step, compute_correlation_coefficients, EPSILON


class TestNormalization(unittest.TestCase):

    def test_normalization(self):
        data = generate_random_filtered_data(10000, num_neurons=1)
        self.assertAlmostEqual(np.mean(data[data > EPSILON]), 1.0, places=5)


class TestRenormalizationStep(unittest.TestCase):
    def test_length_of_neuronal_data_stays_equal(self):
        data = generate_random_filtered_data(10000, num_neurons=100)
        corr = compute_correlation_coefficients(data)
        rdata = renormalization_step(data, corr)
        self.assertEqual(rdata.shape, (50, 10000))


if __name__ == '__main__':
    unittest.main()
