import unittest
import numpy as np
from neurenorm import generate_random_filtered_data, EPSILON

class TestNormalization(unittest.TestCase):

    def test_normalization(self):
        data = generate_random_filtered_data(10000)
        self.assertAlmostEqual(np.mean(data[data > EPSILON]), 1.0, places=5)


if __name__ == '__main__':
    unittest.main()
