import pytest
import numpy as np
from scipy.stats import circmean
from utils import circular_mean

# content of test_class.py
class TestUtils:
    def test_circular_mean(self):
        angles = np.radians([-179, 179])
        cm = circular_mean(angles)
        assert np.degrees(cm) == 180

        weights = np.array([0.8, 0.2])
        cm = circular_mean(angles, weights)
        assert np.degrees(cm) == pytest.approx(-179.40)
