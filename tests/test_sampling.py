import pytest
import numpy as np
from ferm.sampling import gaussian_distribution_max, sample_max_distribution, expectation


def test_gaussian_distribution_max_scalar():
    """Ensure output is a float scalar."""
    out = gaussian_distribution_max(sigma=1.0, mu=0.0, n=50)
    assert isinstance(out, float)


def test_gaussian_distribution_max_small_n():
    """Ensure brute-force path is triggered and output is still valid."""
    out = gaussian_distribution_max(sigma=1.0, mu=0.0, n=3)
    assert isinstance(out, float)


def test_monotonicity_with_mu():
    """Check that increasing mu increases expected max value."""
    s1 = np.mean([gaussian_distribution_max(1.0, 0.0, 50) for _ in range(100)])
    s2 = np.mean([gaussian_distribution_max(1.0, 1.0, 50) for _ in range(100)])
    assert s2 > s1


def test_sample_max_distribution_bounds():
    """Check that lower < upper and both are finite."""
    low, up = sample_max_distribution(mu=0.5, sigma=1.0, pop=100, alpha=3.0)
    assert np.isfinite(low) and np.isfinite(up)
    assert low < up


def test_expectation_bounds():
    """Expectation bounds should be ordered correctly."""
    low, high = expectation(mu=0.0, sigma=1.0, n=100)
    assert low < high

