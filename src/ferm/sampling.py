import numpy as np
from scipy import stats
from arspy.ars import adaptive_rejection_sampling

def sample_max_distribution(mu: float, sigma: float, pop: int, alpha: float = 3.0) -> tuple[float, float]:
    """
    Estimate lower and upper bounds for the maximum of Gaussian distributions.

    Parameters
    ----------
    mu : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation of the Gaussian distribution.
    pop : int
        Number of samples (population size).
    alpha : float, optional
        Multiplier for standard deviation to define bounds, by default 3.0

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds of the maximum distribution.
    """
    mm = (np.sqrt(np.log(pop**2 / (2 * np.pi * np.log(pop**2 / (2 * np.pi)))))
          * (1 + 0.5772156649 / np.log(pop))) * sigma + mu
    ss = np.sqrt(sigma**2 * np.pi**2 / (12 * np.log(pop)))

    low = mm - alpha * ss
    up = mm + alpha * ss
    return low, up

def gaussian_distribution_max(sigma: float, mu: float, n: int) -> float:
    """
    Sample the maximum of `n` Gaussian variables using ARS (if n > 5).

    Parameters
    ----------
    sigma : float
        Standard deviation of the Gaussian distributions.
    mu : float
        Mean of the Gaussian distributions.
    n : int
        Number of samples (population size).

    Returns
    -------
    float
        Sample from the distribution of the maximum value.
    """
    if n <= 5:
        return np.max(stats.norm.rvs(loc=mu, scale=sigma, size=n))
    else:
        a, b = sample_max_distribution(mu, sigma, n)
        domain = (float("-inf"), float("inf"))
        log_pdf_max_gaussian = lambda x: np.log(
            (n / sigma) * stats.norm.pdf(x, loc=mu, scale=sigma) *
            stats.norm.cdf(x, loc=mu, scale=sigma)**(n - 1)
        )
        sample = adaptive_rejection_sampling(log_pdf_max_gaussian, a=a, b=b, domain=domain, n_samples=1)[0]
        return sample

def expectation(mu: float, sigma: float, n: int) -> tuple[float, float]:
    """
    Compute an estimated confidence interval for the maximum of `n` Gaussian samples.

    Parameters
    ----------
    mu : float
        Mean of the Gaussian distribution.
    sigma : float
        Standard deviation.
    n : int
        Population size (number of samples).

    Returns
    -------
    tuple[float, float]
        Lower and upper bounds for the expected maximum.
    """
    lower_bound = mu + sigma * np.sqrt(np.log(n)) * (1 / (np.pi * np.log(2)))
    upper_bound = mu + sigma * np.sqrt(2 * np.log(n))
    return lower_bound, upper_bound

def test_gaussian_max_comparison(sigma: float, n1: int, n2: int, n_trials: int = 1000) -> tuple[int, int]:
    """
    Empirically compare how often one population produces a higher maximum than another.

    Parameters
    ----------
    sigma : float
        Standard deviation for both populations.
    n1 : int
        Size of the first population.
    n2 : int
        Size of the second population.
    n_trials : int, optional
        Number of trials to run, by default 1000

    Returns
    -------
    tuple[int, int]
        Number of times first population wins, number of times second wins.
    """
    s1, s2 = 0, 0
    for _ in range(n_trials):
        z1 = gaussian_distribution_max(sigma, 0, n1)
        z2 = gaussian_distribution_max(sigma, 0.5, n2)
        if z1 > z2:
            s1 += 1
        else:
            s2 += 1
    return s1, s2
