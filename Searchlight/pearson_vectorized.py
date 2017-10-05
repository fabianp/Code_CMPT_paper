import numpy as np


def pearsonr_vectorized(a, b):
    """Vectorized version of Pearson correlation coefficient between each
    row vector in a and the corresponding one in b.
    """
    n = a.shape[1]
    a_sum = a.sum(1)
    b_sum = b.sum(1)
    p1 = n * (a * b).sum(1) - a_sum * b_sum
    p2 = n * (a * a).sum(1) - a_sum * a_sum
    p3 = n * (b * b).sum(1) - b_sum * b_sum
    return p1 / np.sqrt(p2 * p3)


def pearsonr_vectorized2(a, b):
    """Vectorized version of Pearson correlation coefficient between each
    row vector in a and the corresponding one in b.
    
    This is an alternative implementation which, in practice, is
    slightly slower.
    """
    n = a.shape[1]
    p1 = (a * b).sum(1) - a.sum(1) * b.sum(1) / float(n)
    p2 = (n - 1.0) * a.std(axis=1, ddof=1) * b.std(axis=1, ddof=1)
    return p1 / p2


if __name__ == '__main__':

    n_voxels = 300
    n_permutations = 1000
    print("Generating %d (fake) permuted mean beta values for two groups of %d voxels." % (n_permutations, n_voxels))
    a = np.random.uniform(size=(n_permutations, n_voxels))
    b = np.random.uniform(size=(n_permutations, n_voxels))

    print("Computing vectorized Pearson correlations")
    rho_vectorized = pearsonr_vectorized(a, b)
    rho_vectorized2 = pearsonr_vectorized2(a, b)

    print("Checking accuracy of results:")
    from scipy.stats import pearsonr
    rho_scipy = np.array([pearsonr(a[i], b[i])[0] for i in range(a.shape[0])])
    print("|| rho_vectorized - rho_scipy || = %s" %
          np.linalg.norm(rho_vectorized - rho_scipy))
    print("|| rho_vectorized2 - rho_scipy || = %s" %
          np.linalg.norm(rho_vectorized2 - rho_scipy))
    print("|| rho_vectorized - rho_vectorized2 || = %s" %
          np.linalg.norm(rho_vectorized - rho_vectorized2))

    from time import time
    repetitions = 100
    print("Estimating the time of one execution (avg of %d repetitions:" %
          repetitions)

    t0 = time()
    for i in range(repetitions):
        rho_vectorized = pearsonr_vectorized(a, b)

    print("rho_vectorized: %s sec." % ((time() - t0) / repetitions))

    t0 = time()
    for i in range(repetitions):
        rho_vectorized2 = pearsonr_vectorized2(a, b)

    print("rho_vectorized2: %s sec." % ((time() - t0) / repetitions))

    t0 = time()
    for i in range(repetitions):
        rho_scipy = np.array([pearsonr(a[i], b[i])[0]
                              for i in range(a.shape[0])])

    print("rho_scipy: %s sec." % ((time() - t0) / repetitions))
