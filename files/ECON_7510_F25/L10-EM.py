
"""ECON 7510 (F24)
Lecture 10
Author: Adam Harris
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


## Generate some data
np.random.seed(1234)
n_samples = 10000

μ1 = 10.0
μ2 = 30.0
μ3 = 35.0
σ1 = 5.0
σ2 = 4.0
σ3 = 4.0
π1 = 0.5
π2 = 0.25
π3 = 0.25
data = np.concatenate([np.random.normal(μ1, σ1, int(π1 * n_samples)),
                       np.random.normal(μ2, σ2, int(π2 * n_samples)),
                       np.random.normal(μ3, σ3, int(π3 * n_samples))])

plt.clf()
plt.hist(data, bins=100, density=True)
plt.show()

def estimate_mixture(data, K):
    # Initialize parameters
    μ = np.random.rand(K) * 50
    σ = np.random.rand(K) * 50
    weights = np.ones(K) / K
    #
    # Number of data points
    n = len(data)
    #
    # Responsibilities
    responsibilities = np.zeros((n, K))
    #
    # EM Algorithm
    max_iter = 10000
    tol = 1e-4
    log_likelihood = -np.inf
    log_likelihood_old = -np.inf
    #
    for iter in range(1, max_iter + 1):
        # E-step: Calculate responsibilities
        for k in range(K):
            responsibilities[:, k] = weights[k] * stats.norm.pdf(data, loc=μ[k], scale=σ[k])
        sum_responsibilities = np.sum(responsibilities, axis=1)
        responsibilities /= sum_responsibilities[:, np.newaxis]
        # Check for empty clusters
        for k in range(K):
            if np.sum(responsibilities[:, k]) == 0:
                print(f"Cluster {k} is empty")
        # M-step: Update parameters
        for k in range(1, K):
            N_k = np.sum(responsibilities[:, k])
            weights[k] = N_k / n
            weighted_mean = np.sum(responsibilities[:, k] * data) / N_k
            weighted_var = np.sum(responsibilities[:, k] * (data - weighted_mean) ** 2) / N_k
            μ[k] = weighted_mean
            σ[k] = np.sqrt(weighted_var)
        #
        # Check for convergence
        log_likelihood = np.sum(np.log(sum_responsibilities))
        if (log_likelihood - log_likelihood_old) < tol:
            print(f"Converged after {iter} iterations")
            break
        log_likelihood_old = log_likelihood
        print(log_likelihood)
    return μ, σ, weights, log_likelihood

## Estimation with known K=3
estimate_mixture(data,3)

# Multistart: Run EM J times and take the result than gives best log likelihood
J = 20
μ, σ, weights, ll = np.zeros((J, 3)), np.zeros((J, 3)), np.zeros((J, 3)), np.zeros(J)
for j in range(J):
    μ[j, :], σ[j, :], weights[j, :], ll[j] = estimate_mixture(data, 3)

plt.scatter(range(J), ll)
best_j = np.argmax(ll)

print("Estimated parameters:")
for k in range(3):
    print(f"Component {k + 1}: μ = {μ[best_j, k]}, σ = {σ[best_j, k]}, π = {weights[best_j, k]}")




## Estimation with unknown K
k_vec = np.repeat(np.arange(1, 6), 10)
ll_vec = np.zeros(len(k_vec))
for k in range(len(k_vec)):
    μ, σ, π, ll = estimate_mixture(data, k_vec[k])
    ll_vec[k] = ll

plt.scatter(k_vec, ll_vec)
plt.show()
bic = -2.0 * ll_vec + (2 * k_vec - 1) * np.log(n_samples)
plt.scatter(k_vec, bic)
plt.show()
k = k_vec[bic == np.min(bic)][0]
