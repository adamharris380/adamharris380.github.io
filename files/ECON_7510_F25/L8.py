"""
ECON 7510 (F25)
Lecture 8
Author: Adam Harris
"""

# Load packages:
import numpy as np
import copy

#===========  PART 1: Numerical optimization  ===========#
#===========  GENERATE DATA  ===========#
np.random.seed(1234) # For reproducibility
N = 1_000_000 # Number of individuals
K = 10 # Number of characteristics

# Generate individuals' characteristics
X = np.random.normal(size=(N, K))
X = np.hstack((np.ones((N, 1)), X)) # Add a column of ones

# Generate parameters:
beta0 = np.random.normal(size=(K+1,))
# Generate idiosyncratic shocks
epsilon = np.random.logistic(size=N)

# Generate actions:
a = X @ beta0 - epsilon > 0


#===========  ESTIMATION  ===========#
def prob(X,beta):
    return 1 / (1 + np.exp(-X @ beta))

def loglikelihood(X,beta,a):
    p = prob(X,beta)
    l = a * np.log(p) + (1.0 - a) * np.log(1.0 - p)
    return np.sum(l)

def grad_loglikelihood(X,beta,a):
    K = len(beta) - 1
    p = prob(X,beta)
    temp = np.tile(a - p, (K+1,1)).T * X
    return np.sum(temp, axis=0)

# Optimization

# With analytical gradient:
def update_beta(X,beta,a, eta=1e-5):
    g = grad_loglikelihood(X,beta,a)
    return beta + eta * g

def gradient_descent(X,beta,a, TOL=1e-12, eta=1e-5):
    obj, last_obj = 10.0, 1.0
    iter = 1
    while abs(obj - last_obj)/len(a) > TOL:
        beta = update_beta(X,beta,a, eta=eta)
        last_obj = copy.deepcopy(obj)
        obj = loglikelihood(X,beta,a)
        print("Iteration: {}".format(iter))
        print("Objective value: {}".format(obj))
        iter += 1
    return beta

beta_MLE = gradient_descent(X,np.zeros(K+1),a)
np.linalg.norm(beta_MLE - beta0)



# Can we speed up the process by increasing the learning rate?
beta_MLE_bigger_step = gradient_descent(X,np.zeros(K+1),a, eta=1e-4)





# No.  Initial guess is poor, so gradient is quite large:
grad_loglikelihood(X,np.zeros(K+1),a)


# Question: What if we didn't know the analytical gradient?




def grad_loglikelihood_fd(X,beta,a, delta=1e-8):
    l0 = loglikelihood(X,beta,a)
    l1 = np.zeros(K+1)
    for k in range(1,K+2):
        l1[k-1] = loglikelihood(X,beta + np.concatenate((np.zeros(k-1),[delta],np.zeros(K+1-k))),a)
    return (l1 - l0) / delta




# A way to check our analytical gradient
np.linalg.norm(
    grad_loglikelihood_fd(X,beta_MLE,a) - grad_loglikelihood(X,beta_MLE,a)
)


# Is smaller δ always better?
np.linalg.norm(
    grad_loglikelihood_fd(X,beta_MLE,a, delta=1e-12) - grad_loglikelihood(X,beta_MLE,a)
)


# Gradient descent with option for finite differences:
def update_beta(X,beta,a, eta=1e-5, finite_differences=False):
    if not finite_differences:
        g = grad_loglikelihood(X,beta,a)
    else:
        g = grad_loglikelihood_fd(X,beta,a)
    return beta + eta * g


def gradient_descent(X,beta,a, TOL=1e-12, eta=1e-5, finite_differences=False):
    obj, last_obj = 10.0, 1.0
    iter = 1
    while abs(obj - last_obj)/len(a) > TOL:
        beta = update_beta(X,beta,a, eta=eta, finite_differences=finite_differences)
        last_obj = copy.deepcopy(obj)
        obj = loglikelihood(X,beta,a)
        print("Iteration: {}".format(iter))
        print("Objective value: {}".format(obj))
        iter += 1
    return beta

beta_MLE_fd = gradient_descent(X,np.zeros(K+1),a, finite_differences=True)
np.linalg.norm(
    beta_MLE_fd - beta_MLE
)

# Time comparisons:
# Python code to time each of these:
import time
start_time = time.time()
beta_MLE = gradient_descent(X,np.zeros(K+1),a)
end_time = time.time()
print("Time taken with analytical gradient: {} seconds".format(end_time - start_time))
start_time = time.time()
beta_MLE_fd = gradient_descent(X,np.zeros(K+1),a, finite_differences=True)
end_time = time.time()
print("Time taken with finite differences: {} seconds".format(end_time - start_time))



# Analytical gradient is much faster!
# But FD is good if you don't have an analytical gradient or you're lazy.

# Question: Can't Python just do the calculus for us?
# Answer: Yes, with automatic differentiation (AD).

import torch
X_torch = torch.tensor(X, dtype=torch.float32)
a_torch = torch.tensor(a, dtype=torch.float32)

def prob_torch(X, beta):
    # Add numerical stability by clamping the logits
    logits = X @ beta
    logits = torch.clamp(logits, -500, 500)  # Prevent overflow/underflow
    return torch.sigmoid(logits)

def loglikelihood_torch(X, beta, a):
    p = prob_torch(X, beta)
    # Add small epsilon for numerical stability
    eps = 1e-8
    l = a * torch.log(p + eps) + (1.0 - a) * torch.log(1.0 - p + eps)
    return torch.sum(l)

def gradient_descent_torch(X, beta, a, TOL=1e-12, eta=1e-5, verbose=False, maxiter=1000):
    beta = beta.clone().detach().requires_grad_(True)
    last_obj = 1.0
    iter = 1
    while True:
        # Forward pass
        obj = loglikelihood_torch(X, beta, a)
        if verbose:
            print("Iteration: {}".format(iter))
            print("Objective value: {}".format(obj.item()))
            print("Beta values: {}".format(beta.detach().numpy()[:5]))  # Show first 5 beta values
        # Check convergence
        if iter > 1:
            convergence_val = abs(obj.item() - last_obj)/len(a)
            if verbose:
                print("Convergence check: {}".format(convergence_val))
            if convergence_val <= TOL:
                break
        # Backward pass
        obj.backward()
        with torch.no_grad():
            beta += eta * beta.grad
        beta.grad.zero_()  # Clear gradients after update
        last_obj = obj.item()
        iter += 1
        # Safety check to prevent infinite loops
        if iter > maxiter:
            print("Maximum iterations reached")
            break
    return beta.detach().numpy()

beta_MLE_torch = gradient_descent_torch(X_torch, torch.zeros(K+1), a_torch)
np.linalg.norm(
    beta_MLE_torch - beta0
)

# ASIDE:
# A fancier version (using classes, which are a useful convenience):
# Note: This is unrelated to using torch; just a way to organize code.
from dataclasses import dataclass
@dataclass
class GDConfig:
    TOL: float = 1e-12
    eta: float = 1e-5
    max_iter: int = 1000
    verbose: bool = False

@dataclass
class GDResult:
    beta: np.ndarray
    n_iter: int
    final_obj: float

# Example:
gdc_example = GDConfig(verbose=True, eta=1e-4)
print(gdc_example.eta)
print(gdc_example.max_iter)

def gradient_descent_torch_fancy(X, beta, a, config: GDConfig):
    beta = beta.clone().detach().requires_grad_(True)
    last_obj = 1.0
    iter = 1
    while True:
        # Forward pass
        obj = loglikelihood_torch(X, beta, a)
        if config.verbose:
            print("Iteration: {}".format(iter))
            print("Objective value: {}".format(obj.item()))
            print("Beta values: {}".format(beta.detach().numpy()[:5]))  # Show first 5 beta values
        # Check convergence
        if iter > 1:
            convergence_val = abs(obj.item() - last_obj)/len(a)
            if config.verbose:
                print("Convergence check: {}".format(convergence_val))
            if convergence_val <= config.TOL:
                break
        # Backward pass
        obj.backward()
        with torch.no_grad():
            beta += config.eta * beta.grad
        beta.grad.zero_()  # Clear gradients after update
        last_obj = obj.item()
        iter += 1
        # Safety check to prevent infinite loops
        if iter > config.max_iter:
            print("Maximum iterations reached")
            break
    return GDResult(beta=beta.detach().numpy(), n_iter=iter, final_obj=obj.item())

beta_MLE_torch_fancy = gradient_descent_torch_fancy(X_torch, torch.zeros(K+1), a_torch, GDConfig(verbose=True))
np.linalg.norm(
    beta_MLE_torch_fancy.beta - beta0
)
beta_MLE_torch_fancy.n_iter, beta_MLE_torch_fancy.final_obj
#===========  PART 2: Numerical integration  ===========#
# X now unidimensional

beta = np.array([1.0])
def elasticity(x,beta):
    p = prob(x,beta)
    return (1.0 - p) * x[:,0] * beta[0]

# Random sampling:
x = np.random.normal(size=(100_000_000,1))
elast_sampling100000000 = np.mean(elasticity(x,beta))

x = np.random.normal(size=(10_000_000,1))
elast_sampling10000000 = np.mean(elasticity(x,beta))

x = np.random.normal(size=(1_000_000,1))
elast_sampling1000000 = np.mean(elasticity(x,beta))

x = np.random.normal(size=(100_000,1))
elast_sampling100000 = np.mean(elasticity(x,beta))

x = np.random.normal(size=(10_000,1))
elast_sampling10000 = np.mean(elasticity(x,beta))

x = np.random.normal(size=(1_000,1))
elast_sampling1000 = np.mean(elasticity(x,beta))

x = np.random.normal(size=(100,1))
elast_sampling100 = np.mean(elasticity(x,beta))

print("Elasticity estimates:")
print("With 100000000 samples: {}".format(elast_sampling100000000))
print("With 10000000 samples: {}".format(elast_sampling10000000))
print("With 1000000 samples: {}".format(elast_sampling1000000))
print("With 100000 samples: {}".format(elast_sampling100000))
print("With 10000 samples: {}".format(elast_sampling10000))
print("With 1000 samples: {}".format(elast_sampling1000))
print("With 100 samples: {}".format(elast_sampling100))



# Gauss-Hermite quadrature
from numpy.polynomial.hermite import hermgauss

x,w = hermgauss(100)
x = x.reshape(-1,1)
h = elasticity(np.sqrt(2)*x, beta) / np.sqrt(np.pi)
elast_gh100 = np.sum(w * h)

x,w = hermgauss(10)
x = x.reshape(-1,1)
h = elasticity(np.sqrt(2)*x, beta) / np.sqrt(np.pi)
elast_gh10 = np.sum(w * h)

x,w = hermgauss(5)
x = x.reshape(-1,1)
h = elasticity(np.sqrt(2)*x, beta) / np.sqrt(np.pi)
elast_gh5 = np.sum(w * h)

elast_gh100, elast_gh10, elast_gh5

elast_sampling100000000


# Expectation over a non-Normal random variable
# Convert Exponential rv to Normal rv
from scipy import stats
x = stats.expon.rvs(scale=0.5, size=(100_000_000,1))

# First, compute expecation by sampling
elast_avg_sampling = np.mean(elasticity(x,beta))

# Convert Normal rv to Exponential rv
x_transformed = stats.norm.ppf(stats.expon.cdf(x))

def standard_normal_to_expon(z, mean=0.5):
    u = stats.norm.cdf(z)
    return stats.expon.ppf(u, scale=mean)


x,w = hermgauss(20)
x = x.reshape(-1,1)
h = elasticity(standard_normal_to_expon(np.sqrt(2)*x,0.5), beta) / np.sqrt(np.pi)
elast_gh20 = np.sum(w * h)

x, w = hermgauss(10)
x = x.reshape(-1,1)
h = elasticity(standard_normal_to_expon(np.sqrt(2)*x,0.5), beta) / np.sqrt(np.pi)
elast_gh10 = np.sum(w * h)

x, w = hermgauss(5)
x = x.reshape(-1,1)
h = elasticity(standard_normal_to_expon(np.sqrt(2)*x,0.5), beta) / np.sqrt(np.pi)
elast_gh5 = np.sum(w * h)

elast_avg_sampling, elast_gh20, elast_gh10, elast_gh5

@dataclass
class Integrator:
    dist_ppf: callable = None
    dist_param: float = None
    gh_points: int = 10
    function_to_integrate: callable = None
    function_param: float = None
    def integrate(self):
        x, w = hermgauss(self.gh_points)
        x = x.reshape(-1,1)
        u = stats.norm.cdf(np.sqrt(2)*x)
        if self.dist_param is not None:
            d = self.dist_ppf(u, self.dist_param)
        else:
            d = self.dist_ppf(u)
        if self.function_param is not None:
            temp_func = lambda z: self.function_to_integrate(z, self.function_param)
        else:
            temp_func = self.function_to_integrate
        h = temp_func(d) / np.sqrt(np.pi)
        return np.sum(w * h)

exp_integrator = Integrator(
    dist_ppf=stats.expon.ppf,
    function_to_integrate=elasticity,
    function_param=beta,
    dist_param=0.5
)
exp_integrator.integrate()

exp_integrator_beta2 = Integrator(
    dist_ppf=stats.expon.ppf,
    function_to_integrate=elasticity,
    function_param=beta,
    dist_param=0.75
)
exp_integrator_beta2.integrate()