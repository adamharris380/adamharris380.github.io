import numpy as np
import copy
from numpy.polynomial.hermite import hermgauss

## Model setup
N = 5 # Number of states
δ = 0.9 # Discount factor

# Static payoffs:
np.random.seed(1234)
u0 = np.random.normal(size=N)
u1 = np.random.normal(size=N)

# Transition matrices
M0 = np.random.rand(N,N)
M1 = np.random.rand(N,N)
# How to make these proper transition matrices?




rowsums0 = np.sum(M0, axis=1)
rowsums1 = np.sum(M1, axis=1)
for i in range(N):
    M0[i,:] /= rowsums0[i]
    M1[i,:] /= rowsums1[i]

np.sum(M0, axis=1)
np.sum(M1, axis=1)



## Value function iteration:
V = np.zeros(N)  # initialization
V_new = np.ones(N)  # initialization

TOL = 1e-6
diff = 1.0
while diff > TOL:
    v0 = u0 + δ * M0 @ V
    v1 = u1 + δ * M1 @ V
    #
    V_new = np.array([max(v0[i],v1[i]) for i in range(N)])
    #
    diff = np.linalg.norm(V_new - V)
    print(diff)
    #
    V = V_new


# Now compute the policy function (vector):
v0 = u0 + δ * M0 @ V
v1 = u1 + δ * M1 @ V
a_star = v1 > v0
# What problems might arise if we tried to take this model to the data?




## Introducting stochastic payoff term ϵ
theta = 2.0
def special_logistic_function(x,y, theta):
    M = min(theta*x, theta*y)  # This is the log-sum-exp trick to deal with value overflow when x,y are large and negative.
    return (1/theta)*(np.log(np.exp(theta*x - M) + np.exp(theta*y - M)) + M)

# Copy over the code from the deterministic-payoff case and edit to make it work for stochastic-payoff case







## Expected value function iteration:
EV = np.zeros(N)  # initialization
EV_new = np.ones(N)  # initialization

TOL = 1e-6
diff = 1.0
while diff > TOL:
    v0 = u0 + δ * M0 @ EV
    v1 = u1 + δ * M1 @ EV
    #
    EV_new = np.array([special_logistic_function(v0[i],v1[i], theta) for i in range(N)])
    #
    diff = np.linalg.norm(EV_new - EV)
    print(diff)
    #
    EV = EV_new

# Now compute the conditional choice probability function (vector):
v0 = u0 + δ * M0 @ EV
v1 = u1 + δ * M1 @ EV
p = 1 / (1 + np.exp(-theta * (v1 - v0)))






## Handling continuous functions:
def example_fcn(x):
    return np.sin(x / 25) + np.log(x)

x_grid = np.arange(10.0, 110.0, 10.0)
y_grid = example_fcn(x_grid)

import matplotlib.pyplot as plt
plt.scatter(x_grid, y_grid)
plt.show()


# Interpolation:
from scipy.interpolate import interp1d
interp_linear = interp1d(x_grid, y_grid, kind='linear')
interp_linear(10.0)
interp_linear(11.0)
plt.scatter(x_grid, y_grid)
x_fine = np.arange(10.0, 100.0, 0.1)
plt.plot(x_fine, interp_linear(x_fine))
plt.show()

# Allow extrapolation:
interp_linear_extrap = interp1d(x_grid, y_grid, fill_value="extrapolate")
plt.plot(np.arange(-10.0, 100.0, 0.1), interp_linear_extrap(np.arange(-10.0, 100.0, 0.1)))
plt.scatter(x_grid, y_grid)
plt.show()
# What's the problem with this?

# Alternative extrapolation behavior:
interp_linear_extrap2 = interp1d(x_grid, y_grid, fill_value=(y_grid[0], y_grid[-1]), bounds_error=False)
plt.plot(np.arange(-10.0, 100.0, 0.1), interp_linear_extrap2(np.arange(-10.0, 100.0, 0.1)))
plt.scatter(x_grid, y_grid)
plt.show()


# Question: Under what conditions might linear interpolation give a poor approximation?



# Higher-order interpolation:
from scipy.interpolate import CubicSpline
interp_cubic = CubicSpline(x_grid, y_grid, bc_type='natural')
plt.plot(x_fine, interp_cubic(x_fine), label='Cubic Spline')
plt.plot(x_fine, interp_linear(x_fine), label='Linear Interpolation')
plt.legend()
plt.scatter(x_grid, y_grid)
plt.show()

# Question: What's the takeaway here?


# Multi-dimensional interpolation:
from scipy.interpolate import RegularGridInterpolator
def f(x,y):
    return np.log(2.0*x+y)

# Multi-dimensional interpolation:
x_grid = np.arange(1, 5, 0.2)
y_grid = np.arange(2, 5, 0.1)
z_grid = np.array([[f(x,y).item() for y in y_grid] for x in x_grid])
plt.scatter(np.repeat(x_grid, len(y_grid)), np.tile(y_grid, len(x_grid)), c=z_grid.flatten())
plt.colorbar()
plt.show()

interp_linear2 = RegularGridInterpolator((x_grid, y_grid), z_grid)

interp_linear2(np.array([1.1,2.1]))
np.log(2.0 * 1.1 + 2.1)

interp_linear2(np.array([20/9,np.pi]))
np.log(2*(20/9)+np.pi)


## Dynamic programming with continuous state
theta = 2e-4
delta = 0.9

N = 101 # number of grid points for interpolation
EV = np.zeros(N) # initialization
EV_new = np.ones(N) # initialization

x_grid = np.linspace(0.0, 30000.0, N)

u0 = -x_grid
u1 = -1e5*np.ones(N)

TOL = 1e-6
diff = 1.0
x,w = hermgauss(10)



# Trick because
	# (i)  Gauss-Hermite x is symmetric around zero and 
	# (ii) Χ^2 distribution is squared Normal.
x = x[5:]
eta = x**2
w = 2.0 * w[5:]

v0, v1 = np.zeros(N), np.zeros(N)
while diff > TOL:
    # Compute expectation of EV for each x ∈ x_grid:
    EV_interp = interp1d(x_grid, EV, fill_value="extrapolate")
    # Question: Why do we need extrapolation here?
    for i in range(N):
        v0[i] = u0[i] + delta * np.sum(w * EV_interp(x_grid[i] + 2*eta)) / np.sqrt(np.pi)
    v1 = u1 + delta*EV[1]
    #
    EV_new = np.array([special_logistic_function(v0[i],v1[i], theta) for i in range(N)])
    #
    diff = np.sum((EV_new - EV) ** 2)
    print(diff)
    # Update the EV for the next iteration
    EV = EV_new


# Now compute the conditional choice probability function (vector):
EV_interp = interp1d(x_grid, EV, fill_value="extrapolate")
for i in range(N):
    v0[i] = u0[i] + delta * np.sum(w * EV_interp(x_grid[i] + 2*eta)) / np.sqrt(np.pi)

v1 = u1 + delta*EV[1]

p = 1 / (1 + np.exp(-theta * (v1 - v0)))

plt.plot(x_grid, p)
plt.xlabel("x")
plt.ylabel("Pr(a=1 | x)")
plt.show()

# Simulation:
v1_v0_interp = interp1d(x_grid, v1 - v0, fill_value="extrapolate")
a = 0 # initialization
t = 1
x = 0.0
while a == 0:
    p = 1 / (1 + np.exp(-theta * v1_v0_interp(x)))
    if np.random.rand() < p:  # a_t = 1
        break
    else:  # a_t = 0
        η = np.random.chisquare(df=1)
        x += η
    t += 1
    print(f"t: {t}")
    print(f"x: {x}")

# Why does a=1 tend to get chosen around x ∈ [5000,7500]?
# After all, p(7500.0) is very small!
1.0/(1.0 + np.exp(-theta * v1_v0_interp(7500.0)))