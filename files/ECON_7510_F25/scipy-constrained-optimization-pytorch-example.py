import numpy as np
import torch
from scipy.optimize import minimize, NonlinearConstraint
from scipy.sparse.linalg import LinearOperator

# ---------- Global numeric hygiene ----------
torch.set_default_dtype(torch.float64)
DEVICE = torch.device("cpu")
EPS = 1e-6  # small positive epsilon for stable logs/positivity

# ---------- Example "data" bundle (captured by closures) ----------
N, p = 500, 3
rng = torch.Generator(device=DEVICE).manual_seed(0)
D = {
    "X": torch.randn(N, p, generator=rng, device=DEVICE, dtype=torch.float64),
    "y": torch.randn(N,    generator=rng, device=DEVICE, dtype=torch.float64),
    "alpha": 0.1,
}
# Data do NOT require grad:
for v in D.values():
    if isinstance(v, torch.Tensor):
        v.requires_grad_(False)

# ---------- Decision vector layout ----------
# Example: x = [beta(3), s(1)] where sigma = softplus(s)+EPS > 0
n = p + 1   # total variables
m = 2       # number of equality constraints

def split_vars(x_t: torch.Tensor):
    """Return differentiable views mapped from a SINGLE leaf x_t."""
    beta = x_t[:p]                   # view into x_t
    s    = x_t[p]                    # scalar view
    sigma = torch.nn.functional.softplus(s) + EPS
    return beta, sigma

# ---------- Your torch-native objective and constraints ----------
def torch_f(x_t: torch.Tensor, D) -> torch.Tensor:
    """Scalar objective (pure Torch)."""
    beta, sigma = split_vars(x_t)
    X, y, alpha = D["X"], D["y"], D["alpha"]
    pred  = X @ beta
    resid = y - pred
    # smooth, well-scaled loss with positivity on sigma via softplus(s)
    loss = 0.5 * torch.mean((resid / sigma)**2) + alpha * (beta @ beta) + 0.5 * torch.log(sigma**2 + EPS)
    return loss  # scalar tensor (requires grad)

def torch_g(x_t: torch.Tensor, D) -> torch.Tensor:
    """Vector of m nonlinear equalities g(x)=0 (pure Torch)."""
    beta, sigma = split_vars(x_t)
    X, y = D["X"], D["y"]
    pred = X @ beta
    err  = y - pred
    g1 = torch.mean(err)                              # = 0  (mean residual)
    g2 = torch.mean(err * (X[:, 0]**2)) - sigma       # = 0  (moment matches sigma)
    return torch.stack([g1, g2])  # shape (m,)

# ---------- Leaf-creation + diagnostics ----------
def to_leaf_torch(x_np: np.ndarray) -> torch.Tensor:
    """Create a SINGLE leaf (requires_grad=True) on the correct device/dtype."""
    x = np.asarray(x_np, dtype=np.float64)
    x_t = torch.from_numpy(x).to(DEVICE).clone().detach()
    x_t.requires_grad_(True)
    return x_t

def _assert_connected(x_t: torch.Tensor, out: torch.Tensor, name: str):
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"{name} must return a torch.Tensor, got {type(out)}")
    if out.grad_fn is None and not out.requires_grad:
        raise RuntimeError(f"{name} returned a tensor with no grad_fn (graph detached). "
                           f"Avoid .item()/.detach()/.numpy() or np./math ops inside {name}.")
    # Non-destructive probe: retain the graph so subsequent grads still work
    try:
        _ = torch.autograd.grad(out.sum(), x_t, allow_unused=True, retain_graph=True)[0]
    except RuntimeError as e:
        raise RuntimeError(f"autograd.grad failed inside {name}: {e}")


# ---------- SciPy callback suite (values, Jacobians, Hessian-vector products) ----------
def fun(x_np):
    x_t = to_leaf_torch(x_np)
    f = torch_f(x_t, D)
    _assert_connected(x_t, f, "torch_f")
    return f.item()

def grad(x_np):
    x_t = to_leaf_torch(x_np)
    f = torch_f(x_t, D)
    _assert_connected(x_t, f, "torch_f (grad)")
    (gx,) = torch.autograd.grad(f, x_t, create_graph=False)
    return gx.detach().cpu().numpy()

def hess_obj_linearop(x_np):
    """Return LinearOperator p -> ∇²f(x) @ p using one reverse/one forward-over-reverse pass."""
    x_t = to_leaf_torch(x_np)
    n_  = x_t.numel()
    def matvec(p_np):
        p_t = torch.from_numpy(np.asarray(p_np, dtype=np.float64)).to(x_t)
        (gxf,) = torch.autograd.grad(torch_f(x_t, D), x_t, create_graph=True)
        dot    = (gxf * p_t).sum()
        (hvp,) = torch.autograd.grad(dot, x_t)
        return hvp.detach().cpu().numpy()
    return LinearOperator((n_, n_), matvec=matvec)

def g_fun(x_np):
    x_t = to_leaf_torch(x_np)
    g = torch_g(x_t, D)
    _assert_connected(x_t, g, "torch_g")
    return g.detach().cpu().numpy()

def g_jac(x_np):
    """Jacobian of g: rows are ∇g_k(x)^T, shape (m, n)."""
    x_t = to_leaf_torch(x_np)
    g = torch_g(x_t, D)
    _assert_connected(x_t, g, "torch_g (jac)")
    rows = []
    for k in range(g.shape[0]):
        (row_k,) = torch.autograd.grad(g[k], x_t, retain_graph=True)
        rows.append(row_k.detach().cpu().numpy())
    return np.stack(rows, axis=0)

def g_hess(x_np, v):
    """
    Return LinearOperator for sum_k v_k ∇² g_k(x). `trust-constr` will combine
    this with the objective Hessian to form the Lagrangian Hessian.
    """
    x_t = to_leaf_torch(x_np)
    v_t = torch.from_numpy(np.asarray(v, dtype=np.float64)).to(x_t)
    n_  = x_t.numel()
    def matvec(p_np):
        p_t = torch.from_numpy(np.asarray(p_np, dtype=np.float64)).to(x_t)
        L   = torch.dot(v_t, torch_g(x_t, D))           # scalar = v^T g(x)
        (gL,) = torch.autograd.grad(L, x_t, create_graph=True)
        dot   = (gL * p_t).sum()
        (hvp,) = torch.autograd.grad(dot, x_t)          # (∑ v_k ∇²g_k) @ p
        return hvp.detach().cpu().numpy()
    return LinearOperator((n_, n_), matvec=matvec)

# ---------- Pre-solve probes (fail fast with clear messages) ----------
x0 = np.zeros(n, dtype=np.float64)
_ = fun(x0)
_ = grad(x0)
_ = g_fun(x0)
_ = g_jac(x0)
# You can also probe Hessian operators:
_ = hess_obj_linearop(x0).matvec(np.ones(n))
_ = g_hess(x0, np.ones(m)).matvec(np.ones(n))

# ---------- Assemble and solve ----------
eq_con = NonlinearConstraint(g_fun, lb=np.zeros(m), ub=np.zeros(m), jac=g_jac, hess=g_hess)

res = minimize(
    fun, x0, method="trust-constr",
    jac=grad,
    hess=hess_obj_linearop,            # objective Hessian (as LinearOperator)
    constraints=[eq_con],
    options={"verbose": 3, "gtol": 1e-8, "xtol": 1e-10, "maxiter": 500}
)

print("\n--- Result ---")
print("x* =", res.x)
print("status:", res.message)
print("g(x*) =", g_fun(res.x))
print("f(x*) =", fun(res.x))
