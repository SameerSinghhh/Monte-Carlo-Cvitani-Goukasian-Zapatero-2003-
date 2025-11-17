"""Budget constraint solver for Lagrange multiplier y.

Solves for y such that budget constraints (Eq. 16-17) are satisfied.
"""

import numpy as np
from scipy.optimize import brentq
from typing import Callable

from .config import ModelParams
from .models import simulate_paths_from
from .utility import optimal_terminal_wealth, optimal_consumption_path, integrate_r


def compute_budget_TW(
    y: float,
    T: float,
    params: ModelParams,
    gamma: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo estimate of LHS of budget constraint for terminal wealth (Eq. 16).
    
    E[ξ_T * exp(-∫_0^T r_s ds) * X_T*(y)] = X_0
    
    Args:
        y: Lagrange multiplier (trial value)
        T: Terminal time
        params: Model parameters
        gamma: Risk aversion parameter
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps
        rng: Random number generator
        
    Returns:
        Monte Carlo estimate of E[ξ_T * exp(-∫_0^T r_s ds) * X_T*(y)]
    """
    dt = T / n_steps
    contributions = np.zeros(n_paths)
    
    for i in range(n_paths):
        # Simulate one path from 0 to T
        paths = simulate_paths_from(
            0.0, T, params.r0, params.lam0, 1.0, params, n_steps, 1, rng
        )
        r_path = paths['r'][0, :]
        xi_path = paths['xi'][0, :]
        
        # Compute X_T*(y) using Eq. (18)
        X_T_star = optimal_terminal_wealth(
            y, params.rho, r_path, xi_path[-1], dt, gamma
        )
        
        # Compute discount factor exp(-∫_0^T r_s ds)
        discount = np.exp(-integrate_r(r_path, dt))
        
        # Contribution to expectation
        contributions[i] = xi_path[-1] * discount * X_T_star
    
    return np.mean(contributions)


def compute_budget_IC(
    y: float,
    T: float,
    params: ModelParams,
    gamma: float,
    n_paths: int,
    n_steps: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo estimate of LHS of budget constraint for consumption (Eq. 17).
    
    E[∫_0^T ξ_s * exp(-∫_0^s r_u du) * c_s*(y) ds] = X_0
    
    Args:
        y: Lagrange multiplier (trial value)
        T: Terminal time
        params: Model parameters
        gamma: Risk aversion parameter
        n_paths: Number of Monte Carlo paths
        n_steps: Number of time steps
        rng: Random number generator
        
    Returns:
        Monte Carlo estimate of E[∫_0^T ξ_s * exp(-∫_0^s r_u du) * c_s*(y) ds]
    """
    dt = T / n_steps
    contributions = np.zeros(n_paths)
    
    for i in range(n_paths):
        # Simulate one path from 0 to T
        paths = simulate_paths_from(
            0.0, T, params.r0, params.lam0, 1.0, params, n_steps, 1, rng
        )
        r_path = paths['r'][0, :]
        xi_path = paths['xi'][0, :]
        
        # Compute optimal consumption path c_s*(y) using Eq. (19)
        c_path = optimal_consumption_path(
            y, params.rho, r_path, xi_path, dt, gamma
        )
        
        # Compute time integral: ∫_0^T ξ_s * exp(-∫_0^s r_u du) * c_s* ds
        time_integral = 0.0
        for j in range(n_steps + 1):
            # Discount factor exp(-∫_0^s r_u du) at time s_j
            r_subpath = r_path[:j+1]
            if len(r_subpath) > 1:
                discount = np.exp(-integrate_r(r_subpath, dt))
            else:
                discount = np.exp(-r_path[0] * dt * 0.5)
            
            # Contribution at time s_j
            term = xi_path[j] * discount * c_path[j]
            if j == 0 or j == n_steps:
                time_integral += 0.5 * term * dt
            else:
                time_integral += term * dt
        
        contributions[i] = time_integral
    
    return np.mean(contributions)


def solve_y_TW(
    T: float,
    params: ModelParams,
    gamma: float,
    n_paths: int = 200,  # Reduced for speed
    n_steps: int = None,
    rng: np.random.Generator = None,
    y_low: float = 1e-6,
    y_high: float = 100.0,
    rtol: float = 1e-3,  # Relaxed tolerance for speed
) -> float:
    """Solve for Lagrange multiplier y for terminal wealth case.
    
    Finds y such that compute_budget_TW(y) = X_0.
    
    Args:
        T: Terminal time
        params: Model parameters
        gamma: Risk aversion parameter
        n_paths: Number of Monte Carlo paths for budget computation
        n_steps: Number of time steps (default: 250 * T)
        rng: Random number generator (default: new generator)
        y_low: Lower bound for root finding
        y_high: Upper bound for root finding
        rtol: Relative tolerance for root finding
        
    Returns:
        Lagrange multiplier y such that budget constraint is satisfied
    """
    if n_steps is None:
        n_steps = max(10, int(50 * T))  # 50 steps per year for speed
    if rng is None:
        rng = np.random.default_rng()
    
    def objective(y: float) -> float:
        return compute_budget_TW(y, T, params, gamma, n_paths, n_steps, rng) - params.X0
    
    # Find bounds that bracket the root
    # Try different y values to find where objective changes sign
    y_test_values = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    y_low_found = None
    y_high_found = None
    
    for y_test in y_test_values:
        try:
            obj_val = objective(y_test)
            if obj_val < 0 and y_low_found is None:
                y_low_found = y_test
            elif obj_val > 0 and y_high_found is None:
                y_high_found = y_test
            if y_low_found is not None and y_high_found is not None:
                break
        except:
            continue
    
    # If we found bounds, use them; otherwise use defaults
    if y_low_found is not None and y_high_found is not None:
        y_low = y_low_found
        y_high = y_high_found
    elif y_low_found is not None:
        y_high = y_low_found * 1000.0
    elif y_high_found is not None:
        y_low = y_high_found / 1000.0
    
    # Root finding using Brent's method
    try:
        y_star = brentq(objective, y_low, y_high, rtol=rtol, maxiter=50)
    except ValueError:
        # If bounds don't bracket root, use bisection with wider range
        try:
            y_star = brentq(objective, 1e-8, 10000.0, rtol=rtol, maxiter=50)
        except ValueError:
            # Last resort: use a simple bisection-like approach
            # Find y such that objective is close to zero
            best_y = 1.0
            best_err = float('inf')
            for y_candidate in np.logspace(-6, 3, 50):
                try:
                    err = abs(objective(y_candidate))
                    if err < best_err:
                        best_err = err
                        best_y = y_candidate
                except:
                    continue
            y_star = best_y
    
    return y_star


def solve_y_IC(
    T: float,
    params: ModelParams,
    gamma: float,
    n_paths: int = 200,  # Reduced for speed
    n_steps: int = None,
    rng: np.random.Generator = None,
    y_low: float = 1e-6,
    y_high: float = 100.0,
    rtol: float = 1e-3,  # Relaxed tolerance for speed
) -> float:
    """Solve for Lagrange multiplier y for intertemporal consumption case.
    
    Finds y such that compute_budget_IC(y) = X_0.
    
    Args:
        T: Terminal time
        params: Model parameters
        gamma: Risk aversion parameter
        n_paths: Number of Monte Carlo paths for budget computation
        n_steps: Number of time steps (default: 250 * T)
        rng: Random number generator (default: new generator)
        y_low: Lower bound for root finding
        y_high: Upper bound for root finding
        rtol: Relative tolerance for root finding
        
    Returns:
        Lagrange multiplier y such that budget constraint is satisfied
    """
    if n_steps is None:
        n_steps = max(10, int(50 * T))  # 50 steps per year for speed
    if rng is None:
        rng = np.random.default_rng()
    
    def objective(y: float) -> float:
        return compute_budget_IC(y, T, params, gamma, n_paths, n_steps, rng) - params.X0
    
    # Find bounds that bracket the root
    # Try different y values to find where objective changes sign
    y_test_values = [1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0, 100.0, 1000.0]
    y_low_found = None
    y_high_found = None
    
    for y_test in y_test_values:
        try:
            obj_val = objective(y_test)
            if obj_val < 0 and y_low_found is None:
                y_low_found = y_test
            elif obj_val > 0 and y_high_found is None:
                y_high_found = y_test
            if y_low_found is not None and y_high_found is not None:
                break
        except:
            continue
    
    # If we found bounds, use them; otherwise use defaults
    if y_low_found is not None and y_high_found is not None:
        y_low = y_low_found
        y_high = y_high_found
    elif y_low_found is not None:
        y_high = y_low_found * 1000.0
    elif y_high_found is not None:
        y_low = y_high_found / 1000.0
    
    # Root finding using Brent's method
    try:
        y_star = brentq(objective, y_low, y_high, rtol=rtol, maxiter=50)
    except ValueError:
        # If bounds don't bracket root, use bisection with wider range
        try:
            y_star = brentq(objective, 1e-8, 10000.0, rtol=rtol, maxiter=50)
        except ValueError:
            # Last resort: use a simple bisection-like approach
            # Find y such that objective is close to zero
            best_y = 1.0
            best_err = float('inf')
            for y_candidate in np.logspace(-6, 3, 50):
                try:
                    err = abs(objective(y_candidate))
                    if err < best_err:
                        best_err = err
                        best_y = y_candidate
                except:
                    continue
            y_star = best_y
    
    return y_star

