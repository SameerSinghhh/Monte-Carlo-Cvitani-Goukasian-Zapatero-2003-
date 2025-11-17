"""SDE models for interest rate, market price of risk, and pricing kernel.

Implements simulation functions for the one-factor and two-factor models.
"""

import numpy as np
from typing import Dict, Tuple

from .config import ModelParams


def step_r_lambda_xi(
    r_t: float,
    lam_t: float,
    xi_t: float,
    dt: float,
    dW: float,
    params: ModelParams,
) -> Tuple[float, float, float]:
    """Single Euler-Maruyama step for (r_t, λ_t, ξ_t).
    
    Implements discretizations of:
    - Interest rate (CIR-type, Eq. 30): dr_t = κ_r(θ_r - r_t)dt - σ_r r_t^v_r dW
    - Market price of risk (OU, Eq. 31): dλ_t = κ_λ(θ_λ - λ_t)dt + σ_λ λ_t^v_λ dW
    - Pricing kernel (Eq. 15): dξ_t = ξ_t(-λ_t dW_t - ½λ_t² dt)
    
    Args:
        r_t: Current interest rate
        lam_t: Current market price of risk
        xi_t: Current pricing kernel
        dt: Time step
        dW: Brownian increment (should be ~N(0, dt))
        params: Model parameters
        
    Returns:
        Tuple of (r_{t+dt}, λ_{t+dt}, ξ_{t+dt})
    """
    # Interest rate update (Eq. 30, with non-affine variant)
    # For v_r = 0.5, this is CIR: -σ_r * sqrt(r_t) * dW
    # For v_r = 0, this is OU: -σ_r * dW
    r_factor = max(r_t, 0.0) ** params.v_r if params.v_r > 0 else 1.0
    r_next = r_t + params.kappa_r * (params.theta_r - r_t) * dt - params.sigma_r * r_factor * dW
    r_next = max(r_next, 0.0)  # Ensure non-negative
    
    # Market price of risk update (Eq. 31, with non-affine variant)
    # For v_lam = 0, this is OU: +σ_λ * dW
    # For v_lam > 0, this is non-affine: +σ_λ * λ_t^v_lam * dW
    lam_factor = abs(lam_t) ** params.v_lam if params.v_lam > 0 else 1.0
    lam_next = lam_t + params.kappa_lam * (params.theta_lam - lam_t) * dt + params.sigma_lam * lam_factor * dW
    
    # Pricing kernel update (Eq. 15)
    # dξ_t = ξ_t(-λ_t dW_t - ½λ_t² dt)
    xi_next = xi_t * np.exp(-0.5 * lam_t**2 * dt - lam_t * dW)
    
    return r_next, lam_next, xi_next


def simulate_path(
    T: float,
    params: ModelParams,
    n_steps: int,
    seed: int | None = None,
) -> Dict[str, np.ndarray]:
    """Simulate a single path from time 0 to T.
    
    Args:
        T: Terminal time
        params: Model parameters
        n_steps: Number of time steps
        seed: Random seed (optional)
        
    Returns:
        Dictionary with keys:
        - 't_grid': Time grid array (n_steps+1,)
        - 'r': Interest rate path (n_steps+1,)
        - 'lam': Market price of risk path (n_steps+1,)
        - 'xi': Pricing kernel path (n_steps+1,)
        - 'S': Stock price path (n_steps+1,) [optional]
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    t_grid = np.linspace(0, T, n_steps + 1)
    
    # Initialize arrays
    r_path = np.zeros(n_steps + 1)
    lam_path = np.zeros(n_steps + 1)
    xi_path = np.zeros(n_steps + 1)
    S_path = np.zeros(n_steps + 1)
    
    # Initial conditions
    r_path[0] = params.r0
    lam_path[0] = params.lam0
    xi_path[0] = 1.0
    S_path[0] = 1.0  # Normalized initial stock price
    
    # Simulate path
    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt))
        r_path[i+1], lam_path[i+1], xi_path[i+1] = step_r_lambda_xi(
            r_path[i], lam_path[i], xi_path[i], dt, dW, params
        )
        
        # Optional: stock price (for visualization)
        mu_t = r_path[i] + lam_path[i] * params.sigma_stock
        S_path[i+1] = S_path[i] * (1 + mu_t * dt + params.sigma_stock * dW)
    
    return {
        't_grid': t_grid,
        'r': r_path,
        'lam': lam_path,
        'xi': xi_path,
        'S': S_path,
    }


def simulate_paths_from(
    t0: float,
    T: float,
    r0: float,
    lam0: float,
    xi0: float,
    params: ModelParams,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Simulate multiple paths from t0 to T given initial state.
    
    Args:
        t0: Starting time
        T: Terminal time
        r0: Initial interest rate at t0
        lam0: Initial market price of risk at t0
        xi0: Initial pricing kernel at t0
        params: Model parameters
        n_steps: Number of time steps from t0 to T
        n_paths: Number of paths to simulate
        rng: Random number generator
        
    Returns:
        Dictionary with keys:
        - 't_grid': Time grid array (n_steps+1,)
        - 'r': Interest rate paths (n_paths, n_steps+1)
        - 'lam': Market price of risk paths (n_paths, n_steps+1)
        - 'xi': Pricing kernel paths (n_paths, n_steps+1)
    """
    dt = (T - t0) / n_steps
    t_grid = np.linspace(t0, T, n_steps + 1)
    
    # Initialize arrays
    r_paths = np.zeros((n_paths, n_steps + 1))
    lam_paths = np.zeros((n_paths, n_steps + 1))
    xi_paths = np.zeros((n_paths, n_steps + 1))
    
    # Set initial conditions
    r_paths[:, 0] = r0
    lam_paths[:, 0] = lam0
    xi_paths[:, 0] = xi0
    
    # Simulate all paths
    for i in range(n_steps):
        dW = rng.normal(0, np.sqrt(dt), size=n_paths)
        for j in range(n_paths):
            r_paths[j, i+1], lam_paths[j, i+1], xi_paths[j, i+1] = step_r_lambda_xi(
                r_paths[j, i], lam_paths[j, i], xi_paths[j, i], dt, dW[j], params
            )
    
    return {
        't_grid': t_grid,
        'r': r_paths,
        'lam': lam_paths,
        'xi': xi_paths,
    }


def simulate_two_factor_paths_from(
    t0: float,
    T: float,
    r0: float,
    lam1_0: float,
    lam2: float,  # Constant for second factor
    xi0: float,
    params: ModelParams,
    sigma1: float,
    sigma2: float,
    n_steps: int,
    n_paths: int,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    """Simulate paths for two-factor model (Table 6).
    
    Two independent Brownian motions W¹, W².
    λ₁(t) follows OU process, λ₂ is constant.
    
    Args:
        t0: Starting time
        T: Terminal time
        r0: Initial interest rate
        lam1_0: Initial value of λ₁
        lam2: Constant value of λ₂
        xi0: Initial pricing kernel
        params: Model parameters (for r and λ₁ dynamics)
        sigma1: Volatility of first stock
        sigma2: Volatility of second stock
        n_steps: Number of time steps
        n_paths: Number of paths
        rng: Random number generator
        
    Returns:
        Dictionary with paths for r, λ₁, ξ, and Brownian increments dW1, dW2
    """
    dt = (T - t0) / n_steps
    t_grid = np.linspace(t0, T, n_steps + 1)
    
    # Initialize arrays
    r_paths = np.zeros((n_paths, n_steps + 1))
    lam1_paths = np.zeros((n_paths, n_steps + 1))
    xi_paths = np.zeros((n_paths, n_steps + 1))
    dW1_paths = np.zeros((n_paths, n_steps))
    dW2_paths = np.zeros((n_paths, n_steps))
    
    # Set initial conditions
    r_paths[:, 0] = r0
    lam1_paths[:, 0] = lam1_0
    xi_paths[:, 0] = xi0
    
    # Simulate paths
    for i in range(n_steps):
        # Independent Brownian increments
        dW1 = rng.normal(0, np.sqrt(dt), size=n_paths)
        dW2 = rng.normal(0, np.sqrt(dt), size=n_paths)
        dW1_paths[:, i] = dW1
        dW2_paths[:, i] = dW2
        
        # Update r and λ₁ using dW1 (same Brownian as in one-factor case)
        for j in range(n_paths):
            r_paths[j, i+1], lam1_paths[j, i+1], _ = step_r_lambda_xi(
                r_paths[j, i], lam1_paths[j, i], 1.0, dt, dW1[j], params
            )
        
        # Update pricing kernel: dξ_t = ξ_t(-λ₁ dW¹ - λ₂ dW² - ½(λ₁² + λ₂²)dt)
        lam1_t = lam1_paths[:, i]
        xi_paths[:, i+1] = xi_paths[:, i] * np.exp(
            -0.5 * (lam1_t**2 + lam2**2) * dt - lam1_t * dW1 - lam2 * dW2
        )
    
    return {
        't_grid': t_grid,
        'r': r_paths,
        'lam1': lam1_paths,
        'lam2': lam2,  # Constant
        'xi': xi_paths,
        'dW1': dW1_paths,
        'dW2': dW2_paths,
    }

