"""Monte Carlo estimator for φ_t and optimal portfolio π_t.

Implements nested Monte Carlo method (Eq. 24, 28, 39-40) to estimate
optimal portfolio weights.
"""

import numpy as np
from typing import Tuple

from .config import ModelParams
from .models import simulate_paths_from, simulate_two_factor_paths_from
from .utility import (
    optimal_terminal_wealth,
    optimal_consumption_path,
    integrate_r,
    integrate_rho_minus_r,
    compute_H_t_u,
)


def conditional_X_star_TW_at_dt(
    T: float,
    t0: float,
    r_t0: float,
    lam_t0: float,
    xi_t0: float,
    y_TW: float,
    gamma: float,
    params: ModelParams,
    M: int,
    n_steps_from_t0_to_T: int,
    rng: np.random.Generator,
    rho_minus_r_integral_0_to_t0: float = 0.0,
) -> float:
    """Estimate X_{t0}* for terminal wealth case using inner Monte Carlo (Eq. 26).
    
    X_{t0}* = E[H_{t0,T} * X_T* | F_{t0}]
    
    Args:
        T: Terminal time
        t0: Current time (typically Δt)
        r_t0: Interest rate at t0
        lam_t0: Market price of risk at t0
        xi_t0: Pricing kernel at t0
        y_TW: Lagrange multiplier for terminal wealth
        gamma: Risk aversion parameter
        params: Model parameters
        M: Number of inner Monte Carlo paths
        n_steps_from_t0_to_T: Number of steps from t0 to T
        rng: Random number generator
        rho_minus_r_integral_0_to_t0: Precomputed ∫_0^{t0} (ρ - r_s) ds (for computing X_T*)
        
    Returns:
        Estimate of X_{t0}*
    """
    dt = (T - t0) / n_steps_from_t0_to_T
    contributions = np.zeros(M)
    
    for j in range(M):
        # Simulate path from t0 to T
        paths = simulate_paths_from(
            t0, T, r_t0, lam_t0, xi_t0, params,
            n_steps_from_t0_to_T, 1, rng
        )
        r_path = paths['r'][0, :]
        xi_path = paths['xi'][0, :]
        
        # To compute X_T*, we need ∫_0^T (ρ - r_s) ds
        # We have ∫_0^{t0} (ρ - r_s) ds from outer path
        # Need ∫_{t0}^T (ρ - r_s) ds from inner path
        rho_minus_r_integral_t0_to_T = integrate_rho_minus_r(r_path, params.rho, dt)
        # Combine integrals
        rho_minus_r_integral_0_to_T = rho_minus_r_integral_0_to_t0 + rho_minus_r_integral_t0_to_T
        
        # Compute X_T* using Eq. (18)
        # X_T* = (y * exp(∫_0^T (ρ - r_s) ds) * ξ_T)^(1/(γ-1))
        X_T_star = (y_TW * np.exp(rho_minus_r_integral_0_to_T) * xi_path[-1]) ** (1.0 / (gamma - 1.0))
        
        # Compute H_{t0,T} = exp(-∫_{t0}^T r_s ds) * ξ_T / ξ_{t0}
        r_integral_t0_to_T_discount = integrate_r(r_path, dt)
        H_t0_T = np.exp(-r_integral_t0_to_T_discount) * xi_path[-1] / xi_t0
        
        contributions[j] = H_t0_T * X_T_star
    
    return np.mean(contributions)


def conditional_X_star_IC_at_dt(
    T: float,
    t0: float,
    r_t0: float,
    lam_t0: float,
    xi_t0: float,
    y_IC: float,
    gamma: float,
    params: ModelParams,
    M: int,
    n_steps_from_t0_to_T: int,
    rng: np.random.Generator,
    rho_minus_r_integral_0_to_t0: float = 0.0,
) -> float:
    """Estimate X_{t0}* for consumption case using inner Monte Carlo (Eq. 27).
    
    X_{t0}* = E[∫_{t0}^T H_{t0,s} * c_s* ds | F_{t0}]
    
    Args:
        T: Terminal time
        t0: Current time (typically Δt)
        r_t0: Interest rate at t0
        lam_t0: Market price of risk at t0
        xi_t0: Pricing kernel at t0
        y_IC: Lagrange multiplier for consumption
        gamma: Risk aversion parameter
        params: Model parameters
        M: Number of inner Monte Carlo paths
        n_steps_from_t0_to_T: Number of steps from t0 to T
        rng: Random number generator
        rho_minus_r_integral_0_to_t0: Precomputed ∫_0^{t0} (ρ - r_s) ds
        
    Returns:
        Estimate of X_{t0}*
    """
    dt = (T - t0) / n_steps_from_t0_to_T
    contributions = np.zeros(M)
    
    for j in range(M):
        # Simulate path from t0 to T
        paths = simulate_paths_from(
            t0, T, r_t0, lam_t0, xi_t0, params,
            n_steps_from_t0_to_T, 1, rng
        )
        r_path = paths['r'][0, :]
        xi_path = paths['xi'][0, :]
        
        # Compute optimal consumption path c_s* along inner path
        # For c_s*, we need ∫_0^s (ρ - r_u) du
        # Approximate: ∫_0^s = ∫_0^{t0} + ∫_{t0}^s
        c_path = np.zeros(n_steps_from_t0_to_T + 1)
        exponent = 1.0 / (gamma - 1.0)
        
        for k in range(n_steps_from_t0_to_T + 1):
            # Time s_k = t0 + k * dt
            r_subpath = r_path[:k+1]
            if len(r_subpath) > 1:
                rho_minus_r_integral_t0_to_s = integrate_rho_minus_r(r_subpath, params.rho, dt)
            else:
                rho_minus_r_integral_t0_to_s = (params.rho - r_t0) * dt * 0.5
            
            # Approximate full integral
            rho_minus_r_integral_0_to_s = rho_minus_r_integral_0_to_t0 + rho_minus_r_integral_t0_to_s
            c_path[k] = (y_IC * np.exp(rho_minus_r_integral_0_to_s) * xi_path[k]) ** exponent
        
        # Compute time integral: ∫_{t0}^T H_{t0,s} * c_s* ds
        time_integral = 0.0
        for k in range(n_steps_from_t0_to_T + 1):
            # H_{t0,s_k} = exp(-∫_{t0}^{s_k} r_u du) * ξ_{s_k} / ξ_{t0}
            r_subpath = r_path[:k+1]
            if len(r_subpath) > 1:
                r_integral_t0_to_s = integrate_r(r_subpath, dt)
            else:
                r_integral_t0_to_s = r_t0 * dt * 0.5
            
            H_t0_s = np.exp(-r_integral_t0_to_s) * xi_path[k] / xi_t0
            
            term = H_t0_s * c_path[k]
            if k == 0 or k == n_steps_from_t0_to_T:
                time_integral += 0.5 * term * dt
            else:
                time_integral += term * dt
        
        contributions[j] = time_integral
    
    return np.mean(contributions)


def estimate_phi_and_portfolio_TW(
    T: float,
    gamma: float,
    params: ModelParams,
    y_TW: float,
    dt_small: float,
    K: int,
    M: int,
    n_steps_T: int,
    rng: np.random.Generator,
) -> float:
    """Estimate φ_0 and optimal portfolio π_0* for terminal wealth case (Eq. 28).
    
    Args:
        T: Terminal time
        gamma: Risk aversion parameter
        params: Model parameters
        y_TW: Lagrange multiplier for terminal wealth
        dt_small: Small time step for outer loop (e.g., 1/250)
        K: Number of outer Monte Carlo paths
        M: Number of inner Monte Carlo paths per outer path
        n_steps_T: Number of steps from 0 to T (for inner paths)
        rng: Random number generator
        
    Returns:
        Estimated optimal portfolio weight π_0* = φ_0 / σ(0)
    """
    X_0_star = params.X0  # For TW, X_0* = X_0
    
    # Outer loop: sample shocks z_i ~ N(0, dt_small)
    phi_contributions = np.zeros(K)
    
    for i in range(K):
        # Sample outer shock
        z_i = rng.normal(0, np.sqrt(dt_small))
        
        # Simulate one step from 0 to dt_small
        r_dt, lam_dt, xi_dt = params.r0, params.lam0, 1.0
        # Single step update
        from .models import step_r_lambda_xi
        r_dt, lam_dt, xi_dt = step_r_lambda_xi(
            params.r0, params.lam0, 1.0, dt_small, z_i, params
        )
        
        # Precompute ∫_0^{dt_small} (ρ - r_s) ds (approximate r as constant)
        rho_minus_r_integral_0_to_dt = (params.rho - params.r0) * dt_small
        
        # Inner Monte Carlo: estimate X_{dt_small}*
        n_steps_inner = max(1, int((T - dt_small) / dt_small))
        X_dt_star = conditional_X_star_TW_at_dt(
            T, dt_small, r_dt, lam_dt, xi_dt, y_TW, gamma, params,
            M, n_steps_inner, rng, rho_minus_r_integral_0_to_dt
        )
        
        # Contribution to φ_0 estimator (Eq. 28)
        phi_contributions[i] = (X_dt_star - X_0_star) * z_i / dt_small
    
    phi_0 = np.mean(phi_contributions)
    
    # Optimal portfolio: π_0* = φ_0 / σ(0)
    pi_0_star = phi_0 / params.sigma_stock
    
    return pi_0_star


def estimate_phi_and_portfolio_IC(
    T: float,
    gamma: float,
    params: ModelParams,
    y_IC: float,
    dt_small: float,
    K: int,
    M: int,
    n_steps_T: int,
    rng: np.random.Generator,
) -> float:
    """Estimate φ_0 and optimal portfolio π_0* for consumption case.
    
    Args:
        T: Terminal time
        gamma: Risk aversion parameter
        params: Model parameters
        y_IC: Lagrange multiplier for consumption
        dt_small: Small time step for outer loop
        K: Number of outer Monte Carlo paths
        M: Number of inner Monte Carlo paths per outer path
        n_steps_T: Number of steps from 0 to T
        rng: Random number generator
        
    Returns:
        Estimated optimal portfolio weight π_0* = φ_0 / σ(0)
    """
    X_0_star = params.X0  # For IC, X_0* = X_0
    
    # Outer loop
    phi_contributions = np.zeros(K)
    
    for i in range(K):
        # Sample outer shock
        z_i = rng.normal(0, np.sqrt(dt_small))
        
        # Simulate one step from 0 to dt_small
        from .models import step_r_lambda_xi
        r_dt, lam_dt, xi_dt = step_r_lambda_xi(
            params.r0, params.lam0, 1.0, dt_small, z_i, params
        )
        
        # Precompute ∫_0^{dt_small} (ρ - r_s) ds (approximate r as constant)
        rho_minus_r_integral_0_to_dt = (params.rho - params.r0) * dt_small
        
        # Inner Monte Carlo: estimate X_{dt_small}*
        n_steps_inner = max(1, int((T - dt_small) / dt_small))
        X_dt_star = conditional_X_star_IC_at_dt(
            T, dt_small, r_dt, lam_dt, xi_dt, y_IC, gamma, params,
            M, n_steps_inner, rng, rho_minus_r_integral_0_to_dt
        )
        
        # Contribution to φ_0 estimator
        phi_contributions[i] = (X_dt_star - X_0_star) * z_i / dt_small
    
    phi_0 = np.mean(phi_contributions)
    
    # Optimal portfolio
    pi_0_star = phi_0 / params.sigma_stock
    
    return pi_0_star


def estimate_phi_and_portfolio_two_factor(
    T: float,
    gamma: float,
    params: ModelParams,
    y_TW: float,
    dt_small: float,
    K: int,
    M: int,
    n_steps_T: int,
    lam2_const: float,
    sigma1: float,
    sigma2: float,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    """Estimate φ₁, φ₂ and optimal portfolios π₁, π₂ for two-factor model (Eq. 39-40).
    
    Args:
        T: Terminal time
        gamma: Risk aversion parameter
        params: Model parameters
        y_TW: Lagrange multiplier for terminal wealth
        dt_small: Small time step
        K: Number of outer paths
        M: Number of inner paths
        n_steps_T: Number of steps to T
        lam2_const: Constant value of λ₂
        sigma1: Volatility of first stock
        sigma2: Volatility of second stock
        rng: Random number generator
        
    Returns:
        Tuple of (π₁_0*, π₂_0*) - optimal portfolio weights for two stocks
    """
    X_0_star = params.X0
    
    # Outer loop: sample independent shocks z1_i, z2_i
    phi1_contributions = np.zeros(K)
    phi2_contributions = np.zeros(K)
    
    for i in range(K):
        # Sample independent outer shocks
        z1_i = rng.normal(0, np.sqrt(dt_small))
        z2_i = rng.normal(0, np.sqrt(dt_small))
        
        # Simulate one step using z1_i for r and λ₁
        from .models import step_r_lambda_xi
        r_dt, lam1_dt, _ = step_r_lambda_xi(
            params.r0, params.lam0, 1.0, dt_small, z1_i, params
        )
        
        # Pricing kernel update with both shocks
        xi_dt = np.exp(
            -0.5 * (params.lam0**2 + lam2_const**2) * dt_small
            - params.lam0 * z1_i - lam2_const * z2_i
        )
        
        # Inner Monte Carlo: estimate X_{dt_small}* conditioned on both shocks
        # This requires simulating two-factor paths
        n_steps_inner = max(1, int((T - dt_small) / dt_small))
        
        # For simplicity, use one-factor inner simulation with adjusted pricing kernel
        # In full implementation, would use simulate_two_factor_paths_from
        # Here we approximate by using the one-factor estimator
        # Estimate X_dt* using inner MC (simplified - full version would use two-factor paths)
        rho_minus_r_integral_0_to_dt = (params.rho - params.r0) * dt_small
        X_dt_star = conditional_X_star_TW_at_dt(
            T, dt_small, r_dt, lam1_dt, xi_dt, y_TW, gamma, params,
            M, n_steps_inner, rng, rho_minus_r_integral_0_to_dt
        )
        
        # Contributions to φ₁ and φ₂ (Eq. 39-40)
        phi1_contributions[i] = (X_dt_star - X_0_star) * z1_i / dt_small
        phi2_contributions[i] = (X_dt_star - X_0_star) * z2_i / dt_small
    
    phi1_0 = np.mean(phi1_contributions)
    phi2_0 = np.mean(phi2_contributions)
    
    # Optimal portfolios: π₁ = φ₁ / σ₁, π₂ = φ₂ / σ₂
    pi1_0_star = phi1_0 / sigma1
    pi2_0_star = phi2_0 / sigma2
    
    return pi1_0_star, pi2_0_star

