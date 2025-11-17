"""Utility functions and optimal wealth/consumption formulas.

Implements CRRA utility and optimal terminal wealth/consumption (Eq. 18-19).
"""

import numpy as np
from typing import Tuple


def integrate_rho_minus_r(
    r_path: np.ndarray,
    rho: float,
    dt: float,
) -> float:
    """Numerically integrate ∫(ρ - r_s)ds using trapezoidal rule.
    
    Args:
        r_path: Interest rate path array
        rho: Discount rate
        dt: Time step
        
    Returns:
        Value of ∫(ρ - r_s)ds
    """
    integrand = rho - r_path
    # Trapezoidal rule
    integral = dt * (0.5 * integrand[0] + np.sum(integrand[1:-1]) + 0.5 * integrand[-1])
    return integral


def integrate_r(
    r_path: np.ndarray,
    dt: float,
) -> float:
    """Numerically integrate ∫r_s ds using trapezoidal rule.
    
    Args:
        r_path: Interest rate path array
        dt: Time step
        
    Returns:
        Value of ∫r_s ds
    """
    # Trapezoidal rule
    integral = dt * (0.5 * r_path[0] + np.sum(r_path[1:-1]) + 0.5 * r_path[-1])
    return integral


def optimal_terminal_wealth(
    y: float,
    rho: float,
    r_path: np.ndarray,
    xi_T: float,
    dt: float,
    gamma: float,
) -> float:
    """Compute optimal terminal wealth X_T* using Eq. (18).
    
    X_T* = (y * exp(∫_0^T (ρ - r_s) ds) * ξ_T)^(1/(γ-1))
    
    Args:
        y: Lagrange multiplier
        rho: Discount rate
        r_path: Interest rate path from 0 to T (n_steps+1,)
        xi_T: Pricing kernel at time T
        dt: Time step
        gamma: Risk aversion parameter (γ)
        
    Returns:
        Optimal terminal wealth X_T*
    """
    integral = integrate_rho_minus_r(r_path, rho, dt)
    exponent = 1.0 / (gamma - 1.0)
    X_T_star = (y * np.exp(integral) * xi_T) ** exponent
    return X_T_star


def optimal_consumption_path(
    y: float,
    rho: float,
    r_path: np.ndarray,
    xi_path: np.ndarray,
    dt: float,
    gamma: float,
) -> np.ndarray:
    """Compute optimal consumption c_t* at each time using Eq. (19).
    
    c_t* = (y * exp(∫_0^t (ρ - r_s) ds) * ξ_t)^(1/(γ-1))
    
    Args:
        y: Lagrange multiplier
        rho: Discount rate
        r_path: Interest rate path from 0 to T (n_steps+1,)
        xi_path: Pricing kernel path from 0 to T (n_steps+1,)
        dt: Time step
        gamma: Risk aversion parameter (γ)
        
    Returns:
        Optimal consumption path c_t* (n_steps+1,)
    """
    n_steps = len(r_path) - 1
    c_path = np.zeros(n_steps + 1)
    exponent = 1.0 / (gamma - 1.0)
    
    # Compute cumulative integral at each time point
    for i in range(n_steps + 1):
        # Integrate from 0 to t_i
        r_subpath = r_path[:i+1]
        if len(r_subpath) > 1:
            integral = integrate_rho_minus_r(r_subpath, rho, dt)
        else:
            integral = (rho - r_path[0]) * dt * 0.5  # Single point approximation
        
        c_path[i] = (y * np.exp(integral) * xi_path[i]) ** exponent
    
    return c_path


def compute_H_t_u(
    r_path: np.ndarray,
    xi_t: float,
    xi_u: float,
    dt: float,
    t_idx: int,
    u_idx: int,
) -> float:
    """Compute H_{t,u} = exp(-∫_t^u r_s ds) * ξ_u / ξ_t (Eq. 25).
    
    Args:
        r_path: Full interest rate path
        xi_t: Pricing kernel at time t
        xi_u: Pricing kernel at time u
        dt: Time step
        t_idx: Index corresponding to time t
        u_idx: Index corresponding to time u
        
    Returns:
        Value of H_{t,u}
    """
    # Integrate r from t to u
    r_subpath = r_path[t_idx:u_idx+1]
    if len(r_subpath) > 1:
        integral = integrate_r(r_subpath, dt)
    else:
        integral = r_path[t_idx] * dt * 0.5
    
    H = np.exp(-integral) * xi_u / xi_t
    return H

