"""Experiments to reproduce Tables 1-6 from the paper."""

import numpy as np
import pandas as pd
from typing import Dict

from .config import (
    ModelParams,
    K_OUTER_PATHS,
    M_INNER_PATHS,
    N_TIME_PER_YEAR,
    RANDOM_SEED,
    TABLE_2_3_PARAMS,
    TABLE_4_5_PARAMS,
    GAMMA_VALUES,
    TIME_HORIZONS,
    SIGMA1,
    SIGMA2,
    LAM2_VALUES,
)
from .budget_solver import solve_y_TW, solve_y_IC
from .estimator import (
    estimate_phi_and_portfolio_TW,
    estimate_phi_and_portfolio_IC,
    estimate_phi_and_portfolio_two_factor,
)
from .comparison import write_single_table_report


def run_table_1(
    K: int = K_OUTER_PATHS,
    M: int = M_INNER_PATHS,
    n_steps_per_year: int = N_TIME_PER_YEAR,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reproduce Table 1: Horizon and risk aversion (IC vs TW).
    
    Args:
        K: Number of outer Monte Carlo paths
        M: Number of inner Monte Carlo paths
        n_steps_per_year: Time steps per year
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        DataFrame with results
    """
    params = ModelParams()
    dt_small = 1.0 / n_steps_per_year
    rng = np.random.default_rng(seed)
    
    results = []
    
    for gamma in GAMMA_VALUES:
        for T in TIME_HORIZONS:
            if verbose:
                print(f"Table 1: γ={gamma}, T={T}")
            
            n_steps_T = int(n_steps_per_year * T)
            
            # Solve for y (using reduced paths for speed)
            y_tw = solve_y_TW(T, params, gamma, n_paths=200, n_steps=n_steps_T, rng=rng)
            y_ic = solve_y_IC(T, params, gamma, n_paths=200, n_steps=n_steps_T, rng=rng)
            
            # Estimate portfolios
            pi_tw = estimate_phi_and_portfolio_TW(
                T, gamma, params, y_tw, dt_small, K, M, n_steps_T, rng
            )
            pi_ic = estimate_phi_and_portfolio_IC(
                T, gamma, params, y_ic, dt_small, K, M, n_steps_T, rng
            )
            
            results.append({
                'γ': gamma,
                'T': T,
                'π_IC': pi_ic,
                'π_TW': pi_tw,
            })
    
    df = pd.DataFrame(results)
    if verbose:
        print("\nTable 1 Results:")
        print(df.to_string(index=False))
    
    # Write comparison report
    write_single_table_report(1, df)
    
    return df


def run_table_2(
    K: int = K_OUTER_PATHS,
    M: int = M_INNER_PATHS,
    n_steps_per_year: int = N_TIME_PER_YEAR,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reproduce Table 2: Effect of κ_r, σ_r, σ_λ for γ = -1.
    
    Args:
        K: Number of outer Monte Carlo paths
        M: Number of inner Monte Carlo paths
        n_steps_per_year: Time steps per year
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        DataFrame with results
    """
    gamma = -1.0
    dt_small = 1.0 / n_steps_per_year
    rng = np.random.default_rng(seed)
    
    results = []
    
    for param_set in TABLE_2_3_PARAMS:
        params = ModelParams(**param_set)
        
        for T in TIME_HORIZONS:
            if verbose:
                print(f"Table 2: κ_r={params.kappa_r}, σ_r={params.sigma_r}, "
                      f"σ_λ={params.sigma_lam}, T={T}")
            
            n_steps_T = int(n_steps_per_year * T)
            
            y_tw = solve_y_TW(T, params, gamma, n_paths=5000, n_steps=n_steps_T, rng=rng)
            y_ic = solve_y_IC(T, params, gamma, n_paths=5000, n_steps=n_steps_T, rng=rng)
            
            pi_tw = estimate_phi_and_portfolio_TW(
                T, gamma, params, y_tw, dt_small, K, M, n_steps_T, rng
            )
            pi_ic = estimate_phi_and_portfolio_IC(
                T, gamma, params, y_ic, dt_small, K, M, n_steps_T, rng
            )
            
            results.append({
                'κ_r': params.kappa_r,
                'σ_r': params.sigma_r,
                'σ_λ': params.sigma_lam,
                'T': T,
                'π_IC': pi_ic,
                'π_TW': pi_tw,
            })
    
    df = pd.DataFrame(results)
    if verbose:
        print("\nTable 2 Results:")
        print(df.to_string(index=False))
    
    # Write comparison report
    write_single_table_report(2, df)
    
    return df


def run_table_3(
    K: int = K_OUTER_PATHS,
    M: int = M_INNER_PATHS,
    n_steps_per_year: int = N_TIME_PER_YEAR,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reproduce Table 3: Effect of κ_r, σ_r, σ_λ for γ = -2.
    
    Args:
        K: Number of outer Monte Carlo paths
        M: Number of inner Monte Carlo paths
        n_steps_per_year: Time steps per year
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        DataFrame with results
    """
    gamma = -2.0
    dt_small = 1.0 / n_steps_per_year
    rng = np.random.default_rng(seed)
    
    results = []
    
    for param_set in TABLE_2_3_PARAMS:
        params = ModelParams(**param_set)
        
        for T in TIME_HORIZONS:
            if verbose:
                print(f"Table 3: κ_r={params.kappa_r}, σ_r={params.sigma_r}, "
                      f"σ_λ={params.sigma_lam}, T={T}")
            
            n_steps_T = int(n_steps_per_year * T)
            
            y_tw = solve_y_TW(T, params, gamma, n_paths=5000, n_steps=n_steps_T, rng=rng)
            y_ic = solve_y_IC(T, params, gamma, n_paths=5000, n_steps=n_steps_T, rng=rng)
            
            pi_tw = estimate_phi_and_portfolio_TW(
                T, gamma, params, y_tw, dt_small, K, M, n_steps_T, rng
            )
            pi_ic = estimate_phi_and_portfolio_IC(
                T, gamma, params, y_ic, dt_small, K, M, n_steps_T, rng
            )
            
            results.append({
                'κ_r': params.kappa_r,
                'σ_r': params.sigma_r,
                'σ_λ': params.sigma_lam,
                'T': T,
                'π_IC': pi_ic,
                'π_TW': pi_tw,
            })
    
    df = pd.DataFrame(results)
    if verbose:
        print("\nTable 3 Results:")
        print(df.to_string(index=False))
    
    # Write comparison report
    write_single_table_report(3, df)
    
    return df


def run_table_4(
    K: int = K_OUTER_PATHS,
    M: int = M_INNER_PATHS,
    n_steps_per_year: int = N_TIME_PER_YEAR,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reproduce Table 4: Non-affine models (v_r, v_λ) for γ = -1.
    
    Args:
        K: Number of outer Monte Carlo paths
        M: Number of inner Monte Carlo paths
        n_steps_per_year: Time steps per year
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        DataFrame with results
    """
    gamma = -1.0
    dt_small = 1.0 / n_steps_per_year
    rng = np.random.default_rng(seed)
    
    results = []
    
    for param_set in TABLE_4_5_PARAMS:
        params = ModelParams(v_r=param_set['v_r'], v_lam=param_set['v_lam'])
        
        for T in TIME_HORIZONS:
            if verbose:
                print(f"Table 4: v_r={params.v_r}, v_λ={params.v_lam}, T={T}")
            
            n_steps_T = int(n_steps_per_year * T)
            
            y_tw = solve_y_TW(T, params, gamma, n_paths=5000, n_steps=n_steps_T, rng=rng)
            y_ic = solve_y_IC(T, params, gamma, n_paths=5000, n_steps=n_steps_T, rng=rng)
            
            pi_tw = estimate_phi_and_portfolio_TW(
                T, gamma, params, y_tw, dt_small, K, M, n_steps_T, rng
            )
            pi_ic = estimate_phi_and_portfolio_IC(
                T, gamma, params, y_ic, dt_small, K, M, n_steps_T, rng
            )
            
            results.append({
                'v_r': params.v_r,
                'v_λ': params.v_lam,
                'T': T,
                'π_IC': pi_ic,
                'π_TW': pi_tw,
            })
    
    df = pd.DataFrame(results)
    if verbose:
        print("\nTable 4 Results:")
        print(df.to_string(index=False))
    
    # Write comparison report
    write_single_table_report(4, df)
    
    return df


def run_table_5(
    K: int = K_OUTER_PATHS,
    M: int = M_INNER_PATHS,
    n_steps_per_year: int = N_TIME_PER_YEAR,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reproduce Table 5: Non-affine models (v_r, v_λ) for γ = -2.
    
    Args:
        K: Number of outer Monte Carlo paths
        M: Number of inner Monte Carlo paths
        n_steps_per_year: Time steps per year
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        DataFrame with results
    """
    gamma = -2.0
    dt_small = 1.0 / n_steps_per_year
    rng = np.random.default_rng(seed)
    time_horizons = [1.0, 5.0]  # Table 5 only uses T=1,5
    
    results = []
    
    for param_set in TABLE_4_5_PARAMS:
        params = ModelParams(v_r=param_set['v_r'], v_lam=param_set['v_lam'])
        
        for T in time_horizons:
            if verbose:
                print(f"Table 5: v_r={params.v_r}, v_λ={params.v_lam}, T={T}")
            
            n_steps_T = int(n_steps_per_year * T)
            
            y_tw = solve_y_TW(T, params, gamma, n_paths=5000, n_steps=n_steps_T, rng=rng)
            y_ic = solve_y_IC(T, params, gamma, n_paths=5000, n_steps=n_steps_T, rng=rng)
            
            pi_tw = estimate_phi_and_portfolio_TW(
                T, gamma, params, y_tw, dt_small, K, M, n_steps_T, rng
            )
            pi_ic = estimate_phi_and_portfolio_IC(
                T, gamma, params, y_ic, dt_small, K, M, n_steps_T, rng
            )
            
            results.append({
                'v_r': params.v_r,
                'v_λ': params.v_lam,
                'T': T,
                'π_IC': pi_ic,
                'π_TW': pi_tw,
            })
    
    df = pd.DataFrame(results)
    if verbose:
        print("\nTable 5 Results:")
        print(df.to_string(index=False))
    
    # Write comparison report
    write_single_table_report(5, df)
    
    return df


def run_table_6(
    K: int = K_OUTER_PATHS,
    M: int = M_INNER_PATHS,
    n_steps_per_year: int = N_TIME_PER_YEAR,
    seed: int = RANDOM_SEED,
    verbose: bool = True,
) -> pd.DataFrame:
    """Reproduce Table 6: Two-stock example with varying λ₂.
    
    Args:
        K: Number of outer Monte Carlo paths
        M: Number of inner Monte Carlo paths
        n_steps_per_year: Time steps per year
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        DataFrame with results
    """
    params = ModelParams()
    dt_small = 1.0 / n_steps_per_year
    rng = np.random.default_rng(seed)
    
    T = 1.0  # Table 6 uses T=1
    n_steps_T = int(n_steps_per_year * T)
    
    results = []
    
    for gamma in [-1.0, -2.0]:
        # Solve for y (using one-factor model, reduced paths for speed)
        y_tw = solve_y_TW(T, params, gamma, n_paths=200, n_steps=n_steps_T, rng=rng)
        
        for lam2 in LAM2_VALUES:
            if verbose:
                print(f"Table 6: γ={gamma}, λ₂={lam2}")
            
            pi1, pi2 = estimate_phi_and_portfolio_two_factor(
                T, gamma, params, y_tw, dt_small, K, M, n_steps_T,
                lam2, SIGMA1, SIGMA2, rng
            )
            
            results.append({
                'γ': gamma,
                'λ₂': lam2,
                'π₁': pi1,
                'π₂': pi2,
            })
    
    df = pd.DataFrame(results)
    if verbose:
        print("\nTable 6 Results:")
        print(df.to_string(index=False))
    
    # Write comparison report
    write_single_table_report(6, df)
    
    return df

