"""Configuration module for model parameters and constants.

Defines ModelParams dataclass and parameter sets for all experiments.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class ModelParams:
    """Model parameters for the one-factor base model.
    
    Attributes:
        theta_r: Long-term mean of interest rate (θ_r)
        kappa_r: Mean reversion speed for interest rate (κ_r)
        sigma_r: Volatility of interest rate (σ_r)
        theta_lam: Long-term mean of market price of risk (θ_λ)
        kappa_lam: Mean reversion speed for market price of risk (κ_λ)
        sigma_lam: Volatility of market price of risk (σ_λ)
        sigma_stock: Stock volatility (σ)
        rho: Subjective discount rate (ρ)
        X0: Initial wealth
        r0: Initial interest rate
        lam0: Initial market price of risk
        v_r: Power for non-affine interest rate model (default 0.5 for CIR)
        v_lam: Power for non-affine market price of risk model (default 0 for OU)
    """
    theta_r: float = 0.06
    kappa_r: float = 0.0824
    sigma_r: float = 0.0364
    theta_lam: float = 0.0871
    kappa_lam: float = 0.6950
    sigma_lam: float = 0.21
    sigma_stock: float = 0.2
    rho: float = 0.0
    X0: float = 1.0
    r0: float = 0.06
    lam0: float = 0.1
    v_r: float = 0.5  # For non-affine models (Tables 4-5)
    v_lam: float = 0.0  # For non-affine models (Tables 4-5)


# Monte Carlo simulation constants (optimized for speed)
K_OUTER_PATHS = 300  # Number of outer Monte Carlo paths (reduced for speed)
M_INNER_PATHS = 5  # Number of inner Monte Carlo paths per outer path (reduced for speed)
N_TIME_PER_YEAR = 50  # Time steps per year (reduced for speed)
RANDOM_SEED = 42  # Default random seed for reproducibility

# Parameter sets for Tables 2-3 (variations of κ_r, σ_r, σ_λ)
TABLE_2_3_PARAMS = [
    # Baseline
    {"kappa_r": 0.0824, "sigma_r": 0.0364, "sigma_lam": 0.21},
    # Higher κ_r
    {"kappa_r": 0.12, "sigma_r": 0.0364, "sigma_lam": 0.21},
    # Higher σ_r
    {"kappa_r": 0.0824, "sigma_r": 0.05, "sigma_lam": 0.21},
    # Higher σ_λ
    {"kappa_r": 0.0824, "sigma_r": 0.0364, "sigma_lam": 0.3},
]

# Parameter sets for Tables 4-5 (non-affine models with v_r, v_λ)
TABLE_4_5_PARAMS = [
    {"v_r": 0.5, "v_lam": 0.0},
    {"v_r": 0.5, "v_lam": 0.5},
    {"v_r": 0.75, "v_lam": 0.0},
]

# Risk aversion parameters (γ values)
GAMMA_VALUES = [0.5, 0.0, -1.0, -2.0, -5.0, -10.0]

# Time horizons (years)
TIME_HORIZONS = [1.0, 5.0, 10.0]

# Two-factor model parameters for Table 6
SIGMA1 = 0.2  # Volatility of first stock
SIGMA2 = 0.1  # Volatility of second stock
LAM2_VALUES = [0.0, 0.03, 0.06, 0.09]  # Constant values for λ₂

