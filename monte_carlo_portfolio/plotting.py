"""Plotting functions for visualization."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def plot_sample_paths(
    t_grid: np.ndarray,
    r_paths: np.ndarray,
    lam_paths: np.ndarray,
    S_paths: Optional[np.ndarray] = None,
    n_show: int = 10,
    figsize: tuple = (12, 8),
) -> None:
    """Plot sample paths of r_t, λ_t, and optionally S_t.
    
    Args:
        t_grid: Time grid array
        r_paths: Interest rate paths (n_paths, n_steps+1) or (n_steps+1,)
        lam_paths: Market price of risk paths (n_paths, n_steps+1) or (n_steps+1,)
        S_paths: Stock price paths (optional)
        n_show: Number of paths to display
        figsize: Figure size
    """
    # Handle both single path and multiple paths
    if r_paths.ndim == 1:
        r_paths = r_paths.reshape(1, -1)
        lam_paths = lam_paths.reshape(1, -1)
        if S_paths is not None:
            S_paths = S_paths.reshape(1, -1)
        n_show = 1
    
    n_paths = min(n_show, r_paths.shape[0])
    
    fig, axes = plt.subplots(2 if S_paths is None else 3, 1, figsize=figsize, sharex=True)
    
    # Plot interest rate
    ax = axes[0] if S_paths is not None else axes[0]
    for i in range(n_paths):
        ax.plot(t_grid, r_paths[i, :], alpha=0.6, linewidth=0.8)
    ax.set_ylabel('Interest Rate $r_t$')
    ax.set_title('Sample Paths')
    ax.grid(True, alpha=0.3)
    
    # Plot market price of risk
    ax = axes[1] if S_paths is not None else axes[1]
    for i in range(n_paths):
        ax.plot(t_grid, lam_paths[i, :], alpha=0.6, linewidth=0.8, color='orange')
    ax.set_ylabel('Market Price of Risk $\\lambda_t$')
    ax.grid(True, alpha=0.3)
    
    # Plot stock price if provided
    if S_paths is not None:
        ax = axes[2]
        for i in range(n_paths):
            ax.plot(t_grid, S_paths[i, :], alpha=0.6, linewidth=0.8, color='green')
        ax.set_ylabel('Stock Price $S_t$')
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Time $t$')
    plt.tight_layout()
    plt.show()


def plot_wealth_distribution(
    X_T_samples: np.ndarray,
    title: str = "",
    bins: int = 50,
    figsize: tuple = (8, 6),
) -> None:
    """Plot histogram of optimal terminal wealth X_T*.
    
    Args:
        X_T_samples: Array of X_T* samples
        title: Plot title
        bins: Number of histogram bins
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(X_T_samples, bins=bins, density=True, alpha=0.7, edgecolor='black')
    ax.set_xlabel('Terminal Wealth $X_T^*$')
    ax.set_ylabel('Density')
    ax.set_title(title if title else 'Distribution of Optimal Terminal Wealth')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_convergence(
    K_values: np.ndarray,
    pi_estimates: np.ndarray,
    true_value: Optional[float] = None,
    title: str = "Monte Carlo Convergence",
    figsize: tuple = (8, 6),
) -> None:
    """Plot convergence of portfolio estimate as function of K.
    
    Args:
        K_values: Array of K (outer path) values
        pi_estimates: Array of portfolio estimates for each K
        true_value: True value (if known) to plot as reference line
        title: Plot title
        figsize: Figure size
    """
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(K_values, pi_estimates, 'o-', label='Estimate')
    if true_value is not None:
        ax.axhline(y=true_value, color='r', linestyle='--', label='True Value')
    ax.set_xlabel('Number of Outer Paths $K$')
    ax.set_ylabel('Portfolio Estimate $\\pi_0^*$')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

