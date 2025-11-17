"""Main entry point for Monte Carlo Portfolio Optimization.

Command-line interface to run experiments and reproduce tables from the paper.
"""

import argparse
import numpy as np

from monte_carlo_portfolio.config import (
    ModelParams,
    K_OUTER_PATHS,
    M_INNER_PATHS,
    N_TIME_PER_YEAR,
    RANDOM_SEED,
)
from monte_carlo_portfolio.budget_solver import solve_y_TW, solve_y_IC
from monte_carlo_portfolio.estimator import (
    estimate_phi_and_portfolio_TW,
    estimate_phi_and_portfolio_IC,
)
from monte_carlo_portfolio.experiments import (
    run_table_1,
    run_table_2,
    run_table_3,
    run_table_4,
    run_table_5,
    run_table_6,
)
from monte_carlo_portfolio.comparison import write_comparison_report
from monte_carlo_portfolio.plotting import plot_sample_paths
from monte_carlo_portfolio.models import simulate_path


def run_single_test_case(
    gamma: float = -1.0,
    T: float = 1.0,
    K: int = K_OUTER_PATHS,
    M: int = M_INNER_PATHS,
    seed: int = RANDOM_SEED,
) -> None:
    """Run a single test case to verify implementation.
    
    Args:
        gamma: Risk aversion parameter
        T: Time horizon
        K: Number of outer paths
        M: Number of inner paths
        seed: Random seed
    """
    print(f"\n{'='*60}")
    print(f"Single Test Case: γ={gamma}, T={T}")
    print(f"{'='*60}\n")
    
    params = ModelParams()
    dt_small = 1.0 / N_TIME_PER_YEAR
    n_steps_T = int(N_TIME_PER_YEAR * T)
    rng = np.random.default_rng(seed)
    
    # Solve for Lagrange multipliers
    print("Solving for Lagrange multipliers...")
    print("  (Using fast settings for speed)")
    y_tw = solve_y_TW(T, params, gamma, n_paths=200, n_steps=n_steps_T, rng=rng)
    print(f"  y_TW = {y_tw:.6f}")
    y_ic = solve_y_IC(T, params, gamma, n_paths=200, n_steps=n_steps_T, rng=rng)
    print(f"  y_IC = {y_ic:.6f}\n")
    
    # Estimate optimal portfolios
    print("Estimating optimal portfolios...")
    print(f"  Using K={K} outer paths, M={M} inner paths per outer path")
    pi_tw = estimate_phi_and_portfolio_TW(
        T, gamma, params, y_tw, dt_small, K, M, n_steps_T, rng
    )
    pi_ic = estimate_phi_and_portfolio_IC(
        T, gamma, params, y_ic, dt_small, K, M, n_steps_T, rng
    )
    
    print(f"\nResults:")
    print(f"  π_0^IC = {pi_ic:.4f}")
    print(f"  π_0^TW = {pi_tw:.4f}")
    print(f"\n{'='*60}\n")


def plot_sample_path_demo(seed: int = RANDOM_SEED) -> None:
    """Generate and plot sample paths for visualization.
    
    Args:
        seed: Random seed
    """
    print("\nGenerating sample paths...")
    params = ModelParams()
    
    # Simulate a few paths
    n_paths = 5
    T = 1.0
    n_steps = int(N_TIME_PER_YEAR * T)
    
    paths_list = []
    for i in range(n_paths):
        path = simulate_path(T, params, n_steps, seed=seed + i)
        paths_list.append(path)
    
    # Combine paths
    t_grid = paths_list[0]['t_grid']
    r_paths = np.array([p['r'] for p in paths_list])
    lam_paths = np.array([p['lam'] for p in paths_list])
    S_paths = np.array([p['S'] for p in paths_list])
    
    plot_sample_paths(t_grid, r_paths, lam_paths, S_paths, n_show=n_paths)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monte Carlo Portfolio Optimization - Reproduce Tables 1-6"
    )
    parser.add_argument(
        '--table',
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        help='Table number to reproduce (1-6)',
    )
    parser.add_argument(
        '--all-tables',
        action='store_true',
        help='Run all tables (may take a long time)',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run single test case (γ=-1, T=1)',
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Plot sample paths',
    )
    parser.add_argument(
        '--K',
        type=int,
        default=K_OUTER_PATHS,
        help=f'Number of outer Monte Carlo paths (default: {K_OUTER_PATHS})',
    )
    parser.add_argument(
        '--M',
        type=int,
        default=M_INNER_PATHS,
        help=f'Number of inner Monte Carlo paths (default: {M_INNER_PATHS})',
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_SEED,
        help=f'Random seed (default: {RANDOM_SEED})',
    )
    
    args = parser.parse_args()
    
    # Run requested experiments
    if args.test:
        run_single_test_case(K=args.K, M=args.M, seed=args.seed)
    
    if args.plot:
        plot_sample_path_demo(seed=args.seed)
    
    if args.table:
        print(f"\n{'='*60}")
        print(f"Reproducing Table {args.table}")
        print(f"{'='*60}\n")
        
        table_functions = {
            1: run_table_1,
            2: run_table_2,
            3: run_table_3,
            4: run_table_4,
            5: run_table_5,
            6: run_table_6,
        }
        
        # Run table (will automatically write comparison report)
        table_functions[args.table](K=args.K, M=args.M, seed=args.seed)
    
    if args.all_tables:
        print("\n" + "="*60)
        print("Running ALL Tables (this may take a long time...)")
        print("="*60 + "\n")
        
        table_functions = {
            1: run_table_1,
            2: run_table_2,
            3: run_table_3,
            4: run_table_4,
            5: run_table_5,
            6: run_table_6,
        }
        
        results = {}
        for table_num in [1, 2, 3, 4, 5, 6]:
            print(f"\n{'='*60}")
            print(f"Table {table_num}")
            print(f"{'='*60}\n")
            
            df = table_functions[table_num](K=args.K, M=args.M, seed=args.seed)
            results[table_num] = df
        
        # Write comprehensive comparison report
        print("\n" + "="*60)
        print("Generating comprehensive comparison report...")
        print("="*60 + "\n")
        write_comparison_report(results, output_dir="results")
    
    # Default: run test case if no arguments
    if not any([args.test, args.plot, args.table, args.all_tables]):
        print("No action specified. Running test case by default.")
        print("Use --help for options.\n")
        run_single_test_case(K=args.K, M=args.M, seed=args.seed)


if __name__ == "__main__":
    main()

