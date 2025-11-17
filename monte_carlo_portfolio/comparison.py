"""Comparison output functions to compare results with paper values."""

import pandas as pd
from typing import Dict, Tuple, Optional
from datetime import datetime

from .paper_values import (
    TABLE_1_PAPER,
    TABLE_2_PAPER,
    TABLE_3_PAPER,
    TABLE_4_PAPER,
    TABLE_5_PAPER,
    TABLE_6_PAPER,
)


def format_comparison_table_1(df: pd.DataFrame) -> str:
    """Format Table 1 with comparison to paper values."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 1: Horizon and Risk Aversion (IC vs TW)")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'γ':<8} {'T':<6} {'π_IC (Ours)':<15} {'π_IC (Paper)':<15} {'Diff IC':<12} "
                 f"{'π_TW (Ours)':<15} {'π_TW (Paper)':<15} {'Diff TW':<12}")
    lines.append("-" * 100)
    
    for _, row in df.iterrows():
        gamma = row['γ']
        T = row['T']
        pi_ic_ours = row['π_IC']
        pi_tw_ours = row['π_TW']
        
        key = (gamma, T)
        if key in TABLE_1_PAPER:
            pi_ic_paper, pi_tw_paper = TABLE_1_PAPER[key]
            diff_ic = pi_ic_ours - pi_ic_paper
            diff_tw = pi_tw_ours - pi_tw_paper
            lines.append(
                f"{gamma:<8.1f} {T:<6.1f} {pi_ic_ours:<15.4f} {pi_ic_paper:<15.3f} {diff_ic:<12.4f} "
                f"{pi_tw_ours:<15.4f} {pi_tw_paper:<15.3f} {diff_tw:<12.4f}"
            )
        else:
            lines.append(
                f"{gamma:<8.1f} {T:<6.1f} {pi_ic_ours:<15.4f} {'N/A':<15} {'N/A':<12} "
                f"{pi_tw_ours:<15.4f} {'N/A':<15} {'N/A':<12}"
            )
    
    lines.append("")
    return "\n".join(lines)


def format_comparison_table_2(df: pd.DataFrame) -> str:
    """Format Table 2 with comparison to paper values."""
    lines = []
    lines.append("=" * 120)
    lines.append("TABLE 2: Effect of κ_r, σ_r, σ_λ for γ = -1")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"{'κ_r':<8} {'σ_r':<8} {'σ_λ':<8} {'T':<6} {'π_IC (Ours)':<15} {'π_IC (Paper)':<15} {'Diff IC':<12} "
                 f"{'π_TW (Ours)':<15} {'π_TW (Paper)':<15} {'Diff TW':<12}")
    lines.append("-" * 120)
    
    for _, row in df.iterrows():
        kappa_r = row['κ_r']
        sigma_r = row['σ_r']
        sigma_lam = row['σ_λ']
        T = row['T']
        pi_ic_ours = row['π_IC']
        pi_tw_ours = row['π_TW']
        
        key = (kappa_r, sigma_r, sigma_lam, T)
        if key in TABLE_2_PAPER:
            pi_ic_paper, pi_tw_paper = TABLE_2_PAPER[key]
            diff_ic = pi_ic_ours - pi_ic_paper
            diff_tw = pi_tw_ours - pi_tw_paper
            lines.append(
                f"{kappa_r:<8.4f} {sigma_r:<8.4f} {sigma_lam:<8.2f} {T:<6.1f} "
                f"{pi_ic_ours:<15.4f} {pi_ic_paper:<15.3f} {diff_ic:<12.4f} "
                f"{pi_tw_ours:<15.4f} {pi_tw_paper:<15.3f} {diff_tw:<12.4f}"
            )
        else:
            lines.append(
                f"{kappa_r:<8.4f} {sigma_r:<8.4f} {sigma_lam:<8.2f} {T:<6.1f} "
                f"{pi_ic_ours:<15.4f} {'N/A':<15} {'N/A':<12} "
                f"{pi_tw_ours:<15.4f} {'N/A':<15} {'N/A':<12}"
            )
    
    lines.append("")
    return "\n".join(lines)


def format_comparison_table_3(df: pd.DataFrame) -> str:
    """Format Table 3 with comparison to paper values."""
    lines = []
    lines.append("=" * 120)
    lines.append("TABLE 3: Effect of κ_r, σ_r, σ_λ for γ = -2")
    lines.append("=" * 120)
    lines.append("")
    lines.append(f"{'κ_r':<8} {'σ_r':<8} {'σ_λ':<8} {'T':<6} {'π_IC (Ours)':<15} {'π_IC (Paper)':<15} {'Diff IC':<12} "
                 f"{'π_TW (Ours)':<15} {'π_TW (Paper)':<15} {'Diff TW':<12}")
    lines.append("-" * 120)
    
    for _, row in df.iterrows():
        kappa_r = row['κ_r']
        sigma_r = row['σ_r']
        sigma_lam = row['σ_λ']
        T = row['T']
        pi_ic_ours = row['π_IC']
        pi_tw_ours = row['π_TW']
        
        key = (kappa_r, sigma_r, sigma_lam, T)
        if key in TABLE_3_PAPER:
            pi_ic_paper, pi_tw_paper = TABLE_3_PAPER[key]
            diff_ic = pi_ic_ours - pi_ic_paper
            diff_tw = pi_tw_ours - pi_tw_paper
            lines.append(
                f"{kappa_r:<8.4f} {sigma_r:<8.4f} {sigma_lam:<8.2f} {T:<6.1f} "
                f"{pi_ic_ours:<15.4f} {pi_ic_paper:<15.3f} {diff_ic:<12.4f} "
                f"{pi_tw_ours:<15.4f} {pi_tw_paper:<15.3f} {diff_tw:<12.4f}"
            )
        else:
            lines.append(
                f"{kappa_r:<8.4f} {sigma_r:<8.4f} {sigma_lam:<8.2f} {T:<6.1f} "
                f"{pi_ic_ours:<15.4f} {'N/A':<15} {'N/A':<12} "
                f"{pi_tw_ours:<15.4f} {'N/A':<15} {'N/A':<12}"
            )
    
    lines.append("")
    return "\n".join(lines)


def format_comparison_table_4(df: pd.DataFrame) -> str:
    """Format Table 4 with comparison to paper values."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 4: Non-affine Models (v_r, v_λ) for γ = -1")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'v_r':<8} {'v_λ':<8} {'T':<6} {'π_IC (Ours)':<15} {'π_IC (Paper)':<15} {'Diff IC':<12} "
                 f"{'π_TW (Ours)':<15} {'π_TW (Paper)':<15} {'Diff TW':<12}")
    lines.append("-" * 100)
    
    for _, row in df.iterrows():
        v_r = row['v_r']
        v_lam = row['v_λ']
        T = row['T']
        pi_ic_ours = row['π_IC']
        pi_tw_ours = row['π_TW']
        
        key = (v_r, v_lam, T)
        if key in TABLE_4_PAPER:
            pi_ic_paper, pi_tw_paper = TABLE_4_PAPER[key]
            diff_ic = pi_ic_ours - pi_ic_paper
            diff_tw = pi_tw_ours - pi_tw_paper
            lines.append(
                f"{v_r:<8.2f} {v_lam:<8.2f} {T:<6.1f} "
                f"{pi_ic_ours:<15.4f} {pi_ic_paper:<15.3f} {diff_ic:<12.4f} "
                f"{pi_tw_ours:<15.4f} {pi_tw_paper:<15.3f} {diff_tw:<12.4f}"
            )
        else:
            lines.append(
                f"{v_r:<8.2f} {v_lam:<8.2f} {T:<6.1f} "
                f"{pi_ic_ours:<15.4f} {'N/A':<15} {'N/A':<12} "
                f"{pi_tw_ours:<15.4f} {'N/A':<15} {'N/A':<12}"
            )
    
    lines.append("")
    return "\n".join(lines)


def format_comparison_table_5(df: pd.DataFrame) -> str:
    """Format Table 5 with comparison to paper values."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 5: Non-affine Models (v_r, v_λ) for γ = -2")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'v_r':<8} {'v_λ':<8} {'T':<6} {'π_IC (Ours)':<15} {'π_IC (Paper)':<15} {'Diff IC':<12} "
                 f"{'π_TW (Ours)':<15} {'π_TW (Paper)':<15} {'Diff TW':<12}")
    lines.append("-" * 100)
    
    for _, row in df.iterrows():
        v_r = row['v_r']
        v_lam = row['v_λ']
        T = row['T']
        pi_ic_ours = row['π_IC']
        pi_tw_ours = row['π_TW']
        
        key = (v_r, v_lam, T)
        if key in TABLE_5_PAPER:
            pi_ic_paper, pi_tw_paper = TABLE_5_PAPER[key]
            diff_ic = pi_ic_ours - pi_ic_paper
            diff_tw = pi_tw_ours - pi_tw_paper
            lines.append(
                f"{v_r:<8.2f} {v_lam:<8.2f} {T:<6.1f} "
                f"{pi_ic_ours:<15.4f} {pi_ic_paper:<15.3f} {diff_ic:<12.4f} "
                f"{pi_tw_ours:<15.4f} {pi_tw_paper:<15.3f} {diff_tw:<12.4f}"
            )
        else:
            lines.append(
                f"{v_r:<8.2f} {v_lam:<8.2f} {T:<6.1f} "
                f"{pi_ic_ours:<15.4f} {'N/A':<15} {'N/A':<12} "
                f"{pi_tw_ours:<15.4f} {'N/A':<15} {'N/A':<12}"
            )
    
    lines.append("")
    return "\n".join(lines)


def format_comparison_table_6(df: pd.DataFrame) -> str:
    """Format Table 6 with comparison to paper values."""
    lines = []
    lines.append("=" * 100)
    lines.append("TABLE 6: Two-Stock Example")
    lines.append("=" * 100)
    lines.append("")
    lines.append(f"{'γ':<8} {'λ₂':<8} {'π₁ (Ours)':<15} {'π₁ (Paper)':<15} {'Diff π₁':<12} "
                 f"{'π₂ (Ours)':<15} {'π₂ (Paper)':<15} {'Diff π₂':<12}")
    lines.append("-" * 100)
    
    for _, row in df.iterrows():
        gamma = row['γ']
        lam2 = row['λ₂']
        pi1_ours = row['π₁']
        pi2_ours = row['π₂']
        
        key = (gamma, lam2)
        if key in TABLE_6_PAPER:
            pi1_paper, pi2_paper = TABLE_6_PAPER[key]
            diff_pi1 = pi1_ours - pi1_paper
            diff_pi2 = pi2_ours - pi2_paper
            lines.append(
                f"{gamma:<8.1f} {lam2:<8.2f} "
                f"{pi1_ours:<15.4f} {pi1_paper:<15.3f} {diff_pi1:<12.4f} "
                f"{pi2_ours:<15.4f} {pi2_paper:<15.3f} {diff_pi2:<12.4f}"
            )
        else:
            lines.append(
                f"{gamma:<8.1f} {lam2:<8.2f} "
                f"{pi1_ours:<15.4f} {'N/A':<15} {'N/A':<12} "
                f"{pi2_ours:<15.4f} {'N/A':<15} {'N/A':<12}"
            )
    
    lines.append("")
    return "\n".join(lines)


def write_single_table_report(
    table_num: int,
    df: pd.DataFrame,
    output_dir: str = "results",
) -> None:
    """Write a comparison report for a single table.
    
    Args:
        table_num: Table number (1-6)
        df: DataFrame with results
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    output_file = os.path.join(output_dir, f"table_{table_num}_comparison.txt")
    
    formatters = {
        1: format_comparison_table_1,
        2: format_comparison_table_2,
        3: format_comparison_table_3,
        4: format_comparison_table_4,
        5: format_comparison_table_5,
        6: format_comparison_table_6,
    }
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write(f"TABLE {table_num} - COMPARISON REPORT\n")
        f.write("Cvitanić, Goukasian & Zapatero (2003)\n")
        f.write("=" * 100 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("Monte Carlo Parameters (Fast Settings):\n")
        f.write("  - K (outer paths): 300\n")
        f.write("  - M (inner paths): 5\n")
        f.write("  - Time steps per year: 50\n")
        f.write("  - Budget solver paths: 200\n")
        f.write("\n")
        f.write("=" * 100 + "\n\n")
        
        if table_num in formatters:
            f.write(formatters[table_num](df))
        
        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    print(f"Table {table_num} comparison written to: {output_file}")


def write_comparison_report(
    results: Dict[int, pd.DataFrame],
    output_file: str = None,
    output_dir: str = "results",
) -> None:
    """Write a comprehensive comparison report to file.
    
    Args:
        results: Dictionary mapping table number to DataFrame
        output_file: Output file path (default: results/all_tables_comparison.txt)
        output_dir: Output directory
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    if output_file is None:
        output_file = os.path.join(output_dir, "all_tables_comparison.txt")
    
    with open(output_file, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("MONTE CARLO PORTFOLIO OPTIMIZATION - RESULTS COMPARISON\n")
        f.write("Cvitanić, Goukasian & Zapatero (2003)\n")
        f.write("=" * 100 + "\n")
        f.write(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("\n")
        f.write("Monte Carlo Parameters (Fast Settings):\n")
        f.write("  - K (outer paths): 300\n")
        f.write("  - M (inner paths): 5\n")
        f.write("  - Time steps per year: 50\n")
        f.write("  - Budget solver paths: 200\n")
        f.write("\n")
        f.write("=" * 100 + "\n\n")
        
        # Format each table
        formatters = {
            1: format_comparison_table_1,
            2: format_comparison_table_2,
            3: format_comparison_table_3,
            4: format_comparison_table_4,
            5: format_comparison_table_5,
            6: format_comparison_table_6,
        }
        
        for table_num in sorted(results.keys()):
            if table_num in formatters:
                f.write(formatters[table_num](results[table_num]))
                f.write("\n")
        
        f.write("=" * 100 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 100 + "\n")
    
    print(f"\nComparison report written to: {output_file}")

