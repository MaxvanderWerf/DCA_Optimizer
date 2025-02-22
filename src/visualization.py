"""
visualization.py

This module contains functions for generating, saving, and returning visualizations of ETF/Stock DCA simulation results.
It provides functionality to plot portfolio values, total returns, and investment decisions over time.
The functions in this module save the figures to files and return them for use in a Streamlit application.

Functions:
- get_visualization_dir: Get the path to the visualization directory.
- generate_filename: Generate a filename for the visualization.
- plot_portfolio_value: Plot the portfolio value over time for each strategy.
- plot_total_return: Plot the total return for each strategy.
- plot_investment_decisions: Plot investment decisions over time for each strategy.
- plot_results: Generate and save all visualizations for the simulation results.
"""

import os
from pathlib import Path
from datetime import datetime
from typing import Dict
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


def get_visualization_dir() -> Path:
    """
    Get the path to the visualization directory.

    Returns:
        Path: The path to the visualization directory.
    """
    base_dir = Path(__file__).resolve().parent.parent
    viz_dir = base_dir / 'visualizations'
    viz_dir.mkdir(exist_ok=True)
    return viz_dir


def generate_filename(prefix: str, identifier: str) -> str:
    """
    Generate a filename for the visualization.

    Args:
        prefix (str): The prefix for the filename (e.g., 'portfolio_value' or 'total_return').
        identifier (str): The identifier for the specific visualization (e.g., 'SPY_15y').

    Returns:
        str: The generated filename.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    return f'{prefix}_{identifier}_{current_date}.png'


def plot_portfolio_value(simulation_results: Dict[str, pd.DataFrame], identifier: str) -> plt.Figure:
    """
    Plot the portfolio value over time for each strategy.

    Args:
        simulation_results (Dict[str, pd.DataFrame]): The results of the simulation for each strategy.
        identifier (str): The identifier for the specific visualization.

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    for strategy, result in simulation_results.items():
        ax.plot(result.index, result['Portfolio_Value'], label=strategy)
    ax.set_title(f'Portfolio Value Over Time - {identifier}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Portfolio Value ($)')
    ax.legend()

    viz_dir = get_visualization_dir()
    filename = generate_filename('portfolio_value', identifier)
    filepath = viz_dir / filename
    plt.savefig(filepath)
    print(f"Portfolio value visualization saved as {filepath}")

    return fig


def plot_total_return(performance_metrics: pd.DataFrame, identifier: str) -> plt.Figure:
    """
    Plot the total return for each strategy.

    Args:
        performance_metrics (pd.DataFrame): The performance metrics for each strategy.
        identifier (str): The identifier for the specific visualization.

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    performance_metrics['Total Return'].plot(kind='bar', ax=ax)
    ax.set_title(f'Total Return by Strategy - {identifier}')
    ax.set_xlabel('Strategy')
    ax.set_ylabel('Total Return (%)')

    viz_dir = get_visualization_dir()
    filename = generate_filename('total_return', identifier)
    filepath = viz_dir / filename
    plt.savefig(filepath)
    print(f"Total return visualization saved as {filepath}")

    return fig


def plot_investment_decisions(simulation_results: Dict[str, pd.DataFrame], identifier: str) -> plt.Figure:
    """
    Plot investment decisions over time for each strategy.

    Args:
        simulation_results (Dict[str, pd.DataFrame]): The results of the simulation for each strategy.
        identifier (str): The identifier for the specific visualization.

    Returns:
        plt.Figure: The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(15, 10))

    for strategy, result in simulation_results.items():
        ax.plot(result.index, result['Invested'], label=f'{strategy} - Invested', marker='o')
        ax.plot(result.index, result['Cash_Balance'], label=f'{strategy} - Cash Balance', linestyle='--')

    ax.set_title(f'Investment Decisions Over Time - {identifier}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Amount ($)')
    ax.legend()

    # Improve x-axis date formatting
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    fig.autofmt_xdate()  # Rotate and align the tick labels

    viz_dir = get_visualization_dir()
    filename = generate_filename('investment_decisions', identifier)
    filepath = viz_dir / filename
    plt.savefig(filepath)
    print(f"Investment decisions visualization saved as {filepath}")

    return fig


def plot_results(simulation_results: Dict[str, pd.DataFrame], performance_metrics: pd.DataFrame,
                 identifier: str) -> Dict[str, plt.Figure]:
    """
    Generate and save visualizations for the simulation results.

    Args:
        simulation_results (Dict[str, pd.DataFrame]): The results of the simulation for each strategy.
        performance_metrics (pd.DataFrame): The performance metrics for each strategy.
        identifier (str): The identifier for the specific visualization.

    Returns:
        Dict[str, plt.Figure]: A dictionary containing the generated figures.
    """
    print(f"Generating visualizations for {identifier}...")

    figures = {}
    figures['portfolio_value'] = plot_portfolio_value(simulation_results, identifier)
    figures['total_return'] = plot_total_return(performance_metrics, identifier)
    figures['investment_decisions'] = plot_investment_decisions(simulation_results, identifier)

    print(f"Visualizations saved in {get_visualization_dir()} folder for {identifier}.")
    return figures