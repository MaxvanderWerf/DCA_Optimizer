"""
performance_comparison.py

This module handles the analysis and comparison of different DCA strategy performances.
It calculates various performance metrics including:
- Total and annualized returns
- Investment utilization rates
- Market timing effectiveness
- Risk-adjusted returns
- Cash flow analysis

The module provides both detailed metrics for each strategy and comparative
analysis across strategies to help identify the most effective approaches
for different market conditions.
"""

import pandas as pd
import numpy as np
from typing import Dict


def compare_performances(simulation_results: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Compares the performance of different DCA strategies.

    Args:
        simulation_results (Dict[str, pd.DataFrame]): Dictionary of strategy names and their respective performance DataFrames

    Returns:
        pd.DataFrame: DataFrame containing performance metrics for each strategy
    """
    print("Comparing strategy performances...")

    performance_metrics = {}

    for strategy, result in simulation_results.items():
        metrics = calculate_metrics(result)
        performance_metrics[strategy] = metrics

    return pd.DataFrame(performance_metrics).T


def calculate_metrics(data: pd.DataFrame) -> Dict[str, float]:
    """
    Calculates various performance metrics for a strategy.

    Args:
        data (pd.DataFrame): DataFrame containing the strategy's performance data.

    Returns:
        Dict[str, float]: Dictionary of calculated performance metrics.
    """
    total_invested = data['Invested'].sum()
    final_value = data['Portfolio_Value'].iloc[-1]
    initial_value = data['Portfolio_Value'].iloc[0]
    
    # Calculate years properly
    days = (data.index[-1] - data.index[0]).days
    years = days / 365.25
    
    # Total return calculation (based on money invested)
    total_return = (final_value - total_invested) / total_invested
    
    # Calculate IRR (annualized return)
    # Create cash flow series: negative for investments, positive for final value
    cash_flows = [-initial_value]  # Initial investment
    dates = [data.index[0]]
    
    # Add all periodic investments
    investments = data[data['Invested'] > 0][['Invested']]
    cash_flows.extend([-inv for inv in investments['Invested']])
    dates.extend(investments.index)
    
    # Add final portfolio value
    cash_flows.append(final_value)
    dates.append(data.index[-1])
    
    # Convert dates to years from start
    years_from_start = [(date - dates[0]).days / 365.25 for date in dates]
    
    try:
        # Calculate IRR using numpy's financial module
        annualized_return = np.irr(cash_flows)
    except:
        # Fallback to simpler calculation if IRR fails to converge
        annualized_return = total_return / years
    
    total_percent_gain = (final_value - total_invested) / total_invested * 100
    total_gain_euros = final_value - total_invested

    print("\nDetailed Performance Metrics:")
    print(f"Initial Value: ${initial_value:.2f}")
    print(f"Final Value: ${final_value:.2f}")
    print(f"Total Invested: ${total_invested:.2f}")
    print(f"Time period: {years:.1f} years")
    print(f"Total Return: {total_return:.2%}")
    print(f"Annualized Return (IRR): {annualized_return:.2%}")
    print(f"Total Gain: ${total_gain_euros:.2f}")
    print(f"Total Percent Gain: {total_percent_gain:.2f}%")
    print(f"Average Money Invested: ${data['Invested'].cumsum().mean():.2f}")
    print(f"Average Cash Balance: ${data['Cash_Balance'].mean():.2f}")
    print(f"Final Cash Balance: ${data['Cash_Balance'].iloc[-1]:.2f}")

    # Add new time in market metrics
    avg_investment_utilization = (data['Portfolio_Value'] - data['Cash_Balance']).mean() / data['Portfolio_Value'].mean()
    time_invested_ratio = len(data[data['Shares_Owned'] > 0]) / len(data)
    
    # Add market timing effectiveness metrics
    avg_purchase_price = (data['Invested'].sum() / data['Shares_Bought'].sum()) if data['Shares_Bought'].sum() > 0 else 0
    market_avg_price = data['Price'].mean()
    timing_effectiveness = (market_avg_price - avg_purchase_price) / market_avg_price

    # Add Time in Market vs Timing Analysis section in calculate_metrics function
    
    # 1. Calculate percentage of time fully invested
    time_invested_pct = len(data[data['Shares_Owned'] > 0]) / len(data) * 100

    # 2. Calculate average purchase price vs market average - we already have this
    price_efficiency = timing_effectiveness * 100  # Convert to percentage

    # 3. Calculate market participation rate
    # Check if Daily_Return exists, if not, calculate it
    if 'Daily_Return' not in data.columns:
        # Calculate daily returns based on Price
        data['Daily_Return'] = data['Price'].pct_change()
    
    up_days = data[data['Daily_Return'] > 0]
    invested_up_days = up_days[up_days['Shares_Owned'] > 0]
    market_participation = len(invested_up_days) / len(up_days) * 100 if len(up_days) > 0 else 0

    return {
        'Total Return': total_return,
        'Annualized Return': annualized_return,
        'Total Percent Gain': total_percent_gain,
        'Total Gain (€)': total_gain_euros,
        'Avg. Money Invested': data['Invested'].cumsum().mean(),
        'Avg. Money in Cash': data['Cash_Balance'].mean(),
        'Final Portfolio Value': final_value,
        'Total Invested': total_invested,
        'Final Cash Balance': data['Cash_Balance'].iloc[-1],
        'Investment Utilization': avg_investment_utilization,
        'Time Invested Ratio': time_invested_ratio,
        'Avg Purchase Price': avg_purchase_price,
        'Market Timing Score': timing_effectiveness,
        'Time Invested (%)': time_invested_pct,
        'Price Efficiency (%)': price_efficiency,
        'Market Participation (%)': market_participation
    }