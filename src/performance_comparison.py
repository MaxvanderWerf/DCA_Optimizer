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


def count_contribution_opportunities(start_date, end_date, freq='BMS'):
    """
    Count the number of contribution opportunities in a date range based on a frequency rule.
    
    Args:
        start_date: The start date of the simulation
        end_date: The end date of the simulation
        freq: The frequency rule for contributions (default: 'BMS' - business month start)
        
    Returns:
        int: The number of contribution opportunities (excluding initial investment)
    """
    # Convert to timestamps if they're not already
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    
    # Generate dates according to the frequency rule
    # Use inclusive='both' to ensure we include both start and end dates if they match the frequency
    contribution_dates = pd.date_range(start=start_ts, end=end_ts, freq=freq, inclusive='both')
    
    # Print for debugging
    print(f"DEBUG - Raw contribution dates: {[d.strftime('%Y-%m-%d') for d in contribution_dates]}")
    
    # If we're in the same month, there are no additional contribution opportunities
    if start_ts.year == end_ts.year and start_ts.month == end_ts.month:
        print("DEBUG - Start and end dates are in the same month, no additional contributions")
        return 0
    
    # Count all months in the range (excluding the initial investment month)
    all_months = set()
    
    # Start from the month after the initial investment
    current_month = (start_ts.year, start_ts.month)
    end_month = (end_ts.year, end_ts.month)
    
    # Add all months in the range (excluding the initial investment month)
    while current_month <= end_month:
        year, month = current_month
        # Skip the initial investment month
        if (year, month) != (start_ts.year, start_ts.month):
            all_months.add((year, month))
        
        # Move to next month
        month += 1
        if month > 12:
            year += 1
            month = 1
        current_month = (year, month)
    
    print(f"DEBUG - All months in range (excluding initial): {all_months}")
    
    # Check if we have any business month start dates in the range
    if len(contribution_dates) == 0:
        print("DEBUG - No business month start dates found in range")
        return 0
    
    # If the first date is the same as start_date (same day), it's the initial investment
    # We need to exclude it from the count
    first_date_is_initial = False
    if len(contribution_dates) > 0:
        first_date_is_initial = contribution_dates[0].normalize() == start_ts.normalize()
        if first_date_is_initial:
            print(f"DEBUG - First date {contribution_dates[0].strftime('%Y-%m-%d')} is initial investment date")
    
    # Count all unique months that have a business month start date
    # This ensures we count each month that has started, even if it's not complete
    contribution_months = set()
    for date in contribution_dates:
        # Skip the initial investment date
        if first_date_is_initial and date == contribution_dates[0]:
            continue
        contribution_months.add((date.year, date.month))
    
    print(f"DEBUG - Contribution months: {contribution_months}")
    
    # Return the number of unique months with contribution opportunities
    return len(contribution_months)


def compare_performances(simulation_results: Dict[str, pd.DataFrame], 
                         initial_investment: float = 10000, 
                         monthly_investment: float = 1000) -> pd.DataFrame:
    """
    Compares the performance of different DCA strategies.

    Args:
        simulation_results (Dict[str, pd.DataFrame]): Dictionary of strategy names and their respective performance DataFrames
        initial_investment (float): The initial investment amount from UI
        monthly_investment (float): The monthly investment amount from UI

    Returns:
        pd.DataFrame: DataFrame containing performance metrics for each strategy
    """
    print("Comparing strategy performances...")

    performance_metrics = {}

    # Get full date range from all strategies combined
    all_dates = pd.DatetimeIndex([])
    for result in simulation_results.values():
        if not result.empty:
            all_dates = all_dates.union(result.index)
    all_dates = all_dates.sort_values()
    
    # Calculate overall market average price for the entire period
    all_prices = pd.Series(dtype=float)
    for result in simulation_results.values():
        if not result.empty:
            all_prices = pd.concat([all_prices, result['Price']])
    market_avg_price = all_prices.mean() if not all_prices.empty else 0

    # Get investment parameters from the function arguments instead of hardcoding
    sample_strategy = next(iter(simulation_results.values()))
    
    if not sample_strategy.empty:
        # Use the parameters passed from the UI
        # Get start and end dates
        start_date = sample_strategy.index[0]
        end_date = sample_strategy.index[-1]
        
        # Count exact number of contribution opportunities
        contribution_count = count_contribution_opportunities(start_date, end_date, freq='BMS')
        
        # Calculate total available capital
        total_available_capital = initial_investment + (monthly_investment * contribution_count)
        
        print(f"CALCULATION WITH UI VALUES:")
        print(f"Initial investment: {initial_investment}")
        print(f"Monthly amount: {monthly_investment}")
        print(f"Start date: {start_date.strftime('%Y-%m-%d')}")
        print(f"End date: {end_date.strftime('%Y-%m-%d')}")
        print(f"Contribution opportunities: {contribution_count}")
        print(f"Total available capital: {total_available_capital}")
    else:
        total_available_capital = 0
        print("Warning: Could not calculate total available capital")

    for strategy, result in simulation_results.items():
        # Skip if result is empty
        if result.empty:
            continue
            
        # Extract data
        initial_date = result.index[0]
        final_date = result.index[-1]
        final_portfolio_value = result['Portfolio_Value'].iloc[-1]
        total_invested = result['Invested'].sum()
        
        # Add warning if total invested exceeds total available capital
        if total_invested > total_available_capital:
            print(f"\n⚠️ WARNING: For strategy '{strategy}', total invested (€{total_invested:.2f}) exceeds total available capital (€{total_available_capital:.2f})!")
            print(f"This suggests a calculation error or unexpected behavior in the simulation.")
            print(f"Please check the investment dates and amounts in the strategy simulation.")
        
        # Calculate traditional return (on invested capital)
        total_gain_on_invested = final_portfolio_value - total_invested
        total_return_on_invested = total_gain_on_invested / total_invested if total_invested > 0 else 0
        
        # Calculate gain and return on total available capital (the main metrics we'll display)
        total_gain = final_portfolio_value - total_available_capital
        total_return = total_gain / total_available_capital if total_available_capital > 0 else 0
        
        # Calculate time in market vs timing metrics
        # Time Invested: Percentage of potential investment days where money was actually in the market
        # This requires daily data to be accurate, but we'll work with what we have
        
        # 1. Calculate time invested as percentage of days with shares owned vs. total days
        # For strategies that always fully invest, this won't capture partial investments
        days_with_investment = result[result['Shares_Owned'] > 0]
        days_in_period = len(all_dates)
        
        # Calculate time invested percentage
        # Compare fully invested days, partial investments, and cash positions
        max_shares_owned = result['Shares_Owned'].max()
        full_investment_threshold = 0.9 * max_shares_owned  # Consider 90% of max as "fully invested"
        
        # Count days based on investment level
        fully_invested_days = len(result[result['Shares_Owned'] >= full_investment_threshold])
        partially_invested_days = len(result[(result['Shares_Owned'] > 0) & 
                                           (result['Shares_Owned'] < full_investment_threshold)])
        
        # Weight days by investment level (fully = 1, partially = 0.5)
        weighted_invested_days = fully_invested_days + (partially_invested_days * 0.5)
        
        # Calculate time invested percentage
        time_invested_pct = (weighted_invested_days / len(result)) * 100 if len(result) > 0 else 0
        
        # 2. Price Efficiency: How well the strategy buys at below-average prices
        buy_prices = []
        buy_amounts = []
        
        for i, row in result.iterrows():
            if row['Shares_Bought'] > 0:
                buy_prices.append(row['Price'])
                buy_amounts.append(row['Invested'])
        
        # Calculate weighted average purchase price
        if buy_amounts and sum(buy_amounts) > 0:
            weighted_avg_price = sum(p * a for p, a in zip(buy_prices, buy_amounts)) / sum(buy_amounts)
            # Compare to overall market average price, not just this strategy's average
            price_efficiency = ((market_avg_price - weighted_avg_price) / market_avg_price) * 100
        else:
            price_efficiency = 0
        
        # 3. Market Participation: Percentage of market up-days where the strategy was meaningfully invested
        # Create a resampled daily dataset to capture true daily returns
        daily_price = result['Price'].resample('D').last().fillna(method='ffill')
        daily_returns = daily_price.pct_change().dropna()
        daily_shares = result['Shares_Owned'].resample('D').last().fillna(method='ffill')
        
        # Calculate investment level relative to maximum investment
        investment_level = daily_shares / max_shares_owned if max_shares_owned > 0 else 0
        
        # Count up days and invested up days with weighting by investment level
        up_days = daily_returns[daily_returns > 0]
        market_participation = 0
        
        if not up_days.empty:
            up_day_participation = 0
            for day in up_days.index:
                if day in investment_level.index:
                    # Weight participation by investment level
                    up_day_participation += investment_level[day]
            
            market_participation = (up_day_participation / len(up_days)) * 100
        
        # Store metrics
        performance_metrics[strategy] = {
            'Total Return': total_return,  # Return on available capital
            'Return on Invested Capital': total_return_on_invested,  # Original "Total Return" renamed 
            'Annualized Return': calculate_annualized_return(total_return, initial_date, final_date),
            'Total Gain (€)': total_gain,  # Now shows gain against all available capital
            'Final Portfolio Value': final_portfolio_value,
            'Total Invested': total_invested,
            'Total Available Capital': total_available_capital,
            'Time Invested (%)': time_invested_pct,
            'Price Efficiency (%)': price_efficiency,
            'Market Participation (%)': market_participation
        }

    return pd.DataFrame(performance_metrics).T


def calculate_annualized_return(total_return, start_date, end_date):
    """Calculate the annualized return from total return and date range."""
    years = (end_date - start_date).days / 365
    if years > 0 and total_return > -1:
        return (1 + total_return) ** (1 / years) - 1
    return 0


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