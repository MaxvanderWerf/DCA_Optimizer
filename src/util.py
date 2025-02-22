"""
util.py

This module provides utility functions used across the DCA optimizer project.
It includes functions for:
- Data validation and cleaning
- Price and investment amount validation
- Technical indicator calculations (RSI, Moving Averages)
- Date handling and validation
- Performance metric calculations
- Error handling and logging utilities

These utilities ensure consistent data handling and validation across
all modules in the project.
"""

import pandas as pd
import numpy as np
from typing import Union, Optional, Tuple
from datetime import datetime


def validate_price(price: Union[float, int, pd.Series]) -> Tuple[bool, Optional[float]]:
    """
    Validates a price value, handling NaN, negative, and zero values.
    
    Args:
        price: The price value to validate
        
    Returns:
        Tuple of (is_valid, cleaned_price)
    """
    # Handle pandas Series
    if isinstance(price, pd.Series):
        if price.empty or price.isna().all() or (price <= 0).all():
            return False, None
        # Get the last valid price
        price = price.iloc[-1]
    
    # Handle single value
    if pd.isna(price) or price <= 0:
        return False, None
    return True, float(price)


def calculate_shares_to_buy(cash_balance: float, price: float) -> int:
    """
    Calculates the number of shares that can be bought with a given cash balance.
    Handles edge cases and ensures integer number of shares.
    
    Args:
        cash_balance: Available cash to invest
        price: Price per share
        
    Returns:
        Number of shares that can be bought
    """
    is_valid, cleaned_price = validate_price(price)
    if not is_valid or cash_balance <= 0:
        return 0
    return int(cash_balance // cleaned_price)


def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calculates the Relative Strength Index (RSI) for a price series.
    Handles missing values and edge cases.
    
    Args:
        prices: Series of price values
        window: RSI calculation window
        
    Returns:
        Series containing RSI values
    """
    if len(prices) < window:
        return pd.Series(index=prices.index)
        
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    # Handle division by zero
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def calculate_moving_average(prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Calculates moving average with proper handling of missing values.
    
    Args:
        prices: Series of price values
        window: Moving average window
        
    Returns:
        Series containing moving average values
    """
    if len(prices) < window:
        return pd.Series(index=prices.index)
    
    return prices.rolling(window=window, min_periods=1).mean()


def validate_investment_amount(amount: float) -> float:
    """
    Validates and cleans investment amount input.
    
    Args:
        amount: The investment amount to validate
        
    Returns:
        Cleaned investment amount
        
    Raises:
        ValueError: If amount is negative or not a number
    """
    if pd.isna(amount) or amount < 0:
        raise ValueError("Investment amount must be a positive number")
    return float(amount)


def calculate_returns(initial_value: float, final_value: float) -> float:
    """
    Calculates returns while handling edge cases.
    
    Args:
        initial_value: Starting value
        final_value: Ending value
        
    Returns:
        Return as a decimal (e.g., 0.10 for 10% return)
    """
    if initial_value <= 0 or pd.isna(initial_value) or pd.isna(final_value):
        return 0.0
    return (final_value - initial_value) / initial_value


def get_date_info(data: pd.DataFrame) -> dict:
    """Gets information about the date range of available data."""
    current_date = datetime.now().date()
    data_start = data.index[0].date()
    data_end = min(data.index[-1].date(), current_date)
    
    # Print detailed info to terminal
    print(f"\nData range: {data_start} to {data_end}")
    print(f"Total days: {len(data)}")
    print(f"Years covered: {(data_end - data_start).days / 365:.1f}")
    
    return {
        'data_start': data_start,
        'data_end': data_end
    }


def validate_dates(start_date: datetime, end_date: datetime, data: pd.DataFrame) -> tuple[datetime, datetime, list]:
    """
    Validates and adjusts input dates against current date and available data range.
    """
    current_date = datetime.now().date()
    messages = []
    
    # Get actual data range (never beyond current date)
    data_start = data.index[0].date()
    data_end = min(data.index[-1].date(), current_date)
    
    # Basic validation
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")
        
    if end_date > current_date:
        messages.append(f"Cannot simulate into the future. Using current date ({current_date.strftime('%Y-%m-%d')}) as end date.")
        end_date = current_date
    
    # Validate against available data
    if start_date < data_start:
        messages.append(f"Data only available from {data_start.strftime('%Y-%m-%d')}. Adjusting start date.")
        start_date = data_start
        
    if end_date > data_end:
        messages.append(f"Using latest available data point ({data_end.strftime('%Y-%m-%d')}).")
        end_date = data_end
        
    return start_date, end_date, messages 