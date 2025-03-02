"""
strategy_simulator.py

This module implements various Dollar-Cost Averaging (DCA) investment strategies:

1. Standard DCA:
   - Invests a fixed amount at regular intervals
   - Maintains consistent market exposure

2. RSI-based Strategy:
   - Adjusts investment amounts based on RSI indicators
   - Increases investment during oversold conditions
   - Reduces investment during overbought conditions

3. Mean Reversion Strategy:
   - Varies investment based on deviation from moving averages
   - Takes advantage of price movements away from historical trends
   - Aims to optimize entry points while maintaining regular investment schedule

Each strategy maintains detailed transaction records and can work with
the investment logger for transparency and analysis.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from util import validate_price, calculate_shares_to_buy, calculate_rsi
from investment_logger import InvestmentLogger


def simulate_dca_strategies(data: pd.DataFrame, 
                          strategies: List[str],
                          investment_amount: float = 1000,
                          investment_frequency: str = 'BMS',
                          initial_investment: float = 0,
                          logger: Optional[InvestmentLogger] = None,
                          strategy_params: Optional[Dict] = None) -> Dict[str, pd.DataFrame]:
    """
    Simulates various DCA strategies on the given ETF data.
    
    Args:
        data (pd.DataFrame): Historical ETF price data
        strategies (List[str]): List of strategies to simulate
        investment_amount (float): Amount to invest at each interval
        investment_frequency (str): Frequency of investments ('D' for daily, 'W' for weekly, 'ME' for monthly end)
        initial_investment (float): Initial investment amount
        logger (Optional[InvestmentLogger]): Investment logger for logging decisions
        strategy_params (Optional[Dict]): Dictionary containing strategy-specific parameters

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of strategy names and their respective performance DataFrames
    """
    timeframe = f"{(data.index[-1] - data.index[0]).days // 365} years"
    print(f"Simulating DCA strategies for {timeframe}: {strategies}")
    print(f"Initial investment: ${initial_investment:.2f}")
    print(f"Investment amount: ${investment_amount:.2f}")
    print(f"Investment frequency: {investment_frequency}")
    results = {}

    # Use default parameters if none provided
    if strategy_params is None:
        strategy_params = {}

    for strategy in strategies:
        if strategy == "standard":
            results[strategy] = simulate_standard_dca(data, investment_amount, investment_frequency, initial_investment, logger)
        elif strategy == "rsi":
            params = strategy_params.get('rsi', {'lower_bound': 30, 'upper_bound': 70})
            results[strategy] = simulate_rsi_strategy(
                data, 
                investment_amount, 
                investment_frequency, 
                initial_investment,
                lower_bound=params['lower_bound'],
                upper_bound=params['upper_bound']
            )
        elif strategy == "mean_reversion":
            params = strategy_params.get('mean_reversion', {'ma_window': 20})
            results[strategy] = simulate_mean_reversion_strategy(
                data, 
                investment_amount, 
                investment_frequency, 
                initial_investment,
                logger,
                ma_window=params['ma_window']
            )
        else:
            print(f"Strategy {strategy} not implemented")

    return results


def record_transaction(date, price, shares_bought, shares_owned, cash_balance) -> Dict:
    """
    Record a transaction in the simulation results.
    """
    portfolio_value = shares_owned * price + cash_balance  # Calculate current portfolio value
    
    transaction = {
        'Date': date,
        'Price': price,
        'Invested': shares_bought * price,
        'Shares_Bought': shares_bought,
        'Shares_Owned': shares_owned,
        'Cash_Balance': cash_balance,
        'Portfolio_Value': portfolio_value
    }
    print(f"Transaction recorded: Price=${price:.2f}, Shares={shares_bought}, "
          f"Portfolio=${portfolio_value:.2f}, Cash=${cash_balance:.2f}")
    return transaction


def record_no_transaction(date, price, shares_owned, cash_balance, last_valid_price=None) -> Dict:
    """
    Record a period where no shares are bought.
    """
    # Use last valid price for portfolio valuation if current price is NaN
    valuation_price = price if not pd.isna(price) else last_valid_price
    portfolio_value = shares_owned * valuation_price + cash_balance if valuation_price is not None else cash_balance
    
    transaction = {
        'Date': date,
        'Price': price,
        'Invested': 0,
        'Shares_Bought': 0,
        'Shares_Owned': shares_owned,
        'Cash_Balance': cash_balance,
        'Portfolio_Value': portfolio_value
    }
    print(f"No transaction: Portfolio=${portfolio_value:.2f}, Cash=${cash_balance:.2f}")
    return transaction


def simulate_standard_dca(data: pd.DataFrame, investment_amount: float, investment_frequency: str,
                          initial_investment: float,
                          logger: Optional[InvestmentLogger] = None) -> pd.DataFrame:
    """
    Simulates a standard Dollar-Cost Averaging (DCA) strategy.
    """
    print("\nSimulating standard DCA strategy")
    print(f"Starting with ${initial_investment:.2f}")

    cash_balance = initial_investment
    shares_owned = 0
    last_valid_price = None
    transactions = []

    # Resample data according to investment frequency and forward fill missing values
    resampled_data = data.resample(investment_frequency).last()
    resampled_data = resampled_data.fillna(method='ffill')  # Forward fill missing values
    
    for date, row in resampled_data.iterrows():
        cash_balance += investment_amount
        
        price = row['Price']
        
        # Skip if price is None (should not happen with ffill)
        if pd.isna(price):
            if logger:
                logger.log_decision(
                    date=date,
                    strategy="standard",
                    action="skip",
                    price=None,
                    shares=0,
                    cash_used=0,
                    cash_balance=cash_balance,
                    reason="No price data available"
                )
            continue
            
        # Always try to invest if we have a valid price
        shares_to_buy = calculate_shares_to_buy(cash_balance, price)
        
        if shares_to_buy > 0:
            cost = shares_to_buy * price
            cash_balance -= cost
            shares_owned += shares_to_buy
            
            if logger:
                logger.log_decision(
                    date=date,
                    strategy="standard",
                    action="buy",
                    price=price,
                    shares=shares_to_buy,
                    cash_used=cost,
                    cash_balance=cash_balance,
                    reason=f"Regular DCA purchase on {investment_frequency} schedule"
                )
            transactions.append(record_transaction(date, price, shares_to_buy, shares_owned, cash_balance))
        else:
            if logger:
                logger.log_decision(
                    date=date,
                    strategy="standard",
                    action="skip",
                    price=price,
                    shares=0,
                    cash_used=0,
                    cash_balance=cash_balance,
                    reason="Insufficient funds for purchase"
                )
            transactions.append(record_no_transaction(date, price, shares_owned, cash_balance))

    final_df = pd.DataFrame(transactions)
    if not final_df.empty:
        final_df.set_index('Date', inplace=True)
        final_df.index = pd.to_datetime(final_df.index)  # Ensure datetime index
        
    print(f"\nFinal Portfolio Status:")
    print(f"Shares owned: {shares_owned}")
    print(f"Cash balance: ${cash_balance:.2f}")
    if not final_df.empty:
        print(f"Final portfolio value: ${final_df['Portfolio_Value'].iloc[-1]:.2f}")
    
    return final_df


def calculate_rsi(prices, window=14):
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def simulate_rsi_strategy(data: pd.DataFrame, 
                         investment_amount: float, 
                         investment_frequency: str,
                         initial_investment: float,
                         lower_bound: int = 30,
                         upper_bound: int = 70,
                         rsi_window: int = 14,
                         oversold_scale: float = 2.0,
                         overbought_scale: float = 0.5,
                         mid_band_scale: float = 1.0) -> pd.DataFrame:
    """
    Simulates an RSI-based DCA strategy with configurable parameters.
    
    Args:
        rsi_window: Number of periods for RSI calculation
        oversold_scale: Multiplier for investment when RSI is below lower_bound
        overbought_scale: Multiplier for investment when RSI is above upper_bound
        mid_band_scale: Multiplier for investment when RSI is between bounds
    """
    data_copy = data.copy()
    data_copy['RSI'] = calculate_rsi(data_copy['Price'], window=rsi_window)

    cash_balance = initial_investment
    shares_owned = 0
    transactions = []
    last_valid_price = None  # Track the last valid price

    for date, monthly_data in data_copy.resample(investment_frequency):
        cash_balance += investment_amount
        price = monthly_data['Price'].iloc[-1]
        rsi = monthly_data['RSI'].iloc[-1]

        is_valid, cleaned_price = validate_price(price)
        
        # Update last valid price when we have a valid price
        if is_valid:
            last_valid_price = cleaned_price
            
        if not is_valid:
            # Pass the last valid price to record_no_transaction
            transactions.append(record_no_transaction(date, price, shares_owned, cash_balance, last_valid_price))
            continue

        if pd.isna(rsi):
            invest_amount = investment_amount
        elif rsi < lower_bound:  # Oversold
            invest_amount = investment_amount * oversold_scale
        elif rsi > upper_bound:  # Overbought
            invest_amount = investment_amount * overbought_scale
        else:  # Normal conditions
            invest_amount = investment_amount * mid_band_scale

        invest_amount = min(invest_amount, cash_balance)
        shares_to_buy = calculate_shares_to_buy(invest_amount, cleaned_price)

        if shares_to_buy > 0:
            cost = shares_to_buy * cleaned_price
            cash_balance -= cost
            shares_owned += shares_to_buy
            transactions.append(
                record_transaction(date, cleaned_price, shares_to_buy, shares_owned, cash_balance))
        else:
            transactions.append(record_no_transaction(date, cleaned_price, shares_owned, cash_balance, last_valid_price))

    final_df = pd.DataFrame(transactions)
    if not final_df.empty:
        final_df.set_index('Date', inplace=True)
        final_df.index = pd.to_datetime(final_df.index)  # Ensure datetime index
    
    print(f"\nFinal Portfolio Status:")
    print(f"Shares owned: {shares_owned}")
    print(f"Cash balance: ${cash_balance:.2f}")
    if not final_df.empty and last_valid_price is not None:
        # Calculate final portfolio value using the last valid price to ensure consistency
        final_portfolio_value = shares_owned * last_valid_price + cash_balance
        print(f"Final portfolio value: ${final_portfolio_value:.2f}")
    elif not final_df.empty:
        print(f"Final portfolio value: ${final_df['Portfolio_Value'].iloc[-1]:.2f}")
    
    return final_df


def simulate_mean_reversion_strategy(data: pd.DataFrame, 
                                   investment_amount: float,
                                   investment_frequency: str, 
                                   initial_investment: float,
                                   logger: Optional[InvestmentLogger] = None,
                                   ma_window: int = 20,
                                   ma_type: str = 'simple',
                                   deviation_threshold: float = 0.1,
                                   max_scale_up: float = 2.0,
                                   max_scale_down: float = 0.5) -> pd.DataFrame:
    """
    Simulates a Mean Reversion strategy with enhanced configuration options.
    
    Args:
        ma_type: Type of moving average ('simple', 'exponential', 'weighted')
        deviation_threshold: Threshold for considering price significantly deviated
        max_scale_up: Maximum multiplier for scaling up investment
        max_scale_down: Minimum multiplier for scaling down investment
    """
    data_copy = data.copy()
    
    if ma_type == 'exponential':
        data_copy[f'MA{ma_window}'] = data_copy['Price'].ewm(span=ma_window).mean()
    elif ma_type == 'weighted':
        weights = np.arange(1, ma_window + 1)
        data_copy[f'MA{ma_window}'] = data_copy['Price'].rolling(window=ma_window).apply(
            lambda x: np.dot(x, weights) / weights.sum())
    else:  # simple
        data_copy[f'MA{ma_window}'] = data_copy['Price'].rolling(window=ma_window).mean()

    cash_balance = initial_investment
    shares_owned = 0
    transactions = []

    for date, monthly_data in data_copy.resample(investment_frequency):
        cash_balance += investment_amount
        price = monthly_data['Price'].iloc[-1]
        ma = monthly_data[f'MA{ma_window}'].iloc[-1]

        is_valid, cleaned_price = validate_price(price)
        if not is_valid:
            if logger:
                logger.log_decision(
                    date=date,
                    strategy="mean_reversion",
                    action="skip",
                    price=None,
                    shares=0,
                    cash_used=0,
                    cash_balance=cash_balance,
                    reason="Invalid price data"
                )
            continue

        deviation = (cleaned_price - ma) / ma if not pd.isna(ma) else 0
        
        if abs(deviation) > deviation_threshold:
            if deviation < 0:  # Price below MA
                scale = 1 + min(abs(deviation), max_scale_up - 1)
            else:  # Price above MA
                scale = max(max_scale_down, 1 - abs(deviation))
        else:
            scale = 1.0

        invest_amount = investment_amount * scale
        invest_amount = max(0, min(invest_amount, cash_balance))

        shares_to_buy = calculate_shares_to_buy(invest_amount, cleaned_price)
        
        if shares_to_buy > 0:
            cost = shares_to_buy * cleaned_price
            cash_balance -= cost
            shares_owned += shares_to_buy
            
            if logger:
                logger.log_decision(
                    date=date,
                    strategy="mean_reversion",
                    action="buy",
                    price=cleaned_price,
                    shares=shares_to_buy,
                    cash_used=cost,
                    cash_balance=cash_balance,
                    reason=f"Price deviation from MA{ma_window}: {deviation:.2%}. " +
                          f"Investing more due to price being {'below' if deviation < 0 else 'above'} MA{ma_window}",
                    metrics={"MA": ma, "Deviation": deviation}
                )
            transactions.append(
                record_transaction(date, cleaned_price, shares_to_buy, shares_owned, cash_balance))
        else:
            transactions.append(record_no_transaction(date, cleaned_price, shares_owned, cash_balance))

    final_df = pd.DataFrame(transactions)
    if not final_df.empty:
        final_df.set_index('Date', inplace=True)
        final_df.index = pd.to_datetime(final_df.index)  # Ensure datetime index
    
    print(f"\nFinal Portfolio Status:")
    print(f"Shares owned: {shares_owned}")
    print(f"Cash balance: ${cash_balance:.2f}")
    if not final_df.empty:
        print(f"Final portfolio value: ${final_df['Portfolio_Value'].iloc[-1]:.2f}")
    
    return final_df
