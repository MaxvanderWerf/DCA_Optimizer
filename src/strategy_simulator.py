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


def simulate_dca_strategies(data: pd.DataFrame, strategies: List[str],
                            investment_amount: float = 1000,
                            investment_frequency: str = 'BMS',
                            initial_investment: float = 0,
                            logger: Optional[InvestmentLogger] = None) -> Dict[str, pd.DataFrame]:
    """
    Simulates various DCA strategies on the given ETF data.

    Args:
        data (pd.DataFrame): Historical ETF price data
        strategies (List[str]): List of strategies to simulate
        investment_amount (float): Amount to invest at each interval
        investment_frequency (str): Frequency of investments ('D' for daily, 'W' for weekly, 'ME' for monthly end)

    Returns:
        Dict[str, pd.DataFrame]: Dictionary of strategy names and their respective performance DataFrames
    """
    timeframe = f"{(data.index[-1] - data.index[0]).days // 365} years"
    print(f"Simulating DCA strategies for {timeframe}: {strategies}")
    print(f"Initial investment: ${initial_investment:.2f}")
    print(f"Investment amount: ${investment_amount:.2f}")
    print(f"Investment frequency: {investment_frequency}")
    results = {}

    for strategy in strategies:
        if strategy == "standard":
            results[strategy] = simulate_standard_dca(data, investment_amount, investment_frequency, initial_investment, logger)
        elif strategy == "rsi":
            results[strategy] = simulate_rsi_strategy(data, investment_amount, investment_frequency, initial_investment)
        elif strategy == "mean_reversion":
            results[strategy] = simulate_mean_reversion_strategy(data, investment_amount, investment_frequency, initial_investment, logger)
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


def simulate_rsi_strategy(data: pd.DataFrame, investment_amount: float, investment_frequency: str,
                          initial_investment: float) -> pd.DataFrame:
    """
    Simulates an RSI-based DCA strategy that follows strict monthly investment limits.
    
    The strategy:
    - Never invests more than the monthly investment amount
    - During oversold conditions (RSI < 30), invests the full monthly amount
    - During overbought conditions (RSI > 70), invests half the amount and saves the rest
    - During normal conditions (30 <= RSI <= 70), invests the standard amount
    """
    # Ensure we use the full date range
    data_copy = data.copy()
    start_date = data.index[0]
    end_date = data.index[-1]
    data_copy = data_copy.loc[start_date:end_date]
    
    data_copy['RSI'] = calculate_rsi(data_copy['Price'])

    cash_balance = initial_investment
    shares_owned = 0
    last_valid_price = None
    transactions = []

    for date, monthly_data in data_copy.resample(investment_frequency):
        cash_balance += investment_amount
        price = monthly_data['Price'].iloc[-1]
        rsi = monthly_data['RSI'].iloc[-1]

        is_valid, cleaned_price = validate_price(price)
        if not is_valid:
            transactions.append(record_no_transaction(date, price, shares_owned, cash_balance, last_valid_price))
            continue

        last_valid_price = cleaned_price  # Update last valid price
        # Determine investment amount based on RSI
        if pd.isna(rsi):
            invest_amount = investment_amount
        elif rsi < 30:  # Oversold
            invest_amount = cash_balance
        elif rsi > 70:  # Overbought
            invest_amount = cash_balance / 2
        else:  # Normal conditions
            invest_amount = investment_amount

        # Use utility function to calculate shares to buy
        shares_to_buy = calculate_shares_to_buy(invest_amount, cleaned_price)

        if shares_to_buy > 0:
            cost = shares_to_buy * cleaned_price
            cash_balance -= cost
            shares_owned += shares_to_buy
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


def simulate_mean_reversion_strategy(data: pd.DataFrame, investment_amount: float,
                                     investment_frequency: str, initial_investment: float,
                                     logger: Optional[InvestmentLogger] = None) -> pd.DataFrame:
    """
    Simulates a Mean Reversion strategy where the investment amount varies based on the deviation 
    from the 20-day moving average.
    """
    # Ensure we use the full date range
    data_copy = data.copy()
    start_date = data.index[0]
    end_date = data.index[-1]
    data_copy = data_copy.loc[start_date:end_date]
    
    data_copy['MA20'] = data_copy['Price'].rolling(window=20).mean()

    cash_balance = initial_investment
    shares_owned = 0
    last_valid_price = None
    transactions = []

    for date, monthly_data in data_copy.resample(investment_frequency):
        cash_balance += investment_amount
        price = monthly_data['Price'].iloc[-1]
        ma20 = monthly_data['MA20'].iloc[-1]

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
                    reason="Invalid price data for this date"
                )
            continue

        deviation = (cleaned_price - ma20) / ma20 if not pd.isna(ma20) else 0
        invest_amount = cash_balance * (1 - deviation)
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
                    reason=f"Price deviation from MA20: {deviation:.2%}. " +
                          f"Investing more due to price being {'below' if deviation < 0 else 'above'} MA20",
                    metrics={"MA20": ma20, "Deviation": deviation}
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
