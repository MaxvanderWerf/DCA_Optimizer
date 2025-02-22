"""
data_retrieval.py

This module handles the fetching and management of ETF/Stock price data. The data retrieval logic follows these steps:

1. Check Local Data:
   - Look for existing data file in the 'data' directory
   - If file exists, load it and check its date range

2. Determine Data Needs:
   - If no local data exists:
     * Fetch entire requested date range from Yahoo Finance
   - If local data exists:
     * If local data is up to date (extends to current date):
       -> Use existing data
     * If local data is outdated:
       -> Fetch only the missing data (from last available date to current date)
       -> Append to existing data

3. Handle Edge Cases:
   - If requested dates are in the future:
     * Warn user and limit to current date
   - If Yahoo Finance fetch fails:
     * Return existing data with warning
   - If no data is available:
     * Return empty DataFrame with error message

4. Data Processing:
   - Clean and validate price data
   - Calculate additional metrics (daily returns, volatility)
   - Save updated data to local file

The module ensures efficient data management by:
- Minimizing API calls to Yahoo Finance
- Maintaining a local cache of historical data
- Only fetching new data when necessary
- Providing clear feedback about data availability
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional


class DataValidationError(Exception):
    """Custom exception for data validation errors"""
    pass

def fetch_etf_data(symbol: str, 
                   start_date: Optional[datetime] = None, 
                   end_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Fetches historical ETF price data from Yahoo Finance or loads from a local file.
    """
    data_dir = 'data'
    
    # Ensure end_date is not in the future
    current_date = datetime.now()
    if end_date and end_date > current_date:
        print(f"Warning: End date {end_date} is in the future. Using current date instead.")
        end_date = current_date
    
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    filename = f"{data_dir}/{symbol}_data.csv"

    if os.path.exists(filename):
        try:
            existing_data = load_data_from_csv(filename)
            if existing_data is None:
                print(f"Invalid data found in {filename}, fetching fresh data...")
                return fetch_fresh_data(symbol, start_date, end_date, filename)
                
            # If we have data up to today, just return it
            if existing_data.index[-1].date() >= current_date.date():
                return process_data(existing_data.loc[:end_date])

            # Only fetch new data from last available date to today
            return update_existing_data(existing_data, symbol, end_date, filename)
    
        except (DataValidationError, pd.errors.EmptyDataError) as e:
            print(f"Error loading existing data: {e}")
            return fetch_fresh_data(symbol, start_date, end_date, filename)
    else:
        return fetch_fresh_data(symbol, start_date, end_date, filename)

def load_data_from_csv(filename: str) -> Optional[pd.DataFrame]:
    """
    Loads and validates data from CSV file.
    Returns None if data is invalid.
    """
    try:
        # Skip the metadata rows (Ticker row) and use the actual data
        data = pd.read_csv(filename, 
                          skiprows=[1, 2],  # Skip Ticker and empty Date rows
                          index_col=0,
                          parse_dates=True)
        
        # Clean up column names and data
        data.index.name = 'Date'
        
        # If we don't have Adj Close, use regular Close
        if 'Adj Close' not in data.columns and 'Close' in data.columns:
            data['Adj Close'] = data['Close']
            
        # Validate required columns exist
        required_columns = ['Close', 'Volume']  # Remove Adj Close from required
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise DataValidationError(f"Missing required columns: {missing_columns}")
            
        # Validate index is properly parsed
        if data.index.isnull().any():
            raise DataValidationError("Invalid dates found in data")
            
        return data
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def fetch_fresh_data(symbol: str, 
                    start_date: Optional[datetime], 
                    end_date: Optional[datetime],
                    filename: str) -> pd.DataFrame:
    """Fetches fresh data from Yahoo Finance"""
    print(f"Fetching new data for {symbol}...")
    data = yf.download(symbol, start=start_date, end=end_date or datetime.now())
    
    if not data.empty:
        # Save with index name
        data.index.name = 'Date'
        data.to_csv(filename)
        return process_data(data)
    
    raise DataValidationError(f"No data available for {symbol}")

def update_existing_data(existing_data: pd.DataFrame,
                        symbol: str,
                        end_date: Optional[datetime],
                        filename: str) -> pd.DataFrame:
    """Updates existing data with new data from Yahoo Finance"""
    current_date = datetime.now()
    new_start = existing_data.index[-1] + timedelta(days=1)
    
    if new_start.date() > current_date.date():
        return process_data(existing_data)
        
    print(f"Fetching new data from {new_start.date()} to {end_date or current_date}")
    try:
        new_data = yf.download(symbol, start=new_start, end=end_date or current_date)
        if not new_data.empty:
            updated_data = pd.concat([existing_data, new_data])
            updated_data.to_csv(filename)
            return process_data(updated_data)
        return process_data(existing_data)
    except Exception as e:
        print(f"Error fetching new data: {e}")
        return process_data(existing_data)

def process_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the ETF data by calculating daily returns and volatility.
    """
    print("Processing data...")
    # Use Adj Close if available, otherwise use Close
    price_column = 'Adj Close' if 'Adj Close' in data.columns else 'Close'
    data = data[[price_column]].rename(columns={price_column: 'Price'})
    data['Daily_Return'] = data['Price'].pct_change(fill_method=None)
    data['Volatility'] = data['Daily_Return'].rolling(window=30).std() * (252 ** 0.5)

    return data

