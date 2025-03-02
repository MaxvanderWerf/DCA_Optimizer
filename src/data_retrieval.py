"""
data_retrieval.py

This module handles the fetching and management of ETF/Stock price data.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple

class DataError(Exception):
    """Custom exception for data-related errors"""
    pass

def fetch_etf_data(symbol: str, 
                  start_date: Optional[datetime] = None, 
                  end_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Main function to get ETF/Stock data, either from local file or Yahoo Finance.
    
    Args:
        symbol: Ticker symbol (e.g., 'SPY')
        start_date: Optional start date for data
        end_date: Optional end date for data
        
    Returns:
        DataFrame with processed price data
    """
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Normalize dates
    end_date = min(end_date or datetime.now(), datetime.now())
    if start_date and start_date > end_date:
        raise DataError(f"Start date {start_date} is after end date {end_date}")
    
    # File path for cached data
    file_path = f"data/{symbol}_data.csv"
    
    try:
        # Try to load from file if it exists
        if os.path.exists(file_path):
            data, needs_update = load_and_validate_file(file_path, end_date)
            
            # If file is up to date, just return the loaded data
            if not needs_update:
                return data
            
            # Otherwise, update the data with new information
            return update_data(data, symbol, end_date, file_path)
        else:
            # Get fresh data if no file exists
            return get_fresh_data(symbol, start_date, end_date, file_path)
            
    except Exception as e:
        # For any errors, log and fetch fresh data
        print(f"Error handling data for {symbol}: {e}")
        return get_fresh_data(symbol, start_date, end_date, file_path)

def load_and_validate_file(file_path: str, end_date: datetime) -> Tuple[pd.DataFrame, bool]:
    """
    Load data from CSV file and validate it.
    
    Returns:
        Tuple of (processed_data, needs_update)
    """
    try:
        # Read the CSV file with explicit date parsing
        data = pd.read_csv(file_path, index_col=0, parse_dates=True)
        
        # Verify the data is properly formatted
        if not isinstance(data.index, pd.DatetimeIndex):
            print(f"Converting index to datetime for {file_path}")
            data.index = pd.to_datetime(data.index)
        
        # Check if we have all required columns
        required_columns = ['Price', 'Daily_Return', 'Volatility']
        if not all(col in data.columns for col in required_columns):
            print(f"Data in {file_path} missing required columns, will regenerate")
            data = process_raw_data(data)
        
        # Check if data is current or needs update
        last_date = data.index[-1].date()
        current_date = datetime.now().date()
        needs_update = last_date < current_date
        
        return data, needs_update
        
    except Exception as e:
        print(f"Error loading data from {file_path}: {e}")
        raise DataError(f"Could not load data from {file_path}")

def get_fresh_data(symbol: str, start_date: Optional[datetime], 
                  end_date: datetime, file_path: str) -> pd.DataFrame:
    """
    Download fresh data from Yahoo Finance and save it.
    """
    print(f"Downloading fresh data for {symbol}...")
    
    # Use a reasonable default start date if none provided
    if start_date is None:
        start_date = end_date - timedelta(days=365*10)  # 10 years of data
    
    try:
        # Download data from Yahoo Finance
        raw_data = yf.download(symbol, start=start_date, end=end_date)
        
        if raw_data.empty:
            raise DataError(f"No data returned for {symbol}")
            
        # Process and save the data
        processed_data = process_raw_data(raw_data)
        save_data(processed_data, file_path)
        
        return processed_data
        
    except Exception as e:
        print(f"Error downloading fresh data for {symbol}: {e}")
        raise DataError(f"Failed to download data for {symbol}")

def update_data(existing_data: pd.DataFrame, symbol: str, 
               end_date: datetime, file_path: str) -> pd.DataFrame:
    """
    Update existing data with new data from Yahoo Finance.
    """
    # Get the day after the last date in our data
    last_date = existing_data.index[-1]
    new_start = last_date + timedelta(days=1)
    
    # Return existing data if we're already up to date
    if new_start.date() > end_date.date():
        return existing_data
    
    print(f"Updating data for {symbol} from {new_start.date()} to {end_date.date()}...")
    
    try:
        # Download only the new data
        new_data = yf.download(symbol, start=new_start, end=end_date)
        
        if new_data.empty:
            return existing_data
            
        # Process the new data
        new_processed = process_raw_data(new_data)
        
        # Combine old and new data
        combined_data = pd.concat([existing_data, new_processed])
        
        # Remove any duplicates (keep the newest)
        combined_data = combined_data[~combined_data.index.duplicated(keep='last')]
        
        # Save the updated data
        save_data(combined_data, file_path)
        
        return combined_data
        
    except Exception as e:
        print(f"Error updating data for {symbol}: {e}")
        return existing_data  # Return existing data in case of error

def process_raw_data(raw_data: pd.DataFrame) -> pd.DataFrame:
    """
    Process raw Yahoo Finance data into our standard format.
    """
    # Create a new DataFrame for processed data
    processed = pd.DataFrame()
    
    # Use Adj Close if available, otherwise use Close
    if 'Adj Close' in raw_data.columns:
        processed['Price'] = raw_data['Adj Close']
    elif 'Close' in raw_data.columns:
        processed['Price'] = raw_data['Close']
    else:
        # Find any column that might be a close price
        close_cols = [col for col in raw_data.columns if isinstance(col, tuple) and 'Close' in col[0]]
        if close_cols:
            processed['Price'] = raw_data[close_cols[0]]
        else:
            raise DataError("No price data found in columns")
    
    # Calculate daily returns
    processed['Daily_Return'] = processed['Price'].pct_change()
    
    # Calculate volatility (30-day rolling standard deviation, annualized)
    processed['Volatility'] = processed['Daily_Return'].rolling(window=30, min_periods=1).std() * (252 ** 0.5)
    
    return processed

def save_data(data: pd.DataFrame, file_path: str) -> None:
    """
    Save processed data to CSV with standard format.
    """
    try:
        # Ensure the directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save data with datetime index
        data.to_csv(file_path, date_format='%Y-%m-%d')
        
        # Verify file was created
        if not os.path.exists(file_path):
            print(f"Warning: File {file_path} was not created")
            
    except Exception as e:
        print(f"Error saving data to {file_path}: {e}")

