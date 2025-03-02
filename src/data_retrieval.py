"""
data_retrieval.py

This module handles the fetching, processing, and management of ETF/Stock price data.

Architecture:
    The module follows a layered approach to data management:
    1. fetch_etf_data: Main entry point that coordinates the data retrieval process
    2. load_and_validate_file: Handles loading and validating existing data files
    3. get_fresh_data: Downloads new data when needed
    4. update_data: Incrementally updates existing data with new information
    5. process_raw_data: Standardizes data format across all operations
    6. save_data: Handles the consistent saving of data files

Data Flow:
    1. Check if data file exists for the requested symbol
    2. If exists: Load and validate the file format
    3. If file is current: Return the loaded data
    4. If file needs update: Fetch only the missing data and append it
    5. If file doesn't exist or is corrupted: Download complete fresh data
    6. Process all data into a standardized format before returning

Data Format:
    All data is standardized to include these columns:
    - Price: Adjusted closing price (or regular closing price if adjusted is unavailable)
    - Daily_Return: Day-to-day percentage change in price
    - Volatility: 30-day rolling standard deviation of returns (annualized)

Error Handling:
    - All operations use try/except blocks to handle errors gracefully
    - The DataError class provides specific error reporting for data issues
    - When errors occur, the system falls back to downloading fresh data
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Optional, Tuple

class DataError(Exception):
    """
    Custom exception for data-related errors.
    
    This exception is raised when there are issues with data fetching, processing,
    or validation that cannot be automatically resolved.
    """
    pass

def fetch_etf_data(symbol: str, 
                  start_date: Optional[datetime] = None, 
                  end_date: Optional[datetime] = None) -> pd.DataFrame:
    """
    Main function to get ETF/Stock data, either from local file or Yahoo Finance.
    
    This function serves as the main entry point for data retrieval. It determines
    whether to use cached data, update existing data, or fetch completely fresh data
    based on what's available and current.
    
    Args:
        symbol: Ticker symbol (e.g., 'SPY', 'QQQ', 'AAPL')
        start_date: Optional start date for data. If None, defaults to 10 years ago.
        end_date: Optional end date for data. If None, defaults to current date.
        
    Returns:
        DataFrame with processed price data containing:
        - Price: Daily adjusted closing price
        - Daily_Return: Day-to-day percentage change
        - Volatility: 30-day rolling volatility (annualized)
        
    Raises:
        DataError: If start_date is after end_date
        
    Notes:
        - Creates the data directory if it doesn't exist
        - Validates date ranges and normalizes inputs
        - Falls back to fresh data download if any errors occur
        - Returns data in a standardized format regardless of source
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
    Load data from CSV file and validate its format and completeness.
    
    This function handles loading cached data files and ensures they're in the
    correct format. It checks for proper datetime index, required columns,
    and determines if the data needs to be updated.
    
    Args:
        file_path: Path to the CSV file to load
        end_date: End date to compare against for determining if update is needed
    
    Returns:
        Tuple containing:
        - processed_data: DataFrame with the loaded and validated data
        - needs_update: Boolean indicating if the data needs to be updated
    
    Raises:
        DataError: If the file cannot be loaded or validated
        
    Notes:
        - Converts index to datetime if needed
        - Checks for required columns and regenerates them if missing
        - Determines if data is current by comparing last date to current date
        - Handles various formatting issues that may exist in the file
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
    Download fresh data from Yahoo Finance and save it to a local file.
    
    This function is called when no cached data exists or when existing data
    is corrupted. It downloads the complete dataset for the requested period,
    processes it into the standard format, and saves it locally.
    
    Args:
        symbol: Ticker symbol to download data for
        start_date: Start date for data download. If None, defaults to 10 years ago.
        end_date: End date for data download
        file_path: Path where the downloaded data should be saved
    
    Returns:
        DataFrame with the processed downloaded data
        
    Raises:
        DataError: If no data is returned from Yahoo Finance or download fails
        
    Notes:
        - Uses yfinance library to download historical price data
        - Processes raw data into standardized format before returning
        - Saves data to the specified file path for future use
        - Provides comprehensive error handling for download issues
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
    
    This function handles incremental updates to existing data. It determines
    what date range is missing, downloads only that data, and appends it to
    the existing dataset.
    
    Args:
        existing_data: DataFrame containing the already loaded data
        symbol: Ticker symbol to update data for
        end_date: End date for the update period
        file_path: Path where the updated data should be saved
    
    Returns:
        DataFrame with the combined existing and newly downloaded data
        
    Notes:
        - Only downloads the missing date range (from last available date to end_date)
        - Combines new data with existing data
        - Removes any duplicate dates (keeping the newest data)
        - Returns existing data unchanged if already up to date or if update fails
        - Saves the updated combined data to the file path
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
    Process raw Yahoo Finance data into standardized format.
    
    This function takes raw data (either from a download or loaded file) and
    transforms it into the standard format used throughout the application.
    
    Args:
        raw_data: DataFrame containing raw price data
    
    Returns:
        DataFrame with standardized columns:
        - Price: Daily adjusted closing price (or closest equivalent)
        - Daily_Return: Day-to-day percentage change in price
        - Volatility: 30-day rolling standard deviation of returns (annualized)
        
    Raises:
        DataError: If no price data can be found in the input columns
        
    Notes:
        - Prioritizes Adjusted Close price when available
        - Falls back to Close price if Adjusted Close isn't available
        - Handles tuple-based column names that may come from Yahoo Finance
        - Calculates derived metrics (returns, volatility) from the price data
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
    Save processed data to CSV with standardized format.
    
    This function handles the consistent saving of data files to ensure
    they're in the correct format for future loading.
    
    Args:
        data: DataFrame containing the processed data to save
        file_path: Path where the data should be saved
    
    Notes:
        - Creates directories in the path if they don't exist
        - Saves with consistent date format in the index
        - Verifies the file was successfully created
        - Provides error handling for file system issues
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

