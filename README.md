# ETF DCA Optimizer

This project develops a Python-based tool to analyze and optimize various Dollar-Cost Averaging (DCA) strategies for ETF investments. It compares optimized DCA approaches with a baseline strategy, exploring the potential for semi-active management within a traditionally passive investment method.

## Context

This project explores alternatives to standard Dollar-Cost Averaging (DCA) for monthly ETF investments. The goal is to optimize entry points for each month's investment without sacrificing significant time in the market. We compare the standard DCA approach with strategies that aim to capitalize on short-term market inefficiencies while maintaining a consistent long-term investment plan.

## Features

- ETF Data Retrieval: Fetches historical ETF price data from Yahoo Finance.
- DCA Strategy Simulation: Simulates multiple DCA strategies including standard, volatility-adjusted, and trend-following.
- Performance Comparison: Evaluates and compares the performance of different DCA strategies.
- Time in Market vs Timing Analysis: Analyzes strategy effectiveness in terms of market participation and timing.
- Visualization: Generates charts to illustrate strategy performance.

## Strategy Overview
The RSI and mean reversion strategies are great fits for our goal of optimizing monthly investments. 
They tap into short-term market inefficiencies without veering too far from our core DCA approach. 
The RSI strategy helps us buy more when the market might be oversold and less when it's potentially overbought. 
Meanwhile, the mean reversion strategy lets us take advantage of temporary price deviations from recent trends. 
Both strategies aim to improve our entry points each month, potentially boosting returns without sacrificing the 
consistency that makes DCA so appealing. They're simple enough to understand and implement, but sophisticated enough to 
potentially outperform a standard DCA approach.

## Usage

1. Ensure you have Python 3.7+ installed.
2. Install required libraries: `pip install pandas yfinance matplotlib`
3. Run the main script: `python src/main.py`

## Project Structure

- `data/`: Contains the price data for the ETFs
- `src/`: Contains the source code files
  - `main.py`: Main script to run the ETF DCA Optimizer
  - `data_retrieval.py`: Handles ETF data fetching and processing
  - `strategy_simulator.py`: Implements various DCA strategies
  - `performance_comparison.py`: Compares strategy performances
  - `visualization.py`: Creates visualizations of results
- `README.md`: This file

## Output

The script generates two PNG files with visualizations:
- `portfolio_value_over_time.png`: Shows the portfolio value over time for each strategy
- `total_return_by_strategy.png`: Compares the total return of each strategy

## Note

This tool is for educational and research purposes only. Always consult with a financial advisor before making investment decisions.

## Data Format
The ETF/Stock data is stored in CSV files with the following columns:
- Date: Trading date (YYYY-MM-DD)
- Open: Opening price
- High: Highest price of the day
- Low: Lowest price of the day
- Close: Closing price
- Adj Close: Adjusted closing price
- Volume: Trading volume

This data is fetched from Yahoo Finance and cached locally in the `data/` directory.