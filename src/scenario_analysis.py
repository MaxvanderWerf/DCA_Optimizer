"""
scenario_analysis.py

This module handles the generation and analysis of different investment scenarios.
It provides functionality to:
- Generate multiple test scenarios by varying parameters
- Test strategies across different time periods
- Analyze strategy performance under various market conditions
- Save and load scenario results for comparison
- Generate comprehensive scenario reports

The module helps in understanding strategy robustness across different
market conditions and time periods.
"""

from typing import List, Dict, Tuple
import pandas as pd
from datetime import datetime, timedelta
import itertools

def generate_scenarios(base_params: Dict) -> List[Dict]:
    """
    Generate multiple test scenarios by varying parameters.
    """
    scenarios = []
    
    # Test different time periods
    start_years = range(2000, 2024, 2)  # Test different starting years
    period_lengths = [3, 5, 7, 10]  # Different investment horizons
    
    # Test different portfolio allocations
    portfolio_weights = [
        {'SPY': 1.0},
        {'SPY': 0.7, 'QQQ': 0.3},
        {'SPY': 0.6, 'QQQ': 0.2, 'AAPL': 0.2}
    ]
    
    # Strategy parameters
    rsi_thresholds = [(20, 80), (30, 70), (40, 60)]
    ma_windows = [10, 20, 50]
    
    for year, length, weights, (rsi_low, rsi_high), ma_window in itertools.product(
        start_years, period_lengths, portfolio_weights, rsi_thresholds, ma_windows
    ):
        scenario = base_params.copy()
        start_date = datetime(year, 1, 1)
        end_date = start_date + timedelta(days=365*length)
        
        scenario.update({
            'start_date': start_date,
            'end_date': end_date,
            'portfolio_weights': weights,
            'rsi_thresholds': (rsi_low, rsi_high),
            'ma_window': ma_window,
            'scenario_id': f"{year}_{length}y_{'-'.join(weights.keys())}"
        })
        scenarios.append(scenario)
    
    return scenarios

def save_results(results: List[Dict], filename: str = "simulation_results.csv") -> None:
    """
    Save simulation results to CSV for further analysis.
    """
    df = pd.DataFrame(results)
    df.to_csv(f"data/results/{filename}", index=False)
    print(f"Results saved to data/results/{filename}") 