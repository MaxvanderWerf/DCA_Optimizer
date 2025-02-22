"""
investment_logger.py

This module provides logging functionality for investment decisions made by various DCA strategies.
It tracks and records detailed information about each investment decision, including:
- Date and price of investment
- Number of shares bought
- Cash used and remaining balance
- Strategy-specific metrics (RSI, Moving Averages, etc.)
- Rationale for each investment decision

The logger helps in post-analysis of strategy performance and provides transparency
into the decision-making process of each strategy.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

@dataclass
class InvestmentLog:
    date: datetime
    strategy: str
    action: str
    price: float
    shares: float
    cash_used: float
    cash_balance: float
    reason: str
    metrics: Optional[dict] = None

class InvestmentLogger:
    def __init__(self):
        self.logs: List[InvestmentLog] = []
        
    def log_decision(self, date: datetime, strategy: str, action: str, price: float, 
                    shares: float, cash_used: float, cash_balance: float, 
                    reason: str, metrics: Optional[dict] = None):
        """Records an investment decision with its rationale"""
        log = InvestmentLog(
            date=date,
            strategy=strategy,
            action=action,
            price=price,
            shares=shares,
            cash_used=cash_used,
            cash_balance=cash_balance,
            reason=reason,
            metrics=metrics
        )
        self.logs.append(log)
        
    def get_logs_for_strategy(self, strategy: str) -> List[InvestmentLog]:
        """Returns all logs for a specific strategy"""
        return [log for log in self.logs if log.strategy == strategy]
    
    def clear_logs(self):
        """Clears all logs"""
        self.logs = [] 