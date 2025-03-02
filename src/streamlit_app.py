"""
streamlit_app.py

This module implements the Streamlit web interface for the ETF DCA Optimizer.
It provides an interactive dashboard where users can:
- Input ETF/Stock symbols and investment parameters
- Visualize portfolio performance across different strategies
- Compare strategy results with detailed metrics
- View investment decision logs

The interface uses a dark theme optimized for financial data visualization
and includes interactive plots and formatted performance metrics.
"""

from datetime import datetime, timedelta
from data_retrieval import fetch_etf_data
from investment_logger import InvestmentLogger, InvestmentLog
from performance_comparison import compare_performances
from strategy_simulator import simulate_dca_strategies
from typing import List
from util import validate_dates, get_date_info

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Popular ETFs and stocks with descriptions
POPULAR_ASSETS = {
    "Popular ETFs": {
        "SPY": "SPDR S&P 500 ETF - Tracks the S&P 500 Index",
        "QQQ": "Invesco QQQ - Tracks the Nasdaq-100 Index",
        "VTI": "Vanguard Total Stock Market ETF - Total US Market Exposure",
        "VOO": "Vanguard S&P 500 ETF - Low-cost S&P 500 Index Fund",
        "ARKK": "ARK Innovation ETF - Disruptive Innovation Companies",
    },
    "Popular Stocks": {
        "AAPL": "Apple Inc. - Consumer Technology",
        "MSFT": "Microsoft Corporation - Technology & Cloud Computing",
        "GOOGL": "Alphabet Inc. - Technology & Digital Advertising",
        "AMZN": "Amazon.com Inc. - E-commerce & Cloud Services",
        "TSLA": "Tesla Inc. - Electric Vehicles & Clean Energy",
    }
}

# https://claude.ai/chat/76e105c9-149e-46f6-bcc9-7430f79c831c


def create_interactive_portfolio_value_plot(simulation_results, identifier):
    fig = go.Figure()
    
    for strategy, result in simulation_results.items():
        # Ensure index is datetime
        if not isinstance(result.index, pd.DatetimeIndex):
            result.index = pd.to_datetime(result.index)
            
        fig.add_trace(go.Scatter(
            x=result.index,
            y=result['Portfolio_Value'],
            mode='lines',
            name=strategy
        ))
    
    # Get date range after conversion
    min_date = min([result.index.min() for result in simulation_results.values()])
    max_date = max([result.index.max() for result in simulation_results.values()])
    
    fig.update_layout(
        title=f'Portfolio Value Over Time - {identifier}',
        xaxis_title='Date',
        yaxis_title='Portfolio Value ($)',
        xaxis_range=[min_date, max_date],
        plot_bgcolor='#112240',
        paper_bgcolor='#112240',
        font_color='white'
    )
    return fig


def create_interactive_investment_decisions_plot(simulation_results, identifier):
    fig = go.Figure()
    for strategy, result in simulation_results.items():
        fig.add_trace(go.Scatter(x=result.index, y=result['Invested'],
                                 mode='lines+markers', name=f'{strategy} - Invested'))
        fig.add_trace(go.Scatter(x=result.index, y=result['Cash_Balance'],
                                 mode='lines', name=f'{strategy} - Cash Balance', line=dict(dash='dash')))
    fig.update_layout(
        title=f'Investment Decisions Over Time - {identifier}',
        xaxis_title='Date',
        yaxis_title='Amount ($)',
        plot_bgcolor='#112240',
        paper_bgcolor='#112240',
        font_color='white'
    )
    return fig


def set_custom_style():
    st.markdown("""
        <style>
        /* Color palette */
        :root {
            --navy: #0A192F;         /* Darkest blue - main background */
            --light-navy: #112240;   /* Lighter blue - container background */
            --darker-navy: #0D1929;  /* Even darker blue for tooltips */
            --cyan: #64FFDA;         /* Cyan - accents and highlights */
            --light-slate: #a8b2d1;  /* Light text color */
            --green: #43D08A;        /* Success color */
            --red: #E06C75;          /* Error color */
        }
        
        /* Main background */
        .stApp {
            background-color: var(--navy);
        }
        
        /* Make the app narrower */
        .block-container {
            max-width: 75% !important;
            padding-top: 1rem;
            padding-right: 1rem;
            padding-left: 1rem;
            padding-bottom: 1rem;
        }
        
        /* Main header */
        .main-header {
            color: white;
            font-size: 42px;
            font-weight: 600;
            margin-bottom: 30px;
        }
        
        /* Metrics */
        [data-testid="stMetricValue"] {
            color: var(--cyan);
        }
        
        /* Container styling */
        [data-testid="stDataFrame"], 
        [data-testid="stPlotlyChart"] {
            background-color: var(--light-navy);
        }

        /* Strategy tags in multiselect */
        [data-testid="stMultiSelect"] span[data-baseweb="tag"] {
            background-color: var(--light-navy);
            color: var(--cyan);
        }

        /* Run Simulation button */
        .stButton button {
            background-color: transparent;
            color: var(--cyan);
            border: 1px solid var(--cyan);
        }
        
        /* Month styling - even more minimal */
        .month {
            color: white;
            font-weight: bold;
            padding: 5px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        /* Reduce padding in columns */
        [data-testid="column"] {
            padding: 0.25rem !important;
        }
        
        /* Transaction styling - ultra minimal */
        .transaction {
            padding: 5px 0;
            margin-bottom: 4px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.05);
            position: relative;
            cursor: pointer;
        }
        
        /* Last transaction in a group */
        .transaction:last-child {
            border-bottom: none;
        }
        
        /* Transaction amount */
        .amount {
            color: var(--cyan);
            font-weight: bold;
            display: inline;
        }
        
        /* Transaction date */
        .date {
            color: white;
        }
        
        /* Transaction shares */
        .shares {
            color: var(--light-slate);
        }
        
        /* Reason styling */
        .reason {
            color: var(--light-slate);
            margin-left: 18px;
            margin-top: 3px;
        }
        
        /* Status indicators */
        .buy {
            color: var(--green);
        }
        
        .skip {
            color: var(--red);
        }
        
        /* Headers */
        .header-row {
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 5px;
            margin-bottom: 10px;
        }
        
        /* Info button */
        .info-button {
            color: var(--cyan);
            font-size: 0.85em;
            margin-left: 5px;
            cursor: pointer;
            vertical-align: middle;
        }
        
        /* Custom tooltip container */
        .tooltip-container {
            display: inline;
            position: relative;
        }
        
        /* Custom tooltip */
        .tooltip-content {
            visibility: hidden;
            width: 300px;
            background-color: var(--darker-navy);
            color: var(--light-slate);
            text-align: left;
            border-radius: 5px;
            padding: 10px;
            position: absolute;
            z-index: 1000;
            bottom: 125%;
            left: 50%;
            margin-left: -150px;
            opacity: 0;
            transition: opacity 0.3s;
            box-shadow: 0 4px 8px rgba(0,0,0,0.3);
            border: 1px solid rgba(255,255,255,0.1);
            pointer-events: none;
        }
        
        /* Show the tooltip when hovering over the container */
        .transaction:hover .tooltip-content {
            visibility: visible;
            opacity: 1;
        }
        
        /* Tooltip arrow */
        .tooltip-content::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: var(--darker-navy) transparent transparent transparent;
        }
        
        /* Tooltip section headers */
        .tooltip-header {
            color: white;
            font-weight: bold;
            margin-bottom: 5px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding-bottom: 3px;
        }
        
        /* Tooltip data rows */
        .tooltip-row {
            display: flex;
            justify-content: space-between;
            margin: 3px 0;
        }
        
        /* Tooltip data label */
        .tooltip-label {
            color: var(--light-slate);
        }
        
        /* Tooltip data value */
        .tooltip-value {
            color: white;
            font-weight: bold;
        }
        
        /* Tooltip value emphasis */
        .tooltip-emphasis {
            color: var(--cyan);
        }
        </style>
    """, unsafe_allow_html=True)


def format_decision_log(log: InvestmentLog) -> str:
    """Format a single investment decision log into a user-friendly string"""
    date_str = log.date.strftime('%Y-%m-%d')
    
    # For skipped investments
    if log.action == "skip":
        return f"❌ {date_str}: No investment made"
    
    # For buy actions
    price_str = f"${log.price:.2f}"
    
    # Basic buy information
    base_text = f"✅ {date_str}: Bought {log.shares:.0f} shares"
    
    # Add simplified strategy-specific context
    if log.metrics:
        if "MA20" in log.metrics:
            deviation = log.metrics['Deviation']
            if deviation < 0:
                base_text += "\n    📉 Price below average - buying more"
            else:
                base_text += "\n    📈 Price above average - buying less"
        elif "RSI" in log.metrics:
            rsi = log.metrics['RSI']
            if rsi < 30:
                base_text += "\n    📉 Market dip - good time to buy"
            elif rsi > 70:
                base_text += "\n    📈 Market high - being cautious"
            else:
                base_text += "\n    ➡️ Normal market conditions"
    
    # Add money info on same line
    base_text += f" (${log.cash_used:.0f} invested)"
    
    return base_text


def display_strategy_logs(logs_by_strategy: dict[str, List[InvestmentLog]]):
    """Display logs for all strategies in an ultra-minimal format with custom tooltips"""
    
    # Group logs by month and year for each strategy
    monthly_logs = {}
    
    for strategy, logs in logs_by_strategy.items():
        for log in logs:
            month_key = log.date.strftime('%Y-%m')
            if month_key not in monthly_logs:
                monthly_logs[month_key] = {}
            
            if strategy not in monthly_logs[month_key]:
                monthly_logs[month_key][strategy] = []
                
            monthly_logs[month_key][strategy].append(log)
    
    # Sort months in reverse chronological order
    sorted_months = sorted(monthly_logs.keys(), reverse=True)
    
    # Create a consistent header for strategy names
    num_strategies = len(logs_by_strategy)
    strategy_headers = st.columns([0.7] + [3] * num_strategies)
    
    # Header row with light bottom border
    st.markdown("<div class='header-row'>", unsafe_allow_html=True)
    
    with strategy_headers[0]:
        st.markdown("**Month**")
    
    for i, strategy in enumerate(logs_by_strategy.keys(), 1):
        with strategy_headers[i]:
            st.markdown(f"**{strategy.replace('_', ' ').title()}**")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create month-by-month comparison view
    for month in sorted_months:
        month_display = datetime.strptime(month, '%Y-%m').strftime('%B %Y')
        
        # Create a row with month on left and strategies on right
        cols = st.columns([0.7] + [3] * num_strategies)
        
        # Display month in the left column - ultra minimal
        with cols[0]:
            st.markdown(f"<div class='month'>{month_display}</div>", unsafe_allow_html=True)
        
        # For each strategy column
        for i, strategy in enumerate(logs_by_strategy.keys(), 1):
            with cols[i]:
                # Get this strategy's logs for this month
                strategy_logs = monthly_logs[month].get(strategy, [])
                
                if not strategy_logs:
                    st.markdown("<span style='color:#777777;'>No transactions</span>", unsafe_allow_html=True)
                    continue
                
                # Display each transaction with minimal styling
                for log in strategy_logs:
                    # Get day of week and format date
                    weekday = log.date.strftime('%a')
                    date_str = f"{weekday} {log.date.strftime('%d %b')}"
                    status_class = "buy" if log.action == "buy" else "skip"
                    
                    # Determine singular or plural for shares
                    shares_text = "1 share" if log.shares == 1 else f"{log.shares:.0f} shares"
                    
                    # Create custom tooltip content
                    tooltip_content = create_tooltip_content(log)
                    
                    # Transaction line with minimal elements and tooltip
                    if log.action == "buy":
                        st.markdown(
                            f"<div class='transaction'>"
                            f"<span class='date'>{date_str}</span> · "
                            f"<span class='{status_class}'>{shares_text} @ €{log.price:.2f}</span> · "
                            f"<span class='amount'>€{log.cash_used:.0f} invested</span>"
                            f"<div class='reason'>{format_reason_clear(log)}</div>"
                            f"<div class='tooltip-content'>{tooltip_content}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f"<div class='transaction'>"
                            f"<span class='date'>{date_str}</span> · "
                            f"<span class='{status_class}'>No investment</span>"
                            f"<div class='tooltip-content'>{tooltip_content}</div>"
                            f"</div>",
                            unsafe_allow_html=True
                        )


def create_tooltip_content(log: InvestmentLog) -> str:
    """Create formatted tooltip content with detailed information about the transaction"""
    portfolio_value = log.shares * log.price + log.cash_balance
    
    # Portfolio section
    tooltip = "<div class='tooltip-header'>Portfolio Status</div>"
    tooltip += f"<p>Cash balance: <strong>€{log.cash_balance:.2f}</strong></p>"
    tooltip += f"<p>Portfolio value: <strong>€{portfolio_value:.2f}</strong></p>"
    
    # Strategy analysis section
    if log.metrics:
        tooltip += "<div class='tooltip-header' style='margin-top:8px;'>Strategy Analysis</div>"
        
        if "RSI" in log.metrics:
            rsi = log.metrics['RSI']
            
            # Determine RSI status
            rsi_status = "Neutral"
            rsi_class = ""
            if rsi < 30:
                rsi_status = "Oversold (Buy)"
                rsi_class = "tooltip-emphasis"
            elif rsi > 70:
                rsi_status = "Overbought (Caution)"
                rsi_class = "buy"
            
            tooltip += f"<p>Current RSI: <strong>{rsi:.1f}</strong></p>"
            tooltip += f"<p>Status: <strong class='{rsi_class}'>{rsi_status}</strong></p>"
            
            if "Investment Scale" in log.metrics:
                scale = log.metrics['Investment Scale']
                tooltip += f"<p>Investment scale: <strong>{scale:.2f}x</strong></p>"
            
            tooltip += "<div style='font-size:0.9em;margin-top:5px;'>"
            tooltip += "<p>RSI &lt; 30: Oversold - invest more</p>"
            tooltip += "<p>RSI &gt; 70: Overbought - invest less</p>"
            tooltip += "<p>30-70: Normal conditions</p>"
            tooltip += "</div>"
            
        elif "Deviation" in log.metrics:
            deviation = log.metrics['Deviation']
            dev_pct = abs(deviation) * 100
            direction = "Below average" if deviation < 0 else "Above average"
            
            dev_class = "tooltip-emphasis" if deviation < 0 else "buy"
            
            tooltip += f"<p>Price deviation: <strong class='{dev_class}'>{dev_pct:.1f}%</strong></p>"
            tooltip += f"<p>Direction: <strong>{direction}</strong></p>"
            
            if "MA" in log.metrics:
                ma = log.metrics['MA']
                tooltip += f"<p>Moving average: <strong>€{ma:.2f}</strong></p>"
                tooltip += f"<p>Current price: <strong>€{log.price:.2f}</strong></p>"
            
            tooltip += "<div style='font-size:0.9em;margin-top:5px;'>"
            tooltip += "<p>Below average: Buy more</p>"
            tooltip += "<p>Above average: Buy less</p>"
            tooltip += "</div>"
    
    # Decision section for skipped investments
    if log.action == "skip":
        tooltip += "<div class='tooltip-header' style='margin-top:8px;'>Decision</div>"
        tooltip += "<p style='color:var(--red);'>No investment this period</p>"
    
    return tooltip


def format_reason_clear(log: InvestmentLog) -> str:
    """Format the reason text to be more clear about the decision logic"""
    if log.action == "skip":
        return "No investment made"
    
    if log.metrics:
        if "RSI" in log.metrics:
            rsi = log.metrics['RSI'] 
            if rsi < 30:
                return f"<strong>Market dip</strong> (RSI: {rsi:.0f}) - investing more"
            elif rsi > 70:
                return f"<strong>Market high</strong> (RSI: {rsi:.0f}) - investing less"
            else:
                return f"<strong>Normal market</strong> (RSI: {rsi:.0f})"
        elif "Deviation" in log.metrics:
            deviation = log.metrics['Deviation']
            dev_pct = abs(deviation) * 100
            if deviation < 0:
                return f"<strong>Price below average</strong> ({dev_pct:.0f}%) - buying more"
            else:
                return f"<strong>Price above average</strong> ({dev_pct:.0f}%) - buying less"
    
    return "Regular scheduled purchase"


def format_performance_metrics(performance_metrics: pd.DataFrame) -> pd.DataFrame:
    """Format performance metrics for display"""
    formatted_df = performance_metrics.copy()
    
    # Rename columns to remove units from names
    formatted_df = formatted_df.rename(columns={
        'Total Return': 'Total Return',
        'Annualized Return': 'Annualized Return',
        'Total Gain (€)': 'Total Gain',
        'Final Portfolio Value': 'Final Portfolio Value',
        'Total Invested': 'Total Invested'
    })
    
    # Format percentages
    formatted_df['Total Return'] = formatted_df['Total Return'].map('{:.1%}'.format)
    formatted_df['Annualized Return'] = formatted_df['Annualized Return'].map('{:.1%}'.format)
    
    # Format currency values with 2 decimal places, no thousands separator
    for col in ['Total Gain', 'Final Portfolio Value', 'Total Invested']:
        formatted_df[col] = formatted_df[col].map('€{:.2f}'.format)
    
    return formatted_df


def main():
    set_custom_style()
    
    st.markdown('<h1 class="main-header">ETF/Stock DCA Optimizer</h1>', unsafe_allow_html=True)

    # Sidebar for inputs
    with st.sidebar:
        st.header("Settings")
        
        # Initialize symbol in session state if not present
        if 'symbol' not in st.session_state:
            st.session_state.symbol = "SPY"
        
        # Asset selector section
        symbol = st.text_input(
            "Enter ETF/Stock symbol",
            value=st.session_state.symbol,
            help="Enter the ticker symbol (e.g., SPY, AAPL)"
        ).upper()
        
        # Handle symbol changes and data loading
        valid_data = True
        if symbol != st.session_state.symbol:
            try:
                # Data validation and loading
                data = fetch_etf_data(symbol)
                if data.empty:
                    st.error("Something went wrong ❌")
                    valid_data = False
                else:
                    st.session_state.symbol = symbol
            except Exception as e:
                st.error("Something went wrong ❌")
                valid_data = False
        else:
            try:
                # Silent initial load
                data = fetch_etf_data(symbol)
                if data.empty:
                    st.error("Something went wrong ❌")
                    valid_data = False
                else:
                    st.session_state.symbol = symbol
            except Exception as e:
                st.error("Something went wrong ❌")
                valid_data = False

            if valid_data:
                date_info = get_date_info(data)
                
                # Time Period inputs
                col1, col2 = st.columns(2)
                with col1:
                    start_date = st.date_input(
                        "Start date",
                        value=datetime.now() - timedelta(days=365 * 5),
                        min_value=date_info['data_start'],
                        max_value=date_info['data_end'],
                        help="Select start date"
                    )
                
                with col2:
                    end_date = st.date_input(
                        "End date",
                        value=date_info['data_end'],
                        min_value=start_date,
                        max_value=date_info['data_end'],
                        help="Select end date"
                    )

                try:
                    start_date, end_date, messages = validate_dates(start_date, end_date, data)
                    
                    if start_date >= end_date:
                        st.error("Start date must be before end date")
                        valid_data = False
                    else:
                        for message in messages:
                            st.warning(message)
                        
                        data = data.loc[start_date:end_date]
                    
                except ValueError as e:
                    st.error(str(e))
                    valid_data = False

                if valid_data:
                    # Investment inputs
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        initial_investment = st.number_input(
                            "Initial ($)",
                            min_value=0,
                            value=10000,
                            step=1000,
                            help="One-time investment at start"
                        )
                        
                        if initial_investment > 0:
                            shares = initial_investment / data['Price'].iloc[0]
                            st.caption(f"≈ {shares:.1f} shares at starting price")

                    with col2:
                        monthly_investment = st.number_input(
                            "Monthly ($)",
                            min_value=0,
                            value=1000,
                            step=100,
                            help="Monthly investment amount"
                        )
                        
                        if monthly_investment > 0:
                            shares = monthly_investment / data['Price'].iloc[0]
                            st.caption(f"≈ {shares:.1f} shares at starting price")
                    
                    # Investment Summary metrics
                    total_months = (end_date - start_date).days / 30.44
                    total_investment = initial_investment + (monthly_investment * total_months)
                    
                    summary_col1, summary_col2 = st.columns(2)
                    with summary_col1:
                        st.metric(
                            "Investment Period",
                            f"{total_months:.1f} mo"
                        )
                    with summary_col2:
                        st.metric(
                            "Total Investment",
                            f"${total_investment:,.0f}"
                        )

        # Strategy Selection and Settings
        st.header("Strategy Settings")
        
        base_strategies = st.multiselect(
            "Select base strategies to simulate:",
            ["standard", "rsi", "mean_reversion"],
            default=["standard"],
            help="Choose one or more investment strategies to simulate and compare"
        )

        # Dictionary to store strategy variations
        strategy_variations = {}
        strategy_params = {}

        # Strategy-specific settings with variations
        if "rsi" in base_strategies:
            st.subheader("RSI Settings")
            
            if 'rsi_variations' not in st.session_state:
                st.session_state.rsi_variations = 1
            
            for i in range(st.session_state.rsi_variations):
                with st.expander(f"RSI Strategy #{i+1} Settings"):
                    col1, col2 = st.columns(2)
                    with col1:
                        rsi_lower = st.number_input(
                            "RSI Lower Bound",
                            min_value=0,
                            max_value=100,
                            value=30,
                            key=f"rsi_lower_{i}"
                        )
                    with col2:
                        rsi_upper = st.number_input(
                            "RSI Upper Bound",
                            min_value=0,
                            max_value=100,
                            value=70,
                            key=f"rsi_upper_{i}"
                        )
                    
                    # Using a single column for RSI Window
                    rsi_window = st.number_input(
                        "RSI Window (days)",
                        min_value=2,
                        max_value=50,
                        value=14,
                        key=f"rsi_window_{i}"
                    )
                    
                    # Removed oversold and overbought scale inputs - using default values instead
                    oversold_scale = 2.0  # Default value
                    overbought_scale = 0.5  # Default value

                    # Store strategy variation
                    internal_name = f"rsi_{rsi_lower}_{rsi_upper}_{rsi_window}"
                    display_name = f"RSI ({rsi_lower}-{rsi_upper}, {rsi_window}d)"
                    strategy_variations[display_name] = "rsi"
                    strategy_params[display_name] = {
                        'lower_bound': rsi_lower,
                        'upper_bound': rsi_upper,
                        'rsi_window': rsi_window,
                        'oversold_scale': oversold_scale,
                        'overbought_scale': overbought_scale
                    }
            
            # Add variation button - improved styling
            if st.session_state.rsi_variations < 3:
                st.button("Add RSI Variation", key="add_rsi", 
                         on_click=lambda: setattr(st.session_state, 'rsi_variations', st.session_state.rsi_variations + 1),
                         use_container_width=True)

        if "mean_reversion" in base_strategies:
            st.subheader("Mean Reversion Settings")
            
            if 'ma_variations' not in st.session_state:
                st.session_state.ma_variations = 1
            
            for i in range(st.session_state.ma_variations):
                with st.expander(f"Mean Reversion Strategy #{i+1} Settings"):
                    col1, col2 = st.columns(2)
                    with col1:
                        ma_window = st.number_input(
                            "Moving Average Window (days)",
                            min_value=5,
                            max_value=200,
                            value=20,
                            key=f"ma_window_{i}"
                        )
                    with col2:
                        ma_type = st.selectbox(
                            "Moving Average Type",
                            options=['simple', 'exponential', 'weighted'],
                            key=f"ma_type_{i}"
                        )
                    
                    # Removed deviation threshold, max scale up/down inputs - using default values
                    deviation_threshold = 0.1  # Default value
                    max_scale_up = 2.0  # Default value
                    max_scale_down = 0.5  # Default value

                    # Store strategy variation
                    internal_name = f"mean_reversion_{ma_window}_{ma_type}"
                    display_name = f"Mean Reversion ({ma_window}d, {ma_type})"
                    strategy_variations[display_name] = "mean_reversion"
                    strategy_params[display_name] = {
                        'ma_window': ma_window,
                        'ma_type': ma_type,
                        'deviation_threshold': deviation_threshold,
                        'max_scale_up': max_scale_up,
                        'max_scale_down': max_scale_down
                    }
            
            # Add variation button - improved styling
            if st.session_state.ma_variations < 3:
                st.button("Add MA Variation", key="add_ma", 
                         on_click=lambda: setattr(st.session_state, 'ma_variations', st.session_state.ma_variations + 1),
                         use_container_width=True)

        # Add standard strategy if selected
        if "standard" in base_strategies:
            strategy_variations["Standard DCA"] = "standard"

    if st.button("Run Simulation", key="run_sim", help="Click to run the simulation"):
        with st.spinner('Running simulation...'):
            # Create logger
            logger = InvestmentLogger()
            
            # Run simulation with variations
            simulation_results = {}
            for variation_name, base_strategy in strategy_variations.items():
                params = strategy_params.get(variation_name, {})
                
                # Run single strategy simulation
                result = simulate_dca_strategies(
                    data,
                    [base_strategy],  # Pass as single-item list
                    investment_amount=monthly_investment,
                    investment_frequency='BMS',
                    initial_investment=initial_investment,
                    logger=logger,
                    strategy_params={base_strategy: params}  # Wrap params for single strategy
                )
                
                # Extract the result for this variation
                simulation_results[variation_name] = result[base_strategy]

            # Compare performances
            performance_metrics = compare_performances(simulation_results)

            # Display results with variation names
            st.subheader("Performance Metrics")
            displayed_metrics = ['Total Return', 'Annualized Return', 'Total Gain (€)', 
                               'Final Portfolio Value', 'Total Invested']
            
            # Format and display metrics
            formatted_metrics = format_performance_metrics(performance_metrics[displayed_metrics])
            
            st.dataframe(
                formatted_metrics,
                hide_index=False,
                use_container_width=True
            )

            # Create and display interactive plots with variations
            portfolio_value_plot = create_interactive_portfolio_value_plot(simulation_results, st.session_state.symbol)
            investment_decisions_plot = create_interactive_investment_decisions_plot(simulation_results, st.session_state.symbol)

            # Display plots in tabs
            st.subheader("Visualization Results")
            tab1, tab2 = st.tabs(["Portfolio Value", "Investment Decisions"])

            with tab1:
                st.plotly_chart(portfolio_value_plot, use_container_width=True)

            with tab2:
                st.subheader("Investment Decisions")
                
                # Create a dictionary of logs by strategy variation
                logs_by_strategy = {}
                for variation_name, base_strategy in strategy_variations.items():
                    # Map strategy variations correctly to the base strategy logs
                    if base_strategy == "rsi":
                        logs_by_strategy[variation_name] = logger.get_logs_for_strategy("rsi")
                    elif base_strategy == "mean_reversion":
                        logs_by_strategy[variation_name] = logger.get_logs_for_strategy("mean_reversion")
                    else:
                        logs_by_strategy[variation_name] = logger.get_logs_for_strategy(base_strategy)
                
                if any(logs_by_strategy.values()):
                    display_strategy_logs(logs_by_strategy)
                else:
                    st.info("No investment decisions logged for any strategy")


if __name__ == "__main__":
    main()
