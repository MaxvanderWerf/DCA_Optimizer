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
        xaxis_range=[min_date, max_date]
    )
    return fig


def create_interactive_investment_decisions_plot(simulation_results, identifier):
    fig = go.Figure()
    for strategy, result in simulation_results.items():
        fig.add_trace(go.Scatter(x=result.index, y=result['Invested'],
                                 mode='lines+markers', name=f'{strategy} - Invested'))
        fig.add_trace(go.Scatter(x=result.index, y=result['Cash_Balance'],
                                 mode='lines', name=f'{strategy} - Cash Balance', line=dict(dash='dash')))
    fig.update_layout(title=f'Investment Decisions Over Time - {identifier}',
                      xaxis_title='Date', yaxis_title='Amount ($)')
    return fig


def set_custom_style():
    st.markdown("""
        <style>
        /* Main theme colors */
        .stApp {
            background-color: #0A192F;
        }
        
        /* Headers */
        .main-header {
            font-family: 'Calibre', sans-serif;
            font-size: 42px;
            font-weight: 600;
            color: #CCD6F6;
            margin-bottom: 30px;
        }
        
        /* Metric cards */
        .metric-card {
            background-color: #112240;
            padding: 20px;
            border-radius: 4px;
            border: 1px solid #1E3A5F;
        }
        
        /* Plots and DataFrames */
        .stPlotlyChart, .stDataFrame {
            background-color: #112240;
            border: 1px solid #1E3A5F;
            border-radius: 4px;
            padding: 10px;
        }
        
        /* Metrics */
        div[data-testid="stMetricValue"] {
            font-size: 24px;
            color: #64FFDA;
        }
        
        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #112240;
            border-right: 1px solid #1E3A5F;
            padding: 1rem;
            width: 300px !important;
        }
        
        section[data-testid="stSidebar"] .block-container {
            padding-top: 2rem;
            font-size: 0.9rem;
        }
        
        section[data-testid="stSidebar"] h1, 
        section[data-testid="stSidebar"] h2, 
        section[data-testid="stSidebar"] h3 {
            font-size: 1.1rem;
            font-weight: 600;
            margin: 0 0 1rem 0;
        }
        
        section[data-testid="stSidebar"] .stTextInput input {
            font-size: 0.9rem;
        }
        
        section[data-testid="stSidebar"] .stNumberInput input {
            font-size: 0.9rem;
        }
        
        section[data-testid="stSidebar"] .stDateInput input {
            font-size: 0.9rem;
        }
        
        /* View Popular button */
        section[data-testid="stSidebar"] [data-testid="baseButton-secondary"] {
            background: none !important;
            border: none !important;
            color: #64FFDA !important;
            padding: 0 !important;
            font-size: 0.85rem !important;
            min-height: 0 !important;
            margin: 0 !important;
            transform: translateY(8px);
        }
        
        section[data-testid="stSidebar"] [data-testid="baseButton-secondary"]:hover {
            color: #8892B0 !important;
            box-shadow: none !important;
        }
        
        /* Caption text */
        section[data-testid="stSidebar"] .stCaption {
            font-size: 0.8rem !important;
            color: #8892B0 !important;
        }
        
        /* Success message */
        section[data-testid="stSidebar"] .stSuccess {
            font-size: 0.85rem !important;
            padding: 0.5rem !important;
            margin: 0.5rem 0 !important;
        }
        
        /* Metrics in sidebar */
        section[data-testid="stSidebar"] [data-testid="stMetricValue"] {
            font-size: 1.1rem !important;
        }
        
        section[data-testid="stSidebar"] [data-testid="stMetricLabel"] {
            font-size: 0.85rem !important;
        }
        
        /* Regular Buttons */
        .stButton button {
            background-color: transparent;
            color: #8892B0;
            border: 1px solid #8892B0;
            border-radius: 4px;
            padding: 8px 16px;
            transition: all 0.2s ease;
            margin: 0 4px;  /* Add spacing between buttons */
            white-space: nowrap;  /* Prevent text wrapping */
            min-width: fit-content;  /* Ensure button fits text */
        }
        
        .stButton button:hover {
            background-color: rgba(136, 146, 176, 0.1);
            border-color: #CCD6F6;
            color: #CCD6F6;
        }
        
        /* Run Simulation Button - Special styling */
        .run-simulation-btn {
            background-color: transparent !important;
            color: #64FFDA !important;
            border: 1px solid #64FFDA !important;
            border-radius: 4px !important;
            padding: 12px 24px !important;
            font-size: 16px !important;
            font-weight: 500 !important;
            transition: all 0.2s ease !important;
        }
        
        .run-simulation-btn:hover {
            background-color: rgba(100, 255, 218, 0.1) !important;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(100, 255, 218, 0.1);
        }
        
        /* Text inputs - specifically for symbol and investment amounts */
        .stTextInput input:not([type="date"]), 
        .stNumberInput input {
            background-color: #0A192F !important;
            border: 1px solid #1E3A5F;
            color: #CCD6F6;
            border-radius: 4px;
        }
        
        /* Keep date inputs with dark blue background */
        input[type="date"] {
            background-color: #112240 !important;
            border: 1px solid #1E3A5F;
            color: #CCD6F6;
            border-radius: 4px;
        }
        
        /* Secondary text */
        .secondary-text {
            color: #8892B0;
        }
        
        /* Tabs styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
            background-color: #112240;
        }
        
        .stTabs [data-baseweb="tab"] {
            color: #8892B0;
            border: 1px solid #1E3A5F;
            border-radius: 4px;
            padding: 8px 16px;  /* Add padding for better text spacing */
            margin: 0 4px;      /* Add margin between tabs */
            min-width: fit-content;  /* Ensure tab fits text */
            white-space: nowrap;     /* Prevent text wrapping */
        }
        
        .stTabs [data-baseweb="tab"]:hover {
            color: #64FFDA;
        }
        
        .stTabs [aria-selected="true"] {
            color: #64FFDA;
            border-color: #64FFDA;
        }
        
        /* MultiSelect */
        .stMultiSelect {
            background-color: #112240;
            border-radius: 4px;
        }
        
        .stMultiSelect [data-baseweb="tag"] {
            background-color: #1E3A5F;
            border: none;
            color: #64FFDA;
            margin: 2px;  /* Add spacing between selected items */
            padding: 4px 8px;  /* Add padding for better text spacing */
        }
        
        /* Strategy selection buttons */
        div[data-testid="stHorizontalBlock"] button {
            margin: 0 4px;  /* Add spacing between strategy buttons */
            padding: 8px 16px;  /* Add padding for better text spacing */
        }
        
        [data-testid="stVerticalBlock"] div:has(div.status-emoji) {
            padding-top: 25px;
        }
        .status-emoji {
            font-size: 1.2rem;
            line-height: 1;
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
    """Display logs for all strategies side by side, aligned by date"""
    
    # Create a dictionary of logs indexed by date for each strategy
    dated_logs = {}
    all_dates = set()
    
    for strategy, logs in logs_by_strategy.items():
        dated_logs[strategy] = {log.date: log for log in logs}
        all_dates.update(dated_logs[strategy].keys())
    
    # Sort dates
    sorted_dates = sorted(all_dates, reverse=True)  # Most recent first
    
    # Create columns for each strategy
    cols = st.columns(len(logs_by_strategy))
    
    # Display strategy headers and descriptions
    for col, strategy in zip(cols, logs_by_strategy.keys()):
        with col:
            title = f"{strategy.replace('_', ' ').title()} Strategy"
            st.subheader(title)
            
            if "standard" in strategy.lower():
                st.markdown("*Invests the same amount monthly*")
            elif "mean_reversion" in strategy.lower():
                st.markdown("*Adjusts investment based on price trends*")
            elif "rsi" in strategy.lower():
                st.markdown("*Adjusts investment based on market momentum*")
    
    # Create the log text for each strategy
    strategy_texts = {strategy: [] for strategy in logs_by_strategy.keys()}
    
    # Build aligned log entries
    for date in sorted_dates:
        for strategy in logs_by_strategy.keys():
            if log := dated_logs[strategy].get(date):
                strategy_texts[strategy].append(format_decision_log(log))
            else:
                # Add placeholder for missing dates to maintain alignment
                date_str = date.strftime('%Y-%m-%d')
                strategy_texts[strategy].append(f"⚪️ {date_str}: No data")
    
    # Display logs in columns
    for col, strategy in zip(cols, logs_by_strategy.keys()):
        with col:
            log_text = "\n\n".join(strategy_texts[strategy])
            st.markdown(f"```\n{log_text}\n```")


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

    # Main page content
    strategies = st.multiselect(
        "Select strategies to simulate:",
        ["standard", "rsi", "mean_reversion"],
        default=["standard"],
        help="Choose one or more investment strategies to simulate and compare"
    )

    if st.button("Run Simulation", key="run_sim", help="Click to run the simulation"):
        with st.spinner('Running simulation...'):
            # Create logger
            logger = InvestmentLogger()
            
            # Run simulation with logger
            simulation_results = simulate_dca_strategies(data, strategies,
                                                       investment_amount=monthly_investment,
                                                       investment_frequency='BMS',
                                                       initial_investment=initial_investment,
                                                       logger=logger)

            # Compare performances
            performance_metrics = compare_performances(simulation_results)

            # Display results
            st.subheader("Performance Metrics")
            displayed_metrics = ['Total Return', 'Annualized Return', 'Total Gain (€)', 
                                'Final Portfolio Value', 'Total Invested']
            
            # Format and display metrics
            formatted_metrics = format_performance_metrics(performance_metrics[displayed_metrics])
            
            # Add custom CSS to style the dataframe
            st.markdown("""
                <style>
                .dataframe {
                    font-size: 16px !important;
                }
                .dataframe td {
                    text-align: right !important;
                    padding: 8px 12px !important;
                }
                .dataframe th {
                    text-align: left !important;
                    padding: 8px 12px !important;
                    font-weight: 600 !important;
                }
                </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                formatted_metrics,
                hide_index=False,
                use_container_width=True
            )

            # Create interactive plots
            portfolio_value_plot = create_interactive_portfolio_value_plot(simulation_results, st.session_state.symbol)
            investment_decisions_plot = create_interactive_investment_decisions_plot(simulation_results, st.session_state.symbol)

            # Display plots in tabs
            st.subheader("Visualization Results")
            tab1, tab2 = st.tabs(["Portfolio Value", "Investment Decisions"])

            with tab1:
                st.plotly_chart(portfolio_value_plot, use_container_width=True)

            with tab2:
                st.subheader("Investment Decisions")
                
                # Create a dictionary of logs by strategy
                logs_by_strategy = {
                    strategy: logger.get_logs_for_strategy(strategy)
                    for strategy in strategies
                }
                
                if any(logs_by_strategy.values()):
                    display_strategy_logs(logs_by_strategy)
                else:
                    st.info("No investment decisions logged for any strategy")


if __name__ == "__main__":
    main()
