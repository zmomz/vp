# plotting.py
import streamlit as st
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from config import PER_PAGE
from helper_functions import format_number


def plot_equity_curve_plotly(eq: pd.Series) -> go.Figure:
    """
    Given an equity-over-time series (indexed by datetime),
    return a Plotly Figure plotting it.
    """
    # Ensure the index is datetime
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = pd.to_datetime(eq.index)
    
    # Resample to daily if needed for smoother plotting
    if len(eq) > 1000:
        eq = eq.resample('D').last().ffill()
    
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=eq.index,
            y=eq.values,
            mode="lines",
            name="Equity",
            line=dict(color='royalblue', width=2)
        )
    )
    fig.update_layout(
        title="Equity Curve",
        xaxis_title="Time",
        yaxis_title="Equity ($)",
        margin=dict(l=40, r=40, t=50, b=40),
        height=500,
        hovermode="x unified"
    )
    
    # Add markers for trades if available
    if hasattr(st.session_state, 'backtest_stats') and hasattr(st.session_state.backtest_stats, '_trades'):
        trades = st.session_state.backtest_stats._trades
        if not trades.empty:
            trades['EntryTime'] = pd.to_datetime(trades['EntryTime'])
            fig.add_trace(
                go.Scatter(
                    x=trades['EntryTime'],
                    y=eq.reindex(trades['EntryTime']).values,
                    mode="markers",
                    name="Trades",
                    marker=dict(
                        color=['green' if x > 0 else 'red' for x in trades['ReturnPct']],
                        size=8,
                        line=dict(width=1, color='DarkSlateGrey')
                    )
                )
            )
    
    return fig

def plot_trade_analysis(stats, trades_df=None):
    """Plot trade analysis charts"""
    if trades_df is None:
        trades = stats._trades.copy()
    else:
        trades = trades_df.copy()
        
    # Create subplot layout
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            "Monthly Returns", 
            "Risk-Reward Ratio", 
            "Trade Duration", 
            "Return by Entry Hour"
        ),
        specs=[
            [{"type": "bar"}, {"type": "scatter"}],
            [{"type": "histogram"}, {"type": "bar"}]
        ]
    )
    
    # Monthly returns
    if 'EntryTime' in trades.columns:
        trades['EntryMonth'] = trades['EntryTime'].dt.strftime('%Y-%m')
        monthly = trades.groupby('EntryMonth')['ReturnPct'].sum()
        
        fig.add_trace(
            go.Bar(
                x=monthly.index, 
                y=monthly.values,
                marker_color=['green' if x > 0 else 'red' for x in monthly.values],
                name="Monthly Return"
            ),
            row=1, col=1
        )
    
    # Risk-reward scatter
    fig.add_trace(
        go.Scatter(
            x=trades["ReturnPct"], 
            y=trades["ReturnPct"] / trades["ReturnPct"].abs(),
            mode="markers",
            marker=dict(
                size=8,
                color=trades["ReturnPct"],
                colorscale="RdYlGn",
                line=dict(width=1, color='DarkSlateGrey')
            ),
            name="Risk-Reward"
        ),
        row=1, col=2
    )
    
    # Trade duration
    if 'Duration' in trades.columns:
        fig.add_trace(
            go.Histogram(
                x=trades['Duration'].dt.total_seconds() / 3600,  # hours
                nbinsx=20,
                marker_color='lightskyblue',
                name="Duration (hours)"
            ),
            row=2, col=1
        )
    
    # Return by hour
    if 'EntryTime' in trades.columns:
        trades['EntryHour'] = trades['EntryTime'].dt.hour
        hourly = trades.groupby('EntryHour')['ReturnPct'].mean().reindex(range(24))
        
        fig.add_trace(
            go.Bar(
                x=hourly.index, 
                y=hourly.values,
                marker_color=['green' if x > 0 else 'red' for x in hourly.values],
                name="Hourly Return"
            ),
            row=2, col=2
        )
    
    fig.update_layout(
        height=600,
        showlegend=False,
        margin=dict(l=40, r=40, t=60, b=40),
    )
    
    fig.update_xaxes(title_text="Month", row=1, col=1)
    fig.update_xaxes(title_text="Return %", row=1, col=2)
    fig.update_xaxes(title_text="Duration (hours)", row=2, col=1)
    fig.update_xaxes(title_text="Entry Hour", row=2, col=2, tickvals=list(range(0, 24, 3)))
    
    fig.update_yaxes(title_text="Return %", row=1, col=1)
    fig.update_yaxes(title_text="Win/Loss", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Avg Return %", row=2, col=2)
    
    return fig

def display_stats_dashboard(stats, data_df, session_defs, bt=None):
    """Clean, readable performance stats dashboard"""
    st.subheader("📊 Strategy Performance Summary")


    # Organize metrics in categories
    summary = {
        "General": {
            "Start": stats['Start'],
            "End": stats['End'],
            "Duration": stats['Duration'],
            "Exposure [%]": stats['Exposure Time [%]'],
            "Equity Final [$]": stats['Equity Final [$]'],
            "Equity Peak [$]": stats['Equity Peak [$]'],
            "Commissions [$]": stats['Commissions [$]'],
            "Buy & Hold Return [%]": stats['Buy & Hold Return [%]'],
        },
        "Returns & Risk": {
            "Return [%]": stats['Return [%]'],
            "CAGR [%]": stats['CAGR [%]'],
            "Volatility [%]": stats['Volatility (Ann.) [%]'],
            "Sharpe Ratio": stats['Sharpe Ratio'],
            "Sortino Ratio": stats['Sortino Ratio'],
            "Calmar Ratio": stats['Calmar Ratio'],
            "Alpha [%]": stats['Alpha [%]'],
            "Beta": stats['Beta'],
        },
        "Drawdowns": {
            "Max Drawdown [%]": stats['Max. Drawdown [%]'],
            "Avg Drawdown [%]": stats['Avg. Drawdown [%]'],
            "Max Drawdown Duration": stats['Max. Drawdown Duration'],
            "Avg Drawdown Duration": stats['Avg. Drawdown Duration'],
        },
        "Trade Stats": {
            "Win Rate [%]": stats['Win Rate [%]'],
            "Best Trade [%]": stats['Best Trade [%]'],
            "Worst Trade [%]": stats['Worst Trade [%]'],
            "Avg Trade [%]": stats['Avg. Trade [%]'],
            "Max Trade Duration": stats['Max. Trade Duration'],
            "Avg Trade Duration": stats['Avg. Trade Duration'],
            "Profit Factor": stats['Profit Factor'],
            "Expectancy [%]": stats['Expectancy [%]'],
            "SQN": stats['SQN'],
            "Kelly Criterion": stats['Kelly Criterion'],
        }
    }

    # Render each section in its own expandable block
    for section, metrics in summary.items():
        with st.expander(f"📂 {section}", expanded=True):
            df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
            df["Value"] = df["Value"].apply(lambda v: format_number(v) if isinstance(v, (int, float)) else str(v))
            st.dataframe(df, use_container_width=True)

def plot_strategy_visualization(strategy, data_df, session_defs, start_date=None, end_date=None):
    """
    Create a visualization of the volume profile strategy showing trades and session profiles.
    """
    # Filter data if date range provided
    if start_date or end_date:
        idx_dates = data_df.index.date
        mask = True
        if start_date:
            mask &= (idx_dates >= start_date)
        if end_date:
            mask &= (idx_dates <= end_date)
        data_df = data_df[mask].copy()
        
    # Get trade log from strategy
    if hasattr(strategy, 'trade_log') and strategy.trade_log:
        trades = pd.DataFrame(strategy.trade_log)
        # Convert any Timedelta objects to strings
        for col in trades.columns:
            if trades[col].dtype == 'timedelta64[ns]' or trades[col].apply(lambda x: isinstance(x, pd.Timedelta)).any():
                trades[col] = trades[col].astype(str)
    else:
        trades = pd.DataFrame()
    
    if not hasattr(strategy, 'assign_sessions') or not hasattr(strategy, 'profiles'):
        st.error("Strategy object missing required methods or attributes")
        return []
    
    try:
        df_labeled = strategy.assign_sessions(data_df.copy())
    except Exception as e:
        st.error(f"Error assigning sessions: {e}")
        return []
    
    # Create session groups
    session_groups = []
    for (session_date, session_name), group in df_labeled.groupby(['session_date', 'session_name']):
        if len(group) > 0:
            session_groups.append((session_date, session_name, group))
    
    if not session_groups:
        st.warning("No session data available for visualization")
        return []
    
    # Determine how many sessions to show per figure
    sessions_per_figure = min(3, len(session_groups))
    figures = []
    
    # Process sessions in batches
    for i in range(0, len(session_groups), sessions_per_figure):
        batch = session_groups[i:i+sessions_per_figure]
        
        # Create subplots with increased width
        fig = make_subplots(
            rows=len(batch), 
            cols=1,
            shared_xaxes=False,
            vertical_spacing=0.1,
            subplot_titles=[f"Session {s_name} - {s_date.strftime('%Y-%m-%d')}" 
                          for s_date, s_name, _ in batch],
            row_heights=[1] * len(batch)
        )
        
        for idx, (session_date, session_name, session_data) in enumerate(batch, 1):
            # Plot price candlesticks
            fig.add_trace(
                go.Candlestick(
                    x=session_data.index,
                    open=session_data['Open'],
                    high=session_data['High'],
                    low=session_data['Low'],
                    close=session_data['Close'],
                    name="Price",
                    opacity=0.7
                ),
                row=idx, col=1
            )
            
            # Get profile for this session
            profile_key = (session_date, session_name)
            profile_data = strategy.profiles.get(profile_key)
            
            if profile_data:
                # Add volume profile visualization
                vol_profile = profile_data['vp']
                max_vol = vol_profile['Volume'].max() if not vol_profile.empty else 0
                
                if max_vol > 0:
                    for price, volume in vol_profile.reset_index().values:
                        width = volume / max_vol * 0.3
                        fig.add_shape(
                            type="rect", 
                            x0=session_data.index[0],
                            x1=session_data.index[0] + pd.Timedelta(seconds=width * 3600),
                            y0=float(price) - 0.0001,
                            y1=float(price) + 0.0001,
                            fillcolor="rgba(100,100,100,0.3)",
                            line=dict(width=0),
                            layer="below",
                            row=idx, col=1
                        )
                
                # Add VAL, POC, VAH lines
                val, poc, vah = profile_data.get('val'), profile_data.get('poc'), profile_data.get('vah')
                
                if val is not None:
                    fig.add_shape(
                        type="line",
                        x0=session_data.index[0],
                        x1=session_data.index[-1],
                        y0=float(val), y1=float(val),
                        line=dict(color="green", width=1, dash="dash"),
                        row=idx, col=1
                    )
                    fig.add_annotation(
                        x=session_data.index[0], y=float(val),
                        text="VAL", showarrow=False,
                        xanchor="left", yanchor="bottom",
                        row=idx, col=1
                    )
                
                if poc is not None:
                    fig.add_shape(
                        type="line",
                        x0=session_data.index[0],
                        x1=session_data.index[-1],
                        y0=float(poc), y1=float(poc),
                        line=dict(color="red", width=1, dash="dash"),
                        row=idx, col=1
                    )
                    fig.add_annotation(
                        x=session_data.index[0], y=float(poc),
                        text="POC", showarrow=False,
                        xanchor="left", yanchor="bottom",
                        row=idx, col=1
                    )
                
                if vah is not None:
                    fig.add_shape(
                        type="line",
                        x0=session_data.index[0],
                        x1=session_data.index[-1],
                        y0=float(vah), y1=float(vah),
                        line=dict(color="green", width=1, dash="dash"),
                        row=idx, col=1
                    )
                    fig.add_annotation(
                        x=session_data.index[0], y=float(vah),
                        text="VAH", showarrow=False,
                        xanchor="left", yanchor="bottom",
                        row=idx, col=1
                    )
            
            # Plot trades for this session
            if not trades.empty and 'session_date' in trades.columns and 'session_name' in trades.columns:
                session_date_str = session_date.strftime('%Y-%m-%d') if isinstance(trades['session_date'].iloc[0], str) else session_date
                session_trades = trades[
                    (trades['session_date'] == session_date_str) & 
                    (trades['session_name'] == session_name)
                ]
                
                for _, trade in session_trades.iterrows():
                    # Entry marker
                    fig.add_trace(
                        go.Scatter(
                            x=[trade['timestamp']],
                            y=[float(trade['entry'])],
                            mode="markers",
                            marker=dict(
                                symbol="triangle-up" if trade['type'] == "LONG" else "triangle-down",
                                size=10,
                                color="green" if trade['type'] == "LONG" else "red",
                                line=dict(width=1, color="black")
                            ),
                            name=f"{trade['type']} Entry",
                            text=f"{trade['type']} @ {float(trade['entry']):.5f}",
                            hoverinfo="text"
                        ),
                        row=idx, col=1
                    )
                    
                    # Stop loss and target levels
                    fig.add_shape(
                        type="line",
                        x0=trade['timestamp'],
                        x1=session_data.index[-1],
                        y0=float(trade['stop']), y1=float(trade['stop']),
                        line=dict(color="red", width=1, dash="dot"),
                        row=idx, col=1
                    )
                    fig.add_shape(
                        type="line",
                        x0=trade['timestamp'],
                        x1=session_data.index[-1],
                        y0=float(trade['target']), y1=float(trade['target']),
                        line=dict(color="green", width=1, dash="dot"),
                        row=idx, col=1
                    )
                    
                    # Labels for stop and target
                    fig.add_annotation(
                        x=trade['timestamp'], y=float(trade['stop']),
                        text="SL", showarrow=False,
                        xanchor="left", yanchor="bottom",
                        row=idx, col=1
                    )
                    fig.add_annotation(
                        x=trade['timestamp'], y=float(trade['target']),
                        text="TP", showarrow=False,
                        xanchor="left", yanchor="bottom",
                        row=idx, col=1
                    )
        
        # Update layout with increased width
        fig.update_layout(
            title=f"Volume Profile Strategy - Sessions {batch[0][0].strftime('%Y-%m-%d')} to {batch[-1][0].strftime('%Y-%m-%d')}",
            height=700 * len(batch),
            width=1200,  # Increased width
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        
        # Update axes
        for idx in range(1, len(batch) + 1):
            fig.update_yaxes(
                title_text="Price", 
                row=idx, col=1,
                tickformat=".5f"
            )
            fig.update_xaxes(
                title_text="Time",
                row=idx, col=1
            )
            
        figures.append(fig)
    
    return figures

def display_strategy_visualization(strategy, data_df, session_defs, page: int):
    """Paginated visualization with trades properly displayed on candlesticks"""
    st.subheader("Strategy Visualization")

    # Assign sessions
    try:
        df_labeled = strategy.assign_sessions(data_df.copy())
    except Exception as e:
        st.error(f"Error assigning sessions: {e}")
        return

    # Get trade log if available
    trades = pd.DataFrame(strategy.trade_log) if hasattr(strategy, 'trade_log') else pd.DataFrame()

    # Group into (date, name) buckets
    groups = [
        ((d, n), grp)
        for (d, n), grp in df_labeled.groupby(['session_date', 'session_name'])
    ]
    if not groups:
        st.warning("No session data available for visualization")
        return

    # Pagination
    total = len(groups)
    max_page = (total - 1) // PER_PAGE + 1
    if page < 1 or page > max_page:
        st.error(f"Page must be between 1 and {max_page}")
        return

    start = (page - 1) * PER_PAGE
    batch = groups[start : start + PER_PAGE]

    # Define consistent styling
    PROFILE_STYLES = {
        'VAL': {'color': '#4E79A7', 'dash': 'dash', 'label': 'VAL (Value Area Low)'},
        'POC': {'color': '#E15759', 'dash': 'solid', 'label': 'POC (Point of Control)'},
        'VAH': {'color': '#59A14F', 'dash': 'dash', 'label': 'VAH (Value Area High)'}
    }

    TRADE_STYLES = {
        'entry': {
            'LONG': {'symbol': 'triangle-up', 'color': 'green', 'size': 12},
            'SHORT': {'symbol': 'triangle-down', 'color': 'red', 'size': 12}
        },
        'exit': {
            'profit': {'symbol': 'circle', 'color': 'limegreen', 'size': 10},
            'loss': {'symbol': 'x', 'color': 'orangered', 'size': 12}
        }
    }

    # Plot each session in the batch
    for (session_date, session_name), group in batch:
        fig = go.Figure()  # Using single figure instead of subplots for better trade visibility

        # Price candles - must be first trace
        fig.add_trace(
            go.Candlestick(
                x=group.index,
                open=group['Open'],
                high=group['High'],
                low=group['Low'],
                close=group['Close'],
                name="Price",
                increasing_line_color='#2ECC71',  # Green
                decreasing_line_color='#E74C3C'   # Red
            )
        )

        # Volume-profile shapes and lines
        profile = strategy.profiles.get((session_date, session_name))
        if profile:
            # Volume profile visualization
            vp = profile['vp']
            max_vol = vp['Volume'].max() if not vp.empty else 0
            if max_vol > 0:
                for price, vol in vp.reset_index().values:
                    width = vol / max_vol * 0.3
                    fig.add_shape(
                        type="rect",
                        x0=group.index[0],
                        x1=group.index[0] + pd.Timedelta(seconds=width * 3600),
                        y0=float(price) - 1e-4,
                        y1=float(price) + 1e-4,
                        fillcolor="rgba(100,100,100,0.3)",
                        line_width=0,
                        layer="below"
                    )
            
            # Profile lines (VAL, POC, VAH)
            for key in ['val', 'poc', 'vah']:
                value = profile.get(key)
                if value is not None:
                    label_type = key.upper()
                    style = PROFILE_STYLES[label_type]
                    
                    fig.add_shape(
                        type="line",
                        x0=group.index[0], 
                        x1=group.index[-1],
                        y0=float(value), 
                        y1=float(value),
                        line=dict(color=style['color'], dash=style['dash'], width=2),
                        name=style['label'],
                        layer="below"
                    )
                    
                    fig.add_annotation(
                        x=group.index[0],
                        y=float(value),
                        text=f"<b>{style['label']}</b><br>{float(value):.5f}",
                        showarrow=True,
                        arrowhead=1,
                        arrowsize=1,
                        arrowwidth=2,
                        arrowcolor=style['color'],
                        ax=20,
                        ay=0,
                        xanchor='left',
                        yanchor='middle',
                        bordercolor=style['color'],
                        borderwidth=1,
                        borderpad=4,
                        bgcolor="rgba(255,255,255,0.8)",
                        opacity=0.9
                    )

        # Add trades if they exist for this session
        if not trades.empty:
            # Convert session_date to match trade log format
            session_date_str = session_date.strftime('%Y-%m-%d')
            session_trades = trades[
                (trades['session_date'] == session_date_str) & 
                (trades['session_name'] == session_name)
            ]
            
            for _, trade in session_trades.iterrows():
                entry_time = pd.to_datetime(trade['entry_time'])
                exit_time = pd.to_datetime(trade['exit_time'])
                
                # Only plot trades that fall within this session's time range
                if entry_time >= group.index[0] and exit_time <= group.index[-1]:
                    # Trade Entry Marker
                    entry_style = TRADE_STYLES['entry'][trade['type']]
                    fig.add_trace(
                        go.Scatter(
                            x=[entry_time],
                            y=[float(trade['entry_price'])],
                            mode="markers+text",
                            marker=dict(
                                symbol=entry_style['symbol'],
                                size=entry_style['size'],
                                color=entry_style['color'],
                                line=dict(width=1, color='black')
                            ),
                            text=["ENTRY"],
                            textposition="top center",
                            textfont=dict(size=10, color='black'),
                            name=f"{trade['type']} Entry",
                            hoverinfo="text",
                            hovertext=(
                                f"<b>{trade['type']} Entry</b><br>"
                                f"Price: {float(trade['entry_price']):.5f}<br>"
                                f"Time: {entry_time.strftime('%H:%M:%S')}<br>"
                                f"Stop: {float(trade['stop_loss']):.5f}<br>"
                                f"Target: {float(trade['take_profit']):.5f}"
                            ),
                            showlegend=False
                        )
                    )
                    
                    # Trade Exit Marker
                    exit_type = 'profit' if float(trade['exit_price']) >= float(trade['entry_price']) else 'loss'
                    exit_style = TRADE_STYLES['exit'][exit_type]
                    fig.add_trace(
                        go.Scatter(
                            x=[exit_time],
                            y=[float(trade['exit_price'])],
                            mode="markers+text",
                            marker=dict(
                                symbol=exit_style['symbol'],
                                size=exit_style['size'],
                                color=exit_style['color'],
                                line=dict(width=1, color='black')
                            ),
                            text=["EXIT"],
                            textposition="bottom center",
                            textfont=dict(size=10, color='black'),
                            name=f"{trade['type']} Exit",
                            hoverinfo="text",
                            hovertext=(
                                f"<b>{trade['type']} Exit</b><br>"
                                f"Price: {float(trade['exit_price']):.5f}<br>"
                                f"Time: {exit_time.strftime('%H:%M:%S')}<br>"
                                f"P/L: {float(trade['pnl']):.2f} {trade['pnl_currency']}"
                            ),
                            showlegend=False
                        )
                    )
                    
                    # Draw line connecting entry and exit
                    fig.add_trace(
                        go.Scatter(
                            x=[entry_time, exit_time],
                            y=[float(trade['entry_price']), float(trade['exit_price'])],
                            mode="lines",
                            line=dict(
                                color='green' if exit_type == 'profit' else 'red',
                                width=1,
                                dash='dot'
                            ),
                            hoverinfo="none",
                            showlegend=False
                        )
                    )
                    
                    # Add stop loss and take profit lines
                    for level, label, color in [
                        (trade['stop_loss'], "SL", "red"),
                        (trade['take_profit'], "TP", "green")
                    ]:
                        fig.add_shape(
                            type="line",
                            x0=entry_time,
                            x1=exit_time,
                            y0=float(level),
                            y1=float(level),
                            line=dict(color=color, width=1, dash="dash"),
                            opacity=0.7,
                            name=f"{label} {float(level):.5f}"
                        )

        # Update layout
        fig.update_layout(
            title=f"{session_name} Session - {session_date.strftime('%Y-%m-%d')}",
            height=700,
            width=1200,
            xaxis_rangeslider_visible=False,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                title_text="Elements:"
            ),
            hovermode="x unified",
            margin=dict(l=50, r=50, t=100, b=50)
        )
        
        # Add explanatory notes
        fig.add_annotation(
            x=0.5,
            y=1.1,
            xref="paper",
            yref="paper",
            text=(
                "<b>Trade Markers:</b> ▲ = Long Entry | ▼ = Short Entry | ○ = Profit Exit | × = Loss Exit<br>"
                "<b>Profile Lines:</b> VAL = Value Area Low | POC = Point of Control | VAH = Value Area High"
            ),
            showarrow=False,
            font=dict(size=11)
        )
        
        st.plotly_chart(fig, use_container_width=True)
