# streamlit_app.py
import os
import pickle
from typing import List
import streamlit as st
import pandas as pd
from backtesting import Backtest
import plotly.graph_objs as go
from datetime import datetime

from strategy import VolumeProfileStrategy
from config import (DEFAULT_SESSIONS, TIMEFRAMES, AVAILABLE_PAIRS, 
                    EPOCHS, STRATEGY_CONFIG, DATA_DIR, CONFIG_DIR, 
                    RESULTS_DIR, _INITIAL_STATE, PER_PAGE)
from ui_widgets import (session_config_ui, strategy_params_ui, 
                        config_management_ui)
from plotting import (display_stats_dashboard, plot_trade_analysis, 
                      display_strategy_visualization)
from helper_functions import build_equity_curve_from_trades


os.makedirs(DATA_DIR,   exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

for k, v in _INITIAL_STATE.items():
    if k not in st.session_state:
        st.session_state[k] = v
if st.session_state["session_defs"] is None:
    st.session_state["session_defs"] = DEFAULT_SESSIONS.copy()

# add a default for the *page* name
if "active_page" not in st.session_state:
    st.session_state["active_page"] = "Configure & Run"
# ─────────────────────────────────────────────────────────────────────────────
# 3) DATA LOADER
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data(pair: str, timeframe: str) -> pd.DataFrame:
    try:
        file_path = f"data/{pair}_{timeframe}.parquet"
        if not os.path.exists(file_path):
            st.error(f"Data file not found: {file_path}")
            return pd.DataFrame()

        df = pd.read_parquet(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# 8) BACKTEST CALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest():
    """Runs Backtest and saves results."""
    # clear any “loaded from file” state so the UI will show analysis panes
    st.session_state.pop("loaded_from_file", None)
    try:
        STRATEGY_CONFIG.update({
            'session_defs': st.session_state['session_defs'],
            'bucket_size_pips': st.session_state['bucket_size_pips'],
            'value_area_pct': st.session_state['value_area_pct'],
            'balanced_poc_center': st.session_state['balanced_poc_center'],
            'balanced_time_inside': st.session_state['balanced_time_inside'],
            'entry_bars': st.session_state['entry_bars'],
            'risk_per_trade': st.session_state['risk_per_trade'],
        })

        # Load and slice the data
        df = load_data(st.session_state["pair"], st.session_state["timeframe"])
        df = df.loc[st.session_state["start_date"] : st.session_state["end_date"]]
        total_bars = len(df)
        if total_bars == 0:
            st.error("No data available for the selected period.")
            return

        # Run backtest
        bt = Backtest(df, VolumeProfileStrategy, cash=10_000, commission=0.0002)

        status_text = st.empty()
        # INSERT: set up a Streamlit progress bar
        progress_text = st.empty()
        progress_bar = st.progress(0)

        # Monkey-patch VolumeProfileStrategy.next to update the bar,
        # using our own counter
        orig_next = VolumeProfileStrategy.next
        counter = {'i': 0}
        def next_with_progress(self, *args, **kwargs):
            result = orig_next(self, *args, **kwargs)
            counter['i'] += 1
            progress_text.text(f"Backtesting: {int(counter['i']/ total_bars*100)}%")
            progress_bar.progress(int(counter['i'] / total_bars*100))
            return result

        VolumeProfileStrategy.next = next_with_progress

        stats = bt.run()

        # Restore the original next()
        VolumeProfileStrategy.next = orig_next
        # once done, swap to a success message
        progress_text.empty()
        progress_bar.empty()
        status_text.success("Backtest complete!")
        # Get equity curve
        eq_series = getattr(stats, "_equity_curve", None)
        if eq_series is None or eq_series.empty:
            eq_series = build_equity_curve_from_trades(stats)

        # Prepare results for saving
        result_data = {
            "timestamp": datetime.now(),
            "summary": {
                "pair": st.session_state["pair"],
                "timeframe": st.session_state["timeframe"],
                "period": f"{st.session_state['start_date']} to {st.session_state['end_date']}",
                "return_pct": stats['Return [%]'],
                "sharpe": stats['Sharpe Ratio'],
                "max_dd": stats['Max. Drawdown [%]'],
                "win_rate": stats['Win Rate [%]'],
            },
            "parameters": {
                "pair": st.session_state["pair"],
                "timeframe": st.session_state["timeframe"],
                "start_date": st.session_state["start_date"],
                "end_date": st.session_state["end_date"],
                "session_defs": st.session_state["session_defs"],
                "bucket_size_pips": st.session_state["bucket_size_pips"],
                "value_area_pct": st.session_state["value_area_pct"],
                "balanced_poc_center": st.session_state["balanced_poc_center"],
                "balanced_time_inside": st.session_state["balanced_time_inside"],
                "entry_bars": st.session_state["entry_bars"],
                "risk_per_trade": st.session_state["risk_per_trade"],
            },
            "results": {
                "stats": stats,
                "equity_curve": eq_series,
                "trades": getattr(stats, "_trades", None),
            }
        }

        # Save results
        saved_filename = save_backtest_result(result_data)
        if saved_filename:
            st.session_state["last_saved_result"] = saved_filename

        # Update session state
        st.session_state["backtest_data"] = df
        st.session_state["backtest_stats"] = stats
        st.session_state["backtest_strategy"] = stats["_strategy"]  # This won't be saved
        st.session_state["backtest_run"] = True

        st.session_state["active_page"] = "Results"
    except Exception as e:
        st.error(f"Backtest failed: {e}")


# ─────────────────────────────────────────────────────────────────────────
# 9) SAVE RESULTS
# ─────────────────────────────────────────────────────────────────────────
def save_backtest_result(result_data: dict) -> str:
    """
    Save backtest results to a file (excluding the strategy object).
    result_data is expected to have keys:
      - 'timestamp'
      - 'summary'
      - 'parameters'
      - 'results': {
            'stats': backtesting.Stats,
            'equity_curve': pd.Series,
            'trades': pd.DataFrame
        }
    """
    try:
        # timestamp for filename
        ts = result_data["timestamp"].strftime("%Y%m%d_%H%M%S")
        pair = result_data["summary"]["pair"]
        tf   = result_data["summary"]["timeframe"]
        filename = f"{pair}_{tf}_{ts}.pkl"
        filepath = os.path.join(RESULTS_DIR, filename)

        # Build a pure‐Python dict, dropping the strategy class
        stats = result_data["results"]["stats"]
        trades_df = getattr(stats, "_trades", None)

        save_data = {
            "timestamp":    result_data["timestamp"],
            "summary":      result_data["summary"],
            "parameters":   result_data["parameters"],
            "results": {
                # convert Stats to dict
                "stats": dict(stats),
                # equity curve is already a Series; convert to list or dict
                "equity_curve": result_data["results"]["equity_curve"].to_dict(),
                # trades DataFrame → dict of lists
                "trades": trades_df.to_dict(orient="list") if trades_df is not None else None,
            }
        }

        # actually write
        with open(filepath, "wb") as f:
            pickle.dump(save_data, f)
        return filename

    except Exception as e:
        st.error(f"Error saving results: {e}")
        return ""

def load_backtest_result(filename: str) -> dict | None:
    """
    Load backtest results and rehydrate pandas objects.
    """
    try:
        filepath = os.path.join(RESULTS_DIR, filename)
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        # rebuild equity curve as Series
        eq_dict = data["results"]["equity_curve"]
        data["results"]["equity_curve"] = pd.Series(eq_dict)

        # rebuild trades DataFrame
        trades_dict = data["results"]["trades"]
        data["results"]["trades"] = pd.DataFrame(trades_dict) if trades_dict else None

        return data

    except Exception as e:
        st.error(f"Error loading results: {e}")
        return None

def get_saved_results() -> List[str]:
    """List all saved result filenames"""
    if not os.path.isdir(RESULTS_DIR):
        return []
    return sorted([
        fn for fn in os.listdir(RESULTS_DIR) 
        if fn.endswith(".pkl")
    ], reverse=True)

# ─────────────────────────────────────────────────────────────────────────
# TAB 1: Configure & Run
# ─────────────────────────────────────────────────────────────────────────
def show_configure_and_run():
    st.subheader("Data & Period")
    col1, col2 = st.columns(2)
    with col1:
        # Instrument selection with safe default index
        stored_pair = st.session_state.get("pair")
        default_pair = stored_pair if stored_pair in AVAILABLE_PAIRS else AVAILABLE_PAIRS[0]
        pair = st.selectbox(
            "Select instrument",
            AVAILABLE_PAIRS,
            index=AVAILABLE_PAIRS.index(default_pair),
        )
        st.session_state["pair"] = pair

        # Timeframe selection with safe default index
        stored_tf = st.session_state.get("timeframe")
        default_tf = stored_tf if stored_tf in TIMEFRAMES else TIMEFRAMES[1]
        tf = st.selectbox(
            "Select timeframe",
            TIMEFRAMES,
            index=TIMEFRAMES.index(default_tf),
        )
        st.session_state["timeframe"] = tf

    with col2:
        st.markdown("**Backtest Period**")
        period_mode = st.radio(
            "Backtest period selection",
            ["Year range", "Special periods"],
        )
        if period_mode == "Year range":
            sy = st.number_input("Start year", 2024, datetime.now().year, key="sy")
            ey = st.number_input("End year", 2025, datetime.now().year, key="ey")
            start_date = f"{sy}-01-01"
            end_date   = f"{ey}-12-31"
        else:
            epoch = st.selectbox("Epoch", list(EPOCHS.keys()))
            start_date, end_date = EPOCHS[epoch]

        st.session_state["start_date"] = start_date
        st.session_state["end_date"]   = end_date

    # Now actually load & preview the data
    with st.spinner(f"Loading {pair} {tf} data…"):
        df = load_data(pair, tf)
        if df.empty:
            st.error("No data file found.")
            return
        df_bt = df.loc[start_date:end_date].copy()
        if df_bt.empty:
            st.error("No data in that range.")
            return
        st.session_state["backtest_data"] = df_bt  # so you can preview here
        st.success(f"Loaded {len(df_bt):,} bars from {df_bt.index[0].date()} to {df_bt.index[-1].date()}")
        with st.expander("Data Sample"):
            st.dataframe(df_bt.head(10))

    st.markdown("---")
    st.subheader("Configure Sessions & Strategy Parameters")
    _session_defs = session_config_ui()
    _strat_params = strategy_params_ui()
    with st.expander("Save / Load Config"):
        config_management_ui()

    st.markdown("---")
    st.subheader("Run Backtest")
    st.button("🚀 Run Backtest", on_click=run_backtest, use_container_width=True)
    
    if st.session_state.backtest_run:
        st.success("Backtest completed! Switch to the Results tab.")

# ─────────────────────────────────────────────────────────────────────────
# TAB 2: Results
# ─────────────────────────────────────────────────────────────────────────
def show_results():
    st.subheader("Results")

# Section 1: Saved Results Management
    with st.expander("💾 Saved Results", expanded=True):
        saved_results = get_saved_results()

        if not saved_results:
            st.info("No saved results yet. Run a backtest to save results.")
        else:
            # Build a list of (label, filename) for the radio widget
            options: list[tuple[str,str]] = []
            for fn in saved_results:
                try:
                    with open(os.path.join(RESULTS_DIR, fn), "rb") as f:
                        data = pickle.load(f)
                    s = data.get("summary", {})
                    label = (
                        f"{fn} → {s.get('pair','')} "
                        f"{s.get('timeframe','')} "
                        f"({s.get('period','')}) "
                        f"R:{s.get('return_pct',0):.1f}%"
                    )
                except Exception:
                    label = fn
                options.append((label, fn))

            labels, fns = zip(*options)

            # Single-choice radio
            choice = st.radio("Select a result to act on", labels)
            selected_filename = fns[labels.index(choice)]

            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🔍 Load Selected Result"):
                    loaded = load_backtest_result(selected_filename)
                    if loaded:
                        for k, v in loaded["parameters"].items():
                            st.session_state[k] = v
                        st.session_state["backtest_stats"] = loaded["results"]["stats"]
                        st.session_state["backtest_run"] = True
                        # mark that we’re viewing a saved result
                        st.session_state["loaded_from_file"] = True
                        st.success("Parameters loaded from file. Analysis panels are now hidden.")
                        st.rerun()

            with col2:
                if st.button("🗑️ Delete Selected Result"):
                    try:
                        os.remove(os.path.join(RESULTS_DIR, selected_filename))
                        st.success(f"Deleted {selected_filename}")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error deleting: {e}")

            with col3:
                with open(os.path.join(RESULTS_DIR, selected_filename), "rb") as f:
                    st.download_button(
                        "📤 Export Selected Result",
                        data=f,
                        file_name=selected_filename,
                        mime="application/octet-stream"
                    )

    if not st.session_state.backtest_run:
        st.info("Run a backtest first.")
        return

    # 1) Stats summary
    display_stats_dashboard(
        st.session_state.backtest_stats,
        st.session_state.backtest_data,
        st.session_state.session_defs,
    )

    st.markdown("---")

    # if we just loaded from file, hide the Equity/Trade/Price views
    if st.session_state.get("loaded_from_file", False):
        st.info("🔒 You’re viewing a saved result; analysis panels are hidden.  Re-run a live backtest to re-enable.")
        return

    # 2) View selector
    view = st.radio(
        "Select view:", 
        ["Equity Curve", "Trade Analysis", "Price Chart"],
        horizontal=True,
        key="results_view"
    )

    if view == "Equity Curve":
        # Get the backtest stats and data
        stats = st.session_state.backtest_stats
        data_df = st.session_state.backtest_data
        
        # Build equity curve from trades
        if hasattr(stats, '_trades') and not stats._trades.empty:
            trades = stats._trades.copy()
            
            # Ensure EntryTime is datetime
            trades['EntryTime'] = pd.to_datetime(trades['EntryTime'])
            
            # Calculate cumulative returns
            trades['CumReturn'] = (1 + trades['ReturnPct']).cumprod()
            
            # Create a time series with all data points
            eq = pd.Series(10000.0, index=data_df.index)  # Start with initial equity
            
            # Update equity at each trade entry
            for _, trade in trades.iterrows():
                # Update equity from trade entry time onward
                eq.loc[trade['EntryTime']:] = 10000 * trade['CumReturn']
            
            # Forward fill any gaps
            eq = eq.ffill()
        else:
            # If no trades, just show flat line at starting equity
            eq = pd.Series([10000.0], index=[data_df.index[0]])
        
        # Plot the equity curve
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
        
        # Add markers for trades if available
        if hasattr(stats, '_trades') and not stats._trades.empty:
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
        
        fig.update_layout(
            title="Equity Curve",
            xaxis_title="Time",
            yaxis_title="Equity ($)",
            margin=dict(l=40, r=40, t=50, b=40),
            height=500,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig, use_container_width=True)

    elif view == "Trade Analysis":
        fig_tr = plot_trade_analysis(st.session_state.backtest_stats)
        if fig_tr:
            st.plotly_chart(fig_tr, use_container_width=True)

    else:  # Price Chart
        strat = st.session_state.backtest_strategy
        df   = st.session_state.backtest_data

        # figure out pagination
        df_labeled = strat.assign_sessions(df.copy())
        total_sessions = len(df_labeled.groupby(["session_date", "session_name"]))
        max_page = (total_sessions - 1) // PER_PAGE + 1

        page = st.slider("Session page:", 1, max_page, key="sess_page_slider")

        display_strategy_visualization(
            strat,
            df,
            st.session_state.session_defs,
            page=page
        )

# ─────────────────────────────────────────────────────────────────────────
# Tab 3: About
# ─────────────────────────────────────────────────────────────────────────
def show_about():
    st.header("About Volume Profile Strategy")
    st.markdown("""
    ### Strategy Overview
    
    This strategy analyzes volume profiles across different trading sessions to identify potential trading opportunities.
    
    #### Key Components:
    
    1. **Session-Based Volume Profiles**
        - Each trading session (A, B, C) has its own volume profile
        - Price levels are divided into buckets to analyze volume distribution
        
    2. **Value Area Identification**
        - Value Area Low (VAL) and Value Area High (VAH) contain the specified percentage of volume
        - Point of Control (POC) is the price level with the highest trading volume
        
    3. **Trading Rules**
        - Long entries: When session opens below VAL and price moves back into range
        - Short entries: When session opens above VAH and price moves back into range
        - Additional filters can be applied (MA crossover, etc.)
        
    4. **Risk Management**
        - Position sizing based on account risk percentage
        - Stop loss and take profit levels derived from volume profile
        
    ### How to Use
    
    1. Select your instrument and timeframe
    2. Configure your trading sessions (A, B, C)
    3. Adjust strategy parameters
    4. Run backtest and analyze results
    5. Save successful configurations for future use
    
    ### Tips for Optimization
    
    - Test different market environments
    - Adjust bucket size based on instrument volatility
    - Fine tune the value area percentage
    - Test different session definitions

    """)

# ─────────────────────────────────────────────────────────────────────────────
# 9) MAIN APPLICATION
# ─────────────────────────────────────────────────────────────────────────────
def main():
    st.set_page_config(page_title="VP Session Backtester", layout="wide")
    st.title("📊 Volume Profile Session Strategy Backtester")

    PAGES = ["Configure & Run", "Results", "About"]

    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Configure & Run"

    page = st.radio(
        "Navigate to:",
        PAGES,
        key="active_page",
        horizontal=True,
    )
 
    if page == "Configure & Run":
        show_configure_and_run()
    elif page == "Results":
        show_results()
    else:
        show_about()

if __name__ == "__main__":
    main()
