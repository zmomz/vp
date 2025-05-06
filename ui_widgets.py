import streamlit as st
from datetime import time, datetime
from config import DEFAULT_SESSIONS, TIMEFRAMES, AVAILABLE_PAIRS
from helper_functions import save_config, get_saved_configs, load_config

# Session configuration UI component
def session_config_ui():
    """UI for configuring trading sessions"""
    st.subheader("Trading Sessions Configuration")
    
    session_defs = st.session_state.get('session_defs', DEFAULT_SESSIONS.copy())
    
    # Session list with editing capability
    for i in range(len(session_defs)):
        col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 0.5])
        
        with col1:
            session_defs[i]['name'] = st.text_input(
                f"Name #{i+1}", 
                value=session_defs[i]['name'],
                key=f"sess_name_{i}"
            )
        
        with col2:
            session_defs[i]['start'] = st.time_input(
                f"Start #{i+1}",
                time.fromisoformat(session_defs[i]['start']),
                key=f"sess_start_{i}"
            ).strftime("%H:%M")
        
        with col3:
            session_defs[i]['end'] = st.time_input(
                f"End #{i+1}",
                time.fromisoformat(session_defs[i]['end']),
                key=f"sess_end_{i}"
            ).strftime("%H:%M")
        
        with col4:
            session_defs[i]['color'] = st.color_picker(
                f"Color #{i+1}",
                session_defs[i].get('color', 'rgba(100, 100, 100, 0.5)'),
                key=f"sess_color_{i}"
            )
        
        with col5:
            if st.button("✕", key=f"delete_sess_{i}"):
                session_defs.pop(i)
                st.session_state['session_defs'] = session_defs
                st.rerun()
    
    # Add new session button
    if st.button("+ Add Session"):
        session_defs.append({
            "name": f"Session {len(session_defs) + 1}",
            "start": "00:00",
            "end": "04:00",
            "color": "rgba(100, 100, 100, 0.5)"
        })
        st.session_state['session_defs'] = session_defs
        st.rerun()
    
    # Reset to default sessions
    if st.button("Reset to Default"):
        st.session_state['session_defs'] = DEFAULT_SESSIONS.copy()
        st.rerun()
    
    st.session_state['session_defs'] = session_defs
    return session_defs

# Strategy parameter UI component
def strategy_params_ui():
    """UI for configuring strategy parameters"""
    st.subheader("Strategy Parameters")
    
    # Create columns for more compact layout
    col1, col2 = st.columns(2)
    
    with col1:
        bucket_size_pips = st.number_input(
            "Bucket size (pips)", 
            min_value=1, 
            max_value=50, 
            value=st.session_state.get('bucket_size_pips', 5),
            help="Size of each price bucket for volume profile calculation"
        )
        st.session_state['bucket_size_pips'] = bucket_size_pips
        
        value_area_pct = st.slider(
            "Value Area %", 
            min_value=0.1, 
            max_value=1.0, 
            value=st.session_state.get('value_area_pct', 0.7),
            format="%.2f",
            help="Percentage of volume to include in value area"
        )
        st.session_state['value_area_pct'] = value_area_pct
        
        entry_bars = st.slider(
            "Consecutive entry bars", 
            min_value=1, 
            max_value=10, 
            value=st.session_state.get('entry_bars', 2),
            help="Number of consecutive bars needed for entry confirmation"
        )
        st.session_state['entry_bars'] = entry_bars
    
    with col2:
        balanced_poc_center = st.slider(
            "POC center tolerance", 
            min_value=0.0, 
            max_value=0.5, 
            value=st.session_state.get('balanced_poc_center', 0.25),
            format="%.2f",
            help="Maximum distance of POC from center for balanced profile"
        )
        st.session_state['balanced_poc_center'] = balanced_poc_center
        
        balanced_time_inside = st.slider(
            "Time inside VA %", 
            min_value=0.0, 
            max_value=1.0, 
            value=st.session_state.get('balanced_time_inside', 0.7),
            format="%.2f",
            help="Minimum percentage of time price must spend inside value area for balanced profile"
        )
        st.session_state['balanced_time_inside'] = balanced_time_inside
        
        risk_per_trade = st.slider(
            "Risk per trade %", 
            min_value=1.0, 
            max_value=10.0, 
            value=st.session_state.get('risk_per_trade', 5.0),
            format="%.1f",
            help="Percentage of account risked per trade"
        )
        st.session_state['risk_per_trade'] = risk_per_trade
    

    return {
        'bucket_size_pips': bucket_size_pips,
        'value_area_pct': value_area_pct,
        'balanced_poc_center': balanced_poc_center,
        'balanced_time_inside': balanced_time_inside,
        'entry_bars': entry_bars,
        'risk_per_trade': risk_per_trade,
    }

# Configuration management
def config_management_ui():
    """UI for saving and loading configurations"""
    st.subheader("Configuration Management")
    
    # Safely fetch pair & timeframe from session state
    current_pair = st.session_state.get("pair", AVAILABLE_PAIRS[0])
    current_tf   = st.session_state.get("timeframe", TIMEFRAMES[1])

    col1, col2 = st.columns(2)
    
    with col1:
        config_name = st.text_input(
            "Configuration Name",
            value=st.session_state.get(
                'config_name', 
                f"{current_pair}_{current_tf}_config"
            ),
        )
        st.session_state['config_name'] = config_name
        
        if st.button("Save Configuration"):
            config = {
                'pair':            current_pair,
                'timeframe':       current_tf,
                'session_defs':    st.session_state['session_defs'],
                'bucket_size_pips':st.session_state['bucket_size_pips'],
                'value_area_pct':  st.session_state['value_area_pct'],
                'balanced_poc_center': st.session_state['balanced_poc_center'],
                'balanced_time_inside': st.session_state['balanced_time_inside'],
                'entry_bars':      st.session_state['entry_bars'],
                'risk_per_trade':  st.session_state['risk_per_trade'],
                'saved_date':      datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            if save_config(config_name, config):
                st.success(f"Configuration '{config_name}' saved!")

    with col2:
        saved_configs = get_saved_configs()
        if not saved_configs:
            st.info("No saved configurations found")
        else:
            # Let the user pick exactly one
            choice = st.radio("Load Configuration:", saved_configs)
            selected = choice  # the filename/key
            if st.button("Load Selected Config"):
                loaded = load_config(selected)
                if loaded:
                    for key in (
                        'pair','timeframe','session_defs','bucket_size_pips',
                        'value_area_pct','balanced_poc_center',
                        'balanced_time_inside','entry_bars','risk_per_trade'
                    ):
                        if key in loaded:
                            st.session_state[key] = loaded[key]
                    st.session_state['config_name'] = selected
                    st.success(f"Loaded '{selected}'")
                    st.rerun()
