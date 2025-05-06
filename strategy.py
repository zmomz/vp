from backtesting import Strategy
from contextlib import contextmanager
from config import STRATEGY_CONFIG
from datetime import time, timedelta
import concurrent.futures
import numpy as np
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# 0) METRICS & bench context manager (stub)
# ─────────────────────────────────────────────────────────────────────────────
METRICS = {}

@contextmanager
def bench(name, metrics):
    """
    Stub context manager for profiling sections.
    Replace with your real timing implementation if desired.
    """
    yield

class VolumeProfileStrategy(Strategy):
    """Volume Profile Strategy with Session Analysis"""
    
    def init(self):
        self.has_session_data, self.trade_log = False, []
        try:
            self.df_labeled = self.assign_sessions(self.data.df.copy())
            self._sess_name = self.df_labeled["session_name"]
            self._sess_date = self.df_labeled["session_date"]
            self._sess_color = self.df_labeled["session_color"]

            # 🔸 PROFILE BUILD – heavy CPU
            with bench("profile_build", STRATEGY_CONFIG.setdefault("_perf", {})):
                with st.spinner("Calculating volume profiles…"):
                    self.profiles = self.calculate_profiles(self.df_labeled)

            self.has_session_data = True

        except Exception as e:
            st.error(f"Strategy initialization error: {e}")
            self.profiles = {}


    def assign_sessions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign each bar to a trading session"""
        if 'session_defs' not in STRATEGY_CONFIG:
            raise ValueError("STRATEGY_CONFIG['session_defs'] missing")
        
        rec, unmatched = [], 0
        
        for ts in df.index:
            t, d, matched = ts.time(), ts.date(), False
            
            for sd in STRATEGY_CONFIG['session_defs']:
                start, end = time.fromisoformat(sd['start']), time.fromisoformat(sd['end'])
                normal = start < end
                in_range = (start <= t < end) if normal else (t >= start or t < end)
                
                if in_range:
                    sess_date = d if normal or t < end else d + timedelta(days=1)
                    rec.append((ts, sd['name'], sess_date, sd.get('color', 'rgba(100, 100, 100, 0.5)')))
                    matched = True
                    break
            
            if not matched:
                unmatched += 1
                rec.append((ts, 'default', d, 'rgba(200, 200, 200, 0.5)'))
        
        if unmatched:
            st.warning(f"{unmatched} bars outside any session definition")
            
        sess_df = pd.DataFrame(rec, columns=['time', 'session_name', 'session_date', 'session_color'])
        return df.join(sess_df.set_index('time'), how='left')

    def calculate_profiles(self, df: pd.DataFrame) -> dict:
        """Calculate volume profiles for each session"""
        profiles = {}
        total_sessions = len(df.groupby(['session_date', 'session_name']))
        progress_text = st.empty()
        progress_bar = st.progress(0)
        
        # Use thread pool for performance with many sessions
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for (s_date, s_name), grp in df.groupby(['session_date', 'session_name']):
                futures.append(
                    executor.submit(self._calculate_single_profile, s_date, s_name, grp)
                )
            
            for i, future in enumerate(concurrent.futures.as_completed(futures)):
                try:
                    result = future.result()
                    if result:
                        profiles[result[0]] = result[1]
                    
                    # Update progress
                    progress = int((i + 1) / total_sessions * 100)
                    progress_text.text(f"Processing profiles: {progress}%")
                    progress_bar.progress(progress / 100)
                    
                except Exception as e:
                    st.error(f"Error calculating profile: {str(e)}")
        
        progress_text.empty()
        progress_bar.empty()
        return profiles

    def _calculate_single_profile(self, s_date, s_name, grp):
        """Calculate a single session profile (for thread pool)"""
        if len(grp) < 5:
            return None
            
        vp = self.compute_volume_profile(grp)
        if vp.empty or vp.Volume.sum() == 0:
            return None
            
        try:
            val, poc, vah = self.compute_val_poc_vah(vp)
            return (
                (s_date, s_name), 
                dict(
                    val=val, 
                    vah=vah, 
                    poc=poc,
                    balanced=self.is_balanced(grp, val, poc, vah),
                    vp=vp  # Store the full volume profile for visualization
                )
            )
        except Exception as e:
            print(e)
            return None

    def compute_volume_profile(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute volume profile from price data"""
        bucket = STRATEGY_CONFIG.get('bucket_size_pips', 5) * 0.0001
        prange = df.High.max() - df.Low.min()
        
        if prange <= 0 or np.isnan(prange):
            return pd.DataFrame(columns=['Price', 'Volume']).set_index('Price')
            
        bins = np.arange(
            np.floor(df.Low.min() / bucket) * bucket,
            np.ceil(df.High.max() / bucket) * bucket + bucket,
            bucket
        )
        
        if len(bins) < 3:
            return pd.DataFrame(columns=['Price', 'Volume']).set_index('Price')
            
        cats = pd.cut(df.Close, bins=bins, right=False)
        vp = df.Volume.groupby(cats, observed=False).sum()
        
        return pd.DataFrame({
            'Price': [i.left + bucket / 2 for i in vp.index.categories],
            'Volume': vp.values
        }).set_index('Price')

    def compute_val_poc_vah(self, vp: pd.DataFrame):
        """Compute Value Area Low, Point of Control, and Value Area High"""
        total = vp.Volume.sum()
        poc = vp.Volume.idxmax()
        
        sorted_bins = vp.sort_values('Volume', ascending=False)
        cum = sorted_bins.Volume.cumsum()
        target = STRATEGY_CONFIG.get('value_area_pct', 0.7) * total
        
        selected = cum[cum <= target].index.tolist()
        if len(selected) < len(sorted_bins):
            selected.append(cum.index[len(selected)])
            
        if not selected:
            raise ValueError("value-area threshold not met")
            
        return min(selected), poc, max(selected)

    def is_balanced(self, df, val, poc, vah):
        """Determine if a session is balanced"""
        if abs(vah - val) < 1e-5:
            return False
            
        midpoint = val + (vah - val) / 2
        poc_off = abs(poc - midpoint) / (vah - val)
        inside = ((df.Close >= val) & (df.Close <= vah)).mean()
        
        return (
            poc_off <= STRATEGY_CONFIG.get('balanced_poc_center', 0.25) and
            inside >= STRATEGY_CONFIG.get('balanced_time_inside', 0.6)
        )

    def next(self):
        """Strategy logic for each bar"""
        if not self.has_session_data or len(self.data) < 100:
            return
            
        ts = self.data.index[-1]
        s_name = self._sess_name.at[ts]
        s_date = self._sess_date.at[ts]
        s_color = self._sess_color.at[ts]
        
        if pd.isna(s_name) or pd.isna(s_date):
            return
            
        # Get previous session profile
        prev = self.profiles.get(self.get_previous_session(s_date, s_name))
        if not prev:
            return
            
        val, vah = prev['val'], prev['vah']
        if abs(vah - val) < 1e-5 or self.position:
            return
            
        # Get session open price
        s_open = self.get_session_open(s_date, s_name)
        if s_open is None:
            return
            
        # Trading rules
        allow_long = s_open < val
        allow_short = s_open > vah
        
        # Skip balanced profiles if starting within the value area
        if (val <= s_open <= vah) and not prev['balanced']:
            return
            
        price = self.data.Close[-1]
        bucket = STRATEGY_CONFIG.get('bucket_size_pips', 5) * 0.0001

        
        # Entry logic with fixed fractional position sizing
        risk_pct = STRATEGY_CONFIG.get('risk_per_trade', 1.0)/100
        
        # Long entry
        if allow_long and price < vah and self.long_trigger(val):
            stop_price = val - bucket/2

            size = risk_pct  # Use percentage of equity directly
             
            if size > 0:
                self.buy(size=self._FULL_EQUITY*risk_pct, sl=stop_price, tp=vah)
                
                # Log trade details
                self.trade_log.append({
                    'timestamp': ts,
                    'session_date': s_date, 
                    'session_name': s_name,
                    'session_color': s_color,
                    'type': 'LONG',
                    'entry': price,
                    'stop': stop_price,
                    'target': vah,
                    'val': val,
                    'vah': vah,
                    'size': size
                })
        

        elif allow_short and price > val and self.short_trigger(vah):
            stop_price = vah + bucket/2
            
            # Fix: Use fixed fractional sizing (% of equity)
            size = risk_pct  # Use percentage of equity directly    


            if size > 0:
                self.sell(size=self._FULL_EQUITY*risk_pct, sl=stop_price, tp=val)
                
                # Log trade details
                self.trade_log.append({
                    'timestamp': ts,
                    'session_date': s_date, 
                    'session_name': s_name,
                    'session_color': s_color,
                    'type': 'SHORT',
                    'entry': price,
                    'stop': stop_price,
                    'target': val,
                    'val': val,
                    'vah': vah,
                    'size': size
                })

    def get_previous_session(self, d, n):
        """Get the previous session date and name"""
        order = {sd['name']: i for i, sd in enumerate(STRATEGY_CONFIG['session_defs'])}
        idx = order.get(n)
        
        if idx is None or idx == 0:
            return (d - timedelta(days=1), STRATEGY_CONFIG['session_defs'][-1]['name'])
        return (d, STRATEGY_CONFIG['session_defs'][idx - 1]['name'])

    def get_session_open(self, d, n):
        """Get the opening price of a session"""
        bars = self.df_labeled[
            (self.df_labeled.session_date == d) &
            (self.df_labeled.session_name == n)
        ]
        return bars.iloc[0].Open if not bars.empty else None

    def long_trigger(self, val):
        """Long entry trigger condition"""
        n = STRATEGY_CONFIG.get('entry_bars', 2)
        return len(self.data) >= n and (self.data.Close[-n:] > val).all()

    def short_trigger(self, vah):
        """Short entry trigger condition"""
        n = STRATEGY_CONFIG.get('entry_bars', 2)
        return len(self.data) >= n and (self.data.Close[-n:] < vah).all()
