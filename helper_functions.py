# helper_functions.py
import os
import json
from typing import Optional
import pandas as pd


# Number formatter
def format_number(x):
    """
    Format numeric values consistently in tables.
    """
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)

# Equity-curve builder fallback
def build_equity_curve_from_trades(stats) -> pd.Series:
    """
    Build an equity curve at each trade timestamp, with starting cash prepended.
    """
    if not hasattr(stats, "_trades") or stats._trades.empty:
        return pd.Series(dtype=float)

    df = stats._trades.copy()
    # ensure EntryTime is datetime
    df["EntryTime"] = pd.to_datetime(df["EntryTime"])
    # compute equity progression
    df["Equity"] = 10000 * (1 + df["ReturnPct"]).cumprod()

    # set the per-trade equity series
    eq = df.set_index("EntryTime")["Equity"].sort_index()

    # insert the starting cash one minute before the first trade
    first_ts = eq.index[0]
    start_ts = first_ts - pd.Timedelta(minutes=1)
    start_eq = pd.Series([10000.0], index=[start_ts])

    full_eq = pd.concat([start_eq, eq]).sort_index()
    return full_eq

# Config save/load helpers
CONFIG_DIR = "configs"
os.makedirs(CONFIG_DIR, exist_ok=True)

def save_config(name: str, config: dict) -> bool:
    """
    Save a configuration dict to JSON under CONFIG_DIR/name.json.
    """
    try:
        path = os.path.join(CONFIG_DIR, f"{name}.json")
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        return True
    except Exception:
        return False

def get_saved_configs() -> list:
    """
    List all saved config names (without .json) in CONFIG_DIR.
    """
    if not os.path.isdir(CONFIG_DIR):
        return []
    return [
        os.path.splitext(fn)[0]
        for fn in os.listdir(CONFIG_DIR)
        if fn.lower().endswith(".json")
    ]

def load_config(name: str) -> Optional[dict]:
    """
    Load and return the JSON config dict for CONFIG_DIR/name.json, or None.
    """
    try:
        path = os.path.join(CONFIG_DIR, f"{name}.json")
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None

