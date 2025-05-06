# helper_functions.py
AVAILABLE_PAIRS = ["EURUSD","GBPUSD","AUDUSD","NZDUSD","USDCAD","USDCHF","USDJPY"]
TIMEFRAMES      = ["5m","10m","15m","1h"]
DEFAULT_SESSIONS = [
    {"name":"A","start":"00:00","end":"08:00","color":"#ff6384"},
    {"name":"B","start":"08:00","end":"16:00","color":"#36a2eb"},
    {"name":"C","start":"16:00","end":"00:00","color":"#4bc0c0"},
]
EPOCHS = {
    '2006/2007 pre Financial Crisis':      ('2006-01-01', '2007-12-31'),
    '2008/2009 Financial Crisis':     ('2008-01-01', '2009-12-31'),
    '2010/2012 Financial Crisis Recovery': ('2010-01-01', '2012-12-31'),
    '2013/2019 bull market':               ('2013-01-01', '2019-12-31'),
    '2020/2021 covid pandemic':            ('2020-01-01', '2021-12-31'),
    '2022/2025 post covid':                ('2022-01-01', '2025-12-31')
}

STRATEGY_CONFIG = {
    'session_defs': [],           # will be overwritten by UI
    'bucket_size_pips': 5,
    'value_area_pct': 0.7,
    'balanced_poc_center': 0.25,
    'balanced_time_inside': 0.7,
    'risk_per_trade': 5.0,
    '_perf': {},                  # for bench metrics
}

# ─────────────────────────────────────────────────────────────────────────────
# 1) CONSTANTS & DIRECTORIES
# ─────────────────────────────────────────────────────────────────────────────
DATA_DIR        = "data"
CONFIG_DIR      = "configs"
RESULTS_DIR     = "results"
PER_PAGE        = 3

# ─────────────────────────────────────────────────────────────────────────────
# 2) SESSION-STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
_INITIAL_STATE = {
    "pair": None,
    "timeframe": None,
    "start_date": None,
    "end_date": None,
    "session_defs": None,
    "bucket_size_pips": 5,
    "value_area_pct": 0.7,
    "entry_bars": 2,
    "balanced_poc_center": 0.25,
    "balanced_time_inside": 0.7,
    "risk_per_trade": 5.0,
    "backtest_run": False,
    "backtest_data": None,
    "backtest_stats": None,
    "backtest_strategy": None,
    "sess_page_slider": 1,
}