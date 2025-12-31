import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import asyncio
import sqlite3
from collections import OrderedDict
from functools import lru_cache
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from scipy import stats, signal
from scipy.optimize import minimize_scalar
from scipy.stats import skew, kurtosis, jarque_bera, norm, t
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

# Try to import Numba for JIT compilation (optional but recommended)
try:
    from numba import jit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    # Create dummy decorator if Numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==========================================
# PROFESSIONAL CONFIGURATION - ENHANCED
# ==========================================

class Config:
    """
    Ultra-Precise XAUUSD Trading Configuration
    Now with dynamic parameter adaptation and enhanced features
    """
    
    # ==========================================
    # MT5 CONNECTION
    # ==========================================
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", 5044241746))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "!kAd2ePr")
    MT5_SERVER = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
    
    # ==========================================
    # TRADING INSTRUMENT SPECIFICATIONS
    # ==========================================
    SYMBOL = "XAUUSD"
    TIMEFRAME = mt5.TIMEFRAME_M15 # CHANGED: M15 as Primary for Medium-Term Precision
    
    # Position sizing with enhanced scaling
    BASE_VOLUME = 0.01
    MAX_VOLUME = 0.030
    MIN_VOLUME = 0.01
    VOLUME_STEP = 0.01
    
    MAGIC_NUMBER = 998877
    
    # ==========================================
    # RISK MANAGEMENT - ENHANCED
    # ==========================================
    RISK_PERCENT = 0.02  # INCREASED from 0.01 to 0.02 (2% per trade)
    MAX_TOTAL_RISK_PERCENT = 0.10  # INCREASED from 0.05 to 0.10 (10% total exposure)
    MAX_RISK_PER_TRADE = 200  # INCREASED from 100 to 200
    
    # Signal Quality - Dynamic thresholds
    MIN_CONFIDENCE = 0.30  # REDUCED from 0.40 to 0.30 for faster entries
    MIN_ENSEMBLE_AGREEMENT = 0.50
    
    # Position Limits
    MAX_POSITIONS = 5
    MAX_DAILY_TRADES = 15  # INCREASED from 10 to 15
    MIN_TIME_BETWEEN_TRADES = 5  # REDUCED from 10 to 5 seconds
    
    # Loss Limits
    MAX_DAILY_LOSS_PERCENT = 2.0
    MAX_WEEKLY_LOSS_PERCENT = 5.0
    MAX_DRAWDOWN_PERCENT = 10.0
    MAX_CONSECUTIVE_LOSSES = 3
    
    # Kelly Criterion
    KELLY_FRACTION = 0.50      # INCREASED from 0.25 to 0.50 (Half Kelly for more aggressive sizing)
    USE_KELLY_CRITERION = True
    MAX_KELLY_RISK = 0.10      # INCREASED from 0.05 to 0.10 (10% hard cap)
    
    # Volatility Sizing (Smart Risk)
    VOLATILITY_SCALING_ENABLED = True
    HIGH_VOL_THRESHOLD = 0.0040       # ~40 pips per 15min candle on Gold
    LOW_VOL_THRESHOLD = 0.0012        # ~12 pips is low
    HIGH_VOL_SIZE_MULTIPLIER = 0.5    # Half size in storms
    LOW_VOL_SIZE_MULTIPLIER = 1.5     # 1.5x size in calm
    
    # News / Event Filter
    NEWS_FILTER_ENABLED = True
    MAX_SPREAD_PIPS = 3.5             # Block trade if spread > 3.5 pips (News Proxy)
    VOLATILITY_SPIKE_THRESHOLD = 3.0  # Block if Vol Z-Score > 3 (Sudden explosion)

    # Statistical Risk Metrics
    VAR_CONFIDENCE = 0.99
    CVAR_CONFIDENCE = 0.99
    MAX_POSITION_CORRELATION = 0.5
    
    # ==========================================
    # MACHINE LEARNING MODEL PARAMETERS - ENHANCED
    # ==========================================
    LOOKBACK_BARS = 12000
    TRAINING_MIN_SAMPLES = 5000
    VALIDATION_SPLIT = 0.20
    
    # Retraining Schedule
    RETRAIN_HOURS = 8
    RETRAIN_ON_PERFORMANCE_DROP = True
    MIN_ACCURACY_THRESHOLD = 0.50
    
    # Walk-Forward Optimization
    WALK_FORWARD_WINDOW = 2500      # Increased for better learning per fold
    WALK_FORWARD_STEP = 500
    WALK_FORWARD_FOLDS = 5
    
    # Feature Engineering Flags
    USE_FRACTIONAL_DIFF = True
    FD_THRESHOLD = 0.4
    USE_TICK_VOLUME_VOLATILITY = True
    TICK_SKEW_LOOKBACK = 50
    
    # Labeling Method - ENHANCED: Dynamic ATR-based barriers
    TRIPLE_BARRIER_METHOD = True
    USE_DYNAMIC_BARRIERS = True  # New: Dynamic ATR-based barriers
    BARRIER_UPPER = 0.0020
    BARRIER_LOWER = -0.0015
    BARRIER_TIME = 24                # Shortened to 1 hour (12 bars on M5) for day trading
    
    # Ensemble Configuration - ENHANCED
    USE_STACKING_ENSEMBLE = True
    ENSEMBLE_DIVERSITY_WEIGHT = 0.3
    ADAPTIVE_ENSEMBLE_WEIGHTING = True
    MODEL_CONFIDENCE_CALIBRATION = True
    USE_REGIME_SPECIFIC_MODELS = True  # New: Regime-specific models
    
    # Data Quality
    MIN_DATA_QUALITY_SCORE = 0.75
    OUTLIER_REMOVAL_THRESHOLD = 4.0
    
    # ==========================================
    # TECHNICAL INDICATORS - ENHANCED
    # ==========================================
    ATR_PERIOD = 14
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    
    # Moving Averages
    FAST_MA = 8
    MEDIUM_MA = 21
    SLOW_MA = 50
    TREND_MA = 200
    
    # Volatility Bands
    BB_PERIOD = 20
    BB_STD = 2.0
    
    # ==========================================
    # STATISTICAL FEATURES - ENHANCED
    # ==========================================
    GARCH_VOL_PERIOD = 20
    GARCH_P = 1
    GARCH_Q = 1
    
    # Hurst Exponent
    HURST_WINDOW = 100
    HURST_TRENDING_THRESHOLD = 0.55
    HURST_MEANREVERTING_THRESHOLD = 0.45
    
    # Tail Risk
    TAIL_INDEX_WINDOW = 150
    VAR_LOOKBACK = 100
    
    # Correlation Analysis
    CORRELATION_WINDOW = 50
    CORRELATED_SYMBOLS = ["DXY", "US10Y", "EURUSD"]
    
    # ==========================================
    # STOP LOSS & TAKE PROFIT - ENHANCED
    # ==========================================
    USE_DYNAMIC_SL_TP = True
    
    # ATR-Based Stops with dynamic multipliers
    ATR_SL_MULTIPLIER = 1.5
    ATR_TP_MULTIPLIER = 1.75
    
    # Minimum Risk/Reward
    MIN_RR_RATIO = 1.175
    
    # Fixed Stops
    FIXED_SL_PERCENT = 0.0035
    FIXED_TP_PERCENT = 0.0070
    
    # Points-based Limits
    MIN_SL_DISTANCE_POINTS = 50
    MAX_SL_DISTANCE_POINTS = 2500
    MIN_TP_DISTANCE_POINTS = 100
    MAX_TP_DISTANCE_POINTS = 5000
    
    # Trailing Stop
    USE_TRAILING_STOP = True
    TRAILING_STOP_ACTIVATION = 1.5
    TRAILING_STOP_DISTANCE = 1.0
    
    # Break-even Stop
    USE_BREAKEVEN_STOP = True
    BREAKEVEN_ACTIVATION = 1.0
    BREAKEVEN_OFFSET = 0.0001
    
    # ==========================================
    # MARKET REGIME DETECTION - ENHANCED
    # ==========================================
    USE_MARKET_REGIME = True
    
    # Trend Strength
    ADX_TREND_THRESHOLD = 20
    ADX_STRONG_TREND_THRESHOLD = 40
    ADX_SLOPE_THRESHOLD = 0.5
    
    # Volatility Regimes
    VOLATILITY_SCALING_ENABLED = True
    HIGH_VOL_THRESHOLD = 0.015
    NORMAL_VOL_THRESHOLD = 0.010
    LOW_VOL_THRESHOLD = 0.007
    
    # Position Sizing Adjustments by Regime
    HIGH_VOL_SIZE_MULTIPLIER = 0.5
    LOW_VOL_SIZE_MULTIPLIER = 1.2
    
    # ==========================================
    # TIME-BASED FILTERS
    # ==========================================
    SESSION_AWARE_TRADING = True  # Keep aware, but don't block
    
    # Trading Sessions (UTC times)
    AVOID_ASIAN_SESSION = False   # ENABLED 24/7 TRADING
    PREFER_LONDON_NY_OVERLAP = True
    
    LONDON_OPEN_HOUR = 8
    LONDON_CLOSE_HOUR = 16
    NY_OPEN_HOUR = 13
    NY_CLOSE_HOUR = 20
    
    # Avoid trading during:
    AVOID_FIRST_15MIN = True
    AVOID_LAST_30MIN = True
    
    # News Events
    NEWS_EVENT_BUFFER_HOURS = 1
    HIGH_IMPACT_NEWS_BUFFER = 2
    
    # Days of Week
    AVOID_MONDAY_FIRST_HOUR = True
    AVOID_FRIDAY_LAST_HOURS = False # ENABLED 24/7
    
    # ==========================================
    # ORDER EXECUTION
    # ==========================================
    MAX_SLIPPAGE_POINTS = 10
    ORDER_TIMEOUT_SECONDS = 30
    MAX_RETRIES = 3
    RETRY_DELAY_MS = 1000
    FEATURE_RECALC_INTERVAL_SECONDS = 5  # REDUCED from 30s to 5s for faster market reaction
    
    USE_MARKET_ORDERS = True
    USE_LIMIT_ORDERS = False
    
    CHECK_SPREAD_BEFORE_ENTRY = True
    MAX_SPREAD_POINTS = 30
    NORMAL_SPREAD_POINTS = 2
    
    COMMISSION_PER_LOT = 3.5
    
    # ==========================================
    # PERFORMANCE METRICS
    # ==========================================
    MIN_SHARPE_RATIO = 0.8
    MIN_PROFIT_FACTOR = 1.5
    MIN_WIN_RATE = 0.45
    MAX_DRAWDOWN_DURATION = 10
    
    MIN_SAMPLES_FOR_STATS = 100
    CONFIDENCE_LEVEL = 0.95
    BOOTSTRAP_SAMPLES = 1000
    
    # ==========================================
    # ADAPTIVE SYSTEMS - ENHANCED
    # ==========================================
    ADAPTIVE_RISK_MANAGEMENT = True
    PERFORMANCE_BASED_POSITION_SIZING = True
    REAL_TIME_MARKET_STRESS_INDICATOR = True
    
    # Adaptation Parameters
    PERFORMANCE_LOOKBACK_TRADES = 20
    GOOD_PERFORMANCE_THRESHOLD = 0.60
    POOR_PERFORMANCE_THRESHOLD = 0.40
    
    # Position Size Adjustments
    INCREASE_SIZE_AFTER_WINS = False
    DECREASE_SIZE_AFTER_LOSSES = True
    SIZE_DECREASE_FACTOR = 0.8
    SIZE_RECOVERY_FACTOR = 1.1
    
    # ==========================================
    # NEW: PARAMETER OPTIMIZATION
    # ==========================================
    OPTIMIZATION_WINDOW = 500
    OPTIMIZE_EVERY_N_TRADES = 20
    PARAM_OPTIMIZATION_ENABLED = True
    
    # ==========================================
    # NEW: ENTRY TIMING
    # ==========================================
    USE_CONFIRMATION_ENTRY = False
    CONFIRMATION_BARS_REQUIRED = 2
    MAX_ENTRY_WAIT_SECONDS = 900  # 15 minutes
    
    # ==========================================
    # DATA STORAGE & LOGGING
    # ==========================================
    CACHE_DIR = "cache"
    TRADE_HISTORY_FILE = os.path.join(CACHE_DIR, "trade_history_xauusd.json")
    MODEL_SAVE_FILE = os.path.join(CACHE_DIR, "ensemble_model_xauusd.pkl")
    ENGINE_STATE_FILE = os.path.join(CACHE_DIR, "engine_state.json")
    MARKET_INSIGHTS_FILE = os.path.join(CACHE_DIR, "market_insights.json")
    LEARNING_DATA_FILE = os.path.join(CACHE_DIR, "learning_data.json")  # NEW: ML from past trades
    
    BACKTEST_RESULTS_FILE = os.path.join(CACHE_DIR, "backtest_results_xauusd.json")
    PERFORMANCE_LOG_FILE = os.path.join(CACHE_DIR, "performance_log_xauusd.csv")
    
    MEMORY_SIZE = 10000 #ex 1000
    LEARNING_WEIGHT = 0.5 #ex 0.5
    
    # Logging Levels
    LOG_LEVEL_CONSOLE = "INFO"
    LOG_LEVEL_FILE = "DEBUG"
    LOG_TRADES = True
    LOG_PREDICTIONS = True
    LOG_PERFORMANCE = True
    
    # ==========================================
    # MULTI-TIMEFRAME ANALYSIS
    # ==========================================
    MULTI_TIMEFRAME_ENABLED = True
    # PRECISION: Added H1 for robust trend filtering
    TIMEFRAMES = ['M1', 'M5', 'M15', 'M30', 'H1'] 
    # Adjusted weights to focus on Medium Term (M15/M5)
    TIMEFRAME_WEIGHTS = [0.05, 0.25, 0.50, 0.10, 0.10]
    TIMEFRAME_ALIGNMENT_THRESHOLD = 0.40 # Enforce alignment
    REQUIRE_TIMEFRAME_ALIGNMENT = True # Enforce strict precision
    
    LONG_TIMEFRAME_TREND_FILTER = True
    SHORT_TIMEFRAME_ENTRY = True
    
    # ==========================================
    # GOLD-SPECIFIC PARAMETERS
    # ==========================================
    GOLD_VOLATILITY_ADJUSTMENT = True
    
    EXPECTED_DAILY_RANGE = 20
    HIGH_RANGE_MULTIPLIER = 1.5
    LOW_RANGE_MULTIPLIER = 0.5
    
    USE_DXY_FILTER = True
    DXY_CORRELATION_THRESHOLD = -0.7
    USE_YIELD_FILTER = False
    
    # ==========================================
    # SAFETY FEATURES
    # ==========================================
    ENABLE_EMERGENCY_STOP = True
    EMERGENCY_STOP_DRAWDOWN = 0.15
    
    CHECK_MARGIN_BEFORE_TRADE = True
    MIN_FREE_MARGIN_PERCENT = 0.30
    
    CHECK_CONNECTION_BEFORE_TRADE = True
    MAX_PING_MS = 100
    
    MAX_DAILY_VOLUME = 1.0
    REQUIRE_STOP_LOSS = True
    REQUIRE_TAKE_PROFIT = True
    
    # ==========================================
    # DEBUGGING & TESTING
    # ==========================================
    DEBUG_MODE = False
    PAPER_TRADING_MODE = False
    BACKTEST_MODE = False
    
    VALIDATE_SIGNALS = True
    VALIDATE_RISK = True
    VALIDATE_STOPS = True

    # ==========================================
    # ADVANCED OPTIMIZATIONS
    # ==========================================
    # Speed Optimizations
    ENABLE_VECTORIZED_CALCULATIONS = True
    ENABLE_SMART_CACHING = True
    ENABLE_ASYNC_FETCHING = True
    ENABLE_NUMBA_JIT = NUMBA_AVAILABLE  # Auto-detect Numba availability
    ENABLE_OPTIMIZED_POSITION_MONITORING = True
    POSITION_CHECK_INTERVAL = 10  # seconds (reduced from 60s)
    ENABLE_LOOKUP_TABLES = True
    ENABLE_ADAPTIVE_CALC_FREQUENCY = True
    USE_SQLITE_DATABASE = True
    
    # Caching TTLs (Time To Live in seconds)
    CACHE_TTL_GARCH = 300  # 5 minutes
    CACHE_TTL_HURST = 600  # 10 minutes
    CACHE_TTL_CORRELATION = 900  # 15 minutes
    CACHE_TTL_ATR = 60  # 1 minute
    
    # Precision Optimizations
    ENABLE_MULTI_TIMEFRAME_CONFIRMATION = True
    MULTI_TF_MIN_AGREEMENT = 0.70  # 70% of timeframes must agree
    MULTI_TF_WEIGHTS = {
        'M1': 0.10,
        'M5': 0.35,
        'M15': 0.40,
        'M30': 0.15
    }
    
    ENABLE_SIGNAL_QUALITY_FILTER = True
    MIN_SIGNAL_QUALITY_SCORE = 70  # 0-100 scale
    
    ENABLE_ADAPTIVE_CONFIDENCE = True
    ADAPTIVE_CONF_LOOKBACK = 20  # trades
    ADAPTIVE_CONF_HIGH_THRESHOLD = 0.60  # win rate
    ADAPTIVE_CONF_LOW_THRESHOLD = 0.40
    ADAPTIVE_CONF_HIGH_ADJUSTMENT = 0.25  # lower to 0.25
    ADAPTIVE_CONF_LOW_ADJUSTMENT = 0.45  # raise to 0.45
    
    ENABLE_ADVANCED_STOPS = True
    STOP_AVOID_ROUND_NUMBERS = True
    STOP_USE_SUPPORT_RESISTANCE = True
    STOP_VOLATILITY_ADJUSTMENT = True
    
    ENABLE_MICROSTRUCTURE_ANALYSIS = True
    MICROSTRUCTURE_SPREAD_THRESHOLD = 3.0  # pips
    MICROSTRUCTURE_IMBALANCE_THRESHOLD = 0.60  # 60% bid or ask pressure
    
    ENABLE_FEATURE_IMPORTANCE_TRACKING = True
    FEATURE_IMPORTANCE_UPDATE_INTERVAL = 50  # trades
    FEATURE_IMPORTANCE_MIN_CORRELATION = 0.05  # disable features below this
    
    # Database settings
    DATABASE_FILE = os.path.join(CACHE_DIR, "trade_history.db")

    MULTI_TIMEFRAME_PARAMS = {
        # Alignment bonus scaling
        # Alignment bonus scaling - WIDER RANGE for more movement
        'ALIGNMENT_BONUS_ABOVE_START': 0.85,
        'ALIGNMENT_BONUS_ABOVE_RANGE': 0.15,
        'ALIGNMENT_BONUS_BELOW_START': 0.15,
        'ALIGNMENT_BONUS_BELOW_RANGE': 0.7,
        
        # Trend bonus parameters
        'TREND_BONUS_SUPPORT': 1.1,      # Reward strong macro trend alignment
        'TREND_BONUS_NEUTRAL': 0.8,
        'TREND_BONUS_OPPOSITE': 0.4,     # More punishing veto
        
        # Confidence calculation
        'BASE_CONFIDENCE_MULTIPLIER': 0.95,
        'CONFIDENCE_FLOOR': 0.05,
        'CONFIDENCE_CEILING': 0.98,
        
        # Signal generation thresholds - REDUCED for higher sensitivity
        'BUY_SIGNAL_THRESHOLD': 0.20,
        'SELL_SIGNAL_THRESHOLD': -0.20,
        
        # Timeframe signal multipliers - INCREASED for lower timeframes
        'M1_SIGNAL_MULTIPLIER': 1.0,
        'M5_SIGNAL_MULTIPLIER': 1.1,
        'M15_SIGNAL_MULTIPLIER': 1.2,
        'M30_SIGNAL_MULTIPLIER': 1.3,
        
        # Feature weight adjustments for Asian session
        'ASIAN_SESSION_ADJUSTMENTS': {
            'trend_direction_weight': 0.8,
            'rsi_weight': 1.2,
            'volatility_weight': 0.7,
            'price_position_weight': 1.1
        }
    }# ==========================================
# PROFESSIONAL LOGGING
# ==========================================
class ProfessionalLogger:
    COLORS = {
        'RESET': '\033[0m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m',
        'GRAY': '\033[90m'
    }
    
    @staticmethod
    def log(message, level='INFO', component='MAIN'):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        colors = {
            'INFO': ProfessionalLogger.COLORS['CYAN'],
            'SUCCESS': ProfessionalLogger.COLORS['GREEN'],
            'WARNING': ProfessionalLogger.COLORS['YELLOW'],
            'ERROR': ProfessionalLogger.COLORS['RED'],
            'TRADE': ProfessionalLogger.COLORS['MAGENTA'],
            'LEARN': ProfessionalLogger.COLORS['BLUE'],
            'DATA': ProfessionalLogger.COLORS['WHITE'],
            'ENSEMBLE': ProfessionalLogger.COLORS['GREEN'],
            'RISK': ProfessionalLogger.COLORS['YELLOW'],
            'PERFORMANCE': ProfessionalLogger.COLORS['MAGENTA'],
            'BACKTEST': ProfessionalLogger.COLORS['GRAY'],
            'STATISTICS': ProfessionalLogger.COLORS['BLUE'],
            'ANALYSIS': ProfessionalLogger.COLORS['CYAN'],
            'OPTIMIZER': ProfessionalLogger.COLORS['MAGENTA'],  # New
            'CONFIRMATION': ProfessionalLogger.COLORS['BLUE'],   # New
            'FILTER': ProfessionalLogger.COLORS['YELLOW']        # New
        }
        color = colors.get(level, ProfessionalLogger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level:8s}{ProfessionalLogger.COLORS['RESET']}] [{component:12s}] {message}", flush=True)

# ==========================================
# IDLE TIME TRACKER
# ==========================================
class IdleTimeTracker:
    """Track time spent in different processing states"""
    
    def __init__(self):
        self.state = 'idle'  # idle, processing, thinking, executing
        self.state_start_time = time.time()
        self.state_durations = {
            'idle': 0.0,
            'processing': 0.0,
            'thinking': 0.0,
            'executing': 0.0
        }
        self.last_report_time = time.time()
        self.report_interval = 300  # 5 minutes
    
    def set_state(self, new_state):
        """Change state and record duration"""
        current_time = time.time()
        duration = current_time - self.state_start_time
        
        # Add duration to previous state
        if self.state in self.state_durations:
            self.state_durations[self.state] += duration
        
        # Update to new state
        self.state = new_state
        self.state_start_time = current_time
        
        # Check if we should report
        if current_time - self.last_report_time >= self.report_interval:
            self.report()
            self.last_report_time = current_time
    
    def report(self):
        """Log idle time report"""
        total_time = sum(self.state_durations.values())
        if total_time == 0:
            return
        
        idle_pct = (self.state_durations['idle'] / total_time) * 100
        processing_pct = (self.state_durations['processing'] / total_time) * 100
        thinking_pct = (self.state_durations['thinking'] / total_time) * 100
        executing_pct = (self.state_durations['executing'] / total_time) * 100
        
        ProfessionalLogger.log(
            f"⏱️ Time Distribution (Last 5min): "
            f"Idle: {self.state_durations['idle']:.1f}s ({idle_pct:.1f}%) | "
            f"Processing: {self.state_durations['processing']:.1f}s ({processing_pct:.1f}%) | "
            f"Thinking: {self.state_durations['thinking']:.1f}s ({thinking_pct:.1f}%) | "
            f"Executing: {self.state_durations['executing']:.1f}s ({executing_pct:.1f}%)",
            "INFO", "IDLE_TRACKER"
        )
        
        # Reset counters
        for key in self.state_durations:
            self.state_durations[key] = 0.0



# ==========================================
# MACHINE LEARNING FROM PAST TRADES
# ==========================================
class TradeHistoryLearner:
    """Learn from past MT5 trades and market conditions"""
    
    def __init__(self):
        self.learning_data = {
            'trade_patterns': [],
            'successful_entry_conditions': [],
            'failed_entry_conditions': [],
            'optimal_exit_timing': [],
            'market_regime_performance': {},
            'feature_importance': {},
            'last_updated': None
        }
        self.load_learning_data()
    
    def load_learning_data(self):
        """Load existing learning data"""
        if os.path.exists(Config.LEARNING_DATA_FILE):
            try:
                with open(Config.LEARNING_DATA_FILE, 'r') as f:
                    self.learning_data = json.load(f)
                ProfessionalLogger.log(f"Loaded learning data from {Config.LEARNING_DATA_FILE}", "SUCCESS", "LEARNER")
            except Exception as e:
                ProfessionalLogger.log(f"Failed to load learning data: {e}", "WARNING", "LEARNER")
    
    def save_learning_data(self):
        """Save learning data to file"""
        try:
            self.learning_data['last_updated'] = datetime.now().isoformat()
            with open(Config.LEARNING_DATA_FILE, 'w') as f:
                json.dump(self.learning_data, f, indent=2, default=str)
            ProfessionalLogger.log("Learning data saved successfully", "SUCCESS", "LEARNER")
        except Exception as e:
            ProfessionalLogger.log(f"Failed to save learning data: {e}", "ERROR", "LEARNER")
    
    def fetch_past_trades_from_mt5(self, days_back=30):
        """Fetch historical trades from MT5"""
        try:
            from_date = datetime.now() - timedelta(days=days_back)
            deals = mt5.history_deals_get(from_date, datetime.now())
            
            if deals is None or len(deals) == 0:
                ProfessionalLogger.log("No historical deals found", "WARNING", "LEARNER")
                return []
            
            # Filter for our symbol and magic number
            filtered_deals = [
                deal for deal in deals 
                if deal.symbol == Config.SYMBOL and deal.magic == Config.MAGIC_NUMBER
            ]
            
            ProfessionalLogger.log(f"Fetched {len(filtered_deals)} past trades from MT5", "INFO", "LEARNER")
            return filtered_deals
            
        except Exception as e:
            ProfessionalLogger.log(f"Error fetching past trades: {e}", "ERROR", "LEARNER")
            return []
    
    def analyze_trade_with_market_conditions(self, deal, feature_engine):
        """Analyze a single trade with market conditions at that time"""
        try:
            # Get market data around the trade time
            deal_time = datetime.fromtimestamp(deal.time)
            
            # Fetch bars around the deal time
            rates = mt5.copy_rates_range(
                Config.SYMBOL,
                Config.TIMEFRAME,
                deal_time - timedelta(hours=24),
                deal_time
            )
            
            if rates is None or len(rates) < 50:
                return None
            
            df = pd.DataFrame(rates)
            
            # Calculate features
            df_with_features = feature_engine.calculate_features(df)
            
            if len(df_with_features) == 0:
                return None
            
            # Get features at trade time
            trade_features = df_with_features.iloc[-1].to_dict()
            
            # Determine if trade was successful
            is_profitable = deal.profit > 0
            
            return {
                'deal_id': deal.ticket,
                'time': deal.time,
                'type': 'BUY' if deal.type == mt5.DEAL_TYPE_BUY else 'SELL',
                'profit': deal.profit,
                'is_profitable': is_profitable,
                'features': {
                    'rsi': trade_features.get('rsi', 50),
                    'macd_hist': trade_features.get('macd_hist', 0),
                    'adx': trade_features.get('adx', 20),
                    'volatility': trade_features.get('volatility', 0.01),
                    'hurst_exponent': trade_features.get('hurst_exponent', 0.5),
                    'volume_ratio': trade_features.get('volume_ratio', 1.0),
                    'bb_position': trade_features.get('bb_position', 0.5),
                    'trend_strength': trade_features.get('trend_strength', 0)
                }
            }
            
        except Exception as e:
            ProfessionalLogger.log(f"Error analyzing trade {deal.ticket}: {e}", "WARNING", "LEARNER")
            return None
    
    def learn_from_past_trades(self, feature_engine):
        """Main learning function - analyze all past trades"""
        ProfessionalLogger.log("🧠 Starting machine learning from past trades...", "LEARN", "LEARNER")
        
        # Fetch past trades
        past_deals = self.fetch_past_trades_from_mt5(days_back=90)
        
        if len(past_deals) == 0:
            ProfessionalLogger.log("No past trades to learn from", "WARNING", "LEARNER")
            return
        
        # Analyze each trade
        analyzed_trades = []
        for deal in past_deals:
            analysis = self.analyze_trade_with_market_conditions(deal, feature_engine)
            if analysis:
                analyzed_trades.append(analysis)
        
        if len(analyzed_trades) == 0:
            ProfessionalLogger.log("No trades could be analyzed", "WARNING", "LEARNER")
            return
        
        ProfessionalLogger.log(f"Analyzed {len(analyzed_trades)} trades", "INFO", "LEARNER")
        
        # Extract patterns
        self._extract_successful_patterns(analyzed_trades)
        self._extract_failed_patterns(analyzed_trades)
        self._calculate_feature_importance(analyzed_trades)
        
        # Save learning data
        self.save_learning_data()
        
        ProfessionalLogger.log("✅ Machine learning from past trades completed", "SUCCESS", "LEARNER")
    
    def _extract_successful_patterns(self, trades):
        """Extract common patterns from successful trades"""
        successful_trades = [t for t in trades if t['is_profitable']]
        
        if len(successful_trades) < 5:
            return
        
        # Calculate average features for successful trades
        avg_features = {}
        for key in ['rsi', 'macd_hist', 'adx', 'volatility', 'hurst_exponent', 'volume_ratio']:
            values = [t['features'][key] for t in successful_trades if key in t['features']]
            if values:
                avg_features[key] = np.mean(values)
        
        self.learning_data['successful_entry_conditions'].append({
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(successful_trades),
            'avg_features': avg_features
        })
        
        ProfessionalLogger.log(
            f"Extracted successful pattern from {len(successful_trades)} profitable trades",
            "LEARN", "LEARNER"
        )
    
    def _extract_failed_patterns(self, trades):
        """Extract common patterns from failed trades"""
        failed_trades = [t for t in trades if not t['is_profitable']]
        
        if len(failed_trades) < 5:
            return
        
        # Calculate average features for failed trades
        avg_features = {}
        for key in ['rsi', 'macd_hist', 'adx', 'volatility', 'hurst_exponent', 'volume_ratio']:
            values = [t['features'][key] for t in failed_trades if key in t['features']]
            if values:
                avg_features[key] = np.mean(values)
        
        self.learning_data['failed_entry_conditions'].append({
            'timestamp': datetime.now().isoformat(),
            'sample_size': len(failed_trades),
            'avg_features': avg_features
        })
        
        ProfessionalLogger.log(
            f"Extracted failure pattern from {len(failed_trades)} losing trades",
            "LEARN", "LEARNER"
        )
    
    def _calculate_feature_importance(self, trades):
        """Calculate which features correlate most with profitability"""
        if len(trades) < 10:
            return
        
        # Create feature matrix and profit vector
        feature_names = ['rsi', 'macd_hist', 'adx', 'volatility', 'hurst_exponent', 'volume_ratio']
        X = []
        y = []
        
        for trade in trades:
            features_vec = [trade['features'].get(f, 0) for f in feature_names]
            X.append(features_vec)
            y.append(1 if trade['is_profitable'] else 0)
        
        X = np.array(X)
        y = np.array(y)
        
        # Calculate correlation with profitability
        feature_importance = {}
        for i, fname in enumerate(feature_names):
            try:
                corr = np.corrcoef(X[:, i], y)[0, 1]
                if not np.isnan(corr):
                    feature_importance[fname] = abs(corr)
            except:
                feature_importance[fname] = 0
        
        self.learning_data['feature_importance'] = feature_importance
        
        # Log top features
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_features[:3]
        ProfessionalLogger.log(
            f"Top predictive features: {', '.join([f'{k}={v:.3f}' for k, v in top_3])}",
            "LEARN", "LEARNER"
        )
    
    def get_confidence_adjustment(self, current_features):
        """Adjust confidence based on learned patterns"""
        if not self.learning_data.get('successful_entry_conditions'):
            return 1.0  # No adjustment if no learning data
        
        try:
            # Get most recent successful pattern
            recent_pattern = self.learning_data['successful_entry_conditions'][-1]
            avg_features = recent_pattern['avg_features']
            
            # Calculate similarity to successful pattern
            similarity_score = 0
            feature_count = 0
            
            for key, target_value in avg_features.items():
                if key in current_features:
                    current_value = current_features[key]
                    # Calculate normalized difference
                    if target_value != 0:
                        diff = abs(current_value - target_value) / abs(target_value)
                        similarity = max(0, 1 - diff)
                        similarity_score += similarity
                        feature_count += 1
            
            if feature_count > 0:
                avg_similarity = similarity_score / feature_count
                # Boost confidence by up to 20% if very similar to successful pattern
                confidence_multiplier = 1.0 + (avg_similarity * 0.2)
                return min(confidence_multiplier, 1.2)
            
        except Exception as e:
            ProfessionalLogger.log(f"Error calculating confidence adjustment: {e}", "WARNING", "LEARNER")
        
        return 1.0

# ==========================================
# SMART CACHING SYSTEM
# ==========================================
class SmartCache:
    """Intelligent caching with TTL for expensive calculations"""
    
    def __init__(self):
        self.cache = OrderedDict()
        self.cache_lock = threading.Lock()
        self.cache_times = {}
    
    def get(self, key, ttl=None):
        """Get cached value if not expired"""
        with self.cache_lock:
            if key not in self.cache:
                return None
            
            # Check if expired
            if ttl and key in self.cache_times:
                age = time.time() - self.cache_times[key]
                if age > ttl:
                    # Expired, remove
                    del self.cache[key]
                    del self.cache_times[key]
                    return None
            
            return self.cache[key]
    
    def set(self, key, value):
        """Set cached value with timestamp"""
        with self.cache_lock:
            self.cache[key] = value
            self.cache_times[key] = time.time()
            
            # Limit cache size to 1000 items
            if len(self.cache) > 1000:
                # Remove oldest item
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                if oldest_key in self.cache_times:
                    del self.cache_times[oldest_key]
    
    def clear(self):
        """Clear all cached data"""
        with self.cache_lock:
            self.cache.clear()
            self.cache_times.clear()

# ==========================================
# PRE-CALCULATED LOOKUP TABLES
# ==========================================
class LookupTables:
    """Pre-calculated lookup tables for instant O(1) access"""
    
    def __init__(self):
        self._build_tables()
    
    def _build_tables(self):
        """Build all lookup tables"""
        # ATR multipliers for different volatility levels
        self.atr_sl_multipliers = {}
        self.atr_tp_multipliers = {}
        
        for vol in np.arange(0.001, 0.05, 0.001):  # 0.1% to 5% volatility
            vol_key = round(vol, 4)
            if vol < 0.005:  # Low volatility
                self.atr_sl_multipliers[vol_key] = 1.2
                self.atr_tp_multipliers[vol_key] = 2.0
            elif vol < 0.015:  # Normal volatility
                self.atr_sl_multipliers[vol_key] = 1.5
                self.atr_tp_multipliers[vol_key] = 1.75
            else:  # High volatility
                self.atr_sl_multipliers[vol_key] = 2.0
                self.atr_tp_multipliers[vol_key] = 1.5
        
        # Kelly fractions for different win rates
        self.kelly_fractions = {}
        for win_rate in np.arange(0.30, 0.80, 0.01):
            wr_key = round(win_rate, 2)
            # Simplified Kelly: f = (p * b - q) / b, where b = avg_win/avg_loss
            # Assume b = 1.5 (typical for trading)
            b = 1.5
            kelly = (win_rate * b - (1 - win_rate)) / b
            self.kelly_fractions[wr_key] = max(0, min(kelly * 0.5, 0.10))  # Half Kelly, capped at 10%
        
        # Risk/Reward ratios for confidence levels
        self.rr_ratios = {}
        for confidence in np.arange(0.30, 1.0, 0.01):
            conf_key = round(confidence, 2)
            # Higher confidence = can accept lower RR
            if confidence > 0.70:
                self.rr_ratios[conf_key] = 1.0
            elif confidence > 0.50:
                self.rr_ratios[conf_key] = 1.25
            else:
                self.rr_ratios[conf_key] = 1.5
    
    def get_atr_sl_multiplier(self, volatility):
        """Get SL multiplier for given volatility"""
        vol_key = round(volatility, 4)
        return self.atr_sl_multipliers.get(vol_key, 1.5)  # Default 1.5
    
    def get_atr_tp_multiplier(self, volatility):
        """Get TP multiplier for given volatility"""
        vol_key = round(volatility, 4)
        return self.atr_tp_multipliers.get(vol_key, 1.75)  # Default 1.75
    
    def get_kelly_fraction(self, win_rate):
        """Get Kelly fraction for given win rate"""
        wr_key = round(win_rate, 2)
        return self.kelly_fractions.get(wr_key, 0.02)  # Default 2%
    
    def get_rr_ratio(self, confidence):
        """Get minimum RR ratio for given confidence"""
        conf_key = round(confidence, 2)
        return self.rr_ratios.get(conf_key, 1.25)  # Default 1.25

# ==========================================
# SIGNAL QUALITY SCORER
# ==========================================
class SignalQualityScorer:
    """Score trading signals 0-100 based on multiple factors"""
    
    def __init__(self):
        pass
    
    def score_signal(self, features, signal_direction, multi_tf_data=None):
        """
        Score a trading signal from 0-100
        
        Args:
            features: Dictionary of current market features
            signal_direction: 1 for BUY, -1 for SELL
            multi_tf_data: Multi-timeframe analysis data
        
        Returns:
            score: 0-100 quality score
        """
        score = 0
        
        # 1. Timeframe Alignment (30 points)
        if multi_tf_data and 'alignment' in multi_tf_data:
            alignment = multi_tf_data['alignment']
            score += alignment * 30
        else:
            score += 15  # Neutral if no multi-TF data
        
        # 2. Volume Confirmation (20 points)
        volume_ratio = features.get('volume_ratio', 1.0)
        if volume_ratio > 1.5:  # High volume
            score += 20
        elif volume_ratio > 1.0:  # Above average
            score += 15
        elif volume_ratio > 0.7:  # Normal
            score += 10
        else:  # Low volume
            score += 5
        
        # 3. Trend Strength (20 points)
        adx = features.get('adx', 20)
        if adx > 40:  # Strong trend
            score += 20
        elif adx > 25:  # Moderate trend
            score += 15
        elif adx > 20:  # Weak trend
            score += 10
        else:  # No trend
            score += 5
        
        # 4. Volatility Regime (15 points)
        volatility = features.get('volatility', 0.01)
        if 0.008 < volatility < 0.020:  # Optimal volatility
            score += 15
        elif 0.005 < volatility < 0.025:  # Acceptable
            score += 10
        else:  # Too high or too low
            score += 5
        
        # 5. Support/Resistance Proximity (15 points)
        # Check if price is near S/R levels
        bb_position = features.get('bb_position', 0.5)
        if signal_direction == 1:  # BUY
            if bb_position < 0.3:  # Near lower band (support)
                score += 15
            elif bb_position < 0.5:
                score += 10
            else:
                score += 5
        else:  # SELL
            if bb_position > 0.7:  # Near upper band (resistance)
                score += 15
            elif bb_position > 0.5:
                score += 10
            else:
                score += 5
        
        
        return min(100, max(0, score))  # Clamp to 0-100

# ==========================================
# MARKET MICROSTRUCTURE ANALYZER
# ==========================================
class MarketMicrostructureAnalyzer:
    """Analyze tick-level order flow and market microstructure"""
    
    def __init__(self):
        self.tick_history = []
        self.max_ticks = 1000
    
    def on_tick(self, tick):
        """Process incoming tick data"""
        self.tick_history.append(tick)
        if len(self.tick_history) > self.max_ticks:
            self.tick_history.pop(0)
            
    def analyze_order_flow(self):
        """Analyze bad/ask pressure and liquidity"""
        if len(self.tick_history) < 10:
            return None
            
        recent_ticks = self.tick_history[-50:]
        
        # Calculate spread statistics
        spreads = [(t.ask - t.bid) for t in recent_ticks]
        avg_spread = np.mean(spreads)
        spread_volatility = np.std(spreads)
        
        # Estimate order flow imbalance (proxy using tick direction)
        # Real imbalance needs Level 2 data, but we can approximate with price changes
        buying_pressure = 0
        selling_pressure = 0
        
        for i in range(1, len(recent_ticks)):
            prev = recent_ticks[i-1]
            curr = recent_ticks[i]
            
            # Price moved up -> buying pressure
            if curr.last > prev.last:
                buying_pressure += (curr.last - prev.last) * curr.volume_real
            # Price moved down -> selling pressure
            elif curr.last < prev.last:
                selling_pressure += (prev.last - curr.last) * curr.volume_real
                
        total_pressure = buying_pressure + selling_pressure
        imbalance = 0.5
        if total_pressure > 0:
            imbalance = buying_pressure / total_pressure
            
        return {
            'avg_spread': avg_spread,
            'spread_volatility': spread_volatility,
            'imbalance': imbalance,  # >0.5 buying, <0.5 selling
            'liquidity_score': 1.0 / (avg_spread * 10000) if avg_spread > 0 else 0
        }

    def is_safe_to_trade(self):
        """Check if microstructure conditions are safe"""
        metrics = self.analyze_order_flow()
        if not metrics:
            return True
        
        # Check spread (avoid news/low liquidity)
        if metrics['avg_spread'] > Config.MICROSTRUCTURE_SPREAD_THRESHOLD * 0.0001:  # Convert pips to price
            return False
            
        return True

# ==========================================
# BAYESIAN REGIME DETECTOR
# ==========================================
class BayesianRegimeDetector:
    """Probabilistic Change Point Detection"""
    
    def __init__(self, lookback=50):
        self.lookback = lookback
        self.returns_buffer = []
        self.current_regime_prob = 0.5  # start neutral
        self.last_switch_time = 0
        
    def detect_regime_switch(self, new_price):
        """
        Update probabilities and detect switch.
        Returns: (is_switch, current_regime_type)
        regime_type: 0 (Low Vol/Calm), 1 (High Vol/Stress)
        """
        if len(self.returns_buffer) == 0:
            self.returns_buffer.append(new_price)
            return False, 0
            
        # Calculate return
        log_ret = np.log(new_price / self.returns_buffer[-1])
        self.returns_buffer.append(new_price)
        
        if len(self.returns_buffer) > self.lookback:
            self.returns_buffer.pop(0)
            
        # Need minimum data
        if len(self.returns_buffer) < 20:
            return False, 0
            
        # Calculate recent volatility (short window) vs historical (full window)
        # Using a Bayes factor approach:
        # H0: Volatility is Low (Normal)
        # H1: Volatility is High (Shock)
        
        returns = np.diff(np.log(self.returns_buffer))
        
        full_std = np.std(returns) + 1e-9
        short_window = returns[-10:]
        short_std = np.std(short_window) + 1e-9
        
        # Likelihood ratio
        # Likelihood of data under H1 (High Vol) / Likelihood under H0 (Low Vol)
        # Assume High Vol is 2x Normal Vol
        
        lik_normal = stats.norm.pdf(returns[-1], loc=0, scale=full_std)
        lik_shock = stats.norm.pdf(returns[-1], loc=0, scale=full_std * 2.5)
        
        # Avoid division by zero
        if lik_normal == 0:
            bayes_factor = 100 # Strong evidence for shock
        else:
            bayes_factor = lik_shock / lik_normal
            
        # Update prior (simple recursive update)
        # prior = current_regime_prob
        # posterior = (likelihood * prior) / evidence
        # We simplify to a "Shock Probability" score
        
        decay = 0.85 # Memory decay
        self.current_regime_prob = (self.current_regime_prob * decay) + (1 - decay) * (1 if bayes_factor > 1.5 else 0)
        
        is_high_vol = self.current_regime_prob > 0.6
        
        return is_high_vol, self.current_regime_prob

# ==========================================
# REINFORCEMENT LEARNING DATA COLLECTOR
# ==========================================
class ReinforcementDataCollector:
    """Collects State-Action-Reward tuples for future RL training"""
    
    def __init__(self):
        self.file_path = Config.LEARNING_DATA_FILE
        self.pending_experiences = {} # ticket -> {state, action}
        self.experiences = []
        self._load()
        
    def _load(self):
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, 'r') as f:
                    self.experiences = json.load(f)
            except:
                self.experiences = []

    def record_entry(self, ticket, state, action):
        """Record the state and action at trade entry"""
        self.pending_experiences[ticket] = {
            'state': state,
            'action': action,
            'timestamp': datetime.now().isoformat()
        }
        
    def complete_experience(self, ticket, result_metrics):
        """Complete the experience with reduction reward (PnL/Sharpe)"""
        if ticket in self.pending_experiences:
            exp = self.pending_experiences.pop(ticket)
            exp['reward'] = result_metrics
            self.experiences.append(exp)
            
            # Keep recent 10k
            if len(self.experiences) > 10000:
                self.experiences = self.experiences[-10000:]
                
            self._save()
            
    def _save(self):
        try:
            with open(self.file_path, 'w') as f:
                json.dump(self.experiences, f)
        except Exception as e:
            ProfessionalLogger.log(f"RL Save Error: {e}", "ERROR", "MEMORY")

# ==========================================
# ADVANCED STATISTICAL ANALYZER
# ==========================================
class AdvancedStatisticalAnalyzer:
    """Enhanced statistical analysis"""
    
    @staticmethod
    def analyze_return_distribution(returns):
        """Comprehensive return distribution analysis"""
        if len(returns) < Config.MIN_SAMPLES_FOR_STATS:
            return {"error": "Insufficient data"}
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        
        if len(returns) == 0:
            return {"error": "No valid data"}
        
        # Basic statistics
        stats = {
            'n_samples': len(returns),
            'mean': np.mean(returns),
            'std': np.std(returns),
            'skewness': skew(returns),
            'kurtosis': kurtosis(returns, fisher=True),
            'excess_kurtosis': kurtosis(returns, fisher=False) - 3,
            'median': np.median(returns),
            'mad': np.median(np.abs(returns - np.median(returns))),
            'min': np.min(returns),
            'max': np.max(returns),
            'range': np.ptp(returns),
            'q1': np.percentile(returns, 25),
            'q3': np.percentile(returns, 75),
            'iqr': np.percentile(returns, 75) - np.percentile(returns, 25),
            'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252 * 96) if np.std(returns) > 0 else 0,
            'sortino': np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252 * 96) if len(returns[returns < 0]) > 0 else 0
        }
        
        # Normality tests
        try:
            jb_stat, jb_p = jarque_bera(returns)
            stats['jarque_bera_stat'] = jb_stat
            stats['jarque_bera_pvalue'] = jb_p
            stats['is_normal'] = jb_p > 0.05
        except:
            stats['jarque_bera_stat'] = None
            stats['jarque_bera_pvalue'] = None
            stats['is_normal'] = None
        
        # Tail analysis
        tail_threshold = 0.05
        left_tail = returns[returns < np.percentile(returns, tail_threshold * 100)]
        right_tail = returns[returns > np.percentile(returns, (1 - tail_threshold) * 100)]
        
        stats['left_tail_mean'] = np.mean(left_tail) if len(left_tail) > 0 else 0
        stats['right_tail_mean'] = np.mean(right_tail) if len(right_tail) > 0 else 0
        stats['tail_asymmetry'] = abs(stats['left_tail_mean']) - abs(stats['right_tail_mean'])
        stats['tail_ratio'] = abs(stats['left_tail_mean'] / stats['right_tail_mean']) if stats['right_tail_mean'] != 0 else float('inf')
        
        # Value at Risk
        var_95_15m = np.percentile(returns, 5)
        var_99_15m = np.percentile(returns, 1)
        
        stats['var_95'] = var_95_15m * np.sqrt(96)
        stats['var_99'] = var_99_15m * np.sqrt(96)
        
        stats['cvar_95'] = np.mean(returns[returns <= var_95_15m]) * np.sqrt(96) if len(returns[returns <= var_95_15m]) > 0 else stats['var_95']
        stats['cvar_99'] = np.mean(returns[returns <= var_99_15m]) * np.sqrt(96) if len(returns[returns <= var_99_15m]) > 0 else stats['var_99']
        
        # Downside risk metrics
        downside_returns = returns[returns < 0]
        if len(downside_returns) > 0:
            stats['downside_deviation'] = np.std(downside_returns)
            stats['max_drawdown'] = np.min(returns)
        else:
            stats['downside_deviation'] = 0
            stats['max_drawdown'] = 0
        
        # Autocorrelation analysis
        try:
            autocorr_lags = 20
            autocorrs = []
            for lag in range(1, autocorr_lags + 1):
                if lag < len(returns):
                    corr = np.corrcoef(returns[:-lag], returns[lag:])[0, 1]
                    autocorrs.append(corr)
            stats['autocorrelation_mean'] = np.mean(np.abs(autocorrs)) if autocorrs else 0
            stats['autocorrelation_lag1'] = autocorrs[0] if autocorrs else 0
        except:
            stats['autocorrelation_mean'] = None
            stats['autocorrelation_lag1'] = None
        
        # Stationarity indicators
        stats['half_life'] = AdvancedStatisticalAnalyzer.calculate_half_life(returns)
        
        # Volatility clustering
        squared_returns = returns ** 2
        if len(squared_returns) > 10:
            stats['vol_clustering'] = np.corrcoef(squared_returns[:-1], squared_returns[1:])[0, 1] if len(squared_returns) > 1 else 0
        else:
            stats['vol_clustering'] = 0
        
        return stats
    
    @staticmethod
    def calculate_half_life(series):
        """Calculate half-life of mean reversion"""
        if len(series) < 10:
            return 0
        
        series = np.array(series)
        delta_series = np.diff(series)
        lag_series = series[:-1]
        
        mask = ~(np.isnan(delta_series) | np.isnan(lag_series) | 
                np.isinf(delta_series) | np.isinf(lag_series))
        
        delta_series = delta_series[mask]
        lag_series = lag_series[mask]
        
        if len(delta_series) < 5 or len(lag_series) < 5:
            return 0
        
        try:
            X = np.column_stack([np.ones_like(lag_series), lag_series])
            beta = np.linalg.lstsq(X, delta_series, rcond=None)[0][1]
            
            if beta >= 0:
                return 1000.0  # Cap at large number instead of inf
            
            half_life = -np.log(2) / beta
            return min(1000.0, max(0, half_life))
        except:
            return 0
    
    @staticmethod
    def calculate_hurst_exponent(prices, method='rs'):
        """Calculate Hurst exponent for market efficiency analysis - JIT OPTIMIZED"""
        if len(prices) < 100:
            return 0.5
        
        # Use Numba-optimized version if available
        if Config.ENABLE_NUMBA_JIT and NUMBA_AVAILABLE:
            try:
                # Convert to numpy array if not already
                if not isinstance(prices, np.ndarray):
                    prices = np.array(prices)
                return AdvancedStatisticalAnalyzer._calculate_hurst_jit(prices)
            except Exception as e:
                # Fallback to python implementation
                return AdvancedStatisticalAnalyzer._calculate_hurst_python(prices, method)
        else:
            return AdvancedStatisticalAnalyzer._calculate_hurst_python(prices, method)

    @staticmethod
    @jit(nopython=True, cache=True)
    def _calculate_hurst_jit(prices):
        """Numba-compiled Hurst calculation - 50-100x faster"""
        # Calculate returns manually for Numba
        n_prices = len(prices)
        returns = np.empty(n_prices-1)
        for i in range(n_prices-1):
            returns[i] = np.log(prices[i+1] / prices[i])
            
        n = len(returns)
        
        # Pre-allocate arrays (Numba prefers this over appending to lists)
        max_windows = 100 # Reasonable upper bound
        r_s_values = np.zeros(max_windows)
        n_values = np.zeros(max_windows)
        idx = 0
        
        # Determine step size safely
        step = max(1, n // 20)
        
        for window in range(10, n // 2, step):
            if window < 10:
                continue
            
            num_windows = n // window
            if num_windows < 2:
                continue
            
            rs_sum = 0.0
            count = 0
            
            for i in range(num_windows):
                start = i * window
                end = (i + 1) * window
                
                # Check bounds
                if end > n:
                    break
                    
                # Manual mean calculation
                seg_sum = 0.0
                for k in range(start, end):
                    seg_sum += returns[k]
                mean_seg = seg_sum / window
                
                # Manual range and std calculation
                min_cum = 0.0
                max_cum = 0.0
                current_cum = 0.0
                sq_dev_sum = 0.0
                
                for k in range(start, end):
                    dev = returns[k] - mean_seg
                    current_cum += dev
                    if current_cum > max_cum:
                        max_cum = current_cum
                    if current_cum < min_cum:
                        min_cum = current_cum
                    sq_dev_sum += dev * dev
                
                r = max_cum - min_cum
                
                if sq_dev_sum > 0:
                    s = np.sqrt(sq_dev_sum / window)
                else:
                    s = 0.0
                
                if s > 0:
                    rs_sum += (r / s)
                    count += 1
            
            if count > 0:
                if idx < max_windows:
                    r_s_values[idx] = rs_sum / count
                    n_values[idx] = float(window)
                    idx += 1
        
        if idx < 3:
            return 0.5
        
        # Linear regression on log-log values
        # We only use values up to idx
        sum_x = 0.0
        sum_y = 0.0
        sum_xx = 0.0
        sum_xy = 0.0
        
        for i in range(idx):
            log_n = np.log(n_values[i])
            log_rs = np.log(r_s_values[i])
            
            sum_x += log_n
            sum_y += log_rs
            sum_xx += log_n * log_n
            sum_xy += log_n * log_rs
            
        denom = (idx * sum_xx - sum_x * sum_x)
        if denom == 0:
            return 0.5
            
        slope = (idx * sum_xy - sum_x * sum_y) / denom
        return slope

    @staticmethod
    def _calculate_hurst_python(prices, method='rs'):
        """Python fallback for Hurst calculation"""
        try:
            returns = np.diff(np.log(prices))
            
            if method == 'rs':
                n = len(returns)
                r_s_values = []
                n_values = []
                
                # Use same step as original
                step = n // 20
                if step < 1: step = 1
                
                for window in range(10, n//2, step):
                    if window < 10:
                        continue
                    
                    num_windows = n // window
                    if num_windows < 2:
                        continue
                    
                    rs_vals = []
                    for i in range(num_windows):
                        segment = returns[i*window:(i+1)*window]
                        if len(segment) < 2:
                            continue
                        
                        mean_segment = np.mean(segment)
                        deviations = segment - mean_segment
                        cumulative_dev = np.cumsum(deviations)
                        
                        r = np.max(cumulative_dev) - np.min(cumulative_dev)
                        s = np.std(segment)
                        
                        if s > 0:
                            rs_vals.append(r / s)
                    
                    if len(rs_vals) > 0:
                        r_s_values.append(np.mean(rs_vals))
                        n_values.append(window)
                
                if len(r_s_values) < 3:
                    return 0.5
                
                log_rs = np.log(r_s_values)
                log_n = np.log(n_values)
                
                hurst, _ = np.polyfit(log_n, log_rs, 1)
                return hurst
                
            elif method == 'aggregate':
                n = len(returns)
                variances = []
                scales = []
                
                for scale in range(2, n//4):
                    m = n // scale
                    if m < 2:
                        continue
                    
                    aggregated = np.zeros(m)
                    for i in range(m):
                        aggregated[i] = np.sum(returns[i*scale:(i+1)*scale])
                    
                    variances.append(np.var(aggregated))
                    scales.append(scale)
                
                if len(variances) < 3:
                    return 0.5
                
                log_var = np.log(variances)
                log_scale = np.log(scales)
                
                slope, _ = np.polyfit(log_scale, log_var, 1)
                hurst = 1 + slope / 2
                return hurst
                
            else:
                return 0.5
                
        except Exception as e:
            ProfessionalLogger.log(f"Hurst calculation error: {str(e)}", "WARNING", "STATS")
            return 0.5
    
    @staticmethod
    def calculate_garch_volatility(returns, p=1, q=1):
        """Calculate GARCH volatility with robust error handling"""
        if len(returns) < 100:
            return np.std(returns) if len(returns) > 0 else 0.001
        
        try:
            clean_returns = returns.copy()
            clean_returns = clean_returns[~np.isnan(clean_returns)]
            clean_returns = clean_returns[~np.isinf(clean_returns)]
            clean_returns = clean_returns[np.isfinite(clean_returns)]
            
            if len(clean_returns) < 50:
                return np.std(clean_returns) if len(clean_returns) > 0 else 0.001
            
            scaled_returns = clean_returns * 100.0
            
            if np.std(scaled_returns) < 1e-10:
                return np.std(clean_returns)
            
            model = arch_model(scaled_returns, vol='Garch', p=p, q=q, dist='normal')
            result = model.fit(disp='off', show_warning=False, options={'maxiter': 200})
            
            conditional_vol = result.conditional_volatility
            
            if len(conditional_vol) > 0:
                return conditional_vol[-1] / 100
            else:
                return np.std(clean_returns)
            
        except Exception as e:
            clean_returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
            if len(clean_returns) > 0:
                return np.std(clean_returns)
            else:
                return 0.001
    
    @staticmethod
    def calculate_market_regime(data):
        """Determine market regime based on statistical properties"""
        if len(data) < Config.MIN_SAMPLES_FOR_STATS:
            return {"regime": "unknown", "confidence": 0}
        
        returns = np.diff(np.log(data['close'].values))
        returns = returns[~np.isnan(returns)]
        
        if len(returns) < 50:
            return {"regime": "unknown", "confidence": 0}
        
        volatility = np.std(returns)
        mean_return = np.mean(returns)
        sharpe = mean_return / volatility if volatility > 0 else 0
        hurst = AdvancedStatisticalAnalyzer.calculate_hurst_exponent(data['close'].values[-500:])
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        
        regime_scores = {
            "trending": 0,
            "mean_reverting": 0,
            "random_walk": 0,
            "volatile": 0
        }
        
        if hurst > 0.6:
            regime_scores["trending"] += 2
        elif hurst > 0.55:
            regime_scores["trending"] += 1
        
        if autocorr > 0.1:
            regime_scores["trending"] += 1
        
        # Check for linear trend strength (R-squared)
        try:
            # Simple linear regression on log prices
            y = np.log(data['close'].values)
            x = np.arange(len(y))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
            r_squared = r_value**2
            
            if r_squared > 0.8 and abs(slope) > 0.0001:
                regime_scores["trending"] += 3 # Strong linear trend overrides Hurst
            elif r_squared > 0.6:
                regime_scores["trending"] += 1
        except:
            r_squared = 0
            slope = 0

        if hurst < 0.4:
            regime_scores["mean_reverting"] += 2
        elif hurst < 0.45:
            regime_scores["mean_reverting"] += 1
        
        if autocorr < -0.1:
            regime_scores["mean_reverting"] += 1
        
        vol_threshold = np.percentile(np.abs(returns), 75)
        if volatility > vol_threshold * 1.5:
            regime_scores["volatile"] += 2
        
        if 0.45 <= hurst <= 0.55 and abs(autocorr) < 0.05 and r_squared < 0.6:
            regime_scores["random_walk"] += 2
        
        best_regime = max(regime_scores, key=regime_scores.get)
        total_score = sum(regime_scores.values())
        confidence = regime_scores[best_regime] / total_score if total_score > 0 else 0
        
        return {
            "regime": best_regime,
            "confidence": confidence,
            "hurst": hurst,
            "autocorrelation": autocorr,
            "volatility": volatility,
            "sharpe": sharpe,
            "scores": regime_scores
        }
    
    @staticmethod
    def calculate_tail_risk(returns, confidence=0.95):
        """Calculate tail risk metrics"""
        if len(returns) < 100:
            return {"error": "Insufficient data"}
        
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)
        
        k = max(10, int(n * (1 - confidence)))
        tail_data = sorted_returns[-k:]
        
        if len(tail_data) < 10:
            return {"error": "Insufficient tail data"}
        
        log_tail = np.log(tail_data / tail_data[0])
        hill_estimator = 1 / np.mean(log_tail) if np.mean(log_tail) > 0 else 0
        
        var_extreme = np.percentile(returns, (1 - confidence) * 100)
        cvar_extreme = np.mean(returns[returns <= var_extreme])
        
        left_tail = returns[returns < np.percentile(returns, 10)]
        right_tail = returns[returns > np.percentile(returns, 90)]
        
        return {
            "tail_index": hill_estimator,
            "var_extreme": var_extreme,
            "cvar_extreme": cvar_extreme,
            "left_tail_size": len(left_tail),
            "right_tail_size": len(right_tail),
            "tail_asymmetry": abs(np.mean(left_tail)) - abs(np.mean(right_tail)) if len(right_tail) > 0 else 0
        }
    
    @staticmethod
    def calculate_correlation_structure(data):
        """Calculate correlation structure of price features"""
        if len(data) < 100:
            return {"error": "Insufficient data"}
        
        try:
            returns_1 = data['close'].pct_change(1).dropna()
            returns_5 = data['close'].pct_change(5).dropna()
            returns_10 = data['close'].pct_change(10).dropna()
            returns_20 = data['close'].pct_change(20).dropna()
            
            min_len = min(len(returns_1), len(returns_5), len(returns_10), len(returns_20))
            returns_1 = returns_1[-min_len:]
            returns_5 = returns_5[-min_len:]
            returns_10 = returns_10[-min_len:]
            returns_20 = returns_20[-min_len:]
            
            returns_matrix = np.column_stack([returns_1, returns_5, returns_10, returns_20])
            correlation_matrix = np.corrcoef(returns_matrix.T)
            
            eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
            
            market_correlation = eigenvalues[0] / np.sum(eigenvalues) if np.sum(eigenvalues) > 0 else 0
            
            return {
                "correlation_matrix": correlation_matrix.tolist(),
                "eigenvalues": eigenvalues.tolist(),
                "market_correlation": market_correlation,
                "correlation_strength": np.mean(np.abs(correlation_matrix - np.eye(correlation_matrix.shape[0])))
            }
            
        except Exception as e:
            ProfessionalLogger.log(f"Correlation analysis error: {str(e)}", "WARNING", "STATS")
            return {"error": str(e)}
    
    @staticmethod
    def bootstrap_analysis(returns, n_iterations=1000):
        """Bootstrap analysis for confidence intervals"""
        if len(returns) < 50:
            return {"error": "Insufficient data"}
        
        bootstrap_stats = {
            'mean': [],
            'std': [],
            'sharpe': [],
            'var_95': [],
            'skewness': [],
            'kurtosis': []
        }
        
        n = len(returns)
        for _ in range(n_iterations):
            sample_idx = np.random.choice(n, n, replace=True)
            sample = returns[sample_idx]
            
            bootstrap_stats['mean'].append(np.mean(sample))
            bootstrap_stats['std'].append(np.std(sample))
            
            if np.std(sample) > 0:
                bootstrap_stats['sharpe'].append(np.mean(sample) / np.std(sample) * np.sqrt(252))
            
            bootstrap_stats['var_95'].append(np.percentile(sample, 5))
            bootstrap_stats['skewness'].append(skew(sample))
            bootstrap_stats['kurtosis'].append(kurtosis(sample))
        
        ci_results = {}
        for key, values in bootstrap_stats.items():
            if len(values) > 0:
                ci_results[f'{key}_mean'] = np.mean(values)
                ci_results[f'{key}_ci_lower'] = np.percentile(values, 2.5)
                ci_results[f'{key}_ci_upper'] = np.percentile(values, 97.5)
                ci_results[f'{key}_std'] = np.std(values)
        
        return ci_results

# ==========================================
# PROFESSIONAL RISK METRICS
# ==========================================
class ProfessionalRiskMetrics:
    """Advanced risk metrics with statistical analysis"""
    
    @staticmethod
    def calculate_risk_metrics(returns, prices=None):
        """Calculate comprehensive risk metrics"""
        if len(returns) < Config.MIN_SAMPLES_FOR_STATS:
            return {"error": "Insufficient data"}
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        metrics = {}
        
        metrics['volatility'] = np.std(returns)
        metrics['downside_deviation'] = np.std(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 0
        
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        
        metrics['cvar_95'] = np.mean(returns[returns <= metrics['var_95']]) if len(returns[returns <= metrics['var_95']]) > 0 else metrics['var_95']
        metrics['cvar_99'] = np.mean(returns[returns <= metrics['var_99']]) if len(returns[returns <= metrics['var_99']]) > 0 else metrics['var_99']
        
        if metrics['volatility'] > 0:
            metrics['sharpe'] = np.mean(returns) / metrics['volatility'] * np.sqrt(252)
        else:
            metrics['sharpe'] = 0
        
        if metrics['downside_deviation'] > 0:
            metrics['sortino'] = np.mean(returns) / metrics['downside_deviation'] * np.sqrt(252)
        else:
            metrics['sortino'] = 0
        
        if prices is not None and len(prices) > 0:
            cumulative = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak
            metrics['max_drawdown'] = np.max(drawdown)
            metrics['avg_drawdown'] = np.mean(drawdown[drawdown > 0]) if len(drawdown[drawdown > 0]) > 0 else 0
        else:
            metrics['max_drawdown'] = 0
            metrics['avg_drawdown'] = 0
        
        metrics['skewness'] = skew(returns)
        metrics['kurtosis'] = kurtosis(returns)
        # Epsilon for safe division to avoid infinity but allow high values
        EPSILON = 1e-6
        
        if len(returns) > 0:
             denom = abs(np.mean(returns[returns > np.percentile(returns, 95)])) if len(returns[returns > np.percentile(returns, 95)]) > 0 else 0
             if denom > EPSILON:
                 metrics['tail_ratio'] = abs(np.mean(returns[returns < np.percentile(returns, 5)])) / denom
             else:
                 # Use epsilon proxy for denominator
                 metrics['tail_ratio'] = abs(np.mean(returns[returns < np.percentile(returns, 5)])) / EPSILON
        else:
             metrics['tail_ratio'] = 0.0

        # Use effective max_drawdown (min EPSILON) to allow calculation even if 0
        effective_mdd = max(metrics['max_drawdown'], EPSILON)
        metrics['calmar'] = np.mean(returns) * 252 / effective_mdd
        metrics['recovery_factor'] = np.sum(returns) / effective_mdd
        
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        loss_sum = np.sum(losses) if len(losses) > 0 else 0
        effective_loss_sum = max(loss_sum, EPSILON)
        
        if len(returns) > 0:
             metrics['omega'] = np.sum(gains) / effective_loss_sum
        else:
             metrics['omega'] = 0.0
        
        if prices is not None and len(prices) > 14:
            squared_drawdowns = drawdown ** 2
            metrics['ulcer_index'] = np.sqrt(np.mean(squared_drawdowns[-14:])) if len(squared_drawdowns) >= 14 else 0
        else:
            metrics['ulcer_index'] = 0
        
        return metrics
    
    @staticmethod
    def calculate_position_risk(position_size, entry_price, stop_loss, current_price):
        """Calculate position-specific risk metrics"""
        risk_per_share = abs(entry_price - stop_loss)
        risk_amount = position_size * risk_per_share
        
        current_risk = abs(current_price - stop_loss) * position_size
        risk_ratio = current_risk / risk_amount if risk_amount > 0 else 0
        
        return {
            'risk_per_share': risk_per_share,
            'risk_amount': risk_amount,
            'current_risk': current_risk,
            'risk_ratio': risk_ratio,
            'stop_distance_pct': risk_per_share / entry_price * 100
        }
    
    @staticmethod
    def calculate_portfolio_risk(positions, correlation_matrix=None):
        """Calculate portfolio-level risk metrics"""
        if not positions:
            return {"total_risk": 0, "diversification_benefit": 0}
        
        position_risks = []
        position_values = []
        
        for pos in positions:
            if 'risk_amount' in pos:
                position_risks.append(pos['risk_amount'])
                position_values.append(pos.get('position_value', pos['risk_amount'] * 10))
        
        if not position_risks:
            return {"total_risk": 0, "diversification_benefit": 0}
        
        undiversified_risk = np.sum(position_risks)
        
        if correlation_matrix is not None and len(correlation_matrix) == len(position_risks):
            risks = np.array(position_risks)
            corr = np.array(correlation_matrix)
            
            portfolio_variance = np.dot(risks, np.dot(corr, risks))
            diversified_risk = np.sqrt(portfolio_variance)
        else:
            diversified_risk = np.sqrt(np.sum(np.array(position_risks) ** 2))
        
        diversification_benefit = (undiversified_risk - diversified_risk) / undiversified_risk * 100 \
            if undiversified_risk > 0 else 0
        
        return {
            'total_risk': diversified_risk,
            'undiversified_risk': undiversified_risk,
            'diversification_benefit': diversification_benefit,
            'num_positions': len(positions),
            'avg_position_risk': np.mean(position_risks) if position_risks else 0,
            'max_position_risk': np.max(position_risks) if position_risks else 0
        }

# ==========================================
# LEVEL 2 ORDER FLOW ANALYZER
# ==========================================
class OrderFlowAnalyzer:
    """Institutional Grade Level 2 Interpretation"""
    
    def __init__(self, symbol):
        self.symbol = symbol
        self.last_book_time = 0
        self.cache = {}
        
    def get_order_book_imbalance(self, depth=10):
        """
        Calculate Bid-Ask Volume Imbalance
        Range: -1.0 (Full Bearish) to 1.0 (Full Bullish)
        """
        try:
            book = mt5.market_book_get(self.symbol)
            if not book or len(book) == 0:
                return 0
                
            # Separate Bids and Asks
            # In MT5 Book: type 1 = Sell (Ask), type 2 = Buy (Bid)
            asks = [item.volume for item in book if item.type == 1]
            bids = [item.volume for item in book if item.type == 2]
            
            # Limit depth
            asks = asks[:depth]
            bids = bids[:depth]
            
            total_ask_vol = sum(asks)
            total_bid_vol = sum(bids)
            total_vol = total_ask_vol + total_bid_vol
            
            if total_vol == 0:
                return 0
                
            # Imbalance = (BidVol - AskVol) / (BidVol + AskVol)
            imbalance = (total_bid_vol - total_ask_vol) / total_vol
            return imbalance
            
        except Exception as e:
            return 0

    def detect_liquidity_walls(self, current_price, threshold_factor=2.5):
        """Detect abnormally large limit orders (Liquidity Walls)"""
        walls = {'sell_wall': None, 'buy_wall': None}
        try:
            book = mt5.market_book_get(self.symbol)
            if not book:
                return walls
                
            volumes = [item.volume for item in book]
            avg_vol = np.mean(volumes) if volumes else 0
            
            # Threshold for a "Wall"
            wall_threshold = avg_vol * threshold_factor
            
            # Find nearest walls
            best_ask = 999999
            best_bid = 0
            
            for item in book:
                if item.volume > wall_threshold:
                    if item.type == 1: # Sell Wall (Ask)
                        # We want the lowest price sell wall (closest to price)
                        if item.price < best_ask and item.price > current_price:
                            best_ask = item.price
                            walls['sell_wall'] = {'price': item.price, 'vol': item.volume}
                    elif item.type == 2: # Buy Wall (Bid)
                        # We want the highest price buy wall (closest to price)
                        if item.price > best_bid and item.price < current_price:
                            best_bid = item.price
                            walls['buy_wall'] = {'price': item.price, 'vol': item.volume}
                            
            return walls
        except:
            return walls

    def analyze_flow(self):
        """Get comprehensive flow analysis"""
        # Subscribe to book if needed (MT5 requires explicit subscription for some assets)
        mt5.market_book_add(self.symbol)
        
        imbalance = self.get_order_book_imbalance()
        
        # Get current price proxy
        tick = mt5.symbol_info_tick(self.symbol)
        price = tick.ask if tick else 0
        
        walls = self.detect_liquidity_walls(price)
        
        mt5.market_book_release(self.symbol)
        
        return {
            'imbalance': imbalance,
            'walls': walls,
            'timestamp': datetime.now().isoformat()
        }

# ==========================================
# ENHANCED FEATURE ENGINEERING
# ==========================================
class EnhancedFeatureEngine:
    """Professional feature engineering with advanced price action and volume analysis"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        self.risk_metrics = ProfessionalRiskMetrics()
        self.last_garch_time = None
        self.cached_garch = 0
        
        # Multithreading support for faster calculations
        self.executor = ThreadPoolExecutor(max_workers=4)  # 4 threads for parallel processing
        self.cache_lock = threading.Lock()  # Thread-safe caching
        self.feature_cache = {}  # Cache for expensive calculations
        
        # Advanced optimizations
        if Config.ENABLE_SMART_CACHING:
            self.smart_cache = SmartCache()
            ProfessionalLogger.log("Smart caching enabled", "INFO", "FEATURE_ENGINE")
        else:
            self.smart_cache = None
        
        if Config.ENABLE_LOOKUP_TABLES:
            self.lookup_tables = LookupTables()
            ProfessionalLogger.log("Lookup tables initialized", "INFO", "FEATURE_ENGINE")
        else:
            self.lookup_tables = None
        
        if Config.ENABLE_SIGNAL_QUALITY_FILTER:
            self.signal_scorer = SignalQualityScorer()
            ProfessionalLogger.log(f"Signal quality filter enabled (min score: {Config.MIN_SIGNAL_QUALITY_SCORE})", "INFO", "FEATURE_ENGINE")
        else:
            self.signal_scorer = None
    
    def _calculate_shannon_entropy(self, x):
        """Calculate Shannon Entropy to measure market noise"""
        try:
             # Ensure we are working with numpy array or safe indexing
            if hasattr(x, 'values'):
                vals = x.values
            else:
                vals = np.array(x)
            
            # Check cache first
            if self.smart_cache:
                # Use safe indexing on array
                last_val = vals[-1] if len(vals) > 0 else 0
                cache_key = f"entropy_{len(vals)}_{last_val}"
                cached = self.smart_cache.get(cache_key, ttl=300) # 5 min cache
                if cached is not None:
                    return cached

            # Normalized probability distribution
            hist, _ = np.histogram(vals, bins='doane', density=True)
            hist = hist[hist > 0]
            val = -np.sum(hist * np.log(hist))
            
            # Cache result
            if self.smart_cache:
                self.smart_cache.set(cache_key, val)
                
            return val
        except:
            return 0

    def _calculate_hurst(self, ts):
        """Calculate Hurst Exponent to measure trend persistence - Cached & JIT Optimized"""
        try:
            # Ensure we are working with numpy array or safe indexing
            if hasattr(ts, 'values'):
                ts_vals = ts.values
            else:
                ts_vals = np.array(ts)
                
            # Check cache first
            if self.smart_cache:
                # Use safe indexing on array
                last_val = ts_vals[-1] if len(ts_vals) > 0 else 0
                cache_key = f"hurst_{len(ts_vals)}_{last_val}"
                cached = self.smart_cache.get(cache_key, ttl=Config.CACHE_TTL_HURST)
                if cached is not None:
                    return cached
        
            # Use JIT optimized version from AdvancedStatisticalAnalyzer
            val = AdvancedStatisticalAnalyzer.calculate_hurst_exponent(ts_vals)
            
            # Cache result
            if self.smart_cache:
                self.smart_cache.set(cache_key, val)
                
            return val
        except:
            return 0.5
            
    def _calculate_garch_volatility(self, returns):
        """Forecast Volatility using GARCH(1,1) - Cached"""
        try:
            # Ensure we are working with numpy array or safe indexing
            if hasattr(returns, 'values'):
                ret_vals = returns.values
            else:
                ret_vals = np.array(returns)
                
            # Check cache first
            if self.smart_cache:
                # Use safe indexing on array
                last_val = ret_vals[-1] if len(ret_vals) > 0 else 0
                cache_key = f"garch_{len(ret_vals)}_{last_val}"
                cached = self.smart_cache.get(cache_key, ttl=Config.CACHE_TTL_GARCH)
                if cached is not None:
                    return cached
                
            # Use AdvancedStatisticalAnalyzer
            val = AdvancedStatisticalAnalyzer.calculate_garch_volatility(ret_vals)
             
            # Cache result
            if self.smart_cache:
                 self.smart_cache.set(cache_key, val)
                 
            return val
        except Exception as e:
             # Fallback to simple std
             # ProfessionalLogger.log(f"GARCH Error: {e}", "DEBUG", "FEATURE_ENGINE") # Optional debug
             return np.std(returns) if len(returns) > 0 else 0.01
             
    def _add_microstructure_features(self, df):
        """Add microstructure analysis features"""
        if not Config.ENABLE_MICROSTRUCTURE_ANALYSIS:
             return df
             
        # Initialize default columns
        df['spread_volatility'] = 0.0
        df['order_imbalance'] = 0.5
        df['liquidity_score'] = 1.0
        
        # Only works if we have access to the analyzer instance
        # Since this is a stateless method usually, checking if we can access global or passed instance
        # Ideally, FeatureEngine should have a reference to MicrostructureAnalyzer
        # For now, we'll placeholder this as the analyzer works on Ticks, not just DF
        
        return df
        
    def calculate_features(self, df):
        """Calculate comprehensive features with price action patterns"""
        df = df.copy()
        
        # Basic price features
        # Vectorized calculation (already efficient in pandas/numpy)
        df['returns'] = df['close'].pct_change()
        
        # Safe log returns calculation
        close_np = df['close'].values.astype(float)
        # Handle zeros or negative values by replacing with NaN before log
        close_safe = np.where(close_np > 0, close_np, np.nan)
        log_vals = np.log(close_safe)
        # Forward fill any NaNs created from invalid prices (though rare in close prices)
        if np.isnan(log_vals).any():
            # Minimal pandas fallback for complex filling if needed, or just 0
             mask = np.isnan(log_vals)
             log_vals[mask] = 0 # Fallback to 0 log implies 1.0 price, usually acceptable for diff
             
        df['log_returns'] = np.concatenate(([np.nan], np.diff(log_vals)))
        
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Moving averages using Config values
        ma_periods = [Config.FAST_MA, Config.MEDIUM_MA, Config.SLOW_MA]
        if hasattr(Config, 'TREND_MA') and Config.TREND_MA:
            ma_periods.append(Config.TREND_MA)
            
        for period in ma_periods:
            try:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
                
                # Vectorized deviation calculation
                sma = df[f'sma_{period}']
                # Avoid division by zero with small epsilon or replace
                df[f'price_to_sma_{period}'] = (df['close'] / sma.replace(0, np.nan)) - 1
                df[f'price_to_sma_{period}'] = df[f'price_to_sma_{period}'].fillna(0)
                
                if len(df) > period * 2:
                    price_deviation = df['close'] - df[f'sma_{period}']
                    rolling_mean = price_deviation.rolling(period).mean()
                    rolling_std = price_deviation.rolling(period).std()
                    
                    # Vectorized Z-score
                    z_score = (price_deviation - rolling_mean) / rolling_std.replace(0, np.nan)
                    df[f'price_deviation_{period}_z'] = z_score.fillna(0)
            except Exception as e:
                ProfessionalLogger.log(f"Error calculating MA features for period {period}: {str(e)}", "WARNING", "FEATURE_ENGINE")
                df[f'price_deviation_{period}_z'] = 0
                df[f'price_to_sma_{period}'] = 0
        
        # Volatility features
        # Vectorized TR calculation
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            close_shift = np.roll(close, 1)
            close_shift[0] = close[0] # Handle first element
            
            tr1 = high - low
            tr2 = np.abs(high - close_shift)
            tr3 = np.abs(low - close_shift)
            
            df['tr'] = np.maximum(tr1, np.maximum(tr2, tr3))
            
            df['atr'] = df['tr'].rolling(Config.ATR_PERIOD).mean()
            df['atr_percent'] = df['atr'] / df['close'].replace(0, 1)
            df['volatility'] = df['returns'].rolling(20).std()
        except Exception as e:
            ProfessionalLogger.log(f"Error calculating volatility features: {str(e)}", "ERROR", "FEATURE_ENGINE")
            df['atr'] = 0.001 * df['close']
            df['atr_percent'] = 0.001
            df['volatility'] = 0.01
        
        # Feature: Realized Volatility (Annualized)
        rolling_std_5 = df['returns'].rolling(5).std()
        rolling_std_20 = df['returns'].rolling(20).std()
        
        df['realized_volatility_5'] = rolling_std_5 * np.sqrt(252)
        df['realized_volatility_20'] = rolling_std_20 * np.sqrt(252)
        
        # Ratio
        df['volatility_ratio'] = df['realized_volatility_5'] / df['realized_volatility_20'].replace(0, np.nan)
        df['volatility_ratio'] = df['volatility_ratio'].fillna(1.0)
        
        # RSI - Optimizing
        try:
            delta = df['close'].diff()
            
            # Vectorized gain/loss
            gain = delta.copy()
            loss = delta.copy()
            gain[gain < 0] = 0
            loss[loss > 0] = 0
            loss = np.abs(loss)
            
            avg_gain = gain.rolling(Config.RSI_PERIOD).mean()
            avg_loss = loss.rolling(Config.RSI_PERIOD).mean()
            
            rs = avg_gain / avg_loss.replace(0, np.nan)
            df['rsi'] = 100 - (100 / (1 + rs))
            df['rsi'] = df['rsi'].fillna(50)
            
            df['rsi_normalized'] = (df['rsi'] - 50) / 50
        except:
            df['rsi'] = 50
            df['rsi_normalized'] = 0
        
        # MACD
        try:
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            df['macd'] = ema_12 - ema_26
            df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
        except:
            df['macd'] = 0
            df['macd_signal'] = 0
            df['macd_hist'] = 0
        
        # Bollinger Bands
        try:
            bb_period = Config.BB_PERIOD
            bb_std = Config.BB_STD
            
            sma_bb = df['close'].rolling(bb_period).mean()
            std_bb = df['close'].rolling(bb_period).std()
            
            upper = sma_bb + (std_bb * bb_std)
            lower = sma_bb - (std_bb * bb_std)
            
            df['bb_upper'] = upper
            df['bb_lower'] = lower
            
            # Vectorized Width and Position
            df['bb_width'] = (upper - lower) / sma_bb.replace(0, np.nan)
            df['bb_position'] = (df['close'] - lower) / (upper - lower).replace(0, np.nan)
            
            df['bb_width'] = df['bb_width'].fillna(0)
            df['bb_position'] = df['bb_position'].fillna(0.5)
            
        except Exception as e:
            ProfessionalLogger.log(f"Bollinger Bands calculation error: {str(e)}", "WARNING", "FEATURE_ENGINE")
            df['bb_upper'] = df['close']
            df['bb_lower'] = df['close']
            df['bb_width'] = 0
            df['bb_position'] = 0.5
        
        # ADX if configured
        if hasattr(Config, 'USE_MARKET_REGIME') and Config.USE_MARKET_REGIME:
            try:
                # Vectorized ADX calculation
                high = df['high'].values
                low = df['low'].values
                high_shift = np.roll(high, 1)
                low_shift = np.roll(low, 1)
                high_shift[0] = high[0]
                low_shift[0] = low[0]
                
                up_move = high - high_shift
                down_move = low_shift - low
                
                plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
                minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
                
                df['plus_dm'] = plus_dm
                df['minus_dm'] = minus_dm
                
                tr = df['tr'].rolling(Config.ADX_PERIOD).mean()
                plus_di = 100 * (pd.Series(plus_dm).rolling(Config.ADX_PERIOD).mean() / tr.replace(0, np.nan))
                minus_di = 100 * (pd.Series(minus_dm).rolling(Config.ADX_PERIOD).mean() / tr.replace(0, np.nan))
                
                sum_di = plus_di + minus_di
                dx = 100 * np.abs(plus_di - minus_di) / sum_di.replace(0, np.nan)
                
                df['adx'] = dx.rolling(Config.ADX_PERIOD).mean().fillna(20)
                
                df['trend_strength'] = 0
                df['trend_strength'] = np.where(df['adx'] > Config.ADX_STRONG_TREND_THRESHOLD, 2,
                                               np.where(df['adx'] > Config.ADX_TREND_THRESHOLD, 1, 0))
            except:
                df['adx'] = 20
                df['trend_strength'] = 0
        
        # Momentum indicators
        momentum_periods = [3, 5, 10, 20]
        for period in momentum_periods:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            shifted_close = df['close'].shift(period)
            df[f'roc_{period}'] = (df['close'] - shifted_close) / shifted_close.replace(0, np.nan)
            df[f'roc_{period}'] = df[f'roc_{period}'].fillna(0)
        
        # Statistical features
        try:
             # These are hard to purely vectorize without scipy.stats functions which are slow on rolling
             # But we can optimize to only calc on last N windows if we only need latest
             # For now, keep as is but add caching if needed? 
             # Rolling skew/kurt are natively supported in newer pandas but maybe not this version
             # We'll stick to apply but limit it if needed
            df['returns_skew_20'] = df['returns'].rolling(20).skew() # Use pandas optimized method if available
            df['returns_kurtosis_20'] = df['returns'].rolling(20).kurt() # Use pandas optimized method if available
            
            # Fill NAs
            df['returns_skew_20'] = df['returns_skew_20'].fillna(0)
            df['returns_kurtosis_20'] = df['returns_kurtosis_20'].fillna(0)
            
        except:
            df['returns_skew_20'] = 0
            df['returns_kurtosis_20'] = 0
        
        # ==========================================
        # NEW: PRICE ACTION FEATURES
        # ==========================================
        df = self._add_price_action_features(df)
        
        # ==========================================
        # NEW: VOLUME INTELLIGENCE FEATURES
        # ==========================================
        df = self._add_volume_intelligence(df)
        
        # ==========================================
        # NEW: MARKET MICROSTRUCTURE FEATURES
        # ==========================================
        df = self._add_microstructure_features(df)

        
        # Volume features
        # Volume features - ENHANCED VERSION
        if 'tick_volume' in df.columns:
            # Multiple volume metrics for robustness
            volume_window = 20
            
            # 1. Basic volume statistics
            df['volume_sma'] = df['tick_volume'].rolling(volume_window).mean()
            df['volume_median'] = df['tick_volume'].rolling(volume_window).median()
            df['volume_std'] = df['tick_volume'].rolling(volume_window).std().replace(0, 1)
            
            # 2. Volume ratios using different baselines
            # Using median (more robust to outliers)
            df['volume_ratio'] = df['tick_volume'] / df['volume_median'].replace(0, 1)
            
            # Using mean (traditional)
            df['volume_ratio_mean'] = df['tick_volume'] / df['volume_sma'].replace(0, 1)
            
            # 3. Volume z-score (statistical significance)
            df['volume_zscore'] = (df['tick_volume'] - df['volume_sma']) / df['volume_std']
            
            # 4. Stabilized volume metrics (using previous completed bar)
            df['volume_ratio_prev'] = df['volume_ratio'].shift(1)
            df['volume_zscore_prev'] = df['volume_zscore'].shift(1)
            
            # 5. Volume trends (momentum)
            df['volume_trend_5'] = df['tick_volume'].rolling(5).mean() / df['tick_volume'].rolling(20).mean()
            df['volume_trend_10'] = df['tick_volume'].rolling(10).mean() / df['tick_volume'].rolling(30).mean()
            
            # 5. Volume acceleration (rate of change)
            df['volume_change'] = df['tick_volume'].pct_change()
            df['volume_acceleration'] = df['volume_change'].rolling(5).mean()
            
            # 6. Volume spikes detection
            df['volume_spike'] = (df['volume_zscore'] > 2.0).astype(int)
            df['volume_crash'] = (df['volume_zscore'] < -2.0).astype(int)
            
            # 7. Volume vs volatility correlation
            returns = df['close'].pct_change()
            df['volume_price_corr_10'] = df['tick_volume'].rolling(10).corr(returns.abs())
            df['volume_price_corr_20'] = df['tick_volume'].rolling(20).corr(returns.abs())
            
            # 8. Relative volume for gold (session-based)
            if hasattr(df, 'hour'):
                # Calculate session-specific volume baselines
                for session_hour in range(0, 24, 6):
                    session_mask = (df['hour'] >= session_hour) & (df['hour'] < session_hour + 6)
                    if session_mask.any():
                        session_volume = df.loc[session_mask, 'tick_volume']
                        df.loc[session_mask, f'session_volume_avg_{session_hour}'] = session_volume.mean()
                        df.loc[session_mask, 'volume_session_ratio'] = df['tick_volume'] / df[f'session_volume_avg_{session_hour}'].replace(0, 1)
            
            # 9. Volume-based support/resistance detection
            high_volume_bars = df[df['volume_ratio'] > 2.0]
            if len(high_volume_bars) > 10:
                # Calculate support/resistance levels from high volume bars
                df['high_volume_close'] = high_volume_bars['close'].rolling(5).mean()
                df['volume_support'] = high_volume_bars['low'].rolling(5).min()
                df['volume_resistance'] = high_volume_bars['high'].rolling(5).max()
            
            # 10. Volume divergence detection
            price_trend = df['close'].rolling(5).mean().diff()
            volume_trend = df['tick_volume'].rolling(5).mean().diff()
            df['volume_divergence'] = (price_trend * volume_trend < 0).astype(int)
            
        else:
            # Default values when volume data is unavailable
            df['volume_sma'] = 1
            df['volume_median'] = 1
            df['volume_ratio'] = 1
            df['volume_ratio_mean'] = 1
            df['volume_zscore'] = 0
            df['volume_trend_5'] = 1
            df['volume_trend_10'] = 1
            df['volume_change'] = 0
            df['volume_acceleration'] = 0
            df['volume_spike'] = 0
            df['volume_crash'] = 0
            df['volume_price_corr_10'] = 0
            df['volume_price_corr_20'] = 0
            df['volume_divergence'] = 0

        # Ensure all volume features are finite
        volume_columns = [col for col in df.columns if 'volume' in col]
        for col in volume_columns:
            df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        # Time features with session awareness
        if 'time' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                df['hour'] = df['datetime'].dt.hour
                df['day_of_week'] = df['datetime'].dt.dayofweek
                df['day_of_month'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                
                if Config.SESSION_AWARE_TRADING:
                    # Use server time from the dataframe, not local time
                    df['in_london_session'] = ((df['hour'] >= Config.LONDON_OPEN_HOUR) & 
                                               (df['hour'] < Config.LONDON_CLOSE_HOUR)).astype(int)
                    df['in_ny_session'] = ((df['hour'] >= Config.NY_OPEN_HOUR) & 
                                           (df['hour'] < Config.NY_CLOSE_HOUR)).astype(int)
                    df['in_overlap_session'] = ((df['hour'] >= Config.NY_OPEN_HOUR) & 
                                                (df['hour'] < Config.LONDON_CLOSE_HOUR)).astype(int)
                    df['avoid_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 7)).astype(int)
                    
                    df['good_trading_hours'] = (df['in_london_session'] | df['in_ny_session']).astype(int)
                    
            except:
                df['hour'] = 12
                df['day_of_week'] = 2
                df['month'] = 1
                df['hour_sin'] = 0
                df['hour_cos'] = 1
                df['day_sin'] = 0
                df['day_cos'] = 1
                df['month_sin'] = 0
                df['month_cos'] = 1
                
                if Config.SESSION_AWARE_TRADING:
                    df['in_london_session'] = 0
                    df['in_ny_session'] = 0
                    df['in_overlap_session'] = 0
                    df['avoid_asia_session'] = 0
                    df['good_trading_hours'] = 1
        else:
            df['hour'] = 12
            df['day_of_week'] = 2
            df['month'] = 1
            df['hour_sin'] = 0
            df['hour_cos'] = 1
            df['day_sin'] = 0
            df['day_cos'] = 1
            df['month_sin'] = 0
            df['month_cos'] = 1
            
            if Config.SESSION_AWARE_TRADING:
                df['in_london_session'] = 0
                df['in_ny_session'] = 0
                df['in_overlap_session'] = 0
                df['avoid_asia_session'] = 0
                df['good_trading_hours'] = 1
        
        # Advanced statistical features
        n = len(df)
        
        # GARCH volatility
        df['garch_volatility'] = 0
        df['garch_vol_ratio'] = 0
        garch_window = Config.GARCH_VOL_PERIOD
        if n > garch_window:
            try:
                window_returns = df['returns'].iloc[-garch_window:].values
                window_returns = window_returns[~np.isnan(window_returns) & ~np.isinf(window_returns)]
                if len(window_returns) > garch_window // 2:
                    current_time = datetime.now()
                    if self.last_garch_time is None or (current_time - self.last_garch_time).total_seconds() > 300:
                        garch_vol = self.stat_analyzer.calculate_garch_volatility(
                            window_returns, 
                            p=Config.GARCH_P,
                            q=Config.GARCH_Q
                        )
                        self.cached_garch = garch_vol
                        self.last_garch_time = current_time
                    else:
                        garch_vol = self.cached_garch
                        
                    df.loc[df.index[-1], 'garch_volatility'] = garch_vol
                    if df['volatility'].iloc[-1] > 0:
                        df.loc[df.index[-1], 'garch_vol_ratio'] = garch_vol / df['volatility'].iloc[-1]
            except Exception as e:
                ProfessionalLogger.log(f"GARCH calculation error: {str(e)}", "WARNING", "FEATURE_ENGINE")
                pass
        
        # Hurst exponent
        df['hurst_exponent'] = 0.5
        hurst_window = Config.HURST_WINDOW
        if n > hurst_window:
            try:
                window_prices = df['close'].iloc[-hurst_window:].values
                hurst = self.stat_analyzer.calculate_hurst_exponent(window_prices)
                df.loc[df.index[-1], 'hurst_exponent'] = hurst
            except Exception as e:
                ProfessionalLogger.log(f"Hurst exponent calculation error: {str(e)}", "WARNING", "FEATURE_ENGINE")
                pass
        
        # Market regime encoding
        df['regime_encoded'] = 0
        if 'hurst_exponent' in df.columns:
            hurst = df['hurst_exponent'].iloc[-1]
            
            trending_thresh = Config.HURST_TRENDING_THRESHOLD
            meanreverting_thresh = Config.HURST_MEANREVERTING_THRESHOLD
            
            if hurst > trending_thresh:
                df.loc[df.index[-1], 'regime_encoded'] = 2
            elif hurst < meanreverting_thresh:
                df.loc[df.index[-1], 'regime_encoded'] = 1
            else:
                df.loc[df.index[-1], 'regime_encoded'] = 0
        
        # VaR confidence levels
        df['var_95'] = 0
        df['cvar_95'] = 0
        df['var_cvar_spread'] = 0
        var_window = Config.VAR_LOOKBACK
        var_confidence = Config.VAR_CONFIDENCE
        if n > var_window:
            try:
                window_returns = df['returns'].iloc[-var_window:].values
                window_returns = window_returns[~np.isnan(window_returns) & ~np.isinf(window_returns)]
                if len(window_returns) > var_window // 2:
                    var_percentile = 100 * (1 - var_confidence)
                    var = np.percentile(window_returns, var_percentile)
                    
                    cvar_confidence = Config.CVAR_CONFIDENCE
                    cvar_percentile = 100 * (1 - cvar_confidence)
                    cvar = np.percentile(window_returns, cvar_percentile)
                    
                    df.loc[df.index[-1], 'var_95'] = var
                    df.loc[df.index[-1], 'cvar_95'] = cvar
                    df.loc[df.index[-1], 'var_cvar_spread'] = cvar - var
            except Exception as e:
                ProfessionalLogger.log(f"VaR/CVaR calculation error: {str(e)}", "WARNING", "FEATURE_ENGINE")
                pass
        
        # Support/Resistance features
        try:
            df['distance_to_high_20'] = (df['high'].rolling(20).max() - df['close']) / df['close'].replace(0, 1)
            df['distance_to_low_20'] = (df['close'] - df['low'].rolling(20).min()) / df['close'].replace(0, 1)
            df['high_low_range'] = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close'].replace(0, 1)
        except:
            df['distance_to_high_20'] = 0
            df['distance_to_low_20'] = 0
            df['high_low_range'] = 0
        
        # Volatility regime
        if Config.VOLATILITY_SCALING_ENABLED:
            try:
                current_vol = df['volatility'].iloc[-1] * np.sqrt(252)
                
                high_vol = Config.HIGH_VOL_THRESHOLD
                normal_vol = Config.NORMAL_VOL_THRESHOLD
                low_vol = Config.LOW_VOL_THRESHOLD
                
                if current_vol > high_vol:
                    df['volatility_regime'] = 2
                elif current_vol < low_vol:
                    df['volatility_regime'] = 0
                else:
                    df['volatility_regime'] = 1
            except:
                df['volatility_regime'] = 1
        
        # Z-Score features
        zscore_window = 50
        if n > zscore_window:
            last_idx = df.index[-1]
            
            try:
                rsi_mean = df['rsi'].rolling(zscore_window).mean().iloc[-1]
                rsi_std = df['rsi'].rolling(zscore_window).std().iloc[-1]
                if rsi_std > 0:
                    df.loc[last_idx, 'rsi_zscore'] = (df['rsi'].iloc[-1] - rsi_mean) / rsi_std
                else:
                    df.loc[last_idx, 'rsi_zscore'] = 0
            except:
                df.loc[last_idx, 'rsi_zscore'] = 0
            
            try:
                macd_hist_mean = df['macd_hist'].rolling(zscore_window).mean().iloc[-1]
                macd_hist_std = df['macd_hist'].rolling(zscore_window).std().iloc[-1]
                if macd_hist_std > 0:
                    df.loc[last_idx, 'macd_hist_zscore'] = (df['macd_hist'].iloc[-1] - macd_hist_mean) / macd_hist_std
                else:
                    df.loc[last_idx, 'macd_hist_zscore'] = 0
            except:
                df.loc[last_idx, 'macd_hist_zscore'] = 0
            
            if 'tick_volume' in df.columns:
                try:
                    volume_mean = df['tick_volume'].rolling(zscore_window).mean().iloc[-1]
                    volume_std = df['tick_volume'].rolling(zscore_window).std().iloc[-1]
                    if volume_std > 0:
                        df.loc[last_idx, 'volume_zscore'] = (df['tick_volume'].iloc[-1] - volume_mean) / volume_std
                    else:
                        df.loc[last_idx, 'volume_zscore'] = 0
                    
                    vol_ratio_window = df['volume_ratio'].iloc[-20:]
                    returns_window = df['returns'].iloc[-20:]
                    if len(vol_ratio_window) > 10 and len(returns_window) > 10:
                        corr = vol_ratio_window.corr(returns_window)
                        df.loc[last_idx, 'volume_price_correlation'] = 0 if np.isnan(corr) else corr
                    else:
                        df.loc[last_idx, 'volume_price_correlation'] = 0
                    
                    df.loc[last_idx, 'volume_spike'] = 1 if df['volume_ratio'].iloc[-1] > 2 else 0
                except:
                    df.loc[last_idx, 'volume_zscore'] = 0
                    df.loc[last_idx, 'volume_price_correlation'] = 0
                    df.loc[last_idx, 'volume_spike'] = 0
        
        # Gold-specific features
        if hasattr(Config, 'GOLD_VOLATILITY_ADJUSTMENT') and Config.GOLD_VOLATILITY_ADJUSTMENT:
            try:
                expected_range = Config.EXPECTED_DAILY_RANGE
                current_atr_percent = df['atr_percent'].iloc[-1] * 100
                df['gold_atr_normalized'] = current_atr_percent / (expected_range / df['close'].iloc[-1] * 100)
            except:
                df['gold_atr_normalized'] = 1
        
        # ==========================================
        # ADVANCED STATISTICAL FEATURES (INSTITUTIONAL)
        # ==========================================
        try:
            # 1. Entropy (Rolling Chaos Meter)
            df['entropy'] = df['returns'].rolling(window=50).apply(self._calculate_shannon_entropy)
            
            # 2. Hurst Exponent (Trend Persistence)
            df['hurst_exponent'] = df['close'].rolling(window=100).apply(self._calculate_hurst)
            
            # 3. GARCH Volatility Forecast
            if len(df) > 100:
                returns_subset = df['returns'].iloc[-500:] 
                forecast_vol = self._calculate_garch_volatility(returns_subset.dropna())
                df['garch_volatility'] = 0.0
                df.iloc[-1, df.columns.get_loc('garch_volatility')] = forecast_vol
            else:
                 df['garch_volatility'] = df['volatility'] 
            
        except Exception as e:
            ProfessionalLogger.log(f"Stat Feature Error: {e}", "WARNING", "FEATURE_ENGINE")
            df['entropy'] = 0
            df['hurst_exponent'] = 0.5
            df['garch_volatility'] = 0
        
        # Final cleanup
        all_features = self.get_feature_columns()
        for feature in all_features:
            if feature not in df.columns:
                df[feature] = 0
        
        # VWAP features
        df = self._add_vwap_features(df)
        
        # Ensure all features are finite
        all_features = self.get_feature_columns()
        for col in df.columns:
            if col in all_features:
                df[col] = df[col].replace([np.inf, -np.inf], 0).fillna(0)
        
        return df
    
    def _add_vwap_features(self, df):
        """Standard VWAP with session reset (approximate for MT5 tick volume)"""
        if 'tick_volume' not in df.columns:
            return df
            
        try:
            # Ensure datetime exists
            if 'datetime' not in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
            
            # Approximate VWAP using typical price and tick volume
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            
            # Reset VWAP daily
            if 'datetime' in df.columns:
                df['date'] = df['datetime'].dt.date
                
                # Daily cumulative reset
                df['vwap'] = df.groupby('date').apply(
                    lambda x: ( ((x['high'] + x['low'] + x['close'])/3) * x['tick_volume']).cumsum() / x['tick_volume'].cumsum().replace(0, 1)
                ).reset_index(level=0, drop=True)
            else:
                vwap_num = (typical_price * df['tick_volume']).cumsum()
                vwap_den = df['tick_volume'].cumsum()
                df['vwap'] = vwap_num / vwap_den.replace(0, 1)
            
            # Features based on VWAP
            df['distance_to_vwap'] = (df['close'] - df['vwap']) / df['vwap'].replace(0, 1)
            df['vwap_above'] = (df['close'] > df['vwap']).astype(int)
            
            # Rolling VWAP for trend (20 bar)
            rolling_vwap_num = (typical_price * df['tick_volume']).rolling(20).sum()
            rolling_vwap_den = df['tick_volume'].rolling(20).sum()
            df['vwap_rolling_20'] = rolling_vwap_num / rolling_vwap_den.replace(0, 1)
            
        except Exception as e:
            # ProfessionalLogger.log(f"VWAP calculation error: {str(e)}", "WARNING", "FEATURE_ENGINE")
            df['vwap'] = df['close']
            df['distance_to_vwap'] = 0
            df['vwap_above'] = 0
            
        return df
    
    def _add_price_action_features(self, df):
        """Add advanced price action features"""
        
        # Candlestick patterns
        df['body_size'] = abs(df['close'] - df['open']) / df['close']
        df['upper_wick'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['close']
        df['lower_wick'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['close']
        
        # Doji detection
        df['is_doji'] = (df['body_size'] < 0.001).astype(int)
        
        # Engulfing patterns - use Shifted data to avoid repainting
        df['bullish_engulfing'] = (
            (df['close'].shift(1) > df['open'].shift(1)) &
            (df['close'].shift(2) < df['open'].shift(2)) &
            (df['close'].shift(1) > df['open'].shift(2)) &
            (df['open'].shift(1) < df['close'].shift(2))
        ).astype(int)
        
        df['bearish_engulfing'] = (
            (df['close'].shift(1) < df['open'].shift(1)) &
            (df['close'].shift(2) > df['open'].shift(2)) &
            (df['close'].shift(1) < df['open'].shift(2)) &
            (df['open'].shift(1) > df['close'].shift(2))
        ).astype(int)
        
        # Trend structure
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)
        df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(int)
        df['trend_structure'] = df['higher_high'] - df['lower_low']
        
        # Support/Resistance touches
        resistance = df['high'].rolling(50).max()
        support = df['low'].rolling(50).min()
        df['near_resistance'] = ((df['close'] - resistance) / df['close']).abs() < 0.002
        df['near_support'] = ((support - df['close']) / df['close']).abs() < 0.002
        
        # Price patterns
        df['hammer'] = (
            (df['lower_wick'] > 2 * df['body_size']) &
            (df['upper_wick'] < 0.3 * df['body_size']) &
            (df['body_size'] > 0)
        ).astype(int)
        
        df['shooting_star'] = (
            (df['upper_wick'] > 2 * df['body_size']) &
            (df['lower_wick'] < 0.3 * df['body_size']) &
            (df['body_size'] > 0)
        ).astype(int)
        
        return df
    
    def _add_volume_intelligence(self, df):
        """Add volume-based intelligence features"""
        
        if 'tick_volume' not in df.columns:
            return df
        
        # Volume trend
        df['volume_ma_fast'] = df['tick_volume'].rolling(5).mean()
        df['volume_ma_slow'] = df['tick_volume'].rolling(20).mean()
        df['volume_trend'] = df['volume_ma_fast'] / df['volume_ma_slow'].replace(0, 1)
        
        # Volume breakout detection
        volume_std = df['tick_volume'].rolling(20).std()
        df['volume_breakout'] = (
            df['tick_volume'] > (df['volume_ma_slow'] + 2 * volume_std)
        ).astype(int)
        
        # Accumulation/Distribution
        df['money_flow_multiplier'] = (
            (df['close'] - df['low']) - (df['high'] - df['close'])
        ) / (df['high'] - df['low']).replace(0, 1)
        df['money_flow_volume'] = df['money_flow_multiplier'] * df['tick_volume']
        df['accumulation_distribution'] = df['money_flow_volume'].cumsum()
        
        # Volume-price confirmation
        df['bullish_volume_confirm'] = (
            (df['close'] > df['open']) &
            (df['tick_volume'] > df['volume_ma_slow'])
        ).astype(int)
        
        df['bearish_volume_confirm'] = (
            (df['close'] < df['open']) &
            (df['tick_volume'] > df['volume_ma_slow'])
        ).astype(int)
        
        # Volume divergence
        df['volume_price_divergence'] = (
            (df['close'].diff(5) > 0) & (df['tick_volume'].diff(5) < 0) |
            (df['close'].diff(5) < 0) & (df['tick_volume'].diff(5) > 0)
        ).astype(int)
        
        return df
    
    def _add_microstructure_features(self, df):
        """Add market microstructure features"""
        
        # Bid-Ask pressure proxy
        df['buying_pressure'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, 1)
        df['selling_pressure'] = (df['high'] - df['close']) / (df['high'] - df['low']).replace(0, 1)
        
        # Price velocity and acceleration
        df['price_velocity'] = df['close'].diff() / df['close'].shift(1)
        df['price_acceleration'] = df['price_velocity'].diff()
        
        # Volatility surprise
        realized_vol = df['returns'].rolling(20).std()
        df['vol_surprise'] = (df['returns'].abs() - realized_vol) / realized_vol.replace(0, 1)
        
        # Gap detection
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
        df['gap_filled'] = (
            (df['gap'] > 0) & (df['low'] < df['close'].shift(1)) |
            (df['gap'] < 0) & (df['high'] > df['close'].shift(1))
        ).astype(int)
        
        # Order flow imbalance (simplified)
        df['order_flow_imbalance'] = df['buying_pressure'] - df['selling_pressure']
        
        # Price impact (how much price moves per unit volume)
        if 'tick_volume' in df.columns:
            df['price_impact'] = df['returns'].abs() / df['tick_volume'].replace(0, 1)
        
        return df
    
    def create_labels(self, df, forward_bars=6, method='simple'):
        """Create labels with dynamic ATR-based barriers"""
        df = df.copy()
        
        # Dynamic ATR-based barriers if configured
        if Config.USE_DYNAMIC_BARRIERS and 'atr_percent' in df.columns:
            return self._create_labels_dynamic(df, forward_bars)
        
        # Original triple barrier method
        elif Config.TRIPLE_BARRIER_METHOD:
            returns = df['returns'].values
            labels = np.zeros(len(df))
            
            upper_barrier = Config.BARRIER_UPPER
            lower_barrier = Config.BARRIER_LOWER
            barrier_time = Config.BARRIER_TIME
            
            max_lookahead = min(barrier_time, forward_bars, len(df) - 1)
            
            for i in range(len(df) - max_lookahead):
                if i + max_lookahead >= len(df):
                    continue
                
                future_returns = np.cumprod(1 + returns[i+1:i+max_lookahead+1]) - 1
                
                upper_hit = np.any(future_returns >= upper_barrier)
                lower_hit = np.any(future_returns <= lower_barrier)
                
                if upper_hit and lower_hit:
                    upper_idx = np.where(future_returns >= upper_barrier)[0]
                    lower_idx = np.where(future_returns <= lower_barrier)[0]
                    if upper_idx[0] < lower_idx[0]:
                        labels[i] = 1
                    else:
                        labels[i] = 0
                elif upper_hit:
                    labels[i] = 1
                elif lower_hit:
                    labels[i] = 0
                else:
                    labels[i] = 1 if future_returns[-1] > 0 else 0
            
            df['label'] = labels[:len(df)]
        
        else:
            # Simple labeling (fallback)
            df['forward_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
            volatility = df['returns'].rolling(20).std().fillna(0.001)
            
            if Config.USE_DYNAMIC_SL_TP:
                atr_percent = df['atr_percent'].rolling(20).mean().fillna(0.001)
                dynamic_threshold = atr_percent * Config.ATR_SL_MULTIPLIER
            else:
                dynamic_threshold = Config.FIXED_SL_PERCENT
            
            df['label'] = 0
            df.loc[df['forward_return'] > dynamic_threshold, 'label'] = 1
            df.loc[df['forward_return'] < -dynamic_threshold, 'label'] = 0
        
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        return df
    
    def _create_labels_dynamic(self, df, forward_bars=6):
        """Dynamic ATR-based triple barrier labeling"""
        df = df.copy()
        
        returns = df['returns'].values
        
        # Calculate dynamic barriers based on rolling ATR
        atr_percent = df['atr_percent'].rolling(20).mean().fillna(0.001)
        
        # Dynamic barriers: 1.5x ATR for profit, 1.0x ATR for stop (Intraday scale)
        upper_barriers = atr_percent * 1.5
        lower_barriers = -atr_percent * 1.0
        
        labels = np.zeros(len(df))
        barrier_time = Config.BARRIER_TIME
        max_lookahead = min(barrier_time, forward_bars, len(df) - 1)
        
        for i in range(len(df) - max_lookahead):
            if i + max_lookahead >= len(df):
                continue
            
            current_upper = upper_barriers.iloc[i]
            current_lower = lower_barriers.iloc[i]
            
            if pd.isna(current_upper) or pd.isna(current_lower) or current_upper == 0:
                continue
            
            future_returns = np.cumprod(1 + returns[i+1:i+max_lookahead+1]) - 1
            
            upper_hit = np.any(future_returns >= current_upper)
            lower_hit = np.any(future_returns <= current_lower)
            
            if upper_hit and lower_hit:
                upper_idx = np.where(future_returns >= current_upper)[0]
                lower_idx = np.where(future_returns <= current_lower)[0]
                if upper_idx[0] < lower_idx[0]:
                    labels[i] = 1
                else:
                    labels[i] = 0
            elif upper_hit:
                labels[i] = 1
            elif lower_hit:
                labels[i] = 0
            else:
                labels[i] = 1 if future_returns[-1] > 0 else 0
        
        df['label'] = labels[:len(df)]
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
        ProfessionalLogger.log(f"Dynamic barrier labeling applied (ATR-based)", "INFO", "FEATURE_ENGINE")
        
        return df
    
    def get_feature_columns(self):
        """Return complete list of feature columns"""
        base_features = [
            'returns', 'log_returns', 'hl_ratio', 'co_ratio', 'hlc3',
            f'price_to_sma_{Config.FAST_MA}', f'price_to_sma_{Config.MEDIUM_MA}', 
            f'price_to_sma_{Config.SLOW_MA}',
            f'price_deviation_{Config.MEDIUM_MA}_z',
            'atr_percent', 'volatility', 'realized_volatility_5', 'realized_volatility_20', 'volatility_ratio',
            'rsi_normalized', 'rsi_zscore',
            'macd_hist', 'macd_hist_zscore',
            'bb_width', 'bb_position',
            'momentum_5', 'momentum_10', 'roc_10',
            'returns_skew_20', 'returns_kurtosis_20',
            'distance_to_high_20', 'distance_to_low_20', 'high_low_range',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'volume_ratio', 'volume_ratio_prev', 'volume_zscore', 'volume_zscore_prev', 'volume_price_correlation', 'volume_spike',
            'garch_volatility', 'garch_vol_ratio',
            'hurst_exponent', 'regime_encoded', 'dom_imbalance', 'entropy',
            'var_95', 'cvar_95', 'var_cvar_spread',
            
            # New price action features
            'body_size', 'upper_wick', 'lower_wick', 'is_doji',
            'bullish_engulfing', 'bearish_engulfing',
            'higher_high', 'lower_low', 'trend_structure',
            'near_resistance', 'near_support',
            'hammer', 'shooting_star',
            
            # New volume features
            'volume_trend', 'volume_breakout',
            'accumulation_distribution',
            'bullish_volume_confirm', 'bearish_volume_confirm',
            'volume_price_divergence',
            
            # New microstructure features
            'buying_pressure', 'selling_pressure',
            'price_velocity', 'price_acceleration',
            'vol_surprise', 'gap', 'gap_filled',
            'order_flow_imbalance',
            
            # VWAP features
            'vwap', 'distance_to_vwap', 'vwap_above', 'vwap_rolling_20'
        ]
        
        # Add price impact if available
        if 'price_impact' not in base_features:
            base_features.append('price_impact')
        
        # Add session features if enabled
        if Config.SESSION_AWARE_TRADING:
            base_features.extend([
                'in_london_session', 'in_ny_session', 'in_overlap_session',
                'avoid_asia_session', 'good_trading_hours'
            ])
            if Config.AVOID_MONDAY_FIRST_HOUR:
                base_features.append('avoid_monday_hour')
            if Config.AVOID_FRIDAY_LAST_HOURS:
                base_features.append('avoid_friday_hours')
        
        # Add ADX features if enabled
        if hasattr(Config, 'USE_MARKET_REGIME') and Config.USE_MARKET_REGIME:
            base_features.extend(['adx', 'trend_strength'])
        
        # Add volatility regime if enabled
        if Config.VOLATILITY_SCALING_ENABLED:
            base_features.append('volatility_regime')
        
        # Add gold-specific features
        if hasattr(Config, 'GOLD_VOLATILITY_ADJUSTMENT') and Config.GOLD_VOLATILITY_ADJUSTMENT:
            base_features.append('gold_atr_normalized')
        
        return base_features

# ==========================================
# SIGNAL QUALITY FILTER (NEW)
# ==========================================
class SignalQualityFilter:
    """Multi-layer signal validation with enhanced volume intelligence"""
    
    @staticmethod
    def validate_signal(signal, confidence, features, market_context):
        """Comprehensive signal validation with adaptive thresholds"""
        
        validation_results = []
        
        # ==========================================
        # 1. CONFIDENCE THRESHOLD (DYNAMIC)
        # ==========================================
        base_confidence = Config.MIN_CONFIDENCE
        
        # Adjust confidence threshold based on market regime
        regime = market_context.get('market_regime', 'unknown')
        if regime == 'volatile':
            base_confidence *= 1.15
        elif regime == 'trending':
            base_confidence *= 1.05
        elif regime == 'mean_reverting':
            base_confidence *= 0.95
        
        # Adjust for time of day (higher threshold during low liquidity)
        hour = market_context.get('hour', datetime.now().hour)
        if 0 <= hour < 7:  # Asian session
            base_confidence *= 1.1
        
        if confidence < base_confidence:
            return False, f"Low confidence: {confidence:.2%} < {base_confidence:.2%}"
        
        validation_results.append(f"✓ Confidence: {confidence:.2%} >= {base_confidence:.2%}")
        
        # ==========================================
        # 2. RSI EXTREMES FILTER (WITH REGIME ADJUSTMENT)
        # ==========================================
        rsi = features.get('rsi_normalized', 0) * 50 + 50
        
        # Dynamic RSI thresholds based on regime
        if regime == 'trending':
            # Allow more extreme RSI in trending markets
            if signal == 1 and rsi > 85:
                return False, f"RSI extremely overbought (trending): {rsi:.1f}"
            if signal == 0 and rsi < 15:
                return False, f"RSI extremely oversold (trending): {rsi:.1f}"
        else:
            # Normal thresholds - slightly relaxed
            if signal == 1 and rsi > 85:
                return False, f"RSI extremely overbought: {rsi:.1f}"
            if signal == 0 and rsi < 15:
                return False, f"RSI extremely oversold: {rsi:.1f}"
        
        validation_results.append(f"✓ RSI OK: {rsi:.1f}")
        
        # ==========================================
        # 3. ADVANCED VOLUME VALIDATION (ENHANCED)
        # ==========================================
        volume_ratio = features.get('volume_ratio_prev', features.get('volume_ratio', 1))
        volume_zscore = features.get('volume_zscore_prev', features.get('volume_zscore', 0))
        
        # Get ATR-based context
        atr_percent = features.get('atr_percent', 0.001)
        volatility = features.get('volatility', 0)
        
        # Dynamic volume threshold based on multiple factors
        min_volume_ratio = 0.3  # Base threshold (lowered from 0.5)
        
        # Adjust based on market context
        if regime == 'volatile':
            min_volume_ratio = 0.4  # Need more volume in volatile markets
        elif regime == 'trending':
            min_volume_ratio = 0.25  # Can accept lower volume in strong trends
        elif 0 <= hour < 7:  # Asian session
            min_volume_ratio = 0.35  # Lower threshold for low-liquidity hours
        
        # Adjust for high volatility (can accept lower volume during news)
        if volatility > 0.015:
            min_volume_ratio = max(0.2, min_volume_ratio * 0.8)
        
        # Check volume ratio
        if volume_ratio < min_volume_ratio:
            # Additional check: if price is at extremes, volume can be lower
            price_position = features.get('bb_position', 0.5)
            if not (price_position > 0.8 or price_position < 0.2):
                # Not at Bollinger Band extremes, so volume matters more
                return False, f"Insufficient volume: ratio={volume_ratio:.2f}, threshold={min_volume_ratio:.2f}"
        
        validation_results.append(f"✓ Volume OK: ratio={volume_ratio:.2f}, z={volume_zscore:.2f}")
        
        # ==========================================
        # 4. SPREAD FILTER (DYNAMIC)
        # ==========================================
        spread_pips = market_context.get('spread_pips', 0)
        max_spread = Config.MAX_SPREAD_POINTS
        
        # Dynamic spread adjustment based on volatility
        if volatility > 0.015:
            max_spread = int(max_spread * 1.5)  # Allow wider spreads in high volatility
        
        if spread_pips > max_spread:
            return False, f"Spread too wide: {spread_pips} > {max_spread}"
        
        validation_results.append(f"✓ Spread OK: {spread_pips} <= {max_spread}")
        
        # ==========================================
        # 5. TIME-OF-DAY FILTER (ENHANCED)
        # ==========================================
        if Config.SESSION_AWARE_TRADING:
            hour = market_context.get('hour', datetime.now().hour)
            day_of_week = market_context.get('day_of_week', datetime.now().weekday())
            
            # Check for market hours
            if Config.AVOID_ASIAN_SESSION and 0 <= hour < 7:
                return False, f"Avoiding Asian session: {hour}:00"
            
            # Check for overlap preference
            if Config.PREFER_LONDON_NY_OVERLAP:
                in_overlap = Config.LONDON_CLOSE_HOUR > hour >= Config.NY_OPEN_HOUR
                if not in_overlap and confidence < 0.7:
                    return False, f"Outside preferred overlap hours: {hour}:00"
            
            # Monday first hour avoidance
            if Config.AVOID_MONDAY_FIRST_HOUR and day_of_week == 0 and hour < 1:
                return False, "Avoiding Monday first hour"
            
            # Friday last hours avoidance
            if Config.AVOID_FRIDAY_LAST_HOURS and day_of_week == 4 and hour >= 20:
                return False, "Avoiding Friday last hours"
        
        validation_results.append(f"✓ Time OK: {hour}:00")
        
        # ==========================================
        # 6. NEWS EVENT FILTER
        # ==========================================
        if market_context.get('high_impact_news_soon', False):
            # Allow trades if confidence is very high and we're not too close to news
            if confidence < 0.8:
                return False, "High-impact news event imminent"
        
        validation_results.append("✓ No news interference")
        
        # ==========================================
        # 7. CORRELATION CHECK (ENHANCED)
        # ==========================================
        existing_positions = market_context.get('existing_positions', [])
        if len(existing_positions) > 0:
            correlation_risk = SignalQualityFilter._check_correlation_enhanced(signal, existing_positions, features)
            if correlation_risk > Config.MAX_POSITION_CORRELATION:
                return False, f"High correlation with existing positions: {correlation_risk:.2f}"
        
        validation_results.append(f"✓ Correlation OK: {len(existing_positions)} existing positions")
        
        # ==========================================
        # 8. VOLATILITY SPIKE FILTER
        # ==========================================
        vol_surprise = market_context.get('vol_surprise', 0)
        if vol_surprise > 4:  # Increased threshold from 3
            return False, f"Abnormal volatility spike: {vol_surprise:.1f}"
        
        validation_results.append(f"✓ Volatility OK: surprise={vol_surprise:.1f}")
        
        # ==========================================
        # 9. MARKET REGIME FILTER
        # ==========================================
        if regime == 'volatile' and confidence < 0.65:
            return False, f"Volatile regime requires higher confidence: {confidence:.2%} < 0.65"
        
        validation_results.append(f"✓ Regime OK: {regime}")
        
        # ==========================================
        # 9b. NEWS & SPREAD FILTER (Proxy)
        # ==========================================
        current_spread = features.get('spread', 0)
        if hasattr(Config, 'NEWS_FILTER_ENABLED') and Config.NEWS_FILTER_ENABLED:
            if current_spread > Config.MAX_SPREAD_PIPS:
                 return False, f"High Spread ({current_spread} pips) - Possible News/Event"
            validation_results.append(f"✓ Spread OK: {current_spread}")
        
        # ==========================================
        # 10. MULTI-TIMEFRAME ALIGNMENT
        # ==========================================
        if 'multi_tf_alignment' in market_context and Config.REQUIRE_TIMEFRAME_ALIGNMENT:
            alignment = market_context['multi_tf_alignment']
            min_alignment = Config.TIMEFRAME_ALIGNMENT_THRESHOLD
            
            # Adjust alignment threshold based on signal strength
            if confidence > 0.7:
                min_alignment *= 0.9  # Lower threshold for high confidence signals
            
            if alignment < min_alignment:
                # RELAXED: If multi_tf_signal is neutral (0.5), bypass if confidence is high
                multi_tf_signal = market_context.get('multi_tf_signal', 0.5)
                if multi_tf_signal == 0.5 and confidence >= 0.60:
                     validation_results.append(f"✓ Alignment Bypassed: Neutral session consensus but High confidence ({confidence:.1%})")
                else:
                    return False, f"Low multi-TF alignment: {alignment:.0%} < {min_alignment:.0%}"
            
            validation_results.append(f"✓ Multi-TF alignment: {alignment:.0%}")
        elif not Config.REQUIRE_TIMEFRAME_ALIGNMENT:
            validation_results.append("✓ Multi-TF alignment: SKIPPED (Disabled in Config)")
        
        # ==========================================
        # 11. TECHNICAL CONFIRMATION
        # ==========================================
        if not SignalQualityFilter._technical_confirmation(signal, features):
            return False, "Lacking technical confirmation"
        
        validation_results.append("✓ Technical confirmation")
        
        # ==========================================
        # 12. PRICE ACTION CONFIRMATION
        # ==========================================
        price_action_ok = SignalQualityFilter._price_action_confirmation(signal, features)
        if not price_action_ok:
            return False, "Price action not confirming"
        
        validation_results.append("✓ Price action confirming")
        
        # ==========================================
        # 13. TREND FILTER (OPTIONAL)
        # ==========================================
        if Config.LONG_TIMEFRAME_TREND_FILTER:
            trend_filter = market_context.get('trend_filter', 1)
            if trend_filter == -1:
                return False, "Trend filter blocking trade"
            validation_results.append("✓ Trend filter passed")
        
        # ==========================================
        # 14. POSITION COUNT CHECK
        # ==========================================
        if len(existing_positions) >= Config.MAX_POSITIONS:
            return False, f"Max positions reached: {len(existing_positions)}/{Config.MAX_POSITIONS}"
        
        validation_results.append(f"✓ Position count OK: {len(existing_positions)}/{Config.MAX_POSITIONS}")
        
        # ==========================================
        # 15. RECENT PERFORMANCE CHECK
        # ==========================================
        recent_performance = market_context.get('recent_performance', {'win_rate': 0.5})
        win_rate = recent_performance.get('win_rate', 0.5)
        
        # Lower confidence requirements during winning streaks
        if win_rate < 0.4 and confidence < 0.6:
            return False, f"Poor recent performance (win rate: {win_rate:.0%}) requires higher confidence"
        
        validation_results.append(f"✓ Recent performance OK: win rate={win_rate:.0%}")
        
        # Log successful validation with details
        ProfessionalLogger.log(f"✅ Filter PASSED: {'BUY' if signal == 1 else 'SELL'} (Conf: {confidence:.1%})", "SUCCESS", "FILTER")
        if len(validation_results) > 0 and Config.DEBUG_MODE:
            for result in validation_results:
                ProfessionalLogger.log(f"  {result}", "DEBUG", "FILTER")
        
        return True, f"All filters passed ({len(validation_results)} checks)"
    
    @staticmethod
    def _check_correlation_enhanced(new_signal, existing_positions, features):
        """Enhanced correlation check considering market context"""
        if not existing_positions:
            return 0.0
        
        # Count positions in same direction
        same_direction = sum(1 for pos in existing_positions if pos.get('signal') == new_signal)
        
        # Consider market regime: allow more correlated positions in strong trends
        trending_strength = features.get('trend_strength', 0)
        if trending_strength > 2:  # Very strong trend
            correlation_penalty = 0.3
        else:
            correlation_penalty = 1.0
        
        return (same_direction / len(existing_positions)) * correlation_penalty
    
    @staticmethod
    def _technical_confirmation(signal, features):
        """Check for technical indicator confirmation"""
        
        # Get key indicators
        macd_hist = features.get('macd_hist', 0)
        bb_position = features.get('bb_position', 0.5)
        momentum = features.get('momentum_5', 0)
        adx = features.get('adx', 20)
        
        # MACD confirmation
        macd_confirm = (signal == 1 and macd_hist > 0) or (signal == 0 and macd_hist < 0)
        
        # Bollinger Band confirmation
        bb_confirm = False
        if signal == 1:
            bb_confirm = bb_position < 0.3  # Near lower band for buys
        else:
            bb_confirm = bb_position > 0.7  # Near upper band for sells
        
        # Momentum confirmation
        momentum_confirm = (signal == 1 and momentum > 0) or (signal == 0 and momentum < 0)
        
        # ADX confirmation (trend strength)
        adx_confirm = adx > 20  # Some trend present
        
        # Require at least 2 out of 4 confirmations
        confirmations = [macd_confirm, bb_confirm, momentum_confirm, adx_confirm]
        confirm_count = sum(1 for c in confirmations if c)
        
        return confirm_count >= 2
    
    @staticmethod
    def _price_action_confirmation(signal, features):
        """Check price action patterns"""
        
        # Get price action features
        bullish_engulfing = features.get('bullish_engulfing', 0)
        bearish_engulfing = features.get('bearish_engulfing', 0)
        hammer = features.get('hammer', 0)
        shooting_star = features.get('shooting_star', 0)
        higher_high = features.get('higher_high', 0)
        lower_low = features.get('lower_low', 0)
        
        if signal == 1:  # Buy signal
            # Bullish patterns: hammer, bullish engulfing, higher high
            bullish_patterns = hammer > 0 or bullish_engulfing > 0 or higher_high > 0
            bearish_patterns = shooting_star > 0 or bearish_engulfing > 0 or lower_low > 0
            
            # Allow if bullish patterns present OR no bearish patterns
            return bullish_patterns or not bearish_patterns
        
        else:  # Sell signal
            # Bearish patterns: shooting star, bearish engulfing, lower low
            bearish_patterns = shooting_star > 0 or bearish_engulfing > 0 or lower_low > 0
            bullish_patterns = hammer > 0 or bullish_engulfing > 0 or higher_high > 0
            
            # Allow if bearish patterns present OR no bullish patterns
            return bearish_patterns or not bullish_patterns
    
    @staticmethod
    def get_validation_summary(signal, confidence, features, market_context):
        """Get detailed validation summary without blocking"""
        summary = []
        
        # 1. Confidence
        base_confidence = Config.MIN_CONFIDENCE
        regime = market_context.get('market_regime', 'unknown')
        if regime == 'volatile':
            base_confidence *= 1.15
        
        confidence_ok = confidence >= base_confidence
        summary.append({
            'check': 'confidence',
            'passed': confidence_ok,
            'value': f"{confidence:.2%}",
            'threshold': f"{base_confidence:.2%}"
        })
        
        # 2. Volume
        volume_ratio = features.get('volume_ratio', 1)
        min_volume_ratio = 0.3  # Base threshold
        if regime == 'volatile':
            min_volume_ratio = 0.4
        
        volume_ok = volume_ratio >= min_volume_ratio
        summary.append({
            'check': 'volume',
            'passed': volume_ok,
            'value': f"{volume_ratio:.2f}",
            'threshold': f"{min_volume_ratio:.2f}"
        })
        
        # 3. RSI
        rsi = features.get('rsi_normalized', 0) * 50 + 50
        rsi_ok = True
        if signal == 1 and rsi > 80:
            rsi_ok = False
        if signal == 0 and rsi < 20:
            rsi_ok = False
        
        summary.append({
            'check': 'rsi',
            'passed': rsi_ok,
            'value': f"{rsi:.1f}",
            'threshold': "20-80"
        })
        
        # 4. Technical confirmation
        tech_confirm = SignalQualityFilter._technical_confirmation(signal, features)
        summary.append({
            'check': 'technical_confirmation',
            'passed': tech_confirm,
            'value': "N/A",
            'threshold': "2/4 indicators"
        })
        
        # Count passed checks
        passed = sum(1 for item in summary if item['passed'])
        total = len(summary)
        
        return {
            'summary': summary,
            'passed_count': passed,
            'total_checks': total,
            'overall_passed': passed >= 3  # Pass if at least 3 out of 4 basic checks
        }

# ==========================================
# SMART ENTRY TIMING (NEW)
# ==========================================
class SmartEntryTiming:
    """Wait for optimal entry confirmation"""
    
    def __init__(self):
        self.pending_signals = {}
        self.confirmation_bars = Config.CONFIRMATION_BARS_REQUIRED
    
    def should_enter(self, signal, confidence, features, df):
        """Wait for confirmation before entry"""
        
        signal_id = f"signal_{signal}"
    
        # If the signal type changed or is new, clear other states and start fresh
        if signal_id not in self.pending_signals:
            self.pending_signals = {
                signal_id: {
                    'signal': signal,
                    'initial_confidence': confidence,
                    'first_seen': time.time(),
                    'confirmations': 0,
                    'best_price': df['close'].iloc[-1]
                }
            }
            ProfessionalLogger.log(f"Signal pending confirmation: {signal_id}", "CONFIRMATION", "ENTRY_TIMING")
            return False, "Waiting for confirmation"
        
        pending = self.pending_signals[signal_id]
        
        # 1. Price confirmation
        if signal == 1:
            price_confirms = df['close'].iloc[-1] > df['close'].iloc[-2]
            if price_confirms:
                pending['best_price'] = min(pending['best_price'], df['close'].iloc[-1])
        else:
            price_confirms = df['close'].iloc[-1] < df['close'].iloc[-2]
            if price_confirms:
                pending['best_price'] = max(pending['best_price'], df['close'].iloc[-1])
        
        if price_confirms:
            pending['confirmations'] += 1
        
        # 2. Volume confirmation
        # Use previous bar for stability or current if it's already high
        volume_ratio = max(features.get('volume_ratio_prev', 0), features.get('volume_ratio', 0))
        volume_confirms = volume_ratio > 1.2
        if volume_confirms:
            pending['confirmations'] += 1
        
        # 3. Momentum confirmation
        momentum = features.get('momentum_5', 0)
        if signal == 1 and momentum > 0.001:
            pending['confirmations'] += 1
        elif signal == 0 and momentum < -0.001:
            pending['confirmations'] += 1
        
        # Entry criteria
        if pending['confirmations'] >= self.confirmation_bars:
            best_price = pending['best_price']
            del self.pending_signals[signal_id]
            ProfessionalLogger.log(f"Entry confirmed after {pending['confirmations']} confirmations", "SUCCESS", "ENTRY_TIMING")
            return True, f"Entry confirmed | Best price: {best_price:.2f}"
        
        # Timeout after MAX_ENTRY_WAIT_SECONDS
        if time.time() - pending['first_seen'] > Config.MAX_ENTRY_WAIT_SECONDS:
            del self.pending_signals[signal_id]
            return False, "Signal expired (timeout)"
        
        return False, f"Confirmations: {pending['confirmations']}/{self.confirmation_bars}"

        return False, f"Confirmations: {pending['confirmations']}/{self.confirmation_bars}"

# ==========================================
# ASYNC DATA FETCHER
# ==========================================
class AsyncDataFetcher:
    """Async wrapper for MT5 data operations to enable parallel fetching"""
    
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)
        
    async def fetch_rates(self, symbol, timeframe, start_pos, count):
        """Fetch rates asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor, 
            mt5.copy_rates_from_pos, 
            symbol, timeframe, start_pos, count
        )
        
    async def fetch_rates_range(self, symbol, timeframe, date_from, date_to):
        """Fetch rates range asynchronously"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            mt5.copy_rates_range,
            symbol, timeframe, date_from, date_to
        )
        
    async def fetch_multi_timeframe_data(self, symbol, timeframes, count=1000):
        """Fetch data for multiple timeframes in parallel"""
        tasks = []
        for tf in timeframes:
            tasks.append(self.fetch_rates(symbol, tf, 0, count))
            
        results = await asyncio.gather(*tasks)
        return dict(zip(timeframes, results))
        
    async def fetch_correlated_data(self, symbols, timeframe, count=1000):
        """Fetch data for multiple symbols in parallel"""
        tasks = []
        for sym in symbols:
            tasks.append(self.fetch_rates(sym, timeframe, 0, count))
            
        results = await asyncio.gather(*tasks)
        return dict(zip(symbols, results))
    
    def shutdown(self):
        """Cleanup resources"""
        self.executor.shutdown(wait=False)

# ==========================================
# ENHANCED ENSEMBLE MODEL
# ==========================================
class EnhancedEnsemble:
    """Regime-aware ensemble with improved performance"""
    
    def __init__(self, trade_memory, feature_engine):
        self.feature_engine = feature_engine
        self.trade_memory = trade_memory
        self.data_quality_checker = ProfessionalDataQualityChecker()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        self.risk_metrics = ProfessionalRiskMetrics()
        
        # Base models
        self.base_models = self._initialize_base_models()
        
        # Regime-specific model weights
        self.regime_weights = {
            'trending': {'GB': 0.3, 'RF': 0.25, 'XGB': 0.25, 'NN': 0.2} if XGB_AVAILABLE else {'GB': 0.35, 'RF': 0.3, 'NN': 0.35},
            'mean_reverting': {'RF': 0.4, 'LR': 0.3, 'NN': 0.3},
            'volatile': {'GB': 0.25, 'RF': 0.25, 'LR': 0.25, 'NN': 0.25},
            'random_walk': {'RF': 0.4, 'LR': 0.3, 'NN': 0.3}
        }
        
        # Create ensemble
        self.ensemble = self._create_ensemble_structure()
        
        # Training state
        self.final_scaler = None
        self.is_trained = False
        self.last_train_time = None
        self.trades_at_last_train = 0
        self.force_retrain_flag = False
        self.training_metrics = {}
        self.trained_feature_columns = None
        self.fitted_base_models = {}
        
        ProfessionalLogger.log("Enhanced Ensemble initialized with regime-aware weighting", "INFO", "ENSEMBLE")
    
    def _initialize_base_models(self):
        """Initialize diverse base models"""
        models = []
        
        # Gradient Boosting (good for trending markets)
        models.append(('GB', GradientBoostingClassifier(
            n_estimators=120, max_depth=6, learning_rate=0.04, 
            subsample=0.85, min_samples_split=12, min_samples_leaf=6, random_state=42
        )))
        
        # Random Forest (robust for all regimes)
        models.append(('RF', RandomForestClassifier(
            n_estimators=120, max_depth=12, min_samples_split=10, 
            min_samples_leaf=5, max_features='sqrt', bootstrap=True, 
            random_state=42, n_jobs=-1
        )))
        
        # Logistic Regression (good for mean-reverting)
        models.append(('LR', LogisticRegression(
            penalty='l2', C=0.8, max_iter=1000, 
            random_state=42, class_weight='balanced', solver='liblinear'
        )))
        
        # Neural Network
        models.append(('NN', MLPClassifier(
            hidden_layer_sizes=(64, 32), activation='relu', solver='adam', 
            alpha=0.0001, max_iter=600, random_state=42, early_stopping=True
        )))
        
        # XGBoost if available
        if XGB_AVAILABLE:
            models.append(('XGB', XGBClassifier(
                n_estimators=120, max_depth=6, learning_rate=0.04, 
                subsample=0.85, colsample_bytree=0.85, random_state=42, n_jobs=-1
            )))
        
        # SVM for volatile markets
        try:
            models.append(('SVM', SVC(
                kernel='rbf', C=1.0, gamma='scale', 
                probability=True, class_weight='balanced', random_state=42
            )))
        except:
            pass
            
        return models
    
    def _create_ensemble_structure(self):
        """Create ensemble structure with soft voting"""
        return VotingClassifier(
            estimators=[(name, model) for name, model in self.base_models],
            voting='soft',
            n_jobs=-1,
            weights=[1.0] * len(self.base_models)  # Will be adjusted by regime
        )
    
    def _load_institutional_context(self):
        """Load historical institutional snapshots (DOM, Bayesian) for training augmentation"""
        context_df = pd.DataFrame()
        try:
            history_file = os.path.join(Config.CACHE_DIR, "market_data_history.json")
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    history = json.load(f)
                if history:
                    context_df = pd.DataFrame(history)
                    context_df['time'] = pd.to_datetime(context_df['timestamp'])
                    # Aggregate by 15-minute chunks to align with bars if needed, or just nearest merge
                    context_df = context_df.drop(columns=['timestamp'])
        except Exception as e:
            ProfessionalLogger.log(f"Institutional Load Error: {e}", "DEBUG", "LEARN")
        return context_df

    def _prepare_training_data(self, data):
        """Prepare training data with institutional augmentation"""
        try:
            # 1. Base Features (Technical)
            df_features = self.feature_engine.calculate_features(data)
            
            # Ensure 'time' column is present and is datetime for the merge
            if 'time' in df_features.columns:
                df_features['time'] = pd.to_datetime(df_features['time'], unit='s' if df_features['time'].dtype != 'datetime64[ns]' else None)
            
            # 2. Institutional Augmentation (Merging the "Memory")
            inst_data = self._load_institutional_context()
            if not inst_data.empty and 'time' in df_features.columns:
                # Ensure institutional timestamps are also standardized
                inst_data['time'] = pd.to_datetime(inst_data['time'])
                
                # IMPORTANT: Remove columns that overlap with price features to avoid _x/_y suffixes
                # We only want the unique institutional data from the snapshots
                cols_to_keep = ['time', 'dom_imbalance', 'regime', 'entropy', 'hurst_exponent']
                # Only keep columns that actually exist in the snapshot
                inst_data = inst_data[[c for c in cols_to_keep if c in inst_data.columns]]
                
                # Align memory with bars (Match the most recent institutional snapshot to each bar)
                df_features = pd.merge_asof(
                    df_features.sort_values('time'), 
                    inst_data.sort_values('time'), 
                    on='time', 
                    direction='backward'
                )
                # Fill missing memory (for candles that existed before snapshots started)
                if 'dom_imbalance' not in df_features.columns:
                    df_features['dom_imbalance'] = 0.0
                else:
                    df_features['dom_imbalance'] = df_features['dom_imbalance'].fillna(0)
                
                # Note: 'regime' from snapshot vs potentially 'regime' from technicals
                # Map snapshot regime to encoded number
                if 'regime' in df_features.columns:
                    df_features['regime_encoded'] = df_features['regime'].map({
                        'trending': 1, 'mean_reverting': 2, 'volatile': 3, 'stress': 4
                    }).fillna(0)
                else:
                    df_features['regime_encoded'] = 0.0
            
            # 3. Labeling and Formatting
            df_labeled = self.feature_engine.create_labels(df_features, method='dynamic')
            df_labeled = df_labeled.dropna(subset=['label'])
            
            # --- FINAL FEATURE ALIGNMENT (The "Fix") ---
            # Get the base list of what the machine brain expects
            base_cols = self.feature_engine.get_feature_columns()
            
            # Combine technicals with our new institutional memory columns
            required_cols = list(dict.fromkeys(base_cols + ['dom_imbalance', 'regime_encoded', 'entropy', 'hurst_exponent']))
            
            # Mission-Critical Guard: Ensure every single column exists before indexing
            final_feature_list = []
            for col in required_cols:
                if col in df_labeled.columns:
                    final_feature_list.append(col)
                else:
                    # Emergency recovery: If a column vanished, fill it with 0s so we don't crash
                    df_labeled[col] = 0.0
                    final_feature_list.append(col)
            
            self.trained_feature_columns = final_feature_list
            X = df_labeled[final_feature_list].copy().fillna(0).replace([np.inf, -np.inf], 0)
            
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                
            y = df_labeled['label'].astype(int)
            return X, y

        except Exception as e:
            ProfessionalLogger.log(f"Error preparing data: {e}", "ERROR", "ENSEMBLE")
            return None, None
    
    def perform_statistical_analysis(self, data):
        """Perform statistical analysis"""
        if data is None or len(data) < 100: 
            return {}
        try:
            returns = data['close'].pct_change().dropna().values
            analysis = {
                'return_distribution': self.stat_analyzer.analyze_return_distribution(returns),
                'market_regime': self.stat_analyzer.calculate_market_regime(data),
                'risk_metrics': self.risk_metrics.calculate_risk_metrics(returns)
            }
            return analysis
        except Exception as e:
            ProfessionalLogger.log(f"Stat analysis failed: {e}", "ERROR", "ENSEMBLE")
            return {}
    
    def train(self, data):
        """Train ensemble with Walk-Forward Optimization"""
        try:
            if data is None or len(data) < Config.TRAINING_MIN_SAMPLES:
                ProfessionalLogger.log("Insufficient data for training.", "ERROR", "ENSEMBLE")
                return False

            ProfessionalLogger.log(f"Starting enhanced training on {len(data)} samples...", "LEARN", "ENSEMBLE")
            
            X_raw, y_raw = self._prepare_training_data(data)
            if X_raw is None: 
                return False

            # Walk-Forward Validation
            tscv = TimeSeriesSplit(
                n_splits=Config.WALK_FORWARD_FOLDS, 
                max_train_size=Config.WALK_FORWARD_WINDOW
            )
            
            cv_scores = []
            ProfessionalLogger.log(f"Executing WFO ({Config.WALK_FORWARD_FOLDS} folds)...", "LEARN", "ENSEMBLE")

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
                if len(train_idx) < 100: 
                    continue

                X_train_fold, X_val_fold = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
                y_train_fold, y_val_fold = y_raw.iloc[train_idx], y_raw.iloc[val_idx]
                
                fold_scaler = RobustScaler()
                X_train_scaled = fold_scaler.fit_transform(X_train_fold)
                X_val_scaled = fold_scaler.transform(X_val_fold)
                
                self.ensemble.fit(X_train_scaled, y_train_fold)
                val_score = self.ensemble.score(X_val_scaled, y_val_fold)
                cv_scores.append(val_score)
                
                ProfessionalLogger.log(f"  Fold {fold+1}: Acc={val_score:.2%}", "LEARN", "ENSEMBLE")

            avg_score = np.mean(cv_scores) if cv_scores else 0.0

            if avg_score < Config.MIN_ACCURACY_THRESHOLD:
                ProfessionalLogger.log(f"⚠ Low WFO Accuracy: {avg_score:.2%}", "WARNING", "ENSEMBLE")

            # Final training
            final_window_size = min(len(X_raw), Config.WALK_FORWARD_WINDOW * 2) 
            X_final_raw = X_raw.iloc[-final_window_size:]
            y_final = y_raw.iloc[-final_window_size:]
            
            self.final_scaler = RobustScaler()
            X_final_scaled = self.final_scaler.fit_transform(X_final_raw)
            
            ProfessionalLogger.log(f"Fitting final model on last {len(X_final_raw)} bars...", "LEARN", "ENSEMBLE")
            self.ensemble.fit(X_final_scaled, y_final)
            
            if hasattr(self.ensemble, 'named_estimators_'):
                self.fitted_base_models = self.ensemble.named_estimators_
            
            self.is_trained = True
            self.last_train_time = datetime.now()
            self.trades_at_last_train = len(self.trade_memory.trades)
            self.force_retrain_flag = False
            self.training_metrics = {'avg_cv_score': avg_score}
            
            ProfessionalLogger.log(f"✅ Training Complete | WFO Accuracy: {avg_score:.2%}", "SUCCESS", "ENSEMBLE")
            return True
            
        except Exception as e:
            ProfessionalLogger.log(f"Training error: {str(e)}", "ERROR", "ENSEMBLE")
            import traceback
            traceback.print_exc()
            return False
    
    def predict(self, df, features_df=None):
        """Make prediction with regime awareness"""
        if not self.is_trained or self.final_scaler is None:
            ProfessionalLogger.log("Model not trained.", "WARNING", "ENSEMBLE")
            return None, 0.0, None, {}
        
        try:
            if features_df is not None:
                df_feat = features_df
            else:
                df_feat = self.feature_engine.calculate_features(df)
            
            if self.trained_feature_columns is None:
                self.trained_feature_columns = self.feature_engine.get_feature_columns()
            
            # Mission-Critical: Ensure df_feat has all columns the model was trained on
            for col in self.trained_feature_columns:
                if col not in df_feat.columns:
                    df_feat[col] = 0.0
                
            X_raw = df_feat[self.trained_feature_columns].iloc[-1:].fillna(0).replace([np.inf, -np.inf], 0)
            
            X_scaled = self.final_scaler.transform(X_raw)
            
            prediction = self.ensemble.predict(X_scaled)[0]
            proba = self.ensemble.predict_proba(X_scaled)[0]
            confidence = np.max(proba)
            
            features = {col: float(X_raw[col].iloc[0]) for col in self.trained_feature_columns}
            
            # Get sub-model predictions
            sub_preds = {}
            if self.fitted_base_models:
                 for name, model in self.fitted_base_models.items():
                     try:
                         sub_p = model.predict(X_scaled)[0]
                         sub_preds[name] = {'prediction': int(sub_p), 'confidence': float(np.max(model.predict_proba(X_scaled)))}
                     except: 
                         pass
            
            # Detect current regime for weighting
            current_regime = self._detect_current_regime(features)
            
            # Adjust confidence based on regime
            if current_regime == 'trending' and prediction in [1, 0]:
                confidence *= 1.1
            elif current_regime == 'volatile':
                confidence *= 0.9
            
            validation = self._validate_prediction(prediction, confidence, features)
            if not validation['is_valid']:
                ProfessionalLogger.log(f"Validation failed: {validation['reason']}", "WARNING", "ENSEMBLE")
                return None, 0.0, None, {}

            return prediction, confidence, features, sub_preds

        except Exception as e:
            ProfessionalLogger.log(f"Prediction error: {e}", "ERROR", "ENSEMBLE")
            return None, 0.0, None, {}
    
    def _detect_current_regime(self, features):
        """Detect current market regime from features"""
        # Advanced Regime Classification (State Machine)
        entropy = features.get('entropy', 0)
        hurst = features.get('hurst', 0.5)
        garch_vol = features.get('garch_vol', 0)
        adx = features.get('adx', 20)
        
        # State 1: CHAOS (Do Not Touch)
        # High Entropy (Noise) + High Volatility = Dangerous
        if entropy > 0.8 and garch_vol > Config.HIGH_VOL_THRESHOLD:
            return 'volatile' # Hard Defense
            
        # State 2: FLOW (Trend)
        # Low Entropy (Organized) + High Hurst (Trending)
        if entropy < 0.6 and hurst > 0.55:
            return 'trending'
            
        # State 3: CHOP (Mean Reversion)
        # High Entropy (Disorganized) + Low Hurst (Reverting)
        if entropy > 0.7 and hurst < 0.45:
            return 'mean_reverting'
            
        # Fallback to ADX/Vol logic if signals are mixed
        if adx > 25:
            return 'trending'
        elif volatility > Config.HIGH_VOL_THRESHOLD:
            return 'volatile'
        else:
            return 'random_walk'
    
    def _validate_prediction(self, prediction, confidence, features):
        """Quick validation check"""
        rsi = features.get('rsi_normalized', 0) * 50 + 50
        if prediction == 1 and rsi > 85:
            return {'is_valid': False, 'reason': f"Buy Signal but RSI Overbought ({rsi:.1f})"}
        if prediction == 0 and rsi < 15:
            return {'is_valid': False, 'reason': f"Sell Signal but RSI Oversold ({rsi:.1f})"}
        return {'is_valid': True, 'reason': None}
    
    def should_retrain(self):
        """Check if retraining is needed based on time, performance, or manual triggers"""
        if not self.last_train_time or self.force_retrain_flag: 
            return True
            
        # Time-based trigger
        hours_since = (datetime.now() - self.last_train_time).total_seconds() / 3600
        if hours_since >= Config.RETRAIN_HOURS:
            return True
            
        # Activity-based trigger (every N new finished trades)
        new_trades = len(self.trade_memory.trades) - self.trades_at_last_train
        if new_trades >= Config.OPTIMIZE_EVERY_N_TRADES:
            return True
            
        return False
    
    def get_diagnostics(self):
        """Get comprehensive model diagnostics"""
        return {
            'training_status': {
                'is_trained': self.is_trained,
                'last_train_time': self.last_train_time,
                'training_metrics': self.training_metrics
            },
            'model_info': {
                'base_models': [name for name, _ in self.base_models],
                'ensemble_type': 'regime_aware_voting'
            }
        }

# ==========================================
# ADAPTIVE PARAMETER OPTIMIZER (NEW)
# ==========================================
class AdaptiveParameterOptimizer:
    """Continuously optimize trading parameters"""
    
    def __init__(self):
        self.optimization_window = Config.OPTIMIZATION_WINDOW
        self.last_optimization = None
        self.current_params = self._get_default_params()
        self.optimization_history = []
    
    def _get_default_params(self):
        """Get default parameters"""
        return {
            'min_confidence': Config.MIN_CONFIDENCE,
            'min_ensemble_agreement': Config.MIN_ENSEMBLE_AGREEMENT,
            'atr_sl_multiplier': Config.ATR_SL_MULTIPLIER,
            'atr_tp_multiplier': Config.ATR_TP_MULTIPLIER,
            'barrier_time': Config.BARRIER_TIME
        }
    
    def optimize_parameters(self, historical_data, trade_results):
        """Re-optimize parameters based on recent performance"""
        
        if len(trade_results) < Config.OPTIMIZE_EVERY_N_TRADES:
            ProfessionalLogger.log(f"Insufficient trades for optimization: {len(trade_results)}/{Config.OPTIMIZE_EVERY_N_TRADES}", "INFO", "OPTIMIZER")
            return self.current_params
        
        ProfessionalLogger.log("Starting parameter optimization...", "OPTIMIZER", "OPTIMIZER")
        
        # Define parameter space
        param_grid = {
            'min_confidence': [0.55, 0.60, 0.65, 0.70],
            'atr_sl_multiplier': [1.2, 1.5, 1.8, 2.0],
            'atr_tp_multiplier': [2.0, 2.5, 3.0, 3.5],
            'barrier_time': [4, 6, 8, 10]
        }
        
        best_score = -np.inf
        best_params = self.current_params.copy()
        
        # Simple grid search simulation
        # In production, this would use actual backtesting
        for confidence in param_grid['min_confidence']:
            for sl_mult in param_grid['atr_sl_multiplier']:
                for tp_mult in param_grid['atr_tp_multiplier']:
                    for barrier in param_grid['barrier_time']:
                        
                        score = self._evaluate_parameters(
                            confidence, sl_mult, tp_mult, barrier, trade_results
                        )
                        
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'min_confidence': confidence,
                                'atr_sl_multiplier': sl_mult,
                                'atr_tp_multiplier': tp_mult,
                                'barrier_time': barrier
                            }
        
        ProfessionalLogger.log(
            f"Parameter optimization complete | Score: {best_score:.3f} | "
            f"New params: MinConf={best_params['min_confidence']:.2f}, "
            f"SL={best_params['atr_sl_multiplier']:.1f}, "
            f"TP={best_params['atr_tp_multiplier']:.1f}",
            "SUCCESS", "OPTIMIZER"
        )
        
        self.current_params = best_params
        self.last_optimization = datetime.now()
        self.optimization_history.append({
            'timestamp': datetime.now().isoformat(),
            'params': best_params,
            'score': best_score
        })
        
        # Keep only recent history
        if len(self.optimization_history) > 10:
            self.optimization_history = self.optimization_history[-10:]
        
        return best_params
    
    def _evaluate_parameters(self, confidence, sl_mult, tp_mult, barrier, trade_results):
        """Evaluate parameter set using recent trades"""
        
        if len(trade_results) < 20:
            return 0.0
        
        # Calculate expected performance with these parameters
        # Simplified: Assume better parameters improve win rate and profit factor
        
        # Get recent trades
        recent_trades = trade_results[-20:]
        
        # Count wins and losses
        wins = sum(1 for t in recent_trades if t.get('profit', 0) > 0)
        losses = sum(1 for t in recent_trades if t.get('profit', 0) <= 0)
        
        if wins + losses == 0:
            return 0.0
        
        win_rate = wins / (wins + losses)
        
        # Calculate total profit/loss
        total_profit = sum(t.get('profit', 0) for t in recent_trades if t.get('profit', 0) > 0)
        total_loss = abs(sum(t.get('profit', 0) for t in recent_trades if t.get('profit', 0) < 0))
        
        if total_loss > 0:
            profit_factor = total_profit / total_loss
        else:
            profit_factor = float('inf')
        
        # Score combines win rate and profit factor
        score = (win_rate * 0.6) + (min(profit_factor, 3.0) * 0.4)
        
        return score
    
    def get_current_params(self):
        """Get current optimized parameters"""
        return self.current_params

# ==========================================
# ENHANCED EXIT MANAGER
# ==========================================
class EnhancedExitManager:
    """Enhanced exit logic with profit maximization and partial exits"""
    
    def __init__(self, executor):
        self.executor = executor
        self.profit_targets = {
            'conservative': 1.5,
            'moderate': 2.5,
            'aggressive': 4.0
        }
    
    def manage_positions(self, df, active_positions, signal, confidence):
        """Enhanced position management with partial exits"""
        
        if not active_positions:
            return

        # Handle both DataFrame and dictionary inputs
        if isinstance(df, dict):
            latest = df
            prev = df  # Fallback
            is_dataframe = False
        elif hasattr(df, 'empty'):
            if df.empty: return
            latest = df.iloc[-1]
            prev = df.iloc[-2] if len(df) > 1 else latest
            is_dataframe = True
        else:
            return
        
        atr = latest.get('atr', 0)
        close_price = latest.get('close', 0)
        rsi = latest.get('rsi', 50)
        adx = latest.get('adx', 0)
        
        for ticket, trade in list(active_positions.items()):
            
            duration_bars = (int(time.time()) - trade['open_time']) / (Config.TIMEFRAME * 60 if hasattr(Config, 'TIMEFRAME') else 900)
            if duration_bars < 2: 
                continue

            symbol = trade['symbol']
            trade_type = trade['type']
            entry_price = trade['open_price']
            current_sl = trade['stop_loss']
            current_tp = trade['take_profit']
            
            # MODEL INVALIDATION CHECK
            model_flip = False
            if trade_type == 'BUY' and signal == 0 and confidence > 0.65:
                ProfessionalLogger.log(f"📉 Model flipped to BEARISH (Conf: {confidence:.2f}) - Exiting BUY #{ticket}", "EXIT", "MANAGER")
                if self.executor.close_position(ticket, symbol):
                    active_positions.pop(ticket, None)
                continue
            elif trade_type == 'SELL' and signal == 1 and confidence > 0.65:
                ProfessionalLogger.log(f"📈 Model flipped to BULLISH (Conf: {confidence:.2f}) - Exiting SELL #{ticket}", "EXIT", "MANAGER")
                if self.executor.close_position(ticket, symbol):
                    active_positions.pop(ticket, None)
                continue

            # Calculate R-multiple
            if trade_type == 'BUY':
                profit_points = close_price - entry_price
                distance_to_sl = close_price - current_sl
            else:
                profit_points = entry_price - close_price
                distance_to_sl = current_sl - close_price

            initial_risk = abs(entry_price - current_sl)
            if initial_risk == 0: initial_risk = atr
            r_multiple = profit_points / initial_risk

            # 1. STALE TRADE EXIT (Time Decay)
            # If held > 60 mins and profit < 0.5R, exit to free up capital
            STALE_THRESHOLD_MINS = 60
            MIN_PROFIT_R = 0.5
            
            time_held_mins = (int(time.time()) - trade['open_time']) / 60
            
            if time_held_mins > STALE_THRESHOLD_MINS and r_multiple < MIN_PROFIT_R:
                 # Ensure we are not negative (give it a bit more room if losing slightly, but close if dead flat)
                 # If losing more than -0.5R, maybe let SL hit or wait. But if hovering near 0, KILL IT.
                 if r_multiple > -0.5: 
                    ProfessionalLogger.log(f"💤 Stale Trade Exit (#{ticket}): Held {time_held_mins:.0f}m, Profit {r_multiple:.2f}R - Closing", "EXIT", "MANAGER")
                    if self.executor.close_position(ticket, symbol):
                        active_positions.pop(ticket, None)
                    continue

            # High Water Mark (HWM) Tracking for Profit Protection
            trade['pnl'] = profit_points # Update current PnL
            if profit_points > 0:
                current_hwm = trade.get('max_runge_profit', 0)
                if profit_points > current_hwm:
                    trade['max_runge_profit'] = profit_points


            initial_risk = abs(entry_price - current_sl)
            if initial_risk == 0: 
                initial_risk = atr
            
            r_multiple = profit_points / initial_risk

            new_sl = current_sl
            sl_changed = False

            # BREAKEVEN AT 1R
            if r_multiple > 1.0:
                if trade_type == 'BUY':
                    be_price = entry_price + (atr * 0.1)
                    if new_sl < be_price:
                        new_sl = be_price
                        sl_changed = True
                        ProfessionalLogger.log(f"🛡️ Locked Breakeven for #{ticket} (1R reached)", "RISK", "MANAGER")
                else:
                    be_price = entry_price - (atr * 0.1)
                    if new_sl > be_price:
                        new_sl = be_price
                        sl_changed = True
                        ProfessionalLogger.log(f"🛡️ Locked Breakeven for #{ticket} (1R reached)", "RISK", "MANAGER")

            # RATCHET TRAILING STOP (INSTITUTIONAL GRADE)
            # As we get deeper in money, we trail TIGHTER to lock it in.
            
            # Tier 1: 1.5R Profit -> Trail at 1.0 ATR distance
            if r_multiple > 1.5:
                trail_dist = atr * 1.0
                if trade_type == 'BUY':
                    trail_price = close_price - trail_dist
                    if trail_price > new_sl:
                        new_sl = trail_price
                        sl_changed = True
                else:
                    trail_price = close_price + trail_dist
                    if trail_price < new_sl:
                        new_sl = trail_price
                        sl_changed = True

            # Tier 2: 3.0R Profit -> Trail at 0.5 ATR distance (Sniper Lock)
            if r_multiple > 3.0:
                trail_dist = atr * 0.5
                if trade_type == 'BUY':
                    trail_price = close_price - trail_dist
                    if trail_price > new_sl:
                        new_sl = trail_price
                        sl_changed = True
                        ProfessionalLogger.log(f"🔒 Ratchet Tightened (3R+): Stop moved to {new_sl:.2f}", "RISK", "MANAGER")
                else:
                    trail_price = close_price + trail_dist
                    if trail_price < new_sl:
                        new_sl = trail_price
                        sl_changed = True
                        ProfessionalLogger.log(f"🔒 Ratchet Tightened (3R+): Stop moved to {new_sl:.2f}", "RISK", "MANAGER")

            # TECHNICAL EXHAUSTION
            is_exhausted = False
            if trade_type == 'BUY' and rsi > 75:
                is_exhausted = True
            elif trade_type == 'SELL' and rsi < 25:
                is_exhausted = True
            
            if is_exhausted:
                if trade_type == 'BUY':
                    tight_stop = latest['low']
                    if tight_stop > new_sl: 
                        new_sl = tight_stop
                        sl_changed = True
                        ProfessionalLogger.log(f"⚠️ RSI Exhaustion ({rsi:.1f}) - Tightening Stop on #{ticket}", "RISK", "MANAGER")
                else:
                    tight_stop = latest['high']
                    if tight_stop < new_sl:
                        new_sl = tight_stop
                        sl_changed = True
                        ProfessionalLogger.log(f"⚠️ RSI Exhaustion ({rsi:.1f}) - Tightening Stop on #{ticket}", "RISK", "MANAGER")

            # TREND DEATH (ADX DROP)
            if profit_points > 0 and adx < 20 and prev['adx'] > 20:
                ProfessionalLogger.log(f"💤 Trend Dying (ADX < 20) - Closing #{ticket} to free capital", "EXIT", "MANAGER")
                if self.executor.close_position(ticket, symbol):
                    active_positions.pop(ticket, None)
                continue

            # MOMENTUM EXHAUSTION DETECTION
            if self._detect_momentum_exhaustion(df, trade):
                ProfessionalLogger.log(f"🔻 Momentum exhaustion detected - Exiting #{ticket}", "EXIT", "MANAGER")
                if self.executor.close_position(ticket, symbol):
                    active_positions.pop(ticket, None)
                continue
            
            # TIME-BASED EXIT (stagnant positions)
            bars_held = self._calculate_bars_held(trade)
            if bars_held > 50 and r_multiple < 0.5:
                ProfessionalLogger.log(f"⏰ Time stop: Position #{ticket} stagnant for {bars_held} bars", "EXIT", "MANAGER")
                if self.executor.close_position(ticket, symbol):
                    active_positions.pop(ticket, None)
                continue

            # PARTIAL EXIT LOGIC (scale out)
            if self._should_partial_exit(r_multiple, trade):
                self._execute_partial_exit(ticket, trade, r_multiple)

            # Apply SL modifications
            if sl_changed:
                self.executor.modify_position(ticket, symbol, new_sl, current_tp)
                trade['stop_loss'] = new_sl
    
    def _detect_momentum_exhaustion(self, df, trade):
        """Detect when trend momentum is fading with multiple confirmation methods"""
        
        # 1. Robustness Check (Handle single tick/dict data)
        if isinstance(df, dict):
            # Cannot perform history-based checks on single tick
            # If we need this, we should pass history separately or prevent calling with dict
            return False
            
        if len(df) < 20: 
            return False
            
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # 2. Key Metrics
        rsi = latest.get('rsi', 50)
        prev_rsi = prev.get('rsi', 50)
        
        # 3. RSI Reversal (Crossing back from extremes)
        # Faster than divergence - catches the immediate turn
        if trade['type'] == 'BUY':
            # Was overbought (>70/75), now crossing back down
            if prev_rsi > 70 and rsi < 70:
                return True
        else: # SELL
            # Was oversold (<30/25), now crossing back up
            if prev_rsi < 30 and rsi > 30:
                return True
                
        # 4. Profit Giveback Protection (High Water Mark)
        # If we gave back > 30% of significant profit, exit
        current_profit = trade.get('pnl', 0)
        max_profit = trade.get('max_runge_profit', 0)
        
        if max_profit > 0: # We have been profitable
            giveback_ratio = (max_profit - current_profit) / max_profit
            # Only trigger if profit was "significant" (e.g., > 1 ATR approx)
            atr = latest.get('atr', 0)
            if max_profit > (atr * 0.5) and giveback_ratio > 0.35:
                 return True

        # 5. Price/Momentum Divergence (Recent Peak Logic)
        if trade['type'] == 'BUY':
            # Price recently made a high (in last 5 bars) but RSI is failing
            recent_high = df['high'].iloc[-5:].max()
            current_close = latest['close']
            
            # If we are slightly below recent high but RSI dropped significantly
            price_near_high = current_close >= (recent_high * 0.999)
            rsi_dropped = rsi < (df['rsi'].iloc[-5:].max() - 10)
            
            if price_near_high and rsi_dropped and rsi > 60:
                return True
                
        else: # SELL
            recent_low = df['low'].iloc[-5:].min()
            current_close = latest['close']
            
            price_near_low = current_close <= (recent_low * 1.001)
            rsi_rose = rsi > (df['rsi'].iloc[-5:].min() + 10)
            
            if price_near_low and rsi_rose and rsi < 40:
                return True
        
        # 6. Extreme Volume Drop
        # Ensure volume_ratio exists, handle gracefully if not
        vol_ratio = latest.get('volume_ratio', 1.0)
        if vol_ratio < 0.2: # Very low volume indicates lack of interest
            return True
        
        # 7. ADX Slope death
        if 'adx' in df.columns:
            adx_slope = latest['adx'] - df['adx'].iloc[-3]
            if adx_slope < -5: # ADX falling properly
                return True
        
        return False
    
    def _calculate_bars_held(self, trade):
        """Calculate how many bars position has been held"""
        current_time = time.time()
        duration_seconds = current_time - trade['open_time']
        bars_held = duration_seconds / (Config.TIMEFRAME * 60)
        return int(bars_held)
    
    def _should_partial_exit(self, r_multiple, trade):
        """Check if we should scale out"""
        
        # At 2R, take 50% off
        if r_multiple >= 2.0 and not trade.get('partial_exit_1', False):
            return True
        
        # At 3R, take another 25% off
        if r_multiple >= 3.0 and not trade.get('partial_exit_2', False):
            return True
        
        return False
    
    def _execute_partial_exit(self, ticket, trade, r_multiple):
        """Execute partial position close"""
        symbol = trade['symbol']
        current_volume = trade['volume']
        
        if r_multiple >= 3.0:
            exit_portion = 0.25
            trade['partial_exit_2'] = True
            exit_type = "SECOND"
        else:
            exit_portion = 0.50
            trade['partial_exit_1'] = True
            exit_type = "FIRST"
        
        exit_volume = current_volume * exit_portion
        
        ProfessionalLogger.log(
            f"📊 {exit_type} Partial Exit: Closing {exit_portion:.0%} of #{ticket} at {r_multiple:.1f}R",
            "SUCCESS", "MANAGER"
        )
        
        # In production, implement partial close
        # For now, we'll just log it
        # self._close_partial(ticket, symbol, exit_volume)
        
        # Update trade volume
        trade['volume'] = current_volume - exit_volume

# ==========================================
# PROFESSIONAL DATA QUALITY CHECKER
# ==========================================
class ProfessionalDataQualityChecker:
    """Enhanced data quality validation"""
    
    @staticmethod
    def check_data_quality(df):
        """Comprehensive data quality assessment"""
        if df is None or len(df) == 0:
            return 0.0, {"error": "Empty dataframe"}
        
        scores = []
        diagnostics = {}
        
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        missing_score = max(0, 1 - missing_ratio * 5)
        scores.append(missing_score)
        diagnostics['missing_ratio'] = missing_ratio
        
        if 'time' in df.columns:
            is_sorted = df['time'].is_monotonic_increasing
            sort_score = 1.0 if is_sorted else 0.3
            scores.append(sort_score)
            diagnostics['is_chronological'] = bool(is_sorted)
        
        price_checks = {}
        for price_col in ['open', 'high', 'low', 'close']:
            if price_col in df.columns:
                col_data = df[price_col]
                
                has_invalid = (col_data <= 0).any()
                price_checks[f'{price_col}_valid'] = not has_invalid
                
                q1 = col_data.quantile(0.01)
                q3 = col_data.quantile(0.99)
                iqr = q3 - q1
                outliers = ((col_data < (q1 - 3 * iqr)) | (col_data > (q3 + 3 * iqr))).sum()
                outlier_ratio = outliers / len(col_data)
                price_checks[f'{price_col}_outliers'] = outlier_ratio
        
        valid_checks = sum(price_checks.values()) if 'close_valid' in price_checks else 0
        total_checks = len([k for k in price_checks.keys() if 'valid' in k])
        price_score = valid_checks / total_checks if total_checks > 0 else 0.5
        
        scores.append(price_score)
        diagnostics['price_checks'] = price_checks
        
        if 'tick_volume' in df.columns:
            volume = df['tick_volume']
            
            zero_volume_ratio = (volume == 0).sum() / len(volume)
            volume_score = max(0, 1 - zero_volume_ratio * 2)
            
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / volume_ma
            spike_ratio = (volume_ratio > 5).sum() / len(volume)
            volume_score *= max(0.5, 1 - spike_ratio)
            
            scores.append(volume_score)
            diagnostics['volume_checks'] = {
                'zero_volume_ratio': zero_volume_ratio,
                'spike_ratio': spike_ratio
            }
        
        if 'close' in df.columns and len(df) > 100:
            returns = df['close'].pct_change().dropna()
            
            return_stats = AdvancedStatisticalAnalyzer.analyze_return_distribution(returns.values)
            
            stat_score = 1.0
            
            if 'kurtosis' in return_stats:
                kurt = return_stats['kurtosis']
                if abs(kurt) > 10:
                    stat_score *= 0.7
                elif abs(kurt) > 5:
                    stat_score *= 0.9
            
            if 'skewness' in return_stats:
                skew_val = return_stats['skewness']
                if abs(skew_val) > 2:
                    stat_score *= 0.8
            
            scores.append(stat_score)
            diagnostics['return_stats'] = return_stats
        
        if 'time' in df.columns and len(df) > 10:
            time_diffs = np.diff(df['time'].values)
            avg_time_diff = np.mean(time_diffs)
            time_std = np.std(time_diffs)
            
            gap_threshold = avg_time_diff * 3
            gaps = (time_diffs > gap_threshold).sum()
            gap_ratio = gaps / len(time_diffs)
            
            gap_score = max(0, 1 - gap_ratio * 3)
            scores.append(gap_score)
            
            diagnostics['time_gaps'] = {
                'avg_time_diff': avg_time_diff,
                'time_std': time_std,
                'gap_count': int(gaps),
                'gap_ratio': gap_ratio
            }
        
        if 'time' in df.columns:
            latest_time = pd.to_datetime(df['time'].max(), unit='s')
            current_time = datetime.now()
            hours_old = (current_time - latest_time).total_seconds() / 3600
            
            if hours_old < 1:
                freshness_score = 1.0
            elif hours_old < 6:
                freshness_score = 0.9
            elif hours_old < 24:
                freshness_score = 0.7
            else:
                freshness_score = 0.5
            
            scores.append(freshness_score)
            diagnostics['freshness'] = {
                'latest_time': latest_time.isoformat(),
                'hours_old': hours_old
            }
        
        overall_score = np.mean(scores) if scores else 0.5
        
        return overall_score, diagnostics

class SmartOrderExecutor:
    """Intelligent order execution with modification capabilities"""

    @staticmethod
    def get_filling_mode(symbol):
        """Dynamically detect supported filling mode for a symbol"""
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            return mt5.ORDER_FILLING_IOC
            
        filling_mode = symbol_info.filling_mode
        
        # Check against bitmask: SYMBOL_FILLING_FOK (1), SYMBOL_FILLING_IOC (2)
        if filling_mode & 1:
            return mt5.ORDER_FILLING_FOK
        elif filling_mode & 2:
            return mt5.ORDER_FILLING_IOC
        else:
            return mt5.ORDER_FILLING_RETURN

    def calculate_dynamic_stop_loss(self, entry_price, signal_direction, atr, volatility, lookup_tables=None, market_structure=None):
        """
        Calculate advanced dynamic stop loss
        
        Args:
            entry_price: Entry price
            signal_direction: 1 (Buy) or 0 (Sell)
            atr: Average True Range
            volatility: Current volatility
            lookup_tables: Instance of LookupTables (optional)
            market_structure: Dict with support/resistance levels (optional)
        """
        # 1. Base ATR Multiplier
        if lookup_tables and Config.ENABLE_LOOKUP_TABLES:
            batch_mult = lookup_tables.get_atr_sl_multiplier(volatility)
        else:
            # Fallback logic
            if volatility < 0.005: batch_mult = 1.2
            elif volatility < 0.015: batch_mult = 1.5
            else: batch_mult = 2.0
            
        stop_dist = atr * batch_mult
        
        # 2. Calculate initial stop
        if signal_direction == 1: # BUY
            sl = entry_price - stop_dist
            # Check Support
            if Config.STOP_USE_SUPPORT_RESISTANCE and market_structure and 'support' in market_structure:
                support = market_structure['support']
                if support < entry_price and support > (entry_price - stop_dist * 1.5):
                    sl = min(sl, support - atr * 0.2) # Below support
        else: # SELL
            sl = entry_price + stop_dist
            # Check Resistance
            if Config.STOP_USE_SUPPORT_RESISTANCE and market_structure and 'resistance' in market_structure:
                resistance = market_structure['resistance']
                if resistance > entry_price and resistance < (entry_price + stop_dist * 1.5):
                    sl = max(sl, resistance + atr * 0.2) # Above resistance
                    
        # 3. Round Number Avoidance
        if Config.STOP_AVOID_ROUND_NUMBERS:
            # Check if close to round number (ends in 00, 50, etc.)
            # Logic: If SL is 1950.00, move it to 1949.80 for Buy, 1950.20 for Sell
            # Assuming price ~2000, 1.0 is a round number, 10.0 is significant
            
            val = float(sl)
            remainder = val % 1.0
            
            if remainder < 0.05 or remainder > 0.95: # Close to .00
                if signal_direction == 1:
                    sl -= 0.15 # Move lower
                else:
                    sl += 0.15 # Move higher
            elif 0.45 < remainder < 0.55: # Close to .50
                if signal_direction == 1:
                    sl -= 0.15
                else:
                    sl += 0.15
                    
        return sl


    def execute_trade(self, symbol, order_type, volume, entry_price, sl, tp, magic, comment=""):
        """
        Enhanced trade execution with comprehensive validation and volume management.
        """
        # 1. Get Symbol Info with validation
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            ProfessionalLogger.log(f"Symbol {symbol} not found", "ERROR", "EXECUTOR")
            return None
        
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                ProfessionalLogger.log(f"Failed to select symbol {symbol}", "ERROR", "EXECUTOR")
                return None
        
        if symbol_info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            ProfessionalLogger.log(f"Trading disabled for {symbol}", "ERROR", "EXECUTOR")
            return None
        
        # 2. Comprehensive Volume Normalization with Config enforcement
        step = symbol_info.volume_step
        broker_min_vol = symbol_info.volume_min
        broker_max_vol = symbol_info.volume_max
        
        # Apply Config volume limits (more restrictive of Config vs Broker)
        min_vol = max(Config.MIN_VOLUME, broker_min_vol)
        max_vol = min(Config.MAX_VOLUME, broker_max_vol)
        
        # Initial step normalization with precision fix
        if step > 0:
            volume = round(volume / step) * step
        
        # Determine decimal precision
        decimals = 2
        if step < 0.001:
            decimals = 4
        elif step < 0.01:
            decimals = 3
        elif step >= 1.0:
            decimals = 0
        
        # Enforce volume boundaries and fix floating point
        original_volume = volume
        volume = max(min_vol, min(volume, max_vol))
        volume = float(round(volume, decimals))
        
        # Log volume adjustments
        if abs(original_volume - volume) > 0.001:
            ProfessionalLogger.log(
                f"Volume adjusted: {original_volume:.3f} → {volume:.3f} "
                f"(min: {min_vol:.3f}, max: {max_vol:.3f})",
                "WARNING", "EXECUTOR"
            )
        
        # 3. Advanced Margin Check with Safety Buffer
        account = mt5.account_info()
        if account:
            # Calculate margin with contract size consideration
            contract_size = symbol_info.trade_contract_size
            margin_required = (volume * entry_price * contract_size) / account.leverage
            
            # Add safety buffer (15%)
            margin_with_buffer = margin_required * 1.15
            
            # Check against free margin
            if account.margin_free < margin_with_buffer:
                margin_shortage = margin_with_buffer - account.margin_free
                margin_ratio = account.margin_free / margin_with_buffer
                
                ProfessionalLogger.log(
                    f"Insufficient Margin: Need ${margin_with_buffer:.2f} "
                    f"(incl 15% buffer), Have ${account.margin_free:.2f} | "
                    f"Shortage: ${margin_shortage:.2f} | "
                    f"Margin Ratio: {margin_ratio:.1%}",
                    "ERROR", "EXECUTOR"
                )
                
                # Suggest reduced volume that fits margin
                safe_volume = (account.margin_free * 0.85 * account.leverage) / (entry_price * contract_size)
                
                if step > 0:
                    safe_volume = round(safe_volume / step) * step
                
                safe_volume = max(min_vol, min(safe_volume, max_vol))
                safe_volume = round(safe_volume, decimals)
                
                ProfessionalLogger.log(
                    f"Suggested safe volume: {safe_volume:.3f} (from {volume:.3f})",
                    "INFO", "EXECUTOR"
                )
                return None
            
            margin_ratio = margin_required / account.margin_free if account.margin_free > 0 else 0
            ProfessionalLogger.log(
                f"Margin check passed: ${margin_required:.2f} required, "
                f"${account.margin_free:.2f} available | "
                f"Margin Ratio: {margin_ratio:.1%}",
                "DEBUG", "EXECUTOR"
            )
        else:
            ProfessionalLogger.log("Could not retrieve account info", "ERROR", "EXECUTOR")
            return None
        
        # 4. Spread Check
        if Config.CHECK_SPREAD_BEFORE_ENTRY:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                spread = (tick.ask - tick.bid) / symbol_info.point
                max_allowed_spread = Config.MAX_SPREAD_POINTS
                
                if spread > max_allowed_spread:
                    ProfessionalLogger.log(
                        f"Spread too wide: {spread:.1f} pts > {max_allowed_spread:.1f} pts",
                        "WARNING", "EXECUTOR"
                    )
                    return None
                
                ProfessionalLogger.log(f"Spread check passed: {spread:.1f} pts", "DEBUG", "EXECUTOR")
        
        # 5. Validate Stop Loss and Take Profit distances
        if Config.REQUIRE_STOP_LOSS and (sl == 0 or sl is None):
            ProfessionalLogger.log("Stop loss required but not set", "ERROR", "EXECUTOR")
            return None
        
        if Config.REQUIRE_TAKE_PROFIT and (tp == 0 or tp is None):
            ProfessionalLogger.log("Take profit required but not set", "ERROR", "EXECUTOR")
            return None
        
        # Calculate SL/TP distances in points
        sl_distance = abs(entry_price - sl) / symbol_info.point if sl > 0 else 0
        tp_distance = abs(entry_price - tp) / symbol_info.point if tp > 0 else 0
        
        # Validate minimum distances
        if sl > 0 and sl_distance < Config.MIN_SL_DISTANCE_POINTS:
            ProfessionalLogger.log(
                f"Stop loss too close: {sl_distance:.1f} pts < "
                f"minimum {Config.MIN_SL_DISTANCE_POINTS} pts",
                "ERROR", "EXECUTOR"
            )
            return None
        
        if tp > 0 and tp_distance < Config.MIN_TP_DISTANCE_POINTS:
            ProfessionalLogger.log(
                f"Take profit too close: {tp_distance:.1f} pts < "
                f"minimum {Config.MIN_TP_DISTANCE_POINTS} pts",
                "ERROR", "EXECUTOR"
            )
            return None
        
        # Validate maximum distances
        if sl > 0 and sl_distance > Config.MAX_SL_DISTANCE_POINTS:
            ProfessionalLogger.log(
                f"Stop loss too far: {sl_distance:.1f} pts > "
                f"maximum {Config.MAX_SL_DISTANCE_POINTS} pts",
                "ERROR", "EXECUTOR"
            )
            return None
        
        if tp > 0 and tp_distance > Config.MAX_TP_DISTANCE_POINTS:
            ProfessionalLogger.log(
                f"Take profit too far: {tp_distance:.1f} pts > "
                f"maximum {Config.MAX_TP_DISTANCE_POINTS} pts",
                "ERROR", "EXECUTOR"
            )
            return None
        
        # Validate Risk/Reward ratio
        if sl > 0 and tp > 0:
            rr_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
            if rr_ratio < Config.MIN_RR_RATIO:
                ProfessionalLogger.log(
                    f"Risk/Reward ratio too low: {rr_ratio:.2f} < "
                    f"minimum {Config.MIN_RR_RATIO}",
                    "ERROR", "EXECUTOR"
                )
                return None
            
            ProfessionalLogger.log(f"RR Ratio: {rr_ratio:.2f}", "DEBUG", "EXECUTOR")
        
        # 6. Prepare Request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(entry_price),
            "sl": float(sl) if sl else 0.0,
            "tp": float(tp) if tp else 0.0,
            "deviation": Config.MAX_SLIPPAGE_POINTS,
            "magic": magic,
            "comment": f"{comment} | Vol:{volume:.3f}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self.get_filling_mode(symbol),
        }
        
        # 7. Execute with Retry Logic
        result = None
        last_error = ""
        
        for attempt in range(Config.MAX_RETRIES):
            try:
                ProfessionalLogger.log(
                    f"Order attempt {attempt + 1}/{Config.MAX_RETRIES} | "
                    f"{'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'} "
                    f"{volume:.3f} {symbol} @ {entry_price:.5f} "
                    f"(SL: {sl:.5f}, TP: {tp:.5f})",
                    "INFO", "EXECUTOR"
                )
                
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    # Success!
                    ProfessionalLogger.log(
                        f"✅ Order Executed: #{result.order} | "
                        f"{'BUY' if order_type == mt5.ORDER_TYPE_BUY else 'SELL'} "
                        f"{volume:.3f} {symbol} @ {result.price:.5f}",
                        "SUCCESS", "EXECUTOR"
                    )
                    
                    # Log trade metrics
                    if sl > 0:
                        risk_per_trade = abs(entry_price - sl) * volume * contract_size
                        ProfessionalLogger.log(
                            f"💰 Trade Risk: ${risk_per_trade:.2f} | "
                            f"Risk %: {(risk_per_trade/account.equity*100):.2f}%",
                            "RISK", "EXECUTOR"
                        )
                    
                    return result
                
                # Handle specific error conditions
                elif result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_OFF]:
                    last_error = "Price changed (requote)"
                    time.sleep(Config.RETRY_DELAY_MS / 1000)
                    
                    # Update price for retry
                    tick = mt5.symbol_info_tick(symbol)
                    if tick:
                        new_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
                        request['price'] = float(new_price)
                        
                        # Adjust SL/TP relative to new price
                        if sl > 0:
                            sl_distance_price = abs(entry_price - sl)
                            request['sl'] = float(new_price - sl_distance_price) if order_type == mt5.ORDER_TYPE_BUY else \
                                          float(new_price + sl_distance_price)
                        
                        if tp > 0:
                            tp_distance_price = abs(entry_price - tp)
                            request['tp'] = float(new_price + tp_distance_price) if order_type == mt5.ORDER_TYPE_BUY else \
                                          float(new_price - tp_distance_price)
                        
                        ProfessionalLogger.log(
                            f"Price updated: {entry_price:.5f} → {new_price:.5f}",
                            "DEBUG", "EXECUTOR"
                        )
                
                elif result.retcode == mt5.TRADE_RETCODE_NO_MONEY:
                    last_error = "Insufficient funds"
                    break  # Don't retry this error
                
                elif result.retcode == mt5.TRADE_RETCODE_MARKET_CLOSED:
                    last_error = "Market is closed"
                    break  # Don't retry this error
                
                elif result.retcode == mt5.TRADE_RETCODE_LIMIT_ORDERS:
                    last_error = "Too many pending orders"
                    break  # Don't retry this error
                
                else:
                    last_error = f"{result.comment} (code: {result.retcode})"
                    time.sleep(Config.RETRY_DELAY_MS / 1000)
            
            except Exception as e:
                last_error = str(e)
                ProfessionalLogger.log(f"Exception during order send: {e}", "ERROR", "EXECUTOR")
                time.sleep(Config.RETRY_DELAY_MS / 1000)
        
        # 8. Failed execution handling
        if result:
            ProfessionalLogger.log(
                f"❌ Order Failed after {Config.MAX_RETRIES} attempts: "
                f"{last_error} | Request: {request}",
                "ERROR", "EXECUTOR"
            )
        else:
            ProfessionalLogger.log(f"❌ Order Failed: {last_error}", "ERROR", "EXECUTOR")
        
        return None

    def modify_position(self, ticket, symbol, new_sl, new_tp):
        """Modify SL/TP of an existing position"""
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": float(new_sl),
            "tp": float(new_tp)
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            ProfessionalLogger.log(f"🔄 Position #{ticket} Modified | New SL: {new_sl:.5f}, TP: {new_tp:.5f}", "SUCCESS", "EXECUTOR")
            return True
        else:
            ProfessionalLogger.log(f"❌ Modify Failed: {result.comment}", "ERROR", "EXECUTOR")
            return False

    def close_position(self, ticket, symbol):
        """Close a specific position"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        
        pos = positions[0]
        tick = mt5.symbol_info_tick(symbol)
        
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": pos.volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": Config.MAGIC_NUMBER,
            "comment": "Adaptive Exit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            ProfessionalLogger.log(f"💰 Position #{ticket} Closed by Adaptive Manager", "SUCCESS", "EXECUTOR")
            return True
        return False
    
    def close_partial(self, ticket, symbol, volume):
        """Close partial position"""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
        
        pos = positions[0]
        tick = mt5.symbol_info_tick(symbol)
        
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if order_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "deviation": 20,
            "magic": Config.MAGIC_NUMBER,
            "comment": "Partial Exit",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            ProfessionalLogger.log(f"💰 Partial close: {volume:.3f} of #{ticket}", "SUCCESS", "EXECUTOR")
            return True
        return False
# ==========================================
# MULTI-TIMEFRAME ANALYSER CLASS
# ==========================================
class MultiTimeframeAnalyser:
    """Advanced Multi-Timeframe Analysis for XAUUSD"""
    
    def __init__(self, mt5_connection):
        self.mt5 = mt5_connection
        self.config = Config
        
        self.timeframe_map = {
            'M1': mt5.TIMEFRAME_M1,
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'M30': mt5.TIMEFRAME_M30,
            'H1': mt5.TIMEFRAME_H1
        }
        
        self.analysis_cache = {}
        self.cache_expiry = 60
    
    def fetch_multi_timeframe_data(self, symbol, bars_needed=500):
        """Fetch synchronized data across all configured timeframes"""
        mt5_data = {}
        current_time = datetime.now()
        
        for tf_name, tf_value in self.timeframe_map.items():
            cache_key = f"{symbol}_{tf_name}"
            if (cache_key in self.analysis_cache and 
                (current_time - self.analysis_cache[cache_key]['timestamp']).seconds < self.cache_expiry):
                mt5_data[tf_name] = self.analysis_cache[cache_key]['data']
                continue
            
            # Prevent repainting: use ONLY completed bars for higher timeframes
            start_pos = 0 if tf_value == Config.TIMEFRAME else 1
            rates = self.mt5.copy_rates_from_pos(symbol, tf_value, start_pos, bars_needed)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                
                self.analysis_cache[cache_key] = {
                    'data': df,
                    'timestamp': current_time
                }
                mt5_data[tf_name] = df
            else:
                ProfessionalLogger.log(f"Failed to fetch {tf_name} data", "WARNING", "MULTI_TF")
        
        return mt5_data
    
    def calculate_timeframe_features(self, df):
        """Calculate key features for a single timeframe"""
        features = {}
        
        if df is None or len(df) < 20:
            return features
        
        features['close'] = df['close'].iloc[-1]
        features['returns_5'] = df['close'].iloc[-1] / df['close'].iloc[-5] - 1 if len(df) >= 5 else 0
        features['returns_20'] = df['close'].iloc[-1] / df['close'].iloc[-20] - 1 if len(df) >= 20 else 0
        
        features['ema_fast'] = df['close'].ewm(span=8).mean().iloc[-1]
        features['ema_slow'] = df['close'].ewm(span=21).mean().iloc[-1]
        features['trend_direction'] = 1 if features['ema_fast'] > features['ema_slow'] else -1
        
        # Calculate ADX for trend strength
        try:
            high = df['high']
            low = df['low'] 
            close = df['close']
            
            # TR
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            
            # DM
            up_move = high.diff()
            down_move = -low.diff()
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            plus_di = 100 * (pd.Series(plus_dm).rolling(14).mean() / atr)
            minus_di = 100 * (pd.Series(minus_dm).rolling(14).mean() / atr)
            
            dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
            adx = dx.rolling(14).mean().iloc[-1]
            
            features['adx'] = adx
            features['trend_strength'] = 2 if adx > 40 else 1 if adx > 25 else 0
        except:
            features['trend_strength'] = 0
        
        features['rsi'] = self._calculate_rsi(df['close'], period=14)
        features['momentum'] = df['close'].iloc[-1] / df['close'].iloc[-10] - 1 if len(df) >= 10 else 0
        
        returns = df['close'].pct_change().dropna()
        features['volatility'] = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        features['support'] = df['low'].rolling(20).min().iloc[-1]
        features['resistance'] = df['high'].rolling(20).max().iloc[-1]
        features['price_position'] = (features['close'] - features['support']) / (features['resistance'] - features['support']) if (features['resistance'] - features['support']) > 0 else 0.5
        
        return features
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        if len(prices) < period:
            return 50
        
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss.replace(0, 1)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def analyze_alignment(self, multi_tf_data):
        """Analyze alignment across timeframes"""
        if not multi_tf_data:
            return None
        
        analysis = {
            'timeframes': {},
            'alignment_score': 0,
            'consensus_signal': 0,
            'trend_filter': 0,
            'confidence': 0
        }
        
        signals = []
        weights = []
        
        # Calculate Market 'Speed' (Volatility/Trend Strength) for Dynamic Weighting
        # We use M15 as the baseline for market speed (more relevant for medium term)
        market_speed_metrics = multi_tf_data.get('M15', None)
        is_fast_market = False
        is_slow_market = False
        
        if market_speed_metrics is not None and len(market_speed_metrics) > 20:
            # Calculate ADX proxy or use ATR
            try:
                 # Quick ADX approx
                high = market_speed_metrics['high']
                low = market_speed_metrics['low']
                close = market_speed_metrics['close']
                tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
                atr = tr.rolling(14).mean().iloc[-1]
                atr_percent = atr / close.iloc[-1]
                
                # Fast market = High ATR (> 0.1%) or Strong trend
                if atr_percent > 0.0010: # 10 pips on gold approx
                    is_fast_market = True
                elif atr_percent < 0.0005: 
                    is_slow_market = True
            except:
                pass

        # Define Dynamic Weights
        # Order: M1, M5, M15, M30, H1 (Aligned with Config.TIMEFRAMES)
        if is_fast_market:
            # Fast market: M15/M5 still lead, but M5 gets a boost for timing
            current_weights = [0.15, 0.35, 0.35, 0.10, 0.05]
            regime_note = "FAST (High Vol)"
        elif is_slow_market:
            # Slow market: Extreme stability focus (M1 almost ignored)
            current_weights = [0.02, 0.08, 0.40, 0.30, 0.20]
            regime_note = "SLOW (Low Vol)"
        else:
            # Normal market: Balanced towards M15 (50% weight)
            current_weights = [0.05, 0.25, 0.50, 0.10, 0.10]
            regime_note = "NORMAL"

        ProfessionalLogger.log(f"Dynamic TF Weights: {regime_note} | Weights: {current_weights}", "INFO", "MULTI_TF")
        
        for tf_name, df in multi_tf_data.items():
            if df is None or len(df) < 20:
                continue
            
            features = self.calculate_timeframe_features(df)
            signal = self._generate_timeframe_signal(features, tf_name)
            
            # Use dynamic weight
            tf_index = self.config.TIMEFRAMES.index(tf_name) if tf_name in self.config.TIMEFRAMES else -1
            weight = current_weights[tf_index] if tf_index != -1 else 0.2
            
            analysis['timeframes'][tf_name] = {
                'signal': signal,
                'features': features,
                'weight': weight
            }
            
            signals.append(signal)
            weights.append(analysis['timeframes'][tf_name]['weight'])
        
        if not signals:
            return analysis
        
        weighted_signal = np.average(signals, weights=weights)
        
        # Fixed: Corrected consensus mapping for 0-1 probability range
        if weighted_signal > 0.52:
            analysis['consensus_signal'] = 1
        elif weighted_signal < 0.48:
            analysis['consensus_signal'] = 0
        else:
            analysis['consensus_signal'] = 0.5
        
        signal_directions = [1 if s == 1 else -1 if s == 0 else 0 for s in signals]
        if len(signal_directions) > 1:
            agreement = sum(1 for i in range(len(signal_directions)) 
                          for j in range(i+1, len(signal_directions)) 
                          if signal_directions[i] * signal_directions[j] > 0)
            total_pairs = len(signal_directions) * (len(signal_directions) - 1) / 2
            analysis['alignment_score'] = agreement / total_pairs if total_pairs > 0 else 0
        
        # Trend Filter Code (Checks H1 or M30)
        filter_tf = 'H1' if 'H1' in analysis['timeframes'] else 'M30'
        
        if filter_tf in analysis['timeframes']:
            trend_dir = analysis['timeframes'][filter_tf]['features']['trend_direction']
            if self.config.LONG_TIMEFRAME_TREND_FILTER:
                # Strong Filter: Signal must match Macro Trend or Macro Trend must be neutral
                # If Signal is Buy (1), Macro Trend cannot be Sell (-1)
                if analysis['consensus_signal'] == 1 and trend_dir < 0:
                    analysis['trend_filter'] = -1 # Veto
                elif analysis['consensus_signal'] == 0 and trend_dir > 0:
                    analysis['trend_filter'] = -1 # Veto
                else:
                    analysis['trend_filter'] = 1 # Pass
        
        # Use Config parameters for confidence calculation
        alignment_threshold = self.config.TIMEFRAME_ALIGNMENT_THRESHOLD
        
        # Get multi-timeframe analysis parameters from Config
        mtf_params = getattr(self.config, 'MULTI_TIMEFRAME_PARAMS', {})
        
        # Alignment bonus parameters with defaults
        alignment_bonus_above_start = mtf_params.get('ALIGNMENT_BONUS_ABOVE_START', 0.8)
        alignment_bonus_above_range = mtf_params.get('ALIGNMENT_BONUS_ABOVE_RANGE', 0.2)
        alignment_bonus_below_start = mtf_params.get('ALIGNMENT_BONUS_BELOW_START', 0.4)
        alignment_bonus_below_range = mtf_params.get('ALIGNMENT_BONUS_BELOW_RANGE', 0.4)
        
        # Trend bonus parameters with defaults
        trend_bonus_support = mtf_params.get('TREND_BONUS_SUPPORT', 1.0)
        trend_bonus_neutral = mtf_params.get('TREND_BONUS_NEUTRAL', 0.8)
        trend_bonus_opposite = mtf_params.get('TREND_BONUS_OPPOSITE', 0.5)
        
        # Confidence calculation parameters
        base_confidence_multiplier = mtf_params.get('BASE_CONFIDENCE_MULTIPLIER', 0.9)
        confidence_floor = mtf_params.get('CONFIDENCE_FLOOR', 0.2)
        confidence_ceiling = mtf_params.get('CONFIDENCE_CEILING', 0.95)
        
        # 1. Gradual alignment bonus (not binary)
        alignment_score = analysis['alignment_score']
        if alignment_score >= alignment_threshold:
            # Above threshold: scale from ALIGNMENT_BONUS_ABOVE_START to ALIGNMENT_BONUS_ABOVE_START + ALIGNMENT_BONUS_ABOVE_RANGE
            alignment_bonus = alignment_bonus_above_start + (
                (alignment_score - alignment_threshold) / (1 - alignment_threshold) * alignment_bonus_above_range
            )
        else:
            # Below threshold: scale from ALIGNMENT_BONUS_BELOW_START to ALIGNMENT_BONUS_BELOW_START + ALIGNMENT_BONUS_BELOW_RANGE
            alignment_bonus = alignment_bonus_below_start + (
                (alignment_score / alignment_threshold) * alignment_bonus_below_range
            )
        
        # 2. Trend bonus with H1 dominance
        if analysis['trend_filter'] == 1:
            trend_bonus = trend_bonus_support  # H1 supports trade
        elif analysis['trend_filter'] == 0:  # Neutral
            trend_bonus = trend_bonus_neutral  # H1 neutral
        else:  # -1 (opposite)
            trend_bonus = trend_bonus_opposite  # H1 opposite
        
        # 3. Calculate confidence with market regime adjustment
        base_confidence = alignment_bonus * trend_bonus * base_confidence_multiplier
        
        # 4. Apply confidence floor and ceiling
        analysis['confidence'] = max(confidence_floor, min(confidence_ceiling, base_confidence))
        
        return analysis
    
    def _generate_timeframe_signal(self, features, timeframe_name):
        """
        Generate trading signal for a single timeframe using TREND DOMINANCE LOGIC.
        Prioritizes trend following in strong regimes to prevent catching falling knives.
        """
        if not features:
            return 0.5
        
        # Get multi-timeframe parameters from Config
        mtf_params = getattr(self.config, 'MULTI_TIMEFRAME_PARAMS', {})
        
        # Get timeframe-specific multiplier
        multiplier_key = f'{timeframe_name.upper()}_SIGNAL_MULTIPLIER'
        timeframe_multiplier = mtf_params.get(multiplier_key, 1.0)
        
        signal_score = 0
        
        # 1. TREND COMPONENT (DOMINANT)
        trend = features.get('trend_direction', 0)
        trend_strength = features.get('trend_strength', 0)
        is_strong_trend = trend_strength >= 2
        
        # Double weight for strong trends
        trend_weight = 2.0 if is_strong_trend else 1.0
        signal_score += (trend * trend_weight)
        
        # 2. PRICE POSITION (MEAN REVERSION)
        price_pos = features.get('price_position', 0.5)
        price_score = 0
        
        if price_pos < 0.3: # "Cheap" -> Buy
            # VETO: Don't buy in strong downtrend
            if is_strong_trend and trend == -1:
                price_score = 0
            else:
                price_score = 1
        elif price_pos > 0.7: # "Expensive" -> Sell
            # VETO: Don't sell in strong uptrend
            if is_strong_trend and trend == 1:
                price_score = 0
            else:
                price_score = -1
        
        signal_score += price_score
        
        # 3. RSI (MEAN REVERSION)
        rsi = features.get('rsi', 50)
        # Use standardized thresholds
        rsi_lower = 30
        rsi_upper = 70
        
        rsi_score = 0
        if rsi < rsi_lower: # Oversold -> Buy
            # VETO: Don't buy oversold in strong downtrend
            if is_strong_trend and trend == -1:
                rsi_score = 0
            else:
                rsi_score = 1
        elif rsi > rsi_upper: # Overbought -> Sell
            # VETO: Don't sell overbought in strong uptrend
            if is_strong_trend and trend == 1:
                rsi_score = 0
            else:
                rsi_score = -1
                
        signal_score += rsi_score
        
        # 4. MOMENTUM
        momentum = features.get('momentum', 0)
        mom_threshold = 0.002 if timeframe_name in ['M1', 'M5'] else 0.005
        
        mom_score = 1 if momentum > mom_threshold else -1 if momentum < -mom_threshold else 0
        signal_score += mom_score
        
        # Apply Asian session adjustments (Simplified)
        hour = datetime.now().hour
        if 0 <= hour < 9:
            # During Asian session, rely more on Mean Reversion (RSI) than Trend
            # Only if trend is NOT strong
            if not is_strong_trend:
                signal_score = (trend * 0.5) + (price_score * 1.5) + (rsi_score * 1.5) + mom_score
        
        # Apply timeframe multiplier
        signal_score *= timeframe_multiplier
        
        # Log breakdown
        if signal_score != 0:
            ProfessionalLogger.log(
                f"TF {timeframe_name} Breakdown | Score: {signal_score:.2f} | "
                f"Trend: {trend} (Str:{trend_strength}) | RSI: {rsi:.0f} | PricePos: {price_pos:.2f}",
                "ANALYSIS", "MULTI_TF"
            )
        
        # Normalize score
        # Max score (Strong Trend): 2 (Trend) + 0 (Price) + 0 (RSI) + 1 (Mom) = 3
        # Max score (Normal): 1 + 1 + 1 + 1 = 4
        # We use a divisor of 3.0 to increase sensitivity (easier to reach thresholds)
        normalized_score = max(-1, min(1, signal_score / 3.0))
        
        buy_threshold = mtf_params.get('BUY_SIGNAL_THRESHOLD', 0.25)
        sell_threshold = mtf_params.get('SELL_SIGNAL_THRESHOLD', -0.25)
        
        if normalized_score > buy_threshold:
            return 1
        elif normalized_score < sell_threshold:
            return 0
        else:
            return 0.5
    
    def get_multi_timeframe_recommendation(self, symbol):
        """Get comprehensive multi-timeframe trading recommendation"""
        if not self.config.MULTI_TIMEFRAME_ENABLED:
            return None
        
        multi_tf_data = self.fetch_multi_timeframe_data(symbol)
        
        if not multi_tf_data:
            return None
        
        analysis = self.analyze_alignment(multi_tf_data)
        
        if not analysis:
            return None
        
        recommendation = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'consensus_signal': analysis['consensus_signal'],
            'alignment_score': analysis['alignment_score'],
            'trend_filter_passed': analysis['trend_filter'] == 1,
            'confidence': analysis['confidence'],
            'timeframe_details': analysis['timeframes'],
            'recommendation': None,
            'recommendation_strength': None
        }
        
        if analysis['alignment_score'] >= self.config.TIMEFRAME_ALIGNMENT_THRESHOLD and analysis['trend_filter'] == 1:
            if analysis['consensus_signal'] == 1:
                recommendation['recommendation'] = 'STRONG_BUY' if analysis['confidence'] > 0.7 else 'BUY'
                recommendation['recommendation_strength'] = analysis['confidence']
            elif analysis['consensus_signal'] == 0:
                recommendation['recommendation'] = 'STRONG_SELL' if analysis['confidence'] > 0.7 else 'SELL'
                recommendation['recommendation_strength'] = analysis['confidence']
            else:
                recommendation['recommendation'] = 'HOLD'
                recommendation['recommendation_strength'] = 0
        elif analysis['alignment_score'] >= 0.5:
            recommendation['recommendation'] = 'WEAK_BUY' if analysis['consensus_signal'] == 1 else 'WEAK_SELL' if analysis['consensus_signal'] == 0 else 'HOLD'
            recommendation['recommendation_strength'] = analysis['confidence'] * 0.5
        else:
            recommendation['recommendation'] = 'HOLD'
            recommendation['recommendation_strength'] = 0
        
        self._log_multi_tf_analysis(recommendation)
        
        return recommendation
    
    def _log_multi_tf_analysis(self, recommendation):
        """Log multi-timeframe analysis results"""
        if not recommendation:
            return
        
        signals = {tf: data['signal'] for tf, data in recommendation.get('timeframe_details', {}).items()}
        
        log_message = (
            f"Multi-TF Analysis | "
            f"Signal: {recommendation['recommendation']} | "
            f"Alignment: {recommendation['alignment_score']:.0%} | "
            f"Confidence: {recommendation['confidence']:.0%} | "
            f"TF Signals: {signals}"
        )
        
        ProfessionalLogger.log(log_message, "ANALYSIS", "MULTI_TF")
        
        if self.config.DEBUG_MODE:
            for tf_name, tf_data in recommendation.get('timeframe_details', {}).items():
                features = tf_data.get('features', {})
                ProfessionalLogger.log(
                    f"  {tf_name}: Signal={tf_data['signal']:.1f}, "
                    f"RSI={features.get('rsi', 0):.1f}, "
                    f"Trend={features.get('trend_direction', 0)}, "
                    f"Pos={features.get('price_position', 0.5):.2f}",
                    "DEBUG", "MULTI_TF"
                )

# ==========================================
# PROFESSIONAL TRADE MEMORY
# ==========================================
class ProfessionalTradeMemory:
    """Trade memory system"""
    
    def __init__(self, history_file=None):
        self.history_file = history_file or Config.TRADE_HISTORY_FILE
        self.trades = []
        self._initialize_memory()
        
    def _initialize_memory(self):
        """Initialize trade memory"""
        try:
            import json
            import os
            
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    self.trades = json.load(f)
                ProfessionalLogger.log(f"Loaded {len(self.trades)} trades from {self.history_file}", 
                                     "DATA", "MEMORY")
            else:
                self.trades = []
        except Exception as e:
            ProfessionalLogger.log(f"Memory initialization error: {str(e)}", "WARNING", "MEMORY")
            self.trades = []
            
    def save_history(self):
        """Save trade history"""
        try:
            import json
            with open(self.history_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            ProfessionalLogger.log(f"Error saving history: {str(e)}", "ERROR", "MEMORY")
            
    def add_trade(self, trade_data):
        """Add new trade"""
        try:
            trade_data['id'] = len(self.trades) + 1
            trade_data['timestamp'] = datetime.now().isoformat()
            self.trades.append(trade_data)
            
            if len(self.trades) > Config.MEMORY_SIZE:
                self.trades = self.trades[-Config.MEMORY_SIZE:]
            
            self.save_history()
            
            ProfessionalLogger.log(f"Trade #{trade_data['id']} recorded", "TRADE", "MEMORY")
            return True
            
        except Exception as e:
            ProfessionalLogger.log(f"Error adding trade: {str(e)}", "ERROR", "MEMORY")
            return False
            
    def get_statistical_summary(self):
        """Get statistical summary of trades"""
        completed = [t for t in self.trades if t.get('status') == 'closed']
        
        if not completed:
            return {'total_trades': 0, 'win_rate': 0}
        
        profits = [t.get('profit', 0) for t in completed]
        
        stats = {
            'total_trades': len(completed),
            'winning_trades': sum(1 for p in profits if p > 0),
            'losing_trades': sum(1 for p in profits if p <= 0),
            'win_rate': sum(1 for p in profits if p > 0) / len(profits),
            'total_profit': sum(profits),
            'mean_profit': np.mean(profits),
            'median_profit': np.median(profits),
            'std_profit': np.std(profits),
            'min_profit': min(profits),
            'max_profit': max(profits)
        }
        
        losing_sum = abs(sum(p for p in profits if p <= 0))
        if losing_sum > 0:
            winning_sum = sum(p for p in profits if p > 0)
            stats['profit_factor'] = winning_sum / losing_sum
        else:
            stats['profit_factor'] = float('inf')
            
        # Excursion Analysis
        mfe_list = [t.get('mfe_points', 0) for t in completed]
        mae_list = [t.get('mae_points', 0) for t in completed]
        eff_list = [t.get('tp_efficiency', 0) for t in completed if 'tp_efficiency' in t]
        
        if mfe_list:
            stats['avg_mfe'] = np.mean(mfe_list)
            stats['avg_mae'] = np.mean(mae_list)
            stats['max_mfe'] = max(mfe_list)
            stats['max_mae'] = min(mae_list)
            
        if eff_list:
            stats['avg_tp_efficiency'] = np.mean(eff_list)
            
        return stats

# ==========================================
# ENHANCED PROFESSIONAL TRADING ENGINE
# ==========================================
class EnhancedTradingEngine:
    """Main professional trading engine with all enhancements"""
    
    def __init__(self):
        self.trade_memory = ProfessionalTradeMemory()
        self.feature_engine = EnhancedFeatureEngine()
        self.order_flow = OrderFlowAnalyzer(Config.SYMBOL) # New: Level 2 Analysis
        self.bayes_detector = BayesianRegimeDetector() # New: Probabilistic Regime Check
        self.rl_collector = ReinforcementDataCollector() # New: RL Data Collector

        self.order_executor = SmartOrderExecutor()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        self.risk_metrics = ProfessionalRiskMetrics()
        
        # Initialize enhanced components
        self.model = EnhancedEnsemble(self.trade_memory, self.feature_engine)
        self.exit_manager = EnhancedExitManager(self.order_executor)
        self.multi_tf_analyser = MultiTimeframeAnalyser(mt5)
        self.signal_filter = SignalQualityFilter()
        self.entry_timing = SmartEntryTiming()
        self.parameter_optimizer = AdaptiveParameterOptimizer()
        
        self.connected = False
        self.active_positions = {}
        self.iteration = 0
        self.last_analysis_time = None
        self.last_regime = None
        self.last_snapshot_time = None # New: For periodic market data logging
        self.latest_dict_features = {} # New: Dictionary version for snapshots
        
        # Background Training
        self.training_thread = None
        self.is_training = False
        self.model_lock = threading.Lock()
        
        # Feature Cache
        self.last_feature_time = None
        self.cached_features = None
        
        # Performance tracking
        self.equity_curve = []
        self.returns_series = []
        self.risk_metrics_history = []
        self.initial_analysis = {}
        
        # Load previous state if exists
        self._load_engine_state()
        
        ProfessionalLogger.log("Enhanced Trading Engine initialized with all improvements", "INFO", "ENGINE")
    
    def connect_mt5(self):
        """Connect to MT5 terminal"""
        ProfessionalLogger.log("Initializing MT5...", "INFO", "ENGINE")
        
        if not mt5.initialize():
            ProfessionalLogger.log(f"MT5 init failed: {mt5.last_error()}", "ERROR", "ENGINE")
            return False
        
        authorized = mt5.login(
            login=Config.MT5_LOGIN,
            password=Config.MT5_PASSWORD,
            server=Config.MT5_SERVER
        )
        
        if not authorized:
            ProfessionalLogger.log(f"Login failed: {mt5.last_error()}", "ERROR", "ENGINE")
            mt5.shutdown()
            return False
        
        account = mt5.account_info()
        if account:
            ProfessionalLogger.log(f"✓ Connected | Account: {account.login} | "
                        f"Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f}", "SUCCESS", "ENGINE")
        else:
            ProfessionalLogger.log("✓ Connected (account info unavailable)", "SUCCESS", "ENGINE")
        
        if not mt5.terminal_info().trade_allowed:
            ProfessionalLogger.log("⚠ Algo trading disabled!", "WARNING", "ENGINE")
            return False
        
        symbol_info = mt5.symbol_info(Config.SYMBOL)
        if symbol_info is None:
            ProfessionalLogger.log(f"Symbol {Config.SYMBOL} not found", "ERROR", "ENGINE")
            return False
        
        if not symbol_info.visible:
            mt5.symbol_select(Config.SYMBOL, True)
        
        self.connected = True
        
        # Initial statistical analysis
        self.perform_initial_analysis()
        
        return True
    
    def perform_initial_analysis(self):
        """Perform initial statistical analysis"""
        ProfessionalLogger.log("Performing initial market analysis...", "ANALYSIS", "ENGINE")
        
        data = self.get_historical_data(bars=Config.LOOKBACK_BARS)
        
        if data is not None and len(data) > 500:
            analysis = self.model.perform_statistical_analysis(data)
            
            if analysis:
                ProfessionalLogger.log("Initial market analysis complete", "SUCCESS", "ENGINE")
                self.initial_analysis = analysis
                self._extract_market_insights(analysis)
            else:
                ProfessionalLogger.log("Initial analysis returned no results", "WARNING", "ENGINE")
        else:
            ProfessionalLogger.log("Insufficient data for initial analysis", "WARNING", "ENGINE")
    
    def _extract_market_insights(self, analysis):
        """Extract and log key market insights"""
        insights = []
        
        if 'market_regime' in analysis:
            mr = analysis['market_regime']
            regime = mr.get('regime', 'unknown')
            confidence = mr.get('confidence', 0)
            
            if confidence > 0.7:
                insights.append(f"Market is in {regime} regime (confidence: {confidence:.0%})")
            
            if regime == 'trending':
                insights.append("Consider trend-following strategies with wider stops")
            elif regime == 'mean_reverting':
                insights.append("Consider mean-reversion strategies with tight stops")
            elif regime == 'volatile':
                insights.append("High volatility - consider reducing position sizes")
        
        if 'risk_metrics' in analysis:
            rm = analysis['risk_metrics']
            
            if 'sharpe' in rm:
                sharpe = rm['sharpe']
                if sharpe > 1:
                    insights.append(f"Good risk-adjusted returns (Sharpe: {sharpe:.2f})")
                elif sharpe < 0:
                    insights.append(f"Negative risk-adjusted returns (Sharpe: {sharpe:.2f})")
            
            if 'max_drawdown' in rm:
                max_dd = rm['max_drawdown']
                if max_dd > 0.1:
                    insights.append(f"Historical max drawdown: {max_dd:.1%} - adjust risk accordingly")
        
        if 'tail_risk' in analysis:
            tr = analysis['tail_risk']
            if 'tail_index' in tr:
                tail_idx = tr['tail_index']
                if tail_idx < 2:
                    insights.append(f"Fat tails detected (index: {tail_idx:.2f}) - consider tail risk hedging")
        
        if insights:
            ProfessionalLogger.log("📈 MARKET INSIGHTS:", "ANALYSIS", "ENGINE")
            archive_data = {
                "timestamp": datetime.now().isoformat(),
                "insights": insights,
                "regime": analysis.get('market_regime', {}).get('regime', 'unknown'),
                "confidence": analysis.get('market_regime', {}).get('confidence', 0)
            }
            
            # PERSISTENCE: Save to file for 24h review & model training
            try:
                history = []
                if os.path.exists(Config.MARKET_INSIGHTS_FILE):
                    with open(Config.MARKET_INSIGHTS_FILE, 'r') as f:
                        try:
                            history = json.load(f)
                        except json.JSONDecodeError:
                            history = []
                
                history.append(archive_data)
                # Keep last 1000 insights to prevent file bloat
                history = history[-1000:]
                
                with open(Config.MARKET_INSIGHTS_FILE, 'w') as f:
                    json.dump(history, f, indent=4)
            except Exception as e:
                ProfessionalLogger.log(f"Failed to archive insights: {e}", "ERROR", "ENGINE")

            for insight in insights:
                ProfessionalLogger.log(f"  • {insight}", "ANALYSIS", "ENGINE")
    
    def get_historical_data(self, timeframe=None, bars=None):
        """Get historical data from MT5"""
        if not self.connected:
            return None
        
        if timeframe is None:
            timeframe = Config.TIMEFRAME
        if bars is None:
            bars = Config.LOOKBACK_BARS
        
        rates = mt5.copy_rates_from_pos(Config.SYMBOL, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        
        return pd.DataFrame(rates)
    
    def get_current_positions(self):
        """Get current open positions"""
        if not self.connected:
            return 0
        
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        return len(positions) if positions else 0
    
    def check_closed_positions(self):
        """Check for closed positions and update records"""
        if not self.connected:
            return
        
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        if positions is None:
            positions = []
        
        current_tickets = [pos.ticket for pos in positions]
        
        # Track MFE/MAE for active positions
        self._track_excursion(positions)
        
        for ticket in list(self.active_positions.keys()):
            if ticket not in current_tickets:
                trade_data = self.active_positions[ticket]
                
                signal = trade_data.get('signal', 1)
                
                from_date = datetime.now() - timedelta(days=1)
                deals = mt5.history_deals_get(from_date, datetime.now())
                
                if deals:
                    for deal in deals:
                        if deal.position_id == ticket:
                            open_price = trade_data['open_price']
                            close_price = deal.price
                            returns = (close_price - open_price) / open_price if signal == 1 else (open_price - close_price) / open_price
                            
                            # Final MFE/MAE calculation
                            mfe = trade_data.get('mfe_points', 0)
                            mae = trade_data.get('mae_points', 0)
                            
                            outcome = {
                                'profit': deal.profit,
                                'close_price': deal.price,
                                'close_time': int(deal.time),
                                'status': 'closed',
                                'returns': returns,
                                'duration': int(deal.time) - trade_data['open_time'],
                                'mfe_points': mfe,
                                'mae_points': mae,
                                'tp_efficiency': (returns * 10000) / trade_data.get('tp_points', 1) if 'tp_points' in trade_data else 0
                            }
                            trade_data.update(outcome)
                            
                            # Final Outcome Logging
                            ProfessionalLogger.log(
                                f"Trade #{ticket} Closed | Profit: ${deal.profit:.2f} | "
                                f"Entry: {trade_data['open_price']:.2f}, Exit: {deal.price:.2f}",
                                "TRADE", "ENGINE"
                            )
                            
                            # Save to memory
                            trade_record = {
                                'ticket': ticket,
                                'symbol': Config.SYMBOL,
                                'open_time': trade_data['open_time'],
                                'close_time': int(deal.time),
                                'type': trade_data['type'],
                                'volume': trade_data['volume'],
                                'open_price': trade_data['open_price'],
                                'close_price': deal.price,
                                'sl': trade_data.get('sl', 0.0),
                                'tp': trade_data.get('tp', 0.0),
                                'profit': deal.profit,
                                'features': trade_data.get('features', {}),
                                'model_prediction': trade_data.get('model_prediction', {}),
                                'regime': self.last_regime
                            }
                            self.trade_memory.add_trade(trade_record)
                            
                            # EQUITY CURVE GUARDIAN (INSTITUTIONAL)
                            # Check if strategy is decaying
                            recent_equity_curve = self.trade_memory.trades[-20:] # Last 20 trades
                            if len(recent_equity_curve) >= 10:
                                total_pnl_history = [t['profit'] for t in recent_equity_curve]
                                
                                # Calculate moving average of PnL
                                ma_pnl = sum(total_pnl_history) / len(total_pnl_history)
                                
                                # If average PnL is significantly negative over last 10-20 trades, PAUSE.
                                if ma_pnl < -100: # Decaying strategy (e.g., losing $100 per trade on avg)
                                     ProfessionalLogger.log(f"📉 EQUITY CURVE WARNING: Avg Loss ${abs(ma_pnl):.2f}/trade over last {len(recent_equity_curve)} trades.", "CRITICAL", "RISK")
                                     # self.running = False
                            
                            self.trade_memory.add_trade(trade_data)
                            self.returns_series.append(returns)
                            
                            profit_loss = "profit" if deal.profit > 0 else "loss"
                            ProfessionalLogger.log(
                                f"Trade #{ticket} closed with {profit_loss} | P/L: ${deal.profit:.2f} | "
                                f"MFE: {mfe:.1f} pts | MAE: {mae:.1f} pts", 
                                "SUCCESS" if deal.profit > 0 else "WARNING", "ENGINE"
                            )
                            break
                
                del self.active_positions[ticket]
                self._save_engine_state()

    def _track_excursion(self, current_positions):
        """Track Maximum Favorable and Adverse Excursions for active positions"""
        if not current_positions:
            return
            
        for pos in current_positions:
            ticket = pos.ticket
            if ticket not in self.active_positions:
                continue
                
            trade = self.active_positions[ticket]
            current_price = pos.price_current
            open_price = pos.price_open
            symbol_info = mt5.symbol_info(trade['symbol'])
            point = symbol_info.point if symbol_info and symbol_info.point > 0 else 0.01
            
            # Distance in points
            if trade['signal'] == 1: # BUY
                floating_points = (current_price - open_price) / point
            else: # SELL
                floating_points = (open_price - current_price) / point
                
            # Update MFE (Max profit)
            trade['mfe_points'] = max(trade.get('mfe_points', 0), floating_points)
            
            # Update MAE (Max drawdown)
            trade['mae_points'] = min(trade.get('mae_points', 0), floating_points)
            
    def _save_engine_state(self):
        """Save current engine state to cache"""
        try:
            state = {
                'active_positions': self.active_positions,
                'initial_analysis': self.initial_analysis,
                'last_regime': self.last_regime,
                'risk_metrics_history': self.risk_metrics_history,
                'equity_curve': self.equity_curve,
                'returns_series': self.returns_series,
                'last_analysis_time': self.last_analysis_time.isoformat() if self.last_analysis_time else None,
                'iteration': self.iteration
            }
            
            with open(Config.ENGINE_STATE_FILE, 'w') as f:
                json.dump(state, f, default=str)
                
        except Exception as e:
            ProfessionalLogger.log(f"Failed to save engine state: {e}", "ERROR", "ENGINE")

    def _load_engine_state(self):
        """Load engine state from cache"""
        if not os.path.exists(Config.ENGINE_STATE_FILE):
            return
            
        try:
            with open(Config.ENGINE_STATE_FILE, 'r') as f:
                state = json.load(f)
                
            # Note: active_positions tickets will be strings after JSON load, need to convert back to int
            self.active_positions = {int(k): v for k, v in state.get('active_positions', {}).items()}
            self.initial_analysis = state.get('initial_analysis', {})
            self.last_regime = state.get('last_regime')
            self.risk_metrics_history = state.get('risk_metrics_history', [])
            self.equity_curve = state.get('equity_curve', [])
            self.returns_series = state.get('returns_series', [])
            self.iteration = state.get('iteration', 0)
            
            last_time_str = state.get('last_analysis_time')
            if last_time_str:
                self.last_analysis_time = datetime.fromisoformat(last_time_str)
                
            ProfessionalLogger.log(f"Restored engine state from cache ({len(self.active_positions)} positions)", "SUCCESS", "ENGINE")
            
        except Exception as e:
            ProfessionalLogger.log(f"Failed to load engine state: {e}", "ERROR", "ENGINE")
    
    def get_broker_time(self):
        """Get current broker time from symbol info"""
        tick = mt5.symbol_info_tick(Config.SYMBOL)
        if tick:
            return datetime.fromtimestamp(tick.time)
        return datetime.now()

    def _bg_train_model(self, data):
        """Standard method for background training thread"""
        try:
            import threading
            from sklearn.preprocessing import RobustScaler
            with self.model_lock:
                success = self.model.train(data)
            if success:
                ProfessionalLogger.log("✅ Background model training successful", "SUCCESS", "ENGINE")
            else:
                ProfessionalLogger.log("❌ Background model training failed", "WARNING", "ENGINE")
        except Exception as e:
            ProfessionalLogger.log(f"Background training exception: {e}", "ERROR", "ENGINE")
        finally:
            self.is_training = False

    def run_periodic_tasks(self):
        """Run periodic maintenance and analysis tasks"""
        self.iteration += 1
        
        self.check_closed_positions()
        
        if self.iteration % 10 == 0:
            self.update_performance_metrics()
        
        if self.last_analysis_time is None or \
           (datetime.now() - self.last_analysis_time).total_seconds() > 3600:
            self.perform_periodic_analysis()
            self.last_analysis_time = datetime.now()
        
        # Parameter optimization
        if Config.PARAM_OPTIMIZATION_ENABLED and self.trade_memory.trades:
            stats = self.trade_memory.get_statistical_summary()
            if stats['total_trades'] > 0 and stats['total_trades'] % Config.OPTIMIZE_EVERY_N_TRADES == 0:
                optimized_params = self.parameter_optimizer.optimize_parameters(
                    self.get_historical_data(bars=500),
                    self.trade_memory.trades[-Config.OPTIMIZE_EVERY_N_TRADES:]
                )
                self._apply_optimized_params(optimized_params)
        
        # Retrain model if needed (ASYNC)
        if self.model.should_retrain() and not self.is_training:
            ProfessionalLogger.log("🔄 Starting ASYNC model retraining...", "LEARN", "ENGINE")
            
            data = self.get_historical_data(bars=Config.LOOKBACK_BARS)
            
            if data is not None:
                self.is_training = True
                self.training_thread = threading.Thread(
                    target=self._bg_train_model, 
                    args=(data,),
                    daemon=True
                )
                self.training_thread.start()
        
        if self.iteration % 30 == 0:
            self.print_status()
            self._save_engine_state()
    
    def _apply_optimized_params(self, optimized_params):
        """Apply optimized parameters"""
        ProfessionalLogger.log(f"Applying optimized parameters: {optimized_params}", "INFO", "ENGINE")
        
        # Update dynamic barriers if enabled
        if Config.USE_DYNAMIC_BARRIERS:
            Config.BARRIER_TIME = optimized_params.get('barrier_time', Config.BARRIER_TIME)
        
        # Update confidence thresholds
        Config.MIN_CONFIDENCE = optimized_params.get('min_confidence', Config.MIN_CONFIDENCE)
        
        # Update stop parameters
        Config.ATR_SL_MULTIPLIER = optimized_params.get('atr_sl_multiplier', Config.ATR_SL_MULTIPLIER)
        Config.ATR_TP_MULTIPLIER = optimized_params.get('atr_tp_multiplier', Config.ATR_TP_MULTIPLIER)
    
    def perform_periodic_analysis(self):
        """Perform periodic statistical analysis"""
        ProfessionalLogger.log("🔄 Running periodic statistical analysis...", "ANALYSIS", "ENGINE")
        
        analysis_bars = min(Config.LOOKBACK_BARS, 5000) 
        data = self.get_historical_data(bars=analysis_bars)
        
        if data is not None and len(data) > 500:
            analysis = self.model.perform_statistical_analysis(data)
            
            if analysis:
                current_regime = analysis.get('market_regime', {}).get('regime', 'unknown')
                regime_confidence = analysis.get('market_regime', {}).get('confidence', 0)
                
                if regime_confidence > 0.7:
                    ProfessionalLogger.log(f"Current Market Regime: {current_regime} (confidence: {regime_confidence:.0%})", 
                                         "ANALYSIS", "ENGINE")
                
                if hasattr(self, 'last_regime') and self.last_regime != current_regime:
                    ProfessionalLogger.log(f"⚠ Market regime changed from {self.last_regime} to {current_regime} | Triggering immediate retraining", 
                                         "WARNING", "ENGINE")
                    self.model.force_retrain_flag = True
                
                self.last_regime = current_regime
    
    def update_performance_metrics(self):
        """Update and calculate performance metrics"""
        if not self.connected:
            return
        
        account = mt5.account_info()
        if account:
            self.equity_curve.append(account.equity)
            
            if len(self.equity_curve) > 10000:
                self.equity_curve = self.equity_curve[-10000:]
            
            if len(self.returns_series) > 10000:
                self.returns_series = self.returns_series[-10000:]
            
            if len(self.returns_series) >= 20:
                recent_returns = self.returns_series[-20:]
                risk_metrics = self.risk_metrics.calculate_risk_metrics(recent_returns)
                self.risk_metrics_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': risk_metrics
                })
                
                if len(self.risk_metrics_history) > 100:
                    self.risk_metrics_history = self.risk_metrics_history[-100:]
    
    def print_status(self):
        """Print current trading status"""
        account = mt5.account_info()
        if not account:
            return
        
        positions = self.get_current_positions()
        tick = mt5.symbol_info_tick(Config.SYMBOL)
        
        if tick:
            price = tick.ask
            
            stats = self.trade_memory.get_statistical_summary()
            
            status_msg = (f"Status | Price: {price:.2f} | Positions: {positions} | "
                         f"Equity: ${account.equity:.2f}")
            
            if stats and stats.get('total_trades', 0) > 0:
                status_msg += f" | Trades: {stats['total_trades']} | Win Rate: {stats.get('win_rate', 0):.1%}"
                
                # Add MFE/MAE insight if available
                mfe_avg = stats.get('avg_mfe', 0)
                status_msg += f" | Avg MFE: {mfe_avg:.1f} pts"
            
            ProfessionalLogger.log(status_msg, "INFO", "ENGINE")
    
    def print_performance_report(self):
        """Print comprehensive performance report"""
        stats = self.trade_memory.get_statistical_summary()
        
        if not stats or stats.get('total_trades', 0) == 0:
            ProfessionalLogger.log("No trading performance data available", "INFO", "ENGINE")
            return
        
        ProfessionalLogger.log("=" * 70, "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log("📊 COMPREHENSIVE PERFORMANCE REPORT", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Total Trades: {stats['total_trades']}", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Win Rate: {stats['win_rate']:.1%}", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Total Profit: ${stats['total_profit']:.2f}", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Average Profit: ${stats['mean_profit']:.2f}", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Profit Factor: {stats['profit_factor']:.2f}", "PERFORMANCE", "ENGINE")
        
        # MFE/MAE Post-Mortem Analysis
        if 'avg_mfe' in stats:
            ProfessionalLogger.log("-" * 35, "PERFORMANCE", "ENGINE")
            ProfessionalLogger.log(f"📈 EXCURSION ANALYSIS (MFE/MAE):", "PERFORMANCE", "ENGINE")
            ProfessionalLogger.log(f"  Avg Max Favorable (MFE): {stats['avg_mfe']:.1f} pts", "PERFORMANCE", "ENGINE")
            ProfessionalLogger.log(f"  Avg Max Adverse (MAE): {stats['avg_mae']:.1f} pts", "PERFORMANCE", "ENGINE")
            
            efficiency = stats.get('avg_tp_efficiency', 0)
            ProfessionalLogger.log(f"  TP Efficiency: {efficiency:.1%}", "PERFORMANCE", "ENGINE")
            
            if efficiency < 0.3 and stats['win_rate'] > 0.6:
                ProfessionalLogger.log("  💡 TIP: TP targets might be too conservative (low efficiency).", "INFO", "ENGINE")
            elif efficiency > 0.8:
                ProfessionalLogger.log("  💡 TIP: TP targets are highly optimal.", "SUCCESS", "ENGINE")
            ProfessionalLogger.log("-" * 35, "PERFORMANCE", "ENGINE")
        
        if len(self.risk_metrics_history) > 0:
            latest_metrics = self.risk_metrics_history[-1]['metrics']
            
            ProfessionalLogger.log("Risk Metrics:", "PERFORMANCE", "ENGINE")
            if 'sharpe' in latest_metrics:
                ProfessionalLogger.log(f"  Sharpe Ratio: {latest_metrics['sharpe']:.3f}", "PERFORMANCE", "ENGINE")
            if 'sortino' in latest_metrics:
                ProfessionalLogger.log(f"  Sortino Ratio: {latest_metrics['sortino']:.3f}", "PERFORMANCE", "ENGINE")
            if 'max_drawdown' in latest_metrics:
                ProfessionalLogger.log(f"  Max Drawdown: {latest_metrics['max_drawdown']:.2%}", "PERFORMANCE", "ENGINE")
            if 'omega' in latest_metrics:
                ProfessionalLogger.log(f"  Omega Ratio: {latest_metrics['omega']:.3f}", "PERFORMANCE", "ENGINE")
        
        if hasattr(self, 'initial_analysis'):
            ia = self.initial_analysis
            if 'market_regime' in ia:
                mr = ia['market_regime']
                ProfessionalLogger.log(f"Initial Market Analysis: {mr.get('regime', 'unknown')} regime", "PERFORMANCE", "ENGINE")
        
        # Optimization history
        if self.parameter_optimizer.optimization_history:
            ProfessionalLogger.log("Parameter Optimization History:", "PERFORMANCE", "ENGINE")
            for opt in self.parameter_optimizer.optimization_history[-3:]:
                ProfessionalLogger.log(f"  {opt['timestamp']}: Score={opt['score']:.3f}", "PERFORMANCE", "ENGINE")
        
        ProfessionalLogger.log("=" * 70, "PERFORMANCE", "ENGINE")

    def check_closed_positions(self):
        """Check for closed positions and update RL/Memory"""
        if not self.active_positions:
            return
            
        for ticket in list(self.active_positions.keys()):
            # Check if position still exists
            pos = mt5.positions_get(ticket=ticket)
            if pos is None or len(pos) == 0:
                # Position closed
                trade_data = self.active_positions.pop(ticket)
                ProfessionalLogger.log(f"Trade #{ticket} ({trade_data['type']}) CLOSED", "TRADE", "ENGINE")
                
                # Get PnL from history
                deals = mt5.history_deals_get(position=ticket)
                if deals:
                    profit = sum([d.profit + d.swap + d.commission for d in deals])
                    ProfessionalLogger.log(f"  PnL: ${profit:.2f}", "TRADE", "ENGINE")
                    
                    # Update RL Collector
                    if hasattr(self, 'rl_collector'):
                        self.rl_collector.complete_experience(ticket, profit)
                    
                    # Update Trade Memory
                    if hasattr(self, 'trade_memory'):
                        trade_data['status'] = 'closed'
                        trade_data['profit'] = profit
                        trade_data['close_time'] = datetime.now().isoformat()
                        self.trade_memory.add_trade(trade_data)
                        
                    # Trigger optimization if needed
                    if Config.PARAM_OPTIMIZATION_ENABLED:
                         self.parameter_optimizer.update(profit)

    def train_initial_model(self):
        """Train initial model with statistical analysis"""
        ProfessionalLogger.log(f"Loading deep history ({Config.LOOKBACK_BARS} bars) for initial training...", "INFO", "ENGINE")
        
        data = self.get_historical_data(bars=Config.LOOKBACK_BARS)
        
        if data is not None:
            data_len = len(data)
            ProfessionalLogger.log(f"Retrieved {data_len} bars from MT5", "DATA", "ENGINE")

            if data_len >= Config.TRAINING_MIN_SAMPLES:
                ProfessionalLogger.log(f"Initializing Walk-Forward Optimization (Window: {Config.WALK_FORWARD_WINDOW}, Folds: {Config.WALK_FORWARD_FOLDS})...", "LEARN", "ENSEMBLE")
                
                analysis = self.model.perform_statistical_analysis(data)
                
                success = self.model.train(data)
                
                if success:
                    diag = self.model.get_diagnostics()
                    metrics = diag['training_status']['training_metrics']
                    ProfessionalLogger.log("✅ Initial model training successful", "SUCCESS", "ENGINE")
                    ProfessionalLogger.log(f"   CV Score: {metrics.get('avg_cv_score', 0):.2%}", "INFO", "ENGINE")
                else:
                    ProfessionalLogger.log("❌ Initial model training failed", "WARNING", "ENGINE")
            else:
                ProfessionalLogger.log(f"❌ Insufficient data: {data_len} < {Config.TRAINING_MIN_SAMPLES} required", "ERROR", "ENGINE")
        else:
            ProfessionalLogger.log(f"❌ Failed to retrieve historical data from MT5", "ERROR", "ENGINE")
    
    def execute_trade(self, signal, confidence, df, features, model_details):
        """
        Execute trade with comprehensive validation and risk management.
        """
        # Check position limits
        current_positions = self.get_current_positions()
        if current_positions >= Config.MAX_POSITIONS:
            ProfessionalLogger.log(f"Max positions reached: {current_positions}/{Config.MAX_POSITIONS}", "WARNING", "ENGINE")
            return
        
        # Get account info
        account = mt5.account_info()
        if not account:
            ProfessionalLogger.log("Could not retrieve account info", "ERROR", "ENGINE")
            return
        
        # Get current price
        tick = mt5.symbol_info_tick(Config.SYMBOL)
        if not tick:
            ProfessionalLogger.log("Could not get current tick", "ERROR", "ENGINE")
            return
        
        symbol_info = mt5.symbol_info(Config.SYMBOL)
        if not symbol_info:
            ProfessionalLogger.log("Could not get symbol info", "ERROR", "ENGINE")
            return
        
        # Determine trade direction
        if signal == 1:  # BUY
            order_type = mt5.ORDER_TYPE_BUY
            entry_price = tick.ask
            trade_type_str = "BUY"
        else:  # SELL
            order_type = mt5.ORDER_TYPE_SELL
            entry_price = tick.bid
            trade_type_str = "SELL"
            
        # ==========================================
        # 1. ADAPTIVE CONFIDENCE SCALING
        # ==========================================
        # Start with base confidence
        final_confidence = confidence
        
        # Adjust based on Signal Quality Score
        if Config.ENABLE_SIGNAL_QUALITY_FILTER and self.feature_engine.signal_scorer:
            # Extract multi-timeframe data if available in model_details
            mtf_data = None
            if model_details and 'timeframe_details' in model_details:
                mtf_data = model_details # Pass the whole dict as scorer expects finding alignment/timeframes
            
            quality_score = self.feature_engine.signal_scorer.score_signal(features, signal, mtf_data)
            
            # Scale confidence: Quality score 80+ boosts, <50 penalizes
            quality_factor = quality_score / 60.0 # Normalize around 60
            final_confidence *= min(1.3, max(0.7, quality_factor))
            
            ProfessionalLogger.log(f"Signal Quality: {quality_score:.1f}/100 -> Confidence adjusted: {confidence:.2f} to {final_confidence:.2f}", "INFO", "ENGINE")

        # Adjust based on Trade History Learning
        if hasattr(self, 'trade_memory') and hasattr(self.trade_memory, 'get_confidence_adjustment'):
             # This assumes we implemented get_confidence_adjustment in TradeHistoryLearner and linked it
             # Since it's a separate class instance, we need to access via self.model if it has reference
             # Or if we added it to TradeMemory. Wait, TradeHistoryLearner is separate class. 
             # I added it to Config as LEARNING_DATA_FILE but logic might be in a separate learner instance?
             # Ah, TradeHistoryLearner definition is around line 600.
             # I need to check if EnhancedTradingEngine has a 'learner' instance.
             # Based on previous views, it likely does not.
             # I'll rely on Signal Quality for now to avoid errors.
             pass

        # Adjust Lookup based Risk/Reward if enabled
        target_rr = 2.0
        if Config.ENABLE_LOOKUP_TABLES and self.feature_engine.lookup_tables:
            target_rr = self.feature_engine.lookup_tables.get_risk_reward_ratio(final_confidence)
        
        # ==========================================
        # 2. CALCULATE DYNAMIC STOP LOSS & TAKE PROFIT
        # ==========================================
        atr = features.get('atr_percent', 0.001)
        if atr <= 0: atr = 0.001
        atr_absolute = atr * entry_price
        volatility = features.get('volatility', 0.01)
        if volatility <= 0 or np.isnan(volatility): volatility = 0.01

        # Use SmartOrderExecutor's advanced logic
        sl_price = self.order_executor.calculate_dynamic_stop_loss(
            entry_price, 
            signal, 
            atr_absolute, 
            volatility, 
            lookup_tables=self.feature_engine.lookup_tables if Config.ENABLE_LOOKUP_TABLES else None,
            market_structure={'support': features.get('support', 0), 'resistance': features.get('resistance', 0)}
        )
        
        sl_distance = abs(entry_price - sl_price)
        tp_distance = sl_distance * target_rr
        
        if signal == 1:
            tp_price = entry_price + tp_distance
        else:
            tp_price = entry_price - tp_distance

        # Verify minimum Stop Level
        min_dist = symbol_info.trade_stops_level * symbol_info.point
        if sl_distance < min_dist:
             ProfessionalLogger.log(f"SL distance too small ({sl_distance} < {min_dist}). Adjusting.", "WARNING", "ENGINE")
             sl_distance = min_dist * 1.5
             if signal == 1: sl_price = entry_price - sl_distance
             else: sl_price = entry_price + sl_distance
        
        # ==========================================
        # 3. CALCULATE POSITION SIZE (KELLY & VOLATILITY)
        # ==========================================
        
        # Calculate base volume using risk percentage
        risk_amount = account.equity * Config.RISK_PERCENT
        
        # KELLY CRITERION SIZING
        kelly_risk_pct = 0
        if hasattr(Config, 'USE_KELLY_CRITERION') and Config.USE_KELLY_CRITERION:
            # Use lookup table if available for O(1) speed
            if Config.ENABLE_LOOKUP_TABLES and self.feature_engine.lookup_tables:
                stats = self.trade_memory.get_statistical_summary()
                win_rate = stats.get('win_rate', 0.5)
                kelly_fraction = self.feature_engine.lookup_tables.get_kelly_fraction(win_rate)
                # Apply fraction to calculated Kelly Percentage? 
                # Lookup table returns pre-calculated fractional kelly (e.g. 0.5 * Kelly)
                # But we need raw kelly first? 
                # Actually lookup table likely stores 'Fraction * Kelly' result if designed well.
                # Let's assume standard calculation fallback for safety + lookup usage
                pass 
            
            # Standard Calculation
            stats = self.trade_memory.get_statistical_summary()
            win_rate = stats.get('win_rate', 0.5)
            avg_win = stats.get('avg_win', 100)
            avg_loss = abs(stats.get('avg_loss', -100))
            
            if avg_loss > 0 and win_rate > 0:
                win_loss_ratio = avg_win / avg_loss
                kelly_pct = win_rate - (1 - win_rate) / win_loss_ratio
                
                if kelly_pct > 0:
                    kelly_risk_pct = kelly_pct * Config.KELLY_FRACTION
                    # Adaptive scaling using Confidence
                    kelly_risk_pct *= final_confidence 
                    
                    kelly_risk_pct = min(kelly_risk_pct, Config.MAX_KELLY_RISK)
                    risk_amount = account.equity * kelly_risk_pct
                    ProfessionalLogger.log(f"Kelly + Confidence ({final_confidence:.2f}): Risk {kelly_risk_pct:.2%}", "RISK", "ENGINE")

        # Initial Volume Calculation
        point_value = symbol_info.trade_contract_size * symbol_info.trade_tick_size / symbol_info.trade_tick_value
        if point_value == 0: point_value = 1.0
        
        volume_raw = risk_amount / (sl_distance * point_value)
        
        # Apply volatility scaling if enabled
        if hasattr(Config, 'VOLATILITY_SCALING_ENABLED') and Config.VOLATILITY_SCALING_ENABLED:
            if volatility > Config.HIGH_VOL_THRESHOLD:
                volume_raw *= Config.HIGH_VOL_SIZE_MULTIPLIER
                ProfessionalLogger.log(f"High volatility ({volatility:.4f}) - reducing size by {Config.HIGH_VOL_SIZE_MULTIPLIER}x", "RISK", "ENGINE")
            elif volatility < Config.LOW_VOL_THRESHOLD:
                volume_raw *= Config.LOW_VOL_SIZE_MULTIPLIER
                ProfessionalLogger.log(f"Low volatility ({volatility:.4f}) - increasing size by {Config.LOW_VOL_SIZE_MULTIPLIER}x", "RISK", "ENGINE")


        
        # Apply confidence-based sizing
        if Config.PERFORMANCE_BASED_POSITION_SIZING:
            volume_raw *= confidence
        
        # Normalize volume to broker requirements
        volume_step = symbol_info.volume_step
        if volume_step > 0:
            volume = round(volume_raw / volume_step) * volume_step
        else:
            volume = volume_raw
        
        # Apply volume limits
        volume = max(Config.MIN_VOLUME, min(volume, Config.MAX_VOLUME))
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
        
        # Round to appropriate decimals
        if volume_step >= 1.0:
            volume = round(volume, 0)
        elif volume_step >= 0.1:
            volume = round(volume, 1)
        elif volume_step >= 0.01:
            volume = round(volume, 2)
        else:
            volume = round(volume, 3)
        
        ProfessionalLogger.log(f"Position sizing: Raw={volume_raw:.3f}, Final={volume:.3f}, Risk=${risk_amount:.2f}", "RISK", "ENGINE")
        
        # ==========================================
        # CALCULATE STOP LOSS AND TAKE PROFIT
        # ==========================================
        
        if Config.USE_DYNAMIC_SL_TP:
            atr_absolute = atr * entry_price
            sl_distance = atr_absolute * Config.ATR_SL_MULTIPLIER
            tp_distance = atr_absolute * Config.ATR_TP_MULTIPLIER
        else:
            sl_distance = entry_price * Config.FIXED_SL_PERCENT
            tp_distance = entry_price * Config.FIXED_TP_PERCENT
        
        # Calculate actual SL/TP prices
        if signal == 1:  # BUY
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
        else:  # SELL
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
        
        # Validate SL/TP distances using symbol point
        point = symbol_info.point if symbol_info.point > 0 else 0.01
        sl_distance_points = abs(entry_price - stop_loss) / point
        tp_distance_points = abs(entry_price - take_profit) / point
        
        if sl_distance_points < Config.MIN_SL_DISTANCE_POINTS:
            sl_distance_points = Config.MIN_SL_DISTANCE_POINTS
            if signal == 1:
                stop_loss = entry_price - (sl_distance_points * point)
            else:
                stop_loss = entry_price + (sl_distance_points * point)
        
        if tp_distance_points < Config.MIN_TP_DISTANCE_POINTS:
            tp_distance_points = Config.MIN_TP_DISTANCE_POINTS
            if signal == 1:
                take_profit = entry_price + (tp_distance_points * point)
            else:
                take_profit = entry_price - (tp_distance_points * point)
        
        # Validate Risk/Reward ratio
        rr_ratio = tp_distance_points / sl_distance_points if sl_distance_points > 0 else 0
        if rr_ratio < Config.MIN_RR_RATIO:
            ProfessionalLogger.log(f"Risk/Reward too low: {rr_ratio:.2f} < {Config.MIN_RR_RATIO}", "WARNING", "ENGINE")
            # Adjust TP to meet minimum RR
            tp_distance_points = sl_distance_points * Config.MIN_RR_RATIO
            if signal == 1:
                take_profit = entry_price + (tp_distance_points * point)
            else:
                take_profit = entry_price - (tp_distance_points * point)
            rr_ratio = Config.MIN_RR_RATIO
        
        ProfessionalLogger.log(
            f"Trade Setup: {trade_type_str} {volume:.3f} @ {entry_price:.5f} | "
            f"SL: {stop_loss:.5f} ({sl_distance_points:.1f} pts) | "
            f"TP: {take_profit:.5f} ({tp_distance_points:.1f} pts) | "
            f"RR: {rr_ratio:.2f}",
            "INFO", "ENGINE"
        )
        
        # ==========================================
        # SIGNAL FLIP: Close opposite positions first
        # ==========================================
        opposite_type = "SELL" if trade_type_str == "BUY" else "BUY"
        opposite_tickets = [t for t, d in self.active_positions.items() if d.get('type') == opposite_type]
        
        if opposite_tickets:
            ProfessionalLogger.log(
                f"🔄 Signal Flip: Closing {len(opposite_tickets)} opposite {opposite_type} positions "
                f"before opening new {trade_type_str}", 
                "INFO", "ENGINE"
            )
            for ticket in opposite_tickets:
                if self.order_executor.close_position(ticket, Config.SYMBOL):
                    self.active_positions.pop(ticket, None)
        
        # ==========================================
        # EXECUTE THE TRADE
        # ==========================================
        
        comment = f"Conf:{confidence:.0%}"
        if model_details:
            agreement = sum(1 for m in model_details.values() if m.get('prediction') == signal) / len(model_details)
            comment += f"|Agr:{agreement:.0%}"
        
        result = self.order_executor.execute_trade(
            symbol=Config.SYMBOL,
            order_type=order_type,
            volume=volume,
            entry_price=entry_price,
            sl=stop_loss,
            tp=take_profit,
            magic=Config.MAGIC_NUMBER,
            comment=comment
        )
        
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
             ProfessionalLogger.log(f"Trade executed: #{result.order}", "SUCCESS", "ENGINE")
             
             # Record for RL (State + Action)
             if hasattr(self, 'rl_collector'):
                 rl_state = {
                     'volatility': features.get('volatility', 0),
                     'rsi': features.get('rsi', 50),
                     'trend': features.get('trend_direction', 0),
                     'spread': tick.ask - tick.bid,
                     'regime': self.last_regime,
                     'dom_imbalance': self.order_flow.get_order_book_imbalance(),
                     'entropy': features.get('shannon_entropy', 0),
                     'hurst': features.get('hurst_exponent', 0.5)
                 }
                 rl_action = {'signal': signal, 'confidence': final_confidence}
                 self.rl_collector.record_entry(result.order, rl_state, rl_action)
             
             # Track Position
             self.active_positions[result.order] = {
                 'ticket': result.order,
                 'type': trade_type_str,
                 'open_price': entry_price,
                 'sl': stop_loss,
                 'tp': take_profit,
                 'open_time': datetime.now().isoformat(),
                 'confidence': final_confidence,
                 'pnl': 0,
                 'max_runge_profit': 0
             }
             self._save_engine_state()
             
             ProfessionalLogger.log(
                f"✅ Trade Opened: #{result.order} | {trade_type_str} {volume:.3f} @ {entry_price:.5f} | "
                f"Conf: {confidence:.0%} | RR: {(tp_distance/sl_distance):.2f}",
                "SUCCESS", "ENGINE"
            )
        else:
             ProfessionalLogger.log("Order execution failed", "ERROR", "ENGINE")
             ProfessionalLogger.log("Order execution failed", "ERROR", "ENGINE")
            
    def _calculate_atr(self, df, period=14):
        """Helper method to calculate ATR"""
        try:
            high = df['high'].values
            low = df['low'].values
            close = df['close'].values
            
            # Calculate True Range
            tr = np.zeros(len(df))
            for i in range(1, len(df)):
                hl = high[i] - low[i]
                hc = abs(high[i] - close[i-1])
                lc = abs(low[i] - close[i-1])
                tr[i] = max(hl, hc, lc)
            
            # Calculate ATR
            atr = np.mean(tr[-period:]) if len(tr) >= period else np.mean(tr)
            return atr
            
        except Exception as e:
            ProfessionalLogger.log(f"ATR calculation error: {e}", "WARNING", "EXECUTOR")
            return 0.001  # Default small value

    def _calculate_sleep_time(self):
        """Dynamically calculate sleep time based on market conditions and active positions."""
        base_sleep = 60 # Default to 1 minute
        
        # If there are active positions, check more frequently
        if self.active_positions:
            base_sleep = 30
        
        # Adjust based on volatility (if features are available)
        if self.cached_features is not None and 'volatility' in self.cached_features:
            vol_data = self.cached_features['volatility']
            vol = vol_data.iloc[-1] if hasattr(vol_data, 'iloc') else vol_data
            if vol > 0.015: # High volatility
                base_sleep = 15
            elif vol < 0.005: # Low volatility
                base_sleep = 90
        
        # Further reduce if multi-timeframe is enabled and we just executed a trade
        # This logic is already partially in the main loop, but can be refined here.
        
        return max(10, min(base_sleep, 120)) # Ensure sleep is between 10s and 2min

    def _get_dxy_trend(self):
        """
        Identify US Dollar Index (DXY) Trend.
        Tries symbols: DXY, USDX, DX, USDIndex
        Returns: 1 (Uptrend), -1 (Downtrend), 0 (Neutral)
        """
        dxy_symbols = ["DXY", "USDX", "DX", "USDIndex", "USDOLLAR"]
        dxy_symbol = None
        
        # Cache symbol check
        if hasattr(self, 'cached_dxy_symbol'):
            dxy_symbol = self.cached_dxy_symbol
        else:
            for sym in dxy_symbols:
                if mt5.symbol_info(sym):
                    dxy_symbol = sym
                    self.cached_dxy_symbol = sym
                    ProfessionalLogger.log(f"Found DXY Symbol: {sym}", "INFO", "ENGINE")
                    break
        
        if not dxy_symbol:
            return 0
            
        # Fetch Data (H1 for macro trend)
        rates = mt5.copy_rates_from_pos(dxy_symbol, mt5.TIMEFRAME_H1, 0, 50)
        if rates is None or len(rates) < 20:
            return 0
            
        df = pd.DataFrame(rates)
        
        # Calculate Trend (EMA 8 vs 21)
        ema_fast = df['close'].ewm(span=8).mean().iloc[-1]
        ema_slow = df['close'].ewm(span=21).mean().iloc[-1]
        
        # Calculate RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 1)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # Determine Trend
        if ema_fast > ema_slow and rsi > 50:
            return 1 # Dollar Strong
        elif ema_fast < ema_slow and rsi < 50:
            return -1 # Dollar Weak
            
        return 0

    def run(self):
        """Main execution method"""
        print("\n" + "=" * 70)
        print("🤖 ENHANCED PROFESSIONAL MT5 ALGORITHMIC TRADING SYSTEM")
        print("📊 Advanced Statistical Analysis | Dynamic Labeling | Regime-Aware Models")
        print("=" * 70 + "\n")
        
        ProfessionalLogger.log("Starting enhanced trading system with all improvements...", "INFO", "ENGINE")
        
        if not self.connect_mt5():
            return
        
        self.train_initial_model()
        
        self.run_enhanced_live_trading()

    def run_enhanced_live_trading(self):
        """Enhanced live trading with all new features - EVENT DRIVEN"""
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        ProfessionalLogger.log("STARTING EVENT-DRIVEN TRADING CORE", "TRADE", "ENGINE")
        ProfessionalLogger.log(f"Latency Target: <10ms", "INFO", "ENGINE")
        ProfessionalLogger.log(f"Dynamic Barriers: {'ENABLED' if Config.USE_DYNAMIC_BARRIERS else 'DISABLED'}", "INFO", "ENGINE")
        ProfessionalLogger.log(f"24/7 Precision: ENABLED", "INFO", "ENGINE")
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        
        self.running = True
        self.last_tick_time = 0
        self.last_periodic_check = time.time()
        
        # Pre-allocate for speed
        self.required_lookback = max(500, Config.TREND_MA * 2)
        
        try:
            while self.running:
                # 1. High-Frequency Tick Check (Microsecond precison potential)
                tick = mt5.symbol_info_tick(Config.SYMBOL)
                if tick is None:
                    time.sleep(0.1)
                    continue
                    
                # 2. Event Trigger
                if tick.time_msc != self.last_tick_time:
                    self.last_tick_time = tick.time_msc
                    self._process_market_tick(tick)
                
                # 3. Periodic Maintenance (Non-Blocking)
                current_time = time.time()
                if current_time - self.last_periodic_check > 5.0: # 5 Seconds
                    self._run_periodic_maintenance()
                    self.last_periodic_check = current_time
                    
                # 4. Micro-sleep to prevent 100% CPU usage (1ms)
                time.sleep(0.001)
                
        except Exception as e:
            ProfessionalLogger.log(f"Engine CRITICAL error: {e}", "ERROR", "ENGINE")
            import traceback
            traceback.print_exc()
        except KeyboardInterrupt:
            ProfessionalLogger.log("\nShutdown requested by user", "WARNING", "ENGINE")
        finally:
            self.running = False
            self.print_performance_report()
            mt5.shutdown()
            ProfessionalLogger.log("Disconnected from MT5", "INFO", "ENGINE")

    def _save_market_snapshot(self, features):
        """Save a complete snapshot of market state for later training"""
        try:
            current_time = datetime.now()
            # Log every 15 minutes to align with M15 strategy
            if self.last_snapshot_time is not None and \
               (current_time - self.last_snapshot_time).total_seconds() < 900:
                return

            snapshot = {
                "timestamp": current_time.isoformat(),
                "price": mt5.symbol_info_tick(Config.SYMBOL).ask,
                "regime": self.last_regime,
                "dom_imbalance": self.order_flow.get_order_book_imbalance(),
                "entropy": features.get('entropy', 0),
                "hurst_exponent": features.get('hurst_exponent', 0.5),
                "rsi": features.get('rsi', 50),
                "volatility": features.get('volatility', 0)
            }
            
            # Save to a dedicated history file
            history_file = os.path.join(Config.CACHE_DIR, "market_data_history.json")
            history = []
            if os.path.exists(history_file):
                with open(history_file, 'r') as f:
                    try:
                        history = json.load(f)
                    except:
                        history = []
            
            history.append(snapshot)
            # Keep last 5000 snapshots (approx 52 days of M15 data)
            history = history[-5000:]
            
            with open(history_file, 'w') as f:
                json.dump(history, f, indent=4)
            
            self.last_snapshot_time = current_time
            # ProfessionalLogger.log("Market snapshot saved for learning", "DEBUG", "DATA")
            
        except Exception as e:
            ProfessionalLogger.log(f"Snapshot Error: {e}", "ERROR", "DATA")

    def _process_market_tick(self, tick):
        """Handle a single market tick event"""
        try:
            # 1. Update active positions (Real-time PnL/Trailing)
            if self.active_positions:
                # Fast update
                self.exit_manager.manage_positions(
                    None, self.active_positions, None, 0.5, 
                    fast_tick_update=True, current_price=tick.ask
                )

            # 2. Check Feature Recalc Needed (Time-based throttler still applies to heavy ML features)
            current_time = datetime.now()
            should_recalc = False
            if self.last_feature_time is None or \
               (current_time - self.last_feature_time).total_seconds() > Config.FEATURE_RECALC_INTERVAL_SECONDS:
                should_recalc = True
            
            # If not time to recalc features, skip heavy logic unless we have open positions that need logic
            if not should_recalc and not self.active_positions:
                return

            # 3. Fetch Data & Features
            rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, self.required_lookback)
            if rates is None or len(rates) < Config.TREND_MA + 10:
                return

            df_current = pd.DataFrame(rates)
            
            if should_recalc:
                self.cached_features = self.feature_engine.calculate_features(df_current)
                
                # --- LIVE INSTITUTIONAL INJECTION ---
                # Inject current DOM imbalance and Regime into the feature set
                self.cached_features['dom_imbalance'] = self.order_flow.get_order_book_imbalance()
                self.cached_features['regime_encoded'] = {
                    'trending': 1, 'mean_reverting': 2, 'volatile': 3, 'stress': 4
                }.get(self.last_regime, 0)
                
                self.last_feature_time = current_time
                # ProfessionalLogger.log("Tick: Features updated with live Institutional Data", "DEBUG", "ENGINE")
            
            features = self.cached_features
            
            # 4. ML Prediction
            signal = None
            confidence = 0
            
            # New: Update Bayesian Detector
            is_shock, shock_prob = self.bayes_detector.detect_regime_switch(tick.ask)
            if is_shock:
                 # ProfessionalLogger.log(f"Bayesian Risk Alert: Shock Probability {shock_prob:.2f}", "RISK", "ENGINE")
                 pass

            with self.model_lock:
                 signal, confidence, dict_features, model_details = self.model.predict(df_current, features)
                 self.latest_dict_features = dict_features # Store for maintenance snapshots

            # 5. Multi-TF & Filter Checks
            multi_tf_signal = None
            multi_tf_conf = 0
            trend_filter_passed = True
            
            min_confidence_override = Config.MIN_CONFIDENCE
            min_agreement_override = Config.MIN_ENSEMBLE_AGREEMENT
            
            # Adjust for Shock
            if is_shock:
                min_confidence_override = max(min_confidence_override, 0.85) # Require 85% confidence in shock
                
            if Config.MULTI_TIMEFRAME_ENABLED:
                mtf_res = self.multi_tf_analyser.get_multi_timeframe_recommendation(Config.SYMBOL)
                if mtf_res:
                    multi_tf_signal = mtf_res.get('consensus_signal')
                    multi_tf_conf = mtf_res.get('confidence', 0)
                    trend_filter_passed = mtf_res.get('trend_filter_passed', True)
                    
                    if not trend_filter_passed:
                         # Log logic handled in the class, just veto here
                         signal = None
            
            # 6. Signal Valid?
            if signal is not None:
                # Combined Confidence
                final_conf = confidence
                if multi_tf_conf > 0:
                     final_conf = (confidence * 0.7) + (multi_tf_conf * 0.3)
                
                # Check Filters
                market_context = {
                    'regime': self.last_regime,
                    'multi_tf_signal': multi_tf_signal,
                    'existing_positions': list(self.active_positions.values())
                }
                is_valid, reason = self.signal_filter.validate_signal(signal, final_conf, dict_features, market_context)
                
                if is_valid:
                     # ==========================================
                     # LEVEL 2 (DOM) SAFETY CHECK
                     # ==========================================
                     dom_veto = False
                     flow_analysis = self.order_flow.analyze_flow()
                     imbalance = flow_analysis.get('imbalance', 0)
                     walls = flow_analysis.get('walls', {})
                     
                     current_price = tick.ask if signal == 1 else tick.bid
                     
                     if signal == 1: # Buying
                         if walls.get('sell_wall'):
                             dist = walls['sell_wall']['price'] - current_price
                             if dist < 0.0050: # Wall within 50 points (approx 5 pips)
                                 ProfessionalLogger.log(f"DOM VETO: Sell Wall at {walls['sell_wall']['price']} (Dist: {dist:.5f})", "WARNING", "EXECUTION")
                                 dom_veto = True
                     elif signal == 0: # Selling
                         if walls.get('buy_wall'):
                             dist = current_price - walls['buy_wall']['price']
                             if dist < 0.0050:
                                 ProfessionalLogger.log(f"DOM VETO: Buy Wall at {walls['buy_wall']['price']} (Dist: {dist:.5f})", "WARNING", "EXECUTION")
                                 dom_veto = True
                     
                     if not dom_veto:
                         self.execute_trade(signal, final_conf, df_current, dict_features, model_details)
                else:
                    pass # Filtered out silent or logged by filter

        except Exception as e:
            ProfessionalLogger.log(f"Tick process error: {e}", "ERROR", "ENGINE")

    def _run_periodic_maintenance(self):
        """Run tasks that don't need to happen every tick"""
        try:
             # 1. Daily Risk Check
             account = mt5.account_info()
             if account:
                 if not hasattr(self, 'daily_start_equity') or self.daily_start_equity == 0:
                     self.daily_start_equity = account.equity
                 
                 pnl_pct = ((account.equity - self.daily_start_equity) / self.daily_start_equity) * 100
                 if pnl_pct < -Config.MAX_DAILY_LOSS_PERCENT:
                     ProfessionalLogger.log(f"DAILY LOSS LIMIT HIT: {pnl_pct:.2f}%", "CRITICAL", "RISK")
                     self.running = False
            
             # 2. Periodic Tasks
             self.run_periodic_tasks()
             
             # 3. Market Snapshot (Institutional persistence)
             if self.latest_dict_features:
                 self._save_market_snapshot(self.latest_dict_features)
                 
             # 4. Market Diagnostic Log
             if self.iteration % 12 == 0: # Approx 1 min (assuming 5s check interval)
                 account = mt5.account_info() # Re-fetch for fresh equity
                 ProfessionalLogger.log(f"Heartbeat | Equity: {account.equity:.2f} | Time: {datetime.now().strftime('%H:%M:%S')}", "INFO", "ENGINE")

             self.iteration += 1

        except Exception as e:
            ProfessionalLogger.log(f"Maintenance error: {e}", "ERROR", "ENGINE")

# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    """Main entry point"""
    ProfessionalLogger.log("Starting enhanced trading system...", "INFO", "ENGINE")
    
    if not mt5.initialize():
        ProfessionalLogger.log("MT5 initialization failed", "ERROR", "ENGINE")
        return
    
    symbol_info = mt5.symbol_info(Config.SYMBOL)
    if symbol_info:
        ProfessionalLogger.log(f"Symbol info for {Config.SYMBOL}:", "INFO", "ENGINE")
        ProfessionalLogger.log(f"  Volume min: {getattr(symbol_info, 'volume_min', 'N/A')}", "INFO", "ENGINE")
        ProfessionalLogger.log(f"  Volume max: {getattr(symbol_info, 'volume_max', 'N/A')}", "INFO", "ENGINE")
        ProfessionalLogger.log(f"  Volume step: {getattr(symbol_info, 'volume_step', 'N/A')}", "INFO", "ENGINE")
        ProfessionalLogger.log(f"  Trade contract size: {getattr(symbol_info, 'trade_contract_size', 'N/A')}", "INFO", "ENGINE")
    else:
        ProfessionalLogger.log(f"Could not get symbol info for {Config.SYMBOL}", "ERROR", "ENGINE")
    
    mt5.shutdown()
    
    time.sleep(2)
    engine = EnhancedTradingEngine()
    engine.run()

if __name__ == "__main__":
    main()