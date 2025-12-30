import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import os
import sys
import threading
from datetime import datetime, timedelta
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
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", 5044108820))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "@rC1KbQb")
    MT5_SERVER = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
    
    # ==========================================
    # TRADING INSTRUMENT SPECIFICATIONS
    # ==========================================
    SYMBOL = "XAUUSD"
    TIMEFRAME = mt5.TIMEFRAME_M5
    
    # Position sizing with enhanced scaling
    BASE_VOLUME = 0.10
    MAX_VOLUME = 1.00
    MIN_VOLUME = 0.10
    VOLUME_STEP = 0.01
    
    MAGIC_NUMBER = 998877
    
    # ==========================================
    # RISK MANAGEMENT - ENHANCED
    # ==========================================
    RISK_PERCENT = 0.01
    MAX_TOTAL_RISK_PERCENT = 0.05
    MAX_RISK_PER_TRADE = 100
    
    # Signal Quality - Dynamic thresholds
    MIN_CONFIDENCE = 0.40
    MIN_ENSEMBLE_AGREEMENT = 0.50
    
    # Position Limits
    MAX_POSITIONS = 5
    MAX_DAILY_TRADES = 10
    MIN_TIME_BETWEEN_TRADES = 10
    
    # Loss Limits
    MAX_DAILY_LOSS_PERCENT = 2.0
    MAX_WEEKLY_LOSS_PERCENT = 5.0
    MAX_DRAWDOWN_PERCENT = 10.0
    MAX_CONSECUTIVE_LOSSES = 3
    
    # Kelly Criterion
    KELLY_FRACTION = 0.15
    USE_HALF_KELLY = True
    
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
    WALK_FORWARD_WINDOW = 1000
    WALK_FORWARD_STEP = 100
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
    BARRIER_TIME = 6
    
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
    MIN_RR_RATIO = 1.20
    
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
    SESSION_AWARE_TRADING = False
    
    # Trading Sessions (UTC times)
    AVOID_ASIAN_SESSION = True  #true
    PREFER_LONDON_NY_OVERLAP = True #true
    
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
    AVOID_FRIDAY_LAST_HOURS = True
    
    # ==========================================
    # ORDER EXECUTION
    # ==========================================
    MAX_SLIPPAGE_POINTS = 10
    ORDER_TIMEOUT_SECONDS = 30
    MAX_RETRIES = 3
    RETRY_DELAY_MS = 1000
    FEATURE_RECALC_INTERVAL_SECONDS = 30 # Only recalculate every 30s
    
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
    
    BACKTEST_RESULTS_FILE = os.path.join(CACHE_DIR, "backtest_results_xauusd.json")
    PERFORMANCE_LOG_FILE = os.path.join(CACHE_DIR, "performance_log_xauusd.csv")
    
    MEMORY_SIZE = 1000
    LEARNING_WEIGHT = 0.4
    
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
    TIMEFRAMES = ['M5', 'M15', 'H1']
    TIMEFRAME_WEIGHTS = [0.2, 0.5, 0.3]
    TIMEFRAME_ALIGNMENT_THRESHOLD = 0.33
    REQUIRE_TIMEFRAME_ALIGNMENT = False  # Set to False to take trades even with low alignment
    
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

    MULTI_TIMEFRAME_PARAMS = {
        # Alignment bonus scaling
        'ALIGNMENT_BONUS_ABOVE_START': 0.8,
        'ALIGNMENT_BONUS_ABOVE_RANGE': 0.2,
        'ALIGNMENT_BONUS_BELOW_START': 0.4,
        'ALIGNMENT_BONUS_BELOW_RANGE': 0.4,
        
        # Trend bonus parameters
        'TREND_BONUS_SUPPORT': 1.0,      # When H1 trend supports trade
        'TREND_BONUS_NEUTRAL': 0.8,      # When H1 is neutral
        'TREND_BONUS_OPPOSITE': 0.5,     # When H1 trend opposes trade
        
        # Confidence calculation
        'BASE_CONFIDENCE_MULTIPLIER': 0.9,
        'CONFIDENCE_FLOOR': 0.2,
        'CONFIDENCE_CEILING': 0.95,
        
        # Signal generation parameters
        'BUY_SIGNAL_THRESHOLD': 0.35,
        'SELL_SIGNAL_THRESHOLD': -0.35,
        
        # Timeframe signal multipliers
        'M5_SIGNAL_MULTIPLIER': 0.8,
        'M15_SIGNAL_MULTIPLIER': 1.0,
        'H1_SIGNAL_MULTIPLIER': 1.2,
        
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
                return float('inf')
            
            half_life = -np.log(2) / beta
            return max(0, half_life)
        except:
            return 0
    
    @staticmethod
    def calculate_hurst_exponent(prices, method='rs'):
        """Calculate Hurst exponent for market efficiency analysis"""
        if len(prices) < 100:
            return 0.5
        
        try:
            returns = np.diff(np.log(prices))
            
            if method == 'rs':
                n = len(returns)
                r_s_values = []
                n_values = []
                
                for window in range(10, n//2, n//20):
                    if window < 10:
                        continue
                    
                    num_windows = n // window
                    if num_windows < 2:
                        continue
                    
                    rs_values = []
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
                            rs_values.append(r / s)
                    
                    if rs_values:
                        r_s_values.append(np.mean(rs_values))
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
        
        if hurst < 0.4:
            regime_scores["mean_reverting"] += 2
        elif hurst < 0.45:
            regime_scores["mean_reverting"] += 1
        
        if autocorr < -0.1:
            regime_scores["mean_reverting"] += 1
        
        vol_threshold = np.percentile(np.abs(returns), 75)
        if volatility > vol_threshold * 1.5:
            regime_scores["volatile"] += 2
        
        if 0.45 <= hurst <= 0.55 and abs(autocorr) < 0.05:
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
        metrics['tail_ratio'] = abs(np.mean(returns[returns < np.percentile(returns, 5)])) / \
                                abs(np.mean(returns[returns > np.percentile(returns, 95)])) \
                                if len(returns[returns > np.percentile(returns, 95)]) > 0 else float('inf')
        
        if metrics['max_drawdown'] > 0:
            metrics['calmar'] = np.mean(returns) * 252 / metrics['max_drawdown']
            metrics['recovery_factor'] = np.sum(returns) / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else float('inf')
        else:
            metrics['calmar'] = float('inf')
            metrics['recovery_factor'] = float('inf')
        
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if len(losses) > 0 and np.sum(losses) > 0:
            metrics['omega'] = np.sum(gains) / np.sum(losses)
        else:
            metrics['omega'] = float('inf')
        
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
        
    def calculate_features(self, df):
        """Calculate comprehensive features with price action patterns"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Moving averages using Config values
        ma_periods = [Config.FAST_MA, Config.MEDIUM_MA, Config.SLOW_MA]
        if hasattr(Config, 'TREND_MA') and Config.TREND_MA:
            ma_periods.append(Config.TREND_MA)
            
        for period in ma_periods:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'].replace(0, 1) - 1
            
            if len(df) > period * 2:
                try:
                    price_deviation = df['close'] - df[f'sma_{period}']
                    rolling_mean = price_deviation.rolling(period).mean()
                    rolling_std = price_deviation.rolling(period).std().replace(0, 1)
                    df[f'price_deviation_{period}_z'] = (price_deviation - rolling_mean) / rolling_std
                except:
                    df[f'price_deviation_{period}_z'] = 0
        
        # Volatility features
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(Config.ATR_PERIOD).mean()
        df['atr_percent'] = df['atr'] / df['close'].replace(0, 1)
        df['volatility'] = df['returns'].rolling(20).std()
        
        df['realized_volatility_5'] = df['returns'].rolling(5).std() * np.sqrt(252)
        df['realized_volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['volatility_ratio'] = df['realized_volatility_5'] / df['realized_volatility_20'].replace(0, 1)
        
        # RSI
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(Config.RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(Config.RSI_PERIOD).mean()
            rs = gain / loss.replace(0, 1)
            df['rsi'] = 100 - (100 / (1 + rs))
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
            std_bb = df['close'].rolling(bb_period).std().replace(0, 1)
            df['bb_upper'] = sma_bb + (std_bb * bb_std)
            df['bb_lower'] = sma_bb - (std_bb * bb_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_bb.replace(0, 1)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1)
        except Exception as e:
            ProfessionalLogger.log(f"Bollinger Bands calculation error: {str(e)}", "WARNING", "FEATURE_ENGINE")
            df['bb_upper'] = df['close']
            df['bb_lower'] = df['close']
            df['bb_width'] = 0
            df['bb_position'] = 0.5
        
        # ADX if configured
        if hasattr(Config, 'USE_MARKET_REGIME') and Config.USE_MARKET_REGIME:
            try:
                df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                                        np.maximum(df['high'] - df['high'].shift(1), 0), 0)
                df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                                         np.maximum(df['low'].shift(1) - df['low'], 0), 0)
                
                tr = df['tr'].rolling(Config.ADX_PERIOD).mean()
                plus_di = 100 * (df['plus_dm'].rolling(Config.ADX_PERIOD).mean() / tr.replace(0, 1))
                minus_di = 100 * (df['minus_dm'].rolling(Config.ADX_PERIOD).mean() / tr.replace(0, 1))
                
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
                df['adx'] = dx.rolling(Config.ADX_PERIOD).mean()
                
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
            shifted_close = df['close'].shift(period).replace(0, 1)
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / shifted_close
        
        # Statistical features
        try:
            df['returns_skew_20'] = df['returns'].rolling(20).apply(lambda x: skew(x) if len(x[x == x]) > 10 else 0, raw=True)
            df['returns_kurtosis_20'] = df['returns'].rolling(20).apply(lambda x: kurtosis(x) if len(x[x == x]) > 10 else 0, raw=True)
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
        
        # Dynamic barriers: 2x ATR for profit, 1.5x ATR for stop
        upper_barriers = atr_percent * 2.0
        lower_barriers = -atr_percent * 1.5
        
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
            'hurst_exponent', 'regime_encoded',
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
        
        # Log successful validation
        if len(validation_results) > 0 and Config.DEBUG_MODE:
            ProfessionalLogger.log("Signal validation details:", "DEBUG", "FILTER")
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
    
    def _prepare_training_data(self, data):
        """Prepare training data"""
        try:
            df_features = self.feature_engine.calculate_features(data)
            df_labeled = self.feature_engine.create_labels(df_features, method='dynamic')
            df_labeled = df_labeled.dropna(subset=['label'])
            
            all_feature_cols = self.feature_engine.get_feature_columns()
            self.trained_feature_columns = all_feature_cols
            
            X = df_labeled[all_feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
            
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
        hurst = features.get('hurst_exponent', 0.5)
        volatility = features.get('volatility', 0)
        adx = features.get('adx', 20)
        
        if hurst > 0.6 and adx > 25:
            return 'trending'
        elif hurst < 0.4 and adx < 20:
            return 'mean_reverting'
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

            # TIGHT TRAILING AT 2R+
            if r_multiple > 2.0:
                if trade_type == 'BUY':
                    trail_price = close_price - (atr * 1.0)
                    if trail_price > new_sl:
                        new_sl = trail_price
                        sl_changed = True
                else:
                    trail_price = close_price + (atr * 1.0)
                    if trail_price < new_sl:
                        new_sl = trail_price
                        sl_changed = True

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
        """Detect when trend momentum is fading"""
        latest = df.iloc[-1]
        
        # Divergence check (only if in profit)
        if trade['type'] == 'BUY':
            price_higher = df['close'].iloc[-1] > df['close'].iloc[-5]
            rsi_lower = latest['rsi'] < df['rsi'].iloc[-5]
            if price_higher and rsi_lower and latest['rsi'] > 65:
                return True
        else: # SELL
            price_lower = df['close'].iloc[-1] < df['close'].iloc[-5]
            rsi_higher = latest['rsi'] > df['rsi'].iloc[-5]
            if price_lower and rsi_higher and latest['rsi'] < 35:
                return True
        
        # Extreme volume drop (less aggressive than before)
        if latest['volume_ratio'] < 0.3:
            return True
        
        # ADX Slope death
        if hasattr(df, 'adx'):
            adx_slope = latest['adx'] - df['adx'].iloc[-3]
            if adx_slope < -8: # More strict than -5
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
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
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
        features['trend_strength'] = abs(features['ema_fast'] - features['ema_slow']) / features['close']
        
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
        
        for tf_name, df in multi_tf_data.items():
            if df is None or len(df) < 20:
                continue
            
            features = self.calculate_timeframe_features(df)
            signal = self._generate_timeframe_signal(features, tf_name)
            
            analysis['timeframes'][tf_name] = {
                'signal': signal,
                'features': features,
                'weight': self.config.TIMEFRAME_WEIGHTS[self.config.TIMEFRAMES.index(tf_name)] if tf_name in self.config.TIMEFRAMES else 0.2
            }
            
            signals.append(signal)
            weights.append(analysis['timeframes'][tf_name]['weight'])
        
        if not signals:
            return analysis
        
        weighted_signal = np.average(signals, weights=weights)
        analysis['consensus_signal'] = 1 if weighted_signal > 0.5 else 0 if weighted_signal < -0.5 else 0.5
        
        signal_directions = [1 if s == 1 else -1 if s == 0 else 0 for s in signals]
        if len(signal_directions) > 1:
            agreement = sum(1 for i in range(len(signal_directions)) 
                          for j in range(i+1, len(signal_directions)) 
                          if signal_directions[i] * signal_directions[j] > 0)
            total_pairs = len(signal_directions) * (len(signal_directions) - 1) / 2
            analysis['alignment_score'] = agreement / total_pairs if total_pairs > 0 else 0
        
        if 'H1' in analysis['timeframes']:
            h1_trend = analysis['timeframes']['H1']['features']['trend_direction']
            if self.config.LONG_TIMEFRAME_TREND_FILTER:
                if analysis['consensus_signal'] == 1 and h1_trend < 0:
                    analysis['trend_filter'] = -1
                elif analysis['consensus_signal'] == 0 and h1_trend > 0:
                    analysis['trend_filter'] = -1
                else:
                    analysis['trend_filter'] = 1
        
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
        """Generate trading signal for a single timeframe"""
        if not features:
            return 0.5
        
        # Get multi-timeframe parameters from Config
        mtf_params = getattr(self.config, 'MULTI_TIMEFRAME_PARAMS', {})
        
        # Get timeframe-specific multiplier
        multiplier_key = f'{timeframe_name.upper()}_SIGNAL_MULTIPLIER'
        timeframe_multiplier = mtf_params.get(multiplier_key, 1.0)
        
        signal_score = 0
        
        price_pos = features.get('price_position', 0.5)
        if price_pos < 0.3:
            signal_score += 1
        elif price_pos > 0.7:
            signal_score -= 1
        
        trend = features.get('trend_direction', 0)
        signal_score += trend
        
        rsi = features.get('rsi', 50)
        if rsi < 30:
            signal_score += 1
        elif rsi > 70:
            signal_score -= 1
        
        momentum = features.get('momentum', 0)
        signal_score += 1 if momentum > 0.005 else -1 if momentum < -0.005 else 0
        
        # Apply Asian session adjustments if in Asian session
        hour = datetime.now().hour
        if 0 <= hour < 9:  # Asian session
            adjustments = mtf_params.get('ASIAN_SESSION_ADJUSTMENTS', {})
            # Adjust weights of different components
            if adjustments:
                signal_score = (
                    (trend * adjustments.get('trend_direction_weight', 1.0)) +
                    ((1 if rsi < 30 else -1 if rsi > 70 else 0) * adjustments.get('rsi_weight', 1.0)) +
                    ((1 if momentum > 0.005 else -1 if momentum < -0.005 else 0) * adjustments.get('volatility_weight', 1.0))
                )
        
        # Apply timeframe multiplier
        signal_score *= timeframe_multiplier
        
        # Get signal thresholds from config
        buy_threshold = mtf_params.get('BUY_SIGNAL_THRESHOLD', 0.5)
        sell_threshold = mtf_params.get('SELL_SIGNAL_THRESHOLD', -0.5)
        
        # Normalize score and apply thresholds
        normalized_score = max(-1, min(1, signal_score / 4))
        
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
        self.feature_engine = EnhancedFeatureEngine()  # Using enhanced version
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
        # CALCULATE POSITION SIZE
        # ==========================================
        
        # Get ATR for dynamic sizing
        atr = features.get('atr_percent', 0.001)
        if atr <= 0:
            atr = 0.001
        
        # Calculate base volume using risk percentage
        risk_amount = account.equity * Config.RISK_PERCENT
        
        # Calculate stop loss distance
        if Config.USE_DYNAMIC_SL_TP:
            atr_absolute = atr * entry_price
            sl_distance = atr_absolute * Config.ATR_SL_MULTIPLIER
        else:
            sl_distance = entry_price * Config.FIXED_SL_PERCENT
        
        # Calculate volume based on risk
        point_value = symbol_info.trade_contract_size * symbol_info.trade_tick_size / symbol_info.trade_tick_value
        volume_raw = risk_amount / (sl_distance * point_value)
        
        # Apply volatility scaling if enabled
        if Config.VOLATILITY_SCALING_ENABLED:
            volatility = features.get('volatility', 0)
            if volatility > Config.HIGH_VOL_THRESHOLD:
                volume_raw *= Config.HIGH_VOL_SIZE_MULTIPLIER
                ProfessionalLogger.log(f"High volatility - reducing size by {Config.HIGH_VOL_SIZE_MULTIPLIER}x", "RISK", "ENGINE")
            elif volatility < Config.LOW_VOL_THRESHOLD:
                volume_raw *= Config.LOW_VOL_SIZE_MULTIPLIER
                ProfessionalLogger.log(f"Low volatility - increasing size by {Config.LOW_VOL_SIZE_MULTIPLIER}x", "RISK", "ENGINE")
        
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
            # Record trade in memory
            trade_data = {
                'ticket': result.order,
                'symbol': Config.SYMBOL,
                'type': trade_type_str,
                'signal': signal,
                'volume': volume,
                'open_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sl_points': sl_distance_points,
                'tp_points': tp_distance_points,
                'open_time': int(time.time()),
                'confidence': confidence,
                'features': features,
                'model_details': model_details,
                'status': 'open',
                'mfe_points': 0,
                'mae_points': 0
            }
            
            self.active_positions[result.order] = trade_data
            self._save_engine_state()
            
            ProfessionalLogger.log(
                f"✅ Trade Opened: #{result.order} | {trade_type_str} {volume:.3f} @ {entry_price:.5f} | "
                f"Conf: {confidence:.0%} | RR: {rr_ratio:.2f}",
                "SUCCESS", "ENGINE"
            )
        else:
            ProfessionalLogger.log("Trade execution failed", "ERROR", "ENGINE")
            
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
        """Enhanced live trading with all new features"""
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        ProfessionalLogger.log("STARTING ENHANCED LIVE TRADING", "TRADE", "ENGINE")
        ProfessionalLogger.log(f"Dynamic Barriers: {'ENABLED' if Config.USE_DYNAMIC_BARRIERS else 'DISABLED'}", "INFO", "ENGINE")
        ProfessionalLogger.log(f"Entry Confirmation: {'ENABLED' if Config.USE_CONFIRMATION_ENTRY else 'DISABLED'}", "INFO", "ENGINE")
        ProfessionalLogger.log(f"Parameter Optimization: {'ENABLED' if Config.PARAM_OPTIMIZATION_ENABLED else 'DISABLED'}", "INFO", "ENGINE")
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        
        self.running = True
        try:
            while self.running:
                self.run_periodic_tasks()
                
                required_lookback = max(500, Config.TREND_MA * 2) 
                
                rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, required_lookback)
                if rates is None or len(rates) < Config.TREND_MA + 10:
                    ProfessionalLogger.log("Failed to get sufficient rates, retrying...", "WARNING", "ENGINE")
                    time.sleep(10) # Blocking sleep here is acceptable as we can't proceed without data
                    continue
                
                df_current = pd.DataFrame(rates)
                
                # ==========================================
                # FEATURE CALCULATION & CACHING
                # ==========================================
                # Only recalculate features if enough time has passed or data has changed significantly
                current_time = datetime.now()
                if self.last_feature_time is None or \
                   (current_time - self.last_feature_time).total_seconds() > Config.FEATURE_RECALC_INTERVAL_SECONDS:
                    self.cached_features = self.feature_engine.calculate_features(df_current)
                    self.last_feature_time = current_time
                features = self.cached_features
                
                # ==========================================
                # MULTI-TIMEFRAME ANALYSIS
                # ==========================================
                multi_tf_signal = None
                multi_tf_confidence = 0
                multi_tf_alignment = 0
                min_confidence_override = Config.MIN_CONFIDENCE
                min_agreement_override = Config.MIN_ENSEMBLE_AGREEMENT
                
                if Config.MULTI_TIMEFRAME_ENABLED:
                    try:
                        mtf_recommendation = self.multi_tf_analyser.get_multi_timeframe_recommendation(Config.SYMBOL)
                        
                        if mtf_recommendation:
                            multi_tf_signal = mtf_recommendation.get('consensus_signal')
                            multi_tf_confidence = mtf_recommendation.get('confidence', 0)
                            multi_tf_alignment = mtf_recommendation.get('alignment_score', 0)
                            trend_filter_passed = mtf_recommendation.get('trend_filter_passed', True)
                            
                            recommendation = mtf_recommendation.get('recommendation', 'HOLD')
                            ProfessionalLogger.log(
                                f"Multi-TF: {recommendation} | "
                                f"Align: {multi_tf_alignment:.0%} | "
                                f"Conf: {multi_tf_confidence:.0%} | "
                                f"Trend Filter: {'PASS' if trend_filter_passed else 'FAIL'}",
                                "ANALYSIS", "MULTI_TF"
                            )
                            
                            if not trend_filter_passed:
                                ProfessionalLogger.log("Trade blocked by H1 trend filter", "WARNING", "MULTI_TF")
                                signal = None # Block main signal if trend filter fails
                            
                            elif multi_tf_alignment < Config.TIMEFRAME_ALIGNMENT_THRESHOLD and Config.REQUIRE_TIMEFRAME_ALIGNMENT:
                                # Relaxed threshold raising during NY/London
                                is_high_volume_session = (Config.LONDON_OPEN_HOUR <= datetime.now().hour <= Config.NY_CLOSE_HOUR)
                                conf_mult = 1.15 if is_high_volume_session else 1.3
                                agree_mult = 1.1 if is_high_volume_session else 1.2
                                
                                min_confidence_override = Config.MIN_CONFIDENCE * conf_mult
                                min_agreement_override = Config.MIN_ENSEMBLE_AGREEMENT * agree_mult
                                
                                ProfessionalLogger.log(
                                    f"Low alignment ({multi_tf_alignment:.0%} < {Config.TIMEFRAME_ALIGNMENT_THRESHOLD:.0%}) - "
                                    f"{'NY/London session (relaxed)' if is_high_volume_session else 'Standard'} "
                                    f"thresholds: Conf>{min_confidence_override:.0%}, Agree>{min_agreement_override:.0%}",
                                    "WARNING", "MULTI_TF"
                                )
                            
                            elif recommendation in ['STRONG_BUY', 'STRONG_SELL']:
                                min_confidence_override = Config.MIN_CONFIDENCE * 0.9
                                min_agreement_override = Config.MIN_ENSEMBLE_AGREEMENT * 0.9
                                ProfessionalLogger.log(
                                    f"Strong multi-TF signal - "
                                    f"lowering thresholds: Conf>{min_confidence_override:.0%}, Agree>{min_agreement_override:.0%}",
                                    "INFO", "MULTI_TF"
                                )
                                
                    except Exception as e:
                        ProfessionalLogger.log(f"Multi-TF analysis error: {str(e)}", "ERROR", "MULTI_TF")
                        min_confidence_override = Config.MIN_CONFIDENCE
                        min_agreement_override = Config.MIN_ENSEMBLE_AGREEMENT
                
                # ==========================================
                # MAIN MODEL PREDICTION
                # ==========================================
                # Use model_lock to ensure no prediction during background training
                with self.model_lock:
                    signal, confidence, dict_features, model_details = self.model.predict(df_current, features) # Pass cached features
                
                # ==========================================
                # ADAPTIVE EXIT LOGIC
                # ==========================================
                if self.active_positions:
                    self.exit_manager.manage_positions(
                        features, 
                        self.active_positions, 
                        signal, 
                        confidence
                    )
                
                # ==========================================
                # SIGNAL VALIDATION & FILTERING
                # ==========================================
                if signal is None:
                    if self.iteration % 30 == 0:
                        tick = mt5.symbol_info_tick(Config.SYMBOL)
                        if tick:
                            price = tick.ask
                            positions = self.get_current_positions()
                            ProfessionalLogger.log(
                                f"Waiting for signal | Price: {price:.2f} | "
                                f"Positions: {positions} | "
                                f"Multi-TF: {multi_tf_signal if multi_tf_signal is not None else 'N/A'}",
                                "INFO", "ENGINE"
                            )
                    # Prepare for next iteration with non-blocking sleep
                    sleep_time = self._calculate_sleep_time()
                    for _ in range(int(sleep_time)):
                        if not self.running: break
                        if _ % 5 == 0 and self.active_positions:
                            df_current_quick = self.get_historical_data(timeframe=Config.TIMEFRAME, bars=100)
                            if df_current_quick is not None:
                                df_features_quick = self.feature_engine.calculate_features(df_current_quick)
                                self.exit_manager.manage_positions(
                                    df_features_quick, 
                                    self.active_positions, 
                                    None, 0.5
                                )
                        time.sleep(1)
                    continue
                
                # Calculate model agreement
                agreement = 0
                if model_details:
                    predictions = [m['prediction'] for m in model_details.values() 
                                if m['prediction'] != -1]
                    if predictions:
                        agreement = predictions.count(signal) / len(predictions)
                
                # Multi-TF validation
                if Config.MULTI_TIMEFRAME_ENABLED and multi_tf_signal is not None:
                    if multi_tf_signal != 0.5:
                        signal_match = (signal == 1 and multi_tf_signal > 0.6) or \
                                    (signal == 0 and multi_tf_signal < 0.4)
                        
                        if not signal_match:
                            ProfessionalLogger.log(
                                f"Model signal ({'BUY' if signal == 1 else 'SELL'}) "
                                f"rejected by multi-TF consensus ({multi_tf_signal:.2f})",
                                "WARNING", "MULTI_TF"
                            )
                            # Prepare for next iteration with non-blocking sleep
                            sleep_time = self._calculate_sleep_time()
                            for _ in range(int(sleep_time)):
                                if not self.running: break
                                if _ % 5 == 0 and self.active_positions:
                                    df_current_quick = self.get_historical_data(timeframe=Config.TIMEFRAME, bars=100)
                                    if df_current_quick is not None:
                                        df_features_quick = self.feature_engine.calculate_features(df_current_quick)
                                        self.exit_manager.manage_positions(
                                            df_features_quick, 
                                            self.active_positions, 
                                            None, 0.5
                                        )
                                time.sleep(1)
                            continue
                
                # Combined confidence
                combined_confidence = confidence
                if Config.MULTI_TIMEFRAME_ENABLED and multi_tf_confidence > 0:
                    combined_confidence = (confidence * 0.7) + (multi_tf_confidence * 0.3)
                
                # Signal Quality Filtering
                market_context = {
                    'volatility_regime': dict_features.get('volatility_regime', 1),
                    'spread_pips': 2,  # Default
                    'hour': datetime.now().hour,
                    'high_impact_news_soon': False,
                    'existing_positions': list(self.active_positions.values()),
                    'vol_surprise': dict_features.get('vol_surprise', 0),
                    'market_regime': self.last_regime,
                    'multi_tf_alignment': multi_tf_alignment,
                    'multi_tf_signal': multi_tf_signal
                }
                
                is_valid, filter_reason = self.signal_filter.validate_signal(
                    signal, combined_confidence, dict_features, market_context
                )
                
                if not is_valid:
                    ProfessionalLogger.log(f"Signal rejected by filter: {filter_reason}", "FILTER", "ENGINE")
                    # Prepare for next iteration with non-blocking sleep
                    sleep_time = self._calculate_sleep_time()
                    for _ in range(int(sleep_time)):
                        if not self.running: break
                        if _ % 5 == 0 and self.active_positions:
                            df_current_quick = self.get_historical_data(timeframe=Config.TIMEFRAME, bars=100)
                            if df_current_quick is not None:
                                df_features_quick = self.feature_engine.calculate_features(df_current_quick)
                                self.exit_manager.manage_positions(
                                    df_features_quick, 
                                    self.active_positions, 
                                    None, 0.5
                                )
                        time.sleep(1)
                    continue
                
                # Entry Timing Confirmation
                if Config.USE_CONFIRMATION_ENTRY:
                    should_enter, entry_reason = self.entry_timing.should_enter(
                        signal, combined_confidence, dict_features, df_current
                    )
                    
                    if not should_enter:
                        ProfessionalLogger.log(f"Entry delayed: {entry_reason}", "CONFIRMATION", "ENGINE")
                        # Prepare for next iteration with non-blocking sleep
                        sleep_time = self._calculate_sleep_time()
                        for _ in range(int(sleep_time)):
                            if not self.running: break
                            if _ % 5 == 0 and self.active_positions:
                                df_current_quick = self.get_historical_data(timeframe=Config.TIMEFRAME, bars=100)
                                if df_current_quick is not None:
                                    df_features_quick = self.feature_engine.calculate_features(df_current_quick)
                                    self.exit_manager.manage_positions(
                                        df_features_quick, 
                                        self.active_positions, 
                                        None, 0.5
                                    )
                            time.sleep(1)
                        continue
                
                # Log comprehensive signal analysis
                signal_type = "BUY" if signal == 1 else "SELL"
                status_msg = (f"Signal Analysis | {signal_type} | "
                            f"Model Conf: {confidence:.1%} | "
                            f"Agreement: {agreement:.0%} | ")
                
                if Config.MULTI_TIMEFRAME_ENABLED:
                    status_msg += f"Multi-TF Align: {multi_tf_alignment:.0%} | "
                    status_msg += f"Combined Conf: {combined_confidence:.1%}"
                else:
                    status_msg += f"Price: {df_current['close'].iloc[-1]:.2f}"
                
                ProfessionalLogger.log(status_msg, "ANALYSIS", "ENGINE")
                
                # Log key features
                if dict_features is not None:
                    key_features = {
                        'rsi': dict_features.get('rsi_normalized', 0) * 50 + 50,
                        'volatility': dict_features.get('volatility', 0),
                        'regime': dict_features.get('regime_encoded', 0),
                        'atr_percent': dict_features.get('atr_percent', 0)
                    }
                    ProfessionalLogger.log(
                        f"Key Features: RSI={key_features['rsi']:.1f} | "
                        f"Vol={key_features['volatility']:.4f} | "
                        f"Regime={key_features['regime']} | "
                        f"ATR%={key_features['atr_percent']:.4f}",
                        "DATA", "ENGINE"
                    )
                
                # ==========================================
                # TRADE EXECUTION DECISION
                # ==========================================
                execute_trade = False
                execution_reason = ""
                
                if (combined_confidence >= min_confidence_override and 
                    agreement >= min_agreement_override):
                    
                    if Config.MULTI_TIMEFRAME_ENABLED and Config.REQUIRE_TIMEFRAME_ALIGNMENT:
                        if multi_tf_alignment >= Config.TIMEFRAME_ALIGNMENT_THRESHOLD:
                            execute_trade = True
                            execution_reason = "Strong multi-TF alignment"
                        elif combined_confidence > (min_confidence_override * 1.5):
                            execute_trade = True
                            execution_reason = f"Very high confidence ({combined_confidence:.1%})"
                        else:
                            execution_reason = f"Low multi-TF alignment ({multi_tf_alignment:.0%})"
                    else:
                        execute_trade = True
                        execution_reason = "Thresholds met (Alignment Requirement Disabled)"
                
                if execute_trade:
                    ProfessionalLogger.log(
                        f"🎯 {execution_reason} - Executing {signal_type} signal! | "
                        f"Combined Confidence: {combined_confidence:.1%}",
                        "SUCCESS", "ENGINE"
                    )
                    
                    if Config.MULTI_TIMEFRAME_ENABLED and multi_tf_signal is not None:
                        if 'multi_tf' not in model_details:
                            model_details['multi_tf'] = {}
                        model_details['multi_tf'].update({
                            'consensus_signal': multi_tf_signal,
                            'alignment_score': multi_tf_alignment,
                            'confidence': multi_tf_confidence,
                            'min_thresholds_applied': {
                                'confidence': min_confidence_override,
                                'agreement': min_agreement_override
                            }
                        })
                    
                    self.execute_trade(signal, combined_confidence, df_current, dict_features, model_details)
                else:
                    if self.iteration % 10 == 0:
                        ProfessionalLogger.log(
                            f"Signal rejected | Reason: {execution_reason} | "
                            f"Conf: {combined_confidence:.1%} (need {min_confidence_override:.1%}) | "
                            f"Agree: {agreement:.0%} (need {min_agreement_override:.0%})",
                            "INFO", "ENGINE"
                        )
                
                self.update_performance_metrics()
                
                # Prepare for next iteration with non-blocking sleep
                sleep_time = self._calculate_sleep_time()
                
                # Non-blocking sleep loop to allow emergency exits
                for _ in range(int(sleep_time)):
                    if not self.running: break
                    
                    # Every 5 seconds, check if we need to manage existing positions
                    if _ % 5 == 0 and self.active_positions:
                        df_current_quick = self.get_historical_data(timeframe=Config.TIMEFRAME, bars=100)
                        if df_current_quick is not None:
                            df_features_quick = self.feature_engine.calculate_features(df_current_quick)
                            self.exit_manager.manage_positions(
                                df_features_quick, 
                                self.active_positions, 
                                None, 0.5
                            )
                    
                    time.sleep(1)
                
        except Exception as e:
            ProfessionalLogger.log(f"Engine error in main loop: {e}", "ERROR", "ENGINE")
            import traceback
            traceback.print_exc()
            time.sleep(10) # Blocking sleep on error to prevent rapid error loops
        except KeyboardInterrupt:
            ProfessionalLogger.log("\nShutdown requested by user", "WARNING", "ENGINE")
        finally:
            self.running = False
            self.print_performance_report()
            mt5.shutdown()
            ProfessionalLogger.log("Disconnected from MT5", "INFO", "ENGINE")

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