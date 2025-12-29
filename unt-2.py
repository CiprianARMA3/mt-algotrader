import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import os
import sys
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, classification_report
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
# PROFESSIONAL CONFIGURATION
# ==========================================
class Config:
    """
    Ultra-Precise XAUUSD Trading Configuration
    Optimized for Gold's unique characteristics:
    - High volatility compared to forex pairs
    - Strong trend persistence
    - Sensitive to USD strength and economic data
    - Active during London/NY sessions
    """
    
    # ==========================================
    # MT5 CONNECTION
    # ==========================================
    MT5_LOGIN = 5044108820
    MT5_PASSWORD = "@rC1KbQb"
    MT5_SERVER = "MetaQuotes-Demo"
    
    # ==========================================
    # TRADING INSTRUMENT SPECIFICATIONS
    # ==========================================
    SYMBOL = "XAUUSD"
    TIMEFRAME = mt5.TIMEFRAME_M5  # 15-min optimal for intraday gold trading
    
    # XAUUSD-specific: Gold typically trades in 0.01 lot increments
    BASE_VOLUME = 0.25  # START SMALL - critical for risk management
    MAX_VOLUME = 1.00   # Maximum position size cap
    MIN_VOLUME = 0.25   # MT5 minimum for gold
    VOLUME_STEP = 0.01  # Standard gold lot increment
    
    MAGIC_NUMBER = 998877
    
    # ==========================================
    # RISK MANAGEMENT - CONSERVATIVE & PRECISE
    # ==========================================
    
    # Position Sizing
    RISK_PERCENT = 0.01  # Risk 1% per trade (Conservative for stacking)
    MAX_TOTAL_RISK_PERCENT = 0.05  # MAX AGGREGATE RISK: Cap total exposure at 5%
    MAX_RISK_PER_TRADE = 100  # Maximum $ risk per trade (failsafe)
    
    # Signal Quality Thresholds
    MIN_CONFIDENCE = 0.50  # 70% minimum model confidence
    MIN_ENSEMBLE_AGREEMENT = 0.60  # 85% model agreement (increased precision)
    
    # Position Limits
    MAX_POSITIONS = 5  # Allow multiple positions (stacking)
    MAX_DAILY_TRADES = 10  # Increased to allow scaling in
    MIN_TIME_BETWEEN_TRADES = 10  # Reduced to 10 minutes for scaling in
    
    # Loss Limits
    MAX_DAILY_LOSS_PERCENT = 2.0  # Stop trading after 2% daily loss
    MAX_WEEKLY_LOSS_PERCENT = 5.0  # Weekly circuit breaker
    MAX_DRAWDOWN_PERCENT = 10.0  # Maximum account drawdown
    MAX_CONSECUTIVE_LOSSES = 3  # Stop after 3 losses in a row
    
    # Kelly Criterion
    KELLY_FRACTION = 0.15  # Conservative 15% of Kelly (was 20%)
    USE_HALF_KELLY = True  # Use half-Kelly for extra safety
    
    # Statistical Risk Metrics
    VAR_CONFIDENCE = 0.99  # 99% Value at Risk
    CVAR_CONFIDENCE = 0.99  # 99% Conditional VaR
    MAX_POSITION_CORRELATION = 0.5  # Max correlation between positions
    
    # ==========================================
    # MACHINE LEARNING MODEL PARAMETERS
    # ==========================================
    
    # Data Collection
    LOOKBACK_BARS = 65000  # 8000 bars for stable statistics 
    TRAINING_MIN_SAMPLES = 6000  # Minimum samples for reliable training
    VALIDATION_SPLIT = 0.20  # 20% validation set
    
    # Retraining Schedule
    RETRAIN_HOURS = 24  # Retrain every 4 hours (was 3 - too frequent)
    RETRAIN_ON_PERFORMANCE_DROP = True  # Retrain if performance degrades
    MIN_ACCURACY_THRESHOLD = 0.50  # Retrain if accuracy drops below 58%
    
    # Walk-Forward Optimization
    WALK_FORWARD_WINDOW = 1000  # 500 bars per window
    WALK_FORWARD_STEP = 100  # 100 bar step (80% overlap)
    WALK_FORWARD_FOLDS = 5  # 5-fold cross-validation
    
    # Feature Engineering Flags
    USE_FRACTIONAL_DIFF = True
    FD_THRESHOLD = 0.4  # Fractional differentiation d=0.4
    USE_TICK_VOLUME_VOLATILITY = True
    TICK_SKEW_LOOKBACK = 50
    
    # Labeling Method
    TRIPLE_BARRIER_METHOD = True  # Use triple-barrier labeling
    BARRIER_UPPER = 0.0030  # 0.30% = $3.00 (easier to hit, more predictable)
    BARRIER_LOWER = -0.0020  # 0.20% = $2.00
    BARRIER_TIME = 6  # 6 bars = 90 minutes (was 10 = too long)
    
    # Ensemble Configuration
    USE_STACKING_ENSEMBLE = True
    ENSEMBLE_DIVERSITY_WEIGHT = 0.3
    ADAPTIVE_ENSEMBLE_WEIGHTING = True
    MODEL_CONFIDENCE_CALIBRATION = True
    
    # Data Quality
    MIN_DATA_QUALITY_SCORE = 0.75  # Require 75% data quality
    OUTLIER_REMOVAL_THRESHOLD = 4.0  # Remove outliers beyond 4 std devs
    
    # ==========================================
    # TECHNICAL INDICATORS - GOLD-OPTIMIZED
    # ==========================================
    
    # Trend Indicators
    ATR_PERIOD = 14  # Standard ATR (14 is industry standard)
    RSI_PERIOD = 14  # Standard RSI
    ADX_PERIOD = 14  # Standard ADX (14 better than 20 for responsiveness)
    
    # Moving Averages (Gold-specific)
    FAST_MA = 8   # Fast MA for gold trends
    MEDIUM_MA = 21  # Medium MA
    SLOW_MA = 50  # Slow MA
    TREND_MA = 200  # Long-term trend filter
    
    # Volatility Bands
    BB_PERIOD = 20  # Bollinger Bands period
    BB_STD = 2.0  # Standard deviations
    
    # ==========================================
    # STATISTICAL FEATURES - PRECISION TUNED
    # ==========================================
    
    # GARCH Volatility
    GARCH_VOL_PERIOD = 20  # GARCH(1,1) lookback
    GARCH_P = 1  # GARCH p parameter
    GARCH_Q = 1  # GARCH q parameter
    
    # Hurst Exponent (mean reversion vs trending)
    HURST_WINDOW = 100  # 100 bars optimal for M15
    HURST_TRENDING_THRESHOLD = 0.55  # H > 0.55 = trending
    HURST_MEANREVERTING_THRESHOLD = 0.45  # H < 0.45 = mean-reverting
    
    # Tail Risk
    TAIL_INDEX_WINDOW = 150  # Tail risk lookback
    VAR_LOOKBACK = 100  # VaR calculation window
    
    # Correlation Analysis
    CORRELATION_WINDOW = 50  # 50 bars for correlations (was 100 - too slow)
    CORRELATED_SYMBOLS = ["DXY", "US10Y", "EURUSD"]  # Dollar index, yields, EUR
    
    # ==========================================
    # STOP LOSS & TAKE PROFIT - PRECISION TUNED
    # ==========================================
    
    USE_DYNAMIC_SL_TP = True  # Use ATR-based dynamic stops
    
    # ATR-Based Stops (Gold-optimized)
    ATR_SL_MULTIPLIER = 1.5  # 1.5x ATR for stop loss
    ATR_TP_MULTIPLIER = 1.75  # 3.0x ATR for take profit (2:1 R:R)
    
    # Minimum Risk/Reward
    MIN_RR_RATIO = 1.25  # Minimum 2:1 reward:risk
    
    # Fixed Stops (backup if ATR unavailable)
    FIXED_SL_PERCENT = 0.0035  # 0.35% stop loss
    FIXED_TP_PERCENT = 0.0070  # 0.70% take profit
    
    # Points-based Limits (XAUUSD point = $0.01)
    MIN_SL_DISTANCE_POINTS = 50  # Minimum $0.50 stop loss
    MAX_SL_DISTANCE_POINTS = 300  # Maximum $3.00 stop loss
    MIN_TP_DISTANCE_POINTS = 100  # Minimum $1.00 take profit
    MAX_TP_DISTANCE_POINTS = 600  # Maximum $6.00 take profit
    
    # Trailing Stop
    USE_TRAILING_STOP = True
    TRAILING_STOP_ACTIVATION = 1.5  # Activate after 1.5x initial risk
    TRAILING_STOP_DISTANCE = 1.0  # Trail at 1.0x ATR
    
    # Break-even Stop
    USE_BREAKEVEN_STOP = True
    BREAKEVEN_ACTIVATION = 1.0  # Move to B/E after 1x initial risk
    BREAKEVEN_OFFSET = 0.0001  # Small profit lock ($0.10)
    
    # ==========================================
    # MARKET REGIME DETECTION
    # ==========================================
    
    USE_MARKET_REGIME = True
    
    # Trend Strength (ADX-based)
    ADX_TREND_THRESHOLD = 25  # ADX > 25 = trending (standard)
    ADX_STRONG_TREND_THRESHOLD = 40  # ADX > 40 = strong trend
    ADX_SLOPE_THRESHOLD = 0.5  # ADX slope threshold
    
    # Volatility Regimes
    VOLATILITY_SCALING_ENABLED = True
    HIGH_VOL_THRESHOLD = 0.012  # 1.2% daily volatility = high
    NORMAL_VOL_THRESHOLD = 0.008  # 0.8% = normal
    LOW_VOL_THRESHOLD = 0.005  # 0.5% = low
    
    # Position Sizing Adjustments by Regime
    HIGH_VOL_SIZE_MULTIPLIER = 0.5  # Half size in high volatility
    LOW_VOL_SIZE_MULTIPLIER = 1.2  # 20% larger in low volatility
    
    # ==========================================
    # TIME-BASED FILTERS - GOLD SESSIONS
    # ==========================================
    
    SESSION_AWARE_TRADING = True
    
    # Trading Sessions (UTC times)
    # Gold is most active during London (8-11 UTC) and NY (13-17 UTC)
    AVOID_ASIAN_SESSION = True  # Skip 0-7 UTC (low liquidity)
    PREFER_LONDON_NY_OVERLAP = True  # 13-16 UTC (highest volume)
    
    LONDON_OPEN_HOUR = 8  # 8 AM UTC
    LONDON_CLOSE_HOUR = 16  # 4 PM UTC
    NY_OPEN_HOUR = 13  # 1 PM UTC (8 AM EST)
    NY_CLOSE_HOUR = 20  # 8 PM UTC (3 PM EST)
    
    # Avoid trading during:
    AVOID_FIRST_15MIN = True  # Skip first 15 min of sessions (volatility spike)
    AVOID_LAST_30MIN = True  # Skip last 30 min before session close
    
    # News Events
    NEWS_EVENT_BUFFER_HOURS = 1  # Stop trading 1hr before major news
    HIGH_IMPACT_NEWS_BUFFER = 2  # 2hr buffer for NFP, FOMC, etc.
    
    # Days of Week
    AVOID_MONDAY_FIRST_HOUR = True  # Monday gaps can be erratic
    AVOID_FRIDAY_LAST_HOURS = True  # Friday position squaring
    
    # ==========================================
    # ORDER EXECUTION - PRECISION SETTINGS
    # ==========================================
    
    # Slippage & Timing
    MAX_SLIPPAGE_POINTS = 10  # Maximum 10 points ($0.10) slippage
    ORDER_TIMEOUT_SECONDS = 30  # Order timeout
    MAX_RETRIES = 3  # Retry failed orders 3 times
    RETRY_DELAY_MS = 500  # 500ms between retries
    
    # Order Types
    USE_MARKET_ORDERS = True  # Use market orders (immediate execution)
    USE_LIMIT_ORDERS = False  # Disable limit orders for now
    
    # Execution Quality
    CHECK_SPREAD_BEFORE_ENTRY = True
    MAX_SPREAD_POINTS = 30  # Maximum 5 points ($0.05) spread
    NORMAL_SPREAD_POINTS = 2  # Normal spread should be ~2 points
    
    # Commission
    COMMISSION_PER_LOT = 3.5  # $3.50 per lot (typical)
    
    # ==========================================
    # PERFORMANCE METRICS & MONITORING
    # ==========================================
    
    # Minimum Performance Standards
    MIN_SHARPE_RATIO = 0.8  # Minimum Sharpe ratio
    MIN_PROFIT_FACTOR = 1.5  # Minimum profit factor
    MIN_WIN_RATE = 0.45  # Minimum 45% win rate acceptable with 2:1 R:R
    MAX_DRAWDOWN_DURATION = 10  # Max 10 trades in drawdown
    
    # Statistical Testing
    MIN_SAMPLES_FOR_STATS = 100  # Need 100 samples for reliable stats
    CONFIDENCE_LEVEL = 0.95  # 95% confidence intervals
    BOOTSTRAP_SAMPLES = 1000  # Bootstrap iterations
    
    # ==========================================
    # ADAPTIVE SYSTEMS
    # ==========================================
    
    ADAPTIVE_RISK_MANAGEMENT = True
    PERFORMANCE_BASED_POSITION_SIZING = True
    REAL_TIME_MARKET_STRESS_INDICATOR = True
    
    # Adaptation Parameters
    PERFORMANCE_LOOKBACK_TRADES = 20  # Review last 20 trades
    GOOD_PERFORMANCE_THRESHOLD = 0.60  # 60% win rate = good
    POOR_PERFORMANCE_THRESHOLD = 0.40  # 40% win rate = poor
    
    # Position Size Adjustments
    INCREASE_SIZE_AFTER_WINS = False  # Don't increase after wins (anti-martingale)
    DECREASE_SIZE_AFTER_LOSSES = True  # Decrease after losses
    SIZE_DECREASE_FACTOR = 0.8  # Reduce to 80% after loss
    SIZE_RECOVERY_FACTOR = 1.1  # Increase back slowly
    
    # ==========================================
    # DATA STORAGE & LOGGING
    # ==========================================
    
    TRADE_HISTORY_FILE = "trade_history_xauusd.json"
    MODEL_SAVE_FILE = "ensemble_model_xauusd.pkl"
    BACKTEST_RESULTS_FILE = "backtest_results_xauusd.json"
    PERFORMANCE_LOG_FILE = "performance_log_xauusd.csv"
    
    MEMORY_SIZE = 1000  # Keep last 1000 trades in memory
    LEARNING_WEIGHT = 0.4  # Learning rate from past trades
    
    # Logging Levels
    LOG_LEVEL_CONSOLE = "INFO"  # Console verbosity
    LOG_LEVEL_FILE = "DEBUG"  # File verbosity (more detailed)
    LOG_TRADES = True
    LOG_PREDICTIONS = True
    LOG_PERFORMANCE = True
    
    # ==========================================
    # MULTI-TIMEFRAME ANALYSIS
    # ==========================================
    
    MULTI_TIMEFRAME_ENABLED = True
    TIMEFRAMES = ['M5', 'M15', 'H1']  # 5min, 15min, 1hour
    TIMEFRAME_WEIGHTS = [0.2, 0.5, 0.3]  # Weight M15 most heavily
    TIMEFRAME_ALIGNMENT_THRESHOLD = 0.75  # Need 75% agreement
    
    # Timeframe-specific parameters
    LONG_TIMEFRAME_TREND_FILTER = True  # Use H1 for trend direction
    SHORT_TIMEFRAME_ENTRY = True  # Use M5 for precise entry
    
    # ==========================================
    # GOLD-SPECIFIC PARAMETERS
    # ==========================================
    
    GOLD_VOLATILITY_ADJUSTMENT = True
    
    # Gold typically moves $10-30 per day
    EXPECTED_DAILY_RANGE = 20  # Expected $20 daily range
    HIGH_RANGE_MULTIPLIER = 1.5  # High day = 1.5x normal
    LOW_RANGE_MULTIPLIER = 0.5  # Low day = 0.5x normal
    
    # Dollar Index Correlation
    USE_DXY_FILTER = True  # Filter trades based on DXY direction
    DXY_CORRELATION_THRESHOLD = -0.7  # Gold/DXY typically -0.7 to -0.9
    
    # Interest Rate Sensitivity
    USE_YIELD_FILTER = False  # Disable for now (complex)
    
    # ==========================================
    # SAFETY FEATURES
    # ==========================================
    
    # Emergency Stop
    ENABLE_EMERGENCY_STOP = True
    EMERGENCY_STOP_DRAWDOWN = 0.15  # Stop all trading at 15% drawdown
    
    # Position Checks
    CHECK_MARGIN_BEFORE_TRADE = True
    MIN_FREE_MARGIN_PERCENT = 0.30  # Need 30% free margin
    
    # Health Checks
    CHECK_CONNECTION_BEFORE_TRADE = True
    MAX_PING_MS = 100  # Maximum 100ms latency
    
    # Account Protection
    MAX_DAILY_VOLUME = 1.0  # Maximum 1 lot total daily volume
    REQUIRE_STOP_LOSS = True  # Always require stop loss
    REQUIRE_TAKE_PROFIT = True  # Always require take profit
    
    # ==========================================
    # DEBUGGING & TESTING
    # ==========================================
    
    DEBUG_MODE = False  # Set True for verbose debugging
    PAPER_TRADING_MODE = False  # Set True for simulation
    BACKTEST_MODE = False  # Set True for backtesting
    
    # Validation
    VALIDATE_SIGNALS = True  # Double-check signals before execution
    VALIDATE_RISK = True  # Verify risk calculations
    VALIDATE_STOPS = True  # Verify SL/TP distances

# ==========================================
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
            'ANALYSIS': ProfessionalLogger.COLORS['CYAN']
        }
        color = colors.get(level, ProfessionalLogger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level:8s}{ProfessionalLogger.COLORS['RESET']}] [{component:12s}] {message}", flush=True)

# ==========================================
# ADVANCED STATISTICAL ANALYZER
# ==========================================
class AdvancedStatisticalAnalyzer:
    """Advanced statistical analysis for MT5 data"""
    
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
            # CORRECTION: Uses 252*96 for M15 annualization
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
        
        # Value at Risk and Expected Shortfall - HISTORICAL METHOD + DAILY SCALING
        # VaR on 15m timeframe
        var_95_15m = np.percentile(returns, 5)
        var_99_15m = np.percentile(returns, 1)
        
        # Scale to Daily VaR (approximate using sqrt(T)) - 96 intervals per day
        # Note: This implies "If I held this risk for a day"
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
        
        # Remove NaN and infinite values
        mask = ~(np.isnan(delta_series) | np.isnan(lag_series) | 
                np.isinf(delta_series) | np.isinf(lag_series))
        
        delta_series = delta_series[mask]
        lag_series = lag_series[mask]
        
        if len(delta_series) < 5 or len(lag_series) < 5:
            return 0
        
        try:
            # Fit OLS: Δy = α + β*y_{t-1} + ε
            X = np.column_stack([np.ones_like(lag_series), lag_series])
            beta = np.linalg.lstsq(X, delta_series, rcond=None)[0][1]
            
            # Calculate half-life
            if beta >= 0:
                return float('inf')  # No mean reversion
            
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
            # Convert to returns
            returns = np.diff(np.log(prices))
            
            if method == 'rs':
                # Rescaled Range method
                n = len(returns)
                r_s_values = []
                n_values = []
                
                for window in range(10, n//2, n//20):
                    if window < 10:
                        continue
                    
                    # Calculate R/S for different window sizes
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
                        
                        r = np.max(cumulative_dev) - np.min(cumulative_dev)  # Range
                        s = np.std(segment)  # Standard deviation
                        
                        if s > 0:
                            rs_values.append(r / s)
                    
                    if rs_values:
                        r_s_values.append(np.mean(rs_values))
                        n_values.append(window)
                
                if len(r_s_values) < 3:
                    return 0.5
                
                # Fit log(R/S) vs log(n)
                log_rs = np.log(r_s_values)
                log_n = np.log(n_values)
                
                hurst, _ = np.polyfit(log_n, log_rs, 1)
                return hurst
                
            elif method == 'aggregate':
                # Aggregate variance method
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
            # Clean returns - remove NaN and infinite values
            clean_returns = returns.copy()
            clean_returns = clean_returns[~np.isnan(clean_returns)]
            clean_returns = clean_returns[~np.isinf(clean_returns)]
            clean_returns = clean_returns[np.isfinite(clean_returns)]
            
            if len(clean_returns) < 50:
                return np.std(clean_returns) if len(clean_returns) > 0 else 0.001
            
            # Scale returns for better numerical stability (x100)
            scaled_returns = clean_returns * 100.0
            
            # Check for all zeros or constant values
            if np.std(scaled_returns) < 1e-10:
                return np.std(clean_returns)
            
            # Fit GARCH model
            model = arch_model(scaled_returns, vol='Garch', p=p, q=q, dist='normal')
            result = model.fit(disp='off', show_warning=False, options={'maxiter': 200})
            
            # Get conditional volatility
            conditional_vol = result.conditional_volatility
            
            # Return last value, scaled back
            if len(conditional_vol) > 0:
                return conditional_vol[-1] / 100
            else:
                return np.std(clean_returns)
            
        except Exception as e:
            # Fall back to simple volatility
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
        
        # Calculate regime indicators
        volatility = np.std(returns)
        mean_return = np.mean(returns)
        sharpe = mean_return / volatility if volatility > 0 else 0
        hurst = AdvancedStatisticalAnalyzer.calculate_hurst_exponent(data['close'].values[-500:])
        autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
        
        # Determine regime
        regime_scores = {
            "trending": 0,
            "mean_reverting": 0,
            "random_walk": 0,
            "volatile": 0
        }
        
        # Trending regime (high Hurst, positive autocorr)
        if hurst > 0.6:
            regime_scores["trending"] += 2
        elif hurst > 0.55:
            regime_scores["trending"] += 1
        
        if autocorr > 0.1:
            regime_scores["trending"] += 1
        
        # Mean reverting regime (low Hurst, negative autocorr)
        if hurst < 0.4:
            regime_scores["mean_reverting"] += 2
        elif hurst < 0.45:
            regime_scores["mean_reverting"] += 1
        
        if autocorr < -0.1:
            regime_scores["mean_reverting"] += 1
        
        # Volatile regime
        vol_threshold = np.percentile(np.abs(returns), 75)
        if volatility > vol_threshold * 1.5:
            regime_scores["volatile"] += 2
        
        # Random walk (Hurst ~ 0.5, low autocorr)
        if 0.45 <= hurst <= 0.55 and abs(autocorr) < 0.05:
            regime_scores["random_walk"] += 2
        
        # Determine winning regime
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
        
        # Sort returns
        sorted_returns = np.sort(returns)
        n = len(sorted_returns)
        
        # Hill estimator for tail index
        k = max(10, int(n * (1 - confidence)))
        tail_data = sorted_returns[-k:]
        
        if len(tail_data) < 10:
            return {"error": "Insufficient tail data"}
        
        # Calculate Hill estimator
        log_tail = np.log(tail_data / tail_data[0])
        hill_estimator = 1 / np.mean(log_tail) if np.mean(log_tail) > 0 else 0
        
        # Extreme Value Theory metrics
        var_extreme = np.percentile(returns, (1 - confidence) * 100)
        cvar_extreme = np.mean(returns[returns <= var_extreme])
        
        # Tail dependence (simplified)
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
            # Calculate returns for different time periods
            returns_1 = data['close'].pct_change(1).dropna()
            returns_5 = data['close'].pct_change(5).dropna()
            returns_10 = data['close'].pct_change(10).dropna()
            returns_20 = data['close'].pct_change(20).dropna()
            
            # Align lengths
            min_len = min(len(returns_1), len(returns_5), len(returns_10), len(returns_20))
            returns_1 = returns_1[-min_len:]
            returns_5 = returns_5[-min_len:]
            returns_10 = returns_10[-min_len:]
            returns_20 = returns_20[-min_len:]
            
            # Create correlation matrix
            returns_matrix = np.column_stack([returns_1, returns_5, returns_10, returns_20])
            correlation_matrix = np.corrcoef(returns_matrix.T)
            
            # Eigenvalue decomposition
            eigenvalues, eigenvectors = np.linalg.eig(correlation_matrix)
            
            # Market correlation (first eigenvalue dominance)
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
            # Resample with replacement
            sample_idx = np.random.choice(n, n, replace=True)
            sample = returns[sample_idx]
            
            # Calculate statistics
            bootstrap_stats['mean'].append(np.mean(sample))
            bootstrap_stats['std'].append(np.std(sample))
            
            if np.std(sample) > 0:
                bootstrap_stats['sharpe'].append(np.mean(sample) / np.std(sample) * np.sqrt(252))
            
            bootstrap_stats['var_95'].append(np.percentile(sample, 5))
            bootstrap_stats['skewness'].append(skew(sample))
            bootstrap_stats['kurtosis'].append(kurtosis(sample))
        
        # Calculate confidence intervals
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
        
        # Basic risk metrics
        metrics['volatility'] = np.std(returns)
        metrics['downside_deviation'] = np.std(returns[returns < 0]) if len(returns[returns < 0]) > 0 else 0
        
        # Value at Risk
        metrics['var_95'] = np.percentile(returns, 5)
        metrics['var_99'] = np.percentile(returns, 1)
        
        # Conditional VaR (Expected Shortfall)
        metrics['cvar_95'] = np.mean(returns[returns <= metrics['var_95']]) if len(returns[returns <= metrics['var_95']]) > 0 else metrics['var_95']
        metrics['cvar_99'] = np.mean(returns[returns <= metrics['var_99']]) if len(returns[returns <= metrics['var_99']]) > 0 else metrics['var_99']
        
        # Performance ratios
        if metrics['volatility'] > 0:
            metrics['sharpe'] = np.mean(returns) / metrics['volatility'] * np.sqrt(252)
        else:
            metrics['sharpe'] = 0
        
        if metrics['downside_deviation'] > 0:
            metrics['sortino'] = np.mean(returns) / metrics['downside_deviation'] * np.sqrt(252)
        else:
            metrics['sortino'] = 0
        
        # Maximum Drawdown
        if prices is not None and len(prices) > 0:
            cumulative = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak
            metrics['max_drawdown'] = np.max(drawdown)
            metrics['avg_drawdown'] = np.mean(drawdown[drawdown > 0]) if len(drawdown[drawdown > 0]) > 0 else 0
        else:
            metrics['max_drawdown'] = 0
            metrics['avg_drawdown'] = 0
        
        # Tail risk metrics
        metrics['skewness'] = skew(returns)
        metrics['kurtosis'] = kurtosis(returns)
        metrics['tail_ratio'] = abs(np.mean(returns[returns < np.percentile(returns, 5)])) / \
                                abs(np.mean(returns[returns > np.percentile(returns, 95)])) \
                                if len(returns[returns > np.percentile(returns, 95)]) > 0 else float('inf')
        
        # Risk-adjusted return metrics
        if metrics['max_drawdown'] > 0:
            metrics['calmar'] = np.mean(returns) * 252 / metrics['max_drawdown']
            metrics['recovery_factor'] = np.sum(returns) / metrics['max_drawdown'] if metrics['max_drawdown'] > 0 else float('inf')
        else:
            metrics['calmar'] = float('inf')
            metrics['recovery_factor'] = float('inf')
        
        # Omega ratio
        threshold = 0
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if len(losses) > 0 and np.sum(losses) > 0:
            metrics['omega'] = np.sum(gains) / np.sum(losses)
        else:
            metrics['omega'] = float('inf')
        
        # Ulcer index
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
        
        # Calculate individual position risks
        position_risks = []
        position_values = []
        
        for pos in positions:
            if 'risk_amount' in pos:
                position_risks.append(pos['risk_amount'])
                position_values.append(pos.get('position_value', pos['risk_amount'] * 10))
        
        if not position_risks:
            return {"total_risk": 0, "diversification_benefit": 0}
        
        # Simple sum (undiversified risk)
        undiversified_risk = np.sum(position_risks)
        
        # If we have correlation matrix, calculate diversified risk
        if correlation_matrix is not None and len(correlation_matrix) == len(position_risks):
            # Convert to numpy arrays
            risks = np.array(position_risks)
            corr = np.array(correlation_matrix)
            
            # Calculate portfolio variance
            portfolio_variance = np.dot(risks, np.dot(corr, risks))
            diversified_risk = np.sqrt(portfolio_variance)
        else:
            # Assume zero correlation for conservative estimate
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
# DATA QUALITY CHECKER WITH STATISTICAL TESTS
# ==========================================
class ProfessionalDataQualityChecker:
    """Enhanced data quality validation with statistical tests"""
    
    @staticmethod
    def check_data_quality(df):
        """Comprehensive data quality assessment"""
        if df is None or len(df) == 0:
            return 0.0, {"error": "Empty dataframe"}
        
        scores = []
        diagnostics = {}
        
        # 1. Missing values check
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        missing_score = max(0, 1 - missing_ratio * 5)  # Penalize heavily
        scores.append(missing_score)
        diagnostics['missing_ratio'] = missing_ratio
        
        # 2. Chronological order check
        if 'time' in df.columns:
            is_sorted = df['time'].is_monotonic_increasing
            sort_score = 1.0 if is_sorted else 0.3
            scores.append(sort_score)
            diagnostics['is_chronological'] = bool(is_sorted)
        
        # 3. Price sanity checks
        price_checks = {}
        for price_col in ['open', 'high', 'low', 'close']:
            if price_col in df.columns:
                col_data = df[price_col]
                
                # Check for zero or negative prices
                has_invalid = (col_data <= 0).any()
                price_checks[f'{price_col}_valid'] = not has_invalid
                
                # Check for extreme outliers
                q1 = col_data.quantile(0.01)
                q3 = col_data.quantile(0.99)
                iqr = q3 - q1
                outliers = ((col_data < (q1 - 3 * iqr)) | (col_data > (q3 + 3 * iqr))).sum()
                outlier_ratio = outliers / len(col_data)
                price_checks[f'{price_col}_outliers'] = outlier_ratio
        
        # Calculate price sanity score
        valid_checks = sum(price_checks.values()) if 'close_valid' in price_checks else 0
        total_checks = len([k for k in price_checks.keys() if 'valid' in k])
        price_score = valid_checks / total_checks if total_checks > 0 else 0.5
        
        scores.append(price_score)
        diagnostics['price_checks'] = price_checks
        
        # 4. Volume checks (if available)
        if 'tick_volume' in df.columns:
            volume = df['tick_volume']
            
            # Check for zero volume
            zero_volume_ratio = (volume == 0).sum() / len(volume)
            volume_score = max(0, 1 - zero_volume_ratio * 2)
            
            # Check for volume spikes
            volume_ma = volume.rolling(20).mean()
            volume_ratio = volume / volume_ma
            spike_ratio = (volume_ratio > 5).sum() / len(volume)
            volume_score *= max(0.5, 1 - spike_ratio)
            
            scores.append(volume_score)
            diagnostics['volume_checks'] = {
                'zero_volume_ratio': zero_volume_ratio,
                'spike_ratio': spike_ratio
            }
        
        # 5. Statistical consistency checks
        if 'close' in df.columns and len(df) > 100:
            returns = df['close'].pct_change().dropna()
            
            # Check return distribution
            return_stats = AdvancedStatisticalAnalyzer.analyze_return_distribution(returns.values)
            
            # Score based on statistical sanity
            stat_score = 1.0
            
            # Penalize extreme kurtosis
            if 'kurtosis' in return_stats:
                kurt = return_stats['kurtosis']
                if abs(kurt) > 10:  # Extremely fat tails
                    stat_score *= 0.7
                elif abs(kurt) > 5:  # Fat tails
                    stat_score *= 0.9
            
            # Penalize extreme skewness
            if 'skewness' in return_stats:
                skew_val = return_stats['skewness']
                if abs(skew_val) > 2:  # Highly skewed
                    stat_score *= 0.8
            
            scores.append(stat_score)
            diagnostics['return_stats'] = return_stats
        
        # 6. Gap analysis
        if 'time' in df.columns and len(df) > 10:
            time_diffs = np.diff(df['time'].values)
            avg_time_diff = np.mean(time_diffs)
            time_std = np.std(time_diffs)
            
            # Check for time gaps
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
        
        # 7. Freshness check
        if 'time' in df.columns:
            latest_time = pd.to_datetime(df['time'].max(), unit='s')
            current_time = datetime.now()
            hours_old = (current_time - latest_time).total_seconds() / 3600
            
            # Score based on data freshness
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
        
        # Calculate overall score
        overall_score = np.mean(scores) if scores else 0.5
        
        return overall_score, diagnostics

# ==========================================
# PROFESSIONAL FEATURE ENGINEERING WITH STATISTICS
# ==========================================
class ProfessionalFeatureEngine:
    def __init__(self):
        self.scaler = RobustScaler()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        self.risk_metrics = ProfessionalRiskMetrics()
        self.multi_tf_analyser = MultiTimeframeAnalyser(mt5)
        
    def calculate_features(self, df):
        """Calculate features with advanced statistical analysis"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        
        # ==========================================
        # POINT 1: Technical indicator periods using Config
        # ==========================================
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
        
        # RSI using Config.RSI_PERIOD
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
        
        # MACD (standard periods for MACD)
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
        
        # ==========================================
        # POINT 2: Bollinger Bands using Config.BB_PERIOD and Config.BB_STD
        # ==========================================
        try:
            # FIXED: Use Config values directly
            bb_period = Config.BB_PERIOD  # This is 20 from your Config
            bb_std = Config.BB_STD  # This is 2.0 from your Config
            
            sma_bb = df['close'].rolling(bb_period).mean()
            std_bb = df['close'].rolling(bb_period).std().replace(0, 1)
            df['bb_upper'] = sma_bb + (std_bb * bb_std)
            df['bb_lower'] = sma_bb - (std_bb * bb_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_bb.replace(0, 1)
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower']).replace(0, 1)
        except Exception as e:
            # FIXED: Better error handling
            ProfessionalLogger.log(f"Bollinger Bands calculation error: {str(e)}", "WARNING", "FEATURE_ENGINE")
            df['bb_upper'] = df['close']
            df['bb_lower'] = df['close']
            df['bb_width'] = 0
            df['bb_position'] = 0.5
        
        # ADX if configured
        if hasattr(Config, 'USE_MARKET_REGIME') and Config.USE_MARKET_REGIME:
            try:
                # ADX calculation
                df['plus_dm'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']), 
                                        np.maximum(df['high'] - df['high'].shift(1), 0), 0)
                df['minus_dm'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)), 
                                         np.maximum(df['low'].shift(1) - df['low'], 0), 0)
                
                tr = df['tr'].rolling(Config.ADX_PERIOD).mean()
                plus_di = 100 * (df['plus_dm'].rolling(Config.ADX_PERIOD).mean() / tr.replace(0, 1))
                minus_di = 100 * (df['minus_dm'].rolling(Config.ADX_PERIOD).mean() / tr.replace(0, 1))
                
                dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, 1)
                df['adx'] = dx.rolling(Config.ADX_PERIOD).mean()
                
                # Market regime based on ADX
                df['trend_strength'] = 0  # 0=sideways, 1=trending, 2=strong trend
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
        
        # Volume features
        if 'tick_volume' in df.columns:
            df['volume_sma'] = df['tick_volume'].rolling(20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma'].replace(0, 1)
        else:
            df['volume_sma'] = 0
            df['volume_ratio'] = 1
        
        # Time features with session awareness
        if 'time' in df.columns:
            try:
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                df['hour'] = df['datetime'].dt.hour
                df['day_of_week'] = df['datetime'].dt.dayofweek
                df['day_of_month'] = df['datetime'].dt.day
                df['month'] = df['datetime'].dt.month
                
                # Cyclical encoding
                df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
                df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
                df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
                df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
                df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
                df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
                
                # Session flags based on Config
                if Config.SESSION_AWARE_TRADING:
                    # London session
                    df['in_london_session'] = ((df['hour'] >= Config.LONDON_OPEN_HOUR) & 
                                               (df['hour'] < Config.LONDON_CLOSE_HOUR)).astype(int)
                    # NY session
                    df['in_ny_session'] = ((df['hour'] >= Config.NY_OPEN_HOUR) & 
                                           (df['hour'] < Config.NY_CLOSE_HOUR)).astype(int)
                    # London-NY overlap (most active)
                    df['in_overlap_session'] = ((df['hour'] >= Config.NY_OPEN_HOUR) & 
                                                (df['hour'] < Config.LONDON_CLOSE_HOUR)).astype(int)
                    # Avoid Asian session
                    df['avoid_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 7)).astype(int)
                    
                    # Good trading hours (combine sessions)
                    df['good_trading_hours'] = (df['in_london_session'] | df['in_ny_session']).astype(int)
                    
                    # Avoid Monday first hour and Friday last hours
                    if Config.AVOID_MONDAY_FIRST_HOUR:
                        df['avoid_monday_hour'] = ((df['day_of_week'] == 0) & (df['hour'] < 1)).astype(int)
                    
                    if Config.AVOID_FRIDAY_LAST_HOURS:
                        df['avoid_friday_hours'] = ((df['day_of_week'] == 4) & (df['hour'] >= 20)).astype(int)
                    
            except:
                # Default values on error
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
        
        # Advanced statistical features using Config
        n = len(df)
        
        # GARCH volatility using Config.GARCH_VOL_PERIOD
        df['garch_volatility'] = 0
        df['garch_vol_ratio'] = 0
        garch_window = Config.GARCH_VOL_PERIOD  # FIXED: Use Config directly
        if n > garch_window:
            try:
                window_returns = df['returns'].iloc[-garch_window:].values
                window_returns = window_returns[~np.isnan(window_returns) & ~np.isinf(window_returns)]
                if len(window_returns) > garch_window // 2:
                    garch_vol = self.stat_analyzer.calculate_garch_volatility(
                        window_returns, 
                        p=Config.GARCH_P,  # FIXED: Use Config directly
                        q=Config.GARCH_Q   # FIXED: Use Config directly
                    )
                    df.loc[df.index[-1], 'garch_volatility'] = garch_vol
                    if df['volatility'].iloc[-1] > 0:
                        df.loc[df.index[-1], 'garch_vol_ratio'] = garch_vol / df['volatility'].iloc[-1]
            except Exception as e:
                ProfessionalLogger.log(f"GARCH calculation error: {str(e)}", "WARNING", "FEATURE_ENGINE")
                pass
        
        # ==========================================
        # POINT 3: Hurst exponent thresholds using Config
        # ==========================================
        df['hurst_exponent'] = 0.5
        hurst_window = Config.HURST_WINDOW  # FIXED: Use Config directly
        if n > hurst_window:
            try:
                window_prices = df['close'].iloc[-hurst_window:].values
                hurst = self.stat_analyzer.calculate_hurst_exponent(window_prices)
                df.loc[df.index[-1], 'hurst_exponent'] = hurst
            except Exception as e:
                ProfessionalLogger.log(f"Hurst exponent calculation error: {str(e)}", "WARNING", "FEATURE_ENGINE")
                pass
        
        # Market regime encoding using Config thresholds
        df['regime_encoded'] = 0
        if 'hurst_exponent' in df.columns:
            hurst = df['hurst_exponent'].iloc[-1]
            
            # FIXED: Use Config thresholds directly
            trending_thresh = Config.HURST_TRENDING_THRESHOLD
            meanreverting_thresh = Config.HURST_MEANREVERTING_THRESHOLD
            
            if hurst > trending_thresh:
                df.loc[df.index[-1], 'regime_encoded'] = 2  # Trending
            elif hurst < meanreverting_thresh:
                df.loc[df.index[-1], 'regime_encoded'] = 1  # Mean-reverting
            else:
                df.loc[df.index[-1], 'regime_encoded'] = 0  # Random
        
        # ==========================================
        # POINT 4: VaR confidence levels using Config.VAR_CONFIDENCE
        # ==========================================
        df['var_95'] = 0
        df['cvar_95'] = 0
        df['var_cvar_spread'] = 0
        var_window = Config.VAR_LOOKBACK  # FIXED: Use Config directly
        var_confidence = Config.VAR_CONFIDENCE  # FIXED: Use Config directly (0.99 from your Config)
        if n > var_window:
            try:
                window_returns = df['returns'].iloc[-var_window:].values
                window_returns = window_returns[~np.isnan(window_returns) & ~np.isinf(window_returns)]
                if len(window_returns) > var_window // 2:
                    # Calculate VaR at specified confidence
                    var_percentile = 100 * (1 - var_confidence)  # For 0.99 confidence, this is 1
                    var = np.percentile(window_returns, var_percentile)
                    
                    # Calculate CVaR at specified confidence
                    cvar_confidence = Config.CVAR_CONFIDENCE  # Use CVaR confidence from Config
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
                current_vol = df['volatility'].iloc[-1] * np.sqrt(252)  # Annualized
                
                # FIXED: Use Config directly
                high_vol = Config.HIGH_VOL_THRESHOLD
                normal_vol = Config.NORMAL_VOL_THRESHOLD
                low_vol = Config.LOW_VOL_THRESHOLD
                
                if current_vol > high_vol:
                    df['volatility_regime'] = 2  # High volatility
                elif current_vol < low_vol:
                    df['volatility_regime'] = 0  # Low volatility
                else:
                    df['volatility_regime'] = 1  # Normal volatility
            except:
                df['volatility_regime'] = 1
        
        # Z-Score features
        zscore_window = 50
        if n > zscore_window:
            last_idx = df.index[-1]
            
            # RSI Z-score
            try:
                rsi_mean = df['rsi'].rolling(zscore_window).mean().iloc[-1]
                rsi_std = df['rsi'].rolling(zscore_window).std().iloc[-1]
                if rsi_std > 0:
                    df.loc[last_idx, 'rsi_zscore'] = (df['rsi'].iloc[-1] - rsi_mean) / rsi_std
                else:
                    df.loc[last_idx, 'rsi_zscore'] = 0
            except:
                df.loc[last_idx, 'rsi_zscore'] = 0
            
            # MACD histogram Z-score
            try:
                macd_hist_mean = df['macd_hist'].rolling(zscore_window).mean().iloc[-1]
                macd_hist_std = df['macd_hist'].rolling(zscore_window).std().iloc[-1]
                if macd_hist_std > 0:
                    df.loc[last_idx, 'macd_hist_zscore'] = (df['macd_hist'].iloc[-1] - macd_hist_mean) / macd_hist_std
                else:
                    df.loc[last_idx, 'macd_hist_zscore'] = 0
            except:
                df.loc[last_idx, 'macd_hist_zscore'] = 0
            
            # Volume Z-score if available
            if 'tick_volume' in df.columns:
                try:
                    volume_mean = df['tick_volume'].rolling(zscore_window).mean().iloc[-1]
                    volume_std = df['tick_volume'].rolling(zscore_window).std().iloc[-1]
                    if volume_std > 0:
                        df.loc[last_idx, 'volume_zscore'] = (df['tick_volume'].iloc[-1] - volume_mean) / volume_std
                    else:
                        df.loc[last_idx, 'volume_zscore'] = 0
                    
                    # Volume-price correlation
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
                # Calculate normalized ATR based on expected daily range
                expected_range = Config.EXPECTED_DAILY_RANGE
                current_atr_percent = df['atr_percent'].iloc[-1] * 100  # Convert to percentage
                df['gold_atr_normalized'] = current_atr_percent / (expected_range / df['close'].iloc[-1] * 100)
            except:
                df['gold_atr_normalized'] = 1
        
        # Final cleanup
        all_features = self.get_feature_columns()
        for feature in all_features:
            if feature not in df.columns:
                df[feature] = 0
        
        for col in df.columns:
            if col in all_features:
                df[col] = df[col].replace([np.inf, -np.inf], 0)
                df[col] = df[col].fillna(0)
        
        return df
    
    def create_labels(self, df, forward_bars=3, method='simple'):
        """Create labels using advanced methods"""
        df = df.copy()
        
        # ==========================================
        # POINT 5: Triple barrier labeling using Config
        # ==========================================
        # Use triple barrier if configured
        if hasattr(Config, 'TRIPLE_BARRIER_METHOD') and Config.TRIPLE_BARRIER_METHOD:
            returns = df['returns'].values
            labels = np.zeros(len(df))
            
            # FIXED: Use Config values directly
            upper_barrier = Config.BARRIER_UPPER  # 0.0015 from your Config
            lower_barrier = Config.BARRIER_LOWER  # -0.0008 from your Config
            barrier_time = Config.BARRIER_TIME    # 10 from your Config
            
            # IMPORTANT FIX: Use min(barrier_time, forward_bars) to avoid index errors
            max_lookahead = min(barrier_time, forward_bars, len(df) - 1)
            
            for i in range(len(df) - max_lookahead):
                if i + max_lookahead >= len(df):
                    continue
                
                # Calculate cumulative returns over the lookahead period
                future_returns = np.cumprod(1 + returns[i+1:i+max_lookahead+1]) - 1
                
                # Check if upper or lower barrier is hit
                upper_hit = np.any(future_returns >= upper_barrier)
                lower_hit = np.any(future_returns <= lower_barrier)
                
                if upper_hit and lower_hit:
                    # Both hit (unlikely), check which happens first
                    upper_idx = np.where(future_returns >= upper_barrier)[0]
                    lower_idx = np.where(future_returns <= lower_barrier)[0]
                    if upper_idx[0] < lower_idx[0]:
                        labels[i] = 1  # Upper hit first
                    else:
                        labels[i] = 0  # Lower hit first
                elif upper_hit:
                    labels[i] = 1  # Take profit hit
                elif lower_hit:
                    labels[i] = 0  # Stop loss hit
                else:
                    # Neither hit, decide based on final return
                    labels[i] = 1 if future_returns[-1] > 0 else 0
            
            df['label'] = labels[:len(df)]
        
        else:
            # Simple labeling (fallback)
            df['forward_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
            volatility = df['returns'].rolling(20).std().fillna(0.001)
            
            # Use dynamic threshold or fixed from Config
            if hasattr(Config, 'USE_DYNAMIC_SL_TP') and Config.USE_DYNAMIC_SL_TP:
                # ATR-based dynamic threshold
                atr_percent = df['atr_percent'].rolling(20).mean().fillna(0.001)
                dynamic_threshold = atr_percent * Config.ATR_SL_MULTIPLIER
            else:
                # Fixed threshold from Config
                dynamic_threshold = Config.FIXED_SL_PERCENT
            
            df['label'] = 0
            df.loc[df['forward_return'] > dynamic_threshold, 'label'] = 1
            df.loc[df['forward_return'] < -dynamic_threshold, 'label'] = 0
        
        df = df.dropna(subset=['label'])
        df['label'] = df['label'].astype(int)
        
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
            'bb_width', 'bb_position', 'bb_width_zscore',
            'momentum_5', 'momentum_10', 'roc_10', 'momentum_5_zscore',
            'returns_skew_20', 'returns_kurtosis_20', 'returns_autocorr_1',
            'distance_to_high_20', 'distance_to_low_20', 'high_low_range',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos',
            'volume_ratio', 'volume_zscore', 'volume_price_correlation', 'volume_spike',
            'garch_volatility', 'garch_vol_ratio',
            'hurst_exponent', 'regime_encoded',
            'var_95', 'cvar_95', 'var_cvar_spread'
        ]
        
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
# FIXED PROFESSIONAL ENSEMBLE (Prevents Data Leakage)
# ==========================================

# ==========================================
# PROFESSIONAL ENSEMBLE MODEL (Fixed Diagnostics Structure)
# ==========================================

class ProfessionalEnsemble:
    """Professional ensemble with statistical analysis and strict anti-leakage protocols"""
    
    def __init__(self, trade_memory, feature_engine):
        self.feature_engine = feature_engine
        self.trade_memory = trade_memory
        self.data_quality_checker = ProfessionalDataQualityChecker()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        self.risk_metrics = ProfessionalRiskMetrics()
        
        # Model components
        self.base_models = self._initialize_base_models()
        self.ensemble = self._create_ensemble_structure()
        
        # SCALER MANAGEMENT
        self.final_scaler = None 
        
        # Training state
        self.is_trained = False
        self.last_train_time = None
        self.training_metrics = {}
        self.feature_importance = {}
        self.statistical_analysis = {}
        self.trained_feature_columns = None
        self.fitted_base_models = {}
        
        ProfessionalLogger.log("Ensemble initialized with strict anti-leakage protocols", "INFO", "ENSEMBLE")

    def _initialize_base_models(self):
        """Initialize diverse base models"""
        models = []
        
        # Gradient Boosting
        models.append(('GB', GradientBoostingClassifier(
            n_estimators=100, max_depth=5, learning_rate=0.05, 
            subsample=0.8, min_samples_split=10, min_samples_leaf=5, random_state=42
        )))
        
        # Random Forest
        models.append(('RF', RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=8, 
            min_samples_leaf=4, max_features='sqrt', bootstrap=True, 
            random_state=42, n_jobs=-1
        )))
        
        # Logistic Regression
        models.append(('LR', LogisticRegression(
            penalty='l2', C=1.0, max_iter=1000, 
            random_state=42, class_weight='balanced', solver='liblinear'
        )))
        
        # Neural Network
        models.append(('NN', MLPClassifier(
            hidden_layer_sizes=(50, 25), activation='relu', solver='adam', 
            alpha=0.0001, max_iter=500, random_state=42
        )))
        
        # XGBoost if available
        if XGB_AVAILABLE:
            models.append(('XGB', XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.05, 
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
            )))
            
        return models

    def _create_ensemble_structure(self):
        """Create ensemble structure"""
        return VotingClassifier(
            estimators=[(name, model) for name, model in self.base_models],
            voting='soft',
            n_jobs=-1
        )

    def _prepare_training_data(self, data):
        """Prepare training data (Calculate features & labels)"""
        try:
            # Calculate features
            df_features = self.feature_engine.calculate_features(data)
            
            # Create labels
            df_labeled = self.feature_engine.create_labels(df_features, method='simple')
            df_labeled = df_labeled.dropna(subset=['label'])
            
            # Get feature columns
            all_feature_cols = self.feature_engine.get_feature_columns()
            self.trained_feature_columns = all_feature_cols # Save columns used
            
            # Extract X and y (Unscaled)
            X = df_labeled[all_feature_cols].copy().fillna(0).replace([np.inf, -np.inf], 0)
            
            # Ensure numeric
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
                
            y = df_labeled['label'].astype(int)
            
            return X, y

        except Exception as e:
            ProfessionalLogger.log(f"Error preparing data: {e}", "ERROR", "ENSEMBLE")
            return None, None

    def perform_statistical_analysis(self, data):
        """Perform statistical analysis (Wrapper)"""
        if data is None or len(data) < 100: return {}
        try:
            returns = data['close'].pct_change().dropna().values
            analysis = {
                'return_distribution': self.stat_analyzer.analyze_return_distribution(returns),
                'market_regime': self.stat_analyzer.calculate_market_regime(data),
                'risk_metrics': self.risk_metrics.calculate_risk_metrics(returns)
            }
            self.statistical_analysis = analysis
            return analysis
        except Exception as e:
            ProfessionalLogger.log(f"Stat analysis failed: {e}", "ERROR", "ENSEMBLE")
            return {}

    def train(self, data):
        """Train ensemble with Strict Walk-Forward Optimization (No Leakage)"""
        try:
            if data is None or len(data) < Config.TRAINING_MIN_SAMPLES:
                ProfessionalLogger.log("Insufficient data for training.", "ERROR", "ENSEMBLE")
                return False

            ProfessionalLogger.log(f"Starting training on {len(data)} samples...", "LEARN", "ENSEMBLE")
            
            # 1. Prepare Raw Data
            X_raw, y_raw = self._prepare_training_data(data)
            if X_raw is None: return False

            # 2. Walk-Forward Validation (WFO)
            # We split the indices first, then scale INSIDE the loop
            tscv = TimeSeriesSplit(
                n_splits=Config.WALK_FORWARD_FOLDS, 
                max_train_size=Config.WALK_FORWARD_WINDOW
            )
            
            cv_scores = []
            ProfessionalLogger.log(f"Executing WFO ({Config.WALK_FORWARD_FOLDS} folds)...", "LEARN", "ENSEMBLE")

            for fold, (train_idx, val_idx) in enumerate(tscv.split(X_raw)):
                # Enforce minimum training size
                if len(train_idx) < 100: continue

                # A. Split Data (Raw)
                X_train_fold, X_val_fold = X_raw.iloc[train_idx], X_raw.iloc[val_idx]
                y_train_fold, y_val_fold = y_raw.iloc[train_idx], y_raw.iloc[val_idx]
                
                # B. Fit Scaler ONLY on Training Fold (Prevents Leakage)
                fold_scaler = RobustScaler()
                X_train_scaled = fold_scaler.fit_transform(X_train_fold)
                
                # C. Transform Validation using Training Scaler
                X_val_scaled = fold_scaler.transform(X_val_fold)
                
                # D. Fit & Score
                self.ensemble.fit(X_train_scaled, y_train_fold)
                val_score = self.ensemble.score(X_val_scaled, y_val_fold)
                cv_scores.append(val_score)
                
                ProfessionalLogger.log(f"  Fold {fold+1}: Acc={val_score:.2%}", "LEARN", "ENSEMBLE")

            avg_score = np.mean(cv_scores) if cv_scores else 0.0

            if avg_score < Config.MIN_ACCURACY_THRESHOLD:
                ProfessionalLogger.log(f"⚠ Low WFO Accuracy: {avg_score:.2%}", "WARNING", "ENSEMBLE")

            # 3. Final Training for Live Trading
            # We train on the most recent window allowed by config
            final_window_size = min(len(X_raw), Config.WALK_FORWARD_WINDOW * 2) 
            X_final_raw = X_raw.iloc[-final_window_size:]
            y_final = y_raw.iloc[-final_window_size:]
            
            # Create and Fit the FINAL scaler (this is what we use for live predict)
            self.final_scaler = RobustScaler()
            X_final_scaled = self.final_scaler.fit_transform(X_final_raw)
            
            ProfessionalLogger.log(f"Fitting final model on last {len(X_final_raw)} bars...", "LEARN", "ENSEMBLE")
            self.ensemble.fit(X_final_scaled, y_final)
            
            # Store fitted base models for individual confidence checks
            if hasattr(self.ensemble, 'named_estimators_'):
                self.fitted_base_models = self.ensemble.named_estimators_
            
            # Update state
            self.is_trained = True
            self.last_train_time = datetime.now()
            self.training_metrics = {'avg_cv_score': avg_score}
            
            ProfessionalLogger.log(f"✅ Training Complete | WFO Accuracy: {avg_score:.2%}", "SUCCESS", "ENSEMBLE")
            return True
            
        except Exception as e:
            ProfessionalLogger.log(f"Training error: {str(e)}", "ERROR", "ENSEMBLE")
            import traceback
            traceback.print_exc()
            return False

    def predict(self, df):
        """Make prediction using the final fitted scaler"""
        if not self.is_trained or self.final_scaler is None:
            ProfessionalLogger.log("Model not trained.", "WARNING", "ENSEMBLE")
            return None, 0.0, None, {}
        
        try:
            # 1. Calculate features
            df_feat = self.feature_engine.calculate_features(df)
            
            # 2. Extract Feature Vector (Raw)
            # Ensure we use exactly the same columns as training
            if self.trained_feature_columns is None:
                self.trained_feature_columns = self.feature_engine.get_feature_columns()
                
            X_raw = df_feat[self.trained_feature_columns].iloc[-1:].fillna(0).replace([np.inf, -np.inf], 0)
            
            # 3. Scale using the Saved Final Scaler
            X_scaled = self.final_scaler.transform(X_raw)
            
            # 4. Predict
            prediction = self.ensemble.predict(X_scaled)[0]
            proba = self.ensemble.predict_proba(X_scaled)[0]
            confidence = np.max(proba)
            
            # Extract features dict for logging
            features = {col: float(X_raw[col].iloc[0]) for col in self.trained_feature_columns}
            
            # 5. Get Sub-model Predictions (for Agreement Logic)
            sub_preds = {}
            if self.fitted_base_models:
                 for name, model in self.fitted_base_models.items():
                     try:
                         sub_p = model.predict(X_scaled)[0]
                         sub_preds[name] = {'prediction': int(sub_p)}
                     except: pass
            
            # 6. Basic Validation
            validation = self._validate_prediction(prediction, confidence, features)
            if not validation['is_valid']:
                ProfessionalLogger.log(f"Validation failed: {validation['reason']}", "WARNING", "ENSEMBLE")
                return None, 0.0, None, {}

            return prediction, confidence, features, sub_preds

        except Exception as e:
            ProfessionalLogger.log(f"Prediction error: {e}", "ERROR", "ENSEMBLE")
            return None, 0.0, None, {}

    def _validate_prediction(self, prediction, confidence, features):
        """Quick validation check"""
        rsi = features.get('rsi_normalized', 0) * 50 + 50
        if prediction == 1 and rsi > 70:
            return {'is_valid': False, 'reason': f"Buy Signal but RSI Overbought ({rsi:.1f})"}
        if prediction == 0 and rsi < 30: 
             pass
        return {'is_valid': True, 'reason': None}

    def should_retrain(self):
        """Check if retraining is needed"""
        if not self.last_train_time: return True
        hours_since = (datetime.now() - self.last_train_time).total_seconds() / 3600
        return hours_since >= Config.RETRAIN_HOURS

    def get_diagnostics(self):
        """Get comprehensive model diagnostics matching Engine expectation"""
        # This structure matches what ProfessionalTradingEngine expects
        return {
            'training_status': {
                'is_trained': self.is_trained,
                'last_train_time': self.last_train_time,
                'training_metrics': self.training_metrics
            },
            'statistical_analysis': self.statistical_analysis,
            'feature_analysis': {
                'top_features': [] # Placeholder to avoid errors if accessed
            },
            'model_info': {
                'base_models': [name for name, _ in self.base_models],
                'ensemble_type': 'voting'
            }
        }
# ==========================================
# SMART ORDER EXECUTOR
# ==========================================

# ==========================================
# SMART ORDER EXECUTOR (Corrected)
# ==========================================
class SmartOrderExecutor:
    """Intelligent order execution with modification capabilities"""

    def execute_trade(self, symbol, order_type, volume, entry_price, sl, tp, magic, comment=""):
        """
        Executes a trade with the specified parameters.
        Must define 'symbol' as the first argument to match the Engine call.
        """
        # 1. Get Symbol Info
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            ProfessionalLogger.log(f"Symbol {symbol} not found", "ERROR", "EXECUTOR")
            return None

        # 2. Strict Volume Normalization
        step = symbol_info.volume_step
        min_vol = symbol_info.volume_min
        max_vol = symbol_info.volume_max
        
        if step > 0:
            volume = round(volume / step) * step
        
        volume = max(min_vol, min(volume, max_vol))
        
        decimals = 2
        if step < 0.01: decimals = 3
        if step == 1.0: decimals = 0
        volume = round(volume, decimals)

        # 3. Check Free Margin
        account = mt5.account_info()
        if account:
            margin_required = (volume * entry_price * symbol_info.trade_contract_size) / account.leverage
            if account.margin_free < margin_required:
                ProfessionalLogger.log(f"Insufficient Margin: Need ${margin_required:.2f}, Have ${account.margin_free:.2f}", "ERROR", "EXECUTOR")
                return None

        # 4. Prepare Request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": float(entry_price),
            "sl": float(sl),
            "tp": float(tp),
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # 5. Execute with Retries
        for i in range(3):
            result = mt5.order_send(request)
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                ProfessionalLogger.log(f"✅ Order Executed: #{result.order} | {symbol} | Vol: {volume}", "SUCCESS", "EXECUTOR")
                return result
            elif result.retcode in [mt5.TRADE_RETCODE_REQUOTE, mt5.TRADE_RETCODE_PRICE_OFF]:
                time.sleep(0.5)
                # Update price for requote
                tick = mt5.symbol_info_tick(symbol)
                if tick: 
                    request['price'] = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
            else:
                ProfessionalLogger.log(f"❌ Order Failed: {result.comment} ({result.retcode})", "ERROR", "EXECUTOR")
                break
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
            ProfessionalLogger.log(f"🔄 Position #{ticket} Modified | New SL: {new_sl}", "SUCCESS", "EXECUTOR")
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
        
        # Determine opposite order type
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



# ==========================================
# MULTI-TIMEFRAME ANALYSER CLASS
# ==========================================
class MultiTimeframeAnalyser:
    """
    Advanced Multi-Timeframe Analysis for XAUUSD
    - Analyzes M5, M15, H1 timeframes for signal alignment
    - Implements weighted consensus voting
    - Provides trend filtering and entry precision
    """
    
    def __init__(self, mt5_connection):
        self.mt5 = mt5_connection
        self.config = Config
        
        # Timeframe mapping
        self.timeframe_map = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1
        }
        
        # Analysis cache
        self.analysis_cache = {}
        self.cache_expiry = 60  # Cache for 60 seconds
        
    def fetch_multi_timeframe_data(self, symbol, bars_needed=500):
        """
        Fetch synchronized data across all configured timeframes
        Returns: dict with timeframe as key and DataFrame as value
        """
        mt5_data = {}
        current_time = datetime.now()
        
        for tf_name, tf_value in self.timeframe_map.items():
            # Check cache first
            cache_key = f"{symbol}_{tf_name}"
            if (cache_key in self.analysis_cache and 
                (current_time - self.analysis_cache[cache_key]['timestamp']).seconds < self.cache_expiry):
                mt5_data[tf_name] = self.analysis_cache[cache_key]['data']
                continue
            
            # Fetch fresh data
            rates = self.mt5.copy_rates_from_pos(symbol, tf_value, 0, bars_needed)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['datetime'] = pd.to_datetime(df['time'], unit='s')
                
                # Cache the data
                self.analysis_cache[cache_key] = {
                    'data': df,
                    'timestamp': current_time
                }
                mt5_data[tf_name] = df
            else:
                ProfessionalLogger.log(f"Failed to fetch {tf_name} data", "WARNING", "MULTI_TF")
        
        return mt5_data
    
    def calculate_timeframe_features(self, df):
        """
        Calculate key features for a single timeframe
        """
        features = {}
        
        if df is None or len(df) < 20:
            return features
        
        # Price features
        features['close'] = df['close'].iloc[-1]
        features['returns_5'] = df['close'].iloc[-1] / df['close'].iloc[-5] - 1 if len(df) >= 5 else 0
        features['returns_20'] = df['close'].iloc[-1] / df['close'].iloc[-20] - 1 if len(df) >= 20 else 0
        
        # Trend indicators
        features['ema_fast'] = df['close'].ewm(span=8).mean().iloc[-1]
        features['ema_slow'] = df['close'].ewm(span=21).mean().iloc[-1]
        features['trend_direction'] = 1 if features['ema_fast'] > features['ema_slow'] else -1
        features['trend_strength'] = abs(features['ema_fast'] - features['ema_slow']) / features['close']
        
        # Momentum
        features['rsi'] = self._calculate_rsi(df['close'], period=14)
        features['momentum'] = df['close'].iloc[-1] / df['close'].iloc[-10] - 1 if len(df) >= 10 else 0
        
        # Volatility
        returns = df['close'].pct_change().dropna()
        features['volatility'] = returns.std() * np.sqrt(252) if len(returns) > 0 else 0
        
        # Support/Resistance levels
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
        """
        Analyze alignment across timeframes
        Returns: dict with alignment scores and consensus signal
        """
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
        
        # Analyze each timeframe
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
        
        # Calculate weighted consensus
        weighted_signal = np.average(signals, weights=weights)
        analysis['consensus_signal'] = 1 if weighted_signal > 0.5 else 0 if weighted_signal < -0.5 else 0.5
        
        # Calculate alignment score (percentage of timeframes agreeing)
        signal_directions = [1 if s > 0 else -1 if s < 0 else 0 for s in signals]
        if len(signal_directions) > 1:
            agreement = sum(1 for i in range(len(signal_directions)) 
                          for j in range(i+1, len(signal_directions)) 
                          if signal_directions[i] * signal_directions[j] > 0)
            total_pairs = len(signal_directions) * (len(signal_directions) - 1) / 2
            analysis['alignment_score'] = agreement / total_pairs if total_pairs > 0 else 0
        
        # Apply trend filter (H1 trend dominates)
        if 'H1' in analysis['timeframes']:
            h1_trend = analysis['timeframes']['H1']['features']['trend_direction']
            if self.config.LONG_TIMEFRAME_TREND_FILTER:
                # Only allow trades in direction of H1 trend
                if analysis['consensus_signal'] == 1 and h1_trend < 0:
                    analysis['trend_filter'] = -1  # Filtered out
                elif analysis['consensus_signal'] == 0 and h1_trend > 0:
                    analysis['trend_filter'] = -1  # Filtered out
                else:
                    analysis['trend_filter'] = 1  # Passed filter
        
        # Calculate confidence
        alignment_bonus = 1.0 if analysis['alignment_score'] >= self.config.TIMEFRAME_ALIGNMENT_THRESHOLD else 0.5
        trend_bonus = 1.0 if analysis['trend_filter'] == 1 else 0.3
        analysis['confidence'] = min(1.0, alignment_bonus * trend_bonus * 0.8)
        
        return analysis
    
    def _generate_timeframe_signal(self, features, timeframe_name):
        """
        Generate trading signal for a single timeframe
        Returns: 1 (buy), 0 (sell), or 0.5 (neutral)
        """
        if not features:
            return 0.5
        
        signal_score = 0
        
        # Price position scoring
        price_pos = features.get('price_position', 0.5)
        if price_pos < 0.3:
            signal_score += 1  # Near support - bullish
        elif price_pos > 0.7:
            signal_score -= 1  # Near resistance - bearish
        
        # Trend scoring
        trend = features.get('trend_direction', 0)
        signal_score += trend
        
        # RSI scoring
        rsi = features.get('rsi', 50)
        if rsi < 30:
            signal_score += 1  # Oversold - bullish
        elif rsi > 70:
            signal_score -= 1  # Overbought - bearish
        
        # Momentum scoring
        momentum = features.get('momentum', 0)
        signal_score += 1 if momentum > 0.005 else -1 if momentum < -0.005 else 0
        
        # Timeframe-specific adjustments
        if timeframe_name == 'M5':
            signal_score *= 0.8  # Less weight to M5 noise
        elif timeframe_name == 'H1':
            signal_score *= 1.2  # More weight to H1 trend
        
        # Normalize to -1 to 1 range
        normalized_score = max(-1, min(1, signal_score / 4))
        
        # Convert to signal
        if normalized_score > 0.2:
            return 1  # Buy
        elif normalized_score < -0.2:
            return 0  # Sell
        else:
            return 0.5  # Neutral
    
    def get_multi_timeframe_recommendation(self, symbol):
        """
        Get comprehensive multi-timeframe trading recommendation
        """
        if not self.config.MULTI_TIMEFRAME_ENABLED:
            return None
        
        # Fetch multi-timeframe data
        multi_tf_data = self.fetch_multi_timeframe_data(symbol)
        
        if not multi_tf_data:
            return None
        
        # Analyze alignment
        analysis = self.analyze_alignment(multi_tf_data)
        
        if not analysis:
            return None
        
        # Generate recommendation
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
        
        # Determine final recommendation
        if analysis['alignment_score'] >= self.config.TIMEFRAME_ALIGNMENT_THRESHOLD and analysis['trend_filter'] == 1:
            if analysis['consensus_signal'] == 1:
                recommendation['recommendation'] = 'STRONG_BUY' if analysis['confidence'] > 0.7 else 'BUY'
                recommendation['recommendation_strength'] = analysis['confidence']
            elif analysis['consensus_signal'] == 0:
                recommendation['recommendation'] = 'STRONG_SELL' if analysis['confidence'] > 0.7 else 'SELL'
                recommendation['recommendation_strength'] = analysis['confidence']
        elif analysis['alignment_score'] >= 0.5:
            recommendation['recommendation'] = 'WEAK_BUY' if analysis['consensus_signal'] == 1 else 'WEAK_SELL' if analysis['consensus_signal'] == 0 else 'HOLD'
            recommendation['recommendation_strength'] = analysis['confidence'] * 0.5
        else:
            recommendation['recommendation'] = 'HOLD'
            recommendation['recommendation_strength'] = 0
        
        # Log the analysis
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
        
        # Detailed logging in debug mode
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
            
        return stats

# ==========================================
# PROFESSIONAL TRADING ENGINE WITH STATISTICAL ANALYSIS
# ==========================================
# ==========================================
# PROFESSIONAL TRADING ENGINE (Fixed Config Usage)
# ==========================================
# ==========================================
# PROFESSIONAL TRADING ENGINE (Adaptive Cashout Enabled)
# ==========================================
class ProfessionalTradingEngine:
    """Main professional trading engine with advanced statistical analysis and adaptive management"""
    
    def __init__(self):
        self.trade_memory = ProfessionalTradeMemory()
        self.feature_engine = ProfessionalFeatureEngine()
        self.order_executor = SmartOrderExecutor() # Ensure this is the updated version
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        self.risk_metrics = ProfessionalRiskMetrics()
        
        # Initialize model with statistical analysis
        self.model = ProfessionalEnsemble(self.trade_memory, self.feature_engine)
        
        # NEW: Initialize Adaptive Exit Manager
        self.exit_manager = AdaptiveExitManager(self.order_executor)
        
        self.connected = False
        self.active_positions = {}
        self.iteration = 0
        self.last_analysis_time = None
        self.last_regime = None
        
        # Performance tracking
        self.equity_curve = []
        self.returns_series = []
        self.risk_metrics_history = []
        self.fitted_base_models = None
        
        ProfessionalLogger.log("Professional Trading Engine initialized with Adaptive Exit Manager", "INFO", "ENGINE")

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
        
        # FIXED: Use Config.LOOKBACK_BARS
        data = self.get_historical_data(bars=Config.LOOKBACK_BARS)
        
        if data is not None and len(data) > 500:
            # Perform comprehensive statistical analysis
            analysis = self.model.perform_statistical_analysis(data)
            
            # Log key findings
            if analysis:
                ProfessionalLogger.log("Initial market analysis complete", "SUCCESS", "ENGINE")
                
                # Store for reference
                self.initial_analysis = analysis
                
                # Extract key insights
                self._extract_market_insights(analysis)
            else:
                ProfessionalLogger.log("Initial analysis returned no results", "WARNING", "ENGINE")
        else:
            ProfessionalLogger.log("Insufficient data for initial analysis", "WARNING", "ENGINE")

    def _extract_market_insights(self, analysis):
        """Extract and log key market insights"""
        insights = []
        
        # Market regime insight
        if 'market_regime' in analysis:
            mr = analysis['market_regime']
            regime = mr.get('regime', 'unknown')
            confidence = mr.get('confidence', 0)
            
            if confidence > 0.7:
                insights.append(f"Market is in {regime} regime (confidence: {confidence:.0%})")
            
            # Trading implications based on regime
            if regime == 'trending':
                insights.append("Consider trend-following strategies with wider stops")
            elif regime == 'mean_reverting':
                insights.append("Consider mean-reversion strategies with tight stops")
            elif regime == 'volatile':
                insights.append("High volatility - consider reducing position sizes")
        
        # Risk insights
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
                if max_dd > 0.1:  # 10% drawdown
                    insights.append(f"Historical max drawdown: {max_dd:.1%} - adjust risk accordingly")
        
        # Tail risk insights
        if 'tail_risk' in analysis:
            tr = analysis['tail_risk']
            if 'tail_index' in tr:
                tail_idx = tr['tail_index']
                if tail_idx < 2:
                    insights.append(f"Fat tails detected (index: {tail_idx:.2f}) - consider tail risk hedging")
        
        # Log insights
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
            positions = [] # Handle None return
        
        current_tickets = [pos.ticket for pos in positions]
        
        # Check which positions are no longer open
        for ticket in list(self.active_positions.keys()):
            if ticket not in current_tickets:
                # Position was closed
                trade_data = self.active_positions[ticket]
                
                # Get the signal from trade_data
                signal = trade_data.get('signal', 1)  # Default to BUY (1) if not found
                
                # Get deal information
                from_date = datetime.now() - timedelta(days=1)
                deals = mt5.history_deals_get(from_date, datetime.now())
                
                if deals:
                    for deal in deals:
                        if deal.position_id == ticket:
                            # Calculate returns
                            open_price = trade_data['open_price']
                            close_price = deal.price
                            returns = (close_price - open_price) / open_price if signal == 1 else (open_price - close_price) / open_price
                            
                            outcome = {
                                'profit': deal.profit,
                                'close_price': deal.price,
                                'close_time': int(deal.time),
                                'status': 'closed',
                                'returns': returns,
                                'duration': int(deal.time) - trade_data['open_time']
                            }
                            trade_data.update(outcome)
                            
                            # Add to trade memory
                            self.trade_memory.add_trade(trade_data)
                            
                            # Update returns series for statistical analysis
                            self.returns_series.append(returns)
                            
                            # Log closure
                            profit_loss = "profit" if deal.profit > 0 else "loss"
                            ProfessionalLogger.log(f"Trade #{ticket} closed with {profit_loss} | P/L: ${deal.profit:.2f}", 
                                                 "SUCCESS" if deal.profit > 0 else "WARNING", "ENGINE")
                            break
                
                del self.active_positions[ticket]

    def run_periodic_tasks(self):
        """Run periodic maintenance and analysis tasks"""
        self.iteration += 1
        
        # Check closed positions
        self.check_closed_positions()
        
        # Update performance metrics every 10 iterations
        if self.iteration % 10 == 0:
            self.update_performance_metrics()
        
        # Perform statistical analysis periodically
        if self.last_analysis_time is None or \
           (datetime.now() - self.last_analysis_time).total_seconds() > 3600:  # Every hour
            self.perform_periodic_analysis()
            self.last_analysis_time = datetime.now()
        
        # Retrain model if needed
        if self.model.should_retrain():
            ProfessionalLogger.log("🔄 Periodic model retraining...", "LEARN", "ENGINE")
            
            # FIXED: Use Config.LOOKBACK_BARS
            data = self.get_historical_data(bars=Config.LOOKBACK_BARS)
            
            if data is not None:
                success = self.model.train(data)
                if success:
                    ProfessionalLogger.log("✅ Model retraining successful", "SUCCESS", "ENGINE")
                else:
                    ProfessionalLogger.log("❌ Model retraining failed", "WARNING", "ENGINE")
        
        # Print status periodically
        if self.iteration % 30 == 0:
            self.print_status()

    def perform_periodic_analysis(self):
        """Perform periodic statistical analysis"""
        ProfessionalLogger.log("🔄 Running periodic statistical analysis...", "ANALYSIS", "ENGINE")
        
        # FIXED: Use Config.LOOKBACK_BARS or safe max
        analysis_bars = min(Config.LOOKBACK_BARS, 5000) 
        data = self.get_historical_data(bars=analysis_bars)
        
        if data is not None and len(data) > 500:
            # Perform analysis
            analysis = self.model.perform_statistical_analysis(data)
            
            # Extract current market insights
            if analysis:
                current_regime = analysis.get('market_regime', {}).get('regime', 'unknown')
                regime_confidence = analysis.get('market_regime', {}).get('confidence', 0)
                
                if regime_confidence > 0.7:
                    ProfessionalLogger.log(f"Current Market Regime: {current_regime} (confidence: {regime_confidence:.0%})", 
                                         "ANALYSIS", "ENGINE")
                
                # Check for regime changes
                if hasattr(self, 'last_regime') and self.last_regime != current_regime:
                    ProfessionalLogger.log(f"⚠ Market regime changed from {self.last_regime} to {current_regime}", 
                                         "WARNING", "ENGINE")
                
                self.last_regime = current_regime

    def update_performance_metrics(self):
        """Update and calculate performance metrics"""
        if not self.connected:
            return
        
        account = mt5.account_info()
        if account:
            # Update equity curve
            self.equity_curve.append(account.equity)
            
            # Keep only recent history
            if len(self.equity_curve) > 1000:
                self.equity_curve = self.equity_curve[-1000:]
            
            # Calculate risk metrics if we have enough data
            if len(self.returns_series) >= 20:
                recent_returns = self.returns_series[-20:]
                risk_metrics = self.risk_metrics.calculate_risk_metrics(recent_returns)
                self.risk_metrics_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'metrics': risk_metrics
                })
                
                # Keep only recent history
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
            
            # Calculate simple P/L if we have trades
            stats = self.trade_memory.get_statistical_summary()
            
            status_msg = (f"Status | Price: {price:.2f} | Positions: {positions} | "
                         f"Equity: ${account.equity:.2f}")
            
            if stats and stats.get('total_trades', 0) > 0:
                status_msg += f" | Trades: {stats['total_trades']} | Win Rate: {stats.get('win_rate', 0):.1%}"
            
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
        ProfessionalLogger.log(f"Profit Std Dev: ${stats['std_profit']:.2f}", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Profit Factor: {stats['profit_factor']:.2f}", "PERFORMANCE", "ENGINE")
        
        # Risk metrics if available
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
        
        # Statistical insights
        if hasattr(self, 'initial_analysis'):
            ia = self.initial_analysis
            if 'market_regime' in ia:
                mr = ia['market_regime']
                ProfessionalLogger.log(f"Initial Market Analysis: {mr.get('regime', 'unknown')} regime", "PERFORMANCE", "ENGINE")
        
        ProfessionalLogger.log("=" * 70, "PERFORMANCE", "ENGINE")

    def train_initial_model(self):
            """Train initial model with statistical analysis using deep history"""
            ProfessionalLogger.log(f"Loading deep history ({Config.LOOKBACK_BARS} bars) for initial training...", "INFO", "ENGINE")
            
            # FIXED: Use Config.LOOKBACK_BARS
            data = self.get_historical_data(bars=Config.LOOKBACK_BARS)
            
            if data is not None:
                data_len = len(data)
                ProfessionalLogger.log(f"Retrieved {data_len} bars from MT5", "DATA", "ENGINE")

                # Check if we satisfy the training minimums
                if data_len >= Config.TRAINING_MIN_SAMPLES:
                    ProfessionalLogger.log(f"Initializing Walk-Forward Optimization (Window: {Config.WALK_FORWARD_WINDOW}, Folds: {Config.WALK_FORWARD_FOLDS})...", "LEARN", "ENSEMBLE")
                    
                    # Perform statistical analysis before training
                    analysis = self.model.perform_statistical_analysis(data)
                    
                    # Train model
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

    def execute_trade(self, signal, confidence, df_current, features, model_details):
        """Execute a trade based on signal"""
        try:
            symbol = Config.SYMBOL
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                ProfessionalLogger.log("Cannot get current price", "ERROR", "EXECUTOR")
                return None
            
            # Determine entry price
            if signal == 1:  # BUY
                entry_price = tick.ask
                order_type = mt5.ORDER_TYPE_BUY
            else:  # SELL
                entry_price = tick.bid
                order_type = mt5.ORDER_TYPE_SELL
            
            # Calculate position size based on risk
            account = mt5.account_info()
            if not account:
                ProfessionalLogger.log("Cannot get account info", "ERROR", "EXECUTOR")
                return None

            # --- AGGREGATE RISK CHECK ---
            positions = mt5.positions_get(symbol=symbol)
            current_total_risk = 0
            
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                ProfessionalLogger.log(f"Symbol {symbol} info not found", "ERROR", "EXECUTOR")
                return None
                
            # Improved Contract Size Fallback (XAUUSD specific)
            contract_size = getattr(symbol_info, 'trade_contract_size', 100000)
            if not contract_size or contract_size == 0:
                if "XAU" in symbol.upper() or "GOLD" in symbol.upper():
                    contract_size = 100
                else:
                    contract_size = 100000
            
            if positions:
                for pos in positions:
                    if pos.sl > 0:
                        risk = abs(pos.price_open - pos.sl) * pos.volume * contract_size
                        current_total_risk += risk
            
            max_total_risk = account.equity * Config.MAX_TOTAL_RISK_PERCENT
            available_risk = max_total_risk - current_total_risk
            
            if available_risk <= 0:
                ProfessionalLogger.log(f"Aggregate Risk Limit Reached: ${current_total_risk:.2f} >= ${max_total_risk:.2f}", "RISK", "ENGINE")
                return None

            # Calculate base risk for this trade
            risk_amount = account.equity * Config.RISK_PERCENT
            
            # Cap risk to available aggregate room
            if risk_amount > available_risk:
                ProfessionalLogger.log(f"Scaling down risk to fit aggregate limit: {risk_amount:.2f} -> {available_risk:.2f}", "RISK", "ENGINE")
                risk_amount = available_risk

            # Calculate stop loss and take profit
            atr = features.get('atr_percent', 0.001) * entry_price
            if atr == 0:
                atr = entry_price * 0.002  # Default 0.2%
            
            if signal == 1:  # BUY
                stop_loss = entry_price - (atr * Config.ATR_SL_MULTIPLIER)
                take_profit = entry_price + (atr * Config.ATR_TP_MULTIPLIER)
            else:  # SELL
                stop_loss = entry_price + (atr * Config.ATR_SL_MULTIPLIER)
                take_profit = entry_price - (atr * Config.ATR_TP_MULTIPLIER)
            
            # Calculate position size (Dynamic Volume Sizing)
            sl_distance = abs(entry_price - stop_loss)
            
            if sl_distance > 0:
                # Volume = Risk / (StopDistance * ContractSize)
                position_size = risk_amount / (sl_distance * contract_size)
                
                # Confidence Scaling: Size * (Confidence / 0.8)
                scaling_factor = confidence / 0.8
                position_size = position_size * scaling_factor
                
                # Round to allowed lot size
                if hasattr(symbol_info, 'volume_step'):
                    step = symbol_info.volume_step
                    position_size = round(position_size / step) * step
                
                # Ensure within limits
                if hasattr(symbol_info, 'volume_min'):
                    position_size = max(position_size, symbol_info.volume_min)
                if hasattr(symbol_info, 'volume_max'):
                    position_size = min(position_size, symbol_info.volume_max)
            
            # Execute the order
            result = self.order_executor.execute_trade(
                symbol=symbol,
                order_type=order_type,
                volume=position_size,
                entry_price=entry_price,
                sl=stop_loss,
                tp=take_profit,
                magic=Config.MAGIC_NUMBER,
                comment=f"AutoTrade_{signal}_{confidence:.2f}"
            )
            
            if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                # Store trade information
                trade_data = {
                    'ticket': result.order,
                    'symbol': symbol,
                    'signal': signal,
                    'type': 'BUY' if signal == 1 else 'SELL',
                    'volume': position_size,
                    'open_price': entry_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'open_time': int(time.time()),
                    'confidence': confidence,
                    'features': features,
                    'model_details': model_details,
                    'status': 'open'
                }
                
                self.active_positions[result.order] = trade_data
                
                ProfessionalLogger.log(f"✅ Trade #{result.order} opened successfully | "
                                     f"{'BUY' if signal == 1 else 'SELL'} {position_size:.2f} lots at {entry_price:.2f}", 
                                     "SUCCESS", "ENGINE")
            
            return result
            
        except Exception as e:
            ProfessionalLogger.log(f"Error executing trade: {str(e)}", "ERROR", "ENGINE")
            import traceback
            traceback.print_exc()
            return None

    def run(self):
        """Main execution method"""
        print("\n" + "=" * 70)
        print("🤖 PROFESSIONAL MT5 ALGORITHMIC TRADING SYSTEM")
        print("📊 Advanced Statistical Analysis | Live Trading")
        print("=" * 70 + "\n")
        
        ProfessionalLogger.log("Starting professional trading system with statistical analysis...", "INFO", "ENGINE")
        
        # Connect to MT5
        if not self.connect_mt5():
            return
        
        # Train initial model
        self.train_initial_model()
        
        # Start live trading
        self.run_live_trading()

    def run_live_trading(self):
        """Run live trading with statistical monitoring and multi-timeframe analysis"""
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        ProfessionalLogger.log("STARTING LIVE TRADING WITH STATISTICAL MONITORING", "TRADE", "ENGINE")
        ProfessionalLogger.log(f"Multi-Timeframe Analysis: {'ENABLED' if Config.MULTI_TIMEFRAME_ENABLED else 'DISABLED'}", "INFO", "ENGINE")
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        
        try:
            while True:
                self.run_periodic_tasks()
                
                # Ensure sufficient buffer for Feature Engineering
                required_lookback = max(500, Config.TREND_MA * 2) 
                
                rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, required_lookback)
                if rates is None or len(rates) < Config.TREND_MA + 10:
                    ProfessionalLogger.log("Failed to get sufficient rates, retrying...", "WARNING", "ENGINE")
                    time.sleep(10)
                    continue
                
                df_current = pd.DataFrame(rates)
                
                # ==========================================
                # MULTI-TIMEFRAME ANALYSIS SECTION
                # ==========================================
                multi_tf_signal = None
                multi_tf_confidence = 0
                multi_tf_alignment = 0
                min_confidence_override = Config.MIN_CONFIDENCE
                min_agreement_override = Config.MIN_ENSEMBLE_AGREEMENT
                
                if Config.MULTI_TIMEFRAME_ENABLED:
                    try:
                        # Get multi-timeframe recommendation
                        mtf_recommendation = self.multi_tf_analyser.get_multi_timeframe_recommendation(Config.SYMBOL)
                        
                        if mtf_recommendation:
                            # Extract multi-TF signals
                            multi_tf_signal = mtf_recommendation.get('consensus_signal')
                            multi_tf_confidence = mtf_recommendation.get('confidence', 0)
                            multi_tf_alignment = mtf_recommendation.get('alignment_score', 0)
                            trend_filter_passed = mtf_recommendation.get('trend_filter_passed', True)
                            
                            # Log multi-TF analysis
                            recommendation = mtf_recommendation.get('recommendation', 'HOLD')
                            ProfessionalLogger.log(
                                f"Multi-TF: {recommendation} | "
                                f"Align: {multi_tf_alignment:.0%} | "
                                f"Conf: {multi_tf_confidence:.0%} | "
                                f"Trend Filter: {'PASS' if trend_filter_passed else 'FAIL'}",
                                "ANALYSIS", "MULTI_TF"
                            )
                            
                            # Adjust execution thresholds based on multi-TF analysis
                            if not trend_filter_passed:
                                ProfessionalLogger.log("Trade blocked by H1 trend filter", "WARNING", "MULTI_TF")
                                signal = None  # Block all trades
                            
                            elif multi_tf_alignment < Config.TIMEFRAME_ALIGNMENT_THRESHOLD:
                                # Weak alignment - require higher confidence
                                min_confidence_override = Config.MIN_CONFIDENCE * 1.3
                                min_agreement_override = Config.MIN_ENSEMBLE_AGREEMENT * 1.2
                                ProfessionalLogger.log(
                                    f"Low alignment ({multi_tf_alignment:.0%} < {Config.TIMEFRAME_ALIGNMENT_THRESHOLD:.0%}) - "
                                    f"raising thresholds: Conf>{min_confidence_override:.0%}, Agree>{min_agreement_override:.0%}",
                                    "WARNING", "MULTI_TF"
                                )
                            
                            elif recommendation in ['STRONG_BUY', 'STRONG_SELL']:
                                # Strong signal - can be more lenient
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
                signal, confidence, features, model_details = self.model.predict(df_current)
                
                # ==========================================
                # ADAPTIVE EXIT LOGIC
                # ==========================================
                df_features = self.feature_engine.calculate_features(df_current)
                
                if self.active_positions:
                    self.exit_manager.manage_positions(
                        df_features, 
                        self.active_positions, 
                        signal, 
                        confidence
                    )
                
                # ==========================================
                # SIGNAL VALIDATION & PROCESSING
                # ==========================================
                if signal is None:
                    # No valid signal
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
                    time.sleep(60)
                    continue
                
                # Calculate model agreement
                agreement = 0
                if model_details:
                    predictions = [m['prediction'] for m in model_details.values() 
                                if m['prediction'] != -1]
                    if predictions:
                        agreement = predictions.count(signal) / len(predictions)
                
                # Apply multi-TF signal validation if enabled
                if Config.MULTI_TIMEFRAME_ENABLED and multi_tf_signal is not None:
                    # Check if multi-TF confirms the signal
                    if multi_tf_signal != 0.5:  # Not neutral
                        signal_match = (signal == 1 and multi_tf_signal > 0.6) or \
                                    (signal == 0 and multi_tf_signal < 0.4)
                        
                        if not signal_match:
                            ProfessionalLogger.log(
                                f"Model signal ({'BUY' if signal == 1 else 'SELL'}) "
                                f"rejected by multi-TF consensus ({multi_tf_signal:.2f})",
                                "WARNING", "MULTI_TF"
                            )
                            # Don't execute, wait for next cycle
                            time.sleep(60)
                            continue
                
                # Calculate combined confidence (model + multi-TF)
                combined_confidence = confidence
                if Config.MULTI_TIMEFRAME_ENABLED and multi_tf_confidence > 0:
                    # Weighted average: 70% model, 30% multi-TF
                    combined_confidence = (confidence * 0.7) + (multi_tf_confidence * 0.3)
                
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
                if features:
                    key_features = {
                        'rsi': features.get('rsi_normalized', 0) * 50 + 50,
                        'volatility': features.get('volatility', 0),
                        'regime': features.get('regime_encoded', 0),
                        'atr_percent': features.get('atr_percent', 0)
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
                
                # Check if all conditions are met
                if (combined_confidence >= min_confidence_override and 
                    agreement >= min_agreement_override):
                    
                    # Additional multi-TF validation if enabled
                    if Config.MULTI_TIMEFRAME_ENABLED:
                        if multi_tf_alignment >= Config.TIMEFRAME_ALIGNMENT_THRESHOLD:
                            execute_trade = True
                            execution_reason = "Strong multi-TF alignment"
                        elif combined_confidence > (min_confidence_override * 1.5):
                            # Very high confidence can override weak alignment
                            execute_trade = True
                            execution_reason = f"Very high confidence ({combined_confidence:.1%})"
                        else:
                            execution_reason = f"Low multi-TF alignment ({multi_tf_alignment:.0%})"
                    else:
                        execute_trade = True
                        execution_reason = "Standard model signal"
                
                # Execute trade if conditions are met
                if execute_trade:
                    ProfessionalLogger.log(
                        f"🎯 {execution_reason} - Executing {signal_type} signal! | "
                        f"Combined Confidence: {combined_confidence:.1%}",
                        "SUCCESS", "ENGINE"
                    )
                    
                    # Add multi-TF data to model_details for tracking
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
                    
                    # Execute trade
                    self.execute_trade(signal, combined_confidence, df_current, features, model_details)
                else:
                    if self.iteration % 10 == 0:  # Log every 10 iterations when not executing
                        ProfessionalLogger.log(
                            f"Signal rejected | Reason: {execution_reason} | "
                            f"Conf: {combined_confidence:.1%} (need {min_confidence_override:.1%}) | "
                            f"Agree: {agreement:.0%} (need {min_agreement_override:.0%})",
                            "INFO", "ENGINE"
                        )
                
                # Update performance metrics
                self.update_performance_metrics()
                
                # Adaptive sleep based on market conditions
                sleep_time = 60  # Default 1 minute
                
                if Config.MULTI_TIMEFRAME_ENABLED:
                    # Adjust sleep time based on volatility and alignment
                    if features and 'volatility' in features:
                        vol = features['volatility']
                        if vol > 0.015:  # High volatility
                            sleep_time = 30  # Check more frequently
                        elif vol < 0.005:  # Low volatility
                            sleep_time = 90  # Check less frequently
                    
                    # If we just executed a trade or have active positions, check more frequently
                    if execute_trade or self.active_positions:
                        sleep_time = max(30, sleep_time // 2)
                
                time.sleep(sleep_time)
                
        except KeyboardInterrupt:
            ProfessionalLogger.log("\nShutdown requested by user", "WARNING", "ENGINE")
        except Exception as e:
            ProfessionalLogger.log(f"Unexpected error in live trading: {str(e)}", "ERROR", "ENGINE")
            import traceback
            traceback.print_exc()
        finally:
            # Final performance report
            self.print_performance_report()
            
            mt5.shutdown()
            ProfessionalLogger.log("Disconnected from MT5", "INFO", "ENGINE")
# ==========================================
# ADAPTIVE EXIT MANAGER (Smart Cashout)
# ==========================================
class AdaptiveExitManager:
    """
    Manages open positions dynamically to prevent giving back profits.
    Active monitoring of:
    1. Technical Exhaustion (RSI, Bollinger Bands)
    2. Trend Weakness (ADX drop)
    3. Profit Protection (Trailing Stops)
    4. Time Decay (Stagnation)
    """
    def __init__(self, executor):
        self.executor = executor

    def manage_positions(self, df, active_positions, current_model_signal, current_confidence):
        """
        Main loop to check all active positions against exit logic.
        """
        if not active_positions or df.empty:
            return

        # Get latest market data
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Calculate dynamic metrics
        atr = latest.get('atr', 0)
        close_price = latest['close']
        rsi = latest.get('rsi', 50)
        adx = latest.get('adx', 0)
        
        # Iterate through a copy of keys to avoid modification errors during iteration
        for ticket, trade in list(active_positions.items()):
            
            # Skip if trade is too new (give it breathing room)
            duration_bars = (int(time.time()) - trade['open_time']) / (Config.TIMEFRAME * 60 if hasattr(Config, 'TIMEFRAME') else 900)
            if duration_bars < 2: 
                continue

            symbol = trade['symbol']
            trade_type = trade['type'] # 'BUY' or 'SELL'
            entry_price = trade['open_price']
            current_sl = trade['stop_loss']
            current_tp = trade['take_profit']
            
            # 1. MODEL INVALIDATION CHECK
            # If the model explicitly predicts the opposite with high confidence, exit immediately.
            model_flip = False
            if trade_type == 'BUY' and current_model_signal == 0 and current_confidence > 0.65:
                ProfessionalLogger.log(f"📉 Model flipped to BEARISH (Conf: {current_confidence:.2f}) - Exiting BUY #{ticket}", "EXIT", "MANAGER")
                self.executor.close_position(ticket, symbol)
                continue
            elif trade_type == 'SELL' and current_model_signal == 1 and current_confidence > 0.65:
                ProfessionalLogger.log(f"📈 Model flipped to BULLISH (Conf: {current_confidence:.2f}) - Exiting SELL #{ticket}", "EXIT", "MANAGER")
                self.executor.close_position(ticket, symbol)
                continue

            # 2. PROFIT PROTECTION (Smart Trailing)
            # Calculate current floating profit in points
            if trade_type == 'BUY':
                profit_points = close_price - entry_price
                distance_to_sl = close_price - current_sl
            else:
                profit_points = entry_price - close_price
                distance_to_sl = current_sl - close_price

            # Define R (Initial Risk)
            initial_risk = abs(entry_price - current_sl)
            if initial_risk == 0: initial_risk = atr # Safety
            
            r_multiple = profit_points / initial_risk

            new_sl = current_sl
            sl_changed = False

            # --- Logic A: Breakeven at 1R ---
            if r_multiple > 1.0:
                # Move to Breakeven + small buffer
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

            # --- Logic B: Tight Trailing at 2R+ (Don't be greedy) ---
            if r_multiple > 2.0:
                # Trail at 1 ATR distance (tight)
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

            # --- Logic C: Technical Exhaustion (Overvaluation) ---
            # If RSI is screaming overbought/oversold, tighten SL dramatically
            is_exhausted = False
            if trade_type == 'BUY' and rsi > 75:
                is_exhausted = True
            elif trade_type == 'SELL' and rsi < 25:
                is_exhausted = True
            
            if is_exhausted:
                # Tighten to candle low/high
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

            # --- Logic D: Trend Death (ADX Drop) ---
            # If we are in a profit but ADX drops below 20, the trend is likely dead.
            if profit_points > 0 and adx < 20 and prev['adx'] > 20:
                ProfessionalLogger.log(f"💤 Trend Dying (ADX < 20) - Closing #{ticket} to free capital", "EXIT", "MANAGER")
                self.executor.close_position(ticket, symbol)
                continue

            # Apply SL modifications if needed
            if sl_changed:
                self.executor.modify_position(ticket, symbol, new_sl, current_tp)
                # Update local memory
                trade['stop_loss'] = new_sl

# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    """Main entry point"""
    # Add diagnostic information
    ProfessionalLogger.log("Starting professional trading system...", "INFO", "ENGINE")
    
    # Test MT5 connection and symbol info
    if not mt5.initialize():
        ProfessionalLogger.log("MT5 initialization failed", "ERROR", "ENGINE")
        return
    
    # Get symbol information
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
    
    # Continue normal startup
    time.sleep(2)
    engine = ProfessionalTradingEngine()
    engine.run()

if __name__ == "__main__":
    main()