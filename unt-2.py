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
    TIMEFRAME = mt5.TIMEFRAME_M15  # 15-min optimal for intraday gold trading
    
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
    ATR_TP_MULTIPLIER = 3.0  # 3.0x ATR for take profit (2:1 R:R)
    
    # Minimum Risk/Reward
    MIN_RR_RATIO = 2.0  # Minimum 2:1 reward:risk
    
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
# PROFESSIONAL ENSEMBLE MODEL WITH STATISTICAL ANALYSIS
# ==========================================
class ProfessionalEnsemble:
    """Professional ensemble with statistical analysis"""
    
    def __init__(self, trade_memory, feature_engine):
        self.feature_engine = feature_engine
        self.trade_memory = trade_memory
        self.data_quality_checker = ProfessionalDataQualityChecker()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        self.risk_metrics = ProfessionalRiskMetrics()
        
        # Model components
        self.base_models = self._initialize_base_models()
        self.ensemble = self._create_ensemble_structure()
        self.scaler = RobustScaler()
        
        # Training state
        self.is_trained = False
        self.last_train_time = None
        self.training_metrics = {}
        self.feature_importance = {}
        self.statistical_analysis = {}
        self.trained_feature_columns = None  # Store feature columns used during training
        
        ProfessionalLogger.log("Ensemble initialized with advanced statistical analysis", "INFO", "ENSEMBLE")
    
    def _initialize_base_models(self):
        """Initialize diverse base models"""
        models = []
        
        # Gradient Boosting
        models.append(('GB', GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=42
        )))
        
        # Random Forest
        models.append(('RF', RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )))
        
        # Logistic Regression
        models.append(('LR', LogisticRegression(
            penalty='l2',
            C=1.0,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )))
        
        # Neural Network
        models.append(('NN', MLPClassifier(
            hidden_layer_sizes=(50, 25),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            max_iter=500,
            random_state=42
        )))
        
        # XGBoost if available
        if XGB_AVAILABLE:
            models.append(('XGB', XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1
            )))
            
        return models
    
    def _create_ensemble_structure(self):
        """Create ensemble structure with proper cross-validation"""
        # Always use VotingClassifier to avoid cross_val_predict issues
        return VotingClassifier(
            estimators=[(name, model) for name, model in self.base_models],
            voting='soft',
            n_jobs=-1
        )
    
    def _prepare_training_data(self, data):
        """Prepare training data with robust feature handling"""
        if data is None or len(data) < 100:
            return None, None
        
        try:
            # Calculate features
            df_features = self.feature_engine.calculate_features(data)
            
            # Create labels
            df_labeled = self.feature_engine.create_labels(df_features, method='simple')
            
            # Remove rows with missing labels
            df_labeled = df_labeled.dropna(subset=['label'])
            
            if len(df_labeled) < 50:
                ProfessionalLogger.log(f"Insufficient labeled data: {len(df_labeled)} samples", "WARNING", "ENSEMBLE")
                return None, None
            
            # Get all feature columns
            all_feature_cols = self.feature_engine.get_feature_columns()
            
            # Ensure all features exist in the dataframe
            for col in all_feature_cols:
                if col not in df_labeled.columns:
                    df_labeled[col] = 0
            
            # Select only the features we need
            X = df_labeled[all_feature_cols].copy()
            
            # Clean the data
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
            
            # Ensure all values are finite
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            y = df_labeled['label'].astype(int)
            
            # Store the feature columns used
            self.trained_feature_columns = all_feature_cols
            
            ProfessionalLogger.log(f"Prepared training data: {len(X)} samples, {len(X.columns)} features", 
                                 "DATA", "ENSEMBLE")
            
            return X, y
            
        except Exception as e:
            ProfessionalLogger.log(f"Error preparing training data: {str(e)}", "ERROR", "ENSEMBLE")
            import traceback
            traceback.print_exc()
            return None, None
    
    def perform_statistical_analysis(self, data):
        """Perform comprehensive statistical analysis on data"""
        ProfessionalLogger.log("Performing advanced statistical analysis...", "STATISTICS", "ENSEMBLE")
        
        analysis_results = {}
        
        if data is None or len(data) < 100:
            ProfessionalLogger.log("Insufficient data for statistical analysis", "WARNING", "ENSEMBLE")
            return analysis_results
        
        try:
            # Calculate returns
            returns = data['close'].pct_change().dropna().values
            
            if len(returns) < Config.MIN_SAMPLES_FOR_STATS:
                return analysis_results
            
            # 1. Return distribution analysis
            analysis_results['return_distribution'] = self.stat_analyzer.analyze_return_distribution(returns)
            
            # 2. Market regime analysis
            analysis_results['market_regime'] = self.stat_analyzer.calculate_market_regime(data)
            
            # 3. Tail risk analysis
            analysis_results['tail_risk'] = self.stat_analyzer.calculate_tail_risk(returns)
            
            # 4. Risk metrics
            analysis_results['risk_metrics'] = self.risk_metrics.calculate_risk_metrics(returns, data['close'].values)
            
            # 5. Bootstrap analysis for confidence intervals
            analysis_results['bootstrap'] = self.stat_analyzer.bootstrap_analysis(returns, min(Config.BOOTSTRAP_SAMPLES, 500))
            
            # Log key findings
            self._log_statistical_analysis(analysis_results)
            
            return analysis_results
            
        except Exception as e:
            ProfessionalLogger.log(f"Statistical analysis error: {str(e)}", "ERROR", "ENSEMBLE")
            return analysis_results
    
    def _log_statistical_analysis(self, analysis):
        """Log key statistical findings"""
        if not analysis:
            return
        
        ProfessionalLogger.log("=" * 60, "STATISTICS", "ENSEMBLE")
        ProfessionalLogger.log("📊 ADVANCED STATISTICAL ANALYSIS", "STATISTICS", "ENSEMBLE")
        
        # Return distribution
        if 'return_distribution' in analysis:
            rd = analysis['return_distribution']
            if 'n_samples' in rd:
                ProfessionalLogger.log(f"Return Distribution (n={rd['n_samples']}):", "STATISTICS", "ENSEMBLE")
                ProfessionalLogger.log(f"  Mean: {rd.get('mean', 0):.6f} | Std: {rd.get('std', 0):.6f}", "STATISTICS", "ENSEMBLE")
                ProfessionalLogger.log(f"  Skew: {rd.get('skewness', 0):.3f} | Kurtosis: {rd.get('kurtosis', 0):.3f}", "STATISTICS", "ENSEMBLE")
                ProfessionalLogger.log(f"  VaR(95%): {rd.get('var_95', 0):.6f} | CVaR(95%): {rd.get('cvar_95', 0):.6f}", "STATISTICS", "ENSEMBLE")
                
                if rd.get('is_normal', False):
                    ProfessionalLogger.log("  ✓ Returns appear normally distributed", "SUCCESS", "ENSEMBLE")
                else:
                    ProfessionalLogger.log("  ⚠ Returns NOT normally distributed", "WARNING", "ENSEMBLE")
        
        # Market regime
        if 'market_regime' in analysis:
            mr = analysis['market_regime']
            ProfessionalLogger.log(f"Market Regime: {mr.get('regime', 'unknown')} (confidence: {mr.get('confidence', 0):.1%})", "STATISTICS", "ENSEMBLE")
            if 'hurst' in mr:
                ProfessionalLogger.log(f"  Hurst Exponent: {mr['hurst']:.3f}", "STATISTICS", "ENSEMBLE")
        
        ProfessionalLogger.log("=" * 60, "STATISTICS", "ENSEMBLE")
    
    def train(self, data):
            """Train ensemble with Walk-Forward Optimization (WFO) logic"""
            try:
                ProfessionalLogger.log(f"Starting training on {len(data) if data is not None else 0} samples...", "LEARN", "ENSEMBLE")
                
                if data is None or len(data) < Config.TRAINING_MIN_SAMPLES:
                    ProfessionalLogger.log(f"Insufficient data: {len(data) if data is not None else 0} < {Config.TRAINING_MIN_SAMPLES}", "ERROR", "ENSEMBLE")
                    return False
                
                # Perform statistical analysis
                self.statistical_analysis = self.perform_statistical_analysis(data)
                
                # Check data quality
                quality_score, quality_diagnostics = self.data_quality_checker.check_data_quality(data)
                if quality_score < Config.MIN_DATA_QUALITY_SCORE:
                    ProfessionalLogger.log(f"Data quality warning: {quality_score:.2%}", "WARNING", "ENSEMBLE")
                
                # Prepare training data
                X, y = self._prepare_training_data(data)
                if X is None or len(X) < Config.TRAINING_MIN_SAMPLES:
                    ProfessionalLogger.log("Insufficient data after preprocessing", "ERROR", "ENSEMBLE")
                    return False
                
                # Scale features
                X_scaled = self.scaler.fit_transform(X)
                
                # ==========================================
                # UPDATED: Walk-Forward Validation Logic
                # ==========================================
                # We use max_train_size to enforce the "Rolling Window" concept 
                # defined by WALK_FORWARD_WINDOW
                tscv = TimeSeriesSplit(
                    n_splits=Config.WALK_FORWARD_FOLDS, 
                    max_train_size=Config.WALK_FORWARD_WINDOW,
                    gap=0 # Optional: Add gap if you want to simulate deployment delay
                )
                
                cv_scores = []
                
                ProfessionalLogger.log(f"Executing Walk-Forward Validation ({Config.WALK_FORWARD_FOLDS} folds)...", "LEARN", "ENSEMBLE")

                for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
                    # Enforce minimum training size for the fold
                    if len(train_idx) < 100: 
                        continue

                    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Fit the ensemble on this specific window
                    self.ensemble.fit(X_train, y_train)
                    
                    # Evaluate
                    val_score = self.ensemble.score(X_val, y_val)
                    cv_scores.append(val_score)
                    
                    ProfessionalLogger.log(f"  Fold {fold+1}/{Config.WALK_FORWARD_FOLDS}: Window Size={len(train_idx)} | Val Acc={val_score:.2%}", 
                                        "LEARN", "ENSEMBLE")
                
                avg_score = np.mean(cv_scores) if cv_scores else 0
                
                # Check if model meets minimum requirements
                if avg_score < Config.MIN_ACCURACY_THRESHOLD:
                    ProfessionalLogger.log(f"Model failed accuracy threshold: {avg_score:.2%} < {Config.MIN_ACCURACY_THRESHOLD:.2%}", "WARNING", "ENSEMBLE")
                    # We still proceed to fit final model, but warn user
                
                # Final training on the most recent window (Config.WALK_FORWARD_WINDOW)
                # This ensures the model is tuned to the current market regime
                final_window_size = min(len(X_scaled), Config.WALK_FORWARD_WINDOW * 2) # Use double window for final fit for stability
                X_final = X_scaled[-final_window_size:]
                y_final = y.iloc[-final_window_size:]
                
                ProfessionalLogger.log(f"Fitting final model on last {len(X_final)} bars...", "LEARN", "ENSEMBLE")
                self.ensemble.fit(X_final, y_final)
                
                # Store fitted base models
                try:
                    if hasattr(self.ensemble, 'named_estimators_'):
                        self.fitted_base_models = self.ensemble.named_estimators_
                except Exception as e:
                    ProfessionalLogger.log(f"Could not store base models: {e}", "WARNING", "ENSEMBLE")

                # Update feature importance (using the RF model from the ensemble)
                for name, model in self.base_models:
                    if name == 'RF' and hasattr(model, 'feature_importances_'):
                        # We need to access the fitted RF within the ensemble
                        try:
                            fitted_rf = self.ensemble.named_estimators_['RF']
                            self.feature_importance = dict(zip(X.columns, fitted_rf.feature_importances_))
                        except:
                            pass
                        break

                # Update training state
                self.is_trained = True
                self.last_train_time = datetime.now()
                self.training_metrics = {
                    'avg_cv_score': avg_score,
                    'std_cv_score': np.std(cv_scores) if cv_scores else 0,
                    'samples': len(X),
                    'features': len(X.columns),
                    'wfo_window': Config.WALK_FORWARD_WINDOW
                }
                
                ProfessionalLogger.log(f"✅ Training Complete | WFO Accuracy: {avg_score:.2%}", "SUCCESS", "ENSEMBLE")
                return True
                
            except Exception as e:
                ProfessionalLogger.log(f"Training error: {str(e)}", "ERROR", "ENSEMBLE")
                import traceback
                traceback.print_exc()
                return False


    def predict(self, df):
        """Make prediction with statistical validation"""
        if not self.is_trained:
            ProfessionalLogger.log("Model not trained, cannot predict", "WARNING", "ENSEMBLE")
            return None, 0.0, None, {}
        
        try:
            # Calculate features
            df_feat = self.feature_engine.calculate_features(df)
            
            # Get feature columns - use trained features if available
            if self.trained_feature_columns is not None:
                required_features = self.trained_feature_columns
            else:
                required_features = self.feature_engine.get_feature_columns()
            
            # Ensure all required features exist
            for feature in required_features:
                if feature not in df_feat.columns:
                    df_feat[feature] = 0
            
            # Prepare input
            X = df_feat[required_features].iloc[-1:].copy()
            X = X.replace([np.inf, -np.inf], 0)
            X = X.fillna(0)
            
            # Ensure all values are numeric
            for col in X.columns:
                X[col] = pd.to_numeric(X[col], errors='coerce').fillna(0)
            
            X_scaled = self.scaler.transform(X)
            
            # Get prediction from ensemble
            prediction = self.ensemble.predict(X_scaled)[0]
            proba = self.ensemble.predict_proba(X_scaled)[0]
            confidence = np.max(proba)
            
            # Create feature dictionary
            features = {col: float(X[col].iloc[0]) for col in required_features}
            
            # Get sub-model predictions
            sub_preds = {}
            
            # First try to use stored fitted models
            if hasattr(self, 'fitted_base_models') and self.fitted_base_models is not None:
                for name, model in self.fitted_base_models.items():
                    try:
                        sub_pred = model.predict(X_scaled)[0]
                        if hasattr(model, 'predict_proba'):
                            sub_proba = model.predict_proba(X_scaled)[0]
                            sub_conf = np.max(sub_proba)
                            sub_proba_list = sub_proba.tolist()
                        else:
                            sub_conf = 1.0
                            sub_proba_list = None
                        
                        sub_preds[name] = {
                            'prediction': int(sub_pred),
                            'confidence': float(sub_conf),
                            'probabilities': sub_proba_list
                        }
                    except Exception as e:
                        ProfessionalLogger.log(f"Submodel {name} error: {str(e)}", "WARNING", "ENSEMBLE")
                        sub_preds[name] = {'prediction': -1, 'confidence': 0.0, 'probabilities': None}
            
            # If no stored models, try to get from the ensemble
            elif hasattr(self.ensemble, 'named_estimators_'):
                for name, model in self.ensemble.named_estimators_.items():
                    try:
                        sub_pred = model.predict(X_scaled)[0]
                        if hasattr(model, 'predict_proba'):
                            sub_proba = model.predict_proba(X_scaled)[0]
                            sub_conf = np.max(sub_proba)
                            sub_proba_list = sub_proba.tolist()
                        else:
                            sub_conf = 1.0
                            sub_proba_list = None
                        
                        sub_preds[name] = {
                            'prediction': int(sub_pred),
                            'confidence': float(sub_conf),
                            'probabilities': sub_proba_list
                        }
                    except Exception as e:
                        ProfessionalLogger.log(f"Submodel {name} error: {str(e)}", "WARNING", "ENSEMBLE")
                        sub_preds[name] = {'prediction': -1, 'confidence': 0.0, 'probabilities': None}
            
            # If no named_estimators_, try estimators_
            elif hasattr(self.ensemble, 'estimators_'):
                for idx, (name, _) in enumerate(self.base_models):
                    if idx < len(self.ensemble.estimators_):
                        model = self.ensemble.estimators_[idx]
                        try:
                            sub_pred = model.predict(X_scaled)[0]
                            if hasattr(model, 'predict_proba'):
                                sub_proba = model.predict_proba(X_scaled)[0]
                                sub_conf = np.max(sub_proba)
                                sub_proba_list = sub_proba.tolist()
                            else:
                                sub_conf = 1.0
                                sub_proba_list = None
                            
                            sub_preds[name] = {
                                'prediction': int(sub_pred),
                                'confidence': float(sub_conf),
                                'probabilities': sub_proba_list
                            }
                        except Exception as e:
                            ProfessionalLogger.log(f"Submodel {name} error: {str(e)}", "WARNING", "ENSEMBLE")
                            sub_preds[name] = {'prediction': -1, 'confidence': 0.0, 'probabilities': None}
            
            # If we couldn't get any sub-model predictions, create placeholder
            if not sub_preds:
                ProfessionalLogger.log("No sub-model predictions available", "INFO", "ENSEMBLE")
                # Create placeholder predictions
                for name, _ in self.base_models:
                    sub_preds[name] = {
                        'prediction': int(prediction),
                        'confidence': float(confidence),
                        'probabilities': proba.tolist()
                    }
            
            # Validate prediction
            validation = self._validate_prediction(prediction, confidence, features, df_feat)
            
            if not validation['is_valid']:
                ProfessionalLogger.log(f"Prediction validation failed: {validation['reason']}", "WARNING", "ENSEMBLE")
                return None, 0.0, None, {}
            
            return prediction, confidence, features, sub_preds
            
        except Exception as e:
            ProfessionalLogger.log(f"Prediction error: {str(e)}", "ERROR", "ENSEMBLE")
            import traceback
            traceback.print_exc()
            return None, 0.0, None, {}
    
    def _validate_prediction(self, prediction, confidence, features, df_feat):
        """Validate prediction using statistical methods"""
        validation = {
            'is_valid': True,
            'reason': None,
            'checks_passed': 0,
            'total_checks': 0
        }
        
        # 1. Confidence threshold check
        validation['total_checks'] += 1
        if confidence >= Config.MIN_CONFIDENCE:
            validation['checks_passed'] += 1
        else:
            validation['is_valid'] = False
            validation['reason'] = f"Low confidence: {confidence:.2%} < {Config.MIN_CONFIDENCE:.0%}"
            return validation
        
        # 2. Technical indicator alignment
        validation['total_checks'] += 1
        rsi = features.get('rsi_normalized', 0) * 50 + 50  # Denormalize
        
        if prediction == 1:  # Buy signal
            if rsi < 70:  # Not overbought
                validation['checks_passed'] += 1
            else:
                validation['is_valid'] = False
                validation['reason'] = f"Buy signal with overbought RSI: {rsi:.1f}"
                return validation
        else:  # Sell signal
            if rsi > 30:  # Not oversold
                validation['checks_passed'] += 1
            else:
                validation['is_valid'] = False
                validation['reason'] = f"Sell signal with oversold RSI: {rsi:.1f}"
                return validation
        
        # 3. Volatility check
        validation['total_checks'] += 1
        volatility = features.get('volatility', 0)
        if volatility < 0.05:  # Reasonable volatility threshold (5%)
            validation['checks_passed'] += 1
        else:
            ProfessionalLogger.log(f"High volatility detected: {volatility:.4f}", "WARNING", "ENSEMBLE")
            validation['checks_passed'] += 1  # Allow but warn
        
        validation['validation_score'] = validation['checks_passed'] / validation['total_checks'] if validation['total_checks'] > 0 else 0
        
        return validation
    
    def get_diagnostics(self):
        """Get comprehensive model diagnostics"""
        return {
            'training_status': {
                'is_trained': self.is_trained,
                'last_train_time': self.last_train_time.isoformat() if self.last_train_time else None,
                'training_metrics': self.training_metrics
            },
            'statistical_analysis': self.statistical_analysis,
            'feature_analysis': {
                'top_features': sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)[:10] 
                if self.feature_importance else []
            },
            'model_info': {
                'base_models': [name for name, _ in self.base_models],
                'ensemble_type': 'voting'
            }
        }
    
    def should_retrain(self):
        """Determine if retraining is needed based on statistical analysis"""
        if not self.last_train_time:
            return True
        
        # Time-based retraining
        hours_since = (datetime.now() - self.last_train_time).total_seconds() / 3600
        if hours_since >= Config.RETRAIN_HOURS:
            ProfessionalLogger.log(f"Scheduled retraining after {hours_since:.1f} hours", "LEARN", "ENSEMBLE")
            return True
        
        return False

# ==========================================
# SMART ORDER EXECUTOR
# ==========================================
class SmartOrderExecutor:
    """Intelligent order execution"""
    
    def __init__(self):
        self.pending_orders = {}
    
    def execute_trade(self, symbol, order_type, volume, entry_price, sl, tp, magic, comment=""):
        """Execute trade with SL/TP validation"""
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            ProfessionalLogger.log(f"Symbol {symbol} not found", "ERROR", "EXECUTOR")
            return None
        
        # Validate volume using Config
        min_volume = getattr(Config, 'MIN_VOLUME', 0.05)
        max_volume = getattr(Config, 'MAX_VOLUME', 0.20)
        volume_step = getattr(Config, 'VOLUME_STEP', 0.01)
        
        volume = max(min_volume, min(volume, max_volume))
        volume = round(volume / volume_step) * volume_step
        
        # Validate SL/TP Levels using Config
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            ProfessionalLogger.log(f"Cannot get tick data for {symbol}", "ERROR", "EXECUTOR")
            return None
        
        current_price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
        point = symbol_info.point if hasattr(symbol_info, 'point') else 0.01
        digits = symbol_info.digits if hasattr(symbol_info, 'digits') else 2
        
        # Use Config distances
        min_sl_points = getattr(Config, 'MIN_SL_DISTANCE_POINTS', 50)
        max_sl_points = getattr(Config, 'MAX_SL_DISTANCE_POINTS', 300)
        min_tp_points = getattr(Config, 'MIN_TP_DISTANCE_POINTS', 100)
        max_tp_points = getattr(Config, 'MAX_TP_DISTANCE_POINTS', 600)
        
        # Validate SL/TP distances
        if order_type == mt5.ORDER_TYPE_BUY:
            sl_distance = current_price - sl
            tp_distance = tp - current_price
            
            # Ensure minimum distances
            if sl_distance < min_sl_points * point:
                sl = current_price - (min_sl_points * point)
                ProfessionalLogger.log(f"SL adjusted to minimum distance", "WARNING", "EXECUTOR")
            
            if tp_distance < min_tp_points * point:
                tp = current_price + (min_tp_points * point)
                ProfessionalLogger.log(f"TP adjusted to minimum distance", "WARNING", "EXECUTOR")
            
            # Ensure risk/reward ratio
            risk_reward_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
            min_rr = getattr(Config, 'MIN_RR_RATIO', 2.0)
            
            if risk_reward_ratio < min_rr and sl_distance > 0:
                tp = current_price + (sl_distance * min_rr)
                ProfessionalLogger.log(f"TP adjusted to maintain {min_rr}:1 R:R ratio", "WARNING", "EXECUTOR")
                
        else:  # SELL order
            sl_distance = sl - current_price
            tp_distance = current_price - tp
            
            # Ensure minimum distances
            if sl_distance < min_sl_points * point:
                sl = current_price + (min_sl_points * point)
                ProfessionalLogger.log(f"SL adjusted to minimum distance", "WARNING", "EXECUTOR")
            
            if tp_distance < min_tp_points * point:
                tp = current_price - (min_tp_points * point)
                ProfessionalLogger.log(f"TP adjusted to minimum distance", "WARNING", "EXECUTOR")
            
            # Ensure risk/reward ratio
            risk_reward_ratio = tp_distance / sl_distance if sl_distance > 0 else 0
            min_rr = getattr(Config, 'MIN_RR_RATIO', 2.0)
            
            if risk_reward_ratio < min_rr and sl_distance > 0:
                tp = current_price - (sl_distance * min_rr)
                ProfessionalLogger.log(f"TP adjusted to maintain {min_rr}:1 R:R ratio", "WARNING", "EXECUTOR")
        
        # Prepare order request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": current_price,
            "sl": sl,
            "tp": tp,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        ProfessionalLogger.log(f"Order: {symbol} {volume} lots at {current_price:.{digits}f}", "DEBUG", "EXECUTOR")
        ProfessionalLogger.log(f"  SL: {sl:.{digits}f} | TP: {tp:.{digits}f} | R:R: {risk_reward_ratio:.2f}:1", "DEBUG", "EXECUTOR")
        
        # Check spread before entry if configured
        if hasattr(Config, 'CHECK_SPREAD_BEFORE_ENTRY') and Config.CHECK_SPREAD_BEFORE_ENTRY:
            spread = tick.ask - tick.bid
            max_spread = getattr(Config, 'MAX_SPREAD_POINTS', 5) * point
            if spread > max_spread:
                ProfessionalLogger.log(f"Spread too high: {spread/point:.1f} points > {max_spread/point:.1f}", "WARNING", "EXECUTOR")
                return None
        
        # Execute with retries if configured
        max_retries = getattr(Config, 'MAX_RETRIES', 3)
        retry_delay = getattr(Config, 'RETRY_DELAY_MS', 500) / 1000
        
        for attempt in range(max_retries):
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                ProfessionalLogger.log(f"✅ Order executed successfully | Ticket: {result.order}", "SUCCESS", "EXECUTOR")
                return result
            else:
                if attempt < max_retries - 1:
                    ProfessionalLogger.log(f"Order attempt {attempt+1} failed: {result.retcode}, retrying...", "WARNING", "EXECUTOR")
                    time.sleep(retry_delay)
                else:
                    ProfessionalLogger.log(f"Order failed after {max_retries} attempts: {result.retcode} - {result.comment}", "ERROR", "EXECUTOR")
        
        return None

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
class ProfessionalTradingEngine:
    """Main professional trading engine with advanced statistical analysis"""
    
    def __init__(self):
        self.trade_memory = ProfessionalTradeMemory()
        self.feature_engine = ProfessionalFeatureEngine()
        self.order_executor = SmartOrderExecutor()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        self.risk_metrics = ProfessionalRiskMetrics()
        
        # Initialize model with statistical analysis
        self.model = ProfessionalEnsemble(self.trade_memory, self.feature_engine)
        
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
        ProfessionalLogger.log("Professional Trading Engine initialized with advanced statistical analysis", "INFO", "ENGINE")
    
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
        
        # Get historical data for analysis
        data = self.get_historical_data(bars=2000)
        
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
            return
        
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
            data = self.get_historical_data(bars=2000)
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
        
        # Get recent data
        data = self.get_historical_data(bars=1000)
        
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
            
            # FIXED: Use Config.LOOKBACK_BARS instead of hardcoded 3000
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
                        metrics = self.model.get_diagnostics()['training_status']['training_metrics']
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
        """Run live trading with statistical monitoring"""
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        ProfessionalLogger.log("STARTING LIVE TRADING WITH STATISTICAL MONITORING", "TRADE", "ENGINE")
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        
        try:
            while True:
                self.run_periodic_tasks()
                
                # Get current data
                rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, 100)
                if rates is None or len(rates) < 50:
                    ProfessionalLogger.log("Failed to get rates, retrying...", "WARNING", "ENGINE")
                    time.sleep(60)
                    continue
                
                df_current = pd.DataFrame(rates)
                
                # Get model prediction with statistical validation
                signal, confidence, features, model_details = self.model.predict(df_current)
                
                if signal is None:
                    # No valid signal
                    if self.iteration % 30 == 0:
                        tick = mt5.symbol_info_tick(Config.SYMBOL)
                        if tick:
                            price = tick.ask
                            positions = self.get_current_positions()
                            ProfessionalLogger.log(f"Waiting for signal | Price: {price:.2f} | Positions: {positions}", 
                                                 "INFO", "ENGINE")
                    time.sleep(60)
                    continue
                
                # Check model agreement
                agreement = 0
                if model_details:
                    predictions = [m['prediction'] for m in model_details.values() 
                                 if m['prediction'] != -1]
                    if predictions:
                        agreement = predictions.count(signal) / len(predictions)
                
                # Log prediction details
                signal_type = "BUY" if signal == 1 else "SELL"
                status_msg = (f"Signal Analysis | {signal_type} | "
                             f"Confidence: {confidence:.1%} | "
                             f"Agreement: {agreement:.0%} | "
                             f"Price: {df_current['close'].iloc[-1]:.2f}")
                ProfessionalLogger.log(status_msg, "ANALYSIS", "ENGINE")
                
                # Log key features
                if features:
                    key_features = {
                        'rsi': features.get('rsi_normalized', 0) * 50 + 50,
                        'volatility': features.get('volatility', 0),
                        'regime': features.get('regime_encoded', 0)
                    }
                    ProfessionalLogger.log(f"Key Features: RSI={key_features['rsi']:.1f} | "
                                         f"Vol={key_features['volatility']:.4f} | "
                                         f"Regime={key_features['regime']}", "DATA", "ENGINE")
                
                # Execute trade if conditions are met
                if (confidence >= Config.MIN_CONFIDENCE and 
                    agreement >= Config.MIN_ENSEMBLE_AGREEMENT):
                    
                    ProfessionalLogger.log(f"🎯 High-confidence {signal_type} signal confirmed!", "SUCCESS", "ENGINE")
                    
                    # Execute trade
                    self.execute_trade(signal, confidence, df_current, features, model_details)
                
                # Update performance metrics
                self.update_performance_metrics()
                
                time.sleep(60)  # Wait 1 minute before next iteration
                
        except KeyboardInterrupt:
            ProfessionalLogger.log("\nShutdown requested by user", "WARNING", "ENGINE")
        except Exception as e:
            ProfessionalLogger.log(f"Unexpected error: {str(e)}", "ERROR", "ENGINE")
            import traceback
            traceback.print_exc()
        finally:
            # Final performance report
            self.print_performance_report()
            
            mt5.shutdown()
            ProfessionalLogger.log("Disconnected from MT5", "INFO", "ENGINE")

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