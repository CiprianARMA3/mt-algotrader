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
    PRIMARY_TIMEFRAME = mt5.TIMEFRAME_M15  # Primary timeframe for trading decisions
    
    # MULTI-TIMEFRAME CONFIGURATION
    MULTI_TIMEFRAME_ENABLED = True
    COMPARISON_TIMEFRAMES = [
        ('M1', mt5.TIMEFRAME_M1),
        ('M5', mt5.TIMEFRAME_M5),
        ('M15', mt5.TIMEFRAME_M15),
        ('M30', mt5.TIMEFRAME_M30),
        ('H1', mt5.TIMEFRAME_H1)
    ]
    
    # Timeframe weights for ensemble decisions
    TIMEFRAME_WEIGHTS = {
        'M1': 0.1,   # 10% weight - noise filtering
        'M5': 0.2,   # 20% weight - short-term momentum
        'M15': 0.35, # 35% weight - primary trading
        'M30': 0.2,  # 20% weight - medium-term trend
        'H1': 0.15   # 15% weight - long-term context
    }
    
    # Timeframe alignment requirements
    TIMEFRAME_ALIGNMENT_THRESHOLD = 0.70  # Need 70% agreement across timeframes
    
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
    MIN_CONFIDENCE = 0.50  # 50% minimum model confidence
    MIN_ENSEMBLE_AGREEMENT = 0.60  # 60% model agreement
    MIN_TIMEFRAME_AGREEMENT = 0.60  # 60% timeframe agreement
    
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
    KELLY_FRACTION = 0.15  # Conservative 15% of Kelly
    USE_HALF_KELLY = True  # Use half-Kelly for extra safety
    
    # Statistical Risk Metrics
    VAR_CONFIDENCE = 0.99  # 99% Value at Risk
    CVAR_CONFIDENCE = 0.99  # 99% Conditional VaR
    MAX_POSITION_CORRELATION = 0.5  # Max correlation between positions
    
    # ==========================================
    # MACHINE LEARNING MODEL PARAMETERS
    # ==========================================
    
    # Data Collection
    LOOKBACK_BARS = 65000  # 65000 bars for stable statistics
    TRAINING_MIN_SAMPLES = 6000  # Minimum samples for reliable training
    VALIDATION_SPLIT = 0.20  # 20% validation set
    
    # Retraining Schedule
    RETRAIN_HOURS = 24  # Retrain every 24 hours
    RETRAIN_ON_PERFORMANCE_DROP = True  # Retrain if performance degrades
    MIN_ACCURACY_THRESHOLD = 0.50  # Retrain if accuracy drops below 50%
    
    # Walk-Forward Optimization
    WALK_FORWARD_WINDOW = 1000  # 1000 bars per window
    WALK_FORWARD_STEP = 100  # 100 bar step (80% overlap)
    WALK_FORWARD_FOLDS = 5  # 5-fold cross-validation
    
    # Feature Engineering Flags
    USE_FRACTIONAL_DIFF = True
    FD_THRESHOLD = 0.4  # Fractional differentiation d=0.4
    USE_TICK_VOLUME_VOLATILITY = True
    TICK_SKEW_LOOKBACK = 50
    
    # Labeling Method
    TRIPLE_BARRIER_METHOD = True  # Use triple-barrier labeling
    BARRIER_UPPER = 0.0030  # 0.30% = $3.00
    BARRIER_LOWER = -0.0020  # -0.20% = $2.00
    BARRIER_TIME = 6  # 6 bars = 90 minutes
    
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
    ATR_PERIOD = 14  # Standard ATR
    RSI_PERIOD = 14  # Standard RSI
    ADX_PERIOD = 14  # Standard ADX
    
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
    CORRELATION_WINDOW = 50  # 50 bars for correlations
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
    MAX_SPREAD_POINTS = 30  # Maximum 30 points ($0.30) spread
    NORMAL_SPREAD_POINTS = 2  # Normal spread should be ~2 points
    
    # Commission
    COMMISSION_PER_LOT = 3.5  # $3.50 per lot (typical)
    
    # ==========================================
    # PERFORMANCE METRICS & MONITORING
    # ==========================================
    
    # Minimum Performance Standards
    MIN_SHARPE_RATIO = 0.8  # Minimum Sharpe ratio
    MIN_PROFIT_FACTOR = 1.5  # Minimum profit factor
    MIN_WIN_RATE = 0.45  # Minimum 45% win rate
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
    INCREASE_SIZE_AFTER_WINS = False  # Don't increase after wins
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
            'ANALYSIS': ProfessionalLogger.COLORS['CYAN'],
            'TIMEFRAME': ProfessionalLogger.COLORS['MAGENTA'],
            'COMPARISON': ProfessionalLogger.COLORS['CYAN']
        }
        color = colors.get(level, ProfessionalLogger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level:8s}{ProfessionalLogger.COLORS['RESET']}] [{component:12s}] {message}", flush=True)

# ==========================================
# MULTI-TIMEFRAME STATISTICAL ANALYZER
# ==========================================
class MultiTimeframeStatisticalAnalyzer:
    """Advanced statistical analysis across multiple timeframes"""
    
    def __init__(self):
        self.timeframe_data = {}
        self.comparison_results = {}
        
    def fetch_multi_timeframe_data(self, symbol, bars_per_tf=1000):
        """Fetch data for all comparison timeframes"""
        data_frames = {}
        
        for tf_name, tf_enum in Config.COMPARISON_TIMEFRAMES:
            try:
                rates = mt5.copy_rates_from_pos(symbol, tf_enum, 0, bars_per_tf)
                if rates is not None and len(rates) > 0:
                    df = pd.DataFrame(rates)
                    df['timeframe'] = tf_name
                    data_frames[tf_name] = df
                    ProfessionalLogger.log(f"Fetched {len(df)} bars for {tf_name}", "DATA", "MTF_ANALYZER")
                else:
                    ProfessionalLogger.log(f"No data for {tf_name}", "WARNING", "MTF_ANALYZER")
            except Exception as e:
                ProfessionalLogger.log(f"Error fetching {tf_name}: {str(e)}", "ERROR", "MTF_ANALYZER")
        
        self.timeframe_data = data_frames
        return data_frames
    
    def calculate_statistical_metrics(self, df):
        """Calculate comprehensive statistical metrics for a timeframe"""
        if df is None or len(df) < 100:
            return None
        
        try:
            # Calculate returns
            returns = df['close'].pct_change().dropna()
            
            if len(returns) < 50:
                return None
            
            # Basic statistics
            stats = {
                'n_samples': len(returns),
                'mean_return': np.mean(returns),
                'median_return': np.median(returns),
                'std_return': np.std(returns),
                'skewness': skew(returns),
                'kurtosis': kurtosis(returns),
                'min_return': np.min(returns),
                'max_return': np.max(returns),
                'range': np.ptp(returns),
                'q1': np.percentile(returns, 25),
                'q3': np.percentile(returns, 75),
                'iqr': np.percentile(returns, 75) - np.percentile(returns, 25),
                'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252 * 96) if np.std(returns) > 0 else 0,
                'sortino': np.mean(returns) / np.std(returns[returns < 0]) * np.sqrt(252 * 96) if len(returns[returns < 0]) > 0 else 0,
                'var_95': np.percentile(returns, 5),
                'var_99': np.percentile(returns, 1),
                'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]) if len(returns[returns <= np.percentile(returns, 5)]) > 0 else np.percentile(returns, 5),
                'cvar_99': np.mean(returns[returns <= np.percentile(returns, 1)]) if len(returns[returns <= np.percentile(returns, 1)]) > 0 else np.percentile(returns, 1),
                'positive_ratio': len(returns[returns > 0]) / len(returns) if len(returns) > 0 else 0,
                'negative_ratio': len(returns[returns < 0]) / len(returns) if len(returns) > 0 else 0,
                'zero_ratio': len(returns[returns == 0]) / len(returns) if len(returns) > 0 else 0
            }
            
            # Volatility metrics
            stats['realized_volatility'] = np.std(returns) * np.sqrt(252 * 96)
            stats['annualized_vol'] = stats['realized_volatility']
            
            # Calculate rolling volatility
            if len(returns) > 100:
                rolling_vol = returns.rolling(20).std().dropna()
                stats['volatility_mean'] = np.mean(rolling_vol)
                stats['volatility_std'] = np.std(rolling_vol)
                stats['volatility_ratio'] = stats['volatility_mean'] / stats['std_return'] if stats['std_return'] > 0 else 1
            
            # Market regime indicators
            hurst = self.calculate_hurst_exponent(df['close'].values)
            stats['hurst_exponent'] = hurst
            
            if hurst > 0.55:
                stats['market_regime'] = 'trending'
            elif hurst < 0.45:
                stats['market_regime'] = 'mean_reverting'
            else:
                stats['market_regime'] = 'random'
            
            # Autocorrelation
            if len(returns) > 20:
                autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1] if len(returns) > 1 else 0
                stats['autocorrelation_lag1'] = autocorr
                
                # Check for momentum/mean-reversion
                if autocorr > 0.1:
                    stats['momentum_bias'] = 'positive'
                elif autocorr < -0.1:
                    stats['momentum_bias'] = 'negative'
                else:
                    stats['momentum_bias'] = 'neutral'
            
            # Price range statistics
            df['range'] = df['high'] - df['low']
            stats['avg_range'] = np.mean(df['range'])
            stats['range_std'] = np.std(df['range'])
            stats['range_to_close'] = stats['avg_range'] / df['close'].mean() if df['close'].mean() > 0 else 0
            
            # Volume analysis (if available)
            if 'tick_volume' in df.columns:
                volume = df['tick_volume']
                stats['avg_volume'] = np.mean(volume)
                stats['volume_std'] = np.std(volume)
                stats['volume_skew'] = skew(volume) if len(volume) > 0 else 0
                stats['volume_kurtosis'] = kurtosis(volume) if len(volume) > 0 else 0
                
                # Volume-price correlation
                if len(returns) > 10:
                    volume_returns_corr = np.corrcoef(volume[-len(returns):], returns)[0, 1]
                    stats['volume_price_correlation'] = volume_returns_corr
            
            return stats
            
        except Exception as e:
            ProfessionalLogger.log(f"Error calculating stats: {str(e)}", "ERROR", "MTF_ANALYZER")
            return None
    
    def calculate_hurst_exponent(self, prices, max_lags=100):
        """Calculate Hurst exponent using R/S method"""
        if len(prices) < 100:
            return 0.5
        
        try:
            lags = range(2, min(max_lags, len(prices)//4))
            tau = []
            lagvec = []
            
            for lag in lags:
                # Calculate R/S for different lags
                rs_values = []
                for i in range(0, len(prices) - lag, lag):
                    segment = prices[i:i+lag]
                    if len(segment) < 2:
                        continue
                    
                    mean_seg = np.mean(segment)
                    deviations = segment - mean_seg
                    z = np.cumsum(deviations)
                    r = np.max(z) - np.min(z)
                    s = np.std(segment)
                    
                    if s > 0:
                        rs_values.append(r / s)
                
                if rs_values:
                    tau.append(np.log(np.mean(rs_values)))
                    lagvec.append(np.log(lag))
            
            if len(tau) < 3:
                return 0.5
            
            # Fit line to log-log plot
            hurst, _ = np.polyfit(lagvec, tau, 1)
            return hurst
            
        except Exception as e:
            ProfessionalLogger.log(f"Hurst calculation error: {str(e)}", "WARNING", "MTF_ANALYZER")
            return 0.5
    
    def compare_timeframes(self):
        """Compare statistical metrics across timeframes"""
        if not self.timeframe_data:
            return None
        
        comparison = {
            'timeframes': {},
            'consistency_metrics': {},
            'alignment_scores': {}
        }
        
        all_stats = {}
        
        # Calculate stats for each timeframe
        for tf_name, df in self.timeframe_data.items():
            stats = self.calculate_statistical_metrics(df)
            if stats:
                all_stats[tf_name] = stats
                comparison['timeframes'][tf_name] = stats
        
        if not all_stats:
            return comparison
        
        # Calculate consistency metrics
        consistency = {}
        
        # Check trend consistency
        trends = {}
        for tf_name, stats in all_stats.items():
            if 'market_regime' in stats:
                trends[tf_name] = stats['market_regime']
        
        # Calculate regime alignment
        regime_counts = {}
        for regime in trends.values():
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        if regime_counts:
            dominant_regime = max(regime_counts, key=regime_counts.get)
            regime_alignment = regime_counts[dominant_regime] / len(trends)
            consistency['regime_alignment'] = regime_alignment
            consistency['dominant_regime'] = dominant_regime
        
        # Check volatility consistency
        volatilities = {}
        for tf_name, stats in all_stats.items():
            if 'realized_volatility' in stats:
                volatilities[tf_name] = stats['realized_volatility']
        
        if volatilities:
            vol_mean = np.mean(list(volatilities.values()))
            vol_std = np.std(list(volatilities.values()))
            consistency['volatility_mean'] = vol_mean
            consistency['volatility_std'] = vol_std
            consistency['volatility_cv'] = vol_std / vol_mean if vol_mean > 0 else 0
        
        # Check return distribution consistency
        returns_mean = {}
        returns_std = {}
        
        for tf_name, stats in all_stats.items():
            if 'mean_return' in stats:
                returns_mean[tf_name] = stats['mean_return']
            if 'std_return' in stats:
                returns_std[tf_name] = stats['std_return']
        
        if returns_mean and returns_std:
            consistency['returns_mean_consistency'] = np.std(list(returns_mean.values())) / abs(np.mean(list(returns_mean.values()))) if np.mean(list(returns_mean.values())) != 0 else 0
            consistency['returns_std_consistency'] = np.std(list(returns_std.values())) / np.mean(list(returns_std.values())) if np.mean(list(returns_std.values())) > 0 else 0
        
        # Calculate timeframe alignment scores
        alignment_scores = {}
        
        # Score based on regime consistency
        if 'regime_alignment' in consistency:
            alignment_scores['regime'] = consistency['regime_alignment']
        
        # Score based on volatility consistency
        if 'volatility_cv' in consistency:
            # Lower CV is better (more consistent)
            volatility_score = max(0, 1 - consistency['volatility_cv'])
            alignment_scores['volatility'] = volatility_score
        
        # Score based on trend direction
        trend_scores = {}
        for tf_name, df in self.timeframe_data.items():
            if len(df) > 20:
                # Simple trend calculation
                returns_20 = df['close'].pct_change(20).iloc[-1] if len(df) > 20 else 0
                trend_scores[tf_name] = 1 if returns_20 > 0 else 0 if returns_20 < 0 else 0.5
        
        if trend_scores:
            trend_alignment = np.mean(list(trend_scores.values()))
            alignment_scores['trend'] = 1 - abs(2 * trend_alignment - 1)  # 1 = perfect alignment, 0 = perfect disagreement
        
        # Overall alignment score
        if alignment_scores:
            weights = {
                'regime': 0.4,
                'volatility': 0.3,
                'trend': 0.3
            }
            
            overall_score = 0
            total_weight = 0
            for component, score in alignment_scores.items():
                weight = weights.get(component, 0)
                overall_score += score * weight
                total_weight += weight
            
            if total_weight > 0:
                alignment_scores['overall'] = overall_score / total_weight
            else:
                alignment_scores['overall'] = 0.5
        
        comparison['consistency_metrics'] = consistency
        comparison['alignment_scores'] = alignment_scores
        
        # Store for future reference
        self.comparison_results = comparison
        
        return comparison
    
    def generate_comparison_report(self):
        """Generate comprehensive comparison report"""
        if not self.comparison_results:
            self.compare_timeframes()
        
        report = []
        
        ProfessionalLogger.log("=" * 80, "COMPARISON", "MTF_ANALYZER")
        ProfessionalLogger.log("📊 MULTI-TIMEFRAME STATISTICAL COMPARISON REPORT", "COMPARISON", "MTF_ANALYZER")
        ProfessionalLogger.log("=" * 80, "COMPARISON", "MTF_ANALYZER")
        
        # Timeframe-specific statistics
        for tf_name, stats in self.comparison_results.get('timeframes', {}).items():
            report.append(f"\n📈 {tf_name} Timeframe:")
            report.append(f"  Samples: {stats.get('n_samples', 0):,}")
            report.append(f"  Mean Return: {stats.get('mean_return', 0):.6f}")
            report.append(f"  Volatility: {stats.get('realized_volatility', 0):.4f}")
            report.append(f"  Sharpe: {stats.get('sharpe', 0):.3f}")
            report.append(f"  Regime: {stats.get('market_regime', 'unknown')} (H={stats.get('hurst_exponent', 0.5):.3f})")
            report.append(f"  VaR(95%): {stats.get('var_95', 0):.6f}")
            report.append(f"  Positive Ratio: {stats.get('positive_ratio', 0):.1%}")
        
        # Consistency metrics
        consistency = self.comparison_results.get('consistency_metrics', {})
        if consistency:
            report.append("\n🔗 CONSISTENCY METRICS:")
            report.append(f"  Dominant Regime: {consistency.get('dominant_regime', 'unknown')}")
            report.append(f"  Regime Alignment: {consistency.get('regime_alignment', 0):.1%}")
            report.append(f"  Volatility CV: {consistency.get('volatility_cv', 0):.3f}")
        
        # Alignment scores
        alignment = self.comparison_results.get('alignment_scores', {})
        if alignment:
            report.append("\n🎯 ALIGNMENT SCORES:")
            for component, score in alignment.items():
                if component != 'overall':
                    report.append(f"  {component.capitalize()}: {score:.3f}")
            
            if 'overall' in alignment:
                overall_score = alignment['overall']
                report.append(f"\n📊 OVERALL ALIGNMENT: {overall_score:.3f}")
                
                if overall_score > 0.7:
                    report.append("  ✅ STRONG TIMEFRAME ALIGNMENT - High confidence")
                elif overall_score > 0.5:
                    report.append("  ⚠ MODERATE ALIGNMENT - Proceed with caution")
                else:
                    report.append("  ❌ POOR ALIGNMENT - Consider avoiding trades")
        
        # Trading implications
        report.append("\n💡 TRADING IMPLICATIONS:")
        
        # Check if we have enough agreement
        if 'overall' in alignment:
            if alignment['overall'] >= Config.MIN_TIMEFRAME_AGREEMENT:
                report.append("  ✓ Timeframes aligned - Trading signals more reliable")
            else:
                report.append("  ✗ Timeframes divergent - Signals less reliable")
        
        # Market regime implications
        dominant_regime = consistency.get('dominant_regime', '')
        if dominant_regime == 'trending':
            report.append("  ✓ Trending market - Trend-following strategies preferred")
        elif dominant_regime == 'mean_reverting':
            report.append("  ✓ Mean-reverting market - Reversal strategies preferred")
        elif dominant_regime == 'random':
            report.append("  ⚠ Random market - Consider reducing position sizes")
        
        ProfessionalLogger.log("\n".join(report), "COMPARISON", "MTF_ANALYZER")
        ProfessionalLogger.log("=" * 80, "COMPARISON", "MTF_ANALYZER")
        
        return "\n".join(report)
    
    def get_trading_recommendation(self):
        """Get trading recommendation based on multi-timeframe analysis"""
        if not self.comparison_results:
            self.compare_timeframes()
        
        recommendation = {
            'trade_signal': 'HOLD',
            'confidence': 0.0,
            'reasoning': [],
            'risk_adjustment': 1.0,
            'timeframe_analysis': {}
        }
        
        alignment = self.comparison_results.get('alignment_scores', {})
        consistency = self.comparison_results.get('consistency_metrics', {})
        
        # Check overall alignment
        overall_alignment = alignment.get('overall', 0.5)
        
        if overall_alignment < Config.MIN_TIMEFRAME_AGREEMENT:
            recommendation['trade_signal'] = 'AVOID'
            recommendation['confidence'] = 0.0
            recommendation['reasoning'].append(f"Poor timeframe alignment ({overall_alignment:.2f})")
            recommendation['risk_adjustment'] = 0.0
            return recommendation
        
        # Analyze trend across timeframes
        trend_signals = {}
        for tf_name, stats in self.comparison_results.get('timeframes', {}).items():
            if 'mean_return' in stats:
                # Simple trend signal based on recent returns
                mean_return = stats['mean_return']
                if mean_return > 0.0001:  # Positive trend
                    trend_signals[tf_name] = 'BULLISH'
                elif mean_return < -0.0001:  # Negative trend
                    trend_signals[tf_name] = 'BEARISH'
                else:
                    trend_signals[tf_name] = 'NEUTRAL'
        
        # Count trend signals
        trend_counts = {
            'BULLISH': 0,
            'BEARISH': 0,
            'NEUTRAL': 0
        }
        
        for signal in trend_signals.values():
            if signal in trend_counts:
                trend_counts[signal] += 1
        
        total_tfs = len(trend_signals)
        if total_tfs > 0:
            bullish_pct = trend_counts['BULLISH'] / total_tfs
            bearish_pct = trend_counts['BEARISH'] / total_tfs
            
            if bullish_pct > 0.6:
                recommendation['trade_signal'] = 'BUY'
                recommendation['confidence'] = min(0.9, bullish_pct)
                recommendation['reasoning'].append(f"Bullish alignment across {bullish_pct:.0%} of timeframes")
            elif bearish_pct > 0.6:
                recommendation['trade_signal'] = 'SELL'
                recommendation['confidence'] = min(0.9, bearish_pct)
                recommendation['reasoning'].append(f"Bearish alignment across {bearish_pct:.0%} of timeframes")
            else:
                recommendation['trade_signal'] = 'HOLD'
                recommendation['confidence'] = 0.3
                recommendation['reasoning'].append(f"Mixed signals: {bullish_pct:.0%} bullish, {bearish_pct:.0%} bearish")
        
        # Adjust risk based on volatility consistency
        vol_cv = consistency.get('volatility_cv', 0)
        if vol_cv > 0.5:
            recommendation['risk_adjustment'] *= 0.5
            recommendation['reasoning'].append(f"High volatility inconsistency (CV: {vol_cv:.2f}) - reducing risk")
        
        # Check market regime
        dominant_regime = consistency.get('dominant_regime', '')
        if dominant_regime == 'trending':
            recommendation['risk_adjustment'] *= 1.2
            recommendation['reasoning'].append("Trending market - increasing position size")
        elif dominant_regime == 'mean_reverting':
            recommendation['risk_adjustment'] *= 0.8
            recommendation['reasoning'].append("Mean-reverting market - reducing position size")
        
        recommendation['timeframe_analysis'] = {
            'alignment_score': overall_alignment,
            'trend_signals': trend_signals,
            'dominant_regime': dominant_regime
        }
        
        return recommendation

# ==========================================
# ADVANCED STATISTICAL ANALYZER (Enhanced)
# ==========================================
class AdvancedStatisticalAnalyzer:
    """Enhanced statistical analysis for MT5 data"""
    
    def __init__(self):
        self.mtf_analyzer = MultiTimeframeStatisticalAnalyzer()
    
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
        
        # Value at Risk and Expected Shortfall
        var_95 = np.percentile(returns, 5)
        var_99 = np.percentile(returns, 1)
        
        stats['var_95'] = var_95
        stats['var_99'] = var_99
        
        stats['cvar_95'] = np.mean(returns[returns <= var_95]) if len(returns[returns <= var_95]) > 0 else stats['var_95']
        stats['cvar_99'] = np.mean(returns[returns <= var_99]) if len(returns[returns <= var_99]) > 0 else stats['var_99']
        
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
    
    def analyze_multi_timeframe(self, symbol):
        """Perform multi-timeframe statistical analysis"""
        ProfessionalLogger.log("Performing multi-timeframe statistical analysis...", "TIMEFRAME", "STATS")
        
        # Fetch data for all timeframes
        timeframe_data = self.mtf_analyzer.fetch_multi_timeframe_data(symbol, bars_per_tf=2000)
        
        if not timeframe_data:
            ProfessionalLogger.log("No timeframe data available", "WARNING", "STATS")
            return None
        
        # Compare timeframes
        comparison = self.mtf_analyzer.compare_timeframes()
        
        # Generate report
        report = self.mtf_analyzer.generate_comparison_report()
        
        # Get trading recommendation
        recommendation = self.mtf_analyzer.get_trading_recommendation()
        
        return {
            'timeframe_data': {k: len(v) for k, v in timeframe_data.items()},
            'comparison': comparison,
            'recommendation': recommendation,
            'report': report
        }
    
    def calculate_market_regime(self, data):
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
        hurst = self.calculate_hurst_exponent_simple(data['close'].values[-500:])
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
    def calculate_hurst_exponent_simple(prices):
        """Simple Hurst exponent calculation"""
        if len(prices) < 100:
            return 0.5
        
        try:
            # Convert to returns
            returns = np.diff(np.log(prices))
            
            # R/S method
            n = len(returns)
            r_s_values = []
            n_values = []
            
            for window in range(10, n//4, n//40):
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
            
            # Fit log(R/S) vs log(n)
            log_rs = np.log(r_s_values)
            log_n = np.log(n_values)
            
            hurst, _ = np.polyfit(log_n, log_rs, 1)
            return hurst
            
        except:
            return 0.5

# ==========================================
# PROFESSIONAL FEATURE ENGINEERING (Enhanced)
# ==========================================
class ProfessionalFeatureEngine:
    def __init__(self):
        self.scaler = RobustScaler()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        
    def calculate_multi_timeframe_features(self, symbol):
        """Calculate features across multiple timeframes"""
        mtf_features = {}
        
        for tf_name, tf_enum in Config.COMPARISON_TIMEFRAMES:
            try:
                # Fetch data for this timeframe
                rates = mt5.copy_rates_from_pos(symbol, tf_enum, 0, 500)
                if rates is None or len(rates) < 100:
                    continue
                
                df = pd.DataFrame(rates)
                
                # Calculate basic features for this timeframe
                features = self.calculate_basic_timeframe_features(df, tf_name)
                mtf_features[tf_name] = features
                
            except Exception as e:
                ProfessionalLogger.log(f"Error calculating {tf_name} features: {str(e)}", "WARNING", "FEATURE_ENGINE")
        
        return mtf_features
    
    def calculate_basic_timeframe_features(self, df, timeframe_name):
        """Calculate basic features for a specific timeframe"""
        if df is None or len(df) < 50:
            return {}
        
        try:
            features = {}
            
            # Price features
            features['close'] = float(df['close'].iloc[-1])
            features['returns_1'] = float(df['close'].pct_change(1).iloc[-1] if len(df) > 1 else 0)
            features['returns_5'] = float(df['close'].pct_change(5).iloc[-1] if len(df) > 5 else 0)
            features['returns_20'] = float(df['close'].pct_change(20).iloc[-1] if len(df) > 20 else 0)
            
            # Volatility
            returns = df['close'].pct_change().dropna()
            if len(returns) > 0:
                features['volatility'] = float(np.std(returns))
                features['realized_vol'] = float(np.std(returns) * np.sqrt(252 * 96))
            
            # Range features
            df['range'] = df['high'] - df['low']
            features['avg_range'] = float(df['range'].mean() if len(df) > 0 else 0)
            features['range_ratio'] = float(df['range'].iloc[-1] / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0)
            
            # Simple moving averages
            if len(df) >= 20:
                features['sma_20'] = float(df['close'].rolling(20).mean().iloc[-1])
                features['price_vs_sma_20'] = float(df['close'].iloc[-1] / features['sma_20'] - 1 if features['sma_20'] > 0 else 0)
            
            if len(df) >= 50:
                features['sma_50'] = float(df['close'].rolling(50).mean().iloc[-1])
                features['price_vs_sma_50'] = float(df['close'].iloc[-1] / features['sma_50'] - 1 if features['sma_50'] > 0 else 0)
            
            # High/Low proximity
            features['distance_to_high_20'] = float((df['high'].rolling(20).max().iloc[-1] - df['close'].iloc[-1]) / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0)
            features['distance_to_low_20'] = float((df['close'].iloc[-1] - df['low'].rolling(20).min().iloc[-1]) / df['close'].iloc[-1] if df['close'].iloc[-1] > 0 else 0)
            
            # Trend indicators
            if len(df) >= 10:
                # Simple trend: 10-bar return
                features['trend_10'] = float(df['close'].iloc[-1] / df['close'].iloc[-10] - 1 if len(df) >= 10 else 0)
                
                # Momentum
                features['momentum_5'] = float(df['close'].iloc[-1] / df['close'].iloc[-5] - 1 if len(df) >= 5 else 0)
            
            # Volume features (if available)
            if 'tick_volume' in df.columns:
                volume = df['tick_volume']
                features['volume'] = float(volume.iloc[-1])
                if len(volume) >= 20:
                    features['volume_sma_20'] = float(volume.rolling(20).mean().iloc[-1])
                    features['volume_ratio'] = float(volume.iloc[-1] / features['volume_sma_20'] if features['volume_sma_20'] > 0 else 1)
            
            # Timeframe-specific metadata
            features['timeframe'] = timeframe_name
            features['sample_size'] = len(df)
            features['timestamp'] = datetime.now().isoformat()
            
            return features
            
        except Exception as e:
            ProfessionalLogger.log(f"Error in timeframe features: {str(e)}", "ERROR", "FEATURE_ENGINE")
            return {}
    
    def calculate_features(self, df):
        """Calculate comprehensive features for primary timeframe"""
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Moving averages
        for period in [Config.FAST_MA, Config.MEDIUM_MA, Config.SLOW_MA]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'].replace(0, 1) - 1
        
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
        
        # RSI
        try:
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(Config.RSI_PERIOD).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(Config.RSI_PERIOD).mean()
            rs = gain / loss.replace(0, 1)
            df['rsi'] = 100 - (100 / (1 + rs))
        except:
            df['rsi'] = 50
        
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
        except:
            df['bb_upper'] = df['close']
            df['bb_lower'] = df['close']
            df['bb_width'] = 0
            df['bb_position'] = 0.5
        
        # Fill NaN values
        df = df.fillna(0)
        
        return df
    
    def get_feature_columns(self):
        """Return list of feature columns"""
        return [
            'returns', 'log_returns', 'hl_ratio', 'co_ratio',
            f'price_to_sma_{Config.FAST_MA}', f'price_to_sma_{Config.MEDIUM_MA}', f'price_to_sma_{Config.SLOW_MA}',
            'atr_percent', 'volatility',
            'rsi', 'macd_hist',
            'bb_width', 'bb_position'
        ]

# ==========================================
# PROFESSIONAL TRADING ENGINE (Enhanced)
# ==========================================
class ProfessionalTradingEngine:
    """Main professional trading engine with multi-timeframe analysis"""
    
    def __init__(self):
        self.trade_memory = ProfessionalTradeMemory()
        self.feature_engine = ProfessionalFeatureEngine()
        self.order_executor = SmartOrderExecutor()
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        
        # Initialize model
        self.model = ProfessionalEnsemble(self.trade_memory, self.feature_engine)
        
        self.connected = False
        self.active_positions = {}
        self.iteration = 0
        self.last_mtf_analysis = None
        
        ProfessionalLogger.log("Professional Trading Engine initialized with multi-timeframe analysis", "INFO", "ENGINE")
    
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
        
        # Perform initial multi-timeframe analysis
        self.perform_initial_mtf_analysis()
        
        return True
    
    def perform_initial_mtf_analysis(self):
        """Perform initial multi-timeframe analysis"""
        ProfessionalLogger.log("Performing initial multi-timeframe analysis...", "ANALYSIS", "ENGINE")
        
        analysis = self.stat_analyzer.analyze_multi_timeframe(Config.SYMBOL)
        
        if analysis:
            self.last_mtf_analysis = analysis
            ProfessionalLogger.log("Initial multi-timeframe analysis complete", "SUCCESS", "ENGINE")
            
            # Log key recommendations
            recommendation = analysis.get('recommendation', {})
            if recommendation:
                ProfessionalLogger.log(f"Initial Recommendation: {recommendation.get('trade_signal', 'HOLD')} "
                                     f"(Confidence: {recommendation.get('confidence', 0):.1%})", "ANALYSIS", "ENGINE")
        else:
            ProfessionalLogger.log("Initial multi-timeframe analysis failed", "WARNING", "ENGINE")
    
    def perform_periodic_mtf_analysis(self):
        """Perform periodic multi-timeframe analysis"""
        ProfessionalLogger.log("🔄 Running periodic multi-timeframe analysis...", "TIMEFRAME", "ENGINE")
        
        analysis = self.stat_analyzer.analyze_multi_timeframe(Config.SYMBOL)
        
        if analysis:
            self.last_mtf_analysis = analysis
            
            # Log key findings
            recommendation = analysis.get('recommendation', {})
            if recommendation:
                signal = recommendation.get('trade_signal', 'HOLD')
                confidence = recommendation.get('confidence', 0)
                
                ProfessionalLogger.log(f"Multi-Timeframe Recommendation: {signal} (Confidence: {confidence:.1%})", 
                                     "TIMEFRAME", "ENGINE")
                
                if signal != 'HOLD' and confidence > Config.MIN_CONFIDENCE:
                    ProfessionalLogger.log(f"✅ Strong {signal} signal across timeframes", "SUCCESS", "ENGINE")
        
        return analysis
    
    def run_periodic_tasks(self):
        """Run periodic maintenance and analysis tasks"""
        self.iteration += 1
        
        # Perform multi-timeframe analysis every 30 minutes
        if self.iteration % 30 == 0:
            self.perform_periodic_mtf_analysis()
        
        # Check closed positions
        self.check_closed_positions()
        
        # Retrain model if needed
        if self.model.should_retrain():
            ProfessionalLogger.log("🔄 Periodic model retraining...", "LEARN", "ENGINE")
            data = self.get_historical_data(bars=2000)
            if data is not None:
                success = self.model.train(data)
                if success:
                    ProfessionalLogger.log("✅ Model retraining successful", "SUCCESS", "ENGINE")
        
        # Print status every 30 iterations
        if self.iteration % 30 == 0:
            self.print_status()
    
    def get_historical_data(self, timeframe=None, bars=None):
        """Get historical data from MT5"""
        if not self.connected:
            return None
        
        if timeframe is None:
            timeframe = Config.PRIMARY_TIMEFRAME
        if bars is None:
            bars = 1000
        
        rates = mt5.copy_rates_from_pos(Config.SYMBOL, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        
        return pd.DataFrame(rates)
    
    def check_closed_positions(self):
        """Check for closed positions"""
        if not self.connected:
            return
        
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        current_tickets = [pos.ticket for pos in positions] if positions else []
        
        for ticket in list(self.active_positions.keys()):
            if ticket not in current_tickets:
                # Position was closed
                trade_data = self.active_positions[ticket]
                
                # Get deal information
                from_date = datetime.now() - timedelta(days=1)
                deals = mt5.history_deals_get(from_date, datetime.now())
                
                if deals:
                    for deal in deals:
                        if deal.position_id == ticket:
                            profit_loss = "profit" if deal.profit > 0 else "loss"
                            ProfessionalLogger.log(f"Trade #{ticket} closed with {profit_loss} | P/L: ${deal.profit:.2f}", 
                                                 "SUCCESS" if deal.profit > 0 else "WARNING", "ENGINE")
                            break
                
                del self.active_positions[ticket]
    
    def print_status(self):
        """Print current status"""
        account = mt5.account_info()
        if not account:
            return
        
        positions = len(self.active_positions)
        tick = mt5.symbol_info_tick(Config.SYMBOL)
        
        if tick:
            price = tick.ask
            
            status_msg = (f"Status | Price: {price:.2f} | Positions: {positions} | "
                         f"Equity: ${account.equity:.2f}")
            
            ProfessionalLogger.log(status_msg, "INFO", "ENGINE")
    
    def train_initial_model(self):
        """Train initial model"""
        ProfessionalLogger.log("Training initial model...", "INFO", "ENGINE")
        
        data = self.get_historical_data(bars=Config.LOOKBACK_BARS)
        
        if data is not None:
            data_len = len(data)
            ProfessionalLogger.log(f"Retrieved {data_len} bars for training", "DATA", "ENGINE")
            
            if data_len >= Config.TRAINING_MIN_SAMPLES:
                success = self.model.train(data)
                if success:
                    ProfessionalLogger.log("✅ Initial model training successful", "SUCCESS", "ENGINE")
                else:
                    ProfessionalLogger.log("❌ Initial model training failed", "WARNING", "ENGINE")
            else:
                ProfessionalLogger.log(f"❌ Insufficient data: {data_len} < {Config.TRAINING_MIN_SAMPLES}", "ERROR", "ENGINE")
        else:
            ProfessionalLogger.log("❌ Failed to retrieve historical data", "ERROR", "ENGINE")
    
    def execute_trade_with_mtf_validation(self, signal, confidence, df_current, features, model_details):
        """Execute trade with multi-timeframe validation"""
        # First, get current multi-timeframe recommendation
        mtf_analysis = self.perform_periodic_mtf_analysis()
        
        if not mtf_analysis:
            ProfessionalLogger.log("No multi-timeframe analysis available", "WARNING", "ENGINE")
            return None
        
        recommendation = mtf_analysis.get('recommendation', {})
        mtf_signal = recommendation.get('trade_signal', 'HOLD')
        mtf_confidence = recommendation.get('confidence', 0)
        
        # Check alignment with multi-timeframe recommendation
        signal_map = {
            1: 'BUY',
            0: 'SELL'
        }
        
        ml_signal = signal_map.get(signal, 'HOLD')
        
        ProfessionalLogger.log(f"Signal Comparison | ML: {ml_signal} ({confidence:.1%}) | MTF: {mtf_signal} ({mtf_confidence:.1%})", 
                             "ANALYSIS", "ENGINE")
        
        # Check if signals align
        if ml_signal != mtf_signal and mtf_signal != 'HOLD':
            ProfessionalLogger.log(f"⚠ Signal misalignment: ML says {ml_signal}, MTF says {mtf_signal}", "WARNING", "ENGINE")
            
            # If MTF confidence is high, override ML signal
            if mtf_confidence > 0.7 and confidence < 0.6:
                ProfessionalLogger.log(f"Overriding ML with MTF signal (MTF confidence: {mtf_confidence:.1%})", "INFO", "ENGINE")
                # Map MTF signal back to ML format
                if mtf_signal == 'BUY':
                    signal = 1
                elif mtf_signal == 'SELL':
                    signal = 0
                confidence = mtf_confidence
        
        # Check overall timeframe alignment
        alignment_scores = mtf_analysis.get('comparison', {}).get('alignment_scores', {})
        overall_alignment = alignment_scores.get('overall', 0.5)
        
        if overall_alignment < Config.MIN_TIMEFRAME_AGREEMENT:
            ProfessionalLogger.log(f"Poor timeframe alignment ({overall_alignment:.2f}), avoiding trade", "WARNING", "ENGINE")
            return None
        
        # Adjust confidence based on timeframe alignment
        adjusted_confidence = confidence * overall_alignment
        
        if adjusted_confidence < Config.MIN_CONFIDENCE:
            ProfessionalLogger.log(f"Adjusted confidence too low: {adjusted_confidence:.1%}", "WARNING", "ENGINE")
            return None
        
        # Execute trade
        return self.execute_trade(signal, adjusted_confidence, df_current, features, model_details)
    
    def execute_trade(self, signal, confidence, df_current, features, model_details):
        """Execute a trade"""
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
            
            # Calculate position size
            account = mt5.account_info()
            if not account:
                ProfessionalLogger.log("Cannot get account info", "ERROR", "EXECUTOR")
                return None
            
            # Risk calculation
            risk_amount = account.equity * Config.RISK_PERCENT
            
            # Adjust risk based on confidence
            risk_adjustment = min(1.0, confidence / 0.8)  # Scale up to 80% confidence
            risk_amount *= risk_adjustment
            
            # Calculate stop loss and take profit
            atr = features.get('atr_percent', 0.001) * entry_price
            if atr == 0:
                atr = entry_price * 0.002
            
            if signal == 1:  # BUY
                stop_loss = entry_price - (atr * Config.ATR_SL_MULTIPLIER)
                take_profit = entry_price + (atr * Config.ATR_TP_MULTIPLIER)
            else:  # SELL
                stop_loss = entry_price + (atr * Config.ATR_SL_MULTIPLIER)
                take_profit = entry_price - (atr * Config.ATR_TP_MULTIPLIER)
            
            # Calculate position size
            symbol_info = mt5.symbol_info(symbol)
            contract_size = getattr(symbol_info, 'trade_contract_size', 100)
            
            sl_distance = abs(entry_price - stop_loss)
            if sl_distance > 0:
                position_size = risk_amount / (sl_distance * contract_size)
                
                # Round to allowed lot size
                if hasattr(symbol_info, 'volume_step'):
                    step = symbol_info.volume_step
                    position_size = round(position_size / step) * step
                
                # Ensure within limits
                position_size = max(Config.MIN_VOLUME, min(position_size, Config.MAX_VOLUME))
            
            # Execute order
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
        print("\n" + "=" * 80)
        print("🤖 PROFESSIONAL MT5 ALGORITHMIC TRADING SYSTEM")
        print("📊 Multi-Timeframe Statistical Analysis | Live Trading")
        print("=" * 80 + "\n")
        
        ProfessionalLogger.log("Starting professional trading system with multi-timeframe analysis...", "INFO", "ENGINE")
        
        # Connect to MT5
        if not self.connect_mt5():
            return
        
        # Train initial model
        self.train_initial_model()
        
        # Start live trading
        self.run_live_trading()
    
    def run_live_trading(self):
        """Run live trading with multi-timeframe analysis"""
        ProfessionalLogger.log("=" * 80, "INFO", "ENGINE")
        ProfessionalLogger.log("STARTING LIVE TRADING WITH MULTI-TIMEFRAME ANALYSIS", "TRADE", "ENGINE")
        ProfessionalLogger.log("=" * 80, "INFO", "ENGINE")
        
        try:
            while True:
                self.run_periodic_tasks()
                
                # Get current data
                rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.PRIMARY_TIMEFRAME, 0, 100)
                if rates is None or len(rates) < 50:
                    ProfessionalLogger.log("Failed to get rates, retrying...", "WARNING", "ENGINE")
                    time.sleep(60)
                    continue
                
                df_current = pd.DataFrame(rates)
                
                # Get model prediction
                signal, confidence, features, model_details = self.model.predict(df_current)
                
                if signal is None:
                    # No valid signal
                    if self.iteration % 30 == 0:
                        tick = mt5.symbol_info_tick(Config.SYMBOL)
                        if tick:
                            price = tick.ask
                            positions = len(self.active_positions)
                            ProfessionalLogger.log(f"Waiting for signal | Price: {price:.2f} | Positions: {positions}", 
                                                 "INFO", "ENGINE")
                    time.sleep(60)
                    continue
                
                # Log prediction
                signal_type = "BUY" if signal == 1 else "SELL"
                ProfessionalLogger.log(f"Signal: {signal_type} | Confidence: {confidence:.1%}", "ANALYSIS", "ENGINE")
                
                # Check if we should execute
                if confidence >= Config.MIN_CONFIDENCE:
                    ProfessionalLogger.log(f"🎯 High-confidence {signal_type} signal detected!", "SUCCESS", "ENGINE")
                    
                    # Execute with multi-timeframe validation
                    self.execute_trade_with_mtf_validation(signal, confidence, df_current, features, model_details)
                
                time.sleep(60)
                
        except KeyboardInterrupt:
            ProfessionalLogger.log("\nShutdown requested by user", "WARNING", "ENGINE")
        except Exception as e:
            ProfessionalLogger.log(f"Unexpected error: {str(e)}", "ERROR", "ENGINE")
            import traceback
            traceback.print_exc()
        finally:
            mt5.shutdown()
            ProfessionalLogger.log("Disconnected from MT5", "INFO", "ENGINE")

# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    """Main entry point"""
    ProfessionalLogger.log("Starting professional trading system...", "INFO", "ENGINE")
    
    # Test MT5 connection
    if not mt5.initialize():
        ProfessionalLogger.log("MT5 initialization failed", "ERROR", "ENGINE")
        return
    
    mt5.shutdown()
    
    # Continue normal startup
    time.sleep(2)
    engine = ProfessionalTradingEngine()
    engine.run()

if __name__ == "__main__":
    main()