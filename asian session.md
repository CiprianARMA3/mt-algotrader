
class Config:
    """
    Ultra-Precise XAUUSD Trading Configuration
    Now with dynamic parameter adaptation and enhanced features
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
    TIMEFRAME = mt5.TIMEFRAME_M5
    
    # Position sizing with enhanced scaling
    BASE_VOLUME = 0.15  # Increased from 0.10
    MAX_VOLUME = 1.50  # Increased from 1.00
    MIN_VOLUME = 0.10
    VOLUME_STEP = 0.01
    
    MAGIC_NUMBER = 998877
    
    # ==========================================
    # RISK MANAGEMENT - ASIAN SESSION AGGRESSIVE
    # ==========================================
    RISK_PERCENT = 0.015  # Increased from 0.01 (50% more risk)
    MAX_TOTAL_RISK_PERCENT = 0.08  # Increased from 0.05
    MAX_RISK_PER_TRADE = 150  # Increased from 100
    
    # Signal Quality - Dynamic thresholds
    MIN_CONFIDENCE = 0.40  # Lowered from 0.45 (more signals)
    MIN_ENSEMBLE_AGREEMENT = 0.50  # Lowered from 0.55
    
    # Asian Session Specific - More Aggressive
    ASIAN_SESSION_AGGRESSIVE = True
    ASIAN_SESSION_RISK_MULTIPLIER = 1.2  # Take more risk in Asian session
    ASIAN_SESSION_CONFIDENCE_MULTIPLIER = 0.85  # Lower confidence requirements
    
    # Position Limits
    MAX_POSITIONS = 7  # Increased from 5
    MAX_DAILY_TRADES = 15  # Increased from 10
    MIN_TIME_BETWEEN_TRADES = 5  # Reduced from 10 (faster trading)
    
    # Loss Limits
    MAX_DAILY_LOSS_PERCENT = 3.0  # Increased from 2.0
    MAX_WEEKLY_LOSS_PERCENT = 7.0  # Increased from 5.0
    MAX_DRAWDOWN_PERCENT = 12.0  # Increased from 10.0
    MAX_CONSECUTIVE_LOSSES = 4  # Increased from 3
    
    # Kelly Criterion - More aggressive
    KELLY_FRACTION = 0.20  # Increased from 0.15
    USE_HALF_KELLY = True
    
    # Statistical Risk Metrics
    VAR_CONFIDENCE = 0.95  # Reduced from 0.99 (accept more tail risk)
    CVAR_CONFIDENCE = 0.95  # Reduced from 0.99
    MAX_POSITION_CORRELATION = 0.6  # Increased from 0.5
    
    # ==========================================
    # MACHINE LEARNING MODEL PARAMETERS - ENHANCED
    # ==========================================
    LOOKBACK_BARS = 8000
    TRAINING_MIN_SAMPLES = 5000
    VALIDATION_SPLIT = 0.20
    
    # Retraining Schedule
    RETRAIN_HOURS = 12  # More frequent retraining
    RETRAIN_ON_PERFORMANCE_DROP = True
    MIN_ACCURACY_THRESHOLD = 0.48  # Lowered from 0.50
    
    # Walk-Forward Optimization
    WALK_FORWARD_WINDOW = 800  # Reduced from 1000 (more adaptive)
    WALK_FORWARD_STEP = 80  # Reduced from 100
    WALK_FORWARD_FOLDS = 5
    
    # Feature Engineering Flags
    USE_FRACTIONAL_DIFF = True
    FD_THRESHOLD = 0.4
    USE_TICK_VOLUME_VOLATILITY = True
    TICK_SKEW_LOOKBACK = 50
    
    # Labeling Method - ENHANCED: Dynamic ATR-based barriers
    TRIPLE_BARRIER_METHOD = True
    USE_DYNAMIC_BARRIERS = True
    BARRIER_UPPER = 0.0025  # Increased from 0.0020 (bigger targets)
    BARRIER_LOWER = -0.0020  # Increased from -0.0015 (wider stops)
    BARRIER_TIME = 3  # Reduced from 4 (faster targets)
    
    # Ensemble Configuration - ENHANCED
    USE_STACKING_ENSEMBLE = True
    ENSEMBLE_DIVERSITY_WEIGHT = 0.3
    ADAPTIVE_ENSEMBLE_WEIGHTING = True
    MODEL_CONFIDENCE_CALIBRATION = True
    USE_REGIME_SPECIFIC_MODELS = True
    
    # Data Quality
    MIN_DATA_QUALITY_SCORE = 0.70  # Reduced from 0.75
    OUTLIER_REMOVAL_THRESHOLD = 5.0  # Increased from 4.0
    
    # ==========================================
    # TECHNICAL INDICATORS - ASIAN SESSION OPTIMIZED
    # ==========================================
    ATR_PERIOD = 10  # Reduced from 14 (more responsive)
    RSI_PERIOD = 10  # Reduced from 14 (more sensitive)
    ADX_PERIOD = 10  # Reduced from 14
    
    # Moving Averages - Faster for Asian session
    FAST_MA = 5  # Reduced from 8
    MEDIUM_MA = 14  # Reduced from 21
    SLOW_MA = 34  # Reduced from 50
    TREND_MA = 144  # Reduced from 200
    
    # Volatility Bands - Tighter for lower volatility
    BB_PERIOD = 16  # Reduced from 20
    BB_STD = 1.8  # Reduced from 2.0
    
    # ==========================================
    # STATISTICAL FEATURES - ENHANCED
    # ==========================================
    GARCH_VOL_PERIOD = 16  # Reduced from 20
    GARCH_P = 1
    GARCH_Q = 1
    
    # Hurst Exponent
    HURST_WINDOW = 80  # Reduced from 100
    HURST_TRENDING_THRESHOLD = 0.53  # Lowered from 0.55
    HURST_MEANREVERTING_THRESHOLD = 0.47  # Raised from 0.45
    
    # Tail Risk
    TAIL_INDEX_WINDOW = 120  # Reduced from 150
    VAR_LOOKBACK = 80  # Reduced from 100
    
    # Correlation Analysis
    CORRELATION_WINDOW = 40  # Reduced from 50
    CORRELATED_SYMBOLS = ["DXY", "US10Y", "EURUSD"]
    
    # ==========================================
    # STOP LOSS & TAKE PROFIT - AGGRESSIVE
    # ==========================================
    USE_DYNAMIC_SL_TP = True
    
    # ATR-Based Stops with dynamic multipliers
    ATR_SL_MULTIPLIER = 1.2  # Reduced from 1.5 (tighter stops)
    ATR_TP_MULTIPLIER = 2.0  # Increased from 1.75 (bigger targets)
    
    # Minimum Risk/Reward
    MIN_RR_RATIO = 1.15  # Reduced from 1.25
    
    # Fixed Stops
    FIXED_SL_PERCENT = 0.0025  # Reduced from 0.0035
    FIXED_TP_PERCENT = 0.0090  # Increased from 0.0070
    
    # Points-based Limits
    MIN_SL_DISTANCE_POINTS = 40  # Reduced from 50
    MAX_SL_DISTANCE_POINTS = 250  # Reduced from 300
    MIN_TP_DISTANCE_POINTS = 80  # Reduced from 100
    MAX_TP_DISTANCE_POINTS = 1200  # Increased from 1000
    
    # Trailing Stop - More aggressive
    USE_TRAILING_STOP = True
    TRAILING_STOP_ACTIVATION = 1.2  # Reduced from 1.5 (trail sooner)
    TRAILING_STOP_DISTANCE = 0.8  # Reduced from 1.0 (tighter trail)
    
    # Break-even Stop
    USE_BREAKEVEN_STOP = True
    BREAKEVEN_ACTIVATION = 0.8  # Reduced from 1.0 (move to breakeven sooner)
    BREAKEVEN_OFFSET = 0.0001
    
    # ==========================================
    # MARKET REGIME DETECTION - ENHANCED
    # ==========================================
    USE_MARKET_REGIME = True
    
    # Trend Strength - More sensitive
    ADX_TREND_THRESHOLD = 18  # Reduced from 20
    ADX_STRONG_TREND_THRESHOLD = 35  # Reduced from 40
    ADX_SLOPE_THRESHOLD = 0.3  # Reduced from 0.5
    
    # Volatility Regimes - Adjusted for Asian session
    VOLATILITY_SCALING_ENABLED = True
    HIGH_VOL_THRESHOLD = 0.012  # Reduced from 0.015
    NORMAL_VOL_THRESHOLD = 0.008  # Reduced from 0.010
    LOW_VOL_THRESHOLD = 0.005  # Reduced from 0.007
    
    # Position Sizing Adjustments by Regime
    HIGH_VOL_SIZE_MULTIPLIER = 0.6  # Increased from 0.5
    LOW_VOL_SIZE_MULTIPLIER = 1.4  # Increased from 1.2
    
    # ==========================================
    # TIME-BASED FILTERS - ASIAN SESSION OPTIMIZED
    # ==========================================
    SESSION_AWARE_TRADING = True
    
    # Trading Sessions (UTC times)
    AVOID_ASIAN_SESSION = False  # Now trading Asian session
    PREFER_LONDON_NY_OVERLAP = False  # Not restricting to overlap
    
    # Asian session (Tokyo) - 0:00 to 9:00 UTC
    ASIAN_SESSION_START = 0
    ASIAN_SESSION_END = 9
    
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
    # ORDER EXECUTION - FASTER
    # ==========================================
    MAX_SLIPPAGE_POINTS = 15  # Increased from 10
    ORDER_TIMEOUT_SECONDS = 20  # Reduced from 30
    MAX_RETRIES = 3
    RETRY_DELAY_MS = 300  # Reduced from 500
    
    USE_MARKET_ORDERS = True
    USE_LIMIT_ORDERS = False
    
    CHECK_SPREAD_BEFORE_ENTRY = True
    MAX_SPREAD_POINTS = 40  # Increased from 30
    NORMAL_SPREAD_POINTS = 2
    
    COMMISSION_PER_LOT = 3.5
    
    # ==========================================
    # PERFORMANCE METRICS - AGGRESSIVE
    # ==========================================
    MIN_SHARPE_RATIO = 0.7  # Reduced from 0.8
    MIN_PROFIT_FACTOR = 1.3  # Reduced from 1.5
    MIN_WIN_RATE = 0.40  # Reduced from 0.45
    MAX_DRAWDOWN_DURATION = 15  # Increased from 10
    
    MIN_SAMPLES_FOR_STATS = 100
    CONFIDENCE_LEVEL = 0.90  # Reduced from 0.95
    BOOTSTRAP_SAMPLES = 1000
    
    # ==========================================
    # ADAPTIVE SYSTEMS - AGGRESSIVE
    # ==========================================
    ADAPTIVE_RISK_MANAGEMENT = True
    PERFORMANCE_BASED_POSITION_SIZING = True
    REAL_TIME_MARKET_STRESS_INDICATOR = True
    
    # Adaptation Parameters
    PERFORMANCE_LOOKBACK_TRADES = 15  # Reduced from 20
    GOOD_PERFORMANCE_THRESHOLD = 0.55  # Reduced from 0.60
    POOR_PERFORMANCE_THRESHOLD = 0.35  # Reduced from 0.40
    
    # Position Size Adjustments - More aggressive
    INCREASE_SIZE_AFTER_WINS = True  # Changed from False
    DECREASE_SIZE_AFTER_LOSSES = True
    SIZE_DECREASE_FACTOR = 0.7  # Reduced from 0.8 (cut losses faster)
    SIZE_RECOVERY_FACTOR = 1.2  # Increased from 1.1 (scale up faster)
    
    # ==========================================
    # NEW: PARAMETER OPTIMIZATION
    # ==========================================
    OPTIMIZATION_WINDOW = 400  # Reduced from 500
    OPTIMIZE_EVERY_N_TRADES = 30  # Reduced from 50
    PARAM_OPTIMIZATION_ENABLED = True
    
    # ==========================================
    # NEW: ENTRY TIMING - FASTER
    # ==========================================
    USE_CONFIRMATION_ENTRY = True
    CONFIRMATION_BARS_REQUIRED = 1  # Reduced from 2
    MAX_ENTRY_WAIT_SECONDS = 300  # Reduced from 900 (5 minutes)
    
    # ==========================================
    # DATA STORAGE & LOGGING
    # ==========================================
    TRADE_HISTORY_FILE = "trade_history_xauusd_aggressive.json"
    MODEL_SAVE_FILE = "ensemble_model_xauusd_aggressive.pkl"
    BACKTEST_RESULTS_FILE = "backtest_results_xauusd_aggressive.json"
    PERFORMANCE_LOG_FILE = "performance_log_xauusd_aggressive.csv"
    
    MEMORY_SIZE = 1000
    LEARNING_WEIGHT = 0.5  # Increased from 0.4
    
    # Logging Levels
    LOG_LEVEL_CONSOLE = "INFO"
    LOG_LEVEL_FILE = "DEBUG"
    LOG_TRADES = True
    LOG_PREDICTIONS = True
    LOG_PERFORMANCE = True
    
    # ==========================================
    # MULTI-TIMEFRAME ANALYSIS - MORE PERMISSIVE
    # ==========================================
    MULTI_TIMEFRAME_ENABLED = True
    TIMEFRAMES = ['M5', 'M15', 'H1']
    TIMEFRAME_WEIGHTS = [0.3, 0.4, 0.3]  # More weight to M5
    TIMEFRAME_ALIGNMENT_THRESHOLD = 0.50  # Reduced from 0.60
    
    LONG_TIMEFRAME_TREND_FILTER = False  # Disabled for Asian session
    SHORT_TIMEFRAME_ENTRY = True
    
    # ==========================================
    # GOLD-SPECIFIC PARAMETERS - ASIAN SESSION OPTIMIZED
    # ==========================================
    GOLD_VOLATILITY_ADJUSTMENT = True
    
    EXPECTED_DAILY_RANGE = 18  # Reduced from 20
    HIGH_RANGE_MULTIPLIER = 1.3  # Reduced from 1.5
    LOW_RANGE_MULTIPLIER = 0.6  # Increased from 0.5
    
    USE_DXY_FILTER = True
    DXY_CORRELATION_THRESHOLD = -0.6  # Reduced from -0.7
    USE_YIELD_FILTER = False
    
    # ==========================================
    # SAFETY FEATURES - BALANCED WITH RISK
    # ==========================================
    ENABLE_EMERGENCY_STOP = True
    EMERGENCY_STOP_DRAWDOWN = 0.20  # Increased from 0.15
    
    CHECK_MARGIN_BEFORE_TRADE = True
    MIN_FREE_MARGIN_PERCENT = 0.20  # Reduced from 0.30
    
    CHECK_CONNECTION_BEFORE_TRADE = True
    MAX_PING_MS = 150  # Increased from 100
    
    MAX_DAILY_VOLUME = 1.5  # Increased from 1.0
    REQUIRE_STOP_LOSS = True
    REQUIRE_TAKE_PROFIT = True
    
    # ==========================================
    # DEBUGGING & TESTING
    # ==========================================
    DEBUG_MODE = False
    PAPER_TRADING_MODE = False  # Set to True for testing
    BACKTEST_MODE = False
    
    VALIDATE_SIGNALS = True
    VALIDATE_RISK = True
    VALIDATE_STOPS = True