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
    MIN_CONFIDENCE = 0.45
    MIN_ENSEMBLE_AGREEMENT = 0.55
    
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
    LOOKBACK_BARS = 8000
    TRAINING_MIN_SAMPLES = 5000
    VALIDATION_SPLIT = 0.20
    
    # Retraining Schedule
    RETRAIN_HOURS = 24
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
    BARRIER_TIME = 4
    
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
    MIN_RR_RATIO = 1.25
    
    # Fixed Stops
    FIXED_SL_PERCENT = 0.0035
    FIXED_TP_PERCENT = 0.0070
    
    # Points-based Limits
    MIN_SL_DISTANCE_POINTS = 50
    MAX_SL_DISTANCE_POINTS = 300
    MIN_TP_DISTANCE_POINTS = 100
    MAX_TP_DISTANCE_POINTS = 1000
    
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
    SESSION_AWARE_TRADING = True
    
    # Trading Sessions (UTC times)
    AVOID_ASIAN_SESSION = False  #true
    PREFER_LONDON_NY_OVERLAP = False #true
    
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
    RETRY_DELAY_MS = 500
    
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
    OPTIMIZE_EVERY_N_TRADES = 50
    PARAM_OPTIMIZATION_ENABLED = True
    
    # ==========================================
    # NEW: ENTRY TIMING
    # ==========================================
    USE_CONFIRMATION_ENTRY = True
    CONFIRMATION_BARS_REQUIRED = 2
    MAX_ENTRY_WAIT_SECONDS = 900  # 15 minutes
    
    # ==========================================
    # DATA STORAGE & LOGGING
    # ==========================================
    TRADE_HISTORY_FILE = "trade_history_xauusd.json"
    MODEL_SAVE_FILE = "ensemble_model_xauusd.pkl"
    BACKTEST_RESULTS_FILE = "backtest_results_xauusd.json"
    PERFORMANCE_LOG_FILE = "performance_log_xauusd.csv"
    
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
    TIMEFRAME_ALIGNMENT_THRESHOLD = 0.60
    
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
        'BUY_SIGNAL_THRESHOLD': 0.5,
        'SELL_SIGNAL_THRESHOLD': -0.5,
        
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
    }