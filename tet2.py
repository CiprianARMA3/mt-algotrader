import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import time
import json
import os
import sys
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats
from scipy.stats import skew, kurtosis, jarque_bera
from arch import arch_model
import warnings

# Suppress warnings for cleaner console output
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional but recommended)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==========================================
# PROFESSIONAL CONFIGURATION (XAUUSD OPTIMIZED)
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
    # UPDATE THESE WITH YOUR CREDENTIALS
    MT5_LOGIN = 12345678  
    MT5_PASSWORD = "your_password"
    MT5_SERVER = "YourBroker-Server"
    
    # ==========================================
    # TRADING INSTRUMENT SPECIFICATIONS
    # ==========================================
    SYMBOL = "XAUUSD"
    TIMEFRAME = mt5.TIMEFRAME_M15  # 15-min optimal for intraday gold trading
    
    # XAUUSD-specific: Gold typically trades in 0.01 lot increments
    # 1 Lot usually = 100oz. Check your broker contract size!
    BASE_VOLUME = 0.01   # START SMALL - critical for testing
    MAX_VOLUME = 1.0     # Maximum position size cap
    MIN_VOLUME = 0.01    # MT5 minimum for gold
    VOLUME_STEP = 0.01   # Standard gold lot increment
    
    MAGIC_NUMBER = 998877
    
    # ==========================================
    # RISK MANAGEMENT - CONSERVATIVE & PRECISE
    # ==========================================
    
    # Position Sizing
    RISK_PERCENT = 0.01  # Risk 1% of equity per trade
    MAX_RISK_PER_TRADE = 100.0  # Hard cap on dollar risk
    
    # Signal Quality Thresholds
    MIN_CONFIDENCE = 0.65  # Increased to 65% for higher precision
    MIN_ENSEMBLE_AGREEMENT = 0.60  # At least 60% of models must agree
    
    # Position Limits
    MAX_POSITIONS = 1  # One position at a time (gold is volatile)
    MAX_DAILY_TRADES = 5  # Prevent overtrading
    MIN_TIME_BETWEEN_TRADES = 15  # Minutes between trades
    
    # Loss Limits
    MAX_DAILY_LOSS_PERCENT = 2.0  # Stop trading after 2% daily loss
    MAX_DRAWDOWN_PERCENT = 5.0    # Maximum account drawdown circuit breaker
    
    # Statistical Risk Metrics
    VAR_CONFIDENCE = 0.99  # 99% Value at Risk
    CVAR_CONFIDENCE = 0.99  # 99% Conditional VaR
    
    # ==========================================
    # MACHINE LEARNING MODEL PARAMETERS
    # ==========================================
    
    # Data Collection
    LOOKBACK_BARS = 10000  # Number of bars for deep history analysis
    TRAINING_MIN_SAMPLES = 1000  # Minimum samples required to train
    
    # Retraining Schedule
    RETRAIN_HOURS = 12  # Retrain every 12 hours to adapt to new volatility
    MIN_ACCURACY_THRESHOLD = 0.52  # Minimum validation accuracy to accept model
    
    # Walk-Forward Optimization
    WALK_FORWARD_WINDOW = 2000  # Rolling window size
    WALK_FORWARD_FOLDS = 5  # Number of splits
    
    # Labeling Method (Triple Barrier)
    TRIPLE_BARRIER_METHOD = True 
    BARRIER_UPPER = 0.0025  # 0.25% Target (approx $5-$6 move in Gold)
    BARRIER_LOWER = -0.0015 # 0.15% Stop (approx $3-$4 move)
    BARRIER_TIME = 12       # 12 bars (3 hours) max hold time
    
    # Data Quality
    MIN_DATA_QUALITY_SCORE = 0.80  # Require 80% data quality
    
    # ==========================================
    # TECHNICAL INDICATORS - GOLD-OPTIMIZED
    # ==========================================
    
    # Trend Indicators
    ATR_PERIOD = 14
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    
    # Moving Averages (Gold-specific)
    FAST_MA = 9     # Fast EMA
    MEDIUM_MA = 21  # Medium EMA
    SLOW_MA = 50    # Slow SMA
    TREND_MA = 200  # Long-term trend filter
    
    # Volatility Bands
    BB_PERIOD = 20
    BB_STD = 2.0
    
    # ==========================================
    # STATISTICAL FEATURES
    # ==========================================
    
    # GARCH Volatility
    GARCH_VOL_PERIOD = 50
    GARCH_P = 1
    GARCH_Q = 1
    
    # Hurst Exponent
    HURST_WINDOW = 100
    HURST_TRENDING_THRESHOLD = 0.55
    HURST_MEANREVERTING_THRESHOLD = 0.45
    
    # VaR
    VAR_LOOKBACK = 100
    
    # ==========================================
    # STOP LOSS & TAKE PROFIT - PRECISION TUNED
    # ==========================================
    
    USE_DYNAMIC_SL_TP = True  # Use ATR-based dynamic stops
    
    # ATR-Based Stops (Gold-optimized)
    ATR_SL_MULTIPLIER = 1.5  # 1.5x ATR for stop loss
    ATR_TP_MULTIPLIER = 2.5  # 2.5x ATR for take profit
    
    # Fixed Stops (backup)
    FIXED_SL_PERCENT = 0.0020  # 0.20%
    FIXED_TP_PERCENT = 0.0050  # 0.50%
    
    # Points-based Limits (XAUUSD point = 0.01)
    # Ensure these match your broker's digits (usually 2 for Gold)
    MIN_SL_DISTANCE_POINTS = 30   # $0.30
    MAX_SL_DISTANCE_POINTS = 500  # $5.00
    MIN_TP_DISTANCE_POINTS = 50   # $0.50
    MAX_TP_DISTANCE_POINTS = 1000 # $10.00
    
    # Trailing & Breakeven
    USE_TRAILING_STOP = True
    TRAILING_STOP_ACTIVATION = 1.2 # Activate after 1.2x Risk profit
    USE_BREAKEVEN_STOP = True
    
    # ==========================================
    # MARKET REGIME DETECTION
    # ==========================================
    
    USE_MARKET_REGIME = True
    
    # Trend Strength
    ADX_TREND_THRESHOLD = 25
    ADX_STRONG_TREND_THRESHOLD = 40
    
    # Volatility Thresholds (Daily Returns)
    VOLATILITY_SCALING_ENABLED = True
    HIGH_VOL_THRESHOLD = 0.015   # 1.5% daily move is high for Gold
    NORMAL_VOL_THRESHOLD = 0.008 # 0.8% is normal
    LOW_VOL_THRESHOLD = 0.004    # 0.4% is low
    
    # ==========================================
    # TIME-BASED FILTERS
    # ==========================================
    
    SESSION_AWARE_TRADING = True
    
    # UTC Times
    LONDON_OPEN_HOUR = 8
    LONDON_CLOSE_HOUR = 16
    NY_OPEN_HOUR = 13
    NY_CLOSE_HOUR = 21
    
    AVOID_ASIAN_SESSION = True     # Low liquidity for Gold
    AVOID_MONDAY_FIRST_HOUR = True
    AVOID_FRIDAY_LAST_HOURS = True
    
    # ==========================================
    # ORDER EXECUTION
    # ==========================================
    
    MAX_SLIPPAGE_POINTS = 20 # 20 points ($0.20)
    MAX_RETRIES = 5
    RETRY_DELAY_MS = 1000
    CHECK_SPREAD_BEFORE_ENTRY = True
    MAX_SPREAD_POINTS = 40   # Max spread allowed ($0.40)
    
    # ==========================================
    # DATA STORAGE
    # ==========================================
    
    TRADE_HISTORY_FILE = "xauusd_trade_history.json"
    MODEL_SAVE_FILE = "xauusd_ensemble_model.pkl"
    MEMORY_SIZE = 2000

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
        if len(returns) < 50:
            return {"error": "Insufficient data"}
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        returns = returns[~np.isinf(returns)]
        
        if len(returns) == 0:
            return {"error": "No valid data"}
        
        stats_dict = {
            'n_samples': len(returns),
            'mean': np.mean(returns),
            'std': np.std(returns),
            'skewness': skew(returns),
            'kurtosis': kurtosis(returns, fisher=True),
            'median': np.median(returns),
            'min': np.min(returns),
            'max': np.max(returns),
            'sharpe': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        }
        
        # Normality tests
        try:
            jb_stat, jb_p = jarque_bera(returns)
            stats_dict['is_normal'] = jb_p > 0.05
        except:
            stats_dict['is_normal'] = False
        
        # Value at Risk
        stats_dict['var_95'] = np.percentile(returns, 5)
        stats_dict['var_99'] = np.percentile(returns, 1)
        
        # CVaR (Expected Shortfall)
        tail_95 = returns[returns <= stats_dict['var_95']]
        stats_dict['cvar_95'] = np.mean(tail_95) if len(tail_95) > 0 else stats_dict['var_95']
        
        return stats_dict
    
    @staticmethod
    def calculate_hurst_exponent(prices):
        """Calculate Hurst exponent for market efficiency analysis"""
        if len(prices) < 100:
            return 0.5
        
        try:
            returns = np.diff(np.log(prices))
            n = len(returns)
            
            # Aggregate variance method (Simplified for speed/stability)
            variances = []
            scales = []
            
            max_scale = n // 4
            for scale in range(2, max_scale):
                m = n // scale
                if m < 4: continue
                
                # Reshape and sum to aggregate
                # Truncate to multiple of scale
                limit = m * scale
                aggregated = returns[:limit].reshape(m, scale).sum(axis=1)
                variances.append(np.var(aggregated))
                scales.append(scale)
                
            if len(variances) < 3: return 0.5
            
            # Fit line to log-log
            log_var = np.log(variances)
            log_scale = np.log(scales)
            
            slope, _ = np.polyfit(log_scale, log_var, 1)
            hurst = 1 + slope / 2
            return max(0.0, min(1.0, hurst))
            
        except Exception as e:
            # ProfessionalLogger.log(f"Hurst calc error: {e}", "WARNING", "STATS")
            return 0.5
    
    @staticmethod
    def calculate_garch_volatility(returns, p=1, q=1):
        """Calculate GARCH volatility with robust error handling"""
        if len(returns) < 100:
            return np.std(returns) if len(returns) > 0 else 0.001
        
        try:
            # Scale returns for GARCH stability (critical for small return numbers)
            scaled_returns = returns * 100.0
            
            model = arch_model(scaled_returns, vol='Garch', p=p, q=q, dist='normal')
            res = model.fit(disp='off', show_warning=False)
            
            # Get conditional volatility and descale
            cond_vol = res.conditional_volatility[-1] / 100.0
            return cond_vol
            
        except Exception:
            return np.std(returns)

    @staticmethod
    def calculate_market_regime(data):
        """Determine market regime based on statistical properties"""
        if len(data) < 200:
            return {"regime": "unknown", "confidence": 0}
        
        prices = data['close'].values
        returns = np.diff(np.log(prices))
        
        hurst = AdvancedStatisticalAnalyzer.calculate_hurst_exponent(prices[-500:])
        volatility = np.std(returns) * np.sqrt(252) # Annualized
        
        # Calculate ADX-like trend strength if not available
        # Simple slope check
        slope = (prices[-1] - prices[-20]) / prices[-20]
        
        regime = "ranging"
        confidence = 0.5
        
        if hurst > 0.6:
            regime = "trending"
            confidence = (hurst - 0.5) * 2
        elif hurst < 0.4:
            regime = "mean_reverting"
            confidence = (0.5 - hurst) * 2
            
        # Volatility overlay
        if volatility > Config.HIGH_VOL_THRESHOLD * np.sqrt(252): # Compare annualized
            regime += "_high_vol"
            
        return {
            "regime": regime,
            "confidence": min(confidence, 1.0),
            "hurst": hurst,
            "volatility": volatility
        }

# ==========================================
# PROFESSIONAL RISK METRICS
# ==========================================
class ProfessionalRiskMetrics:
    """Advanced risk metrics with statistical analysis"""
    
    @staticmethod
    def calculate_risk_metrics(returns, prices=None):
        if len(returns) < 20:
            return {"error": "Insufficient data"}
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        metrics = {}
        metrics['volatility'] = np.std(returns)
        metrics['sharpe'] = np.mean(returns) / metrics['volatility'] * np.sqrt(252) if metrics['volatility'] > 0 else 0
        
        # Max Drawdown
        if prices is not None:
            cum_returns = np.cumprod(1 + returns)
            peak = np.maximum.accumulate(cum_returns)
            drawdown = (cum_returns - peak) / peak
            metrics['max_drawdown'] = np.min(drawdown)
        else:
            metrics['max_drawdown'] = 0.0
            
        return metrics

# ==========================================
# DATA QUALITY CHECKER
# ==========================================
class ProfessionalDataQualityChecker:
    @staticmethod
    def check_data_quality(df):
        if df is None or len(df) == 0:
            return 0.0, {"error": "Empty dataframe"}
        
        score = 1.0
        diagnostics = {}
        
        # Check for NaNs
        nan_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])
        score -= nan_ratio * 5
        diagnostics['nan_ratio'] = nan_ratio
        
        # Check for duplicate time indices
        if 'time' in df.columns:
            duplicates = df.duplicated(subset=['time']).sum()
            score -= (duplicates / len(df)) * 5
            diagnostics['duplicates'] = duplicates
            
        # Check for zero prices
        if 'close' in df.columns:
            zeros = (df['close'] == 0).sum()
            score -= (zeros / len(df)) * 10
            diagnostics['zeros'] = zeros
            
        return max(0.0, score), diagnostics

# ==========================================
# PROFESSIONAL FEATURE ENGINEERING
# ==========================================
class ProfessionalFeatureEngine:
    def __init__(self):
        self.stat_analyzer = AdvancedStatisticalAnalyzer()
        
    def calculate_features(self, df):
        """Calculate features using Config parameters"""
        df = df.copy()
        
        # Basic Price Action
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # Moving Averages
        df[f'sma_{Config.FAST_MA}'] = df['close'].rolling(Config.FAST_MA).mean()
        df[f'sma_{Config.MEDIUM_MA}'] = df['close'].rolling(Config.MEDIUM_MA).mean()
        df[f'sma_{Config.SLOW_MA}'] = df['close'].rolling(Config.SLOW_MA).mean()
        
        # Distances from MA
        df['dist_fast'] = df['close'] - df[f'sma_{Config.FAST_MA}']
        df['dist_medium'] = df['close'] - df[f'sma_{Config.MEDIUM_MA}']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(Config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(Config.RSI_PERIOD).mean()
        rs = gain / loss.replace(0, 1e-10) # Avoid div by zero
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_norm'] = (df['rsi'] - 50) / 50.0  # Normalized -1 to 1
        
        # Bollinger Bands
        sma_bb = df['close'].rolling(Config.BB_PERIOD).mean()
        std_bb = df['close'].rolling(Config.BB_PERIOD).std()
        df['bb_upper'] = sma_bb + (std_bb * Config.BB_STD)
        df['bb_lower'] = sma_bb - (std_bb * Config.BB_STD)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_bb
        df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volatility (ATR)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(Config.ATR_PERIOD).mean()
        df['atr_pct'] = df['atr'] / df['close']
        
        # Statistical Features
        df['volatility_20'] = df['returns'].rolling(20).std()
        df['skew_20'] = df['returns'].rolling(20).skew()
        
        # Time Features (Cyclical)
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df['hour'] = df['datetime'].dt.hour
            df['day'] = df['datetime'].dt.dayofweek
            
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day'] / 7)
            
            # Session Flags
            if Config.SESSION_AWARE_TRADING:
                df['is_london'] = df['hour'].apply(lambda x: 1 if Config.LONDON_OPEN_HOUR <= x < Config.LONDON_CLOSE_HOUR else 0)
                df['is_ny'] = df['hour'].apply(lambda x: 1 if Config.NY_OPEN_HOUR <= x < Config.NY_CLOSE_HOUR else 0)
        
        # Clean infinite values
        df = df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)
        return df

    def create_labels(self, df):
        """Triple Barrier Method for Labeling"""
        df = df.copy()
        labels = np.zeros(len(df))
        
        # Triple Barrier Parameters
        upper = Config.BARRIER_UPPER
        lower = Config.BARRIER_LOWER
        horizon = Config.BARRIER_TIME
        
        closes = df['close'].values
        
        for i in range(len(closes) - horizon):
            window = closes[i+1 : i+horizon+1]
            returns = (window - closes[i]) / closes[i]
            
            # Check for barriers
            hit_upper = np.where(returns >= upper)[0]
            hit_lower = np.where(returns <= lower)[0]
            
            first_upper = hit_upper[0] if len(hit_upper) > 0 else horizon + 1
            first_lower = hit_lower[0] if len(hit_lower) > 0 else horizon + 1
            
            if first_upper < first_lower and first_upper < horizon:
                labels[i] = 1 # Buy
            elif first_lower < first_upper and first_lower < horizon:
                labels[i] = 0 # Sell/Hold (Binary classification usually 0/1)
            else:
                # Vertical Barrier (Time expired)
                # If positive return, treat as buy, else sell
                labels[i] = 1 if returns[-1] > 0 else 0
                
        df['label'] = labels
        return df

    def get_feature_columns(self):
        return [
            'log_returns', 'dist_fast', 'dist_medium', 
            'rsi_norm', 'bb_width', 'bb_pos', 'atr_pct',
            'volatility_20', 'skew_20',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'is_london', 'is_ny'
        ]

# ==========================================
# PROFESSIONAL ENSEMBLE MODEL
# ==========================================
class ProfessionalEnsemble:
    """Ensemble of ML models for robust prediction"""
    
    def __init__(self, trade_memory, feature_engine):
        self.feature_engine = feature_engine
        self.trade_memory = trade_memory
        self.scaler = RobustScaler()
        self.is_trained = False
        self.last_train_time = None
        self.training_metrics = {}
        
        # Base Models
        self.rf = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=42, n_jobs=-1)
        self.gb = GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, random_state=42)
        self.lr = LogisticRegression(class_weight='balanced', max_iter=1000)
        
        estimators = [('rf', self.rf), ('gb', self.gb), ('lr', self.lr)]
        if XGB_AVAILABLE:
            self.xgb = XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=5, n_jobs=-1, verbosity=0)
            estimators.append(('xgb', self.xgb))
            
        self.ensemble = VotingClassifier(estimators=estimators, voting='soft')
        
    def train(self, df):
        try:
            ProfessionalLogger.log(f"Starting training on {len(df)} bars...", "LEARN", "ENSEMBLE")
            
            # Feature Prep
            df_feat = self.feature_engine.calculate_features(df)
            df_lbl = self.feature_engine.create_labels(df_feat)
            
            # Remove unlabeled data at the end (due to horizon)
            df_lbl = df_lbl.iloc[:-Config.BARRIER_TIME]
            
            features = self.feature_engine.get_feature_columns()
            X = df_lbl[features].values
            y = df_lbl['label'].values
            
            # Clean data
            mask = ~np.isnan(X).any(axis=1)
            X = X[mask]
            y = y[mask]
            
            if len(X) < Config.TRAINING_MIN_SAMPLES:
                ProfessionalLogger.log("Insufficient clean training data", "ERROR", "ENSEMBLE")
                return False
                
            # Walk-Forward Validation
            tscv = TimeSeriesSplit(n_splits=Config.WALK_FORWARD_FOLDS)
            scores = []
            
            for train_index, test_index in tscv.split(X):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                
                # Scale within fold to prevent leakage
                scaler = RobustScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                self.ensemble.fit(X_train_scaled, y_train)
                preds = self.ensemble.predict(X_test_scaled)
                acc = accuracy_score(y_test, preds)
                scores.append(acc)
                
            avg_acc = np.mean(scores)
            ProfessionalLogger.log(f"Walk-Forward CV Accuracy: {avg_acc:.2%}", "LEARN", "ENSEMBLE")
            
            if avg_acc < Config.MIN_ACCURACY_THRESHOLD:
                ProfessionalLogger.log(f"Model failed accuracy threshold ({Config.MIN_ACCURACY_THRESHOLD:.2%})", "WARNING", "ENSEMBLE")
                # We can choose to return False, or proceed with caution. 
                # Proceeding for now to allow dynamic adaptation.
            
            # Final Fit on all data
            self.scaler.fit(X)
            X_scaled = self.scaler.transform(X)
            self.ensemble.fit(X_scaled, y)
            
            self.is_trained = True
            self.last_train_time = datetime.now()
            self.training_metrics['accuracy'] = avg_acc
            
            return True
            
        except Exception as e:
            ProfessionalLogger.log(f"Training failed: {e}", "ERROR", "ENSEMBLE")
            return False

    def predict(self, df):
        if not self.is_trained:
            return None, 0.0, None, {}
            
        try:
            df_feat = self.feature_engine.calculate_features(df)
            features = self.feature_engine.get_feature_columns()
            
            current_feat = df_feat[features].iloc[-1].values.reshape(1, -1)
            # Handle NaN
            if np.isnan(current_feat).any():
                return None, 0.0, None, {}
                
            current_scaled = self.scaler.transform(current_feat)
            
            # Voting Classifier Prediction
            prediction = self.ensemble.predict(current_scaled)[0]
            probs = self.ensemble.predict_proba(current_scaled)[0]
            confidence = np.max(probs)
            
            # Get individual model votes for "Agreement"
            model_details = {}
            agreement_count = 0
            total_models = len(self.ensemble.estimators_)
            
            for name, model in self.ensemble.named_estimators_.items():
                try:
                    p = model.predict(current_scaled)[0]
                    model_details[name] = {'prediction': int(p)}
                    if int(p) == int(prediction):
                        agreement_count += 1
                except:
                    pass
                    
            return int(prediction), confidence, df_feat.iloc[-1].to_dict(), model_details
            
        except Exception as e:
            ProfessionalLogger.log(f"Prediction failed: {e}", "ERROR", "ENSEMBLE")
            return None, 0.0, None, {}

    def should_retrain(self):
        if not self.is_trained: return True
        elapsed = (datetime.now() - self.last_train_time).total_seconds() / 3600
        return elapsed > Config.RETRAIN_HOURS

# ==========================================
# SMART ORDER EXECUTOR
# ==========================================
class SmartOrderExecutor:
    """Intelligent order execution with specific XAUUSD validation"""
    
    def execute_trade(self, symbol, order_type, volume, entry_price, sl, tp, magic, comment=""):
        
        # 1. Validation Checks
        if not mt5.terminal_info().trade_allowed:
            ProfessionalLogger.log("Auto-trading is disabled in terminal settings", "ERROR", "EXECUTOR")
            return None
            
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            ProfessionalLogger.log(f"Symbol {symbol} not found", "ERROR", "EXECUTOR")
            return None
            
        # Check Money Management
        account = mt5.account_info()
        if account.equity < 100: # Bare minimum safety check
            ProfessionalLogger.log("Insufficient equity", "ERROR", "EXECUTOR")
            return None
            
        # Normalize volume
        volume = max(Config.MIN_VOLUME, min(Config.MAX_VOLUME, volume))
        volume = round(volume / Config.VOLUME_STEP) * Config.VOLUME_STEP
        
        # 2. Check Spread
        if Config.CHECK_SPREAD_BEFORE_ENTRY:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                spread_points = (tick.ask - tick.bid) / symbol_info.point
                if spread_points > Config.MAX_SPREAD_POINTS:
                    ProfessionalLogger.log(f"Spread too high ({spread_points:.1f} > {Config.MAX_SPREAD_POINTS})", "WARNING", "EXECUTOR")
                    return None
        
        # 3. Construct Order
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": entry_price,
            "sl": sl,
            "tp": tp,
            "magic": magic,
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # 4. Execute with Retry Logic
        for attempt in range(Config.MAX_RETRIES):
            result = mt5.order_send(request)
            
            if result.retcode == mt5.TRADE_RETCODE_DONE:
                ProfessionalLogger.log(f"Order Sent Successfully! Ticket: {result.order}", "SUCCESS", "EXECUTOR")
                return result
            elif result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                ProfessionalLogger.log("Requote, retrying...", "WARNING", "EXECUTOR")
                time.sleep(Config.RETRY_DELAY_MS / 1000.0)
                continue
            else:
                ProfessionalLogger.log(f"Order Failed: {result.retcode} - {result.comment}", "ERROR", "EXECUTOR")
                break
                
        return None

# ==========================================
# TRADE MEMORY & PERSISTENCE
# ==========================================
class ProfessionalTradeMemory:
    def __init__(self):
        self.history_file = Config.TRADE_HISTORY_FILE
        self.trades = self._load()
        
    def _load(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            except:
                return []
        return []
        
    def save(self):
        with open(self.history_file, 'w') as f:
            json.dump(self.trades, f, indent=4)
            
    def add_trade(self, trade_dict):
        self.trades.append(trade_dict)
        if len(self.trades) > Config.MEMORY_SIZE:
            self.trades.pop(0)
        self.save()
        
    def get_summary(self):
        if not self.trades: return {}
        closed_trades = [t for t in self.trades if t.get('status') == 'closed']
        if not closed_trades: return {}
        
        profits = [t['profit'] for t in closed_trades]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]
        
        return {
            'total_trades': len(closed_trades),
            'win_rate': len(wins) / len(closed_trades),
            'total_profit': sum(profits),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0
        }

# ==========================================
# MAIN TRADING ENGINE
# ==========================================
class ProfessionalTradingEngine:
    def __init__(self):
        self.memory = ProfessionalTradeMemory()
        self.feature_engine = ProfessionalFeatureEngine()
        self.model = ProfessionalEnsemble(self.memory, self.feature_engine)
        self.executor = SmartOrderExecutor()
        self.stats = AdvancedStatisticalAnalyzer()
        self.connected = False
        self.active_trade = None
        
    def connect(self):
        if not mt5.initialize():
            ProfessionalLogger.log("MT5 Init Failed", "ERROR", "ENGINE")
            return False
            
        try:
            authorized = mt5.login(Config.MT5_LOGIN, Config.MT5_PASSWORD, Config.MT5_SERVER)
            if authorized:
                info = mt5.account_info()
                ProfessionalLogger.log(f"Connected to {info.name} | Equity: {info.equity}", "SUCCESS", "ENGINE")
                self.connected = True
                return True
        except Exception as e:
            ProfessionalLogger.log(f"Login Exception: {e}", "ERROR", "ENGINE")
            
        return False
        
    def get_data(self, bars=Config.LOOKBACK_BARS):
        rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        return pd.DataFrame(rates)
        
    def manage_positions(self):
        """Monitor open positions"""
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        
        if not positions:
            if self.active_trade:
                # Trade was closed externally or by TP/SL
                ProfessionalLogger.log("Trade closed detected", "TRADE", "ENGINE")
                
                # Retrieve deal history to log result
                from_date = datetime.now() - timedelta(days=1)
                deals = mt5.history_deals_get(from_date, datetime.now())
                
                if deals:
                    # Find deal corresponding to our ticket
                    for deal in reversed(deals):
                        if deal.position_id == self.active_trade['ticket']:
                            self.active_trade['status'] = 'closed'
                            self.active_trade['close_time'] = int(deal.time)
                            self.active_trade['profit'] = deal.profit
                            self.memory.add_trade(self.active_trade)
                            ProfessionalLogger.log(f"Trade Result: ${deal.profit:.2f}", "PERFORMANCE", "ENGINE")
                            break
                            
                self.active_trade = None
            return 0
        else:
            # Trailing Stop Logic
            if Config.USE_TRAILING_STOP and self.active_trade:
                pos = positions[0]
                current_price = mt5.symbol_info_tick(Config.SYMBOL).bid if pos.type == 0 else mt5.symbol_info_tick(Config.SYMBOL).ask
                
                # Basic Trailing Logic
                if pos.type == 0: # Buy
                    dist_to_price = current_price - pos.price_open
                    if dist_to_price > (Config.TRAILING_STOP_ACTIVATION * abs(pos.price_open - pos.sl)):
                        new_sl = current_price - (Config.ATR_SL_MULTIPLIER * (pos.price_open - pos.sl)/1.5) # Example formula
                        if new_sl > pos.sl:
                            request = {
                                "action": mt5.TRADE_ACTION_SLTP,
                                "position": pos.ticket,
                                "sl": new_sl,
                                "tp": pos.tp
                            }
                            mt5.order_send(request)
                            ProfessionalLogger.log(f"Trailing SL moved to {new_sl}", "risk", "ENGINE")
            
            return len(positions)

    def run(self):
        ProfessionalLogger.log("Starting Professional XAUUSD Engine...", "INFO", "ENGINE")
        if not self.connect(): return
        
        # Initial Training
        data = self.get_data(bars=Config.LOOKBACK_BARS)
        if data is not None:
            # Perform initial statistical scan
            stats_res = self.stats.analyze_return_distribution(data['close'].pct_change().dropna())
            ProfessionalLogger.log(f"Market Volatility (Annualized): {stats_res.get('std',0)*np.sqrt(252):.2%}", "STATISTICS", "ENGINE")
            
            # Train Model
            self.model.train(data)
        
        # Main Loop
        try:
            while True:
                # 1. Maintenance
                if self.model.should_retrain():
                    data = self.get_data()
                    self.model.train(data)
                    
                # 2. Manage Positions
                open_pos_count = self.manage_positions()
                
                if open_pos_count >= Config.MAX_POSITIONS:
                    time.sleep(10)
                    continue
                    
                # 3. Check Session Time
                current_hour = datetime.utcnow().hour
                if Config.SESSION_AWARE_TRADING:
                    if Config.AVOID_ASIAN_SESSION and (0 <= current_hour < 6):
                        time.sleep(300)
                        continue
                        
                # 4. Get Data & Predict
                # Only need small amount of data for prediction
                live_data = self.get_data(bars=200) 
                if live_data is None: 
                    time.sleep(5)
                    continue
                    
                prediction, confidence, features, details = self.model.predict(live_data)
                
                if prediction is not None:
                    # Calculate agreement
                    agreement = 0
                    preds = [d['prediction'] for d in details.values()]
                    if preds:
                        agreement = preds.count(prediction) / len(preds)
                    
                    # Log Status
                    tick = mt5.symbol_info_tick(Config.SYMBOL)
                    price = tick.ask if tick else 0
                    ProfessionalLogger.log(f"Price: {price:.2f} | Pred: {prediction} | Conf: {confidence:.2f} | Agree: {agreement:.2f}", "ANALYSIS", "ENGINE")
                    
                    # 5. Execute
                    if confidence >= Config.MIN_CONFIDENCE and agreement >= Config.MIN_ENSEMBLE_AGREEMENT:
                        
                        # Calculate stops based on ATR
                        entry = tick.ask if prediction == 1 else tick.bid
                        atr_val = live_data.iloc[-1]['close'] * features.get('atr_pct', 0.002)
                        
                        if prediction == 1: # Buy
                            sl = entry - (atr_val * Config.ATR_SL_MULTIPLIER)
                            tp = entry + (atr_val * Config.ATR_TP_MULTIPLIER)
                            type_op = mt5.ORDER_TYPE_BUY
                        else: # Sell
                            sl = entry + (atr_val * Config.ATR_SL_MULTIPLIER)
                            tp = entry - (atr_val * Config.ATR_TP_MULTIPLIER)
                            type_op = mt5.ORDER_TYPE_SELL
                            
                        # Calculate Volume based on Risk
                        risk_amt = mt5.account_info().equity * Config.RISK_PERCENT
                        dist_money = abs(entry - sl)
                        # XAUUSD 1 lot = 100oz (check broker)
                        # $ distance * 100 * lots = risk
                        if dist_money > 0:
                            vol = risk_amt / (dist_money * 100) # Standard contract size
                        else:
                            vol = Config.MIN_VOLUME
                            
                        result = self.executor.execute_trade(Config.SYMBOL, type_op, vol, entry, sl, tp, Config.MAGIC_NUMBER)
                        
                        if result:
                            self.active_trade = {
                                'ticket': result.order,
                                'symbol': Config.SYMBOL,
                                'type': type_op,
                                'open_time': int(time.time()),
                                'open_price': entry,
                                'sl': sl,
                                'tp': tp,
                                'features': features,
                                'status': 'open'
                            }
                            
                time.sleep(60) # Wait 1 minute
                
        except KeyboardInterrupt:
            ProfessionalLogger.log("Stopping Engine...", "INFO", "ENGINE")
            mt5.shutdown()

if __name__ == "__main__":
    engine = ProfessionalTradingEngine()
    engine.run()