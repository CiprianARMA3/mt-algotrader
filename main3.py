import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.feature_selection import SelectFromModel
from scipy import stats, signal
from scipy.optimize import minimize_scalar
from scipy.stats import skew, kurtosis, jarque_bera, norm, t
from arch import arch_model
import warnings
import json
import os
import shutil
import time
from datetime import datetime, timedelta
import traceback

warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    
# ==========================================
# HYBRID CONFIGURATION (Main.py Data + Main2.py Tech)
# ==========================================
class Config:
    # MT5 Credentials
    MT5_LOGIN = 5044108820
    MT5_PASSWORD = "@rC1KbQb"
    MT5_SERVER = "MetaQuotes-Demo"
    
    # Trading Parameters
    SYMBOL = "XAUUSD"
    TIMEFRAME = mt5.TIMEFRAME_M15
    BASE_VOLUME = 0.2
    MAGIC_NUMBER = 998877
    
    # Advanced Risk Management
    RISK_PERCENT = 0.02
    MIN_CONFIDENCE = 0.65
    MIN_ENSEMBLE_AGREEMENT = 0.70
    MAX_POSITIONS = 2
    MAX_DAILY_LOSS_PERCENT = 3.0
    MAX_WEEKLY_LOSS_PERCENT = 7.0
    MAX_DRAWDOWN_PERCENT = 15.0
    MAX_CONSECUTIVE_LOSSES = 3
    KELLY_FRACTION = 0.25
    
    # Statistical Risk Metrics
    VAR_CONFIDENCE = 0.95
    CVAR_CONFIDENCE = 0.95
    MAX_POSITION_CORRELATION = 0.7
    
    # Model Parameters
    LOOKBACK_BARS = 1000
    RETRAIN_HOURS = 24
    TRAINING_MIN_SAMPLES = 500
    WALK_FORWARD_WINDOW = 200
    WALK_FORWARD_STEP = 50
    
    # Hybrid Dataset Configuration (Optimized)
    USE_HISTORICAL_DATASET = True
    HISTORICAL_DATA_LIMIT = 25000  # Higher limit enabled by caching optimization
    HISTORICAL_WEIGHT = 0.4
    RECENT_WEIGHT = 0.6
    
    # Ensemble Configuration
    USE_STACKING_ENSEMBLE = True
    MIN_DATA_QUALITY_SCORE = 0.6
    
    # Learning Parameters
    TRADE_HISTORY_FILE = "trade_history.json"
    MODEL_SAVE_FILE = "ensemble_model.joblib"
    BACKTEST_RESULTS_FILE = "backtest_results.json"
    MEMORY_SIZE = 1000
    LEARNING_WEIGHT = 0.4
    
    # Technical Parameters
    ATR_PERIOD = 14
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    
    # Statistical Feature Parameters
    GARCH_VOL_PERIOD = 10
    HURST_WINDOW = 100
    TAIL_INDEX_WINDOW = 50
    
    # Price Action Parameters
    USE_SMART_ENTRY = True
    USE_DYNAMIC_SL_TP = True
    MIN_RR_RATIO = 1.8
    LOOKBACK_SWING_POINTS = 50
    
    # Multi-timeframe Parameters
    MULTI_TIMEFRAME_ENABLED = True
    TIMEFRAMES = ['M5', 'M15', 'H1']
    TIMEFRAME_ALIGNMENT_THRESHOLD = 0.70
    
    # Market Regime Parameters
    USE_MARKET_REGIME = True
    ADX_TREND_THRESHOLD = 25
    ADX_STRONG_TREND_THRESHOLD = 40
    ADX_SLOPE_THRESHOLD = 0.5
    
    # Order Execution
    MAX_SLIPPAGE_PIPS = 3
    ORDER_TIMEOUT_SECONDS = 30
    COMMISSION_PER_LOT = 3.5

# ==========================================
# PROFESSIONAL LOGGING
# ==========================================
class ProfessionalLogger:
    COLORS = {
        'RESET': '\033[0m', 'RED': '\033[91m', 'GREEN': '\033[92m',
        'YELLOW': '\033[93m', 'BLUE': '\033[94m', 'MAGENTA': '\033[95m',
        'CYAN': '\033[96m', 'WHITE': '\033[97m', 'GRAY': '\033[90m'
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
            'PERFORMANCE': ProfessionalLogger.COLORS['MAGENTA']
        }
        color = colors.get(level, ProfessionalLogger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level:8s}{ProfessionalLogger.COLORS['RESET']}] [{component:12s}] {message}", flush=True)

# ==========================================
# FAST DATA LOADING (OPTIMIZED)
# ==========================================
class DataLoader:
    """Optimized Data Loader with Caching and Slicing"""
    
    @staticmethod
    def load_huggingface_dataset():
        cache_file = "historical_data_cache_v3.pkl"
        
        # 1. Try local cache first
        if os.path.exists(cache_file):
            try:
                # Cache valid for 24h
                if (datetime.now().timestamp() - os.path.getmtime(cache_file)) < 86400:
                    ProfessionalLogger.log("Loading dataset from local cache...", "DATA", "LOADER")
                    df = pd.read_pickle(cache_file)
                    if hasattr(Config, 'HISTORICAL_DATA_LIMIT'):
                        df = df.tail(Config.HISTORICAL_DATA_LIMIT)
                    return df
            except Exception as e:
                ProfessionalLogger.log(f"Cache load failed: {e}", "WARNING", "LOADER")

        try:
            ProfessionalLogger.log("Loading historical XAU/USD dataset (Optimized)...", "DATA", "LOADER")
            
            # Load dataset in streaming mode (split only)
            ds = load_dataset("ZombitX64/xauusd-gold-price-historical-data-2004-2025", split='train')
            
            # Optimization: Fetch only needed rows using slicing
            needed_rows = getattr(Config, 'HISTORICAL_DATA_LIMIT', 25000) * 2
            needed_rows = max(needed_rows, 10000)
            
            total_rows = len(ds)
            start_idx = max(0, total_rows - needed_rows)
            
            ProfessionalLogger.log(f"Fetching last {needed_rows} rows from {total_rows} total...", "DATA", "LOADER")
            
            data_slice = ds[start_idx:]
            df = pd.DataFrame(data_slice)
            
            # Standardize columns
            column_mapping = {
                'Date': 'time', 'Open': 'open', 'High': 'high', 
                'Low': 'low', 'Close': 'close', 'Volume': 'tick_volume'
            }
            
            for old_col, new_col in column_mapping.items():
                matching_cols = [col for col in df.columns if col.lower() == old_col.lower()]
                if matching_cols:
                    df.rename(columns={matching_cols[0]: new_col}, inplace=True)
            
            # Time processing
            if 'time' in df.columns:
                if df['time'].dtype == 'object':
                    df['time'] = pd.to_datetime(df['time']).view('int64') // 10**9
                elif np.issubdtype(df['time'].dtype, np.datetime64):
                     df['time'] = df['time'].view('int64') // 10**9
            
            df = df.sort_values('time').reset_index(drop=True)
            
            # Save to cache
            try:
                df.to_pickle(cache_file)
            except:
                pass
                
            if hasattr(Config, 'HISTORICAL_DATA_LIMIT'):
                df = df.tail(Config.HISTORICAL_DATA_LIMIT)
                
            ProfessionalLogger.log(f"Loaded {len(df)} rows successfully", "SUCCESS", "LOADER")
            return df
            
        except Exception as e:
            ProfessionalLogger.log(f"Failed to load dataset: {str(e)}", "ERROR", "LOADER")
            return None

# ==========================================
# ADVANCED FEATURE ENGINE
# ==========================================
class ProfessionalFeatureEngine:
    """Advanced feature engineering including Market Regime"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_features(self, df):
        df = df.copy()
        
        # Basic price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Moving Averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            
        # MA Crossovers
        df['ma_cross_5_20'] = df['sma_5'] - df['sma_20']
        df['ma_cross_10_50'] = df['sma_10'] - df['sma_50']
        df['ma_cross_50_200'] = df['sma_50'] - df['sma_200']
        
        # Volatility & ATR
        df['tr'] = np.maximum(df['high'] - df['low'], 
                            np.maximum(abs(df['high'] - df['close'].shift(1)), 
                                     abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(Config.ATR_PERIOD).mean()
        df['atr_percent'] = df['atr'] / df['close']
        df['volatility'] = df['returns'].rolling(20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(Config.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(Config.RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50
        
        # MACD
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # ADX
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        plus_dm = np.where((plus_dm > minus_dm) & (plus_dm > 0), plus_dm, 0.0)
        minus_dm = np.where((minus_dm > plus_dm) & (minus_dm > 0), minus_dm, 0.0)
        tr_s = df['tr'].rolling(Config.ADX_PERIOD).sum()
        plus_di = 100 * (pd.Series(plus_dm).rolling(Config.ADX_PERIOD).sum() / tr_s)
        minus_di = 100 * (pd.Series(minus_dm).rolling(Config.ADX_PERIOD).sum() / tr_s)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        df['adx'] = dx.rolling(Config.ADX_PERIOD).mean()
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        df['adx_slope'] = df['adx'].diff()
        df['di_spread'] = abs(plus_di - minus_di)
        
        # Market Regime
        conditions = [
            (df['adx'] < Config.ADX_TREND_THRESHOLD),
            (df['adx'] >= Config.ADX_TREND_THRESHOLD) & (df['adx'] < Config.ADX_STRONG_TREND_THRESHOLD),
            (df['adx'] >= Config.ADX_STRONG_TREND_THRESHOLD)
        ]
        df['regime'] = np.select(conditions, [0, 1, 2], default=0)
        
        # Momentum
        for p in [3, 5, 10, 20]:
            df[f'momentum_{p}'] = df['close'].pct_change(p)
            
        # Time features
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volatility_lag_{lag}'] = df['volatility'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi_normalized'].shift(lag)
            
        return df.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(0)

    def create_labels(self, df, forward_bars=5, threshold=0.001):
        df = df.copy()
        df['future_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
        vol_threshold = threshold * (1 + df['volatility'] * 10)
        df['label'] = -1
        df.loc[df['future_return'] > vol_threshold, 'label'] = 1
        df.loc[df['future_return'] < -vol_threshold, 'label'] = 0
        return df

    def get_feature_columns(self, df):
        exclude = ['time', 'datetime', 'open', 'high', 'low', 'close', 'tick_volume', 
                  'spread', 'real_volume', 'label', 'future_return', 'target', 'volatility_100']
        return [c for c in df.columns if c not in exclude]

# ==========================================
# ADVANCED ML COMPONENTS (OPTIMIZED)
# ==========================================
class TripleBarrierLabeling:
    @staticmethod
    def get_events(close, t_events, pt_sl, target, min_ret, vertical_barrier_times=None):
        target = target.loc[t_events]
        target = target[target > min_ret]
        if vertical_barrier_times is None:
            vertical_barrier_times = pd.Series(pd.NaT, index=t_events)
        out = []
        for loc, t in t_events.items():
            if loc not in close.index: continue
            close_subset = close[loc:]
            path = close_subset / close_subset.iloc[0] - 1
            t1 = vertical_barrier_times.get(loc, pd.NaT)
            top = pt_sl[0] * target.get(loc, 0)
            bottom = -pt_sl[1] * target.get(loc, 0)
            
            touch_times = pd.concat([
                path[path > top].head(1).index,
                path[path < bottom].head(1).index,
                pd.Index([t1]) if not pd.isna(t1) else pd.Index([])
            ]).sort_values()
            
            if len(touch_times) > 0:
                out.append([loc, touch_times[0]])
        return pd.DataFrame(out, columns=['t0', 't1']).set_index('t0')

    @staticmethod
    def get_bins(events, close):
        events = events.dropna(subset=['t1'])
        px = events.index.union(events['t1'].values).drop_duplicates()
        px = close.reindex(px, method='bfill')
        out = pd.DataFrame(index=events.index)
        out['ret'] = px.loc[events['t1'].values].values / px.loc[events.index].values - 1
        out['bin'] = np.sign(out['ret'])
        out['bin'].loc[out['ret'] == 0] = 0
        out['bin'] = out['bin'].apply(lambda x: 1 if x > 0 else 0)
        return out

class HyperparameterTuner:
    @staticmethod
    def tune_model(model, param_grid, X, y, cv=3, n_iter=5):
        if len(X) < 100: return model
        try:
            search = RandomizedSearchCV(
                estimator=model, param_distributions=param_grid, n_iter=n_iter,
                cv=TimeSeriesSplit(n_splits=cv), scoring='accuracy',
                n_jobs=-1, random_state=42, verbose=0, error_score='raise'
            )
            search.fit(X, y)
            ProfessionalLogger.log(f"Best params for {type(model).__name__}: {search.best_params_}", "LEARN", "TUNER")
            return search.best_estimator_
        except:
            return model

class FeatureSelector:
    def __init__(self, n_features_to_select=20):
        self.n_features = n_features_to_select
        self.selected_features = None
    
    def fit(self, X, y, estimator=None):
        try:
            if estimator is None:
                estimator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
            estimator.fit(X, y)
            selector = SelectFromModel(estimator, max_features=self.n_features, threshold=-np.inf, prefit=True)
            feature_idx = selector.get_support()
            self.selected_features = X.columns[feature_idx].tolist()
            if len(self.selected_features) < 5:
                importances = estimator.feature_importances_
                indices = np.argsort(importances)[::-1]
                self.selected_features = X.columns[indices[:self.n_features]].tolist()
            ProfessionalLogger.log(f"Selected {len(self.selected_features)} features", "LEARN", "SELECTOR")
            return self.selected_features
        except:
            return X.columns.tolist()

# ==========================================
# PROFESSIONAL TRADING ENGINE COMPONENTS
# ==========================================
class ProfessionalTradeMemory:
    def __init__(self):
        self.history = []
    def add_trade(self, trade): pass
    def update_trade(self, ticket, data): pass

class ProfessionalEnsemble:
    def __init__(self, trade_memory, feature_engine):
        self.trade_memory = trade_memory
        self.feature_engine = feature_engine
        self.tuner = HyperparameterTuner()
        self.selector = FeatureSelector(n_features_to_select=25)
        self.labeler = TripleBarrierLabeling()
        self.scaler = StandardScaler()
        self.base_models = self._initialize_base_models()
        self.ensemble = self._create_ensemble()
        self.is_trained = False
        self.selected_features_names = []
        
    def _initialize_base_models(self):
        models = [
            ('GB', GradientBoostingClassifier(n_estimators=100, max_depth=3, learning_rate=0.1)),
            ('RF', RandomForestClassifier(n_estimators=100, max_depth=7, n_jobs=-1)),
            ('LR', LogisticRegression(max_iter=1000, class_weight='balanced'))
        ]
        if XGB_AVAILABLE:
            models.append(('XGB', XGBClassifier(n_estimators=100, max_depth=3, n_jobs=-1)))
        return models

    def _create_ensemble(self):
        estimators = [(n, m) for n, m in self.base_models]
        if Config.USE_STACKING_ENSEMBLE:
            return StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(), cv=3, n_jobs=-1)
        return VotingClassifier(estimators=estimators, voting='soft')

    def _create_advanced_labels(self, df):
        try:
            df['volatility_100'] = df['close'].pct_change().rolling(100).std()
            t_events = df.index
            pt_sl = [2.0, 1.0]
            target = df['volatility_100']
            vertical_barrier = pd.Series(df.index + 20, index=df.index).apply(lambda x: min(x, df.index[-1]))
            
            events = self.labeler.get_events(df['close'], pd.Series(t_events, index=t_events), 
                                           pt_sl, target, 0.001, vertical_barrier)
            labels = self.labeler.get_bins(events, df['close'])
            df_labeled = df.copy()
            df_labeled.loc[labels.index, 'label'] = labels['bin']
            df_labeled.dropna(subset=['label'], inplace=True)
            
            if len(df_labeled) < 50: raise ValueError("Not enough labels")
            return df_labeled
        except:
            return self.feature_engine.create_labels(df)

    def train(self, data):
        try:
            df = self.feature_engine.calculate_features(data)
            df = self._create_advanced_labels(df)
            
            feature_cols = self.feature_engine.get_feature_columns(df)
            X = df[feature_cols].fillna(0)
            y = df['label']
            
            if len(X) < 100: return False
            
            # Select Features
            self.selected_features_names = self.selector.fit(X, y)
            X = X[self.selected_features_names]
            
            # Tune Models (Simplified)
            tuned_models = []
            for name, model in self.base_models:
                if name == 'RF':
                    param_grid = {'max_depth': [5, 10], 'n_estimators': [50, 100]}
                    tuned_models.append((name, self.tuner.tune_model(model, param_grid, X, y, n_iter=3)))
                else:
                    tuned_models.append((name, model))
            self.base_models = tuned_models
            self.ensemble = self._create_ensemble()
            
            # Train
            X_scaled = self.scaler.fit_transform(X)
            self.ensemble.fit(X_scaled, y)
            self.is_trained = True
            ProfessionalLogger.log("Ensemble training complete", "SUCCESS", "ENSEMBLE")
            return True
        except Exception as e:
            ProfessionalLogger.log(f"Training error: {e}", "ERROR", "ENSEMBLE")
            return False

    def predict(self, df):
        if not self.is_trained:
            return None, 0.0, {}, {}
            
        try:
            df_feat = self.feature_engine.calculate_features(df)
            feature_cols = self.feature_engine.get_feature_columns(df_feat)
            
            X = df_feat[feature_cols].iloc[-1:].fillna(0)
            if self.selected_features_names:
                X = X[self.selected_features_names]
            
            X_scaled = self.scaler.transform(X)
            prob = self.ensemble.predict_proba(X_scaled)[0]
            prediction = np.argmax(prob)
            confidence = np.max(prob)
            
            # ADX Filter
            if Config.USE_MARKET_REGIME and 'adx' in df_feat.columns:
                adx = df_feat['adx'].iloc[-1]
                if adx < 20 and confidence < 0.7: confidence *= 0.8
                
            return prediction, confidence, {}, {}
        except Exception as e:
            ProfessionalLogger.log(f"Prediction error: {e}", "ERROR", "ENSEMBLE")
            return None, 0.0, {}, {}

class ProfessionalTradingEngine:
    def __init__(self):
        self.feature_engine = ProfessionalFeatureEngine()
        self.trade_memory = ProfessionalTradeMemory()
        self.model = ProfessionalEnsemble(self.trade_memory, self.feature_engine)
        self.connected = False
        
    def connect_mt5(self):
        if not mt5.initialize(login=Config.MT5_LOGIN, password=Config.MT5_PASSWORD, server=Config.MT5_SERVER):
            ProfessionalLogger.log(f"MT5 Init Failed: {mt5.last_error()}", "ERROR")
            if not mt5.initialize(): return False
        self.connected = True
        ProfessionalLogger.log(f"Connected to MT5. Account: {mt5.account_info().login}", "SUCCESS")
        return True

    def run(self):
        print("="*60)
        print("  MAIN3.PY - HYBRID OPTIMIZED TRADING ENGINE")
        print("="*60)
        
        self.connect_mt5()
        
        # 1. Load Data (Hybrid)
        all_data = []
        
        # Historical (Fast Load)
        if Config.USE_HISTORICAL_DATASET:
            df_hist = DataLoader.load_huggingface_dataset()
            if df_hist is not None:
                all_data.append(df_hist)
                
        # Recent (MT5)
        recent_bars = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, 5000)
        if recent_bars is not None:
            df_recent = pd.DataFrame(recent_bars)
            df_recent['time'] = df_recent['time'].astype(np.int64) 
            all_data.append(df_recent)
            
        if not all_data:
            ProfessionalLogger.log("No data available!", "ERROR")
            return
            
        # Combine
        full_data = pd.concat(all_data).drop_duplicates(subset='time').sort_values('time').reset_index(drop=True)
        ProfessionalLogger.log(f"Combined Training Data: {len(full_data)} bars", "INFO")
        
        # 2. Train
        if not self.model.train(full_data):
            ProfessionalLogger.log("Initial Training Failed", "ERROR")
            return
            
        # 3. Live Loop
        ProfessionalLogger.log("Starting Live Trading Loop...", "INFO")
        try:
            while True:
                time.sleep(1 if datetime.now().second == 0 else 0.1)
                if datetime.now().second != 0: continue
                
                # Get live data
                rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, 200)
                if rates is None: continue
                
                df_live = pd.DataFrame(rates)
                df_live['time'] = df_live['time'].astype(np.int64)
                
                signal, conf, _, _ = self.model.predict(df_live)
                
                if signal is not None:
                    # Basic logging
                    price = df_live['close'].iloc[-1]
                    sig_str = "BUY" if signal == 1 else "SELL"
                    ProfessionalLogger.log(f"Price: {price:.2f} | Signal: {sig_str} | Conf: {conf:.1%}", "INFO")
                    
                    if conf > Config.MIN_CONFIDENCE:
                         ProfessionalLogger.log(f"⚠️ HIGH CONFIDENCE {sig_str} SIGNAL!", "TRADE")
                         
        except KeyboardInterrupt:
            mt5.shutdown()
            print("\nStopped.")

if __name__ == "__main__":
    ProfessionalTradingEngine().run()
