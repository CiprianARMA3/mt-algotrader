import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
import warnings
import sys

warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==========================================
# CONFIGURATION (Adapted from main.py)
# ==========================================
class Config:
    SYMBOL = "XAUUSD"
    BASE_VOLUME = 0.2
    MAGIC_NUMBER = 998877
    
    # Risk Management
    RISK_PERCENT = 0.02
    MIN_CONFIDENCE = 0.6
    MIN_ENSEMBLE_AGREEMENT = 0.67
    MAX_POSITIONS = 2
    KELLY_FRACTION = 0.5
    
    # Model Parameters
    LOOKBACK_BARS = 1000
    TRAINING_DAYS = 60         # Days of history to load for training from JSON
    TESTING_DAYS = 7           # Days of history to load for testing from JSON
    TRAINING_MIN_SAMPLES = 100
    
    # Ensemble Configuration
    USE_STACKING_ENSEMBLE = True
    MIN_DATA_QUALITY_SCORE = 0.6 # Slightly lower for local data tolerance
    
    # Technical Parameters
    ATR_PERIOD = 14
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    
    # Price Action Parameters
    USE_SMART_ENTRY = True
    USE_DYNAMIC_SL_TP = True
    MIN_RR_RATIO = 1.5
    LOOKBACK_SWING_POINTS = 50
    
    # Market Regime & ADX Parameters
    USE_MARKET_REGIME = True
    ADX_TREND_THRESHOLD = 25
    ADX_STRONG_TREND_THRESHOLD = 40
    ADX_SLOPE_THRESHOLD = 0.5
    
    # Filesystem
    DATABASE_ROOT = "database"
    TEST_TRADES_DIR = "test-trades"
    INITIAL_BALANCE = 3000.0

# ==========================================
# LOGGING
# ==========================================
class Logger:
    COLORS = {
        'RESET': '\033[0m', 'RED': '\033[91m', 'GREEN': '\033[92m',
        'YELLOW': '\033[93m', 'BLUE': '\033[94m', 'MAGENTA': '\033[95m',
        'CYAN': '\033[96m', 'WHITE': '\033[97m'
    }
    
    @staticmethod
    def log(message, level='INFO'):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        colors = {
            'INFO': Logger.COLORS['CYAN'], 'SUCCESS': Logger.COLORS['GREEN'],
            'WARNING': Logger.COLORS['YELLOW'], 'ERROR': Logger.COLORS['RED'],
            'TRADE': Logger.COLORS['MAGENTA'], 'DATA': Logger.COLORS['WHITE'],
            'ENSEMBLE': Logger.COLORS['GREEN'], 'PERFORMANCE': Logger.COLORS['MAGENTA']
        }
        color = colors.get(level, Logger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level}{Logger.COLORS['RESET']}] {message}", flush=True)

# ==========================================
# DATA QUALITY CHECKER (From main.py)
# ==========================================
class DataQualityChecker:
    @staticmethod
    def check_data_quality(df):
        scores = []
        # Missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        scores.append(max(0, 1 - missing_ratio * 2))
        
        # Variance check
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variance_score = 1.0
        for col in numeric_cols:
            if df[col].std() == 0: variance_score *= 0.3
            elif df[col].nunique() < 10: variance_score *= 0.7
        scores.append(variance_score)
        
        # Chronological order
        if 'time' in df.columns:
            is_sorted = df['time'].is_monotonic_increasing
            scores.append(1.0 if is_sorted else 0.5)
            
        overall_score = np.mean(scores)
        return overall_score, scores

    @staticmethod
    def validate_features(df, feature_cols):
        invalid_features = []
        for col in feature_cols:
            if col not in df.columns:
                invalid_features.append(f"{col} (missing)")
                continue
            if np.isinf(df[col]).any():
                invalid_features.append(f"{col} (has inf)")
            nan_ratio = df[col].isnull().sum() / len(df)
            if nan_ratio > 0.2:
                invalid_features.append(f"{col} ({nan_ratio:.1%} NaN)")
            if df[col].std() == 0:
                invalid_features.append(f"{col} (no variance)")
        return invalid_features

# ==========================================
# ADVANCED FEATURE ENGINE (From main.py)
# ==========================================
class AdvancedFeatureEngine:
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
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        
        df['ma_cross_5_20'] = df['sma_5'] - df['sma_20']
        df['ma_cross_10_50'] = df['sma_10'] - df['sma_50']
        
        # ATR and volatility
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
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
        
        # ENHANCED ADX & REGIME
        if Config.USE_MARKET_REGIME:
            adx_data = self.calculate_adx(df)
            df['adx'] = adx_data['adx']
            df['plus_di'] = adx_data['plus_di']
            df['minus_di'] = adx_data['minus_di']
            
            df['trend_strength'] = df['adx'] / 100
            df['trend_direction'] = np.where(df['plus_di'] > df['minus_di'], 1, -1)
            df['adx_slope'] = df['adx'].diff()
            df['di_spread'] = abs(df['plus_di'] - df['minus_di'])
            
            conditions = [
                (df['adx'] < Config.ADX_TREND_THRESHOLD),
                (df['adx'] >= Config.ADX_TREND_THRESHOLD) & (df['adx'] < Config.ADX_STRONG_TREND_THRESHOLD),
                (df['adx'] >= Config.ADX_STRONG_TREND_THRESHOLD)
            ]
            choices = [0, 1, 2]
            df['regime'] = np.select(conditions, choices, default=0)
        
        # Momentum
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # Support/Resistance
        df['distance_to_high'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['distance_to_low'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        
        # Time features
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'])
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
        else:
            df['hour'] = 12
            df['day_of_week'] = 2
            df['month'] = 1
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['gold_seasonal'] = df['month'].apply(lambda x: 1 if x in [9, 10, 11, 12] else 0)
        df['session'] = df['hour'].apply(self._get_market_session)
        
        return df
    
    def calculate_adx(self, df, period=Config.ADX_PERIOD):
        high, low, close = df['high'], df['low'], df['close']
        tr = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        tr_smooth = pd.Series(tr).rolling(period).sum()
        plus_dm_smooth = pd.Series(plus_dm).rolling(period).sum()
        minus_dm_smooth = pd.Series(minus_dm).rolling(period).sum()
        
        plus_di = 100 * (plus_dm_smooth / tr_smooth)
        minus_di = 100 * (minus_dm_smooth / tr_smooth)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return pd.DataFrame({'adx': adx.fillna(0), 'plus_di': plus_di.fillna(0), 'minus_di': minus_di.fillna(0)})
    
    def _get_market_session(self, hour):
        if 0 <= hour < 8: return 0
        elif 8 <= hour < 16: return 1
        else: return 2
    
    def create_labels(self, df, forward_bars=3, threshold=0.001):
        df = df.copy()
        df['forward_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
        volatility = df['returns'].rolling(20).std().fillna(0.001)
        dynamic_threshold = threshold * (1 + volatility * 10)
        
        df['label'] = -1
        df.loc[df['forward_return'] > dynamic_threshold, 'label'] = 1
        df.loc[df['forward_return'] < -dynamic_threshold, 'label'] = 0
        return df
    
    def get_feature_columns(self):
        base_features = [
            'returns', 'log_returns', 'hl_ratio', 'co_ratio', 'hlc3',
            'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50',
            'ma_cross_5_20', 'ma_cross_10_50',
            'atr_percent', 'volatility', 'rsi_normalized', 'macd_hist',
            'bb_width', 'bb_position', 'momentum_5', 'momentum_10', 'roc_10',
            'distance_to_high', 'distance_to_low',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'gold_seasonal', 'session'
        ]
        if Config.USE_MARKET_REGIME:
            base_features.extend(['trend_strength', 'regime', 'plus_di', 'minus_di', 'adx_slope', 'di_spread'])
        return base_features

# ==========================================
# ADVANCED ENSEMBLE MODEL (From main.py)
# ==========================================
class AdvancedEnsemble:
    def __init__(self):
        self.feature_engine = AdvancedFeatureEngine()
        self.data_quality_checker = DataQualityChecker()
        self.base_models = self._initialize_base_models()
        
        if Config.USE_STACKING_ENSEMBLE:
            self.ensemble = self._create_stacking_ensemble()
        else:
            self.ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in self.base_models],
                voting='soft', weights=[1.0, 1.2, 1.0, 1.1]
            )
            
        self.calibrated_ensemble = CalibratedClassifierCV(self.ensemble, method='sigmoid', cv=3)
        self.is_trained = False
        
    def _initialize_base_models(self):
        models = [
            ('GB', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)),
            ('RF', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1, class_weight='balanced')),
            ('LR', LogisticRegression(max_iter=1000, random_state=42, penalty='l2', class_weight='balanced')),
            ('NN', MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=1000, random_state=42))
        ]
        if XGB_AVAILABLE:
            models.append(('XGB', XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, n_jobs=-1)))
        return models
    
    def _create_stacking_ensemble(self):
        base_estimators = [(name, model) for name, model in self.base_models if name != 'NN']
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=3, passthrough=True, n_jobs=-1
        )
    
    def train(self, df):
        Logger.log("Training Advanced Ensemble on local data...", "ENSEMBLE")
        
        # 1. Feature Engineering
        df_processed = self.feature_engine.calculate_features(df)
        df_labeled = self.feature_engine.create_labels(df_processed).dropna()
        df_labeled = df_labeled[df_labeled['label'] != -1]
        
        # 2. Quality Check
        quality, _ = self.data_quality_checker.check_data_quality(df_labeled)
        if quality < Config.MIN_DATA_QUALITY_SCORE:
            Logger.log(f"Data quality {quality:.2%} too low", "WARNING")
            
        if len(df_labeled) < Config.TRAINING_MIN_SAMPLES:
            Logger.log(f"Insufficient training data: {len(df_labeled)}", "ERROR")
            return False
            
        feature_cols = self.feature_engine.get_feature_columns()
        X = df_labeled[feature_cols].fillna(0)
        y = df_labeled['label']
        
        X_scaled = self.feature_engine.scaler.fit_transform(X)
        
        try:
            self.calibrated_ensemble.fit(X_scaled, y)
            self.is_trained = True
            Logger.log(f"✅ Ensemble trained on {len(df_labeled)} samples", "SUCCESS")
            return True
        except Exception as e:
            Logger.log(f"Training failed: {str(e)}", "ERROR")
            return False
            
    def predict(self, df):
        if not self.is_trained:
            return None, 0.0, None, {}
            
        df_feat = self.feature_engine.calculate_features(df)
        feature_cols = self.feature_engine.get_feature_columns()
        
        X = df_feat[feature_cols].iloc[-1:].fillna(0).values
        f_dict = {col: float(X[0][i]) for i, col in enumerate(feature_cols)}
        
        X_scaled = self.feature_engine.scaler.transform(X)
        
        # Get base model details
        model_details = {}
        if hasattr(self.calibrated_ensemble, 'calibrated_classifiers_'):
            # Approximation for calibrated stacking
            pass 
            
        final_p = self.calibrated_ensemble.predict(X_scaled)[0]
        proba = self.calibrated_ensemble.predict_proba(X_scaled)[0]
        final_c = np.max(proba)
        
        # --- ENHANCED ADX FILTERING LOGIC (From main.py) ---
        if Config.USE_MARKET_REGIME and 'regime' in f_dict:
            regime = f_dict['regime']
            adx = f_dict.get('adx', 0)
            plus_di = f_dict.get('plus_di', 0)
            minus_di = f_dict.get('minus_di', 0)
            adx_slope = f_dict.get('adx_slope', 0)

            # 1. Filter: Do not fight a Strong Trend (ADX > 25)
            if adx > Config.ADX_TREND_THRESHOLD:
                if plus_di > minus_di and final_p == 0: return None, 0.0, None, {}
                if minus_di > plus_di and final_p == 1: return None, 0.0, None, {}

            # 2. Logic: Adjust Confidence based on ADX Slope
            confidence_threshold = Config.MIN_CONFIDENCE
            if adx_slope > Config.ADX_SLOPE_THRESHOLD: confidence_threshold *= 1.1
            if regime == 2: confidence_threshold *= 1.3
            
            if final_c < confidence_threshold: return None, 0.0, None, {}
            
        return final_p, final_c, f_dict, model_details

# ==========================================
# ENHANCED PRICE ACTION (From main.py)
# ==========================================
class EnhancedPriceActionAnalyzer:
    @staticmethod
    def find_support_resistance_clusters(df, current_price, lookback=100):
        # Simplified cluster finding using highs/lows for backtest speed
        df_slice = df.tail(lookback)
        highs = df_slice['high'].values
        lows = df_slice['low'].values
        
        resistances = [h for h in highs if h > current_price]
        supports = [l for l in lows if l < current_price]
        
        resistance = min(resistances) if resistances else current_price * 1.01
        support = max(supports) if supports else current_price * 0.99
        
        return support, resistance

    @staticmethod
    def calculate_fibonacci_levels(df):
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        price_range = recent_high - recent_low
        return {
            '0.0': recent_low,
            '0.236': recent_low + price_range * 0.236,
            '0.382': recent_low + price_range * 0.382,
            '0.5': recent_low + price_range * 0.5,
            '0.618': recent_low + price_range * 0.618,
            '1.0': recent_high
        }

    @staticmethod
    def calculate_optimal_entry_sl_tp(df, signal, current_price, atr, risk_reward_ratio=Config.MIN_RR_RATIO):
        support, resistance = EnhancedPriceActionAnalyzer.find_support_resistance_clusters(df, current_price, Config.LOOKBACK_SWING_POINTS)
        fib_levels = EnhancedPriceActionAnalyzer.calculate_fibonacci_levels(df)
        
        if signal == 1: # BUY
            optimal_entry = current_price
            sl_candidates = [support - (atr * 0.5), current_price - (atr * 1.5)]
            sl = max(sl_candidates)
            
            tp_candidates = [
                resistance,
                fib_levels['1.0'],
                current_price + (atr * 3),
                current_price + (abs(current_price - sl) * risk_reward_ratio)
            ]
            valid_tps = [tp for tp in tp_candidates if tp > optimal_entry]
            
            if valid_tps:
                final_tp = min([tp for tp in sorted(valid_tps) if (tp - optimal_entry) / (optimal_entry - sl) >= risk_reward_ratio])
            else:
                final_tp = optimal_entry + (atr * 3)
                
        else: # SELL
            optimal_entry = current_price
            sl_candidates = [resistance + (atr * 0.5), current_price + (atr * 1.5)]
            sl = min(sl_candidates)
            
            tp_candidates = [
                support,
                fib_levels['0.0'],
                current_price - (atr * 3),
                current_price - (abs(sl - current_price) * risk_reward_ratio)
            ]
            valid_tps = [tp for tp in tp_candidates if tp < optimal_entry]
            
            if valid_tps:
                final_tp = max([tp for tp in sorted(valid_tps, reverse=True) if (optimal_entry - tp) / (sl - optimal_entry) >= risk_reward_ratio])
            else:
                final_tp = optimal_entry - (atr * 3)
                
        # Smart Entry Logic (Wait for pullback)
        if Config.USE_SMART_ENTRY:
             # In backtest, we assume we get the optimal entry if price touches it in near future
             # For simplicity in loop, we set entry slightly better than close
             if signal == 1: optimal_entry = max(support, current_price - (atr * 0.2))
             else: optimal_entry = min(resistance, current_price + (atr * 0.2))
        
        risk_amount = abs(optimal_entry - sl)
        reward_amount = abs(final_tp - optimal_entry)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return {'optimal_entry': optimal_entry, 'sl': sl, 'tp': final_tp, 'rr_ratio': rr_ratio}

# ==========================================
# BACKTEST ENGINE
# ==========================================
class BacktestEngine:
    def __init__(self, start_datetime):
        self.start_datetime = start_datetime
        self.model = AdvancedEnsemble()
        self.price_action = EnhancedPriceActionAnalyzer()
        self.test_trades_dir = Config.TEST_TRADES_DIR
        
        os.makedirs(self.test_trades_dir, exist_ok=True)
        
        self.balance = Config.INITIAL_BALANCE
        self.open_positions = []
        self.closed_trades = []
        self.trade_counter = 0

    def load_historical_data(self):
        """Loads Old (Training) and New (Testing) data from JSONs"""
        Logger.log(f"Loading data centered on {self.start_datetime}...", "DATA")
        
        train_start = self.start_datetime - timedelta(days=Config.TRAINING_DAYS)
        test_end = self.start_datetime + timedelta(days=Config.TESTING_DAYS)
        
        # 1. Training Data (Old)
        train_df = self._fetch_data_from_json(train_start, self.start_datetime)
        
        # 2. Testing Data (Future)
        test_df = self._fetch_data_from_json(self.start_datetime, test_end)
        
        if train_df is None or train_df.empty:
            Logger.log("No historical data found for training", "ERROR")
            return None, None
            
        if test_df is None or test_df.empty:
            Logger.log("No future data found for testing", "ERROR")
            return train_df, None
            
        Logger.log(f"Loaded {len(train_df)} training bars and {len(test_df)} testing bars", "SUCCESS")
        return train_df, test_df

    def _fetch_data_from_json(self, start_dt, end_dt):
        """Traverse /database/YYYY/MM/DD/data.json"""
        combined_data = []
        delta = end_dt - start_dt
        days_count = delta.days + 1
        
        for i in range(days_count):
            current_date = start_dt + timedelta(days=i)
            year, month, day = str(current_date.year), f"{current_date.month:02d}", f"{current_date.day:02d}"
            file_path = os.path.join(Config.DATABASE_ROOT, year, month, day, "data.json")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        day_data = json.load(f)
                        if day_data: combined_data.extend(day_data)
                except Exception as e:
                    Logger.log(f"Error reading {file_path}: {e}", "WARNING")
                    
        if not combined_data: return None
            
        df = pd.DataFrame(combined_data)
        if 'time' not in df.columns: return None
        
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').drop_duplicates(subset=['time'])
        df = df[(df['time'] >= start_dt) & (df['time'] < end_dt)]
        
        for col in ['open', 'high', 'low', 'close']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
        return df.reset_index(drop=True)

    def execute_backtest(self, train_data, test_data):
        Logger.log("=" * 60, "BACKTEST")
        Logger.log("🔬 STARTING SIMULATION (Main.py Logic)", "BACKTEST")
        Logger.log("=" * 60, "BACKTEST")
        
        # 1. Train Stacking Ensemble on Old Data
        if not self.model.train(train_data): return

        # 2. Fix Cold Start: Pre-fill buffer with end of training data
        context_buffer = train_data.tail(Config.LOOKBACK_BARS).copy()
        
        # 3. Simulation Loop
        for idx, row in test_data.iterrows():
            current_bar = row
            current_time = current_bar['time']
            current_price = current_bar['close']
            
            # --- Manage Existing Positions ---
            for position in self.open_positions[:]:
                self._check_exit(position, current_bar)

            # --- Update Buffer ---
            context_buffer = pd.concat([context_buffer, pd.DataFrame([row])], ignore_index=True)
            if len(context_buffer) > Config.LOOKBACK_BARS:
                context_buffer = context_buffer.iloc[-Config.LOOKBACK_BARS:]
            
            # --- Generate Signal (Enhanced Logic) ---
            if len(self.open_positions) < Config.MAX_POSITIONS:
                # This calls the AdvancedEnsemble predict which includes ADX/Regime filtering
                signal, confidence, features, details = self.model.predict(context_buffer)
                
                if signal is not None and confidence >= Config.MIN_CONFIDENCE:
                    self._execute_entry_logic(signal, confidence, current_price, current_time, context_buffer, features)

        # 4. Cleanup
        self._close_all_positions(test_data.iloc[-1])
        self.save_results()
        self.print_summary()

    def _execute_entry_logic(self, signal, confidence, current_price, current_time, context_df, features):
        # Calculate ATR
        atr = context_df['close'].rolling(Config.ATR_PERIOD).std().iloc[-1] * 2
        if pd.isna(atr) or atr == 0: atr = current_price * 0.001
        
        # Calculate Enhanced SL/TP/Entry
        levels = self.price_action.calculate_optimal_entry_sl_tp(
            context_df, signal, current_price, atr
        )
        
        if levels['rr_ratio'] >= Config.MIN_RR_RATIO:
            self.trade_counter += 1
            # Simulate Limit Order fill: In backtest loop we assume fill at optimal_entry 
            # if it's close to current_price, otherwise we might miss it. 
            # For strict simulation, we use current_price but keep the calculated SL/TP structure.
            entry_price = current_price 
            
            position = {
                'id': self.trade_counter,
                'entry_time': str(current_time),
                'entry_price': entry_price,
                'signal': int(signal),
                'signal_name': 'BUY' if signal == 1 else 'SELL',
                'confidence': float(confidence),
                'sl': levels['sl'],
                'tp': levels['tp'],
                'rr_ratio': levels['rr_ratio'],
                'features': features
            }
            self.open_positions.append(position)
            Logger.log(f"📊 New {position['signal_name']} | Conf: {confidence:.1%} | R:R: {levels['rr_ratio']:.2f}", "TRADE")

    def _check_exit(self, position, current_bar):
        high, low = current_bar['high'], current_bar['low']
        
        exit_price = None
        reason = None
        
        if position['signal'] == 1: # BUY
            if high >= position['tp']:
                exit_price = position['tp']
                reason = "TP"
            elif low <= position['sl']:
                exit_price = position['sl']
                reason = "SL"
        else: # SELL
            if low <= position['tp']:
                exit_price = position['tp']
                reason = "TP"
            elif high >= position['sl']:
                exit_price = position['sl']
                reason = "SL"
                
        if exit_price:
            self._finalize_trade(position, exit_price, current_bar['time'], reason)

    def _finalize_trade(self, position, exit_price, exit_time, reason):
        if position['signal'] == 1: pnl_raw = exit_price - position['entry_price']
        else: pnl_raw = position['entry_price'] - exit_price
            
        pnl_dollars = pnl_raw * Config.BASE_VOLUME * 100 
        self.balance += pnl_dollars
        
        closed_trade = {
            **position, 'exit_price': exit_price, 'exit_time': str(exit_time),
            'exit_reason': reason, 'pnl': pnl_dollars, 'balance_after': self.balance
        }
        
        self.closed_trades.append(closed_trade)
        self.open_positions.remove(position)
        
        Logger.log(f"🛑 Close {reason} | P&L: ${pnl_dollars:.2f} | Bal: ${self.balance:.0f}", 
                   "SUCCESS" if pnl_dollars > 0 else "WARNING")

    def _close_all_positions(self, final_bar):
        for pos in self.open_positions[:]:
            self._finalize_trade(pos, final_bar['close'], final_bar['time'], "END_TEST")

    def save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'summary': {
                'start': str(self.start_datetime),
                'initial_balance': Config.INITIAL_BALANCE,
                'final_balance': self.balance,
                'net_pnl': self.balance - Config.INITIAL_BALANCE
            },
            'trades': self.closed_trades
        }
        filename = f"backtest_{timestamp}.json"
        filepath = os.path.join(self.test_trades_dir, filename)
        with open(filepath, 'w') as f: json.dump(results, f, indent=2)
        Logger.log(f"📄 Saved results to {filepath}", "SUCCESS")

    def print_summary(self):
        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        Logger.log("\n" + "="*60, "PERFORMANCE")
        Logger.log(f"Final Balance: ${self.balance:.2f}", "PERFORMANCE")
        Logger.log(f"Total Trades: {len(self.closed_trades)}", "PERFORMANCE")
        if self.closed_trades:
            win_rate = len(wins) / len(self.closed_trades) * 100
            Logger.log(f"Win Rate: {win_rate:.2f}%", "SUCCESS" if win_rate > 50 else "WARNING")
        Logger.log("="*60 + "\n", "PERFORMANCE")

# ==========================================
# MAIN EXECUTION
# ==========================================
def get_user_input():
    print("\n--- ADVANCED BACKTEST CONFIGURATION ---")
    while True:
        try:
            d = input("Enter Start Date (YYYY-MM-DD): ").strip()
            t = input("Enter Start Time (HH:MM): ").strip()
            return datetime.strptime(f"{d} {t}", "%Y-%m-%d %H:%M")
        except ValueError:
            print("Invalid format. Try again.")

def main():
    start_dt = get_user_input()
    engine = BacktestEngine(start_dt)
    train_data, test_data = engine.load_historical_data()
    
    if train_data is not None and test_data is not None:
        engine.execute_backtest(train_data, test_data)

if __name__ == "__main__":
    main()