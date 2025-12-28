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
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ==========================================
# LOGGING SYSTEM
# ==========================================
class Logger:
    COLORS = {
        'RESET': '\033[0m',
        'RED': '\033[91m',
        'GREEN': '\033[92m',
        'YELLOW': '\033[93m',
        'BLUE': '\033[94m',
        'MAGENTA': '\033[95m',
        'CYAN': '\033[96m',
        'WHITE': '\033[97m'
    }
    
    @staticmethod
    def log(message, level='INFO'):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        colors = {
            'INFO': Logger.COLORS['CYAN'],
            'SUCCESS': Logger.COLORS['GREEN'],
            'WARNING': Logger.COLORS['YELLOW'],
            'ERROR': Logger.COLORS['RED'],
            'TRADE': Logger.COLORS['MAGENTA'],
            'LEARN': Logger.COLORS['BLUE'],
            'DATA': Logger.COLORS['WHITE'],
            'ENSEMBLE': Logger.COLORS['GREEN'],
            'RISK': Logger.COLORS['YELLOW'],
            'PERFORMANCE': Logger.COLORS['MAGENTA'],
            'BACKTEST': Logger.COLORS['CYAN']
        }
        color = colors.get(level, Logger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level}{Logger.COLORS['RESET']}] {message}", flush=True)

# ==========================================
# CONFIGURATION (Same as main.py)
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
    
    # Model Parameters
    LOOKBACK_BARS = 1000
    TRAINING_MIN_SAMPLES = 100
    
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
    
    # Backtesting specific
    DATABASE_ROOT = "database"
    TEST_TRADES_DIR = "test-trades"
    INITIAL_BALANCE = 10000.0
    POINT_VALUE = 0.01  # For XAUUSD

# ==========================================
# FEATURE ENGINE (Same as main.py)
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
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
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
        
        # ADX
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
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = np.maximum(high - low,
                        np.maximum(abs(high - close.shift(1)),
                                   abs(low - close.shift(1))))
        
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
        
        return pd.DataFrame({
            'adx': adx.fillna(0),
            'plus_di': plus_di.fillna(0),
            'minus_di': minus_di.fillna(0)
        })
    
    def _get_market_session(self, hour):
        if 0 <= hour < 8:
            return 0
        elif 8 <= hour < 16:
            return 1
        else:
            return 2
    
    def create_labels(self, df, forward_bars=3, threshold=0.001):
        df = df.copy()
        df['forward_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
        
        volatility = df['returns'].rolling(20).std().fillna(0.001)
        dynamic_threshold = threshold * (1 + volatility * 10)
        
        df['label'] = -1
        df.loc[df['forward_return'] > dynamic_threshold, 'label'] = 1
        df.loc[df['forward_return'] < -dynamic_threshold, 'label'] = 0
        df['label_confidence'] = abs(df['forward_return']) / dynamic_threshold
        
        return df
    
    def get_feature_columns(self):
        base_features = [
            'returns', 'log_returns', 'hl_ratio', 'co_ratio', 'hlc3',
            'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50',
            'ma_cross_5_20', 'ma_cross_10_50',
            'atr_percent', 'volatility',
            'rsi_normalized', 'macd_hist',
            'bb_width', 'bb_position',
            'momentum_5', 'momentum_10', 'roc_10',
            'distance_to_high', 'distance_to_low',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'gold_seasonal', 'session'
        ]
        
        if Config.USE_MARKET_REGIME:
            base_features.extend([
                'trend_strength', 'regime', 'plus_di', 'minus_di', 
                'adx_slope', 'di_spread'
            ])
        
        return base_features

# ==========================================
# ENSEMBLE MODEL (Same as main.py)
# ==========================================
class BacktestEnsemble:
    def __init__(self):
        self.feature_engine = AdvancedFeatureEngine()
        self.base_models = self._initialize_base_models()
        self.ensemble = VotingClassifier(
            estimators=[(name, model) for name, model in self.base_models],
            voting='soft',
            weights=[1.0, 1.2, 1.0, 1.1]
        )
        self.calibrated_ensemble = CalibratedClassifierCV(
            self.ensemble, method='sigmoid', cv=3
        )
        self.is_trained = False
        
    def _initialize_base_models(self):
        models = [
            ('GB', GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.1,
                subsample=0.8, random_state=42
            )),
            ('RF', RandomForestClassifier(
                n_estimators=100, max_depth=10, min_samples_split=5,
                random_state=42, n_jobs=-1, class_weight='balanced'
            )),
            ('LR', LogisticRegression(
                max_iter=1000, random_state=42, penalty='l2',
                C=1.0, class_weight='balanced'
            )),
            ('NN', MLPClassifier(
                hidden_layer_sizes=(50, 25), max_iter=1000,
                random_state=42, early_stopping=True,
                learning_rate='adaptive'
            ))
        ]
        
        if XGB_AVAILABLE:
            models.append(('XGB', XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1
            )))
        
        return models
    
    def train(self, df):
        Logger.log("Training ensemble model...", "ENSEMBLE")
        
        if df is None or len(df) < Config.TRAINING_MIN_SAMPLES:
            Logger.log(f"Insufficient training data: {len(df) if df is not None else 0} samples", "ERROR")
            return False
        
        df_processed = self.feature_engine.calculate_features(df)
        df_labeled = self.feature_engine.create_labels(df_processed).dropna()
        df_labeled = df_labeled[df_labeled['label'] != -1]
        
        if len(df_labeled) < Config.TRAINING_MIN_SAMPLES:
            Logger.log(f"Insufficient labeled data: {len(df_labeled)} samples", "ERROR")
            return False
        
        feature_cols = self.feature_engine.get_feature_columns()
        X = df_labeled[feature_cols].fillna(0)
        y = df_labeled['label']
        
        X_scaled = self.feature_engine.scaler.fit_transform(X)
        
        try:
            self.calibrated_ensemble.fit(X_scaled, y)
            self.is_trained = True
            Logger.log(f"✅ Model trained on {len(df_labeled)} samples", "SUCCESS")
            return True
        except Exception as e:
            Logger.log(f"Training failed: {str(e)}", "ERROR")
            return False
    
    def predict(self, df):
        if not self.is_trained:
            return None, 0.0, None
        
        df_feat = self.feature_engine.calculate_features(df)
        feature_cols = self.feature_engine.get_feature_columns()
        
        X = df_feat[feature_cols].iloc[-1:].fillna(0).values
        f_dict = {col: float(X[0][i]) for i, col in enumerate(feature_cols)}
        
        X_scaled = self.feature_engine.scaler.transform(X)
        
        final_p = self.calibrated_ensemble.predict(X_scaled)[0]
        proba = self.calibrated_ensemble.predict_proba(X_scaled)[0]
        final_c = np.max(proba)
        
        # ADX Filtering (same as main.py)
        if Config.USE_MARKET_REGIME and 'regime' in f_dict:
            adx = f_dict.get('adx', 0)
            plus_di = f_dict.get('plus_di', 0)
            minus_di = f_dict.get('minus_di', 0)
            adx_slope = f_dict.get('adx_slope', 0)

            if adx > Config.ADX_TREND_THRESHOLD:
                if plus_di > minus_di and final_p == 0:
                    return None, 0.0, None
                if minus_di > plus_di and final_p == 1:
                    return None, 0.0, None

            confidence_threshold = Config.MIN_CONFIDENCE
            if adx_slope > Config.ADX_SLOPE_THRESHOLD:
                confidence_threshold = Config.MIN_CONFIDENCE * 1.1
            
            if f_dict.get('regime') == 2:
                confidence_threshold = Config.MIN_CONFIDENCE * 1.3
            
            if final_c < confidence_threshold:
                return None, 0.0, None
        
        return final_p, final_c, f_dict

# ==========================================
# PRICE ACTION ANALYZER
# ==========================================
class PriceActionAnalyzer:
    @staticmethod
    def calculate_optimal_entry_sl_tp(df, signal, current_price, atr, risk_reward_ratio=Config.MIN_RR_RATIO):
        support, resistance = PriceActionAnalyzer.find_support_resistance(df, current_price)
        
        if signal == 1:  # BUY
            optimal_entry = current_price
            
            sl_candidates = [
                support - (atr * 0.5),
                current_price - (atr * 1.5)
            ]
            sl = max(sl_candidates)
            
            tp_candidates = [
                resistance,
                current_price + (atr * 3),
                current_price + (abs(current_price - sl) * risk_reward_ratio)
            ]
            
            valid_tps = [tp for tp in tp_candidates if tp > optimal_entry]
            if valid_tps:
                final_tp = min([tp for tp in sorted(valid_tps) 
                               if (tp - optimal_entry) / (optimal_entry - sl) >= risk_reward_ratio])
            else:
                final_tp = optimal_entry + (atr * 3)
                
        else:  # SELL
            optimal_entry = current_price
            
            sl_candidates = [
                resistance + (atr * 0.5),
                current_price + (atr * 1.5)
            ]
            sl = min(sl_candidates)
            
            tp_candidates = [
                support,
                current_price - (atr * 3),
                current_price - (abs(sl - current_price) * risk_reward_ratio)
            ]
            
            valid_tps = [tp for tp in tp_candidates if tp < optimal_entry]
            if valid_tps:
                final_tp = max([tp for tp in sorted(valid_tps, reverse=True)
                               if (optimal_entry - tp) / (sl - optimal_entry) >= risk_reward_ratio])
            else:
                final_tp = optimal_entry - (atr * 3)
        
        risk_amount = abs(optimal_entry - sl)
        reward_amount = abs(final_tp - optimal_entry)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return {
            'optimal_entry': optimal_entry,
            'sl': sl,
            'tp': tp,
            'rr_ratio': rr_ratio,
            'support': support,
            'resistance': resistance
        }
    
    @staticmethod
    def find_support_resistance(df, current_price):
        highs = df['high'].tail(Config.LOOKBACK_SWING_POINTS)
        lows = df['low'].tail(Config.LOOKBACK_SWING_POINTS)
        
        resistances = [h for h in highs if h > current_price]
        supports = [l for l in lows if l < current_price]
        
        resistance = min(resistances) if resistances else current_price * 1.01
        support = max(supports) if supports else current_price * 0.99
        
        return support, resistance

# ==========================================
# BACKTEST ENGINE
# ==========================================
class BacktestEngine:
    def __init__(self, start_datetime):
        self.start_datetime = start_datetime
        self.model = BacktestEnsemble()
        self.price_action = PriceActionAnalyzer()
        self.test_trades_dir = Config.TEST_TRADES_DIR
        
        # Create test-trades directory
        os.makedirs(self.test_trades_dir, exist_ok=True)
        
        # Backtest state
        self.balance = Config.INITIAL_BALANCE
        self.open_positions = []
        self.closed_trades = []
        self.trade_counter = 0
        
    def load_historical_data(self, days_before=30, days_after=7):
        """Load data from database"""
        Logger.log(f"Loading historical data from {self.start_datetime}...", "DATA")
        
        # Load training data (before start_datetime)
        train_start = self.start_datetime - timedelta(days=days_before)
        train_end = self.start_datetime - timedelta(hours=1)
        
        # Load testing data (from start_datetime forward)
        test_start = self.start_datetime
        test_end = self.start_datetime + timedelta(days=days_after)
        
        train_data = self._load_data_range(train_start, train_end)
        test_data = self._load_data_range(test_start, test_end)
        
        if train_data is None:
            Logger.log("No training data found", "ERROR")
            return None, None
        
        if test_data is None:
            Logger.log("No testing data found", "ERROR")
            return train_data, None
        
        Logger.log(f"Loaded {len(train_data)} training bars and {len(test_data)} testing bars", "SUCCESS")
        return train_data, test_data
    
    def _load_data_range(self, start_dt, end_dt):
        """Load data from database directory structure"""
        combined_data = []
        
        delta = end_dt - start_dt
        days_count = delta.days + 1
        
        for i in range(days_count):
            current_date = start_dt + timedelta(days=i)
            year = str(current_date.year)
            month = f"{current_date.month:02d}"
            day = f"{current_date.day:02d}"
            
            file_path = os.path.join(Config.DATABASE_ROOT, year, month, day, "data.json")
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        day_data = json.load(f)
                        combined_data.extend(day_data)
                except Exception as e:
                    Logger.log(f"Error loading {file_path}: {e}", "WARNING")
        
        if not combined_data:
            return None
        
        df = pd.DataFrame(combined_data)
        df['time'] = pd.to_datetime(df['time'])
        df = df.sort_values('time').drop_duplicates(subset=['time'])
        
        # Filter by exact datetime range
        df = df[(df['time'] >= start_dt) & (df['time'] <= end_dt)]
        
        return df.reset_index(drop=True)
    
    def calculate_position_outcome(self, entry_price, sl, tp, signal, exit_price):
        """Calculate P&L for a position"""
        if signal == 1:  # BUY
            if exit_price >= tp:
                pnl = tp - entry_price
                outcome = "TP"
            elif exit_price <= sl:
                pnl = sl - entry_price
                outcome = "SL"
            else:
                pnl = exit_price - entry_price
                outcome = "PARTIAL"
        else:  # SELL
            if exit_price <= tp:
                pnl = entry_price - tp
                outcome = "TP"
            elif exit_price >= sl:
                pnl = entry_price - sl
                outcome = "SL"
            else:
                pnl = entry_price - exit_price
                outcome = "PARTIAL"
        
        pnl_dollars = pnl * Config.BASE_VOLUME * 100  # Simplified P&L calculation
        return pnl_dollars, outcome
    
    def check_position_exit(self, position, current_bar):
        """Check if position should be closed"""
        high = current_bar['high']
        low = current_bar['low']
        close = current_bar['close']
        
        if position['signal'] == 1:  # BUY
            if high >= position['tp']:
                return True, position['tp'], "TP"
            elif low <= position['sl']:
                return True, position['sl'], "SL"
        else:  # SELL
            if low <= position['tp']:
                return True, position['tp'], "TP"
            elif high >= position['sl']:
                return True, position['sl'], "SL"
        
        return False, close, "OPEN"
    
    def execute_backtest(self, train_data, test_data):
        """Run the backtest simulation"""
        Logger.log("=" * 60, "BACKTEST")
        Logger.log("🔬 STARTING BACKTEST SIMULATION", "BACKTEST")
        Logger.log("=" * 60, "BACKTEST")
        
        # Train model
        if not self.model.train(train_data):
            Logger.log("Model training failed", "ERROR")
            return
        
        # Run through test data bar by bar
        for idx in range(len(test_data)):
            current_bar = test_data.iloc[idx]
            current_time = current_bar['time']
            current_price = current_bar['close']
            
            # Check existing positions
            for position in self.open_positions[:]:
                should_exit, exit_price, exit_reason = self.check_position_exit(position, current_bar)
                
                if should_exit:
                    pnl, outcome = self.calculate_position_outcome(
                        position['entry_price'],
                        position['sl'],
                        position['tp'],
                        position['signal'],
                        exit_price
                    )
                    
                    self.balance += pnl
                    
                    closed_trade = {
                        **position,
                        'exit_price': exit_price,
                        'exit_time': str(current_time),
                        'exit_reason': exit_reason,
                        'pnl': pnl,
                        'balance_after': self.balance
                    }
                    
                    self.closed_trades.append(closed_trade)
                    self.open_positions.remove(position)
                    
                    result = "✅ WIN" if pnl > 0 else "❌ LOSS"
                    Logger.log(f"{result} | {exit_reason} | P&L: ${pnl:.2f} | Balance: ${self.balance:.2f}", 
                             "SUCCESS" if pnl > 0 else "WARNING")
            
            # Check for new signals (only if we have enough historical data)
            if idx >= 100 and len(self.open_positions) < Config.MAX_POSITIONS:
                # Get data up to current point (no future peeking!)
                historical_context = test_data.iloc[max(0, idx-Config.LOOKBACK_BARS):idx+1]
                
                signal, confidence, features = self.model.predict(historical_context)
                
                if signal is not None and confidence >= Config.MIN_CONFIDENCE:
                    # Calculate trade parameters
                    atr = historical_context['close'].rolling(Config.ATR_PERIOD).std().iloc[-1] * 2
                    if pd.isna(atr) or atr == 0:
                        atr = current_price * 0.001
                    
                    levels = self.price_action.calculate_optimal_entry_sl_tp(
                        historical_context, signal, current_price, atr
                    )
                    
                    if levels['rr_ratio'] >= Config.MIN_RR_RATIO:
                        self.trade_counter += 1
                        
                        position = {
                            'id': self.trade_counter,
                            'entry_time': str(current_time),
                            'entry_price': current_price,
                            'signal': int(signal),
                            'signal_name': 'BUY' if signal == 1 else 'SELL',
                            'confidence': float(confidence),
                            'sl': levels['sl'],
                            'tp': levels['tp'],
                            'rr_ratio': levels['rr_ratio'],
                            'atr': atr,
                            'support': levels['support'],
                            'resistance': levels['resistance'],
                            'features': features
                        }
                        
                        self.open_positions.append(position)
                        
                        Logger.log("=" * 60, "TRADE")
                        Logger.log(f"📊 NEW {position['signal_name']} SIGNAL", "TRADE")
                        Logger.log(f"  Time: {current_time}", "TRADE")
                        Logger.log(f"  Entry: {current_price:.2f}", "TRADE")
                        Logger.log(f"  SL: {levels['sl']:.2f} | TP: {levels['tp']:.2f}", "TRADE")
                        Logger.log(f"  R:R: {levels['rr_ratio']:.2f} | Confidence: {confidence:.1%}", "TRADE")
                        Logger.log(f"  Support: {levels['support']:.2f} | Resistance: {levels['resistance']:.2f}", "TRADE")
                        Logger.log("=" * 60, "TRADE")
        
        # Close any remaining positions at final price
        if self.open_positions:
            final_price = test_data.iloc[-1]['close']
            final_time = test_data.iloc[-1]['time']
            
            Logger.log(f"\n⏰ Test period ended. Closing {len(self.open_positions)} open positions...", "WARNING")
            
            for position in self.open_positions:
                pnl, outcome = self.calculate_position_outcome(
                    position['entry_price'],
                    position['sl'],
                    position['tp'],
                    position['signal'],
                    final_price
                )
                
                self.balance += pnl
                
                closed_trade = {
                    **position,
                    'exit_price': final_price,
                    'exit_time': str(final_time),
                    'exit_reason': 'END_OF_TEST',
                    'pnl': pnl,
                    'balance_after': self.balance
                }
                
                self.closed_trades.append(closed_trade)
        
        self.save_results()
        self.print_summary()
    
    def save_results(self):
        """Save backtest results to JSON"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        test_date = self.start_datetime.strftime("%Y%m%d")
        
        results = {
            'test_info': {
                'start_datetime': str(self.start_datetime),
                'test_date': test_date,
                'timestamp': timestamp,
                'initial_balance': Config.INITIAL_BALANCE,
                'final_balance': self.balance,
                'total_return': self.balance - Config.INITIAL_BALANCE,
                'return_pct': ((self.balance - Config.INITIAL_BALANCE) / Config.INITIAL_BALANCE) * 100
            },
            'trades': self.closed_trades,
            'config': {
                'symbol': Config.SYMBOL,
                'min_confidence': Config.MIN_CONFIDENCE,
                'min_rr_ratio': Config.MIN_RR_RATIO,
                'max_positions': Config.MAX_POSITIONS,
                'use_market_regime': Config.USE_MARKET_REGIME
            }
        }
        
        filename = f"backtest_{test_date}_{timestamp}.json"
        filepath = os.path.join(self.test_trades_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        Logger.log(f"📄 Results saved to: {filepath}", "SUCCESS")
    
    def print_summary(self):
        """Print comprehensive backtest summary"""
        Logger.log("\n" + "=" * 60, "PERFORMANCE")
        Logger.log("📊 BACKTEST RESULTS SUMMARY", "PERFORMANCE")
        Logger.log("=" * 60, "PERFORMANCE")
        
        total_trades = len(self.closed_trades)
        
        if total_trades == 0:
            Logger.log("❌ No trades executed during test period", "WARNING")
            return
        
        wins = [t for t in self.closed_trades if t['pnl'] > 0]
        losses = [t for t in self.closed_trades if t['pnl'] <= 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades) * 100
        
        total_profit = sum(t['pnl'] for t in wins) if wins else 0
        total_loss = sum(t['pnl'] for t in losses) if losses else 0
        net_profit = self.balance - Config.INITIAL_BALANCE
        
        avg_win = total_profit / win_count if win_count > 0 else 0
        avg_loss = total_loss / loss_count if loss_count > 0 else 0
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else float('inf')
        
        # Trade breakdown by exit reason
        tp_count = len([t for t in self.closed_trades if t['exit_reason'] == 'TP'])
        sl_count = len([t for t in self.closed_trades if t['exit_reason'] == 'SL'])
        
        Logger.log(f"\n📈 Performance Metrics:", "PERFORMANCE")
        Logger.log(f"  Initial Balance: ${Config.INITIAL_BALANCE:.2f}", "PERFORMANCE")
        Logger.log(f"  Final Balance: ${self.balance:.2f}", "PERFORMANCE")
        Logger.log(f"  Net P&L: ${net_profit:.2f} ({(net_profit/Config.INITIAL_BALANCE)*100:.2f}%)", 
                  "SUCCESS" if net_profit > 0 else "ERROR")
        
        Logger.log(f"\n📊 Trade Statistics:", "PERFORMANCE")
        Logger.log(f"  Total Trades: {total_trades}", "PERFORMANCE")
        Logger.log(f"  Wins: {win_count} ({win_rate:.1f}%)", "SUCCESS")
        Logger.log(f"  Losses: {loss_count} ({100-win_rate:.1f}%)", "WARNING")
        Logger.log(f"  TP Hits: {tp_count}", "SUCCESS")
        Logger.log(f"  SL Hits: {sl_count}", "WARNING")
        
        Logger.log(f"\n💰 P&L Analysis:", "PERFORMANCE")
        Logger.log(f"  Average Win: ${avg_win:.2f}", "SUCCESS")
        Logger.log(f"  Average Loss: ${avg_loss:.2f}", "WARNING")
        Logger.log(f"  Profit Factor: {profit_factor:.2f}", "PERFORMANCE")
        Logger.log(f"  Expectancy: ${(total_profit + total_loss) / total_trades:.2f}", "PERFORMANCE")
        
        # Best and worst trades
        if self.closed_trades:
            best_trade = max(self.closed_trades, key=lambda x: x['pnl'])
            worst_trade = min(self.closed_trades, key=lambda x: x['pnl'])
            
            Logger.log(f"\n🏆 Best Trade:", "SUCCESS")
            Logger.log(f"  {best_trade['signal_name']} | P&L: ${best_trade['pnl']:.2f} | "
                      f"Exit: {best_trade['exit_reason']}", "SUCCESS")
            
            Logger.log(f"\n💔 Worst Trade:", "ERROR")
            Logger.log(f"  {worst_trade['signal_name']} | P&L: ${worst_trade['pnl']:.2f} | "
                      f"Exit: {worst_trade['exit_reason']}", "ERROR")
        
        Logger.log("\n" + "=" * 60, "PERFORMANCE")
        
        # Signal accuracy breakdown
        Logger.log(f"\n🎯 Signal Accuracy:", "PERFORMANCE")
        buy_trades = [t for t in self.closed_trades if t['signal'] == 1]
        sell_trades = [t for t in self.closed_trades if t['signal'] == 0]
        
        if buy_trades:
            buy_wins = len([t for t in buy_trades if t['pnl'] > 0])
            buy_wr = (buy_wins / len(buy_trades)) * 100
            Logger.log(f"  BUY Signals: {len(buy_trades)} trades | {buy_wr:.1f}% win rate", "PERFORMANCE")
        
        if sell_trades:
            sell_wins = len([t for t in sell_trades if t['pnl'] > 0])
            sell_wr = (sell_wins / len(sell_trades)) * 100
            Logger.log(f"  SELL Signals: {len(sell_trades)} trades | {sell_wr:.1f}% win rate", "PERFORMANCE")
        
        Logger.log("\n" + "=" * 60 + "\n", "PERFORMANCE")

# ==========================================
# USER INTERACTION
# ==========================================
def get_backtest_datetime():
    """Interactive function to get backtest start datetime from user"""
    print("\n" + "=" * 60)
    print("  🔬 BACKTESTING SYSTEM - Historical Performance Testing")
    print("=" * 60 + "\n")
    
    print("This system will:")
    print("  1. Load data BEFORE your chosen datetime for model training")
    print("  2. Make predictions starting FROM your datetime (no future peeking)")
    print("  3. Simulate trades and show results")
    print()
    
    while True:
        try:
            date_str = input("Enter test START DATE (YYYY-MM-DD): ").strip()
            time_str = input("Enter test START TIME (HH:MM): ").strip()
            
            datetime_str = f"{date_str} {time_str}"
            test_datetime = datetime.strptime(datetime_str, "%Y-%m-%d %H:%M")
            
            print(f"\n✅ Test will start from: {test_datetime}")
            confirm = input("Proceed? (y/n): ").strip().lower()
            
            if confirm == 'y':
                return test_datetime
            
        except ValueError:
            Logger.log("Invalid format. Please use YYYY-MM-DD for date and HH:MM for time", "ERROR")

# ==========================================
# MAIN EXECUTION
# ==========================================
def main():
    """Main backtest execution"""
    start_datetime = get_backtest_datetime()
    
    Logger.log(f"\n🚀 Initializing backtest from {start_datetime}...", "INFO")
    
    engine = BacktestEngine(start_datetime)
    
    # Load historical data
    train_data, test_data = engine.load_historical_data(days_before=30, days_after=7)
    
    if train_data is None or test_data is None:
        Logger.log("❌ Cannot proceed without sufficient data", "ERROR")
        return
    
    # Run backtest
    engine.execute_backtest(train_data, test_data)
    
    Logger.log("\n✅ Backtest complete! Check test-trades/ folder for detailed results.", "SUCCESS")

if __name__ == "__main__":
    main()