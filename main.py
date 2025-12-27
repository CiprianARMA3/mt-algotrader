import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime, timedelta
import time
import sys
import json
import os
import warnings
warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
class Config:
    # MT5 Credentials
    MT5_LOGIN = 5044108820
    MT5_PASSWORD = "@rC1KbQb"
    MT5_SERVER = "MetaQuotes-Demo"
    
    # Trading Parameters
    SYMBOL = "XAUUSD"
    TIMEFRAME = mt5.TIMEFRAME_M15
    BASE_VOLUME = 0.01
    MAGIC_NUMBER = 998877
    
    # Risk Management
    RISK_PERCENT = 0.02
    MIN_CONFIDENCE = 0.6  # Increased threshold for ensemble
    MIN_ENSEMBLE_AGREEMENT = 0.67  # At least 2/3 models must agree
    MAX_POSITIONS = 2
    
    # Model Parameters
    LOOKBACK_BARS = 1000
    RETRAIN_HOURS = 12
    
    # Dataset Configuration
    USE_HISTORICAL_DATASET = True
    HISTORICAL_DATA_LIMIT = 5000
    HISTORICAL_WEIGHT = 0.3
    RECENT_WEIGHT = 0.7
    
    # Ensemble Configuration
    USE_ENSEMBLE = True  # Multiple models for cross-validation
    MIN_DATA_QUALITY_SCORE = 0.7  # Filter low-quality data
    
    # Learning Parameters
    TRADE_HISTORY_FILE = "trade_history.json"
    MEMORY_SIZE = 500
    LEARNING_WEIGHT = 0.4
    
    # Technical Parameters
    ATR_PERIOD = 14
    RSI_PERIOD = 14
    
    # Price Action Parameters
    USE_SMART_ENTRY = True  # Wait for optimal entry price
    USE_DYNAMIC_SL_TP = True  # Calculate SL/TP based on S/R levels
    MIN_RR_RATIO = 1.5  # Minimum Risk:Reward ratio
    LOOKBACK_SWING_POINTS = 50  # Bars to analyze for swing highs/lows
    
# ==========================================
# LOGGING
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
            'ENSEMBLE': Logger.COLORS['GREEN']
        }
        color = colors.get(level, Logger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level}{Logger.COLORS['RESET']}] {message}", flush=True)

# ==========================================
# DATA QUALITY CHECKER
# ==========================================
class DataQualityChecker:
    """Validates and scores data quality before training"""
    
    @staticmethod
    def check_data_quality(df):
        """Return quality score 0-1 based on data characteristics"""
        scores = []
        
        # 1. Check for missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        scores.append(1 - missing_ratio)
        
        # 2. Check for data variety (not all same values)
        variance_score = 1.0
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].std() == 0:
                variance_score *= 0.5
        scores.append(variance_score)
        
        # 3. Check for outliers (extreme values)
        outlier_score = 1.0
        for col in ['close', 'open', 'high', 'low']:
            if col in df.columns:
                q1, q3 = df[col].quantile([0.25, 0.75])
                iqr = q3 - q1
                outliers = ((df[col] < (q1 - 3*iqr)) | (df[col] > (q3 + 3*iqr))).sum()
                outlier_ratio = outliers / len(df)
                outlier_score *= max(0.5, 1 - outlier_ratio)
        scores.append(outlier_score)
        
        # 4. Check chronological order
        if 'time' in df.columns:
            is_sorted = df['time'].is_monotonic_increasing
            scores.append(1.0 if is_sorted else 0.7)
        
        overall_score = np.mean(scores)
        return overall_score, scores
    
    @staticmethod
    def validate_features(df, feature_cols):
        """Check if features are valid for prediction"""
        invalid_features = []
        
        for col in feature_cols:
            if col not in df.columns:
                continue
            
            # Check for inf values
            if np.isinf(df[col]).any():
                invalid_features.append(f"{col} (has inf)")
                
            # Check for too many NaN
            nan_ratio = df[col].isnull().sum() / len(df)
            if nan_ratio > 0.3:
                invalid_features.append(f"{col} (>{nan_ratio:.0%} NaN)")
        
        return invalid_features

# ==========================================
# TRADE MEMORY SYSTEM
# ==========================================
class TradeMemory:
    """Stores and manages historical trade data for learning"""
    
    def __init__(self):
        self.history_file = Config.TRADE_HISTORY_FILE
        self.trades = self.load_history()
        
    def load_history(self):
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    data = json.load(f)
                Logger.log(f"Loaded {len(data)} trades from history", "LEARN")
                return data
            except Exception as e:
                Logger.log(f"Error loading history: {e}", "WARNING")
                return []
        return []
    
    def save_history(self):
        try:
            if len(self.trades) > Config.MEMORY_SIZE:
                self.trades = self.trades[-Config.MEMORY_SIZE:]
            with open(self.history_file, 'w') as f:
                json.dump(self.trades, f, indent=2)
        except Exception as e:
            Logger.log(f"Error saving history: {e}", "ERROR")
    
    def add_trade(self, trade_data):
        self.trades.append(trade_data)
        self.save_history()
        Logger.log(f"Trade recorded | Total in memory: {len(self.trades)}", "LEARN")
    
    def get_trade_by_ticket(self, ticket):
        for trade in self.trades:
            if trade['ticket'] == ticket:
                return trade
        return None
    
    def update_trade_outcome(self, ticket, outcome_data):
        for trade in self.trades:
            if trade['ticket'] == ticket:
                trade.update(outcome_data)
                self.save_history()
                profit = outcome_data.get('profit', 0)
                result = "WIN" if profit > 0 else "LOSS"
                Logger.log(
                    f"Trade #{ticket} closed | {result} | P/L: ${profit:.2f}",
                    "SUCCESS" if profit > 0 else "WARNING"
                )
                return True
        return False
    
    def get_completed_trades(self):
        return [t for t in self.trades if 'profit' in t]
    
    def get_performance_stats(self):
        completed = self.get_completed_trades()
        if not completed:
            return None
        
        wins = [t for t in completed if t['profit'] > 0]
        losses = [t for t in completed if t['profit'] <= 0]
        
        total_profit = sum(t['profit'] for t in completed)
        win_rate = len(wins) / len(completed) if completed else 0
        avg_win = np.mean([t['profit'] for t in wins]) if wins else 0
        avg_loss = np.mean([t['profit'] for t in losses]) if losses else 0
        
        total_win_profit = sum(t['profit'] for t in wins) if wins else 0
        total_loss_profit = abs(sum(t['profit'] for t in losses)) if losses else 1
        profit_factor = total_win_profit / total_loss_profit if total_loss_profit > 0 else 0
        
        return {
            'total_trades': len(completed),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }

# ==========================================
# DATA LOADER
# ==========================================
class DataLoader:
    """Handles loading and preprocessing data from multiple sources"""
    
    @staticmethod
    def load_huggingface_dataset():
        try:
            Logger.log("Loading historical XAU/USD dataset from Hugging Face...", "DATA")
            ds = load_dataset("ZombitX64/xauusd-gold-price-historical-data-2004-2025")
            df = ds['train'].to_pandas()
            
            Logger.log(f"Loaded {len(df)} historical records", "SUCCESS")
            
            # Standardize columns
            column_mapping = {
                'Date': 'time', 'Open': 'open', 'High': 'high', 
                'Low': 'low', 'Close': 'close', 'Volume': 'tick_volume'
            }
            
            for old_col, new_col in column_mapping.items():
                matching_cols = [col for col in df.columns if col.lower() == old_col.lower()]
                if matching_cols:
                    df.rename(columns={matching_cols[0]: new_col}, inplace=True)
            
            # Convert timestamps
            if 'time' in df.columns and df['time'].dtype == 'object':
                df['time'] = pd.to_datetime(df['time']).astype(int) // 10**9
            
            # Filter to recent years
            if 'time' in df.columns:
                cutoff_date = datetime.now() - timedelta(days=3*365)
                cutoff_timestamp = int(cutoff_date.timestamp())
                df = df[df['time'] >= cutoff_timestamp]
                Logger.log(f"Filtered to last 3 years: {len(df)} records", "DATA")
            
            # Quality check
            quality_score, _ = DataQualityChecker.check_data_quality(df)
            Logger.log(f"Dataset quality score: {quality_score:.2%}", "DATA")
            
            if quality_score < Config.MIN_DATA_QUALITY_SCORE:
                Logger.log(f"Dataset quality below threshold ({Config.MIN_DATA_QUALITY_SCORE:.0%})", "WARNING")
            
            df = df.sort_values('time').reset_index(drop=True)
            return df
            
        except Exception as e:
            Logger.log(f"Failed to load dataset: {str(e)}", "ERROR")
            return None

# ==========================================
# PRICE ACTION ANALYZER
# ==========================================
class PriceActionAnalyzer:
    """Analyzes price action to find optimal entry, SL, and TP levels"""
    
    @staticmethod
    def find_swing_points(df, lookback=50):
        """Identify swing highs and lows"""
        df = df.tail(lookback).copy()
        
        swing_highs = []
        swing_lows = []
        
        for i in range(2, len(df) - 2):
            # Swing High: higher than 2 bars on each side
            if (df['high'].iloc[i] > df['high'].iloc[i-1] and 
                df['high'].iloc[i] > df['high'].iloc[i-2] and
                df['high'].iloc[i] > df['high'].iloc[i+1] and 
                df['high'].iloc[i] > df['high'].iloc[i+2]):
                swing_highs.append({
                    'price': df['high'].iloc[i],
                    'index': i,
                    'time': df['time'].iloc[i] if 'time' in df.columns else i
                })
            
            # Swing Low: lower than 2 bars on each side
            if (df['low'].iloc[i] < df['low'].iloc[i-1] and 
                df['low'].iloc[i] < df['low'].iloc[i-2] and
                df['low'].iloc[i] < df['low'].iloc[i+1] and 
                df['low'].iloc[i] < df['low'].iloc[i+2]):
                swing_lows.append({
                    'price': df['low'].iloc[i],
                    'index': i,
                    'time': df['time'].iloc[i] if 'time' in df.columns else i
                })
        
        return swing_highs, swing_lows
    
    @staticmethod
    def find_support_resistance(df, current_price, lookback=50):
        """Find nearest support and resistance levels"""
        swing_highs, swing_lows = PriceActionAnalyzer.find_swing_points(df, lookback)
        
        # Find resistance (swing highs above current price)
        resistances = [s['price'] for s in swing_highs if s['price'] > current_price]
        resistance = min(resistances) if resistances else current_price * 1.01
        
        # Find support (swing lows below current price)
        supports = [s['price'] for s in swing_lows if s['price'] < current_price]
        support = max(supports) if supports else current_price * 0.99
        
        return support, resistance
    
    @staticmethod
    def calculate_pivot_points(df):
        """Calculate classic pivot points"""
        last_bar = df.iloc[-1]
        high = last_bar['high']
        low = last_bar['low']
        close = last_bar['close']
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        r2 = pivot + (high - low)
        r3 = high + 2 * (pivot - low)
        
        s1 = 2 * pivot - high
        s2 = pivot - (high - low)
        s3 = low - 2 * (high - pivot)
        
        return {
            'pivot': pivot,
            'r1': r1, 'r2': r2, 'r3': r3,
            's1': s1, 's2': s2, 's3': s3
        }
    
    @staticmethod
    def find_nearest_level(current_price, levels):
        """Find the nearest price level"""
        if not levels:
            return None
        return min(levels, key=lambda x: abs(x - current_price))
    
    @staticmethod
    def calculate_optimal_entry_sl_tp(df, signal, current_price, atr):
        """
        Calculate optimal entry, stop loss, and take profit
        based on price action and support/resistance
        """
        # Get swing points and S/R levels
        support, resistance = PriceActionAnalyzer.find_support_resistance(
            df, current_price, Config.LOOKBACK_SWING_POINTS
        )
        
        # Get pivot points
        pivots = PriceActionAnalyzer.calculate_pivot_points(df)
        
        if signal == 1:  # BUY Signal
            # Entry: Current price or slight pullback to support
            entry_price = current_price
            
            # Stop Loss: Below nearest support or recent swing low
            # Use the closer of: support level or ATR-based
            atr_sl = current_price - (atr * 1.5)
            support_sl = support - (atr * 0.3)  # Slightly below support for buffer
            sl = max(atr_sl, support_sl)  # Use the tighter stop
            
            # Take Profit: At resistance or pivot R1/R2
            # Check multiple TP targets
            tp_targets = [
                resistance,
                pivots['r1'],
                pivots['r2'],
                current_price + (atr * 3)  # Fallback: 3x ATR
            ]
            
            # Filter realistic targets (above entry)
            valid_targets = [t for t in tp_targets if t > entry_price]
            
            if valid_targets:
                # Use the nearest resistance that gives good R:R
                for tp in sorted(valid_targets):
                    risk = entry_price - sl
                    reward = tp - entry_price
                    rr_ratio = reward / risk if risk > 0 else 0
                    
                    if rr_ratio >= Config.MIN_RR_RATIO:
                        break
            else:
                tp = entry_price + (atr * 3)
            
            # Smart entry: wait for pullback if price too far from support
            distance_to_support = (current_price - support) / atr
            if distance_to_support > 1.5 and Config.USE_SMART_ENTRY:
                optimal_entry = support + (atr * 0.5)  # Enter near support
            else:
                optimal_entry = entry_price
                
        else:  # SELL Signal
            entry_price = current_price
            
            # Stop Loss: Above nearest resistance
            atr_sl = current_price + (atr * 1.5)
            resistance_sl = resistance + (atr * 0.3)
            sl = min(atr_sl, resistance_sl)
            
            # Take Profit: At support or pivot S1/S2
            tp_targets = [
                support,
                pivots['s1'],
                pivots['s2'],
                current_price - (atr * 3)
            ]
            
            valid_targets = [t for t in tp_targets if t < entry_price]
            
            if valid_targets:
                for tp in sorted(valid_targets, reverse=True):
                    risk = sl - entry_price
                    reward = entry_price - tp
                    rr_ratio = reward / risk if risk > 0 else 0
                    
                    if rr_ratio >= Config.MIN_RR_RATIO:
                        break
            else:
                tp = entry_price - (atr * 3)
            
            # Smart entry: wait for retest of resistance
            distance_to_resistance = (resistance - current_price) / atr
            if distance_to_resistance > 1.5 and Config.USE_SMART_ENTRY:
                optimal_entry = resistance - (atr * 0.5)
            else:
                optimal_entry = entry_price
        
        # Calculate final risk:reward ratio
        risk = abs(optimal_entry - sl)
        reward = abs(tp - optimal_entry)
        rr_ratio = reward / risk if risk > 0 else 0
        
        return {
            'optimal_entry': optimal_entry,
            'sl': sl,
            'tp': tp,
            'current_price': current_price,
            'support': support,
            'resistance': resistance,
            'rr_ratio': rr_ratio,
            'distance_to_optimal': abs(current_price - optimal_entry)
        }

# ==========================================
# FEATURE ENGINEERING
# ==========================================
class FeatureEngine:
    def __init__(self):
        self.scaler = StandardScaler()
        
    def calculate_features(self, df):
        df = df.copy()
        
        # Price features
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        
        # Moving Averages
        for period in [5, 10, 20, 50]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        
        # ATR
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(Config.ATR_PERIOD).mean()
        df['normalized_atr'] = df['atr'] / df['close']
        
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
        df['macd_normalized'] = df['macd_hist'] / df['atr']
        
        # Bollinger Bands
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume
        if 'tick_volume' in df.columns:
            df['volume_sma'] = df['tick_volume'].rolling(20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
        
        # Momentum
        for period in [5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
        
        # Time features
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
        else:
            df['hour'] = 12
            df['day_of_week'] = 2
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        return df
    
    def create_labels(self, df, forward_bars=3, threshold=0.0008):
        df = df.copy()
        df['forward_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
        df['label'] = 0
        df.loc[df['forward_return'] > threshold, 'label'] = 1
        df.loc[df['forward_return'] < -threshold, 'label'] = 0
        df.loc[abs(df['forward_return']) <= threshold, 'label'] = -1
        return df
    
    def get_feature_columns(self):
        return [
            'returns', 'log_returns', 'hl_ratio', 'co_ratio',
            'price_to_sma_5', 'price_to_sma_10', 'price_to_sma_20', 'price_to_sma_50',
            'normalized_atr', 'rsi_normalized', 'macd_normalized', 'bb_position',
            'momentum_5', 'momentum_10', 'momentum_20',
            'hour_sin', 'hour_cos', 'day_of_week'
        ]

# ==========================================
# ENSEMBLE MODEL (REWRITTEN & INTEGRATED)
# ==========================================
class EnsembleModel:
    """Ensemble learning system using GB, RF, and LR with internal agreement checks."""
    
    def __init__(self, trade_memory):
        self.feature_engine = FeatureEngine()
        self.trade_memory = trade_memory
        self.data_quality_checker = DataQualityChecker()
        
        # Base model definitions
        self.base_models = [
            ('GB', GradientBoostingClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, subsample=0.8, random_state=42)),
            ('RF', RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=5, random_state=42, n_jobs=-1)),
            ('LR', LogisticRegression(max_iter=1000, random_state=42))
        ]
        
        # Initialize Voting Classifier
        self.ensemble = VotingClassifier(
            estimators=self.base_models,
            voting='soft'
        )
        
        self.is_trained = False
        self.last_train_time = None
        self.model_scores = {}
        self.historical_data_cache = None

    def load_all_training_data(self, recent_mt5_data):
        """Merges historical data, recent market data, and memory of past trades."""
        all_dfs = []
        
        # 1. Historical Data Source
        if Config.USE_HISTORICAL_DATASET:
            if self.historical_data_cache is None:
                self.historical_data_cache = DataLoader.load_huggingface_dataset()
            
            if self.historical_data_cache is not None:
                df_h = self.feature_engine.calculate_features(self.historical_data_cache)
                df_h = self.feature_engine.create_labels(df_h).dropna()
                df_h = df_h[df_h['label'] != -1].tail(Config.HISTORICAL_DATA_LIMIT)
                df_h['weight'] = Config.HISTORICAL_WEIGHT
                all_dfs.append(df_h)

        # 2. Recent MT5 Live Data
        df_r = self.feature_engine.calculate_features(recent_mt5_data)
        df_r = self.feature_engine.create_labels(df_r).dropna()
        df_r = df_r[df_r['label'] != -1]
        df_r['weight'] = Config.RECENT_WEIGHT
        all_dfs.append(df_r)

        # 3. Trade Experience (Reinforcement Learning light)
        trades = self.trade_memory.get_completed_trades()
        if trades:
            t_samples = []
            for t in trades:
                if 'features' in t:
                    label = t['signal'] if t['profit'] > 0 else 1 - t['signal']
                    sample = {**t['features'], 'label': label, 'weight': Config.LEARNING_WEIGHT}
                    t_samples.append(sample)
            if t_samples:
                all_dfs.append(pd.DataFrame(t_samples))

        if not all_dfs: return None
        
        combined = pd.concat(all_dfs, ignore_index=True)
        q_score, _ = self.data_quality_checker.check_data_quality(combined)
        Logger.log(f"Data Quality: {q_score:.2%} | Samples: {len(combined)}", "DATA")
        
        return combined

    def train(self, recent_mt5_data):
        """Fits the ensemble and validates using TimeSeriesSplit."""
        Logger.log("=" * 40, "INFO")
        Logger.log("🚀 STARTING ENSEMBLE TRAINING", "ENSEMBLE")
        
        df = self.load_all_training_data(recent_mt5_data)
        if df is None or len(df) < 100:
            Logger.log("Training aborted: Insufficient data.", "ERROR")
            return False

        f_cols = self.feature_engine.get_feature_columns()
        X = df[f_cols].fillna(0)
        y = df['label']
        weights = df['weight'].values
        
        X_scaled = self.feature_engine.scaler.fit_transform(X)
        tscv = TimeSeriesSplit(n_splits=3)

        # Validate individual components
        for name, model in self.base_models:
            acc = np.mean(cross_val_score(model, X_scaled, y, cv=tscv))
            self.model_scores[name] = acc
            Logger.log(f"Component {name} Accuracy: {acc:.2%}", "ENSEMBLE")

        # Fit the actual ensemble
        self.ensemble.fit(X_scaled, y, sample_weight=weights)
        
        self.is_trained = True
        self.last_train_time = datetime.now()
        Logger.log("✅ Ensemble Training Complete", "SUCCESS")
        return True

    def predict(self, df):
        """
        Predicts using the fitted ensemble. 
        Accesses .estimators_ to prevent NotFittedError on sub-models.
        """
        if not self.is_trained: return None, 0.0, None, {}

        df_feat = self.feature_engine.calculate_features(df)
        f_cols = self.feature_engine.get_feature_columns()
        X = df_feat[f_cols].iloc[-1:].fillna(0).values
        
        f_dict = {col: float(X[0][i]) for i, col in enumerate(f_cols)}
        X_scaled = self.feature_engine.scaler.transform(X)

        # Get results from fitted sub-models
        sub_preds = {}
        # Scikit-learn stores fitted clones in estimators_ after fit()
        for (name, _), fitted_model in zip(self.base_models, self.ensemble.estimators_):
            p = fitted_model.predict(X_scaled)[0]
            c = np.max(fitted_model.predict_proba(X_scaled)[0])
            sub_preds[name] = {'prediction': p, 'confidence': c}

        # Ensemble Final Vote
        final_p = self.ensemble.predict(X_scaled)[0]
        final_c = np.max(self.ensemble.predict_proba(X_scaled)[0])

        # Agreement Logic
        raw_votes = [m['prediction'] for m in sub_preds.values()]
        agreement = raw_votes.count(final_p) / len(raw_votes)

        if agreement < Config.MIN_ENSEMBLE_AGREEMENT:
            Logger.log(f"Signal filtered: Low Agreement ({agreement:.0%})", "WARNING")
            return None, 0.0, None, sub_preds

        return final_p, final_c, f_dict, sub_preds

    def should_retrain(self):
        if not self.last_train_time: return True
        return (datetime.now() - self.last_train_time).total_seconds() / 3600 >= Config.RETRAIN_HOURS

# ==========================================
# TRADING ENGINE
# ==========================================
class TradingEngine:
    def __init__(self):
        self.trade_memory = TradeMemory()
        self.model = EnsembleModel(self.trade_memory)
        self.price_action = PriceActionAnalyzer()
        self.connected = False
        self.active_positions = {}
        self.pending_signals = {}  # Store signals waiting for optimal entry
        
    def connect_mt5(self):
        Logger.log("Initializing MT5...", "INFO")
        
        if not mt5.initialize():
            Logger.log(f"MT5 init failed: {mt5.last_error()}", "ERROR")
            return False
        
        authorized = mt5.login(
            login=Config.MT5_LOGIN,
            password=Config.MT5_PASSWORD,
            server=Config.MT5_SERVER
        )
        
        if not authorized:
            Logger.log(f"Login failed: {mt5.last_error()}", "ERROR")
            mt5.shutdown()
            return False
        
        account = mt5.account_info()
        Logger.log(f"✓ Connected | Account: {account.login} | Balance: ${account.balance:.2f}", "SUCCESS")
        
        if not mt5.terminal_info().trade_allowed:
            Logger.log("⚠ Algo trading disabled!", "ERROR")
            return False
        
        symbol_info = mt5.symbol_info(Config.SYMBOL)
        if symbol_info is None:
            Logger.log(f"Symbol {Config.SYMBOL} not found", "ERROR")
            return False
        
        if not symbol_info.visible:
            mt5.symbol_select(Config.SYMBOL, True)
        
        self.connected = True
        return True
    
    def get_historical_data(self):
        rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, Config.LOOKBACK_BARS)
        if rates is None or len(rates) == 0:
            return None
        return pd.DataFrame(rates)
    
    def check_closed_positions(self):
        from_date = datetime.now() - timedelta(days=1)
        deals = mt5.history_deals_get(from_date, datetime.now())
        
        if deals is None:
            return
        
        for deal in deals:
            if deal.magic != Config.MAGIC_NUMBER or deal.entry != mt5.DEAL_ENTRY_OUT:
                continue
            
            ticket = deal.position_id
            existing_trade = self.trade_memory.get_trade_by_ticket(ticket)
            
            if existing_trade and 'profit' not in existing_trade:
                outcome = {
                    'profit': deal.profit,
                    'close_price': deal.price,
                    'close_time': deal.time,
                    'duration_seconds': deal.time - existing_trade['open_time']
                }
                self.trade_memory.update_trade_outcome(ticket, outcome)
                
                if ticket in self.active_positions:
                    del self.active_positions[ticket]
    
    def get_current_positions(self):
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        return len(positions) if positions else 0
    
    def calculate_position_size(self, stop_loss_pips):
        account = mt5.account_info()
        if not account:
            return Config.BASE_VOLUME
        
        balance = account.balance
        risk_amount = balance * Config.RISK_PERCENT
        
        symbol_info = mt5.symbol_info(Config.SYMBOL)
        pip_value = symbol_info.trade_tick_value
        
        volume = risk_amount / (stop_loss_pips * pip_value)
        volume_step = symbol_info.volume_step
        volume = round(volume / volume_step) * volume_step
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
        
        return volume
    
    def execute_trade(self, signal, confidence, df_current, features, model_agreement):
        """Execute trade with optimal entry, SL, and TP based on price action"""
        
        if self.get_current_positions() >= Config.MAX_POSITIONS:
            Logger.log("Max positions reached", "WARNING")
            return False
        
        tick = mt5.symbol_info_tick(Config.SYMBOL)
        if not tick:
            return False
        
        current_price = df_current.iloc[-1]['close']
        atr = df_current['close'].rolling(Config.ATR_PERIOD).std().iloc[-1] * 2
        
        # Calculate optimal levels using price action
        if Config.USE_DYNAMIC_SL_TP:
            levels = self.price_action.calculate_optimal_entry_sl_tp(
                df_current, signal, current_price, atr
            )
            
            optimal_entry = levels['optimal_entry']
            sl = levels['sl']
            tp = levels['tp']
            rr_ratio = levels['rr_ratio']
            
            Logger.log(
                f"📊 Price Action Analysis | Entry: {optimal_entry:.2f} | "
                f"SL: {sl:.2f} | TP: {tp:.2f} | R:R {rr_ratio:.2f}",
                "DATA"
            )
            Logger.log(
                f"📍 Levels | Support: {levels['support']:.2f} | "
                f"Resistance: {levels['resistance']:.2f}",
                "DATA"
            )
            
            # Check if current price is close to optimal entry
            distance_to_entry = abs(current_price - optimal_entry)
            entry_tolerance = atr * 0.3  # Within 30% of ATR
            
            if distance_to_entry > entry_tolerance and Config.USE_SMART_ENTRY:
                Logger.log(
                    f"⏳ Waiting for better entry | Current: {current_price:.2f} | "
                    f"Target: {optimal_entry:.2f} | Distance: {distance_to_entry:.2f}",
                    "WARNING"
                )
                # Store signal for later execution
                signal_id = f"{signal}_{int(datetime.now().timestamp())}"
                self.pending_signals[signal_id] = {
                    'signal': signal,
                    'confidence': confidence,
                    'optimal_entry': optimal_entry,
                    'sl': sl,
                    'tp': tp,
                    'features': features,
                    'model_agreement': model_agreement,
                    'created_at': datetime.now(),
                    'atr': atr
                }
                return False
            
            # Validate R:R ratio
            if rr_ratio < Config.MIN_RR_RATIO:
                Logger.log(
                    f"⚠ Poor R:R ratio {rr_ratio:.2f} (min: {Config.MIN_RR_RATIO}), skipping",
                    "WARNING"
                )
                return False
                
        else:
            # Fallback: Simple ATR-based SL/TP
            atr_multiplier = 1.5
            sl_distance = atr * atr_multiplier
            tp_distance = atr * atr_multiplier * 2
            
            if signal == 1:
                sl = current_price - sl_distance
                tp = current_price + tp_distance
            else:
                sl = current_price + sl_distance
                tp = current_price - tp_distance
            
            rr_ratio = 2.0
        
        # Execute order at current market price
        if signal == 1:
            order_type = mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        
        # Calculate position size based on actual SL
        sl_pips = abs(price - sl) / 0.01
        volume = self.calculate_position_size(sl_pips)
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": Config.SYMBOL,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": Config.MAGIC_NUMBER,
            "comment": f"ENS_{int(confidence*100)}_RR{int(rr_ratio*10)}",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            Logger.log(f"Order failed: {result.retcode} - {result.comment}", "ERROR")
            return False
        
        trade_data = {
            'ticket': result.order,
            'signal': int(signal),
            'confidence': float(confidence),
            'model_agreement': model_agreement,
            'open_price': price,
            'sl': sl,
            'tp': tp,
            'rr_ratio': rr_ratio,
            'volume': volume,
            'open_time': int(datetime.now().timestamp()),
            'features': features
        }
        
        self.trade_memory.add_trade(trade_data)
        self.active_positions[result.order] = trade_data
        
        signal_type = "BUY" if signal == 1 else "SELL"
        Logger.log(
            f"✅ {signal_type} EXECUTED | Ticket #{result.order} | "
            f"Entry: {price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}",
            "TRADE"
        )
        Logger.log(
            f"📊 Trade Details | Vol: {volume} | R:R: {rr_ratio:.2f} | "
            f"Conf: {confidence:.1%} | Agreement: {model_agreement['agreement']:.0%}",
            "TRADE"
        )
        return True
    
    def check_pending_signals(self, df_current):
        """Check if any pending signals can now be executed"""
        if not self.pending_signals:
            return
        
        current_price = df_current.iloc[-1]['close']
        signals_to_remove = []
        
        for signal_id, pending in self.pending_signals.items():
            # Check if signal is too old (expire after 1 hour)
            age = (datetime.now() - pending['created_at']).total_seconds() / 3600
            if age > 1:
                signals_to_remove.append(signal_id)
                Logger.log(f"⏰ Pending signal expired", "WARNING")
                continue
            
            # Check if price reached optimal entry
            optimal_entry = pending['optimal_entry']
            distance = abs(current_price - optimal_entry)
            tolerance = pending['atr'] * 0.2
            
            if distance <= tolerance:
                Logger.log(f"🎯 Optimal entry reached! Executing pending signal...", "SUCCESS")
                
                # Execute the trade
                tick = mt5.symbol_info_tick(Config.SYMBOL)
                if not tick:
                    continue
                
                signal = pending['signal']
                if signal == 1:
                    order_type = mt5.ORDER_TYPE_BUY
                    price = tick.ask
                else:
                    order_type = mt5.ORDER_TYPE_SELL
                    price = tick.bid
                
                sl_pips = abs(price - pending['sl']) / 0.01
                volume = self.calculate_position_size(sl_pips)
                
                request = {
                    "action": mt5.TRADE_ACTION_DEAL,
                    "symbol": Config.SYMBOL,
                    "volume": volume,
                    "type": order_type,
                    "price": price,
                    "sl": pending['sl'],
                    "tp": pending['tp'],
                    "magic": Config.MAGIC_NUMBER,
                    "comment": f"ENS_DELAYED_{int(pending['confidence']*100)}",
                    "type_time": mt5.ORDER_TIME_GTC,
                    "type_filling": mt5.ORDER_FILLING_IOC,
                }
                
                result = mt5.order_send(request)
                
                if result.retcode == mt5.TRADE_RETCODE_DONE:
                    trade_data = {
                        'ticket': result.order,
                        'signal': int(signal),
                        'confidence': float(pending['confidence']),
                        'model_agreement': pending['model_agreement'],
                        'open_price': price,
                        'sl': pending['sl'],
                        'tp': pending['tp'],
                        'volume': volume,
                        'open_time': int(datetime.now().timestamp()),
                        'features': pending['features']
                    }
                    
                    self.trade_memory.add_trade(trade_data)
                    self.active_positions[result.order] = trade_data
                    
                    signal_type = "BUY" if signal == 1 else "SELL"
                    Logger.log(
                        f"✅ {signal_type} (Delayed Entry) | Ticket #{result.order} | "
                        f"Price: {price:.2f} | SL: {pending['sl']:.2f} | TP: {pending['tp']:.2f}",
                        "TRADE"
                    )
                
                signals_to_remove.append(signal_id)
        
        # Clean up executed/expired signals
        for signal_id in signals_to_remove:
            del self.pending_signals[signal_id]
    
    def run(self):
        print("\n" + "=" * 70)
        print("🤖 MT5 ENSEMBLE AI TRADING SYSTEM v3.0")
        print("📊 Multi-Model Cross-Validation | Data Quality Checks")
        print("=" * 70 + "\n")
        
        if not self.connect_mt5():
            return
        
        df = self.get_historical_data()
        if df is None:
            Logger.log("Cannot proceed without data", "ERROR")
            return
        
        if not self.model.train(df):
            Logger.log("Training failed", "ERROR")
            return
        
        Logger.log(
            f"🎯 Trading {Config.SYMBOL} M15 | Min Confidence: {Config.MIN_CONFIDENCE:.0%} | "
            f"Min Agreement: {Config.MIN_ENSEMBLE_AGREEMENT:.0%}",
            "INFO"
        )
        
        try:
            iteration = 0
            while True:
                iteration += 1
                
                self.check_closed_positions()
                
                if self.model.should_retrain():
                    Logger.log("🔄 Retraining ensemble...", "LEARN")
                    df = self.get_historical_data()
                    if df is not None:
                        self.model.train(df)
                
                rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, 100)
                if rates is None or len(rates) < 50:
                    time.sleep(60)
                    continue
                
                df_current = pd.DataFrame(rates)
                signal, confidence, features, model_details = self.model.predict(df_current)
                
                if signal is None:
                    time.sleep(60)
                    continue
                
                current_price = df_current.iloc[-1]['close']
                atr = df_current['close'].rolling(Config.ATR_PERIOD).std().iloc[-1] * 2
                signal_type = "BUY" if signal == 1 else "SELL"
                positions = self.get_current_positions()
                
                # Calculate model agreement
                if model_details:
                    predictions = [p['prediction'] for p in model_details.values()]
                    agreement = predictions.count(signal) / len(predictions)
                    model_agreement = {'agreement': agreement, 'details': model_details}
                else:
                    model_agreement = {'agreement': 1.0, 'details': {}}
                
                if iteration % 1 == 0:
                    agreement_str = f"Agreement: {model_agreement['agreement']:.0%}" if model_details else ""
                    Logger.log(
                        f"Price: {current_price:.2f} | Signal: {signal_type} | "
                        f"Conf: {confidence:.1%} | {agreement_str} | Pos: {positions}",
                        "INFO"
                    )
                
                if confidence >= Config.MIN_CONFIDENCE and model_agreement['agreement'] >= Config.MIN_ENSEMBLE_AGREEMENT:
                    Logger.log(f"🎯 High confidence {signal_type} with strong model agreement!", "SUCCESS")
                    self.execute_trade(signal, confidence, df_current, features, model_agreement)
                
                # Check pending signals for optimal entry
                self.check_pending_signals(df_current)
                
                time.sleep(60)
                
        except KeyboardInterrupt:
            Logger.log("Shutdown requested", "WARNING")
        except Exception as e:
            Logger.log(f"Error: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
        finally:
            mt5.shutdown()
            Logger.log("Disconnected", "INFO")

def main():
    engine = TradingEngine()
    engine.run()

if __name__ == "__main__":
    main()


    #claude