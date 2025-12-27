import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datasets import load_dataset
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.calibration import CalibratedClassifierCV
from datetime import datetime, timedelta
import time
import sys
import json
import os
import warnings
import yaml
warnings.filterwarnings('ignore')

# Try to import XGBoost (optional)
try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    Logger.log("XGBoost not available, using alternative models", "WARNING")

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
    BASE_VOLUME = 0.2
    MAGIC_NUMBER = 998877
    
    # Risk Management
    RISK_PERCENT = 0.02
    MIN_CONFIDENCE = 0.6
    MIN_ENSEMBLE_AGREEMENT = 0.67
    MAX_POSITIONS = 2
    MAX_DAILY_LOSS_PERCENT = 5.0
    MAX_CONSECUTIVE_LOSSES = 3
    KELLY_FRACTION = 0.5  # Fractional Kelly for safety
    
    # Model Parameters
    LOOKBACK_BARS = 1000
    RETRAIN_HOURS = 12
    TRAINING_MIN_SAMPLES = 100
    
    # Dataset Configuration
    USE_HISTORICAL_DATASET = True
    HISTORICAL_DATA_LIMIT = 5000
    HISTORICAL_WEIGHT = 0.3
    RECENT_WEIGHT = 0.7
    
    # Ensemble Configuration
    USE_STACKING_ENSEMBLE = True
    MIN_DATA_QUALITY_SCORE = 0.7
    
    # Learning Parameters
    TRADE_HISTORY_FILE = "trade_history.json"
    MODEL_SAVE_FILE = "ensemble_model.joblib"
    MEMORY_SIZE = 500
    LEARNING_WEIGHT = 0.4
    
    # Technical Parameters
    ATR_PERIOD = 14
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    
    # Price Action Parameters
    USE_SMART_ENTRY = True
    USE_DYNAMIC_SL_TP = True
    MIN_RR_RATIO = 1.5
    LOOKBACK_SWING_POINTS = 50
    
    # Multi-timeframe Parameters
    MULTI_TIMEFRAME_ENABLED = True
    TIMEFRAMES = ['M5', 'M15', 'H1']
    TIMEFRAME_ALIGNMENT_THRESHOLD = 0.67
    
    # Market Regime Parameters
    USE_MARKET_REGIME = True
    ADX_TREND_THRESHOLD = 25
    
    # Order Execution
    MAX_SLIPPAGE_PIPS = 5
    ORDER_TIMEOUT_SECONDS = 30
    
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
            'ENSEMBLE': Logger.COLORS['GREEN'],
            'RISK': Logger.COLORS['YELLOW'],
            'PERFORMANCE': Logger.COLORS['MAGENTA']
        }
        color = colors.get(level, Logger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level}{Logger.COLORS['RESET']}] {message}", flush=True)

# ==========================================
# ADVANCED RISK MANAGEMENT
# ==========================================
class AdvancedRiskManager:
    """Dynamic risk management with Kelly Criterion and drawdown control"""
    
    def __init__(self):
        self.daily_pnl = 0
        self.consecutive_losses = 0
        self.daily_start_balance = None
        self.equity_curve = []
        self.trade_history = []
        
    def update_daily_start(self, balance):
        """Reset daily tracking at market open"""
        if self.daily_start_balance is None:
            self.daily_start_balance = balance
            self.daily_pnl = 0
            Logger.log(f"Daily tracking started | Starting balance: ${balance:.2f}", "RISK")
    
    def record_trade(self, profit):
        """Record trade outcome for risk management"""
        self.daily_pnl += profit
        self.trade_history.append(profit)
        
        if profit <= 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # Keep only recent trades
        if len(self.trade_history) > 100:
            self.trade_history.pop(0)
    
    def calculate_kelly_position_size(self, account_balance):
        """Calculate optimal position size using Kelly Criterion"""
        if len(self.trade_history) < 20:
            return Config.BASE_VOLUME
        
        # Calculate win rate and average win/loss
        wins = [p for p in self.trade_history if p > 0]
        losses = [p for p in self.trade_history if p <= 0]
        
        if not wins or not losses:
            return Config.BASE_VOLUME
        
        win_rate = len(wins) / len(self.trade_history)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return Config.BASE_VOLUME
        
        win_loss_ratio = avg_win / avg_loss
        
        # Kelly formula
        kelly_f = win_rate - ((1 - win_rate) / win_loss_ratio)
        
        # Apply fractional Kelly for safety
        fractional_kelly = max(0.01, kelly_f * Config.KELLY_FRACTION)
        
        # Convert to volume (simplified)
        position_size = Config.BASE_VOLUME * fractional_kelly * 10
        
        Logger.log(
            f"Kelly Calc | Win Rate: {win_rate:.1%} | W/L Ratio: {win_loss_ratio:.2f} | "
            f"Kelly: {kelly_f:.3f} | Position Size: {position_size:.3f}",
            "RISK"
        )
        
        return max(Config.BASE_VOLUME * 0.5, min(position_size, Config.BASE_VOLUME * 3))
    
    def check_risk_limits(self, account_balance):
        """Check if we should continue trading"""
        violations = []
        
        # Daily loss limit
        daily_loss_pct = (self.daily_pnl / account_balance) * 100
        if daily_loss_pct <= -Config.MAX_DAILY_LOSS_PERCENT:
            violations.append(f"Daily loss limit: {daily_loss_pct:.1f}%")
        
        # Consecutive losses
        if self.consecutive_losses >= Config.MAX_CONSECUTIVE_LOSSES:
            violations.append(f"Consecutive losses: {self.consecutive_losses}")
        
        # Margin safety
        account_info = mt5.account_info()
        if account_info and account_info.margin_level < 100:
            violations.append(f"Low margin: {account_info.margin_level:.1f}%")
        
        return len(violations) == 0, violations
    
    def get_risk_multiplier(self):
        """Get risk multiplier based on performance"""
        if self.consecutive_losses >= 2:
            return 0.5  # Reduce risk after losses
        if len(self.trade_history) > 10:
            recent_profits = self.trade_history[-10:]
            if sum(recent_profits) > 0:
                return 1.2  # Increase risk during winning streak
        return 1.0

# ==========================================
# PERFORMANCE MONITOR
# ==========================================
class PerformanceMonitor:
    """Monitor trading performance and provide insights"""
    
    def __init__(self):
        self.performance_history = []
        self.sharpe_ratios = []
        self.max_drawdown = 0
        self.current_drawdown = 0
        self.start_time = datetime.now()
        
    def update(self, trade_result):
        """Update performance metrics"""
        self.performance_history.append(trade_result)
        
        if len(self.performance_history) > 100:
            self.performance_history.pop(0)
        
        self.calculate_metrics()
    
    def calculate_metrics(self):
        """Calculate key performance indicators"""
        if len(self.performance_history) < 10:
            return
        
        returns = [t.get('profit', 0) / max(abs(t.get('risk', 1)), 1) 
                  for t in self.performance_history]
        
        # Sharpe Ratio (annualized)
        if len(returns) >= 10 and np.std(returns) > 0:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
            self.sharpe_ratios.append(sharpe)
            if len(self.sharpe_ratios) > 20:
                self.sharpe_ratios.pop(0)
        
        # Drawdown calculation
        cumulative = np.cumsum(returns)
        if len(cumulative) > 0:
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / peak * 100
            self.current_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
            self.max_drawdown = max(self.max_drawdown, self.current_drawdown)
    
    def get_performance_summary(self):
        """Get comprehensive performance summary"""
        if not self.performance_history:
            return None
        
        profits = [t.get('profit', 0) for t in self.performance_history]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]
        
        total_profit = sum(profits)
        win_rate = len(wins) / len(profits) if profits else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
        
        # Recovery factor (profit / max drawdown)
        recovery_factor = total_profit / self.max_drawdown if self.max_drawdown > 0 else 0
        
        # Average Sharpe
        avg_sharpe = np.mean(self.sharpe_ratios) if self.sharpe_ratios else 0
        
        runtime = datetime.now() - self.start_time
        
        return {
            'total_trades': len(self.performance_history),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown,
            'current_drawdown': self.current_drawdown,
            'sharpe_ratio': avg_sharpe,
            'recovery_factor': recovery_factor,
            'runtime_hours': runtime.total_seconds() / 3600,
            'profit_per_hour': total_profit / (runtime.total_seconds() / 3600) if runtime.total_seconds() > 0 else 0
        }
    
    def print_performance_report(self):
        """Print formatted performance report"""
        summary = self.get_performance_summary()
        if not summary:
            return
        
        Logger.log("=" * 60, "PERFORMANCE")
        Logger.log(f"📊 PERFORMANCE REPORT", "PERFORMANCE")
        Logger.log(f"Total Trades: {summary['total_trades']}", "PERFORMANCE")
        Logger.log(f"Win Rate: {summary['win_rate']:.1%}", "PERFORMANCE")
        Logger.log(f"Total Profit: ${summary['total_profit']:.2f}", "PERFORMANCE")
        Logger.log(f"Profit/Hour: ${summary['profit_per_hour']:.2f}", "PERFORMANCE")
        Logger.log(f"Profit Factor: {summary['profit_factor']:.2f}", "PERFORMANCE")
        Logger.log(f"Max Drawdown: {summary['max_drawdown']:.1f}%", "PERFORMANCE")
        Logger.log(f"Sharpe Ratio: {summary['sharpe_ratio']:.2f}", "PERFORMANCE")
        Logger.log("=" * 60, "PERFORMANCE")

# ==========================================
# DATA QUALITY CHECKER
# ==========================================
class DataQualityChecker:
    """Enhanced data quality validation"""
    
    @staticmethod
    def check_data_quality(df):
        scores = []
        
        # Missing values
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        scores.append(max(0, 1 - missing_ratio * 2))  # Penalize heavily
        
        # Variance check
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        variance_score = 1.0
        for col in numeric_cols:
            if df[col].std() == 0:
                variance_score *= 0.3  # Heavy penalty for no variance
            elif df[col].nunique() < 10:
                variance_score *= 0.7  # Penalize low diversity
        scores.append(variance_score)
        
        # Outlier detection
        outlier_score = 1.0
        price_cols = ['close', 'open', 'high', 'low']
        for col in price_cols:
            if col in df.columns:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers = ((df[col] < (q1 - 3*iqr)) | (df[col] > (q3 + 3*iqr))).sum()
                outlier_ratio = outliers / len(df)
                outlier_score *= max(0.5, 1 - outlier_ratio * 2)
        scores.append(outlier_score)
        
        # Chronological order
        if 'time' in df.columns:
            is_sorted = df['time'].is_monotonic_increasing
            scores.append(1.0 if is_sorted else 0.5)
        
        # Data freshness
        if 'time' in df.columns:
            latest_time = pd.to_datetime(df['time'].max(), unit='s')
            days_old = (datetime.now() - latest_time).days
            freshness_score = max(0, 1 - days_old / 30)  # 30 days max
            scores.append(freshness_score)
        
        overall_score = np.mean(scores)
        return overall_score, scores
    
    @staticmethod
    def validate_features(df, feature_cols):
        invalid_features = []
        
        for col in feature_cols:
            if col not in df.columns:
                invalid_features.append(f"{col} (missing)")
                continue
            
            # Check for inf values
            if np.isinf(df[col]).any():
                invalid_features.append(f"{col} (has inf)")
            
            # Check for too many NaN
            nan_ratio = df[col].isnull().sum() / len(df)
            if nan_ratio > 0.2:
                invalid_features.append(f"{col} ({nan_ratio:.1%} NaN)")
            
            # Check variance
            if df[col].std() == 0:
                invalid_features.append(f"{col} (no variance)")
        
        return invalid_features

# ==========================================
# TRADE MEMORY SYSTEM
# ==========================================
class TradeMemory:
    """Enhanced trade memory with performance tracking"""
    
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
        trade_data['id'] = len(self.trades)
        trade_data['timestamp'] = datetime.now().isoformat()
        self.trades.append(trade_data)
        self.save_history()
        Logger.log(f"Trade #{trade_data['id']} recorded", "LEARN")
    
    def update_trade_outcome(self, ticket, outcome_data):
        for trade in self.trades:
            if trade.get('ticket') == ticket:
                trade.update(outcome_data)
                trade['close_time'] = datetime.now().isoformat()
                
                # Calculate additional metrics
                if 'profit' in outcome_data and 'open_price' in trade:
                    profit = outcome_data['profit']
                    risk = abs(trade['open_price'] - trade.get('sl', trade['open_price']))
                    if risk > 0:
                        trade['rr_achieved'] = abs(profit) / risk
                
                self.save_history()
                
                result = "WIN" if profit > 0 else "LOSS"
                Logger.log(
                    f"Trade #{ticket} closed | {result} | P/L: ${profit:.2f} | "
                    f"Duration: {outcome_data.get('duration_seconds', 0)/60:.1f}min",
                    "SUCCESS" if profit > 0 else "WARNING"
                )
                return True
        return False
    
    def get_completed_trades(self):
        return [t for t in self.trades if 'profit' in t]
    
    def get_recent_trades(self, n=50):
        completed = self.get_completed_trades()
        return sorted(completed, key=lambda x: x.get('close_time', ''), reverse=True)[:n]
    
    def get_performance_stats(self, period_days=None):
        completed = self.get_completed_trades()
        
        if period_days:
            cutoff = datetime.now() - timedelta(days=period_days)
            completed = [t for t in completed if 
                        datetime.fromisoformat(t.get('close_time', '2000-01-01')) > cutoff]
        
        if not completed:
            return None
        
        profits = [t.get('profit', 0) for t in completed]
        wins = [p for p in profits if p > 0]
        losses = [p for p in profits if p <= 0]
        
        total_profit = sum(profits)
        win_rate = len(wins) / len(completed)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        profit_factor = abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf')
        
        # Calculate expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss))
        
        # Calculate consistency score
        if len(profits) >= 5:
            consistency = 1 - (np.std(profits) / abs(np.mean(profits))) if np.mean(profits) != 0 else 0
        else:
            consistency = 0
        
        return {
            'total_trades': len(completed),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'consistency': consistency
        }

# ==========================================
# ADVANCED FEATURE ENGINEERING
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
        
        # Moving Averages with crossovers
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        
        # MA Crossovers
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
        
        # ADX for trend strength
        if Config.USE_MARKET_REGIME:
            df['adx'] = self.calculate_adx(df)
            df['trend_strength'] = df['adx'] / 100
            df['regime'] = np.where(df['adx'] > Config.ADX_TREND_THRESHOLD, 1, 0)
        
        # Volume features
        if 'tick_volume' in df.columns:
            df['volume_sma'] = df['tick_volume'].rolling(20).mean()
            df['volume_ratio'] = df['tick_volume'] / df['volume_sma']
            df['volume_price_trend'] = df['volume_ratio'] * df['returns']
        
        # Momentum
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # Support/Resistance features
        df['distance_to_high'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['distance_to_low'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        
        # Time features
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
            df['hour'] = df['datetime'].dt.hour
            df['day_of_week'] = df['datetime'].dt.dayofweek
            df['month'] = df['datetime'].dt.month
        else:
            df['hour'] = 12
            df['day_of_week'] = 2
            df['month'] = 1
        
        # Cyclical encoding for time
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Gold seasonal pattern (strong in Q4)
        df['gold_seasonal'] = df['month'].apply(lambda x: 1 if x in [9, 10, 11, 12] else 0)
        
        # Market session (Asian, European, American)
        df['session'] = df['hour'].apply(self._get_market_session)
        
        return df
    
    def calculate_adx(self, df, period=Config.ADX_PERIOD):
        """Calculate Average Directional Index"""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr = np.maximum(high - low,
                       np.maximum(abs(high - close.shift(1)),
                                 abs(low - close.shift(1))))
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        tr_smooth = tr.rolling(period).mean()
        plus_dm_smooth = plus_dm.rolling(period).mean()
        minus_dm_smooth = minus_dm.rolling(period).mean()
        
        # Directional Indicators
        plus_di = 100 * plus_dm_smooth / tr_smooth
        minus_di = 100 * minus_dm_smooth / tr_smooth
        
        # DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(period).mean()
        
        return adx.fillna(0)
    
    def _get_market_session(self, hour):
        """Classify market session based on hour"""
        if 0 <= hour < 8:  # Asian session
            return 0
        elif 8 <= hour < 16:  # European session
            return 1
        else:  # American session
            return 2
    
    def create_labels(self, df, forward_bars=3, threshold=0.001):
        """Create labels for classification"""
        df = df.copy()
        
        # Forward returns
        df['forward_return'] = df['close'].shift(-forward_bars) / df['close'] - 1
        
        # Dynamic threshold based on volatility
        volatility = df['returns'].rolling(20).std().fillna(0.001)
        dynamic_threshold = threshold * (1 + volatility * 10)
        
        # Create labels: 1=Buy, 0=Sell, -1=Hold
        df['label'] = -1
        df.loc[df['forward_return'] > dynamic_threshold, 'label'] = 1
        df.loc[df['forward_return'] < -dynamic_threshold, 'label'] = 0
        
        # Add label confidence based on magnitude
        df['label_confidence'] = abs(df['forward_return']) / dynamic_threshold
        
        return df
    
    def get_feature_columns(self):
        """Return list of feature columns"""
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
            base_features.extend(['trend_strength', 'regime'])
        
        return base_features

# ==========================================
# MULTI-TIMEFRAME ANALYZER
# ==========================================
class MultiTimeframeAnalyzer:
    """Analyze multiple timeframes for signal confirmation"""
    
    def __init__(self):
        self.timeframe_map = {
            'M5': mt5.TIMEFRAME_M5,
            'M15': mt5.TIMEFRAME_M15,
            'H1': mt5.TIMEFRAME_H1,
            'H4': mt5.TIMEFRAME_H4
        }
        self.active_timeframes = Config.TIMEFRAMES
    
    def get_multi_timeframe_data(self):
        """Fetch data for all active timeframes"""
        mtf_data = {}
        
        for tf_name in self.active_timeframes:
            if tf_name in self.timeframe_map:
                rates = mt5.copy_rates_from_pos(Config.SYMBOL, self.timeframe_map[tf_name], 0, 300)
                if rates is not None and len(rates) > 50:
                    df = pd.DataFrame(rates)
                    mtf_data[tf_name] = df
                else:
                    Logger.log(f"Could not fetch {tf_name} data", "WARNING")
        
        return mtf_data
    
    def analyze_timeframe_alignment(self, model, mtf_data):
        """Check signal alignment across timeframes"""
        if not Config.MULTI_TIMEFRAME_ENABLED or len(mtf_data) < 2:
            return None, 0.0
        
        signals = {}
        confidences = {}
        
        for tf_name, df in mtf_data.items():
            signal, confidence, _, _ = model.predict(df)
            if signal is not None:
                signals[tf_name] = signal
                confidences[tf_name] = confidence
        
        if not signals:
            return None, 0.0
        
        # Count signals in same direction
        buy_signals = sum(1 for s in signals.values() if s == 1)
        sell_signals = sum(1 for s in signals.values() if s == 0)
        total = len(signals)
        
        if total == 0:
            return None, 0.0
        
        # Calculate alignment
        alignment_ratio = max(buy_signals, sell_signals) / total
        avg_confidence = np.mean(list(confidences.values())) if confidences else 0
        
        if buy_signals / total >= Config.TIMEFRAME_ALIGNMENT_THRESHOLD:
            return 1, alignment_ratio * avg_confidence
        elif sell_signals / total >= Config.TIMEFRAME_ALIGNMENT_THRESHOLD:
            return 0, alignment_ratio * avg_confidence
        
        return None, alignment_ratio * avg_confidence
    
    def get_timeframe_summary(self, signals):
        """Get summary of timeframe signals"""
        summary = {}
        for tf_name, signal in signals.items():
            summary[tf_name] = "BUY" if signal == 1 else "SELL" if signal == 0 else "NEUTRAL"
        return summary

# ==========================================
# ADVANCED ENSEMBLE MODEL
# ==========================================
class AdvancedEnsemble:
    """Enhanced ensemble with stacking and calibration"""
    
    def __init__(self, trade_memory):
        self.feature_engine = AdvancedFeatureEngine()
        self.trade_memory = trade_memory
        self.data_quality_checker = DataQualityChecker()
        
        # Initialize base models
        self.base_models = self._initialize_base_models()
        
        # Create ensemble
        if Config.USE_STACKING_ENSEMBLE:
            self.ensemble = self._create_stacking_ensemble()
        else:
            self.ensemble = VotingClassifier(
                estimators=[(name, model) for name, model in self.base_models],
                voting='soft',
                weights=[1.0, 1.2, 1.0, 1.1]  # Weight RF more
            )
        
        # Calibrate for better probabilities
        self.calibrated_ensemble = CalibratedClassifierCV(
            self.ensemble, method='sigmoid', cv=3
        )
        
        self.is_trained = False
        self.last_train_time = None
        self.model_scores = {}
        self.historical_data_cache = None
        self.feature_importance = {}
    
    def _initialize_base_models(self):
        """Initialize diverse base models"""
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
        
        # Add XGBoost if available
        if XGB_AVAILABLE:
            models.append(('XGB', XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, n_jobs=-1
            )))
        
        return models
    
    def _create_stacking_ensemble(self):
        """Create stacking ensemble with meta-learner"""
        # Use all but NN as base estimators for stacking
        base_estimators = [(name, model) for name, model in self.base_models 
                          if name != 'NN']
        
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=LogisticRegression(max_iter=1000, random_state=42),
            cv=3,
            passthrough=True,
            n_jobs=-1
        )
    
    def load_all_training_data(self, recent_mt5_data):
        """Load and combine data from multiple sources"""
        all_dfs = []
        
        # 1. Historical dataset
        if Config.USE_HISTORICAL_DATASET:
            if self.historical_data_cache is None:
                self.historical_data_cache = DataLoader.load_huggingface_dataset()
            
            if self.historical_data_cache is not None:
                df_h = self.feature_engine.calculate_features(self.historical_data_cache)
                df_h = self.feature_engine.create_labels(df_h).dropna()
                df_h = df_h[df_h['label'] != -1].tail(Config.HISTORICAL_DATA_LIMIT)
                df_h['weight'] = Config.HISTORICAL_WEIGHT
                df_h['data_source'] = 'historical'
                all_dfs.append(df_h)
                Logger.log(f"Added {len(df_h)} historical samples", "DATA")
        
        # 2. Recent MT5 data
        if recent_mt5_data is not None and len(recent_mt5_data) > 100:
            df_r = self.feature_engine.calculate_features(recent_mt5_data)
            df_r = self.feature_engine.create_labels(df_r).dropna()
            df_r = df_r[df_r['label'] != -1]
            df_r['weight'] = Config.RECENT_WEIGHT
            df_r['data_source'] = 'recent'
            all_dfs.append(df_r)
            Logger.log(f"Added {len(df_r)} recent samples", "DATA")
        
        # 3. Trade experience (reinforcement learning)
        trades = self.trade_memory.get_completed_trades()
        if trades:
            t_samples = []
            for t in trades:
                if 'features' in t and 'profit' in t:
                    # Enhanced learning: consider profit magnitude
                    profit = t['profit']
                    signal = t['signal']
                    
                    if profit > 0:
                        # Winning trade: reinforce the signal
                        label = signal
                        weight = Config.LEARNING_WEIGHT * (1 + min(profit / 100, 2))
                    else:
                        # Losing trade: opposite signal
                        label = 1 - signal
                        weight = Config.LEARNING_WEIGHT * (1 + min(abs(profit) / 100, 1))
                    
                    sample = {**t['features'], 'label': label, 'weight': weight}
                    t_samples.append(sample)
            
            if t_samples:
                df_t = pd.DataFrame(t_samples)
                df_t['data_source'] = 'experience'
                all_dfs.append(df_t)
                Logger.log(f"Added {len(df_t)} experience samples", "LEARN")
        
        if not all_dfs:
            return None
        
        # Combine all data
        combined = pd.concat(all_dfs, ignore_index=True, sort=False)
        
        # Quality check
        quality_score, _ = self.data_quality_checker.check_data_quality(combined)
        Logger.log(f"Training Data Quality: {quality_score:.2%} | Samples: {len(combined)}", "DATA")
        
        if quality_score < Config.MIN_DATA_QUALITY_SCORE:
            Logger.log(f"Data quality below threshold ({Config.MIN_DATA_QUALITY_SCORE:.0%})", "WARNING")
            return None
        
        return combined
    
    def train(self, recent_mt5_data):
        """Train the ensemble with comprehensive validation"""
        Logger.log("=" * 50, "INFO")
        Logger.log("🚀 STARTING ADVANCED ENSEMBLE TRAINING", "ENSEMBLE")
        
        df = self.load_all_training_data(recent_mt5_data)
        if df is None or len(df) < Config.TRAINING_MIN_SAMPLES:
            Logger.log(f"Training aborted: Insufficient data ({len(df) if df else 0} samples)", "ERROR")
            return False
        
        # Prepare features and labels
        feature_cols = self.feature_engine.get_feature_columns()
        
        # Validate features
        invalid_features = self.data_quality_checker.validate_features(df, feature_cols)
        if invalid_features:
            Logger.log(f"Invalid features: {', '.join(invalid_features)}", "WARNING")
            # Remove invalid features
            feature_cols = [f for f in feature_cols if f not in [x.split(' ')[0] for x in invalid_features]]
        
        X = df[feature_cols].fillna(0)
        y = df['label']
        
        # Check class balance
        class_distribution = y.value_counts(normalize=True)
        Logger.log(f"Class distribution: {dict(class_distribution)}", "DATA")
        
        if len(class_distribution) < 2:
            Logger.log("Insufficient class variety for training", "ERROR")
            return False
        
        weights = df['weight'].values if 'weight' in df.columns else None
        
        # Scale features
        X_scaled = self.feature_engine.scaler.fit_transform(X)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Train and validate base models
        Logger.log("Training base models...", "ENSEMBLE")
        for name, model in self.base_models:
            try:
                cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, 
                                           scoring='accuracy', n_jobs=-1)
                avg_score = np.mean(cv_scores)
                self.model_scores[name] = avg_score
                Logger.log(f"  {name}: {avg_score:.2%} accuracy", "ENSEMBLE")
                
                # Train individual model for feature importance
                model.fit(X_scaled, y)
                
            except Exception as e:
                Logger.log(f"  {name} failed: {str(e)}", "WARNING")
                self.model_scores[name] = 0
        
        # Train the main ensemble
        Logger.log("Training ensemble...", "ENSEMBLE")
        try:
            if weights is not None:
                self.calibrated_ensemble.fit(X_scaled, y, sample_weight=weights)
            else:
                self.calibrated_ensemble.fit(X_scaled, y)
            
            # Calculate feature importance from Random Forest
            if 'RF' in [name for name, _ in self.base_models]:
                rf_model = next(model for name, model in self.base_models if name == 'RF')[1]
                if hasattr(rf_model, 'feature_importances_'):
                    self.feature_importance = dict(zip(feature_cols, rf_model.feature_importances_))
                    
                    # Log top features
                    top_features = sorted(self.feature_importance.items(), 
                                        key=lambda x: x[1], reverse=True)[:10]
                    Logger.log("Top 10 features by importance:", "ENSEMBLE")
                    for feature, importance in top_features:
                        Logger.log(f"  {feature}: {importance:.3f}", "ENSEMBLE")
            
            self.is_trained = True
            self.last_train_time = datetime.now()
            
            # Validate ensemble performance
            ensemble_score = np.mean(cross_val_score(
                self.calibrated_ensemble, X_scaled, y, cv=tscv, scoring='accuracy'
            ))
            Logger.log(f"✅ Ensemble Training Complete | CV Accuracy: {ensemble_score:.2%}", "SUCCESS")
            
            return True
            
        except Exception as e:
            Logger.log(f"Ensemble training failed: {str(e)}", "ERROR")
            return False
    
    def predict(self, df):
        """Make prediction with calibrated probabilities"""
        if not self.is_trained:
            return None, 0.0, None, {}
        
        # Calculate features
        df_feat = self.feature_engine.calculate_features(df)
        feature_cols = self.feature_engine.get_feature_columns()
        
        # Get latest features
        X = df_feat[feature_cols].iloc[-1:].fillna(0).values
        
        # Create feature dictionary
        f_dict = {col: float(X[0][i]) for i, col in enumerate(feature_cols)}
        
        # Scale features
        X_scaled = self.feature_engine.scaler.transform(X)
        
        # Get predictions from base models
        sub_preds = {}
        if hasattr(self.calibrated_ensemble, 'calibrated_classifiers_'):
            # For calibrated ensemble, get base predictions
            for name, model in self.base_models:
                try:
                    p = model.predict(X_scaled)[0]
                    proba = model.predict_proba(X_scaled)[0]
                    c = np.max(proba)
                    sub_preds[name] = {'prediction': p, 'confidence': c}
                except:
                    sub_preds[name] = {'prediction': -1, 'confidence': 0}
        
        # Get ensemble prediction
        final_p = self.calibrated_ensemble.predict(X_scaled)[0]
        proba = self.calibrated_ensemble.predict_proba(X_scaled)[0]
        final_c = np.max(proba)
        
        # Get prediction probabilities for each class
        class_probs = {i: proba[i] for i in range(len(proba))}
        
        # Check agreement
        if sub_preds:
            raw_votes = [m['prediction'] for m in sub_preds.values() 
                        if m['prediction'] != -1]
            if raw_votes:
                agreement = raw_votes.count(final_p) / len(raw_votes)
                
                if agreement < Config.MIN_ENSEMBLE_AGREEMENT:
                    Logger.log(f"Signal filtered: Low agreement ({agreement:.0%})", "WARNING")
                    return None, 0.0, None, sub_preds
        
        # Add market regime context
        if Config.USE_MARKET_REGIME and 'regime' in f_dict:
            regime = f_dict['regime']
            if regime > 0.5:  # Trending market
                # Require higher confidence in trending markets
                confidence_threshold = Config.MIN_CONFIDENCE * 1.2
            else:  # Ranging market
                confidence_threshold = Config.MIN_CONFIDENCE * 0.9
            
            if final_c < confidence_threshold:
                Logger.log(f"Signal filtered: Low confidence for regime {regime:.1f}", "WARNING")
                return None, 0.0, None, sub_preds
        
        return final_p, final_c, f_dict, sub_preds
    
    def should_retrain(self):
        """Check if retraining is needed"""
        if not self.last_train_time:
            return True
        
        hours_since = (datetime.now() - self.last_train_time).total_seconds() / 3600
        
        # Dynamic retraining based on market conditions
        if hours_since >= Config.RETRAIN_HOURS:
            return True
        
        # Retrain if performance is degrading
        stats = self.trade_memory.get_performance_stats(period_days=1)
        if stats and stats['win_rate'] < 0.4 and stats['total_trades'] >= 10:
            Logger.log("Performance degradation detected, retraining...", "LEARN")
            return True
        
        return False

# ==========================================
# PRICE ACTION ANALYZER (ENHANCED)
# ==========================================
class EnhancedPriceActionAnalyzer:
    """Enhanced price action analysis with multiple techniques"""
    
    @staticmethod
    def find_swing_points(df, lookback=50, strength=2):
        """Find swing highs and lows with configurable strength"""
        df = df.tail(lookback).copy().reset_index(drop=True)
        
        swing_highs = []
        swing_lows = []
        
        for i in range(strength, len(df) - strength):
            # Check for swing high
            is_high = True
            for j in range(1, strength + 1):
                if df['high'].iloc[i] <= df['high'].iloc[i - j] or \
                   df['high'].iloc[i] <= df['high'].iloc[i + j]:
                    is_high = False
                    break
            
            if is_high:
                swing_highs.append({
                    'price': df['high'].iloc[i],
                    'index': i,
                    'time': df['time'].iloc[i] if 'time' in df.columns else i,
                    'strength': strength
                })
            
            # Check for swing low
            is_low = True
            for j in range(1, strength + 1):
                if df['low'].iloc[i] >= df['low'].iloc[i - j] or \
                   df['low'].iloc[i] >= df['low'].iloc[i + j]:
                    is_low = False
                    break
            
            if is_low:
                swing_lows.append({
                    'price': df['low'].iloc[i],
                    'index': i,
                    'time': df['time'].iloc[i] if 'time' in df.columns else i,
                    'strength': strength
                })
        
        return swing_highs, swing_lows
    
    @staticmethod
    def find_support_resistance_clusters(df, current_price, lookback=100):
        """Find support/resistance clusters using price density"""
        prices = pd.concat([df['high'], df['low'], df['close']]).tail(lookback * 3)
        
        # Use KDE to find price clusters
        from scipy import stats
        
        try:
            kde = stats.gaussian_kde(prices)
            x_range = np.linspace(prices.min(), prices.max(), 200)
            density = kde(x_range)
            
            # Find local maxima as S/R levels
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(density, height=np.mean(density) * 1.2)
            
            if len(peaks) > 0:
                levels = x_range[peaks]
                
                # Separate support and resistance
                support_levels = [l for l in levels if l < current_price]
                resistance_levels = [l for l in levels if l > current_price]
                
                # Find nearest levels
                nearest_support = max(support_levels) if support_levels else current_price * 0.99
                nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.01
                
                return nearest_support, nearest_resistance, levels
            
        except:
            pass
        
        # Fallback to swing points
        swing_highs, swing_lows = EnhancedPriceActionAnalyzer.find_swing_points(df, lookback)
        
        resistances = [s['price'] for s in swing_highs if s['price'] > current_price]
        supports = [s['price'] for s in swing_lows if s['price'] < current_price]
        
        support = max(supports) if supports else current_price * 0.99
        resistance = min(resistances) if resistances else current_price * 1.01
        
        return support, resistance, []
    
    @staticmethod
    def calculate_fibonacci_levels(df):
        """Calculate Fibonacci retracement levels"""
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        price_range = recent_high - recent_low
        
        fib_levels = {
            '0.0': recent_low,
            '0.236': recent_low + price_range * 0.236,
            '0.382': recent_low + price_range * 0.382,
            '0.5': recent_low + price_range * 0.5,
            '0.618': recent_low + price_range * 0.618,
            '0.786': recent_low + price_range * 0.786,
            '1.0': recent_high
        }
        
        return fib_levels
    
    @staticmethod
    def calculate_optimal_entry_sl_tp(df, signal, current_price, atr, risk_reward_ratio=Config.MIN_RR_RATIO):
        """Calculate optimal trade parameters with multiple techniques"""
        
        # Get multiple S/R estimates
        support, resistance, all_levels = EnhancedPriceActionAnalyzer.find_support_resistance_clusters(
            df, current_price, Config.LOOKBACK_SWING_POINTS
        )
        
        # Calculate Fibonacci levels
        fib_levels = EnhancedPriceActionAnalyzer.calculate_fibonacci_levels(df)
        
        # Calculate pivot points
        pivots = PriceActionAnalyzer.calculate_pivot_points(df)
        
        if signal == 1:  # BUY Signal
            # Entry logic
            optimal_entry = current_price
            
            # Multiple stop loss candidates
            sl_candidates = [
                support - (atr * 0.5),  # Below support with buffer
                current_price - (atr * 1.5),  # ATR-based
                fib_levels['0.618'] if current_price > fib_levels['0.618'] else current_price - (atr * 2)
            ]
            
            # Choose the tightest reasonable stop
            sl = max(sl_candidates)
            
            # Multiple take profit candidates
            tp_candidates = [
                resistance,
                pivots['r1'],
                fib_levels['1.0'],
                current_price + (atr * 3),
                current_price + (abs(current_price - sl) * risk_reward_ratio)
            ]
            
            # Filter and choose best TP
            valid_tps = [tp for tp in tp_candidates if tp > optimal_entry]
            if valid_tps:
                # Choose TP that meets risk:reward
                for tp in sorted(valid_tps):
                    risk = optimal_entry - sl
                    reward = tp - optimal_entry
                    if risk > 0 and reward / risk >= risk_reward_ratio:
                        final_tp = tp
                        break
                else:
                    # Fallback to minimum R:R
                    final_tp = optimal_entry + (risk * risk_reward_ratio)
            else:
                final_tp = optimal_entry + (atr * 3)
            
            # Smart entry: wait for pullback
            if Config.USE_SMART_ENTRY:
                pullback_targets = [
                    support + (atr * 0.3),
                    fib_levels['0.618'] if fib_levels['0.618'] < current_price else None,
                    pivots['pivot']
                ]
                
                valid_targets = [t for t in pullback_targets if t is not None and t < current_price]
                if valid_targets:
                    optimal_entry = max(valid_targets)
            
        else:  # SELL Signal
            optimal_entry = current_price
            
            # Stop loss candidates
            sl_candidates = [
                resistance + (atr * 0.5),
                current_price + (atr * 1.5),
                fib_levels['0.382'] if current_price < fib_levels['0.382'] else current_price + (atr * 2)
            ]
            
            sl = min(sl_candidates)
            
            # Take profit candidates
            tp_candidates = [
                support,
                pivots['s1'],
                fib_levels['0.0'],
                current_price - (atr * 3),
                current_price - (abs(sl - current_price) * risk_reward_ratio)
            ]
            
            valid_tps = [tp for tp in tp_candidates if tp < optimal_entry]
            if valid_tps:
                for tp in sorted(valid_tps, reverse=True):
                    risk = sl - optimal_entry
                    reward = optimal_entry - tp
                    if risk > 0 and reward / risk >= risk_reward_ratio:
                        final_tp = tp
                        break
                else:
                    final_tp = optimal_entry - (risk * risk_reward_ratio)
            else:
                final_tp = optimal_entry - (atr * 3)
            
            # Smart entry: wait for retracement
            if Config.USE_SMART_ENTRY:
                retracement_targets = [
                    resistance - (atr * 0.3),
                    fib_levels['0.382'] if fib_levels['0.382'] > current_price else None,
                    pivots['pivot']
                ]
                
                valid_targets = [t for t in retracement_targets if t is not None and t > current_price]
                if valid_targets:
                    optimal_entry = min(valid_targets)
        
        # Calculate final R:R ratio
        risk_amount = abs(optimal_entry - sl)
        reward_amount = abs(final_tp - optimal_entry)
        rr_ratio = reward_amount / risk_amount if risk_amount > 0 else 0
        
        return {
            'optimal_entry': optimal_entry,
            'sl': sl,
            'tp': final_tp,
            'current_price': current_price,
            'support': support,
            'resistance': resistance,
            'fib_levels': fib_levels,
            'rr_ratio': rr_ratio,
            'risk_amount': risk_amount,
            'reward_amount': reward_amount,
            'distance_to_optimal': abs(current_price - optimal_entry),
            'all_levels': all_levels
        }

# ==========================================
# SMART ORDER EXECUTOR
# ==========================================
class SmartOrderExecutor:
    """Intelligent order execution with slippage control"""
    
    def __init__(self):
        self.pending_orders = {}
        self.order_timeout = Config.ORDER_TIMEOUT_SECONDS
    
    def execute_trade(self, symbol, order_type, volume, entry_price, sl, tp, magic, comment=""):
        """Execute trade with intelligent order placement"""
        
        symbol_info = mt5.symbol_info(symbol)
        if not symbol_info:
            Logger.log(f"Symbol {symbol} not found", "ERROR")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            Logger.log(f"Cannot get tick data for {symbol}", "ERROR")
            return None
        
        # Determine actual entry price based on order type
        if order_type == mt5.ORDER_TYPE_BUY:
            current_price = tick.ask
            slippage = abs(current_price - entry_price) / symbol_info.point
            
            # Check if slippage is acceptable
            if slippage > Config.MAX_SLIPPAGE_PIPS:
                Logger.log(f"Slippage too high: {slippage:.1f} pips", "WARNING")
                
                # Try limit order
                if Config.USE_SMART_ENTRY and entry_price < current_price:
                    return self.place_limit_order(
                        symbol, mt5.ORDER_TYPE_BUY_LIMIT, volume, 
                        entry_price, sl, tp, magic, comment
                    )
            
            # Market order
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY,
                "price": current_price,
                "sl": sl,
                "tp": tp,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            
        else:  # SELL order
            current_price = tick.bid
            slippage = abs(current_price - entry_price) / symbol_info.point
            
            if slippage > Config.MAX_SLIPPAGE_PIPS:
                Logger.log(f"Slippage too high: {slippage:.1f} pips", "WARNING")
                
                if Config.USE_SMART_ENTRY and entry_price > current_price:
                    return self.place_limit_order(
                        symbol, mt5.ORDER_TYPE_SELL_LIMIT, volume,
                        entry_price, sl, tp, magic, comment
                    )
            
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_SELL,
                "price": current_price,
                "sl": sl,
                "tp": tp,
                "magic": magic,
                "comment": comment,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
        
        # Send order
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            Logger.log(f"Order failed: {result.retcode} - {result.comment}", "ERROR")
            
            # Try with different filling mode
            if result.retcode == mt5.TRADE_RETCODE_REQUOTE:
                request["type_filling"] = mt5.ORDER_FILLING_RETURN
                result = mt5.order_send(request)
        
        return result
    
    def place_limit_order(self, symbol, order_type, volume, price, sl, tp, magic, comment):
        """Place a limit order for better entry"""
        
        request = {
            "action": mt5.TRADE_ACTION_PENDING,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": magic,
            "comment": f"LIMIT_{comment}",
            "type_time": mt5.ORDER_TIME_DAY,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        
        result = mt5.order_send(request)
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            order_ticket = result.order
            self.pending_orders[order_ticket] = {
                'placed_at': datetime.now(),
                'symbol': symbol,
                'price': price
            }
            Logger.log(f"Limit order placed: #{order_ticket} at {price:.2f}", "INFO")
        
        return result
    
    def check_pending_orders(self):
        """Check and manage pending orders"""
        expired_orders = []
        
        for order_ticket, order_info in self.pending_orders.items():
            # Check if order is still pending
            order = mt5.order_get(ticket=order_ticket)
            
            if order is None or order.time_done > 0:
                # Order filled or cancelled
                expired_orders.append(order_ticket)
                continue
            
            # Check if order expired
            age = (datetime.now() - order_info['placed_at']).total_seconds()
            if age > self.order_timeout:
                mt5.order_delete(order_ticket)
                expired_orders.append(order_ticket)
                Logger.log(f"Limit order #{order_ticket} expired", "INFO")
        
        # Clean up expired orders
        for ticket in expired_orders:
            if ticket in self.pending_orders:
                del self.pending_orders[ticket]

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
# ENHANCED TRADING ENGINE
# ==========================================
class EnhancedTradingEngine:
    """Main trading engine with all enhancements"""
    
    def __init__(self):
        self.trade_memory = TradeMemory()
        self.risk_manager = AdvancedRiskManager()
        self.performance_monitor = PerformanceMonitor()
        self.price_action = EnhancedPriceActionAnalyzer()
        self.order_executor = SmartOrderExecutor()
        self.mtf_analyzer = MultiTimeframeAnalyzer()
        self.model = AdvancedEnsemble(self.trade_memory)
        
        self.connected = False
        self.active_positions = {}
        self.pending_signals = {}
        self.iteration = 0
        self.last_performance_report = datetime.now()
        
    def connect_mt5(self):
        """Connect to MT5 terminal"""
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
        if account:
            Logger.log(f"✓ Connected | Account: {account.login} | "
                      f"Balance: ${account.balance:.2f} | Equity: ${account.equity:.2f}", "SUCCESS")
            
            # Initialize risk manager with starting balance
            self.risk_manager.update_daily_start(account.balance)
        else:
            Logger.log("✓ Connected (account info unavailable)", "SUCCESS")
        
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
    
    def get_historical_data(self, timeframe=None, bars=None):
        """Get historical data from MT5"""
        if timeframe is None:
            timeframe = Config.TIMEFRAME
        if bars is None:
            bars = Config.LOOKBACK_BARS
        
        rates = mt5.copy_rates_from_pos(Config.SYMBOL, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        return pd.DataFrame(rates)
    
    def check_closed_positions(self):
        """Check for closed positions and update records"""
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
                
                # Update risk manager and performance monitor
                self.risk_manager.record_trade(deal.profit)
                self.performance_monitor.update({
                    'profit': deal.profit,
                    'risk': abs(existing_trade.get('open_price', 0) - existing_trade.get('sl', 0))
                })
                
                if ticket in self.active_positions:
                    del self.active_positions[ticket]
    
    def get_current_positions(self):
        """Get current open positions"""
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        return len(positions) if positions else 0
    
    def calculate_position_size(self, stop_loss_pips, risk_multiplier=1.0):
        """Calculate position size based on risk management"""
        account = mt5.account_info()
        if not account:
            return Config.BASE_VOLUME
        
        # Get dynamic position size from risk manager
        kelly_size = self.risk_manager.calculate_kelly_position_size(account.balance)
        
        # Adjust for current risk
        risk_amount = account.balance * Config.RISK_PERCENT * risk_multiplier
        
        symbol_info = mt5.symbol_info(Config.SYMBOL)
        if not symbol_info:
            return kelly_size
        
        pip_value = symbol_info.trade_tick_value
        
        # Calculate volume based on risk
        if stop_loss_pips > 0 and pip_value > 0:
            volume = risk_amount / (stop_loss_pips * pip_value)
        else:
            volume = kelly_size
        
        # Apply broker constraints
        volume_step = symbol_info.volume_step
        volume = round(volume / volume_step) * volume_step
        volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
        
        # Blend with Kelly size
        final_volume = (volume + kelly_size) / 2
        
        Logger.log(f"Position Size | Risk: ${risk_amount:.2f} | "
                  f"Kelly: {kelly_size:.3f} | Final: {final_volume:.3f}", "RISK")
        
        return final_volume
    
    def execute_enhanced_trade(self, signal, confidence, df_current, features, model_agreement):
        """Execute trade with all enhancements"""
        
        # Check position limits
        if self.get_current_positions() >= Config.MAX_POSITIONS:
            Logger.log("Max positions reached", "WARNING")
            return False
        
        # Check risk limits
        account = mt5.account_info()
        if account:
            can_trade, violations = self.risk_manager.check_risk_limits(account.balance)
            if not can_trade:
                Logger.log(f"Trading halted: {', '.join(violations)}", "RISK")
                return False
        
        tick = mt5.symbol_info_tick(Config.SYMBOL)
        if not tick:
            return False
        
        current_price = df_current.iloc[-1]['close']
        
        # Calculate ATR for risk assessment
        atr = df_current['close'].rolling(Config.ATR_PERIOD).std().iloc[-1] * 2
        if pd.isna(atr) or atr == 0:
            atr = current_price * 0.001
        
        # Calculate optimal trade parameters
        if Config.USE_DYNAMIC_SL_TP:
            levels = self.price_action.calculate_optimal_entry_sl_tp(
                df_current, signal, current_price, atr, Config.MIN_RR_RATIO
            )
            
            optimal_entry = levels['optimal_entry']
            sl = levels['sl']
            tp = levels['tp']
            rr_ratio = levels['rr_ratio']
            
            Logger.log(
                f"📊 Enhanced Price Analysis", "DATA"
            )
            Logger.log(
                f"  Entry: {optimal_entry:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | R:R: {rr_ratio:.2f}", "DATA"
            )
            Logger.log(
                f"  Support: {levels['support']:.2f} | Resistance: {levels['resistance']:.2f}", "DATA"
            )
            
            # Validate R:R ratio
            if rr_ratio < Config.MIN_RR_RATIO:
                Logger.log(f"Poor R:R ratio {rr_ratio:.2f} (min: {Config.MIN_RR_RATIO})", "WARNING")
                return False
            
            # Check for smart entry opportunity
            distance_to_entry = abs(current_price - optimal_entry)
            entry_tolerance = atr * 0.3
            
            if distance_to_entry > entry_tolerance and Config.USE_SMART_ENTRY:
                Logger.log(
                    f"⏳ Waiting for better entry | Current: {current_price:.2f} | "
                    f"Target: {optimal_entry:.2f} | Distance: {distance_to_entry:.2f}", "WARNING"
                )
                
                # Store as pending signal
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
                    'atr': atr,
                    'rr_ratio': rr_ratio
                }
                return False
        else:
            # Simple ATR-based approach (fallback)
            sl_distance = atr * 1.5
            tp_distance = atr * 3
            
            if signal == 1:
                optimal_entry = current_price
                sl = current_price - sl_distance
                tp = current_price + tp_distance
            else:
                optimal_entry = current_price
                sl = current_price + sl_distance
                tp = current_price - tp_distance
            
            rr_ratio = 2.0
        
        # Determine order type
        if signal == 1:
            order_type = mt5.ORDER_TYPE_BUY
        else:
            order_type = mt5.ORDER_TYPE_SELL
        
        # Calculate position size
        stop_loss_pips = abs(optimal_entry - sl) / 0.01  # Assuming 0.01 is 1 pip for XAUUSD
        risk_multiplier = self.risk_manager.get_risk_multiplier()
        volume = self.calculate_position_size(stop_loss_pips, risk_multiplier)
        
        # Prepare order comment
        comment = (f"ENS_{int(confidence*100)}_RR{int(rr_ratio*10)}_"
                  f"AG{int(model_agreement.get('agreement', 1)*100)}")
        
        # Execute trade
        result = self.order_executor.execute_trade(
            symbol=Config.SYMBOL,
            order_type=order_type,
            volume=volume,
            entry_price=optimal_entry,
            sl=sl,
            tp=tp,
            magic=Config.MAGIC_NUMBER,
            comment=comment
        )
        
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            return False
        
        # Record trade
        trade_data = {
            'ticket': result.order,
            'signal': int(signal),
            'confidence': float(confidence),
            'model_agreement': model_agreement,
            'open_price': result.price,
            'sl': sl,
            'tp': tp,
            'rr_ratio': rr_ratio,
            'volume': volume,
            'open_time': int(datetime.now().timestamp()),
            'features': features,
            'comment': comment,
            'atr': atr,
            'optimal_entry': optimal_entry
        }
        
        self.trade_memory.add_trade(trade_data)
        self.active_positions[result.order] = trade_data
        
        # Log trade execution
        signal_type = "BUY" if signal == 1 else "SELL"
        Logger.log("=" * 50, "TRADE")
        Logger.log(f"✅ {signal_type} EXECUTED | Ticket #{result.order}", "TRADE")
        Logger.log(f"  Entry: {result.price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}", "TRADE")
        Logger.log(f"  Volume: {volume:.3f} | R:R: {rr_ratio:.2f}", "TRADE")
        Logger.log(f"  Confidence: {confidence:.1%} | Agreement: {model_agreement.get('agreement', 1):.0%}", "TRADE")
        Logger.log("=" * 50, "TRADE")
        
        return True
    
    def check_pending_signals(self, df_current):
        """Check and execute pending signals"""
        if not self.pending_signals:
            return
        
        current_price = df_current.iloc[-1]['close']
        signals_to_remove = []
        
        for signal_id, pending in self.pending_signals.items():
            # Check signal age
            age = (datetime.now() - pending['created_at']).total_seconds() / 3600
            if age > 1:  # Expire after 1 hour
                signals_to_remove.append(signal_id)
                Logger.log(f"Pending signal expired (age: {age:.1f}h)", "WARNING")
                continue
            
            # Check if price reached optimal entry
            optimal_entry = pending['optimal_entry']
            distance = abs(current_price - optimal_entry)
            tolerance = pending['atr'] * 0.2
            
            if distance <= tolerance:
                Logger.log(f"🎯 Optimal entry reached! Executing pending signal...", "SUCCESS")
                
                # Execute the trade
                success = self.execute_enhanced_trade(
                    pending['signal'],
                    pending['confidence'],
                    df_current,
                    pending['features'],
                    pending['model_agreement']
                )
                
                if success:
                    signals_to_remove.append(signal_id)
        
        # Clean up executed/expired signals
        for signal_id in signals_to_remove:
            del self.pending_signals[signal_id]
    
    def check_and_manage_positions(self):
        """Check and manage open positions"""
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        if not positions:
            return
        
        for position in positions:
            if position.magic != Config.MAGIC_NUMBER:
                continue
            
            # Check if position needs management
            current_price = position.price_current
            open_price = position.price_open
            sl = position.sl
            tp = position.tp
            
            # Move stop loss to breakeven if in profit
            if sl != 0 and tp != 0:
                profit_pips = abs(current_price - open_price) / 0.01
                
                # Move SL to breakeven after 1x risk
                if profit_pips > abs(open_price - sl) / 0.01:
                    new_sl = open_price
                    
                    # Only modify if different
                    if abs(new_sl - sl) > 0.01:
                        request = {
                            "action": mt5.TRADE_ACTION_SLTP,
                            "symbol": Config.SYMBOL,
                            "sl": new_sl,
                            "tp": tp,
                            "position": position.ticket
                        }
                        
                        result = mt5.order_send(request)
                        if result.retcode == mt5.TRADE_RETCODE_DONE:
                            Logger.log(f"Position #{position.ticket} SL moved to breakeven", "TRADE")
    
    def run_periodic_tasks(self):
        """Run periodic maintenance tasks"""
        self.iteration += 1
        
        # Check closed positions
        self.check_closed_positions()
        
        # Check pending orders
        self.order_executor.check_pending_orders()
        
        # Check and manage open positions
        self.check_and_manage_positions()
        
        # Check pending signals
        if self.pending_signals:
            df = self.get_historical_data(bars=100)
            if df is not None:
                self.check_pending_signals(df)
        
        # Performance reporting (every 100 iterations or 1 hour)
        if (self.iteration % 100 == 0 or 
            (datetime.now() - self.last_performance_report).total_seconds() > 3600):
            
            self.performance_monitor.print_performance_report()
            self.last_performance_report = datetime.now()
            
            # Log risk status
            account = mt5.account_info()
            if account:
                daily_pnl_pct = (self.risk_manager.daily_pnl / account.balance) * 100
                Logger.log(f"Risk Status | Daily P/L: {daily_pnl_pct:.1f}% | "
                          f"Consecutive Losses: {self.risk_manager.consecutive_losses}", "RISK")
        
        # Model retraining check
        if self.model.should_retrain():
            Logger.log("🔄 Retraining ensemble model...", "LEARN")
            df = self.get_historical_data()
            if df is not None:
                success = self.model.train(df)
                if success:
                    Logger.log("✅ Model retraining successful", "SUCCESS")
                else:
                    Logger.log("❌ Model retraining failed", "ERROR")
    
    def run(self):
        """Main trading loop"""
        print("\n" + "=" * 70)
        print("🤖 ENHANCED MT5 AI TRADING SYSTEM v4.0")
        print("📊 Advanced Ensemble | Risk Management | Multi-Timeframe")
        print("=" * 70 + "\n")
        
        if not self.connect_mt5():
            return
        
        # Initial data fetch and model training
        df = self.get_historical_data()
        if df is None:
            Logger.log("Cannot proceed without historical data", "ERROR")
            return
        
        if not self.model.train(df):
            Logger.log("Initial model training failed", "ERROR")
            return
        
        Logger.log(
            f"🎯 Trading {Config.SYMBOL} {Config.TIMEFRAMES} | "
            f"Min Confidence: {Config.MIN_CONFIDENCE:.0%} | "
            f"Min Agreement: {Config.MIN_ENSEMBLE_AGREEMENT:.0%}", "INFO"
        )
        
        try:
            while True:
                # Run periodic tasks
                self.run_periodic_tasks()
                
                # Get current market data
                rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, 100)
                if rates is None or len(rates) < 50:
                    time.sleep(60)
                    continue
                
                df_current = pd.DataFrame(rates)
                
                # Multi-timeframe analysis
                mtf_data = {}
                if Config.MULTI_TIMEFRAME_ENABLED:
                    mtf_data = self.mtf_analyzer.get_multi_timeframe_data()
                
                # Get prediction from model
                signal, confidence, features, model_details = self.model.predict(df_current)
                
                if signal is None:
                    time.sleep(60)
                    continue
                
                # Multi-timeframe alignment check
                mtf_signal = None
                mtf_confidence = 0
                
                if mtf_data:
                    mtf_signal, mtf_confidence = self.mtf_analyzer.analyze_timeframe_alignment(
                        self.model, mtf_data
                    )
                
                # Determine final signal
                final_signal = signal
                final_confidence = confidence
                
                if mtf_signal is not None and mtf_confidence > 0:
                    # Combine signals
                    if mtf_signal == signal:
                        # Alignment strengthens signal
                        final_confidence = (confidence + mtf_confidence) / 2
                        Logger.log(f"✅ Multi-timeframe alignment confirmed ({mtf_confidence:.0%})", "SUCCESS")
                    else:
                        # Conflict weakens signal
                        final_confidence = confidence * 0.7
                        Logger.log(f"⚠ Multi-timeframe conflict", "WARNING")
                
                # Calculate model agreement
                model_agreement = {'agreement': 1.0, 'details': model_details}
                if model_details:
                    predictions = [m['prediction'] for m in model_details.values() 
                                 if m['prediction'] != -1]
                    if predictions:
                        agreement = predictions.count(signal) / len(predictions)
                        model_agreement['agreement'] = agreement
                        model_agreement['details'] = model_details
                
                # Log current status
                current_price = df_current.iloc[-1]['close']
                positions = self.get_current_positions()
                signal_type = "BUY" if final_signal == 1 else "SELL"
                
                status_msg = (f"Price: {current_price:.2f} | Signal: {signal_type} | "
                             f"Conf: {final_confidence:.1%}")
                
                if model_agreement['agreement'] < 1.0:
                    status_msg += f" | Agreement: {model_agreement['agreement']:.0%}"
                
                if mtf_confidence > 0:
                    status_msg += f" | MTF: {mtf_confidence:.0%}"
                
                status_msg += f" | Positions: {positions}"
                
                if self.iteration % 10 == 0:  # Log every 10 iterations
                    Logger.log(status_msg, "INFO")
                
                # Check if we should execute
                if (final_confidence >= Config.MIN_CONFIDENCE and 
                    model_agreement['agreement'] >= Config.MIN_ENSEMBLE_AGREEMENT):
                    
                    Logger.log(f"🎯 High-confidence {signal_type} signal detected!", "SUCCESS")
                    
                    # Execute trade
                    self.execute_enhanced_trade(
                        final_signal, 
                        final_confidence, 
                        df_current, 
                        features, 
                        model_agreement
                    )
                
                # Sleep before next iteration
                time.sleep(60)
                
        except KeyboardInterrupt:
            Logger.log("\nShutdown requested by user", "WARNING")
        except Exception as e:
            Logger.log(f"Unexpected error: {str(e)}", "ERROR")
            import traceback
            traceback.print_exc()
        finally:
            # Final performance report
            self.performance_monitor.print_performance_report()
            
            # Disconnect from MT5
            mt5.shutdown()
            Logger.log("Disconnected from MT5", "INFO")

# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    """Main entry point"""
    engine = EnhancedTradingEngine()
    engine.run()

if __name__ == "__main__":
    main()

# dSEEK