import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datasets import load_dataset
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
    
    # Dataset Configuration
    USE_HISTORICAL_DATASET = False
    HISTORICAL_DATA_LIMIT = 5000
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
    
    # Monte Carlo Parameters
    MC_SIMULATIONS = 1000
    MC_CONFIDENCE = 0.90
    
    # Statistical Testing
    MIN_SHARPE_RATIO = 0.5
    MIN_PROFIT_FACTOR = 1.3
    MAX_DRAWDOWN_DURATION = 20

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
            'STATISTICS': ProfessionalLogger.COLORS['BLUE']
        }
        color = colors.get(level, ProfessionalLogger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level:8s}{ProfessionalLogger.COLORS['RESET']}] [{component:12s}] {message}", flush=True)

# ==========================================
# PROFESSIONAL STATISTICAL ANALYZER
# ==========================================
class StatisticalAnalyzer:
    """Advanced statistical analysis for financial time series"""
    
    @staticmethod
    def analyze_return_distribution(returns):
        """Comprehensive return distribution analysis"""
        stats_dict = {
            'mean': np.mean(returns),
            'std': np.std(returns),
            'skewness': skew(returns),
            'kurtosis': kurtosis(returns),
            'jarque_bera_stat': jarque_bera(returns)[0],
            'jarque_bera_pvalue': jarque_bera(returns)[1],
            'min': np.min(returns),
            'max': np.max(returns),
            'median': np.median(returns),
            'mad': np.median(np.abs(returns - np.median(returns))),
            'iqr': np.percentile(returns, 75) - np.percentile(returns, 25)
        }
        
        # Tail analysis
        left_tail = returns[returns < np.percentile(returns, 5)]
        right_tail = returns[returns > np.percentile(returns, 95)]
        
        stats_dict['left_tail_mean'] = np.mean(left_tail) if len(left_tail) > 0 else 0
        stats_dict['right_tail_mean'] = np.mean(right_tail) if len(right_tail) > 0 else 0
        stats_dict['tail_ratio'] = abs(stats_dict['left_tail_mean'] / stats_dict['right_tail_mean']) \
            if stats_dict['right_tail_mean'] != 0 else float('inf')
        
        return stats_dict
    
    @staticmethod
    def calculate_hurst_exponent(price_series, window=None):
        """Calculate Hurst exponent for market efficiency analysis"""
        if window:
            series = price_series[-window:]
        else:
            series = price_series
            
        lags = range(2, 100)
        tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
        
        reg = np.polyfit(np.log(lags), np.log(tau), 1)
        return reg[0]
    
    @staticmethod
    def estimate_tail_index(returns, confidence=0.95):
        """Estimate tail index using Hill estimator"""
        sorted_returns = np.sort(returns)
        k = int(len(returns) * (1 - confidence))
        
        if k < 10:
            return None
            
        tail_returns = sorted_returns[-k:]
        hill_estimator = 1 / np.mean(np.log(tail_returns / tail_returns[0]))
        
        return hill_estimator
    
    @staticmethod
    def calculate_garch_volatility(returns, p=1, q=1):
        """Calculate GARCH volatility"""
        try:
            am = arch_model(returns * 100, vol='Garch', p=p, q=q, dist='ged')
            res = am.fit(disp='off')
            return res.conditional_volatility[-1] / 100
        except:
            return np.std(returns)
    
    @staticmethod
    def calculate_autocorrelation(series, lags=20):
        """Calculate autocorrelation with confidence intervals"""
        acf = []
        conf_int = []
        n = len(series)
        
        for lag in range(1, lags + 1):
            corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            acf.append(corr)
            conf_int.append(1.96 / np.sqrt(n - lag))
        
        return np.array(acf), np.array(conf_int)

# ==========================================
# PROFESSIONAL RISK METRICS
# ==========================================
class ProfessionalRiskMetrics:
    """Advanced risk metrics for professional trading"""
    
    @staticmethod
    def calculate_var(returns, confidence=0.95, method='historical'):
        """Calculate Value at Risk using multiple methods"""
        if method == 'historical':
            return np.percentile(returns, (1 - confidence) * 100)
        elif method == 'parametric':
            mu = np.mean(returns)
            sigma = np.std(returns)
            return norm.ppf(1 - confidence, mu, sigma)
        elif method == 'cornish_fisher':
            mu = np.mean(returns)
            sigma = np.std(returns)
            skewness = skew(returns)
            kurt = kurtosis(returns)
            
            z = norm.ppf(1 - confidence)
            z_cf = (z + 
                   (z**2 - 1) * skewness / 6 +
                   (z**3 - 3*z) * (kurt - 3) / 24 -
                   (2*z**3 - 5*z) * skewness**2 / 36)
            
            return mu + sigma * z_cf
    
    @staticmethod
    def calculate_cvar(returns, confidence=0.95):
        """Calculate Conditional VaR (Expected Shortfall)"""
        var = ProfessionalRiskMetrics.calculate_var(returns, confidence)
        cvar = returns[returns <= var].mean()
        return cvar if not np.isnan(cvar) else var
    
    @staticmethod
    def calculate_maximum_drawdown(equity_curve):
        """Calculate maximum drawdown with duration"""
        cumulative = np.array(equity_curve)
        peak = np.maximum.accumulate(cumulative)
        drawdown = (peak - cumulative) / peak
        
        max_dd = np.max(drawdown)
        max_dd_idx = np.argmax(drawdown)
        
        peak_idx = np.argmax(cumulative[:max_dd_idx])
        duration = max_dd_idx - peak_idx
        
        return max_dd, duration, peak_idx, max_dd_idx
    
    @staticmethod
    def calculate_ulcer_index(equity_curve, period=14):
        """Calculate Ulcer Index - measures downside volatility"""
        returns = np.diff(equity_curve) / equity_curve[:-1]
        downside_returns = np.where(returns < 0, returns**2, 0)
        
        if len(downside_returns) < period:
            return np.sqrt(np.mean(downside_returns))
        
        ui_values = []
        for i in range(period, len(downside_returns)):
            ui = np.sqrt(np.mean(downside_returns[i-period:i]))
            ui_values.append(ui)
        
        return np.mean(ui_values) if ui_values else 0
    
    @staticmethod
    def calculate_omega_ratio(returns, threshold=0):
        """Calculate Omega Ratio - considers entire distribution"""
        gains = returns[returns > threshold] - threshold
        losses = threshold - returns[returns <= threshold]
        
        if len(losses) == 0:
            return float('inf')
        
        return np.sum(gains) / np.sum(losses)
    
    @staticmethod
    def calculate_calmar_ratio(returns, equity_curve, period=252):
        """Calculate Calmar Ratio (return / max drawdown)"""
        annual_return = np.mean(returns) * period
        max_dd, _, _, _ = ProfessionalRiskMetrics.calculate_maximum_drawdown(equity_curve)
        
        if max_dd == 0:
            return float('inf')
        
        return annual_return / max_dd

# ==========================================
# DATA QUALITY CHECKER (ENHANCED)
# ==========================================
class ProfessionalDataQualityChecker:
    """Enhanced data quality validation with statistical tests"""
    
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
            if df[col].std() == 0:
                variance_score *= 0.3
            elif df[col].nunique() < 10:
                variance_score *= 0.7
        scores.append(variance_score)
        
        # Outlier detection using multiple methods
        outlier_score = 1.0
        price_cols = ['close', 'open', 'high', 'low']
        for col in price_cols:
            if col in df.columns:
                # Method 1: IQR
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                outliers_iqr = ((df[col] < (q1 - 3*iqr)) | (df[col] > (q3 + 3*iqr))).sum()
                
                # Method 2: Z-score
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers_z = (z_scores > 3).sum()
                
                # Use the worst case
                outliers = max(outliers_iqr, outliers_z)
                outlier_ratio = outliers / len(df)
                outlier_score *= max(0.5, 1 - outlier_ratio * 2)
        scores.append(outlier_score)
        
        # Chronological order
        if 'time' in df.columns:
            is_sorted = df['time'].is_monotonic_increasing
            scores.append(1.0 if is_sorted else 0.3)
        
        # Data freshness
        if 'time' in df.columns:
            latest_time = pd.to_datetime(df['time'].max(), unit='s')
            days_old = (datetime.now() - latest_time).days
            freshness_score = max(0, 1 - days_old / 30)
            scores.append(freshness_score)
        
        # Statistical sanity checks
        stat_score = 1.0
        if 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            if len(returns) > 50:
                # Check for stationarity (simplified)
                if np.abs(returns.mean()) > 0.001:
                    stat_score *= 0.8
                
                # Check for autocorrelation
                autocorr = returns.autocorr(lag=1)
                if np.abs(autocorr) > 0.1:
                    stat_score *= 0.9
        scores.append(stat_score)
        
        overall_score = np.mean(scores)
        return overall_score, scores

# ==========================================
# TRADE MEMORY SYSTEM (ENHANCED)
# ==========================================
class ProfessionalTradeMemory:
    """Enhanced trade memory with statistical tracking and fault tolerance"""
    
    def __init__(self, history_file=None, memory_size=None):
        self.history_file = history_file or Config.TRADE_HISTORY_FILE
        self.memory_size = memory_size or Config.MEMORY_SIZE
        self.trades = []
        self.stat_analyzer = StatisticalAnalyzer()
        self.risk_metrics = ProfessionalRiskMetrics()
        self._initialize_memory()
        
    def _initialize_memory(self):
        """Initialize trade memory with validation"""
        try:
            if os.path.exists(self.history_file):
                self.trades = self.load_history()
                ProfessionalLogger.log(f"Loaded {len(self.trades)} trades from {self.history_file}", 
                                     "DATA", "MEMORY")
            else:
                ProfessionalLogger.log(f"No existing trade history found. Starting fresh.", 
                                     "INFO", "MEMORY")
                self.trades = []
                
            # Validate loaded trades
            self._validate_trades()
            
        except Exception as e:
            ProfessionalLogger.log(f"Memory initialization error: {str(e)}", "ERROR", "MEMORY")
            self.trades = []
            
    def load_history(self):
        """Load trade history from file with robust error handling"""
        try:
            with open(self.history_file, 'r') as f:
                raw_data = json.load(f)
                
            # Validate and sanitize loaded data
            validated_trades = []
            for i, trade in enumerate(raw_data):
                try:
                    validated = self._validate_trade_structure(trade)
                    if validated:
                        validated_trades.append(validated)
                    else:
                        ProfessionalLogger.log(f"Skipping invalid trade at index {i}", 
                                             "WARNING", "MEMORY")
                except Exception as e:
                    ProfessionalLogger.log(f"Trade validation error at index {i}: {str(e)}", 
                                         "WARNING", "MEMORY")
                    
            return validated_trades
            
        except json.JSONDecodeError as e:
            ProfessionalLogger.log(f"JSON decode error in {self.history_file}: {str(e)}", 
                                 "ERROR", "MEMORY")
            return []
        except Exception as e:
            ProfessionalLogger.log(f"Unexpected error loading history: {str(e)}", 
                                 "ERROR", "MEMORY")
            return []
            
    def _validate_trade_structure(self, trade):
        """Validate and normalize trade structure"""
        required_fields = ['id', 'timestamp', 'signal', 'open_price']
        
        # Check required fields
        for field in required_fields:
            if field not in trade:
                ProfessionalLogger.log(f"Missing required field '{field}' in trade", 
                                     "WARNING", "MEMORY")
                return None
                
        # Ensure correct data types
        normalized = trade.copy()
        
        # Ensure numeric fields
        numeric_fields = ['id', 'signal', 'open_price', 'sl', 'tp', 'volume', 
                         'profit', 'rr_achieved', 'expectancy']
        for field in numeric_fields:
            if field in normalized and normalized[field] is not None:
                try:
                    normalized[field] = float(normalized[field])
                except (ValueError, TypeError):
                    normalized[field] = 0.0 if field in ['profit', 'rr_achieved', 'expectancy'] else None
                    
        # Ensure timestamp format
        if 'timestamp' in normalized:
            try:
                # Try to parse timestamp
                datetime.fromisoformat(normalized['timestamp'].replace('Z', '+00:00'))
            except (ValueError, AttributeError):
                normalized['timestamp'] = datetime.now().isoformat()
                
        # Add missing optional fields with defaults
        defaults = {
            'confidence': 0.0,
            'volume': 0.1,
            'risk_amount': 0.0,
            'reward_amount': 0.0,
            'duration_seconds': 0,
            'model_agreement': 1.0,
            'atr': 0.0,
            'features': {}
        }
        
        for key, default in defaults.items():
            if key not in normalized:
                normalized[key] = default
                
        return normalized
        
    def _validate_trades(self):
        """Validate all trades in memory"""
        original_count = len(self.trades)
        validated_trades = []
        
        for trade in self.trades:
            validated = self._validate_trade_structure(trade)
            if validated:
                validated_trades.append(validated)
                
        self.trades = validated_trades
        
        if original_count != len(self.trades):
            ProfessionalLogger.log(f"Filtered {original_count - len(self.trades)} invalid trades", 
                                 "WARNING", "MEMORY")
            
    def save_history(self):
        """Save trade history with atomic write and backup"""
        if not self.trades:
            ProfessionalLogger.log("No trades to save", "INFO", "MEMORY")
            return
            
        try:
            # Create backup if file exists
            if os.path.exists(self.history_file):
                backup_file = f"{self.history_file}.backup.{int(datetime.now().timestamp())}"
                shutil.copy2(self.history_file, backup_file)
                
            # Apply memory size limit
            if len(self.trades) > self.memory_size:
                self.trades = self.trades[-self.memory_size:]
                ProfessionalLogger.log(f"Trimmed history to {self.memory_size} most recent trades", 
                                     "INFO", "MEMORY")
                
            # Write to temporary file first
            temp_file = f"{self.history_file}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(self.trades, f, indent=2, default=self._json_serializer)
                
            # Atomic move to final location
            shutil.move(temp_file, self.history_file)
            
            ProfessionalLogger.log(f"Saved {len(self.trades)} trades to {self.history_file}", 
                                 "DATA", "MEMORY")
                
        except Exception as e:
            ProfessionalLogger.log(f"Error saving history: {str(e)}", "ERROR", "MEMORY")
            
    def _json_serializer(self, obj):
        """Custom JSON serializer for non-serializable objects"""
        if isinstance(obj, (datetime, pd.Timestamp)):
            return obj.isoformat()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
    def add_trade(self, trade_data):
        """Add new trade with comprehensive validation"""
        try:
            # Generate unique ID
            trade_data['id'] = self._generate_trade_id()
            trade_data['timestamp'] = datetime.now().isoformat()
            
            # Validate and normalize
            validated = self._validate_trade_structure(trade_data)
            if not validated:
                ProfessionalLogger.log("Failed to validate trade data", "ERROR", "MEMORY")
                return False
                
            # Check for duplicates
            if self._is_duplicate_trade(validated):
                ProfessionalLogger.log("Duplicate trade detected, skipping", "WARNING", "MEMORY")
                return False
                
            # Calculate additional metrics if not provided
            self._enrich_trade_data(validated)
            
            self.trades.append(validated)
            self.save_history()
            
            ProfessionalLogger.log(f"Trade #{validated['id']} recorded | Signal: {validated['signal']} | "
                                 f"Price: {validated['open_price']:.2f}", 
                                 "TRADE", "MEMORY")
            return True
            
        except Exception as e:
            ProfessionalLogger.log(f"Error adding trade: {str(e)}", "ERROR", "MEMORY")
            return False
            
    def _generate_trade_id(self):
        """Generate unique trade ID"""
        if not self.trades:
            return 1
        return max(trade.get('id', 0) for trade in self.trades) + 1
        
    def _is_duplicate_trade(self, trade):
        """Check for duplicate trades based on timestamp and price"""
        if not self.trades:
            return False
            
        recent_trades = self.trades[-10:]  # Check last 10 trades
        for existing in recent_trades:
            # Check if same timestamp (within 60 seconds)
            try:
                existing_time = datetime.fromisoformat(existing['timestamp'].replace('Z', '+00:00'))
                new_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                
                time_diff = abs((new_time - existing_time).total_seconds())
                price_diff = abs(trade['open_price'] - existing.get('open_price', 0))
                
                if time_diff < 60 and price_diff < 0.1:  # Same minute and similar price
                    return True
            except:
                continue
                
        return False
        
    def _enrich_trade_data(self, trade):
        """Enrich trade data with calculated metrics"""
        # Calculate risk/reward if not provided
        if 'sl' in trade and 'tp' in trade and 'open_price' in trade:
            risk = abs(trade['open_price'] - trade['sl'])
            reward = abs(trade['tp'] - trade['open_price'])
            
            if risk > 0:
                trade['risk_amount'] = risk
                trade['reward_amount'] = reward
                trade['planned_rr'] = reward / risk if risk > 0 else 0
                
        # Calculate position size metrics
        if 'volume' in trade and 'open_price' in trade:
            # Assuming standard gold contract: 100 oz per standard lot
            trade['position_value'] = trade['volume'] * trade['open_price'] * 100
            
    def update_trade_outcome(self, ticket, outcome_data):
        """Update trade with outcome data (closing details)"""
        try:
            trade = self.get_trade_by_ticket(ticket)
            if not trade:
                ProfessionalLogger.log(f"Trade #{ticket} not found for update", "WARNING", "MEMORY")
                return False
                
            # Validate outcome data
            required_outcome_fields = ['profit', 'close_price']
            for field in required_outcome_fields:
                if field not in outcome_data:
                    ProfessionalLogger.log(f"Missing {field} in outcome data", "ERROR", "MEMORY")
                    return False
                    
            # Calculate duration if not provided
            if 'duration_seconds' not in outcome_data:
                try:
                    open_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                    close_time = datetime.now()
                    outcome_data['duration_seconds'] = (close_time - open_time).total_seconds()
                except:
                    outcome_data['duration_seconds'] = 0
                    
            # Update trade with outcome
            trade.update(outcome_data)
            trade['close_time'] = datetime.now().isoformat()
            trade['status'] = 'closed'
            
            # Calculate realized metrics
            self._calculate_realized_metrics(trade)
            
            self.save_history()
            
            # Log outcome
            profit = outcome_data['profit']
            result = "WIN" if profit > 0 else "LOSS"
            duration_min = outcome_data['duration_seconds'] / 60
            
            ProfessionalLogger.log(
                f"Trade #{ticket} {result} | P/L: ${profit:.2f} | "
                f"Duration: {duration_min:.1f}min | "
                f"R/R Achieved: {trade.get('rr_achieved', 0):.2f}",
                "SUCCESS" if profit > 0 else "WARNING", "MEMORY"
            )
            
            return True
            
        except Exception as e:
            ProfessionalLogger.log(f"Error updating trade outcome: {str(e)}", "ERROR", "MEMORY")
            return False
            
    def _calculate_realized_metrics(self, trade):
        """Calculate realized metrics after trade closure"""
        if 'profit' not in trade or 'open_price' not in trade:
            return
            
        profit = trade['profit']
        
        # Calculate realized R/R
        if 'sl' in trade and trade['sl']:
            risk = abs(trade['open_price'] - trade['sl'])
            if risk > 0:
                trade['rr_achieved'] = abs(profit) / risk
                
        # Calculate expectancy
        if 'risk_amount' in trade and trade['risk_amount'] > 0:
            trade['expectancy'] = profit / trade['risk_amount']
            
        # Calculate additional performance metrics
        trade['profit_percent'] = (profit / trade.get('position_value', 1)) * 100 if trade.get('position_value', 0) > 0 else 0
        
    def get_trade_by_ticket(self, ticket):
        """Get trade by ticket number"""
        for trade in self.trades:
            if trade.get('ticket') == ticket:
                return trade
        return None
        
    def get_completed_trades(self, min_profit_filter=None, date_range=None):
        """Get completed trades with optional filtering"""
        completed = [t for t in self.trades if t.get('status') == 'closed' and 'profit' in t]
        
        # Apply filters
        if min_profit_filter is not None:
            completed = [t for t in completed if abs(t.get('profit', 0)) >= min_profit_filter]
            
        if date_range:
            start_date, end_date = date_range
            filtered = []
            for trade in completed:
                try:
                    trade_time = datetime.fromisoformat(trade.get('close_time', trade['timestamp']).replace('Z', '+00:00'))
                    if start_date <= trade_time <= end_date:
                        filtered.append(trade)
                except:
                    continue
            completed = filtered
            
        return completed
        
    def get_open_trades(self):
        """Get currently open trades"""
        return [t for t in self.trades if t.get('status') != 'closed']
        
    def get_statistical_summary(self, period_days=None, include_advanced=True):
        """Get comprehensive statistical summary of trades"""
        completed = self.get_completed_trades()
        
        if period_days:
            cutoff_date = datetime.now() - timedelta(days=period_days)
            completed = [t for t in completed if 
                        datetime.fromisoformat(t.get('close_time', '2000-01-01').replace('Z', '+00:00')) > cutoff_date]
        
        if not completed:
            return {
                'total_trades': 0,
                'message': 'No completed trades in specified period',
                'period_days': period_days
            }
        
        # Extract metrics
        profits = np.array([t.get('profit', 0) for t in completed])
        durations = np.array([t.get('duration_seconds', 0) for t in completed])
        volumes = np.array([t.get('volume', 0.1) for t in completed])
        
        # Basic statistics
        stats = {
            'total_trades': len(completed),
            'winning_trades': sum(1 for p in profits if p > 0),
            'losing_trades': sum(1 for p in profits if p <= 0),
            'win_rate': np.mean(profits > 0),
            'total_profit': profits.sum(),
            'mean_profit': profits.mean(),
            'median_profit': np.median(profits),
            'std_profit': profits.std(),
            'min_profit': profits.min(),
            'max_profit': profits.max(),
            'profit_factor': abs(profits[profits > 0].sum() / profits[profits <= 0].sum()) 
                           if profits[profits <= 0].sum() != 0 else float('inf'),
            'avg_trade_duration_minutes': np.mean(durations) / 60 if len(durations) > 0 else 0,
            'avg_volume': np.mean(volumes),
            'period_days': period_days,
            'date_range': {
                'start': min(t.get('timestamp') for t in completed),
                'end': max(t.get('close_time', t['timestamp']) for t in completed)
            }
        }
        
        # Advanced statistics
        if include_advanced and len(profits) >= 10:
            # Return distribution
            dist_stats = self.stat_analyzer.analyze_return_distribution(profits)
            stats.update({f'dist_{k}': v for k, v in dist_stats.items()})
            
            # Risk metrics
            stats['var_95'] = self.risk_metrics.calculate_var(profits, 0.95)
            stats['cvar_95'] = self.risk_metrics.calculate_cvar(profits, 0.95)
            stats['omega_ratio'] = self.risk_metrics.calculate_omega_ratio(profits)
            
            # Performance metrics
            if len(profits) >= 20:
                equity_curve = np.cumsum(profits)
                max_dd, dd_duration, _, _ = self.risk_metrics.calculate_maximum_drawdown(equity_curve)
                stats['max_drawdown'] = max_dd
                stats['max_drawdown_duration'] = dd_duration
                stats['recovery_factor'] = abs(profits.sum() / max_dd) if max_dd != 0 else float('inf')
                
            # Time-based analysis
            hourly_profits = self._analyze_by_time_of_day(completed)
            stats['best_hour'] = max(hourly_profits.items(), key=lambda x: x[1])[0] if hourly_profits else None
            stats['worst_hour'] = min(hourly_profits.items(), key=lambda x: x[1])[0] if hourly_profits else None
            
        return stats
        
    def _analyze_by_time_of_day(self, trades):
        """Analyze performance by hour of day"""
        hourly_profits = {}
        
        for trade in trades:
            try:
                trade_time = datetime.fromisoformat(trade['timestamp'].replace('Z', '+00:00'))
                hour = trade_time.hour
                profit = trade.get('profit', 0)
                
                if hour not in hourly_profits:
                    hourly_profits[hour] = {'total': 0, 'count': 0}
                    
                hourly_profits[hour]['total'] += profit
                hourly_profits[hour]['count'] += 1
            except:
                continue
                
        # Convert to average profit per hour
        return {hour: data['total'] / data['count'] for hour, data in hourly_profits.items() 
                if data['count'] > 0}
                
    def get_performance_trend(self, window_size=20):
        """Get performance trend over time"""
        completed = self.get_completed_trades()
        if len(completed) < window_size:
            return None
            
        # Sort by close time
        completed.sort(key=lambda x: x.get('close_time', x['timestamp']))
        
        rolling_profits = []
        rolling_wins = []
        
        for i in range(window_size, len(completed)):
            window = completed[i-window_size:i]
            window_profits = [t.get('profit', 0) for t in window]
            rolling_profits.append(sum(window_profits))
            rolling_wins.append(sum(1 for p in window_profits if p > 0) / len(window_profits))
            
        return {
            'rolling_profits': rolling_profits,
            'rolling_win_rates': rolling_wins,
            'trend_slope': np.polyfit(range(len(rolling_profits)), rolling_profits, 1)[0] 
                          if rolling_profits else 0
        }
        
    def clear_history(self, confirm=False):
        """Clear trade history (with confirmation)"""
        if not confirm:
            ProfessionalLogger.log("Clear history requires confirmation", "WARNING", "MEMORY")
            return False
            
        try:
            self.trades = []
            if os.path.exists(self.history_file):
                os.remove(self.history_file)
            ProfessionalLogger.log("Trade history cleared", "INFO", "MEMORY")
            return True
        except Exception as e:
            ProfessionalLogger.log(f"Error clearing history: {str(e)}", "ERROR", "MEMORY")
            return False
            
    def export_to_dataframe(self):
        """Export trade history to pandas DataFrame"""
        if not self.trades:
            return pd.DataFrame()
            
        return pd.DataFrame(self.trades)
        
    def backup_history(self, backup_path=None):
        """Create backup of trade history"""
        try:
            if not self.history_file or not os.path.exists(self.history_file):
                ProfessionalLogger.log("No history file to backup", "WARNING", "MEMORY")
                return False
                
            backup_path = backup_path or f"{self.history_file}.backup.{int(datetime.now().timestamp())}"
            shutil.copy2(self.history_file, backup_path)
            ProfessionalLogger.log(f"Backup created: {backup_path}", "INFO", "MEMORY")
            return True
            
        except Exception as e:
            ProfessionalLogger.log(f"Error creating backup: {str(e)}", "ERROR", "MEMORY")
            return False
# ==========================================
# ADVANCED FEATURE ENGINEERING (PROFESSIONAL)
# ==========================================
class ProfessionalFeatureEngine:
    def __init__(self):
        self.scaler = RobustScaler()
        self.stat_analyzer = StatisticalAnalyzer()
        
    def calculate_features(self, df):
        """Calculate features with statistical rigor"""
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
        
        # --- ENHANCED ADX & DIRECTIONAL SYSTEM ---
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
        
        # Gold seasonal pattern
        df['gold_seasonal'] = df['month'].apply(lambda x: 1 if x in [9, 10, 11, 12] else 0)
        
        # Market session
        df['session'] = df['hour'].apply(self._get_market_session)
        
        # Statistical features
        df['returns_skew_20'] = df['returns'].rolling(20).apply(skew, raw=True)
        df['returns_kurtosis_20'] = df['returns'].rolling(20).apply(kurtosis, raw=True)
        
        # GARCH volatility
        if len(df) > 100:
            try:
                garch_vol = []
                for i in range(100, len(df)):
                    window_returns = df['returns'].iloc[i-100:i].fillna(0)
                    vol = self.stat_analyzer.calculate_garch_volatility(window_returns.values)
                    garch_vol.append(vol)
                df.loc[df.index[100:], 'garch_vol'] = garch_vol
            except:
                df['garch_vol'] = df['volatility']
        
        return df
    
    def calculate_adx(self, df, period=Config.ADX_PERIOD):
        """Calculate Average Directional Index"""
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
        
        tr_s = pd.Series(tr)
        plus_dm_s = pd.Series(plus_dm)
        minus_dm_s = pd.Series(minus_dm)
        
        tr_smooth = tr_s.rolling(period).sum()
        plus_dm_smooth = plus_dm_s.rolling(period).sum()
        minus_dm_smooth = minus_dm_s.rolling(period).sum()
        
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
        """Classify market session based on hour"""
        if 0 <= hour < 8:
            return 0
        elif 8 <= hour < 16:
            return 1
        else:
            return 2
    
    def create_labels(self, df, forward_bars=3, threshold=0.001):
        """Create labels for classification with statistical validation"""
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
            'gold_seasonal', 'session',
            'returns_skew_20', 'returns_kurtosis_20',
            'garch_vol'
        ]
        
        if Config.USE_MARKET_REGIME:
            base_features.extend([
                'trend_strength', 'regime', 'plus_di', 'minus_di', 
                'adx_slope', 'di_spread'
            ])
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
            df[f'volatility_lag_{lag}'] = df['volatility'].shift(lag)
            df[f'rsi_lag_{lag}'] = df['rsi_normalized'].shift(lag)
            if 'tick_volume' in df.columns:
                df[f'volume_lag_{lag}'] = df['tick_volume'].shift(lag) / df['tick_volume'].rolling(20).mean().shift(lag)
        
        return base_features + [f'returns_lag_{lag}' for lag in [1, 2, 3, 5]] + \
               [f'volatility_lag_{lag}' for lag in [1, 2, 3, 5]] + \
               [f'rsi_lag_{lag}' for lag in [1, 2, 3, 5]] + \
               ([f'volume_lag_{lag}' for lag in [1, 2, 3, 5]] if 'tick_volume' in df.columns else [])

# ==========================================
# PRICE ACTION ANALYZER (ENHANCED)
# ==========================================
class ProfessionalPriceActionAnalyzer:
    """Enhanced price action analysis with statistical methods"""
    
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
        """Find support/resistance clusters using KDE"""
        prices = pd.concat([df['high'], df['low'], df['close']]).tail(lookback * 3)
        
        try:
            kde = stats.gaussian_kde(prices)
            x_range = np.linspace(prices.min(), prices.max(), 200)
            density = kde(x_range)
            
            peaks, _ = signal.find_peaks(density, height=np.mean(density) * 1.2)
            
            if len(peaks) > 0:
                levels = x_range[peaks]
                
                support_levels = [l for l in levels if l < current_price]
                resistance_levels = [l for l in levels if l > current_price]
                
                nearest_support = max(support_levels) if support_levels else current_price * 0.99
                nearest_resistance = min(resistance_levels) if resistance_levels else current_price * 1.01
                
                return nearest_support, nearest_resistance, levels
            
        except:
            pass
        
        # Fallback
        swing_highs, swing_lows = ProfessionalPriceActionAnalyzer.find_swing_points(df, lookback)
        
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
    def calculate_pivot_points(df):
        """Calculate standard pivot points"""
        last_bar = df.iloc[-2]
        high = last_bar['high']
        low = last_bar['low']
        close = last_bar['close']
        
        pivot = (high + low + close) / 3
        r1 = (2 * pivot) - low
        s1 = (2 * pivot) - high
        
        return {'pivot': pivot, 'r1': r1, 's1': s1}

    @staticmethod
    def calculate_optimal_entry_sl_tp(df, signal, current_price, atr, risk_reward_ratio=Config.MIN_RR_RATIO):
        """Calculate optimal trade parameters with multiple techniques"""
        
        support, resistance, all_levels = ProfessionalPriceActionAnalyzer.find_support_resistance_clusters(
            df, current_price, Config.LOOKBACK_SWING_POINTS
        )
        
        fib_levels = ProfessionalPriceActionAnalyzer.calculate_fibonacci_levels(df)
        pivots = ProfessionalPriceActionAnalyzer.calculate_pivot_points(df)
        
        if signal == 1:  # BUY Signal
            optimal_entry = current_price
            
            sl_candidates = [
                support - (atr * 0.5),
                current_price - (atr * 1.5),
                fib_levels['0.618'] if current_price > fib_levels['0.618'] else current_price - (atr * 2)
            ]
            
            sl = max(sl_candidates)
            
            tp_candidates = [
                resistance,
                pivots['r1'],
                fib_levels['1.0'],
                current_price + (atr * 3),
                current_price + (abs(current_price - sl) * risk_reward_ratio)
            ]
            
            valid_tps = [tp for tp in tp_candidates if tp > optimal_entry]
            if valid_tps:
                for tp in sorted(valid_tps):
                    risk = optimal_entry - sl
                    reward = tp - optimal_entry
                    if risk > 0 and reward / risk >= risk_reward_ratio:
                        final_tp = tp
                        break
                else:
                    final_tp = optimal_entry + ((optimal_entry - sl) * risk_reward_ratio)
            else:
                final_tp = optimal_entry + (atr * 3)
            
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
            
            sl_candidates = [
                resistance + (atr * 0.5),
                current_price + (atr * 1.5),
                fib_levels['0.382'] if current_price < fib_levels['0.382'] else current_price + (atr * 2)
            ]
            
            sl = min(sl_candidates)
            
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
                    final_tp = optimal_entry - ((sl - optimal_entry) * risk_reward_ratio)
            else:
                final_tp = optimal_entry - (atr * 3)
            
            if Config.USE_SMART_ENTRY:
                retracement_targets = [
                    resistance - (atr * 0.3),
                    fib_levels['0.382'] if fib_levels['0.382'] > current_price else None,
                    pivots['pivot']
                ]
                
                valid_targets = [t for t in retracement_targets if t is not None and t > current_price]
                if valid_targets:
                    optimal_entry = min(valid_targets)
        
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
# MULTI-TIMEFRAME ANALYZER (ENHANCED)
# ==========================================
class ProfessionalMultiTimeframeAnalyzer:
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
                    ProfessionalLogger.log(f"Could not fetch {tf_name} data", "WARNING", "MTF")
        
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
        
        buy_signals = sum(1 for s in signals.values() if s == 1)
        sell_signals = sum(1 for s in signals.values() if s == 0)
        total = len(signals)
        
        if total == 0:
            return None, 0.0
        
        alignment_ratio = max(buy_signals, sell_signals) / total
        avg_confidence = np.mean(list(confidences.values())) if confidences else 0
        
        if buy_signals / total >= Config.TIMEFRAME_ALIGNMENT_THRESHOLD:
            return 1, alignment_ratio * avg_confidence
        elif sell_signals / total >= Config.TIMEFRAME_ALIGNMENT_THRESHOLD:
            return 0, alignment_ratio * avg_confidence
        
        return None, alignment_ratio * avg_confidence

# ==========================================
# PROFESSIONAL BACKTESTER
# ==========================================
class ProfessionalBacktester:
    """Professional backtesting framework"""
    
    def __init__(self, model, feature_engine):
        self.model = model
        self.feature_engine = feature_engine
        self.results = []
        
    def walk_forward_test(self, data, train_size, test_size, step_size=None):
        """Walk-forward testing with realistic assumptions"""
        if step_size is None:
            step_size = test_size
        
        total_bars = len(data)
        results = []
        
        for start in range(0, total_bars - train_size - test_size, step_size):
            train_end = start + train_size
            test_end = min(train_end + test_size, total_bars)
            
            train_data = data.iloc[start:train_end]
            test_data = data.iloc[train_end:test_end]
            
            ProfessionalLogger.log(f"Training on bars {start}:{train_end}", "BACKTEST", "BACKTESTER")
            
            # Simulate training (would use model.train() in reality)
            test_results = self._simulate_test(test_data)
            test_results['train_period'] = (start, train_end)
            test_results['test_period'] = (train_end, test_end)
            
            results.append(test_results)
            
            ProfessionalLogger.log(
                f"Test Results: Trades={test_results['total_trades']}, "
                f"WinRate={test_results['win_rate']:.1%}, "
                f"Profit=${test_results['total_profit']:.2f}",
                "BACKTEST", "BACKTESTER"
            )
        
        return results
    
    def _simulate_test(self, data):
        """Simulate trading on test data"""
        # Simplified simulation
        trades = []
        position = None
        entry_price = None
        
        for i in range(1, len(data)):
            current_price = data['close'].iloc[i]
            
            # Simple trading logic for simulation
            if data['rsi'].iloc[i] < 30 and position != 'long':
                if position == 'short':
                    # Close short
                    pnl = (entry_price - current_price) * 10000
                    trades.append({'pnl': pnl, 'type': 'short_close'})
                
                # Open long
                position = 'long'
                entry_price = current_price
                
            elif data['rsi'].iloc[i] > 70 and position != 'short':
                if position == 'long':
                    # Close long
                    pnl = (current_price - entry_price) * 10000
                    trades.append({'pnl': pnl, 'type': 'long_close'})
                
                # Open short
                position = 'short'
                entry_price = current_price
        
        # Close final position
        if position:
            last_price = data['close'].iloc[-1]
            if position == 'long':
                pnl = (last_price - entry_price) * 10000
                trades.append({'pnl': pnl, 'type': 'long_close'})
            else:
                pnl = (entry_price - last_price) * 10000
                trades.append({'pnl': pnl, 'type': 'short_close'})
        
        return self._calculate_performance_metrics(trades)
    
    def _calculate_performance_metrics(self, trades):
        """Calculate comprehensive performance metrics"""
        if not trades:
            return {'total_trades': 0, 'total_profit': 0, 'win_rate': 0}
        
        pnls = [t['pnl'] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        total_profit = sum(pnls)
        win_rate = len(wins) / len(pnls)
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        
        equity_curve = np.cumsum(pnls)
        returns = np.diff(equity_curve) / np.abs(equity_curve[:-1]) if len(equity_curve) > 1 else [0]
        
        risk_metrics = ProfessionalRiskMetrics()
        max_dd, dd_duration, _, _ = risk_metrics.calculate_maximum_drawdown(equity_curve)
        
        if len(returns) > 10:
            sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
            var = risk_metrics.calculate_var(returns, Config.VAR_CONFIDENCE)
            cvar = risk_metrics.calculate_cvar(returns, Config.CVAR_CONFIDENCE)
            calmar = risk_metrics.calculate_calmar_ratio(returns, equity_curve)
            omega = risk_metrics.calculate_omega_ratio(returns)
        else:
            sharpe = var = cvar = calmar = omega = 0
        
        return {
            'total_trades': len(trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf'),
            'expectancy': (win_rate * avg_win) - ((1 - win_rate) * abs(avg_loss)),
            'max_drawdown': max_dd,
            'drawdown_duration': dd_duration,
            'sharpe_ratio': sharpe,
            'var_95': var,
            'cvar_95': cvar,
            'calmar_ratio': calmar,
            'omega_ratio': omega
        }
    
    def monte_carlo_simulation(self, trades, n_simulations=1000):
        """Run Monte Carlo simulations"""
        if len(trades) < 20:
            return None
        
        pnls = [t['pnl'] for t in trades]
        mc_results = []
        
        for _ in range(n_simulations):
            sample_idx = np.random.choice(len(pnls), size=len(pnls), replace=True)
            sample_pnls = [pnls[i] for i in sample_idx]
            mc_results.append(sum(sample_pnls))
        
        mc_results = np.array(mc_results)
        
        return {
            'mean': np.mean(mc_results),
            'std': np.std(mc_results),
            'percentile_5': np.percentile(mc_results, 5),
            'percentile_95': np.percentile(mc_results, 95),
            'probability_profit': np.mean(mc_results > 0),
            'probability_ruin': np.mean(mc_results < -Config.MAX_DRAWDOWN_PERCENT * 10000)
        }

# ==========================================
# ADVANCED ML COMPONENTS
# ==========================================
class TripleBarrierLabeling:
    """Advanced Triple Barrier Method for labeling"""
    
    @staticmethod
    def get_events(close, t_events, pt_sl, target, min_ret, vertical_barrier_times=None, side_prediction=None):
        """
        Finds the time of the first barrier touch.
        """
        target = target.loc[t_events]
        target = target[target > min_ret]
        
        if vertical_barrier_times is None:
            vertical_barrier_times = pd.Series(pd.NaT, index=t_events)
            
        out = []
        for loc, t in t_events.items():
            if loc not in close.index: continue
            
            close_subset = close[loc:]
            # Simple approximate barrier approach for speed
            path = close_subset / close_subset.iloc[0] - 1
            
            # Vertical Barrier
            t1 = vertical_barrier_times.get(loc, pd.NaT)
            
            # Top Barrier
            if pt_sl[0] > 0:
                top = pt_sl[0] * target.get(loc, 0)
            else:
                top = pd.Series(index=path.index) # NaNs
                
            # Bottom Barrier
            if pt_sl[1] > 0:
                bottom = -pt_sl[1] * target.get(loc, 0)
            else:
                bottom = pd.Series(index=path.index) # NaNs
                
            # Find earliest touch
            touch_times = pd.concat([
                path[path > top].head(1).index if not isinstance(top, pd.Series) or not top.isna().all() else pd.Index([]),
                path[path < bottom].head(1).index if not isinstance(bottom, pd.Series) or not bottom.isna().all() else pd.Index([]),
                pd.Index([t1]) if not pd.isna(t1) else pd.Index([])
            ]).sort_values()
            
            if len(touch_times) > 0:
                out.append([loc, touch_times[0]])
                
        return pd.DataFrame(out, columns=['t0', 't1']).set_index('t0')

    @staticmethod
    def get_bins(events, close):
        """
        Generates labels: 1 if top barrier touched, 0 if bottom barrier or time limit.
        """
        events = events.dropna(subset=['t1'])
        px = events.index.union(events['t1'].values).drop_duplicates()
        px = close.reindex(px, method='bfill')
        
        out = pd.DataFrame(index=events.index)
        out['ret'] = px.loc[events['t1'].values].values / px.loc[events.index].values - 1
        
        # Label 1 if return is positive (hit top barrier)
        out['bin'] = np.sign(out['ret'])
        out['bin'].loc[out['ret'] == 0] = 0
        
        # Map -1 to 0 for binary classification
        out['bin'] = out['bin'].apply(lambda x: 1 if x > 0 else 0)
        
        return out

class HyperparameterTuner:
    """Automated Hyperparameter Tuning"""
    
    @staticmethod
    def tune_model(model, param_grid, X, y, cv=3, n_iter=5):
        if len(X) < 100:
            return model
            
        try:
            # Check if model has simple params optimization first (faster)
            # Otherwise use RandomizedSearch with few iterations
            from sklearn.model_selection import RandomizedSearchCV
            
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_grid,
                n_iter=n_iter, # Reduced iterations for speed
                cv=TimeSeriesSplit(n_splits=cv), # TimeSeries partition
                scoring='accuracy',
                n_jobs=-1,
                random_state=42,
                verbose=0,
                error_score='raise' 
            )
            
            search.fit(X, y)
            ProfessionalLogger.log(f"Best params for {type(model).__name__}: {search.best_params_}", "LEARN", "TUNER")
            return search.best_estimator_
            
        except Exception as e:
            # ProfessionalLogger.log(f"Tuning failed: {str(e)}", "WARNING", "TUNER")
            return model

class FeatureSelector:
    """Fast Feature Selection using Feature Importances"""
    
    def __init__(self, n_features_to_select=20):
        self.n_features = n_features_to_select
        self.selected_features = None
        
    def fit(self, X, y, estimator=None):
        try:
            from sklearn.feature_selection import SelectFromModel
            
            if estimator is None:
                estimator = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42, n_jobs=-1)
                
            # Fit estimator first
            estimator.fit(X, y)
            
            # Select features based on importance threshold
            selector = SelectFromModel(estimator,  max_features=self.n_features, threshold=-np.inf, prefit=True)
            
            feature_idx = selector.get_support()
            self.selected_features = X.columns[feature_idx].tolist()
            
            # Ensure we don't drop too many if SelectFromModel is too aggressive
            if len(self.selected_features) < 5:
                # Fallback to taking top N by importance manually
                importances = estimator.feature_importances_
                indices = np.argsort(importances)[::-1]
                self.selected_features = X.columns[indices[:self.n_features]].tolist()
            
            ProfessionalLogger.log(f"Selected {len(self.selected_features)} features", "LEARN", "SELECTOR")
            return self.selected_features
            
        except Exception as e:
            ProfessionalLogger.log(f"Feature selection failed: {str(e)}", "WARNING", "SELECTOR")
            return X.columns.tolist()

# ==========================================
# PROFESSIONAL RISK MANAGER
# ==========================================
class ProfessionalRiskManager:
    """Professional risk management with fat-tail adjustments"""
    
    def __init__(self):
        self.equity_curve = []
        self.trade_history = []
        self.consecutive_losses = 0
        self.daily_pnl = 0
        self.weekly_pnl = 0
        self.daily_start_balance = None
        self.weekly_start_balance = None
        
    def calculate_optimal_position_size(self, account_balance, trade_stats, market_conditions):
        """Calculate position size using multiple methods"""
        
        kelly_size = self._calculate_fat_tail_kelly(trade_stats)
        risk_parity_size = self._calculate_risk_parity_size(account_balance, market_conditions)
        vol_target_size = self._calculate_volatility_target_size(account_balance, market_conditions)
        
        final_size = (
            kelly_size * 0.4 +
            risk_parity_size * 0.3 +
            vol_target_size * 0.3
        )
        
        final_size = max(Config.BASE_VOLUME * 0.1, min(final_size, Config.BASE_VOLUME * 2))
        
        ProfessionalLogger.log(
            f"Position Size: Kelly={kelly_size:.3f}, "
            f"RiskParity={risk_parity_size:.3f}, "
            f"VolTarget={vol_target_size:.3f}, "
            f"Final={final_size:.3f}",
            "RISK", "RISK_MANAGER"
        )
        
        return final_size
    
    def _calculate_fat_tail_kelly(self, trade_stats):
        """Kelly criterion adjusted for fat tails"""
        if len(trade_stats) < 30:
            return Config.BASE_VOLUME
        
        returns = np.array([t['pnl'] / abs(t['risk']) if t['risk'] != 0 else 0 
                           for t in trade_stats[-30:]])
        
        winsorized = self._winsorize(returns, limits=[0.05, 0.05])
        
        wins = winsorized[winsorized > 0]
        losses = winsorized[winsorized <= 0]
        
        if len(wins) == 0 or len(losses) == 0:
            return Config.BASE_VOLUME
        
        win_rate = len(wins) / len(winsorized)
        avg_win = np.mean(wins)
        avg_loss = abs(np.mean(losses))
        
        if avg_loss == 0:
            return Config.BASE_VOLUME
        
        kelly_basic = win_rate - ((1 - win_rate) / (avg_win / avg_loss))
        
        skewness = skew(returns)
        kurt = kurtosis(returns)
        
        fat_tail_adjustment = 1 - (abs(skewness) * 0.1) - (max(0, kurt - 3) * 0.05)
        fat_tail_adjustment = max(0.3, min(fat_tail_adjustment, 1.0))
        
        fractional_kelly = kelly_basic * Config.KELLY_FRACTION * fat_tail_adjustment
        
        return Config.BASE_VOLUME * fractional_kelly * 10
    
    def _calculate_risk_parity_size(self, account_balance, market_conditions):
        """Risk parity position sizing"""
        if 'volatility' in market_conditions:
            target_vol = 0.15
            current_vol = market_conditions['volatility']
            
            if current_vol > 0:
                size = (target_vol / current_vol) * Config.BASE_VOLUME
                return max(Config.BASE_VOLUME * 0.5, min(size, Config.BASE_VOLUME * 2))
        
        return Config.BASE_VOLUME
    
    def _calculate_volatility_target_size(self, account_balance, market_conditions):
        """Volatility targeting position sizing"""
        risk_amount = account_balance * Config.RISK_PERCENT
        
        if 'atr' in market_conditions and market_conditions['atr'] > 0:
            size = risk_amount / (market_conditions['atr'] * 10000)
            return max(Config.BASE_VOLUME * 0.5, min(size, Config.BASE_VOLUME * 3))
        
        return Config.BASE_VOLUME
    
    def _winsorize(self, data, limits=[0.05, 0.05]):
        """Winsorize data to reduce tail impact"""
        data = np.array(data)
        lower = np.percentile(data, limits[0] * 100)
        upper = np.percentile(data, (1 - limits[1]) * 100)
        
        data[data < lower] = lower
        data[data > upper] = upper
        
        return data
    
    def check_risk_limits(self, account_balance, current_positions):
        """Check all risk limits"""
        violations = []
        
        daily_loss_pct = (self.daily_pnl / account_balance) * 100
        if daily_loss_pct <= -Config.MAX_DAILY_LOSS_PERCENT:
            violations.append(f"Daily loss: {daily_loss_pct:.1f}%")
        
        weekly_loss_pct = (self.weekly_pnl / account_balance) * 100
        if weekly_loss_pct <= -Config.MAX_WEEKLY_LOSS_PERCENT:
            violations.append(f"Weekly loss: {weekly_loss_pct:.1f}%")
        
        if len(self.equity_curve) > 10:
            current_equity = account_balance + sum(self.equity_curve[-10:])
            max_dd, _, _, _ = ProfessionalRiskMetrics.calculate_maximum_drawdown(
                [account_balance] + self.equity_curve
            )
            if max_dd > Config.MAX_DRAWDOWN_PERCENT / 100:
                violations.append(f"Max drawdown: {max_dd:.1%}")
        
        if len(current_positions) > 1:
            if len(current_positions) >= Config.MAX_POSITIONS:
                violations.append("Maximum positions reached")
        
        account_info = mt5.account_info()
        if account_info:
            if account_info.margin_level is not None and account_info.margin_level < 200:
                violations.append(f"Low margin: {account_info.margin_level:.1f}%")
            
            if account_info.margin_free is not None and account_info.margin_free < account_balance * 0.1:
                violations.append("Low free margin")
        
        return len(violations) == 0, violations

# ==========================================
# PROFESSIONAL ENSEMBLE MODEL
# ==========================================
class ProfessionalEnsemble:
    """Professional ensemble with advanced training, validation, and adaptive learning"""
    
    def __init__(self, trade_memory, feature_engine):
        self.feature_engine = feature_engine
        self.trade_memory = trade_memory
        self.data_quality_checker = ProfessionalDataQualityChecker()
        
        # New Components
        self.tuner = HyperparameterTuner()
        self.selector = FeatureSelector(n_features_to_select=25)
        self.labeler = TripleBarrierLabeling()
        
        # Model components
        self.base_models = self._initialize_base_models()
        self.ensemble = self._create_ensemble_structure()
        self.calibrated_ensemble = None
        self.scaler = RobustScaler()
        
        # Training state
        self.is_trained = False
        self.last_train_time = None
        self.training_metrics = {}
        self.model_performance = {}
        self.feature_importance = {}
        self.historical_data_cache = None
        
        # Validation components
        self.validation_scores = {}
        self.cv_folds = 5
        self.early_stopping_rounds = 10
        
        # Adaptive learning
        self.learning_history = []
        self.model_weights = {}
        self.performance_decay_factor = 0.95
        
        # Cache for predictions
        self.prediction_cache = {}
        self.cache_size = 1000
        
        ProfessionalLogger.log("Ensemble initialized with professional configuration", "INFO", "ENSEMBLE")
    
    def _initialize_base_models(self):
        """Initialize diverse base models with optimized hyperparameters"""
        models = []
        
        # Gradient Boosting - optimized for financial data
        models.append(('GB', GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10
        )))
        
        # Random Forest - robust and stable
        models.append(('RF', RandomForestClassifier(
            n_estimators=150,
            max_depth=12,
            min_samples_split=8,
            min_samples_leaf=4,
            max_features=0.7,
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced_subsample'
        )))
        
        # Logistic Regression - interpretable baseline
        models.append(('LR', LogisticRegression(
            penalty='elasticnet',
            C=1.0,
            l1_ratio=0.5,
            solver='saga',
            max_iter=2000,
            random_state=42,
            class_weight='balanced',
            warm_start=True
        )))
        
        # Neural Network - for capturing complex patterns
        models.append(('NN', MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size='auto',
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=42
        )))
        
        # XGBoost if available
        if XGB_AVAILABLE:
            models.append(('XGB', XGBClassifier(
                n_estimators=150,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,
                reg_alpha=0.01,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                eval_metric='logloss',
                use_label_encoder=False
            )))
        
        # Additional models for diversity
        try:
            from sklearn.svm import SVC
            models.append(('SVM', SVC(
                C=1.0,
                kernel='rbf',
                gamma='scale',
                probability=True,
                random_state=42,
                class_weight='balanced'
            )))
        except ImportError:
            pass
            
        ProfessionalLogger.log(f"Initialized {len(models)} base models", "INFO", "ENSEMBLE")
        return models
    
    def _create_ensemble_structure(self):
        """Create ensemble structure based on configuration"""
        if Config.USE_STACKING_ENSEMBLE:
            return self._create_stacking_ensemble()
        else:
            return self._create_voting_ensemble()
    
    def _create_stacking_ensemble(self):
        """Create stacking ensemble with meta-learner and cross-validation"""
        # Exclude models that might not work well in stacking
        base_estimators = []
        for name, model in self.base_models:
            if name not in ['NN', 'SVM']:  # NN can be unstable in stacking
                base_estimators.append((name, model))
        
        meta_learner = LogisticRegression(
            penalty='l2',
            C=0.5,
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )
        
        return StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_learner,
            cv=TimeSeriesSplit(n_splits=3),
            passthrough=True,
            n_jobs=-1,
            verbose=0
        )
    
    def _create_voting_ensemble(self):
        """Create weighted voting ensemble"""
        # Calculate initial weights based on model characteristics
        weights = {
            'GB': 1.2,  # Gradient Boosting - strong performance
            'RF': 1.0,  # Random Forest - stable
            'LR': 0.8,  # Logistic Regression - interpretable
            'NN': 1.1,  # Neural Network - complex patterns
            'XGB': 1.3,  # XGBoost - high performance
            'SVM': 0.9   # SVM - good margins
        }
        
        available_weights = [weights.get(name, 1.0) for name, _ in self.base_models]
        
        return VotingClassifier(
            estimators=[(name, model) for name, model in self.base_models],
            voting='soft',
            weights=available_weights,
            n_jobs=-1
        )
    
    def train_with_advanced_validation(self, data):
        """Train with advanced validation techniques"""
        ProfessionalLogger.log("Starting advanced training with comprehensive validation", "LEARN", "ENSEMBLE")
        
        # Prepare data
        X, y = self._prepare_training_data(data)
        if X is None or len(X) < Config.TRAINING_MIN_SAMPLES:
            ProfessionalLogger.log("Insufficient training data", "ERROR", "ENSEMBLE")
            return False
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Store performance metrics
        fold_scores = []
        fold_importances = []
        training_history = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            ProfessionalLogger.log(f"Training fold {fold_idx + 1}/{self.cv_folds}", "LEARN", "ENSEMBLE")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train ensemble
            self.ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.ensemble.score(X_train_scaled, y_train)
            val_score = self.ensemble.score(X_val_scaled, y_val)
            
            fold_scores.append({
                'fold': fold_idx + 1,
                'train_score': train_score,
                'val_score': val_score,
                'train_size': len(X_train),
                'val_size': len(X_val)
            })
            
            # Feature importance from Random Forest
            if 'RF' in [name for name, _ in self.base_models]:
                rf_model = next(model for name, model in self.base_models if name == 'RF')
                if hasattr(rf_model, 'feature_importances_'):
                    fold_importances.append(rf_model.feature_importances_)
            
            ProfessionalLogger.log(
                f"Fold {fold_idx + 1}: Train Acc={train_score:.2%}, Val Acc={val_score:.2%}",
                "LEARN", "ENSEMBLE"
            )
        
        # Final training on full dataset
        ProfessionalLogger.log("Training final ensemble on full dataset", "LEARN", "ENSEMBLE")
        
        X_scaled = self.scaler.fit_transform(X)
        self.ensemble.fit(X_scaled, y)
        
        # Calibrate probabilities
        self._calibrate_ensemble(X_scaled, y)
        
        # Calculate feature importance
        self._calculate_feature_importance(X.columns, fold_importances)
        
        # Update training state
        self.is_trained = True
        self.last_train_time = datetime.now()
        self.training_metrics = {
            'cv_scores': [score['val_score'] for score in fold_scores],
            'avg_val_score': np.mean([score['val_score'] for score in fold_scores]),
            'std_val_score': np.std([score['val_score'] for score in fold_scores]),
            'training_history': fold_scores
        }
        
        # Log summary
        self._log_training_summary()
        
        return True
    
    def _calibrate_ensemble(self, X, y):
        """Calibrate ensemble probabilities"""
        ProfessionalLogger.log("Calibrating ensemble probabilities", "LEARN", "ENSEMBLE")
        
        self.calibrated_ensemble = CalibratedClassifierCV(
            estimator=self.ensemble,
            method='isotonic',  # Isotonic for better calibration than sigmoid
            cv=TimeSeriesSplit(n_splits=3),
            ensemble=True
        )
        
        self.calibrated_ensemble.fit(X, y)
        
        # Evaluate calibration
        from sklearn.calibration import calibration_curve
        prob_pos = self.calibrated_ensemble.predict_proba(X)[:, 1]
        fraction_of_positives, mean_predicted_value = calibration_curve(y, prob_pos, n_bins=10)
        
        calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
        ProfessionalLogger.log(f"Calibration error: {calibration_error:.4f}", "LEARN", "ENSEMBLE")
    
    def _calculate_feature_importance(self, feature_names, fold_importances):
        """Calculate and analyze feature importance"""
        if not fold_importances:
            return
        
        avg_importance = np.mean(fold_importances, axis=0)
        std_importance = np.std(fold_importances, axis=0)
        
        self.feature_importance = {
            'mean': dict(zip(feature_names, avg_importance)),
            'std': dict(zip(feature_names, std_importance)),
            'ranking': sorted(zip(feature_names, avg_importance), 
                            key=lambda x: x[1], reverse=True)
        }
        
        # Log top features
        ProfessionalLogger.log("Top 15 features by importance:", "LEARN", "ENSEMBLE")
        for feature, importance in self.feature_importance['ranking'][:15]:
            std = self.feature_importance['std'].get(feature, 0)
            ProfessionalLogger.log(f"  {feature}: {importance:.4f} (±{std:.4f})", "LEARN", "ENSEMBLE")
    
    def _log_training_summary(self):
        """Log comprehensive training summary"""
        ProfessionalLogger.log("=" * 60, "LEARN", "ENSEMBLE")
        ProfessionalLogger.log("📊 TRAINING SUMMARY", "LEARN", "ENSEMBLE")
        ProfessionalLogger.log(f"Final Validation Accuracy: {self.training_metrics['avg_val_score']:.2%}", "LEARN", "ENSEMBLE")
        ProfessionalLogger.log(f"Validation Std Dev: {self.training_metrics['std_val_score']:.2%}", "LEARN", "ENSEMBLE")
        
        # Class distribution info
        if 'class_distribution' in self.training_metrics:
            dist = self.training_metrics['class_distribution']
            ProfessionalLogger.log(f"Class Distribution: Buy={dist.get(1, 0):.1%}, Sell={dist.get(0, 0):.1%}", "LEARN", "ENSEMBLE")
        
        ProfessionalLogger.log("=" * 60, "LEARN", "ENSEMBLE")
    
    def _prepare_training_data(self, data):
        """Prepare training data with advanced preprocessing"""
        if data is None or len(data) < 100:
            return None, None
        
        # Calculate features
        df_features = self.feature_engine.calculate_features(data)
        
        # Create labels with Triple Barrier Method
        df_labeled = self._create_advanced_labels(df_features)
        
        # Handle class imbalance
        df_balanced = self._balance_dataset(df_labeled)
        
        # Extract features and labels
        feature_cols = self.feature_engine.get_feature_columns()
        X = df_balanced[feature_cols].fillna(0)
        y = df_balanced['label']
        
        # Store class distribution
        self.training_metrics['class_distribution'] = dict(y.value_counts(normalize=True))
        
        return X, y
    
    def _balance_dataset(self, df):
        """Balance dataset using SMOTE or adaptive sampling"""
        if 'label' not in df.columns:
            return df
        
        class_counts = df['label'].value_counts()
        min_samples = class_counts.min()
        
        if len(class_counts) < 2:
            return df
        
        # If imbalance is severe, use undersampling
        max_imbalance_ratio = 3.0
        if class_counts.max() / class_counts.min() > max_imbalance_ratio:
            balanced_dfs = []
            for label, count in class_counts.items():
                label_df = df[df['label'] == label]
                if count > min_samples * 1.5:
                    # Undersample majority class
                    sample_size = int(min_samples * 1.5)
                    balanced_dfs.append(label_df.sample(n=min(sample_size, len(label_df)), 
                                                       random_state=42))
                else:
                    balanced_dfs.append(label_df)
            
            return pd.concat(balanced_dfs, ignore_index=True)
        
        return df
    
    def train(self, recent_mt5_data):
        """Main training method with fallback strategies"""
        try:
            ProfessionalLogger.log("🚀 Starting ensemble training pipeline", "LEARN", "ENSEMBLE")
            
            # Load and combine training data
            df = self._load_training_data(recent_mt5_data)
            if df is None or len(df) < Config.TRAINING_MIN_SAMPLES:
                ProfessionalLogger.log(f"Insufficient data: {len(df) if df else 0} samples", "ERROR", "ENSEMBLE")
                return False
            
            # Data quality check
            quality_score, quality_details = self.data_quality_checker.check_data_quality(df)
            if quality_score < Config.MIN_DATA_QUALITY_SCORE:
                ProfessionalLogger.log(f"Data quality too low: {quality_score:.2%}", "WARNING", "ENSEMBLE")
                # Try to clean data
                df = self._clean_training_data(df)
            
            # Train with advanced validation
            success = self.train_with_advanced_validation(df)
            
            if success:
                # Update model performance tracking
                self._update_model_performance()
                
                # Save model if configured
                if hasattr(Config, 'MODEL_SAVE_FILE'):
                    self._save_model()
                
                ProfessionalLogger.log("✅ Ensemble training completed successfully", "SUCCESS", "ENSEMBLE")
            else:
                ProfessionalLogger.log("❌ Ensemble training failed", "ERROR", "ENSEMBLE")
            
            return success
            
        except Exception as e:
            ProfessionalLogger.log(f"Training error: {str(e)}", "ERROR", "ENSEMBLE")
            import traceback
            traceback.print_exc()
            return False
    
    def _create_advanced_labels(self, df):
        """Use Triple Barrier Method for robust labeling"""
        try:
            # Volatility for dynamic barriers
            df['volatility_100'] = df['close'].pct_change().rolling(100).std()
            
            # Events
            t_events = df.index
            
            # Barriers (TP: 2.0 * vol, SL: 1.0 * vol)
            pt_sl = [2.0, 1.0]
            target = df['volatility_100']
            min_ret = 0.001
            
            # Vertical barrier (hold time limit) - e.g., 20 bars
            vertical_barrier = pd.Series(df.index + 20, index=df.index)
            # Clip to max index
            max_idx = df.index[-1]
            vertical_barrier = vertical_barrier.apply(lambda x: min(x, max_idx))
            
            events = self.labeler.get_events(
                df['close'], 
                pd.Series(t_events, index=t_events), 
                pt_sl, 
                target, 
                min_ret,
                vertical_barrier_times=vertical_barrier
            )
            
            labels = self.labeler.get_bins(events, df['close'])
            
            # Join labels
            df_labeled = df.copy()
            df_labeled.loc[labels.index, 'label'] = labels['bin']
            
            # Fill missing labels
            df_labeled.dropna(subset=['label'], inplace=True)
            
            # CRITICAL: Check if we lost too much data
            if len(df_labeled) < 50:
                ProfessionalLogger.log(f"Advanced labeling resulted in too few samples ({len(df_labeled)}). Falling back.", "WARNING", "ENSEMBLE")
                return self.feature_engine.create_labels(df)
            
            return df_labeled
            
        except Exception as e:
            ProfessionalLogger.log(f"Labeling error: {str(e)}", "ERROR", "ENSEMBLE")
            # Fallback to simple labeling
            return self.feature_engine.create_labels(df)
            
    def _tune_models(self, X, y):
        """Tune base models"""
        ProfessionalLogger.log("Tuning hyperparameters...", "LEARN", "ENSEMBLE")
        
        tuned_models = []
        for name, model in self.base_models:
            param_grid = {}
            if name == 'GB':
                param_grid = {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [3, 5, 7],
                    'learning_rate': [0.01, 0.05, 0.1]
                }
            elif name == 'RF':
                param_grid = {
                    'n_estimators': [100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            
            if param_grid:
                # Use reduced n_iter for speed
                tuned = self.tuner.tune_model(model, param_grid, X, y, n_iter=3)
                tuned_models.append((name, tuned))
            else:
                tuned_models.append((name, model))
                
        self.base_models = tuned_models
        # Re-create ensemble
        self.ensemble = self._create_ensemble_structure()

    def train_with_advanced_validation(self, data):
        """Train with advanced validation techniques"""
        ProfessionalLogger.log("Starting professional training pipeline", "LEARN", "ENSEMBLE")
        
        # Prepare data
        X, y = self._prepare_training_data(data)
        if X is None or len(X) < Config.TRAINING_MIN_SAMPLES:
            ProfessionalLogger.log("Insufficient training data", "ERROR", "ENSEMBLE")
            return False
            
        # Feature Selection
        selected_features = self.selector.fit(X, y)
        X = X[selected_features]
        # Store selected features for prediction time
        self.selected_features_names = selected_features
        
        # Hyperparameter Tuning
        self._tune_models(X, y)
        
        # Time-series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_folds)
        
        # Store performance metrics
        fold_scores = []
        fold_importances = []
        
        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            ProfessionalLogger.log(f"Training fold {fold_idx + 1}/{self.cv_folds}", "LEARN", "ENSEMBLE")
            
            # Split data
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)
            
            # Train ensemble
            self.ensemble.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.ensemble.score(X_train_scaled, y_train)
            val_score = self.ensemble.score(X_val_scaled, y_val)
            
            fold_scores.append({
                'fold': fold_idx + 1,
                'train_score': train_score,
                'val_score': val_score
            })
            
            ProfessionalLogger.log(
                f"Fold {fold_idx + 1}: Train Acc={train_score:.2%}, Val Acc={val_score:.2%}",
                "LEARN", "ENSEMBLE"
            )
        
        # Final training on full dataset
        ProfessionalLogger.log("Training final ensemble on full dataset", "LEARN", "ENSEMBLE")
        
        X_scaled = self.scaler.fit_transform(X)
        self.ensemble.fit(X_scaled, y)
        
        # Calibrate probabilities
        self._calibrate_ensemble(X_scaled, y)
        
        # Update training state
        self.is_trained = True
        self.last_train_time = datetime.now()
        self.training_metrics = {
            'avg_val_score': np.mean([score['val_score'] for score in fold_scores]),
            'std_val_score': np.std([score['val_score'] for score in fold_scores])
        }
        
        self._log_training_summary()
        return True
    
    def _load_training_data(self, recent_mt5_data):
        """Load training data from multiple sources with caching"""
        all_dfs = []
        
        # 1. Historical data (Optional)
        if Config.USE_HISTORICAL_DATASET:
            historical_data = self._get_historical_data()
            if historical_data is not None:
                df_hist = self.feature_engine.calculate_features(historical_data)
                df_hist = self._create_advanced_labels(df_hist)
                df_hist = df_hist[df_hist['label'] != -1].tail(Config.HISTORICAL_DATA_LIMIT)
                df_hist['weight'] = Config.HISTORICAL_WEIGHT
                df_hist['data_source'] = 'historical'
                all_dfs.append(df_hist)
                ProfessionalLogger.log(f"Added {len(df_hist)} historical samples", "DATA", "ENSEMBLE")
        
        # 2. Recent MT5 data - CRITICAL update: Ensure we use it if passed
        if recent_mt5_data is not None and len(recent_mt5_data) > 100:
            df_recent = self.feature_engine.calculate_features(recent_mt5_data)
            # Use advanced labels here too
            df_recent = self._create_advanced_labels(df_recent)
            df_recent = df_recent[df_recent['label'] != -1]
            df_recent['weight'] = Config.RECENT_WEIGHT
            df_recent['data_source'] = 'recent'
            all_dfs.append(df_recent)
            ProfessionalLogger.log(f"Added {len(df_recent)} recent samples", "DATA", "ENSEMBLE")
        
        # 3. Trade experience data
        trades = self.trade_memory.get_completed_trades()
        if trades and len(trades) >= 10:
            experience_data = self._create_experience_dataset(trades)
            if experience_data is not None:
                all_dfs.append(experience_data)
                ProfessionalLogger.log(f"Added {len(experience_data)} experience samples", "LEARN", "ENSEMBLE")
        
        if not all_dfs:
            return None
        
        # Combine all data sources
        combined = pd.concat(all_dfs, ignore_index=True, sort=False)
        
        # Shuffle while maintaining time-series characteristics for recent data
        if len(combined) > 1000:
            # Keep recent data in order, shuffle older data
            recent_cutoff = int(len(combined) * 0.3)
            recent_data = combined.iloc[-recent_cutoff:]
            older_data = combined.iloc[:-recent_cutoff].sample(frac=1, random_state=42)
            combined = pd.concat([older_data, recent_data], ignore_index=True)
        
        ProfessionalLogger.log(f"Total training samples: {len(combined)}", "DATA", "ENSEMBLE")
        return combined
    
    def _get_historical_data(self):
        """Get historical data with caching"""
        if self.historical_data_cache is None:
            self.historical_data_cache = DataLoader.load_huggingface_dataset()
        return self.historical_data_cache
    
    def _create_experience_dataset(self, trades):
        """Create training dataset from trading experience"""
        samples = []
        
        for trade in trades[-100:]:  # Use most recent 100 trades
            if 'features' in trade and 'profit' in trade:
                features = trade['features']
                profit = trade['profit']
                signal = trade.get('signal', 0)
                
                # Create label based on profitability
                if abs(profit) > 0.1:  # Significant profit/loss
                    label = 1 if profit > 0 else 0
                    
                    # Weight based on profit magnitude and recency
                    weight = Config.LEARNING_WEIGHT * (1 + min(abs(profit) / 50, 2))
                    
                    sample = {**features, 'label': label, 'weight': weight}
                    samples.append(sample)
        
        if samples:
            df = pd.DataFrame(samples)
            df['data_source'] = 'experience'
            return df
        
        return None
    
    def _clean_training_data(self, df):
        """Clean training data by removing outliers and handling missing values"""
        original_len = len(df)
        
        # Remove rows with too many missing values
        df = df.dropna(thresh=int(df.shape[1] * 0.7))
        
        # Remove extreme outliers in numerical features
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col not in ['label', 'weight', 'data_source']:
                q1 = df[col].quantile(0.05)
                q3 = df[col].quantile(0.95)
                iqr = q3 - q1
                lower_bound = q1 - 3 * iqr
                upper_bound = q3 + 3 * iqr
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
        
        removed = original_len - len(df)
        if removed > 0:
            ProfessionalLogger.log(f"Cleaned data: removed {removed} problematic samples", "DATA", "ENSEMBLE")
        
        return df
    
    def predict(self, df, use_cache=True):
        """Make prediction with comprehensive validation and caching"""
        if not self.is_trained:
            ProfessionalLogger.log("Model not trained, cannot predict", "WARNING", "ENSEMBLE")
            return None, 0.0, None, {}
        
        try:
            # Check cache first
            cache_key = hash(tuple(df['close'].tail(10).values.tolist()))
            if use_cache and cache_key in self.prediction_cache:
                cached = self.prediction_cache[cache_key]
                ProfessionalLogger.log("Using cached prediction", "INFO", "ENSEMBLE")
                return cached
            
            # Calculate features
            df_feat = self.feature_engine.calculate_features(df)
            feature_cols = self.feature_engine.get_feature_columns()
            
            # Prepare input
            X = df_feat[feature_cols].iloc[-1:].fillna(0)
            
            # Filter selected features if trained
            if hasattr(self, 'selected_features_names') and self.selected_features_names:
                X = X[self.selected_features_names]
                
            X_scaled = self.scaler.transform(X)
            
            # Get predictions from all models
            sub_preds = self._get_submodel_predictions(X_scaled)
            
            # Get ensemble prediction
            if self.calibrated_ensemble is not None:
                final_p = self.calibrated_ensemble.predict(X_scaled)[0]
                proba = self.calibrated_ensemble.predict_proba(X_scaled)[0]
                final_c = np.max(proba)
            else:
                final_p = self.ensemble.predict(X_scaled)[0]
                proba = self.ensemble.predict_proba(X_scaled)[0]
                final_c = np.max(proba)
            
            # Create feature dictionary
            f_dict = {col: float(X[col].iloc[0]) for col in feature_cols}
            
            # Validate prediction
            if not self._validate_prediction(final_p, final_c, sub_preds, f_dict):
                return None, 0.0, None, sub_preds
            
            # Update cache
            result = (final_p, final_c, f_dict, sub_preds)
            self._update_prediction_cache(cache_key, result)
            
            return result
            
        except Exception as e:
            ProfessionalLogger.log(f"Prediction error: {str(e)}", "ERROR", "ENSEMBLE")
            return None, 0.0, None, {}
    
    def _get_submodel_predictions(self, X_scaled):
        """Get predictions from all sub-models"""
        sub_preds = {}
        
        for name, model in self.base_models:
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_scaled)[0]
                    prediction = model.predict(X_scaled)[0]
                    confidence = np.max(proba)
                else:
                    prediction = model.predict(X_scaled)[0]
                    confidence = 1.0
                
                sub_preds[name] = {
                    'prediction': int(prediction),
                    'confidence': float(confidence),
                    'proba': proba.tolist() if hasattr(model, 'predict_proba') else None
                }
            except Exception as e:
                ProfessionalLogger.log(f"Submodel {name} prediction failed: {str(e)}", "WARNING", "ENSEMBLE")
                sub_preds[name] = {'prediction': -1, 'confidence': 0.0, 'proba': None}
        
        return sub_preds
    
    def _validate_prediction(self, prediction, confidence, sub_preds, features):
        """Validate prediction against multiple criteria"""
        # 1. Check confidence threshold
        if confidence < Config.MIN_CONFIDENCE:
            ProfessionalLogger.log(f"Low confidence: {confidence:.2%} < {Config.MIN_CONFIDENCE:.0%}", 
                                 "WARNING", "ENSEMBLE")
            return False
        
        # 2. Check ensemble agreement
        valid_predictions = [p['prediction'] for p in sub_preds.values() 
                           if p['prediction'] != -1 and p['confidence'] > 0.5]
        
        if valid_predictions:
            agreement = valid_predictions.count(prediction) / len(valid_predictions)
            if agreement < Config.MIN_ENSEMBLE_AGREEMENT:
                ProfessionalLogger.log(f"Low agreement: {agreement:.0%} < {Config.MIN_ENSEMBLE_AGREEMENT:.0%}", 
                                     "WARNING", "ENSEMBLE")
                return False
        
        # 3. Check market regime constraints
        if Config.USE_MARKET_REGIME:
            regime = features.get('regime', 0)
            adx = features.get('adx', 0)
            plus_di = features.get('plus_di', 0)
            minus_di = features.get('minus_di', 0)
            
            # Adjust confidence threshold based on regime
            regime_multipliers = {0: 1.0, 1: 1.1, 2: 1.3}
            required_confidence = Config.MIN_CONFIDENCE * regime_multipliers.get(regime, 1.0)
            
            if confidence < required_confidence:
                ProfessionalLogger.log(f"Insufficient confidence for regime {regime}: "
                                     f"{confidence:.2%} < {required_confidence:.2%}", 
                                     "WARNING", "ENSEMBLE")
                return False
            
            # Check trend alignment
            if adx > Config.ADX_TREND_THRESHOLD:
                if plus_di > minus_di and prediction == 0:
                    ProfessionalLogger.log(f"Prediction contradicts uptrend (ADX={adx:.1f})", 
                                         "WARNING", "ENSEMBLE")
                    return False
                if minus_di > plus_di and prediction == 1:
                    ProfessionalLogger.log(f"Prediction contradicts downtrend (ADX={adx:.1f})", 
                                         "WARNING", "ENSEMBLE")
                    return False
        
        # 4. Check feature consistency
        if not self._check_feature_consistency(features):
            ProfessionalLogger.log("Feature consistency check failed", "WARNING", "ENSEMBLE")
            return False
        
        return True
    
    def _check_feature_consistency(self, features):
        """Check if features are internally consistent"""
        # Check if RSI is extreme but prediction doesn't align
        rsi = features.get('rsi_normalized', 0) * 50 + 50  # Denormalize
        prediction = features.get('prediction', -1)
        
        if rsi > 70 and prediction == 1:  # Overbought but predicting buy
            return False
        if rsi < 30 and prediction == 0:  # Oversold but predicting sell
            return False
        
        # Check if volatility is too high
        volatility = features.get('volatility', 0)
        if volatility > 0.02:  # 2% daily volatility threshold
            ProfessionalLogger.log(f"High volatility: {volatility:.2%}", "WARNING", "ENSEMBLE")
        
        return True
    
    def _update_prediction_cache(self, key, prediction):
        """Update prediction cache with LRU strategy"""
        if len(self.prediction_cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.prediction_cache))
            del self.prediction_cache[oldest_key]
        
        self.prediction_cache[key] = prediction
    
    def _update_model_performance(self):
        """Update model performance tracking"""
        self.learning_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': self.training_metrics.copy(),
            'feature_importance': self.feature_importance.copy()
        })
        
        # Keep only recent history
        if len(self.learning_history) > 20:
            self.learning_history = self.learning_history[-20:]
    
    def _save_model(self):
        """Save trained model to disk"""
        try:
            import joblib
            model_data = {
                'ensemble': self.ensemble,
                'calibrated_ensemble': self.calibrated_ensemble,
                'scaler': self.scaler,
                'feature_importance': self.feature_importance,
                'training_metrics': self.training_metrics,
                'last_train_time': self.last_train_time
            }
            
            joblib.dump(model_data, Config.MODEL_SAVE_FILE)
            ProfessionalLogger.log(f"Model saved to {Config.MODEL_SAVE_FILE}", "INFO", "ENSEMBLE")
            
        except Exception as e:
            ProfessionalLogger.log(f"Error saving model: {str(e)}", "ERROR", "ENSEMBLE")
    
    def load_model(self, model_file=None):
        """Load trained model from disk"""
        try:
            import joblib
            
            model_file = model_file or Config.MODEL_SAVE_FILE
            if not os.path.exists(model_file):
                ProfessionalLogger.log(f"Model file not found: {model_file}", "WARNING", "ENSEMBLE")
                return False
            
            model_data = joblib.load(model_file)
            
            self.ensemble = model_data['ensemble']
            self.calibrated_ensemble = model_data.get('calibrated_ensemble')
            self.scaler = model_data['scaler']
            self.feature_importance = model_data.get('feature_importance', {})
            self.training_metrics = model_data.get('training_metrics', {})
            self.last_train_time = model_data.get('last_train_time')
            self.is_trained = True
            
            ProfessionalLogger.log(f"Model loaded from {model_file}", "SUCCESS", "ENSEMBLE")
            return True
            
        except Exception as e:
            ProfessionalLogger.log(f"Error loading model: {str(e)}", "ERROR", "ENSEMBLE")
            return False
    
    def should_retrain(self):
        """Determine if retraining is needed with adaptive thresholds"""
        if not self.last_train_time:
            return True
        
        # Time-based retraining
        hours_since = (datetime.now() - self.last_train_time).total_seconds() / 3600
        if hours_since >= Config.RETRAIN_HOURS:
            ProfessionalLogger.log(f"Scheduled retraining after {hours_since:.1f} hours", 
                                 "LEARN", "ENSEMBLE")
            return True
        
        # Performance-based retraining
        stats = self.trade_memory.get_statistical_summary(period_days=1)
        if stats and stats.get('total_trades', 0) >= 10:
            win_rate = stats.get('win_rate', 0)
            profit_factor = stats.get('profit_factor', 0)
            
            # Dynamic thresholds based on market conditions
            if win_rate < 0.4 or profit_factor < 1.2:
                ProfessionalLogger.log(f"Performance degradation detected: "
                                     f"WinRate={win_rate:.1%}, ProfitFactor={profit_factor:.2f}", 
                                     "LEARN", "ENSEMBLE")
                return True
        
        # Data drift detection (simplified)
        if self._detect_data_drift():
            ProfessionalLogger.log("Data drift detected, retraining recommended", 
                                 "LEARN", "ENSEMBLE")
            return True
        
        return False
    
    def _detect_data_drift(self):
        """Simple data drift detection based on feature statistics"""
        # This is a simplified version - in production, use proper drift detection
        if not self.feature_importance or 'mean' not in self.feature_importance:
            return False
        
        # Check if top features have changed significantly
        top_features = [f[0] for f in self.feature_importance.get('ranking', [])[:5]]
        
        # In a real implementation, compare current feature distributions
        # with training feature distributions
        
        return False
    
    def get_diagnostics(self):
        """Get comprehensive model diagnostics"""
        return {
            'training_status': {
                'is_trained': self.is_trained,
                'last_train_time': self.last_train_time.isoformat() if self.last_train_time else None,
                'training_samples': len(self.learning_history)
            },
            'performance': {
                'avg_val_score': self.training_metrics.get('avg_val_score', 0),
                'val_score_std': self.training_metrics.get('std_val_score', 0),
                'calibration_error': self.training_metrics.get('calibration_error', 0)
            },
            'model_info': {
                'base_models': [name for name, _ in self.base_models],
                'ensemble_type': 'stacking' if Config.USE_STACKING_ENSEMBLE else 'voting',
                'feature_count': len(self.feature_engine.get_feature_columns())
            },
            'feature_analysis': {
                'top_features': self.feature_importance.get('ranking', [])[:10],
                'feature_count': len(self.feature_importance.get('mean', {}))
            }
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
            ProfessionalLogger.log(f"Symbol {symbol} not found", "ERROR", "EXECUTOR")
            return None
        
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            ProfessionalLogger.log(f"Cannot get tick data for {symbol}", "ERROR", "EXECUTOR")
            return None
        
        if order_type == mt5.ORDER_TYPE_BUY:
            current_price = tick.ask
            slippage = abs(current_price - entry_price) / symbol_info.point
            
            if slippage > Config.MAX_SLIPPAGE_PIPS:
                ProfessionalLogger.log(f"Slippage too high: {slippage:.1f} pips", "WARNING", "EXECUTOR")
                
                if Config.USE_SMART_ENTRY and entry_price < current_price:
                    return self.place_limit_order(
                        symbol, mt5.ORDER_TYPE_BUY_LIMIT, volume, 
                        entry_price, sl, tp, magic, comment
                    )
            
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
            
        else:
            current_price = tick.bid
            slippage = abs(current_price - entry_price) / symbol_info.point
            
            if slippage > Config.MAX_SLIPPAGE_PIPS:
                ProfessionalLogger.log(f"Slippage too high: {slippage:.1f} pips", "WARNING", "EXECUTOR")
                
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
        
        result = mt5.order_send(request)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            ProfessionalLogger.log(f"Order failed: {result.retcode} - {result.comment}", "ERROR", "EXECUTOR")
            
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
            ProfessionalLogger.log(f"Limit order placed: #{order_ticket} at {price:.2f}", "INFO", "EXECUTOR")
        
        return result
    
    def check_pending_orders(self):
        """Check and manage pending orders"""
        expired_orders = []
        
        for order_ticket, order_info in self.pending_orders.items():
            order = mt5.order_get(ticket=order_ticket)
            
            if order is None or order.time_done > 0:
                expired_orders.append(order_ticket)
                continue
            
            age = (datetime.now() - order_info['placed_at']).total_seconds()
            if age > self.order_timeout:
                mt5.order_delete(order_ticket)
                expired_orders.append(order_ticket)
                ProfessionalLogger.log(f"Limit order #{order_ticket} expired", "INFO", "EXECUTOR")
        
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
        cache_file = "historical_data_cache.pkl"
        
        # 1. Try to load from local cache
        if os.path.exists(cache_file):
            try:
                # Check cache age (expire after 24 hours)
                if (datetime.now().timestamp() - os.path.getmtime(cache_file)) < 86400:
                    ProfessionalLogger.log("Loading dataset from local cache...", "DATA", "LOADER")
                    df = pd.read_pickle(cache_file)
                    
                    # Apply limit if configured
                    if hasattr(Config, 'HISTORICAL_DATA_LIMIT'):
                        df = df.tail(Config.HISTORICAL_DATA_LIMIT)
                        
                    return df
            except Exception as e:
                ProfessionalLogger.log(f"Cache load failed: {e}", "WARNING", "LOADER")

        try:
            ProfessionalLogger.log("Loading historical XAU/USD dataset (Optimized)...", "DATA", "LOADER")
            
            # Load dataset in streaming mode or just metadata first
            ds = load_dataset("ZombitX64/xauusd-gold-price-historical-data-2004-2025", split='train')
            
            # Determine how much data we actually need
            # If we only need the last N rows (e.g. 50000 for plenty of history, or Config limit)
            # We fetch a reasonable buffer to allow for technical indicator calculation
            needed_rows = getattr(Config, 'HISTORICAL_DATA_LIMIT', 5000) * 2
            needed_rows = max(needed_rows, 10000) # Ensure minimal baseline
            
            total_rows = len(ds)
            start_idx = max(0, total_rows - needed_rows)
            
            ProfessionalLogger.log(f"Fetching last {needed_rows} rows from {total_rows} total...", "DATA", "LOADER")
            
            # Slice the dataset - this returns a dict of lists, which is much faster than to_pandas() on the whole object
            data_slice = ds[start_idx:]
            
            # Convert to DataFrame
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
                # Handle varying time formats
                if df['time'].dtype == 'object':
                    df['time'] = pd.to_datetime(df['time']).view('int64') // 10**9
                elif np.issubdtype(df['time'].dtype, np.datetime64):
                     df['time'] = df['time'].view('int64') // 10**9
            
            # Sort and clean
            df = df.sort_values('time').reset_index(drop=True)
            
            # Check quality
            quality_score, _ = ProfessionalDataQualityChecker.check_data_quality(df)
            ProfessionalLogger.log(f"Dataset loaded. Quality: {quality_score:.2%}", "SUCCESS", "LOADER")
            
            # Save to cache (save the larger slice to avoid re-downloading next time we increase limit)
            try:
                df.to_pickle(cache_file)
            except:
                pass
                
            # Apply strict limit for return
            if hasattr(Config, 'HISTORICAL_DATA_LIMIT'):
                df = df.tail(Config.HISTORICAL_DATA_LIMIT)
                
            return df
            
        except Exception as e:
            ProfessionalLogger.log(f"Failed to load dataset: {str(e)}", "ERROR", "LOADER")
            return None

# ==========================================
# PROFESSIONAL TRADING ENGINE
# ==========================================
class ProfessionalTradingEngine:
    """Main professional trading engine"""
    
    def __init__(self):
        self.trade_memory = ProfessionalTradeMemory()
        self.risk_manager = ProfessionalRiskManager()
        self.feature_engine = ProfessionalFeatureEngine()
        self.price_action = ProfessionalPriceActionAnalyzer()
        self.order_executor = SmartOrderExecutor()
        self.mtf_analyzer = ProfessionalMultiTimeframeAnalyzer()
        
        # Initialize model components
        self.model = None
        self.stat_analyzer = StatisticalAnalyzer()
        
        self.connected = False
        self.active_positions = {}
        self.pending_signals = {}
        self.iteration = 0
        self.last_performance_report = datetime.now()
    
    # ==========================================
    # CORE METHODS
    # ==========================================
    
    def run(self):
        """Main execution method - runs backtest then live trading"""
        print("\n" + "=" * 70)
        print("🤖 PROFESSIONAL MT5 ALGORITHMIC TRADING SYSTEM")
        print("📊 Advanced Statistics | Fat-Tail Risk | Professional Backtesting")
        print("=" * 70 + "\n")
        
        ProfessionalLogger.log("Starting professional trading system...", "INFO", "MAIN")
        
        # Run backtest first
        backtest_passed = self.run_comprehensive_backtest()
        
        if backtest_passed:
            ProfessionalLogger.log("\n" + "=" * 70, "INFO", "MAIN")
            ProfessionalLogger.log("BACKTEST PASSED - PROCEEDING TO LIVE TRADING", "SUCCESS", "MAIN")
            ProfessionalLogger.log("=" * 70 + "\n", "INFO", "MAIN")
            
            self.run_live_trading()
        else:
            ProfessionalLogger.log("\n" + "=" * 70, "ERROR", "MAIN")
            ProfessionalLogger.log("BACKTEST FAILED - ABORTING LIVE TRADING", "ERROR", "MAIN")
            ProfessionalLogger.log("=" * 70, "ERROR", "MAIN")
    
    def get_historical_data(self, timeframe=None, bars=None):
        """Get historical data from MT5"""
        if not self.connected:
            if not self.connect_mt5():
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
                
                # Update risk manager
                self.risk_manager.record_trade(deal.profit)
                
                if ticket in self.active_positions:
                    del self.active_positions[ticket]
    
    def calculate_position_size(self, stop_loss_pips, risk_multiplier=1.0):
        """Calculate position size based on risk management"""
        if not self.connected:
            return Config.BASE_VOLUME
        
        account = mt5.account_info()
        if not account:
            return Config.BASE_VOLUME
        
        trade_stats = self.trade_memory.get_completed_trades()[-30:] if len(self.trade_memory.get_completed_trades()) >= 30 else []
        market_conditions = {'atr': self.get_current_atr()}
        
        position_size = self.risk_manager.calculate_optimal_position_size(
            account.balance, trade_stats, market_conditions
        )
        
        return position_size
    
    def get_current_atr(self):
        """Get current ATR"""
        df = self.get_historical_data(bars=100)
        if df is not None and len(df) > Config.ATR_PERIOD:
            high = df['high']
            low = df['low']
            close = df['close']
            
            tr = np.maximum(
                high - low,
                np.maximum(
                    abs(high - close.shift(1)),
                    abs(low - close.shift(1))
                )
            )
            atr = tr.rolling(Config.ATR_PERIOD).mean().iloc[-1]
            return atr
        return 0
    
    def execute_enhanced_trade(self, signal, confidence, df_current, features, model_agreement):
        """Execute trade with all enhancements"""
        
        if self.get_current_positions() >= Config.MAX_POSITIONS:
            ProfessionalLogger.log("Max positions reached", "WARNING", "ENGINE")
            return False
        
        account = mt5.account_info()
        if account:
            can_trade, violations = self.risk_manager.check_risk_limits(account.balance, self.active_positions)
            if not can_trade:
                ProfessionalLogger.log(f"Trading halted: {', '.join(violations)}", "RISK", "ENGINE")
                return False
        
        tick = mt5.symbol_info_tick(Config.SYMBOL)
        if not tick:
            return False
        
        current_price = df_current.iloc[-1]['close']
        
        atr = df_current['close'].rolling(Config.ATR_PERIOD).std().iloc[-1] * 2
        if pd.isna(atr) or atr == 0:
            atr = current_price * 0.001
        
        if Config.USE_DYNAMIC_SL_TP:
            levels = self.price_action.calculate_optimal_entry_sl_tp(
                df_current, signal, current_price, atr, Config.MIN_RR_RATIO
            )
            
            optimal_entry = levels['optimal_entry']
            sl = levels['sl']
            tp = levels['tp']
            rr_ratio = levels['rr_ratio']
            
            ProfessionalLogger.log(f"📊 Enhanced Price Analysis", "DATA", "ENGINE")
            ProfessionalLogger.log(f"  Entry: {optimal_entry:.2f} | SL: {sl:.2f} | TP: {tp:.2f} | R:R: {rr_ratio:.2f}", "DATA", "ENGINE")
            ProfessionalLogger.log(f"  Support: {levels['support']:.2f} | Resistance: {levels['resistance']:.2f}", "DATA", "ENGINE")
            
            if rr_ratio < Config.MIN_RR_RATIO:
                ProfessionalLogger.log(f"Poor R:R ratio {rr_ratio:.2f} (min: {Config.MIN_RR_RATIO})", "WARNING", "ENGINE")
                return False
            
            distance_to_entry = abs(current_price - optimal_entry)
            entry_tolerance = atr * 0.3
            
            if distance_to_entry > entry_tolerance and Config.USE_SMART_ENTRY:
                ProfessionalLogger.log(
                    f"⏳ Waiting for better entry | Current: {current_price:.2f} | "
                    f"Target: {optimal_entry:.2f} | Distance: {distance_to_entry:.2f}", "WARNING", "ENGINE"
                )
                
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
        
        if signal == 1:
            order_type = mt5.ORDER_TYPE_BUY
        else:
            order_type = mt5.ORDER_TYPE_SELL
        
        stop_loss_pips = abs(optimal_entry - sl) / 0.01
        volume = self.calculate_position_size(stop_loss_pips)
        
        comment = (f"PRO_{int(confidence*100)}_RR{int(rr_ratio*10)}_"
                  f"AG{int(model_agreement.get('agreement', 1)*100)}")
        
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
        
        signal_type = "BUY" if signal == 1 else "SELL"
        ProfessionalLogger.log("=" * 50, "TRADE", "ENGINE")
        ProfessionalLogger.log(f"✅ {signal_type} EXECUTED | Ticket #{result.order}", "TRADE", "ENGINE")
        ProfessionalLogger.log(f"  Entry: {result.price:.2f} | SL: {sl:.2f} | TP: {tp:.2f}", "TRADE", "ENGINE")
        ProfessionalLogger.log(f"  Volume: {volume:.3f} | R:R: {rr_ratio:.2f}", "TRADE", "ENGINE")
        ProfessionalLogger.log(f"  Confidence: {confidence:.1%} | Agreement: {model_agreement.get('agreement', 1):.0%}", "TRADE", "ENGINE")
        ProfessionalLogger.log("=" * 50, "TRADE", "ENGINE")
        
        return True
    
    def check_pending_signals(self, df_current):
        """Check and execute pending signals"""
        if not self.pending_signals:
            return
        
        current_price = df_current.iloc[-1]['close']
        signals_to_remove = []
        
        for signal_id, pending in self.pending_signals.items():
            age = (datetime.now() - pending['created_at']).total_seconds() / 3600
            if age > 1:
                signals_to_remove.append(signal_id)
                ProfessionalLogger.log(f"Pending signal expired (age: {age:.1f}h)", "WARNING", "ENGINE")
                continue
            
            optimal_entry = pending['optimal_entry']
            distance = abs(current_price - optimal_entry)
            tolerance = pending['atr'] * 0.2
            
            if distance <= tolerance:
                ProfessionalLogger.log(f"🎯 Optimal entry reached! Executing pending signal...", "SUCCESS", "ENGINE")
                
                success = self.execute_enhanced_trade(
                    pending['signal'],
                    pending['confidence'],
                    df_current,
                    pending['features'],
                    pending['model_agreement']
                )
                
                if success:
                    signals_to_remove.append(signal_id)
        
        for signal_id in signals_to_remove:
            del self.pending_signals[signal_id]
    
    def check_and_manage_positions(self):
        """Check and manage open positions"""
        if not self.connected:
            return
        
        positions = mt5.positions_get(symbol=Config.SYMBOL)
        if not positions:
            return
        
        for position in positions:
            if position.magic != Config.MAGIC_NUMBER:
                continue
            
            current_price = position.price_current
            open_price = position.price_open
            sl = position.sl
            tp = position.tp
            
            if sl != 0 and tp != 0:
                profit_pips = abs(current_price - open_price) / 0.01
                
                if profit_pips > abs(open_price - sl) / 0.01:
                    new_sl = open_price
                    
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
                            ProfessionalLogger.log(f"Position #{position.ticket} SL moved to breakeven", "TRADE", "ENGINE")
    
    def run_periodic_tasks(self):
        """Run periodic maintenance tasks"""
        self.iteration += 1
        
        self.check_closed_positions()
        self.order_executor.check_pending_orders()
        self.check_and_manage_positions()
        
        if self.pending_signals:
            df = self.get_historical_data(bars=100)
            if df is not None:
                self.check_pending_signals(df)
        
        if (self.iteration % 100 == 0 or 
            (datetime.now() - self.last_performance_report).total_seconds() > 3600):
            
            self._print_performance_report()
            self.last_performance_report = datetime.now()
            
            account = mt5.account_info()
            if account:
                daily_pnl_pct = (self.risk_manager.daily_pnl / account.balance) * 100
                ProfessionalLogger.log(f"Risk Status | Daily P/L: {daily_pnl_pct:.1f}% | "
                          f"Consecutive Losses: {self.risk_manager.consecutive_losses}", "RISK", "ENGINE")
        
        if self.model and self.model.should_retrain():
            ProfessionalLogger.log("🔄 Retraining ensemble model...", "LEARN", "ENGINE")
            df = self.get_historical_data()
            if df is not None:
                success = self.model.train(df)
                if success:
                    ProfessionalLogger.log("✅ Model retraining successful", "SUCCESS", "ENGINE")
                else:
                    ProfessionalLogger.log("❌ Model retraining failed", "ERROR", "ENGINE")
    
    def _print_performance_report(self):
        """Print performance report"""
        stats = self.trade_memory.get_statistical_summary()
        if not stats:
            return
        
        ProfessionalLogger.log("=" * 60, "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"📊 PROFESSIONAL PERFORMANCE REPORT", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Total Trades: {stats['total_trades']}", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Win Rate: {stats['win_rate']:.1%}", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Total Profit: ${stats['total_profit']:.2f}", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Avg Profit: ${stats['mean_profit']:.2f}", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Profit Std Dev: ${stats['std_profit']:.2f}", "PERFORMANCE", "ENGINE")
        ProfessionalLogger.log(f"Profit Factor: {stats['profit_factor']:.2f}", "PERFORMANCE", "ENGINE")
        
        if 'var_95' in stats:
            ProfessionalLogger.log(f"VaR (95%): ${stats['var_95']:.2f}", "PERFORMANCE", "ENGINE")
            ProfessionalLogger.log(f"CVaR (95%): ${stats['cvar_95']:.2f}", "PERFORMANCE", "ENGINE")
            ProfessionalLogger.log(f"Omega Ratio: {stats['omega_ratio']:.2f}", "PERFORMANCE", "ENGINE")
        
        if 'skewness' in stats:
            ProfessionalLogger.log(f"Skewness: {stats['skewness']:.3f}", "PERFORMANCE", "ENGINE")
            ProfessionalLogger.log(f"Kurtosis: {stats['kurtosis']:.3f}", "PERFORMANCE", "ENGINE")
        
        ProfessionalLogger.log("=" * 60, "PERFORMANCE", "ENGINE")
    
    # ==========================================
    # BACKTESTING METHODS
    # ==========================================
    
    def _perform_statistical_analysis(self, data):
        """Perform statistical analysis on market data"""
        returns = data['close'].pct_change().dropna()
        
        stats = self.stat_analyzer.analyze_return_distribution(returns.values)
        
        ProfessionalLogger.log("Return Distribution Analysis:", "STATISTICS", "ENGINE")
        ProfessionalLogger.log(f"  Mean: {stats['mean']:.6f}", "STATISTICS", "ENGINE")
        ProfessionalLogger.log(f"  Std: {stats['std']:.6f}", "STATISTICS", "ENGINE")
        ProfessionalLogger.log(f"  Skewness: {stats['skewness']:.3f}", "STATISTICS", "ENGINE")
        ProfessionalLogger.log(f"  Kurtosis: {stats['kurtosis']:.3f}", "STATISTICS", "ENGINE")
        ProfessionalLogger.log(f"  JB p-value: {stats['jarque_bera_pvalue']:.6f}", "STATISTICS", "ENGINE")
        
        if stats['jarque_bera_pvalue'] < 0.05:
            ProfessionalLogger.log("  WARNING: Returns are NOT normally distributed", "WARNING", "ENGINE")
        else:
            ProfessionalLogger.log("  Returns appear normally distributed", "STATISTICS", "ENGINE")
    
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
            ProfessionalLogger.log("⚠ Algo trading disabled!", "ERROR", "ENGINE")
            return False
        
        symbol_info = mt5.symbol_info(Config.SYMBOL)
        if symbol_info is None:
            ProfessionalLogger.log(f"Symbol {Config.SYMBOL} not found", "ERROR", "ENGINE")
            return False
        
        if not symbol_info.visible:
            mt5.symbol_select(Config.SYMBOL, True)
        
        self.connected = True
        return True
    
    def _load_historical_data_for_backtest(self):
        """Load historical data for backtesting"""
        if not self.connected:
            ProfessionalLogger.log("Connecting to MT5 for backtest data...", "INFO", "ENGINE")
            if not self.connect_mt5():
                return None
        
        try:
            # Try to get larger dataset for backtesting
            rates = mt5.copy_rates_from_pos(Config.SYMBOL, Config.TIMEFRAME, 0, 5000)
            if rates is None or len(rates) == 0:
                ProfessionalLogger.log("Failed to fetch rates from MT5", "ERROR", "ENGINE")
                return None
            
            data = pd.DataFrame(rates)
            ProfessionalLogger.log(f"Loaded {len(data)} bars for backtesting", "SUCCESS", "ENGINE")
            
            # Calculate features
            data = self.feature_engine.calculate_features(data)
            
            # Check data quality
            quality_score, _ = ProfessionalDataQualityChecker.check_data_quality(data)
            ProfessionalLogger.log(f"Backtest data quality: {quality_score:.2%}", "DATA", "ENGINE")
            
            return data
            
        except Exception as e:
            ProfessionalLogger.log(f"Error loading backtest data: {str(e)}", "ERROR", "ENGINE")
            return None
    
    def run_comprehensive_backtest(self):
        """Run comprehensive backtesting"""
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        ProfessionalLogger.log("RUNNING COMPREHENSIVE BACKTESTING", "BACKTEST", "ENGINE")
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        
        # Initialize model before backtesting
        if self.model is None:
            self.model = ProfessionalEnsemble(self.trade_memory, self.feature_engine)
        
        data = self._load_historical_data_for_backtest()
        if data is None or len(data) < 1000:
            ProfessionalLogger.log(f"Insufficient data for backtesting ({len(data) if data else 0} bars)", "ERROR", "ENGINE")
            return False
        
        ProfessionalLogger.log("Step 1: Statistical Analysis", "STATISTICS", "ENGINE")
        self._perform_statistical_analysis(data)
        
        ProfessionalLogger.log("Step 2: Train-Test Split", "BACKTEST", "ENGINE")
        train_size = int(len(data) * 0.7)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        ProfessionalLogger.log(f"Training on {len(train_data)} bars, testing on {len(test_data)} bars", "BACKTEST", "ENGINE")
        
        # Train the model
        ProfessionalLogger.log("Training model on historical data...", "LEARN", "ENSEMBLE")
        train_success = self.model.train(train_data)
        
        if not train_success:
            ProfessionalLogger.log("Model training failed", "ERROR", "ENGINE")
            return False
        
        ProfessionalLogger.log("Step 3: Simulated Backtest", "BACKTEST", "ENGINE")
        
        # Simple backtest simulation
        test_results = self._simplified_backtest(test_data)
        
        ProfessionalLogger.log("Step 4: Strategy Evaluation", "BACKTEST", "ENGINE")
        passes = self._evaluate_backtest_results(test_results)
        
        if passes:
            ProfessionalLogger.log("✅ Strategy PASSES backtest criteria", "SUCCESS", "ENGINE")
            return True
        else:
            ProfessionalLogger.log("❌ Strategy FAILS backtest criteria", "ERROR", "ENGINE")
            return False
    
    def _simplified_backtest(self, test_data):
        """Simplified backtest using the trained model"""
        trades = []
        equity = 10000
        position = None
        entry_price = None
        
        for i in range(100, len(test_data) - 5):  # Start from 100 to have enough features
            current_slice = test_data.iloc[:i+1]
            
            try:
                signal, confidence, features, model_details = self.model.predict(current_slice)
                
                if signal is None:
                    continue
                
                current_price = current_slice['close'].iloc[-1]
                
                # Simulate trading logic
                if signal == 1 and position != 'long':
                    if position == 'short':
                        # Close short
                        profit = (entry_price - current_price) * 10000
                        trades.append({'pnl': profit, 'type': 'short_close'})
                        equity += profit
                        position = None
                    
                    # Open long
                    position = 'long'
                    entry_price = current_price
                    trades.append({'action': 'open_long', 'price': current_price})
                    
                elif signal == 0 and position != 'short':
                    if position == 'long':
                        # Close long
                        profit = (current_price - entry_price) * 10000
                        trades.append({'pnl': profit, 'type': 'long_close'})
                        equity += profit
                        position = None
                    
                    # Open short
                    position = 'short'
                    entry_price = current_price
                    trades.append({'action': 'open_short', 'price': current_price})
                    
            except Exception as e:
                ProfessionalLogger.log(f"Backtest error at index {i}: {str(e)}", "WARNING", "ENGINE")
                continue
        
        # Close final position
        if position:
            last_price = test_data['close'].iloc[-1]
            if position == 'long':
                profit = (last_price - entry_price) * 10000
                trades.append({'pnl': profit, 'type': 'long_close'})
                equity += profit
            elif position == 'short':
                profit = (entry_price - last_price) * 10000
                trades.append({'pnl': profit, 'type': 'short_close'})
                equity += profit
        
        # Calculate metrics
        pnl_trades = [t for t in trades if 'pnl' in t]
        if not pnl_trades:
            return None
        
        pnls = [t['pnl'] for t in pnl_trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        
        return {
            'total_trades': len(pnl_trades),
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': len(wins) / len(pnl_trades) if pnl_trades else 0,
            'total_profit': sum(pnls),
            'avg_win': np.mean(wins) if wins else 0,
            'avg_loss': np.mean(losses) if losses else 0,
            'profit_factor': abs(sum(wins) / sum(losses)) if sum(losses) != 0 else float('inf'),
            'final_equity': equity
        }
    
    def _evaluate_backtest_results(self, results):
        """Evaluate if strategy passes all criteria"""
        if not results:
            ProfessionalLogger.log("No backtest results", "ERROR", "ENGINE")
            return False
        
        passes = (
            results['total_trades'] >= 20 and
            results['win_rate'] >= 0.45 and
            results['total_profit'] > 0 and
            results['profit_factor'] > 1.3
        )
        
        ProfessionalLogger.log("Backtest Results:", "BACKTEST", "ENGINE")
        ProfessionalLogger.log(f"  Total Trades: {results['total_trades']}", "BACKTEST", "ENGINE")
        ProfessionalLogger.log(f"  Win Rate: {results['win_rate']:.1%}", "BACKTEST", "ENGINE")
        ProfessionalLogger.log(f"  Total Profit: ${results['total_profit']:.2f}", "BACKTEST", "ENGINE")
        ProfessionalLogger.log(f"  Profit Factor: {results['profit_factor']:.2f}", "BACKTEST", "ENGINE")
        ProfessionalLogger.log(f"  Final Equity: ${results['final_equity']:.2f}", "BACKTEST", "ENGINE")
        
        return passes
    
    def run_live_trading(self):
        """Run live trading after successful backtest"""
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        ProfessionalLogger.log("STARTING LIVE TRADING", "TRADE", "ENGINE")
        ProfessionalLogger.log("=" * 70, "INFO", "ENGINE")
        
        # Make sure we're connected
        if not self.connected:
            if not self.connect_mt5():
                ProfessionalLogger.log("Failed to connect to MT5", "ERROR", "ENGINE")
                return
        
        # Initialize model if not already done
        if self.model is None:
            self.model = ProfessionalEnsemble(self.trade_memory, self.feature_engine)
        
        # Load recent data for training
        ProfessionalLogger.log("Loading recent data from MT5 for initial training...", "INFO", "ENGINE")
        
        # Fetch significant history from MT5 for training
        training_bars = 5000 # Enough for ML
        data = self.get_historical_data(bars=training_bars)
        
        if data is not None and len(data) > Config.TRAINING_MIN_SAMPLES:
            ProfessionalLogger.log(f"Training on {len(data)} recent bars from MT5", "LEARN", "ENSEMBLE")
            success = self.model.train(data)
            if not success:
                ProfessionalLogger.log("Initial training failed", "WARNING", "ENGINE")
        else:
            ProfessionalLogger.log("Insufficient data for training, using model as-is", "WARNING", "ENGINE")
        
        ProfessionalLogger.log(
            f"🎯 Trading {Config.SYMBOL} {Config.TIMEFRAMES} | "
            f"Min Confidence: {Config.MIN_CONFIDENCE:.0%} | "
            f"Min Agreement: {Config.MIN_ENSEMBLE_AGREEMENT:.0%}", "INFO", "ENGINE"
        )
        
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
                
                # Get multi-timeframe data
                mtf_data = {}
                if Config.MULTI_TIMEFRAME_ENABLED:
                    mtf_data = self.mtf_analyzer.get_multi_timeframe_data()
                
                # Get model prediction
                signal, confidence, features, model_details = self.model.predict(df_current)
                
                if signal is None:
                    # No valid signal
                    if self.iteration % 30 == 0:
                        ProfessionalLogger.log(f"Waiting for signal... | Price: {df_current['close'].iloc[-1]:.2f} | Positions: {self.get_current_positions()}", "INFO", "ENGINE")
                    time.sleep(60)
                    continue
                
                # Check multi-timeframe alignment
                mtf_signal = None
                mtf_confidence = 0
                if mtf_data:
                    mtf_signal, mtf_confidence = self.mtf_analyzer.analyze_timeframe_alignment(
                        self.model, mtf_data
                    )
                
                final_signal = signal
                final_confidence = confidence
                
                if mtf_signal is not None and mtf_confidence > 0:
                    if mtf_signal == signal:
                        final_confidence = (confidence + mtf_confidence) / 2
                        ProfessionalLogger.log(f"✅ Multi-timeframe alignment confirmed ({mtf_confidence:.0%})", "SUCCESS", "ENGINE")
                    else:
                        final_confidence = confidence * 0.7
                        ProfessionalLogger.log(f"⚠ Multi-timeframe conflict", "WARNING", "ENGINE")
                
                # Check model agreement
                model_agreement = {'agreement': 1.0, 'details': model_details}
                if model_details:
                    predictions = [m['prediction'] for m in model_details.values() 
                                 if m['prediction'] != -1]
                    if predictions:
                        agreement = predictions.count(signal) / len(predictions)
                        model_agreement['agreement'] = agreement
                
                # Log status
                if self.iteration % 10 == 0:
                    signal_type = "BUY" if final_signal == 1 else "SELL"
                    status_msg = (f"Price: {df_current['close'].iloc[-1]:.2f} | Signal: {signal_type} | "
                                 f"Conf: {final_confidence:.1%} | "
                                 f"Agreement: {model_agreement['agreement']:.0%} | "
                                 f"Positions: {self.get_current_positions()}")
                    ProfessionalLogger.log(status_msg, "INFO", "ENGINE")
                
                # Execute trade if conditions are met
                if (final_confidence >= Config.MIN_CONFIDENCE and 
                    model_agreement['agreement'] >= Config.MIN_ENSEMBLE_AGREEMENT):
                    
                    signal_type = "BUY" if final_signal == 1 else "SELL"
                    ProfessionalLogger.log(f"🎯 High-confidence {signal_type} signal detected!", "SUCCESS", "ENGINE")
                    
                    self.execute_enhanced_trade(
                        final_signal, 
                        final_confidence, 
                        df_current, 
                        features, 
                        model_agreement
                    )
                
                time.sleep(60)  # Wait 1 minute before next iteration
                
        except KeyboardInterrupt:
            ProfessionalLogger.log("\nShutdown requested by user", "WARNING", "ENGINE")
        except Exception as e:
            ProfessionalLogger.log(f"Unexpected error: {str(e)}", "ERROR", "ENGINE")
            import traceback
            traceback.print_exc()
        finally:
            self._print_performance_report()
            mt5.shutdown()
            ProfessionalLogger.log("Disconnected from MT5", "INFO", "ENGINE")
# ==========================================
# MAIN FUNCTION
# ==========================================
def main():
    """Main entry point"""
    engine = ProfessionalTradingEngine()
    engine.run()

if __name__ == "__main__":
    from datetime import datetime, timedelta
    import time
    import json
    import os
    import sys
    
    main()