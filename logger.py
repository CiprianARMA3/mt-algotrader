import os
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta
import MetaTrader5 as mt5

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
            'PERFORMANCE': Logger.COLORS['MAGENTA']
        }
        color = colors.get(level, Logger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level}{Logger.COLORS['RESET']}] {message}", flush=True)

# ==========================================
# CONFIGURATION
# ==========================================
class LoggerConfig:
    SYMBOL = "XAUUSD"
    TIMEFRAME = mt5.TIMEFRAME_M15
    ATR_PERIOD = 14
    RSI_PERIOD = 14
    ADX_PERIOD = 14
    USE_MARKET_REGIME = True
    ADX_TREND_THRESHOLD = 25
    ADX_STRONG_TREND_THRESHOLD = 40
    DATABASE_ROOT = "database"
    UPDATE_INTERVAL_SEC = 60    # Check frequency when market is open
    CLOSED_SLEEP_SEC = 300      # Check frequency when market is closed (5 mins)
    INDICATOR_BUFFER_DAYS = 7   # Days to fetch BEFORE the current month to ensure accurate indicators

class DataLogger:
    def __init__(self):
        self.symbol = LoggerConfig.SYMBOL
        self.timeframe = LoggerConfig.TIMEFRAME
        self.root_dir = LoggerConfig.DATABASE_ROOT

    def is_market_open(self):
        """Checks if the market is currently open for the symbol"""
        symbol_info = mt5.symbol_info(self.symbol)
        if symbol_info is None:
            return False
            
        # Check time since last quote 
        last_quote_time = datetime.fromtimestamp(symbol_info.time)
        time_diff = datetime.now() - last_quote_time
        
        # If last quote was more than 10 minutes ago, assume closed
        if time_diff.total_seconds() > 600: 
            return False
            
        return True

    def calculate_adx(self, df, period=LoggerConfig.ADX_PERIOD):
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

    def process_data(self, df):
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        # --- Basic Features ---
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['hl_ratio'] = (df['high'] - df['low']) / df['close']
        df['co_ratio'] = (df['close'] - df['open']) / df['open']
        df['hlc3'] = (df['high'] + df['low'] + df['close']) / 3
        
        # --- Moving Averages ---
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
            df[f'price_to_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
        
        df['ma_cross_5_20'] = df['sma_5'] - df['sma_20']
        df['ma_cross_10_50'] = df['sma_10'] - df['sma_50']
        
        # --- Volatility (ATR) ---
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['atr'] = df['tr'].rolling(LoggerConfig.ATR_PERIOD).mean()
        df['atr_percent'] = df['atr'] / df['close']
        df['volatility'] = df['returns'].rolling(20).std()
        
        # --- RSI ---
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(LoggerConfig.RSI_PERIOD).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(LoggerConfig.RSI_PERIOD).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_normalized'] = (df['rsi'] - 50) / 50
        
        # --- MACD ---
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # --- Bollinger Bands ---
        sma_20 = df['close'].rolling(20).mean()
        std_20 = df['close'].rolling(20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / sma_20
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # --- ENHANCED ADX ---
        if LoggerConfig.USE_MARKET_REGIME:
            adx_data = self.calculate_adx(df)
            df['adx'] = adx_data['adx']
            df['plus_di'] = adx_data['plus_di']
            df['minus_di'] = adx_data['minus_di']
            
            df['trend_strength'] = df['adx'] / 100
            df['trend_direction'] = np.where(df['plus_di'] > df['minus_di'], 1, -1)
            df['adx_slope'] = df['adx'].diff()
            df['di_spread'] = abs(df['plus_di'] - df['minus_di'])
            
            conditions = [
                (df['adx'] < LoggerConfig.ADX_TREND_THRESHOLD),
                (df['adx'] >= LoggerConfig.ADX_TREND_THRESHOLD) & (df['adx'] < LoggerConfig.ADX_STRONG_TREND_THRESHOLD),
                (df['adx'] >= LoggerConfig.ADX_STRONG_TREND_THRESHOLD)
            ]
            choices = [0, 1, 2]
            df['regime'] = np.select(conditions, choices, default=0)
        
        # --- Momentum ---
        for period in [3, 5, 10, 20]:
            df[f'momentum_{period}'] = df['close'].pct_change(period)
            df[f'roc_{period}'] = (df['close'] - df['close'].shift(period)) / df['close'].shift(period)
        
        # --- Time Features ---
        if 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'], unit='s')
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
            
        return df.dropna()

    def get_sync_dates(self):
        """
        Calculates the start date of the current month.
        Returns:
            start_of_month: The 1st day of the current month (for saving).
            buffer_start: Date to start fetching from (includes buffer for indicators).
        """
        now = datetime.now()
        start_of_month = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        
        # We need a buffer before the 1st of the month to calculate indicators (like EMA 100) correctly
        # for the first few days of the month.
        buffer_start = start_of_month - timedelta(days=LoggerConfig.INDICATOR_BUFFER_DAYS)
        
        return start_of_month, buffer_start

    def save_to_database(self, df):
        """Saves data to /database/year/month/day/data.json structure"""
        if df is None or df.empty:
            return []

        if not np.issubdtype(df['time'].dtype, np.datetime64):
            df['time'] = pd.to_datetime(df['time'], unit='s')

        grouped = df.groupby(df['time'].dt.date)
        saved_files = []

        for date_obj, group_df in grouped:
            year = str(date_obj.year)
            month = f"{date_obj.month:02d}"
            day = f"{date_obj.day:02d}"

            dir_path = os.path.join(self.root_dir, year, month, day)
            os.makedirs(dir_path, exist_ok=True)
            file_path = os.path.join(dir_path, "data.json")
            
            # --- JSON FIX: Convert all timestamps/dates to strings ---
            export_df = group_df.copy()
            export_df['time'] = export_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            for col in export_df.select_dtypes(include=['datetime64', 'datetimetz']).columns:
                export_df[col] = export_df[col].astype(str)
                
            data_list = export_df.to_dict(orient='records')

            try:
                with open(file_path, 'w') as f:
                    json.dump(data_list, f, indent=2)
                saved_files.append(file_path)
            except Exception as e:
                Logger.log(f"Error saving {file_path}: {e}", "ERROR")
        
        return saved_files
    
    def perform_sync(self):
        """
        Fetches data from the beginning of the month (plus buffer) 
        and updates the database. This auto-completes any missing data 
        for the current month.
        """
        start_of_month, fetch_start = self.get_sync_dates()
        now = datetime.now()
        
        # Add 1 day to 'now' to ensure we get everything up to the last second
        fetch_end = now + timedelta(days=1)
        
        # Use copy_rates_range to get the exact historical window
        rates = mt5.copy_rates_range(self.symbol, self.timeframe, fetch_start, fetch_end)
        
        if rates is None or len(rates) == 0:
            Logger.log("Failed to fetch rates (Empty response)", "ERROR")
            return []

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')

        # Process Indicators
        df_processed = self.process_data(df)
        
        # Filter: Only save data belonging to the current month
        # (The buffer data was only used for calculation)
        df_month_clean = df_processed[df_processed['time'] >= start_of_month].copy()

        # Save
        saved_paths = self.save_to_database(df_month_clean)
        return saved_paths

    def check_saved_data(self):
        """
        Checks if data for the last available market tick is saved.
        Returns: (True/False if saved, Date of last tick)
        """
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            return False, None
            
        last_tick_time = datetime.fromtimestamp(tick.time)
        year = str(last_tick_time.year)
        month = f"{last_tick_time.month:02d}"
        day = f"{last_tick_time.day:02d}"
        
        expected_path = os.path.join(self.root_dir, year, month, day, "data.json")
        return os.path.exists(expected_path), last_tick_time.date()

    def run_realtime(self):
        """Runs the logger in a real-time loop"""
        if not mt5.initialize():
            Logger.log("MT5 Initialization failed", "ERROR")
            return

        Logger.log(f"Starting Logger for {self.symbol}...", "INFO")
        Logger.log(f"Ensuring full sync for current month...", "INFO")
        
        # --- INITIAL SYNC ON STARTUP ---
        # This ensures that even before we loop, the month is complete.
        try:
            initial_paths = self.perform_sync()
            if initial_paths:
                Logger.log(f"Initial Sync Complete: Updated {len(initial_paths)} daily files.", "SUCCESS")
            else:
                Logger.log("Initial Sync: No data found or connection issue.", "WARNING")
        except Exception as e:
            Logger.log(f"Initial Sync Failed: {e}", "ERROR")

        try:
            while True:
                # 1. Check Market Status
                if not self.is_market_open():
                    is_saved, last_date = self.check_saved_data()
                    
                    if is_saved:
                        Logger.log(f"Market Closed. Data for {last_date} is up to date.", "WARNING")
                    else:
                        Logger.log(f"Market Closed. Data for {last_date} incomplete. Syncing...", "WARNING")
                        saved_paths = self.perform_sync()
                        if saved_paths:
                            Logger.log(f"Sync complete: {len(saved_paths)} files saved.", "SUCCESS")
                    
                    time.sleep(LoggerConfig.CLOSED_SLEEP_SEC)
                    continue

                # 2. Market is OPEN
                # perform_sync() fetches the whole month every time.
                # This guarantees "auto-complete" functionality in real-time.
                saved_paths = self.perform_sync()
                
                if saved_paths:
                    # We print just the number of files touched (usually just today's file changes)
                    Logger.log(f"Real-time update: {len(saved_paths)} files refreshed.", "SUCCESS")
                
                time.sleep(LoggerConfig.UPDATE_INTERVAL_SEC)

        except KeyboardInterrupt:
            Logger.log("\nLogger stopped by user.", "WARNING")
        except Exception as e:
            Logger.log(f"Unexpected error: {str(e)}", "ERROR")
        finally:
            mt5.shutdown()

if __name__ == "__main__":
    logger = DataLogger()
    logger.run_realtime()