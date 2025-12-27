import os
import json
import time
import threading
from datetime import datetime, timedelta
import MetaTrader5 as mt5

# ==========================================
# CUSTOM LOGGER
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
            'DATA': Logger.COLORS['WHITE']
        }
        color = colors.get(level, Logger.COLORS['RESET'])
        print(f"{timestamp} [{color}{level}{Logger.COLORS['RESET']}] {message}", flush=True)

# ==========================================
# CONFIGURATION
# ==========================================
class TradeConfig:
    SYMBOL = "XAUUSD" # Set to None to fetch ALL symbols
    ROOT_DIR = "trades"
    ACTIVE_DIR = "trades/active"
    HISTORY_DIR = "trades/history"
    UPDATE_INTERVAL = 5  # Update active trades every 5 seconds
    HISTORY_SYNC_INTERVAL = 60 # Sync closed history every 60 seconds

class TradeLogger:
    def __init__(self):
        self.root_dir = TradeConfig.ROOT_DIR
        self.active_path = os.path.join(self.root_dir, TradeConfig.ACTIVE_DIR)
        self.history_path = os.path.join(self.root_dir, TradeConfig.HISTORY_DIR)
        
        # Ensure base directories exist
        os.makedirs(self.active_path, exist_ok=True)
        os.makedirs(self.history_path, exist_ok=True)

    def _convert_to_dict(self, trade_tuple):
        """Converts MT5 tuple/object to a standard dictionary"""
        if trade_tuple is None:
            return None
            
        trade_dict = trade_tuple._asdict()
        
        # Convert any non-serializable objects (like timestamps)
        for key, value in trade_dict.items():
            if key == 'time' or key == 'time_msc' or key == 'time_setup':
                # Convert timestamp to readable string
                trade_dict[f'{key}_str'] = datetime.fromtimestamp(value).strftime('%Y-%m-%d %H:%M:%S')
        
        return trade_dict

    def save_active_positions(self):
        """Fetches current open positions and overwrites the snapshot file"""
        try:
            # Fetch positions
            if TradeConfig.SYMBOL:
                positions = mt5.positions_get(symbol=TradeConfig.SYMBOL)
            else:
                positions = mt5.positions_get()

            if positions is None:
                positions = []

            # Convert to list of dicts
            data = [self._convert_to_dict(pos) for pos in positions]
            
            # Define path: database/trades/active/active_positions.json
            file_path = os.path.join(self.active_path, "active_positions.json")
            
            # Overwrite file
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
                
            return len(data)
            
        except Exception as e:
            Logger.log(f"Error saving active positions: {e}", "ERROR")
            return 0

    def sync_closed_history(self, days_back=30):
        """
        Fetches closed deals (history) and organizes them into folders by date.
        """
        try:
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days_back)
            
            # Fetch history deals
            if TradeConfig.SYMBOL:
                deals = mt5.history_deals_get(from_date, to_date, group=TradeConfig.SYMBOL)
            else:
                deals = mt5.history_deals_get(from_date, to_date)

            if deals is None or len(deals) == 0:
                return 0

            # Convert to nice dictionaries
            deals_data = [self._convert_to_dict(d) for d in deals]
            
            # Filter: We only care about entries (0) and exits (1), usually type 1 (DEAL_ENTRY_OUT) is a closed trade
            # But saving all deals is safer for auditing.
            
            # Group by Date (Year-Month-Day)
            # We use a dictionary to group them in memory first
            grouped_deals = {}
            
            for deal in deals_data:
                # Get the date from the deal timestamp
                deal_time = datetime.fromtimestamp(deal['time'])
                date_key = deal_time.date() # Object like 2024-05-20
                
                if date_key not in grouped_deals:
                    grouped_deals[date_key] = []
                grouped_deals[date_key].append(deal)

            saved_count = 0
            
            # Save into folders
            for date_key, group_list in grouped_deals.items():
                year = str(date_key.year)
                month = f"{date_key.month:02d}"
                day = f"{date_key.day:02d}"
                
                # Path: database/trades/history/2024/05/20/closed_deals.json
                dir_path = os.path.join(self.history_path, year, month, day)
                os.makedirs(dir_path, exist_ok=True)
                
                file_path = os.path.join(dir_path, "closed_deals.json")
                
                # We overwrite the daily file with the full data for that day to ensure updates
                with open(file_path, 'w') as f:
                    json.dump(group_list, f, indent=2)
                
                saved_count += 1
                
            return len(deals_data)

        except Exception as e:
            Logger.log(f"Error syncing history: {e}", "ERROR")
            return 0

    def run(self):
        if not mt5.initialize():
            Logger.log("MT5 Initialization Failed", "ERROR")
            return

        Logger.log(f"Trade Logger Started for {TradeConfig.SYMBOL or 'ALL'}", "INFO")
        Logger.log("Monitoring active positions and history...", "INFO")

        last_history_sync = 0
        
        try:
            while True:
                # 1. Update Active Positions (Fast Loop)
                active_count = self.save_active_positions()
                
                # 2. Update Closed History (Slower Interval)
                now = time.time()
                if now - last_history_sync > TradeConfig.HISTORY_SYNC_INTERVAL:
                    # Sync last 5 days just to be sure we catch recent closes
                    deals_count = self.sync_closed_history(days_back=5)
                    last_history_sync = now
                    
                    if deals_count > 0:
                         # Use TRADE color (Magenta) for trade-related logs
                        Logger.log(f"Synced History: {deals_count} deals processed.", "TRADE")

                # Log active status periodically
                if active_count > 0:
                    # Optional: Don't spam active count every 5 seconds unless needed
                    # Logger.log(f"Active Positions: {active_count}", "TRADE")
                    pass

                time.sleep(TradeConfig.UPDATE_INTERVAL)

        except KeyboardInterrupt:
            Logger.log("Trade Logger Stopped.", "WARNING")
        finally:
            mt5.shutdown()

if __name__ == "__main__":
    logger = TradeLogger()
    logger.run()