import os
import json
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta

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
DATABASE_ROOT = "database"

def get_single_date_input(label):
    """Helper to get a validated date from user"""
    print(f"\n--- {label} ---")
    while True:
        try:
            date_str = input(f"Enter {label} (YYYY-MM-DD): ").strip()
            # Try parsing to ensure validity
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            return dt
        except ValueError:
            Logger.log("Invalid format. Please use YYYY-MM-DD (e.g., 2024-10-25)", "ERROR")

def load_data_range(start_date, end_date):
    """Iterates through all dates in range and combines data"""
    combined_data = []
    
    # Calculate total days for iteration
    delta = end_date - start_date
    days_count = delta.days + 1
    
    Logger.log(f"Scanning database from {start_date.date()} to {end_date.date()}...", "DATA")
    
    files_found = 0
    
    for i in range(days_count):
        current_date = start_date + timedelta(days=i)
        
        year = str(current_date.year)
        month = f"{current_date.month:02d}"
        day = f"{current_date.day:02d}"
        
        file_path = os.path.join(DATABASE_ROOT, year, month, day, "data.json")
        
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    day_data = json.load(f)
                    combined_data.extend(day_data)
                    files_found += 1
            except Exception as e:
                Logger.log(f"Corrupt file at {file_path}: {e}", "ERROR")
        else:
            # Optional: Log missing days if you want verbose output
            # Logger.log(f"No data for {current_date.date()}", "WARNING")
            pass

    if not combined_data:
        return None

    Logger.log(f"Successfully loaded {files_found} daily files.", "SUCCESS")
    
    df = pd.DataFrame(combined_data)
    # Convert time and sort
    df['time'] = pd.to_datetime(df['time'])
    df = df.sort_values('time').drop_duplicates(subset=['time'])
    
    return df

def plot_mt5_style(df, start_date, end_date):
    """Creates an interactive Plotly chart resembling MT5"""
    Logger.log("Generating interactive chart...", "INFO")

    # Create figure with secondary y-axis (if we wanted volume, but sticking to overlays for now)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                        vertical_spacing=0.03, subplot_titles=('Price Action', 'RSI'),
                        row_width=[0.2, 0.7])

    # -------------------------------------------------------
    # 1. CANDLESTICK CHART (Main)
    # -------------------------------------------------------
    fig.add_trace(go.Candlestick(
        x=df['time'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='XAUUSD',
        increasing_line_color='#26a69a', # MT5 Greenish
        decreasing_line_color='#ef5350'  # MT5 Reddish
    ), row=1, col=1)

    # -------------------------------------------------------
    # 2. MOVING AVERAGES (Overlays)
    # -------------------------------------------------------
    # SMA 20 (Yellow)
    if 'sma_20' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['sma_20'],
            mode='lines', name='SMA 20',
            line=dict(color='yellow', width=1.5)
        ), row=1, col=1)

    # SMA 50 (Cyan)
    if 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['sma_50'],
            mode='lines', name='SMA 50',
            line=dict(color='cyan', width=1.5)
        ), row=1, col=1)

    # SMA 100 (Red)
    if 'sma_100' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['sma_100'],
            mode='lines', name='SMA 100',
            line=dict(color='magenta', width=2)
        ), row=1, col=1)

    # -------------------------------------------------------
    # 3. BOLLINGER BANDS
    # -------------------------------------------------------
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        # Upper Band
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['bb_upper'],
            mode='lines', name='BB Upper',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'),
            showlegend=False
        ), row=1, col=1)

        # Lower Band
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['bb_lower'],
            mode='lines', name='Bollinger Bands',
            line=dict(color='rgba(255, 255, 255, 0.3)', width=1, dash='dot'),
            fill='tonexty', # Fills area between Upper and Lower
            fillcolor='rgba(255, 255, 255, 0.05)'
        ), row=1, col=1)

    # -------------------------------------------------------
    # 4. RSI (Subplot)
    # -------------------------------------------------------
    if 'rsi' in df.columns:
        fig.add_trace(go.Scatter(
            x=df['time'], y=df['rsi'],
            mode='lines', name='RSI (14)',
            line=dict(color='#A68CFF', width=1.5)
        ), row=2, col=1)
        
        # Add 70/30 Levels
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)

    # -------------------------------------------------------
    # LAYOUT STYLING (Dark Mode)
    # -------------------------------------------------------
    range_title = f"{start_date.date()} to {end_date.date()}"
    
    fig.update_layout(
        title=f"Market Data Visualizer - {range_title}",
        template="plotly_dark", # This gives the MT5 dark theme look
        xaxis_rangeslider_visible=False, # Hide the bottom slider for cleaner look
        height=800,
        hovermode="x unified", # Shows all data points on hover line
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    # Update axes to remove grid lines for cleaner look (optional, MT5 has grids)
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#333')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#333')
    
    fig.show()
    Logger.log("Chart opened in browser.", "SUCCESS")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    print("\n" + "="*40)
    print("   MT5 DATABASE VISUALIZER   ")
    print("="*40 + "\n")
    
    # 1. Get User Selection
    start_dt = get_single_date_input("Start Date")
    end_dt = get_single_date_input("End Date")
    
    if start_dt > end_dt:
        Logger.log("Start date cannot be after End date. Swapping them...", "WARNING")
        start_dt, end_dt = end_dt, start_dt

    # 2. Load Data
    df = load_data_range(start_dt, end_dt)
    
    # 3. Plot
    if df is not None:
        Logger.log(f"Processing {len(df)} records...", "INFO")
        plot_mt5_style(df, start_dt, end_dt)
    else:
        Logger.log("No data found for the selected range.", "ERROR")