import tkinter as tk
from tkinter import ttk, scrolledtext
import subprocess
import threading
import queue
import os
import sys
import signal
import re

# ==========================================
# CONFIGURATION
# ==========================================
# Define the scripts you want to control here
SCRIPTS = {
    "Market Data": "logger.py",
    "Trade Logger": "trade_logger.py",
    "Trading Bot": "main.py"
}

# Define colors for the ANSI parser
ANSI_COLORS = {
    '91': '#ff5555', # Red
    '92': '#50fa7b', # Green
    '93': '#f1fa8c', # Yellow
    '94': '#bd93f9', # Blue
    '95': '#ff79c6', # Magenta
    '96': '#8be9fd', # Cyan
    '97': '#f8f8f2', # White
    '0':  '#f8f8f2'  # Reset (White)
}

class ProcessHandler:
    """
    Manages a single subprocess: starting, stopping, and capturing output.
    """
    def __init__(self, name, script_name, log_queue, status_callback):
        self.name = name
        self.script_name = script_name
        self.log_queue = log_queue
        self.status_callback = status_callback
        self.process = None
        self.thread = None
        self.stop_event = threading.Event()

    def start(self):
        if self.process and self.process.poll() is None:
            return # Already running

        # Start the subprocess with unbuffered output (-u)
        # creationflags=subprocess.CREATE_NO_WINDOW hides the console window on Windows
        startupinfo = None
        if os.name == 'nt':
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW

        try:
            self.process = subprocess.Popen(
                [sys.executable, "-u", self.script_name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                universal_newlines=True,
                startupinfo=startupinfo
            )
            self.stop_event.clear()
            self.status_callback(self.name, True)
            
            # Start reader thread
            self.thread = threading.Thread(target=self._read_output, daemon=True)
            self.thread.start()
            
            self.log_queue.put((self.name, f"SYSTEM: Started {self.script_name} (PID: {self.process.pid})\n", "92"))
        except Exception as e:
            self.log_queue.put((self.name, f"SYSTEM: Failed to start {self.script_name}: {e}\n", "91"))

    def stop(self):
        if self.process:
            self.log_queue.put((self.name, f"SYSTEM: Stopping {self.script_name}...\n", "93"))
            self.process.terminate() # Send SIGTERM
            self.log_queue.put((self.name, f"SYSTEM: Process stopped {self.script_name}...\n", "91"))
            self.process = None
            self.status_callback(self.name, False)

    def _read_output(self):
        """Reads stdout/stderr from the process line by line"""
        try:
            # We read stdout. Assuming the scripts print logs to stdout.
            for line in iter(self.process.stdout.readline, ''):
                if line:
                    self.log_queue.put((self.name, line, None)) # None means "parse ANSI inside"
                elif self.process.poll() is not None:
                    break
        except Exception:
            pass
        finally:
            self.status_callback(self.name, False)

class DashboardApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MT5 Trading Bot - Master Control")
        self.root.geometry("1100x700")
        self.root.configure(bg="#1e1e1e")

        # Data Structures
        self.handlers = {}
        self.log_queue = queue.Queue()
        self.tabs = {}
        self.status_indicators = {}

        # Style Config
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#1e1e1e", borderwidth=0)
        style.configure("TNotebook.Tab", background="#333", foreground="white", padding=[10, 5])
        style.map("TNotebook.Tab", background=[("selected", "#555")])
        style.configure("TFrame", background="#1e1e1e")

        # --- LAYOUT ---
        # Left Sidebar (Controls)
        sidebar = tk.Frame(root, bg="#252526", width=250)
        sidebar.pack(side=tk.LEFT, fill=tk.Y)
        sidebar.pack_propagate(False)

        # Title in Sidebar
        tk.Label(sidebar, text="PROCESSES", bg="#252526", fg="#888", font=("Segoe UI", 10, "bold")).pack(pady=(20, 10), anchor="w", padx=15)

        # Create Controls for each script
        for name, script in SCRIPTS.items():
            self.create_process_control(sidebar, name, script)

        # Visualizer Button (Special Case - External Window)
        tk.Label(sidebar, text="TOOLS", bg="#252526", fg="#888", font=("Segoe UI", 10, "bold")).pack(pady=(30, 10), anchor="w", padx=15)
        btn_viz = tk.Button(sidebar, text="Open Visualizer", bg="#444", fg="white", relief="flat",
                            command=self.launch_visualizer, font=("Segoe UI", 10))
        btn_viz.pack(fill=tk.X, padx=15, pady=5)

        # Right Area (Logs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH, padx=10, pady=10)

        # Create Tabs for each script
        for name in SCRIPTS.keys():
            self.create_log_tab(name)

        # Start Queue Polling
        self.root.after(100, self.process_log_queue)

    def create_process_control(self, parent, name, script):
        frame = tk.Frame(parent, bg="#252526")
        frame.pack(fill=tk.X, padx=15, pady=8)

        # Status Dot
        canvas = tk.Canvas(frame, width=12, height=12, bg="#252526", highlightthickness=0)
        indicator = canvas.create_oval(2, 2, 10, 10, fill="#ff5555", outline="") # Start red
        canvas.pack(side=tk.LEFT)
        self.status_indicators[name] = (canvas, indicator)

        # Label
        tk.Label(frame, text=name, bg="#252526", fg="white", font=("Segoe UI", 11)).pack(side=tk.LEFT, padx=10)

        # Buttons Frame
        btn_frame = tk.Frame(parent, bg="#252526")
        btn_frame.pack(fill=tk.X, padx=15, pady=(0, 10))

        # Start Handler
        handler = ProcessHandler(name, script, self.log_queue, self.update_status)
        self.handlers[name] = handler

        # Start Button
        btn_start = tk.Button(btn_frame, text="Start", bg="#28a745", fg="white", relief="flat", width=8,
                              command=lambda: handler.start())
        btn_start.pack(side=tk.LEFT, padx=(22, 5))

        # Stop Button
        btn_stop = tk.Button(btn_frame, text="Stop", bg="#dc3545", fg="white", relief="flat", width=8,
                             command=lambda: handler.stop())
        btn_stop.pack(side=tk.LEFT)

    def create_log_tab(self, name):
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text=name)
        
        # Scrolled Text
        text_widget = scrolledtext.ScrolledText(tab_frame, bg="#1e1e1e", fg="#d4d4d4", 
                                                font=("Consolas", 10), state='disabled', borderwidth=0)
        text_widget.pack(expand=True, fill='both')
        
        # Configure Tags based on ANSI_COLORS
        for code, color in ANSI_COLORS.items():
            text_widget.tag_config(code, foreground=color)
        
        self.tabs[name] = text_widget

    def update_status(self, name, is_running):
        """Updates the status dot color"""
        canvas, indicator = self.status_indicators[name]
        color = "#50fa7b" if is_running else "#ff5555" # Green if running, Red if stopped
        canvas.itemconfig(indicator, fill=color)

    def launch_visualizer(self):
        """Launches visualizer in a new external terminal window"""
        script = "visualize.py"
        if os.name == 'nt':
            os.system(f'start cmd /k python {script}')
        else:
            # Linux/Mac (adjust terminal app as needed, e.g., gnome-terminal)
            os.system(f'x-terminal-emulator -e "python3 {script}"')

    def process_log_queue(self):
        """Reads logs from queue and inserts into GUI"""
        try:
            while True:
                name, line, direct_color = self.log_queue.get_nowait()
                text_widget = self.tabs.get(name)
                
                if text_widget:
                    text_widget.config(state='normal')
                    
                    if direct_color:
                        # System message with forced color
                        text_widget.insert(tk.END, line, direct_color)
                    else:
                        # Parse ANSI codes in the line
                        self.insert_ansi_text(text_widget, line)
                    
                    text_widget.see(tk.END)
                    text_widget.config(state='disabled')
        except queue.Empty:
            pass
        finally:
            self.root.after(50, self.process_log_queue)

    def insert_ansi_text(self, widget, text):
        """Parses ANSI color codes and inserts text with tags"""
        # Regex to find ANSI codes like \033[92m
        parts = re.split(r'(\033\[\d+m)', text)
        
        current_tag = '0' # Default white
        
        for part in parts:
            if part.startswith('\033['):
                # Extract code (e.g., '92')
                code = part[2:-1]
                if code in ANSI_COLORS:
                    current_tag = code
            else:
                if part:
                    widget.insert(tk.END, part, current_tag)

if __name__ == "__main__":
    # Ensure database folder exists so scripts don't crash on start
    os.makedirs("database", exist_ok=True)
    
    root = tk.Tk()
    app = DashboardApp(root)
    
    # Handle window close to kill processes
    def on_close():
        for handler in app.handlers.values():
            if handler.process:
                handler.process.terminate()
        root.destroy()
        
    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()