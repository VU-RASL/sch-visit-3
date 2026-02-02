import tkinter as tk
from tkinter import ttk, messagebox
import queue
import os
import datetime
import threading
import time
import json
import math
import sv_ttk  # Modern theme for tkinter
import logging

from setup_page import SetupPage
from sensor_page import SensorPage
from session_page import SessionPage
import sensor_utils  # Import the sensor utilities module

logger = logging.getLogger(__name__)

class IISCAApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("IISCA Session Manager")
        
        # Set a fixed window size
        self.geometry("400x350")  # Width x Height
        
        # Apply modern theme
        try:
            sv_ttk.set_theme("dark")  # Use "light" or "dark"
        except:
            # Fallback to standard themes if sv_ttk is not available
            try:
                self.style = ttk.Style()
                available_themes = self.style.theme_names()
                if "clam" in available_themes:
                    self.style.theme_use("clam")
                elif "vista" in available_themes:
                    self.style.theme_use("vista")
            except:
                pass
        
        # Application data
        self.participant_data = {}
        self.sensors = {}
        self.session_data = {}
        self.current_session = 0
        self.session_types = []
        self.data_queue = queue.Queue()
        self.running = False
        self.paused = False
        self.data_thread = None
        self.ml_thread = None
        
        # Create container for frames
        self.container = tk.Frame(self)
        self.container.pack(fill="both", expand=True)
        
        # Configure grid
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        # Initialize frames
        self.frames = {}
        for F in (SetupPage, SensorPage, SessionPage):
            frame = F(self.container, self)
            self.frames[F] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        
        # Add status bar
        self.status_bar = ttk.Label(self, text="Ready", relief=tk.SUNKEN, anchor=tk.W)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Show initial frame
        self.show_frame(SetupPage)
        
        # Bind close event
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def show_frame(self, cont):
        # Import all page classes at the beginning to avoid reference errors
        from setup_page import SetupPage
        from sensor_page import SensorPage
        from session_page import SessionPage
        
        # Handle string references to frames
        if isinstance(cont, str):
            if cont == "SetupPage":
                cont = SetupPage
            elif cont == "SensorPage":
                cont = SensorPage
            elif cont == "SessionPage":
                cont = SessionPage
        
        frame = self.frames[cont]
        frame.tkraise()
        
        # Update status bar
        if cont == SetupPage:
            self.status_bar.config(text="Setup Page: Enter participant information")
        elif cont == SensorPage:
            self.status_bar.config(text="Sensor Page: Configure and connect sensors")
        elif cont == SessionPage:
            if self.current_session < len(self.session_types):
                session_type = self.session_types[self.current_session]
                
                # Update ML prediction status based on session type
                if session_type == "Standard":
                    self.frames[SessionPage].ml_prediction_active = False
        
        # Call the frame's update method
        if hasattr(frame, "update_frame"):
            frame.update_frame()
        
    def start_data_collection(self):
        """Start collecting data from sensors"""
        self.running = True
        self.paused = False
        
        # Create directory for data
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        participant_id = self.participant_data.get("participant_id", "unknown")
        
        self.data_dir = f"data_{participant_id}_{timestamp}"
        os.makedirs(self.data_dir, exist_ok=True)
        # Also make `data_dir` available to ML for correct output path
        self.participant_data["data_dir"] = self.data_dir
        
        # Save participant data
        with open(os.path.join(self.data_dir, "participant_info.json"), "w") as f:
            json.dump(self.participant_data, f, indent=4)
        
        # Start data collection thread
        self.data_thread = threading.Thread(target=self.collect_data)
        self.data_thread.daemon = True
        self.data_thread.start()
        
        # Start ML prediction thread (1 Hz)
        self.ml_thread = threading.Thread(target=self.run_ml_loop)
        self.ml_thread.daemon = True
        self.ml_thread.start()
        
        # Update status
        self.status_bar.config(text="Data collection started")
        
        # Initialize session data file immediately (atomic write)
        self.save_session_data()
    
    def save_session_data(self):
        """Atomically write session data to disk immediately.
        Uses a temporary file and os.replace to avoid partial writes.
        """
        try:
            if not hasattr(self, "data_dir") or not self.data_dir:
                return
            os.makedirs(self.data_dir, exist_ok=True)
            path = os.path.join(self.data_dir, "session_data.json")
            tmp_path = f"{path}.tmp"
            with open(tmp_path, "w") as f:
                json.dump(self.session_data, f, indent=4)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, path)
        except Exception as e:
            logger.exception("Error saving session data")

    def collect_data(self):
        """Collect data from sensors in a separate thread"""
        while self.running:
            if not self.paused:
                # Collect data from all sensors using the sensor_utils module
                sensor_utils.collect_sensor_data(self.sensors, self.data_queue, self.running, self.paused)
                
                # Process data in queue
                while not self.data_queue.empty():
                    data_item = self.data_queue.get()
                    sensor_utils.process_sensor_data(data_item, self.data_dir, self.current_session, self.session_types)
            
            # Sleep to control data collection rate
            time.sleep(0.01)
    
    def run_ml_loop(self):
        """Dedicated ML prediction loop running at ~1 Hz"""
        next_tick = time.time()
        while self.running:
            # Align to 1 Hz
            next_tick += 1.0
            
            try:
                if (not self.paused and
                    self.current_session < len(self.session_types)):
                    
                    # Generate prediction and get distances + sample counts using the sensor_utils module
                    current_session_type = None
                    try:
                        if self.current_session < len(self.session_types):
                            current_session_type = self.session_types[self.current_session]
                    except Exception:
                        current_session_type = None
                    prediction_value, closest_prototype, distances, sample_counts = sensor_utils.run_ml_prediction(self.participant_data, session_type=current_session_type)
                    
                    # Update UI when active and not Standard session
                    if (hasattr(self.frames[SessionPage], "ml_prediction_active") and
                        self.frames[SessionPage].ml_prediction_active and
                        self.session_types[self.current_session] != "Standard"):
                        # Update the ML prediction display in the session page with prototype distances
                        self.frames[SessionPage].update_ml_display(distances, sample_counts)
            except Exception:
                logger.exception("Error in ML loop")
            
            # Sleep until next tick
            sleep_for = max(0.0, next_tick - time.time())
            time.sleep(sleep_for)
    
    def pause_data_collection(self):
        """Pause or resume data collection"""
        self.paused = not self.paused
        
        if self.paused:
            self.status_bar.config(text="Data collection paused")
        else:
            self.status_bar.config(text="Data collection resumed")
        
        return self.paused
    
    def stop_data_collection(self):
        """Stop data collection"""
        self.running = False
        
        if self.data_thread and self.data_thread.is_alive():
            self.data_thread.join(timeout=1.0)
        if self.ml_thread and self.ml_thread.is_alive():
            self.ml_thread.join(timeout=1.0)
        
        # Save session data immediately using atomic write
        self.save_session_data()
        
        self.status_bar.config(text="Data collection stopped")
    
    def on_closing(self):
        """Handle window closing"""
        if messagebox.askokcancel("Quit", "Do you want to quit the application?"):
            if self.running:
                self.stop_data_collection()
            self.destroy()

if __name__ == "__main__":
    # Basic structured logging configuration
    log_level = os.environ.get("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    app = IISCAApp()
    app.mainloop()