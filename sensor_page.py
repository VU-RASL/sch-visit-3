import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import asyncio
import threading
import sensor_utils
from audio_utils import AudioProcessor
import os
import datetime
import logging

class SensorPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.audio_processor = None
        
        # Create widgets
        self.create_widgets()
        
        # Bind cleanup to window close event
        self.bind_all("<Destroy>", self._on_destroy)
    
    def _on_destroy(self, event):
        """Handle window destruction"""
        if event.widget == self:
            self.cleanup_audio()
    
    def cleanup_audio(self):
        """Clean up audio resources"""
        if self.audio_processor is not None:
            try:
                self.audio_processor.cleanup()
            except Exception as e:
                print(f"Error cleaning up audio processor: {str(e)}")
            self.audio_processor = None
    
    def create_widgets(self):
        # Title with improved styling
        title = ttk.Label(self, text="Sensor Setup", font=("Helvetica", 18, "bold"))
        title.pack(pady=15)
        
        # Sensor selection frame with better styling
        sensor_frame = ttk.LabelFrame(self, text="Select Sensors")
        sensor_frame.pack(pady=15, padx=30, fill="both", expand=True)
        
        # BLE Sensors with improved layout
        ble_frame = ttk.LabelFrame(sensor_frame, text="BLE IMU Sensors")
        ble_frame.pack(pady=10, padx=15, fill="x")
        
        self.ble_var = tk.StringVar(value="simulate")
        ttk.Radiobutton(ble_frame, text="Connect to Real Sensors", variable=self.ble_var, 
                        value="real", padding=5).pack(anchor="w", pady=5, padx=10)
        ttk.Radiobutton(ble_frame, text="Simulate 5 IMU Sensors", variable=self.ble_var, 
                        value="simulate", padding=5).pack(anchor="w", pady=5, padx=10)
        
        # OSC Sensors
        osc_frame = ttk.LabelFrame(sensor_frame, text="OSC Sensors")
        osc_frame.pack(pady=10, padx=15, fill="x")
        
        self.osc_var = tk.StringVar(value="simulate")
        ttk.Radiobutton(osc_frame, text="Connect to Real EmotiBit OSC Sensors", variable=self.osc_var, 
                        value="real", padding=5).pack(anchor="w", pady=5, padx=10)
        ttk.Radiobutton(osc_frame, text="Simulate EmotiBit OSC Sensors", variable=self.osc_var, 
                        value="simulate", padding=5).pack(anchor="w", pady=5, padx=10)
        
        # Audio Sensor
        audio_frame = ttk.LabelFrame(sensor_frame, text="Audio Sensor")
        audio_frame.pack(pady=10, padx=15, fill="x")
        
        self.audio_var = tk.StringVar(value="simulate")
        ttk.Radiobutton(audio_frame, text="Connect to Real Audio Sensor", variable=self.audio_var, 
                        value="real", padding=5).pack(anchor="w", pady=5, padx=10)
        ttk.Radiobutton(audio_frame, text="Simulate Audio Sensor", variable=self.audio_var, 
                        value="simulate", padding=5).pack(anchor="w", pady=5, padx=10)
        
        # Status indicators
        status_frame = ttk.LabelFrame(self, text="Connection Status")
        status_frame.pack(pady=10, padx=30, fill="x")
        
        # Create a grid of status indicators
        self.status_indicators = {}
        sensors = [("BLE_IMU_1", 0, 0), ("BLE_IMU_2", 0, 1), ("BLE_IMU_3", 0, 2), 
                  ("BLE_IMU_4", 0, 3), ("BLE_IMU_5", 0, 4),
                  ("OSC_EmotiBit", 1, 0), 
                  ("Audio_1", 1, 2)]
                  
        for sensor_id, row, col in sensors:
            frame = ttk.Frame(status_frame, padding=5)
            frame.grid(row=row, column=col, padx=10, pady=5)
            
            # Create colored indicator
            canvas = tk.Canvas(frame, width=15, height=15, bg="gray", highlightthickness=0)
            canvas.pack(side="left", padx=5)
            
            # Create label
            label = ttk.Label(frame, text=sensor_id)
            label.pack(side="left")
            
            self.status_indicators[sensor_id] = canvas
        
        # Buttons with improved styling
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=20)
        
        ttk.Button(button_frame, text="Back", width=12, 
                  command=lambda: self.controller.show_frame("SetupPage")).pack(side="left", padx=10)
        
        self.connect_button = ttk.Button(button_frame, text="Connect & Start", width=15, 
                                        command=self.connect_and_start)
        self.connect_button.pack(side="left", padx=10)
    
    def connect_and_start(self):
        # Show connecting status
        self.connect_button.config(state="disabled", text="Connecting...")
        self.update()
        
        # Configure sensors based on selections
        self.controller.sensors = {}
        
        # Start a thread to handle BLE connections (if needed)
        if self.ble_var.get() == "real":
            # Create a progress dialog
            progress_window = tk.Toplevel(self)
            progress_window.title("Connecting to BLE Sensors")
            progress_window.geometry("400x150")
            progress_window.transient(self)
            progress_window.grab_set()
            
            progress_label = ttk.Label(progress_window, text="Scanning for BLE sensors...", font=("Helvetica", 12))
            progress_label.pack(pady=15)
            
            progress_bar = ttk.Progressbar(progress_window, orient="horizontal", length=300, mode="indeterminate")
            progress_bar.pack(pady=10)
            progress_bar.start()
            
            # Start BLE connection in a separate thread
            threading.Thread(target=self.connect_ble_sensors, args=(progress_window,), daemon=True).start()
        else:
            # If not using real BLE, just proceed with simulated sensors
            self.setup_simulated_ble()
            self.setup_other_sensors()
            self.finish_connection()
    
    def connect_ble_sensors(self, progress_window):
        try:
            # Create a new event loop for the thread
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Add a text widget to display connection messages
            progress_text = scrolledtext.ScrolledText(progress_window, height=8, width=50)
            progress_text.pack(pady=10, padx=15, fill="both", expand=True)
            
            # Attach a logging handler to stream sensor_utils logs into the progress_text
            class _ProgressTextHandler(logging.Handler):
                def __init__(self, ui_self, text_widget):
                    super().__init__()
                    self.ui_self = ui_self
                    self.text_widget = text_widget
                def emit(self, record):
                    msg = self.format(record)
                    # Ensure UI update happens on main thread
                    try:
                        self.ui_self.after(0, lambda: self.text_widget.insert(tk.END, msg + "\n"))
                        self.ui_self.after(0, lambda: self.text_widget.see(tk.END))
                    except Exception:
                        pass
            sensor_logger = logging.getLogger("sensor_utils")
            handler = _ProgressTextHandler(self, progress_text)
            handler.setLevel(logging.INFO)
            handler.setFormatter(logging.Formatter("%(message)s"))
            sensor_logger.addHandler(handler)
            
            # Connect to BLE sensors
            connected_devices = loop.run_until_complete(sensor_utils.connect_ble_sensors())
            # Remove handler after use
            sensor_logger.removeHandler(handler)
            
            # Update UI with connected devices
            for i, device in enumerate(connected_devices):
                if i < 5:  # We support up to 5 BLE sensors
                    sensor_id = f"BLE_IMU_{i+1}"
                    self.controller.sensors[sensor_id] = {
                        "type": "real",
                        "connected": True,
                        "device": device,  # Store the device object for later use
                        "loop": loop  # Store the event loop for this device
                    }
                    
                    # Update status indicator in the main thread
                    self.after(0, lambda sid=sensor_id: self.status_indicators[sid].config(bg="green"))
            
            # Fill in any missing sensors with simulated ones
            for i in range(len(connected_devices), 5):
                sensor_id = f"BLE_IMU_{i+1}"
                self.controller.sensors[sensor_id] = {
                    "type": "simulated",
                    "connected": True
                }
                # Update status indicator in the main thread
                self.after(0, lambda sid=sensor_id: self.status_indicators[sid].config(bg="yellow"))
            
            # Setup other sensors
            self.after(0, self.setup_other_sensors)
            
            # Start a thread to keep the event loop running
            def run_event_loop():
                try:
                    loop.run_forever()
                except Exception as e:
                    print(f"Event loop error: {str(e)}")
            
            threading.Thread(target=run_event_loop, daemon=True).start()
            
            # Close the progress window and finish connection in the main thread
            self.after(0, progress_window.destroy)
            self.after(100, self.finish_connection)
            
        except Exception as e:
            # Handle any errors
            try:
                # Attempt to remove handler if it exists
                sensor_logger = logging.getLogger("sensor_utils")
                for h in list(sensor_logger.handlers):
                    if isinstance(h, logging.Handler) and getattr(h, "ui_self", None) is self:
                        sensor_logger.removeHandler(h)
            except Exception:
                pass
            self.after(0, lambda: messagebox.showerror("Connection Error", f"Failed to connect to BLE sensors: {str(e)}"))
            self.after(0, progress_window.destroy)
            self.after(0, lambda: self.connect_button.config(state="normal", text="Connect & Start"))
    
    def setup_simulated_ble(self):
        # Setup simulated BLE sensors
        for i in range(5):
            sensor_id = f"BLE_IMU_{i+1}"
            self.controller.sensors[sensor_id] = {
                "type": "simulated",
                "connected": True
            }
            self.status_indicators[sensor_id].config(bg="green")
    
    def setup_other_sensors(self):
        # OSC Sensors - EmotiBit
        from sensor_utils import EMOTIBIT_SIGNAL_TYPES, EmotiBitConnector
        
        if self.osc_var.get() == "simulate":
            # Add a single entry for the EmotiBit connection status
            self.controller.sensors["OSC_EmotiBit"] = {
                "type": "simulated",
                "connected": True
            }
            self.status_indicators["OSC_EmotiBit"].config(bg="green")
            
            # Create individual sensor entries for each EmotiBit signal
            for signal_type in EMOTIBIT_SIGNAL_TYPES.keys():
                sensor_id = f"OSC_{signal_type}"
                self.controller.sensors[sensor_id] = {
                    "type": "simulated",
                    "connected": True
                }
        else:
            # Create EmotiBit connector
            emotibit_connector = EmotiBitConnector()
            
            # Try to connect to real EmotiBit via OSC
            success = emotibit_connector.connect()
            
            # Update status based on connection success
            if success:
                # Add a single entry for the EmotiBit connection status
                self.controller.sensors["OSC_EmotiBit"] = {
                    "type": "real",
                    "connected": True,
                    "connector": emotibit_connector  # Store connector for later use
                }
                self.status_indicators["OSC_EmotiBit"].config(bg="green")
                
                # Create individual sensor entries for each EmotiBit signal
                for signal_type in EMOTIBIT_SIGNAL_TYPES.keys():
                    sensor_id = f"OSC_{signal_type}"
                    self.controller.sensors[sensor_id] = {
                        "type": "real",
                        "connected": True
                    }
            else:
                print("Failed to connect to EmotiBit and fallback to simulation")
                # Fall back to simulation if connection fails
                self.controller.sensors["OSC_EmotiBit"] = {
                    "type": "simulated",
                    "connected": True
                }
                self.status_indicators["OSC_EmotiBit"].config(bg="yellow")
                
                # Create individual sensor entries for each EmotiBit signal
                for signal_type in EMOTIBIT_SIGNAL_TYPES.keys():
                    sensor_id = f"OSC_{signal_type}"
                    self.controller.sensors[sensor_id] = {
                        "type": "simulated",
                        "connected": True
                    }
        
        # Audio Sensor
        if self.audio_var.get() == "simulate":
            self.controller.sensors["Audio_1"] = {
                "type": "simulated",
                "connected": True
            }
            self.status_indicators["Audio_1"].config(bg="green")
        else:
            try:
                # Clean up any existing audio processor
                self.cleanup_audio()
                
                # Initialize real audio processor
                self.audio_processor = AudioProcessor()
                
                # Store the processor in the sensor info, but don't start recording yet
                self.controller.sensors["Audio_1"] = {
                    "type": "real",
                    "connected": True,
                    "processor": self.audio_processor
                }
                self.status_indicators["Audio_1"].config(bg="green")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to initialize audio sensor: {str(e)}")
                self.controller.sensors["Audio_1"] = {
                    "type": "simulated",
                    "connected": True
                }
                self.status_indicators["Audio_1"].config(bg="green")
    
    def finish_connection(self):
        # Start data collection (this will create the data directory)
        self.controller.start_data_collection()
        
        # Sync any buffered EmotiBit data
        if "OSC_EmotiBit" in self.controller.sensors:
            osc_sensor = self.controller.sensors["OSC_EmotiBit"]
            if osc_sensor["type"] == "real" and "connector" in osc_sensor:
                try:
                    # Sync any buffered data
                    osc_sensor["connector"].sync_buffered_data(self.controller.data_queue)
                except Exception as e:
                    print(f"Error syncing EmotiBit buffered data: {str(e)}")
        
        # Now that data directory is created, start audio recording if using real audio
        if self.audio_var.get() == "real" and "Audio_1" in self.controller.sensors:
            audio_sensor = self.controller.sensors["Audio_1"]
            if audio_sensor["type"] == "real" and "processor" in audio_sensor:
                try:
                    # Start recording in the controller's data directory
                    audio_sensor["processor"].start_recording(output_dir=self.controller.data_dir)
                except Exception as e:
                    print(f"Error starting audio recording: {str(e)}")
        
        # Show success message
        print("All sensors connected successfully!")
        
        # Move to session page
        self.controller.current_session = 0
        self.controller.show_frame("SessionPage")
        
        # Reset button state
        self.connect_button.config(state="normal", text="Connect & Start")
    
    def update_frame(self):
        # Reset status indicators when the frame is shown
        for canvas in self.status_indicators.values():
            canvas.config(bg="gray")
        
        # Clean up audio processor
        self.cleanup_audio()