import tkinter as tk
from tkinter import ttk, scrolledtext
import queue
import random
import time
from PIL import Image, ImageTk
import os
from datetime import datetime

class NonBlockingPopup(tk.Toplevel):
    def __init__(self, parent, title, message, auto_dismiss_ms=3000):
        super().__init__(parent)
        self.title(title)
        self.geometry("300x150")
        self.transient(parent)
        
        # Add padding around all widgets
        frame = ttk.Frame(self, padding=20)
        frame.pack(fill="both", expand=True)
        
        # Message
        ttk.Label(
            frame, 
            text=message,
            font=("Helvetica", 12),
            wraplength=250
        ).pack(pady=(0, 20))
        
        # Dismiss button
        ttk.Button(
            frame, 
            text="Dismiss", 
            command=self.destroy,
            padding=5
        ).pack()
        
        # Center the popup on the screen
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        
        # Auto-dismiss after specified milliseconds
        if auto_dismiss_ms > 0:
            self.after(auto_dismiss_ms, self.destroy)

class SessionPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Session state
        self.sr_timer_running = False
        self.eo_timer_running = False
        self.sr_time = 0
        self.current_state = "SR"  # Start in SR state
        self.ml_prediction_active = False  # Start with ML inactive
        self.last_prediction_time = 0
        self.last_prediction_value = 0
        self.trial_count = 0  # Track number of trials (SR to EO transitions)
        self.eo_time = 0  # Add this line to track EO timer
        self.timer_id = None  # Add timer ID to track scheduled timers
        # Cache last displayed texts to avoid unnecessary UI updates (reduce flicker)
        self._last_ml_label_text = None
        self._last_ml_prob_text = None
        self._last_ml_counts_text = None
        
        # Create widgets
        self.create_widgets()
    
    def create_widgets(self):
        # Main container with padding
        main_container = ttk.Frame(self, padding=20)
        main_container.pack(fill="both", expand=True)
        
        # Title with improved styling
        self.title_label = ttk.Label(main_container, text="IISCA Session", font=("Helvetica", 22, "bold"))
        self.title_label.pack(pady=15)
        
        # Session info frame with better layout
        info_frame = ttk.Frame(main_container)
        info_frame.pack(pady=10, fill="x")
        
        self.session_label = ttk.Label(info_frame, text="Session Type: ", font=("Helvetica", 12))
        self.session_label.pack(side="left", padx=20)
        
        self.state_label = ttk.Label(info_frame, text="Current State: SR", font=("Helvetica", 12, "bold"))
        self.state_label.pack(side="right", padx=20)
        
        # Trial counter
        self.trial_label = ttk.Label(info_frame, text="Trials: 0", font=("Helvetica", 12))
        self.trial_label.pack(side="right", padx=20)
        
        # Timer frame with improved visibility
        self.timer_frame = ttk.Frame(main_container)
        self.timer_frame.pack(pady=15)
        
        self.timer_label = ttk.Label(self.timer_frame, text="SR Timer: 00:00", font=("Helvetica", 28))
        self.timer_label.pack()
        
        # Create a 'Start 30s Timer' button (initially visible; enabled per session)
        # Place it under the timer label
        # Control buttons with improved layout
        control_frame = ttk.LabelFrame(main_container, text="Session Controls")
        control_frame.pack(pady=15, padx=10, fill="x")
        
        # Standard session buttons with better spacing and styling
        self.standard_frame = ttk.Frame(control_frame)
        self.standard_frame.pack(pady=15, padx=10)
        
        # Create a more modern button style
        button_style = {"width": 14, "padding": 8}
        
        # Now that style is defined, create the Start 30s Timer button in the timer area
        self.start_30s_button = ttk.Button(
            self.timer_frame,
            text="Start 30s Timer",
            command=self.start_sr_30s,
            **button_style
        )
        self.start_30s_button.pack(pady=8)
        
        # First row of buttons
        button_row1 = ttk.Frame(self.standard_frame)
        button_row1.pack(pady=5, fill="x")
        
        self.eo_button = ttk.Button(button_row1, text="EO", command=self.toggle_eo_sr, **button_style)
        self.eo_button.pack(side="left", padx=10)
        
        self.sr_button = ttk.Button(button_row1, text="SR", command=self.toggle_eo_sr, **button_style)
        self.sr_button.pack(side="left", padx=10)
        self.sr_button.config(state="disabled")
        
        self.pause_button = ttk.Button(button_row1, text="Pause", command=self.toggle_pause, **button_style)
        self.pause_button.pack(side="left", padx=10)
        
        # Second row of buttons
        button_row2 = ttk.Frame(self.standard_frame)
        button_row2.pack(pady=5, fill="x")
        
        self.notify_eo_button = ttk.Button(button_row2, text="Notify EO", 
                                          command=lambda: self.show_notification("EO Notification"), **button_style)
        self.notify_eo_button.pack(side="left", padx=10)
        
        self.notify_sr_button = ttk.Button(button_row2, text="Notify SR", 
                                          command=lambda: self.show_notification("SR Notification"), **button_style)
        self.notify_sr_button.pack(side="left", padx=10)
        
        self.notify_prompt_button = ttk.Button(button_row2, text="Notify Prompt", 
                                              command=lambda: self.show_notification("Prompt Notification"), **button_style)
        self.notify_prompt_button.pack(side="left", padx=10)
        
        # ML prediction display with improved styling
        self.ml_frame = ttk.LabelFrame(main_container, text="ML Predictions")
        self.ml_frame.pack(pady=15, padx=10, fill="x")
        
        ml_inner_frame = ttk.Frame(self.ml_frame)
        ml_inner_frame.pack(pady=10, fill="x")
        
        # Large, stable header for closest prototype (centered)
        self.ml_label = ttk.Label(
            ml_inner_frame,
            text="Closest Prototype: N/A",
            font=("Helvetica", 16, "bold"),
            justify="center",
            anchor="center"
        )
        self.ml_label.pack(pady=5, fill="x")
        
        # Monospace, fixed-width distances label to reduce layout changes (centered)
        self.ml_probability = ttk.Label(
            ml_inner_frame,
            text="",
            font=("Courier New", 12),
            justify="center",
            anchor="center",
            width=60
        )
        self.ml_probability.pack(pady=5, fill="x", expand=True)
        
        # Separate counts line to avoid changing the distances block each update (centered)
        self.ml_counts = ttk.Label(
            ml_inner_frame,
            text="",
            font=("Helvetica", 11),
            anchor="center",
            justify="center"
        )
        self.ml_counts.pack(pady=2, fill="x")
        
        self.ml_progress = ttk.Progressbar(ml_inner_frame, orient="horizontal", length=300, mode="determinate")
        self.ml_progress.pack(pady=5)
        
        # Session notes
        notes_frame = ttk.LabelFrame(main_container, text="Session Notes")
        notes_frame.pack(pady=15, padx=10, fill="both", expand=True)
        
        self.notes_text = scrolledtext.ScrolledText(notes_frame, height=5, font=("Helvetica", 11))
        self.notes_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Next session button
        self.next_button = ttk.Button(main_container, text="Next Session", command=self.next_session, **button_style)
        self.next_button.pack(pady=15)
    
    def toggle_eo_sr(self):
        """Toggle between EO and SR states"""
        if self.current_state == "SR":
            # Switching from SR to EO
            self.current_state = "EO"
            self.state_label.config(text="Current State: EO")
            self.sr_timer_running = False
            
            # Check if this is a Standard session
            if self.controller.current_session < len(self.controller.session_types):
                session_type = self.controller.session_types[self.controller.current_session]
                if session_type == "Standard":
                    # Start 30 second EO timer for Standard sessions
                    self.start_eo_timer()
                else:
                    self.timer_label.config(text="")  # Clear timer for non-Standard sessions
            
            # Enable/disable appropriate buttons
            self.eo_button.config(state="disabled")
            self.sr_button.config(state="normal")
            
            # Increment trial count
            self.trial_count += 1
            self.trial_label.config(text=f"Trials: {self.trial_count}")
            
            # Check session type for ML prediction
            if self.controller.current_session < len(self.controller.session_types):
                session_type = self.controller.session_types[self.controller.current_session]
                # Enable ML prediction for non-Standard sessions
                if session_type != "Standard":
                    self.ml_prediction_active = True
        else:
            # Switching from EO to SR
            self.current_state = "SR"
            self.state_label.config(text="Current State: SR")
            
            # Enable/disable appropriate buttons
            self.eo_button.config(state="normal")
            self.sr_button.config(state="disabled")
            
            # Start SR timer
            self.start_sr_timer()
            
            # Check session type for ML prediction
            if self.controller.current_session < len(self.controller.session_types):
                session_type = self.controller.session_types[self.controller.current_session]
                # Enable ML prediction for non-Standard sessions
                if session_type != "Standard":
                    self.ml_prediction_active = True
                else:
                    self.ml_prediction_active = False
                    # Reset ML display for Standard sessions
                    self.ml_label.config(text="No prediction in Standard session")
                    self.ml_probability.config(text="Probability: N/A")
                    self.ml_progress["value"] = 0
    
    def start_sr_timer(self, time=None):
        """Start the SR timer"""
        # Cancel any existing timer
        if self.timer_id is not None:
            self.after_cancel(self.timer_id)
            self.timer_id = None
            
        # Set random time between 30-90 seconds
        if time is None:
            self.sr_time = random.randint(30, 90)
        else:
            self.sr_time = time

        self.eo_time = 0
        
        # Format time as MM:SS
        minutes = self.sr_time // 60
        seconds = self.sr_time % 60
        self.timer_label.config(text=f"SR Timer: {minutes:02d}:{seconds:02d}")
        
        # Start timer
        self.sr_timer_running = True
        self.update_timer()
    
    def update_timer(self):
        """Update the SR timer"""
        if self.sr_timer_running and self.sr_time > 0:
            # Format time as MM:SS
            minutes = self.sr_time // 60
            seconds = self.sr_time % 60
            self.timer_label.config(text=f"SR Timer: {minutes:02d}:{seconds:02d}")
            
            # Decrement timer
            self.sr_time -= 1
            
            # Schedule next update
            self.timer_id = self.after(1000, self.update_timer)
        elif self.sr_timer_running:
            self.timer_label.config(text="SR Timer: 00:00")
            self.sr_timer_running = False
            self.timer_id = None
            
            if self.controller.current_session < len(self.controller.session_types):
                NonBlockingPopup(self, "SR Ended", "SR period has ended.")
    
    def toggle_pause(self):
        paused = self.controller.pause_data_collection()
        if paused:
            self.pause_button.config(text="Resume")
            if self.sr_timer_running:
                self.sr_timer_running = False
            self.ml_prediction_active = False
        else:
            self.pause_button.config(text="Pause")
            if self.current_state == "SR":
                self.sr_timer_running = True
                self.update_timer()
            
            # Check session type for ML prediction
            if self.controller.current_session < len(self.controller.session_types):
                session_type = self.controller.session_types[self.controller.current_session]
                # Enable ML prediction for non-Standard sessions
                if session_type != "Standard":
                    self.ml_prediction_active = True
    
    def next_session(self):
        # Cancel any existing timer
        if self.timer_id is not None:
            self.after_cancel(self.timer_id)
            self.timer_id = None
            
        # Save session notes
        notes = self.notes_text.get("1.0", "end-1c")
        
        # Make sure we have a valid session index
        if self.controller.current_session < len(self.controller.session_types):
            self.controller.session_data[str(self.controller.current_session)] = {
                "type": self.controller.session_types[self.controller.current_session],
                "notes": notes,
                "trials": self.trial_count
            }
            # Persist session data immediately to avoid loss on crash
            if hasattr(self.controller, "save_session_data"):
                self.controller.save_session_data()
            
            # Move to next session
            self.controller.current_session += 1
            self.sr_timer_running = False
            self.eo_timer_running = False
            self.sr_time = 0
            self.eo_time = 0
            
            # End of all sessions
            if self.controller.current_session < len(self.controller.session_types):
                # Create custom popup for session confirmation
                self.show_begin_session_popup()
            else:
                # End of all sessions
                self.controller.stop_data_collection()
                NonBlockingPopup(self, "Complete", "All sessions completed. Data has been saved.")
                self.controller.destroy()
    
    def show_begin_session_popup(self):
        """Show a popup that allows the user to begin the next session"""
        session_type = self.controller.session_types[self.controller.current_session]
        
        # Create a custom dialog
        popup = tk.Toplevel(self)
        popup.title("Begin Next Session")
        popup.geometry("400x200")
        popup.transient(self)  # Set to be on top of the main window
        popup.grab_set()  # Modal dialog
        
        # Add padding around all widgets
        frame = ttk.Frame(popup, padding=20)
        frame.pack(fill="both", expand=True)
        
        # Title and session info
        ttk.Label(
            frame, 
            text=f"Session {self.controller.current_session + 1} of {len(self.controller.session_types)}",
            font=("Helvetica", 14, "bold")
        ).pack(pady=(0, 10))
        
        ttk.Label(
            frame, 
            text=f"Type: {session_type}",
            font=("Helvetica", 12)
        ).pack(pady=(0, 20))
        
        # Begin button
        begin_button = ttk.Button(
            frame, 
            text="Begin Session", 
            command=lambda: self.begin_next_session(popup),
            padding=10
        )
        begin_button.pack(pady=10)
        
        # Center the popup on the screen
        popup.update_idletasks()
        width = popup.winfo_width()
        height = popup.winfo_height()
        x = (popup.winfo_screenwidth() // 2) - (width // 2)
        y = (popup.winfo_screenheight() // 2) - (height // 2)
        popup.geometry(f"{width}x{height}+{x}+{y}")
    
    def begin_next_session(self, popup):
        """Begin the next session after confirmation"""
        # Close the popup
        popup.destroy()
        
        # Update the session frame
        self.update_frame()
    
    def update_frame(self):
        # Update session information
        if self.controller.current_session < len(self.controller.session_types):
            session_type = self.controller.session_types[self.controller.current_session]
            self.title_label.config(text=f"Session {self.controller.current_session + 1} of {len(self.controller.session_types)}")
            self.session_label.config(text=f"Session Type: {session_type}")
            
            # If this session hasn't been recorded yet, create and save it at session begin
            session_key = str(self.controller.current_session)
            if session_key not in self.controller.session_data:
                self.controller.session_data[session_key] = {
                    "type": session_type,
                    "notes": "",
                    "trials": 0,
                    "started_at": datetime.now().isoformat()
                }
                if hasattr(self.controller, "save_session_data"):
                    self.controller.save_session_data()

            # Reset state
            self.current_state = "SR"
            self.state_label.config(text="Current State: SR")
            self.sr_timer_running = False
            self.sr_time = 0
            self.timer_label.config(text="SR Timer: 00:00")
            self.notes_text.delete("1.0", "end")
            
            # Reset trial count
            self.trial_count = 0
            self.trial_label.config(text="Trials: 0")
            
            # ML prediction should be inactive for Standard sessions
            if session_type == "Standard":
                self.ml_prediction_active = False
                self.ml_label.config(text="No prediction in Standard session")
                self.ml_probability.config(text="Probability: N/A")
            else:
                self.ml_prediction_active = True
                self.ml_label.config(text="No prediction yet")
                self.ml_probability.config(text="Probability: 0.000")
            
            self.ml_progress["value"] = 0
            
            # Reset button states
            self.eo_button.config(state="normal")
            self.sr_button.config(state="disabled")
            
            # Do not auto-start SR timer; show enable button for manual start
            if hasattr(self, "start_30s_button"):
                self.start_30s_button.config(state="normal")
    
    def update_ml_display(self, distances, sample_counts=None):
        """Update ML display to show closest prototype (large) and distances list without flicker."""
        try:
            if isinstance(distances, dict) and distances:
                # Sort by increasing distance for readability
                sorted_items = sorted(distances.items(), key=lambda kv: kv[1])
                lines = [f"{k}: {v:.3f}" for k, v in sorted_items]
                # Closest prototype overall (first after sorting)
                closest_label, closest_dist = sorted_items[0]
                closest_text = f"Closest Prototype: {closest_label} ({closest_dist:.3f})"
                distances_text = "\n".join(lines)
                # Only update labels if content changed (prevents flicker)
                if closest_text != self._last_ml_label_text:
                    self.ml_label.config(text=closest_text)
                    self._last_ml_label_text = closest_text
                if distances_text != self._last_ml_prob_text:
                    self.ml_probability.config(text=distances_text)
                    self._last_ml_prob_text = distances_text
            else:
                if "Closest Prototype: N/A" != self._last_ml_label_text:
                    self.ml_label.config(text="Closest Prototype: N/A")
                    self._last_ml_label_text = "Closest Prototype: N/A"
                if "No distances available" != self._last_ml_prob_text:
                    self.ml_probability.config(text="No distances available")
                    self._last_ml_prob_text = "No distances available"
            
            # We no longer use the progress bar for raw score
            self.ml_progress["value"] = 0
            
            # Sample counts (optional) - update separately to avoid touching distances label
            if sample_counts:
                counts_text = (f"Samples: IMU={sample_counts.get('imu_samples',0)}, "
                               f"OSC={sample_counts.get('osc_samples',0)}, "
                               f"Audio={sample_counts.get('audio_samples',0)}")
                if counts_text != self._last_ml_counts_text:
                    self.ml_counts.config(text=counts_text)
                    self._last_ml_counts_text = counts_text
            else:
                if "" != self._last_ml_counts_text:
                    self.ml_counts.config(text="")
                    self._last_ml_counts_text = ""
        except Exception:
            # Fail-soft to avoid UI crashes
            if "Prototype distances" != self._last_ml_label_text:
                self.ml_label.config(text="Prototype distances")
                self._last_ml_label_text = "Prototype distances"
            if "Error displaying distances" != self._last_ml_prob_text:
                self.ml_probability.config(text="Error displaying distances")
                self._last_ml_prob_text = "Error displaying distances"
    
    def show_notification(self, message):
        """Show a notification to the participant"""
        NonBlockingPopup(self, "Notification", message)

    def start_sr_30s(self):
        """Manually start a 30-second SR timer from the button."""
        # Disable the button to prevent multiple starts
        if hasattr(self, "start_30s_button"):
            self.start_30s_button.config(state="disabled")
        self.start_sr_timer(time=30)

    def start_eo_timer(self):
        """Start the EO timer"""
        # Cancel any existing timer
        if self.timer_id is not None:
            self.after_cancel(self.timer_id)
            self.timer_id = None
            
        self.eo_time = 30
        self.sr_time = 0
        self.eo_timer_running = True
        self.update_eo_timer()
        
    def update_eo_timer(self):
        """Update the EO timer"""
        if self.eo_timer_running and self.eo_time > 0:
            # Format time as MM:SS
            minutes = self.eo_time // 60
            seconds = self.eo_time % 60
            self.timer_label.config(text=f"EO Timer: {minutes:02d}:{seconds:02d}")
            
            # Decrement timer
            self.eo_time -= 1
            
            # Schedule next update
            self.timer_id = self.after(1000, self.update_eo_timer)
        elif self.eo_timer_running:
            self.timer_label.config(text="EO Timer: 00:00")
            self.eo_timer_running = False
            self.timer_id = None