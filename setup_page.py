import tkinter as tk
from tkinter import ttk, messagebox
import random
from sensor_page import SensorPage
import os

class SetupPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Create form
        self.create_widgets()
    
    def create_widgets(self):
        # Title with improved styling
        title = ttk.Label(self, text="Participant Setup", font=("Helvetica", 16, "bold"))  # Reduced font size
        title.pack(pady=10)  # Reduced padding
        
        # Form frame with better styling
        form_frame = ttk.LabelFrame(self, text="Participant Information")
        form_frame.pack(pady=10, padx=5, fill="both", expand=True)  # Reduced padding
        
        # Add some padding inside the form
        inner_frame = ttk.Frame(form_frame, padding=10)  # Reduced padding
        inner_frame.pack(fill="both", expand=True)
        
        # Participant ID with improved layout
        ttk.Label(inner_frame, text="Participant ID:", font=("Helvetica", 11)).grid(row=0, column=0, sticky="w", pady=8)
        self.participant_id = ttk.Entry(inner_frame, width=30, font=("Helvetica", 11))
        self.participant_id.insert(0, "1")
        self.participant_id.grid(row=0, column=1, sticky="w", pady=8, padx=5)
        
        # Age
        ttk.Label(inner_frame, text="Age:", font=("Helvetica", 11)).grid(row=1, column=0, sticky="w", pady=8)
        self.age = ttk.Entry(inner_frame, width=30, font=("Helvetica", 11))
        self.age.insert(0, "5")
        self.age.grid(row=1, column=1, sticky="w", pady=8, padx=5)
        
        # Location
        ttk.Label(inner_frame, text="Location:", font=("Helvetica", 11)).grid(row=2, column=0, sticky="w", pady=8)
        self.location = ttk.Entry(inner_frame, width=30, font=("Helvetica", 11))
        self.location.insert(0, "Nashville")
        self.location.grid(row=2, column=1, sticky="w", pady=8, padx=5)
        
        # Number of Sessions with input box
        ttk.Label(inner_frame, text="Number of Sessions:", font=("Helvetica", 11)).grid(row=3, column=0, sticky="w", pady=8)
        self.sessions_var = ttk.Entry(inner_frame, width=30, font=("Helvetica", 11))
        self.sessions_var.insert(0, "10")  # Default value
        self.sessions_var.grid(row=3, column=1, sticky="w", pady=8, padx=5)
        
        # ML Window Size (seconds)
        ttk.Label(inner_frame, text="Window Size (s):", font=("Helvetica", 11)).grid(row=4, column=0, sticky="w", pady=8)
        self.window_size_var = ttk.Entry(inner_frame, width=30, font=("Helvetica", 11))
        self.window_size_var.insert(0, "30")
        self.window_size_var.grid(row=4, column=1, sticky="w", pady=8, padx=5)
        
        # Removed "Seconds in Advance" input
        # Model Selection (participant/group)
        ttk.Label(inner_frame, text="Model Folder:", font=("Helvetica", 11)).grid(row=5, column=0, sticky="w", pady=8)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(inner_frame, width=30, textvariable=self.model_var, state="readonly")
        # Discover available model folders inside 'models'
        model_options = []
        try:
            base_dir = "models"
            if os.path.isdir(base_dir):
                for name in sorted(os.listdir(base_dir)):
                    path = os.path.join(base_dir, name)
                    # Only include directories that look like model folders (contain model.pt)
                    if os.path.isdir(path) and os.path.exists(os.path.join(path, "model.pt")) and name.lower() != "group":
                        model_options.append(name)
        except Exception:
            model_options = []
        self.model_combo["values"] = model_options
        # Default to first participant option (exclude 'group')
        default_choice = model_options[0] if model_options else ""
        if default_choice:
            self.model_var.set(default_choice)
        self.model_combo.grid(row=5, column=1, sticky="w", pady=5, padx=0)
        
        # Next button with improved styling
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=5)
        
        next_button = ttk.Button(button_frame, text="Next", width=15, command=self.save_and_continue)
        next_button.pack()
    
    def update_sessions_label(self, event=None):
        self.sessions_label.config(text=str(self.sessions_var.get()))
    
    # Removed tooltip helpers since ML threshold is no longer used
    
    def save_and_continue(self):
        # Validate inputs
        if not self.participant_id.get().strip():
            messagebox.showerror("Error", "Please enter a Participant ID")
            return
            
        # Validate number of sessions
        try:
            num_sessions = int(self.sessions_var.get())
            if num_sessions < 1:
                messagebox.showerror("Error", "Number of Sessions must be at least 1")
                return
        except ValueError:
            messagebox.showerror("Error", "Number of Sessions must be a valid integer")
            return
        
        # Validate window size seconds
        try:
            window_seconds = int(float(self.window_size_var.get()))
            if window_seconds <= 0:
                messagebox.showerror("Error", "Window Size must be greater than 0 seconds")
                return
        except ValueError:
            messagebox.showerror("Error", "Window Size must be a valid number")
            return
        
        # Removed "Seconds in Advance" validation
        
        # Save participant data
        self.controller.participant_data = {
            "participant_id": self.participant_id.get(),
            "age": self.age.get(),
            "location": self.location.get(),
            "num_sessions": str(self.sessions_var.get()),
            "ml_window_seconds": str(window_seconds),
            # Removed "ml_seconds_in_advance"
            "model_participant": self.model_var.get() if hasattr(self, "model_var") else "",
        }
        
        # Validate selected model folder exists; do not fall back to 'group'
        try:
            base_dir = "models"
            chosen = self.controller.participant_data.get("model_participant", "")
            chosen_path = os.path.join(base_dir, chosen) if chosen else ""
            if not chosen:
                messagebox.showerror("Error", "Please select a participant model (excluding 'group').")
                return
            if not (os.path.isdir(chosen_path) and os.path.exists(os.path.join(chosen_path, "model.pt"))):
                messagebox.showerror("Error", "Selected participant model folder is invalid or missing 'model.pt'.")
                return
        except Exception:
            messagebox.showerror("Error", "Failed to validate models folder.")
            return
        
        # Generate random session types
        try:
            session_types = ["Standard", "Individual", "Group"]
            self.controller.session_types = random.choices(session_types, k=num_sessions)
        except ValueError:
            self.controller.session_types = ["Standard"]
        
        # Move to sensor page
        self.controller.show_frame(SensorPage)
    
    def update_frame(self):
        # This method is called when the frame is shown
        pass