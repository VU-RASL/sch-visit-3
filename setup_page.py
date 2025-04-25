import tkinter as tk
from tkinter import ttk, messagebox
import random
from sensor_page import SensorPage

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
        
        # ML Threshold with tooltip
        ttk.Label(inner_frame, text="ML Threshold:", font=("Helvetica", 11)).grid(row=3, column=0, sticky="w", pady=8)
        threshold_frame = ttk.Frame(inner_frame)
        threshold_frame.grid(row=3, column=1, sticky="w", pady=8, padx=5)
        
        self.ml_threshold = ttk.Entry(threshold_frame, width=30, font=("Helvetica", 11))
        self.ml_threshold.insert(0, "0.95")
        self.ml_threshold.pack(side="left")
        
        threshold_info = ttk.Label(threshold_frame, text="ⓘ", font=("Helvetica", 11))
        threshold_info.pack(side="left", padx=5)
        threshold_info.bind("<Enter>", lambda e: self.show_tooltip("Value between 0 and 1 for ML prediction threshold"))
        threshold_info.bind("<Leave>", lambda e: self.hide_tooltip())
        
        # Number of Sessions with input box
        ttk.Label(inner_frame, text="Number of Sessions:", font=("Helvetica", 11)).grid(row=4, column=0, sticky="w", pady=8)
        self.sessions_var = ttk.Entry(inner_frame, width=30, font=("Helvetica", 11))
        self.sessions_var.insert(0, "10")  # Default value
        self.sessions_var.grid(row=4, column=1, sticky="w", pady=8, padx=5)
        
        # Next button with improved styling
        button_frame = ttk.Frame(self)
        button_frame.pack(pady=5)
        
        next_button = ttk.Button(button_frame, text="Next", width=15, command=self.save_and_continue)
        next_button.pack()
    
    def update_sessions_label(self, event=None):
        self.sessions_label.config(text=str(self.sessions_var.get()))
    
    def show_tooltip(self, text):
        x, y, _, _ = self.winfo_toplevel().winfo_geometry().split('+')
        x, y = int(x), int(y)
        
        self.tooltip = tk.Toplevel(self)
        self.tooltip.wm_overrideredirect(True)
        self.tooltip.wm_geometry(f"+{x+100}+{y+300}")
        
        label = ttk.Label(self.tooltip, text=text, background="#FFFFCC", relief="solid", borderwidth=1)
        label.pack()
    
    def hide_tooltip(self):
        if hasattr(self, "tooltip"):
            self.tooltip.destroy()
    
    def save_and_continue(self):
        # Validate inputs
        if not self.participant_id.get().strip():
            messagebox.showerror("Error", "Please enter a Participant ID")
            return
            
        try:
            ml_threshold = float(self.ml_threshold.get())
            if not 0 <= ml_threshold <= 1:
                messagebox.showerror("Error", "ML Threshold must be between 0 and 1")
                return
        except ValueError:
            messagebox.showerror("Error", "ML Threshold must be a number between 0 and 1")
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
        
        # Save participant data
        self.controller.participant_data = {
            "participant_id": self.participant_id.get(),
            "age": self.age.get(),
            "location": self.location.get(),
            "ml_threshold": self.ml_threshold.get(),
            "num_sessions": str(self.sessions_var.get())
        }
        
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