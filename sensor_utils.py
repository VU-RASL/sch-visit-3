import random
import numpy as np
import time
import math
from bleak import BleakClient, BleakScanner
import os
import datetime
import queue
import asyncio
import struct
from scipy.spatial.transform import Rotation
from audio_utils import process_audio_data
import threading
import atexit

import pickle
import pandas as pd
import torch
import torch.nn as nn
from typing import List, Dict, Tuple, Any, Optional, Union
from scipy.ndimage import gaussian_filter1d

# Data buffer to store recent sensor data
sensor_data_buffer = {}
buffer_max_age = 40  # Maximum age of data to keep in buffer (seconds) - older data will be removed

# Batching system for file I/O operations
file_write_batches = {}
file_batch_lock = threading.Lock()
FILE_BATCH_SIZE = 50  # Number of samples to batch before writing
FILE_BATCH_TIMEOUT = 1.0  # Maximum time (seconds) to hold data before forced write

# UUIDs for BLE characteristics (from test.py)
UUID_A = "F00044DC-0451-4000-B000-000000000000"  # Data characteristic
UUID_X = "F000ABCD-0451-4000-B000-000000000000"  # Command characteristic
UUID_SERVICE = "F0002642-0451-4000-B000-000000000000"  # Service UUID

# Global variables for BLE
ble_devices = []
freq = 100  # Default frequency

# BLE reconnection parameters
BLE_RECONNECT_ATTEMPTS = 5  # Maximum reconnection attempts
BLE_RECONNECT_DELAY = 3.0   # Initial delay between reconnection attempts (seconds)
BLE_RECONNECT_MAX_DELAY = 30.0  # Maximum delay between attempts (seconds)
BLE_CONNECTION_CHECK_INTERVAL = 5.0  # How often to check connection status (seconds)

# Connection monitor thread
ble_connection_monitor = None
ble_monitor_stop_event = threading.Event()
ble_main_event_loop = None  # Store reference to main thread's event loop
ble_reconnection_lock = threading.Lock()  # Lock to prevent concurrent reconnections to the same device
ble_reconnecting_devices = {}  # Track devices that are currently in reconnection process

# File batch worker thread
file_batch_thread = None
file_batch_stop_event = threading.Event()

# Machine learning model
model_loader = None

# Register cleanup function to run at exit
atexit.register(lambda: cleanup_sensor_system())

# Define sampling rates and buffer sizes for EmotiBit sensors
EMOTIBIT_SAMPLING_RATES = {
    "motion": 25,  # AX, AY, AZ, GX, GY, GZ, MX, MY, MZ at 25Hz
    "ppg": 25,     # PI, PG, PR at 25Hz
    "temperature": 7.5,  # T0, TH at 7.5Hz
    "eda": 15      # EA, EL, ER at 15Hz
}

# EmotiBit OSC signal types mapped to human-readable names
EMOTIBIT_SIGNAL_TYPES = {
    # Motion signals
    "ACC:X": "Accelerometer X",
    "ACC:Y": "Accelerometer Y", 
    "ACC:Z": "Accelerometer Z",
    "GYRO:X": "Gyroscope X",
    "GYRO:Y": "Gyroscope Y",
    "GYRO:Z": "Gyroscope Z",
    "MAG:X": "Magnetometer X",
    "MAG:Y": "Magnetometer Y",
    "MAG:Z": "Magnetometer Z",
    
    # PPG signals
    "PPG:RED": "PPG Red",
    "PPG:IR": "PPG Infrared",
    "PPG:GRN": "PPG Green",
    
    # Temperature signals
    "TEMP": "Temperature",
    "TEMP:T1": "Temperature T1",
    "THERM": "Thermopile",
    
    # EDA signals
    "EDA": "Electrodermal Activity",
    "EDL": "Electrodermal Level",
    "EDR": "Electrodermal Response",
    
    # Derived metrics
    "SCR:AMP": "SCR Amplitude",
    "SCR:RISE": "SCR Rise Time",
    "SCR:FREQ": "SCR Frequency",
    
    # Heart metrics
    "HR": "Heart Rate",
    "IBI": "Inter-beat Interval",
    
    # Humidity
    "HUMIDITY": "Humidity"
}

# Global variables for EmotiBit OSC
emotibit_osc_server = None
emotibit_data_buffers = {}
emotibit_active_signals = []
emotibit_connected = False

# Global variables for data collection
data_queue = None
sensors = None

def simulate_imu_data(time_val):
    """Simulate IMU sensor data"""
    # Generate quaternion (normalized)
    qw = 1.0 + 0.1 * math.sin(time_val * 0.1)
    qx = 0.1 * math.sin(time_val * 0.2)
    qy = 0.1 * math.cos(time_val * 0.3)
    qz = 0.1 * math.sin(time_val * 0.4)
    
    # Normalize quaternion
    norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Calculate Euler angles (roll, pitch, yaw)
    roll = math.atan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
    pitch = math.asin(2*(qw*qy - qz*qx))
    yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
    
    return [qw, qx, qy, qz, roll, pitch, yaw]

def simulate_osc_data(time_val):
    """Simulate OSC sensor data"""
    value1 = math.sin(time_val * 0.1) * 10
    value2 = math.cos(time_val * 0.2) * 5
    value3 = math.sin(time_val * 0.3) * 2
    
    return [value1, value2, value3]

def simulate_audio_data(time_val):
    """Simulate audio sensor data"""
    amplitude = 0.5 + 0.5 * math.sin(time_val * 0.1)
    frequency = 440 + 100 * math.sin(time_val * 0.05)
    
    return [amplitude, frequency]

def collect_sensor_data(sensors_dict, queue, running, paused):
    """Collect data from all sensors and put in queue"""
    global data_queue, sensors
    
    # Store these as module variables so they can be accessed by handlers
    # since the handlers are called outside of this function's context
    data_queue = queue
    sensors = sensors_dict
    
    if not running or paused:
        return
    
    # Get current time
    current_time = time.time()
    
    # Sync any buffered EmotiBit data every second
    if "OSC_EmotiBit" in sensors and sensors["OSC_EmotiBit"]["connected"]:
        sensor_info = sensors["OSC_EmotiBit"]
        if sensor_info["type"] == "real" and "connector" in sensor_info:
            # Use a timestamp-based approach to sync only once per second
            if not hasattr(collect_sensor_data, "last_sync_time"):
                collect_sensor_data.last_sync_time = 0
            
            if current_time - collect_sensor_data.last_sync_time >= 1.0:
                try:
                    # First trim buffers to remove old data
                    sensor_info["connector"].trim_buffers()
                    
                    # Don't clear the buffer, as the OSC thread might still be adding to it
                    sensor_info["connector"].sync_buffered_data(data_queue, clear_buffer=False)
                    collect_sensor_data.last_sync_time = current_time
                except Exception as e:
                    print(f"Error syncing EmotiBit data: {str(e)}")
    
    # Collect data from all sensors
    for sensor_id, sensor_info in sensors.items():
        if sensor_info["connected"]:
            if sensor_info["type"] == "simulated":
                # Generate simulated data based on sensor type
                if "BLE_IMU" in sensor_id:
                    data = simulate_imu_data(current_time)
                elif "OSC_" in sensor_id and sensor_id.count('_') == 1:
                    # The old style OSC generic simulation
                    data = simulate_osc_data(current_time)
                elif "OSC_" in sensor_id:
                    # EmotiBit-style OSC simulation for specific signals
                    signal_type = sensor_id.replace("OSC_", "")
                    data = [simulate_emotibit_data(signal_type, current_time)]
                elif "Audio" in sensor_id:
                    data = simulate_audio_data(current_time)
                else:
                    data = None
                    
                if data:
                    # Put data in queue for processing
                    data_queue.put({
                        "sensor_id": sensor_id,
                        "timestamp": current_time,
                        "data": data
                    })
            elif sensor_info["type"] == "real":
                if "BLE_IMU" in sensor_id:
                    # For real BLE sensors, get all data from the device buffer
                    if "device" in sensor_info and hasattr(sensor_info["device"], "data_buffer") and sensor_info["device"].data_buffer:
                        # Process all samples in the buffer
                        for sample in sensor_info["device"].data_buffer:
                            if sample is not None:
                                # Extract quaternion and Euler angles
                                # Format: [timestamp, battery, cal, qw, qx, qy, qz, roll, pitch, yaw]
                                qw, qx, qy, qz = sample[3:7]
                                roll, pitch, yaw = sample[7:10]
                                
                                # Ensure data is in the correct format: [qw, qx, qy, qz, roll, pitch, yaw]
                                data = [float(qw), float(qx), float(qy), float(qz), 
                                      float(roll), float(pitch), float(yaw)]
                                
                                # Put data in queue for processing with the sample's timestamp
                                data_queue.put({
                                    "sensor_id": sensor_id,
                                    "timestamp": float(sample[0]),  # Ensure timestamp is float
                                    "data": data
                                })
                        
                        # Clear the buffer after processing
                        sensor_info["device"].data_buffer = []
                elif "Audio" in sensor_id and "processor" in sensor_info:
                    try:
                        # For real audio sensor, get features from the processor
                        audio_features = sensor_info["processor"].get_audio_features()
                        if audio_features is not None:
                            # Process audio features to match simulated data format [amplitude, frequency]
                            data = process_audio_data(audio_features)
                            if data is not None:
                                # Put data in queue for processing
                                data_queue.put({
                                    "sensor_id": sensor_id,
                                    "timestamp": audio_features['timestamp'],
                                    "data": data
                                })
                    except Exception as e:
                        print(f"Error processing audio data: {str(e)}")
                # Note: OSC sensors with type="real" don't need handling here
                # they're automatically handled by the emotibit_handler callback

def simulate_emotibit_data(signal_type, time_val):
    """Simulate EmotiBit sensor data for a specific signal type"""
    if "ACC:" in signal_type:
        return math.sin(time_val * 0.5) * 0.5  # Simulate accelerometer data
    elif "GYRO:" in signal_type:
        return math.cos(time_val * 0.3) * 10.0  # Simulate gyroscope data
    elif "MAG:" in signal_type:
        return math.sin(time_val * 0.1) * 50.0  # Simulate magnetometer data
    elif "PPG:" in signal_type:
        return math.sin(time_val * 1.0) * 100.0 + 500.0  # Simulate PPG signal
    elif "TEMP" in signal_type or "THERM" in signal_type:
        return 36.5 + math.sin(time_val * 0.05) * 0.5  # Simulate temperature ~37°C
    elif "ED" in signal_type:  # EDA, EDL, EDR
        return 2.0 + math.sin(time_val * 0.2) * 0.5  # Simulate EDA around 2 µS
    elif "SCR:" in signal_type:
        return math.sin(time_val * 0.3) * 0.2  # Simulate SCR metrics
    elif "HR" == signal_type:
        return 70.0 + math.sin(time_val * 0.1) * 5.0  # Simulate HR ~70 BPM
    elif "IBI" == signal_type:
        return 0.85 + math.sin(time_val * 0.1) * 0.05  # Simulate IBI ~850ms
    elif "HUMIDITY" == signal_type:
        return 40.0 + math.sin(time_val * 0.05) * 5.0  # Simulate humidity ~40%
    else:
        return math.sin(time_val * 0.2) * 5.0  # Default simulation

def start_file_batch_writer():
    """Start the background thread for batch file writing"""
    global file_batch_thread, file_batch_stop_event
    
    if file_batch_thread is not None and file_batch_thread.is_alive():
        return  # Thread already running
    
    file_batch_stop_event = threading.Event()
    file_batch_thread = threading.Thread(target=file_batch_worker, daemon=True)
    file_batch_thread.start()

def stop_file_batch_writer():
    """Stop the background thread for batch file writing"""
    global file_batch_thread, file_batch_stop_event
    
    if file_batch_thread is not None and file_batch_thread.is_alive():
        file_batch_stop_event.set()
        file_batch_thread.join(timeout=2.0)
        
        # Flush any remaining data
        with file_batch_lock:
            for file_key, batch_info in file_write_batches.items():
                if batch_info["data"]:
                    write_batch_to_file(file_key)
        
        file_batch_thread = None

def file_batch_worker():
    """Background worker that periodically writes batched data to files"""
    last_check_time = time.time()
    
    while not file_batch_stop_event.is_set():
        current_time = time.time()
        flush_needed = False
        
        # Check if any batches need to be written due to timeout
        if current_time - last_check_time >= 0.1:  # Check every 100ms
            with file_batch_lock:
                for file_key, batch_info in list(file_write_batches.items()):
                    if (batch_info["data"] and 
                        (current_time - batch_info["last_update"] >= FILE_BATCH_TIMEOUT or 
                         len(batch_info["data"]) >= FILE_BATCH_SIZE)):
                        write_batch_to_file(file_key)
                        flush_needed = True
            
            last_check_time = current_time
        
        # Sleep a bit to avoid busy waiting
        time.sleep(0.01)

def write_batch_to_file(file_key):
    """Write a batch of data to a file"""
    if file_key not in file_write_batches:
        return
    
    batch_info = file_write_batches[file_key]
    if not batch_info["data"]:
        return
    
    data_dir, filename = file_key
    full_path = os.path.join(data_dir, filename)
    
    try:
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        # Check if we need to write a header
        file_exists = os.path.exists(full_path) and os.path.getsize(full_path) > 0
        
        # Open the file with explicit closing to ensure file handles are released
        f = None
        try:
            f = open(full_path, "a")
            if not file_exists:
                # Write header if file is new
                f.write(batch_info["header"] + "\n")
            
            # Write all data at once
            f.write("".join(batch_info["data"]))
            
            # Explicitly flush to ensure data is written
            f.flush()
        finally:
            # Ensure file is closed even if an error occurs
            if f is not None:
                f.close()
        
        # Clear the batch after successful write
        batch_info["data"] = []
        
    except Exception as e:
        print(f"Error writing batch to file {full_path}: {str(e)}")
        # Don't clear the batch on error, so we can retry later

def get_sensor_file_header(sensor_id):
    """Get the CSV header for a specific sensor type"""
    if "BLE_IMU" in sensor_id:
        return "timestamp,qw,qx,qy,qz,roll,pitch,yaw"
    elif "OSC_" in sensor_id and sensor_id.count('_') == 1:
        # Old generic OSC style
        return "timestamp,value"
    elif "OSC_" in sensor_id:
        # EmotiBit-style OSC for specific signals
        signal_type = sensor_id.replace("OSC_", "")
        if signal_type in EMOTIBIT_SIGNAL_TYPES:
            return f"timestamp,{EMOTIBIT_SIGNAL_TYPES[signal_type]}"
        else:
            return "timestamp,value"
    elif "Audio" in sensor_id:
        return "timestamp,amplitude,frequency"
    return "timestamp,data"

def add_to_write_batch(sensor_id, timestamp, data, data_dir):
    """Add a data item to the write batch for a specific sensor"""
    # Get device name for BLE sensors
    device_name = ""
    if "BLE_IMU" in sensor_id:
        for device in ble_devices:
            if device.idx == int(sensor_id.split("_")[-1]) - 1:  # Subtract 1 to convert from 1-based to 0-based indexing
                device_name = f"_{device.name}"
                break
    
    # Create filename
    filename = f"{sensor_id}{device_name}.csv"
    file_key = (data_dir, filename)
    
    # Format data for writing
    formatted_data = [f"{x:.3f}" for x in data]
    formatted_timestamp = f"{timestamp:.3f}"
    line = f"{formatted_timestamp},{','.join(formatted_data)}\n"
    
    with file_batch_lock:
        # Initialize batch if it doesn't exist
        if file_key not in file_write_batches:
            file_write_batches[file_key] = {
                "data": [],
                "header": get_sensor_file_header(sensor_id),
                "last_update": time.time()
            }
        
        # Add line to batch
        file_write_batches[file_key]["data"].append(line)
        file_write_batches[file_key]["last_update"] = time.time()
        
        # Write immediately if batch is full
        if len(file_write_batches[file_key]["data"]) >= FILE_BATCH_SIZE:
            write_batch_to_file(file_key)

def process_sensor_data(data_item, data_dir, current_session, session_types):
    """Process and save sensor data"""
    sensor_id = data_item["sensor_id"]
    timestamp = data_item["timestamp"]
    data = data_item["data"]
    
    # Add data to in-memory buffer
    if sensor_id not in sensor_data_buffer:
        sensor_data_buffer[sensor_id] = []
    
    sensor_data_buffer[sensor_id].append((timestamp, data))
    
    # Trim buffer to keep only recent data
    current_time = time.time()
    sensor_data_buffer[sensor_id] = [
        (ts, d) for ts, d in sensor_data_buffer[sensor_id] 
        if current_time - ts <= buffer_max_age
    ]
    
    # Add to write batch (batch file I/O)
    add_to_write_batch(sensor_id, timestamp, data, data_dir)

def get_recent_sensor_data(seconds=5):
    """
    Get the last N seconds of sensor data for all sensors
    
    Args:
        seconds: Number of seconds of data to retrieve
        
    Returns:
        Dictionary containing recent sensor data organized by sensor type
    """
    current_time = time.time()
    min_timestamp = current_time - seconds
    
    # Create a dictionary to hold all recent sensor data
    recent_data = {
        "timestamp_range": (min_timestamp, current_time),
        "imu_sensors": {},
        "osc_sensors": {},
        "audio_sensors": {}
    }
    
    # Process each sensor's data
    for sensor_id, data_list in sensor_data_buffer.items():
        # Filter data by timestamp
        recent_sensor_data = [
            (ts, d) for ts, d in data_list 
            if ts >= min_timestamp
        ]
        
        if not recent_sensor_data:
            continue
            
        # Organize data by sensor type
        if "BLE_IMU" in sensor_id:
            recent_data["imu_sensors"][sensor_id] = {
                "timestamps": [ts for ts, _ in recent_sensor_data],
                "data": [d for _, d in recent_sensor_data]
            }
        elif "OSC_" in sensor_id:
            recent_data["osc_sensors"][sensor_id] = {
                "timestamps": [ts for ts, _ in recent_sensor_data],
                "data": [d for _, d in recent_sensor_data]
            }
        elif "Audio" in sensor_id:
            recent_data["audio_sensors"][sensor_id] = {
                "timestamps": [ts for ts, _ in recent_sensor_data],
                "data": [d for _, d in recent_sensor_data]
            }
    
    return recent_data

def print_ble_roll(recent_data, ble_idx):
    """
    Print the roll values from BLE_IMU sensor
    
    Returns:
        List of roll values if sensor data exists, otherwise None
    """
    
    # Look for BLE_IMU_ble_idx in the imu_sensors
    ble_data = None
    for sensor_id, sensor_data in recent_data["imu_sensors"].items():
        if sensor_id == f"BLE_IMU_{ble_idx}" or (sensor_id.startswith("BLE_IMU") and f"{ble_idx}" in sensor_id):
            ble_data = sensor_data
            break
    
    if ble_data is None:
        print(f"No data found for BLE_IMU_{ble_idx}")
        return None
    
    # Extract roll values (roll is at index 4 in IMU data format [qw, qx, qy, qz, roll, pitch, yaw])
    roll_values = []
    for data_point in ble_data["data"]:
        if len(data_point) >= 5:  # Make sure data has enough elements
            roll_values.append(data_point[4])
    
    # Print the roll values
    if roll_values:
        print(f"Roll values from BLE_IMU_{ble_idx}:")
        for i, roll in enumerate(roll_values):
            print(f"Sample {i+1}: {roll:.4f} degrees")
        return roll_values
    else:
        print(f"No roll data available for BLE_IMU_{ble_idx}")
        return None

def print_emotibit_ppg_ir(recent_data):
    """
    Print the PPG:IR values from EmotiBit sensor
    
    Args:
        recent_data: Dictionary containing recent sensor data
        
    Returns:
        List of PPG:IR values if sensor data exists, otherwise None
    """
    # Look for OSC_PPG:IR in the osc_sensors
    ppg_ir_data = None
    sensor_id = "OSC_PPG:IR"
    
    if "osc_sensors" in recent_data:
        for sid, sensor_data in recent_data["osc_sensors"].items():
            if sid == sensor_id:
                ppg_ir_data = sensor_data
                break
    
    if ppg_ir_data is None:
        print("No data found for EmotiBit PPG:IR")
        return None
    
    # Extract PPG:IR values (data is a list of lists, with one value per sample)
    ppg_ir_values = []
    for data_point in ppg_ir_data["data"]:
        if len(data_point) >= 1:  # Make sure data has at least one element
            ppg_ir_values.append(data_point[0])
    
    # Print the PPG:IR values
    if ppg_ir_values:
        print(f"PPG:IR values from EmotiBit:")
        for i, value in enumerate(ppg_ir_values):
            print(f"Sample {i+1}: {value:.2f}")
        return ppg_ir_values
    else:
        print("No PPG:IR data available from EmotiBit")
        return None

def get_samples():

    """Get the last N seconds of sensor data for all sensors"""
    requested_seconds = 30
    recent_data = get_recent_sensor_data(seconds=requested_seconds)

    # Count samples for each sensor type
    imu_sample_count = 0
    osc_sample_count = 0
    audio_sample_count = 0
    
    # Count IMU sensor samples per device
    imu_samples_by_device = {}
    for sensor_id, sensor_data in recent_data["imu_sensors"].items():
        device_name = f"BLE_IMU_{sensor_id.split('_')[-1]}"
        imu_samples_by_device[device_name] = len(sensor_data["data"])
        imu_sample_count += len(sensor_data["data"])
    
    # Count OSC sensor samples
    osc_samples_by_signal = {}
    for sensor_id, sensor_data in recent_data["osc_sensors"].items():
        signal_count = len(sensor_data["data"])
        osc_sample_count += signal_count
        
        # For EmotiBit signals, track by signal type
        if sensor_id.replace("OSC_", "") in EMOTIBIT_SIGNAL_TYPES:
            signal_type = sensor_id.replace("OSC_", "")
            signal_name = EMOTIBIT_SIGNAL_TYPES[signal_type]
            osc_samples_by_signal[signal_name] = signal_count
    
    # Count Audio sensor samples
    for sensor_id, sensor_data in recent_data["audio_sensors"].items():
        audio_sample_count += len(sensor_data["data"])
    
    # Store sample counts in a dictionary
    sample_counts = {
        "imu_samples": imu_sample_count,
        "imu_samples_by_device": imu_samples_by_device,
        "osc_samples": osc_sample_count,
        "osc_samples_by_signal": osc_samples_by_signal,
        "audio_samples": audio_sample_count,
        "total_samples": imu_sample_count + osc_sample_count + audio_sample_count
    }

    print("Sample counts:")
    print(f"Total IMU samples: {imu_sample_count}")
    print("IMU samples by device:")
    for device, count in imu_samples_by_device.items():
        # Get actual device name from ble_devices list
        actual_device_name = "Unknown"
        device_idx = device.split("_")[-1]
        if device_idx.isdigit():
            for ble_device in ble_devices:
                if ble_device.idx == int(device_idx) - 1:  # Convert from 1-based to 0-based indexing
                    actual_device_name = ble_device.name
                    break
                
        print(f"  {device} ({actual_device_name}): {count} samples")
    print(f"Total OSC samples: {osc_sample_count}")
    print("OSC samples by signal:")
    for signal, count in osc_samples_by_signal.items():
        print(f"  {signal}: {count} samples")
    print(f"Audio samples: {audio_sample_count}")
    print(f"Total samples: {sample_counts['total_samples']}")

    return recent_data, sample_counts

class EmbeddingNet(nn.Module):
    """
    A multi-layer perceptron that embeds input features into a latent space.
    Replicates the architecture from the original model.
    """

    def __init__(self, input_dim: int, embedding_dim: int = 64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),
            nn.Linear(128, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DeepPrototypeModelLoader:
    """
    Standalone class for loading and using a saved deep prototype model.
    """
    
    def __init__(self, model_dir: str, device: str = "cpu"):
        """
        Initialize the model loader by loading all necessary components.
        
        Args:
            model_dir: Directory containing the saved model artifacts
            device: Device to load the model onto ("cpu" or "cuda")
        """
        self.model_dir = model_dir
        self.device = torch.device(device)
        
        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
        # Load feature names
        feature_names_path = os.path.join(model_dir, "feature_names.pkl")
        if not os.path.exists(feature_names_path):
            raise FileNotFoundError(f"Feature names file not found: {feature_names_path}")
        with open(feature_names_path, "rb") as f:
            self.feature_names = pickle.load(f)
        
        # Load scaler
        scaler_path = os.path.join(model_dir, "scaler.pkl")
        if not os.path.exists(scaler_path):
            raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        
        # Load model configuration
        config_path = os.path.join(model_dir, "model_config.pkl")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Model config file not found: {config_path}")
        with open(config_path, "rb") as f:
            model_config = pickle.load(f)
        
        input_dim = model_config["input_dim"]
        embedding_dim = model_config["embedding_dim"]
        
        # Initialize and load model
        model_path = os.path.join(model_dir, "model.pt")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found: {model_path}")
            
        self.model = EmbeddingNet(input_dim, embedding_dim).to(self.device)
        self.model.load_state_dict(torch.load(
            model_path, 
            map_location=self.device
        ))
        self.model.eval()
        
        # Load prototypes
        prototypes_path = os.path.join(model_dir, "prototypes.pkl")
        if not os.path.exists(prototypes_path):
            raise FileNotFoundError(f"Prototypes file not found: {prototypes_path}")
        with open(prototypes_path, "rb") as f:
            prototype_dict = pickle.load(f)
            # Convert numpy arrays back to torch tensors
            self.prototypes = {k: torch.tensor(v, device=self.device) for k, v in prototype_dict.items()}
        
        # Create a mapping for prototype labels
        self.prototype_labels = ['Negative']
        for key in self.prototypes.keys():
            if key != 0:  # 0 is the negative prototype
                self.prototype_labels.append(str(key))
    
    def preprocess_features(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Preprocess the input features using the saved scaler.
        Ensures all expected features are present, adding zero columns for missing features.
        
        Args:
            features_df: DataFrame containing raw features
            
        Returns:
            Scaled feature array ready for model input
        """
        # Create a copy to avoid modifying the original DataFrame
        features_df = features_df.copy()
        
        # Check and add missing features all at once using a dictionary
        missing_features = {}
        for feature in self.feature_names:
            if feature not in features_df.columns:
                missing_features[feature] = 0
                
        # Add all missing columns at once if any
        if missing_features:
            for feature, value in missing_features.items():
                features_df[feature] = value
        
        # Keep only needed features and in the right order
        X = features_df[self.feature_names]
        
        # Scale the features
        X_scaled = self.scaler.transform(X)
        return X_scaled
    
    def predict(self, features_df: pd.DataFrame, smoothing_sigma: float = 2.0) -> pd.DataFrame:
        """
        Generate predictions for input features.
        
        Args:
            features_df: DataFrame containing features for prediction
            smoothing_sigma: Sigma parameter for Gaussian smoothing of probabilities
            
        Returns:
            DataFrame with prediction results
        """
        # Preprocess features
        X_scaled = self.preprocess_features(features_df)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32).to(self.device)
        
        # Get timestamps if available, otherwise create index-based timestamps
        if "timestamp" in features_df.columns:
            timestamps = features_df["timestamp"].values
        else:
            timestamps = np.arange(len(features_df))
        
        # Generate embeddings
        with torch.no_grad():
            embeddings = self.model(X_tensor)
        
        # Prepare prototype tensors for distance calculation
        all_prototype_tensors = []
        prototype_labels = []
        
        # Add the negative prototype first
        all_prototype_tensors.append(self.prototypes[0].unsqueeze(0))
        prototype_labels.append('Negative')
        
        # Add all positive prototypes
        for category, proto in self.prototypes.items():
            if category != 0:
                all_prototype_tensors.append(proto.unsqueeze(0))
                prototype_labels.append(str(category))
        
        # Stack all prototypes
        all_prototypes_stacked = torch.cat(all_prototype_tensors, dim=0)
        
        # Compute distances to all prototypes
        dists = torch.cdist(embeddings, all_prototypes_stacked, p=2) ** 2
        
        # Apply softmax to get probabilities across all prototypes
        probs = nn.functional.softmax(-dists, dim=1)
        
        # Sum probabilities of all positive prototypes (all except 'Negative' which is at index 0)
        prob_positive = 1 - probs[:, 0].cpu().numpy()
        
        # Apply Gaussian smoothing if requested
        if smoothing_sigma > 0:
            prob_positive = gaussian_filter1d(prob_positive, sigma=smoothing_sigma)
        
        # Get the most likely prototype for each embedding
        most_likely_prototype_idx = torch.argmin(dists, dim=1).cpu().numpy()
        most_likely_prototype = [prototype_labels[idx] for idx in most_likely_prototype_idx]
        
        # Get the most likely POSITIVE prototype (ignoring the negative prototype)
        # First, create a version of distances with the negative prototype set to infinity
        positive_only_dists = dists.clone()
        positive_only_dists[:, 0] = float('inf')  # Set distance to negative prototype to infinity
        
        # Now find the most likely positive prototype
        most_likely_positive_idx = torch.argmin(positive_only_dists, dim=1).cpu().numpy()
        most_likely_positive = [prototype_labels[idx] for idx in most_likely_positive_idx]
        
        # Get the probability of the most likely positive prototype
        most_likely_positive_prob = probs.gather(1, torch.tensor(most_likely_positive_idx, device=self.device).unsqueeze(1)).squeeze().cpu().numpy()
        
        # Create predictions DataFrame
        predictions = pd.DataFrame({
            "timestamp": timestamps,
            "prob_positive": prob_positive,
            "most_likely_prototype": most_likely_prototype,
            "most_likely_positive": most_likely_positive,
            "most_likely_positive_prob": most_likely_positive_prob,
            "is_positive": (most_likely_prototype_idx != 0).astype(int)
        })
        
        return predictions

def run_ml_prediction(participant_data):
    """Run machine learning prediction on sensor data"""

    global model_loader
    if model_loader is None:
        models_base_dir = "models"
        available_models = os.listdir(models_base_dir)
        model_loader = DeepPrototypeModelLoader(model_dir="models/")

    # Get the last 5 seconds of sensor data
    recent_data, sample_counts = get_samples()
    
    # Map quaternion components to expected feature naming
    component_mapping = {
        "qw": "W",
        "qx": "X",
        "qy": "Y",
        "qz": "Z"
    }
    
    # Initialize feature dictionary with a single timestamp
    feature_dict = {"timestamp": [time.time()]}
    
    # Initialize all features to 0.0 to handle missing data
    for feature in model_loader.feature_names:
        feature_dict[feature] = [0.0]
    
    # Calculate statistics for each IMU sensor and map to feature dict
    if sample_counts["imu_samples"] > 0:
        for sensor_id, sensor_data in recent_data["imu_sensors"].items():
            # Extract the node name from the sensor ID
            device_idx = sensor_id.split("_")[-1]
            node_name = None
            if device_idx.isdigit():
                for device in ble_devices:
                    if device.idx == int(device_idx) - 1:  # Convert from 1-based to 0-based indexing
                        node_name = device.name
                        break
            
            if node_name is None:
                print(f"Could not find device name for {sensor_id}, skipping")
                continue  # Skip if we can't find the device name
            
            # Get quaternion values (qw, qx, qy, qz are at indices 0, 1, 2, 3 in the data format)
            quat_data = np.array([data_point[:4] for data_point in sensor_data["data"]])
            
            # Only calculate if we have enough data points
            if len(quat_data) > 0:
                component_names = ["qw", "qx", "qy", "qz"]
                
                # Calculate statistics for each quaternion component
                for i, comp_name in enumerate(component_names):
                    values = quat_data[:, i]
                    
                    # Skip if not enough values
                    if len(values) < 2:
                        continue
                    
                    # Map component name to expected format
                    comp_mapped = component_mapping[comp_name]
                    
                    # Basic statistics
                    mean_val = float(np.mean(values))
                    min_val = float(np.min(values))
                    max_val = float(np.max(values))
                    std_val = float(np.std(values))
                    
                    # Calculate rolling statistics (window of 3 samples or 1/3 of data points, whichever is larger)
                    window_size = max(3, len(values) // 3)
                    if len(values) >= window_size:
                        rolling_mean = np.mean(values[-window_size:])
                        rolling_std = np.std(values[-window_size:])
                    else:
                        rolling_mean = mean_val
                        rolling_std = std_val
                    
                    # Calculate trend (slope of linear regression)
                    x = np.arange(len(values))
                    A = np.vstack([x, np.ones(len(x))]).T
                    slope, _ = np.linalg.lstsq(A, values, rcond=None)[0]
                    trend = float(slope)
                    
                    # Construct feature names
                    feature_prefix = f"Wing_{node_name}_0000_{comp_mapped}"
                    
                    # Store values in feature_dict (just one value per feature)
                    for stat, value in [
                        ("mean", mean_val),
                        ("min", min_val),
                        ("max", max_val),
                        ("std", std_val),
                        ("rolling_mean", rolling_mean),
                        ("rolling_std", rolling_std),
                        ("trend", trend)
                    ]:
                        feature_name = f"{feature_prefix}_{stat}"
                        if feature_name in feature_dict:
                            # Set single value
                            feature_dict[feature_name] = [value]
    
    # Create DataFrame with features
    example_features = pd.DataFrame(feature_dict)
    
    # Generate predictions
    predictions = model_loader.predict(example_features)
    print("\nModel predictions:")
    print(predictions)

    # Write predictions to file - use the same data_dir as other sensor files
    current_time = time.time()
    
    # Look for an existing data directory in file_write_batches
    data_dir = None
    with file_batch_lock:
        # Find the first active data directory from existing batch files
        for file_key in file_write_batches.keys():
            if file_key and isinstance(file_key, tuple) and len(file_key) > 0:
                data_dir = file_key[0]
                if data_dir and not data_dir.startswith("data/"):
                    # Found a valid data directory
                    break
    
    # If no data directory found in batches, try participant_data
    if not data_dir and participant_data and "data_dir" in participant_data and participant_data["data_dir"]:
        data_dir = participant_data["data_dir"]
    
    # If still no data directory, use fallback
    if not data_dir:
        data_dir = os.path.join("data", datetime.datetime.now().strftime("%Y-%m-%d"))
    
    filename = "ML_Predictions.csv"
    file_key = (data_dir, filename)
    
    # Prepare prediction data for writing
    prediction_data = {
        "timestamp": current_time,
        "prob_positive": predictions['prob_positive'].values[0]
    }
    
    # Format as CSV line
    formatted_timestamp = f"{prediction_data['timestamp']:.3f}"
    line = f"{formatted_timestamp},{prediction_data['prob_positive']:.4f}\n"
    
    with file_batch_lock:
        # Initialize batch if it doesn't exist
        if file_key not in file_write_batches:
            file_write_batches[file_key] = {
                "data": [],
                "header": "timestamp,prob_positive",
                "last_update": time.time()
            }
        
        # Add line to batch
        file_write_batches[file_key]["data"].append(line)
        file_write_batches[file_key]["last_update"] = time.time()
        
        # Write immediately if batch is full
        if len(file_write_batches[file_key]["data"]) >= FILE_BATCH_SIZE:
            write_batch_to_file(file_key)
    
    # Return both the prediction and sample counts
    return predictions['prob_positive'].values[0], sample_counts

# BLE connection utilities from test.py
async def connect_ble_sensors():
    """Connect to real BLE sensors using code from test.py"""
    global ble_devices, freq, ble_connection_monitor, ble_monitor_stop_event, ble_main_event_loop
    
    # Store reference to the current event loop (main thread's loop)
    ble_main_event_loop = asyncio.get_event_loop()
    
    # Ensure the batch file writer thread is running
    if file_batch_thread is None or not file_batch_thread.is_alive():
        start_file_batch_writer()
    
    # Scan for BLE devices
    devices = await BleakScanner.discover()
    ble_nodes = [d for d in devices if d.name and "Node" in d.name]
    
    if not ble_nodes:
        return []
    
    # Create device objects for each BLE node
    ble_devices = []
    for i, device in enumerate(ble_nodes):
        ble_devices.append(BLEDevice(device.address, device.name, i, ble_main_event_loop))
    
    # Connect to each device
    connected_devices = []
    for device in ble_devices:
        success = await device.connect()
        if success:
            connected_devices.append(device)
    
    # Start the connection monitor if not already running
    if (ble_connection_monitor is None or not ble_connection_monitor.is_alive()) and connected_devices:
        ble_monitor_stop_event.clear()
        ble_connection_monitor = threading.Thread(target=ble_connection_monitor_thread, daemon=True)
        ble_connection_monitor.start()
        print("BLE connection monitor started")
    
    return connected_devices

def ble_connection_monitor_thread():
    """Background thread that monitors BLE connections and attempts reconnection"""
    global ble_devices, ble_monitor_stop_event, ble_main_event_loop, ble_reconnecting_devices
    
    print("BLE connection monitor thread started")
    
    while not ble_monitor_stop_event.is_set():
        try:
            # Sleep at the start to give devices time to initialize
            time.sleep(BLE_CONNECTION_CHECK_INTERVAL)
            
            # Check each device's connection status
            for device in ble_devices:
                if device.client is not None:
                    # Check if disconnected
                    device_key = f"{device.idx}_{device.name}"
                    if not device.client.is_connected and device.connected:
                        print(f"Detected disconnection of {device.name} (idx: {device.idx})")
                        device.connected = False
                        
                        # Only start reconnection if not already trying to reconnect this device
                        if device_key not in ble_reconnecting_devices:
                            ble_reconnecting_devices[device_key] = True
                            
                            # Start reconnection in a background thread
                            threading.Thread(
                                target=lambda idx=device.idx: handle_device_reconnection(idx), 
                                daemon=True
                            ).start()
        except Exception as e:
            print(f"Error in BLE connection monitor: {str(e)}")
    
    print("BLE connection monitor thread stopped")

def handle_device_reconnection(device_idx):
    """Handle reconnection with automatic retries and backoff"""
    global ble_devices, ble_reconnecting_devices
    
    if device_idx < 0 or device_idx >= len(ble_devices):
        print(f"Invalid device index: {device_idx}")
        return
    
    device = ble_devices[device_idx]
    device_key = f"{device_idx}_{device.name}"
    
    # Reset reconnection attempts to ensure we get a fresh start
    device.reconnect_attempts = 0
    
    # Try to reconnect with exponential backoff
    success = False
    current_delay = BLE_RECONNECT_DELAY
    
    while not success and device.reconnect_attempts < BLE_RECONNECT_ATTEMPTS:
        try:
            print(f"Attempting to reconnect to {device.name} (idx: {device_idx})...")
            
            # Run the reconnection in an async context
            success = asyncio.run(reconnect_ble_device(device_idx))
            
            if success:
                print(f"Successfully reconnected to {device.name}")
                break
            
            # If we get here, reconnection failed but didn't raise an exception
            # Calculate next delay with exponential backoff (max 30 seconds)
            current_delay = min(current_delay * 1.5, BLE_RECONNECT_MAX_DELAY)
            print(f"Reconnection attempt {device.reconnect_attempts}/{BLE_RECONNECT_ATTEMPTS} failed. " +
                  f"Retrying in {current_delay:.1f} seconds...")
            
            # Wait before next attempt
            time.sleep(current_delay)
            
        except Exception as e:
            print(f"Error during reconnection of {device.name}: {str(e)}")
            device.reconnect_attempts += 1
            
            # Calculate next delay with exponential backoff
            current_delay = min(current_delay * 1.5, BLE_RECONNECT_MAX_DELAY)
            print(f"Reconnection attempt {device.reconnect_attempts}/{BLE_RECONNECT_ATTEMPTS} failed with error. " +
                  f"Retrying in {current_delay:.1f} seconds...")
            
            # Wait before next attempt
            time.sleep(current_delay)
    
    # Remove from reconnecting devices list
    if device_key in ble_reconnecting_devices:
        del ble_reconnecting_devices[device_key]
    
    if not success:
        print(f"All reconnection attempts for {device.name} failed after {device.reconnect_attempts} tries")

def stop_ble_connection_monitor():
    """Stop the BLE connection monitor thread"""
    global ble_connection_monitor, ble_monitor_stop_event
    
    if ble_connection_monitor is not None and ble_connection_monitor.is_alive():
        ble_monitor_stop_event.set()
        ble_connection_monitor.join(timeout=2.0)
        print("BLE connection monitor stopped")

async def reconnect_ble_device(device_idx):
    """Reconnect to a specific BLE device by index"""
    global ble_devices, ble_main_event_loop, ble_reconnection_lock
    
    # Use a lock to prevent multiple simultaneous reconnection attempts to the same device
    with ble_reconnection_lock:
        if device_idx < 0 or device_idx >= len(ble_devices):
            print(f"Invalid device index: {device_idx}")
            return False
        
        device = ble_devices[device_idx]
        
        # Set the event loop for reconnection to be the main event loop
        if ble_main_event_loop and ble_main_event_loop.is_running():
            # This reconnection is happening in a new thread, so we need to run it in the main loop
            future = asyncio.run_coroutine_threadsafe(device.reconnect(), ble_main_event_loop)
            try:
                success = future.result(timeout=10.0)  # Wait for up to 10 seconds
            except Exception as e:
                print(f"Error in reconnection: {str(e)}")
                success = False
        else:
            # Fallback if main loop isn't available
            success = await device.reconnect()
        
        return success

class BLEDevice:
    """Class for handling BLE device connections (from test.py)"""
    def __init__(self, address, name, idx, event_loop=None):
        self.address = address
        self.name = name
        self.idx = idx
        self.client = None
        self.data_buffer = []
        self.raw_buffer = []
        self.connected = False
        self.reconnect_attempts = 0
        self.last_disconnect_time = 0
        self.event_loop = event_loop or asyncio.get_event_loop()
        self.is_reconnecting = False
    
    async def notification_handler(self, sender, data):
        """Handle incoming notifications from this device"""
        try:
            # Store raw data
            self.raw_buffer.append(data)
            
            # Process data
            processed_data = self.process_data(data, freq)
            
            # Append processed data
            for sample in processed_data:
                self.data_buffer.append(sample)
                # print(f"Added sample to buffer: {sample[0]:.3f}")  # Print timestamp of added sample
        except Exception as e:
            print(f"Error in notification handler: {str(e)}")
    
    async def connect(self):
        """Connect to this BLE device"""
        print(f"Connecting to {self.name}...")
        
        try:
            # Initialize client with explicit loop argument
            self.client = BleakClient(self.address, loop=self.event_loop)
            await self.client.connect()
            
            # Subscribe to notifications with this device's handler
            await self.client.start_notify(UUID_A, self.notification_handler)
            print(f"Subscribed to notifications for device {self.idx} ({self.name})")
            
            # Send time and start command
            time_bytes = self.clock_in()
            command_time = bytearray([1, 0]) + time_bytes
            await self.client.write_gatt_char(UUID_X, command_time)
            
            # Start device command
            start_command = bytearray([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            await self.client.write_gatt_char(UUID_X, start_command)
            
            print(f"{self.name} connected and started.")
            self.connected = True
            self.reconnect_attempts = 0
            return True
        
        except Exception as e:
            print(f"Failed to connect to {self.name}: {str(e)}")
            self.connected = False
            return False
    
    async def reconnect(self):
        """Reconnect to this BLE device with backoff strategy"""
        if self.reconnect_attempts >= BLE_RECONNECT_ATTEMPTS:
            print(f"Maximum reconnect attempts ({BLE_RECONNECT_ATTEMPTS}) reached for {self.name}")
            return False
        
        # Check if we need to enforce a delay between reconnection attempts
        current_time = time.time()
        if current_time - self.last_disconnect_time < BLE_RECONNECT_DELAY:
            delay = BLE_RECONNECT_DELAY - (current_time - self.last_disconnect_time)
            print(f"Waiting {delay:.1f}s before reconnecting to {self.name}")
            await asyncio.sleep(delay)
        
        self.reconnect_attempts += 1
        print(f"Reconnection attempt {self.reconnect_attempts}/{BLE_RECONNECT_ATTEMPTS} for {self.name}")
        
        # Set reconnecting flag
        self.is_reconnecting = True
        
        # Clean up any existing client
        if self.client:
            try:
                # Only attempt disconnect if it thinks it's connected
                if self.client.is_connected:
                    await self.client.disconnect()
            except Exception as e:
                print(f"Error disconnecting during reconnect: {str(e)}")
            self.client = None
        
        try:
            # Create a new client with the same event loop
            self.client = BleakClient(self.address, loop=self.event_loop)
            
            # Connect to the device
            await self.client.connect(timeout=10.0)  # Add explicit timeout for connection
            
            # Subscribe to notifications with this device's handler
            await self.client.start_notify(UUID_A, self.notification_handler)
            print(f"Resubscribed to notifications for device {self.idx} ({self.name})")
            
            # Send time and start command
            time_bytes = self.clock_in()
            command_time = bytearray([1, 0]) + time_bytes
            await self.client.write_gatt_char(UUID_X, command_time)
            
            # Start device command
            start_command = bytearray([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
            await self.client.write_gatt_char(UUID_X, start_command)
            
            print(f"{self.name} reconnected successfully after {self.reconnect_attempts} attempts")
            self.connected = True
            self.is_reconnecting = False
            return True
            
        except Exception as e:
            print(f"Failed to reconnect to {self.name} (attempt {self.reconnect_attempts}): {str(e)}")
            self.connected = False
            self.last_disconnect_time = time.time()  # Update time for backoff
            self.is_reconnecting = False
            return False
    
    async def disconnect(self):
        """Disconnect from this BLE device"""
        if self.client and self.client.is_connected:
            try:
                await self.client.disconnect()
                print(f"{self.name} disconnected.")
                self.connected = False
                self.last_disconnect_time = time.time()
            except Exception as e:
                print(f"Error disconnecting from {self.name}: {str(e)}")
    
    def bytes_to_double(self, bytes_data):
        """Convert two bytes of data into a signed integer"""
        combined = (bytes_data[1] << 8) | bytes_data[0]
        # Handle sign (two's complement)
        if combined & 0x8000:
            return -((~combined & 0xFFFF) + 1)
        return combined
    
    def clock_in(self):
        """Takes current time and converts to form HET can read"""
        current_time = time.time()
        
        # Split into integer and fractional parts
        int_part = int(current_time)
        frac_part = int((current_time - int_part) * 1e9)
        
        # Convert to bytes (8 bytes total: 4 for integer, 4 for fraction)
        time_bytes = bytearray(8)
        
        # Integer part (big endian)
        time_bytes[0] = (int_part >> 24) & 0xFF
        time_bytes[1] = (int_part >> 16) & 0xFF
        time_bytes[2] = (int_part >> 8) & 0xFF
        time_bytes[3] = int_part & 0xFF
        
        # Fractional part (big endian)
        time_bytes[4] = (frac_part >> 24) & 0xFF
        time_bytes[5] = (frac_part >> 16) & 0xFF
        time_bytes[6] = (frac_part >> 8) & 0xFF
        time_bytes[7] = frac_part & 0xFF
        
        return time_bytes
    
    def clock_out(self, time_bytes):
        """Convert read HET time data into unix timestamp"""
        # Check if we have enough bytes
        if len(time_bytes) < 7:
            print(f"Warning: Received incomplete timestamp data ({len(time_bytes)} bytes)")
            return time.time()
        
        # Integer part (first 4 bytes)
        int_part = (time_bytes[0] << 24) | (time_bytes[1] << 16) | (time_bytes[2] << 8) | time_bytes[3]
        
        # Fractional part (last 3 bytes)
        frac_part = ((time_bytes[4] << 16) | (time_bytes[5] << 8) | time_bytes[6]) / 1e9
        
        return int_part + frac_part
    
    def process_data(self, raw_data, freq):
        """Process raw BLE data into readable form"""
        
        num_samples = 21
        out = np.zeros((num_samples, 10))
        
        # Extract time data - make sure indexing matches MATLAB
        # Using 7 bytes for timestamp as specified
        t = self.clock_out(raw_data[2:9])
        
        # Battery and calibration - note the indexing
        # Using 2 bytes for each value to match MATLAB
        bat = self.bytes_to_double(raw_data[178:180])
        cal = self.bytes_to_double(raw_data[180:182])
        
        # Fill in common data
        out[:, 1] = bat
        out[:, 2] = cal

        # Process each sample
        for i in range(num_samples):
            # Calculate timestamp for this sample
            out[i, 0] = t + (i - 11) * (1 / freq)
            
            # Extract quaternions - using 2 bytes for each value
            qw = self.bytes_to_double(raw_data[10 + i*2 : 12 + i*2]) / 2**14
            qx = self.bytes_to_double(raw_data[52 + i*2 : 54 + i*2]) / 2**14
            qy = self.bytes_to_double(raw_data[94 + i*2 : 96 + i*2]) / 2**14
            qz = self.bytes_to_double(raw_data[136 + i*2 : 138 + i*2]) / 2**14
            
            out[i, 3:7] = [qw, qx, qy, qz]
            
            # Calculate Euler angles from normalized quaternion
            rot = Rotation.from_quat([qx, qy, qz, qw])
            eul = rot.as_euler('xyz', degrees=True)
            out[i, 7:10] = eul
            
        return out

class EmotiBitConnector:
    """Class to connect with EmotiBit OSC server and manage data"""
    def __init__(self):
        self.osc_server = None
        self.is_connected = False
        self.signals = {}
        self.signal_counts = {}
        self.last_cleanup_time = time.time()
    
    def connect(self):
        """Connect to EmotiBit via OSC"""
        try:
            # Check if emotibit.py exists in the current directory
            if not os.path.exists("emotibit.py"):
                print("EmotiBit OSC server script not found")
                return False
            
            # Start EmotiBit OSC server using setup_emotibit_osc
            success = setup_emotibit_osc(enable=True)
            if success:
                self.is_connected = True
                print("Successfully connected to EmotiBit OSC server")
                return True
            else:
                self.is_connected = False
                print("Failed to connect to EmotiBit OSC server")
                return False
        
        except Exception as e:
            print(f"Error connecting to EmotiBit OSC server: {str(e)}")
            self.is_connected = False
            return False
    
    def disconnect(self):
        """Disconnect from EmotiBit OSC server"""
        try:
            # Stop EmotiBit OSC server
            success = setup_emotibit_osc(enable=False)
            self.is_connected = False
            return success
        except Exception as e:
            print(f"Error disconnecting from EmotiBit OSC server: {str(e)}")
            return False
    
    def trim_buffers(self):
        """Trim all EmotiBit buffers to keep only recent data"""
        if not self.is_connected:
            return 0
        
        # Only perform cleanup periodically
        current_time = time.time()
        if current_time - self.last_cleanup_time < 5.0:  # Clean up every 5 seconds
            return 0
            
        total_removed = 0
        
        # Clean up global EmotiBit buffers
        for signal_type in list(emotibit_data_buffers.keys()):
            if signal_type in emotibit_data_buffers:
                buffer = emotibit_data_buffers[signal_type]
                original_len = len(buffer)
                
                # Trim buffer to keep only recent data
                emotibit_data_buffers[signal_type] = [
                    (ts, v) for ts, v in buffer 
                    if current_time - ts <= buffer_max_age
                ]
                
                removed = original_len - len(emotibit_data_buffers[signal_type])
                total_removed += removed
        
        self.last_cleanup_time = current_time
        
        if total_removed > 0:
            print(f"Connector cleaned up {total_removed} outdated EmotiBit samples (older than {buffer_max_age}s)")
            
        return total_removed
    
    def get_available_signals(self):
        """Get list of available signal types"""
        return list(EMOTIBIT_SIGNAL_TYPES.keys())
    
    def get_signal_stats(self):
        """Get statistics for each signal"""
        # First trim buffers to get accurate stats
        self.trim_buffers()
        
        stats = {}
        
        # Count samples for each signal type
        for sensor_id, buffer in sensor_data_buffer.items():
            if sensor_id.startswith("OSC_"):
                signal_type = sensor_id.replace("OSC_", "")
                if signal_type in EMOTIBIT_SIGNAL_TYPES:
                    stats[signal_type] = {
                        "count": len(buffer),
                        "name": EMOTIBIT_SIGNAL_TYPES[signal_type]
                    }
        
        # Also check the local emotibit buffers
        for signal_type, buffer in emotibit_data_buffers.items():
            if signal_type in EMOTIBIT_SIGNAL_TYPES:
                stats[signal_type] = {
                    "count": len(buffer),
                    "name": EMOTIBIT_SIGNAL_TYPES[signal_type],
                    "buffer_only": True
                }
        
        return stats
    
    def sync_buffered_data(self, target_data_queue, clear_buffer=True):
        """Process any buffered data and add it to the data queue
        
        Args:
            target_data_queue: The queue to add data to
            clear_buffer: Whether to clear the buffer after syncing (default: True)
        
        Returns:
            Number of samples synced
        """
        if not self.is_connected:
            return 0
        
        # Trim buffers first to avoid processing old data
        self.trim_buffers()
        
        synced_count = 0
        current_time = time.time()
        
        # Process each signal's buffered data
        for signal_type, buffer in emotibit_data_buffers.items():
            if not buffer:
                continue
                
            # Trim buffer by timestamp first, to avoid processing old data
            buffer_copy = [(ts, val) for ts, val in buffer if current_time - ts <= buffer_max_age]
            
            # Only continue if we have data after trimming
            if not buffer_copy:
                continue
            
            sensor_id = f"OSC_{signal_type}"
            
            # Process all buffered data
            for timestamp, value in buffer_copy:
                # Add to data queue
                target_data_queue.put({
                    "sensor_id": sensor_id,
                    "timestamp": timestamp,
                    "data": [value]
                })
                synced_count += 1
            
            # Clear the buffer after processing if requested
            if clear_buffer:
                buffer.clear()
            else:
                # If not clearing, at least update the buffer with age-limited version
                emotibit_data_buffers[signal_type] = buffer_copy
        return synced_count

def emotibit_handler(signal_type, address, *args):
    """Generic handler for all EmotiBit OSC messages"""
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                # Create the sensor_id from the signal type
                sensor_id = f"OSC_{signal_type}"
                
                # Always store in local buffer regardless of sensors dict state
                # This ensures data is captured even before the app is fully set up
                if signal_type not in emotibit_data_buffers:
                    emotibit_data_buffers[signal_type] = []
                
                # Store the data with timestamp
                emotibit_data_buffers[signal_type].append((timestamp, float(value)))
                
                # Trim buffer to keep only recent data, same as main sensor buffer
                current_time = time.time()
                emotibit_data_buffers[signal_type] = [
                    (ts, v) for ts, v in emotibit_data_buffers[signal_type] 
                    if current_time - ts <= buffer_max_age
                ]
                
                # If sensors dict is available, add to data queue
                global sensors, data_queue
                if sensors is not None and data_queue is not None:
                    try:
                        for app_sensor_id, sensor_info in sensors.items():
                            if app_sensor_id == sensor_id and sensor_info["connected"]:
                                # In real data mode, we get a single value per message
                                data_value = [float(value)]
                                
                                # Add to data queue
                                data_queue.put({
                                    "sensor_id": sensor_id,
                                    "timestamp": timestamp,
                                    "data": data_value
                                })
                                break
                    except (AttributeError, TypeError):
                        # Safely handle the case where sensors isn't ready yet
                        pass

def setup_emotibit_osc(enable=True):
    """Initialize or stop the EmotiBit OSC server"""
    global emotibit_osc_server, emotibit_connected, emotibit_active_signals, emotibit_data_buffers
    
    if enable:
        # If server already exists, don't create another one
        if emotibit_osc_server is not None:
            print("EmotiBit OSC server is already running")
            return True
            
        try:
            from pythonosc import dispatcher
            from pythonosc import osc_server
            import threading
            
            # Reset the data buffers
            emotibit_active_signals = []
            
            # Keep existing data if any, or initialize empty buffers
            if not emotibit_data_buffers:
                emotibit_data_buffers = {}
            
            # Create dispatcher
            osc_dispatcher = dispatcher.Dispatcher()
            
            # Map OSC addresses to handler functions
            for signal_type in EMOTIBIT_SIGNAL_TYPES.keys():
                address = f"/EmotiBit/0/{signal_type}"
                # Use a nested function to avoid the closure issue with the lambda
                def create_handler(sig_type):
                    return lambda addr, *args: emotibit_handler(sig_type, addr, *args)
                
                handler = create_handler(signal_type)
                osc_dispatcher.map(address, handler)
                emotibit_active_signals.append(signal_type)
                
                # Initialize buffer for each signal if it doesn't exist
                if signal_type not in emotibit_data_buffers:
                    emotibit_data_buffers[signal_type] = []
            
            # Set a default handler for unmapped addresses
            osc_dispatcher.set_default_handler(lambda addr, *args: None)
            
            try:
                # Create server with a timeout for operations
                server = osc_server.ThreadingOSCUDPServer(
                    ("127.0.0.1", 12345), osc_dispatcher)
                server.timeout = 0.5  # Add a timeout to avoid blocking forever
                
                # Create a thread to run the server
                server_thread = threading.Thread(target=server.serve_forever, daemon=True)
                server_thread.start()
                
                # Start a buffer cleanup thread
                cleanup_thread = threading.Thread(target=emotibit_buffer_cleanup_worker, daemon=True)
                cleanup_thread.start()
                
                emotibit_osc_server = server
                emotibit_connected = True
                
                print(f"EmotiBit OSC server listening on 127.0.0.1:12345")
                print(f"Monitoring {len(emotibit_active_signals)} EmotiBit signals")
                return True
            except Exception as e:
                print(f"Error starting EmotiBit OSC server: {str(e)}")
                return False
            
        except Exception as e:
            print(f"Error setting up EmotiBit OSC server: {str(e)}")
            emotibit_connected = False
            return False
    
    elif not enable:
        if emotibit_osc_server is not None:
            try:
                # Stop the server
                emotibit_osc_server.shutdown()
                emotibit_osc_server.server_close()
                emotibit_osc_server = None
                emotibit_connected = False
                print("EmotiBit OSC server stopped")
                return True
            except Exception as e:
                print(f"Error stopping EmotiBit OSC server: {str(e)}")
                return False
        else:
            # Server is already stopped
            return True
    
    return True

def emotibit_buffer_cleanup_worker():
    """Background thread that periodically cleans up EmotiBit data buffers"""
    cleanup_stop_event = threading.Event()
    
    try:
        while not cleanup_stop_event.is_set() and emotibit_connected:
            # Sleep for a short time
            time.sleep(2.0)
            
            # Get current time
            current_time = time.time()
            
            # Clean up all EmotiBit buffers
            total_removed = 0
            for signal_type in list(emotibit_data_buffers.keys()):
                if signal_type in emotibit_data_buffers:
                    buffer = emotibit_data_buffers[signal_type]
                    original_len = len(buffer)
                    
                    # Trim buffer to keep only recent data
                    emotibit_data_buffers[signal_type] = [
                        (ts, v) for ts, v in buffer 
                        if current_time - ts <= buffer_max_age
                    ]
                    
                    removed = original_len - len(emotibit_data_buffers[signal_type])
                    total_removed += removed
            
            if total_removed > 0:
                print(f"Cleaned up {total_removed} outdated EmotiBit samples (older than {buffer_max_age}s)")
                
    except Exception as e:
        print(f"Error in EmotiBit buffer cleanup worker: {str(e)}")

def cleanup_sensor_system():
    """Clean up resources when the system is shutting down"""
    # Stop the file batch writer
    stop_file_batch_writer()
    
    # Stop BLE connection monitor
    stop_ble_connection_monitor()
    
    # Stop EmotiBit OSC server if running
    setup_emotibit_osc(enable=False)
    
    # Disconnect all BLE devices
    for device in ble_devices:
        if hasattr(device, 'client') and device.client and device.client.is_connected:
            try:
                # Use the device's event loop for clean disconnection
                if hasattr(device, 'event_loop') and device.event_loop and device.event_loop.is_running():
                    future = asyncio.run_coroutine_threadsafe(device.disconnect(), device.event_loop)
                    future.result(timeout=5.0)
                else:
                    # Fallback to default loop if device loop isn't available
                    asyncio.get_event_loop().run_until_complete(device.disconnect())
            except:
                pass
    
    print("Sensor system cleanup complete")

def init_sensor_system():
    """Initialize sensor system"""
    # Start the file batch writer thread
    start_file_batch_writer()
    
    # Make sure pythonosc package is available
    try:
        import pythonosc
    except ImportError:
        print("Warning: pythonosc package not found. EmotiBit OSC will not work.")
        print("Install with: pip install python-osc")
    
    print("Sensor system initialized")

# Initialize the system when module is imported
init_sensor_system()