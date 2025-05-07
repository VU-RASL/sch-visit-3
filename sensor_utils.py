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

# Data buffer to store recent sensor data
sensor_data_buffer = {}
buffer_max_age = 60  # Maximum age of data to keep in buffer (seconds)

# UUIDs for BLE characteristics (from test.py)
UUID_A = "F00044DC-0451-4000-B000-000000000000"  # Data characteristic
UUID_X = "F000ABCD-0451-4000-B000-000000000000"  # Command characteristic
UUID_SERVICE = "F0002642-0451-4000-B000-000000000000"  # Service UUID

# Global variables for BLE
ble_devices = []
freq = 100  # Default frequency

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

def simulate_tcp_data(time_val):
    """Simulate TCP sensor data"""
    value1 = math.sin(time_val * 0.1) * 10
    value2 = math.cos(time_val * 0.2) * 5
    value3 = math.sin(time_val * 0.3) * 2
    
    return [value1, value2, value3]

def simulate_audio_data(time_val):
    """Simulate audio sensor data"""
    amplitude = 0.5 + 0.5 * math.sin(time_val * 0.1)
    frequency = 440 + 100 * math.sin(time_val * 0.05)
    
    return [amplitude, frequency]

def collect_sensor_data(sensors, data_queue, running, paused):
    """Collect data from all sensors and put in queue"""
    if not running or paused:
        return
    
    # Get current time
    current_time = time.time()
    
    # Collect data from all sensors
    for sensor_id, sensor_info in sensors.items():
        if sensor_info["connected"]:
            if sensor_info["type"] == "simulated":
                # Generate simulated data based on sensor type
                if "BLE_IMU" in sensor_id:
                    data = simulate_imu_data(current_time)
                elif "TCP" in sensor_id:
                    data = simulate_tcp_data(current_time)
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
                    # For real audio sensor, get features from the processor
                    audio_features = sensor_info["processor"].get_audio_features()
                    if audio_features is not None:
                        # Process audio features for ML prediction
                        prediction = process_audio_data(audio_features)
                        if prediction is not None:
                            # Format data as [amplitude, frequency] for consistency with simulated data
                            data = [prediction, audio_features['zero_crossing_rate']]
                            
                            # Put data in queue for processing
                            data_queue.put({
                                "sensor_id": sensor_id,
                                "timestamp": audio_features['timestamp'],
                                "data": data
                            })

def process_sensor_data(data_item, data_dir, current_session, session_types):
    """Process and save sensor data"""
    sensor_id = data_item["sensor_id"]
    timestamp = data_item["timestamp"]
    data = data_item["data"]
    
    # Add data to buffer
    if sensor_id not in sensor_data_buffer:
        sensor_data_buffer[sensor_id] = []
    
    sensor_data_buffer[sensor_id].append((timestamp, data))
    
    # Trim buffer to keep only recent data
    current_time = time.time()
    sensor_data_buffer[sensor_id] = [
        (ts, d) for ts, d in sensor_data_buffer[sensor_id] 
        if current_time - ts <= buffer_max_age
    ]
    
    # Save data to file
    if current_session < len(session_types):
        session_type = session_types[current_session]
        filename = os.path.join(data_dir, f"session_{current_session}_{session_type}_{sensor_id}.csv")
        
        # Append data to file
        with open(filename, "a") as f:
            if os.path.getsize(filename) == 0:
                # Write header if file is empty
                if "BLE_IMU" in sensor_id:
                    f.write("timestamp,qw,qx,qy,qz,roll,pitch,yaw\n")
                elif "TCP" in sensor_id:
                    f.write("timestamp,value1,value2,value3\n")
                elif "Audio" in sensor_id:
                    f.write("timestamp,amplitude,frequency\n")
            
            # Format all numeric values to 2 decimal places
            formatted_data = [f"{x:.3f}" for x in data]
            formatted_timestamp = f"{timestamp:.3f}"
            
            # Write data
            if "BLE_IMU" in sensor_id:
                f.write(f"{formatted_timestamp},{','.join(formatted_data)}\n")
            elif "TCP" in sensor_id:
                f.write(f"{formatted_timestamp},{','.join(formatted_data)}\n")
            elif "Audio" in sensor_id:
                f.write(f"{formatted_timestamp},{','.join(formatted_data)}\n")

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
        "tcp_sensors": {},
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
        elif "TCP" in sensor_id:
            recent_data["tcp_sensors"][sensor_id] = {
                "timestamps": [ts for ts, _ in recent_sensor_data],
                "data": [d for _, d in recent_sensor_data]
            }
        elif "Audio" in sensor_id:
            recent_data["audio_sensors"][sensor_id] = {
                "timestamps": [ts for ts, _ in recent_sensor_data],
                "data": [d for _, d in recent_sensor_data]
            }
    
    return recent_data

def run_ml_prediction(participant_data):
    """Run machine learning prediction on sensor data"""
    # Get the last 5 seconds of sensor data
    recent_data = get_recent_sensor_data(seconds=5)
    
    # Count samples for each sensor type
    imu_sample_count = 0
    tcp_sample_count = 0
    audio_sample_count = 0
    
    # Count IMU sensor samples
    for sensor_id, sensor_data in recent_data["imu_sensors"].items():
        imu_sample_count += len(sensor_data["data"])
    
    # Count TCP sensor samples
    for sensor_id, sensor_data in recent_data["tcp_sensors"].items():
        tcp_sample_count += len(sensor_data["data"])
    
    # Count Audio sensor samples
    for sensor_id, sensor_data in recent_data["audio_sensors"].items():
        audio_sample_count += len(sensor_data["data"])
    
    # Store sample counts in a dictionary
    sample_counts = {
        "imu_samples": imu_sample_count,
        "tcp_samples": tcp_sample_count,
        "audio_samples": audio_sample_count,
        "total_samples": imu_sample_count + tcp_sample_count + audio_sample_count
    }

    print(sample_counts)
    
    # For now, just generate a random prediction as placeholder
    threshold = float(participant_data.get("ml_threshold", 0.5))
    prediction = random.random()
    
    # Return both the prediction and sample counts
    return prediction, sample_counts

# BLE connection utilities from test.py
async def connect_ble_sensors():
    """Connect to real BLE sensors using code from test.py"""
    global ble_devices, freq
    
    # Scan for BLE devices
    devices = await BleakScanner.discover()
    ble_nodes = [d for d in devices if d.name and "Node" in d.name]
    
    if not ble_nodes:
        return []
    
    # Create device objects for each BLE node
    ble_devices = []
    for i, device in enumerate(ble_nodes):
        ble_devices.append(BLEDevice(device.address, device.name, i))
    
    # Connect to each device
    connected_devices = []
    for device in ble_devices:
        success = await device.connect()
        if success:
            connected_devices.append(device)
    
    return connected_devices

class BLEDevice:
    """Class for handling BLE device connections (from test.py)"""
    def __init__(self, address, name, idx):
        self.address = address
        self.name = name
        self.idx = idx
        self.client = None
        self.data_buffer = []
        self.raw_buffer = []
    
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
        
        self.client = BleakClient(self.address)
        try:
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
            return True
        
        except Exception as e:
            print(f"Failed to connect to {self.name}: {str(e)}")
            return False
    
    async def disconnect(self):
        """Disconnect from this BLE device"""
        if self.client and self.client.is_connected:
            await self.client.disconnect()
            print(f"{self.name} disconnected.")
    
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
        
        num_samples = 12
        out = np.zeros((num_samples, 10))
        
        # Extract time data - make sure indexing matches MATLAB
        # Using 7 bytes for timestamp as specified
        t = self.clock_out(raw_data[2:9])
        print(f"Processing data at time {t}")
        print(f"Length of raw data: {len(raw_data)}")
        
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
            print(f"Timestamp for sample {i}: {out[i, 0]:.3f}")
            
            # Extract quaternions - using 2 bytes for each value
            qw = self.bytes_to_double(raw_data[10 + i*2 : 12 + i*2])
            qx = self.bytes_to_double(raw_data[52 + i*2 : 54 + i*2])
            qy = self.bytes_to_double(raw_data[94 + i*2 : 96 + i*2])
            qz = self.bytes_to_double(raw_data[136 + i*2 : 138 + i*2])
            
            # Normalize quaternion
            # norm = math.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
            # if norm > 0:  # Avoid division by zero
            #     qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
            
            out[i, 3:7] = [qw, qx, qy, qz]
            
            # Calculate Euler angles from normalized quaternion
            rot = Rotation.from_quat([qx, qy, qz, qw])
            eul = rot.as_euler('xyz', degrees=False)
            out[i, 7:10] = eul
            
            if i == 0:  # Print first sample for debugging
                print(f"Sample {i} - Quaternion: [{qw:.3f}, {qx:.3f}, {qy:.3f}, {qz:.3f}], Euler: [{eul[0]:.3f}, {eul[1]:.3f}, {eul[2]:.3f}]")
        
        return out