import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
from typing import List, Any, Dict, Deque
from collections import deque
import time
import numpy as np

LISTEN_IP_ADDRESS = "127.0.0.1"
LISTEN_PORT = 12345  # The port specified in your XML's <output><port>

# Define sampling rates and buffer sizes
SAMPLING_RATES = {
    "motion": 25,  # AX, AY, AZ, GX, GY, GZ, MX, MY, MZ at 25Hz
    "ppg": 25,     # PI, PG, PR at 25Hz
    "temperature": 7.5,  # T0, TH at 7.5Hz
    "eda": 15      # EA, EL, ER at 15Hz
}

# Calculate buffer sizes based on sampling rates (60-second buffer)
BUFFER_SIZES = {signal: int(rate * 60) for signal, rate in SAMPLING_RATES.items()}

# Create data buffers for each signal type
data_buffers = {
    # Motion signals
    "ACC:X": deque(maxlen=BUFFER_SIZES["motion"]),
    "ACC:Y": deque(maxlen=BUFFER_SIZES["motion"]),
    "ACC:Z": deque(maxlen=BUFFER_SIZES["motion"]),
    "GYRO:X": deque(maxlen=BUFFER_SIZES["motion"]),
    "GYRO:Y": deque(maxlen=BUFFER_SIZES["motion"]),
    "GYRO:Z": deque(maxlen=BUFFER_SIZES["motion"]),
    "MAG:X": deque(maxlen=BUFFER_SIZES["motion"]),
    "MAG:Y": deque(maxlen=BUFFER_SIZES["motion"]),
    "MAG:Z": deque(maxlen=BUFFER_SIZES["motion"]),
    
    # PPG signals
    "PPG:RED": deque(maxlen=BUFFER_SIZES["ppg"]),
    "PPG:IR": deque(maxlen=BUFFER_SIZES["ppg"]),
    "PPG:GRN": deque(maxlen=BUFFER_SIZES["ppg"]),
    
    # Temperature signals
    "TEMP": deque(maxlen=BUFFER_SIZES["temperature"]),
    "TEMP:T1": deque(maxlen=BUFFER_SIZES["temperature"]),
    "THERM": deque(maxlen=BUFFER_SIZES["temperature"]),
    
    # EDA signals
    "EDA": deque(maxlen=BUFFER_SIZES["eda"]),
    "EDL": deque(maxlen=BUFFER_SIZES["eda"]),
    "EDR": deque(maxlen=BUFFER_SIZES["eda"]),
    
    # Derived metrics (using EDA sampling rate)
    "SCR:AMP": deque(maxlen=BUFFER_SIZES["eda"]),
    "SCR:RISE": deque(maxlen=BUFFER_SIZES["eda"]),
    "SCR:FREQ": deque(maxlen=BUFFER_SIZES["eda"]),
    
    # Heart metrics (using PPG sampling rate)
    "HR": deque(maxlen=BUFFER_SIZES["ppg"]),
    "IBI": deque(maxlen=BUFFER_SIZES["ppg"]),
    
    # Humidity (using temperature sampling rate)
    "HUMIDITY": deque(maxlen=BUFFER_SIZES["temperature"]),
}

def store_data(signal_type: str, timestamp: float, value: float):
    """Store data in the appropriate buffer with timestamp."""
    if signal_type in data_buffers:
        data_buffers[signal_type].append((timestamp, value))

def ppg_red_handler(address: str, *args: List[Any]):
    """Handles messages for PPG Red channel."""
    # print(f"RECEIVED [PPG:RED] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("PPG:RED", timestamp, value)

def ppg_ir_handler(address: str, *args: List[Any]):
    """Handles messages for PPG Infrared channel."""
    # print(f"RECEIVED [PPG:IR] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("PPG:IR", timestamp, value)

def ppg_grn_handler(address: str, *args: List[Any]):
    """Handles messages for PPG Green channel."""
    # print(f"RECEIVED [PPG:GRN] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("PPG:GRN", timestamp, value)

# Electrodermal Activity Handlers
def eda_handler(address: str, *args: List[Any]):
    """Handles messages for Electrodermal Activity (EDA)."""
    # print(f"RECEIVED [EDA] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("EDA", timestamp, value)

def edl_handler(address: str, *args: List[Any]):
    """Handles messages for Electrodermal Level (EDL)."""
    # print(f"RECEIVED [EDL] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("EDL", timestamp, value)

def edr_handler(address: str, *args: List[Any]):
    """Handles messages for Electrodermal Response (EDR)."""
    # print(f"RECEIVED [EDR] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("EDR", timestamp, value)

# Temperature Handlers
def temp_t0_handler(address: str, *args: List[Any]):
    """Handles messages for Temperature (T0 - typically older EmotiBit versions)."""
    # print(f"RECEIVED [TEMP_T0] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("TEMP", timestamp, value)

def temp_t1_handler(address: str, *args: List[Any]):
    """Handles messages for Temperature (T1)."""
    # print(f"RECEIVED [TEMP_T1] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("TEMP:T1", timestamp, value)

def therm_handler(address: str, *args: List[Any]):
    """Handles messages for Temperature via Medical-grade Thermopile (TH)."""
    # print(f"RECEIVED [THERM_MD] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("THERM", timestamp, value)

# Accelerometer Handlers
def acc_x_handler(address: str, *args: List[Any]):
    """Handles messages for Accelerometer X-axis."""
    # print(f"RECEIVED [ACC:X] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("ACC:X", timestamp, value)

def acc_y_handler(address: str, *args: List[Any]):
    """Handles messages for Accelerometer Y-axis."""
    # print(f"RECEIVED [ACC:Y] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("ACC:Y", timestamp, value)

def acc_z_handler(address: str, *args: List[Any]):
    """Handles messages for Accelerometer Z-axis."""
    # print(f"RECEIVED [ACC:Z] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("ACC:Z", timestamp, value)

# Gyroscope Handlers
def gyro_x_handler(address: str, *args: List[Any]):
    """Handles messages for Gyroscope X-axis."""
    # print(f"RECEIVED [GYRO:X] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("GYRO:X", timestamp, value)

def gyro_y_handler(address: str, *args: List[Any]):
    """Handles messages for Gyroscope Y-axis."""
    # print(f"RECEIVED [GYRO:Y] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("GYRO:Y", timestamp, value)

def gyro_z_handler(address: str, *args: List[Any]):
    """Handles messages for Gyroscope Z-axis."""
    # print(f"RECEIVED [GYRO:Z] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("GYRO:Z", timestamp, value)

# Magnetometer Handlers
def mag_x_handler(address: str, *args: List[Any]):
    """Handles messages for Magnetometer X-axis."""
    # print(f"RECEIVED [MAG:X] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("MAG:X", timestamp, value)

def mag_y_handler(address: str, *args: List[Any]):
    """Handles messages for Magnetometer Y-axis."""
    # print(f"RECEIVED [MAG:Y] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("MAG:Y", timestamp, value)

def mag_z_handler(address: str, *args: List[Any]):
    """Handles messages for Magnetometer Z-axis."""
    # print(f"RECEIVED [MAG:Z] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("MAG:Z", timestamp, value)

# Skin Conductance Response (SCR) Handlers
def scr_amp_handler(address: str, *args: List[Any]):
    """Handles messages for Skin Conductance Response (SCR) Amplitude (SA)."""
    # print(f"RECEIVED [SCR:AMP] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("SCR:AMP", timestamp, value)

def scr_rise_handler(address: str, *args: List[Any]):
    """Handles messages for Skin Conductance Response (SCR) Rise Time (SR)."""
    # print(f"RECEIVED [SCR:RISE] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("SCR:RISE", timestamp, value)

def scr_freq_handler(address: str, *args: List[Any]):
    """Handles messages for Skin Conductance Response (SCR) Frequency (SF)."""
    # print(f"RECEIVED [SCR:FREQ] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("SCR:FREQ", timestamp, value)

# Heart Rate Metric Handlers
def hr_handler(address: str, *args: List[Any]):
    """Handles messages for Heart Rate (HR)."""
    # print(f"RECEIVED [HR] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("HR", timestamp, value)

def ibi_handler(address: str, *args: List[Any]):
    """Handles messages for Inter-beat Interval (IBI)."""
    # print(f"RECEIVED [IBI] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("IBI", timestamp, value)

# Humidity Handler
def humidity_handler(address: str, *args: List[Any]):
    """Handles messages for Humidity (H0 - typically older EmotiBit versions)."""
    # print(f"RECEIVED [HUMIDITY] - Address: {address}, Data: {args}")
    if args:
        timestamp = time.time()
        for value in args:
            if isinstance(value, (int, float)):
                store_data("HUMIDITY", timestamp, value)

# Default Handler for Unmapped Messages
def default_handler(address: str, *args: List[Any]):
    """Handles any messages that don't match a specific mapped address."""
    # print(f"RECEIVED [UNMAPPED] - Address: {address}, Data: {args}")

def get_buffer_stats():
    """Return statistics for all buffers."""
    stats = {}
    for signal, buffer in data_buffers.items():
        if buffer:
            values = [value for _, value in buffer]
            stats[signal] = {
                "count": len(buffer),
                "capacity": buffer.maxlen,
                "min": min(values) if values else None,
                "max": max(values) if values else None,
                "mean": np.mean(values) if values else None,
                "latest": values[-1] if values else None,
                "buffer_full_pct": (len(buffer) / buffer.maxlen) * 100 if buffer.maxlen else 0
            }
        else:
            stats[signal] = {
                "count": 0,
                "capacity": buffer.maxlen,
                "min": None,
                "max": None,
                "mean": None,
                "latest": None,
                "buffer_full_pct": 0
            }
    return stats

if __name__ == "__main__":
    # --- Initialize OSC Dispatcher and Server ---
    osc_dispatcher = dispatcher.Dispatcher()

    # Map OSC addresses from your XML to the handler functions
    # PPG
    osc_dispatcher.map("/EmotiBit/0/PPG:RED", ppg_red_handler)
    osc_dispatcher.map("/EmotiBit/0/PPG:IR", ppg_ir_handler)
    osc_dispatcher.map("/EmotiBit/0/PPG:GRN", ppg_grn_handler)

    # EDA Family
    osc_dispatcher.map("/EmotiBit/0/EDA", eda_handler)
    osc_dispatcher.map("/EmotiBit/0/EDL", edl_handler)
    osc_dispatcher.map("/EmotiBit/0/EDR", edr_handler)

    # Temperature
    osc_dispatcher.map("/EmotiBit/0/TEMP", temp_t0_handler) # Corresponds to T0 input in XML
    osc_dispatcher.map("/EmotiBit/0/TEMP:T1", temp_t1_handler) # Corresponds to T1 input in XML
    osc_dispatcher.map("/EmotiBit/0/THERM", therm_handler) # Corresponds to TH input in XML

    # IMU: Accelerometer
    osc_dispatcher.map("/EmotiBit/0/ACC:X", acc_x_handler)
    osc_dispatcher.map("/EmotiBit/0/ACC:Y", acc_y_handler)
    osc_dispatcher.map("/EmotiBit/0/ACC:Z", acc_z_handler)

    # IMU: Gyroscope
    osc_dispatcher.map("/EmotiBit/0/GYRO:X", gyro_x_handler)
    osc_dispatcher.map("/EmotiBit/0/GYRO:Y", gyro_y_handler)
    osc_dispatcher.map("/EmotiBit/0/GYRO:Z", gyro_z_handler)

    # IMU: Magnetometer
    osc_dispatcher.map("/EmotiBit/0/MAG:X", mag_x_handler)
    osc_dispatcher.map("/EmotiBit/0/MAG:Y", mag_y_handler)
    osc_dispatcher.map("/EmotiBit/0/MAG:Z", mag_z_handler)

    # SCR Metrics
    osc_dispatcher.map("/EmotiBit/0/SCR:AMP", scr_amp_handler)
    osc_dispatcher.map("/EmotiBit/0/SCR:RISE", scr_rise_handler)
    osc_dispatcher.map("/EmotiBit/0/SCR:FREQ", scr_freq_handler)

    # Heart Rate Metrics
    osc_dispatcher.map("/EmotiBit/0/HR", hr_handler)
    osc_dispatcher.map("/EmotiBit/0/IBI", ibi_handler)

    # Humidity
    osc_dispatcher.map("/EmotiBit/0/HUMIDITY", humidity_handler) # Corresponds to H0 input in XML

    # Optional: Set a default handler for messages to addresses not explicitly mapped
    osc_dispatcher.set_default_handler(default_handler)

    server = osc_server.ThreadingOSCUDPServer(
        (LISTEN_IP_ADDRESS, LISTEN_PORT), osc_dispatcher)

    print(f"Python OSC Server listening on {server.server_address[0]}:{server.server_address[1]}")
    print("Ready to receive EmotiBit OSC messages...")
    print("Buffer sizes configured for 60-second history:")
    for signal_group, rate in SAMPLING_RATES.items():
        print(f"  - {signal_group.upper()}: {rate} Hz → {BUFFER_SIZES[signal_group]} samples")
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()  # This will block and keep the server running
    except KeyboardInterrupt:
        print("\nStopping OSC server...")
        # Print buffer statistics on exit
        stats = get_buffer_stats()
        print("\nBuffer Statistics:")
        for signal, stat in stats.items():
            if stat["count"] > 0:
                print(f"  {signal}: {stat['count']}/{stat['capacity']} samples ({stat['buffer_full_pct']:.1f}% full)")
    finally:
        server.server_close()
        print("OSC Server closed.")
