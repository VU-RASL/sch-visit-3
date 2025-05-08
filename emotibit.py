import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
from typing import List, Any

LISTEN_IP_ADDRESS = "127.0.0.1"
LISTEN_PORT = 12345  # The port specified in your XML's <output><port>

def ppg_red_handler(address: str, *args: List[Any]):
    """Handles messages for PPG Red channel."""
    print(f"RECEIVED [PPG:RED] - Address: {address}, Data: {args}")

def ppg_ir_handler(address: str, *args: List[Any]):
    """Handles messages for PPG Infrared channel."""
    print(f"RECEIVED [PPG:IR] - Address: {address}, Data: {args}")

def ppg_grn_handler(address: str, *args: List[Any]):
    """Handles messages for PPG Green channel."""
    print(f"RECEIVED [PPG:GRN] - Address: {address}, Data: {args}")

# Electrodermal Activity Handlers
def eda_handler(address: str, *args: List[Any]):
    """Handles messages for Electrodermal Activity (EDA)."""
    print(f"RECEIVED [EDA] - Address: {address}, Data: {args}")

def edl_handler(address: str, *args: List[Any]):
    """Handles messages for Electrodermal Level (EDL)."""
    print(f"RECEIVED [EDL] - Address: {address}, Data: {args}")

def edr_handler(address: str, *args: List[Any]):
    """Handles messages for Electrodermal Response (EDR)."""
    # Note: EmotiBit V4+ might combine EDR into EA. This handler is for explicit EDR signals.
    print(f"RECEIVED [EDR] - Address: {address}, Data: {args}")

# Temperature Handlers
def temp_t0_handler(address: str, *args: List[Any]):
    """Handles messages for Temperature (T0 - typically older EmotiBit versions)."""
    # Note: Original TEMP, often associated with T0 TypeTag
    print(f"RECEIVED [TEMP_T0] - Address: {address}, Data: {args}")

def temp_t1_handler(address: str, *args: List[Any]):
    """Handles messages for Temperature (T1)."""
    print(f"RECEIVED [TEMP_T1] - Address: {address}, Data: {args}")

def therm_handler(address: str, *args: List[Any]):
    """Handles messages for Temperature via Medical-grade Thermopile (TH)."""
    print(f"RECEIVED [THERM_MD] - Address: {address}, Data: {args}")

# Accelerometer Handlers
def acc_x_handler(address: str, *args: List[Any]):
    """Handles messages for Accelerometer X-axis."""
    print(f"RECEIVED [ACC:X] - Address: {address}, Data: {args}")

def acc_y_handler(address: str, *args: List[Any]):
    """Handles messages for Accelerometer Y-axis."""
    print(f"RECEIVED [ACC:Y] - Address: {address}, Data: {args}")

def acc_z_handler(address: str, *args: List[Any]):
    """Handles messages for Accelerometer Z-axis."""
    print(f"RECEIVED [ACC:Z] - Address: {address}, Data: {args}")

# Gyroscope Handlers
def gyro_x_handler(address: str, *args: List[Any]):
    """Handles messages for Gyroscope X-axis."""
    print(f"RECEIVED [GYRO:X] - Address: {address}, Data: {args}")

def gyro_y_handler(address: str, *args: List[Any]):
    """Handles messages for Gyroscope Y-axis."""
    print(f"RECEIVED [GYRO:Y] - Address: {address}, Data: {args}")

def gyro_z_handler(address: str, *args: List[Any]):
    """Handles messages for Gyroscope Z-axis."""
    print(f"RECEIVED [GYRO:Z] - Address: {address}, Data: {args}")

# Magnetometer Handlers
def mag_x_handler(address: str, *args: List[Any]):
    """Handles messages for Magnetometer X-axis."""
    print(f"RECEIVED [MAG:X] - Address: {address}, Data: {args}")

def mag_y_handler(address: str, *args: List[Any]):
    """Handles messages for Magnetometer Y-axis."""
    print(f"RECEIVED [MAG:Y] - Address: {address}, Data: {args}")

def mag_z_handler(address: str, *args: List[Any]):
    """Handles messages for Magnetometer Z-axis."""
    print(f"RECEIVED [MAG:Z] - Address: {address}, Data: {args}")

# Skin Conductance Response (SCR) Handlers
def scr_amp_handler(address: str, *args: List[Any]):
    """Handles messages for Skin Conductance Response (SCR) Amplitude (SA)."""
    print(f"RECEIVED [SCR:AMP] - Address: {address}, Data: {args}")

def scr_rise_handler(address: str, *args: List[Any]):
    """Handles messages for Skin Conductance Response (SCR) Rise Time (SR)."""
    print(f"RECEIVED [SCR:RISE] - Address: {address}, Data: {args}")

def scr_freq_handler(address: str, *args: List[Any]):
    """Handles messages for Skin Conductance Response (SCR) Frequency (SF)."""
    print(f"RECEIVED [SCR:FREQ] - Address: {address}, Data: {args}")

# Heart Rate Metric Handlers
def hr_handler(address: str, *args: List[Any]):
    """Handles messages for Heart Rate (HR)."""
    print(f"RECEIVED [HR] - Address: {address}, Data: {args}")

def ibi_handler(address: str, *args: List[Any]):
    """Handles messages for Inter-beat Interval (IBI)."""
    print(f"RECEIVED [IBI] - Address: {address}, Data: {args}")

# Humidity Handler
def humidity_handler(address: str, *args: List[Any]):
    """Handles messages for Humidity (H0 - typically older EmotiBit versions)."""
    print(f"RECEIVED [HUMIDITY] - Address: {address}, Data: {args}")

# Default Handler for Unmapped Messages
def default_handler(address: str, *args: List[Any]):
    """Handles any messages that don't match a specific mapped address."""
    print(f"RECEIVED [UNMAPPED] - Address: {address}, Data: {args}")


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
    print("Press Ctrl+C to stop.")

    try:
        server.serve_forever()  # This will block and keep the server running
    except KeyboardInterrupt:
        print("\nStopping OSC server...")
    finally:
        server.server_close()
        print("OSC Server closed.")
