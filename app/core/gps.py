import threading
import time
import serial
import pynmea2
from app.core import config

class GPSState:
    """Thread-safe GPS state container"""
    def __init__(self):
        self.lock = threading.Lock()
        self.latitude = None
        self.longitude = None
        self.timestamp = None
        self.is_valid = False

    def update(self, lat, lon, ts):
        with self.lock:
            self.latitude = lat
            self.longitude = lon
            self.timestamp = ts
            self.is_valid = True

    def get(self):
        with self.lock:
            return self.latitude, self.longitude, self.timestamp, self.is_valid

# Global GPS State object
gps_state = GPSState()

def gps_reader_thread():
    """Asynchronous daemon thread to read GPS data without blocking YOLO."""
    while True:
        try:
            # Setup serial connection
            # Timeout is important to avoid blocking forever
            ser = serial.Serial(config.GPS_PORT, baudrate=config.GPS_BAUDRATE, timeout=1.0)
            print(f"GPS connected on {config.GPS_PORT}")
            
            while True:
                line = ser.readline().decode('ascii', errors='replace').strip()
                if line.startswith('$GPGGA') or line.startswith('$GPRMC'):
                    try:
                        msg = pynmea2.parse(line)
                        if hasattr(msg, 'latitude') and hasattr(msg, 'longitude') and msg.latitude != 0.0:
                            # Parse out the valid latitude and longitude
                            lat = msg.latitude
                            lon = msg.longitude
                            ts = time.time()
                            gps_state.update(lat, lon, ts)
                    except pynmea2.ParseError:
                        pass
                
        except serial.SerialException as e:
            print(f"GPS Serial Error: {e}, retrying in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"GPS Unexpected Error: {e}")
            time.sleep(5)
        finally:
            try:
                if 'ser' in locals() and ser.is_open:
                    ser.close()
            except:
                pass
