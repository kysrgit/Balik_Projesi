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
        self.altitude = None
        self.satellites = None
        self.hdop = None
        self.timestamp = None
        self.is_valid = False

    def update(self, lat, lon, ts, msg=None):
        with self.lock:
            self.latitude = lat
            self.longitude = lon
            self.timestamp = ts
            if msg:
                self.altitude = getattr(msg, 'altitude', None)
                self.satellites = getattr(msg, 'num_sats', None)
                self.hdop = getattr(msg, 'horizontal_dil', None)
            self.is_valid = True

    def get(self):
        with self.lock:
            # Check staleness
            valid = self.is_valid and self.timestamp is not None and (time.time() - self.timestamp <= getattr(config, 'GPS_STALE_TIMEOUT', 10.0))
            return self.latitude, self.longitude, self.timestamp, valid
            
    def get_dict(self):
        with self.lock:
            valid = self.is_valid and self.timestamp is not None and (time.time() - self.timestamp <= getattr(config, 'GPS_STALE_TIMEOUT', 10.0))
            return {
                'latitude': self.latitude,
                'longitude': self.longitude,
                'altitude': self.altitude,
                'satellites': self.satellites,
                'hdop': self.hdop,
                'timestamp': self.timestamp,
                'is_valid': valid
            }

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
                if line.startswith(('$GPGGA', '$GNGGA', '$GPRMC', '$GNRMC')):
                    try:
                        msg = pynmea2.parse(line)
                        if hasattr(msg, 'latitude') and hasattr(msg, 'longitude') and msg.latitude != 0.0:
                            # Parse out the valid latitude and longitude
                            lat = msg.latitude
                            lon = msg.longitude
                            ts = time.time()
                            gps_state.update(lat, lon, ts, msg)
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

