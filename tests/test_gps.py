import time
import pytest
from app.core import config
from app.core.gps import GPSState

def test_gps_state_initialization():
    state = GPSState()
    lat, lon, ts, valid = state.get()
    assert lat is None
    assert lon is None
    assert not valid

def test_gps_state_update_and_validity():
    state = GPSState()
    now = time.time()
    
    # Mocking a message object
    class MockMsg:
        altitude = 15.5
        num_sats = 8
        horizontal_dil = 1.2
    
    state.update(36.5, 30.0, now, MockMsg())
    
    lat, lon, ts, valid = state.get()
    assert lat == 36.5
    assert lon == 30.0
    assert ts == now
    assert valid is True
    
    d = state.get_dict()
    assert d['altitude'] == 15.5
    assert d['satellites'] == 8
    assert d['hdop'] == 1.2

def test_gps_staleness():
    state = GPSState()
    old_time = time.time() - 20.0 # 20 seconds ago, assuming config.GPS_STALE_TIMEOUT is 10.0
    
    state.update(36.5, 30.0, old_time)
    
    # It should be marked invalid due to staleness
    lat, lon, ts, valid = state.get()
    assert lat == 36.5
    assert valid is False
