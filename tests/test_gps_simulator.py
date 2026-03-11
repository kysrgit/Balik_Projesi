import pytest
from scripts.gps_simulator import generate_gga_sentence

def test_generate_gga_sentence_basic():
    # Test with standard positive coordinates
    lat = 36.8848
    lon = 30.7040
    time_str = "123456.00"

    sentence = generate_gga_sentence(lat, lon, time_str)

    assert sentence.startswith("$GPGGA,")
    assert sentence.endswith("\r\n")

    parts = sentence.strip().split(',')

    assert parts[1] == time_str

    # 36.8848 deg
    # 36 deg + 0.8848 * 60 min
    # 0.8848 * 60 = 53.088
    # 3653.0880
    assert parts[2] == "3653.0880"
    assert parts[3] == "N"

    # 30.7040 deg
    # 30 deg + 0.7040 * 60 min
    # 0.7040 * 60 = 42.24
    # 03042.2400
    assert parts[4] == "03042.2400"
    assert parts[5] == "E"

    # Rest of the string
    assert parts[6] == "1"
    assert parts[7] == "08"

def test_generate_gga_sentence_negative_coords():
    # Test with negative coordinates (South, West)
    lat = -36.8848
    lon = -30.7040
    time_str = "123456.00"

    sentence = generate_gga_sentence(lat, lon, time_str)
    parts = sentence.strip().split(',')

    assert parts[3] == "S"
    assert parts[5] == "W"

def test_generate_gga_sentence_zero_coords():
    # Test with equator / prime meridian
    lat = 0.0
    lon = 0.0
    time_str = "000000.00"

    sentence = generate_gga_sentence(lat, lon, time_str)
    parts = sentence.strip().split(',')

    assert parts[2] == "0000.0000"
    assert parts[3] == "N"  # Code gives 'N' for lat >= 0
    assert parts[4] == "00000.0000"
    assert parts[5] == "E"  # Code gives 'E' for lon >= 0

def test_generate_gga_sentence_checksum():
    # Ensure checksum is calculated correctly
    # Checksum is XOR of all bytes between $ and *
    lat = 36.8848
    lon = 30.7040
    time_str = "123456.00"

    sentence = generate_gga_sentence(lat, lon, time_str)

    # Extract core string to re-calculate checksum
    core = sentence.split('$')[1].split('*')[0]
    expected_checksum = 0
    for char in core:
        expected_checksum ^= ord(char)

    actual_checksum = int(sentence.split('*')[1][:2], 16)

    assert actual_checksum == expected_checksum
