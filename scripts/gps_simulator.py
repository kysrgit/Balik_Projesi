#!/usr/bin/env python3
"""
Balon Baligi Tespit Sistemi
Laboratuvar Testi icin GPS (NMEA GPGGA) Simulatoru

Kullanim:
Linux/Mac: Sanal seri port (pty) uzerinden calisir. Ciktidaki portu config.py'a yazin.
Windows: Soket (TCP) uzerinden yayin yapar (Uyumluluk modu). Ana kodda pyserial yerine TCP client gerekebilir.
"""
import os
import time
import math
import platform
import socket
from datetime import datetime

# Baslangic konumu (Ornek: Akdeniz - Antalya aciklari)
START_LAT = 36.8848
START_LON = 30.7040

def generate_gga_sentence(lat, lon, time_str):
    """Verilen koordinat ve zamana gore NMEA GPGGA cumlesi uretir."""
    
    # Enlem ve boylami NMEA DDMM.MMMMM formatina cevir
    lat_deg = int(abs(lat))
    lat_min = (abs(lat) - lat_deg) * 60
    lat_str = f"{lat_deg:02d}{lat_min:07.4f}"
    lat_dir = 'N' if lat >= 0 else 'S'

    lon_deg = int(abs(lon))
    lon_min = (abs(lon) - lon_deg) * 60
    lon_str = f"{lon_deg:03d}{lon_min:07.4f}"
    lon_dir = 'E' if lon >= 0 else 'W'

    # GGA Cumlesi Olustur
    sentence_core = f"GPGGA,{time_str},{lat_str},{lat_dir},{lon_str},{lon_dir},1,08,0.9,5.4,M,46.9,M,,"
    
    # Checksum hesapla
    checksum = 0
    for char in sentence_core:
        checksum ^= ord(char)
    
    return f"${sentence_core}*{checksum:02X}\r\n"

def run_pty_simulator():
    """Linux/Mac icin PTY (Sanal Seri Port) modunda calistir"""
    try:
        import pty
        import tty
    except ImportError:
        print("PTY modulu bu isletim sisteminde desteklenmiyor.")
        return False
        
    master_fd, slave_fd = pty.openpty()
    tty.setraw(slave_fd)
    
    slave_name = os.ttyname(slave_fd)
    print("=" * 40)
    print("🛳️  GPS Simulator (PTY Modu) Baslatildi!")
    print(f"📡 Lutfen config.py icinde GPS_PORT degerini soyle degistirin:")
    print(f"👉 GPS_PORT = '{slave_name}'")
    print("=" * 40)
    
    run_simulation_loop(lambda data: os.write(master_fd, data.encode('ascii')))

def run_tcp_simulator():
    """Windows icin TCP Soket modunda calistir (Com0Com alternatifi)"""
    HOST = '127.0.0.1'
    PORT = 9090
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    
    try:
        server.bind((HOST, PORT))
        server.listen(1)
    except Exception as e:
        print(f"Soket baslatilamadi: {e}")
        return
        
    print("=" * 40)
    print("🛳️  GPS Simulator (TCP Soket Modu) Baslatildi!")
    print(f"📡 Dinleniyor: {HOST}:{PORT}")
    print(f"Tavsiye: Windows'ta serial port emulator (com0com) kullanarak bu TCP portunu COM portuna yonlendirebilirsiniz.")
    print("=" * 40)
    print("Istemci bekleniyor...")
    
    conn, addr = server.accept()
    print(f"Istemci baglandi: {addr}")
    
    try:
        run_simulation_loop(lambda data: conn.sendall(data.encode('ascii')))
    except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
        print("Istemci baglantiyi kesti.")
    finally:
        conn.close()
        server.close()

def run_simulation_loop(write_callback):
    """Ortak dongu: Surekli GPS cumleleri uretip gonderir."""
    current_lat = START_LAT
    current_lon = START_LON
    
    radius = 0.005 # Yaklasik yari cap
    angle = 0.0
    
    print("NMEA Cümleleri basliyor. (Durdurmak icin Ctrl+C)\n")
    try:
        while True:
            time_str = datetime.utcnow().strftime("%H%M%S.00")
            
            angle += 0.05
            sim_lat = current_lat + math.sin(angle) * radius
            sim_lon = current_lon + math.cos(angle) * radius
            
            nmea_sentence = generate_gga_sentence(sim_lat, sim_lon, time_str)
            
            # Alt sisteme yaz (PTY veya TCP Soket)
            try:
                write_callback(nmea_sentence)
            except Exception as e:
                print(f"Yazma hatasi: {e}")
                break
                
            print(f"Yayinlaniyor: {nmea_sentence.strip()}", end='\r')
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n\nGPS Simulatoru Kapatiliyor...")

if __name__ == "__main__":
    if platform.system() == "Windows":
        run_tcp_simulator()
    else:
        run_pty_simulator()
