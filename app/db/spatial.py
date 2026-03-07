import sqlite3
import os
from contextlib import contextmanager
from app.core import config

def get_db_connection():
    """Create a thread-local database connection with SpatiaLite enabled"""
    conn = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    # Enable extension loading
    conn.enable_load_extension(True)
    
    # Try different extension paths depending on the OS / sqlite build
    try:
        conn.execute("SELECT load_extension('mod_spatialite')")
    except sqlite3.OperationalError:
        try:
            # Maybe linux format
            conn.execute("SELECT load_extension('libspatialite.so')")
        except sqlite3.OperationalError:
            try:
                # Maybe macos
                conn.execute("SELECT load_extension('mod_spatialite.dylib')")
            except sqlite3.OperationalError:
                try:
                    # Windows alternative
                    conn.execute("SELECT load_extension('mod_spatialite.dll')")
                except sqlite3.OperationalError:
                    print("WARNING: Could not load SpatiaLite extension. Spatial indexing will not work.")
    
    conn.row_factory = sqlite3.Row
    return conn

@contextmanager
def db_session():
    """Context manager for DB connections"""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()

def init_db():
    """Initialize SpatiaLite database and tables if they don't exist"""
    print(f"Initializing SpatiaLite database at {config.DB_PATH} ...")
    
    with db_session() as conn:
        cursor = conn.cursor()
        
        # Iniliaize SpatiaLite internal metadata tables
        cursor.execute("SELECT InitSpatialMetaData(1);")
        
        # Create our detection log table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS spatial_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp REAL NOT NULL
            )
        ''')
        
        # Add the geometry column (POINT)
        # Check if geometry column already exists
        cursor.execute("PRAGMA table_info(spatial_log)")
        columns = [col['name'] for col in cursor.fetchall()]
        if 'geom' not in columns:
            cursor.execute("SELECT AddGeometryColumn('spatial_log', 'geom', 4326, 'POINT', 'XY')")
            
            # Create a spatial index
            cursor.execute("SELECT CreateSpatialIndex('spatial_log', 'geom')")
            
    print("Database initialization successful.")

def insert_detection(species, confidence, lat, lon, timestamp):
    """Insert a new detection with spatial coordinates into the database"""
    with db_session() as conn:
        cursor = conn.cursor()
        
        # RULE 3: ST_GeomFromText('POINT(lon lat)', 4326)
        wkt_point = f"POINT({lon} {lat})"
        
        query = '''
            INSERT INTO spatial_log (species, confidence, timestamp, geom)
            VALUES (?, ?, ?, ST_GeomFromText(?, 4326))
        '''
        cursor.execute(query, (species, confidence, timestamp, wkt_point))

def query_detections(limit=100):
    """Son N tespiti döndür. Koordinatları okurken ST_X (lon) ve ST_Y (lat) kullanır."""
    with db_session() as conn:
        cursor = conn.cursor()
        query = '''
            SELECT id, species, confidence, timestamp, ST_Y(geom) as latitude, ST_X(geom) as longitude
            FROM spatial_log
            ORDER BY timestamp DESC
            LIMIT ?
        '''
        cursor.execute(query, (limit,))
        return [dict(row) for row in cursor.fetchall()]

def query_detections_bbox(min_lat, min_lon, max_lat, max_lon):
    """Bölgesel sorgu - Bounding box"""
    with db_session() as conn:
        cursor = conn.cursor()
        # SpatiaLite BuildMbr / MbrWithin veya basit lat/lon
        query = '''
            SELECT id, species, confidence, timestamp, ST_Y(geom) as latitude, ST_X(geom) as longitude
            FROM spatial_log
            WHERE ST_Y(geom) BETWEEN ? AND ?
              AND ST_X(geom) BETWEEN ? AND ?
            ORDER BY timestamp DESC
        '''
        cursor.execute(query, (min_lat, max_lat, min_lon, max_lon))
        return [dict(row) for row in cursor.fetchall()]
