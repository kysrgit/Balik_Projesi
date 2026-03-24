import sqlite3
from contextlib import contextmanager

from app.core import config


SPATIALITE_EXTENSIONS = [
    "mod_spatialite",
    "libspatialite.so",
    "mod_spatialite.dylib",
    "mod_spatialite.dll",
]


def _load_spatialite(conn: sqlite3.Connection) -> bool:
    """Try to load the SpatiaLite extension for the current connection."""
    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.OperationalError):
        return False

    for extension in SPATIALITE_EXTENSIONS:
        try:
            conn.execute(f"SELECT load_extension('{extension}')")
            return True
        except sqlite3.OperationalError:
            continue

    return False


def supports_spatialite(conn: sqlite3.Connection) -> bool:
    try:
        conn.execute("SELECT spatialite_version()").fetchone()
        return True
    except sqlite3.OperationalError:
        return False


def table_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    cursor = conn.execute(f"PRAGMA table_info({table_name})")
    return {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}


def _has_spatial_metadata(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='spatial_ref_sys'"
    ).fetchone()
    return row is not None


def _geom_query_fragment(conn: sqlite3.Connection) -> tuple[str, bool]:
    columns = table_columns(conn, "spatial_log")
    if {"latitude", "longitude"}.issubset(columns):
        return "latitude, longitude", False

    if "geom" in columns and supports_spatialite(conn):
        return "ST_Y(geom) as latitude, ST_X(geom) as longitude", True

    return "NULL as latitude, NULL as longitude", False


def get_db_connection(db_path=None):
    """Create a database connection and try to enable SpatiaLite if available."""
    conn = sqlite3.connect(db_path or config.DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row

    if not _load_spatialite(conn):
        print("WARNING: Could not load SpatiaLite extension. Falling back to plain SQLite coordinates.")

    return conn


@contextmanager
def db_session(db_path=None):
    """Context manager for DB connections."""
    conn = get_db_connection(db_path=db_path)
    try:
        yield conn
    finally:
        conn.commit()
        conn.close()


def init_db(db_path=None):
    """Initialize the database for both SpatiaLite and plain SQLite modes."""
    target_path = db_path or config.DB_PATH
    print(f"Initializing database at {target_path} ...")

    with db_session(db_path=target_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS spatial_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                species TEXT NOT NULL,
                confidence REAL NOT NULL,
                timestamp REAL NOT NULL,
                latitude REAL,
                longitude REAL
            )
            """
        )

        columns = table_columns(conn, "spatial_log")
        if "latitude" not in columns:
            cursor.execute("ALTER TABLE spatial_log ADD COLUMN latitude REAL")
        if "longitude" not in columns:
            cursor.execute("ALTER TABLE spatial_log ADD COLUMN longitude REAL")

        if supports_spatialite(conn):
            if not _has_spatial_metadata(conn):
                cursor.execute("SELECT InitSpatialMetaData(1);")

            columns = table_columns(conn, "spatial_log")
            if "geom" not in columns:
                cursor.execute("SELECT AddGeometryColumn('spatial_log', 'geom', 4326, 'POINT', 'XY')")
                cursor.execute("SELECT CreateSpatialIndex('spatial_log', 'geom')")

    print("Database initialization successful.")


def insert_detection(species, confidence, lat, lon, timestamp, db_path=None):
    """Insert a new detection into the database with or without SpatiaLite."""
    with db_session(db_path=db_path) as conn:
        cursor = conn.cursor()
        columns = table_columns(conn, "spatial_log")
        use_geom = "geom" in columns and supports_spatialite(conn)

        if use_geom:
            wkt_point = f"POINT({lon} {lat})"
            cursor.execute(
                """
                INSERT INTO spatial_log (species, confidence, timestamp, latitude, longitude, geom)
                VALUES (?, ?, ?, ?, ?, ST_GeomFromText(?, 4326))
                """,
                (species, confidence, timestamp, lat, lon, wkt_point),
            )
        else:
            cursor.execute(
                """
                INSERT INTO spatial_log (species, confidence, timestamp, latitude, longitude)
                VALUES (?, ?, ?, ?, ?)
                """,
                (species, confidence, timestamp, lat, lon),
            )


def query_detections(limit=100, db_path=None):
    """Return the latest detections regardless of DB backend mode."""
    with db_session(db_path=db_path) as conn:
        cursor = conn.cursor()
        coords_fragment, _ = _geom_query_fragment(conn)
        cursor.execute(
            f"""
            SELECT id, species, confidence, timestamp, {coords_fragment}
            FROM spatial_log
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [dict(row) for row in cursor.fetchall()]


def query_detections_bbox(min_lat, min_lon, max_lat, max_lon, db_path=None):
    """Bounding-box query that works in both SQLite fallback and SpatiaLite mode."""
    with db_session(db_path=db_path) as conn:
        cursor = conn.cursor()
        coords_fragment, using_geom = _geom_query_fragment(conn)

        if using_geom:
            where_clause = "ST_Y(geom) BETWEEN ? AND ? AND ST_X(geom) BETWEEN ? AND ?"
        else:
            where_clause = "latitude BETWEEN ? AND ? AND longitude BETWEEN ? AND ?"

        cursor.execute(
            f"""
            SELECT id, species, confidence, timestamp, {coords_fragment}
            FROM spatial_log
            WHERE {where_clause}
            ORDER BY timestamp DESC
            """,
            (min_lat, max_lat, min_lon, max_lon),
        )
        return [dict(row) for row in cursor.fetchall()]
