"""
Veri Export Formatları
- GeoJSON (harita servisleri, QGIS, Leaflet)
- CSV download (araştırmacılar)
- DarwinCore Archive (GBIF / OBIS uluslararası standart)
"""
import csv
import io
import json
import os
import zipfile
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional


# ─────────────────────────────────────────────────────────────────
# Ortak: CSV dosyasından tespitleri oku
# ─────────────────────────────────────────────────────────────────
def _read_detections_csv(csv_path: str) -> List[Dict[str, Any]]:
    """CSV log dosyasından tüm tespitleri sözlük listesi olarak oku"""
    rows = []
    if not os.path.exists(csv_path):
        return rows
    with open(csv_path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _read_detections_db(db_path: str) -> List[Dict[str, Any]]:
    """SpatiaLite veritabanından tespitleri oku (GPS verileriyle)"""
    rows = []
    try:
        import sqlite3
        if not os.path.exists(db_path):
            return rows
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT species, confidence, timestamp, latitude, longitude FROM spatial_log ORDER BY timestamp"
        )
        for row in cursor:
            rows.append(dict(row))
        conn.close()
    except Exception:
        pass
    return rows


# ─────────────────────────────────────────────────────────────────
# 1. GeoJSON Export
# ─────────────────────────────────────────────────────────────────
def to_geojson(csv_path: str, db_path: Optional[str] = None) -> dict:
    """
    Tespit verilerini GeoJSON FeatureCollection olarak döndürür.
    GPS verisi varsa DB'den, yoksa CSV'den (koordinatsız) alır.
    
    Uyumlu: Leaflet, QGIS, MapBox, ArcGIS Online, Google Earth
    """
    features = []

    # Önce DB'den GPS'li kayıtları dene
    if db_path:
        db_rows = _read_detections_db(db_path)
        for row in db_rows:
            lat = row.get('latitude')
            lon = row.get('longitude')
            if lat is not None and lon is not None:
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [float(lon), float(lat)]  # GeoJSON: [lon, lat]
                    },
                    "properties": {
                        "species": row.get('species', 'Lagocephalus sceleratus'),
                        "confidence": float(row.get('confidence', 0)),
                        "timestamp": row.get('timestamp', ''),
                        "source": "antigravity_pufferfish_detector"
                    }
                }
                features.append(feature)

    # GPS verisi yoksa CSV'den al (koordinatsız)
    if not features:
        csv_rows = _read_detections_csv(csv_path)
        for i, row in enumerate(csv_rows):
            feature = {
                "type": "Feature",
                "geometry": None,  # Koordinat yok
                "properties": {
                    "species": "Lagocephalus sceleratus",
                    "confidence": float(row.get('Confidence', 0)),
                    "date": row.get('Date', ''),
                    "time": row.get('Time', ''),
                    "bbox": f"{row.get('BBox_X1','')},{row.get('BBox_Y1','')},{row.get('BBox_X2','')},{row.get('BBox_Y2','')}",
                    "source": "antigravity_pufferfish_detector"
                }
            }
            features.append(feature)

    return {
        "type": "FeatureCollection",
        "name": "pufferfish_detections",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"}
        },
        "features": features
    }


# ─────────────────────────────────────────────────────────────────
# 2. CSV Download
# ─────────────────────────────────────────────────────────────────
def to_csv_download(csv_path: str, db_path: Optional[str] = None) -> str:
    """
    İndirilebilir CSV string'i döndürür.
    DB verisi varsa GPS koordinatlarıyla zenginleştirir.
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # DB'den GPS'li veriler
    db_rows = _read_detections_db(db_path) if db_path else []

    if db_rows:
        writer.writerow(["Species", "Confidence", "Timestamp", "Latitude", "Longitude"])
        for row in db_rows:
            writer.writerow([
                row.get('species', 'Lagocephalus sceleratus'),
                round(float(row.get('confidence', 0)), 4),
                row.get('timestamp', ''),
                row.get('latitude', ''),
                row.get('longitude', '')
            ])
    else:
        # Sadece CSV log
        csv_rows = _read_detections_csv(csv_path)
        writer.writerow(["Timestamp", "Date", "Time", "Confidence",
                         "BBox_X1", "BBox_Y1", "BBox_X2", "BBox_Y2"])
        for row in csv_rows:
            writer.writerow([
                row.get('Timestamp', ''), row.get('Date', ''),
                row.get('Time', ''), row.get('Confidence', ''),
                row.get('BBox_X1', ''), row.get('BBox_Y1', ''),
                row.get('BBox_X2', ''), row.get('BBox_Y2', '')
            ])

    return output.getvalue()


# ─────────────────────────────────────────────────────────────────
# 3. DarwinCore Archive (GBIF / OBIS standart format)
# ─────────────────────────────────────────────────────────────────
def to_darwincore_archive(csv_path: str, db_path: Optional[str] = None) -> bytes:
    """
    DarwinCore Archive (DwC-A) ZIP dosyası oluşturur.
    GBIF ve OBIS'e doğrudan yüklenebilir format.
    
    İçerik:
    - occurrence.csv  (Tespit kayıtları)
    - meta.xml        (Arşiv tanımlayıcı)
    - eml.xml         (Veri seti metadata)
    
    Referans: https://dwc.tdwg.org/terms/
    """
    # Veri topla
    db_rows = _read_detections_db(db_path) if db_path else []
    csv_rows = _read_detections_csv(csv_path) if not db_rows else []

    # ── occurrence.csv ──
    occ_output = io.StringIO()
    occ_writer = csv.writer(occ_output, delimiter='\t')

    # DarwinCore standart sütunları
    dwc_header = [
        "occurrenceID", "basisOfRecord", "eventDate",
        "scientificName", "vernacularName", "kingdom", "phylum",
        "class", "order", "family", "genus", "specificEpithet",
        "decimalLatitude", "decimalLongitude",
        "coordinateUncertaintyInMeters", "geodeticDatum",
        "occurrenceStatus", "individualCount",
        "identificationVerificationStatus",
        "measurementValue", "measurementType", "measurementUnit",
        "institutionCode", "datasetName", "informationWithheld"
    ]
    occ_writer.writerow(dwc_header)

    occurrence_id = 0

    if db_rows:
        for row in db_rows:
            occurrence_id += 1
            ts = row.get('timestamp', '')
            # Unix timestamp'i ISO formatına çevir
            try:
                event_date = datetime.fromtimestamp(float(ts), tz=timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
            except (ValueError, TypeError):
                event_date = str(ts)

            occ_writer.writerow([
                f"AG-PF-{occurrence_id:06d}",          # occurrenceID
                "MachineObservation",                    # basisOfRecord
                event_date,                              # eventDate
                "Lagocephalus sceleratus",               # scientificName
                "Silver-cheeked toadfish",               # vernacularName
                "Animalia",                              # kingdom
                "Chordata",                              # phylum
                "Actinopterygii",                        # class
                "Tetraodontiformes",                     # order
                "Tetraodontidae",                        # family
                "Lagocephalus",                          # genus
                "sceleratus",                            # specificEpithet
                row.get('latitude', ''),                 # decimalLatitude
                row.get('longitude', ''),                # decimalLongitude
                "10",                                    # coordinateUncertaintyInMeters (GPS ~10m)
                "WGS84",                                 # geodeticDatum
                "present",                               # occurrenceStatus
                "1",                                     # individualCount
                "MachineLearningPrediction",             # identificationVerificationStatus
                round(float(row.get('confidence', 0)), 4),  # measurementValue
                "confidence_score",                      # measurementType
                "probability",                           # measurementUnit
                "Antigravity",                           # institutionCode
                "Pufferfish Detection System",           # datasetName
                ""                                       # informationWithheld
            ])
    else:
        for row in csv_rows:
            occurrence_id += 1
            event_date = row.get('Date', '')
            event_time = row.get('Time', '')
            if event_date and event_time:
                event_date = f"{event_date}T{event_time}Z"

            occ_writer.writerow([
                f"AG-PF-{occurrence_id:06d}",
                "MachineObservation",
                event_date,
                "Lagocephalus sceleratus",
                "Silver-cheeked toadfish",
                "Animalia", "Chordata", "Actinopterygii",
                "Tetraodontiformes", "Tetraodontidae",
                "Lagocephalus", "sceleratus",
                "", "",  # No GPS
                "", "WGS84",
                "present", "1",
                "MachineLearningPrediction",
                row.get('Confidence', ''),
                "confidence_score", "probability",
                "Antigravity", "Pufferfish Detection System", ""
            ])

    # ── meta.xml ──
    meta_xml = """<?xml version="1.0" encoding="UTF-8"?>
<archive xmlns="http://rs.tdwg.org/dwc/text/"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="http://rs.tdwg.org/dwc/text/ http://rs.tdwg.org/dwc/text/tdwg_dwc_text.xsd">
  <core encoding="UTF-8" linesTerminatedBy="\\r\\n" fieldsTerminatedBy="\\t"
        fieldsEnclosedBy="" ignoreHeaderLines="1" rowType="http://rs.tdwg.org/dwc/terms/Occurrence">
    <files><location>occurrence.csv</location></files>
    <id index="0"/>
    <field index="1" term="http://rs.tdwg.org/dwc/terms/basisOfRecord"/>
    <field index="2" term="http://rs.tdwg.org/dwc/terms/eventDate"/>
    <field index="3" term="http://rs.tdwg.org/dwc/terms/scientificName"/>
    <field index="4" term="http://rs.tdwg.org/dwc/terms/vernacularName"/>
    <field index="5" term="http://rs.tdwg.org/dwc/terms/kingdom"/>
    <field index="6" term="http://rs.tdwg.org/dwc/terms/phylum"/>
    <field index="7" term="http://rs.tdwg.org/dwc/terms/class"/>
    <field index="8" term="http://rs.tdwg.org/dwc/terms/order"/>
    <field index="9" term="http://rs.tdwg.org/dwc/terms/family"/>
    <field index="10" term="http://rs.tdwg.org/dwc/terms/genus"/>
    <field index="11" term="http://rs.tdwg.org/dwc/terms/specificEpithet"/>
    <field index="12" term="http://rs.tdwg.org/dwc/terms/decimalLatitude"/>
    <field index="13" term="http://rs.tdwg.org/dwc/terms/decimalLongitude"/>
    <field index="14" term="http://rs.tdwg.org/dwc/terms/coordinateUncertaintyInMeters"/>
    <field index="15" term="http://rs.tdwg.org/dwc/terms/geodeticDatum"/>
    <field index="16" term="http://rs.tdwg.org/dwc/terms/occurrenceStatus"/>
    <field index="17" term="http://rs.tdwg.org/dwc/terms/individualCount"/>
    <field index="18" term="http://rs.tdwg.org/dwc/terms/identificationVerificationStatus"/>
    <field index="19" term="http://purl.org/dc/terms/measurementValue"/>
    <field index="20" term="http://purl.org/dc/terms/measurementType"/>
    <field index="21" term="http://purl.org/dc/terms/measurementUnit"/>
    <field index="22" term="http://rs.tdwg.org/dwc/terms/institutionCode"/>
    <field index="23" term="http://rs.tdwg.org/dwc/terms/datasetName"/>
    <field index="24" term="http://rs.tdwg.org/dwc/terms/informationWithheld"/>
  </core>
</archive>"""

    # ── eml.xml (Ecological Metadata Language) ──
    eml_xml = f"""<?xml version="1.0" encoding="UTF-8"?>
<eml:eml xmlns:eml="eml://ecoinformatics.org/eml-2.1.1"
         xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
         xsi:schemaLocation="eml://ecoinformatics.org/eml-2.1.1 eml.xsd"
         packageId="antigravity-pufferfish-{datetime.now(timezone.utc).strftime('%Y%m%d')}"
         system="Antigravity">
  <dataset>
    <title>Pufferfish (Lagocephalus sceleratus) Detection Dataset - Mediterranean</title>
    <creator>
      <organizationName>Antigravity Pufferfish Detection Project</organizationName>
    </creator>
    <pubDate>{datetime.now(timezone.utc).strftime('%Y-%m-%d')}</pubDate>
    <language>en</language>
    <abstract>
      <para>
        Automated underwater detection records of silver-cheeked toadfish
        (Lagocephalus sceleratus) collected by the Antigravity AI-powered
        detection system. Detections are made using a YOLO11m deep learning
        model deployed on Raspberry Pi 5 with real-time GPS geotagging.
        This invasive species poses significant ecological and public health
        risks in the Mediterranean Sea.
      </para>
    </abstract>
    <keywordSet>
      <keyword>Lagocephalus sceleratus</keyword>
      <keyword>pufferfish</keyword>
      <keyword>invasive species</keyword>
      <keyword>Mediterranean</keyword>
      <keyword>machine learning</keyword>
      <keyword>computer vision</keyword>
      <keyword>YOLO</keyword>
    </keywordSet>
    <intellectualRights>
      <para>Creative Commons Attribution 4.0 International (CC BY 4.0)</para>
    </intellectualRights>
    <coverage>
      <geographicCoverage>
        <geographicDescription>Mediterranean Sea coastal waters</geographicDescription>
        <boundingCoordinates>
          <westBoundingCoordinate>-5.5</westBoundingCoordinate>
          <eastBoundingCoordinate>36.0</eastBoundingCoordinate>
          <northBoundingCoordinate>46.0</northBoundingCoordinate>
          <southBoundingCoordinate>30.0</southBoundingCoordinate>
        </boundingCoordinates>
      </geographicCoverage>
      <taxonomicCoverage>
        <taxonomicClassification>
          <taxonRankName>Species</taxonRankName>
          <taxonRankValue>Lagocephalus sceleratus</taxonRankValue>
          <commonName>Silver-cheeked toadfish</commonName>
        </taxonomicClassification>
      </taxonomicCoverage>
    </coverage>
    <methods>
      <methodStep>
        <description>
          <para>
            Detection via YOLO11m deep learning model (INT8 quantized ONNX)
            running on Raspberry Pi 5. Camera: Pi Camera Module 3 at 640x480.
            CLAHE preprocessing applied for underwater visibility enhancement.
            Confidence threshold: configurable (default 0.60).
          </para>
        </description>
      </methodStep>
    </methods>
  </dataset>
</eml:eml>"""

    # ── ZIP arşivi oluştur ──
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('occurrence.csv', occ_output.getvalue())
        zf.writestr('meta.xml', meta_xml)
        zf.writestr('eml.xml', eml_xml)

    return zip_buffer.getvalue()
