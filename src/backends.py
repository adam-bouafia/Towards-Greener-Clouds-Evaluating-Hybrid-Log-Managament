# src/backends.py
import math
import time
from datetime import date, datetime

import mysql.connector
from mysql.connector import Error as MySQLError

from elasticsearch import Elasticsearch
from elasticsearch import exceptions as es_exceptions

import ipfshttpclient

try:
    import numpy as np
except Exception:
    np = None


class BackendManager:
    def __init__(self, config):
        """
        config may expose:
          - MYSQL_DATABASE (str)
          - MYSQL_USER (str, default "log_user")
          - MYSQL_PASSWORD (str, default "your_strong_password")
          - ES_INDEX_NAME (str, default "logs_raw")
        """
        self.config = config
        self.mysql_conn = None
        self.es_client = None
        self.ipfs_client = None
        self._es_index_ready = False

    # ----------------- Helpers -----------------
    def _to_jsonable(self, v):
        """Convert Python/NumPy values to JSON-safe types for Elasticsearch."""
        if v is None:
            return None
        if isinstance(v, (str, bool, int)):
            return v
        if isinstance(v, float):
            if math.isnan(v) or math.isinf(v):
                return None
            return v
        if np is not None:
            if isinstance(v, (np.bool_,)):
                return bool(v)
            if isinstance(v, (np.integer,)):
                return int(v)
            if isinstance(v, (np.floating,)):
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
        if isinstance(v, (datetime, date)):
            return v.isoformat()
        # Fallback: stringify unknown types
        return str(v)

    def _json_safe_doc(self, doc: dict) -> dict:
        return {k: self._to_jsonable(v) for k, v in doc.items()}

    # ----------------- MySQL -----------------
    def setup_mysql(self, retries=30, delay=1.0):
        if self.mysql_conn:
            return
        db_name = getattr(self.config, "MYSQL_DATABASE", "logs_db")
        user = getattr(self.config, "MYSQL_USER", "log_user")
        pwd = getattr(self.config, "MYSQL_PASSWORD", "your_strong_password")

        for attempt in range(1, retries + 1):
            try:
                self.mysql_conn = mysql.connector.connect(
                    host="127.0.0.1", user=user, password=pwd, database=db_name
                )
                cursor = self.mysql_conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS logs (
                        id BIGINT AUTO_INCREMENT PRIMARY KEY,
                        log_timestamp VARCHAR(255),
                        log_level VARCHAR(32),
                        source_host VARCHAR(255),
                        service_name VARCHAR(128),
                        log_message TEXT
                    )
                    """
                )
                self.mysql_conn.commit()
                cursor.close()
                print(f"[MYSQL] Connected to {db_name} as {user}.")
                return
            except MySQLError as e:
                print(f"[MYSQL] connect attempt {attempt}/{retries} failed: {e}")
                time.sleep(delay)

        print("[MYSQL] Could not connect; MySQL writes will be skipped.")
        self.mysql_conn = None

    def write_to_mysql(self, log_data):
        """
        Insert one row into MySQL.
        Returns:
          dict {"success": True, "rowid": <lastrowid>} on success
          False on failure (so bool(...) works in caller)
        """
        if not self.mysql_conn:
            self.setup_mysql()
        if not self.mysql_conn:
            return False
        try:
            cursor = self.mysql_conn.cursor()
            sql = (
                "INSERT INTO logs (log_timestamp, log_level, source_host, "
                "service_name, log_message) VALUES (%s, %s, %s, %s, %s)"
            )
            val = (
                (log_data.get("Time") or None),
                (log_data.get("Level") or None),
                (log_data.get("Node") or None),
                (log_data.get("Component") or None),
                (log_data.get("Content") or None),
            )
            cursor.execute(sql, val)
            self.mysql_conn.commit()
            rowid = cursor.lastrowid
            cursor.close()
            return {"success": True, "rowid": rowid}
        except MySQLError as e:
            print(f"[MYSQL ERROR] {e}")
            return False

    # ----------------- Elasticsearch -----------------
    def setup_elk(self, retries=30, delay=1.0):
        if self.es_client:
            return
        self.es_client = Elasticsearch(
            hosts=[{"host": "127.0.0.1", "port": 9200, "scheme": "http"}],
            request_timeout=30,
            retry_on_timeout=True,
            max_retries=5,
        )
        for attempt in range(1, retries + 1):
            try:
                if self.es_client.ping():
                    print("[ELK] Connected to Elasticsearch on http://127.0.0.1:9200")
                    return
            except es_exceptions.ConnectionError as e:
                print(f"[ELK] ping attempt {attempt}/{retries} failed: {e}")
            time.sleep(delay)

        print("[ELK] Could not connect; ELK writes will be skipped.")
        self.es_client = None

    def _ensure_elk_index(self):
        """Create a safe index with Date/Time as keyword and date_detection disabled."""
        if self._es_index_ready or not self.es_client:
            return
        index = getattr(self.config, "ES_INDEX_NAME", "logs_raw")
        try:
            if not self.es_client.indices.exists(index=index):
                self.es_client.indices.create(
                    index=index,
                    mappings={
                        "date_detection": False,  # do NOT auto-parse dates
                        "properties": {
                            "Date": {"type": "keyword", "ignore_above": 512},
                            "Time": {"type": "keyword", "ignore_above": 512},
                            # Optional: dual fields for common text fields
                            "Level": {"type": "keyword", "ignore_above": 256},
                            "Component": {"type": "keyword", "ignore_above": 512},
                            "LogSource": {"type": "keyword", "ignore_above": 512},
                        },
                    },
                    settings={
                        "index.mapping.ignore_malformed": True
                    },
                )
                print(f"[ELK] Created index '{index}' with date_detection=false (Date/Time as keyword).")
            self._es_index_ready = True
        except Exception as e:
            print(f"[ELK] Failed to ensure index mapping: {e}")

    def write_to_elk(self, log_data):
        """
        Index one document into Elasticsearch.
        Returns:
          Elasticsearch response dict (contains '_id' on success)
          False on failure
        """
        if not self.es_client:
            self.setup_elk()
        if not self.es_client:
            return False
        self._ensure_elk_index()

        try:
            safe_doc = self._json_safe_doc(log_data)
            index = getattr(self.config, "ES_INDEX_NAME", "logs_raw")
            resp = self.es_client.index(index=index, document=safe_doc)
            # Typical resp keys: {'_index','_id','_version','result','_shards',...}
            return resp
        except (es_exceptions.TransportError, es_exceptions.ConnectionError) as e:
            print(f"[ELK ERROR] {e}")
            return False
        except Exception as e:
            print(f"[ELK ERROR] {e}")
            return False

    # ----------------- IPFS -----------------
    def setup_ipfs(self):
        if self.ipfs_client:
            return
        try:
            # Kubo API on localhost
            self.ipfs_client = ipfshttpclient.connect("/ip4/127.0.0.1/tcp/5001/http")
            _ = self.ipfs_client.id()
            print("[IPFS] Connected to IPFS daemon on 127.0.0.1:5001.")
        except Exception as e:
            print(f"[IPFS] Not connected ({e}); will simulate writes.")
            self.ipfs_client = None

    def write_to_ipfs(self, log_data):
        """
        Store the Content bytes in IPFS and return the CID (string).
        On failure or when not connected, returns a deterministic hash string.
        """
        content = (log_data.get("Content") or "")
        content_bytes = content.encode("utf-8")
        if self.ipfs_client is None:
            import hashlib
            return hashlib.sha256(content_bytes).hexdigest()[:46]
        try:
            # Use add_bytes to avoid temp files
            cid = self.ipfs_client.add_bytes(content_bytes)
            # Optionally pin: self.ipfs_client.pin.add(cid)
            return cid
        except Exception as e:
            print(f"[IPFS ERROR] {e}; simulating.")
            import hashlib
            return hashlib.sha256(content_bytes).hexdigest()[:46]

    # ----------------- Cleanup -----------------
    def close_connections(self):
        if self.mysql_conn:
            try:
                self.mysql_conn.close()
            except Exception:
                pass
            self.mysql_conn = None
