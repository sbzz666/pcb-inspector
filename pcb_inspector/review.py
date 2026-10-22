"""Append-only human review snapshots for reproducible inspection decisions."""
import json
from pathlib import Path
import sqlite3
from datetime import datetime, timezone
from . import ROOT


def connect(path=None):
    path = Path(path or ROOT / "reports/reviews.sqlite3")
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS reviews (id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, image_sha256 TEXT NOT NULL, image_name TEXT NOT NULL, payload TEXT NOT NULL)")
    return conn


def save_review(image_hash, image_name, payload, path=None):
    with connect(path) as conn:
        result = conn.execute("INSERT INTO reviews(created_at,image_sha256,image_name,payload) VALUES(?,?,?,?)", (datetime.now(timezone.utc).isoformat(), image_hash, image_name, json.dumps(payload, ensure_ascii=False)))
        return result.lastrowid


def list_reviews(path=None):
    with connect(path) as conn:
        rows = conn.execute("SELECT id, created_at, image_sha256, image_name, payload FROM reviews ORDER BY id DESC LIMIT 200").fetchall()
    return [{"id": row[0], "created_at": row[1], "image_sha256": row[2], "image_name": row[3], "payload": json.loads(row[4])} for row in rows]
