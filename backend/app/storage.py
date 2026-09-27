"""Persistent anonymous browser profiles. Tokens are stored only as hashes."""
import hashlib
import json
import os
import secrets
import sqlite3
import time
from pathlib import Path

DB_PATH = Path(os.getenv('DATABASE_PATH', str(Path(__file__).resolve().parents[1] / 'data' / 'scanner.sqlite3')))


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.execute('CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, state TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS scan_events (id INTEGER PRIMARY KEY AUTOINCREMENT, slug TEXT NOT NULL, created_at REAL NOT NULL)')
    db.execute('CREATE INDEX IF NOT EXISTS scan_events_created_at ON scan_events (created_at DESC)')
    return db


def profile(token):
    if os.getenv('DATABASE_URL'):
        return pg_profile(token)
    key = hashlib.sha256((token or '').encode()).hexdigest()
    with connect() as db:
        row = db.execute('SELECT state FROM profiles WHERE id = ?', (key,)).fetchone()
        if row:
            return token, json.loads(row[0])
        token = secrets.token_urlsafe(32)
        key = hashlib.sha256(token.encode()).hexdigest()
        state = {'saved': [], 'ratings': {}}
        db.execute('INSERT INTO profiles VALUES (?, ?)', (key, json.dumps(state)))
        return token, state


def save(token, state):
    if os.getenv('DATABASE_URL'):
        from .database import connect as pg_connect
        from psycopg.types.json import Jsonb
        with pg_connect() as db:
            db.execute('UPDATE scanner_profiles SET state = %s, updated_at = now() WHERE id = %s', (Jsonb(state), hashlib.sha256(token.encode()).hexdigest()))
        return
    key = hashlib.sha256(token.encode()).hexdigest()
    with connect() as db:
        db.execute('UPDATE profiles SET state = ? WHERE id = ?', (json.dumps(state, ensure_ascii=False), key))


def pg_profile(token):
    from .database import connect as pg_connect
    from psycopg.types.json import Jsonb
    key = hashlib.sha256((token or '').encode()).hexdigest()
    with pg_connect() as db:
        row = db.execute('SELECT state FROM scanner_profiles WHERE id = %s', (key,)).fetchone()
        if row:
            return token, row['state']
        token = secrets.token_urlsafe(32)
        state = {'saved': [], 'ratings': {}}
        db.execute('INSERT INTO scanner_profiles (id, state) VALUES (%s, %s)',
                   (hashlib.sha256(token.encode()).hexdigest(), Jsonb(state)))
        return token, state


SCAN_RETENTION_DAYS = 30


def record_scan(slug):
    """Remember a matched wine for the public feed; old events are pruned on write."""
    if os.getenv('DATABASE_URL'):
        from .database import connect as pg_connect
        with pg_connect() as db:
            db.execute('INSERT INTO scan_events (slug) VALUES (%s)', (slug,))
            db.execute(f"DELETE FROM scan_events WHERE created_at < now() - interval '{SCAN_RETENTION_DAYS} days'")
        return
    now = time.time()
    with connect() as db:
        db.execute('INSERT INTO scan_events (slug, created_at) VALUES (?, ?)', (slug, now))
        db.execute('DELETE FROM scan_events WHERE created_at < ?', (now - SCAN_RETENTION_DAYS * 86400,))


def recent_scans(limit):
    """Latest distinct wines as (slug, unix seconds), newest first."""
    if os.getenv('DATABASE_URL'):
        from .database import connect as pg_connect
        with pg_connect() as db:
            rows = db.execute(
                'SELECT slug, extract(epoch FROM max(created_at)) AS last FROM scan_events '
                'GROUP BY slug ORDER BY last DESC LIMIT %s', (limit,)).fetchall()
        return [(row['slug'], float(row['last'])) for row in rows]
    with connect() as db:
        rows = db.execute(
            'SELECT slug, max(created_at) AS last FROM scan_events GROUP BY slug ORDER BY last DESC LIMIT ?',
            (limit,)).fetchall()
    return [(slug, float(last)) for slug, last in rows]
