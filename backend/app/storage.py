"""Persistent anonymous browser profiles. Tokens are stored only as hashes."""
import hashlib
import json
import os
import secrets
import sqlite3
from pathlib import Path

DB_PATH = Path(os.getenv('DATABASE_PATH', str(Path(__file__).resolve().parents[1] / 'data' / 'scanner.sqlite3')))


def connect():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(DB_PATH, timeout=10)
    db.execute('CREATE TABLE IF NOT EXISTS profiles (id TEXT PRIMARY KEY, state TEXT NOT NULL)')
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
