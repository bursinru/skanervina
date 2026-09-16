import os
from pathlib import Path
import psycopg
from psycopg.rows import dict_row


def connect():
    return psycopg.connect(os.environ['DATABASE_URL'], row_factory=dict_row, connect_timeout=5)


def migrate():
    with connect() as db:
        db.execute((Path(__file__).resolve().parents[1] / 'migrations' / '001_initial.sql').read_text())
